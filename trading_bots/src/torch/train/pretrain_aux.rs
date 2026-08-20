//! The auxiliary-resolution training stream.
//!
//! # Why this exists
//!
//! The deployment corpus is `long_data/bars/*.300.bars`, 2021-08-17 onward. It contains no 2000
//! dot-com unwind, no 2008 credit crisis and only the tail of 2020, and no amount of widening it
//! sideways fixes that: equity returns are cross-sectionally correlated, so more tickers over the
//! same five years buy bars, not regimes.
//!
//! That gap is not a theoretical worry, it is a MEASURED defect in the shipped predictor. The
//! Kelly bench on the promoted checkpoint beats the unconditional-marginal null by +4.69 bps/bar
//! on val and +4.28 on test, both intervals excluding zero — but 84.8% of its bars sit pinned at
//! the 4x leverage cap, and one window was wiped out entirely. Kelly saturating at the cap means
//! `E[log(1 + fR)]` is still increasing at `f = 4`, and that happens for exactly one reason: the
//! predictive distribution's LEFT TAIL IS TOO THIN. The model does not believe in crashes, because
//! it has never seen one. `long_data/bars/*.86400.bars` — 4,748 symbols, 21,499,137 daily bars back
//! to 1970-01-02 — is the only available cure, and 74.9% of those bars predate the intraday
//! corpus entirely.
//!
//! # Why it is safe to mix timeframes
//!
//! All five degrees of freedom are scale-invariant ratios (`ln(close/prev_close)`, `ln(high/low)`,
//! two positions inside the log range, `ln(volume/ema_volume)`), so the daily corpus's dividend
//! adjustment — every OHLC scaled by a piecewise-constant `adjclose/close` — cancels everywhere
//! except the single ex-dividend bar, where it correctly turns the price return into a total
//! return. The synthetic daily `vwap` and `trades = 0` are never encoded at all.
//!
//! What is NOT safe is scoring a daily bar against 5-minute bins. The two DOF distributions are
//! nowhere near each other: measured over the whole corpus, `r` has a 0.01% quantile of -0.182 at
//! five minutes against -0.409 daily, and the median `s` is 0.0015 against 0.0211, 14x wider. So
//! every resolution gets its OWN fitted supports (`bar_supports.<res>.json`), the trunk gets the
//! resolution as a conditioning id ([`crate::torch::dataset::TIME_RESOLUTION`]), and a batch is
//! never mixed: an auxiliary step is a WHOLE step drawn from one resolution and scored against
//! that resolution's supports.
//!
//! # Why it is not a promotion criterion
//!
//! Selection and promotion stay on the 300s held-out `nll_bar_conditional`, unchanged. A model
//! that got better at daily bars and worse at five-minute bars must lose. Daily is training
//! signal only.
//!
//! # Survivorship, stated rather than hidden
//!
//! Yahoo purges delisted tickers, so the daily corpus is 98.5% survivors: 70 of its 4,748 symbols
//! are delisted, 1.5%, against 1,029 of the 5,728 intraday files, 18.0%. That biases the
//! UNCONDITIONAL marginal — how often a name dies — which is not what this model predicts. It
//! does not bias the conditional law `p(next bar DOF | history)` inside a crash, which is the
//! entire thing the auxiliary corpus is here to teach, and which survivors lived through in full.

use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use shared::report::{Report, ReportKind, ReportSeries, ScaleKind};
use tch::Device;

use crate::torch::bar_dist::{BarScoring, BarSupports};
use crate::torch::dataset::{
    load_auxiliary, BarBatch, BarCorpus, BarSampler, CorpusAnomalies, CoverageAudit, PassLayout,
    PassLedger, PassPlan, Split, AUXILIARY_MIN_BARS,
};

/// Ramp contexts for an auxiliary resolution, in bars, shortest first.
///
/// Deliberately NOT the deployment ramp of 896/1472/2048. A context is a number of BARS, and 896
/// daily bars is 3.5 years of conditioning where 896 five-minute bars is five sessions — the same
/// integer is not the same curriculum. Sized off the measured per-symbol daily train axis instead
/// (p10 422, median 3,326, max 14,062 bars under the campaign's pinned bounds):
///
/// * At a shortest context of 256 exactly 349 of 4,748 symbols cannot tile one window, carrying
///   36,414 bars — 0.18% of the 20,498,862 auxiliary train bars. Those land in the coverage
///   audit's short-symbol remainder, counted, not silently dropped. Reusing the deployment's 896
///   would strand 752 symbols and 253,720 bars for nothing.
/// * At a longest context of 1024 — four years of sessions — 3,885 symbols and 20,136,882 bars
///   (98.2%) are reachable. Reusing the deployment's 2048 reaches only 3,056 symbols and 18.9M
///   bars, and buys no crisis structure a four-year window does not already span: the 2007-10 ..
///   2009-03 crisis is 375 sessions, so a 1024-bar window holds the whole arc plus three years of
///   the regime before it.
pub const AUXILIARY_CONTEXTS: [i64; 3] = [256, 512, 1024];

/// Context the auxiliary held-out NLL is measured at.
///
/// Forced by the geometry, not chosen. The campaign's pinned bounds put `train | val` at
/// 2025-10-07 and `val | test` at 2026-03-13, five months apart, so EVERY daily symbol holds
/// exactly 108 val bars and 106 test bars however deep its history. No daily symbol can host a
/// 256-bar window inside the held-out window, let alone a 2048-bar one, so the auxiliary held-out
/// number is measured at 64 bars — three months of conditioning — over 4,748 symbols.
///
/// That number is therefore NOT comparable to the 300s held-out NLL, which is measured at a far
/// longer context on a completely different bin geometry. It is comparable to ITSELF across runs,
/// which is the question it exists to answer: did the auxiliary corpus get learned, ignored, or
/// actively damaged.
pub const AUXILIARY_HELDOUT_CONTEXT: i64 = 64;

/// One auxiliary resolution: its corpus, its own supports, its own pass partition and ledger.
pub struct AuxiliaryStream {
    res_secs: u32,
    corpus: BarCorpus,
    /// Host-side, for the checkpoint sidecar.
    supports: BarSupports,
    supports_dev: BarSupports,
    /// Additive constant the density scoring rule contributes, so the loss-term shares of an
    /// auxiliary step are on the same categorical scale as a primary step's.
    share_scale_offset: f64,
    /// One per entry of [`AUXILIARY_CONTEXTS`].
    samplers: Vec<BarSampler>,
    pass: PassPlan,
    layout: Arc<PassLayout>,
    ledger: PassLedger,
    cursor: Vec<usize>,
    steps_per_epoch: usize,
    batch: Vec<usize>,
    /// Running totals for the epoch row.
    pub bars_seen: u64,
    pub nll_sum: f64,
    pub steps: usize,
}

impl std::fmt::Debug for AuxiliaryStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuxiliaryStream")
            .field("res_secs", &self.res_secs)
            .field("symbols", &self.corpus.symbols().len())
            .field("train_bars", &self.pass.covered_bars())
            .field("steps_per_epoch", &self.steps_per_epoch)
            .finish()
    }
}

/// Everything the caller must supply to open the auxiliary streams, gathered so the signature
/// does not grow a seventh positional `usize`.
pub struct AuxiliaryConfig<'a> {
    pub resolutions: &'a [u32],
    pub base_batch: usize,
    pub batch_ramp: &'a [usize],
    pub seed: u64,
    pub scoring: BarScoring,
    pub device: Device,
}

impl AuxiliaryStream {
    /// Open every requested auxiliary resolution against the deployment corpus's split instants.
    ///
    /// `fit` fits (or reuses) one resolution's supports; it is passed in rather than reimplemented
    /// so there is exactly one provenance check in the process, the one `fit_supports_at` does.
    /// Empty `resolutions` returns an empty vector and costs nothing: the auxiliary corpus enters
    /// a run only when it is asked for.
    pub fn open(
        deployment: &BarCorpus,
        cfg: &AuxiliaryConfig<'_>,
        mut fit: impl FnMut(&BarCorpus, &str) -> Result<BarSupports>,
    ) -> Result<Vec<Self>> {
        if cfg.resolutions.is_empty() {
            return Ok(Vec::new());
        }
        ensure!(
            cfg.batch_ramp.len() == AUXILIARY_CONTEXTS.len(),
            "the auxiliary ramp has {} contexts but the batch ramp has {} stages",
            AUXILIARY_CONTEXTS.len(),
            cfg.batch_ramp.len()
        );
        let requested: Vec<(u32, usize)> = cfg
            .resolutions
            .iter()
            .map(|&res| (res, AUXILIARY_MIN_BARS))
            .collect();
        let corpora = load_auxiliary(deployment, &requested)
            .context("failed opening the auxiliary corpora")?;

        let batch: Vec<usize> = cfg
            .batch_ramp
            .iter()
            .map(|multiple| (cfg.base_batch * multiple).max(1))
            .collect();
        // `batch[s] * context[s]`: the bar-token budget of the stage, which is what the pass
        // partition apportions by. Same rule the deployment ramp uses.
        let weights: Vec<f64> = (0..AUXILIARY_CONTEXTS.len())
            .map(|stage| (batch[stage] as i64 * AUXILIARY_CONTEXTS[stage]) as f64)
            .collect();

        let mut out = Vec::with_capacity(corpora.len());
        for corpus in corpora {
            let res_secs = corpus.res_secs();
            let fingerprint = corpus.identity_fingerprint();
            let supports = fit(&corpus, &fingerprint)?;
            // Mixed with the resolution so two auxiliary resolutions in one run never share a
            // partition geometry, and so the auxiliary pass can never collide with the primary's
            // stream for any `(seed, epoch)`.
            let seed = cfg.seed ^ ((res_secs as u64) << 32);
            let pass = PassPlan::new(
                &corpus,
                Split::Train,
                &AUXILIARY_CONTEXTS,
                &weights,
                seed,
            )
            .with_context(|| {
                format!("failed partitioning the {res_secs}s auxiliary training split")
            })?;
            let samplers: Vec<BarSampler> = AUXILIARY_CONTEXTS
                .iter()
                .map(|&context| BarSampler::new(&corpus, Split::Train, context, seed))
                .collect();
            let stage_steps: Vec<usize> = pass
                .windows_per_stage()
                .iter()
                .zip(batch.iter())
                .map(|(windows, batch)| windows.div_ceil(*batch))
                .collect();
            let steps_per_epoch = stage_steps.iter().sum();
            let layout = pass.layout(0);
            let share_scale_offset = if cfg.scoring.is_density() {
                supports.log_measure_bar()
            } else {
                0.0
            };
            println!(
                "[aux {res_secs}s] {} symbols, {} train bars, contexts {:?}, {} steps/epoch, \
                 {} bar-tokens/epoch",
                corpus.symbols().len(),
                pass.covered_bars(),
                AUXILIARY_CONTEXTS,
                steps_per_epoch,
                (0..AUXILIARY_CONTEXTS.len())
                    .map(|s| stage_steps[s] as u64 * batch[s] as u64 * AUXILIARY_CONTEXTS[s] as u64)
                    .sum::<u64>()
            );
            let remainder = pass.remainder();
            println!(
                "[aux {res_secs}s] pass remainder {} bars: {} head, {} in {} symbols shorter \
                 than the {}-bar shortest context, {} sub-context holes",
                remainder.total(),
                remainder.head_bars,
                remainder.short_symbol_bars,
                remainder.short_symbols,
                AUXILIARY_CONTEXTS[0],
                remainder.hole_bars
            );
            out.push(Self {
                res_secs,
                supports_dev: supports.to_device(cfg.device),
                supports,
                share_scale_offset,
                cursor: vec![0; AUXILIARY_CONTEXTS.len()],
                ledger: PassLedger::new(&layout),
                layout,
                pass,
                samplers,
                steps_per_epoch,
                batch: batch.clone(),
                corpus,
                bars_seen: 0,
                nll_sum: 0.0,
                steps: 0,
            });
        }
        Ok(out)
    }

    pub fn res_secs(&self) -> u32 {
        self.res_secs
    }

    pub fn corpus(&self) -> &BarCorpus {
        &self.corpus
    }

    pub fn supports(&self) -> &BarSupports {
        &self.supports
    }

    pub fn supports_dev(&self) -> &BarSupports {
        &self.supports_dev
    }

    pub fn share_scale_offset(&self) -> f64 {
        self.share_scale_offset
    }

    /// Optimizer steps one full pass over this corpus takes. ADDITIVE to the deployment's
    /// `steps_per_epoch`: an auxiliary step must never consume a primary one, or the primary
    /// stage ends its epoch short and [`CoverageAudit::require_full_pass`] kills the run.
    pub fn steps_per_epoch(&self) -> usize {
        self.steps_per_epoch
    }

    /// Whether an auxiliary step fires after primary step `primary_step` of a pass that runs
    /// `primary_steps` steps.
    ///
    /// Bresenham: fires exactly [`Self::steps_per_epoch`] times per primary pass and spreads them
    /// evenly, so the auxiliary distribution is present throughout the epoch rather than in a
    /// block at one end where it would read as a regime change to the optimizer.
    pub fn fires_after(&self, primary_step: usize, primary_steps: usize) -> bool {
        if self.steps_per_epoch == 0 || primary_steps == 0 {
            return false;
        }
        let within = primary_step % primary_steps;
        let scaled = |p: usize| p * self.steps_per_epoch / primary_steps;
        scaled(within + 1) > scaled(within)
    }

    /// The next auxiliary batch, or `None` once this pass is exhausted.
    ///
    /// Stages are drained in order, shortest context first, so the auxiliary runs the same
    /// short-to-long curriculum the deployment ramp does.
    pub fn draw(&mut self, device: Device) -> Option<(usize, BarBatch, usize)> {
        for stage in 0..AUXILIARY_CONTEXTS.len() {
            let cursor = self.cursor[stage];
            let refs = self.layout.draw(stage, cursor, self.batch[stage]).to_vec();
            if refs.is_empty() {
                continue;
            }
            let sample = self.samplers[stage].batch_of(&refs, device);
            // Marked from the DRAW length, never the planned batch: the last draw of a stage is
            // short, and marking the planned count would record windows that were never issued.
            let drawn = refs.len();
            self.ledger.mark(stage, cursor, drawn);
            self.cursor[stage] = cursor + drawn;
            self.bars_seen += drawn as u64 * AUXILIARY_CONTEXTS[stage] as u64;
            return Some((stage, sample, drawn));
        }
        None
    }

    pub fn context(&self, stage: usize) -> i64 {
        AUXILIARY_CONTEXTS[stage]
    }

    /// Coverage of the pass just finished. Audited exactly like the primary's: an auxiliary
    /// shortfall is no more forgivable than a deployment one.
    pub fn audit(&self) -> CoverageAudit {
        self.pass.audit(&self.layout, &self.ledger)
    }

    /// Redraw the partition for `epoch` and reset every cursor and the ledger together.
    pub fn begin_pass(&mut self, epoch: usize) {
        self.layout = self.pass.layout(epoch);
        self.ledger = PassLedger::new(&self.layout);
        self.cursor.iter_mut().for_each(|cursor| *cursor = 0);
    }

    /// Mean NLL over this epoch's auxiliary steps, and a reset. NaN when no step ran, which is
    /// the honest reading for "the auxiliary stream was configured but never fired".
    pub fn take_epoch_nll(&mut self) -> f64 {
        let mean = if self.steps == 0 {
            f64::NAN
        } else {
            self.nll_sum / self.steps as f64
        };
        self.nll_sum = 0.0;
        self.steps = 0;
        mean
    }

    pub fn record_step(&mut self, nll_bar: f64) {
        self.nll_sum += nll_bar;
        self.steps += 1;
    }

    /// The anomaly audit for this resolution, for the multi-resolution corpus report.
    pub fn scan_anomalies(&self) -> CorpusAnomalies {
        self.corpus.scan_anomalies()
    }

    /// Bars one pass over this corpus reaches. Kept OUT of the deployment's `unique_bar_reuse`
    /// denominator on purpose: that ratio measures how many times the deployment corpus is
    /// walked, and folding a second corpus into it would make it measure neither.
    pub fn covered_bars(&self) -> u64 {
        self.pass.covered_bars()
    }

    /// Bars one pass CANNOT target, by named bucket. `covered_bars() + pass_remainder_total()`
    /// must equal the split's bars exactly, which is the auxiliary half of the same accounting
    /// invariant the deployment pass is held to.
    pub fn pass_remainder_total(&self) -> u64 {
        self.pass.remainder().total()
    }
}

/// Report base name; registered in [`shared::report::PRETRAIN_REPORT_BASES`].
pub const AUXILIARY_REPORT_BASE: &str = "pretrain_auxiliary_nll";

/// One epoch-indexed row per auxiliary resolution: the pass's mean TRAINING NLL and the
/// HELD-OUT NLL on that resolution's own val windows.
///
/// Both are needed and neither substitutes for the other. Training NLL falling proves the
/// auxiliary stream is being optimized; only the held-out number can distinguish "the daily
/// corpus was learned" from "the daily corpus was memorized". And the pair is what answers the
/// question this whole channel exists for — did the auxiliary corpus help, hurt, or get ignored
/// — which a run that reported only the deployment curve cannot answer at all.
#[derive(Clone, Debug, Default)]
pub struct AuxiliaryReport {
    /// Parallel to [`Self::rows`]: the resolution each row series belongs to.
    res_secs: Vec<u32>,
    /// `[resolution][epoch]` train and held-out NLL, in nats per bar under the run's scoring.
    train: Vec<Vec<f32>>,
    held_out: Vec<Vec<f32>>,
    /// Symbols and train bars per resolution, for the title. Fixed for the run.
    shape: Vec<(usize, u64)>,
}

impl AuxiliaryReport {
    pub fn new(streams: &[AuxiliaryStream]) -> Self {
        Self {
            res_secs: streams.iter().map(AuxiliaryStream::res_secs).collect(),
            train: vec![Vec::new(); streams.len()],
            held_out: vec![Vec::new(); streams.len()],
            shape: streams
                .iter()
                .map(|s| (s.corpus.symbols().len(), s.covered_bars()))
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.res_secs.is_empty()
    }

    /// Push one epoch's pair. A resolution that ran no step this epoch pushes NaN, which the
    /// chart renders as a GAP rather than as a zero that reads like a perfect fit.
    pub fn record(&mut self, index: usize, train: f64, held_out: f64) {
        self.train[index].push(train as f32);
        self.held_out[index].push(held_out as f32);
    }

    pub fn report(&self) -> Report {
        let mut series = Vec::with_capacity(2 * self.res_secs.len());
        for (index, res) in self.res_secs.iter().enumerate() {
            series.push(ReportSeries {
                label: format!("train@{res}s"),
                values: self.train[index].clone(),
            });
            series.push(ReportSeries {
                label: format!("held-out@{res}s"),
                values: self.held_out[index].clone(),
            });
        }
        let shape = self
            .res_secs
            .iter()
            .zip(self.shape.iter())
            .map(|(res, (symbols, bars))| format!("{res}s: {symbols} symbols, {bars} train bars"))
            .collect::<Vec<_>>()
            .join(" | ");
        Report {
            title: format!(
                "AUXILIARY resolutions — training signal only, NEVER a promotion criterion \
                 ({shape}). Held-out measured at {AUXILIARY_HELDOUT_CONTEXT} bars on each \
                 resolution's OWN supports, so these curves are comparable across runs but NOT \
                 to the deployment resolution's NLL."
            ),
            x_label: Some("epoch".to_string()),
            y_label: Some("nats per bar (own bin geometry)".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine { series },
        }
    }

    /// Write `<dir>/pretrain_auxiliary_nll.report.bin`. A run with no auxiliary resolution
    /// writes nothing, so the panel is absent rather than blank.
    pub fn write_report(&self, dir: &Path) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let path = dir.join(format!("{AUXILIARY_REPORT_BASE}.report.bin"));
        shared::report::write_report(&path, &self.report())
            .with_context(|| format!("writing {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::universe::eligible_bar_universe;
    use shared::bars::{write_bar_file, PackedBar};
    use shared::report::read_report;
    use std::path::PathBuf;

    /// `count` synthetic bars of `res_secs` ending just before `end_ms`, on a deterministic
    /// random walk so the fitted supports of two resolutions genuinely differ.
    fn synth(count: usize, res_secs: u32, end_ms: i64, seed: u64) -> Vec<PackedBar> {
        let step = res_secs as i64 * 1000;
        let mut state = seed | 1;
        let mut close = 100.0f32;
        (0..count)
            .map(|i| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Scaled by the resolution so a daily bar's return is genuinely wider than a
                // five-minute one's, which is the whole reason the two need separate supports.
                let unit = ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
                let width = 0.001 * (res_secs as f32 / 300.0).sqrt();
                let open = close;
                close = (open * (1.0 + unit * width)).max(1.0);
                let (high, low) = (open.max(close) * 1.001, open.min(close) * 0.999);
                PackedBar {
                    ts_ms: end_ms - (count - i) as i64 * step,
                    open,
                    high,
                    low,
                    close,
                    volume: 10_000.0 + unit.abs() * 5_000.0,
                    vwap: (high + low) * 0.5,
                    trades: 100,
                }
            })
            .collect()
    }

    struct Fixture {
        dir: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A deployment corpus at `RES` plus two auxiliary resolutions whose bars all sit strictly
    /// BEFORE the deployment corpus starts — the real shape, where the deep-history corpus
    /// predates the intraday one.
    fn fixture(name: &str) -> (Fixture, BarCorpus) {
        const RES: u32 = 300;
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_aux_{name}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("fixture dir");
        // 2024-01-01, so every auxiliary bar is older than every deployment bar.
        let deployment_start = 1_704_067_200_000i64;
        for (index, symbol) in ["AAA", "BBB", "CCC"].iter().enumerate() {
            let bars = synth(
                6_000,
                RES,
                deployment_start + 6_000 * RES as i64 * 1000,
                index as u64 + 11,
            );
            write_bar_file(&dir.join(format!("{symbol}.{RES}.bars")), symbol, RES, &bars)
                .expect("deployment bars");
            for res in [3_600u32, 86_400] {
                let aux = synth(3_000, res, deployment_start, index as u64 * 31 + res as u64);
                write_bar_file(&dir.join(format!("{symbol}.{res}.bars")), symbol, res, &aux)
                    .expect("auxiliary bars");
            }
        }
        let corpus = BarCorpus::load(&dir, RES, 100).expect("deployment corpus");
        (Fixture { dir }, corpus)
    }

    /// The writer named in `pretrain_reports::CYCLE_EXEMPT` for `pretrain_auxiliary_nll`.
    ///
    /// Drives the whole auxiliary path — open, fit per-resolution supports, tile the pass, draw
    /// every window, accumulate, roll the epoch, write the report — and reads the artifact back.
    /// Exists because an exemption from the full-cycle walk is only honest if some test actually
    /// executes the writer; a stated reason is not coverage.
    #[test]
    fn the_auxiliary_report_lands_with_one_distinguishable_series_pair_per_resolution() {
        let (fx, deployment) = fixture("report");
        let mut streams = AuxiliaryStream::open(
            &deployment,
            &AuxiliaryConfig {
                resolutions: &[3_600, 86_400],
                base_batch: 2,
                batch_ramp: &[1, 1, 1],
                seed: 0xA11CE,
                scoring: BarScoring::Density,
                device: Device::Cpu,
            },
            |corpus, _fingerprint| Ok(corpus.fit_supports(4_096, 7)),
        )
        .expect("auxiliary streams open");
        assert_eq!(streams.len(), 2, "both auxiliary resolutions must load");
        assert_eq!(
            streams.iter().map(AuxiliaryStream::res_secs).collect::<Vec<_>>(),
            vec![3_600, 86_400],
            "streams must come back in the order requested"
        );

        // Every auxiliary corpus must actually hold bars. This is the assertion that would have
        // caught the `DEFAULT_MIN_BARS` bug: a 20,480-bar floor rejects every daily file and the
        // run comes up green having loaded nothing.
        for stream in &streams {
            assert_eq!(
                stream.corpus().symbols().len(),
                3,
                "the {}s corpus lost symbols to the eligibility floor",
                stream.res_secs()
            );
            assert!(
                stream.covered_bars() > 0,
                "the {}s pass covers no bar",
                stream.res_secs()
            );
            assert!(
                stream.steps_per_epoch() > 0,
                "the {}s stream would never fire",
                stream.res_secs()
            );
        }

        let mut report = AuxiliaryReport::new(&streams);
        assert!(!report.is_empty());

        // One pass, driven exactly as the trainer drives it: the Bresenham cadence over a
        // synthetic primary pass, then the epoch roll.
        let primary_steps = 64usize;
        let mut drawn_per_stream = vec![0usize; streams.len()];
        for step in 0..primary_steps {
            for (index, stream) in streams.iter_mut().enumerate() {
                if !stream.fires_after(step, primary_steps) {
                    continue;
                }
                let Some((stage, sample, drawn)) = stream.draw(Device::Cpu) else {
                    continue;
                };
                assert_eq!(
                    sample.dof.size()[1],
                    stream.context(stage) + 1,
                    "a draw must carry context + 1 bars"
                );
                assert!(drawn > 0);
                // A real, resolution-dependent number rather than a constant: the marginal NLL
                // of this resolution's OWN fitted supports. Two resolutions cannot produce the
                // same value, so a writer that mixed up its rows cannot pass.
                let nll = stream.supports().marginal_nll_bar(BarScoring::Density);
                assert!(nll.is_finite());
                stream.record_step(nll);
                drawn_per_stream[index] += drawn;
            }
        }
        for (index, stream) in streams.iter_mut().enumerate() {
            assert!(
                drawn_per_stream[index] > 0,
                "the {}s stream never fired across {primary_steps} primary steps",
                stream.res_secs()
            );
            let train = stream.take_epoch_nll();
            assert!(train.is_finite(), "the epoch mean must be measured");
            report.record(index, train, train - 0.25);
        }

        report.write_report(&fx.dir).expect("report writes");
        let path = fx
            .dir
            .join(format!("{AUXILIARY_REPORT_BASE}.report.bin"));
        assert!(path.exists(), "{AUXILIARY_REPORT_BASE} was never written");
        let read = read_report(&path).expect("report reads back");
        let series = match read.kind {
            shared::report::ReportKind::MultiLine { series } => series,
            other => panic!("unexpected kind {other:?}"),
        };
        assert_eq!(
            series.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(),
            vec![
                "train@3600s",
                "held-out@3600s",
                "train@86400s",
                "held-out@86400s"
            ],
            "the point of this base is reading each resolution's curve side by side"
        );
        for s in &series {
            assert_eq!(s.values.len(), 1, "one point per pass");
            assert!(
                s.values.iter().all(|v| v.is_finite()),
                "{} holds a non-finite value",
                s.label
            );
        }
        assert_ne!(
            series[0].values[0], series[2].values[0],
            "the two resolutions' training curves are identical, so the rows are not really \
             per-resolution"
        );
        assert!(
            read.title.contains("NEVER a promotion criterion"),
            "the title must state what the curve is not: {}",
            read.title
        );
    }

    /// The specific bug that would make a multi-resolution run look like it worked while
    /// loading nothing: a floor sized for five-minute bars applied to daily files.
    #[test]
    fn the_auxiliary_floor_admits_daily_files_the_deployment_floor_rejects() {
        let (fx, _deployment) = fixture("floor");
        let admitted = eligible_bar_universe(&fx.dir, 86_400, AUXILIARY_MIN_BARS);
        assert_eq!(
            admitted.len(),
            3,
            "the auxiliary floor must admit every daily symbol"
        );
        assert!(
            eligible_bar_universe(&fx.dir, 86_400, crate::torch::dataset::DEFAULT_MIN_BARS)
                .is_empty(),
            "the deployment floor is expected to reject every daily file; if it stopped doing \
             so, AUXILIARY_MIN_BARS is no longer load-bearing and this test is the record of it"
        );
    }

    /// PROOF, against the real corpus, that `--auxiliary-resolutions 86400` loads bars.
    ///
    /// `#[ignore]`d because it reads `long_data/bars` — 5,297 intraday files and 4,748 daily
    /// ones — and FITS `bar_supports.86400.json` if it is absent, which is minutes of CPU. It is
    /// the only thing that can answer "does the daily corpus actually enter training", because
    /// every cheap synthetic fixture answers a question about the fixture. Run it with
    /// `cargo test -p trading_bot_0 --lib the_real_daily_corpus -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads the real long_data/bars corpus and fits bar_supports.86400.json"]
    fn the_real_daily_corpus_enters_training_and_no_held_out_bar_does() {
        use crate::data::ingest::{bars_dir, PINNED_SPLIT_BOUNDS};
        use crate::torch::dataset::DEFAULT_MIN_BARS;
        use crate::torch::train::pretrain::{fit_supports_at, SupportsFit};

        let dir = bars_dir();
        let deployment =
            BarCorpus::load_with_bounds(&dir, 300, DEFAULT_MIN_BARS, PINNED_SPLIT_BOUNDS)
                .expect("the 300s deployment corpus loads");
        println!(
            "[300s] {} symbols, {} bars, split {} | {} | {}",
            deployment.symbols().len(),
            deployment.unique_bars(),
            deployment.split_bars(Split::Train),
            deployment.split_bars(Split::Val),
            deployment.split_bars(Split::Test),
        );

        let fit = SupportsFit {
            samples: 4_000_000,
            seed: 0x5EED,
            freeze: false,
        };
        let streams = AuxiliaryStream::open(
            &deployment,
            &AuxiliaryConfig {
                resolutions: &[86_400],
                base_batch: 24,
                batch_ramp: &[1, 1, 1],
                seed: 0x5EED,
                scoring: BarScoring::Density,
                device: Device::Cpu,
            },
            |corpus, fingerprint| {
                let path = corpus.supports_path();
                Ok(fit_supports_at(corpus, &path, fit, fingerprint)?.0)
            },
        )
        .expect("the daily auxiliary corpus opens");
        let daily = &streams[0];
        let corpus = daily.corpus();
        let (train, val, test) = (
            corpus.split_bars(Split::Train) as u64,
            corpus.split_bars(Split::Val) as u64,
            corpus.split_bars(Split::Test) as u64,
        );
        println!(
            "[86400s] {} symbols, {} bars, split {train} | {val} | {test}, pass covers {} \
             bars in {} steps/epoch",
            corpus.symbols().len(),
            corpus.unique_bars(),
            daily.covered_bars(),
            daily.steps_per_epoch(),
        );
        println!("[86400s] {}", corpus.supports_path().display());

        // The defect this whole wiring exists to remove: a daily corpus on disk that trains
        // nothing. A floor sized for five-minute bars makes this zero.
        assert!(
            corpus.symbols().len() > 4_000,
            "only {} daily symbols were admitted; the auxiliary floor is rejecting the corpus",
            corpus.symbols().len()
        );
        assert!(
            daily.covered_bars() > 15_000_000,
            "the daily pass covers only {} bars",
            daily.covered_bars()
        );
        assert!(daily.steps_per_epoch() > 0);

        // And the leak it must not introduce. `PINNED_SPLIT_BOUNDS` are 2025/2026 instants and
        // the daily corpus runs to 2026-08-14, so a daily bar CAN land inside the window the
        // deployment resolution is scored on. Those bars exist and are excluded: the pass is
        // built on `Split::Train` alone, so `covered_bars + remainder == train`, never `train +
        // val + test`.
        assert!(
            val + test > 0,
            "expected some daily bars inside the held-out window; if there are none this \
             assertion is no longer testing the exclusion"
        );
        assert_eq!(
            daily.covered_bars() + daily.pass_remainder_total(),
            train,
            "the daily pass must account for exactly the TRAIN bars: {} covered + {} remainder \
             against {train} train, {val} val and {test} test",
            daily.covered_bars(),
            daily.pass_remainder_total()
        );
        println!(
            "[86400s] {} val + {} test daily bars ({:.2}% of {}) are EXCLUDED from training",
            val,
            test,
            100.0 * (val + test) as f64 / corpus.unique_bars() as f64,
            corpus.unique_bars()
        );
    }
}
