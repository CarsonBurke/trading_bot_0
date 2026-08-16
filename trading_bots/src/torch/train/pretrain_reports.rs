//! Report emission for discrete distributional bar pretraining.
//!
//! Every metric the pretrainer produces leaves the process through this module
//! and through `shared::report::write_report` only; there is no second channel.
//! The base names written here are mirrored in `meta_chart_bases` in
//! `tui/src/main.rs`, which is what makes them visible.
//!
//! All scalar curves share one x-axis, the *record tick*. [`PretrainReporter::record_step`]
//! mean-aggregates [`STEP_DECIMATION`] optimizer steps into one tick, and
//! [`PretrainReporter::record_epoch`] commits a tick of its own carrying the
//! validation numbers. Series that only exist on one of the two paths are
//! NaN-padded on the other, which the report renderer already filters. The
//! practical effect is a dense training curve with validation markers
//! interleaved, instead of a chart with one point per epoch.
//!
//! Two validation contexts are reported separately and must not be conflated.
//! `pretrain_nll_bar` carries the *promotion* metric, measured at the full
//! context and therefore only defined once the context ramp has finished.
//! `pretrain_nll_bar_diag896` and every per-DOF, calibration and diagnostic
//! series carry the *fixed 896-context* evaluation, which is configured
//! identically in every run and is the only curve comparable across ablations.

use std::array;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ring::digest::{Context as DigestContext, SHA256};
use shared::report::{write_report, CandleBar, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{
    decode_dof, BarDof, BarScoring, BarSupports, BAR_DOF, BAR_DOF_NAMES, DOF_U, DOF_V,
};

/// Resolution of the per-DOF PIT histogram.
pub const PIT_HIST_BINS: usize = 16;
/// Rollout horizons, in bars, reported by `pretrain_rollout_nll`.
pub const ROLLOUT_HORIZONS: [usize; 4] = [1, 4, 16, 64];
/// Context the comparable diagnostic evaluation is pinned to, for axis labels.
pub const DIAGNOSTIC_CONTEXT: i64 = 896;
/// Recommended ancestral sample count per snapshot window.
pub const SNAPSHOT_SAMPLES: usize = 256;
/// Optimizer steps folded into one record tick.
const STEP_DECIMATION: usize = 20;
/// Record ticks between report flushes on the step path. Epoch and snapshot
/// records always flush.
const FLUSH_EVERY_TICKS: usize = 5;
const BAND_LOW: f64 = 0.10;
const BAND_HIGH: f64 = 0.90;
/// Share of the objective's total magnitude at which an AUXILIARY term is considered to be
/// competing with the likelihood rather than shaping the latent.
///
/// Drawn on `pretrain_loss_shares` and enforced by the pretrainer's consecutive-step
/// warning. Not a hard cap: the right response to a term crossing it is a decision about
/// `--lambda-dyn`, not a silent clamp that would hide the miscalibration.
pub const AUX_SHARE_WARN: f64 = 0.25;
/// Consecutive steps an auxiliary term may sit above [`AUX_SHARE_WARN`] before the
/// pretrainer warns. One step is minibatch noise; a hundred is the objective.
pub const AUX_SHARE_WARN_STREAK: usize = 100;
/// Fraction of realized closes a calibrated 10/90 band should contain.
const NOMINAL_COVERAGE: f64 = BAND_HIGH - BAND_LOW;

/// Nats per bar of the uniform CATEGORICAL chain, `BAR_DOF * ln(NUM_BAR_BINS)`.
///
/// This is the discrete part only. Under `BarScoring::Density` a uniform head also pays the
/// measure term, so the mode-aware line a chart must use is
/// [`HeldOutBaselines::uniform_nll_bar`].
pub fn uniform_categorical_nll_bar() -> f64 {
    BarSupports::uniform_categorical_nll_bar()
}

// ---------------------------------------------------------------------------
// Caller-facing metric structs
// ---------------------------------------------------------------------------

/// Per-optimizer-step training metrics. Build with [`StepMetrics::nan`] and set
/// what is available; non-finite fields are skipped rather than plotted.
#[derive(Clone, Copy, Debug)]
pub struct StepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub nll_bar: f64,
    pub nll_dof: [f64; BAR_DOF],
    pub dyn_loss: f64,
    pub kl_loss: f64,
    pub total_loss: f64,
    /// Share of the objective's total MAGNITUDE carried by each term, i.e. the weighted
    /// term over the sum of the three weighted magnitudes. They sum to one.
    ///
    /// Magnitudes and not the signed total: under `BarScoring::Density` the likelihood term
    /// is a log density and is routinely NEGATIVE, so a signed denominator would pass
    /// through zero and make every share meaningless exactly when the objective is most
    /// worth watching.
    pub nll_share: f64,
    pub dyn_share: f64,
    pub kl_share: f64,
    /// Mean `cos(h_t, h_{t+1})` over the batch. A trunk that wins on the NextLat term by
    /// making beliefs SLOWLY VARYING — which the zero-init identity dynamics predicts
    /// perfectly — drives this to one while destroying the trajectory's temporal
    /// resolution. Diagnostic only; nothing optimizes it.
    pub belief_autocorr: f64,
    /// `dyn` divided by the TRIVIAL-IDENTITY baseline `smooth_l1(h_t, sg[h_{t+k}])`, i.e.
    /// what the term would score if the dynamics MLP returned its input unchanged. At one
    /// the MLP contributes nothing and `dyn` is measuring belief smoothness alone.
    pub dyn_vs_identity: f64,
    pub lr_mult: f64,
    pub muon_momentum: f64,
    /// Observed gradient norm. Nothing clips on it; Muon orthogonalization does
    /// that job, so this is a pure diagnostic.
    pub grad_norm: f64,
    pub context: i64,
    pub batch_size: usize,
    pub bars_seen: u64,
}

impl StepMetrics {
    pub fn nan() -> Self {
        Self {
            epoch: 0,
            step: 0,
            nll_bar: f64::NAN,
            nll_dof: [f64::NAN; BAR_DOF],
            dyn_loss: f64::NAN,
            kl_loss: f64::NAN,
            total_loss: f64::NAN,
            nll_share: f64::NAN,
            dyn_share: f64::NAN,
            kl_share: f64::NAN,
            belief_autocorr: f64::NAN,
            dyn_vs_identity: f64::NAN,
            lr_mult: f64::NAN,
            muon_momentum: f64::NAN,
            grad_norm: f64::NAN,
            context: 0,
            batch_size: 0,
            bars_seen: 0,
        }
    }
}

/// Per-validation metrics. Build with [`EpochMetrics::nan`] and set what is
/// available.
///
/// Everything except `val_nll_bar` / `best_val_nll_bar` comes from the fixed
/// [`DIAGNOSTIC_CONTEXT`] evaluation, so it stays comparable across runs.
#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub global_step: usize,
    pub train_nll_bar: f64,
    /// Promotion metric at the full context; NaN before the ramp completes.
    pub val_nll_bar: f64,
    pub best_val_nll_bar: f64,
    /// Path the promoted checkpoint was written to, or `None` if this validation
    /// did not promote. The reporter fingerprints the artifact here so the
    /// end-of-run test battery can prove it scored that exact file.
    pub promoted_checkpoint: Option<PathBuf>,
    /// Across-run diagnostic at the fixed [`DIAGNOSTIC_CONTEXT`].
    pub val_nll_bar_diag: f64,
    pub train_nll_dof: [f64; BAR_DOF],
    pub val_nll_dof: [f64; BAR_DOF],
    pub val_crps_dof: [f64; BAR_DOF],
    pub val_pit: PitHistogram,
    /// Sign accuracy of `E[r]` against the realized return, validation only.
    pub val_dir_acc: f64,
    /// Teacher-forced rollout NLL at [`ROLLOUT_HORIZONS`] with beliefs advanced
    /// by the trunk. NaN for horizons the evaluation window cannot reach.
    pub rollout_nll_exact: [f64; ROLLOUT_HORIZONS.len()],
    /// The same horizons with beliefs advanced by `BarDynamics`. The gap against
    /// the exact series is what the dynamics KL term is there to close.
    pub rollout_nll_dynamics: [f64; ROLLOUT_HORIZONS.len()],
    /// Bars consumed divided by the corpus unique-bar count.
    pub unique_bar_reuse: f64,
    /// Participation ratio of the belief covariance; see [`belief_effective_rank`].
    pub effective_rank: f64,
    /// Block-bootstrap standard error of `val_nll_bar`, resampling by
    /// `(symbol, calendar month)`. NaN when the promotion set was not evaluated.
    pub val_nll_bar_se: f64,
    /// 95% block-bootstrap interval of `val_nll_bar`, charted as a band.
    pub val_nll_bar_ci: (f64, f64),
    /// The same standard error resampling CALENDAR MONTHS, i.e. treating the shared
    /// wall-clock slots as the resampling unit. This is the honest error bar on the
    /// absolute level and runs ~4x the `(symbol, month)` figure, because all 4096 windows
    /// sit in a handful of shared months and a regime shift moves them together.
    pub val_nll_bar_se_level: f64,
    /// `val_nll_bar` with the encoding tautology excluded: `u` and `v` are scored only on
    /// bars with `s != 0`, where the encoding does not already determine them.
    pub val_nll_bar_conditional: f64,
    pub val_nll_dof_conditional: [f64; BAR_DOF],
    /// Fraction of each ramp stage's anchor list actually issued so far. One entry per
    /// stage; empty when the caller does not track it. Each stage owns its own stride-C
    /// anchor list and restarts at index 0, and the bar budget splits unevenly across the
    /// ramp, so an early stage sees far less than one pass over its list.
    pub stage_coverage: Vec<f64>,
    /// Per-DOF split of the diagnostic NLL into the degeneracy class and the continuous
    /// shape. A head that only learned which bars are degenerate posts its whole gain in
    /// `class`, which the undivided number cannot distinguish from intra-bar skill.
    pub val_nll_dof_class: [f64; BAR_DOF],
    pub val_nll_dof_shape: [f64; BAR_DOF],
}

impl EpochMetrics {
    pub fn nan() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            train_nll_bar: f64::NAN,
            val_nll_bar: f64::NAN,
            best_val_nll_bar: f64::NAN,
            promoted_checkpoint: None,
            val_nll_bar_diag: f64::NAN,
            train_nll_dof: [f64::NAN; BAR_DOF],
            val_nll_dof: [f64::NAN; BAR_DOF],
            val_crps_dof: [f64::NAN; BAR_DOF],
            val_pit: PitHistogram::default(),
            val_dir_acc: f64::NAN,
            rollout_nll_exact: [f64::NAN; ROLLOUT_HORIZONS.len()],
            rollout_nll_dynamics: [f64::NAN; ROLLOUT_HORIZONS.len()],
            unique_bar_reuse: f64::NAN,
            effective_rank: f64::NAN,
            val_nll_bar_se: f64::NAN,
            val_nll_bar_ci: (f64::NAN, f64::NAN),
            val_nll_bar_se_level: f64::NAN,
            val_nll_bar_conditional: f64::NAN,
            val_nll_dof_class: [f64::NAN; BAR_DOF],
            val_nll_dof_shape: [f64::NAN; BAR_DOF],
            val_nll_dof_conditional: [f64::NAN; BAR_DOF],
            stage_coverage: Vec::new(),
        }
    }
}

/// End-of-run held-out battery, emitted exactly once as `pretrain_test`.
///
/// The validation split drives promotion, so across an ablation campaign it
/// stops being an unbiased estimate of generalization: we select against it
/// repeatedly and it drifts optimistic. This battery is the split that is
/// touched once, after the last promotion decision, and never feeds back into
/// any decision. [`PretrainReporter::finish`] enforces both properties; see its
/// documentation for exactly what is checked and what is not.
#[derive(Clone, Debug)]
pub struct TestBattery {
    /// The checkpoint passed to `BarWorldModel::load`. Must be the promoted one.
    pub checkpoint: PathBuf,
    /// `BarWorldModel::lineage_sha256()` of the reloaded model. Evidence that the
    /// numbers came from an artifact read back off disk rather than from the
    /// in-memory training model.
    pub model_lineage: String,
    pub nll_bar: f64,
    pub nll_dof: [f64; BAR_DOF],
    pub crps_dof: [f64; BAR_DOF],
    pub rollout_nll_exact: [f64; ROLLOUT_HORIZONS.len()],
    pub rollout_nll_dynamics: [f64; ROLLOUT_HORIZONS.len()],
    pub pit: PitHistogram,
    /// Sign accuracy of `E[r]` against the realized return.
    pub dir_acc: f64,
    /// `BarCorpus::identity_fingerprint()` of the corpus this battery scored. The corpus is
    /// live and the split instants are percentiles of it, so without this two batteries a
    /// week apart are not the same measurement and nothing says so.
    pub corpus_fingerprint: String,
    /// `(train|val, val|test)` instants, in epoch millis.
    pub split_bounds: (i64, i64),
    /// `nll_bar` with the `s == 0 => u = v = 0.5` encoding tautology excluded.
    pub nll_bar_conditional: f64,
    pub nll_dof_conditional: [f64; BAR_DOF],
    /// Block-bootstrap standard error and 95% interval of `nll_bar`.
    pub nll_bar_se: f64,
    pub nll_bar_ci: (f64, f64),
}

impl TestBattery {
    pub fn nan(checkpoint: PathBuf, model_lineage: String) -> Self {
        Self {
            checkpoint,
            model_lineage,
            nll_bar: f64::NAN,
            nll_dof: [f64::NAN; BAR_DOF],
            crps_dof: [f64::NAN; BAR_DOF],
            rollout_nll_exact: [f64::NAN; ROLLOUT_HORIZONS.len()],
            rollout_nll_dynamics: [f64::NAN; ROLLOUT_HORIZONS.len()],
            pit: PitHistogram::default(),
            dir_acc: f64::NAN,
            corpus_fingerprint: String::new(),
            split_bounds: (0, 0),
            nll_bar_conditional: f64::NAN,
            nll_dof_conditional: [f64::NAN; BAR_DOF],
            nll_bar_se: f64::NAN,
            nll_bar_ci: (f64::NAN, f64::NAN),
        }
    }
}

/// Reference lines that need more than the fitted supports to compute, plus the scoring
/// rule they are all expressed in.
///
/// Two of them exist because the headline "X nats better than the calibrated marginal"
/// claim was, until now, comparing a held-out number against a TRAIN-fitted baseline that
/// also credits an arithmetic identity of the encoding as skill. Every line here is
/// recomputed per scoring mode, so a chart can never draw a `smoothed` yardstick under a
/// `density` curve.
#[derive(Clone, Copy, Debug)]
pub struct HeldOutBaselines {
    /// Scoring rule every figure in this struct, and every `nll` series it is drawn against,
    /// is measured under.
    pub scoring: BarScoring,
    /// Nats/bar a UNIFORM-over-bins head pays under `scoring`, which is where the
    /// zero-initialized emission head starts and the zero of the gain-vs-baselines chart.
    pub uniform_nll_bar: f64,
    /// Per-DOF marginal entropy with the `s == 0 => u = v = 0.5` identity removed: the `u`
    /// and `v` references are conditioned on a non-flat bar. This is the yardstick for
    /// `val_nll_dof_conditional`.
    pub marginal_nll_dof_conditional: [f64; BAR_DOF],
    /// The TRAIN-fitted `q*` scored as a fixed prediction against the pinned VALIDATION
    /// windows, per DOF. For `r` and `w` the equal-mass binning makes `q*` nearly uniform
    /// and this barely moves; for `s`, `u` and `v` the entire advantage over uniform is four
    /// measured point masses, and those are liquidity statistics that shift with the regime.
    /// The gap against the train figure IS the distribution shift.
    pub marginal_nll_dof_val: [f64; BAR_DOF],
    /// Nats/bar any model collects for free from `s == 0 => u = v = 0.5`. Roughly 0.690 on
    /// the live supports, i.e. ~19% of the gain a trained model currently reports over the
    /// calibrated marginal.
    pub encoding_identity_nats: f64,
    /// Nats/bar of the scoring rule's unreachable floor, so the reachable range is
    /// `marginal - floor`, not `marginal`. Exactly zero for `hard` and `density`, which
    /// score the bin the observation actually landed in; the label-smoothing entropy for
    /// `smoothed`.
    pub scoring_floor_bar: f64,
    /// Per-DOF split of the MARGINAL into the atom-vs-continuous indicator and the
    /// intra-continuous shape: the reference the measured `class` / `shape` curves are read
    /// against.
    pub marginal_class_dof: [f64; BAR_DOF],
    pub marginal_shape_dof: [f64; BAR_DOF],
}

impl HeldOutBaselines {
    pub fn nan() -> Self {
        Self {
            scoring: BarScoring::default(),
            uniform_nll_bar: f64::NAN,
            marginal_nll_dof_conditional: [f64::NAN; BAR_DOF],
            marginal_nll_dof_val: [f64::NAN; BAR_DOF],
            encoding_identity_nats: f64::NAN,
            scoring_floor_bar: f64::NAN,
            marginal_class_dof: [f64::NAN; BAR_DOF],
            marginal_shape_dof: [f64::NAN; BAR_DOF],
        }
    }

    fn sum(values: &[f64; BAR_DOF]) -> f64 {
        if values.iter().all(|v| v.is_finite()) {
            values.iter().sum()
        } else {
            f64::NAN
        }
    }

    /// Nats/bar of the conditional reference.
    pub fn marginal_nll_bar_conditional(&self) -> f64 {
        Self::sum(&self.marginal_nll_dof_conditional)
    }

    /// Nats/bar of the train-fitted marginal measured on the held-out windows.
    pub fn marginal_nll_bar_val(&self) -> f64 {
        Self::sum(&self.marginal_nll_dof_val)
    }
}

/// Ancestral samples for the candle snapshot pictures.
///
/// The reporter never invokes the model: the caller draws the samples with
/// `BarWorldModel::rollout` and hands the tensor over, which keeps report
/// emission independent of the model surface and directly testable.
pub struct SnapshotInput<'a> {
    /// `[W, samples, H, BAR_DOF]` ancestral rollout of the pinned windows.
    pub rollout: &'a Tensor,
    /// `[W, H, BAR_DOF]` realized continuation of the same windows.
    pub future_dof: &'a Tensor,
    pub epoch: usize,
    pub global_step: usize,
}

// ---------------------------------------------------------------------------
// PIT histogram
// ---------------------------------------------------------------------------

/// Per-DOF histogram of the probability integral transform. A calibrated
/// predictive gives a uniform PIT, so a flat histogram is the target shape.
#[derive(Clone, Debug, Default)]
pub struct PitHistogram {
    counts: [[u64; PIT_HIST_BINS]; BAR_DOF],
}

impl PitHistogram {
    /// Fold a `[..., BAR_DOF]` PIT tensor, as produced by
    /// `bar_dist::bar_pit_from_logits(logits, target_dof, supports, seed)`, into
    /// the running counts. Pass the run seed there so the atom-randomized PIT is
    /// reproducible and these histograms stay diffable across ablations.
    pub fn accumulate(&mut self, pit: &Tensor) {
        let flat = pit
            .detach()
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .reshape([-1, BAR_DOF as i64])
            .contiguous();
        let numel = flat.numel();
        let mut values = vec![0.0f32; numel];
        flat.copy_data(&mut values, numel);
        for row in values.chunks_exact(BAR_DOF) {
            for (dof, &p) in row.iter().enumerate() {
                if !p.is_finite() {
                    continue;
                }
                let bin = ((p as f64) * PIT_HIST_BINS as f64).floor() as isize;
                let bin = bin.clamp(0, PIT_HIST_BINS as isize - 1) as usize;
                self.counts[dof][bin] += 1;
            }
        }
    }

    pub fn reset(&mut self) {
        self.counts = [[0; PIT_HIST_BINS]; BAR_DOF];
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|dof| dof.iter().all(|&c| c == 0))
    }

    /// Bin densities scaled so a perfectly calibrated DOF sits flat at `1.0`.
    pub fn density(&self) -> [[f64; PIT_HIST_BINS]; BAR_DOF] {
        array::from_fn(|dof| {
            let total: u64 = self.counts[dof].iter().sum();
            array::from_fn(|bin| {
                if total == 0 {
                    f64::NAN
                } else {
                    self.counts[dof][bin] as f64 * PIT_HIST_BINS as f64 / total as f64
                }
            })
        })
    }

    /// Per-DOF total variation distance from the uniform PIT, in `[0, 1]`.
    /// Zero is perfect calibration; it is the scalar summary the end-of-run test
    /// battery reports instead of a whole histogram.
    pub fn total_variation(&self) -> [f64; BAR_DOF] {
        let density = self.density();
        array::from_fn(|dof| {
            if density[dof].iter().any(|v| !v.is_finite()) {
                return f64::NAN;
            }
            0.5 * density[dof]
                .iter()
                .map(|v| (v / PIT_HIST_BINS as f64 - 1.0 / PIT_HIST_BINS as f64).abs())
                .sum::<f64>()
        })
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Participation ratio of the belief covariance, `(sum lambda)^2 / sum lambda^2`.
///
/// For a symmetric covariance that equals `trace(C)^2 / ||C||_F^2`, so no
/// eigendecomposition is needed. Ranges from `1` (rank-one collapse) to the
/// belief width (isotropic). Purely a diagnostic: nothing optimizes it.
pub fn belief_effective_rank(beliefs: &Tensor) -> f64 {
    tch::no_grad(|| {
        let dim = match beliefs.size().last() {
            Some(&d) if d > 0 => d,
            _ => return f64::NAN,
        };
        let flat = beliefs
            .detach()
            .to_kind(Kind::Float)
            .reshape([-1, dim])
            .contiguous();
        let rows = flat.size()[0];
        if rows < 2 {
            return f64::NAN;
        }
        let centered = &flat - flat.mean_dim([0i64].as_slice(), true, Kind::Float);
        let cov = centered.transpose(0, 1).matmul(&centered) / (rows - 1) as f64;
        let trace = cov.diagonal(0, 0, 1).sum(Kind::Float).double_value(&[]);
        let frobenius = cov.pow_tensor_scalar(2.0).sum(Kind::Float).double_value(&[]);
        if !trace.is_finite() || !frobenius.is_finite() || frobenius <= 0.0 {
            f64::NAN
        } else {
            trace * trace / frobenius
        }
    })
}

// ---------------------------------------------------------------------------
// Series plumbing
// ---------------------------------------------------------------------------

/// A sparse curve on the shared record-tick axis. Ticks never written stay NaN.
#[derive(Clone, Debug, Default)]
struct Series(Vec<f32>);

impl Series {
    fn set(&mut self, tick: usize, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.0.len() <= tick {
            self.0.resize(tick + 1, f32::NAN);
        }
        self.0[tick] = value as f32;
    }

    fn padded(&self, len: usize) -> Vec<f32> {
        let mut values = self.0.clone();
        values.resize(len, f32::NAN);
        values
    }

    fn labeled(&self, label: &str, len: usize) -> ReportSeries {
        ReportSeries {
            label: label.to_owned(),
            values: self.padded(len),
        }
    }
}

/// Running mean that ignores non-finite contributions.
#[derive(Clone, Copy, Debug, Default)]
struct Mean {
    sum: f64,
    count: usize,
}

impl Mean {
    fn push(&mut self, value: f64) {
        if value.is_finite() {
            self.sum += value;
            self.count += 1;
        }
    }

    fn value(self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StepAccumulator {
    steps: usize,
    nll_bar: Mean,
    nll_dof: [Mean; BAR_DOF],
    dyn_loss: Mean,
    kl_loss: Mean,
    total_loss: Mean,
    nll_share: Mean,
    dyn_share: Mean,
    kl_share: Mean,
    belief_autocorr: Mean,
    dyn_vs_identity: Mean,
    lr_mult: Mean,
    muon_momentum: Mean,
    grad_norm: Mean,
    context: Mean,
    batch_size: Mean,
    bars_seen: u64,
}

impl StepAccumulator {
    fn push(&mut self, step: &StepMetrics) {
        self.steps += 1;
        self.nll_bar.push(step.nll_bar);
        for (slot, &value) in self.nll_dof.iter_mut().zip(step.nll_dof.iter()) {
            slot.push(value);
        }
        self.dyn_loss.push(step.dyn_loss);
        self.kl_loss.push(step.kl_loss);
        self.total_loss.push(step.total_loss);
        self.nll_share.push(step.nll_share);
        self.dyn_share.push(step.dyn_share);
        self.kl_share.push(step.kl_share);
        self.belief_autocorr.push(step.belief_autocorr);
        self.dyn_vs_identity.push(step.dyn_vs_identity);
        self.lr_mult.push(step.lr_mult);
        self.muon_momentum.push(step.muon_momentum);
        self.grad_norm.push(step.grad_norm);
        self.context.push(step.context as f64);
        self.batch_size.push(step.batch_size as f64);
        self.bars_seen = self.bars_seen.max(step.bars_seen);
    }
}

// ---------------------------------------------------------------------------
// Reporter
// ---------------------------------------------------------------------------

/// Accumulates every pretraining metric and writes the `.report.bin` set.
pub struct PretrainReporter {
    gens_dir: PathBuf,
    tick: usize,
    ticks_since_flush: usize,
    epoch: usize,
    global_step: usize,
    accumulator: StepAccumulator,
    promotions: usize,
    /// Canonical path and SHA-256 of the most recently promoted checkpoint; the
    /// end-of-run test battery is checked against this.
    promoted_checkpoint: Option<(PathBuf, [u8; 32])>,
    /// Per-DOF entropy of the fitted marginals, the honest yardstick. A head that
    /// only learned the unconditional marginals sits exactly here; beating it is
    /// the first evidence of anything conditional. NaN when the caller has not
    /// supplied it.
    marginal_nll_dof: [f64; BAR_DOF],
    marginal_nll_bar: f64,

    nll_bar_train: Series,
    nll_bar_val: Series,
    nll_bar_best: Series,
    nll_bar_diag: Series,
    nll_dof_train: [Series; BAR_DOF],
    nll_dof_val: [Series; BAR_DOF],
    vs_uniform_train: Series,
    vs_uniform_val: Series,
    vs_uniform_diag: Series,
    marginal_dof: [Series; BAR_DOF],
    crps_dof: [Series; BAR_DOF],
    dyn_loss: Series,
    kl_loss: Series,
    total_loss: Series,
    nll_share: Series,
    dyn_share: Series,
    kl_share: Series,
    belief_autocorr: Series,
    dyn_vs_identity: Series,
    rollout_exact: [Series; ROLLOUT_HORIZONS.len()],
    rollout_dynamics: [Series; ROLLOUT_HORIZONS.len()],
    dir_acc: Series,
    lr: Series,
    muon_momentum: Series,
    grad_norm: Series,
    unique_bar_reuse: Series,
    effective_rank: Series,
    promotion_trace: Series,
    context: Series,
    batch_size: Series,
    bars_seen: Series,

    pit: Option<[[f64; PIT_HIST_BINS]; BAR_DOF]>,

    candle_mse: Vec<f32>,
    candle_dclose: Vec<f32>,
    candle_band: Vec<f32>,
    candle_coverage: Vec<f32>,

    /// Reference lines that need the corpus and the pinned val set, not just the supports.
    /// See [`HeldOutBaselines`]; all NaN until the caller supplies them.
    baselines: HeldOutBaselines,

    nll_bar_ci_low: Series,
    nll_bar_ci_high: Series,
    nll_bar_conditional: Series,
    nll_dof_conditional: [Series; BAR_DOF],
    nll_dof_class: [Series; BAR_DOF],
    nll_dof_shape: [Series; BAR_DOF],
    /// One coverage curve per ramp stage, grown on demand: the reporter does not know how
    /// many stages the schedule has.
    stage_coverage: Vec<Series>,
}

impl PretrainReporter {
    /// `gens_dir` is the run's `gens` directory; reports land in `gens_dir/<epoch>`,
    /// which is where the TUI scans for them.
    ///
    /// `marginal_nll_dof` is the per-DOF entropy of the fitted supports, i.e. what
    /// a perfectly calibrated marginal head scores. Pass `[f64::NAN; BAR_DOF]` if
    /// it is not available and the marginal reference lines are simply omitted.
    pub fn new(gens_dir: &Path, marginal_nll_dof: [f64; BAR_DOF]) -> Self {
        let marginal_nll_bar = if marginal_nll_dof.iter().all(|v| v.is_finite()) {
            marginal_nll_dof.iter().sum()
        } else {
            f64::NAN
        };
        Self {
            gens_dir: gens_dir.to_path_buf(),
            tick: 0,
            ticks_since_flush: 0,
            epoch: 0,
            global_step: 0,
            accumulator: StepAccumulator::default(),
            promotions: 0,
            promoted_checkpoint: None,
            marginal_nll_dof,
            marginal_nll_bar,
            nll_bar_train: Series::default(),
            nll_bar_val: Series::default(),
            nll_bar_best: Series::default(),
            nll_bar_diag: Series::default(),
            nll_dof_train: array::from_fn(|_| Series::default()),
            nll_dof_val: array::from_fn(|_| Series::default()),
            vs_uniform_train: Series::default(),
            vs_uniform_val: Series::default(),
            vs_uniform_diag: Series::default(),
            marginal_dof: array::from_fn(|_| Series::default()),
            crps_dof: array::from_fn(|_| Series::default()),
            dyn_loss: Series::default(),
            kl_loss: Series::default(),
            total_loss: Series::default(),
            nll_share: Series::default(),
            dyn_share: Series::default(),
            kl_share: Series::default(),
            belief_autocorr: Series::default(),
            dyn_vs_identity: Series::default(),
            rollout_exact: array::from_fn(|_| Series::default()),
            rollout_dynamics: array::from_fn(|_| Series::default()),
            dir_acc: Series::default(),
            lr: Series::default(),
            muon_momentum: Series::default(),
            grad_norm: Series::default(),
            unique_bar_reuse: Series::default(),
            effective_rank: Series::default(),
            promotion_trace: Series::default(),
            context: Series::default(),
            batch_size: Series::default(),
            bars_seen: Series::default(),
            pit: None,
            candle_mse: Vec::new(),
            candle_dclose: Vec::new(),
            candle_band: Vec::new(),
            candle_coverage: Vec::new(),
            baselines: HeldOutBaselines::nan(),
            nll_bar_ci_low: Series::default(),
            nll_bar_ci_high: Series::default(),
            nll_bar_conditional: Series::default(),
            nll_dof_conditional: array::from_fn(|_| Series::default()),
            nll_dof_class: array::from_fn(|_| Series::default()),
            nll_dof_shape: array::from_fn(|_| Series::default()),
            stage_coverage: Vec::new(),
        }
    }

    /// Supply the reference lines that only exist once the corpus and the pinned validation
    /// set are known. Call once, before the first [`Self::record_epoch`].
    pub fn set_held_out_baselines(&mut self, baselines: HeldOutBaselines) {
        self.baselines = baselines;
    }

    /// Fold one optimizer step in. Every [`STEP_DECIMATION`] steps this commits a
    /// record tick, and every [`FLUSH_EVERY_TICKS`] ticks it rewrites the charts.
    pub fn record_step(&mut self, step: &StepMetrics) -> Result<()> {
        self.epoch = step.epoch;
        self.global_step = step.step;
        self.accumulator.push(step);
        if self.accumulator.steps < STEP_DECIMATION {
            return Ok(());
        }
        self.commit_steps();
        if self.ticks_since_flush >= FLUSH_EVERY_TICKS {
            self.flush()?;
        }
        Ok(())
    }

    /// Commit a validation tick and rewrite every chart.
    pub fn record_epoch(&mut self, metrics: &EpochMetrics) -> Result<()> {
        self.epoch = metrics.epoch;
        self.global_step = metrics.global_step;
        if self.accumulator.steps > 0 {
            self.commit_steps();
        }

        let tick = self.tick;
        let uniform = self.baselines.uniform_nll_bar;
        self.nll_bar_train.set(tick, metrics.train_nll_bar);
        self.nll_bar_val.set(tick, metrics.val_nll_bar);
        self.nll_bar_best.set(tick, metrics.best_val_nll_bar);
        self.nll_bar_diag.set(tick, metrics.val_nll_bar_diag);
        self.vs_uniform_train
            .set(tick, uniform - metrics.train_nll_bar);
        self.vs_uniform_val.set(tick, uniform - metrics.val_nll_bar);
        self.vs_uniform_diag
            .set(tick, uniform - metrics.val_nll_bar_diag);
        for dof in 0..BAR_DOF {
            self.nll_dof_train[dof].set(tick, metrics.train_nll_dof[dof]);
            self.nll_dof_val[dof].set(tick, metrics.val_nll_dof[dof]);
            self.crps_dof[dof].set(tick, metrics.val_crps_dof[dof]);
            self.marginal_dof[dof].set(tick, self.marginal_nll_dof[dof]);
        }
        self.nll_bar_conditional
            .set(tick, metrics.val_nll_bar_conditional);
        self.nll_bar_ci_low.set(tick, metrics.val_nll_bar_ci.0);
        self.nll_bar_ci_high.set(tick, metrics.val_nll_bar_ci.1);
        for dof in 0..BAR_DOF {
            self.nll_dof_conditional[dof].set(tick, metrics.val_nll_dof_conditional[dof]);
            self.nll_dof_class[dof].set(tick, metrics.val_nll_dof_class[dof]);
            self.nll_dof_shape[dof].set(tick, metrics.val_nll_dof_shape[dof]);
        }
        // Each ramp stage owns its own anchor list, so coverage is per stage and the stage
        // count is the schedule's business, not the reporter's.
        for (stage, fraction) in metrics.stage_coverage.iter().enumerate() {
            if self.stage_coverage.len() <= stage {
                self.stage_coverage.resize_with(stage + 1, Series::default);
            }
            self.stage_coverage[stage].set(tick, *fraction);
        }
        for horizon in 0..ROLLOUT_HORIZONS.len() {
            self.rollout_exact[horizon].set(tick, metrics.rollout_nll_exact[horizon]);
            self.rollout_dynamics[horizon].set(tick, metrics.rollout_nll_dynamics[horizon]);
        }
        self.dir_acc.set(tick, metrics.val_dir_acc);
        self.unique_bar_reuse.set(tick, metrics.unique_bar_reuse);
        self.effective_rank.set(tick, metrics.effective_rank);
        if let Some(checkpoint) = &metrics.promoted_checkpoint {
            // A promotion whose artifact is not on disk means the end-of-run test
            // battery would score the wrong file. Fail here, not in hour forty.
            let canonical = checkpoint.canonicalize().with_context(|| {
                format!(
                    "promoted checkpoint {} is not readable; save it before reporting the promotion",
                    checkpoint.display()
                )
            })?;
            let digest = file_digest(&canonical)?;
            self.promoted_checkpoint = Some((canonical, digest));
            self.promotions += 1;
        }
        self.promotion_trace.set(tick, self.promotions as f64);
        if !metrics.val_pit.is_empty() {
            self.pit = Some(metrics.val_pit.density());
        }
        self.advance_tick();
        self.flush()
    }

    /// Ancestral candle snapshots: one `CandleCompare` of the realized bars
    /// against the sample-wise median path per window, one close-path fan chart,
    /// and the scalar band/coverage diagnostics.
    pub fn record_snapshot(&mut self, input: &SnapshotInput<'_>) -> Result<()> {
        self.epoch = input.epoch;
        self.global_step = input.global_step;
        let steps = match input.future_dof.size().as_slice() {
            [_, horizon, dof] if *dof == BAR_DOF as i64 && *horizon > 0 => *horizon as usize,
            other => anyhow::bail!("snapshot future_dof must be [W, H, {BAR_DOF}], got {other:?}"),
        };
        let windows = input.future_dof.size()[0] as usize;

        let summary = tch::no_grad(|| {
            self.write_candle_windows(&input.rollout.detach(), input.future_dof, windows, steps)
        })?;

        self.candle_mse.push(summary.mse as f32);
        self.candle_dclose.push(summary.dclose as f32);
        self.candle_band.push(summary.band as f32);
        self.candle_coverage.push(summary.coverage as f32);
        self.flush()
    }

    /// Emit the end-of-run held-out battery as `pretrain_test` and consume the
    /// reporter.
    ///
    /// Two properties are enforced here rather than left to convention, because
    /// a test number that leaks into model selection stops being a test number.
    ///
    /// 1. **It cannot precede the final promotion decision.** Taking `self` by
    ///    value means every promotion must already have been reported: once this
    ///    returns there is no reporter left to call [`Self::record_epoch`] on, so
    ///    a later promotion is a compile error, not a review comment. The old
    ///    loop ordered the battery before the last promotion only by luck.
    /// 2. **It must score the promoted artifact, read back off disk.** The
    ///    checkpoint named by the battery must be the last one reported promoted,
    ///    and its SHA-256 must still match the fingerprint taken at promotion
    ///    time, so scoring the in-memory model, a stale `*_best.ot`, or a file
    ///    rewritten since promotion all fail loudly. `model_lineage` must be
    ///    non-empty, which only a real `BarWorldModel::load` can supply.
    pub fn finish(mut self, battery: &TestBattery) -> Result<()> {
        let (promoted, promoted_digest) = self.promoted_checkpoint.clone().context(
            "no promotion was ever reported, so there is no checkpoint the held-out battery could \
             legitimately score; report the promotion through EpochMetrics::promoted_checkpoint first",
        )?;
        if battery.model_lineage.trim().is_empty() {
            anyhow::bail!(
                "held-out battery carries no model lineage; it must be scored on a model reloaded \
                 through BarWorldModel::load, not on the in-memory training model"
            );
        }
        let scored = battery.checkpoint.canonicalize().with_context(|| {
            format!(
                "held-out battery checkpoint {} is not readable",
                battery.checkpoint.display()
            )
        })?;
        if scored != promoted {
            anyhow::bail!(
                "held-out battery scored {} but the promoted checkpoint is {}; the test split must \
                 only ever measure the artifact that was actually selected",
                scored.display(),
                promoted.display()
            );
        }
        if file_digest(&scored)? != promoted_digest {
            anyhow::bail!(
                "{} changed on disk after it was promoted; the held-out battery would not be \
                 measuring the selected weights",
                scored.display()
            );
        }

        self.flush()?;
        let dir = self.gens_dir.join(self.epoch.to_string());
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let uniform = self.baselines.uniform_nll_bar;
        let mut series = vec![
            point_series("nll_bar", battery.nll_bar),
            point_series("nll_bar vs uniform", uniform - battery.nll_bar),
            point_series(
                "nll_bar vs marginal",
                self.marginal_nll_bar - battery.nll_bar,
            ),
            point_series("uniform", uniform),
            point_series("marginal", self.marginal_nll_bar),
            point_series(
                "scoring mode (0 smoothed, 1 hard, 2 density)",
                match self.baselines.scoring {
                    BarScoring::Smoothed => 0.0,
                    BarScoring::Hard => 1.0,
                    BarScoring::Density => 2.0,
                },
            ),
        ];
        let pit_tv = battery.pit.total_variation();
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            series.push(point_series(&format!("nll {name}"), battery.nll_dof[dof]));
            series.push(point_series(&format!("crps {name}"), battery.crps_dof[dof]));
            series.push(point_series(&format!("pit tv {name}"), pit_tv[dof]));
        }
        for (i, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            series.push(point_series(
                &format!("rollout h{horizon} exact"),
                battery.rollout_nll_exact[i],
            ));
            series.push(point_series(
                &format!("rollout h{horizon} dynamics"),
                battery.rollout_nll_dynamics[i],
            ));
        }
        series.push(point_series("dir acc", battery.dir_acc));
        series.push(point_series("nll_bar se", battery.nll_bar_se));
        series.push(point_series("nll_bar ci95 low", battery.nll_bar_ci.0));
        series.push(point_series("nll_bar ci95 high", battery.nll_bar_ci.1));
        series.push(point_series(
            "nll_bar conditional",
            battery.nll_bar_conditional,
        ));
        series.push(point_series(
            "marginal conditional",
            self.baselines.marginal_nll_bar_conditional(),
        ));
        series.push(point_series(
            "encoding identity nats",
            self.baselines.encoding_identity_nats,
        ));
        for dof in [DOF_U, DOF_V] {
            series.push(point_series(
                &format!("nll {} | s!=0", BAR_DOF_NAMES[dof]),
                battery.nll_dof_conditional[dof],
            ));
        }
        // Per-DOF deltas against the marginal, stated rather than left to be subtracted by
        // eye. The aggregate hides that `w` has exactly zero headroom below uniform while
        // `u` and `v` have over a nat each.
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            series.push(point_series(
                &format!("nll {name} vs marginal"),
                self.marginal_nll_dof[dof] - battery.nll_dof[dof],
            ));
        }

        let name = scored
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| scored.display().to_string());
        let lineage: String = battery.model_lineage.chars().take(12).collect();
        let corpus: String = battery.corpus_fingerprint.chars().take(12).collect();
        write_chart(
            &dir,
            "pretrain_test",
            format!(
                "Pretrain Held-out Test Battery - {name} - lineage {lineage} - corpus {corpus} \
                 - split {}|{} - step {}",
                battery.split_bounds.0, battery.split_bounds.1, self.global_step
            ),
            "single evaluation",
            "held-out test split, scored once, never used for selection",
            ScaleKind::Linear,
            series,
        )
    }

    fn commit_steps(&mut self) {
        let tick = self.tick;
        let acc = self.accumulator;
        let nll = acc.nll_bar.value();
        self.nll_bar_train.set(tick, nll);
        self.vs_uniform_train
            .set(tick, self.baselines.uniform_nll_bar - nll);
        for dof in 0..BAR_DOF {
            self.nll_dof_train[dof].set(tick, acc.nll_dof[dof].value());
        }
        self.dyn_loss.set(tick, acc.dyn_loss.value());
        self.kl_loss.set(tick, acc.kl_loss.value());
        self.total_loss.set(tick, acc.total_loss.value());
        self.nll_share.set(tick, acc.nll_share.value());
        self.dyn_share.set(tick, acc.dyn_share.value());
        self.kl_share.set(tick, acc.kl_share.value());
        self.belief_autocorr.set(tick, acc.belief_autocorr.value());
        self.dyn_vs_identity.set(tick, acc.dyn_vs_identity.value());
        self.lr.set(tick, acc.lr_mult.value());
        self.muon_momentum.set(tick, acc.muon_momentum.value());
        self.grad_norm.set(tick, acc.grad_norm.value());
        self.context.set(tick, acc.context.value());
        self.batch_size.set(tick, acc.batch_size.value());
        if acc.bars_seen > 0 {
            self.bars_seen.set(tick, acc.bars_seen as f64 / 1.0e6);
        }
        self.accumulator = StepAccumulator::default();
        self.advance_tick();
    }

    fn advance_tick(&mut self) {
        self.tick += 1;
        self.ticks_since_flush += 1;
    }

    fn flush(&mut self) -> Result<()> {
        self.ticks_since_flush = 0;
        let dir = self.gens_dir.join(self.epoch.to_string());
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let len = self.tick;
        let epoch = self.epoch;
        let step = self.global_step;
        // Every nats axis below is in the units of the scoring rule in force, and the three
        // rules differ by tens of nats. The mode belongs in the title of every chart, not
        // only in the banner of a log nobody opens next to the picture.
        let suffix = format!(
            "epoch {epoch} step {step} - scoring {}",
            self.baselines.scoring
        );
        let diag = DIAGNOSTIC_CONTEXT;

        write_chart(
            &dir,
            "pretrain_nll_bar",
            format!("Pretrain Bar NLL - {suffix}"),
            "record",
            "nats/bar (val = promotion metric, full context; band = 95% block bootstrap)",
            ScaleKind::Linear,
            vec![
                self.nll_bar_train.labeled("train", len),
                self.nll_bar_val.labeled("val", len),
                // The band is the whole point: a val curve without one invites reading a
                // 0.05-nat wiggle as an effect when the interval is four times that wide.
                self.nll_bar_ci_low.labeled("val ci95 low", len),
                self.nll_bar_ci_high.labeled("val ci95 high", len),
                self.nll_bar_best.labeled("best val", len),
                // Excludes the s == 0 => u = v = 0.5 identity, which is ~0.69 nats of the
                // reported gain and is arithmetic rather than prediction.
                self.nll_bar_conditional.labeled("val conditional", len),
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_nll_bar_diag896",
            format!("Pretrain Bar NLL (fixed {diag} context) - {suffix}"),
            "record",
            &format!("nats/bar at a pinned {diag} context, comparable across runs"),
            ScaleKind::Linear,
            vec![self.nll_bar_diag.labeled("val diag", len)],
        )?;

        let mut nll_dof = Vec::with_capacity(3 * BAR_DOF);
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.nll_dof_train[dof].labeled(&format!("{name} train"), len));
        }
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.nll_dof_val[dof].labeled(&format!("{name} val diag"), len));
        }
        // Per-DOF marginal floors. Without them a DOF like u, whose marginal
        // entropy is far below ln(128) because 42% of bars pin the close to a bar
        // extreme, looks like it is learning when it has only found its marginal.
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.marginal_dof[dof].labeled(&format!("{name} marginal"), len));
        }
        // The `u` and `v` curves conditioned on a non-flat bar, against the matching
        // conditional floors. Without this pair, a head that has learned only the flat-bar
        // identity shows a large gain on both DOF and nothing distinguishes it from one that
        // learned intra-bar shape.
        for dof in [DOF_U, DOF_V] {
            let name = BAR_DOF_NAMES[dof];
            nll_dof.push(
                self.nll_dof_conditional[dof].labeled(&format!("{name} val | s!=0"), len),
            );
            nll_dof.push(constant_series(
                &format!("{name} marginal | s!=0"),
                self.baselines.marginal_nll_dof_conditional[dof],
                len,
            ));
        }
        // Degeneracy class vs intra-continuous shape, against the marginal's own split. A
        // head that only learned which bars are degenerate posts its whole gain in `class`.
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.nll_dof_class[dof].labeled(&format!("{name} class"), len));
            nll_dof.push(self.nll_dof_shape[dof].labeled(&format!("{name} shape"), len));
            nll_dof.push(constant_series(
                &format!("{name} class marginal"),
                self.baselines.marginal_class_dof[dof],
                len,
            ));
            nll_dof.push(constant_series(
                &format!("{name} shape marginal"),
                self.baselines.marginal_shape_dof[dof],
                len,
            ));
        }
        write_chart(
            &dir,
            "pretrain_nll_dof",
            format!("Pretrain Bar NLL per DOF - {suffix}"),
            "record",
            &format!("nats (val series at the fixed {diag} context)"),
            ScaleKind::Linear,
            nll_dof,
        )?;

        // Gain over the uniform chain, with every yardstick drawn flat. Crossing `uniform`
        // only proves the head found the unconditional marginals. `marginal` is the
        // TRAIN-fitted reference; `marginal (val)` is that same fixed prediction scored on
        // the held-out windows, and the gap between the two IS the distribution shift.
        // `marginal + encoding identity` adds the ~0.69 nats that `s == 0 => u = v = 0.5`
        // hands out for free — that, not `marginal`, is the line a claim of conditional
        // structure has to clear.
        let uniform = self.baselines.uniform_nll_bar;
        let mut baseline_series = vec![
            self.vs_uniform_train.labeled("train", len),
            self.vs_uniform_val.labeled("val", len),
            self.vs_uniform_diag.labeled("val diag", len),
            constant_series("uniform", 0.0, len),
            constant_series("marginal", uniform - self.marginal_nll_bar, len),
            constant_series(
                "marginal (val)",
                uniform - self.baselines.marginal_nll_bar_val(),
                len,
            ),
            constant_series(
                "marginal + encoding identity",
                uniform - self.marginal_nll_bar + self.baselines.encoding_identity_nats,
                len,
            ),
        ];
        // Only the smoothed rule has a floor. Under `hard` and `density` the line would sit
        // exactly on `uniform` and assert something false — that an oracle cannot reach the
        // zero of this axis — so it is omitted rather than drawn at zero.
        if self.baselines.scoring_floor_bar > 0.0 {
            baseline_series.push(constant_series(
                "smoothing floor (unreachable)",
                uniform - self.baselines.scoring_floor_bar,
                len,
            ));
        }
        write_chart(
            &dir,
            "pretrain_nll_vs_baselines",
            format!("Pretrain NLL Gain vs Baselines - {suffix}"),
            "record",
            "nats/bar below the uniform chain",
            ScaleKind::Linear,
            baseline_series,
        )?;

        // Per-stage coverage of the ramp. One epoch is one pass worth of BAR-TOKENS, not one
        // pass over unique bars: each stage walks its own stride-C anchor list from index 0
        // and the token budget splits ~9% / 30% / 61% across the three stages, so stage 0
        // sees roughly a quarter of the corpus at its context. That is a defensible
        // curriculum, and it belongs on a chart rather than in a derivation.
        if !self.stage_coverage.is_empty() {
            write_chart(
                &dir,
                "pretrain_stage_coverage",
                format!("Pretrain Ramp Stage Coverage - {suffix}"),
                "record",
                "distinct anchors issued / anchors available, per ramp stage",
                ScaleKind::Linear,
                self.stage_coverage
                    .iter()
                    .enumerate()
                    .map(|(stage, series)| series.labeled(&format!("stage {stage}"), len))
                    .collect(),
            )?;
        }

        write_chart(
            &dir,
            "pretrain_crps_dof",
            format!("Pretrain CRPS per DOF - {suffix}"),
            "record",
            &format!("CRPS at the fixed {diag} context"),
            ScaleKind::Linear,
            BAR_DOF_NAMES
                .iter()
                .enumerate()
                .map(|(dof, name)| self.crps_dof[dof].labeled(name, len))
                .collect(),
        )?;

        if let Some(density) = self.pit {
            let mut series: Vec<ReportSeries> = BAR_DOF_NAMES
                .iter()
                .enumerate()
                .map(|(dof, name)| ReportSeries {
                    label: (*name).to_owned(),
                    values: density[dof].iter().map(|&v| v as f32).collect(),
                })
                .collect();
            series.push(constant_series("uniform", 1.0, PIT_HIST_BINS));
            write_chart(
                &dir,
                "pretrain_pit_hist",
                format!("Pretrain PIT Histogram - {suffix}"),
                "PIT bin",
                &format!("density at the fixed {diag} context (1.0 == calibrated)"),
                ScaleKind::Linear,
                series,
            )?;
        }

        write_chart(
            &dir,
            "pretrain_dyn_loss",
            format!("Pretrain Dynamics Loss - {suffix}"),
            "record",
            "smooth L1 to the stop-grad belief",
            ScaleKind::Linear,
            vec![self.dyn_loss.labeled("train", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_kl_loss",
            format!("Pretrain Dynamics KL - {suffix}"),
            "record",
            "nats/bar",
            ScaleKind::Linear,
            vec![self.kl_loss.labeled("train", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_total_loss",
            format!("Pretrain Total Loss - {suffix}"),
            "record",
            "loss",
            ScaleKind::Linear,
            vec![self.total_loss.labeled("train", len)],
        )?;

        // What each term is actually WORTH in the objective. The absolute curves above
        // cannot show a term taking over: `dyn` rising 20x while `nll` drifts up looks like
        // two unrelated curves, and at a weight of 1.0 it was 62% of the loss.
        write_chart(
            &dir,
            "pretrain_loss_shares",
            format!("Pretrain Loss Term Shares - {suffix}"),
            "record",
            "weighted term / sum of weighted magnitudes (0.25 = the warning threshold)",
            ScaleKind::Linear,
            vec![
                self.nll_share.labeled("nll", len),
                self.dyn_share.labeled("dyn", len),
                self.kl_share.labeled("kl", len),
                constant_series("aux warning threshold", AUX_SHARE_WARN, len),
            ],
        )?;

        // The collapse direction the NextLat term has and `rms_norm` does not stop: nothing
        // prevents the trunk from making `h_{t+1} ~ h_t`, which the zero-init identity
        // dynamics predicts perfectly. That reduces `dyn` by destroying the belief
        // trajectory's temporal resolution rather than by learning any dynamics.
        write_chart(
            &dir,
            "pretrain_belief_autocorr",
            format!("Pretrain Lag-1 Belief Autocorrelation - {suffix}"),
            "record",
            "mean cos(h_t, h_t+1) over the batch (1.0 = the belief trajectory is frozen)",
            ScaleKind::Linear,
            vec![
                self.belief_autocorr.labeled("train", len),
                constant_series("frozen trajectory", 1.0, len),
            ],
        )?;

        // `dyn` against what it would be if the dynamics MLP returned its input unchanged.
        // At 1.0 the MLP contributes nothing and the term is a pure smoothness penalty on
        // the trunk; below 1.0 it is predicting something the identity cannot.
        write_chart(
            &dir,
            "pretrain_dyn_vs_identity",
            format!("Pretrain NextLat vs Identity Baseline - {suffix}"),
            "record",
            "dyn / smooth_l1(h_t, sg[h_t+k]) (1.0 = the dynamics MLP does nothing)",
            ScaleKind::Linear,
            vec![
                self.dyn_vs_identity.labeled("train", len),
                constant_series("MLP contributes nothing", 1.0, len),
            ],
        )?;

        let mut rollout = Vec::with_capacity(2 * ROLLOUT_HORIZONS.len());
        for (i, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            rollout.push(self.rollout_exact[i].labeled(&format!("h{horizon} exact"), len));
        }
        for (i, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            rollout.push(self.rollout_dynamics[i].labeled(&format!("h{horizon} dynamics"), len));
        }
        write_chart(
            &dir,
            "pretrain_rollout_nll",
            format!("Pretrain Rollout NLL by Horizon - {suffix}"),
            "record",
            "nats/bar (exact vs dynamics belief advance)",
            ScaleKind::Linear,
            rollout,
        )?;

        write_chart(
            &dir,
            "pretrain_dir_acc",
            format!("Pretrain Return Sign Accuracy - {suffix}"),
            "record",
            &format!("sign-agreement fraction at the fixed {diag} context"),
            ScaleKind::Linear,
            vec![
                self.dir_acc.labeled("val diag", len),
                constant_series("coin flip", 0.5, len),
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_lr",
            format!("Pretrain LR Multiplier - {suffix}"),
            "record",
            "global lr multiplier",
            ScaleKind::Linear,
            vec![self.lr.labeled("lr", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_muon_momentum",
            format!("Pretrain Muon Momentum - {suffix}"),
            "record",
            "momentum",
            ScaleKind::Linear,
            vec![self.muon_momentum.labeled("momentum", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_grad_norm",
            format!("Pretrain Gradient Norm - {suffix}"),
            "record",
            "observed L2 norm (never clipped)",
            ScaleKind::Symlog,
            vec![self.grad_norm.labeled("grad norm", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_unique_bar_reuse",
            format!("Pretrain Unique Bar Reuse - {suffix}"),
            "record",
            "bars seen / unique corpus bars",
            ScaleKind::Linear,
            vec![self.unique_bar_reuse.labeled("reuse", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_effective_rank",
            format!("Pretrain Belief Effective Rank - {suffix}"),
            "record",
            "participation ratio (diagnostic only)",
            ScaleKind::Linear,
            vec![self.effective_rank.labeled("effective rank", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_promotions",
            format!("Pretrain Checkpoint Promotions - {suffix}"),
            "record",
            "cumulative promotions",
            ScaleKind::Linear,
            vec![self.promotion_trace.labeled("promotions", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_schedule",
            format!("Pretrain Ramp Schedule - {suffix}"),
            "record",
            "context / batch / bars seen (M)",
            ScaleKind::Symlog,
            vec![
                self.context.labeled("context", len),
                self.batch_size.labeled("batch", len),
                self.bars_seen.labeled("bars seen (M)", len),
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_candle_rollout_mse",
            format!("Pretrain Candle Rollout MSE - {suffix}"),
            "snapshot",
            "median-path close MSE",
            ScaleKind::Linear,
            vec![ReportSeries {
                label: "mse".to_owned(),
                values: self.candle_mse.clone(),
            }],
        )?;

        write_chart(
            &dir,
            "pretrain_candle_rollout_dclose",
            format!("Pretrain Candle Rollout Close Drift - {suffix}"),
            "snapshot",
            "mean median-path log return",
            ScaleKind::Linear,
            vec![ReportSeries {
                label: "dclose".to_owned(),
                values: self.candle_dclose.clone(),
            }],
        )?;

        write_chart(
            &dir,
            "pretrain_candle_rollout_band",
            format!("Pretrain Candle Rollout Band Width - {suffix}"),
            "snapshot",
            "mean ln(p90 / p10)",
            ScaleKind::Linear,
            vec![ReportSeries {
                label: "band".to_owned(),
                values: self.candle_band.clone(),
            }],
        )?;

        write_chart(
            &dir,
            "pretrain_candle_rollout_coverage",
            format!("Pretrain Candle Rollout Coverage - {suffix}"),
            "snapshot",
            "fraction of realized closes inside the 10/90 band",
            ScaleKind::Linear,
            vec![
                ReportSeries {
                    label: "coverage".to_owned(),
                    values: self.candle_coverage.clone(),
                },
                constant_series("nominal", NOMINAL_COVERAGE, self.candle_coverage.len()),
            ],
        )?;

        Ok(())
    }

    /// Per-window `CandleCompare` (realized vs ancestral median) plus the
    /// close-path fan chart, and the scalar summary feeding
    /// `pretrain_candle_rollout_*`.
    fn write_candle_windows(
        &self,
        drawn: &Tensor,
        future_dof: &Tensor,
        windows: usize,
        steps: usize,
    ) -> Result<CandleSummary> {
        let samples = match drawn.size().as_slice() {
            [w, s, t, d]
                if *w == windows as i64 && *t == steps as i64 && *d == BAR_DOF as i64 && *s > 0 =>
            {
                *s as usize
            }
            other => anyhow::bail!(
                "rollout must be [{windows}, samples, {steps}, {BAR_DOF}], got {other:?}"
            ),
        };

        let drawn_values = tensor_values(drawn);
        let future_values = tensor_values(future_dof);
        let dir = self
            .gens_dir
            .join(self.epoch.to_string())
            .join("candle_snapshots");
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;

        let mut squared_error = Mean::default();
        let mut drift = Mean::default();
        let mut band = Mean::default();
        let mut covered = 0usize;
        let mut counted = 0usize;

        let window_stride = samples * steps * BAR_DOF;
        let sample_stride = steps * BAR_DOF;
        let mut column = vec![0.0f32; samples];

        for window in 0..windows {
            let actual = chained_candles(
                &future_values[window * steps * BAR_DOF..(window + 1) * steps * BAR_DOF],
            );
            let paths: Vec<Vec<CandleBar>> = (0..samples)
                .map(|sample| {
                    let start = window * window_stride + sample * sample_stride;
                    chained_candles(&drawn_values[start..start + sample_stride])
                })
                .collect();

            let mut median = Vec::with_capacity(steps);
            let mut low = Vec::with_capacity(steps);
            let mut high = Vec::with_capacity(steps);
            for t in 0..steps {
                // Coordinate-wise medians preserve low <= open, close <= high, so
                // the median candle is itself a well-formed bar.
                let fields: [f32; 4] = array::from_fn(|field| {
                    for (slot, path) in column.iter_mut().zip(paths.iter()) {
                        let bar = &path[t];
                        *slot = match field {
                            0 => bar.open,
                            1 => bar.high,
                            2 => bar.low,
                            _ => bar.close,
                        };
                    }
                    column.sort_by(|a, b| a.total_cmp(b));
                    quantile_sorted(&column, 0.5)
                });
                for (slot, path) in column.iter_mut().zip(paths.iter()) {
                    *slot = path[t].close;
                }
                column.sort_by(|a, b| a.total_cmp(b));
                let p10 = quantile_sorted(&column, BAND_LOW);
                let p90 = quantile_sorted(&column, BAND_HIGH);
                median.push(CandleBar {
                    open: fields[0],
                    high: fields[1],
                    low: fields[2],
                    close: fields[3],
                });
                low.push(p10);
                high.push(p90);

                let realized = actual[t].close;
                squared_error.push(((fields[3] - realized) as f64).powi(2));
                band.push(((p90.max(1e-12) / p10.max(1e-12)) as f64).ln());
                if realized >= p10 && realized <= p90 {
                    covered += 1;
                }
                counted += 1;
                let previous = if t == 0 { 1.0 } else { median[t - 1].close };
                drift.push(((fields[3].max(1e-12) / previous.max(1e-12)) as f64).ln());
            }

            let tag = format!("step{}_window{:02}", self.global_step, window + 1);
            write_report_at(
                &dir.join(format!("{tag}_candles.report.bin")),
                &Report {
                    title: format!(
                        "Pretrain Candle Snapshot (ancestral median) - step {} - window {:02}",
                        self.global_step,
                        window + 1
                    ),
                    x_label: Some("forecast bar".to_owned()),
                    y_label: Some("relative price".to_owned()),
                    scale: ScaleKind::Linear,
                    kind: ReportKind::CandleCompare {
                        actual: actual.clone(),
                        predicted: median.clone(),
                    },
                },
            )?;
            write_report_at(
                &dir.join(format!("{tag}_band_candles.report.bin")),
                &Report {
                    title: format!(
                        "Pretrain Rollout Band - step {} - window {:02}",
                        self.global_step,
                        window + 1
                    ),
                    x_label: Some("forecast bar".to_owned()),
                    y_label: Some("relative close".to_owned()),
                    scale: ScaleKind::Linear,
                    kind: ReportKind::MultiLine {
                        series: vec![
                            ReportSeries {
                                label: "actual".to_owned(),
                                values: actual.iter().map(|bar| bar.close).collect(),
                            },
                            ReportSeries {
                                label: "p10".to_owned(),
                                values: low,
                            },
                            ReportSeries {
                                label: "p50".to_owned(),
                                values: median.iter().map(|bar| bar.close).collect(),
                            },
                            ReportSeries {
                                label: "p90".to_owned(),
                                values: high,
                            },
                        ],
                    },
                },
            )?;
        }

        Ok(CandleSummary {
            mse: squared_error.value(),
            dclose: drift.value(),
            band: band.value(),
            coverage: if counted == 0 {
                f64::NAN
            } else {
                covered as f64 / counted as f64
            },
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CandleSummary {
    mse: f64,
    dclose: f64,
    band: f64,
    coverage: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Decode a `[T, BAR_DOF]` DOF path into a chained candle path anchored at a
/// previous close of `1.0`, so every window is on a comparable relative scale.
/// Volume is not a candle field, so the EMA reference is irrelevant here.
fn chained_candles(dof_path: &[f32]) -> Vec<CandleBar> {
    let mut previous = 1.0f32;
    dof_path
        .chunks_exact(BAR_DOF)
        .map(|row| {
            let dof = BarDof::from_array([row[0], row[1], row[2], row[3], row[4]]);
            let bar = decode_dof(previous, &dof, 1.0);
            previous = bar.close;
            CandleBar {
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
            }
        })
        .collect()
}

fn quantile_sorted(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = (position - lower as f64) as f32;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn tensor_values(tensor: &Tensor) -> Vec<f32> {
    let flat = tensor
        .detach()
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .view([-1]);
    let numel = flat.numel();
    let mut values = vec![0.0f32; numel];
    flat.copy_data(&mut values, numel);
    values
}

fn constant_series(label: &str, value: f64, len: usize) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: vec![value as f32; len],
    }
}

/// One-point series, the shape the end-of-run battery reports each scalar in.
fn point_series(label: &str, value: f64) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: vec![value as f32],
    }
}

/// SHA-256 of a checkpoint, streamed so a multi-hundred-megabyte artifact never
/// lands in memory. Fingerprinting the promotion is what lets the held-out
/// battery prove it scored the selected weights and not something else.
fn file_digest(path: &Path) -> Result<[u8; 32]> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open {} for fingerprinting", path.display()))?;
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(context.finish().as_ref());
    Ok(digest)
}

/// Write a chart, skipping it while no series holds a finite value so the run
/// directory never fills with all-NaN placeholders.
fn write_chart(
    dir: &Path,
    base: &str,
    title: String,
    x_label: &str,
    y_label: &str,
    scale: ScaleKind,
    series: Vec<ReportSeries>,
) -> Result<()> {
    if !series
        .iter()
        .any(|s| s.values.iter().any(|value| value.is_finite()))
    {
        return Ok(());
    }
    write_report_at(
        &dir.join(format!("{base}.report.bin")),
        &Report {
            title,
            x_label: Some(x_label.to_owned()),
            y_label: Some(y_label.to_owned()),
            scale,
            kind: ReportKind::MultiLine { series },
        },
    )
}

fn write_report_at(path: &Path, report: &Report) -> Result<()> {
    write_report(path, report).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::read_report;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Every base this module can write. The TUI's `meta_chart_bases` must
    /// contain each of these or the chart is invisible.
    const EXPECTED_BASES: [&str; 25] = [
        "pretrain_nll_bar",
        "pretrain_nll_bar_diag896",
        "pretrain_nll_dof",
        "pretrain_nll_vs_baselines",
        "pretrain_crps_dof",
        "pretrain_pit_hist",
        "pretrain_dyn_loss",
        "pretrain_kl_loss",
        "pretrain_total_loss",
        "pretrain_loss_shares",
        "pretrain_belief_autocorr",
        "pretrain_dyn_vs_identity",
        "pretrain_rollout_nll",
        "pretrain_dir_acc",
        "pretrain_lr",
        "pretrain_muon_momentum",
        "pretrain_grad_norm",
        "pretrain_unique_bar_reuse",
        "pretrain_effective_rank",
        "pretrain_promotions",
        "pretrain_schedule",
        "pretrain_candle_rollout_mse",
        "pretrain_candle_rollout_dclose",
        "pretrain_candle_rollout_band",
        "pretrain_candle_rollout_coverage",
    ];

    static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "pretrain_reports_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The two discrete rules both start a zero-init head at `BAR_DOF * ln(NUM_BAR_BINS)`.
    /// The density rule adds the measure term, which is why the mode-aware line lives on
    /// [`HeldOutBaselines`] and this one is explicitly the CATEGORICAL figure.
    #[test]
    fn uniform_baseline_is_five_log_one_twenty_eight() {
        // 5 * ln(128) = 24.2601513..., not the 24.2536 quoted in the brief.
        assert!((uniform_categorical_nll_bar() - 24.260_151_3).abs() < 1.0e-6);
        assert_eq!(uniform_categorical_nll_bar(), 5.0 * 128.0f64.ln());
    }

    #[test]
    fn series_nan_pads_skipped_ticks() {
        let mut series = Series::default();
        series.set(0, 1.0);
        series.set(3, 4.0);
        series.set(4, f64::NAN);
        let values = series.padded(6);
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan() && values[2].is_nan());
        assert_eq!(values[3], 4.0);
        assert!(values[4].is_nan() && values[5].is_nan());
    }

    #[test]
    fn mean_ignores_non_finite_samples() {
        let mut mean = Mean::default();
        mean.push(f64::NAN);
        mean.push(2.0);
        mean.push(4.0);
        mean.push(f64::INFINITY);
        assert_eq!(mean.value(), 3.0);
        assert!(Mean::default().value().is_nan());
    }

    #[test]
    fn pit_density_is_flat_for_a_uniform_transform() {
        let rows = 4096i64;
        let uniform = Tensor::arange(rows, (Kind::Float, Device::Cpu)) / rows as f64;
        let pit = uniform.unsqueeze(-1).repeat([1, BAR_DOF as i64]);
        let mut histogram = PitHistogram::default();
        histogram.accumulate(&pit);
        assert!(!histogram.is_empty());
        for dof in histogram.density() {
            for bin in dof {
                assert!((bin - 1.0).abs() < 1.0e-6, "expected flat PIT, got {bin}");
            }
        }
        histogram.reset();
        assert!(histogram.is_empty());
    }

    #[test]
    fn pit_clamps_out_of_range_values_into_the_edge_bins() {
        let pit =
            Tensor::from_slice(&[-0.5f32, 1.5, 0.5, f32::NAN, 0.0]).reshape([1, BAR_DOF as i64]);
        let mut histogram = PitHistogram::default();
        histogram.accumulate(&pit);
        let density = histogram.density();
        assert_eq!(density[0][0], PIT_HIST_BINS as f64);
        assert_eq!(density[1][PIT_HIST_BINS - 1], PIT_HIST_BINS as f64);
        assert_eq!(density[2][PIT_HIST_BINS / 2], PIT_HIST_BINS as f64);
        assert!(density[3][0].is_nan());
        assert_eq!(density[4][0], PIT_HIST_BINS as f64);
    }

    #[test]
    fn effective_rank_spans_collapse_to_isotropy() {
        let device = Device::Cpu;
        let rank_one = Tensor::arange(64, (Kind::Float, device))
            .unsqueeze(-1)
            .matmul(&Tensor::ones([1, 8], (Kind::Float, device)));
        let collapsed = belief_effective_rank(&rank_one);
        assert!(
            (collapsed - 1.0).abs() < 1.0e-3,
            "rank-one beliefs should score 1, got {collapsed}"
        );

        let isotropic = Tensor::randn([8192, 8], (Kind::Float, device));
        let spread = belief_effective_rank(&isotropic);
        assert!(
            spread > 7.0 && spread <= 8.0 + 1.0e-6,
            "isotropic beliefs should approach the width, got {spread}"
        );

        assert!(belief_effective_rank(&Tensor::zeros([1, 8], (Kind::Float, device))).is_nan());
    }

    #[test]
    fn chained_candles_stay_ordered_and_anchored() {
        let path = [
            0.01f32, 0.02, 0.25, 0.75, 0.0, //
            -0.03, 0.05, 0.9, 0.1, 0.5,
        ];
        let candles = chained_candles(&path);
        assert_eq!(candles.len(), 2);
        for candle in &candles {
            assert!(candle.low <= candle.open.min(candle.close));
            assert!(candle.high >= candle.open.max(candle.close));
        }
        assert!((candles[0].close as f64 - 0.01f64.exp()).abs() < 1.0e-5);
        assert!(
            (candles[1].close as f64 - (0.01f64 - 0.03).exp()).abs() < 1.0e-5,
            "the second bar must chain off the first close"
        );
    }

    #[test]
    fn quantiles_interpolate_between_order_statistics() {
        let sorted = [0.0f32, 1.0, 2.0, 3.0];
        assert_eq!(quantile_sorted(&sorted, 0.0), 0.0);
        assert_eq!(quantile_sorted(&sorted, 1.0), 3.0);
        assert_eq!(quantile_sorted(&sorted, 0.5), 1.5);
        assert!((quantile_sorted(&sorted, 0.1) - 0.3).abs() < 1.0e-6);
        assert!(quantile_sorted(&[], 0.5).is_nan());
    }

    const MARGINAL_DOF: [f64; BAR_DOF] = [4.760, 4.693, 3.920, 3.836, 4.852];

    fn checkpoint(root: &Path, name: &str, body: &[u8]) -> PathBuf {
        let path = root.join(name);
        fs::write(&path, body).expect("checkpoint fixture");
        path
    }

    fn populated_epoch(epoch: usize, step: usize, promoted: Option<PathBuf>) -> EpochMetrics {
        let mut metrics = EpochMetrics::nan();
        metrics.epoch = epoch;
        metrics.global_step = step;
        metrics.train_nll_bar = 20.0;
        metrics.val_nll_bar = 21.0;
        metrics.best_val_nll_bar = 21.0;
        metrics.val_nll_bar_diag = 21.5;
        metrics.promoted_checkpoint = promoted;
        metrics.train_nll_dof = [4.0; BAR_DOF];
        metrics.val_nll_dof = [4.2; BAR_DOF];
        metrics.val_crps_dof = [0.1; BAR_DOF];
        metrics.val_dir_acc = 0.52;
        metrics.rollout_nll_exact = [21.0, 22.0, 23.0, 24.0];
        metrics.rollout_nll_dynamics = [21.1, 22.4, 23.9, 25.2];
        metrics.unique_bar_reuse = 0.25;
        metrics.effective_rank = 42.0;
        metrics
            .val_pit
            .accumulate(&(Tensor::arange(512, (Kind::Float, Device::Cpu)) / 512.0)
                .unsqueeze(-1)
                .repeat([1, BAR_DOF as i64]));
        metrics
    }

    #[test]
    fn a_full_cycle_writes_every_registered_base() {
        let root = scratch_dir("full_cycle");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);

        for step in 0..STEP_DECIMATION {
            let mut metrics = StepMetrics::nan();
            metrics.epoch = 0;
            metrics.step = step;
            metrics.nll_bar = 24.0 - step as f64 * 0.01;
            metrics.nll_dof = [4.8; BAR_DOF];
            metrics.dyn_loss = 0.5;
            metrics.kl_loss = 0.25;
            metrics.total_loss = 24.75;
            metrics.lr_mult = 1.0;
            metrics.muon_momentum = 0.85;
            metrics.grad_norm = 3.5;
            metrics.context = 896;
            metrics.batch_size = 16;
            metrics.bars_seen = 1_000_000 * (step as u64 + 1);
            reporter.record_step(&metrics).unwrap();
        }

        let windows = 2i64;
        let samples = 8i64;
        let horizon = 4i64;
        let rollout = Tensor::rand(
            [windows, samples, horizon, BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        ) * 0.01;
        let future = Tensor::rand(
            [windows, horizon, BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        ) * 0.01;
        reporter
            .record_snapshot(&SnapshotInput {
                rollout: &rollout,
                future_dof: &future,
                epoch: 0,
                global_step: STEP_DECIMATION,
            })
            .unwrap();
        let weights = checkpoint(&root, "pretrain_best.ot", b"promoted-weights");
        reporter
            .record_epoch(&populated_epoch(0, STEP_DECIMATION, Some(weights.clone())))
            .unwrap();

        let dir = root.join("0");
        for base in EXPECTED_BASES {
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was never written");
            let report = read_report(&path).expect("report reads back");
            match report.kind {
                ReportKind::MultiLine { series } => assert!(
                    series
                        .iter()
                        .any(|s| s.values.iter().any(|v| v.is_finite())),
                    "{base} holds no finite value"
                ),
                other => panic!("{base} has unexpected kind {other:?}"),
            }
        }

        let snapshots = dir.join("candle_snapshots");
        for window in 1..=windows {
            for suffix in ["candles", "band_candles"] {
                let path = snapshots.join(format!(
                    "step{}_window{window:02}_{suffix}.report.bin",
                    STEP_DECIMATION
                ));
                assert!(path.exists(), "missing snapshot {}", path.display());
            }
        }
        let compare = read_report(
            &snapshots.join(format!("step{}_window01_candles.report.bin", STEP_DECIMATION)),
        )
        .unwrap();
        match compare.kind {
            ReportKind::CandleCompare { actual, predicted } => {
                assert_eq!(actual.len(), horizon as usize);
                assert_eq!(predicted.len(), horizon as usize);
                for bar in predicted {
                    assert!(bar.low <= bar.open.min(bar.close));
                    assert!(bar.high >= bar.open.max(bar.close));
                }
            }
            other => panic!("expected CandleCompare, got {other:?}"),
        }

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn train_is_dense_and_validation_is_sparse_on_the_shared_axis() {
        let root = scratch_dir("axis");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        for step in 0..STEP_DECIMATION * 3 {
            let mut metrics = StepMetrics::nan();
            metrics.step = step;
            metrics.nll_bar = 24.0;
            reporter.record_step(&metrics).unwrap();
        }
        reporter
            .record_epoch(&populated_epoch(0, STEP_DECIMATION * 3, None))
            .unwrap();

        let report = read_report(&root.join("0").join("pretrain_nll_bar.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let train = &series[0];
        let val = &series[1];
        assert_eq!(train.label, "train");
        assert_eq!(val.label, "val");
        assert_eq!(train.values.len(), 4, "3 step ticks plus the epoch tick");
        assert_eq!(val.values.len(), 4);
        assert_eq!(train.values.iter().filter(|v| v.is_finite()).count(), 4);
        assert_eq!(
            val.values.iter().filter(|v| v.is_finite()).count(),
            1,
            "validation must be a single marker, not a dense curve"
        );
        assert_eq!(val.values[3], 21.0);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn promotions_accumulate_and_absent_metrics_stay_unwritten() {
        let root = scratch_dir("promotions");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let weights = checkpoint(&root, "best.ot", b"weights-v1");
        for (epoch, promoted) in [
            (0usize, Some(weights.clone())),
            (1, None),
            (2, Some(weights.clone())),
        ] {
            let mut metrics = populated_epoch(epoch, epoch * 10, promoted);
            metrics.effective_rank = f64::NAN;
            reporter.record_epoch(&metrics).unwrap();
        }

        let report = read_report(&root.join("2").join("pretrain_promotions.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        assert_eq!(series[0].values, vec![1.0, 1.0, 2.0]);

        for epoch in 0..3 {
            assert!(
                !root
                    .join(epoch.to_string())
                    .join("pretrain_effective_rank.report.bin")
                    .exists(),
                "an all-NaN series must not produce a file"
            );
        }

        fs::remove_dir_all(&root).ok();
    }

    fn populated_battery(checkpoint: PathBuf) -> TestBattery {
        let mut battery = TestBattery::nan(checkpoint, "0f1e2d3c4b5a".to_owned());
        battery.nll_bar = 21.4;
        battery.nll_dof = [4.2, 4.3, 4.2, 4.3, 4.4];
        battery.crps_dof = [0.003, 0.002, 0.19, 0.21, 0.44];
        battery.rollout_nll_exact = [21.4, 22.1, 22.9, 23.8];
        battery.rollout_nll_dynamics = [21.5, 22.4, 23.6, 25.1];
        battery.dir_acc = 0.514;
        battery
            .pit
            .accumulate(&(Tensor::arange(1024, (Kind::Float, Device::Cpu)) / 1024.0)
                .unsqueeze(-1)
                .repeat([1, BAR_DOF as i64]));
        battery.corpus_fingerprint = "c".repeat(64);
        battery.split_bounds = (1_600_000_000_000, 1_650_000_000_000);
        battery.nll_bar_conditional = 22.1;
        battery.nll_dof_conditional = [4.2, 4.3, 4.55, 4.65, 4.4];
        battery.nll_bar_se = 0.031;
        battery.nll_bar_ci = (21.34, 21.46);
        battery
    }

    fn promoted_reporter(root: &Path, weights: &Path) -> PretrainReporter {
        let mut reporter = PretrainReporter::new(root, MARGINAL_DOF);
        reporter.set_held_out_baselines(smoothed_baselines());
        reporter
            .record_epoch(&populated_epoch(0, 10, Some(weights.to_path_buf())))
            .unwrap();
        reporter
    }

    /// The reference set a `smoothed` run reports, i.e. the one mode that has a floor.
    fn smoothed_baselines() -> HeldOutBaselines {
        HeldOutBaselines {
            scoring: BarScoring::Smoothed,
            uniform_nll_bar: uniform_categorical_nll_bar(),
            marginal_nll_dof_conditional: [4.74, 4.66, 3.83, 3.74, 4.85],
            marginal_nll_dof_val: [4.75, 4.67, 3.78, 3.70, 4.86],
            encoding_identity_nats: 0.690,
            scoring_floor_bar: 4.648,
            marginal_class_dof: [0.272, 0.345, 1.195, 1.217, 0.0],
            marginal_shape_dof: [4.469, 4.314, 2.554, 2.451, 4.851],
        }
    }

    #[test]
    fn the_held_out_battery_is_written_once_with_every_scalar() {
        let root = scratch_dir("battery");
        let weights = checkpoint(&root, "best.ot", b"promoted");
        let reporter = promoted_reporter(&root, &weights);
        reporter.finish(&populated_battery(weights)).unwrap();

        let report = read_report(&root.join("0").join("pretrain_test.report.bin")).unwrap();
        assert!(
            report.title.contains("best.ot") && report.title.contains("0f1e2d3c4b5a"),
            "the battery must name the artifact and lineage it scored: {}",
            report.title
        );
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let labels: Vec<&str> = series.iter().map(|s| s.label.as_str()).collect();
        for expected in [
            "nll_bar",
            "nll_bar vs uniform",
            "nll_bar vs marginal",
            "uniform",
            "marginal",
            "nll r",
            "crps w",
            "pit tv u",
            "rollout h64 exact",
            "rollout h64 dynamics",
            "dir acc",
            "nll_bar se",
            "nll_bar ci95 low",
            "nll_bar conditional",
            "marginal conditional",
            "encoding identity nats",
            "nll u | s!=0",
            "nll r vs marginal",
        ] {
            assert!(labels.contains(&expected), "battery is missing {expected}");
        }
        for entry in &series {
            assert_eq!(entry.values.len(), 1, "{} is not a single point", entry.label);
            assert!(
                entry.values[0].is_finite(),
                "{} is not finite",
                entry.label
            );
        }
        let marginal_total: f64 = MARGINAL_DOF.iter().sum();
        let gain = series
            .iter()
            .find(|s| s.label == "nll_bar vs marginal")
            .unwrap()
            .values[0] as f64;
        assert!((gain - (marginal_total - 21.4)).abs() < 1.0e-4);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_battery_refuses_a_checkpoint_that_was_never_promoted() {
        let root = scratch_dir("battery_wrong");
        let promoted = checkpoint(&root, "best.ot", b"promoted");
        let other = checkpoint(&root, "final.ot", b"not-promoted");
        let reporter = promoted_reporter(&root, &promoted);
        let error = reporter
            .finish(&populated_battery(other))
            .expect_err("scoring an unpromoted artifact must fail");
        assert!(
            error.to_string().contains("actually selected"),
            "unexpected error: {error}"
        );
        assert!(!root.join("0").join("pretrain_test.report.bin").exists());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_battery_refuses_a_checkpoint_rewritten_after_promotion() {
        let root = scratch_dir("battery_mutated");
        let weights = checkpoint(&root, "best.ot", b"promoted");
        let reporter = promoted_reporter(&root, &weights);
        fs::write(&weights, b"rewritten-after-promotion").unwrap();
        let error = reporter
            .finish(&populated_battery(weights))
            .expect_err("a mutated artifact must fail");
        assert!(
            error.to_string().contains("changed on disk"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_battery_refuses_an_in_memory_model_and_an_unpromoted_run() {
        let root = scratch_dir("battery_guards");
        let weights = checkpoint(&root, "best.ot", b"promoted");

        let never_promoted = PretrainReporter::new(&root, MARGINAL_DOF);
        let error = never_promoted
            .finish(&populated_battery(weights.clone()))
            .expect_err("a run that never promoted has nothing to score");
        assert!(
            error.to_string().contains("no promotion was ever reported"),
            "unexpected error: {error}"
        );

        let reporter = promoted_reporter(&root, &weights);
        let mut battery = populated_battery(weights);
        battery.model_lineage = "   ".to_owned();
        let error = reporter
            .finish(&battery)
            .expect_err("an empty lineage means no reload happened");
        assert!(
            error.to_string().contains("reloaded through BarWorldModel::load"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_promotion_whose_artifact_is_missing_fails_immediately() {
        let root = scratch_dir("missing_artifact");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let error = reporter
            .record_epoch(&populated_epoch(0, 10, Some(root.join("absent.ot"))))
            .expect_err("promoting a file that is not on disk must fail at once");
        assert!(
            error.to_string().contains("save it before reporting the promotion"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn marginal_reference_lines_track_the_fitted_supports() {
        let root = scratch_dir("baselines");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let baselines = smoothed_baselines();
        reporter.set_held_out_baselines(baselines);
        reporter.record_epoch(&populated_epoch(0, 10, None)).unwrap();

        let report =
            read_report(&root.join("0").join("pretrain_nll_vs_baselines.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let marginal_total: f64 = MARGINAL_DOF.iter().sum();
        let uniform = baselines.uniform_nll_bar;
        let line = series.iter().find(|s| s.label == "marginal").unwrap();
        assert!(
            (line.values[0] as f64 - (uniform - marginal_total)).abs() < 1.0e-4,
            "marginal reference should sit {} nats above uniform, got {}",
            uniform - marginal_total,
            line.values[0]
        );
        assert!(series.iter().any(|s| s.label == "uniform"));
        // The smoothed rule is the only one with a floor, so the unreachable line is drawn.
        let floor = series
            .iter()
            .find(|s| s.label == "smoothing floor (unreachable)")
            .expect("the smoothed rule must draw its floor");
        assert!(
            (floor.values[0] as f64 - (uniform - baselines.scoring_floor_bar)).abs() < 1.0e-4
        );

        let dof_report = read_report(&root.join("0").join("pretrain_nll_dof.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = dof_report.kind else {
            panic!("expected MultiLine");
        };
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            let label = format!("{name} marginal");
            let line = series.iter().find(|s| s.label == label).unwrap();
            assert!((line.values[0] as f64 - MARGINAL_DOF[dof]).abs() < 1.0e-4);
        }

        // Absent marginals must simply omit the reference, never invent one.
        let bare_root = scratch_dir("baselines_bare");
        let mut bare = PretrainReporter::new(&bare_root, [f64::NAN; BAR_DOF]);
        bare.record_epoch(&populated_epoch(0, 10, None)).unwrap();
        let report =
            read_report(&bare_root.join("0").join("pretrain_nll_vs_baselines.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let line = series.iter().find(|s| s.label == "marginal").unwrap();
        assert!(line.values.iter().all(|v| !v.is_finite()));
        let dof_report =
            read_report(&bare_root.join("0").join("pretrain_nll_dof.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = dof_report.kind else {
            panic!("expected MultiLine");
        };
        for name in BAR_DOF_NAMES {
            let line = series
                .iter()
                .find(|s| s.label == format!("{name} marginal"))
                .unwrap();
            assert!(
                line.values.iter().all(|v| !v.is_finite()),
                "{name} marginal must stay absent, never be invented"
            );
        }

        fs::remove_dir_all(&root).ok();
        fs::remove_dir_all(&bare_root).ok();
    }
}
