//! Held-out dispersion, and paired comparison of two pretraining runs.
//!
//! A held-out mean with no standard error is not a measurement, and until this module
//! existed the pretrainer produced exactly that: `evaluate()` reduced every chunk to a
//! scalar and discarded the per-window values, so no dispersion of `nll_bar` was ever
//! available and every ablation delta was read against zero.
//!
//! Two facts about this corpus set the whole design.
//!
//! * **The windows are not independent.** A 2048-bar window at ~93 bars/day spans ~22
//!   trading days, the validation split holds ~108, so there are only about FOUR
//!   non-overlapping time slots per symbol and every symbol shares the same four wall-clock
//!   slots. The naive iid standard error over 4096 windows is ~0.026 nats; the true standard
//!   error of the LEVEL is ~0.10, because the market-common regime term does not average
//!   down. [`WindowScores::level_dispersion`] blocks by calendar month for exactly this
//!   reason, and it is the number to quote when stating an absolute level.
//! * **Paired comparison is what makes the campaign viable.** At an unpaired standard error
//!   of ~0.10 the minimum detectable difference at 80% power is ~0.41 nats, larger than most
//!   effects worth chasing. On the IDENTICAL pinned windows the per-window correlation
//!   between two runs is 0.95-0.99, and the paired MDE falls to 0.04-0.09 nats. So the
//!   deliverable that matters is [`paired_comparison`]: two runs' per-window vectors,
//!   differenced window by window, with a block-bootstrap interval on the difference.
//!
//! Every per-window vector is persisted next to the checkpoint it was measured on
//! ([`window_scores_path`]) together with the corpus fingerprint, the split instants and the
//! evaluation seed, so a pairing that is not actually comparable fails loudly instead of
//! quietly returning a number.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use chrono::{DateTime, Datelike, Utc};
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::torch::bar_dist::{BAR_DOF, BAR_DOF_NAMES};
use crate::torch::dataset::iso_ms;

/// Resamples per bootstrap. 1000 draws over ~4k f64 is microseconds, and the 2.5/97.5
/// percentiles of 1000 draws are stable to about a percent of the interval width.
pub const BOOTSTRAP_DRAWS: usize = 1000;

/// Fixed bootstrap stream. The interval is a property of the data, not of the run, so two
/// reports of the same vector must agree to the last digit.
pub const BOOTSTRAP_SEED: u64 = 0xB10C_B007_5EED_0001;

/// Two-sided interval the reported CI covers.
pub const CI_MASS: f64 = 0.95;

/// Schema of the persisted per-window vector.
pub const WINDOW_SCORES_FORMAT_VERSION: u32 = 1;

/// `z` for a two-sided 95% interval times `sqrt(2)`, i.e. the multiple of the paired
/// standard error a difference must clear to be detectable at 80% power.
const MDE_MULTIPLIER: f64 = 2.802;

/// Where a checkpoint's per-window held-out vector lives: `pretrain_best.windows.json`
/// beside `pretrain_best.ot`.
pub fn window_scores_path(checkpoint: &Path) -> PathBuf {
    checkpoint.with_extension("windows.json")
}

/// Calendar month of an instant as `year * 12 + (month - 1)`, in UTC.
///
/// UTC rather than ET on purpose: this is a blocking key, not a session label, and a bar at
/// 20:00 ET on the last day of a month is in the same regime as one at 04:00 ET the next
/// morning either way. What matters is that windows a month apart land in different blocks.
pub fn calendar_month(ts_ms: i64) -> i32 {
    DateTime::<Utc>::from_timestamp_millis(ts_ms)
        .map(|stamp| stamp.year() * 12 + stamp.month0() as i32)
        .unwrap_or(i32::MIN)
}

// ---------------------------------------------------------------------------
// Block bootstrap
// ---------------------------------------------------------------------------

/// A mean with a block-bootstrap interval around it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dispersion {
    pub mean: f64,
    /// Standard deviation of the bootstrap means, i.e. the standard error of `mean`.
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// Resampling units. This, not `samples`, is what the interval width is governed by.
    pub blocks: usize,
    pub samples: usize,
}

impl Dispersion {
    pub fn nan() -> Self {
        Self {
            mean: f64::NAN,
            se: f64::NAN,
            ci_low: f64::NAN,
            ci_high: f64::NAN,
            blocks: 0,
            samples: 0,
        }
    }

    /// Smallest difference this dispersion could detect at 80% power, two-sided alpha 0.05,
    /// if it were the standard error of a difference.
    pub fn minimum_detectable_effect(&self) -> f64 {
        MDE_MULTIPLIER * self.se
    }
}

impl fmt::Display for Dispersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.4} +/- {:.4} (95% CI {:.4}..{:.4}, {} blocks / {} windows)",
            self.mean, self.se, self.ci_low, self.ci_high, self.blocks, self.samples
        )
    }
}

/// Nonparametric block bootstrap: resample BLOCKS with replacement, never individual
/// observations.
///
/// `blocks[i]` is the resampling unit observation `i` belongs to. Every draw picks as many
/// blocks as there are blocks, with replacement, and averages every observation inside them,
/// so a block with many windows carries its natural weight. The interval is the empirical
/// 2.5/97.5 percentile of the draws and `se` is their standard deviation.
///
/// Resampling blocks rather than windows is the whole point: windows inside one block share
/// a regime, so treating them as independent draws would divide the variance by a sample
/// size the data does not have.
pub fn block_bootstrap(values: &[f64], blocks: &[u64], draws: usize, seed: u64) -> Dispersion {
    assert_eq!(
        values.len(),
        blocks.len(),
        "every value needs a block assignment"
    );
    let finite: Vec<(u64, f64)> = blocks
        .iter()
        .copied()
        .zip(values.iter().copied())
        .filter(|(_, v)| v.is_finite())
        .collect();
    if finite.is_empty() {
        return Dispersion::nan();
    }

    // (sum, count) per block, in a deterministic order.
    let mut grouped: BTreeMap<u64, (f64, u64)> = BTreeMap::new();
    for (block, value) in &finite {
        let entry = grouped.entry(*block).or_insert((0.0, 0));
        entry.0 += *value;
        entry.1 += 1;
    }
    let totals: Vec<(f64, u64)> = grouped.values().copied().collect();
    let samples = finite.len();
    let mean = totals.iter().map(|(sum, _)| *sum).sum::<f64>() / samples as f64;
    if totals.len() < 2 || draws == 0 {
        // One block is one observation: there is no dispersion to estimate, and pretending
        // otherwise would report a zero-width interval as if it were precision.
        return Dispersion {
            mean,
            se: f64::NAN,
            ci_low: f64::NAN,
            ci_high: f64::NAN,
            blocks: totals.len(),
            samples,
        };
    }

    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut sum = 0.0;
        let mut count = 0u64;
        for _ in 0..totals.len() {
            let (block_sum, block_count) = totals
                .choose(&mut rng)
                .copied()
                .expect("totals is non-empty");
            sum += block_sum;
            count += block_count;
        }
        means.push(sum / count as f64);
    }
    means.sort_by(f64::total_cmp);

    let draw_mean = means.iter().sum::<f64>() / means.len() as f64;
    let variance = means
        .iter()
        .map(|m| (m - draw_mean) * (m - draw_mean))
        .sum::<f64>()
        / (means.len() - 1) as f64;
    let tail = (1.0 - CI_MASS) / 2.0;
    Dispersion {
        mean,
        se: variance.sqrt(),
        ci_low: percentile(&means, tail),
        ci_high: percentile(&means, 1.0 - tail),
        blocks: totals.len(),
        samples,
    }
}

/// Linear-interpolated percentile of an ascending slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

// ---------------------------------------------------------------------------
// Persisted per-window vectors
// ---------------------------------------------------------------------------

/// One pinned window's held-out score.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowScore {
    pub symbol: String,
    /// Bar index of the window's first DOF-carrying bar within its symbol file.
    pub bar_index: u32,
    /// Open timestamp of that bar, which places the window on the calendar.
    pub ts_ms: i64,
    /// Mean nats per bar for each of the five chain factors, in `[r, s, u, v, w]` order.
    pub nll_dof: [f64; BAR_DOF],
    /// `nll_bar` with the encoding tautology excluded: `u` and `v` are averaged only over
    /// bars with `s != 0`, where they are not determined by the encoding.
    pub nll_bar_conditional: f64,
}

impl WindowScore {
    pub fn nll_bar(&self) -> f64 {
        self.nll_dof.iter().sum()
    }
}

/// A run's per-window held-out vector, with everything needed to decide whether another
/// run's vector is comparable to it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowScores {
    pub format_version: u32,
    /// Run directory name, for the comparison report.
    pub run: String,
    pub global_step: usize,
    /// `"val"` or `"test"`.
    pub split: String,
    pub context: i64,
    /// The campaign-fixed evaluation seed the windows were drawn with.
    pub eval_window_seed: u64,
    pub corpus_fingerprint: String,
    pub split_bounds: (i64, i64),
    /// Calibrated-marginal reference at the time of measurement, for context in reports.
    pub marginal_nll_bar: f64,
    /// Scoring rule every number in this vector was measured under, by name.
    ///
    /// `None` on a vector written before the rule became a flag, which is NOT the same as
    /// "smoothed": it means the file cannot be checked, and [`paired_comparison`] refuses
    /// it rather than assuming. The three modes differ by additive constants that depend on
    /// the binning, so pairing across them would difference two different quantities.
    #[serde(default)]
    pub scoring: Option<String>,
    pub windows: Vec<WindowScore>,
}

impl WindowScores {
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
        let body = serde_json::to_vec(self).context("serializing per-window held-out scores")?;
        std::fs::write(path, body).with_context(|| format!("writing {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let body = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let scores: Self =
            serde_json::from_slice(&body).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(
            scores.format_version == WINDOW_SCORES_FORMAT_VERSION,
            "{} has window-score format version {}, expected {WINDOW_SCORES_FORMAT_VERSION}",
            path.display(),
            scores.format_version
        );
        ensure!(
            !scores.windows.is_empty(),
            "{} holds no windows",
            path.display()
        );
        Ok(scores)
    }

    pub fn nll_bar(&self) -> Vec<f64> {
        self.windows.iter().map(WindowScore::nll_bar).collect()
    }

    pub fn nll_bar_conditional(&self) -> Vec<f64> {
        self.windows.iter().map(|w| w.nll_bar_conditional).collect()
    }

    pub fn nll_dof(&self, dof: usize) -> Vec<f64> {
        self.windows.iter().map(|w| w.nll_dof[dof]).collect()
    }

    /// Dense block ids keyed by `(symbol, calendar month)`.
    ///
    /// This is the finest blocking that is still defensible: two windows of the same ticker
    /// in the same month are one draw. It does NOT capture the market-common regime term,
    /// because every symbol shares the same calendar months — use [`Self::month_blocks`] for
    /// that, and see [`Self::level_dispersion`].
    pub fn symbol_month_blocks(&self) -> Vec<u64> {
        let mut ids: BTreeMap<(&str, i32), u64> = BTreeMap::new();
        let mut next = 0u64;
        self.windows
            .iter()
            .map(|w| {
                let key = (w.symbol.as_str(), calendar_month(w.ts_ms));
                *ids.entry(key).or_insert_with(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect()
    }

    /// Dense block ids keyed by calendar month alone: every symbol in one month is a single
    /// resampling unit.
    pub fn month_blocks(&self) -> Vec<u64> {
        let mut ids: BTreeMap<i32, u64> = BTreeMap::new();
        let mut next = 0u64;
        self.windows
            .iter()
            .map(|w| {
                *ids.entry(calendar_month(w.ts_ms)).or_insert_with(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect()
    }

    /// `nll_bar` with a `(symbol, month)` block-bootstrap interval. Reported per validation
    /// as `val_nll_bar_se`.
    pub fn dispersion(&self) -> Dispersion {
        block_bootstrap(
            &self.nll_bar(),
            &self.symbol_month_blocks(),
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        )
    }

    /// `nll_bar` with a CALENDAR-MONTH block bootstrap, i.e. the honest standard error of
    /// the absolute level.
    ///
    /// Almost every `(symbol, month)` block holds a single window, so [`Self::dispersion`]
    /// is close to an iid bootstrap and lands near 0.026 nats. That understates the level by
    /// about 4x, because all 4096 windows sit in a handful of shared wall-clock months and a
    /// market-wide regime shift moves all of them together. This estimator resamples those
    /// months, so it sees the term that actually dominates. It is coarse — single-digit
    /// blocks — which is a statement about the split, not about the estimator.
    pub fn level_dispersion(&self) -> Dispersion {
        block_bootstrap(
            &self.nll_bar(),
            &self.month_blocks(),
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        )
    }
}

// ---------------------------------------------------------------------------
// Paired comparison
// ---------------------------------------------------------------------------

/// Two runs differenced window by window.
#[derive(Clone, Debug)]
pub struct PairedComparison {
    pub baseline_run: String,
    pub candidate_run: String,
    pub windows: usize,
    /// Scoring rule both runs were measured under. Every nats figure below is in that
    /// rule's units and is comparable to nothing measured under another.
    pub scoring: String,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    /// `candidate - baseline`; negative means the candidate is better.
    pub difference: Dispersion,
    /// Same difference on the conditional metric, which excludes the encoding tautology.
    pub conditional_difference: Dispersion,
    /// Per-DOF paired differences, in `[r, s, u, v, w]` order.
    pub dof_difference: [Dispersion; BAR_DOF],
    /// Pearson correlation of the two per-window vectors. This is what buys the paired
    /// design its power; below ~0.9 the pairing is barely helping and something differs
    /// between the runs beyond the change under test.
    pub correlation: f64,
    /// Windows on which the candidate scored worse.
    pub worse_windows: usize,
}

impl PairedComparison {
    /// True when zero lies outside the difference's 95% interval.
    pub fn significant(&self) -> bool {
        self.difference.ci_low.is_finite()
            && self.difference.ci_high.is_finite()
            && (self.difference.ci_low > 0.0 || self.difference.ci_high < 0.0)
    }
}

impl fmt::Display for PairedComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "paired comparison over {} identical pinned windows, scoring {}",
            self.windows, self.scoring
        )?;
        writeln!(
            f,
            "  baseline  {:<28} {:.4} nats/bar",
            self.baseline_run, self.baseline_mean
        )?;
        writeln!(
            f,
            "  candidate {:<28} {:.4} nats/bar",
            self.candidate_run, self.candidate_mean
        )?;
        writeln!(f, "  paired delta (candidate - baseline) {}", self.difference)?;
        writeln!(
            f,
            "  conditional delta (u,v scored only where s != 0) {}",
            self.conditional_difference
        )?;
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            writeln!(f, "  delta {name:<2} {}", self.dof_difference[dof])?;
        }
        writeln!(
            f,
            "  per-window correlation {:.4}, candidate worse on {} of {} windows",
            self.correlation, self.worse_windows, self.windows
        )?;
        writeln!(
            f,
            "  detectable at 80% power: {:.4} nats; verdict: {}",
            self.difference.minimum_detectable_effect(),
            if self.significant() {
                "SIGNIFICANT at 95%"
            } else {
                "not distinguishable from zero"
            }
        )
    }
}

/// Difference two runs' per-window vectors on the identical pinned windows.
///
/// Refuses anything that is not actually a pairing: a different corpus, different split
/// instants, a different evaluation seed, a different context, or a window list that does
/// not match element for element. Every one of those silently turns a paired comparison back
/// into an unpaired one, whose minimum detectable effect is an order of magnitude worse.
pub fn paired_comparison(
    baseline: &WindowScores,
    candidate: &WindowScores,
) -> Result<PairedComparison> {
    ensure!(
        baseline.corpus_fingerprint == candidate.corpus_fingerprint,
        "the two runs were scored on different corpora ({} vs {}); pin --split-bounds and \
         re-score, or compare levels with their own intervals and accept the ~0.41 nat MDE",
        &baseline.corpus_fingerprint[..12.min(baseline.corpus_fingerprint.len())],
        &candidate.corpus_fingerprint[..12.min(candidate.corpus_fingerprint.len())],
    );
    ensure!(
        baseline.split_bounds == candidate.split_bounds,
        "REFUSING to pair: the two runs were scored against different split instants. \
         baseline {} | {} ({} | {} ms); candidate {} | {} ({} | {} ms). The corpus grows \
         under running jobs, so the boundary drifts ~0.8 days per ingestion day. Pin \
         --split-bounds for the whole campaign and re-score, or compare the two levels with \
         their own intervals and accept the ~0.41 nat unpaired MDE.",
        iso_ms(baseline.split_bounds.0),
        iso_ms(baseline.split_bounds.1),
        baseline.split_bounds.0,
        baseline.split_bounds.1,
        iso_ms(candidate.split_bounds.0),
        iso_ms(candidate.split_bounds.1),
        candidate.split_bounds.0,
        candidate.split_bounds.1
    );
    ensure!(
        baseline.eval_window_seed == candidate.eval_window_seed,
        "evaluation window seeds differ: {:#x} vs {:#x}; the two runs were not scored on the \
         same windows",
        baseline.eval_window_seed,
        candidate.eval_window_seed
    );
    let scoring = match (baseline.scoring.as_deref(), candidate.scoring.as_deref()) {
        (Some(a), Some(b)) if a == b => a.to_owned(),
        (Some(a), Some(b)) => bail!(
            "REFUSING to pair: the two runs were scored under different rules ({a} vs {b}). \
             The bar scoring modes differ by additive constants that depend on the binning — \
             a density figure sits tens of nats below a hard one on the identical model — so \
             differencing them measures the rule, not the model. Re-score one arm with \
             --scoring {a}."
        ),
        (None, _) | (_, None) => bail!(
            "REFUSING to pair: at least one per-window vector does not record its scoring \
             rule (baseline {:?}, candidate {:?}). It was written before --scoring existed, \
             so which rule produced it cannot be established from the artifact; re-score \
             that arm rather than assuming.",
            baseline.scoring,
            candidate.scoring
        ),
    };
    ensure!(
        baseline.split == candidate.split && baseline.context == candidate.context,
        "the two runs were scored on different sets: {} at context {} vs {} at context {}",
        baseline.split,
        baseline.context,
        candidate.split,
        candidate.context
    );
    ensure!(
        baseline.windows.len() == candidate.windows.len(),
        "window counts differ: {} vs {}",
        baseline.windows.len(),
        candidate.windows.len()
    );
    for (index, (a, b)) in baseline
        .windows
        .iter()
        .zip(candidate.windows.iter())
        .enumerate()
    {
        if a.symbol != b.symbol || a.bar_index != b.bar_index {
            bail!(
                "window {index} differs: {}@{} vs {}@{}",
                a.symbol,
                a.bar_index,
                b.symbol,
                b.bar_index
            );
        }
    }

    let blocks = baseline.symbol_month_blocks();
    let base_nll = baseline.nll_bar();
    let cand_nll = candidate.nll_bar();
    let deltas: Vec<f64> = cand_nll
        .iter()
        .zip(base_nll.iter())
        .map(|(c, b)| c - b)
        .collect();
    let difference = block_bootstrap(&deltas, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);

    let conditional: Vec<f64> = candidate
        .nll_bar_conditional()
        .iter()
        .zip(baseline.nll_bar_conditional().iter())
        .map(|(c, b)| c - b)
        .collect();
    let conditional_difference =
        block_bootstrap(&conditional, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);

    let dof_difference = std::array::from_fn(|dof| {
        let per_dof: Vec<f64> = candidate
            .nll_dof(dof)
            .iter()
            .zip(baseline.nll_dof(dof).iter())
            .map(|(c, b)| c - b)
            .collect();
        block_bootstrap(&per_dof, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    });

    Ok(PairedComparison {
        baseline_run: baseline.run.clone(),
        candidate_run: candidate.run.clone(),
        windows: base_nll.len(),
        scoring,
        baseline_mean: base_nll.iter().sum::<f64>() / base_nll.len() as f64,
        candidate_mean: cand_nll.iter().sum::<f64>() / cand_nll.len() as f64,
        difference,
        conditional_difference,
        dof_difference,
        correlation: pearson(&base_nll, &cand_nll),
        worse_windows: deltas.iter().filter(|d| **d > 0.0).count(),
    })
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return f64::NAN;
    }
    cov / (var_a * var_b).sqrt()
}

/// Load two persisted vectors and print their paired comparison. This is the entry point
/// behind `trading_bot pretrain-compare`.
pub fn compare_runs(baseline: &Path, candidate: &Path) -> Result<PairedComparison> {
    let baseline = WindowScores::load(baseline)?;
    let candidate = WindowScores::load(candidate)?;
    paired_comparison(&baseline, &candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// `values[i] = level + block_effect[block(i)] + noise`, so the true standard error of
    /// the mean is dominated by the block effect and is analytically known.
    fn clustered(blocks: usize, per_block: usize, block_sd: f64, noise_sd: f64) -> (Vec<f64>, Vec<u64>) {
        let mut rng = ChaCha12Rng::seed_from_u64(0xC1057E4);
        let mut values = Vec::with_capacity(blocks * per_block);
        let mut ids = Vec::with_capacity(blocks * per_block);
        for block in 0..blocks {
            let effect: f64 = normal(&mut rng) * block_sd;
            for _ in 0..per_block {
                values.push(18.0 + effect + normal(&mut rng) * noise_sd);
                ids.push(block as u64);
            }
        }
        (values, ids)
    }

    fn normal(rng: &mut ChaCha12Rng) -> f64 {
        // Box-Muller; the bootstrap under test never sees this generator.
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// On data whose dispersion is known by construction, the block bootstrap must recover
    /// the CLUSTERED standard error, not the iid one — that difference is the entire reason
    /// this exists.
    ///
    /// 200 blocks of 20, block sd 1.0, noise sd 2.0. The true SE of the mean is
    /// `sqrt(1^2/200 + 2^2/4000) = 0.0775`, of which the block term contributes 83% of the
    /// variance and does NOT shrink with the 4000 observations. The naive iid formula
    /// `sd/sqrt(n) = sqrt(5)/sqrt(4000) = 0.0354` understates it 2.2x — the same shape as
    /// the real validation split, where ~4 shared calendar slots dominate 4096 windows.
    #[test]
    fn block_bootstrap_recovers_a_known_clustered_standard_error() {
        let (values, blocks) = clustered(200, 20, 1.0, 2.0);
        let d = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);

        assert_eq!(d.blocks, 200);
        assert_eq!(d.samples, 4000);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!((d.mean - mean).abs() < 1e-12, "{} != {mean}", d.mean);

        let truth = (1.0f64 / 200.0 + 4.0 / 4000.0).sqrt();
        assert!(
            (d.se / truth - 1.0).abs() < 0.15,
            "block bootstrap SE {:.5} is not within 15% of the analytic {truth:.5}",
            d.se
        );

        let iid_sd = {
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (values.len() - 1) as f64;
            var.sqrt() / (values.len() as f64).sqrt()
        };
        assert!(
            d.se > 1.8 * iid_sd,
            "the whole point is that clustering inflates the SE: {:.5} vs iid {iid_sd:.5}",
            d.se
        );

        // The interval brackets the mean and is roughly symmetric around it.
        assert!(d.ci_low < d.mean && d.mean < d.ci_high);
        let half = 0.5 * (d.ci_high - d.ci_low);
        assert!(
            (half / (1.96 * d.se) - 1.0).abs() < 0.2,
            "95% half-width {half:.5} is not ~1.96 SE ({:.5})",
            1.96 * d.se
        );
    }

    /// Ignoring the blocks must give the smaller, wrong answer. Same data, one block per
    /// observation.
    #[test]
    fn treating_every_window_as_its_own_block_understates_the_error() {
        let (values, blocks) = clustered(200, 20, 1.0, 2.0);
        let clustered_se = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED).se;
        let singleton: Vec<u64> = (0..values.len() as u64).collect();
        let iid_se = block_bootstrap(&values, &singleton, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED).se;
        assert!(
            clustered_se > 1.8 * iid_se,
            "clustered {clustered_se:.5} should dominate iid {iid_se:.5}"
        );
    }

    /// A single block carries no information about dispersion, and must say so rather than
    /// reporting a zero-width interval.
    #[test]
    fn a_single_block_reports_no_interval() {
        let d = block_bootstrap(&[1.0, 2.0, 3.0], &[7, 7, 7], BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        assert_eq!(d.blocks, 1);
        assert!((d.mean - 2.0).abs() < 1e-12);
        assert!(d.se.is_nan() && d.ci_low.is_nan());
    }

    /// The bootstrap is a property of the data, so the same vector must give the same
    /// interval every time it is reported.
    #[test]
    fn the_bootstrap_is_deterministic() {
        let (values, blocks) = clustered(40, 5, 0.3, 1.0);
        let a = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        let b = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        assert_eq!(a, b);
    }

    fn scores(run: &str, fingerprint: &str, offset: f64) -> WindowScores {
        let windows = (0..64)
            .map(|i| {
                let base = 18.0 + (i % 7) as f64 * 0.05 + offset;
                WindowScore {
                    symbol: format!("SYM{}", i % 8),
                    bar_index: 1000 + i as u32,
                    // Eight windows per calendar month, so the blocking has something to do.
                    ts_ms: 1_700_000_000_000 + (i as i64 / 8) * 30 * 86_400_000,
                    nll_dof: [base * 0.2; BAR_DOF],
                    nll_bar_conditional: base - 0.7,
                }
            })
            .collect();
        WindowScores {
            format_version: WINDOW_SCORES_FORMAT_VERSION,
            run: run.to_owned(),
            global_step: 9000,
            split: "val".to_owned(),
            context: 2048,
            eval_window_seed: 0xE7A1,
            corpus_fingerprint: fingerprint.to_owned(),
            split_bounds: (1_600_000_000_000, 1_650_000_000_000),
            marginal_nll_bar: 21.6686,
            scoring: Some("density".to_owned()),
            windows,
        }
    }

    /// The pairing must recover a constant shift exactly, and must refuse a comparison whose
    /// windows are not actually the same measurement.
    #[test]
    fn pairing_recovers_a_constant_shift_and_refuses_incomparable_runs() {
        let baseline = scores("base", "ff", 0.0);
        let candidate = scores("cand", "ff", -0.25);
        let paired = paired_comparison(&baseline, &candidate).expect("comparable");
        assert_eq!(paired.windows, 64);
        assert!(
            (paired.difference.mean + 0.25).abs() < 1e-9,
            "{}",
            paired.difference.mean
        );
        // A pure shift leaves zero residual dispersion, so the interval collapses onto it.
        assert!(paired.difference.se < 1e-9);
        assert!(paired.significant() || paired.difference.se.is_nan());
        assert!((paired.correlation - 1.0).abs() < 1e-9);
        assert_eq!(paired.worse_windows, 0);

        let other_corpus = scores("cand", "ee", -0.25);
        assert!(paired_comparison(&baseline, &other_corpus).is_err());

        let mut moved = scores("cand", "ff", -0.25);
        moved.windows[3].bar_index += 1;
        assert!(paired_comparison(&baseline, &moved).is_err());

        let mut reseeded = scores("cand", "ff", -0.25);
        reseeded.eval_window_seed = 0x1234;
        assert!(paired_comparison(&baseline, &reseeded).is_err());

        // Two runs scored under different rules are two different quantities: the density
        // and hard figures differ by the log measure, so differencing them measures the
        // rule. Refused, and so is a vector too old to say which rule it used.
        let mut other_rule = scores("cand", "ff", -0.25);
        other_rule.scoring = Some("hard".to_owned());
        let err = paired_comparison(&baseline, &other_rule)
            .expect_err("two scoring rules must not be paired")
            .to_string();
        assert!(err.contains("different rules"), "{err}");
        let mut unrecorded = scores("cand", "ff", -0.25);
        unrecorded.scoring = None;
        assert!(paired_comparison(&baseline, &unrecorded).is_err());
    }

    /// Month blocking must collapse the 64 windows onto the 8 calendar months they occupy,
    /// which is the coarse-but-honest unit for a statement about the absolute level.
    #[test]
    fn level_dispersion_blocks_by_calendar_month() {
        let scored = scores("base", "ff", 0.0);
        assert_eq!(scored.symbol_month_blocks().len(), 64);
        let months = scored.month_blocks();
        let distinct: std::collections::BTreeSet<u64> = months.iter().copied().collect();
        assert_eq!(distinct.len(), 8);
        assert_eq!(scored.level_dispersion().blocks, 8);
        assert!(scored.dispersion().blocks > scored.level_dispersion().blocks);
    }

    /// Identity must survive the round trip exactly — a pairing is only valid if the window
    /// list matches element for element. The scores themselves round-trip to well under a
    /// micro-nat, which serde_json's default float parser does not promise to the last bit
    /// and which is ~13 orders of magnitude below anything the campaign acts on.
    #[test]
    fn window_scores_round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_window_scores_{}",
            uuid::Uuid::new_v4()
        ));
        let path = dir.join("pretrain_best.windows.json");
        let scored = scores("base", "ff", 0.0);
        scored.save(&path).expect("save");
        let loaded = WindowScores::load(&path).expect("load");

        assert_eq!(loaded.windows.len(), scored.windows.len());
        for (got, want) in loaded.windows.iter().zip(scored.windows.iter()) {
            assert_eq!(got.symbol, want.symbol);
            assert_eq!(got.bar_index, want.bar_index);
            assert_eq!(got.ts_ms, want.ts_ms);
            for dof in 0..BAR_DOF {
                assert!((got.nll_dof[dof] - want.nll_dof[dof]).abs() < 1e-12);
            }
            assert!((got.nll_bar_conditional - want.nll_bar_conditional).abs() < 1e-12);
        }
        assert_eq!(loaded.corpus_fingerprint, scored.corpus_fingerprint);
        assert_eq!(loaded.split_bounds, scored.split_bounds);
        assert_eq!(loaded.eval_window_seed, scored.eval_window_seed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scores_path_sits_beside_the_checkpoint() {
        assert_eq!(
            window_scores_path(Path::new("runs/x/weights/pretrain_best.ot")),
            PathBuf::from("runs/x/weights/pretrain_best.windows.json")
        );
    }
}
