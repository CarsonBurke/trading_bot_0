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
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::torch::bar_dist::{BAR_DOF, BAR_DOF_NAMES};
use crate::torch::dataset::iso_ms;

use super::trade_bench::{self, TradeBench, COST_GRID_BPS, POLICY_NAMES};

/// Resamples per bootstrap. 1000 draws over ~4k f64 is microseconds, and the 2.5/97.5
/// percentiles of 1000 draws are stable to about a percent of the interval width.
pub const BOOTSTRAP_DRAWS: usize = 1000;

/// Fixed bootstrap stream. The interval is a property of the data, not of the run, so two
/// reports of the same vector must agree to the last digit.
pub const BOOTSTRAP_SEED: u64 = 0xB10C_B007_5EED_0001;

/// Two-sided interval the reported CI covers.
pub const CI_MASS: f64 = 0.95;

/// Schema of the persisted per-window vector. v2 replaces the mean-of-window conditional
/// score with the numerator and denominator of every per-DOF conditional ratio, so a pooled
/// point estimate and every bootstrap resample measure the same estimand. v3 adds the two
/// per-window economic vectors and the traded prefix's blocking, without which no ablation
/// arm can be judged economically, plus the bin-geometry digest that lets
/// [`paired_comparison`] refuse a pair whose supports were refitted.
pub const WINDOW_SCORES_FORMAT_VERSION: u32 = 3;

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
    moving_block_bootstrap(values, blocks, 1, draws, seed)
}

/// Circular moving-block bootstrap over the ordered distinct block IDs.
///
/// A draw samples consecutive groups in chunks of `block_len`, wrapping at the end, until
/// exactly the original number of groups has been selected. `Dispersion::blocks` reports the
/// resulting number of independently selected chunks, not the raw group count. Group totals
/// and counts travel together, preserving the natural observation weighting even when group
/// sizes differ. `values` may already be a paired difference series; resampling it directly
/// preserves the pairing without making either arm an independent draw.
pub fn moving_block_bootstrap(
    values: &[f64],
    blocks: &[u64],
    block_len: usize,
    draws: usize,
    seed: u64,
) -> Dispersion {
    assert_eq!(
        values.len(),
        blocks.len(),
        "every value needs a block assignment"
    );
    assert!(block_len >= 1, "moving block length must be at least one");
    let finite: Vec<(u64, f64)> = blocks
        .iter()
        .copied()
        .zip(values.iter().copied())
        .filter(|(_, value)| value.is_finite())
        .collect();
    if finite.is_empty() {
        return Dispersion::nan();
    }

    // BTreeMap order is the time order when callers use monotone calendar block IDs.
    let mut grouped: BTreeMap<u64, (f64, u64)> = BTreeMap::new();
    for (block, value) in &finite {
        let entry = grouped.entry(*block).or_insert((0.0, 0));
        entry.0 += *value;
        entry.1 += 1;
    }
    let totals: Vec<(f64, u64)> = grouped.into_values().collect();
    let samples = finite.len();
    let mean = totals.iter().map(|(sum, _)| *sum).sum::<f64>() / samples as f64;
    let resampling_blocks = totals.len() / block_len + usize::from(totals.len() % block_len != 0);
    if resampling_blocks < 2 || draws == 0 {
        // One effective chunk always carries the full circular sample, so a zero-width
        // interval would falsely present deterministic resampling as measured precision.
        return Dispersion {
            mean,
            se: f64::NAN,
            ci_low: f64::NAN,
            ci_high: f64::NAN,
            blocks: resampling_blocks,
            samples,
        };
    }

    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut sum = 0.0;
        let mut count = 0u64;
        if block_len == 1 {
            // Keep the ordinary block bootstrap's seeded draw sequence exactly unchanged.
            for _ in 0..totals.len() {
                let (block_sum, block_count) = totals
                    .choose(&mut rng)
                    .copied()
                    .expect("totals is non-empty");
                sum += block_sum;
                count += block_count;
            }
        } else {
            let mut selected = 0usize;
            while selected < totals.len() {
                let start = rng.random_range(0..totals.len());
                let chunk_len = block_len.min(totals.len() - selected);
                for offset in 0..chunk_len {
                    let (block_sum, block_count) = totals[(start + offset) % totals.len()];
                    sum += block_sum;
                    count += block_count;
                }
                selected += chunk_len;
            }
        }
        means.push(sum / count as f64);
    }
    means.sort_by(f64::total_cmp);

    let draw_mean = means.iter().sum::<f64>() / means.len() as f64;
    let variance = means
        .iter()
        .map(|sample_mean| (sample_mean - draw_mean).powi(2))
        .sum::<f64>()
        / (means.len() - 1) as f64;
    let tail = (1.0 - CI_MASS) / 2.0;
    Dispersion {
        mean,
        se: variance.sqrt(),
        ci_low: percentile(&means, tail),
        ci_high: percentile(&means, 1.0 - tail),
        blocks: resampling_blocks,
        samples,
    }
}
/// Paired block bootstrap of the pooled conditional-NLL difference.
///
/// Each draw resamples blocks once and applies that same multiplicity to both checkpoints,
/// then recomputes each checkpoint's sum of per-DOF ratios from its resampled numerators and
/// denominators. Differencing precomputed per-window ratios would target a mean-of-ratios
/// instead and can move both the point estimate and the promotion decision when live-bar
/// counts differ across windows.
pub fn block_bootstrap_conditional_difference(
    baseline: &[ConditionalNllStats],
    candidate: &[ConditionalNllStats],
    blocks: &[u64],
    draws: usize,
    seed: u64,
) -> Dispersion {
    assert_eq!(
        baseline.len(),
        candidate.len(),
        "paired score counts differ"
    );
    assert_eq!(
        candidate.len(),
        blocks.len(),
        "every conditional score needs a block assignment"
    );
    if candidate.is_empty() {
        return Dispersion::nan();
    }

    let mut grouped: BTreeMap<u64, (ConditionalNllStats, ConditionalNllStats, usize)> =
        BTreeMap::new();
    for ((base, cand), block) in baseline.iter().zip(candidate).zip(blocks) {
        let entry = grouped.entry(*block).or_insert((
            ConditionalNllStats::default(),
            ConditionalNllStats::default(),
            0,
        ));
        entry.0.absorb(base);
        entry.1.absorb(cand);
        entry.2 += 1;
    }
    let totals: Vec<_> = grouped.into_values().collect();
    let point = ConditionalNllStats::pooled(candidate).point_estimate()
        - ConditionalNllStats::pooled(baseline).point_estimate();
    if totals.len() < 2 || draws == 0 {
        return Dispersion {
            mean: point,
            se: f64::NAN,
            ci_low: f64::NAN,
            ci_high: f64::NAN,
            blocks: totals.len(),
            samples: candidate.len(),
        };
    }

    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut differences = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut base = ConditionalNllStats::default();
        let mut cand = ConditionalNllStats::default();
        for _ in 0..totals.len() {
            let (block_base, block_cand, _) = totals
                .choose(&mut rng)
                .expect("conditional bootstrap has at least one block");
            base.absorb(block_base);
            cand.absorb(block_cand);
        }
        differences.push(cand.point_estimate() - base.point_estimate());
    }
    differences.sort_by(f64::total_cmp);
    let draw_mean = differences.iter().sum::<f64>() / differences.len() as f64;
    let variance = differences
        .iter()
        .map(|value| (value - draw_mean).powi(2))
        .sum::<f64>()
        / (differences.len() - 1) as f64;
    let tail = (1.0 - CI_MASS) / 2.0;
    Dispersion {
        mean: point,
        se: variance.sqrt(),
        ci_low: percentile(&differences, tail),
        ci_high: percentile(&differences, 1.0 - tail),
        blocks: totals.len(),
        samples: candidate.len(),
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

/// Per-window sufficient statistics for the encoding-adjusted conditional NLL.
///
/// The `r`, `s`, and `w` factors use every bar, while `u` and `v` use only live bars. Their
/// pooled bar-level estimand is therefore a sum of five ratios, not a mean of per-window
/// ratios. Keeping both sides of each ratio is what lets a block-bootstrap resample recompute
/// that exact estimand after windows have been repeated or omitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConditionalNllStats {
    pub numerator: [f64; BAR_DOF],
    pub denominator: [f64; BAR_DOF],
}

impl ConditionalNllStats {
    pub fn point_estimate_dof(&self) -> [f64; BAR_DOF] {
        std::array::from_fn(|dof| {
            if self.denominator[dof] > 0.0 {
                self.numerator[dof] / self.denominator[dof]
            } else {
                0.0
            }
        })
    }

    pub fn point_estimate(&self) -> f64 {
        self.point_estimate_dof().iter().sum()
    }

    pub fn absorb(&mut self, other: &Self) {
        for dof in 0..BAR_DOF {
            self.numerator[dof] += other.numerator[dof];
            self.denominator[dof] += other.denominator[dof];
        }
    }

    pub fn pooled(stats: &[Self]) -> Self {
        let mut pooled = Self::default();
        for row in stats {
            pooled.absorb(row);
        }
        pooled
    }
}

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
    /// Sufficient statistics for the displayed pooled conditional NLL. A window-local ratio
    /// is deliberately not persisted: averaging those ratios would overweight windows with
    /// few live bars and would make promotion guard a different quantity than it displays.
    pub conditional_nll: ConditionalNllStats,
}

impl WindowScore {
    pub fn nll_bar(&self) -> f64 {
        self.nll_dof.iter().sum()
    }
}

/// A JSON-safe `f64`.
///
/// `serde_json` maps every non-finite float to `null` on the way out and then refuses to
/// read it back as an `f64`, which would make an epoch sidecar unloadable the moment the
/// bench reported one. Non-finite is not an error here: `break_even_bps` is `NaN` when
/// there is no gross edge to lose and `+inf` when no cost ever removes it, and those are
/// DIFFERENT findings that a shared `null` would merge. So the three non-finite values
/// round-trip through their own names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JsonF64(pub f64);

impl From<f64> for JsonF64 {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl fmt::Display for JsonF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Serialize for JsonF64 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            serializer.serialize_f64(self.0)
        } else if self.0.is_nan() {
            serializer.serialize_str("nan")
        } else if self.0 > 0.0 {
            serializer.serialize_str("inf")
        } else {
            serializer.serialize_str("-inf")
        }
    }
}

impl<'de> Deserialize<'de> for JsonF64 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Number(f64),
            Name(String),
            Absent,
        }
        Ok(Self(match Repr::deserialize(deserializer)? {
            Repr::Number(value) => value,
            // `null` is what an older writer, or `serde_json` itself, leaves behind for a
            // non-finite value; it cannot say WHICH one, so it reads back as `NaN`.
            Repr::Absent => f64::NAN,
            Repr::Name(name) => match name.as_str() {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => {
                    return Err(serde::de::Error::custom(format!(
                        "expected a number or one of `nan`, `inf`, `-inf`, got `{other}`"
                    )))
                }
            },
        }))
    }
}

/// One policy's realized performance and its own paired verdict against the null, as an epoch
/// artifact records it.
///
/// Mirrors `trade_bench::PolicyStats` field for field, plus this policy's row of the bench's
/// per-policy edge, break-even and ceiling share. It is a separate type rather than a derive
/// on the bench's own struct because every float here has to survive JSON, and because the
/// persisted schema must not move whenever the bench grows an internal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TradePolicySummary {
    /// `trade_bench::POLICY_NAMES` entry, so the file is readable without the index table.
    pub policy: String,
    pub gross_growth: JsonF64,
    pub net_growth: JsonF64,
    pub sharpe: JsonF64,
    pub hit_rate: JsonF64,
    pub turnover: JsonF64,
    pub time_in_market: JsonF64,
    pub mean_abs_position: JsonF64,
    pub clamped_fraction: JsonF64,
    pub mean_drawdown: JsonF64,
    pub max_drawdown: JsonF64,
    pub ruin_bars: usize,
    /// This policy's paired `policy - marginal null` net log growth per bar, with the same
    /// block-bootstrap interval the headline uses. Present per policy so a fractional-Kelly
    /// row can be quoted as the verdict without anyone re-deriving its interval by hand; the
    /// null's own row is identically zero, by construction.
    pub edge: JsonF64,
    pub edge_se: JsonF64,
    pub edge_ci_low: JsonF64,
    pub edge_ci_high: JsonF64,
    /// Cost at which this policy's edge reaches zero. `nan` = no gross edge, `inf` = never.
    pub break_even_bps: JsonF64,
    pub ceiling_capture: JsonF64,
    /// This policy's edge at each [`TradeSummary::cost_grid_bps`] level.
    pub cost_curve: Vec<JsonF64>,
}

/// The moment-correct quadratic Kelly bench of one pinned pass, persisted beside the
/// checkpoint it scored. Model and marginal positions use `E[R] / E[R²]` reduced from
/// train-fitted within-bin moments; the oracle remains a separate realized-return ceiling.
///
/// Growth figures are realized natural log growth PER BAR, not the basis points charts draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TradeSummary {
    pub policies: Vec<TradePolicySummary>,
    /// Paired `model - marginal null` net log growth per bar: THE number.
    pub edge: JsonF64,
    pub edge_se: JsonF64,
    pub edge_ci_low: JsonF64,
    pub edge_ci_high: JsonF64,
    /// Resampling units behind the interval, which is what its width is governed by.
    pub edge_blocks: usize,
    /// Cost at which `edge` reaches zero. `nan` = no gross edge, `inf` = never.
    pub break_even_bps: JsonF64,
    pub ceiling_capture: JsonF64,
    /// The cost axis `cost_curve` is evaluated on, so the pair is self-describing.
    pub cost_grid_bps: Vec<JsonF64>,
    pub cost_curve: Vec<JsonF64>,
    pub cost_bps: JsonF64,
    pub leverage_cap: JsonF64,
    pub bars: usize,
    pub windows: usize,
    pub blocks: usize,
    /// One entry per [`trade_bench::CAP_GRID`] point: the same verdict re-derived at each
    /// leverage cap. Persisted because the headline is only interpretable beside it — an
    /// edge that vanishes as the cap falls was bought with leverage, not with prediction.
    pub cap_curve: Vec<TradeCapPointSummary>,
    /// Distribution of the UNCAPPED optimum `|f*|` over traded bars, and the share of bars
    /// the cap rather than the distribution sized.
    pub free_kelly_bucket_floor: Vec<JsonF64>,
    pub free_kelly_share: Vec<JsonF64>,
    pub free_kelly_saturated: JsonF64,
    pub free_kelly_median: JsonF64,
    pub free_kelly_p95: JsonF64,
    pub free_kelly_mean_signed: JsonF64,
    /// Far-tail calibration of the traded law, lower tail then upper, one entry per
    /// [`trade_bench::TAIL_LEVELS`] level.
    pub tail_lower: Vec<TradeTailSummary>,
    pub tail_upper: Vec<TradeTailSummary>,
    pub tail_bars: JsonF64,
    pub tail_windows: usize,
}

impl From<&TradeBench> for TradeSummary {
    fn from(bench: &TradeBench) -> Self {
        Self {
            policies: POLICY_NAMES
                .iter()
                .enumerate()
                .map(|(policy, name)| {
                    let stats = &bench.policies[policy];
                    let edge = bench.edge[policy];
                    TradePolicySummary {
                        policy: (*name).to_owned(),
                        gross_growth: stats.gross_growth.into(),
                        net_growth: stats.net_growth.into(),
                        sharpe: stats.sharpe.into(),
                        hit_rate: stats.hit_rate.into(),
                        turnover: stats.turnover.into(),
                        time_in_market: stats.time_in_market.into(),
                        mean_abs_position: stats.mean_abs_position.into(),
                        clamped_fraction: stats.clamped_fraction.into(),
                        mean_drawdown: stats.mean_drawdown.into(),
                        max_drawdown: stats.max_drawdown.into(),
                        ruin_bars: stats.ruin_bars,
                        edge: edge.mean.into(),
                        edge_se: edge.se.into(),
                        edge_ci_low: edge.ci_low.into(),
                        edge_ci_high: edge.ci_high.into(),
                        break_even_bps: bench.break_even_bps[policy].into(),
                        ceiling_capture: bench.ceiling_capture[policy].into(),
                        cost_curve: bench.cost_curve[policy]
                            .iter()
                            .map(|edge| JsonF64(*edge))
                            .collect(),
                    }
                })
                .collect(),
            edge: bench.model_edge().mean.into(),
            edge_se: bench.model_edge().se.into(),
            edge_ci_low: bench.model_edge().ci_low.into(),
            edge_ci_high: bench.model_edge().ci_high.into(),
            edge_blocks: bench.model_edge().blocks,
            break_even_bps: bench.model_break_even().into(),
            ceiling_capture: bench.model_capture().into(),
            cost_grid_bps: COST_GRID_BPS.iter().map(|bps| JsonF64(*bps)).collect(),
            cost_curve: bench
                .model_cost_curve()
                .iter()
                .map(|edge| JsonF64(*edge))
                .collect(),
            cost_bps: bench.cost_bps.into(),
            leverage_cap: bench.leverage_cap.into(),
            bars: bench.bars,
            windows: bench.windows,
            blocks: bench.blocks,
            cap_curve: bench.cap_curve.iter().map(Into::into).collect(),
            free_kelly_bucket_floor: trade_bench::FREE_KELLY_EDGES
                [..trade_bench::FREE_KELLY_EDGES.len() - 1]
                .iter()
                .map(|edge| JsonF64(*edge))
                .collect(),
            free_kelly_share: bench
                .free_kelly
                .histogram
                .iter()
                .map(|share| JsonF64(*share))
                .collect(),
            free_kelly_saturated: bench.free_kelly.saturated.into(),
            free_kelly_median: bench.free_kelly.median.into(),
            free_kelly_p95: bench.free_kelly.p95.into(),
            free_kelly_mean_signed: bench.free_kelly.mean_signed.into(),
            tail_lower: bench.tail.lower.iter().map(Into::into).collect(),
            tail_upper: bench.tail.upper.iter().map(Into::into).collect(),
            tail_bars: bench.tail.bars.into(),
            tail_windows: bench.tail.windows,
        }
    }
}

/// One leverage cap's re-derived verdict, as the artifact records it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TradeCapPointSummary {
    pub cap: JsonF64,
    pub edge: JsonF64,
    pub break_even_bps: JsonF64,
    pub sharpe: JsonF64,
    pub ceiling_capture: JsonF64,
    pub mean_abs_position: JsonF64,
    pub clamped_fraction: JsonF64,
    pub max_drawdown: JsonF64,
    pub ruin_bars: usize,
}

/// One tail level's promise against what happened.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TradeTailSummary {
    pub nominal: JsonF64,
    pub realized: JsonF64,
    /// `realized / nominal`: one is honest, four means a wipeout the model called impossible.
    pub ratio: JsonF64,
    pub exceedances: JsonF64,
    /// Binomial interval treating bars as independent: a FLOOR on the uncertainty.
    pub wilson_low: JsonF64,
    pub wilson_high: JsonF64,
    /// Window-blocked interval, the honest one, in the same units as `realized`.
    pub blocked_low: JsonF64,
    pub blocked_high: JsonF64,
}

impl From<&trade_bench::CapPoint> for TradeCapPointSummary {
    fn from(point: &trade_bench::CapPoint) -> Self {
        Self {
            cap: point.cap.into(),
            edge: point.edge.into(),
            break_even_bps: point.break_even_bps.into(),
            sharpe: point.sharpe.into(),
            ceiling_capture: point.ceiling_capture.into(),
            mean_abs_position: point.mean_abs_position.into(),
            clamped_fraction: point.clamped_fraction.into(),
            max_drawdown: point.max_drawdown.into(),
            ruin_bars: point.ruin_bars,
        }
    }
}

impl From<&trade_bench::TailPoint> for TradeTailSummary {
    fn from(point: &trade_bench::TailPoint) -> Self {
        Self {
            nominal: point.nominal.into(),
            realized: point.realized.into(),
            ratio: point.ratio.into(),
            exceedances: point.exceedances.into(),
            wilson_low: point.wilson.0.into(),
            wilson_high: point.wilson.1.into(),
            blocked_low: point.blocked.0.into(),
            blocked_high: point.blocked.1.into(),
        }
    }
}

/// The Mincer-Zarnowitz calibration of the traded law, reduced to the three numbers a
/// cross-run comparison needs.
///
/// A slope is a ratio of a realized quantity to a predicted one, so it is DIMENSIONLESS and
/// invariant to the bin geometry the prediction was decoded through. That makes it the one
/// calibration ruler that still means the same thing after an arm refits the supports, which
/// is why it is persisted at every validation rather than only by the offline
/// `pretrain-calibration` command.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSlopes {
    /// Slope of realized `r` on the traded conditional mean. `1.0` is perfect calibration;
    /// below one says the forecast moves more than the outcome it predicts, which is
    /// exactly the overstated edge a Kelly bettor pays for quadratically.
    pub mean_beta: JsonF64,
    /// Block-bootstrap standard error of [`Self::mean_beta`], over the traded prefix's
    /// `(symbol, month)` blocks.
    pub mean_beta_se: JsonF64,
    /// Slope of the realized squared residual on the predicted variance. Below one says the
    /// predicted variance is too large, i.e. the law is under-confident and under-leverages;
    /// above one says it is too confident. A mean-only recalibration cannot correct either.
    pub variance_beta: JsonF64,
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
    /// Position-sizing rule active when the artifact was scored.
    ///
    /// This is artifact identity and may deliberately differ between an A/B pair: sizing
    /// policy is the treatment in A1, not a comparability precondition. `None` on artifacts
    /// written before the rule was persisted.
    #[serde(default)]
    pub sizing_rule: Option<String>,
    pub windows: Vec<WindowScore>,
    /// The Kelly trading bench measured on THIS pass, when the caller ran one.
    ///
    /// Present on the epoch-boundary artifacts, whose whole purpose is to be evaluable
    /// without being re-scored: an epoch checkpoint that carries its per-window NLL vector
    /// but not what those windows were WORTH still forces a reload to answer the only
    /// question a reader has. Absent — not zero — on every sidecar written from a pass the
    /// bench did not ride.
    #[serde(default)]
    pub trade: Option<TradeSummary>,
    /// Base batch the run ACTUALLY executed, and the total step count priced from it.
    ///
    /// `None` on a vector written before these were recorded, which is NOT the same as
    /// "matched": it means the pair cannot be checked, and [`paired_comparison`] refuses it
    /// rather than assuming, exactly as it does for [`Self::scoring`].
    ///
    /// # Why an A/B needs these and why `--batch-size` is not enough
    ///
    /// The startup capacity probe clamps the base batch to what the card can hold, which
    /// depends on what ELSE was resident at launch. Measured, on two arms of the
    /// expected-log-growth ablation launched identically at `--batch-size 24` with the same
    /// seed: 16.37 GiB free gave base 23 and 10818 steps, 14.94 GiB gave base 21 and 11847.
    /// The bar budget is fixed by `--epochs`, so a smaller batch buys MORE steps — a different
    /// gradient-noise level and a different-length lr and momentum schedule. Both runs
    /// REQUESTED 24, so the requested figure cannot detect it and only the realized one can.
    /// [`super::pretrain::PretrainArgs::exact_batch`] prevents it at launch; this refuses the
    /// pair afterwards, for the runs that were launched before anyone knew to set it.
    #[serde(default)]
    pub realized_batch: Option<usize>,
    #[serde(default)]
    pub realized_steps: Option<usize>,
    /// SHA-256 of the bin supports this run was scored against, i.e. the identity of the
    /// DISCRETIZATION every nats figure here lives in.
    ///
    /// The same digest [`crate::torch::world_model::BarWorldModelMetadata::supports_sha256`]
    /// records beside the weights. Nothing else on this struct is a function of the bin
    /// geometry — not the corpus fingerprint, not the split instants, not the scoring rule,
    /// which is a rule NAME — so without this a run whose supports were refitted pairs
    /// cleanly against one whose were not and returns a meaningless nats difference with a
    /// tight interval. `None` on a pre-v3 vector, which is NOT "same geometry":
    /// [`paired_comparison`] refuses it rather than assuming.
    #[serde(default)]
    pub supports_sha256: Option<String>,
    /// Per-window net Kelly edge over the unconditional-marginal null at the selection cap,
    /// in BPS per bar, over the TRADED PREFIX of [`Self::windows`] in pinned order.
    ///
    /// This is the vector selection itself reads, persisted so a cross-run comparison can
    /// re-difference it instead of differencing two already-collapsed aggregates. The null
    /// leg is built from the bin supports, so the DIFFERENCE is only meaningful within one
    /// [`Self::supports_sha256`]; [`Self::model_growth_bps`] is the leg that is not.
    ///
    /// `None` on a pre-v3 vector, which is NOT "no edge".
    #[serde(default)]
    pub selection_edge_bps: Option<Vec<f64>>,
    /// The MODEL LEG ALONE of the same measurement: net log growth per bar of the capped
    /// model policy at the same cap and cost, in BPS, with no null subtracted.
    ///
    /// Pure realized-return arithmetic over the positions the model took, so it is the only
    /// economic ruler that survives a change of bin geometry and the only one an arm that
    /// refits the supports may be judged on. `None` on a pre-v3 vector.
    #[serde(default)]
    pub model_growth_bps: Option<Vec<f64>>,
    /// `(symbol, calendar month)` block ids of the traded prefix, in the same order as the
    /// two vectors above.
    ///
    /// [`Self::symbol_month_blocks`] truncated to the traded length, persisted rather than
    /// re-derived so a consumer bootstraps the economics over exactly the resampling units
    /// the run's own selection used. `None` on a pre-v3 vector.
    #[serde(default)]
    pub traded_blocks: Option<Vec<u64>>,
    /// Mincer-Zarnowitz calibration of the traded law on THIS pass.
    ///
    /// Absent — not zeroed — on a pass that ran no bench, and on every pre-v3 vector.
    #[serde(default)]
    pub calibration: Option<CalibrationSlopes>,
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

    /// The pooled encoding-adjusted conditional NLL displayed by evaluation and consumed by
    /// selection. This is a ratio-of-sums within each DOF, then a sum across DOFs.
    pub fn conditional_nll(&self) -> f64 {
        let mut pooled = ConditionalNllStats::default();
        for window in &self.windows {
            pooled.absorb(&window.conditional_nll);
        }
        pooled.point_estimate()
    }

    pub fn conditional_nll_stats(&self) -> Vec<ConditionalNllStats> {
        self.windows
            .iter()
            .map(|window| window.conditional_nll)
            .collect()
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

/// Which sign of a `candidate - baseline` difference favours the candidate.
///
/// The predictive and economic differences below run in OPPOSITE directions — an NLL is a
/// loss and an edge is a gain — and a comparator whose reader has to remember which is which
/// is a comparator that will eventually be read backwards. So no difference is printed
/// without its own direction, and [`Self::advantage`] is the one place the flip happens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BetterWhen {
    /// Loss-like, so a NEGATIVE difference favours the candidate: every nats figure.
    Negative,
    /// Gain-like, so a POSITIVE difference favours the candidate: every bps figure.
    Positive,
}

impl BetterWhen {
    /// The candidate's advantage over the baseline in "more is better" orientation,
    /// whatever the raw difference's sign convention is.
    pub fn advantage(self, difference: f64) -> f64 {
        match self {
            Self::Negative => -difference,
            Self::Positive => difference,
        }
    }

    fn direction(self) -> &'static str {
        match self {
            Self::Negative => "NEGATIVE is better",
            Self::Positive => "POSITIVE is better",
        }
    }
}

/// Two runs differenced window by window.
#[derive(Clone, Debug)]
pub struct PairedComparison {
    pub baseline_run: String,
    pub candidate_run: String,
    /// Sizing identities are displayed but never required to match: a sizing rule may be the
    /// deliberate treatment.
    pub baseline_sizing_rule: Option<String>,
    pub candidate_sizing_rule: Option<String>,
    pub windows: usize,
    /// Scoring rule both runs were measured under. Every nats figure below is in that
    /// rule's units and is comparable to nothing measured under another.
    pub scoring: String,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    /// `candidate - baseline` in nats/bar. NEGATIVE means the candidate is better; see
    /// [`BetterWhen::Negative`], and contrast the two economic differences below, whose
    /// convention is the opposite one.
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
    /// `candidate - baseline` on [`WindowScores::selection_edge_bps`], in BPS per bar over
    /// the traded prefix, block-bootstrapped over that prefix's own `(symbol, month)` units.
    ///
    /// POSITIVE means the candidate is better — the OPPOSITE of [`Self::difference`],
    /// because an edge is a gain and a negative log-likelihood is a loss. See
    /// [`BetterWhen::Positive`].
    ///
    /// `None` when either side is a pre-v3 vector or recorded no traded windows. Valid only
    /// within one bin geometry, which [`paired_comparison`] has already enforced.
    pub edge_difference: Option<Dispersion>,
    /// The same paired difference on [`WindowScores::model_growth_bps`], the model leg with
    /// no support-derived null subtracted. POSITIVE means the candidate is better.
    ///
    /// This is the economic difference that stays meaningful when an arm changes the bin
    /// geometry, because nothing about it is decoded through the supports.
    pub model_growth_difference: Option<Dispersion>,
    /// The two runs' bin geometries, when they DIFFER and the operator asserted
    /// [`GeometryPolicy::ModelGrowthOnly`], as `(baseline, candidate)`.
    ///
    /// Every nats field above, and [`Self::edge_difference`], is then withheld:
    /// `Dispersion::nan()`, `NaN`, or `None`. [`Self::predictive_measured`] is the predicate
    /// to read before touching any of them. `None` on every ordinary comparison, because
    /// [`GeometryPolicy::Require`] refuses a geometry change rather than reporting one — so
    /// this being `Some` is itself the record that an operator asserted the change.
    pub geometry_change: Option<(String, String)>,
}

impl PairedComparison {
    /// True when the nats figures were measured, i.e. the two runs shared one bin geometry.
    ///
    /// Read this before [`Self::difference`], [`Self::conditional_difference`],
    /// [`Self::dof_difference`], the two means or [`Self::correlation`]: a
    /// [`GeometryPolicy::ModelGrowthOnly`] comparison leaves every one of them unmeasured.
    pub fn predictive_measured(&self) -> bool {
        self.geometry_change.is_none()
    }

    /// True when zero lies outside the difference's 95% interval.
    pub fn significant(&self) -> bool {
        Self::resolvable(&self.difference)
    }

    fn resolvable(dispersion: &Dispersion) -> bool {
        dispersion.ci_low.is_finite()
            && dispersion.ci_high.is_finite()
            && (dispersion.ci_low > 0.0 || dispersion.ci_high < 0.0)
    }

    /// One difference line: the interval, the direction its sign is read in, the candidate's
    /// advantage in "more is better" orientation, and the MDE the interval implies.
    ///
    /// A difference that does not exist prints WHY. Skipping the row would let a stale
    /// artifact, or a metric withheld across a geometry change, read as an arm with no
    /// economic effect.
    fn write_difference(
        f: &mut fmt::Formatter<'_>,
        label: &str,
        units: &str,
        difference: Option<&Dispersion>,
        better: BetterWhen,
    ) -> fmt::Result {
        let Some(difference) = difference else {
            return Self::write_withheld(
                f,
                label,
                units,
                &format!("pre-v{WINDOW_SCORES_FORMAT_VERSION} artifact, or a pass with no traded windows"),
            );
        };
        writeln!(
            f,
            "  {label} [{units}, candidate - baseline, {}] {difference}; candidate advantage \
             {:+.4}, MDE {:.4}, verdict: {}",
            better.direction(),
            better.advantage(difference.mean),
            difference.minimum_detectable_effect(),
            if Self::resolvable(difference) {
                "SIGNIFICANT at 95%"
            } else {
                "not distinguishable from zero"
            }
        )
    }

    /// A metric this comparison may not state, and the reason. One uniform form, so a reader
    /// or a driver sees an explicit verdict per metric rather than a missing line.
    fn write_withheld(
        f: &mut fmt::Formatter<'_>,
        label: &str,
        units: &str,
        reason: &str,
    ) -> fmt::Result {
        writeln!(f, "  {label} [{units}]: NOT COMPARABLE ({reason})")
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
            "  sizing rule: baseline {}; candidate {}",
            self.baseline_sizing_rule.as_deref().unwrap_or("unrecorded"),
            self.candidate_sizing_rule
                .as_deref()
                .unwrap_or("unrecorded")
        )?;
        match &self.geometry_change {
            None => {
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
                writeln!(
                    f,
                    "  paired delta [nats/bar, candidate - baseline, {}] {}",
                    BetterWhen::Negative.direction(),
                    self.difference
                )?;
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
                )?;
                // The economics print with the opposite sign convention spelled out on every
                // line, because these are the only rows an economic ablation may be culled on
                // and reading one of them backwards inverts the decision.
                Self::write_difference(
                    f,
                    "selection edge delta",
                    "bps/bar",
                    self.edge_difference.as_ref(),
                    BetterWhen::Positive,
                )?;
            }
            // Named at the point of use, because months from now the ledger has to be
            // readable without the conversation that produced this row.
            Some((baseline_supports, candidate_supports)) => {
                writeln!(
                    f,
                    "  BIN GEOMETRY CHANGED and --allow-geometry-change was given: baseline \
                     supports {baseline_supports}, candidate supports {candidate_supports}. \
                     Every quantity decoded through the bins is WITHHELD rather than \
                     corrected, because no correction makes two discretizations' log \
                     densities comparable and a corrected number would look usable. `model \
                     growth delta` is the ONLY valid cull metric below."
                )?;
                for (label, units) in [
                    ("paired delta", "nats/bar"),
                    ("conditional delta", "nats/bar"),
                    ("per-DOF deltas", "nats/bar"),
                    ("per-window correlation", "unitless"),
                    // Its null leg is `marginal_position(&supports, ..)`, so it is as
                    // support-derived as the nats rows and is a CULL metric — the single most
                    // likely way someone would judge a geometry arm on a meaningless number.
                    ("selection edge delta", "bps/bar"),
                ] {
                    Self::write_withheld(f, label, units, "forbidden across bin geometries")?;
                }
            }
        }
        Self::write_difference(
            f,
            "model growth delta",
            "bps/bar",
            self.model_growth_difference.as_ref(),
            BetterWhen::Positive,
        )
    }
}

/// Whether a comparison may proceed across a change of bin geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryPolicy {
    /// Refuse the pair outright. The default, and the only correct choice for an arm that did
    /// not deliberately refit the supports.
    Require,
    /// Proceed on the geometry-free quantities ALONE, for an arm whose treatment IS the
    /// discretization.
    ///
    /// Every figure that passes through the supports is WITHHELD, never adjusted: the nats
    /// levels and deltas, all five per-DOF deltas, the per-window correlation, and the
    /// selection edge, whose null leg is `marginal_position(&supports, ..)`. There is no
    /// correction that makes two discretizations' log densities comparable, and a corrected
    /// number would be worse than none because it would look usable. What survives is
    /// [`PairedComparison::model_growth_difference`], which is realized-return arithmetic
    /// over the positions taken and never touches a bin.
    ModelGrowthOnly,
}

/// Difference two runs' per-window vectors on the identical pinned windows.
///
/// Refuses anything that is not actually a pairing: a different corpus, a different bin
/// geometry, different split instants, a different evaluation seed, a different context, a
/// different traded prefix, or a window list that does not match element for element. Every
/// one of those silently turns a paired comparison back into an unpaired one, whose minimum
/// detectable effect is an order of magnitude worse — or, for the geometry, into no
/// measurement of the model at all.
pub fn paired_comparison(
    baseline: &WindowScores,
    candidate: &WindowScores,
) -> Result<PairedComparison> {
    paired_comparison_with(baseline, candidate, GeometryPolicy::Require)
}

/// [`paired_comparison`] with an explicit bin-geometry policy. Every other refusal is
/// unchanged: `geometry` relaxes the supports check ALONE, and only in the direction of
/// reporting less.
pub fn paired_comparison_with(
    baseline: &WindowScores,
    candidate: &WindowScores,
    geometry: GeometryPolicy,
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
    // The deepest refusal here, and the one nothing else on the artifact stands in for.
    // `corpus_fingerprint` names the BARS and `scoring` names the RULE; neither is a function
    // of the discretization the rule is evaluated on, so a refitted-support arm passes every
    // other check and returns a meaningless nats difference with a tight interval. Precedent:
    // `mem_probe::assert_one_geometry`, which already refuses this for checkpoints.
    let geometry_change = match (
        baseline.supports_sha256.as_deref(),
        candidate.supports_sha256.as_deref(),
    ) {
        (Some(a), Some(b)) if a == b => None,
        (Some(a), Some(b)) => match geometry {
            GeometryPolicy::Require => bail!(
                "REFUSING to pair: the two runs were scored against DIFFERENT BIN GEOMETRIES \
                 — baseline supports {a}, candidate supports {b}. An NLL is the log mass the \
                 model put on the bin the outcome fell in, so refitting the supports moves \
                 the whole scale by a binning-dependent constant, and differencing across the \
                 two measures the discretization rather than the model. If the discretization \
                 IS the treatment, re-run with --allow-geometry-change, which reports the \
                 model growth leg alone; otherwise re-score one arm against the other's \
                 supports with --supports <path> --freeze-supports."
            ),
            GeometryPolicy::ModelGrowthOnly => Some((a.to_owned(), b.to_owned())),
        },
        // Unknown geometry is never treated as matching, under EITHER policy: the operator
        // can assert that the supports changed, but nobody can assert what they were.
        (None, _) | (_, None) => bail!(
            "REFUSING to pair: at least one per-window vector does not record the bin \
             geometry it was scored against (baseline {:?}, candidate {:?}). It was written \
             before format v{WINDOW_SCORES_FORMAT_VERSION}, so whether the two arms share a \
             discretization cannot be established from the artifact — and assuming they do is \
             exactly what would let a refitted-support arm read as a real gain. Re-score that \
             arm rather than assuming.",
            baseline.supports_sha256,
            candidate.supports_sha256
        ),
    };
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
    match (
        baseline.realized_batch.zip(baseline.realized_steps),
        candidate.realized_batch.zip(candidate.realized_steps),
    ) {
        (Some(a), Some(b)) if a == b => {}
        (Some((ab, asteps)), Some((bb, bsteps))) => bail!(
            "REFUSING to pair: the two runs executed different schedules — baseline base \
             batch {ab} over {asteps} steps, candidate {bb} over {bsteps}. The startup \
             capacity probe clamps the base batch to whatever the card had free, and a \
             fixed bar budget then buys a different number of steps, so the pair differs \
             in gradient noise and in lr/momentum schedule length as well as in whatever \
             was being tested. Relaunch both arms with --exact-batch so a card that cannot \
             hold the requested batch refuses instead of quietly reducing it."
        ),
        (None, _) | (_, None) => bail!(
            "REFUSING to pair: at least one per-window vector does not record the batch and \
             step count its run actually executed (baseline {:?}/{:?}, candidate {:?}/{:?}). \
             It was written before those were recorded, so whether the two arms ran the same \
             schedule cannot be established from the artifact; re-score that arm rather than \
             assuming it matched.",
            baseline.realized_batch,
            baseline.realized_steps,
            candidate.realized_batch,
            candidate.realized_steps
        ),
    }
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
    // Across a bin-geometry change every nats figure is WITHHELD rather than computed. There
    // is no correction that makes two discretizations' log densities comparable, and a number
    // in these fields would be read as one; `Dispersion::nan()` is the house idiom for a
    // quantity this pass did not measure. `PairedComparison::predictive_measured` reports it.
    let predictive = geometry_change.is_none();
    let difference = if predictive {
        block_bootstrap(&deltas, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    } else {
        Dispersion::nan()
    };

    let conditional_difference = if predictive {
        block_bootstrap_conditional_difference(
            &baseline.conditional_nll_stats(),
            &candidate.conditional_nll_stats(),
            &blocks,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        )
    } else {
        Dispersion::nan()
    };

    let dof_difference = std::array::from_fn(|dof| {
        if !predictive {
            return Dispersion::nan();
        }
        let per_dof: Vec<f64> = candidate
            .nll_dof(dof)
            .iter()
            .zip(baseline.nll_dof(dof).iter())
            .map(|(c, b)| c - b)
            .collect();
        block_bootstrap(&per_dof, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    });
    // The bench trades a PREFIX of the pinned set, so two arms can agree on every pinned
    // window and still have priced two different books. Refused rather than reported as an
    // absent difference: "not recorded" reads as "this arm had no economic effect", which is
    // the same silent-invalid-comparison hazard the geometry check above exists for.
    let traded_blocks = match (
        baseline.traded_blocks.as_deref(),
        candidate.traded_blocks.as_deref(),
    ) {
        (Some(a), Some(b)) if a == b => a,
        (Some(a), Some(b)) => bail!(
            "REFUSING to pair: the two runs traded different prefixes of the identical pinned \
             set — baseline {} windows, candidate {}. The per-window economic vectors are \
             indexed by that prefix, so differencing them would subtract one arm's window i \
             from a different bar of the other's. Re-score both arms with the same \
             --validation-windows.",
            a.len(),
            b.len()
        ),
        (None, _) | (_, None) => bail!(
            "REFUSING to pair: at least one per-window vector does not record its traded \
             prefix's blocking (baseline {:?} entries, candidate {:?}). It was written before \
             format v{WINDOW_SCORES_FORMAT_VERSION}, so the two economic vectors cannot be \
             aligned window for window; re-score that arm rather than assuming they line up.",
            baseline.traded_blocks.as_ref().map(Vec::len),
            candidate.traded_blocks.as_ref().map(Vec::len)
        ),
    };
    // The edge's null leg is `marginal_position(&supports, ..)`, so it is as support-derived
    // as the nats rows and is withheld with them. It is also a CULL metric, which makes it the
    // most dangerous row to leave standing across a geometry change.
    let edge_difference = predictive
        .then(|| {
            traded_difference(
                "selection_edge_bps",
                baseline.selection_edge_bps.as_deref(),
                candidate.selection_edge_bps.as_deref(),
                traded_blocks,
            )
        })
        .transpose()?
        .flatten();
    let model_growth_difference = traded_difference(
        "model_growth_bps",
        baseline.model_growth_bps.as_deref(),
        candidate.model_growth_bps.as_deref(),
        traded_blocks,
    )?;

    Ok(PairedComparison {
        baseline_run: baseline.run.clone(),
        candidate_run: candidate.run.clone(),
        baseline_sizing_rule: baseline.sizing_rule.clone(),
        candidate_sizing_rule: candidate.sizing_rule.clone(),
        windows: base_nll.len(),
        scoring,
        baseline_mean: if predictive {
            base_nll.iter().sum::<f64>() / base_nll.len() as f64
        } else {
            f64::NAN
        },
        candidate_mean: if predictive {
            cand_nll.iter().sum::<f64>() / cand_nll.len() as f64
        } else {
            f64::NAN
        },
        difference,
        conditional_difference,
        dof_difference,
        correlation: if predictive {
            pearson(&base_nll, &cand_nll)
        } else {
            f64::NAN
        },
        worse_windows: if predictive {
            deltas.iter().filter(|d| **d > 0.0).count()
        } else {
            0
        },
        edge_difference,
        model_growth_difference,
        geometry_change,
    })
}

/// Paired difference of one per-traded-window economic vector, in BPS per bar.
///
/// Bootstrapped over the traded prefix's OWN `(symbol, month)` ids, persisted by the writer
/// rather than re-derived here, so the interval is bit for bit the object the run's selection
/// rule read. `Ok(None)` only when neither side recorded the vector or the prefix is empty;
/// anything else that would misalign the two books is an error, because a missing economic
/// difference reads as an arm with no economic effect.
fn traded_difference(
    field: &str,
    baseline: Option<&[f64]>,
    candidate: Option<&[f64]>,
    blocks: &[u64],
) -> Result<Option<Dispersion>> {
    let (baseline, candidate) = match (baseline, candidate) {
        (Some(baseline), Some(candidate)) => (baseline, candidate),
        (None, None) => return Ok(None),
        (baseline, candidate) => bail!(
            "REFUSING to pair: only one of the two runs recorded `{field}` (baseline {}, \
             candidate {}). Comparing an arm that priced its windows against one that did not \
             would report the missing side as no economic effect; re-score that arm.",
            baseline.map_or("absent", |_| "present"),
            candidate.map_or("absent", |_| "present")
        ),
    };
    ensure!(
        baseline.len() == blocks.len() && candidate.len() == blocks.len(),
        "REFUSING to pair: `{field}` does not span the recorded traded prefix — baseline {} \
         values, candidate {}, blocking {}. The vector is indexed by the traded prefix, so a \
         length that disagrees with it is misaligned window for window.",
        baseline.len(),
        candidate.len(),
        blocks.len()
    );
    if blocks.is_empty() {
        return Ok(None);
    }
    let deltas: Vec<f64> = candidate
        .iter()
        .zip(baseline)
        .map(|(candidate, baseline)| candidate - baseline)
        .collect();
    Ok(Some(block_bootstrap(
        &deltas,
        blocks,
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    )))
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
    compare_runs_with(baseline, candidate, GeometryPolicy::Require)
}

/// [`compare_runs`] with an explicit bin-geometry policy, for an arm whose treatment IS the
/// discretization. See [`GeometryPolicy::ModelGrowthOnly`] for what such a comparison may
/// still say.
pub fn compare_runs_with(
    baseline: &Path,
    candidate: &Path,
    geometry: GeometryPolicy,
) -> Result<PairedComparison> {
    let baseline = WindowScores::load(baseline)?;
    let candidate = WindowScores::load(candidate)?;
    paired_comparison_with(&baseline, &candidate, geometry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// `values[i] = level + block_effect[block(i)] + noise`, so the true standard error of
    /// the mean is dominated by the block effect and is analytically known.
    fn clustered(
        blocks: usize,
        per_block: usize,
        block_sd: f64,
        noise_sd: f64,
    ) -> (Vec<f64>, Vec<u64>) {
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
            let var =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
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
        let d = block_bootstrap(
            &[1.0, 2.0, 3.0],
            &[7, 7, 7],
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        );
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

    #[test]
    fn moving_blocks_of_one_are_the_ordinary_block_bootstrap() {
        let values = [1.0, 2.0, f64::NAN, 4.0, 8.0, 16.0];
        let blocks = [30, 10, 20, 20, 40, 40];
        let ordinary = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        assert_eq!(
            ordinary,
            Dispersion {
                mean: 6.2,
                se: 2.708_208_107_744_120_2,
                ci_low: 1.5,
                ci_high: 10.571_428_571_428_571,
                blocks: 4,
                samples: 5,
            },
            "block_len=1 must preserve the legacy seeded bootstrap draws"
        );
        let moving = moving_block_bootstrap(&values, &blocks, 1, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        assert_eq!(moving, ordinary);
    }

    #[test]
    fn longer_moving_blocks_capture_serial_group_dependence() {
        let blocks: Vec<u64> = (0..80).collect();
        let values: Vec<f64> = (0..80)
            .map(|group| if (group / 10) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let daily = moving_block_bootstrap(&values, &blocks, 1, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        let moving = moving_block_bootstrap(&values, &blocks, 5, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        assert_eq!(moving.blocks, 16);
        assert_eq!(moving.mean, daily.mean);
        assert_eq!(moving.samples, 80);
        assert!(
            moving.se > 1.7 * daily.se,
            "serially aware SE {:.5} should materially exceed daily {:.5}",
            moving.se,
            daily.se
        );
    }

    #[test]
    fn moving_bootstrap_nonfinite_single_block_contract_matches_ordinary_bootstrap() {
        let values = [f64::NAN, 1.0, f64::INFINITY, 3.0];
        let blocks = [7, 7, 8, 7];
        let ordinary = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        let moving = moving_block_bootstrap(&values, &blocks, 3, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);

        assert_eq!(moving.mean, ordinary.mean);
        assert_eq!(moving.blocks, 1);
        assert_eq!(moving.samples, 2);
        assert!(moving.se.is_nan() && moving.ci_low.is_nan() && moving.ci_high.is_nan());

        let empty = moving_block_bootstrap(&[f64::NAN], &[7], 3, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        assert!(empty.mean.is_nan() && empty.se.is_nan());
        assert_eq!(empty.blocks, 0);
        assert_eq!(empty.samples, 0);

        let full_sample_chunk = moving_block_bootstrap(
            &[1.0, 2.0, 3.0],
            &[1, 2, 3],
            3,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        );
        assert_eq!(full_sample_chunk.mean, 2.0);
        assert_eq!(full_sample_chunk.blocks, 1);
        assert_eq!(full_sample_chunk.samples, 3);
        assert!(
            full_sample_chunk.se.is_nan()
                && full_sample_chunk.ci_low.is_nan()
                && full_sample_chunk.ci_high.is_nan(),
            "one effective moving chunk cannot identify dispersion"
        );
    }

    #[test]
    #[should_panic(expected = "moving block length must be at least one")]
    fn moving_bootstrap_rejects_zero_length_blocks() {
        let _ = moving_block_bootstrap(&[1.0], &[0], 0, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
    }
    /// Conditional NLL pools bar-level sufficient statistics. A mean of window ratios can say
    /// "no change" on the same data, so both the reported point and every paired bootstrap
    /// draw must recompute the ratio of sums.
    #[test]
    fn conditional_bootstrap_recomputes_ratio_of_sums_deterministically() {
        let row = |numerator: f64, denominator: f64| {
            let mut stats = ConditionalNllStats::default();
            stats.numerator[0] = numerator;
            stats.denominator[0] = denominator;
            stats
        };
        let baseline = [row(1.0, 1.0), row(900.0, 100.0)];
        let candidate = [row(2.0, 1.0), row(800.0, 100.0)];
        let blocks = [0, 1];
        let expected: f64 = 802.0 / 101.0 - 901.0 / 101.0;
        let mean_of_window_ratio_deltas = ((2.0 - 1.0) + (8.0 - 9.0)) / 2.0;
        assert_eq!(mean_of_window_ratio_deltas, 0.0);
        assert!(expected.abs() > 0.9);

        let a = block_bootstrap_conditional_difference(
            &baseline,
            &candidate,
            &blocks,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        );
        let b = block_bootstrap_conditional_difference(
            &baseline,
            &candidate,
            &blocks,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        );
        assert_eq!(a, b);
        assert_eq!(a.mean, expected);
        assert_eq!(
            a.mean,
            ConditionalNllStats::pooled(&candidate).point_estimate()
                - ConditionalNllStats::pooled(&baseline).point_estimate()
        );
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
                    conditional_nll: ConditionalNllStats {
                        numerator: [(base - 0.7) * 0.2; BAR_DOF],
                        denominator: [1.0; BAR_DOF],
                    },
                }
            })
            .collect();
        // The bench trades a PREFIX of the pinned set, so the fixture does too: a fixture
        // whose economics spanned all 64 windows would never exercise the truncation every
        // real artifact carries.
        const TRADED: usize = 40;
        let mut scored = WindowScores {
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
            sizing_rule: Some(trade_bench::SizingRule::CONTROL.label().to_owned()),
            windows,
            trade: None,
            // Set, because a real run always sets them: a fixture that left them absent
            // would make every pairing test exercise the "unrecorded" refusal instead of
            // the comparison.
            realized_batch: Some(23),
            realized_steps: Some(10818),
            supports_sha256: Some("a".repeat(64)),
            // The economics run the OPPOSITE way from the nats — `offset` below zero is a
            // better NLL, so the same arm has to show a HIGHER edge — and the two vectors are
            // given different magnitudes so no test can pass by reading one for the other.
            selection_edge_bps: Some(
                (0..TRADED)
                    .map(|i| 0.38 + (i % 5) as f64 * 0.01 - offset)
                    .collect(),
            ),
            model_growth_bps: Some(
                (0..TRADED)
                    .map(|i| 0.50 + (i % 5) as f64 * 0.01 - 2.0 * offset)
                    .collect(),
            ),
            traded_blocks: None,
            calibration: Some(CalibrationSlopes {
                mean_beta: 0.71.into(),
                mean_beta_se: 0.04.into(),
                variance_beta: 0.62.into(),
            }),
        };
        let mut traded = scored.symbol_month_blocks();
        traded.truncate(TRADED);
        scored.traded_blocks = Some(traded);
        scored
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
        assert!((paired.conditional_difference.mean + 0.25).abs() < 1e-9);
        // A pure shift leaves zero residual dispersion, so the interval collapses onto it.
        assert!(paired.difference.se < 1e-9);
        assert!(paired.significant() || paired.difference.se.is_nan());
        assert!((paired.correlation - 1.0).abs() < 1e-9);
        assert_eq!(paired.worse_windows, 0);

        // Sizing is artifact identity, not a refusal: this A/B deliberately changes policy.
        let mut other_sizing = candidate.clone();
        other_sizing.sizing_rule = Some(trade_bench::SizingRule::CUMULANT_LOG.label().to_owned());
        let sized = paired_comparison(&baseline, &other_sizing)
            .expect("a deliberate sizing-policy comparison remains pairable");
        assert_eq!(sized.baseline_sizing_rule, baseline.sizing_rule);
        assert_eq!(sized.candidate_sizing_rule, other_sizing.sizing_rule);
        let rendered = sized.to_string();
        assert!(rendered.contains("sizing rule: baseline"), "{rendered}");
        assert!(
            rendered.contains("second-cumulant expected-log approximation"),
            "{rendered}"
        );

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

        // The two arms of the growth ablation were launched identically at --batch-size 24
        // and the startup capacity probe gave one 23 and the other 21, which bought 10818
        // steps against 11847 — a different gradient-noise level and a different-length
        // schedule, invisible in the requested figure. Refused on the realized one.
        let mut clamped_lower = scores("cand", "ff", -0.25);
        clamped_lower.realized_batch = Some(21);
        clamped_lower.realized_steps = Some(11847);
        let err = paired_comparison(&baseline, &clamped_lower)
            .expect_err("two schedules must not be paired")
            .to_string();
        assert!(err.contains("different schedules"), "{err}");
        assert!(err.contains("--exact-batch"), "{err}");

        // Absent is not "matched": a vector too old to record what it ran is refused rather
        // than assumed equal, exactly as an unrecorded scoring rule is.
        for (batch, steps) in [(None, Some(10818)), (Some(23), None), (None, None)] {
            let mut unrecorded = scores("cand", "ff", -0.25);
            unrecorded.realized_batch = batch;
            unrecorded.realized_steps = steps;
            let err = paired_comparison(&baseline, &unrecorded)
                .expect_err("an unrecorded schedule must not be paired")
                .to_string();
            assert!(err.contains("actually executed"), "{err}");
        }
    }

    /// Nothing else on a per-window vector is a function of the bin geometry, so a run whose
    /// supports were refitted clears every other check and returns a meaningless nats
    /// difference with a tight interval. Both the mismatch and the un-recorded case have to
    /// be refusals; treating absent as equal is what makes the failure silent.
    #[test]
    fn pairing_refuses_a_changed_bin_geometry_and_an_unrecorded_one() {
        let baseline = scores("base", "ff", 0.0);

        let mut refitted = scores("cand", "ff", -0.25);
        refitted.supports_sha256 = Some("b".repeat(64));
        let err = paired_comparison(&baseline, &refitted)
            .expect_err("two bin geometries must not be paired")
            .to_string();
        assert!(err.contains("DIFFERENT BIN GEOMETRIES"), "{err}");
        // The message has to NAME both hashes, or the reader cannot tell which arm moved.
        assert!(err.contains(&"a".repeat(64)), "{err}");
        assert!(err.contains(&"b".repeat(64)), "{err}");
        // And it has to name the escape hatch, so a geometry-CHANGING arm is re-judged on the
        // one ruler that survives rather than discarded.
        assert!(err.contains("--allow-geometry-change"), "{err}");

        for (base_geometry, candidate_geometry) in [
            (None, Some("a".repeat(64))),
            (Some("a".repeat(64)), None),
            (None, None),
        ] {
            let mut old_baseline = scores("base", "ff", 0.0);
            old_baseline.supports_sha256 = base_geometry;
            let mut old_candidate = scores("cand", "ff", -0.25);
            old_candidate.supports_sha256 = candidate_geometry;
            // Unknown geometry is refused under BOTH policies: an operator can assert that
            // the supports changed, but nobody can assert what they were.
            for policy in [GeometryPolicy::Require, GeometryPolicy::ModelGrowthOnly] {
                let err = paired_comparison_with(&old_baseline, &old_candidate, policy)
                    .expect_err("an unrecorded bin geometry must not be assumed equal")
                    .to_string();
                assert!(err.contains("does not record the bin geometry"), "{err}");
            }
        }
    }

    /// Wave E changes the discretization on purpose, so the pair MUST still yield the model
    /// growth leg — that is the entire reason both economic vectors are persisted. Everything
    /// decoded through the bins must be withheld, and the selection edge counts as decoded
    /// through them because its null leg is support-derived.
    #[test]
    fn an_asserted_geometry_change_reports_the_model_leg_and_withholds_the_rest() {
        let baseline = scores("base", "ff", 0.0);
        let mut refitted = scores("cand", "ff", -0.25);
        refitted.supports_sha256 = Some("b".repeat(64));

        let paired = paired_comparison_with(&baseline, &refitted, GeometryPolicy::ModelGrowthOnly)
            .expect("an asserted geometry change is comparable on the model leg");
        assert!(!paired.predictive_measured());
        assert_eq!(
            paired.geometry_change,
            Some(("a".repeat(64), "b".repeat(64)))
        );

        // Withheld, not computed: a number here would be read as a measurement.
        assert!(paired.difference.mean.is_nan());
        assert!(paired.conditional_difference.mean.is_nan());
        assert!(paired.dof_difference.iter().all(|d| d.mean.is_nan()));
        assert!(paired.baseline_mean.is_nan() && paired.candidate_mean.is_nan());
        assert!(paired.correlation.is_nan());
        assert!(!paired.significant());
        assert!(
            paired.edge_difference.is_none(),
            "the selection edge's null leg is support-derived, so it cannot survive a \
             geometry change"
        );

        // The one ruler that does survive, with the same value it has within one geometry:
        // the model leg never touches a bin.
        let growth = paired
            .model_growth_difference
            .expect("the model leg is valid across geometries");
        assert!((growth.mean - 0.50).abs() < 1e-9, "{}", growth.mean);

        let printed = paired.to_string();
        assert!(printed.contains("BIN GEOMETRY CHANGED"), "{printed}");
        // The banner must name both hashes and the only valid cull metric, so the ledger is
        // readable years later without this conversation.
        assert!(printed.contains(&"a".repeat(64)), "{printed}");
        assert!(printed.contains(&"b".repeat(64)), "{printed}");
        assert!(printed.contains("ONLY valid cull metric"), "{printed}");
        assert_eq!(
            printed
                .matches("NOT COMPARABLE (forbidden across bin geometries)")
                .count(),
            5,
            "every suppressed metric needs an explicit verdict, never a missing line: \
             {printed}"
        );
        // And the surviving row is still printed with its sign convention.
        assert!(printed.contains("model growth delta [bps/bar"), "{printed}");
        assert!(printed.contains("POSITIVE is better"), "{printed}");

        // The default policy still refuses the identical pair.
        assert!(paired_comparison(&baseline, &refitted).is_err());
    }

    /// The economics and the nats run in OPPOSITE directions, and the whole campaign culls on
    /// these two numbers. A sign flip here would invert every economic verdict, so the
    /// convention is pinned by construction: the fixture's candidate is better on all three
    /// metrics at once, which means a NEGATIVE nats difference and POSITIVE economic ones.
    #[test]
    fn the_economic_differences_are_positive_when_the_candidate_is_better() {
        let baseline = scores("base", "ff", 0.0);
        let candidate = scores("cand", "ff", -0.25);
        let paired = paired_comparison(&baseline, &candidate).expect("comparable");

        assert!(
            paired.difference.mean < 0.0,
            "the better arm must score LOWER in nats: {}",
            paired.difference.mean
        );
        let edge = paired.edge_difference.expect("v3 vectors carry an edge");
        let growth = paired
            .model_growth_difference
            .expect("v3 vectors carry a model leg");
        // The fixture shifts the edge by +0.25 bps and the model leg by +0.50 on every traded
        // window, so the two are distinguishable and neither can be standing in for the other.
        assert!((edge.mean - 0.25).abs() < 1e-9, "{}", edge.mean);
        assert!((growth.mean - 0.50).abs() < 1e-9, "{}", growth.mean);
        assert!(BetterWhen::Positive.advantage(edge.mean) > 0.0);
        assert!(BetterWhen::Positive.advantage(growth.mean) > 0.0);
        // Both conventions must agree that this candidate won.
        assert!(BetterWhen::Negative.advantage(paired.difference.mean) > 0.0);

        // Blocked over the TRADED prefix, not the pinned set: 40 windows, and the prefix's own
        // symbol-month grouping rather than all 64 windows' worth.
        assert_eq!(edge.samples, 40);
        assert_eq!(edge.blocks, {
            let mut blocks = baseline.symbol_month_blocks();
            blocks.truncate(40);
            blocks
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        });

        let printed = paired.to_string();
        assert!(printed.contains("selection edge delta"), "{printed}");
        assert!(printed.contains("model growth delta"), "{printed}");
        assert!(printed.contains("POSITIVE is better"), "{printed}");
        assert!(printed.contains("NEGATIVE is better"), "{printed}");

        // An arm that priced a different number of windows is misaligned window for window,
        // and must be refused rather than reported as having no economic effect.
        let mut short_prefix = scores("cand", "ff", -0.25);
        short_prefix.traded_blocks = short_prefix.traded_blocks.map(|mut blocks| {
            blocks.truncate(32);
            blocks
        });
        let err = paired_comparison(&baseline, &short_prefix)
            .expect_err("two traded prefixes must not be paired")
            .to_string();
        assert!(err.contains("different prefixes"), "{err}");

        // Absence must print, not vanish: a stale artifact that silently skipped the row
        // would read as an arm with no economic effect.
        let mut unrecorded = paired.clone();
        unrecorded.edge_difference = None;
        unrecorded.model_growth_difference = None;
        let printed = unrecorded.to_string();
        assert_eq!(
            printed
                .matches(&format!(
                    "NOT COMPARABLE (pre-v{WINDOW_SCORES_FORMAT_VERSION} artifact"
                ))
                .count(),
            2,
            "{printed}"
        );
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
            for dof in 0..BAR_DOF {
                assert!(
                    (got.conditional_nll.numerator[dof] - want.conditional_nll.numerator[dof])
                        .abs()
                        < 1e-12
                );
                assert!(
                    (got.conditional_nll.denominator[dof] - want.conditional_nll.denominator[dof])
                        .abs()
                        < 1e-12
                );
            }
        }
        assert_eq!(loaded.corpus_fingerprint, scored.corpus_fingerprint);
        assert_eq!(loaded.split_bounds, scored.split_bounds);
        assert_eq!(loaded.eval_window_seed, scored.eval_window_seed);
        assert_eq!(loaded.supports_sha256, scored.supports_sha256);
        assert_eq!(loaded.traded_blocks, scored.traded_blocks);
        assert_eq!(loaded.sizing_rule, scored.sizing_rule);
        // Identity fields are exact; the bps vectors are held to the same tolerance as the
        // nats above, because serde_json's default parser does not promise the last bit and a
        // pico-bp is ~13 orders of magnitude below anything selection acts on.
        let close = |a: &Option<Vec<f64>>, b: &Option<Vec<f64>>| {
            let (a, b) = (a.as_ref().expect("recorded"), b.as_ref().expect("recorded"));
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-12));
        };
        close(&loaded.selection_edge_bps, &scored.selection_edge_bps);
        close(&loaded.model_growth_bps, &scored.model_growth_bps);
        let (got, want) = (
            loaded.calibration.expect("recorded"),
            scored.calibration.expect("recorded"),
        );
        assert!((got.mean_beta.0 - want.mean_beta.0).abs() < 1e-12);
        assert!((got.mean_beta_se.0 - want.mean_beta_se.0).abs() < 1e-12);
        assert!((got.variance_beta.0 - want.variance_beta.0).abs() < 1e-12);
        // A v2 artifact must fail at LOAD rather than pair as though it recorded economics.
        let mut stale = scored.clone();
        stale.format_version = WINDOW_SCORES_FORMAT_VERSION - 1;
        let stale_path = dir.join("stale.windows.json");
        stale.save(&stale_path).expect("save");
        assert!(WindowScores::load(&stale_path).is_err());
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
