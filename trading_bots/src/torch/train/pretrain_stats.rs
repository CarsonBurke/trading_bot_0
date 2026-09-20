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
/// point estimate and every bootstrap resample measure the same estimand.
pub const WINDOW_SCORES_FORMAT_VERSION: u32 = 2;

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
// Generic deterministic statistics
// ---------------------------------------------------------------------------

/// Product moments of a stream of finite `(x, y)` pairs.
///
/// The six stored values are additive sufficient statistics: callers can accumulate one
/// instance per block or shard and [`Self::absorb`] them without retaining any observations.
/// Non-finite pairs are deliberately ignored, matching the Skill diagnostic's historical
/// convention.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProductMoments {
    n: f64,
    x: f64,
    y: f64,
    xx: f64,
    yy: f64,
    xy: f64,
}

impl ProductMoments {
    pub fn push(&mut self, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.n += 1.0;
        self.x += x;
        self.y += y;
        self.xx += x * x;
        self.yy += y * y;
        self.xy += x * y;
    }

    pub fn absorb(&mut self, other: &Self) {
        self.n += other.n;
        self.x += other.x;
        self.y += other.y;
        self.xx += other.xx;
        self.yy += other.yy;
        self.xy += other.xy;
    }

    pub fn count(&self) -> f64 {
        self.n
    }

    pub fn mean_x(&self) -> f64 {
        if self.n > 0.0 {
            self.x / self.n
        } else {
            f64::NAN
        }
    }

    /// Pearson `corr(x, y)`, or NaN when fewer than two pairs were seen or either side is
    /// constant.
    pub fn corr(&self) -> f64 {
        if self.n < 2.0 {
            return f64::NAN;
        }
        let sxx = self.n * self.xx - self.x * self.x;
        let syy = self.n * self.yy - self.y * self.y;
        let sxy = self.n * self.xy - self.x * self.y;
        if !(sxx > 0.0) || !(syy > 0.0) {
            return f64::NAN;
        }
        sxy / (sxx * syy).sqrt()
    }
}

/// Additive sufficient statistics that can be resampled block by block.
pub trait BlockSums: Copy + Default {
    fn absorb(&mut self, other: &Self);
    /// Number of observations represented by this accumulator.
    fn count(&self) -> f64;
}

impl BlockSums for ProductMoments {
    fn absorb(&mut self, other: &Self) {
        ProductMoments::absorb(self, other);
    }

    fn count(&self) -> f64 {
        ProductMoments::count(self)
    }
}

/// Pooled mid-ranks of `values`, `1..=n`, with tied values sharing their mean rank.
///
/// Ordering uses [`f64::total_cmp`], while ties use floating-point equality. This preserves
/// the established Skill behavior, including one common rank for an all-tied input.
pub fn mid_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut ranks = vec![f64::NAN; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let mean = 0.5 * ((start + 1) + end) as f64;
        for slot in &order[start..end] {
            ranks[*slot] = mean;
        }
        start = end;
    }
    ranks
}

/// Ascending interior percentile cutpoints for `buckets` groups.
///
/// Non-finite values are excluded. Assignment against these population cutpoints is performed
/// by [`bucket_of`]; tied observations at a boundary stay together in the lower bucket, which
/// is the Skill diagnostic's existing convention.
pub fn percentile_cutpoints(values: &[f64], buckets: usize) -> Vec<f64> {
    if buckets < 2 {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    sorted.sort_by(f64::total_cmp);
    (1..buckets)
        .map(|k| sorted_percentile(&sorted, k as f64 / buckets as f64))
        .collect()
}

/// Bucket index against ascending interior `cutpoints`.
///
/// A value equal to a boundary remains in the lower bucket. With `k` cutpoints the result is
/// always in `0..=k`.
pub fn bucket_of(cutpoints: &[f64], value: f64) -> usize {
    cutpoints.partition_point(|cut| *cut < value)
}

/// Exact equal-count bucket assignment without arbitrary tie breaking.
///
/// Every finite observation receives the bucket implied by its sorted position and bucket
/// sizes differ by at most one. If a required boundary splits equal values, including every
/// boundary of an all-tied sample, the assignment is rejected instead of using input or symbol
/// order to manufacture a ranking that the data do not contain.
pub fn equal_count_buckets(values: &[f64], buckets: usize) -> Result<Vec<usize>> {
    ensure!(
        buckets >= 2,
        "equal-count bucketing needs at least two buckets"
    );
    ensure!(
        values.len() >= buckets,
        "cannot divide {} observations into {buckets} non-empty buckets",
        values.len()
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "equal-count bucketing requires finite values"
    );

    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    if values[order[0]] == values[*order.last().expect("values is non-empty")] {
        bail!(
            "all {} observations are tied at {}; exact {buckets}-bucket coverage is ambiguous",
            values.len(),
            values[order[0]]
        );
    }
    for bucket in 1..buckets {
        let boundary = (bucket * values.len()).div_ceil(buckets);
        if values[order[boundary - 1]] == values[order[boundary]] {
            bail!(
                "equal-count bucket boundary {bucket}/{buckets} splits tied value {}; \
                 exact coverage is ambiguous",
                values[order[boundary]]
            );
        }
    }

    let mut assignment = vec![0usize; values.len()];
    for (rank, slot) in order.into_iter().enumerate() {
        assignment[slot] = rank * buckets / values.len();
    }
    Ok(assignment)
}

/// Resample additive statistics block by block and refit `stat` on each deterministic draw.
///
/// Non-finite draw results are omitted rather than imputed. This is the generic form of the
/// paired block-statistic convention used by Skill: fixed bootstrap constants, block order,
/// RNG, sample standard deviation, and linear-interpolated interval percentiles.
pub fn paired_block_statistic<S: BlockSums>(blocks: &[S], stat: impl Fn(&S) -> f64) -> Dispersion {
    let mut pooled = S::default();
    for block in blocks {
        pooled.absorb(block);
    }
    let mut out = Dispersion {
        mean: stat(&pooled),
        se: f64::NAN,
        ci_low: f64::NAN,
        ci_high: f64::NAN,
        blocks: blocks.len(),
        samples: pooled.count() as usize,
    };
    if blocks.len() < 2 {
        return out;
    }

    let mut rng = ChaCha12Rng::seed_from_u64(BOOTSTRAP_SEED);
    let mut draws = Vec::with_capacity(BOOTSTRAP_DRAWS);
    for _ in 0..BOOTSTRAP_DRAWS {
        let mut draw = S::default();
        for _ in 0..blocks.len() {
            draw.absorb(blocks.choose(&mut rng).expect("blocks is non-empty"));
        }
        let value = stat(&draw);
        if value.is_finite() {
            draws.push(value);
        }
    }
    if draws.len() < 2 {
        return out;
    }
    draws.sort_by(f64::total_cmp);
    out.se = sample_standard_deviation(&draws);
    let tail = (1.0 - CI_MASS) / 2.0;
    out.ci_low = sorted_percentile(&draws, tail);
    out.ci_high = sorted_percentile(&draws, 1.0 - tail);
    out
}

/// Sample standard deviation with Bessel's correction, or NaN for fewer than two values.
pub fn sample_standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Streaming f64 sufficient statistics for a ridge regression with an intercept.
///
/// Memory is `O(p²)` regardless of row count. The intercept is represented as the first
/// normal-equation column and is never penalized.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeSums {
    features: usize,
    rows: u64,
    xtx: Vec<f64>,
    xty: Vec<f64>,
}

impl RidgeSums {
    pub fn new(features: usize) -> Result<Self> {
        let dimension = features
            .checked_add(1)
            .context("ridge feature dimension overflow")?;
        let matrix_len = dimension
            .checked_mul(dimension)
            .context("ridge normal-matrix dimension overflow")?;
        Ok(Self {
            features,
            rows: 0,
            xtx: vec![0.0; matrix_len],
            xty: vec![0.0; dimension],
        })
    }

    pub fn feature_count(&self) -> usize {
        self.features
    }

    pub fn row_count(&self) -> u64 {
        self.rows
    }

    /// Add one row. Dimension and finiteness are validated before any statistic is changed.
    pub fn push(&mut self, features: &[f64], target: f64) -> Result<()> {
        ensure!(
            features.len() == self.features,
            "ridge row has {} features, expected {}",
            features.len(),
            self.features
        );
        ensure!(target.is_finite(), "ridge target must be finite");
        ensure!(
            features.iter().all(|value| value.is_finite()),
            "ridge features must all be finite"
        );
        let next_rows = self
            .rows
            .checked_add(1)
            .context("ridge row count overflow")?;
        let dimension = self.features + 1;
        for i in 0..dimension {
            let xi = if i == 0 { 1.0 } else { features[i - 1] };
            let target_increment = xi * target;
            ensure!(
                target_increment.is_finite() && (self.xty[i] + target_increment).is_finite(),
                "ridge target cross-product overflowed at feature column {i}"
            );
            for j in 0..dimension {
                let xj = if j == 0 { 1.0 } else { features[j - 1] };
                let matrix_increment = xi * xj;
                ensure!(
                    matrix_increment.is_finite()
                        && (self.xtx[i * dimension + j] + matrix_increment).is_finite(),
                    "ridge feature cross-product overflowed at matrix entry ({i}, {j})"
                );
            }
        }
        for i in 0..dimension {
            let xi = if i == 0 { 1.0 } else { features[i - 1] };
            self.xty[i] += xi * target;
            for j in 0..dimension {
                let xj = if j == 0 { 1.0 } else { features[j - 1] };
                self.xtx[i * dimension + j] += xi * xj;
            }
        }
        self.rows = next_rows;
        Ok(())
    }

    /// Merge another accumulator in a fixed, deterministic element order.
    pub fn absorb(&mut self, other: &Self) -> Result<()> {
        ensure!(
            self.features == other.features,
            "cannot merge ridge statistics with {} and {} features",
            self.features,
            other.features
        );
        let rows = self
            .rows
            .checked_add(other.rows)
            .context("ridge row count overflow while merging")?;
        ensure!(
            self.xtx
                .iter()
                .chain(&self.xty)
                .chain(&other.xtx)
                .chain(&other.xty)
                .all(|value| value.is_finite()),
            "cannot merge non-finite ridge sufficient statistics"
        );
        ensure!(
            self.xtx
                .iter()
                .zip(&other.xtx)
                .all(|(left, right)| (left + right).is_finite())
                && self
                    .xty
                    .iter()
                    .zip(&other.xty)
                    .all(|(left, right)| (left + right).is_finite()),
            "ridge sufficient statistics overflow while merging"
        );
        for (left, right) in self.xtx.iter_mut().zip(&other.xtx) {
            *left += right;
        }
        for (left, right) in self.xty.iter_mut().zip(&other.xty) {
            *left += right;
        }
        self.rows = rows;
        Ok(())
    }

    /// Solve the regularized normal equations with a deterministic Cholesky factorization.
    ///
    /// `lambda` is added to feature diagonals only. Empty, non-finite, asymmetric, singular,
    /// or numerically non-positive systems return an actionable error.
    pub fn fit(&self, lambda: f64) -> Result<RidgeFit> {
        ensure!(
            lambda.is_finite() && lambda >= 0.0,
            "ridge lambda must be finite and non-negative, got {lambda}"
        );
        ensure!(self.rows > 0, "cannot fit ridge regression without rows");
        ensure!(
            self.xtx
                .iter()
                .chain(&self.xty)
                .all(|value| value.is_finite()),
            "ridge sufficient statistics contain a non-finite value"
        );

        let dimension = self.features + 1;
        let scale = self
            .xtx
            .iter()
            .map(|value| value.abs())
            .fold(0.0f64, f64::max)
            .max(1.0);
        let symmetry_tolerance = 32.0 * f64::EPSILON * scale;
        for i in 0..dimension {
            for j in 0..i {
                let difference = (self.xtx[i * dimension + j] - self.xtx[j * dimension + i]).abs();
                ensure!(
                    difference <= symmetry_tolerance,
                    "ridge normal matrix is asymmetric at ({i}, {j}): difference \
                     {difference:e} exceeds tolerance {symmetry_tolerance:e}"
                );
            }
        }

        let mut matrix = self.xtx.clone();
        for diagonal in 1..dimension {
            matrix[diagonal * dimension + diagonal] += lambda;
        }
        ensure!(
            matrix.iter().all(|value| value.is_finite()),
            "ridge regularization overflowed the normal matrix at lambda {lambda}"
        );

        let pivot_floor = f64::EPSILON * dimension as f64 * scale;
        let mut lower = vec![0.0; dimension * dimension];
        for i in 0..dimension {
            for j in 0..=i {
                let mut value = matrix[i * dimension + j];
                for k in 0..j {
                    value -= lower[i * dimension + k] * lower[j * dimension + k];
                }
                ensure!(
                    value.is_finite(),
                    "ridge Cholesky produced a non-finite value at ({i}, {j})"
                );
                if i == j {
                    ensure!(
                        value > pivot_floor,
                        "ridge normal matrix is singular or ill-conditioned at pivot {i}: \
                         {value:e} <= {pivot_floor:e}; increase lambda or remove a degenerate \
                         feature"
                    );
                    lower[i * dimension + j] = value.sqrt();
                } else {
                    lower[i * dimension + j] = value / lower[j * dimension + j];
                }
            }
        }

        let mut intermediate = vec![0.0; dimension];
        for i in 0..dimension {
            let mut value = self.xty[i];
            for j in 0..i {
                value -= lower[i * dimension + j] * intermediate[j];
            }
            intermediate[i] = value / lower[i * dimension + i];
        }
        let mut solution = vec![0.0; dimension];
        for i in (0..dimension).rev() {
            let mut value = intermediate[i];
            for j in i + 1..dimension {
                value -= lower[j * dimension + i] * solution[j];
            }
            solution[i] = value / lower[i * dimension + i];
        }
        ensure!(
            solution.iter().all(|value| value.is_finite()),
            "ridge solve produced non-finite coefficients"
        );

        Ok(RidgeFit {
            intercept: solution[0],
            coefficients: solution[1..].to_vec(),
            lambda,
        })
    }
}

/// A solved ridge model. `coefficients` excludes the intercept.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeFit {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub lambda: f64,
}

impl RidgeFit {
    pub fn predict(&self, features: &[f64]) -> Result<f64> {
        ensure!(
            features.len() == self.coefficients.len(),
            "ridge prediction has {} features, expected {}",
            features.len(),
            self.coefficients.len()
        );
        ensure!(
            features.iter().all(|value| value.is_finite()),
            "ridge prediction features must all be finite"
        );
        let prediction = self
            .coefficients
            .iter()
            .zip(features)
            .fold(self.intercept, |sum, (coefficient, feature)| {
                sum + coefficient * feature
            });
        ensure!(prediction.is_finite(), "ridge prediction is non-finite");
        Ok(prediction)
    }

    pub fn predict_all<T: AsRef<[f64]>>(&self, rows: &[T]) -> Result<Vec<f64>> {
        rows.iter().map(|row| self.predict(row.as_ref())).collect()
    }
}

/// Positive geometric ridge candidates: `first * ratio.powi(index)`.
pub fn geometric_lambdas(first: f64, ratio: f64, count: usize) -> Result<Vec<f64>> {
    ensure!(
        first.is_finite() && first > 0.0,
        "first geometric ridge lambda must be finite and positive"
    );
    ensure!(
        ratio.is_finite() && ratio > 1.0,
        "geometric ridge lambda ratio must be finite and greater than one"
    );
    ensure!(count > 0, "geometric ridge lambda count must be positive");
    let mut candidates = Vec::with_capacity(count);
    let mut candidate = first;
    for index in 0..count {
        ensure!(
            candidate.is_finite(),
            "geometric ridge lambda overflowed at candidate {index}"
        );
        candidates.push(candidate);
        candidate *= ratio;
    }
    Ok(candidates)
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
        ci_low: sorted_percentile(&means, tail),
        ci_high: sorted_percentile(&means, 1.0 - tail),
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
        ci_low: sorted_percentile(&differences, tail),
        ci_high: sorted_percentile(&differences, 1.0 - tail),
        blocks: totals.len(),
        samples: candidate.len(),
    }
}

/// Linear-interpolated percentile of an ascending slice.
pub fn sorted_percentile(sorted: &[f64], q: f64) -> f64 {
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
        writeln!(
            f,
            "  paired delta (candidate - baseline) {}",
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
    let difference = block_bootstrap(&deltas, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);

    let base_conditional = baseline.conditional_nll_stats();
    let cand_conditional = candidate.conditional_nll_stats();
    let conditional_difference = block_bootstrap_conditional_difference(
        &base_conditional,
        &cand_conditional,
        &blocks,
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    );

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
            trade: None,
            // Set, because a real run always sets them: a fixture that left them absent
            // would make every pairing test exercise the "unrecorded" refusal instead of
            // the comparison.
            realized_batch: Some(23),
            realized_steps: Some(10818),
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
        assert!((paired.conditional_difference.mean + 0.25).abs() < 1e-9);
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scores_path_sits_beside_the_checkpoint() {
        assert_eq!(
            window_scores_path(Path::new("runs/x/weights/pretrain_best.ot")),
            PathBuf::from("runs/x/weights/pretrain_best.windows.json")
        );
    }

    #[test]
    fn shared_mid_ranks_preserve_ties_and_all_tied_inputs() {
        assert_eq!(
            mid_ranks(&[30.0, 10.0, 20.0, 10.0]),
            vec![4.0, 1.5, 3.0, 1.5]
        );
        assert_eq!(mid_ranks(&[7.0, 7.0, 7.0, 7.0]), vec![2.5; 4]);
        assert!(mid_ranks(&[]).is_empty());
    }

    #[test]
    fn product_moments_matches_centered_pearson_and_is_block_additive() {
        let x = [1.0, 2.0, 4.0, 8.0, 16.0];
        let y = [-2.0, 3.0, 1.0, 9.0, 7.0];
        let mean_x = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        let covariance = x
            .iter()
            .zip(y)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>();
        let variance_x = x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
        let variance_y = y.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
        let expected = covariance / (variance_x * variance_y).sqrt();

        let mut left = ProductMoments::default();
        let mut right = ProductMoments::default();
        for (&x, &y) in x[..2].iter().zip(&y[..2]) {
            left.push(x, y);
        }
        for (&x, &y) in x[2..].iter().zip(&y[2..]) {
            right.push(x, y);
        }
        left.absorb(&right);
        assert!((left.corr() - expected).abs() < 1e-14);
        assert_eq!(left.count(), x.len() as f64);

        let mut all_tied = ProductMoments::default();
        all_tied.push(1.0, 2.0);
        all_tied.push(1.0, 3.0);
        assert!(all_tied.corr().is_nan());
    }

    #[test]
    fn exact_deciles_cover_every_row_and_refuse_ambiguous_ties() {
        let values: Vec<f64> = (0..100).map(|value| value as f64).collect();
        let assignment = equal_count_buckets(&values, 10).expect("distinct exact deciles");
        let mut counts = [0usize; 10];
        for bucket in assignment {
            counts[bucket] += 1;
        }
        assert_eq!(counts, [10; 10]);

        let cuts = percentile_cutpoints(&values, 10);
        assert_eq!(cuts.len(), 9);
        assert_eq!(bucket_of(&cuts, 0.0), 0);
        assert_eq!(bucket_of(&cuts, 99.0), 9);

        let tied = equal_count_buckets(&[0.0, 1.0, 1.0, 2.0], 2)
            .expect_err("a boundary must not split equal observations")
            .to_string();
        assert!(tied.contains("splits tied value"), "{tied}");
        let all_tied = equal_count_buckets(&[3.0; 10], 10)
            .expect_err("all-tied deciles are undefined")
            .to_string();
        assert!(all_tied.contains("ambiguous"), "{all_tied}");
    }

    #[test]
    fn streaming_ridge_recovers_multivariate_coefficients_and_predicts() {
        let rows = [[-2.0, 4.0], [-1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 4.0]];
        let mut sums = RidgeSums::new(2).expect("dimensions");
        for row in rows {
            sums.push(&row, 4.0 + 2.0 * row[0] - 0.5 * row[1])
                .expect("finite row");
        }
        let fit = sums.fit(0.0).expect("full-rank exact solve");
        assert!((fit.intercept - 4.0).abs() < 1e-12);
        assert!((fit.coefficients[0] - 2.0).abs() < 1e-12);
        assert!((fit.coefficients[1] + 0.5).abs() < 1e-12);
        assert!((fit.predict(&[3.0, 2.0]).expect("prediction") - 9.0).abs() < 1e-12);
        assert_eq!(
            fit.predict_all(&[[0.0, 0.0], [1.0, 1.0]])
                .expect("batch prediction")
                .len(),
            2
        );
    }

    #[test]
    fn ridge_regularizes_slopes_but_never_the_intercept() {
        let mut line = RidgeSums::new(1).expect("dimensions");
        for x in -5..=5 {
            line.push(&[x as f64], 5.0 + 3.0 * x as f64)
                .expect("finite row");
        }
        let unregularized = line.fit(0.0).expect("ordinary least squares");
        let regularized = line.fit(1_000.0).expect("ridge");
        assert!((unregularized.intercept - 5.0).abs() < 1e-12);
        assert!((unregularized.coefficients[0] - 3.0).abs() < 1e-12);
        assert!((regularized.intercept - 5.0).abs() < 1e-12);
        assert!(regularized.coefficients[0].abs() < unregularized.coefficients[0].abs());

        let mut intercept_only = RidgeSums::new(0).expect("intercept-only dimensions");
        for target in [7.0, 7.0, 7.0] {
            intercept_only.push(&[], target).expect("finite target");
        }
        let intercept = intercept_only
            .fit(1.0e12)
            .expect("unpenalized intercept")
            .intercept;
        assert!((intercept - 7.0).abs() < 1e-12);
    }

    #[test]
    fn merged_ridge_sufficient_statistics_match_one_pass_accumulation() {
        let rows = [
            ([1.0, -1.0], 2.0),
            ([2.0, 0.5], 4.0),
            ([-3.0, 2.0], -1.0),
            ([0.25, 4.0], 3.0),
            ([5.0, -2.0], 8.0),
        ];
        let mut whole = RidgeSums::new(2).expect("dimensions");
        let mut left = RidgeSums::new(2).expect("dimensions");
        let mut right = RidgeSums::new(2).expect("dimensions");
        for (index, (features, target)) in rows.iter().enumerate() {
            whole.push(features, *target).expect("whole row");
            if index < 2 {
                left.push(features, *target).expect("left row");
            } else {
                right.push(features, *target).expect("right row");
            }
        }
        left.absorb(&right).expect("compatible merge");
        assert_eq!(left.row_count(), whole.row_count());
        let merged = left.fit(0.25).expect("merged fit");
        let direct = whole.fit(0.25).expect("direct fit");
        assert!((merged.intercept - direct.intercept).abs() < 1e-14);
        for (merged, direct) in merged.coefficients.iter().zip(direct.coefficients) {
            assert!((merged - direct).abs() < 1e-14);
        }
    }

    #[test]
    fn ridge_rejects_nonfinite_asymmetric_and_degenerate_inputs() {
        let mut finite = RidgeSums::new(2).expect("dimensions");
        assert!(finite.push(&[f64::NAN, 1.0], 2.0).is_err());
        assert!(finite.push(&[1.0, 2.0], f64::INFINITY).is_err());
        assert!(finite.push(&[f64::MAX, 1.0], 2.0).is_err());
        assert_eq!(
            finite.row_count(),
            0,
            "rejected rows must not partially mutate sums"
        );
        assert!(finite.fit(0.0).is_err());
        assert!(geometric_lambdas(1e-6, 10.0, 4).is_ok());
        assert!(geometric_lambdas(0.0, 10.0, 4).is_err());

        let mut degenerate = RidgeSums::new(2).expect("dimensions");
        for x in 0..4 {
            degenerate
                .push(&[x as f64, x as f64], x as f64)
                .expect("finite row");
        }
        assert!(degenerate.fit(0.0).is_err());
        assert!(degenerate.fit(1.0).is_ok());

        let mut asymmetric = degenerate.clone();
        asymmetric.xtx[1] += 1.0;
        let error = asymmetric
            .fit(1.0)
            .expect_err("asymmetric normal matrix")
            .to_string();
        assert!(error.contains("asymmetric"), "{error}");
    }
}
