//! Does a CONTINUOUS mixed likelihood per DOF reproduce the bar law the 128-way equal-mass
//! discrete support encodes, and does it survive the leverage licence the bins currently supply?
//!
//! This is a GATE, not a migration. It fits candidate families to the UNCONDITIONAL law on the
//! SAME 4,000,000-row train-region draw the live supports were fitted from, scores them against
//! the discrete marginal on one explicit footing, and either licenses replacing the bins or names
//! the measured fact that kills each family. It touches no trainer, no head, no loss and no
//! `BAR_CHAIN`; it reads [`crate::torch::bar_dist`] and writes reports.
//!
//! WHY THE COMPARISON IS ALREADY FAIR. [`BarScoring::Density`] adds `E[ln width(bin)]` to the
//! categorical loss, so [`BarSupports::marginal_nll_dof`] under that rule is already the mixed
//! measure entropy of a STEP DENSITY: mass `m_b` spread over width `w_b` is a density `m_b / w_b`,
//! and an atom bin has zero width and keeps its probability MASS. Both sides of the comparison
//! below are therefore nats per bar of `-ln` of a density with respect to the same mixed measure
//! (counting measure on the atoms, Lebesgue on the remainder). No rescaling, no offset, and the
//! discrete figure is NOT a bin probability.
//!
//! WHY EVERY FAMILY IS MIXED. A pure continuous density on `u` is ill-defined: half the rows sit
//! on `{0, 0.5, 1}` exactly. Every family here is `atom probabilities x continuous density on the
//! remainder`, and the atom probabilities are the EMPIRICAL shares by construction, not fitted —
//! the log-likelihood of a mixed law separates into a multinomial term over the class indicator
//! and a density term inside the continuous class, and the multinomial MLE of the class
//! probabilities IS the empirical share. So deliverable (a) is exact by construction and is
//! reported as such; what this pass actually MEASURES is that the redrawn sample reproduces the
//! atom shares the persisted artifact recorded, which is the check that the geometry and this fit
//! are looking at the same rows.
//!
//! WHY `r` IS TRUNCATED. For a position at fraction `F` with `R = exp(r) - 1`, `E[ln(1 + F R)]`
//! is finite only if `1 + F R > 0` almost surely. Unbounded support forces `|F| < 1`. A bounded
//! support turns the leverage ceiling into an auditable modeling constant instead of an artifact
//! of `lo[0]`, and [`RuinRow`] inverts the condition on both sides of the book.
//!
//! WHY THERE IS A RESOLUTION FLOOR, and why it is not a fudge. The data is tick-quantized, so the
//! log-density of ANY continuous family is unbounded above: collapse a component onto a repeated
//! value and the NLL goes to `-inf`. The discrete competitor cannot do that, because its bin
//! widths are bounded below by the tiling. Every mixture here is therefore floored at the
//! narrowest nonzero bin of the support it is being scored against, so neither side can buy nats
//! with resolution the other does not have. That floor is reported per DOF.
//!
//! RESOURCE SHAPE, load-bearing. ONE bounded buffer: the drawn `Vec<BarDof>`, the same allocation
//! [`crate::torch::dataset::BarCorpus::fit_supports`] already makes. Every fit is a streaming
//! rayon fold whose accumulator is `O(K)` doubles; the only other buffers are an
//! [`INIT_SUBSAMPLE`]-element quantile probe and a [`TAIL_BUFFER`]-element upper-order-statistic
//! heap, both fixed at compile time and both independent of the corpus.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f64::consts::{FRAC_1_SQRT_2, LN_2, PI};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::torch::bar_dist::{
    BarDof, BarScoring, BarSupports, BAR_DOF, BAR_DOF_NAMES, DOF_R, DOF_S, DOF_U, DOF_V,
    NUM_BAR_BINS,
};
use crate::torch::train::pretrain::{load_corpus, CorpusFlags};
use crate::torch::train::trade_bench::{LEVERAGE_CAP, MAX_LEVERAGE};

use super::pretrain_reports::write_bar_family;

/// Rows folded per rayon task.
const EM_CHUNK: usize = 1 << 16;

/// Upper bound on the deterministic quantile probe used to initialize every mixture.
///
/// Initialization needs order statistics and order statistics need a sort, so this is the one
/// place a sorted buffer appears. Strided rather than random, so the probe is a function of the
/// draw alone and every fit is reproducible from the seed.
const INIT_SUBSAMPLE: usize = 1 << 16;

/// Upper order statistics of `|r|` retained for the tail estimators.
///
/// 65,536 of 4,000,000 rows is the top 1.64%, which covers every level in [`TAIL_LEVELS_P`] and
/// every fragment in [`HILL_K`] exactly. A bounded min-heap, so this costs 512 KiB and one
/// comparison per row rather than a sort of the draw.
pub const TAIL_BUFFER: usize = 1 << 16;

/// Holdout stride: one row in ten is withheld from every fit and scored separately.
///
/// A fixed stride rather than a hash, so the split is reproducible from the row index alone, and
/// offset off zero so it does not align with the block anchors `sample_train_dof` lays down at
/// multiples of its `SUPPORT_BLOCK`.
const HOLDOUT_STRIDE: usize = 10;
const HOLDOUT_RESIDUE: usize = 7;

/// EM sweeps per fit, and the mean-log-likelihood movement that ends them early. The tolerance is
/// six orders below any figure this pass reports, so stopping on it cannot move a reported number.
const EM_ITERATIONS: usize = 120;
const EM_TOLERANCE: f64 = 1.0e-8;

/// Newton sweeps per Beta M-step, and the score norm that ends them.
const NEWTON_ITERATIONS: usize = 60;
const NEWTON_TOLERANCE: f64 = 1.0e-12;

/// Smallest mixture weight a component keeps. A component that collects no responsibility is
/// pinned here rather than deleted, so a `K`-component fit has `K` components at every sweep and
/// the component sweep compares like with like.
const MIN_WEIGHT: f64 = 1.0e-9;

/// Exceedance levels the six pairwise log-log slopes are read at.
///
/// Four levels give `4 choose 2 = 6` pairs, which is the count the measured 1.66-1.84 spread was
/// quoted from.
pub const TAIL_LEVELS_P: [f64; 4] = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4];

/// Fragment sizes the Hill index is fitted at.
///
/// Hill is consistent only as `k -> inf` with `k / n -> 0`, so a single `k` is a choice and not an
/// estimate. The sweep is the estimate, and the presence or absence of a plateau across it is what
/// carries the finding. The range extends down to `k = 50` because the quoted 1.66-1.84 spread is
/// not reproduced anywhere in the `k >= 500` range, and a heavier index can only live FURTHER out;
/// `k = 50` on 3.6e6 continuous rows is a `1.4e-5` exceedance, which is as far as the draw reaches
/// before the estimator is counting single bars.
const HILL_K: [usize; 10] =
    [50, 100, 200, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000];

/// Points on the log-log tail chart, geometric in threshold so a power law reads as a line.
const TAIL_GRID: usize = 40;

/// Interior nodes per support bin used to average a fitted density over that bin.
///
/// A composite MIDPOINT rule, never a rule with endpoint nodes: a Beta component with `alpha < 1`
/// has an integrable singularity at zero, and an endpoint evaluation there returns `+inf` and
/// destroys the closure check the average feeds. The comparison target is the empirical
/// `mass / width`, i.e. the bin AVERAGE, so a single midpoint would disagree with it wherever the
/// density curves across the bin — which is exactly where the interesting structure is.
const DENSITY_NODES: usize = 32;

/// Candidate max-leverage values the ruin licence is tabulated over. Contains both live constants,
/// so the table passes through the numbers actually in force.
const RUIN_LEVERAGES: [f64; 11] = [
    1.5,
    2.0,
    3.0,
    LEVERAGE_CAP,
    6.0,
    8.0,
    10.0,
    11.0,
    MAX_LEVERAGE,
    16.0,
    24.0,
];

/// Lattice positions of `u` / `v` probed for exact-equality mass.
///
/// Not a fitted quantity and not part of any family: a diagnostic that separates "the interior law
/// is a smooth density with three atoms" from "the interior law is a price LATTICE whose rational
/// positions carry mass the 0.5%-threshold atom detector never promoted".
const LATTICE_PROBES: [f64; 7] = [0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0];

/// Every base [`write_bar_family`] writes. The single source of truth for this module's panel set,
/// walked by this module's registry test and mirrored in
/// [`shared::report::PRETRAIN_REPORT_BASES`].
pub const BAR_FAMILY_BASES: &[&str] = &[
    "bar_family_density_r",
    "bar_family_density_s",
    "bar_family_density_u",
    "bar_family_density_v",
    "bar_family_density_w",
    "bar_family_tail_r",
    "bar_family_k_sweep",
    "bar_family_nll",
    "bar_family_atoms",
    "bar_family_ruin_bound",
];

/// The per-DOF density base, in tensor order. Indexed by DOF, so a reorder of `BAR_DOF_NAMES`
/// cannot silently retarget a panel.
pub const DENSITY_BASES: [&str; BAR_DOF] = [
    "bar_family_density_r",
    "bar_family_density_s",
    "bar_family_density_u",
    "bar_family_density_v",
    "bar_family_density_w",
];

/// Sampling slack on an atom share.
///
/// The redraw is the SAME draw by construction — same accessor, same budget, same seed — so the
/// shares agree to the last bit and any tolerance is generous. It is not zero only because the
/// artifact's masses have been through decimal JSON. `1e-9` is five orders below the binomial
/// standard error a 20% share carries on four million rows (2.0e-4), so it cannot hide a real
/// population difference while still tolerating a re-serialized file.
pub const DEFAULT_ATOM_TOLERANCE: f64 = 1.0e-9;

/// The measured pairwise-slope band a fitted tail index is held against.
///
/// SIX PAIRWISE LOG-LOG SLOPES, not a fitted index: no point estimate, no standard error. Carried
/// as a pair so every consumer draws it as a BAND and never as a value.
pub const MEASURED_TAIL_BAND: (f64, f64) = (1.66, 1.84);

/// Which continuous family sits under a DOF's atoms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilyKind {
    /// Gaussian mixture on the value itself, truncated to `[-r_max, +r_max]`.
    TruncatedGaussianMixture,
    /// Gaussian mixture on `ln` of the value, i.e. a log-normal mixture.
    LogNormalMixture,
    /// Beta mixture on the open unit interval.
    BetaMixture,
    /// Gaussian mixture on the value itself, untruncated.
    GaussianMixture,
}

impl FamilyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TruncatedGaussianMixture => "truncated gaussian mixture",
            Self::LogNormalMixture => "log-normal mixture",
            Self::BetaMixture => "beta mixture",
            Self::GaussianMixture => "gaussian mixture",
        }
    }
}

// ---------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------

/// `ln Gamma(x)` for `x > 0`, by upward recurrence onto the Stirling series.
///
/// The series is the Bernoulli asymptotic expansion
/// `(z - 1/2) ln z - z + ln(2 pi) / 2 + sum_n B_2n / (2n (2n - 1) z^(2n-1))`, truncated after
/// `B_10`; shifting the argument to `z >= 16` first puts the first dropped term at
/// `|B_12| / (12 * 11 * 16^11) ~ 1e-16`. Written from the expansion rather than from a table of
/// Lanczos coefficients so every constant is derivable, and checked against libtorch's `lgamma`.
fn ln_gamma(x: f64) -> f64 {
    debug_assert!(x > 0.0);
    let mut z = x;
    let mut shift = 0.0;
    while z < 16.0 {
        shift += z.ln();
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    let series = inv
        * (1.0 / 12.0
            + inv2
                * (-1.0 / 360.0
                    + inv2 * (1.0 / 1260.0 + inv2 * (-1.0 / 1680.0 + inv2 / 1188.0))));
    (z - 0.5) * z.ln() - z + 0.5 * (2.0 * PI).ln() + series - shift
}

/// `psi(x)` for `x > 0`, by upward recurrence onto `ln z - 1/(2z) - sum_n B_2n / (2n z^2n)`.
fn digamma(x: f64) -> f64 {
    debug_assert!(x > 0.0);
    let mut z = x;
    let mut shift = 0.0;
    while z < 16.0 {
        shift += 1.0 / z;
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    let series = inv2
        * (-1.0 / 12.0
            + inv2 * (1.0 / 120.0 + inv2 * (-1.0 / 252.0 + inv2 * (1.0 / 240.0 - inv2 / 132.0))));
    z.ln() - 0.5 * inv + series - shift
}

/// `psi'(x)` for `x > 0`, by upward recurrence onto `1/z + 1/(2 z^2) + sum_n B_2n / z^(2n+1)`.
fn trigamma(x: f64) -> f64 {
    debug_assert!(x > 0.0);
    let mut z = x;
    let mut shift = 0.0;
    while z < 16.0 {
        shift += 1.0 / (z * z);
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    let series = inv
        * (1.0
            + 0.5 * inv
            + inv2
                * (1.0 / 6.0 + inv2 * (-1.0 / 30.0 + inv2 * (1.0 / 42.0 - inv2 / 30.0))));
    series + shift
}

/// Regularized lower incomplete gamma `P(a, x)` by its ascending series, for `x < a + 1`.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut n = 0.0;
    while n < 1_000.0 {
        n += 1.0;
        term *= x / (a + n);
        sum += term;
        if term.abs() < sum.abs() * 1.0e-17 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularized upper incomplete gamma `Q(a, x)` by the modified Lentz continued fraction, for
/// `x > a + 1`.
fn gamma_q_continued(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i = 0.0;
    while i < 1_000.0 {
        i += 1.0;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1.0e-17 {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// `erfc(x)` to full double precision, as the regularized upper incomplete gamma `Q(1/2, x^2)`,
/// which is an identity and not an approximation. Checked against libtorch's `erfc`, including
/// RELATIVE accuracy in the far tail, because the tail chart reads exceedance probabilities off it.
fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // `erfc(30)` is 3e-393, below the smallest subnormal, so every larger argument is exactly
    // zero. The early return also keeps `+inf` out of the continued fraction, where
    // `-x + a ln x` would evaluate to `NaN`.
    if x > 30.0 {
        return 0.0;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    let t = x * x;
    if t < 1.5 {
        1.0 - gamma_p_series(0.5, t)
    } else {
        gamma_q_continued(0.5, t)
    }
}

/// `P(Z <= z)` for a standard normal.
fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z * FRAC_1_SQRT_2)
}

/// `P(Z > z)` for a standard normal, on the side that does not cancel.
fn normal_sf(z: f64) -> f64 {
    0.5 * erfc(z * FRAC_1_SQRT_2)
}

/// `ln(exp(a) + exp(b))` without overflow.
fn log_add(a: f64, b: f64) -> f64 {
    if a > b {
        a + (-(a - b)).exp().ln_1p()
    } else if b > a {
        b + (-(b - a)).exp().ln_1p()
    } else if a.is_finite() {
        a + LN_2
    } else {
        a
    }
}

/// `ln sum exp(terms)`, and the responsibilities `exp(terms - total)` written back in place.
///
/// One `ln` and `2K` `exp` per row instead of the `K` `exp` plus `K` `ln_1p` a pairwise fold
/// costs. This is the innermost loop of every EM sweep over four million rows, so the difference
/// is the difference between minutes and tens of minutes.
fn normalize_responsibilities(terms: &mut [f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    for term in terms.iter() {
        if *term > peak {
            peak = *term;
        }
    }
    if !peak.is_finite() {
        return peak;
    }
    let mut sum = 0.0;
    for term in terms.iter_mut() {
        let value = (*term - peak).exp();
        *term = value;
        sum += value;
    }
    let total = peak + sum.ln();
    let inverse = 1.0 / sum;
    for term in terms.iter_mut() {
        *term *= inverse;
    }
    total
}

// ---------------------------------------------------------------------------
// Gaussian mixture
// ---------------------------------------------------------------------------

/// A `K`-component Gaussian mixture on the real line, optionally read as a density truncated to a
/// symmetric interval.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianMixture {
    pub weights: Vec<f64>,
    pub means: Vec<f64>,
    pub sds: Vec<f64>,
    /// Symmetric truncation half-width, or `None` for the untruncated reading.
    ///
    /// Truncation RENORMALIZES; it never moves a fitted parameter. The fit is the untruncated
    /// MLE and the reported density is that fit divided by its mass inside the bound, which is a
    /// valid density and an upper bound on the truncated family's own MLE nats. Stated because it
    /// is not the same thing as maximizing the truncated likelihood.
    pub truncation: Option<f64>,
}

impl GaussianMixture {
    fn components(&self) -> usize {
        self.weights.len()
    }

    /// `ln P(|X| <= truncation)` under the untruncated mixture; zero when untruncated.
    fn log_normalizer(&self) -> f64 {
        let Some(a) = self.truncation else {
            return 0.0;
        };
        let mass: f64 = (0..self.components())
            .map(|k| {
                let sd = self.sds[k];
                let mu = self.means[k];
                self.weights[k] * (normal_cdf((a - mu) / sd) - normal_cdf((-a - mu) / sd))
            })
            .sum();
        mass.max(f64::MIN_POSITIVE).ln()
    }

    fn log_density_with(&self, x: f64, log_normalizer: f64) -> f64 {
        if let Some(a) = self.truncation {
            if x < -a || x > a {
                return f64::NEG_INFINITY;
            }
        }
        let mut acc = f64::NEG_INFINITY;
        let log_root = 0.5 * (2.0 * PI).ln();
        for k in 0..self.components() {
            let sd = self.sds[k];
            let z = (x - self.means[k]) / sd;
            acc = log_add(
                acc,
                self.weights[k].ln() - sd.ln() - log_root - 0.5 * z * z,
            );
        }
        acc - log_normalizer
    }

    fn log_density(&self, x: f64) -> f64 {
        self.log_density_with(x, self.log_normalizer())
    }

    fn density(&self, x: f64) -> f64 {
        self.log_density(x).exp()
    }

    /// `P(X > x)` under the (possibly truncated) density.
    fn survival(&self, x: f64) -> f64 {
        match self.truncation {
            Some(a) if x >= a => 0.0,
            Some(a) => {
                let mass: f64 = (0..self.components())
                    .map(|k| {
                        let sd = self.sds[k];
                        let mu = self.means[k];
                        self.weights[k] * (normal_cdf((a - mu) / sd) - normal_cdf((x - mu) / sd))
                    })
                    .sum();
                (mass / self.log_normalizer().exp()).clamp(0.0, 1.0)
            }
            None => (0..self.components())
                .map(|k| self.weights[k] * normal_sf((x - self.means[k]) / self.sds[k]))
                .sum(),
        }
    }

    /// `P(X < x)` under the (possibly truncated) density.
    fn cumulative(&self, x: f64) -> f64 {
        match self.truncation {
            Some(a) if x <= -a => 0.0,
            Some(a) => {
                let mass: f64 = (0..self.components())
                    .map(|k| {
                        let sd = self.sds[k];
                        let mu = self.means[k];
                        self.weights[k] * (normal_cdf((x - mu) / sd) - normal_cdf((-a - mu) / sd))
                    })
                    .sum();
                (mass / self.log_normalizer().exp()).clamp(0.0, 1.0)
            }
            None => (0..self.components())
                .map(|k| self.weights[k] * normal_cdf((x - self.means[k]) / self.sds[k]))
                .sum(),
        }
    }

    fn free_parameters(&self) -> usize {
        3 * self.components() - 1
    }
}

/// Streaming E-step accumulator: `O(K)` doubles, independent of the row count.
#[derive(Clone)]
struct GaussianAccum {
    rows: f64,
    log_lik: f64,
    weight: Vec<f64>,
    weight_x: Vec<f64>,
    weight_xx: Vec<f64>,
}

impl GaussianAccum {
    fn new(k: usize) -> Self {
        Self {
            rows: 0.0,
            log_lik: 0.0,
            weight: vec![0.0; k],
            weight_x: vec![0.0; k],
            weight_xx: vec![0.0; k],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.rows += other.rows;
        self.log_lik += other.log_lik;
        for k in 0..self.weight.len() {
            self.weight[k] += other.weight[k];
            self.weight_x[k] += other.weight_x[k];
            self.weight_xx[k] += other.weight_xx[k];
        }
        self
    }
}

/// Which rows of the draw a fit or a score looks at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rows {
    /// Every row. The footing the headline comparison against the discrete marginal uses, because
    /// that marginal is the entropy of the SAME rows' own histogram.
    All,
    /// Rows the holdout does not claim.
    Fit,
    /// The withheld one row in [`HOLDOUT_STRIDE`].
    Holdout,
}

impl Rows {
    fn takes(self, index: usize) -> bool {
        match self {
            Self::All => true,
            Self::Fit => index % HOLDOUT_STRIDE != HOLDOUT_RESIDUE,
            Self::Holdout => index % HOLDOUT_STRIDE == HOLDOUT_RESIDUE,
        }
    }
}

/// One EM sweep over the selected rows, returning the updated mixture, the mean log-likelihood of
/// the PREVIOUS parameters, and the rows it saw.
fn gaussian_em_step<E>(
    samples: &[BarDof],
    rows: Rows,
    value_of: &E,
    current: &GaussianMixture,
    sd_floor: f64,
) -> (GaussianMixture, f64, f64)
where
    E: Fn(&BarDof) -> Option<f64> + Send + Sync,
{
    let k = current.components();
    let log_weights: Vec<f64> = current
        .weights
        .iter()
        .map(|w| w.max(MIN_WEIGHT).ln())
        .collect();
    let offsets: Vec<f64> = (0..k)
        .map(|c| log_weights[c] - current.sds[c].ln() - 0.5 * (2.0 * PI).ln())
        .collect();

    let parts: Vec<GaussianAccum> = samples
        .par_chunks(EM_CHUNK)
        .enumerate()
        .map(|(chunk, block)| {
            let base = chunk * EM_CHUNK;
            let mut acc = GaussianAccum::new(k);
            let mut terms = vec![0.0f64; k];
            for (offset, row) in block.iter().enumerate() {
                if !rows.takes(base + offset) {
                    continue;
                }
                let Some(x) = value_of(row) else { continue };
                for c in 0..k {
                    let z = (x - current.means[c]) / current.sds[c];
                    terms[c] = offsets[c] - 0.5 * z * z;
                }
                let total = normalize_responsibilities(&mut terms);
                if !total.is_finite() {
                    continue;
                }
                acc.rows += 1.0;
                acc.log_lik += total;
                for c in 0..k {
                    let resp = terms[c];
                    acc.weight[c] += resp;
                    acc.weight_x[c] += resp * x;
                    acc.weight_xx[c] += resp * x * x;
                }
            }
            acc
        })
        .collect();

    let acc = parts
        .into_iter()
        .fold(GaussianAccum::new(k), |a, b| a.merge(b));
    let rows_seen = acc.rows;
    let mean_log_lik = if rows_seen > 0.0 {
        acc.log_lik / rows_seen
    } else {
        f64::NAN
    };

    let mut next = current.clone();
    for c in 0..k {
        let mass = acc.weight[c];
        if !mass.is_finite() || mass <= rows_seen * MIN_WEIGHT {
            next.weights[c] = MIN_WEIGHT;
            continue;
        }
        next.weights[c] = mass / rows_seen;
        let mean = acc.weight_x[c] / mass;
        let variance = (acc.weight_xx[c] / mass - mean * mean).max(0.0);
        next.means[c] = mean;
        next.sds[c] = variance.sqrt().max(sd_floor);
    }
    let total: f64 = next.weights.iter().sum();
    for w in &mut next.weights {
        *w /= total;
    }
    (next, mean_log_lik, rows_seen)
}

/// Deterministic strided quantile probe of the extracted values, sorted ascending.
fn quantile_probe<E>(samples: &[BarDof], rows: Rows, value_of: &E) -> Vec<f64>
where
    E: Fn(&BarDof) -> Option<f64> + Send + Sync,
{
    let stride = (samples.len() / INIT_SUBSAMPLE).max(1);
    let mut probe: Vec<f64> = samples
        .iter()
        .enumerate()
        .filter(|(index, _)| index % stride == 0 && rows.takes(*index))
        .filter_map(|(_, row)| value_of(row))
        .filter(|x| x.is_finite())
        .collect();
    probe.sort_unstable_by(f64::total_cmp);
    probe
}

fn probe_quantile(probe: &[f64], q: f64) -> f64 {
    if probe.is_empty() {
        return f64::NAN;
    }
    let position = q.clamp(0.0, 1.0) * (probe.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = (low + 1).min(probe.len() - 1);
    let frac = position - low as f64;
    probe[low] * (1.0 - frac) + probe[high] * frac
}

/// Fit a `k`-component Gaussian mixture by EM from the quantile initialization.
fn fit_gaussian_mixture<E>(
    samples: &[BarDof],
    rows: Rows,
    value_of: &E,
    k: usize,
    sd_floor: f64,
    truncation: Option<f64>,
) -> (GaussianMixture, usize)
where
    E: Fn(&BarDof) -> Option<f64> + Send + Sync,
{
    let probe = quantile_probe(samples, rows, value_of);
    // Component means at the K interior quantiles, widths at the interquartile spread over K: a
    // deterministic function of the draw with no random restarts, so the fit is reproducible from
    // the seed alone.
    let spread = (probe_quantile(&probe, 0.75) - probe_quantile(&probe, 0.25)).abs();
    let mut mixture = GaussianMixture {
        weights: vec![1.0 / k as f64; k],
        means: (0..k)
            .map(|c| probe_quantile(&probe, (c as f64 + 0.5) / k as f64))
            .collect(),
        sds: vec![(spread / k as f64).max(sd_floor); k],
        truncation,
    };
    drop(probe);

    let mut previous = f64::NEG_INFINITY;
    let mut sweeps = 0usize;
    for _ in 0..EM_ITERATIONS {
        let (next, log_lik, seen) = gaussian_em_step(samples, rows, value_of, &mixture, sd_floor);
        if seen == 0.0 || !log_lik.is_finite() {
            break;
        }
        sweeps += 1;
        mixture = next;
        if (log_lik - previous).abs() < EM_TOLERANCE {
            break;
        }
        previous = log_lik;
    }
    (mixture, sweeps)
}

// ---------------------------------------------------------------------------
// Beta mixture
// ---------------------------------------------------------------------------

/// A `K`-component Beta mixture on the open unit interval.
#[derive(Clone, Debug, PartialEq)]
pub struct BetaMixture {
    pub weights: Vec<f64>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl BetaMixture {
    fn components(&self) -> usize {
        self.weights.len()
    }

    fn log_density(&self, x: f64) -> f64 {
        if !(x > 0.0 && x < 1.0) {
            return f64::NEG_INFINITY;
        }
        let log_x = x.ln();
        let log_1mx = (-x).ln_1p();
        let mut acc = f64::NEG_INFINITY;
        for k in 0..self.components() {
            let a = self.alpha[k];
            let b = self.beta[k];
            let log_norm = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
            acc = log_add(
                acc,
                self.weights[k].ln() + (a - 1.0) * log_x + (b - 1.0) * log_1mx - log_norm,
            );
        }
        acc
    }

    fn density(&self, x: f64) -> f64 {
        self.log_density(x).exp()
    }

    fn free_parameters(&self) -> usize {
        3 * self.components() - 1
    }
}

#[derive(Clone)]
struct BetaAccum {
    rows: f64,
    log_lik: f64,
    weight: Vec<f64>,
    weight_log_x: Vec<f64>,
    weight_log_1mx: Vec<f64>,
    weight_x: Vec<f64>,
    weight_xx: Vec<f64>,
}

impl BetaAccum {
    fn new(k: usize) -> Self {
        Self {
            rows: 0.0,
            log_lik: 0.0,
            weight: vec![0.0; k],
            weight_log_x: vec![0.0; k],
            weight_log_1mx: vec![0.0; k],
            weight_x: vec![0.0; k],
            weight_xx: vec![0.0; k],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.rows += other.rows;
        self.log_lik += other.log_lik;
        for k in 0..self.weight.len() {
            self.weight[k] += other.weight[k];
            self.weight_log_x[k] += other.weight_log_x[k];
            self.weight_log_1mx[k] += other.weight_log_1mx[k];
            self.weight_x[k] += other.weight_x[k];
            self.weight_xx[k] += other.weight_xx[k];
        }
        self
    }
}

/// Weighted Beta MLE from the sufficient statistics `E[ln x]` and `E[ln(1 - x)]`, by damped
/// Newton on the score equations `psi(a) - psi(a + b) = E[ln x]` and
/// `psi(b) - psi(a + b) = E[ln(1 - x)]`, started at the method-of-moments solution.
///
/// Returns the solution and whether Newton actually reached the MLE. A component that ended at the
/// moment start or at the concentration cap is a valid Beta but NOT the maximizer, and the count
/// of those is reported rather than hidden.
fn beta_mle(
    mean_log_x: f64,
    mean_log_1mx: f64,
    mean: f64,
    variance: f64,
    concentration_cap: f64,
) -> (f64, f64, bool) {
    let mean = mean.clamp(1.0e-12, 1.0 - 1.0e-12);
    let concentration = if variance > 0.0 {
        (mean * (1.0 - mean) / variance - 1.0).clamp(1.0e-6, concentration_cap)
    } else {
        concentration_cap
    };
    let mut a = (mean * concentration).max(1.0e-8);
    let mut b = ((1.0 - mean) * concentration).max(1.0e-8);

    let mut converged = false;
    for _ in 0..NEWTON_ITERATIONS {
        let psi_ab = digamma(a + b);
        let g0 = digamma(a) - psi_ab - mean_log_x;
        let g1 = digamma(b) - psi_ab - mean_log_1mx;
        let base = g0.abs().max(g1.abs());
        if base < NEWTON_TOLERANCE {
            converged = true;
            break;
        }
        let t_ab = trigamma(a + b);
        let j00 = trigamma(a) - t_ab;
        let j11 = trigamma(b) - t_ab;
        let j01 = -t_ab;
        let det = j00 * j11 - j01 * j01;
        if !det.is_finite() || det.abs() < 1.0e-300 {
            break;
        }
        let da = (j11 * g0 - j01 * g1) / det;
        let db = (j00 * g1 - j01 * g0) / det;
        // Backtrack until the step stays in the positive quadrant AND lowers the score norm. The
        // score equations are severely ill-conditioned at the concentrations a near-degenerate
        // spike demands, where an undamped Newton step overshoots straight into `a < 0`.
        let mut scale = 1.0;
        let mut stepped = false;
        for _ in 0..80 {
            let na = a - scale * da;
            let nb = b - scale * db;
            if na > 0.0 && nb > 0.0 {
                let psi = digamma(na + nb);
                let n0 = digamma(na) - psi - mean_log_x;
                let n1 = digamma(nb) - psi - mean_log_1mx;
                if n0.abs().max(n1.abs()) < base {
                    a = na;
                    b = nb;
                    stepped = true;
                    break;
                }
            }
            scale *= 0.5;
        }
        if !stepped {
            break;
        }
    }

    // The cap IS the resolution floor in Beta coordinates: a Beta with mean `m` and concentration
    // `c` has variance `m(1 - m) / (c + 1)`, so capping `c` caps the sharpness at the discrete
    // competitor's narrowest bin. A capped component is deliberately NOT reported as converged.
    let total = a + b;
    if total > concentration_cap {
        let m = a / total;
        a = (m * concentration_cap).max(1.0e-8);
        b = ((1.0 - m) * concentration_cap).max(1.0e-8);
        converged = false;
    }
    (a, b, converged)
}

/// One EM sweep of a Beta mixture: updated mixture, mean log-likelihood of the previous
/// parameters, rows seen, and how many components failed to reach the MLE.
fn beta_em_step<E>(
    samples: &[BarDof],
    rows: Rows,
    value_of: &E,
    current: &BetaMixture,
    concentration_cap: f64,
) -> (BetaMixture, f64, f64, usize)
where
    E: Fn(&BarDof) -> Option<f64> + Send + Sync,
{
    let k = current.components();
    let log_weights: Vec<f64> = current
        .weights
        .iter()
        .map(|w| w.max(MIN_WEIGHT).ln())
        .collect();
    let offsets: Vec<f64> = (0..k)
        .map(|c| {
            let a = current.alpha[c];
            let b = current.beta[c];
            log_weights[c] - (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b))
        })
        .collect();

    let parts: Vec<BetaAccum> = samples
        .par_chunks(EM_CHUNK)
        .enumerate()
        .map(|(chunk, block)| {
            let base = chunk * EM_CHUNK;
            let mut acc = BetaAccum::new(k);
            let mut terms = vec![0.0f64; k];
            for (offset, row) in block.iter().enumerate() {
                if !rows.takes(base + offset) {
                    continue;
                }
                let Some(x) = value_of(row) else { continue };
                let log_x = x.ln();
                let log_1mx = (-x).ln_1p();
                if !log_x.is_finite() || !log_1mx.is_finite() {
                    continue;
                }
                for c in 0..k {
                    terms[c] = offsets[c]
                        + (current.alpha[c] - 1.0) * log_x
                        + (current.beta[c] - 1.0) * log_1mx;
                }
                let total = normalize_responsibilities(&mut terms);
                if !total.is_finite() {
                    continue;
                }
                acc.rows += 1.0;
                acc.log_lik += total;
                for c in 0..k {
                    let resp = terms[c];
                    acc.weight[c] += resp;
                    acc.weight_log_x[c] += resp * log_x;
                    acc.weight_log_1mx[c] += resp * log_1mx;
                    acc.weight_x[c] += resp * x;
                    acc.weight_xx[c] += resp * x * x;
                }
            }
            acc
        })
        .collect();

    let acc = parts.into_iter().fold(BetaAccum::new(k), |a, b| a.merge(b));
    let rows_seen = acc.rows;
    let mean_log_lik = if rows_seen > 0.0 {
        acc.log_lik / rows_seen
    } else {
        f64::NAN
    };

    let mut next = current.clone();
    let mut unconverged = 0usize;
    for c in 0..k {
        let mass = acc.weight[c];
        if !mass.is_finite() || mass <= rows_seen * MIN_WEIGHT {
            next.weights[c] = MIN_WEIGHT;
            continue;
        }
        next.weights[c] = mass / rows_seen;
        let mean = acc.weight_x[c] / mass;
        let variance = (acc.weight_xx[c] / mass - mean * mean).max(0.0);
        let (a, b, ok) = beta_mle(
            acc.weight_log_x[c] / mass,
            acc.weight_log_1mx[c] / mass,
            mean,
            variance,
            concentration_cap,
        );
        next.alpha[c] = a;
        next.beta[c] = b;
        if !ok {
            unconverged += 1;
        }
    }
    let total: f64 = next.weights.iter().sum();
    for w in &mut next.weights {
        *w /= total;
    }
    (next, mean_log_lik, rows_seen, unconverged)
}

fn fit_beta_mixture<E>(
    samples: &[BarDof],
    rows: Rows,
    value_of: &E,
    k: usize,
    concentration_cap: f64,
) -> (BetaMixture, usize, usize)
where
    E: Fn(&BarDof) -> Option<f64> + Send + Sync,
{
    let probe = quantile_probe(samples, rows, value_of);
    // Interior quantiles as component locations, each started at a concentration matching the
    // local quantile spacing, so a component lands on a near-degenerate spike from the first
    // sweep instead of having to find it by chance.
    let mut mixture = BetaMixture {
        weights: vec![1.0 / k as f64; k],
        alpha: vec![0.0; k],
        beta: vec![0.0; k],
    };
    for c in 0..k {
        let lo = probe_quantile(&probe, c as f64 / k as f64);
        let hi = probe_quantile(&probe, (c as f64 + 1.0) / k as f64);
        let m = probe_quantile(&probe, (c as f64 + 0.5) / k as f64).clamp(1.0e-6, 1.0 - 1.0e-6);
        let width = ((hi - lo).abs() / 4.0).max(1.0e-6);
        let concentration = (m * (1.0 - m) / (width * width) - 1.0).clamp(0.1, concentration_cap);
        mixture.alpha[c] = (m * concentration).max(1.0e-6);
        mixture.beta[c] = ((1.0 - m) * concentration).max(1.0e-6);
    }
    drop(probe);

    let mut previous = f64::NEG_INFINITY;
    let mut sweeps = 0usize;
    let mut unconverged = 0usize;
    for _ in 0..EM_ITERATIONS {
        let (next, log_lik, seen, bad) =
            beta_em_step(samples, rows, value_of, &mixture, concentration_cap);
        if seen == 0.0 || !log_lik.is_finite() {
            break;
        }
        sweeps += 1;
        mixture = next;
        unconverged = bad;
        if (log_lik - previous).abs() < EM_TOLERANCE {
            break;
        }
        previous = log_lik;
    }
    (mixture, sweeps, unconverged)
}

// ---------------------------------------------------------------------------
// The mixed likelihood
// ---------------------------------------------------------------------------

/// The continuous half of a DOF's mixed likelihood, on the DOF's OWN coordinate.
#[derive(Clone, Debug)]
pub enum Continuous {
    Gaussian(GaussianMixture),
    /// A Gaussian mixture on `ln x`, read as a density on `x > 0` through the `1 / x` Jacobian.
    LogNormal(GaussianMixture),
    Beta(BetaMixture),
}

impl Continuous {
    /// `ln f(x)` on the DOF's own coordinate, Jacobian included.
    fn log_density(&self, x: f64) -> f64 {
        match self {
            Self::Gaussian(m) => m.log_density(x),
            Self::LogNormal(m) => {
                if x <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    let log_x = x.ln();
                    m.log_density(log_x) - log_x
                }
            }
            Self::Beta(m) => m.log_density(x),
        }
    }

    fn density(&self, x: f64) -> f64 {
        match self {
            Self::Gaussian(m) => m.density(x),
            Self::LogNormal(_) => self.log_density(x).exp(),
            Self::Beta(m) => m.density(x),
        }
    }

    /// Mass this density places OUTSIDE `[lo, hi]` on the DOF's own coordinate.
    ///
    /// The closure check needs it: the charted grid is the support's own bin range, which for an
    /// untruncated family does not cover the whole support, so `sum(density * width)` legitimately
    /// falls short of the continuous class share and the shortfall must be accounted rather than
    /// tolerated.
    fn mass_outside(&self, lo: f64, hi: f64) -> f64 {
        match self {
            Self::Gaussian(m) => (1.0 - (m.cumulative(hi) - m.cumulative(lo))).clamp(0.0, 1.0),
            Self::LogNormal(m) => {
                let below = if lo > 0.0 { m.cumulative(lo.ln()) } else { 0.0 };
                let above = if hi > 0.0 { 1.0 - m.cumulative(hi.ln()) } else { 1.0 };
                (below + above).clamp(0.0, 1.0)
            }
            // A Beta lives on `[0, 1]` and the `u` / `v` grid spans exactly that, so there is
            // nothing outside to account for.
            Self::Beta(_) => 0.0,
        }
    }

    fn free_parameters(&self) -> usize {
        match self {
            Self::Gaussian(m) | Self::LogNormal(m) => m.free_parameters(),
            Self::Beta(m) => m.free_parameters(),
        }
    }

    fn parameter_lines(&self, kind: FamilyKind) -> Vec<String> {
        let mut lines = vec![format!("      family {}", kind.as_str())];
        match self {
            Self::Gaussian(m) | Self::LogNormal(m) => {
                for k in 0..m.components() {
                    lines.push(format!(
                        "      [{k}] w {:.6} mean {:+.8} sd {:.8}",
                        m.weights[k], m.means[k], m.sds[k]
                    ));
                }
                if let Some(a) = m.truncation {
                    lines.push(format!(
                        "      truncated at +/-{:.8} ({:.2} bps); renormalizer ln Z {:+.3e}. The \
                         FIT is the untruncated MLE and the reported density is that fit \
                         renormalized, which upper-bounds the truncated family's own MLE nats.",
                        a,
                        a * 10_000.0,
                        m.log_normalizer()
                    ));
                }
            }
            Self::Beta(m) => {
                for k in 0..m.components() {
                    let total = m.alpha[k] + m.beta[k];
                    lines.push(format!(
                        "      [{k}] w {:.6} alpha {:.6e} beta {:.6e} mean {:.8} concentration \
                         {:.4e} sd {:.4e}",
                        m.weights[k],
                        m.alpha[k],
                        m.beta[k],
                        m.alpha[k] / total,
                        total,
                        (m.alpha[k] * m.beta[k] / (total * total * (total + 1.0))).sqrt()
                    ));
                }
            }
        }
        lines
    }
}

/// One atom of a DOF's mixed likelihood, with the check that the redraw sees what the artifact
/// recorded.
#[derive(Clone, Copy, Debug)]
pub struct AtomCheck {
    pub dof: usize,
    pub value: f64,
    /// Share of THIS draw sitting exactly on `value`, which is the family's atom parameter.
    pub drawn_share: f64,
    /// Share the persisted artifact recorded for the same value.
    pub artifact_mass: f64,
}

impl AtomCheck {
    pub fn deviation(&self) -> f64 {
        (self.drawn_share - self.artifact_mass).abs()
    }
}

/// Exact-equality mass at a probed lattice position of `u` / `v`.
#[derive(Clone, Copy, Debug)]
pub struct LatticeProbe {
    pub dof: usize,
    pub value: f64,
    pub share: f64,
    /// Whether the discrete support promoted this value to an atom bin.
    pub is_artifact_atom: bool,
}

/// A `(K, fit, holdout)` point of one DOF's component sweep.
#[derive(Clone, Copy, Debug)]
pub struct SweepPoint {
    pub dof: usize,
    pub components: usize,
    /// Nats per bar on the rows the fit used, mixed-measure density footing.
    pub fit_nll: f64,
    /// Nats per bar on the WITHHELD rows, scored with the fit rows' own class probabilities, so
    /// it is out of sample in the density AND in the class term.
    pub holdout_nll: f64,
    pub free_parameters: usize,
    pub aic_per_bar: f64,
    pub bic_per_bar: f64,
    pub em_sweeps: usize,
    /// Beta components that ended a sweep at the moment start or the concentration cap rather than
    /// at the MLE. Always zero for the Gaussian families.
    pub unconverged_components: usize,
}

/// Everything measured and fitted for one DOF.
pub struct DofFit {
    pub dof: usize,
    pub kind: FamilyKind,
    pub selected_components: usize,
    /// Atom probabilities. Exact empirical shares BY CONSTRUCTION; see the module docs.
    pub atoms: Vec<AtomCheck>,
    pub continuous: Continuous,
    /// Share of the draw in the continuous class.
    pub continuous_share: f64,
    /// Nats per bar of the mixed likelihood on the FULL draw, mixed-measure density footing.
    pub family_nll: f64,
    /// The same family shape fitted on the 90% and scored on the withheld 10%.
    pub holdout_nll: f64,
    /// `BarSupports::marginal_nll_dof(BarScoring::Density)[dof]`, off the persisted artifact.
    pub discrete_nll: f64,
    pub discrete_free_parameters: usize,
    /// The DISCRETE competitor on the FAMILY's protocol: a 128-bin histogram refitted on the same
    /// 90% and scored on the same withheld 10%. The symmetric counterpart of `holdout_nll`.
    pub discrete_holdout_nll: f64,
    /// Holdout rows that landed in a bin the fit rows left empty, which is why
    /// `discrete_holdout_nll` would be infinite.
    pub discrete_holdout_zero_mass_rows: usize,
    /// Narrowest nonzero bin of the discrete competitor, in the coordinate the mixture was fitted
    /// on. The resolution floor; see the module docs for why one has to exist.
    pub resolution_floor: f64,
    /// Chart grid: the support's own CONTINUOUS bin edges, atom bins dropped.
    pub grid_lo: Vec<f64>,
    pub grid_hi: Vec<f64>,
    /// Empirical `mass / width` per continuous bin, from THIS draw.
    pub empirical_density: Vec<f64>,
    /// The family's bin-AVERAGE density over the same bins.
    pub fitted_density: Vec<f64>,
    /// Atoms plus the fitted mass over the charted range plus the fitted mass outside it: a
    /// closure check that the fitted object is a probability density and not a positive function.
    pub integrated_mass: f64,
    pub sweep: Vec<SweepPoint>,
    pub em_sweeps: usize,
    pub unconverged_components: usize,
}

impl DofFit {
    pub fn name(&self) -> &'static str {
        BAR_DOF_NAMES[self.dof]
    }

    pub fn atom_mass(&self) -> f64 {
        self.atoms.iter().map(|a| a.drawn_share).sum()
    }

    pub fn worst_atom_deviation(&self) -> f64 {
        self.atoms.iter().fold(0.0f64, |w, a| w.max(a.deviation()))
    }

    /// Positive when the continuous family beats the 128-way discrete marginal.
    pub fn nats_gained(&self) -> f64 {
        self.discrete_nll - self.family_nll
    }

    /// The same contrast on the SYMMETRIC footing: both sides fitted on the 90%, both scored on the
    /// withheld 10%. This is the figure that cannot be explained by the histogram's 127 parameters.
    pub fn holdout_nats_gained(&self) -> f64 {
        self.discrete_holdout_nll - self.holdout_nll
    }

    pub fn free_parameters(&self) -> usize {
        self.continuous.free_parameters() + self.atoms.len()
    }
}

// ---------------------------------------------------------------------------
// Tail
// ---------------------------------------------------------------------------

/// One pairwise log-log slope, read off two exceedance levels.
#[derive(Clone, Copy, Debug)]
pub struct PairSlope {
    pub p_low: f64,
    pub p_high: f64,
    pub x_low: f64,
    pub x_high: f64,
    /// `-(ln p_i - ln p_j) / (ln x_i - ln x_j)` on the EMPIRICAL exceedances.
    pub empirical: f64,
    /// The identical functional on the FITTED family's own survival function at the SAME two
    /// thresholds. Not a refit and not a second estimate of anything: the same estimator, applied
    /// to the model instead of to the data.
    pub fitted: f64,
}

/// One pairwise log-log slope of the EMPIRICAL exceedance curve, with no family in it.
///
/// The estimator half of [`PairSlope`], split out because it is the only half that can be applied
/// to a second sample: an audit that removes rows from the draw and re-reads the tail needs the
/// SAME functional at the SAME levels, and re-deriving it beside this module is exactly how two
/// tail numbers stop being comparable.
#[derive(Clone, Copy, Debug)]
pub struct EmpiricalSlope {
    pub p_low: f64,
    pub p_high: f64,
    /// The `1 - p_low` quantile of `|r|`, i.e. the FURTHER-OUT threshold.
    pub x_low: f64,
    /// The `1 - p_high` quantile of `|r|`.
    pub x_high: f64,
    /// `-(ln p_low - ln p_high) / (ln x_low - ln x_high)`.
    pub alpha: f64,
}

/// One Hill fragment.
#[derive(Clone, Copy, Debug)]
pub struct HillPoint {
    pub k: usize,
    pub threshold: f64,
    /// `k / sum_{i <= k} ln(X_(i) / X_(k+1))`.
    pub alpha: f64,
    /// `alpha / sqrt(k)`: the asymptotic standard error under an exact Pareto tail, which is the
    /// only regime in which it is the standard error of anything.
    pub standard_error: f64,
}

/// One point of the log-log exceedance chart.
#[derive(Clone, Copy, Debug)]
pub struct TailPoint {
    pub threshold: f64,
    pub empirical_exceedance: f64,
    pub empirical_count: u64,
    pub fitted_exceedance: f64,
}

/// The `r` tail, measured on the draw and read off the fitted family with the same estimators.
pub struct TailFit {
    pub rows: u64,
    /// Rows off the `r == 0` atom, i.e. the rows the continuous density covers.
    pub continuous_rows: u64,
    pub max_abs: f64,
    pub min_r: f64,
    pub max_r: f64,
    pub pairs: Vec<PairSlope>,
    pub hill: Vec<HillPoint>,
    pub grid: Vec<TailPoint>,
    /// The measured band this fit is held against: six pairwise slopes, NOT a fitted index, so it
    /// has no point estimate and no standard error.
    pub measured_band: (f64, f64),
}

impl TailFit {
    pub fn empirical_span(&self) -> (f64, f64) {
        self.pairs
            .iter()
            .filter(|p| p.empirical.is_finite())
            .fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                (lo.min(p.empirical), hi.max(p.empirical))
            })
    }

    pub fn fitted_span(&self) -> (f64, f64) {
        self.pairs
            .iter()
            .filter(|p| p.fitted.is_finite())
            .fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                (lo.min(p.fitted), hi.max(p.fitted))
            })
    }

    /// Whether any Hill fragment's `+/- 2 se` interval reaches into the measured pairwise band.
    ///
    /// The honest form of "is it consistent": one side is a fitted index WITH a standard error and
    /// the other is a SPREAD of six slopes with neither, so the only well-posed question is
    /// whether the estimate's interval intersects the spread — not whether two estimates agree.
    pub fn hill_reaches_measured_band(&self) -> bool {
        self.hill.iter().any(|h| {
            let lo = h.alpha - 2.0 * h.standard_error;
            let hi = h.alpha + 2.0 * h.standard_error;
            hi >= self.measured_band.0 && lo <= self.measured_band.1
        })
    }

    /// Whether the FAMILY's own pairwise slopes overlap the measured band at all.
    pub fn family_reaches_measured_band(&self) -> bool {
        let (lo, hi) = self.fitted_span();
        hi >= self.measured_band.0 && lo <= self.measured_band.1
    }
}

// ---------------------------------------------------------------------------
// Ruin licence
// ---------------------------------------------------------------------------

/// The truncation bound a declared max leverage licenses, on both sides of the book.
///
/// `1 + F R > 0` with `R = exp(r) - 1`. A LONG at `F` is ruined by the down move, so it needs
/// `1 - F (1 - exp(-r_max)) > 0`, i.e. `r_max < -ln(1 - 1/F)`. A SHORT at `F` is ruined by the up
/// move: `1 - F (exp(r_max) - 1) > 0`, i.e. `r_max < ln(1 + 1/F)`. Since `ln(1 + y) < -ln(1 - y)`
/// for `y` in `(0, 1)`, the SHORT side always binds first, and a bound derived from the worst DOWN
/// bar alone OVERSTATES the licensed leverage.
#[derive(Clone, Copy, Debug)]
pub struct RuinRow {
    pub leverage: f64,
    /// `-ln(1 - 1/F)`: largest symmetric log bound a long at `F` survives.
    pub long_log_bound: f64,
    /// `ln(1 + 1/F)`: the same for a short at `F`.
    pub short_log_bound: f64,
    /// `min` of the two, which is always the short side.
    pub binding_log_bound: f64,
    /// `exp(binding) - 1`: the binding bound as a simple return, i.e. `R_max`.
    pub binding_simple_return: f64,
    /// Whether the DRAW fits inside the binding bound. When false, truncating `r` at this
    /// leverage's licence would assign ZERO density to bars that actually happened.
    pub licensed_by_draw: bool,
    /// Whether the DISCRETE SUPPORT's own `r` range fits inside the binding bound. This is the
    /// licence today's cap actually rests on: the model cannot predict outside `[lo[0], hi[127]]`,
    /// so the Kelly solve's `1 + F R > 0` is decided by the SUPPORT's edges and not by the corpus.
    pub licensed_by_support: bool,
}

/// The ruin table, what the draw itself licenses, and what the discrete support licenses.
pub struct RuinLicence {
    pub rows: Vec<RuinRow>,
    /// `1 / (1 - exp(r_min))`: most leverage a long survives against the worst DOWN bar drawn.
    pub draw_long_max_leverage: f64,
    /// `1 / (exp(r_max) - 1)`: most leverage a short survives against the worst UP bar drawn.
    pub draw_short_max_leverage: f64,
    pub draw_min_r: f64,
    pub draw_max_r: f64,
    /// `lo[DOF_R][0]` and `hi[DOF_R][bins - 1]`: the discrete support's own reachable `r` range.
    pub support_min_r: f64,
    pub support_max_r: f64,
    pub support_long_max_leverage: f64,
    pub support_short_max_leverage: f64,
}

impl RuinLicence {
    pub fn draw_max_leverage(&self) -> f64 {
        self.draw_long_max_leverage.min(self.draw_short_max_leverage)
    }

    /// The leverage the SUPPORT's bounded range licenses, short side binding.
    pub fn support_max_leverage(&self) -> f64 {
        self.support_long_max_leverage
            .min(self.support_short_max_leverage)
    }
}

// ---------------------------------------------------------------------------
// Everything the pass produced
// ---------------------------------------------------------------------------

pub struct BarFamilyFit {
    pub rows: usize,
    pub seed: u64,
    pub split_bounds: (i64, i64),
    pub corpus_fingerprint: String,
    pub dofs: Vec<DofFit>,
    pub lattice: Vec<LatticeProbe>,
    pub tail: TailFit,
    pub ruin: RuinLicence,
    /// Sum over the five chain factors of the fitted families' nats per bar.
    pub family_nll_bar: f64,
    /// `BarSupports::marginal_nll_bar(BarScoring::Density)`, off the artifact.
    pub discrete_nll_bar: f64,
    /// Worst absolute atom-share deviation between the redraw and the artifact.
    pub worst_atom_deviation: f64,
}

impl BarFamilyFit {
    pub fn nats_gained(&self) -> f64 {
        self.discrete_nll_bar - self.family_nll_bar
    }

    /// Whether every DOF's family beats the discrete marginal it would replace. NECESSARY, not
    /// sufficient: the tail and the atom structure carry their own verdicts.
    pub fn every_dof_improves(&self) -> bool {
        self.dofs.iter().all(|d| d.nats_gained() > 0.0)
    }

    /// The same verdict on the symmetric 90/10 footing, where the histogram's 127 parameters have
    /// to be paid for out of sample too.
    pub fn every_dof_improves_on_holdout(&self) -> bool {
        self.dofs.iter().all(|d| d.holdout_nats_gained() > 0.0)
    }

    /// DOFs whose fitted family left a component at the concentration cap or the moment start,
    /// i.e. where the reported density is not the family's maximizer.
    pub fn dofs_not_at_the_mle(&self) -> Vec<&'static str> {
        self.dofs
            .iter()
            .filter(|d| d.unconverged_components > 0)
            .map(|d| d.name())
            .collect()
    }

    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "bar family fit: {} rows, seed {} (0x{:X}), split {:?}, corpus {}",
            self.rows, self.seed, self.seed, self.split_bounds, self.corpus_fingerprint
        ));

        lines.push(
            "(a) ATOMS. The atom probabilities of a mixed likelihood are the empirical shares BY \
             CONSTRUCTION, not fitted: the log-likelihood separates into a multinomial over the \
             class indicator and a density inside the continuous class, and the multinomial MLE is \
             the share. What is MEASURED below is that this redraw reproduces the shares the \
             persisted artifact recorded, which is the check that the discrete geometry and this \
             fit are looking at the same rows."
                .to_owned(),
        );
        for dof in &self.dofs {
            if dof.atoms.is_empty() {
                lines.push(format!(
                    "  {} — the support promoted NO atom, so the family is purely continuous and \
                     reproduces zero atomic mass, which is what the draw shows",
                    dof.name()
                ));
                continue;
            }
            for atom in &dof.atoms {
                lines.push(format!(
                    "  {} atom at {:+.6}: draw {:.6}%, artifact {:.6}%, |dev| {:.3e} — exact by \
                     construction as a family parameter",
                    dof.name(),
                    atom.value,
                    100.0 * atom.drawn_share,
                    100.0 * atom.artifact_mass,
                    atom.deviation()
                ));
            }
            lines.push(format!(
                "  {} total atomic mass {:.4}%, continuous class {:.4}%",
                dof.name(),
                100.0 * dof.atom_mass(),
                100.0 * dof.continuous_share
            ));
        }
        lines.push(format!(
            "  worst atom-share deviation over every DOF: {:.3e}",
            self.worst_atom_deviation
        ));
        // The three atoms are not three independent facts. `encode_series` sets `u = v = 0.5` when
        // `high == low`, so the `s == 0` atom and the `u == 0.5` / `v == 0.5` atoms are the SAME
        // rows. A per-DOF factorized marginal cannot represent that; the autoregressive chain can,
        // conditionally, which is exactly why this is a statement about the fit and not the head.
        let share_at = |dof: usize, value: f64| -> Option<f64> {
            self.dofs[dof]
                .atoms
                .iter()
                .find(|atom| atom.value == value)
                .map(|atom| atom.drawn_share)
        };
        if let (Some(flat), Some(u_half), Some(v_half)) = (
            share_at(DOF_S, 0.0),
            share_at(DOF_U, 0.5),
            share_at(DOF_V, 0.5),
        ) {
            lines.push(format!(
                "  ATOM COUPLING, measured: the s == 0 atom carries {:.6}% and the u == 0.5 / \
                 v == 0.5 atoms carry {:.6}% / {:.6}%, differing by {:.3e} / {:.3e}. These are the \
                 SAME rows: a flat bar has high == low, and the DOF encoder assigns u = v = 0.5 \
                 when the range is zero, so the 0.5 spike is a coordinate convention and not a \
                 market fact. A per-DOF FACTORIZED marginal cannot carry that dependence at all; \
                 the autoregressive chain can carry it conditionally, so this bounds the marginal \
                 fit reported here and not the emission head.",
                100.0 * flat,
                100.0 * u_half,
                100.0 * v_half,
                (u_half - flat).abs(),
                (v_half - flat).abs()
            ));
        }
        lines.push(
            "  LATTICE PROBE (diagnostic, no family parameter): exact-equality mass at rational \
             positions of u / v. A position carrying mass that the 0.5% promotion threshold did \
             not turn into an atom is a singularity NO smooth density on the interior can carry."
                .to_owned(),
        );
        for probe in &self.lattice {
            lines.push(format!(
                "  {} at {:.6}: {:.4}% of the draw; promoted to an artifact atom: {}",
                BAR_DOF_NAMES[probe.dof],
                probe.value,
                100.0 * probe.share,
                probe.is_artifact_atom
            ));
        }

        lines.push(format!(
            "(b) TAIL. The measured {:.2}-{:.2} is a SPREAD OF SIX PAIRWISE SLOPES, not a fitted \
             index: it carries no point estimate and no standard error, so nothing below is two \
             estimates being compared.",
            self.tail.measured_band.0, self.tail.measured_band.1
        ));
        lines.push(format!(
            "  {} rows, {} of them off the r == 0 atom; |r| reaches {:.2} bps",
            self.tail.rows,
            self.tail.continuous_rows,
            self.tail.max_abs * 10_000.0
        ));
        let (elo, ehi) = self.tail.empirical_span();
        let (flo, fhi) = self.tail.fitted_span();
        lines.push(format!(
            "  six pairwise slopes on THIS draw's |r|: {:.4}-{:.4}. The SAME estimator on the \
             fitted family's own survival function at the SAME thresholds: {:.4}-{:.4}",
            elo, ehi, flo, fhi
        ));
        for pair in &self.tail.pairs {
            lines.push(format!(
                "    p {:.0e}/{:.0e} at x {:8.2}/{:8.2} bps: empirical {:8.4}, fitted {:10.4}",
                pair.p_high,
                pair.p_low,
                pair.x_high * 10_000.0,
                pair.x_low * 10_000.0,
                pair.empirical,
                pair.fitted
            ));
        }
        lines.push(
            "  Hill index — a genuinely FITTED index with its asymptotic standard error. Hill is \
             consistent only as k -> inf with k/n -> 0, so a single k is a choice; the sweep is \
             the estimate and the presence or absence of a plateau is the finding."
                .to_owned(),
        );
        for hill in &self.tail.hill {
            lines.push(format!(
                "    k {:6}: threshold {:8.2} bps, alpha {:.4} +/- {:.4}",
                hill.k,
                hill.threshold * 10_000.0,
                hill.alpha,
                hill.standard_error
            ));
        }
        lines.push(format!(
            "  does any Hill fragment's +/-2 se interval reach the measured band: {}. Does the \
             FAMILY's own pairwise-slope range overlap the measured band: {}.",
            self.tail.hill_reaches_measured_band(),
            self.tail.family_reaches_measured_band()
        ));
        lines.push(format!(
            "  OVERLAP IS NOT CONSISTENCY. A power law's pairwise slopes must AGREE across \
             thresholds; a family whose slopes SWEEP across a band merely crosses it. The draw's \
             slopes span {:.4} and the family's span {:.4}, a ratio of {:.2}x, so the family's \
             overlap is a crossing and not a matched tail.",
            ehi - elo,
            fhi - flo,
            (fhi - flo) / (ehi - elo).max(1.0e-12)
        ));

        lines.push(
            "(c) MARGINAL NLL, one footing, stated. `scoring: density` adds E[ln width(bin)] to \
             the categorical loss, so the discrete figure is ALREADY a log density against the \
             mixed measure (counting on the atoms, Lebesgue elsewhere): mass m_b over width w_b is \
             a step density m_b/w_b, and an atom bin has zero width and keeps its MASS. Both \
             columns below are nats per bar of -ln of a density on that same measure. No offset is \
             applied and the discrete number is NOT a bin probability."
                .to_owned(),
        );
        for dof in &self.dofs {
            lines.push(format!(
                "  {} K={} {}: family {:+.6} nats (holdout {:+.6}), discrete {:+.6} nats, gain \
                 {:+.6}; free params {} vs {}; resolution floor {:.4e}; EM sweeps {}; components \
                 not at the MLE {}",
                dof.name(),
                dof.selected_components,
                dof.kind.as_str(),
                dof.family_nll,
                dof.holdout_nll,
                dof.discrete_nll,
                dof.nats_gained(),
                dof.free_parameters(),
                dof.discrete_free_parameters,
                dof.resolution_floor,
                dof.em_sweeps,
                dof.unconverged_components
            ));
            lines.push(format!(
                "      SYMMETRIC footing, both sides fitted on the same 90% and scored on the same \
                 withheld 10%: family {:+.6}, discrete histogram refitted {:+.6}, gain {:+.6}; \
                 holdout rows in a bin the fit left empty: {}",
                dof.holdout_nll,
                dof.discrete_holdout_nll,
                dof.holdout_nats_gained(),
                dof.discrete_holdout_zero_mass_rows
            ));
            for line in dof.continuous.parameter_lines(dof.kind) {
                lines.push(line);
            }
            lines.push(format!(
                "      density closure: atoms + fitted mass on and off the charted range = {:.6}",
                dof.integrated_mass
            ));
            lines.push("      K sweep (selection is minimum HOLDOUT nats, declared):".to_owned());
            for point in &dof.sweep {
                lines.push(format!(
                    "        K {}: fit {:+.6}, holdout {:+.6}, aic/bar {:+.6}, bic/bar {:+.6}, \
                         params {}, sweeps {}, not-at-MLE {}",
                    point.components,
                    point.fit_nll,
                    point.holdout_nll,
                    point.aic_per_bar,
                    point.bic_per_bar,
                    point.free_parameters,
                    point.em_sweeps,
                    point.unconverged_components
                ));
            }
        }
        lines.push(format!(
            "  bar total: family {:+.6} nats/bar, discrete {:+.6} nats/bar, gain {:+.6}",
            self.family_nll_bar,
            self.discrete_nll_bar,
            self.nats_gained()
        ));

        lines.push(
            "(d) RUIN LICENCE. 1 + F(exp(r) - 1) > 0. A long at F needs r > ln(1 - 1/F); a short \
             at F needs r < ln(1 + 1/F). The SHORT side always binds, so a bound taken from the \
             worst DOWN bar alone overstates licensed leverage. TWO candidate bounds can license a \
             cap and they are NOT the same quantity: the discrete SUPPORT's reachable range, which \
             is a declared modeling constant, and the corpus DRAW's own extreme, which is a fact \
             about the data."
                .to_owned(),
        );
        for row in &self.ruin.rows {
            lines.push(format!(
                "  F {:6.2}x: long {:9.2} bps, short {:9.2} bps, binding {:9.2} bps (R_max \
                 {:+.6}), licensed by the support: {:5}, licensed by the draw: {}",
                row.leverage,
                row.long_log_bound * 10_000.0,
                row.short_log_bound * 10_000.0,
                row.binding_log_bound * 10_000.0,
                row.binding_simple_return,
                row.licensed_by_support,
                row.licensed_by_draw
            ));
        }
        lines.push(format!(
            "  the SUPPORT licenses at most {:.4}x long (lo[0] {:.4} bps) and {:.4}x short (hi[127] \
             {:+.4} bps), so its binding licence is {:.4}x. This is the licence today's \
             {LEVERAGE_CAP}x cap actually rests on; a long-only reading of it reports {:.4}x, which \
             overstates the binding figure by {:.4}x.",
            self.ruin.support_long_max_leverage,
            self.ruin.support_min_r * 10_000.0,
            self.ruin.support_short_max_leverage,
            self.ruin.support_max_r * 10_000.0,
            self.ruin.support_max_leverage(),
            self.ruin.support_long_max_leverage,
            self.ruin.support_long_max_leverage - self.ruin.support_max_leverage()
        ));
        lines.push(format!(
            "  the DRAW licenses at most {:.4}x long (worst down bar {:.4} bps, r = {:.9}) and \
             {:.4}x short (worst up bar {:+.4} bps, r = {:+.9}), so its binding licence is {:.4}x. \
             Declaring R_max at the DRAW's extreme therefore licenses NO leverage at all, which is \
             the finding: a bounded support wide enough to give every drawn bar positive density is \
             not a bound that licenses 4x.",
            self.ruin.draw_long_max_leverage,
            self.ruin.draw_min_r * 10_000.0,
            self.ruin.draw_min_r,
            self.ruin.draw_short_max_leverage,
            self.ruin.draw_max_r * 10_000.0,
            self.ruin.draw_max_r,
            self.ruin.draw_max_leverage()
        ));

        lines.push(format!(
            "SUMMARY: every DOF improves on the discrete marginal in sample: {}; on the symmetric \
             90/10 footing: {}. DOFs whose reported density is not the family maximizer: {:?}. Hill \
             reaches the measured band: {}. The family's own tail crosses it: {}. The support \
             licenses {:.4}x and the draw licenses {:.4}x.",
            self.every_dof_improves(),
            self.every_dof_improves_on_holdout(),
            self.dofs_not_at_the_mle(),
            self.tail.hill_reaches_measured_band(),
            self.tail.family_reaches_measured_band(),
            self.ruin.support_max_leverage(),
            self.ruin.draw_max_leverage()
        ));
        lines
    }
}

// ---------------------------------------------------------------------------
// Args and entry point
// ---------------------------------------------------------------------------

/// Everything the family fit needs. Deliberately separate from `PretrainArgs`: this touches no
/// model, no device and no schedule.
#[derive(Clone, Debug)]
pub struct BarFamilyArgs {
    pub corpus: CorpusFlags,
    /// Discrete support the families are scored against, and the source of the atom set, the chart
    /// grid and the resolution floors. Read, never written.
    pub supports: String,
    /// Reports directory, i.e. a run's `gens/<n>`.
    pub output: String,
    /// Rows to draw. MUST match the `sample_count` the support's provenance records.
    pub samples: usize,
    /// Draw seed. MUST be the `train_seed` of the run that fitted the support.
    pub seed: u64,
    pub k_min: usize,
    pub k_max: usize,
    /// Largest absolute atom-share deviation accepted between the redraw and the artifact.
    pub atom_tolerance: f64,
}

/// Fit the continuous mixed likelihoods to the unconditional bar law on the support's own draw,
/// score them against the discrete marginal, and emit the verdict.
pub fn fit_bar_families(args: BarFamilyArgs) -> Result<()> {
    ensure!(args.samples > 0, "--samples must be positive");
    ensure!(
        args.k_min >= 1 && args.k_max >= args.k_min,
        "--k-min must be at least 1 and no greater than --k-max"
    );
    ensure!(
        args.atom_tolerance >= 0.0 && args.atom_tolerance.is_finite(),
        "--atom-tolerance must be a finite non-negative probability"
    );

    let source = Path::new(&args.supports);
    let supports = BarSupports::load(source).with_context(|| {
        format!(
            "reading the discrete support to score against, {}",
            source.display()
        )
    })?;
    ensure!(
        supports.num_bins() == NUM_BAR_BINS,
        "{} has {} bins, this build uses {NUM_BAR_BINS}",
        source.display(),
        supports.num_bins()
    );

    let corpus = load_corpus(&args.corpus)?;
    // Without provenance the draw cannot be identified as the one the discrete geometry was fitted
    // on, and the two sides of the NLL comparison would describe different populations.
    let provenance = supports.provenance().with_context(|| {
        format!(
            "{} carries no provenance, so the draw this pass makes cannot be identified against \
             the discrete masses it is scored beside",
            source.display()
        )
    })?;
    ensure!(
        provenance.split_bounds == corpus.split_bounds(),
        "{} was fitted against split bounds {:?} but this corpus resolves {:?}",
        source.display(),
        provenance.split_bounds,
        corpus.split_bounds()
    );
    ensure!(
        provenance.sample_count == args.samples,
        "{} records a fit sample of {} rows but --samples is {}; the families must be fitted on \
         the SAME draw the discrete masses were",
        source.display(),
        provenance.sample_count,
        args.samples
    );

    println!(
        "fitting continuous mixed likelihoods against {}: {} rows from the train region of {:?}, \
         seed {} (0x{:X}), corpus fingerprint {}",
        source.display(),
        args.samples,
        provenance.split_bounds,
        args.seed,
        args.seed,
        provenance.corpus_fingerprint,
    );

    // THE one bounded buffer. Same accessor, same budget, same seed as `fit_supports`, so these
    // are the identical rows the discrete masses were measured on.
    let samples: Vec<BarDof> = corpus
        .sample_train_dof(args.samples, args.seed)
        .into_iter()
        .map(|(_, dof)| dof)
        .collect();
    ensure!(
        !samples.is_empty(),
        "the train region yielded no DOF rows, so there is nothing to fit"
    );

    let fit = build_fit(&samples, &supports, &args)?;
    drop(samples);

    for line in fit.report_lines() {
        println!("{line}");
    }
    write_bar_family(Path::new(&args.output), &fit)?;
    println!("bar family fit charts written to {}", args.output);
    Ok(())
}

/// Fit every family, run every check, and assemble the deliverable. Split out from
/// [`fit_bar_families`] so a test can drive it on a synthetic draw with no corpus on disk.
pub fn build_fit(
    samples: &[BarDof],
    supports: &BarSupports,
    args: &BarFamilyArgs,
) -> Result<BarFamilyFit> {
    let rows = samples.len();
    let bins = NUM_BAR_BINS as usize;
    let discrete_dof = supports.marginal_nll_dof(BarScoring::Density);

    // The truncation for `r` is declared as the draw's own extreme, which makes it an auditable
    // modeling constant with a provenance instead of an artifact of `lo[0]`. It is also the
    // TIGHTEST symmetric bound under which no drawn bar has zero density.
    let extremes = measure_r_extremes(samples);
    ensure!(
        extremes.max_abs > 0.0 && extremes.max_abs.is_finite(),
        "the draw's |r| extreme is {}, so no truncation bound can be declared",
        extremes.max_abs
    );

    let mut dofs = Vec::with_capacity(BAR_DOF);
    for dof in 0..BAR_DOF {
        dofs.push(fit_one_dof(
            samples,
            supports,
            dof,
            args,
            discrete_dof[dof],
            bins,
            extremes.max_abs,
        ));
    }

    // The STOP condition. A redraw that does not reproduce the artifact's atom shares means the
    // geometry and this fit are looking at different data, and every number beside them would
    // describe a population the discrete masses do not.
    for dof in &dofs {
        for atom in &dof.atoms {
            ensure!(
                atom.deviation() <= args.atom_tolerance,
                "the redraw puts {:.9}% of rows on the {} atom at {:+.9} but {} records {:.9}%, a \
                 deviation of {:.3e} above the {:.3e} tolerance. The geometry and this fit are \
                 looking at different data; refusing to report a fit against a population the \
                 discrete masses do not describe.",
                100.0 * atom.drawn_share,
                dof.name(),
                atom.value,
                args.supports,
                100.0 * atom.artifact_mass,
                atom.deviation(),
                args.atom_tolerance
            );
        }
    }

    let lattice = probe_lattice(samples, supports);
    let r_family = match &dofs[DOF_R].continuous {
        Continuous::Gaussian(m) => m.clone(),
        Continuous::LogNormal(_) => bail!("the r family resolved to a log-normal mixture"),
        Continuous::Beta(_) => bail!("the r family resolved to a beta mixture"),
    };
    let tail = measure_tail(samples, &r_family, dofs[DOF_R].continuous_share, &extremes);
    let ruin = tabulate_ruin(
        &extremes,
        supports.lower_bounds(DOF_R)[0],
        supports.upper_bounds(DOF_R)[bins - 1],
    );

    let family_nll_bar: f64 = dofs.iter().map(|d| d.family_nll).sum();
    let discrete_nll_bar = supports.marginal_nll_bar(BarScoring::Density);
    let worst_atom_deviation = dofs
        .iter()
        .fold(0.0f64, |w, d| w.max(d.worst_atom_deviation()));
    let (split_bounds, corpus_fingerprint) = supports
        .provenance()
        .map(|p| (p.split_bounds, p.corpus_fingerprint.clone()))
        .unwrap_or(((0, 0), "unrecorded".to_owned()));

    Ok(BarFamilyFit {
        rows,
        seed: args.seed,
        split_bounds,
        corpus_fingerprint,
        dofs,
        lattice,
        tail,
        ruin,
        family_nll_bar,
        discrete_nll_bar,
        worst_atom_deviation,
    })
}

/// The draw's `r` extremes, which fix both the truncation bound and the ruin licence.
#[derive(Clone, Copy, Debug)]
struct RExtremes {
    min_r: f64,
    max_r: f64,
    max_abs: f64,
}

fn measure_r_extremes(samples: &[BarDof]) -> RExtremes {
    let (min_r, max_r) = samples
        .par_chunks(EM_CHUNK)
        .map(|block| {
            block.iter().fold((f64::MAX, f64::MIN), |(lo, hi), row| {
                let r = row.r as f64;
                if r.is_finite() {
                    (lo.min(r), hi.max(r))
                } else {
                    (lo, hi)
                }
            })
        })
        .reduce(
            || (f64::MAX, f64::MIN),
            |(alo, ahi), (blo, bhi)| (alo.min(blo), ahi.max(bhi)),
        );
    RExtremes {
        min_r,
        max_r,
        max_abs: min_r.abs().max(max_r.abs()),
    }
}

/// Exact-equality share of a value, as one bounded counting pass.
fn exact_share(samples: &[BarDof], dof: usize, value: f64) -> f64 {
    let target = value as f32;
    let hits: u64 = samples
        .par_chunks(EM_CHUNK)
        .map(|block| {
            block
                .iter()
                .filter(|row| row.to_array()[dof] == target)
                .count() as u64
        })
        .sum();
    hits as f64 / samples.len() as f64
}

fn probe_lattice(samples: &[BarDof], supports: &BarSupports) -> Vec<LatticeProbe> {
    let mut out = Vec::with_capacity(2 * LATTICE_PROBES.len());
    for dof in [DOF_U, DOF_V] {
        for value in LATTICE_PROBES {
            let is_artifact_atom = supports
                .atoms(dof)
                .iter()
                .any(|a| a.value as f32 == value as f32);
            out.push(LatticeProbe {
                dof,
                value,
                share: exact_share(samples, dof, value),
                is_artifact_atom,
            });
        }
    }
    out
}

/// Which family a DOF takes.
fn family_of(dof: usize) -> FamilyKind {
    match dof {
        DOF_R => FamilyKind::TruncatedGaussianMixture,
        DOF_S => FamilyKind::LogNormalMixture,
        DOF_U | DOF_V => FamilyKind::BetaMixture,
        _ => FamilyKind::GaussianMixture,
    }
}

/// Narrowest nonzero bin of the discrete competitor, in the coordinate the mixture is fitted on.
fn resolution_floor(supports: &BarSupports, dof: usize, bins: usize) -> f64 {
    let widths = supports.widths(dof);
    let lower = supports.lower_bounds(dof);
    let upper = supports.upper_bounds(dof);
    let mut floor = f64::MAX;
    for bin in 0..bins {
        if widths[bin] <= 0.0 {
            continue;
        }
        // The `s` mixture lives on `ln s`, so its floor has to be a LOG width or it would be
        // comparing a width in `s` against a scale in `ln s`.
        let width = if dof == DOF_S {
            if lower[bin] > 0.0 && upper[bin] > lower[bin] {
                upper[bin].ln() - lower[bin].ln()
            } else {
                continue;
            }
        } else {
            widths[bin]
        };
        floor = floor.min(width);
    }
    if floor.is_finite() && floor > 0.0 {
        floor
    } else {
        f64::EPSILON
    }
}

/// The DISCRETE competitor scored on the family's OWN protocol: a 128-bin histogram refitted on the
/// same 90% of the draw and scored on the same withheld 10%, on the same mixed-measure density
/// footing (`mass` on a zero-width atom bin, `mass / width` on a continuous one).
///
/// Without this, (c) is asymmetric in the one direction that matters. The artifact's
/// `marginal_nll_dof(Density)` is the entropy of the FULL draw's own histogram: an in-sample figure
/// with 127 free parameters, held against a family that is reported both in and out of sample. The
/// histogram's 127 parameters over 3.6e6 rows should cost it almost nothing out of sample, and the
/// point of measuring rather than asserting that is that "almost nothing" is the claim under test.
///
/// Returns the holdout nats per row and the number of holdout rows that landed in a bin the fit
/// rows left EMPTY. A nonzero count is an infinite figure and is reported as one rather than
/// smoothed away, because an unsmoothed histogram assigning zero mass to an event that happened is
/// a real property of the discrete family and not a numerical inconvenience.
fn measure_discrete_holdout(
    samples: &[BarDof],
    supports: &BarSupports,
    dof: usize,
    bins: usize,
) -> (f64, usize) {
    // `BarSupports` holds a `Tensor` and is therefore not `Sync`, so the geometry is copied out
    // before the parallel walk and `bin_of` is reproduced against the copy. Exactly the artifact's
    // rule: narrow to `f32`, clamp onto the support, take the atom bin on an exact atom match,
    // otherwise the last bin whose lower bound the value reaches.
    let lower: Vec<f64> = supports.lower_bounds(dof).to_vec();
    let upper_last = supports.upper_bounds(dof)[bins - 1];
    let atom_bins: Vec<(f64, usize)> = supports
        .atoms(dof)
        .iter()
        .map(|atom| (atom.value, atom.bin))
        .collect();
    let bin_of = |row: &BarDof| -> usize {
        let clamped = ((row.to_array()[dof]) as f64).clamp(lower[0], upper_last);
        if let Some((_, bin)) = atom_bins.iter().find(|(value, _)| *value == clamped) {
            return *bin;
        }
        lower
            .partition_point(|&bound| bound <= clamped)
            .saturating_sub(1)
            .min(bins - 1)
    };
    let tally = |rows: Rows| -> (Vec<u64>, u64) {
        samples
            .par_chunks(EM_CHUNK)
            .enumerate()
            .map(|(chunk, block)| {
                let base = chunk * EM_CHUNK;
                let mut counts = vec![0u64; bins];
                let mut seen = 0u64;
                for (offset, row) in block.iter().enumerate() {
                    if !rows.takes(base + offset) {
                        continue;
                    }
                    counts[bin_of(row)] += 1;
                    seen += 1;
                }
                (counts, seen)
            })
            .reduce(
                || (vec![0u64; bins], 0u64),
                |mut a, b| {
                    for (slot, add) in a.0.iter_mut().zip(b.0.iter()) {
                        *slot += add;
                    }
                    a.1 += b.1;
                    (a.0, a.1)
                },
            )
    };

    let (fit_counts, fit_rows) = tally(Rows::Fit);
    let (holdout_counts, holdout_rows) = tally(Rows::Holdout);
    if fit_rows == 0 || holdout_rows == 0 {
        return (f64::NAN, 0);
    }
    let widths = supports.widths(dof);
    let mut nats = 0.0f64;
    let mut zero_mass_rows = 0usize;
    for bin in 0..bins {
        if holdout_counts[bin] == 0 {
            continue;
        }
        if fit_counts[bin] == 0 {
            zero_mass_rows += holdout_counts[bin] as usize;
            continue;
        }
        let mass = fit_counts[bin] as f64 / fit_rows as f64;
        // A zero-width bin is an ATOM and keeps its MASS; a positive-width bin is a step density
        // `mass / width`. Exactly the rule `BarScoring::Density` applies.
        let log_density = if widths[bin] > 0.0 {
            mass.ln() - widths[bin].ln()
        } else {
            mass.ln()
        };
        nats -= holdout_counts[bin] as f64 * log_density;
    }
    if zero_mass_rows > 0 {
        return (f64::INFINITY, zero_mass_rows);
    }
    (nats / holdout_rows as f64, 0)
}

/// The row's value ON THE DOF'S OWN COORDINATE, or `None` when the row is on an atom or outside
/// the continuous class.
///
/// This is the coordinate every DENSITY is expressed on — [`Continuous::log_density`] applies each
/// family's own change of variable internally — and it is deliberately NOT the coordinate a
/// mixture is fitted on. Conflating the two double-applies the `ln` for `s`, which pushes every
/// live row outside the log-normal's domain and returns an infinite NLL.
fn continuous_value(dof: usize, row: &BarDof, atoms: &[f32]) -> Option<f64> {
    let raw = row.to_array()[dof];
    if !raw.is_finite() || atoms.contains(&raw) {
        return None;
    }
    match dof {
        DOF_S => (raw > 0.0).then_some(raw as f64),
        DOF_U | DOF_V => {
            let x = raw as f64;
            (x > 0.0 && x < 1.0).then_some(x)
        }
        _ => Some(raw as f64),
    }
}

/// The coordinate the DOF's MIXTURE is fitted on: `ln s` for the log-normal family, the value
/// itself everywhere else. The inverse of the Jacobian [`Continuous::log_density`] applies.
fn fit_coordinate(dof: usize, value: f64) -> f64 {
    if dof == DOF_S {
        value.ln()
    } else {
        value
    }
}

/// Empirical class probabilities over `[atom_0, .., atom_{m-1}, continuous]` on the selected rows.
///
/// These ARE the mixed likelihood's class parameters: the multinomial MLE is the empirical share.
/// Carried explicitly so a holdout score can be given the FIT rows' probabilities and therefore be
/// out of sample in the class term as well as in the density.
fn class_probabilities(
    samples: &[BarDof],
    rows: Rows,
    dof: usize,
    atom_values: &[f32],
) -> Vec<f64> {
    let classes = atom_values.len() + 1;
    let counts = samples
        .par_chunks(EM_CHUNK)
        .enumerate()
        .map(|(chunk, block)| {
            let base = chunk * EM_CHUNK;
            let mut local = vec![0u64; classes];
            for (offset, row) in block.iter().enumerate() {
                if !rows.takes(base + offset) {
                    continue;
                }
                let raw = row.to_array()[dof];
                if !raw.is_finite() {
                    continue;
                }
                match atom_values.iter().position(|a| *a == raw) {
                    Some(slot) => local[slot] += 1,
                    None => local[classes - 1] += 1,
                }
            }
            local
        })
        .reduce(
            || vec![0u64; classes],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return vec![f64::NAN; classes];
    }
    counts
        .iter()
        .map(|c| *c as f64 / total as f64)
        .collect()
}

/// Nats per bar of the mixed likelihood over `score_rows`, on the mixed-measure density footing:
/// `-ln P(class)` on an atom row and `-ln P(continuous) - ln f(x)` on a continuous one.
fn mixed_nll(
    samples: &[BarDof],
    score_rows: Rows,
    dof: usize,
    atom_values: &[f32],
    continuous: &Continuous,
    class: &[f64],
) -> f64 {
    let classes = atom_values.len() + 1;
    let log_class: Vec<f64> = class.iter().map(|p| p.max(f64::MIN_POSITIVE).ln()).collect();
    let parts: Vec<(f64, u64)> = samples
        .par_chunks(EM_CHUNK)
        .enumerate()
        .map(|(chunk, block)| {
            let base = chunk * EM_CHUNK;
            let mut nats = 0.0f64;
            let mut scored = 0u64;
            for (offset, row) in block.iter().enumerate() {
                if !score_rows.takes(base + offset) {
                    continue;
                }
                let raw = row.to_array()[dof];
                if !raw.is_finite() {
                    continue;
                }
                scored += 1;
                match atom_values.iter().position(|a| *a == raw) {
                    Some(slot) => nats -= log_class[slot],
                    None => {
                        nats -= log_class[classes - 1];
                        match continuous_value(dof, row, atom_values) {
                            Some(x) => nats -= continuous.log_density(x),
                            None => nats = f64::INFINITY,
                        }
                    }
                }
            }
            (nats, scored)
        })
        .collect();
    let mut nats = 0.0f64;
    let mut scored = 0u64;
    for (n, s) in parts {
        nats += n;
        scored += s;
    }
    if scored == 0 {
        return f64::NAN;
    }
    nats / scored as f64
}

fn count_continuous<E>(samples: &[BarDof], rows: Rows, value_of: &E) -> u64
where
    E: Fn(&BarDof) -> Option<f64> + Send + Sync,
{
    samples
        .par_chunks(EM_CHUNK)
        .enumerate()
        .map(|(chunk, block)| {
            let base = chunk * EM_CHUNK;
            block
                .iter()
                .enumerate()
                .filter(|(offset, row)| rows.takes(base + offset) && value_of(row).is_some())
                .count() as u64
        })
        .sum()
}

fn fit_continuous<E>(
    samples: &[BarDof],
    rows: Rows,
    value_of: &E,
    kind: FamilyKind,
    k: usize,
    sd_floor: f64,
    truncation: Option<f64>,
    concentration_cap: f64,
) -> (Continuous, usize, usize)
where
    E: Fn(&BarDof) -> Option<f64> + Send + Sync,
{
    match kind {
        FamilyKind::BetaMixture => {
            let (m, sweeps, bad) = fit_beta_mixture(samples, rows, value_of, k, concentration_cap);
            (Continuous::Beta(m), sweeps, bad)
        }
        FamilyKind::LogNormalMixture => {
            let (m, sweeps) = fit_gaussian_mixture(samples, rows, value_of, k, sd_floor, None);
            (Continuous::LogNormal(m), sweeps, 0)
        }
        FamilyKind::TruncatedGaussianMixture | FamilyKind::GaussianMixture => {
            let (m, sweeps) =
                fit_gaussian_mixture(samples, rows, value_of, k, sd_floor, truncation);
            (Continuous::Gaussian(m), sweeps, 0)
        }
    }
}

fn fit_one_dof(
    samples: &[BarDof],
    supports: &BarSupports,
    dof: usize,
    args: &BarFamilyArgs,
    discrete_nll: f64,
    bins: usize,
    r_truncation: f64,
) -> DofFit {
    let kind = family_of(dof);
    let atom_values: Vec<f32> = supports.atoms(dof).iter().map(|a| a.value as f32).collect();
    let atoms: Vec<AtomCheck> = supports
        .atoms(dof)
        .iter()
        .map(|atom| AtomCheck {
            dof,
            value: atom.value,
            drawn_share: exact_share(samples, dof, atom.value),
            artifact_mass: atom.mass,
        })
        .collect();
    let atom_mass: f64 = atoms.iter().map(|a| a.drawn_share).sum();

    let floor = resolution_floor(supports, dof, bins);
    let extract_atoms = atom_values.clone();
    // The EM and the quantile probe see the FIT coordinate; every density evaluation downstream
    // sees the DOF's own coordinate and applies the Jacobian itself.
    let value_of = move |row: &BarDof| {
        continuous_value(dof, row, &extract_atoms).map(|x| fit_coordinate(dof, x))
    };

    let truncation = (dof == DOF_R).then_some(r_truncation);
    let concentration_cap = (0.25 / (floor * floor) - 1.0).max(1.0);

    let continuous_share = count_continuous(samples, Rows::All, &value_of) as f64
        / samples.len().max(1) as f64;
    let fit_class = class_probabilities(samples, Rows::Fit, dof, &atom_values);
    let all_class = class_probabilities(samples, Rows::All, dof, &atom_values);
    let fit_rows = count_continuous(samples, Rows::Fit, &value_of).max(1) as f64;

    let mut sweep = Vec::with_capacity(args.k_max - args.k_min + 1);
    let mut selected = (args.k_min, f64::INFINITY);
    for k in args.k_min..=args.k_max {
        let (fitted, sweeps, unconverged) = fit_continuous(
            samples,
            Rows::Fit,
            &value_of,
            kind,
            k,
            floor,
            truncation,
            concentration_cap,
        );
        let fit_nll = mixed_nll(samples, Rows::Fit, dof, &atom_values, &fitted, &fit_class);
        // Scored with the FIT rows' class probabilities, so the holdout figure is out of sample in
        // the class term too.
        let holdout_nll = mixed_nll(
            samples,
            Rows::Holdout,
            dof,
            &atom_values,
            &fitted,
            &fit_class,
        );
        let params = fitted.free_parameters() + atoms.len();
        let total_nats = fit_nll * fit_rows;
        sweep.push(SweepPoint {
            dof,
            components: k,
            fit_nll,
            holdout_nll,
            free_parameters: params,
            aic_per_bar: (2.0 * total_nats + 2.0 * params as f64) / fit_rows,
            bic_per_bar: (2.0 * total_nats + params as f64 * fit_rows.ln()) / fit_rows,
            em_sweeps: sweeps,
            unconverged_components: unconverged,
        });
        // Selection rule, declared up front: minimum HOLDOUT nats per bar.
        if holdout_nll.is_finite() && holdout_nll < selected.1 {
            selected = (k, holdout_nll);
        }
    }
    let (selected_components, holdout_nll) = selected;

    // The reported family is REFITTED on the full draw at the selected K, because the discrete
    // figure it is compared against is the entropy of the full draw's own histogram. The holdout
    // number that selected K is carried beside it rather than replaced by it.
    let (continuous, em_sweeps, unconverged_components) = fit_continuous(
        samples,
        Rows::All,
        &value_of,
        kind,
        selected_components,
        floor,
        truncation,
        concentration_cap,
    );
    let family_nll = mixed_nll(
        samples,
        Rows::All,
        dof,
        &atom_values,
        &continuous,
        &all_class,
    );

    // The discrete competitor on the family's own 90/10 protocol, so (c) has a symmetric footing as
    // well as the in-sample one the artifact records.
    let (discrete_holdout_nll, discrete_holdout_zero_mass_rows) =
        measure_discrete_holdout(samples, supports, dof, bins);

    let (grid_lo, grid_hi, empirical_density) =
        empirical_bin_density(samples, supports, dof, bins, &atom_values);
    let fitted_density: Vec<f64> = grid_lo
        .iter()
        .zip(grid_hi.iter())
        .map(|(lo, hi)| bin_average_density(&continuous, *lo, *hi) * continuous_share)
        .collect();
    let charted: f64 = grid_lo
        .iter()
        .zip(grid_hi.iter())
        .zip(fitted_density.iter())
        .map(|((lo, hi), density)| density * (hi - lo))
        .sum();
    let outside = match (grid_lo.first(), grid_hi.last()) {
        (Some(lo), Some(hi)) => continuous_share * continuous.mass_outside(*lo, *hi),
        _ => 0.0,
    };

    DofFit {
        dof,
        kind,
        selected_components,
        atoms,
        continuous,
        continuous_share,
        family_nll,
        holdout_nll,
        discrete_nll,
        discrete_free_parameters: bins - 1,
        discrete_holdout_nll,
        discrete_holdout_zero_mass_rows,
        resolution_floor: floor,
        grid_lo,
        grid_hi,
        empirical_density,
        fitted_density,
        integrated_mass: atom_mass + charted + outside,
        sweep,
        em_sweeps,
        unconverged_components,
    }
}

/// The draw's own per-bin density on the support's CONTINUOUS bins, and their edges. Atom rows are
/// excluded, so the histogram integrates to the continuous class share exactly as the fitted
/// density does.
fn empirical_bin_density(
    samples: &[BarDof],
    supports: &BarSupports,
    dof: usize,
    bins: usize,
    atom_values: &[f32],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let lower = supports.lower_bounds(dof);
    let upper = supports.upper_bounds(dof);
    let widths = supports.widths(dof);
    let keep: Vec<usize> = (0..bins).filter(|b| widths[*b] > 0.0).collect();
    let grid_lo: Vec<f64> = keep.iter().map(|b| lower[*b]).collect();
    let grid_hi: Vec<f64> = keep.iter().map(|b| upper[*b]).collect();

    let counts: Vec<u64> = samples
        .par_chunks(EM_CHUNK)
        .map(|block| {
            let mut local = vec![0u64; keep.len()];
            for row in block {
                let raw = row.to_array()[dof];
                if !raw.is_finite() || atom_values.contains(&raw) {
                    continue;
                }
                let value = raw as f64;
                // Kept bin lower bounds ascend, so the containing bin is the last one whose lower
                // bound does not exceed the value.
                let slot = grid_lo.partition_point(|lo| *lo <= value);
                if slot == 0 {
                    continue;
                }
                let slot = slot - 1;
                if value <= grid_hi[slot] {
                    local[slot] += 1;
                }
            }
            local
        })
        .reduce(
            || vec![0u64; keep.len()],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );

    let total = samples.len() as f64;
    let density: Vec<f64> = counts
        .iter()
        .zip(keep.iter())
        .map(|(count, bin)| *count as f64 / total / widths[*bin])
        .collect();
    (grid_lo, grid_hi, density)
}

/// Bin-AVERAGE density of the continuous family over `[lo, hi]`, by the composite midpoint rule.
///
/// Midpoint and never a rule with endpoint nodes: an `alpha < 1` Beta component has an integrable
/// singularity at zero, and evaluating there returns `+inf` and destroys the closure check.
fn bin_average_density(continuous: &Continuous, lo: f64, hi: f64) -> f64 {
    if !(hi > lo) {
        return f64::NAN;
    }
    let step = (hi - lo) / DENSITY_NODES as f64;
    let mut total = 0.0;
    for i in 0..DENSITY_NODES {
        total += continuous.density(lo + step * (i as f64 + 0.5));
    }
    total / DENSITY_NODES as f64
}

// ---------------------------------------------------------------------------
// Tail measurement
// ---------------------------------------------------------------------------

/// The [`TAIL_BUFFER`] largest `|r|` off the `r == 0` atom, ascending, plus the row counts.
///
/// Returns `(ordered, rows, continuous_rows)`. `rows` counts every FINITE row including the atom;
/// `continuous_rows` counts only those off it. The distinction is load-bearing downstream:
/// [`empirical_tail_slopes`] reads unconditional `1 - p` quantiles, so it must divide by `rows`,
/// and it stays correct on a buffer holding only off-atom magnitudes only because `r == 0` sorts
/// below every retained value.
pub fn upper_order_statistics(samples: &[BarDof]) -> (Vec<f64>, u64, u64) {
    let push = |heap: &mut BinaryHeap<Reverse<OrderedFloat<f64>>>, magnitude: OrderedFloat<f64>| {
        if heap.len() < TAIL_BUFFER {
            heap.push(Reverse(magnitude));
        } else if heap
            .peek()
            .map(|Reverse(m)| magnitude > *m)
            .unwrap_or(false)
        {
            heap.pop();
            heap.push(Reverse(magnitude));
        }
    };

    let parts: Vec<(BinaryHeap<Reverse<OrderedFloat<f64>>>, u64, u64)> = samples
        .par_chunks(EM_CHUNK)
        .map(|block| {
            let mut heap = BinaryHeap::with_capacity(TAIL_BUFFER + 1);
            let mut rows = 0u64;
            let mut continuous = 0u64;
            for row in block {
                let r = row.r as f64;
                if !r.is_finite() {
                    continue;
                }
                rows += 1;
                if r == 0.0 {
                    continue;
                }
                continuous += 1;
                push(&mut heap, OrderedFloat(r.abs()));
            }
            (heap, rows, continuous)
        })
        .collect();

    let mut merged = BinaryHeap::with_capacity(TAIL_BUFFER + 1);
    let mut rows = 0u64;
    let mut continuous = 0u64;
    for (heap, r, c) in parts {
        rows += r;
        continuous += c;
        for Reverse(magnitude) in heap {
            push(&mut merged, magnitude);
        }
    }
    let mut ordered: Vec<f64> = merged.into_iter().map(|Reverse(m)| m.into_inner()).collect();
    ordered.sort_unstable_by(f64::total_cmp);
    (ordered, rows, continuous)
}

/// Exceedance counts of `|r|` at each threshold, one bounded pass.
fn exceedance_counts(samples: &[BarDof], thresholds: &[f64]) -> Vec<u64> {
    samples
        .par_chunks(EM_CHUNK)
        .map(|block| {
            let mut local = vec![0u64; thresholds.len()];
            for row in block {
                let r = row.r as f64;
                if !r.is_finite() {
                    continue;
                }
                let magnitude = r.abs();
                for (slot, threshold) in thresholds.iter().enumerate() {
                    if magnitude > *threshold {
                        local[slot] += 1;
                    }
                }
            }
            local
        })
        .reduce(
            || vec![0u64; thresholds.len()],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        )
}

/// `P(|R| > x)` under the fitted mixed law for `r`, for `x > 0`: the atom at zero contributes
/// nothing, so it is the continuous class share times both truncated tails.
fn family_abs_exceedance(mixture: &GaussianMixture, continuous_share: f64, x: f64) -> f64 {
    continuous_share * (mixture.survival(x) + mixture.cumulative(-x))
}

/// The [`TAIL_LEVELS_P`] thresholds and the six pairwise log-log slopes between them.
///
/// `ordered` is [`upper_order_statistics`]'s ascending buffer and `rows` is its TOTAL finite row
/// count, atom rows included: the thresholds are unconditional `1 - p` quantiles of `|r|` over the
/// whole sample, and reading them off a buffer that holds only off-atom magnitudes is correct
/// precisely because `r == 0` sorts below everything retained. Passing `continuous_rows` here
/// instead would shift every threshold outward and silently change the slopes.
///
/// A power law `P(|r| > x) = C x^-alpha` gives `alpha = -(ln p_low - ln p_high) / (ln x_low -
/// ln x_high)` for any two levels, so the SPREAD of the six pairs over the four levels is the
/// finding: a single value would hide the curvature that decides whether there is a tail index at
/// all. `NaN` where two levels share a threshold, which happens when the buffer cannot reach the
/// outer level.
pub fn empirical_tail_slopes(ordered: &[f64], rows: u64) -> (Vec<f64>, Vec<EmpiricalSlope>) {
    let total = rows.max(1) as f64;
    let quantile_at = |p: f64| -> f64 {
        let from_top = (p * total).round().max(1.0) as usize;
        if from_top > ordered.len() {
            return f64::NAN;
        }
        ordered[ordered.len() - from_top]
    };
    let thresholds: Vec<f64> = TAIL_LEVELS_P.iter().map(|p| quantile_at(*p)).collect();

    let mut slopes = Vec::with_capacity(6);
    for i in 0..TAIL_LEVELS_P.len() {
        for j in (i + 1)..TAIL_LEVELS_P.len() {
            let (p_high, p_low) = (TAIL_LEVELS_P[i], TAIL_LEVELS_P[j]);
            let (x_high, x_low) = (thresholds[i], thresholds[j]);
            let log_x = x_low.ln() - x_high.ln();
            let alpha = if log_x != 0.0 && log_x.is_finite() {
                -(p_low.ln() - p_high.ln()) / log_x
            } else {
                f64::NAN
            };
            slopes.push(EmpiricalSlope {
                p_low,
                p_high,
                x_low,
                x_high,
                alpha,
            });
        }
    }
    (thresholds, slopes)
}

fn measure_tail(
    samples: &[BarDof],
    mixture: &GaussianMixture,
    continuous_share: f64,
    extremes: &RExtremes,
) -> TailFit {
    let (ordered, rows, continuous_rows) = upper_order_statistics(samples);
    let total = rows.max(1) as f64;

    let (_, empirical) = empirical_tail_slopes(&ordered, rows);
    // The SAME functional applied to the model instead of to the data, at the SAME thresholds.
    // Not a refit and not a second estimate of anything.
    let pairs: Vec<PairSlope> = empirical
        .iter()
        .map(|slope| {
            let log_x = slope.x_low.ln() - slope.x_high.ln();
            let s_high = family_abs_exceedance(mixture, continuous_share, slope.x_high);
            let s_low = family_abs_exceedance(mixture, continuous_share, slope.x_low);
            let fitted = if log_x != 0.0 && log_x.is_finite() && s_high > 0.0 && s_low > 0.0 {
                -(s_low.ln() - s_high.ln()) / log_x
            } else {
                f64::NAN
            };
            PairSlope {
                p_low: slope.p_low,
                p_high: slope.p_high,
                x_low: slope.x_low,
                x_high: slope.x_high,
                empirical: slope.alpha,
                fitted,
            }
        })
        .collect();

    let hill: Vec<HillPoint> = HILL_K
        .iter()
        .filter(|k| **k + 1 < ordered.len())
        .map(|k| {
            let threshold = ordered[ordered.len() - k - 1];
            let log_threshold = threshold.ln();
            let sum: f64 = ordered[ordered.len() - k..]
                .iter()
                .map(|x| x.ln() - log_threshold)
                .sum();
            let alpha = if sum > 0.0 {
                *k as f64 / sum
            } else {
                f64::NAN
            };
            HillPoint {
                k: *k,
                threshold,
                alpha,
                standard_error: alpha / (*k as f64).sqrt(),
            }
        })
        .collect();

    // Geometric grid, so the chart index is linear in `ln x` and a power law reads as a straight
    // line. Anchored at 10 bps: below that the equal-mass bins are dense and no tail question
    // arises.
    let x_lo = 1.0e-3f64;
    let x_hi = extremes.max_abs.max(2.0 * x_lo);
    let grid_x: Vec<f64> = (0..TAIL_GRID)
        .map(|i| {
            let t = i as f64 / (TAIL_GRID - 1) as f64;
            (x_lo.ln() + t * (x_hi.ln() - x_lo.ln())).exp()
        })
        .collect();
    let counts = exceedance_counts(samples, &grid_x);
    let grid: Vec<TailPoint> = grid_x
        .iter()
        .zip(counts.iter())
        .map(|(x, count)| TailPoint {
            threshold: *x,
            empirical_exceedance: *count as f64 / total,
            empirical_count: *count,
            fitted_exceedance: family_abs_exceedance(mixture, continuous_share, *x),
        })
        .collect();

    TailFit {
        rows,
        continuous_rows,
        max_abs: extremes.max_abs,
        min_r: extremes.min_r,
        max_r: extremes.max_r,
        pairs,
        hill,
        grid,
        measured_band: MEASURED_TAIL_BAND,
    }
}

/// The ruin table, held against BOTH bounds that could license a leverage cap: the corpus draw's
/// own extreme, and the discrete support's reachable range. They are not the same quantity and the
/// distinction is the whole content of (d) — the cap in force is licensed by the SUPPORT, which is
/// a declared modeling constant, and NOT by the data, which licenses almost nothing.
fn tabulate_ruin(extremes: &RExtremes, support_min_r: f64, support_max_r: f64) -> RuinLicence {
    let rows = RUIN_LEVERAGES
        .iter()
        .map(|leverage| {
            let inverse = 1.0 / leverage;
            let long = if inverse < 1.0 {
                -(1.0 - inverse).ln()
            } else {
                f64::INFINITY
            };
            let short = (1.0 + inverse).ln();
            let binding = long.min(short);
            RuinRow {
                leverage: *leverage,
                long_log_bound: long,
                short_log_bound: short,
                binding_log_bound: binding,
                binding_simple_return: binding.exp() - 1.0,
                // Side-wise, not against `max_abs`: the two sides carry different bounds and a
                // symmetric test would report the wrong side as the binding one.
                licensed_by_draw: -extremes.min_r <= long && extremes.max_r <= short,
                licensed_by_support: -support_min_r <= long && support_max_r <= short,
            }
        })
        .collect();
    RuinLicence {
        rows,
        draw_long_max_leverage: 1.0 / (1.0 - extremes.min_r.exp()),
        draw_short_max_leverage: 1.0 / (extremes.max_r.exp() - 1.0),
        draw_min_r: extremes.min_r,
        draw_max_r: extremes.max_r,
        support_min_r,
        support_max_r,
        support_long_max_leverage: 1.0 / (1.0 - support_min_r.exp()),
        support_short_max_leverage: 1.0 / (support_max_r.exp() - 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::DOF_W;
    use shared::report::{read_report, ReportKind};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tch::{Kind, Tensor};

    static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let unique = SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bar_family_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Deterministic xorshift, so every fixture is a function of its seed alone.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let u = self.uniform().max(1.0e-12);
            let v = self.uniform();
            (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
        }
    }

    /// A draw with the atom structure the real corpus has: a point mass on `r == 0`, flat bars
    /// with `s == 0` forcing `u == v == 0.5`, and `u` / `v` mass on both endpoints. Fitting on a
    /// smooth fixture would let a purely continuous family pass, which is the whole thing under
    /// test.
    fn atom_heavy_draw(rows: usize, seed: u64) -> Vec<BarDof> {
        let mut rng = Rng(seed | 1);
        (0..rows)
            .map(|_| {
                let flat = rng.uniform() < 0.15;
                let s = if flat {
                    0.0
                } else {
                    (0.004f64 * (1.0 + rng.normal().abs())).min(0.12)
                };
                let r = if rng.uniform() < 0.09 {
                    0.0
                } else {
                    (0.003 * rng.normal()).clamp(-0.085, 0.085)
                };
                let pick = |rng: &mut Rng| -> f64 {
                    let q = rng.uniform();
                    if q < 0.2 {
                        0.0
                    } else if q < 0.4 {
                        1.0
                    } else {
                        rng.uniform().clamp(1.0e-6, 1.0 - 1.0e-6)
                    }
                };
                let (u, v) = if flat {
                    (0.5, 0.5)
                } else {
                    (pick(&mut rng), pick(&mut rng))
                };
                BarDof {
                    r: r as f32,
                    s: s as f32,
                    u: u as f32,
                    v: v as f32,
                    w: (0.4 * rng.normal()) as f32,
                }
            })
            .collect()
    }

    fn fixture_args() -> BarFamilyArgs {
        BarFamilyArgs {
            corpus: CorpusFlags {
                data_dir: String::new(),
                resolution_secs: 300,
                min_bars: 0,
                split_bounds: None,
                derive_split_bounds: false,
                min_dollar_volume: 0.0,
            },
            supports: "<in-memory fixture>".to_owned(),
            output: String::new(),
            samples: 0,
            seed: 0x5EED,
            k_min: 2,
            k_max: 3,
            atom_tolerance: DEFAULT_ATOM_TOLERANCE,
        }
    }

    /// `ln_gamma`, `digamma`, `trigamma` and `erfc` are written out from the Bernoulli expansions
    /// and the incomplete-gamma identity, so they are CHECKED against libtorch's own
    /// implementations rather than trusted.
    #[test]
    fn the_special_functions_match_libtorch() {
        let probes: Vec<f64> = vec![
            1.0e-3, 0.05, 0.25, 0.5, 0.999, 1.0, 1.5, 2.0, 7.0, 15.5, 16.0, 33.0, 500.0, 1.0e5,
            5.0e5, 3.7e8,
        ];
        let tensor = Tensor::from_slice(&probes).to_kind(Kind::Double);
        let host = |t: Tensor| -> Vec<f64> { Vec::<f64>::try_from(t).expect("host copy") };

        for (i, (mine, theirs)) in probes
            .iter()
            .map(|x| ln_gamma(*x))
            .zip(host(tensor.lgamma()))
            .enumerate()
        {
            assert!(
                (mine - theirs).abs() <= 1.0e-10 * theirs.abs().max(1.0),
                "ln_gamma({}) = {mine}, libtorch {theirs}",
                probes[i]
            );
        }
        for (i, (mine, theirs)) in probes
            .iter()
            .map(|x| digamma(*x))
            .zip(host(tensor.digamma()))
            .enumerate()
        {
            assert!(
                (mine - theirs).abs() <= 1.0e-10 * theirs.abs().max(1.0),
                "digamma({}) = {mine}, libtorch {theirs}",
                probes[i]
            );
        }
        // `trigamma` is checked against CLOSED FORMS rather than against libtorch, because
        // libtorch is the less accurate of the two here: `polygamma(1, 0.5)` returns
        // 4.934802202073678 against the exact `pi^2 / 2 = 4.934802200544679`, an error of 1.5e-9,
        // while this implementation lands 2e-15 from it. Cross-checking against libtorch as well,
        // at a tolerance that states that gap rather than hiding it.
        for (x, exact) in [
            (0.5f64, PI * PI / 2.0),
            (1.0, PI * PI / 6.0),
            (2.0, PI * PI / 6.0 - 1.0),
            (3.0, PI * PI / 6.0 - 1.0 - 0.25),
        ] {
            let mine = trigamma(x);
            assert!(
                (mine - exact).abs() <= 1.0e-13 * exact.abs(),
                "trigamma({x}) = {mine}, exact {exact}"
            );
        }
        // The recurrence `psi'(x) - psi'(x + 1) = 1 / x^2` is an identity, so it pins the
        // implementation across the whole range including the shift boundary at 16.
        for x in &probes {
            let residual = trigamma(*x) - trigamma(*x + 1.0) - 1.0 / (x * x);
            assert!(
                residual.abs() <= 1.0e-13 * trigamma(*x).abs(),
                "the trigamma recurrence left {residual} at {x}"
            );
        }
        for (i, (mine, theirs)) in probes
            .iter()
            .map(|x| trigamma(*x))
            .zip(host(tensor.polygamma(1)))
            .enumerate()
        {
            assert!(
                (mine - theirs).abs() <= 1.0e-8 * theirs.abs().max(1.0e-9),
                "trigamma({}) = {mine}, libtorch {theirs}",
                probes[i]
            );
        }

        // `erfc` must hold RELATIVE accuracy into the far tail, because the tail chart reads
        // exceedance probabilities down to 1e-8 off it and an absolute tolerance would accept
        // returning zero there.
        let z: Vec<f64> = vec![
            -6.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 1.2247, 3.0, 5.0, 6.0, 8.0, 12.0,
        ];
        let z_tensor = Tensor::from_slice(&z).to_kind(Kind::Double);
        for (i, (mine, theirs)) in z
            .iter()
            .map(|x| erfc(*x))
            .zip(host(z_tensor.erfc()))
            .enumerate()
        {
            assert!(
                (mine - theirs).abs() <= 1.0e-11 * theirs.abs().max(1.0e-300),
                "erfc({}) = {mine:e}, libtorch {theirs:e}",
                z[i]
            );
        }
        assert_eq!(erfc(40.0), 0.0, "erfc must underflow to exactly zero, not NaN");
    }

    /// A truncated mixture is a DENSITY: it integrates to one over its truncation, its survival
    /// and cumulative functions are complements, and both are monotone.
    #[test]
    fn the_truncated_mixture_is_a_normalized_density() {
        let mixture = GaussianMixture {
            weights: vec![0.7, 0.3],
            means: vec![0.0, 0.004],
            sds: vec![0.001, 0.01],
            truncation: Some(0.09),
        };
        let panels = 200_000usize;
        let (lo, hi) = (-0.09f64, 0.09f64);
        let step = (hi - lo) / panels as f64;
        let mut total = mixture.density(lo) + mixture.density(hi);
        for i in 1..panels {
            let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
            total += weight * mixture.density(lo + step * i as f64);
        }
        let mass = total * step / 3.0;
        assert!(
            (mass - 1.0).abs() < 1.0e-8,
            "the truncated mixture integrates to {mass}, not one"
        );
        assert_eq!(mixture.density(0.1), 0.0, "density outside the truncation");
        for x in [-0.05f64, -0.001, 0.0, 0.002, 0.05] {
            let sum = mixture.survival(x) + mixture.cumulative(x);
            assert!(
                (sum - 1.0).abs() < 1.0e-10,
                "S({x}) + F({x}) = {sum}, not one"
            );
        }
        let mut previous = 1.0;
        for i in 0..64 {
            let x = -0.09 + 0.18 * i as f64 / 63.0;
            let survival = mixture.survival(x);
            assert!(survival <= previous + 1.0e-15, "survival rose at {x}");
            previous = survival;
        }
    }

    /// EM must RECOVER a mixture it was generated from. Without this every NLL below is a number
    /// with no claim attached.
    #[test]
    fn em_recovers_a_planted_gaussian_mixture() {
        let mut rng = Rng(0xBEEF_0001);
        let rows = 200_000usize;
        let samples: Vec<BarDof> = (0..rows)
            .map(|_| {
                let value = if rng.uniform() < 0.3 {
                    -0.02 + 0.002 * rng.normal()
                } else {
                    0.01 + 0.006 * rng.normal()
                };
                BarDof {
                    w: value as f32,
                    ..BarDof::default()
                }
            })
            .collect();
        let value_of = |row: &BarDof| Some(row.w as f64);
        let (fitted, sweeps) = fit_gaussian_mixture(&samples, Rows::All, &value_of, 2, 1.0e-7, None);
        assert!(sweeps > 1, "EM took {sweeps} sweeps");

        // Components come out in whatever order the quantile init put them; identify by mean.
        let (low, high) = if fitted.means[0] < fitted.means[1] {
            (0, 1)
        } else {
            (1, 0)
        };
        assert!(
            (fitted.means[low] + 0.02).abs() < 5.0e-4,
            "low mean {}",
            fitted.means[low]
        );
        assert!(
            (fitted.means[high] - 0.01).abs() < 5.0e-4,
            "high mean {}",
            fitted.means[high]
        );
        assert!(
            (fitted.sds[low] - 0.002).abs() < 2.0e-4,
            "low sd {}",
            fitted.sds[low]
        );
        assert!(
            (fitted.sds[high] - 0.006).abs() < 3.0e-4,
            "high sd {}",
            fitted.sds[high]
        );
        assert!(
            (fitted.weights[low] - 0.3).abs() < 0.01,
            "low weight {}",
            fitted.weights[low]
        );
    }

    /// The Beta M-step is a Newton MLE on the score equations, so it must recover a planted Beta —
    /// including at the concentration a near-degenerate spike demands.
    #[test]
    fn the_beta_m_step_solves_the_score_equations() {
        for (a, b) in [(2.0f64, 5.0f64), (0.5, 0.5), (1.0, 1.0), (5.0e5, 5.0e5)] {
            let total = a + b;
            let mean = a / total;
            let variance = a * b / (total * total * (total + 1.0));
            // The EXACT sufficient statistics of a Beta(a, b), so this tests the solver and not a
            // sampler.
            let (fit_a, fit_b, converged) = beta_mle(
                digamma(a) - digamma(total),
                digamma(b) - digamma(total),
                mean,
                variance,
                1.0e10,
            );
            assert!(converged, "Newton did not converge on Beta({a}, {b})");
            assert!(
                (fit_a - a).abs() <= 1.0e-5 * a.max(1.0),
                "alpha {fit_a} vs {a}"
            );
            assert!(
                (fit_b - b).abs() <= 1.0e-5 * b.max(1.0),
                "beta {fit_b} vs {b}"
            );
        }
    }

    /// The concentration cap is the resolution floor, and it MUST bind — otherwise a component
    /// collapsing onto a tick-repeated value would buy unbounded nats and the comparison against
    /// the discrete competitor would be meaningless. A capped component must NOT be reported as
    /// having reached the MLE.
    #[test]
    fn the_concentration_cap_binds_and_is_reported() {
        let a = 5.0e5f64;
        let total = 2.0 * a;
        let (fit_a, fit_b, converged) = beta_mle(
            digamma(a) - digamma(total),
            digamma(a) - digamma(total),
            0.5,
            a * a / (total * total * (total + 1.0)),
            1_000.0,
        );
        assert!(!converged);
        assert!(
            (fit_a + fit_b - 1_000.0).abs() < 1.0e-6,
            "the cap did not bind: {fit_a} + {fit_b}"
        );
    }

    /// The SHORT side binds, and the two candidate licences are kept apart: the DISCRETE SUPPORT's
    /// reachable range licenses the live cap, the CORPUS DRAW licenses essentially nothing, and a
    /// long-only reading of either overstates both.
    #[test]
    fn the_short_side_of_the_ruin_licence_always_binds() {
        // `lo[0]` and `hi[127]` of the live 300s support's `r` row, read off the artifact.
        let support_min_r = -0.088_331_513_106_822_97;
        let support_max_r = 0.088_038_101_792_335_51;
        // The draw's own extremes, as this pass measures them on the 4e6-row train-region sample.
        let extremes = RExtremes {
            min_r: -5.0f64.ln(),
            max_r: 5.0f64.ln(),
            max_abs: 5.0f64.ln(),
        };
        let licence = tabulate_ruin(&extremes, support_min_r, support_max_r);
        for row in &licence.rows {
            assert!(
                row.short_log_bound < row.long_log_bound,
                "at {}x the short bound {} did not bind against the long bound {}",
                row.leverage,
                row.short_log_bound,
                row.long_log_bound
            );
            assert_eq!(row.binding_log_bound, row.short_log_bound);
            // `1 + F * R_max` is exactly the ruin point on the short side, by construction.
            let residual = 1.0 - row.leverage * row.binding_simple_return;
            assert!(
                residual.abs() < 1.0e-12,
                "at {}x the binding return {} leaves {residual}",
                row.leverage,
                row.binding_simple_return
            );
            assert!(
                !row.licensed_by_draw,
                "a +/-{:.0} bps draw cannot license {}x",
                extremes.max_abs * 10_000.0,
                row.leverage
            );
        }
        assert!(licence.draw_short_max_leverage < licence.draw_long_max_leverage);
        assert_eq!(licence.draw_max_leverage(), licence.draw_short_max_leverage);
        assert!(
            licence.draw_max_leverage() < 1.0,
            "the draw's own extreme licenses {}x, which would be a licence for leverage",
            licence.draw_max_leverage()
        );

        // The support side is the licence in force, and the short side is what binds it. The
        // long-only figure is materially larger, which is exactly the overstatement being named.
        assert!(licence.support_short_max_leverage < licence.support_long_max_leverage);
        assert_eq!(
            licence.support_max_leverage(),
            licence.support_short_max_leverage
        );
        assert!(
            (licence.support_long_max_leverage - 11.829).abs() < 0.01,
            "the long-only support licence is {}",
            licence.support_long_max_leverage
        );
        assert!(
            (licence.support_short_max_leverage - 10.866).abs() < 0.01,
            "the binding support licence is {}",
            licence.support_short_max_leverage
        );

        let cap = licence
            .rows
            .iter()
            .find(|r| r.leverage == LEVERAGE_CAP)
            .expect("the live cap is tabulated");
        assert!(
            cap.licensed_by_support,
            "the live {LEVERAGE_CAP}x cap must be licensed by the support's own +/-{:.0} bps range",
            support_min_r.abs().max(support_max_r) * 10_000.0
        );
        let ceiling = licence
            .rows
            .iter()
            .find(|r| r.leverage == MAX_LEVERAGE)
            .expect("the declared ceiling is tabulated");
        assert!(
            !ceiling.licensed_by_support,
            "a {MAX_LEVERAGE}x ceiling needs r under {:.2} bps on the short side but the support \
             reaches {:.2} bps",
            ceiling.binding_log_bound * 10_000.0,
            support_max_r * 10_000.0
        );
    }

    /// The mixed NLL must be the mixed-measure quantity it claims: the class term plus the
    /// within-class mean negative log density, reproducing an ANALYTIC value on a law whose answer
    /// is known in closed form.
    #[test]
    fn the_mixed_nll_is_the_analytic_mixed_measure_entropy() {
        const ATOM: f64 = 0.25;
        const SPAN: f64 = 0.1;
        let rows = 400_000usize;
        let mut rng = Rng(0x0A70_D1);
        let samples: Vec<BarDof> = (0..rows)
            .map(|_| {
                let r = if rng.uniform() < ATOM {
                    0.0
                } else {
                    (rng.uniform() - 0.5) * SPAN
                };
                BarDof {
                    r: r as f32,
                    ..BarDof::default()
                }
            })
            .collect();
        // A one-component mixture with an enormous sd, truncated to the span, IS the uniform to
        // machine precision, so the analytic answer is exact rather than approximate.
        let continuous = Continuous::Gaussian(GaussianMixture {
            weights: vec![1.0],
            means: vec![0.0],
            sds: vec![1.0e6],
            truncation: Some(SPAN / 2.0),
        });
        let atoms = [0.0f32];
        let class = class_probabilities(&samples, Rows::All, DOF_R, &atoms);
        let measured = mixed_nll(&samples, Rows::All, DOF_R, &atoms, &continuous, &class);
        let atom_share = class[0];
        let expected = -atom_share * atom_share.ln()
            - (1.0 - atom_share) * (1.0 - atom_share).ln()
            + (1.0 - atom_share) * SPAN.ln();
        assert!(
            (measured - expected).abs() < 1.0e-6,
            "mixed nll {measured} is not the analytic {expected}"
        );
        // And it is NEGATIVE, which is the tell that it is a log DENSITY and not a probability.
        assert!(
            measured < 0.0,
            "a log-density nll must be able to go negative, got {measured}"
        );
        assert!(
            (atom_share - ATOM).abs() < 0.01,
            "the class probability is the empirical share by construction, got {atom_share}"
        );
    }

    /// The Hill index recovers a planted Pareto tail, and the same estimator applied to a
    /// THIN-tailed family comes out far steeper. That contrast is the entire content of
    /// deliverable (b).
    #[test]
    fn the_hill_index_recovers_a_planted_pareto_tail() {
        const ALPHA: f64 = 1.75;
        let rows = 2_000_000usize;
        let mut rng = Rng(0x7A11_0001);
        let samples: Vec<BarDof> = (0..rows)
            .map(|_| {
                let u = rng.uniform().max(1.0e-12);
                let magnitude = (1.0e-4 * u.powf(-1.0 / ALPHA)).min(0.5);
                let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                BarDof {
                    r: (sign * magnitude) as f32,
                    ..BarDof::default()
                }
            })
            .collect();
        let extremes = measure_r_extremes(&samples);
        let thin = GaussianMixture {
            weights: vec![1.0],
            means: vec![0.0],
            sds: vec![0.01],
            truncation: Some(extremes.max_abs),
        };
        let tail = measure_tail(&samples, &thin, 1.0, &extremes);

        for hill in &tail.hill {
            assert!(
                (hill.alpha - ALPHA).abs() < 6.0 * hill.standard_error + 0.02,
                "Hill at k={} gave {} +/- {} against a planted {ALPHA}",
                hill.k,
                hill.alpha,
                hill.standard_error
            );
        }
        let (lo, hi) = tail.empirical_span();
        assert!(
            (lo - ALPHA).abs() < 0.15 && (hi - ALPHA).abs() < 0.15,
            "the six pairwise slopes {lo}-{hi} do not bracket a planted {ALPHA}"
        );
        // The contrast that is the entire content of deliverable (b). A Gaussian family is NOT
        // uniformly steeper than a Pareto at every threshold — over the measured decade its
        // log-log slope can be shallower, because the thresholds sit inside its bulk. What it
        // cannot do is be CONSTANT: a Pareto's six pairwise slopes agree with each other, and the
        // Gaussian's fan out by a factor of several across the same four thresholds. That spread,
        // not the level, is what a power law does and a mixture of Gaussians does not.
        let (flo, fhi) = tail.fitted_span();
        let (elo, ehi) = tail.empirical_span();
        assert!(
            fhi < ALPHA - 0.3,
            "the thin family's steepest pairwise slope {fhi} is not materially below the planted \
             {ALPHA}"
        );
        assert!(
            fhi - flo > 4.0 * (ehi - elo),
            "the thin family's slopes span {flo}-{fhi} while the Pareto's span {elo}-{ehi}; the \
             discriminator is that a power law's slopes agree and a Gaussian's do not"
        );
        assert!(
            !tail.family_reaches_measured_band(),
            "a Gaussian tail must not be reported as reaching a 1.66-1.84 band"
        );
    }

    /// The discrete competitor's holdout figure is on the SAME footing as the artifact's own
    /// `scoring: density` marginal, which is the claim deliverable (c) rests on.
    ///
    /// Refitting the histogram on 90% and scoring the withheld 10% must land within sampling error
    /// of the full-sample plug-in entropy `marginal_nll_dof(Density)`, because 127 masses over
    /// hundreds of thousands of rows cost essentially nothing out of sample. Any disagreement of
    /// order a nat means the two sides are not measuring against the same measure — a dropped
    /// `ln(width)` term moves this by tens of nats, which is exactly the mistake that would make
    /// the whole (c) comparison meaningless.
    #[test]
    fn the_discrete_holdout_shares_the_artifacts_density_footing() {
        let rows = 400_000usize;
        let samples = atom_heavy_draw(rows, 0x9C31_0001);
        let supports = BarSupports::fit(&samples);
        let bins = NUM_BAR_BINS as usize;
        let in_sample = supports.marginal_nll_dof(BarScoring::Density);
        for dof in 0..BAR_DOF {
            let (holdout, zero_mass) = measure_discrete_holdout(&samples, &supports, dof, bins);
            assert_eq!(
                zero_mass, 0,
                "{} put {zero_mass} holdout rows in a bin the fit left empty on an equal-mass \
                 geometry over {rows} rows",
                BAR_DOF_NAMES[dof]
            );
            assert!(
                holdout.is_finite(),
                "{} discrete holdout is {holdout}",
                BAR_DOF_NAMES[dof]
            );
            assert!(
                (holdout - in_sample[dof]).abs() < 0.05,
                "{} discrete holdout {holdout} against the artifact's own density marginal {}; a \
                 gap this size means the two are not on the same measure",
                BAR_DOF_NAMES[dof],
                in_sample[dof]
            );
        }
    }

    /// A holdout row in a bin the fit rows left EMPTY is an infinite figure, and it is reported as
    /// one rather than smoothed. An unsmoothed histogram assigning zero mass to an event that
    /// happened is a real property of the discrete family, and hiding it behind a floor would make
    /// the discrete column look better than it is for a reason the reader could not see.
    #[test]
    fn a_discrete_bin_the_fit_never_saw_is_reported_as_infinite() {
        let rows = 5_000usize;
        // The withheld rows are exactly the top decile BY VALUE, so every bin above the 90th
        // percentile of `w` contains holdout rows and no fit rows at all.
        let samples: Vec<BarDof> = (0..rows)
            .map(|index| BarDof {
                w: if index % HOLDOUT_STRIDE == HOLDOUT_RESIDUE {
                    (1_000.0 + index as f64 * 1.0e-3) as f32
                } else {
                    (index as f64 * 1.0e-3) as f32
                },
                ..BarDof::default()
            })
            .collect();
        let supports = BarSupports::fit(&samples);
        let (holdout, zero_mass) =
            measure_discrete_holdout(&samples, &supports, DOF_W, NUM_BAR_BINS as usize);
        assert!(
            zero_mass > 0,
            "the fixture did not produce a bin the fit rows left empty"
        );
        assert!(
            holdout.is_infinite() && holdout > 0.0,
            "a zero-mass bin must score +inf, got {holdout}"
        );
    }

    /// The writer named in `pretrain_reports::CYCLE_EXEMPT` for every base this module owns.
    ///
    /// The exemption is honest only if something EXECUTES the writer: a stated reason is not
    /// coverage. Every base needs a whole fitted family battery over a drawn sample, which an
    /// in-run reporter cycle over step metrics does not have, which is why they are exempt and why
    /// this test is the exemption's entire justification.
    #[test]
    fn the_bar_family_fit_writes_every_registered_base() {
        let rows = 60_000usize;
        let samples = atom_heavy_draw(rows, 0x8A12_0001);
        let supports = BarSupports::fit(&samples);
        let args = fixture_args();
        let fit = build_fit(&samples, &supports, &args).expect("the family battery runs");

        // Atom reproduction is exact BY CONSTRUCTION on the same rows, which is what claim (a)
        // says; if this drifts, the claim is wrong and not merely imprecise.
        assert!(
            fit.worst_atom_deviation <= args.atom_tolerance,
            "worst atom deviation {:.3e}",
            fit.worst_atom_deviation
        );
        assert_eq!(fit.dofs.len(), BAR_DOF);
        for dof in &fit.dofs {
            assert!(
                dof.family_nll.is_finite(),
                "{} family nll is {}",
                dof.name(),
                dof.family_nll
            );
            assert!(
                dof.discrete_nll.is_finite(),
                "{} discrete nll is {}",
                dof.name(),
                dof.discrete_nll
            );
            assert!(
                dof.discrete_holdout_nll.is_finite() && dof.discrete_holdout_zero_mass_rows == 0,
                "{} discrete holdout is {} with {} rows in an empty bin",
                dof.name(),
                dof.discrete_holdout_nll,
                dof.discrete_holdout_zero_mass_rows
            );
            assert!(
                dof.holdout_nats_gained().is_finite(),
                "{} symmetric-footing gain is {}",
                dof.name(),
                dof.holdout_nats_gained()
            );
            assert!(
                (dof.integrated_mass - 1.0).abs() < 0.02,
                "{} fitted density plus atoms integrates to {}",
                dof.name(),
                dof.integrated_mass
            );
            assert_eq!(dof.sweep.len(), args.k_max - args.k_min + 1);
            assert_eq!(
                dof.grid_lo.len(),
                dof.empirical_density.len(),
                "{} grid and histogram disagree in length",
                dof.name()
            );
        }
        // `r` MUST carry an atom, or the fixture is not exercising the mixed likelihood at all —
        // and a point mass is exactly the fact a pure Gaussian mixture cannot reproduce.
        assert!(
            !fit.dofs[DOF_R].atoms.is_empty(),
            "the fixture must plant an r atom"
        );
        assert!(fit.tail.pairs.len() == 6, "six pairwise slopes are required");
        assert!(fit.ruin.rows.len() == RUIN_LEVERAGES.len());
        assert!(!fit.report_lines().is_empty());

        let dir = scratch_dir("bases");
        write_bar_family(&dir, &fit).expect("every chart writes");
        for base in BAR_FAMILY_BASES {
            assert!(
                shared::report::PRETRAIN_REPORT_BASES.contains(base),
                "{base} must be registered in shared::report::PRETRAIN_REPORT_BASES or the TUI \
                 never scans for it"
            );
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was not written");
            let read = read_report(&path).expect("the report reads back");
            let ReportKind::MultiLine { series } = &read.kind else {
                panic!("{base} must be a MultiLine chart");
            };
            assert!(
                series.iter().any(|s| s.values.iter().any(|v| v.is_finite())),
                "{base} carries no finite value, so it is a blank panel"
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
