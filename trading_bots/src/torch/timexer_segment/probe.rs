//! The frozen-trunk latent probe: how much long-horizon information the trunk latent already
//! carries that the trained head fails to route.
//!
//! # The question this exists to settle, and why nothing cheaper settles it
//!
//! Oracle per-horizon rescaling takes the h = 192 close MSE ratio from `1.0212` to `0.99695`
//! on `timexer-control-4k` step 3000, and by construction moves the within-timestamp IC by
//! nothing at all. Amplitude repair is therefore ~2.4% of MSE at the long end and exactly zero
//! information. Every remaining proposal that only re-scales, re-weights or re-calibrates the
//! existing forecast is bounded by that number. New long-horizon edge has to come from
//! information reaching the forecast that does not reach it today.
//!
//! There are exactly two places that information can be missing from, and they imply opposite
//! programmes:
//!
//! - **World A, extraction failure.** The frozen latent linearly predicts the h = 192 target
//!   BETTER than the trained head does. The trunk knows something the dense 192-row head and
//!   its count-normalized objective do not route, and an objective that concentrates gradient
//!   on fewer, higher-SNR pathways has something to collect.
//! - **World B, information ceiling.** A linear probe on the latent merely MATCHES the head.
//!   Then the head is already extracting what the latent knows, no auxiliary loss defined on
//!   latents can manufacture edge, and the honest levers are context, features, capacity or
//!   the target definition.
//!
//! The decision is pre-registered as a number in [`WORLD_A_GAIN`] and [`WORLD_B_BAND`], on the
//! PAIRED IC difference at `h >= `[`VERDICT_HORIZON_FLOOR`], and [`verdict`] applies it.
//!
//! # Which tensor, and at which point
//!
//! [`super::model::CausalPatchModel::backbone`] ends with `rms_norm(&state)` and then, under
//! `last_only`, narrows to token position `origins - 1` - the origin's own position. That
//! narrowed post-norm tensor is what this module probes, and it is probed because it is
//! *literally the tensor [`super::model::CausalPatchModel::head`] consumes as its `state`
//! argument*. Three consequences, all of them load-bearing:
//!
//! - **Post-final-RMSNorm, not pre.** The head reads the normalized stream. RMSNorm's gain is
//!   a per-token positive scalar, not a global one, so pre-norm and post-norm are NOT related
//!   by any single linear map: a probe fitted before the norm would be fitted on a different
//!   object than the head sees and a win could be attributed to the row scaling instead of to
//!   the representation. The comparison is only clean at the head's own input.
//! - **The origin token only.** Every scored quantity in this project is a final-origin
//!   quantity, so the probe reads the one position the forecast is emitted from.
//! - **Latent only, covariates excluded.** [`super::model::CausalPatchModel::head`]
//!   concatenates a 256-wide projection of the known-future calendar/gap covariates onto the
//!   512-wide latent before its hidden GEMM. The probe is NOT given that block. Its input is
//!   therefore a strict SUBSET of the head's input, which makes the test conservative in the
//!   only direction that matters: a probe that beats the head does so while seeing strictly
//!   less, and that gap cannot be explained by extra inputs.
//!
//! # Why a linear probe is the right instrument, and its limits
//!
//! A linear probe is a LOWER bound on the information in the latent and an honest one on the
//! head's own terms: the head is a two-layer MLP over that same latent, so anything a linear
//! map extracts, a trained head with a GELU hidden layer of 1024 units is expressively capable
//! of extracting. A probe win is then unambiguously an OPTIMIZATION/ROUTING failure rather
//! than a capacity one - which is exactly the claim a latent auxiliary objective would have to
//! rest on. The converse is weaker and is stated as such: a probe that ties the head bounds
//! only the LINEARLY decodable information, so World B is a verdict about linear structure,
//! and it is the right verdict to act on only because the head itself is a shallow decoder.
//!
//! # Why closed form, and why a rank-restricted twin
//!
//! No SGD anywhere. Every probe here is one symmetric eigendecomposition of the centred latent
//! second moment plus a diagonal solve, so the result is a deterministic function of the fit
//! partition and carries no learning-rate, no seed and no stopping rule that a sceptic could
//! attribute the answer to.
//!
//! Two classes, because an unconstrained 512-dimensional probe fitted on 433 k origins and
//! scored on a chronologically later block is credible but not self-evidently so:
//!
//! - [`ProbeClass::Ridge`] uses all 512 directions, `d + 1` fitted parameters.
//! - [`ProbeClass::Principal`] keeps only the top `r` eigenvectors of the latent's OWN second
//!   moment (`r = 8, 32`), so `r + 1` fitted parameters. The projector is estimated from the
//!   latent covariance and sees NO target, so a rank-8 probe cannot select 8 directions
//!   because they happen to correlate with the future - it is handed the 8 directions the
//!   representation itself spends the most variance on. That is a strictly harder test than
//!   reduced-rank regression, and it is the one that makes "the probe just memorized 512 free
//!   directions" unavailable as an explanation of a win.
//!
//! Both report the condition number of the system actually solved, beside the unregularized
//! one, because a probe that only wins through an ill-conditioned inverse is not evidence.
//!
//! # Leakage
//!
//! Impossible by construction rather than by discipline. [`Partitions::split`] is the ONLY way
//! to obtain origins here and it takes both populations at once, so no caller can hand the fit
//! and the score the same rows. It enforces, on realized timestamps and not on the boundary
//! arithmetic that produced them:
//!
//! 1. the fit and scored origin sets are disjoint as `(ticker, origin)` pairs;
//! 2. every bar any fit target reads completes strictly before the first scored origin
//!    ([`Blocks::spanning`]);
//! 3. the ridge coefficient is chosen on a chronologically LAST slice of the fit partition,
//!    itself purged from the inner fit block by the same target-reach rule, so the penalty is
//!    never selected on the scored split either.
//!
//! The fit partition is the corpus's reserved `[70%, 80%)` band, which checkpoint selection
//! never read (selection minimizes held-out-full NLL), so the probe is clean of selection bias
//! on the fit side as well as of leakage.
//!
//! # Cost
//!
//! Two forward passes over held-out data with grad disabled and no backward: one over the
//! ~434 k fit origins, one over the ~433 k scored origins. The added arithmetic is one fp64
//! `[H, d, B] x [B, d]` batched GEMM per batch for the Gram - `H·d²·B·2` = 0.94 GFLOP per
//! 256-row batch at `H = 7`, `d = 512` - against a trunk forward that is three orders of
//! magnitude larger. Host algebra is two `512x512` symmetric eigendecompositions per horizon,
//! about 0.4 GFLOP once.

use anyhow::{ensure, Context, Result};
use std::collections::HashSet;
use tch::{Device, Kind, Tensor};

use super::{calibration::Blocks, corpus::WindowRef, runner::CROSS_SECTION_MIN};

/// Share of the fit partition's origins reserved, chronologically LAST, for choosing the ridge
/// coefficient. Last rather than random: the penalty has to be chosen under the same
/// forward-in-time transfer the scored block will demand of it, and a random slice would let
/// the penalty be tuned on interleaved neighbours of its own training rows.
const LAMBDA_HOLDOUT_SHARE: f64 = 0.2;
/// Ceiling on the share of the scored population the outer wall-clock alignment may drop.
///
/// Pre-registered at five times the measured price (1.0% on the real corpus at the time of
/// writing) so it is a real bar rather than a rubber stamp: a corpus whose reserved partitions
/// interleave badly enough to cost a twentieth of the scored block is not the pairing this
/// experiment was designed on, and [`Partitions::split`] refuses instead of reporting a number
/// nobody can interpret.
const OUTER_PURGE_CEILING: f64 = 0.05;
/// Ridge grid, geometric, in units of the MEAN eigenvalue of the centred latent second moment,
/// so the grid is invariant to the latent's scale and to the population size. The latent is
/// RMS-normalized, so that mean is near 1 and the grid spans "nine decades below the typical
/// direction" to "three above it" - unregularized through fully collapsed.
const RIDGE_DECADES: (f64, f64) = (-6.0, 3.0);
const RIDGE_GRID: usize = 37;
/// Retained principal directions for the rank-restricted probes.
pub const PROBE_RANKS: &[usize] = &[8, 32];

/// The pre-registered decision, as a number rather than an adjective. Both thresholds are on
/// the PAIRED within-timestamp IC difference `probe - head`, at horizons at or beyond
/// [`VERDICT_HORIZON_FLOOR`], because the long end is the only place the question is open.
///
/// `+0.010` is about 3.5 iid standard errors at the measured full-split IC precision
/// (`SE ~= 0.0028`), and it is also an economically meaningful step beside a measured h = 64
/// IC of `0.0655`: a gain of that size is a 15% relative improvement in the decision metric,
/// not a rounding of it.
pub const WORLD_A_GAIN: f64 = 0.010;
/// Inside `+-0.005` in BOTH directions at every long horizon is World B and retires the whole
/// latent-objective line. Half the World A bar, so the two verdicts cannot both fire and the
/// band between them is reported as indeterminate rather than resolved by rounding.
pub const WORLD_B_BAND: f64 = 0.005;
/// Horizons at or beyond this bar count carry the verdict.
pub const VERDICT_HORIZON_FLOOR: usize = 64;

/// Which world the paired gaps put us in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum World {
    /// At least one probe class clears [`WORLD_A_GAIN`] at some horizon past the floor.
    ExtractionFailure,
    /// Every probe class is inside [`WORLD_B_BAND`] at every horizon past the floor.
    InformationCeiling,
    /// Neither. A gap between the two thresholds is not a verdict and is not reported as one.
    Indeterminate,
}

impl World {
    pub fn label(self) -> &'static str {
        match self {
            Self::ExtractionFailure => "World A, extraction failure",
            Self::InformationCeiling => "World B, information ceiling",
            Self::Indeterminate => "indeterminate, between the pre-registered thresholds",
        }
    }
}

/// Equal-population chronological tranches of the fit partition.
///
/// Three, not two: two points give a slope with no way to see that it is a slope rather than a
/// pair of noisy levels, and the residual of a three-point line is the cheapest available check
/// that the recency effect is monotone rather than a single odd tranche.
pub const RECENCY_TRANCHES: usize = 3;

/// Milliseconds in a Julian year, the unit the recency slope is quoted in.
const MS_PER_YEAR: f64 = 365.25 * 86_400_000.;

/// Assign each fit row to one of `count` equal-POPULATION chronological tranches.
///
/// Equal population and not equal duration: the tranches exist to hold fit-sample size fixed
/// while fit-block AGE varies, which is exactly the control the extraction ladder does not
/// provide, and equal-duration tranches would confound the two.
pub fn chronological_tranches(stamps: &[i64], count: usize) -> Result<Vec<usize>> {
    ensure!(
        count >= 2,
        "a recency slope needs at least two tranches, got {count}"
    );
    ensure!(!stamps.is_empty(), "no fit origins to tranche");
    let mut sorted: Vec<i64> = stamps.to_vec();
    sorted.sort_unstable();
    // Cut points by rank, then membership by timestamp, so every row on one instant lands in one
    // tranche and no cross-section is split across the boundary.
    let cuts: Vec<i64> = (1..count)
        .map(|index| sorted[(sorted.len() * index / count).min(sorted.len() - 1)])
        .collect();
    Ok(stamps
        .iter()
        .map(|stamp| cuts.partition_point(|cut| *cut <= *stamp))
        .collect())
}

/// How much of the probe's edge is recency rather than latent information.
///
/// The probe's fit block is strictly MORE RECENT than anything the trained head saw, so under a
/// non-stationary market part of any probe win is an advantage the head does not have. This
/// prices it from data the probe already reads: identical-size chronological tranches of the fit
/// partition, each fitted and scored on the SAME held-out block, give `dIC/d(fit-block age)` with
/// sample size held constant, and multiplying by the head's own age deficit converts it into a
/// threshold adjustment.
#[derive(Clone, Debug)]
pub struct Recency {
    /// Per tranche, oldest first: origin count and the mean origin instant.
    pub tranches: Vec<(usize, i64)>,
    pub horizons: Vec<usize>,
    /// Per horizon, IC of each tranche's probe on the scored block, oldest first.
    pub tranche_ic: Vec<Vec<f64>>,
    /// Per horizon, `dIC/dyear` of fit-block recency, by least squares over the tranches.
    pub slope_per_year: Vec<f64>,
    /// Years between the head's last training target bar and the fit block's mean instant.
    pub age_gap_years: f64,
    /// Per horizon, `slope · age_gap`, the number added to the World A bar.
    pub discount: Vec<f64>,
}

impl Recency {
    /// `None` when the horizon was not probed, so a caller can never silently read zero for a
    /// quantity that was not measured.
    pub fn discount(&self, horizon: usize) -> Option<f64> {
        self.horizons
            .iter()
            .position(|probed| *probed == horizon)
            .map(|index| self.discount[index])
    }

    /// Slope by least squares on (age in years, IC), then the discount at the head's age deficit.
    ///
    /// `head_last_target_ms` is the last bar the trained head's targets read. Everything is
    /// measured against the fit tranches' own mean instants, so nothing here depends on the
    /// nominal partition boundaries that the per-ticker ordinal construction makes unreliable.
    pub fn fit(
        tranches: Vec<(usize, i64)>,
        horizons: &[usize],
        tranche_ic: Vec<Vec<f64>>,
        head_last_target_ms: i64,
    ) -> Result<Self> {
        ensure!(
            tranches.len() >= 2,
            "a recency slope needs at least two tranches, got {}",
            tranches.len()
        );
        ensure!(
            tranche_ic.len() == horizons.len(),
            "{} IC rows against {} horizons",
            tranche_ic.len(),
            horizons.len()
        );
        let weighted: f64 = tranches
            .iter()
            .map(|(count, mid)| *count as f64 * *mid as f64)
            .sum();
        let total: f64 = tranches.iter().map(|(count, _)| *count as f64).sum();
        ensure!(total > 0., "the fit tranches hold no origins");
        let centre = weighted / total;
        let age_gap_years = (centre - head_last_target_ms as f64) / MS_PER_YEAR;
        // Recency in years, POSITIVE for a more recent tranche, so a positive slope means "a
        // fresher fit block scores better" and the discount comes out positive.
        let age: Vec<f64> = tranches
            .iter()
            .map(|(_, mid)| (*mid as f64 - centre) / MS_PER_YEAR)
            .collect();
        let mean_age = age.iter().sum::<f64>() / age.len() as f64;
        let spread: f64 = age.iter().map(|value| (value - mean_age).powi(2)).sum();
        let mut slope_per_year = Vec::with_capacity(horizons.len());
        let mut discount = Vec::with_capacity(horizons.len());
        for row in &tranche_ic {
            let usable = row.iter().all(|ic| ic.is_finite());
            let value = if usable && spread > 0. {
                let mean_ic = row.iter().sum::<f64>() / row.len() as f64;
                age.iter()
                    .zip(row)
                    .map(|(age, ic)| (age - mean_age) * (ic - mean_ic))
                    .sum::<f64>()
                    / spread
            } else {
                f64::NAN
            };
            slope_per_year.push(value);
            discount.push(value * age_gap_years);
        }
        Ok(Self {
            tranches,
            horizons: horizons.to_vec(),
            tranche_ic,
            slope_per_year,
            age_gap_years,
            discount,
        })
    }
}

/// Apply the pre-registered rule to the scored probes. The head's own row is skipped: it is the
/// baseline the gap is measured against and its gap is identically zero.
pub fn verdict(report: &ProbeReport, recency: Option<&Recency>) -> World {
    let long = |score: &HorizonScore| score.horizon >= VERDICT_HORIZON_FLOOR;
    let discount = |horizon: usize| {
        recency
            .and_then(|recency| recency.discount(horizon))
            .filter(|value| value.is_finite() && *value > 0.)
            .unwrap_or(0.)
    };
    let gaps: Vec<(usize, f64)> = report
        .forecasters
        .iter()
        .filter(|f| f.parameters > 0)
        .flat_map(|f| f.per_horizon.iter())
        .filter(|score| long(score) && score.gap.is_finite())
        .map(|score| (score.horizon, score.gap))
        .collect();
    if gaps.is_empty() {
        return World::Indeterminate;
    }
    // The World A bar is RAISED by the measured recency advantage and the World B band is NOT.
    // The asymmetry is the point: the probe is fitted on a window more recent than anything the
    // head trained on, so part of any win is recency rather than latent information the head
    // failed to route, while a FAILURE under that advantage is if anything stronger evidence of
    // an information ceiling than the flat band suggests. A discount that is missing, negative
    // or not finite contributes nothing, so an unmeasured discount can never loosen the gate.
    if gaps
        .iter()
        .any(|(horizon, gap)| *gap >= WORLD_A_GAIN + discount(*horizon))
    {
        return World::ExtractionFailure;
    }
    if gaps.iter().all(|(_, gap)| gap.abs() <= WORLD_B_BAND) {
        return World::InformationCeiling;
    }
    World::Indeterminate
}

/// One probe family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeClass {
    /// All `d` latent directions, ridge-penalized.
    Ridge,
    /// The top `r` eigenvectors of the latent's own centred second moment, ridge-penalized
    /// inside that subspace. The subspace is chosen without ever looking at a target.
    Principal(usize),
}

impl ProbeClass {
    /// Fitted parameters, intercept included. Labels carry this number because it is the whole
    /// reason the rank-restricted twin exists.
    pub fn parameters(self, width: usize) -> usize {
        match self {
            Self::Ridge => width + 1,
            Self::Principal(rank) => rank + 1,
        }
    }

    /// Report label. No `=` anywhere: `report_cli --var` splits a token on its first one.
    pub fn label(self, width: usize) -> String {
        match self {
            Self::Ridge => format!(
                "ridge linear latent probe over all {width} directions ({} fitted parameters)",
                self.parameters(width)
            ),
            Self::Principal(rank) => format!(
                "rank-{rank} principal latent probe ({} fitted parameters)",
                self.parameters(width)
            ),
        }
    }

    /// Every class the probe fits, in report order.
    pub fn all() -> Vec<Self> {
        std::iter::once(Self::Ridge)
            .chain(PROBE_RANKS.iter().map(|rank| Self::Principal(*rank)))
            .collect()
    }
}

/// The three origin populations, produced together so no caller can pair the wrong two.
///
/// `inner` fits the coefficients, `holdout` chooses the penalty, `scored` is where every
/// reported number is measured. All three are pairwise disjoint as `(ticker, origin)` pairs and
/// separated in TARGET-bar time, which is the separation that matters: a `pred_len`-bar
/// cumulative target reaches far past its own origin, so blocks split on origin time alone
/// would still share bars.
#[derive(Debug)]
pub struct Partitions {
    pub inner: Vec<WindowRef>,
    pub holdout: Vec<WindowRef>,
    pub scored: Vec<WindowRef>,
    /// Fit partition against scored partition.
    pub outer: Blocks,
    /// Inner fit block against the penalty-selection holdout.
    pub inner_blocks: Blocks,
    /// Fit origins dropped into the inner purge band because their targets reach into the
    /// penalty-selection holdout. Reported, not hidden: a purge that swallowed most of the fit
    /// partition would be a defect and has to be visible as one.
    pub purged: usize,
    /// Scored origins dropped because they begin before the fit block's last target bar
    /// completes. See [`Partitions::split`]: this is not leakage repair, it is the price of
    /// stating a wall-clock claim on a universe whose per-ticker partitions are not
    /// wall-clock aligned. Reported for the same reason `purged` is.
    pub outer_purged: usize,
}

/// `(reference, origin timestamp, timestamp of the last bar this origin's targets read)`.
pub type DatedRef = (WindowRef, i64, i64);

impl Partitions {
    /// Cut the three populations and PROVE the cuts, on realized timestamps.
    ///
    /// Refuses rather than repairs, with ONE deliberate exception stated here in full.
    ///
    /// # The outer wall-clock cut
    ///
    /// The corpus's reserved partitions are ordered per ticker: `boundaries[i]` is each
    /// ticker's OWN valid-bar ordinal, so `[70%, 80%)` and `[80%, 90%)` are fractions of that
    /// ticker's history. Across a universe of unequal histories the global extrema therefore
    /// interleave. MEASURED on the real corpus by
    /// `the_real_reserved_partitions_are_dated_and_their_ordering_is_measured`: the fit block's
    /// targets reach `1723739700000` (2024-08-15) while the earliest scored origin begins
    /// `1550178000000` (2019-02-14), five years earlier, on a different and longer-lived
    /// ticker. Per ticker the ordering is exact - 0 of 4,498 violate it - and not one of the
    /// 433,303 scored origins even shares a TIMESTAMP with a fit origin, so no bar and no
    /// instant is common to the two populations.
    ///
    /// A wall-clock claim still has to be true on the wall clock. So the scored block is cut
    /// to the origins that begin strictly after every fit target has completed, rather than
    /// the guard being softened to the per-ticker claim that would pass unaltered. The cut is
    /// a construction step with a MEASURED price - 1.0% of the scored population, 433,303 to
    /// ~428,900 - and it is refused outright if that price exceeds
    /// [`OUTER_PURGE_CEILING`], because past that point the pairing is no longer the
    /// experiment this subcommand was designed to run and the honest move is to report nothing.
    ///
    /// Every other ensure below is a leakage mode with no such escape: overlapping origin
    /// sets, a fit target that reads a bar the scored block also reads, a penalty chosen on
    /// rows the coefficients were fitted on, or an empty block that would make one of those
    /// checks vacuous.
    pub fn split(fit: &[DatedRef], scored: &[DatedRef]) -> Result<Self> {
        ensure!(
            !fit.is_empty() && !scored.is_empty(),
            "the latent probe needs a nonempty fit partition and a nonempty scored partition, \
             got {} and {}",
            fit.len(),
            scored.len()
        );
        let dated = |rows: &[DatedRef]| -> Vec<(i64, i64)> {
            rows.iter()
                .map(|(_, origin, reach)| (*origin, *reach))
                .collect()
        };
        // Origin disjointness FIRST, on the full populations, before anything is trimmed. A
        // shared origin is a different bug from an interleaved block and deserves to be named
        // as one; running it after the cut would let a genuinely overlapping pair be reported
        // as a wall-clock alignment failure.
        let scored_set: HashSet<(usize, usize)> = scored
            .iter()
            .map(|(r, _, _)| (r.ticker, r.origin))
            .collect();
        let shared = fit
            .iter()
            .filter(|(r, _, _)| scored_set.contains(&(r.ticker, r.origin)))
            .count();
        ensure!(
            shared == 0,
            "{shared} of {} fit origins are also scored origins; the probe would be fitted on \
             rows it reports out-of-sample scores for",
            fit.len()
        );
        let fit_reach = fit
            .iter()
            .map(|(_, _, reach)| *reach)
            .max()
            .expect("nonempty");
        let kept: Vec<DatedRef> = scored
            .iter()
            .copied()
            .filter(|(_, origin, _)| *origin > fit_reach)
            .collect();
        let outer_purged = scored.len() - kept.len();
        let share = outer_purged as f64 / scored.len() as f64;
        ensure!(
            share <= OUTER_PURGE_CEILING,
            "aligning the scored block behind the fit block's last target bar ({fit_reach}) \
             would drop {outer_purged} of {} scored origins ({:.1}%), past the {:.0}% ceiling; \
             the two reserved partitions are too badly interleaved on the wall clock to pair",
            scored.len(),
            100. * share,
            100. * OUTER_PURGE_CEILING
        );
        let scored = kept;
        ensure!(
            !scored.is_empty(),
            "no scored origin begins after the fit block's last target bar at {fit_reach}"
        );
        // Passes by construction after the cut above, and is kept as the postcondition that
        // proves it: the inequality this whole struct exists to guarantee is asserted on the
        // realized timestamps of the populations actually used, never inferred from the cut.
        let outer = Blocks::spanning(&dated(fit), &dated(&scored))?;
        // The penalty-selection cut: the chronologically last `LAMBDA_HOLDOUT_SHARE` of the fit
        // partition's ORIGINS, then the inner block purged back to the targets that complete
        // before the first held-back origin.
        let mut stamps: Vec<i64> = fit.iter().map(|(_, origin, _)| *origin).collect();
        stamps.sort_unstable();
        let index = ((stamps.len() as f64 * (1. - LAMBDA_HOLDOUT_SHARE)).floor() as usize)
            .min(stamps.len() - 1);
        let cutoff = stamps[index];
        let holdout: Vec<DatedRef> = fit
            .iter()
            .copied()
            .filter(|(_, origin, _)| *origin >= cutoff)
            .collect();
        ensure!(
            !holdout.is_empty(),
            "the penalty-selection holdout is empty at cutoff {cutoff}"
        );
        let first_holdout_origin = holdout
            .iter()
            .map(|(_, origin, _)| *origin)
            .min()
            .expect("nonempty");
        let inner: Vec<DatedRef> = fit
            .iter()
            .copied()
            .filter(|(_, _, reach)| *reach < first_holdout_origin)
            .collect();
        ensure!(
            !inner.is_empty(),
            "every fit origin's targets reach into the penalty-selection holdout, so there is \
             no purged inner block to fit coefficients on"
        );
        let inner_blocks = Blocks::spanning(&dated(&inner), &dated(&holdout))?;
        let holdout_set: HashSet<(usize, usize)> = holdout
            .iter()
            .map(|(r, _, _)| (r.ticker, r.origin))
            .collect();
        let inner_shared = inner
            .iter()
            .filter(|(r, _, _)| holdout_set.contains(&(r.ticker, r.origin)))
            .count();
        ensure!(
            inner_shared == 0,
            "{inner_shared} coefficient-fit origins are also penalty-selection origins"
        );
        Ok(Self {
            purged: fit.len() - inner.len() - holdout.len(),
            outer_purged,
            inner: inner.iter().map(|(r, _, _)| *r).collect(),
            holdout: holdout.iter().map(|(r, _, _)| *r).collect(),
            scored: scored.iter().map(|(r, _, _)| *r).collect(),
            outer,
            inner_blocks,
        })
    }
}

/// Device-resident fp64 sufficient statistics of the (latent, close target) joint moments, one
/// set per probed horizon.
///
/// Everything downstream - every ridge fit, every rank restriction, every penalty evaluation -
/// is a function of these six accumulators and nothing else, which is why the fit needs exactly
/// one pass over the fit partition regardless of how many penalties or ranks are tried.
///
/// fp64, not fp32: the Gram is a sum of ~434 k rank-one updates of a unit-RMS vector, so its
/// diagonal reaches `4e5` while the signal in the centred cross moment sits near `1e-2`. An
/// fp32 accumulator would lose the cross moment's low bits into the Gram's exponent.
pub struct Moments {
    width: i64,
    horizons: i64,
    count: Tensor,
    latent_sum: Tensor,
    gram: Tensor,
    cross: Tensor,
    target_sum: Tensor,
    target_square: Tensor,
}

impl Moments {
    pub fn new(width: i64, horizons: i64, device: Device) -> Self {
        let opts = (Kind::Double, device);
        Self {
            width,
            horizons,
            count: Tensor::zeros([horizons], opts),
            latent_sum: Tensor::zeros([horizons, width], opts),
            gram: Tensor::zeros([horizons, width, width], opts),
            cross: Tensor::zeros([horizons, width], opts),
            target_sum: Tensor::zeros([horizons], opts),
            target_square: Tensor::zeros([horizons], opts),
        }
    }

    /// One batch. `latent` is `[rows, width]`, `target` and `mask` are `[rows, horizons]`, all
    /// fp64 and all on the accumulator's device. The mask is applied here and only here, so a
    /// horizon's statistics are over exactly its own valid target bars.
    pub fn push(&mut self, latent: &Tensor, target: &Tensor, mask: &Tensor) {
        let masked_latent = mask.transpose(0, 1).unsqueeze(-1) * latent.unsqueeze(0);
        let masked_target = target * mask;
        self.gram += masked_latent.transpose(1, 2).matmul(&latent.unsqueeze(0));
        self.latent_sum += masked_latent.sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
        self.cross += (&masked_latent * masked_target.transpose(0, 1).unsqueeze(-1))
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
        self.count += mask.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        self.target_sum += masked_target.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        self.target_square +=
            (target.square() * mask).sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
    }

    /// One host transfer for the whole set.
    pub fn to_host(&self) -> Result<HostMoments> {
        let host = |tensor: &Tensor| tensor.to_device(Device::Cpu);
        let moments = HostMoments {
            width: self.width,
            horizons: self.horizons,
            count: host(&self.count),
            latent_sum: host(&self.latent_sum),
            gram: host(&self.gram),
            cross: host(&self.cross),
            target_sum: host(&self.target_sum),
            target_square: host(&self.target_square),
        };
        ensure!(
            moments.count.isfinite().all().int64_value(&[]) == 1
                && moments.gram.isfinite().all().int64_value(&[]) == 1
                && moments.cross.isfinite().all().int64_value(&[]) == 1,
            "the latent moment accumulators went nonfinite; a probe fitted on them would be \
             arithmetic, not measurement"
        );
        Ok(moments)
    }
}

/// [`Moments`] on the host, and the block algebra that turns two of them into one.
///
/// `Clone` is a tensor-level `shallow_clone` of six accumulators and exists for the scaling
/// ladder's prefix merge, which needs to retain each cumulative block while continuing to fold
/// the next shell into it.
#[derive(Debug)]
pub struct HostMoments {
    pub width: i64,
    pub horizons: i64,
    pub count: Tensor,
    pub latent_sum: Tensor,
    pub gram: Tensor,
    pub cross: Tensor,
    pub target_sum: Tensor,
    pub target_square: Tensor,
}

impl Clone for HostMoments {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            horizons: self.horizons,
            count: self.count.shallow_clone(),
            latent_sum: self.latent_sum.shallow_clone(),
            gram: self.gram.shallow_clone(),
            cross: self.cross.shallow_clone(),
            target_sum: self.target_sum.shallow_clone(),
            target_square: self.target_square.shallow_clone(),
        }
    }
}

impl HostMoments {
    /// Sufficient statistics are additive over disjoint blocks, which is what lets the final
    /// coefficients be refitted on the WHOLE fit partition at the penalty its held-back slice
    /// chose, without a second pass over the data.
    pub fn merge(&self, other: &Self) -> Result<Self> {
        ensure!(
            self.width == other.width && self.horizons == other.horizons,
            "moment blocks disagree on shape"
        );
        Ok(Self {
            width: self.width,
            horizons: self.horizons,
            count: &self.count + &other.count,
            latent_sum: &self.latent_sum + &other.latent_sum,
            gram: &self.gram + &other.gram,
            cross: &self.cross + &other.cross,
            target_sum: &self.target_sum + &other.target_sum,
            target_square: &self.target_square + &other.target_square,
        })
    }

    /// One horizon's centred, per-observation-normalized system: the latent covariance `A`, the
    /// latent/target covariance `c`, the latent mean, the target mean and the count.
    fn centred(&self, index: i64) -> Result<Centred> {
        let count = self.count.double_value(&[index]);
        ensure!(
            count > self.width as f64,
            "horizon index {index} has {count} valid target bars against {} latent dimensions; \
             the normal equations are rank deficient before any penalty",
            self.width
        );
        let latent_mean = self.latent_sum.get(index) / count;
        let target_mean = self.target_sum.double_value(&[index]) / count;
        let covariance = self.gram.get(index) / count
            - latent_mean
                .reshape([-1, 1])
                .matmul(&latent_mean.reshape([1, -1]));
        let cross = self.cross.get(index) / count - &latent_mean * target_mean;
        Ok(Centred {
            count,
            latent_mean,
            target_mean,
            target_variance: self.target_square.double_value(&[index]) / count
                - target_mean * target_mean,
            covariance,
            cross,
        })
    }
}

struct Centred {
    count: f64,
    latent_mean: Tensor,
    target_mean: f64,
    target_variance: f64,
    covariance: Tensor,
    cross: Tensor,
}

/// One fitted probe: the frozen map, and everything a sceptic needs to discount it.
pub struct ProbeFit {
    pub horizon: usize,
    pub class: ProbeClass,
    pub parameters: usize,
    /// The selected penalty, in units of the mean eigenvalue.
    pub ridge: f64,
    /// Condition number of the system actually solved.
    pub condition: f64,
    /// Condition number of the same system with no penalty at all. A probe whose win only
    /// appears when this is astronomical is inverting noise.
    pub raw_condition: f64,
    /// Pooled correlation the selected penalty achieved on the penalty-selection holdout. This
    /// is a within-fit-partition number and is NOT the reported result; it exists so a probe
    /// that transfers to the scored block far worse than to its own holdout is visible.
    pub holdout_correlation: f64,
    /// `[width]` fp64 on the host.
    pub weight: Tensor,
    pub intercept: f64,
}

/// Fit every class at every horizon: coefficients on `inner`, penalty on `holdout`, then a
/// refit of the coefficients on `inner + holdout` at the chosen penalty.
///
/// The refit is not a shortcut around the holdout. The penalty is a single scalar per (horizon,
/// class) chosen on rows the coefficients never saw; spending the held-back rows on the
/// coefficients afterwards cannot leak anything into the SCORED block, which neither step ever
/// touched, and it is what keeps the reported probe fitted on the whole reserved partition
/// rather than on 80% of it.
pub fn fit_all(
    inner: &HostMoments,
    holdout: &HostMoments,
    horizons: &[usize],
) -> Result<Vec<ProbeFit>> {
    ensure!(
        inner.horizons == horizons.len() as i64 && holdout.horizons == horizons.len() as i64,
        "the moment blocks cover {} and {} horizons against {} probed",
        inner.horizons,
        holdout.horizons,
        horizons.len()
    );
    let full = inner.merge(holdout)?;
    let width = inner.width;
    let grid: Vec<f64> = (0..RIDGE_GRID)
        .map(|i| {
            let (low, high) = RIDGE_DECADES;
            10f64.powf(low + (high - low) * i as f64 / (RIDGE_GRID - 1) as f64)
        })
        .collect();
    let mut fits = Vec::with_capacity(horizons.len() * ProbeClass::all().len());
    for (index, horizon) in horizons.iter().enumerate() {
        let index = index as i64;
        let selection = inner.centred(index)?;
        let (eigenvalues, eigenvectors) = selection.covariance.linalg_eigh("L");
        let smallest = eigenvalues.double_value(&[0]);
        let largest = eigenvalues.double_value(&[width - 1]);
        ensure!(
            largest > 0. && smallest > -1e-6 * largest,
            "the centred latent second moment at h={horizon} is not positive semidefinite \
             (eigenvalues span {smallest:.3e} to {largest:.3e}); the accumulation is wrong"
        );
        let mean_eigenvalue = eigenvalues.mean(Kind::Double).double_value(&[]);
        let rotated = eigenvectors
            .transpose(0, 1)
            .matmul(&selection.cross.reshape([-1, 1]))
            .reshape([-1]);
        let evaluation = holdout.centred(index)?;
        let full_moments = full.centred(index)?;
        let (full_eigenvalues, full_eigenvectors) = full_moments.covariance.linalg_eigh("L");
        let full_mean = full_eigenvalues.mean(Kind::Double).double_value(&[]);
        let full_rotated = full_eigenvectors
            .transpose(0, 1)
            .matmul(&full_moments.cross.reshape([-1, 1]))
            .reshape([-1]);
        for class in ProbeClass::all() {
            let kept = match class {
                ProbeClass::Ridge => width,
                ProbeClass::Principal(rank) => rank as i64,
            };
            ensure!(
                kept > 0 && kept <= width,
                "a rank-{kept} probe is not expressible in {width} latent dimensions"
            );
            // MINIMUM held-out squared error, not maximum correlation. Correlation is
            // scale-invariant, so it is indifferent between a probe and the same probe shrunk
            // by a thousand: the grid's argmax would then be decided by rounding, and the
            // selected probe's amplitude - which the reported MSE ratio is quadratic in - would
            // be arbitrary. Squared error is the criterion the ridge is a solution to, it is
            // scale-sensitive, and it is the one criterion under which a shrunk probe is
            // correctly refused. The correlation is still measured, and reported, as the
            // diagnostic that says whether the transfer to the scored split degraded.
            let mut best: Option<(f64, HoldoutFit)> = None;
            for penalty in &grid {
                let weight = reconstruct(
                    &eigenvectors,
                    &eigenvalues,
                    &rotated,
                    kept,
                    penalty * mean_eigenvalue,
                );
                let intercept =
                    selection.target_mean - weight.dot(&selection.latent_mean).double_value(&[]);
                let measured = holdout_fit(holdout, index, &evaluation, &weight, intercept);
                if measured.error.is_finite()
                    && best
                        .as_ref()
                        .is_none_or(|(_, seen)| measured.error < seen.error)
                {
                    best = Some((*penalty, measured));
                }
            }
            let (penalty, measured) = best.unwrap_or((
                grid[grid.len() / 2],
                HoldoutFit {
                    error: f64::NAN,
                    correlation: f64::NAN,
                },
            ));
            let scaled = penalty * full_mean;
            let weight = reconstruct(
                &full_eigenvectors,
                &full_eigenvalues,
                &full_rotated,
                kept,
                scaled,
            );
            let intercept =
                full_moments.target_mean - weight.dot(&full_moments.latent_mean).double_value(&[]);
            let retained_low = full_eigenvalues.double_value(&[width - kept]);
            let retained_high = full_eigenvalues.double_value(&[width - 1]);
            fits.push(ProbeFit {
                horizon: *horizon,
                class,
                parameters: class.parameters(width as usize),
                ridge: penalty,
                condition: (retained_high + scaled) / (retained_low + scaled),
                raw_condition: if retained_low > 0. {
                    retained_high / retained_low
                } else {
                    f64::INFINITY
                },
                holdout_correlation: measured.correlation,
                weight,
                intercept,
            });
        }
    }
    ensure!(
        fits.len() == horizons.len() * ProbeClass::all().len(),
        "the fit produced {} probes for {} horizons and {} classes",
        fits.len(),
        horizons.len(),
        ProbeClass::all().len()
    );
    Ok(fits)
}

/// `w = V_r (Λ_r + λI)⁻¹ V_rᵀ c`, with `V_r` the top-`kept` eigenvectors. Eigenvalues arrive
/// ascending from `linalg_eigh`, so "top" is the tail.
fn reconstruct(
    eigenvectors: &Tensor,
    eigenvalues: &Tensor,
    rotated: &Tensor,
    kept: i64,
    penalty: f64,
) -> Tensor {
    let width = eigenvalues.size()[0];
    let filter = Tensor::zeros([width], (Kind::Double, Device::Cpu));
    let _ = filter.narrow(0, width - kept, kept).fill_(1.0);
    let scaled = rotated * filter / (eigenvalues + penalty).clamp_min(f64::MIN_POSITIVE);
    eigenvectors.matmul(&scaled.reshape([-1, 1])).reshape([-1])
}

/// What one candidate probe scores on the penalty-selection block.
struct HoldoutFit {
    /// Mean squared error per valid target bar. The selection criterion.
    error: f64,
    /// Pooled Pearson correlation, scale-invariant, reported as a diagnostic only.
    correlation: f64,
}

/// Score `w·z + a` on the penalty-selection block entirely from that block's sufficient
/// statistics: `Σ(y - f)² = Σy² - 2Σfy + Σf²`, and every term on the right is a contraction of
/// the six accumulators. No second pass over any data, which is why a 37-point grid crossed
/// with three classes costs nothing at all.
fn holdout_fit(
    moments: &HostMoments,
    index: i64,
    centred: &Centred,
    weight: &Tensor,
    intercept: f64,
) -> HoldoutFit {
    let count = centred.count;
    let latent_sum = moments.latent_sum.get(index);
    let linear = weight.dot(&latent_sum).double_value(&[]);
    let forecast_sum = linear + intercept * count;
    let quadratic = weight
        .dot(
            &moments
                .gram
                .get(index)
                .matmul(&weight.reshape([-1, 1]))
                .reshape([-1]),
        )
        .double_value(&[]);
    let forecast_square = quadratic + 2. * intercept * linear + intercept * intercept * count;
    let joint = weight.dot(&moments.cross.get(index)).double_value(&[])
        + intercept * moments.target_sum.double_value(&[index]);
    let target_square = moments.target_square.double_value(&[index]);
    let forecast_mean = forecast_sum / count;
    let covariance = joint / count - forecast_mean * centred.target_mean;
    let variance = forecast_square / count - forecast_mean * forecast_mean;
    HoldoutFit {
        error: (target_square - 2. * joint + forecast_square) / count,
        correlation: if variance > 0. && centred.target_variance > 0. {
            covariance / (variance * centred.target_variance).sqrt()
        } else {
            f64::NAN
        },
    }
}

/// Rungs in the extraction scaling ladder: seven nested fit sets spanning a 64x range of fit
/// rows. The exponent of a power law is read from the lever arm in `log N`, so a 64x span over
/// seven rungs pins it better than a wider population over a narrower range would.
pub const SCALING_LEVELS: usize = 7;

/// Which disjoint shell each fit row belongs to, given a nested ladder of keep-masks.
///
/// `ladder` comes from `teacher::whole_timestamp_ladder`, coarsest first, and its nesting is
/// asserted here rather than trusted: a silently non-nested ladder yields a curve that looks
/// perfectly well behaved and means nothing, because the rungs would then carry independent
/// sampling noise instead of a shared common mode. Shell `i` is "in rung `i` but not rung
/// `i-1`", so the shells are disjoint, they partition the population, and prefix-merging their
/// moments reconstructs every nested fit set at the cost of ONE pass.
pub fn scaling_shells(ladder: &[Vec<bool>]) -> Result<Vec<usize>> {
    ensure!(
        ladder.len() >= 3,
        "a scaling curve needs at least three rungs, got {}",
        ladder.len()
    );
    let rows = ladder[0].len();
    for (index, rung) in ladder.iter().enumerate() {
        ensure!(
            rung.len() == rows,
            "ladder rung {index} covers {} rows against {rows} on the coarsest",
            rung.len()
        );
    }
    for index in 1..ladder.len() {
        let escaped = (0..rows)
            .filter(|row| ladder[index - 1][*row] && !ladder[index][*row])
            .count();
        ensure!(
            escaped == 0,
            "ladder rung {} drops {escaped} rows that rung {} kept, so the rungs are not nested \
             and the fitted exponent would be read off independent draws",
            index,
            index - 1
        );
    }
    let finest = ladder.last().expect("checked nonempty");
    let missing = finest.iter().filter(|kept| !**kept).count();
    ensure!(
        missing == 0,
        "the finest ladder rung omits {missing} of {rows} rows; it must be the whole fit \
         population or `IC(N)` at the largest N is not the number the rest of this report quotes"
    );
    Ok((0..rows)
        .map(|row| {
            ladder
                .iter()
                .position(|rung| rung[row])
                .expect("the finest rung keeps every row")
        })
        .collect())
}

/// One rung: how many fit rows it had, and what the probe fitted on them scored.
#[derive(Clone, Copy, Debug)]
pub struct ScalingRung {
    /// REALIZED fit origins, counted rather than taken from the nominal share. Cross-sections
    /// are unequal in width, so a 64x-coarser stamp set is not a 64x-smaller row count.
    pub origins: usize,
    pub timestamps: usize,
    /// Selected penalty, so a rung that regularized its way out of the curve is visible.
    pub ridge: f64,
    pub ic: f64,
    pub ic_se: f64,
}

/// `IC(N) = IC_inf - a·N^-b`, fitted deterministically.
#[derive(Clone, Debug)]
pub struct ScalingCurve {
    pub rungs: Vec<ScalingRung>,
    /// Extrapolated IC at infinite fit sample. This, not the largest rung, is the quantity the
    /// experiment exists to estimate.
    pub asymptote: f64,
    pub amplitude: f64,
    pub exponent: f64,
    /// Root mean squared IC residual of the fit, in IC units, so a curve the power law does not
    /// describe is visible as one rather than reported as a clean asymptote.
    pub residual: f64,
}

/// Exponent by grid, intercept and amplitude in closed form at each candidate.
///
/// No SGD and no iterative solver: for a FIXED `b` the model is linear in `(IC_inf, a)` against
/// the regressor `N^-b`, so each candidate costs one two-parameter normal equation and the grid
/// is a deterministic scan. That is the whole reason the exponent is gridded rather than
/// optimized - it makes the result a function of the data and the grid alone.
pub fn fit_scaling(rungs: &[ScalingRung]) -> Result<ScalingCurve> {
    let usable: Vec<&ScalingRung> = rungs
        .iter()
        .filter(|rung| rung.ic.is_finite() && rung.origins > 0)
        .collect();
    ensure!(
        usable.len() >= 3,
        "a three-parameter curve needs at least three rungs with a measured IC, got {}",
        usable.len()
    );
    let mut best: Option<(f64, f64, f64, f64)> = None;
    for step in 0..=60 {
        let exponent = 0.05 * (2.0f64 / 0.05).powf(step as f64 / 60.);
        let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0., 0., 0., 0., 0.);
        for rung in &usable {
            let x = (rung.origins as f64).powf(-exponent);
            sx += x;
            sy += rung.ic;
            sxx += x * x;
            sxy += x * rung.ic;
            n += 1.;
        }
        let denominator = n * sxx - sx * sx;
        if denominator.abs() < f64::MIN_POSITIVE {
            continue;
        }
        let slope = (n * sxy - sx * sy) / denominator;
        let intercept = (sy - slope * sx) / n;
        let error: f64 = usable
            .iter()
            .map(|rung| {
                let x = (rung.origins as f64).powf(-exponent);
                (rung.ic - intercept - slope * x).powi(2)
            })
            .sum();
        if best.is_none_or(|(_, _, _, seen)| error < seen) {
            best = Some((exponent, intercept, -slope, error));
        }
    }
    let (exponent, asymptote, amplitude, error) = best.context("no usable exponent on the grid")?;
    Ok(ScalingCurve {
        rungs: rungs.to_vec(),
        asymptote,
        amplitude,
        exponent,
        residual: (error / usable.len() as f64).sqrt(),
    })
}

/// The frozen probes as one device-resident GEMM: `[width, classes·horizons]` weights and the
/// matching intercepts, so scoring every class at every horizon costs one matmul per batch.
pub struct ProbeBank {
    pub weight: Tensor,
    pub intercept: Tensor,
    pub columns: i64,
}

impl ProbeBank {
    /// Column order is class-major then horizon, which is the layout [`PairedScorer`] assumes
    /// for its forecaster blocks.
    pub fn new(fits: &[ProbeFit], horizons: &[usize], width: i64, device: Device) -> Result<Self> {
        let classes = ProbeClass::all();
        ensure!(
            fits.len() == classes.len() * horizons.len() && !horizons.is_empty(),
            "the bank needs one fit per class per horizon, got {} for {} classes and {} horizons",
            fits.len(),
            classes.len(),
            horizons.len()
        );
        let mut columns = Vec::with_capacity(fits.len());
        let mut intercepts = Vec::with_capacity(fits.len());
        for class in &classes {
            for horizon in horizons {
                let fit = fits
                    .iter()
                    .find(|fit| fit.class == *class && fit.horizon == *horizon)
                    .with_context(|| {
                        format!(
                            "no {} was fitted at h={horizon}",
                            class.label(width as usize)
                        )
                    })?;
                columns.push(fit.weight.shallow_clone());
                intercepts.push(fit.intercept);
            }
        }
        let weight = Tensor::stack(&columns, 1).to_device(device);
        ensure!(
            weight.size() == vec![width, columns.len() as i64],
            "the probe bank is {:?}, not [{width}, {}]",
            weight.size(),
            columns.len()
        );
        Ok(Self {
            columns: columns.len() as i64,
            weight,
            intercept: Tensor::from_slice(&intercepts)
                .to_device(device)
                .reshape([1, -1]),
        })
    }

    /// `[rows, columns]` forecasts from `[rows, width]` fp64 latents.
    pub fn forecast(&self, latent: &Tensor) -> Tensor {
        latent.matmul(&self.weight) + &self.intercept
    }
}

/// Per-timestamp and pooled moments for several forecasters on ONE pass over ONE origin set.
///
/// Paired by construction: the head's forecast and every probe's forecast are reduced from the
/// same rows in the same batch against the same target and the same mask, so the reported
/// difference is a difference of statistics on one draw rather than a difference across two
/// draws whose sampling noise would not cancel.
pub struct PairedScorer {
    horizons: i64,
    forecasters: i64,
    group_count: i64,
    /// `[groups, forecasters·horizons]`, forecaster-major.
    valid: Tensor,
    forecast: Tensor,
    forecast_square: Tensor,
    target: Tensor,
    target_square: Tensor,
    joint: Tensor,
    /// `[7, forecasters·horizons]`: count, Σf, Σf², Σy, Σy², Σfy, Σ(y-f)².
    pooled: Tensor,
}

impl PairedScorer {
    pub fn new(horizons: i64, forecasters: i64, group_count: i64, device: Device) -> Self {
        let opts = (Kind::Double, device);
        let width = forecasters * horizons;
        let groups = || Tensor::zeros([group_count, width], opts);
        Self {
            horizons,
            forecasters,
            group_count,
            valid: groups(),
            forecast: groups(),
            forecast_square: groups(),
            target: groups(),
            target_square: groups(),
            joint: groups(),
            pooled: Tensor::zeros([7, width], opts),
        }
    }

    /// `group` is `[rows]` int64 dense timestamp ranks; `forecast` is `[rows,
    /// forecasters·horizons]`; `target` and `mask` are `[rows, horizons]`. All fp64 on device.
    pub fn push(&mut self, group: &Tensor, forecast: &Tensor, target: &Tensor, mask: &Tensor) {
        let repeats = self.forecasters;
        let mask = mask.repeat([1, repeats]);
        let target = target.repeat([1, repeats]) * &mask;
        let forecast = forecast * &mask;
        let scatter = |accumulator: &mut Tensor, source: &Tensor| {
            *accumulator = accumulator.index_add(0, group, source);
        };
        scatter(&mut self.valid, &mask);
        scatter(&mut self.forecast, &forecast);
        scatter(&mut self.forecast_square, &(&forecast * &forecast));
        scatter(&mut self.target, &target);
        scatter(&mut self.target_square, &(&target * &target));
        scatter(&mut self.joint, &(&forecast * &target));
        let column = |t: &Tensor| t.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        self.pooled += Tensor::stack(
            &[
                column(&mask),
                column(&forecast),
                column(&(&forecast * &forecast)),
                column(&target),
                column(&(&target * &target)),
                column(&(&forecast * &target)),
                column(&((&target - &forecast).square() * &mask)),
            ],
            0,
        );
    }

    /// Per-forecaster, per-horizon scores plus the paired difference against forecaster 0.
    ///
    /// The within-timestamp IC follows the scorer's own convention exactly - a timestamp
    /// contributes only when it holds at least [`CROSS_SECTION_MIN`] valid names and both the
    /// forecast and the target have spread in it - because a probe measured on a different
    /// eligibility rule than the head would not be comparable to the head at all.
    ///
    /// The PAIRED difference is restricted further, to timestamps usable for EVERY forecaster
    /// at that horizon. Averaging each forecaster over its own eligible set and subtracting
    /// would let a difference come from a difference in populations.
    pub fn finish(&self, labels: &[String], horizons: &[usize]) -> Result<ProbeReport> {
        ensure!(
            labels.len() as i64 == self.forecasters && horizons.len() as i64 == self.horizons,
            "the scorer covers {} forecasters over {} horizons against {} labels and {} horizons",
            self.forecasters,
            self.horizons,
            labels.len(),
            horizons.len()
        );
        let inverse = self.valid.clamp_min(1.).reciprocal();
        let mean_forecast = &self.forecast * &inverse;
        let mean_target = &self.target * &inverse;
        let covariance = &self.joint * &inverse - &mean_forecast * &mean_target;
        let spread = ((&self.forecast_square * &inverse - mean_forecast.square()).clamp_min(0.)
            * (&self.target_square * &inverse - mean_target.square()).clamp_min(0.))
        .sqrt();
        let usable = self
            .valid
            .ge(CROSS_SECTION_MIN as f64)
            .logical_and(&spread.gt(1e-12))
            .to_kind(Kind::Double);
        let ic = covariance / spread.clamp_min(1e-30) * &usable;
        // Common eligibility per horizon: the product over the forecaster blocks of one
        // horizon's usable flags, broadcast back over the blocks.
        let blocked = usable.reshape([self.group_count, self.forecasters, self.horizons]);
        let common = blocked
            .prod_dim_int(1, true, Kind::Double)
            .expand([self.group_count, self.forecasters, self.horizons], false)
            .reshape([self.group_count, self.forecasters * self.horizons]);
        let baseline = ic
            .reshape([self.group_count, self.forecasters, self.horizons])
            .narrow(1, 0, 1)
            .expand([self.group_count, self.forecasters, self.horizons], false)
            .reshape([self.group_count, self.forecasters * self.horizons]);
        let gap = (&ic - baseline) * &common;
        let column = |t: &Tensor| t.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        let reduced = Tensor::stack(
            &[
                column(&usable),
                column(&ic),
                column(&ic.square()),
                column(&common),
                column(&gap),
                column(&gap.square()),
            ],
            0,
        );
        let width = (self.forecasters * self.horizons) as usize;
        let stats = Vec::<f64>::try_from(
            Tensor::cat(&[reduced, self.pooled.shallow_clone()], 0)
                .to_device(Device::Cpu)
                .flatten(0, -1),
        )?;
        let at = |row: usize, column: usize| stats[row * width + column];
        let mut forecasters = Vec::with_capacity(labels.len());
        for (block, label) in labels.iter().enumerate() {
            let mut per_horizon = Vec::with_capacity(horizons.len());
            for (index, horizon) in horizons.iter().enumerate() {
                let column = block * horizons.len() + index;
                let moments = at(0, column);
                let ic_mean = at(1, column) / moments.max(1.);
                let ic_variance = (at(2, column) / moments.max(1.) - ic_mean * ic_mean).max(0.);
                let paired = at(3, column);
                let gap_mean = at(4, column) / paired.max(1.);
                let gap_variance = (at(5, column) / paired.max(1.) - gap_mean * gap_mean).max(0.);
                let count = at(6, column);
                ensure!(
                    count > 0.,
                    "{label} scored no valid target bar at h={horizon}"
                );
                let forecast_mean = at(7, column) / count;
                let target_mean = at(9, column) / count;
                let persistence = at(10, column) / count;
                ensure!(
                    persistence > 0.,
                    "the persistence baseline at h={horizon} is zero; no ratio is defined"
                );
                let forecast_variance = at(8, column) / count - forecast_mean * forecast_mean;
                let joint = at(11, column) / count - forecast_mean * target_mean;
                let offset = (2. * forecast_mean * target_mean - forecast_mean * forecast_mean)
                    / persistence;
                let demeaned = if forecast_variance > 0. {
                    joint * joint / forecast_variance / persistence
                } else {
                    0.
                };
                per_horizon.push(HorizonScore {
                    horizon: *horizon,
                    ic: if moments > 0. { ic_mean } else { f64::NAN },
                    ic_se: if moments >= 2. {
                        (ic_variance / moments).sqrt()
                    } else {
                        f64::NAN
                    },
                    cross_sections: moments,
                    mse_ratio: at(12, column) / count / persistence,
                    oracle_mse_ratio: 1. - offset - demeaned,
                    optimal_gain: if forecast_variance > 0. {
                        joint / forecast_variance
                    } else {
                        f64::NAN
                    },
                    gap: if block == 0 {
                        0.
                    } else if paired > 0. {
                        gap_mean
                    } else {
                        f64::NAN
                    },
                    gap_se: if block == 0 {
                        0.
                    } else if paired >= 2. {
                        (gap_variance / paired).sqrt()
                    } else {
                        f64::NAN
                    },
                    paired_cross_sections: paired,
                    ridge: f64::NAN,
                    condition: f64::NAN,
                    raw_condition: f64::NAN,
                    holdout_correlation: f64::NAN,
                });
            }
            forecasters.push(Forecaster {
                label: label.clone(),
                parameters: 0,
                per_horizon,
            });
        }
        Ok(ProbeReport {
            horizons: horizons.to_vec(),
            forecasters,
        })
    }
}

/// One forecaster's score at one horizon, all on the scored split.
#[derive(Clone, Debug)]
pub struct HorizonScore {
    pub horizon: usize,
    /// Mean over scored timestamps of the within-timestamp cross-ticker correlation.
    pub ic: f64,
    pub ic_se: f64,
    pub cross_sections: f64,
    /// Close-channel market-neutral MSE against close-anchored persistence.
    pub mse_ratio: f64,
    /// The same forecast with its amplitude corrected and nothing else changed. Reported for
    /// probe and head alike so the two are compared with the amplitude defect neutralized on
    /// BOTH sides - without it the panel re-measures amplitude instead of information.
    pub oracle_mse_ratio: f64,
    pub optimal_gain: f64,
    /// This forecaster's IC minus the head's, averaged over timestamps usable for every
    /// forecaster. Zero on the head's own row by definition.
    pub gap: f64,
    /// Standard error of that PAIRED mean, from the dispersion of the per-timestamp
    /// differences. Not the difference of two independent standard errors: the two ICs share a
    /// draw and their sampling errors are strongly positively correlated, so treating them as
    /// independent would overstate the error by a large factor and make World B unfalsifiable.
    pub gap_se: f64,
    pub paired_cross_sections: f64,
    pub ridge: f64,
    pub condition: f64,
    pub raw_condition: f64,
    pub holdout_correlation: f64,
}

#[derive(Clone, Debug)]
pub struct Forecaster {
    pub label: String,
    /// Fitted parameters. `0` marks the trained head, which was not fitted here.
    pub parameters: usize,
    pub per_horizon: Vec<HorizonScore>,
}

#[derive(Clone, Debug)]
pub struct ProbeReport {
    pub horizons: Vec<usize>,
    pub forecasters: Vec<Forecaster>,
}

impl ProbeReport {
    /// Fold the fit diagnostics onto the scored rows they belong to, and stamp the parameter
    /// counts. Called once, after scoring, so a row always carries both halves of its evidence.
    pub fn attach(&mut self, fits: &[ProbeFit], width: usize) {
        for forecaster in self.forecasters.iter_mut() {
            for score in forecaster.per_horizon.iter_mut() {
                let Some(fit) = fits.iter().find(|fit| {
                    fit.horizon == score.horizon && fit.class.label(width) == forecaster.label
                }) else {
                    continue;
                };
                forecaster.parameters = fit.parameters;
                score.ridge = fit.ridge;
                score.condition = fit.condition;
                score.raw_condition = fit.raw_condition;
                score.holdout_correlation = fit.holdout_correlation;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `count` origins per ticker, one bar apart, targets reaching `reach` bars past the origin.
    fn dated(ticker: usize, first: usize, count: usize, reach: i64) -> Vec<DatedRef> {
        (0..count)
            .map(|i| {
                let origin = first + i;
                (
                    WindowRef { ticker, origin },
                    origin as i64 * 1000,
                    (origin as i64 + reach) * 1000,
                )
            })
            .collect()
    }

    /// A non-nested ladder is refused, and a nested one partitions the rows exactly once.
    ///
    /// This is the assertion the whole scaling curve rests on. Nested rungs share their common
    /// sampling mode, so the DIFFERENCES between rungs - which is all a power law is - are far
    /// better determined than the levels; independent draws would give each rung its own noise
    /// of order `1/sqrt(timestamps)`, which at the coarse end is the same size as the curvature
    /// the exponent is read from. A silently non-nested ladder therefore produces a curve that
    /// looks well behaved and means nothing.
    #[test]
    fn a_ladder_that_is_not_nested_is_refused_and_a_nested_one_partitions_the_rows() {
        let rows = 32;
        let nested: Vec<Vec<bool>> = [8usize, 4, 2, 1]
            .iter()
            .map(|stride| (0..rows).map(|row| row % stride == 0).collect())
            .collect();
        let shells = scaling_shells(&nested).unwrap();
        assert_eq!(shells.len(), rows);
        // Every row lands in exactly one shell, and the prefix counts are the rung sizes.
        for (index, rung) in nested.iter().enumerate() {
            let prefix = shells.iter().filter(|shell| **shell <= index).count();
            assert_eq!(prefix, rung.iter().filter(|kept| **kept).count());
        }
        // A rung that DROPS a row a coarser rung kept is the failure mode, and it is named.
        let mut broken = nested.clone();
        broken[2][0] = false;
        let error = scaling_shells(&broken).unwrap_err().to_string();
        assert!(error.contains("not nested"), "{error}");
        // A finest rung that is not the whole population is refused too: `IC(N)` at the largest
        // N has to be the same number the rest of the report quotes.
        let mut partial = nested.clone();
        partial[3][1] = false;
        let error = scaling_shells(&partial).unwrap_err().to_string();
        assert!(error.contains("whole fit population"), "{error}");
    }

    /// The curve fit recovers a planted power law and reports its own misfit.
    #[test]
    fn the_scaling_fit_recovers_a_planted_power_law() {
        let rung = |origins: usize| ScalingRung {
            origins,
            timestamps: origins / 10,
            ridge: 1e-3,
            ic: 0.08 - 0.9 * (origins as f64).powf(-0.5),
            ic_se: 0.001,
        };
        let rungs: Vec<ScalingRung> = [
            6_800usize, 13_600, 27_200, 54_400, 108_800, 217_600, 433_721,
        ]
        .map(rung)
        .to_vec();
        let curve = fit_scaling(&rungs).unwrap();
        // Tolerances are the GRID's resolution, not a hope: the exponent grid is 61 geometric
        // points from 0.05 to 2.0, so consecutive candidates differ by a factor of 40^(1/60) =
        // 1.0634 and the nearest one to 0.5 can be 3.2% away. The intercept absorbs that
        // mismatch, which is why the asymptote lands within 2e-4 rather than exactly.
        assert!(
            (curve.asymptote - 0.08).abs() < 5e-4,
            "asymptote {}",
            curve.asymptote
        );
        assert!(
            (curve.exponent - 0.5).abs() < 0.5 * 0.032,
            "exponent {}",
            curve.exponent
        );
        assert!(curve.residual < 5e-5, "residual {}", curve.residual);
        // Fewer than three rungs cannot pin three parameters and is refused rather than fitted.
        assert!(fit_scaling(&rungs[..2]).is_err());
    }

    /// The leakage proof. Every population the probe touches is cut by one function, and this
    /// asserts the three disjointness facts that make the reported scores out of sample:
    /// coefficient rows never appear among scored rows, penalty rows never appear among either,
    /// and no fit target reads a bar at or after the first scored origin.
    #[test]
    fn no_origin_the_probe_fits_on_is_ever_an_origin_it_scores() {
        let reach = 192;
        let fit: Vec<DatedRef> = (0..3).flat_map(|t| dated(t, 0, 4000, reach)).collect();
        let scored: Vec<DatedRef> = (0..3).flat_map(|t| dated(t, 4400, 4000, reach)).collect();
        let partitions = Partitions::split(&fit, &scored).unwrap();
        let key = |refs: &[WindowRef]| -> HashSet<(usize, usize)> {
            refs.iter().map(|r| (r.ticker, r.origin)).collect()
        };
        let (inner, holdout, scored_keys) = (
            key(&partitions.inner),
            key(&partitions.holdout),
            key(&partitions.scored),
        );
        assert!(!inner.is_empty() && !holdout.is_empty() && !scored_keys.is_empty());
        assert!(
            inner.is_disjoint(&holdout),
            "penalty rows were also fitted on"
        );
        assert!(
            inner.is_disjoint(&scored_keys),
            "coefficients were fitted on scored rows"
        );
        assert!(
            holdout.is_disjoint(&scored_keys),
            "the penalty was chosen on scored rows"
        );
        // Disjointness of ORIGINS is necessary and not sufficient: cumulative targets overlap.
        // Both purges are stated in target-bar time and both must be strictly positive.
        assert!(partitions.outer.purge_gap_ms > 0);
        assert!(partitions.inner_blocks.purge_gap_ms > 0);
        assert_eq!(
            partitions.inner_blocks.calibration_last_target_ms
                + partitions.inner_blocks.purge_gap_ms,
            partitions.inner_blocks.evaluation_first_origin_ms
        );
        // The penalty holdout is the chronologically LAST fifth, and the purge band is exactly
        // the origins whose targets reach into it - one target reach per ticker.
        assert_eq!(partitions.holdout.len(), 3 * 800);
        assert_eq!(partitions.purged, 3 * reach as usize);
        // Every purged origin is one whose target reaches into the holdout, and no more.
        assert!(partitions
            .inner
            .iter()
            .all(|r| (r.origin as i64 + reach) * 1000
                < partitions.inner_blocks.evaluation_first_origin_ms));
    }

    /// Each leakage mode is named by the error it raises, because "leakage" is three different
    /// bugs and a probe that reports the wrong one sends the reader to the wrong fix.
    ///
    /// A shared ORIGIN is always refused. A shared BAR - scored origins that begin before a fit
    /// target completes - is trimmed when the trim is cheap and refused when it is not, which
    /// is the one repair [`Partitions::split`] performs and the reason `outer_purged` is a
    /// reported field rather than an internal detail.
    #[test]
    fn overlapping_populations_are_refused_or_priced() {
        // Origins 0..499, targets reaching to 507.
        let fit = dated(0, 0, 500, 8);
        // Shares its first 100 origins with the fit block: always refused, never trimmed.
        let error = Partitions::split(&fit, &dated(0, 400, 500, 8))
            .unwrap_err()
            .to_string();
        assert!(error.contains("also scored origins"), "{error}");
        // Origin-disjoint but target-overlapping by 8 of 500 scored origins: 1.6% is under the
        // 5% ceiling, so those 8 are dropped and the rest is scored.
        let trimmed = Partitions::split(&fit, &dated(0, 500, 500, 8)).unwrap();
        assert_eq!(trimmed.outer_purged, 8);
        assert_eq!(trimmed.scored.len(), 492);
        assert!(trimmed.outer.purge_gap_ms > 0);
        // A whole scored block sitting under the fit block's reach - the shape the real corpus
        // produced, on another ticker so no origin is shared - blows the ceiling and is refused.
        let error = Partitions::split(&fit, &dated(1, 0, 500, 8))
            .unwrap_err()
            .to_string();
        assert!(error.contains("past the 5% ceiling"), "{error}");
        // Origin-disjoint AND target-disjoint costs nothing.
        let clean = Partitions::split(&fit, &dated(0, 510, 500, 8)).unwrap();
        assert_eq!(clean.outer_purged, 0);
        assert_eq!(clean.scored.len(), 500);
    }

    /// The selected ridge must not move when only the TARGET's scale moves.
    ///
    /// Raised against this design directly: the market-neutral target's within-window variance
    /// ratio is measured at 0.50 / 0.39 / 0.35 of random-walk scaling at h = 8 / 32 / 192, so
    /// target variance is a strong and non-monotone function of horizon, and the penalty is
    /// chosen on squared error whose floor moves with exactly that variance. If the criterion
    /// were scale-sensitive, the probe would be silently more regularized at one end of the
    /// horizon axis than the other for no reason connected to signal.
    ///
    /// It is not, and the reason is structural rather than lucky: the penalty is expressed in
    /// units of the mean eigenvalue of the LATENT second moment, which no horizon touches, and
    /// scaling `y` by `s` scales the solution `w = (A + λI)⁻¹c` by `s` and every candidate's
    /// holdout squared error by `s²`, leaving the argmin fixed. This asserts that identity end
    /// to end through the real solve, on a horizon pair differing by 100x in scale and in
    /// nothing else, so the remaining horizon-to-horizon variation in the reported `ridge` is
    /// signal-to-noise and nothing else.
    #[test]
    fn the_selected_penalty_does_not_move_when_only_the_target_scale_does() {
        let (width, rows, horizons) = (32i64, 4096i64, 2i64);
        let cpu = Device::Cpu;
        let truth = Tensor::zeros([width], (Kind::Double, cpu));
        let _ = truth.narrow(0, 0, 1).fill_(0.4);
        let mut inner = Moments::new(width, horizons, cpu);
        let mut holdout = Moments::new(width, horizons, cpu);
        let mut stream = 0u64;
        for (block, batches) in [(&mut inner, 12), (&mut holdout, 6)] {
            for _ in 0..batches {
                stream += 1;
                let latent =
                    Tensor::from_slice(&normals((rows * width) as usize, stream * 977 + 1))
                        .reshape([rows, width]);
                let signal = latent.matmul(&truth.reshape([-1, 1])).reshape([-1]);
                let noise = Tensor::from_slice(&normals(rows as usize, stream * 977 + 2));
                let base = &signal + &noise;
                // Horizon 0 and horizon 1 are the SAME target, 100x apart.
                let target = Tensor::stack(&[base.shallow_clone(), base * 100.], 1);
                block.push(
                    &latent,
                    &target,
                    &Tensor::ones([rows, horizons], (Kind::Double, cpu)),
                );
            }
        }
        let fits = fit_all(
            &inner.to_host().unwrap(),
            &holdout.to_host().unwrap(),
            &[1, 8],
        )
        .unwrap();
        for class in ProbeClass::all() {
            let small = fits
                .iter()
                .find(|f| f.horizon == 1 && f.class == class)
                .unwrap();
            let large = fits
                .iter()
                .find(|f| f.horizon == 8 && f.class == class)
                .unwrap();
            assert_eq!(
                small.ridge, large.ridge,
                "{class:?} chose {} at unit scale and {} at 100x, so the penalty axis is \
                 contaminated by target variance",
                small.ridge, large.ridge
            );
            // And the fitted map really did rescale, so the invariance is not a degenerate
            // "both collapsed to zero" agreement.
            let ratio = large.weight.double_value(&[0]) / small.weight.double_value(&[0]);
            assert!(
                (ratio - 100.).abs() < 1e-6,
                "{class:?} weight ratio {ratio} is not the 100x target ratio"
            );
        }
    }

    /// Standard normals from a self-contained xorshift, NOT from `tch::manual_seed`.
    ///
    /// The global torch RNG is process-wide and the test harness runs these threads in
    /// parallel, so a seeded `Tensor::randn` here draws a different sample depending on which
    /// other test happened to run beside it - measured during development as the same assertion
    /// producing 0.0326 and 0.0213 on two runs of identical code. A statistical assertion on a
    /// shared RNG is a flake generator, so the fixture owns its own stream.
    fn normals(count: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(count + 1);
        while out.len() < count {
            let radius = (-2.0 * uniform().ln()).sqrt();
            let angle = std::f64::consts::TAU * uniform();
            out.push(radius * angle.cos());
            out.push(radius * angle.sin());
        }
        out.truncate(count);
        out
    }

    /// A probe fitted on a latent that genuinely carries the target must recover it, and one
    /// fitted on noise must not. Both are checked through the real closed-form path - moment
    /// accumulation, eigendecomposition, penalty selection on a held-back block - so the test
    /// exercises the estimator rather than a paraphrase of it.
    #[test]
    fn the_closed_form_probe_recovers_a_planted_direction_and_refuses_noise() {
        // Wide enough that every entry of `PROBE_RANKS` is expressible, which is itself part of
        // the contract: a rank the latent cannot carry is refused rather than silently clamped.
        let (width, rows, horizons) = (64i64, 4096i64, 2i64);
        let cpu = Device::Cpu;
        let truth = Tensor::zeros([width], (Kind::Double, cpu));
        let _ = truth.narrow(0, 0, 1).fill_(0.5);
        let mut inner = Moments::new(width, horizons, cpu);
        let mut holdout = Moments::new(width, horizons, cpu);
        let mut stream = 0u64;
        // 65,536 fit rows and 32,768 penalty rows. Sized, not guessed: the out-of-sample
        // correlation of a fitted 64-dimensional probe on `m` rows has sampling scale `1/√m`,
        // so at 8,192 penalty rows the planted signal's own 0.05 correlation and pure noise are
        // two standard errors apart and every assertion below would be a coin flip.
        for (block, batches) in [(&mut inner, 16), (&mut holdout, 8)] {
            for _ in 0..batches {
                stream += 1;
                let draw = |offset: u64, count: usize| {
                    Tensor::from_slice(&normals(count, stream * 977 + offset))
                };
                let latent = draw(1, (rows * width) as usize).reshape([rows, width]);
                // Horizon 0 is a planted linear signal at a realistic 5% correlation; horizon 1
                // is pure noise with no dependence on the latent at all.
                let signal = latent.matmul(&truth.reshape([-1, 1])).reshape([-1]);
                let noisy = &signal * 0.1 + draw(2, rows as usize);
                let target = Tensor::stack(&[noisy, draw(3, rows as usize)], 1);
                block.push(
                    &latent,
                    &target,
                    &Tensor::ones([rows, horizons], (Kind::Double, cpu)),
                );
            }
        }
        let fits = fit_all(
            &inner.to_host().unwrap(),
            &holdout.to_host().unwrap(),
            &[1, 8],
        )
        .unwrap();
        let ridge_signal = fits
            .iter()
            .find(|f| f.horizon == 1 && f.class == ProbeClass::Ridge)
            .unwrap();
        let ridge_noise = fits
            .iter()
            .find(|f| f.horizon == 8 && f.class == ProbeClass::Ridge)
            .unwrap();
        // The planted coefficient is 0.1 * 0.5 = 0.05 on direction 0 and 0 on the other 63.
        //
        // The probe must recover it SHRUNK, not exactly. At a planted coefficient variance of
        // 2.5e-3 against unit noise over 64 directions and 32,768 fit rows, the squared-error
        // optimal ridge is a real penalty: `λ* ≈ p·σ²/(n·Σβ²) ≈ 0.78` in mean-eigenvalue units,
        // so the MSE-optimal coefficient is about `1/(1 + λ*) ≈ 0.56` of the truth. Asserting
        // the unshrunk value would be asserting that the estimator ignores its own noise. What
        // must hold is the sign, the order of magnitude, and - the part a broken estimator
        // would fail - that the mass sits on the planted direction, not spread over 63 empty
        // ones.
        let recovered = ridge_signal.weight.double_value(&[0]);
        assert!(
            (0.3 * 0.05..=0.05).contains(&recovered),
            "recovered {recovered}, outside the shrinkage band"
        );
        // Against the RMS of the 63 empty directions, not their max. Each empty coefficient is
        // a shrunk OLS estimate with standard deviation `shrink·σ/√n ≈ 0.0024` here, so the
        // LARGEST of 63 such draws sits near 2.8 of those by extreme-value arithmetic alone and
        // a bar set against it would be measuring the tail of a null distribution rather than
        // the estimator. The planted direction must stand well clear of the null SCALE.
        let null_rms = ((1..width)
            .map(|i| ridge_signal.weight.double_value(&[i]).powi(2))
            .sum::<f64>()
            / (width - 1) as f64)
            .sqrt();
        assert!(
            recovered > 4. * null_rms,
            "the planted direction carries {recovered} against a null coefficient scale of \
             {null_rms}; the probe is spreading mass over noise"
        );
        // Out of sample the probe keeps most of the planted 0.05 correlation: the 64-dimensional
        // fit loses roughly `p/n` of the R² and the shrinkage recovers part of that back.
        assert!(
            ridge_signal.holdout_correlation > 0.025,
            "the planted horizon transferred at only {}",
            ridge_signal.holdout_correlation
        );
        // On the noise horizon there is nothing to find, so the held-back block's own squared
        // error selects a penalty whose out-of-sample correlation is indistinguishable from
        // zero - within 2 of the `1/√32768 ≈ 0.0055` sampling scale. This is the direction that
        // matters: a probe that could manufacture correlation from 64 free directions would make
        // World A unfalsifiable.
        assert!(
            ridge_noise.holdout_correlation.abs() < 0.012,
            "a probe on noise claimed correlation {}",
            ridge_noise.holdout_correlation
        );
        assert!(
            ridge_signal.holdout_correlation > 3. * ridge_noise.holdout_correlation.abs(),
            "signal {} against noise {}",
            ridge_signal.holdout_correlation,
            ridge_noise.holdout_correlation
        );
        // The rank restriction is real: a rank-8 probe fits 9 parameters regardless of how wide
        // the latent is, which is the property that makes a win by it hard to dismiss.
        let reduced = fits
            .iter()
            .find(|f| f.horizon == 1 && f.class == ProbeClass::Principal(8))
            .unwrap();
        assert_eq!(reduced.parameters, 9);
        assert_eq!(ridge_signal.parameters, width as usize + 1);
        // The rank restriction is what makes the reduced probe better conditioned, and the
        // penalty-free ratio is the λ-independent way to say so: the retained eigenvalues are
        // the top 8 of the same spectrum, so their spread cannot exceed the full one.
        assert!(reduced.raw_condition <= ridge_signal.raw_condition);
        assert!(ridge_signal.raw_condition.is_finite() && ridge_signal.raw_condition > 1.);
    }

    /// The paired reduction. Two forecasters over one draw: a head that is pure noise and a
    /// probe that is the target itself. The gap must equal the difference of the two ICs
    /// exactly, because both are averaged over the SAME eligible timestamps.
    #[test]
    fn the_paired_gap_is_measured_on_one_common_set_of_timestamps() {
        // `manual_seed` mutates the ONE process-global generator, so an unguarded call here
        // rewinds the stream a concurrently-running seeded test is reading - measured as
        // `model::disabling_the_x0_injection_removes_its_parameters_and_its_kernel` failing
        // one run in three with `left 3.96875, right 0.0`. See `torch::test_rng`.
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(3);
        let (groups, per_group, horizons) = (40i64, 32i64, 2i64);
        let rows = groups * per_group;
        let cpu = Device::Cpu;
        let mut scorer = PairedScorer::new(horizons, 2, groups, cpu);
        let group = Tensor::arange(rows, (Kind::Int64, cpu)).remainder(groups);
        let target = Tensor::randn([rows, horizons], (Kind::Double, cpu));
        let head = Tensor::randn([rows, horizons], (Kind::Double, cpu));
        // Forecaster 0 is the head, forecaster 1 is the target scaled by 3 - a perfect rank
        // forecast with a badly wrong amplitude, which is exactly the pair of properties the
        // oracle-rescaled ratio has to separate.
        let forecast = Tensor::cat(&[&head, &(&target * 3.)], 1);
        scorer.push(
            &group,
            &forecast,
            &target,
            &Tensor::ones([rows, horizons], (Kind::Double, cpu)),
        );
        let report = scorer
            .finish(&["head".to_owned(), "oracle".to_owned()], &[1, 64])
            .unwrap();
        for index in 0..2 {
            let head_score = &report.forecasters[0].per_horizon[index];
            let oracle = &report.forecasters[1].per_horizon[index];
            assert!(
                (oracle.ic - 1.).abs() < 1e-9,
                "a perfect forecast scored {}",
                oracle.ic
            );
            assert!((oracle.gap - (oracle.ic - head_score.ic)).abs() < 1e-9);
            assert_eq!(head_score.gap, 0.);
            assert_eq!(oracle.paired_cross_sections, groups as f64);
            // Amplitude: 3x over-amplified is MSE-worse than persistence while carrying all of
            // the information, and the oracle rescale is what says so.
            assert!(oracle.mse_ratio > 3., "ratio {}", oracle.mse_ratio);
            assert!((oracle.optimal_gain - 1. / 3.).abs() < 1e-6);
            // The oracle series corrects AMPLITUDE and nothing else - it is the project's
            // `best_scale_mse_ratio`, `1 - offset - demeaned`, so the tilt term survives. For a
            // forecast that is exactly `3y` that residual is `4·mean(y)²/mean(y²)` in closed
            // form, and pinning the identity rather than a loose bound is what would catch the
            // oracle silently becoming a full affine refit, which would flatter every probe.
            let column = target.select(1, index as i64);
            let mean = column.mean(Kind::Double).double_value(&[]);
            let persistence = column.square().mean(Kind::Double).double_value(&[]);
            assert!(
                (oracle.oracle_mse_ratio - 4. * mean * mean / persistence).abs() < 1e-9,
                "oracle {} against the closed-form tilt {}",
                oracle.oracle_mse_ratio,
                4. * mean * mean / persistence
            );
        }
    }

    /// Timestamps too thin to carry a cross-section are dropped, and a forecaster with no
    /// usable timestamp reports NaN rather than 0 - an unmeasured IC is not a measured zero.
    #[test]
    fn a_draw_with_no_usable_cross_section_reports_gaps_not_zeros() {
        let cpu = Device::Cpu;
        let rows = 16i64;
        let mut scorer = PairedScorer::new(1, 1, rows, cpu);
        scorer.push(
            &Tensor::arange(rows, (Kind::Int64, cpu)),
            &Tensor::ones([rows, 1], (Kind::Double, cpu)),
            &Tensor::ones([rows, 1], (Kind::Double, cpu)),
            &Tensor::ones([rows, 1], (Kind::Double, cpu)),
        );
        let report = scorer.finish(&["head".to_owned()], &[1]).unwrap();
        let score = &report.forecasters[0].per_horizon[0];
        assert!(score.ic.is_nan());
        assert!(score.ic_se.is_nan());
        assert_eq!(score.cross_sections, 0.);
    }

    /// The refusal that stopped job 5458, measured on the REAL corpus rather than on a fixture.
    ///
    /// [`Blocks::spanning`] compares a max over one population's target timestamps against a
    /// min over the other's origin timestamps. Both sides come from [`CorpusTicker::timestamp`]
    /// over one universe, so they are the same wall clock - but they are GLOBAL extrema over
    /// 4,873 tickers whose histories start and end at different dates. This test measures
    /// whether the two reserved partitions are ordered per ticker, globally, or neither, and
    /// prints the dates, because the fix for a per-ticker-ordered but globally interleaved pair
    /// is completely different from the fix for a genuinely leaking one.
    #[test]
    #[ignore = "loads the real 28 GB corpus"]
    fn the_real_reserved_partitions_are_dated_and_their_ordering_is_measured() {
        use super::super::corpus::{Corpus, CorpusContract};
        let published: CorpusContract = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(shared::paths::TRAINING_PATH)
                    .join("runs/timexer-control-4k/timexer-segment-data-contract.json"),
            )
            .expect("the published contract of the run the probe scores"),
        )
        .unwrap();
        let corpus = Corpus::load(
            &crate::data::ingest::bars_dir(),
            &[],
            published.context,
            published.pred_len,
            published.common_context,
            &published.features,
            published.market_min_cross_section,
            published.in_period_sections,
        )
        .unwrap();
        let pred_len = corpus.contract.pred_len;
        let reach = |reference: &WindowRef| {
            corpus
                .ticker(*reference)
                .timestamp(reference.origin + pred_len)
        };
        let start = |reference: &WindowRef| corpus.ticker(*reference).timestamp(reference.origin);
        let global_reach = corpus.calibration_refs.iter().map(reach).max().unwrap();
        let global_first = corpus.validation_refs.iter().map(start).min().unwrap();
        // Per ticker, which is what the corpus's boundary arithmetic actually guarantees.
        let mut per_ticker_reach: std::collections::BTreeMap<usize, i64> = Default::default();
        let mut per_ticker_first: std::collections::BTreeMap<usize, i64> = Default::default();
        for reference in &corpus.calibration_refs {
            let entry = per_ticker_reach.entry(reference.ticker).or_insert(i64::MIN);
            *entry = (*entry).max(reach(reference));
        }
        for reference in &corpus.validation_refs {
            let entry = per_ticker_first.entry(reference.ticker).or_insert(i64::MAX);
            *entry = (*entry).min(start(reference));
        }
        let mut violations = 0usize;
        let mut shared = 0usize;
        for (ticker, calibration_reach) in &per_ticker_reach {
            shared += 1;
            if let Some(validation_first) = per_ticker_first.get(ticker) {
                if calibration_reach >= validation_first {
                    violations += 1;
                }
            }
        }
        println!(
            "GLOBAL: last calibration target {global_reach}, first validation origin \
             {global_first}, gap {} ms ({})",
            global_first - global_reach,
            if global_first > global_reach {
                "ordered"
            } else {
                "INTERLEAVED"
            }
        );
        println!(
            "PER TICKER: {violations} of {shared} tickers have a calibration target at or after \
             their own first validation origin"
        );
        // What a GLOBAL wall-clock cut would cost. `T` is a candidate instant: fit origins whose
        // targets complete before it, scored origins that begin after it. Both counts at once,
        // because a cut that leaves either side thin is not a usable experiment.
        let mut fit_dates: Vec<i64> = corpus.calibration_refs.iter().map(reach).collect();
        let mut scored_dates: Vec<i64> = corpus.validation_refs.iter().map(start).collect();
        fit_dates.sort_unstable();
        scored_dates.sort_unstable();
        println!(
            "fit target reach spans [{}, {}], scored origin starts span [{}, {}]",
            fit_dates[0],
            fit_dates[fit_dates.len() - 1],
            scored_dates[0],
            scored_dates[scored_dates.len() - 1]
        );
        for percentile in [1usize, 5, 10, 25, 50, 75, 90, 99] {
            let cut = scored_dates[scored_dates.len() * percentile / 100];
            let kept_fit = fit_dates.partition_point(|d| *d < cut);
            let kept_scored = scored_dates.len() - scored_dates.partition_point(|d| *d <= cut);
            println!(
                "cut at scored p{percentile} = {cut}: {kept_fit} fit origins ({:.1}%) and \
                 {kept_scored} scored origins ({:.1}%) survive",
                100. * kept_fit as f64 / fit_dates.len() as f64,
                100. * kept_scored as f64 / scored_dates.len() as f64
            );
        }
        // The residual channel a per-ticker guard leaves open: a fit row and a scored row on
        // DIFFERENT tickers sharing one timestamp. No bar is shared, but a market-neutral target
        // is a cross-sectional residual, so same-instant rows are weakly coupled.
        let fit_stamps: HashSet<i64> = corpus.calibration_refs.iter().map(start).collect();
        let contemporaneous = corpus
            .validation_refs
            .iter()
            .filter(|reference| fit_stamps.contains(&start(reference)))
            .count();
        println!(
            "CONTEMPORANEOUS: {contemporaneous} of {} scored origins ({:.2}%) sit on a timestamp \
             that also carries a fit origin, over {} distinct fit timestamps",
            corpus.validation_refs.len(),
            100. * contemporaneous as f64 / corpus.validation_refs.len() as f64,
            fit_stamps.len()
        );
        assert_eq!(
            violations, 0,
            "the corpus's own per-ticker boundary arithmetic is broken, which is a corpus defect \
             and not a probe pairing defect"
        );
    }

    /// Does the TRAINING population overlap the held-out draws on the wall clock?
    ///
    /// The same instrument that caught the probe's own pairing, pointed at the split every
    /// number in this project rests on. `boundaries[0]` and `boundaries[1]` are each ticker's
    /// own valid-bar ordinals, so training and `held-out full` are per-ticker fractions too. If
    /// they interleave the way the reserved partitions do, the model is trained on calendar
    /// dates that are held-out dates for OTHER tickers - and because the target is a
    /// market-neutral cross-sectional residual, two tickers on one date are not independent.
    ///
    /// Same-ticker disjointness is already assured by the corpus, so every hit this counts is
    /// necessarily cross-ticker; the per-ticker ordering is re-asserted here rather than
    /// assumed, so that inference is sound rather than inherited.
    #[test]
    #[ignore = "loads the real 28 GB corpus and scans every training target bar"]
    fn the_training_population_is_dated_against_every_held_out_draw() {
        use super::super::corpus::{Corpus, CorpusContract};
        let published: CorpusContract = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(shared::paths::TRAINING_PATH)
                    .join("runs/timexer-control-4k/timexer-segment-data-contract.json"),
            )
            .expect("the published contract"),
        )
        .unwrap();
        let corpus = Corpus::load(
            &crate::data::ingest::bars_dir(),
            &[],
            published.context,
            published.pred_len,
            published.common_context,
            &published.features,
            published.market_min_cross_section,
            published.in_period_sections,
        )
        .unwrap();
        let pred_len = corpus.contract.pred_len;
        let cross_section =
            super::super::runner::cross_section_origins(&corpus, &corpus.validation_refs).unwrap();
        let draws: [(&str, &[WindowRef]); 3] = [
            ("held-out full", &corpus.validation_refs),
            ("held-out cross-section", &cross_section),
            ("calibration", &corpus.calibration_refs),
        ];
        // Per ticker, the contiguous ordinal range every training target bar falls in, and the
        // wall clock of its last bar. Training origins are non-overlapping `pred_len` blocks, so
        // the union of their targets is one interval per ticker and costs one scan, not one
        // scan per origin.
        let mut span: std::collections::BTreeMap<usize, (usize, usize)> = Default::default();
        for reference in &corpus.train_refs {
            let entry = span.entry(reference.ticker).or_insert((usize::MAX, 0));
            entry.0 = entry.0.min(reference.origin + 1);
            entry.1 = entry.1.max(reference.origin + pred_len);
        }
        let mut train_first = i64::MAX;
        let mut train_last = i64::MIN;
        let mut train_last_of: std::collections::BTreeMap<usize, i64> = Default::default();
        for (ticker, (lo, hi)) in &span {
            let data = &corpus.contract.tickers[*ticker];
            let hi = (*hi).min(data.valid_bars.saturating_sub(1));
            let (first, last) = (
                corpus
                    .ticker(WindowRef {
                        ticker: *ticker,
                        origin: *lo,
                    })
                    .timestamp(*lo),
                corpus
                    .ticker(WindowRef {
                        ticker: *ticker,
                        origin: hi,
                    })
                    .timestamp(hi),
            );
            train_first = train_first.min(first);
            train_last = train_last.max(last);
            train_last_of.insert(*ticker, last);
        }
        // Every distinct origin timestamp any held-out draw uses, so ONE scan over the training
        // target bars answers the exact-match question for all three at once.
        let mut candidates: HashSet<i64> = HashSet::new();
        for (_, refs) in &draws {
            for reference in refs.iter() {
                candidates.insert(corpus.ticker(*reference).timestamp(reference.origin));
            }
        }
        let floor = *candidates.iter().min().unwrap();
        let mut hit: HashSet<i64> = HashSet::new();
        for (ticker, (lo, hi)) in &span {
            let data = &corpus.contract.tickers[*ticker];
            let hi = (*hi).min(data.valid_bars.saturating_sub(1));
            if train_last_of[ticker] < floor {
                continue;
            }
            for ordinal in *lo..=hi {
                let stamp = corpus
                    .ticker(WindowRef {
                        ticker: *ticker,
                        origin: ordinal,
                    })
                    .timestamp(ordinal);
                if stamp >= floor && candidates.contains(&stamp) {
                    hit.insert(stamp);
                }
            }
        }
        println!(
            "TRAINING target bars span [{train_first}, {train_last}] over {} tickers; {} of {} \
             distinct held-out origin timestamps are EXACTLY a training target bar somewhere in \
             the universe",
            span.len(),
            hit.len(),
            candidates.len()
        );
        for (name, refs) in &draws {
            let mut before = 0usize;
            let mut exact = 0usize;
            let mut same_ticker = 0usize;
            let (mut first, mut last) = (i64::MAX, i64::MIN);
            for reference in refs.iter() {
                let stamp = corpus.ticker(*reference).timestamp(reference.origin);
                first = first.min(stamp);
                last = last.max(stamp);
                if stamp <= train_last {
                    before += 1;
                }
                if hit.contains(&stamp) {
                    exact += 1;
                    if train_last_of
                        .get(&reference.ticker)
                        .is_some_and(|end| *end >= stamp)
                    {
                        same_ticker += 1;
                    }
                }
            }
            let total = refs.len().max(1);
            println!(
                "{name}: {} origins over [{first}, {last}]; {before} ({:.2}%) at or before the \
                 last training target bar; {exact} ({:.2}%) sit EXACTLY on a training target \
                 bar, of which {same_ticker} on their own ticker and {} cross-ticker; \
                 training-to-draw overlap {} days",
                refs.len(),
                100. * before as f64 / total as f64,
                100. * exact as f64 / total as f64,
                exact - same_ticker,
                (train_last - first).max(0) / 86_400_000
            );
            assert_eq!(
                same_ticker, 0,
                "{name} has an origin on a bar its OWN ticker trained on, which is same-ticker \
                 leakage and a corpus defect rather than a cross-ticker exposure"
            );
        }
    }
}
