//! Post-hoc amplitude calibration of the conditional MEAN, fitted out of sample INSIDE the run.
//!
//! # What this fixes
//!
//! The measured long-horizon failure is not absent signal, it is over-amplitude. On
//! `timexer_segment_horizon_steps_gain` at step 2000 the close channel's MSE-optimal gain
//! sweeps `3.662` at h = 1 to `0.195` at h = 192 - a factor of 18.8 across one axis - while the
//! within-timestamp IC stays flat at 0.13-0.16. At h = 128 and h = 192 that costs the whole
//! result: the market-neutral close MSE ratio reads 1.0391 and 1.0386, WORSE than persistence,
//! against 0.9968 and 0.9975 at those horizons' own best scale. Amplitude destroys MSE and
//! leaves rank order, and therefore cross-sectional tradability, untouched.
//!
//! A positive per-horizon gain is exactly the transform that repairs the first and cannot
//! touch the second: within one timestamp and one horizon it multiplies every ticker's forecast
//! by the same positive number, so every rank, every sign, every decile membership and
//! therefore every rank statistic is invariant, by construction rather than by tolerance. That
//! invariance is the strongest available self-check on this whole mechanism and
//! [`super::runner`] asserts it on the real reduction rather than trusting it.
//!
//! # The two coordinates, and why not four channels
//!
//! [`super::model::decode_joint`] does not emit four independent forecasts. It emits ONE close
//! anchor and three monotone offsets built from the range and position coordinates:
//!
//! ```text
//! close = z·√h,  low = close - a/σ,  high = close + (r - a)/σ,  open = close + (o - a)/σ
//! ```
//!
//! with `0 ≤ a ≤ r` and `0 ≤ o ≤ r`, which is what makes
//! `high ≥ max(open, close) ≥ min(open, close) ≥ low` survive rounding. So a decoded candle has
//! exactly TWO amplitude degrees of freedom, and this module fits exactly two curves:
//!
//! ```text
//! f_c(h) = g_anchor(h)·close(h) + g_offset(h)·offset_c(h)
//! ```
//!
//! Both gains are strictly positive, so every offset keeps its sign and its ordering relative
//! to the anchor: the calibrated candle is as valid as the emitted one, for the same reason.
//! Scaling the four DECODED channels independently - four curves - would rescale those offsets
//! by four different numbers and invert them. That is not a tolerance question: at h = 192 the
//! anchor is a ~14σ cumulative move while the intrabar offsets are ~1σ, so a 7% disagreement
//! between two channel gains already crosses the ordering, and four separately smoothed curves
//! disagree by far more than that.
//!
//! What the pair fixes that the old close-only curve did not: the anchor gain is now fitted to
//! minimize the FOUR-CHANNEL squared error rather than the close channel's alone, and the
//! offset amplitude - the intrabar spread, previously frozen at 1 whatever the data said - is
//! calibrated. The old curve moved all four channels (they all contain the anchor) but aimed
//! only at the close residual, and it left the spread uncorrected.
//!
//! # The estimator
//!
//! Per horizon, `(g_anchor, g_offset)` is the exact 2x2 least-squares solution pooled over the
//! four channels and every origin in the fit block - no intercept, because a pure gain is what
//! is applied - and the fit weight is the inverse delta-method variance of `ln g`, taken from
//! that same solve's Gram matrix with the channel multiplicity charged as a design effect (the
//! four candle channels of one bar share one anchor error, so one bar is one independent
//! residual, not four).
//!
//! **Pure gain, no offset.** Refusing an intercept is not an assumption, it is a measured
//! refusal enforced by [`MeanCalibration::fit`], PER HORIZON and on the CLOSE channel:
//! `ȳ_close²/E[y_close²]` is the ENTIRE share of persistence MSE any constant forecast could
//! earn at that horizon, and the fit declines that horizon unless it is small against the
//! amplitude cost a gain competes for THERE. Two things it deliberately is not. It is not
//! pooled over the four DECODED channels: a bar's high is never below its close, so the high
//! and low target means are structurally nonzero on any candle data whatsoever, and their
//! pooled share measures the INTRABAR OFFSET - it goes as `1/h`, the signature of a
//! non-accumulating per-bar constant, where market drift would have to GROW as `h·µ²` - which
//! is the quantity the offset COORDINATE exists to absorb. It is also not a max-over-horizon
//! comparison: the intercept share peaks at h = 1 and the amplitude cost past h = 170, so
//! worst-against-worst lets one short horizon blank the other 191. A refused horizon is
//! dropped from both curves, carried by its neighbours through the roughness prior, and gated
//! out of SIZING under its own named reason.
//!
//! Only the anchor carries this ceiling, because only the anchor's column can be beaten by a
//! constant: within a timestamp it is market-neutral, so no gain on it reaches any constant at
//! all. The offset column's mean IS the structural spread - `0 ≤ a ≤ r` fixes the sign of every
//! intrabar offset - so a strictly positive `g_offset` already spans the constant it would be
//! tested against, and testing it there would refuse the estimator on the very structure its
//! second coordinate models. An intercept is also the least stationary thing in the data - a
//! bet on the next block's mean drift - and it is the one part of an affine map that a
//! within-timestamp IC cannot see; a mean intrabar range is neither.
//!
//! **The shape prior.** Adjacent horizons' true gains cannot jump: `g` is a smooth functional
//! of the joint law of `(f_h, y_h)` and neighbouring cumulative returns overlap in 191 of 192
//! bars. So each curve is a roughness-penalized weighted least squares on `ln g` against
//! `ln h` - positive by construction, curvature-penalized rather than shape-imposed, and
//! reducing to a straight log-log line as the penalty grows. It is deliberately NOT a two-
//! parameter power law: the measured profile is well APPROXIMATED by one (`192^-0.57` reproduces
//! the 18.8x sweep to two digits) but the approximation is not exact, and weighted by their own
//! measured precision the control checkpoint's residuals leave `χ²/dof = 23.6` against a
//! log-log line and `13.6` against a log-quadratic, on a curve whose own second differences sit
//! well below the noise scale. The fine structure is real and a global shape would erase it.
//!
//! **The bound, and why it is not 1.** Applying gain `g` to a forecast whose true amplitude is
//! `β` leaves a cross term `-(β - g)²·Var(f)/P`, so MSE supplies no ceiling anywhere and a
//! measured gain above 1 is under-amplification that a shrinkage-only clamp simply refuses to
//! collect. What bounds the amplifying direction is ESTIMATION ERROR, because a gain multiplies
//! deployed notional in every sizing rule affine in the forecast:
//!
//! ```text
//! ceiling_h = max(1, exp(ln ĝ_h - z·SE(ln ĝ_h))),  SE(ln ĝ_h) = 1/√weight_h,  z = 3
//! g_h       = min(fitted_h, ceiling_h)
//! ```
//!
//! A horizon whose gain is within `z` standard errors of 1 is never amplified; a horizon with
//! no measurable amplitude gets `ceiling_h = 1`, so the roughness prior may SHRINK it from its
//! neighbours but may not amplify it on their evidence.
//!
//! # Where it is fitted, and where it is applied
//!
//! Fitted on the corpus's reserved `[70%, 80%)` calibration partition, at every evaluation of
//! the run that produced the checkpoint, and carried in that checkpoint's manifest
//! ([`FrozenGain`]). Applied by [`super::model::CausalPatchModel::set_mean_gain`] at the point
//! a checkpoint is loaded, so the gained mean is the model's actual output and no consumer -
//! reports, `evaluate`, trading, portfolio sizing - can disagree about what was applied. The
//! training loss never reads the gain buffers, so a run's gradient is untouched by its own
//! calibration.
//!
//! # Why a refusal ABORTS instead of producing the identity
//!
//! Every failure path here returns `Err`. A gain of 1 is a legitimate MEASURED result - the
//! block says this coordinate is already at the right amplitude - and it is written to the
//! chart as 1.0 for exactly that reason. An unfittable amplitude rendered as a gain of 1 is
//! the same bits, so the two would be indistinguishable on every downstream artifact: the
//! `calibrated` MSE ratio is produced by applying `g` in closed form
//! ([`super::reports`]), so `g = 1` makes it bit-identical to the `uncalibrated` series and a
//! run reports a calibration it never performed. That is what job 6004 shipped. A refusal is
//! therefore fatal at the evaluation that measured it, with the numbers that refused it in the
//! message, and the arm is rerun under an estimator the block admits rather than read as if it
//! had been calibrated.
//!
//! # Cost
//!
//! Zero on the training path: nothing here runs during a step and the loss buffers are not
//! touched. The moments are eight `[pred_len, CHANNELS]` f64 column reductions of tensors the
//! evaluation already has resident. The host fit is two 192x192 Cholesky sweeps per penalty
//! grid point, about 0.5 GFLOP per evaluation. The checkpoint grows by the two frozen curves,
//! `2·pred_len` f64 (≈ 8 KB of JSON at pred_len = 192).

use super::model::CHANNELS;
use crate::torch::dataset::iso_ms;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Decoded channel index of the close coordinate: `decode_joint` emits `(open, high, low,
/// close)`, so the anchor is last and its offset is identically zero.
pub const CLOSE_CHANNEL: usize = CHANNELS as usize - 1;

/// The pre-registered refusal threshold on the intercept: at any horizon, the best constant
/// forecast may not be able to earn more than this share of what the amplitude error costs
/// THERE, or a pure gain is the wrong parameterization at that horizon and the fit applies none
/// to it. The value is registered and unchanged; what was wrong was the quantity it read - four
/// pooled channels, not the close channel it was pinned against - and the axis it read it on.
/// Measured close-channel margin on the control checkpoint is `4.63e-5` against a `2.995e-2`
/// amplitude cost, i.e. 65x below this.
pub const INTERCEPT_CEILING_SHARE: f64 = 0.1;
/// Standard errors of its OWN estimate that an amplification must clear before any of it is
/// applied. Shrinkage is unbounded below by the fit; amplification is bounded above by
/// `exp(ln ĝ - AMPLIFICATION_SIGMAS·SE(ln ĝ))`, floored at 1.
const AMPLIFICATION_SIGMAS: f64 = 3.0;
/// Penalty strengths tried, geometric, in units of the mean fit weight so the grid is
/// invariant to the population size and to the correlation level.
const PENALTY_DECADES: (f64, f64) = (-4.0, 6.0);
const PENALTY_GRID: usize = 41;

/// Everything that has to agree for a stored artifact to be applicable to a running model.
///
/// Not the mean gain's guard any more - that rides inside the authenticated checkpoint manifest
/// and cannot be paired with anything else - but the portfolio tape cache's binding, which is a
/// function of exactly these identities. Each field is one way a cached artifact can silently
/// describe a different forecaster: a different checkpoint of the same architecture has
/// different head weights; a different architecture stamp has a different mean
/// parameterization; a different corpus contract has different targets, splits and market
/// demeaning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pairing {
    /// The checkpoint's own `format` stamp: architecture, x0 mode and mean parameterization.
    pub checkpoint_format: String,
    /// The training objective the weights were fitted under.
    pub objective: String,
    /// SHA-256 of `model.safetensors` - the identity of the exact weights.
    pub weights_sha256: String,
    /// SHA-256 of the checkpoint's own authenticated manifest, so a manifest edited after the
    /// fact (a re-stamped selection, a corrected step) is also caught.
    pub manifest_sha256: String,
    /// Optimizer step the weights are from, carried for legibility.
    pub step: usize,
    /// Horizons the forecaster emits.
    pub pred_len: usize,
    /// The corpus contract's schema string, so a data-contract change is named as one.
    pub corpus_schema: String,
    /// SHA-256 over the whole serialized corpus contract: universe, fingerprints, boundaries,
    /// market construction, feature set.
    pub corpus_sha256: String,
}

/// The two chronological blocks, dated, plus the fact that makes them disjoint.
///
/// `calibration_last_target_ms` is the load-bearing field: cumulative targets reach `pred_len`
/// valid bars past their origin, so blocks separated by ORIGIN time alone would share bars -
/// the last calibration origin's h = 192 target is realized after its origin. This is the
/// latest bar any calibration origin's target reads, and every evaluation origin is strictly
/// after it, so no observation is on both sides of the split.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blocks {
    pub calibration_first_origin_ms: i64,
    pub calibration_last_origin_ms: i64,
    /// Completion time of the last bar any calibration-block target reads.
    pub calibration_last_target_ms: i64,
    pub calibration_origins: usize,
    pub evaluation_first_origin_ms: i64,
    pub evaluation_last_origin_ms: i64,
    pub evaluation_origins: usize,
    /// The tightest separation the constructor PROVED, and the whole of the disjointness
    /// claim: [`Self::spanning`] proves one pooled block pair and reports that pair's gap;
    /// [`Self::per_ticker`] proves every ticker's own pair and reports the smallest of them.
    /// Strictly positive either way - a gap of zero would mean a fit target and a scored
    /// origin read the same bar. Stated in milliseconds rather than as an origin count because
    /// the corpus's reserved purge band holds no origins at all - counting them would report 0
    /// for a separation of many days.
    pub purge_gap_ms: i64,
}

/// `(ticker index, origin timestamp, timestamp of the last bar this origin's `pred_len`
/// cumulative targets read)`.
///
/// The ticker index is load-bearing for [`Blocks::per_ticker`]: the corpus cuts its held-out
/// blocks on each ticker's OWN valid-bar ordinals, so disjointness is a per-ticker claim and
/// the index is what makes it checkable.
pub type DatedOrigin = (usize, i64, i64);

impl Blocks {
    /// Date two POOLED populations and prove they are disjoint on the global wall clock.
    ///
    /// Each entry is `(origin timestamp, timestamp of the last bar this origin's `pred_len`
    /// cumulative targets read)`. The comparison is a max over one population's target
    /// timestamps against a min over the other's origin timestamps, so it is a claim about the
    /// whole universe at once, and it holds only where the caller has CUT the scored block to
    /// earn it - [`super::probe::Partitions::split`] does exactly that and pays a measured 1%
    /// of its scored population for it.
    ///
    /// The corpus's own reserved bands are not such a pair. They are cut per ticker and their
    /// global extrema interleave; proving them is [`Self::per_ticker`]'s job, and pointing
    /// this function at them refuses every corpus this universe can produce.
    pub fn spanning(calibration: &[(i64, i64)], evaluation: &[(i64, i64)]) -> Result<Self> {
        ensure!(
            !calibration.is_empty(),
            "the calibration partition produced no origins; there is nothing to fit a gain \
             curve on"
        );
        ensure!(
            !evaluation.is_empty(),
            "the evaluation partition produced no origins; there is nothing to score"
        );
        let span = |rows: &[(i64, i64)]| {
            rows.iter().fold(
                (i64::MAX, i64::MIN, i64::MIN),
                |(first, last, reach), (origin, target)| {
                    (first.min(*origin), last.max(*origin), reach.max(*target))
                },
            )
        };
        let (calibration_first, calibration_last, calibration_reach) = span(calibration);
        let (evaluation_first, evaluation_last, _) = span(evaluation);
        let blocks = Self {
            calibration_first_origin_ms: calibration_first,
            calibration_last_origin_ms: calibration_last,
            calibration_last_target_ms: calibration_reach,
            calibration_origins: calibration.len(),
            evaluation_first_origin_ms: evaluation_first,
            evaluation_last_origin_ms: evaluation_last,
            evaluation_origins: evaluation.len(),
            purge_gap_ms: evaluation_first - calibration_reach,
        };
        ensure!(
            blocks.purge_gap_ms > 0,
            "the evaluation block's first origin is at {evaluation_first} but the calibration \
             block's targets reach {calibration_reach}: the two populations share bars and a \
             gain fitted on the first would be scored on data it has already seen",
        );
        Ok(blocks)
    }

    /// PROVE the disjointness the pooled corpus actually guarantees: per ticker, exactly, for
    /// every ticker, never on global extrema.
    ///
    /// `corpus.rs` cuts both reserved bands at `boundaries[k]`, which is each ticker's OWN
    /// valid-bar ordinal at the shared boundary timestamp (`index_at_or_after`). Within one
    /// ticker the ordering is exact: the calibration band stops at `retained_partition_end`,
    /// `purge >= 100` bars before `boundaries[1]`, while the validation band's first origin is
    /// `boundaries[1] - 1`. ACROSS tickers those ordinals are different wall clocks - a ticker
    /// that stops trading, or thins out, long before the shared boundary has its own band edge
    /// years earlier than a continuously traded one - so the two populations' global timestamp
    /// extrema interleave and comparing them proves nothing and refuses everything. Measured on
    /// the 4,873-ticker corpus by
    /// `probe::tests::the_real_reserved_partitions_are_dated_and_their_ordering_is_measured`:
    /// fit targets reach 2024-08-15 while the earliest scored origin is 2019-02-14, and 0 of
    /// 4,498 tickers violate their own ordering.
    ///
    /// RESIDUAL ASSUMPTION, stated because it is real and deliberately not fixed here: with
    /// per-ticker bands the two populations DO overlap in absolute wall-clock time across
    /// DIFFERENT tickers. No ticker shares a bar, an origin or an instant with itself across
    /// the split - and on the real corpus not one scored origin shares even a TIMESTAMP with a
    /// fit origin - but the fit period and the scoring period are the same calendar span, so
    /// the gain sees market-wide contemporaneous information. For `pred_len` amplitude scalars
    /// that is a weak leak; it is not a nil one, and closing it would cost a global wall-clock
    /// cut of the scored block, which is a different experiment.
    pub fn per_ticker(
        calibration: &[DatedOrigin],
        evaluation: &[DatedOrigin],
        horizons: usize,
        name: impl Fn(usize) -> String,
    ) -> Result<Self> {
        ensure!(
            !calibration.is_empty(),
            "the calibration partition produced no origins; there is nothing to fit a gain \
             curve on"
        );
        ensure!(
            !evaluation.is_empty(),
            "the evaluation partition produced no origins; there is nothing to score"
        );
        // Ordered, not hashed: the ticker a refusal names has to be the same ticker on every
        // run, and ties in the tightest separation are common on a purged grid.
        let mut reach: BTreeMap<usize, i64> = BTreeMap::new();
        let mut calibration_span = (i64::MAX, i64::MIN, i64::MIN);
        for &(ticker, origin, target) in calibration {
            calibration_span.0 = calibration_span.0.min(origin);
            calibration_span.1 = calibration_span.1.max(origin);
            calibration_span.2 = calibration_span.2.max(target);
            let entry = reach.entry(ticker).or_insert(i64::MIN);
            *entry = (*entry).max(target);
        }
        let mut opens: BTreeMap<usize, i64> = BTreeMap::new();
        let mut evaluation_span = (i64::MAX, i64::MIN);
        for &(ticker, origin, _) in evaluation {
            evaluation_span.0 = evaluation_span.0.min(origin);
            evaluation_span.1 = evaluation_span.1.max(origin);
            let entry = opens.entry(ticker).or_insert(i64::MAX);
            *entry = (*entry).min(origin);
        }
        // Every ticker that carries origins on BOTH sides, which is the complete set of ways
        // this split can leak: a ticker on one side alone shares nothing with itself.
        let mut shared = 0usize;
        let mut violations = 0usize;
        let mut tightest: Option<(usize, i64, i64)> = None;
        for (&ticker, &first_origin) in &opens {
            let Some(&last_target) = reach.get(&ticker) else {
                continue;
            };
            shared += 1;
            if last_target >= first_origin {
                violations += 1;
            }
            if tightest
                .is_none_or(|(_, target, origin)| first_origin - last_target < origin - target)
            {
                tightest = Some((ticker, last_target, first_origin));
            }
        }
        let Some((offender, last_target, first_origin)) = tightest else {
            bail!(
                "no ticker carries origins in both the calibration block ({} origins over {} \
                 tickers) and the evaluation block ({} origins over {} tickers), so the fit is \
                 scored on instruments it never saw",
                calibration.len(),
                reach.len(),
                evaluation.len(),
                opens.len()
            );
        };
        ensure!(
            violations == 0,
            "{violations} of {shared} tickers fit and score a gain on their own data: {} is the \
             worst - its calibration block's last {horizons}-step target completes at {} \
             ({last_target} ms epoch) but its own first evaluation origin is at {} \
             ({first_origin} ms epoch), {} ms EARLIER, so every one of those {horizons} \
             horizons is fitted on bars that ticker is then scored on",
            name(offender),
            iso_ms(last_target),
            iso_ms(first_origin),
            last_target - first_origin
        );
        Ok(Self {
            calibration_first_origin_ms: calibration_span.0,
            calibration_last_origin_ms: calibration_span.1,
            calibration_last_target_ms: calibration_span.2,
            calibration_origins: calibration.len(),
            evaluation_first_origin_ms: evaluation_span.0,
            evaluation_last_origin_ms: evaluation_span.1,
            evaluation_origins: evaluation.len(),
            purge_gap_ms: first_origin - last_target,
        })
    }
}

/// Which of the decoded candle's two amplitude degrees of freedom a curve calibrates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Coordinate {
    /// The `√h`-scaled close mean every channel is built on.
    CloseAnchor,
    /// The intrabar offsets `open - close`, `high - close`, `low - close`, jointly.
    IntrabarOffset,
}

impl Coordinate {
    pub fn label(self) -> &'static str {
        match self {
            Self::CloseAnchor => "close anchor",
            Self::IntrabarOffset => "intrabar offset",
        }
    }
}

/// Per-(horizon, channel) second moments of the emitted mean against its target, masked, in
/// market-neutral σ units - everything the fit, the un-gained ratio and the gained ratio are
/// functions of, and the only thing a scoring pass has to carry to produce any of them.
///
/// Layout is row-major `[horizon][channel]`, index 0 being one bar ahead. The forecast is split
/// into the anchor `C` (the close channel) and the offset `O_c = f_c - C`, which is exactly the
/// two-coordinate decomposition [`decode_joint`](super::model::decode_joint) builds, so
/// `O_close ≡ 0` and the close rows carry the anchor alone.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Moments {
    pub pred_len: usize,
    pub channels: usize,
    /// Valid target bars per horizon. Channel-independent: the mask is per (origin, bar).
    pub bars: Vec<f64>,
    /// `Σ y`
    pub target: Vec<f64>,
    /// `Σ y²` - the persistence MSE numerator every ratio here is a share of.
    pub target_square: Vec<f64>,
    /// `Σ C²`
    pub anchor_square: Vec<f64>,
    /// `Σ C·O`
    pub anchor_offset: Vec<f64>,
    /// `Σ O²`
    pub offset_square: Vec<f64>,
    /// `Σ C·y`
    pub anchor_target: Vec<f64>,
    /// `Σ O·y`
    pub offset_target: Vec<f64>,
}

/// The five channel-pooled sums one horizon's 2x2 solve needs.
struct Pooled {
    anchor_square: f64,
    anchor_offset: f64,
    offset_square: f64,
    anchor_target: f64,
    offset_target: f64,
    persistence: f64,
}

impl Moments {
    pub fn validate(&self) -> Result<()> {
        let elements = self.pred_len * self.channels;
        ensure!(
            self.pred_len > 0 && self.channels > 0,
            "amplitude moments cover {} horizons and {} channels",
            self.pred_len,
            self.channels
        );
        ensure!(
            self.bars.len() == self.pred_len
                && [
                    self.target.len(),
                    self.target_square.len(),
                    self.anchor_square.len(),
                    self.anchor_offset.len(),
                    self.offset_square.len(),
                    self.anchor_target.len(),
                    self.offset_target.len(),
                ]
                .iter()
                .all(|length| *length == elements),
            "amplitude moments must be {elements} (horizon, channel) cells and {} bar counts",
            self.pred_len
        );
        ensure!(
            self.bars
                .iter()
                .chain(&self.target)
                .chain(&self.target_square)
                .chain(&self.anchor_square)
                .chain(&self.anchor_offset)
                .chain(&self.offset_square)
                .chain(&self.anchor_target)
                .chain(&self.offset_target)
                .all(|value| value.is_finite()),
            "nonfinite amplitude moment"
        );
        ensure!(
            self.channels > CLOSE_CHANNEL,
            "the close channel is index {CLOSE_CHANNEL} of a decoded candle and the intercept \
             refusal is measured on it alone, so {} channels of moments cannot carry it",
            self.channels
        );
        Ok(())
    }

    fn cell(&self, horizon: usize, channel: usize) -> usize {
        horizon * self.channels + channel
    }

    /// `Σ f²` for decoded channel `c`, from `f = C + O_c`.
    pub fn forecast_square(&self, horizon: usize, channel: usize) -> f64 {
        let cell = self.cell(horizon, channel);
        self.anchor_square[cell] + 2. * self.anchor_offset[cell] + self.offset_square[cell]
    }

    /// `Σ f·y` for decoded channel `c`.
    pub fn forecast_target(&self, horizon: usize, channel: usize) -> f64 {
        let cell = self.cell(horizon, channel);
        self.anchor_target[cell] + self.offset_target[cell]
    }

    /// The anchor energy `Σ C²`, the anchor covariance `Σ C·y` and the persistence `Σ y²` of
    /// one cell: the denominator, the numerator and the scale of a measured gain. Together they
    /// are what separates "this forecast is genuinely too small" from "this forecast has no
    /// amplitude left and the quotient means nothing", which no function of the gain alone can.
    pub fn gain_moments(&self, horizon: usize, channel: usize) -> (f64, f64, f64) {
        let cell = self.cell(horizon, channel);
        (
            self.anchor_square[cell],
            self.anchor_target[cell],
            self.target_square[cell],
        )
    }

    /// The MSE-optimal single gain of decoded channel `c`, `Σfy/Σf²`. This is the raw,
    /// un-demeaned quantity - the scale a pure gain would take, matching what is applied - and
    /// it is `NaN` where the channel emits no amplitude at all, because 0 would read as
    /// "infinitely over-amplified", the opposite of what a flat forecast is.
    pub fn channel_gain(&self, horizon: usize, channel: usize) -> f64 {
        let square = self.forecast_square(horizon, channel);
        if square > 0. {
            self.forecast_target(horizon, channel) / square
        } else {
            f64::NAN
        }
    }

    /// [`Self::channel_gain`] over the whole axis, `[channel][horizon]`.
    pub fn channel_gains(&self) -> Vec<Vec<f64>> {
        (0..self.channels)
            .map(|channel| {
                (0..self.pred_len)
                    .map(|horizon| self.channel_gain(horizon, channel))
                    .collect()
            })
            .collect()
    }

    /// One channel's MSE against persistence, as emitted.
    pub fn channel_ratio(&self, horizon: usize, channel: usize) -> f64 {
        let cell = self.cell(horizon, channel);
        let persistence = self.target_square[cell];
        if persistence <= 0. {
            return f64::NAN;
        }
        (self.forecast_square(horizon, channel) - 2. * self.forecast_target(horizon, channel)
            + persistence)
            / persistence
    }

    /// One channel's MSE against persistence with the two gains applied, in closed form.
    ///
    /// `Σ(g_a·C + g_o·O - y)²` expands into the six moments already accumulated, so the gained
    /// ratio is exact arithmetic on the un-gained pass rather than a second scoring pass over
    /// the same origins.
    pub fn channel_gained_ratio(
        &self,
        horizon: usize,
        channel: usize,
        anchor: f64,
        offset: f64,
    ) -> f64 {
        let cell = self.cell(horizon, channel);
        let persistence = self.target_square[cell];
        if persistence <= 0. {
            return f64::NAN;
        }
        (anchor * anchor * self.anchor_square[cell]
            + offset * offset * self.offset_square[cell]
            + 2. * anchor * offset * self.anchor_offset[cell]
            - 2. * anchor * self.anchor_target[cell]
            - 2. * offset * self.offset_target[cell]
            + persistence)
            / persistence
    }

    /// The four-channel MSE ratio - the primary metric - as emitted.
    pub fn pooled_ratio(&self, horizon: usize) -> f64 {
        let pooled = self.pooled(horizon);
        if pooled.persistence <= 0. {
            return f64::NAN;
        }
        (pooled.anchor_square + 2. * pooled.anchor_offset + pooled.offset_square
            - 2. * (pooled.anchor_target + pooled.offset_target)
            + pooled.persistence)
            / pooled.persistence
    }

    /// The four-channel MSE ratio with the two gains applied, same closed form.
    pub fn pooled_gained_ratio(&self, horizon: usize, anchor: f64, offset: f64) -> f64 {
        let pooled = self.pooled(horizon);
        if pooled.persistence <= 0. {
            return f64::NAN;
        }
        (anchor * anchor * pooled.anchor_square
            + offset * offset * pooled.offset_square
            + 2. * anchor * offset * pooled.anchor_offset
            - 2. * anchor * pooled.anchor_target
            - 2. * offset * pooled.offset_target
            + pooled.persistence)
            / pooled.persistence
    }

    fn pooled(&self, horizon: usize) -> Pooled {
        let mut pooled = Pooled {
            anchor_square: 0.,
            anchor_offset: 0.,
            offset_square: 0.,
            anchor_target: 0.,
            offset_target: 0.,
            persistence: 0.,
        };
        for channel in 0..self.channels {
            let cell = self.cell(horizon, channel);
            pooled.anchor_square += self.anchor_square[cell];
            pooled.anchor_offset += self.anchor_offset[cell];
            pooled.offset_square += self.offset_square[cell];
            pooled.anchor_target += self.anchor_target[cell];
            pooled.offset_target += self.offset_target[cell];
            pooled.persistence += self.target_square[cell];
        }
        pooled
    }

    /// `ȳ_close²/E[y_close²]`: the whole share of the persistence MSE that the best constant
    /// forecast could earn at this horizon, on the CLOSE channel alone.
    ///
    /// The close channel and not the four decoded ones, because that is the population the
    /// refusal is a statement about. `O_close ≡ 0`, so the close row carries the ANCHOR alone,
    /// and the anchor is the only coordinate a constant can compete with: it is market-neutral
    /// within a timestamp, so no gain on it reaches a constant. Pooling the four channels
    /// instead measures the intrabar spread - `ȳ_high` and `ȳ_low` are structurally nonzero on
    /// candle data, and the pooled share then goes as `1/h` rather than growing as `h·µ²` the
    /// way a drift intercept must - which is what the offset coordinate absorbs, so charging it
    /// here refuses the estimator on its own second degree of freedom. Measured on the control
    /// block: `4.63e-5` on this channel against `7.535e-2` pooled, a factor of 1600.
    ///
    /// `NaN` where the block measured nothing at this horizon - no valid target bars, or no
    /// persistence to be a share of. NOT 0: zero is the most permissive value this quantity
    /// has, so an unmeasured horizon folded to 0 would pass the refusal gate below on evidence
    /// that does not exist, and would draw on the moment panel as a measured absence of drift.
    fn intercept_ceiling(&self, horizon: usize) -> f64 {
        let bars = self.bars[horizon];
        if bars <= 0. {
            return f64::NAN;
        }
        let cell = self.cell(horizon, CLOSE_CHANNEL);
        let persistence = self.target_square[cell];
        if persistence > 0. {
            (self.target[cell] * self.target[cell] / bars / persistence).clamp(0., 1.)
        } else {
            f64::NAN
        }
    }
}

/// One horizon's two-coordinate least-squares solution.
///
/// Each coordinate is `None` where the block's design carries no energy in it - a decode whose
/// intrabar range has collapsed leaves the offset column identically zero, and a zero-init head
/// leaves the anchor column identically zero - which is a different fact from a measured zero
/// and is why it is an option rather than a `0.`.
struct Solved {
    anchor: Option<f64>,
    offset: Option<f64>,
    /// Inverse delta-method variance of `ln g`, per coordinate; 0 where that coordinate has no
    /// energy or the residual leaves it no precision.
    anchor_weight: f64,
    offset_weight: f64,
    /// What fixing the amplitude at this horizon is worth, as a share of the persistence MSE.
    amplitude_cost: f64,
}

/// The fitted curve for one coordinate, and every quantity a reader needs to judge it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurveFit {
    pub coordinate: Coordinate,
    /// The frozen curve: what a calibrated decode multiplies this coordinate by. Strictly
    /// positive; two-sided, bounded above by [`Self::amplification_ceiling`] rather than by 1.
    pub gain: Vec<f64>,
    /// The raw per-horizon least-squares gain this was smoothed from, `NaN` where the block
    /// could not identify it.
    pub measured_gain: Vec<f64>,
    /// `SE(ln ĝ) = 1/√weight` per horizon, `NaN` where there is no measurable amplitude. This
    /// is what the amplifying direction is charged against.
    pub standard_error: Vec<f64>,
    /// `max(1, exp(ln ĝ - z·SE))`: the largest gain this horizon's own data proves, and the
    /// only thing that bounds the curve from above.
    pub amplification_ceiling: Vec<f64>,
    /// Inverse delta-method variance of `ln ĝ` per horizon; 0 where unidentified.
    pub weight: Vec<f64>,
    /// The selected roughness penalty, in units of the mean fit weight.
    pub penalty: f64,
    /// `tr((W + λR)⁻¹W)`: how many of the `pred_len` coefficients the data actually paid for.
    pub effective_dof: f64,
    /// Horizons whose amplitude the block identified.
    pub identified: usize,
}

/// One horizon whose pure-gain parameterization the calibration block refused, with both
/// competing quantities.
///
/// Named rather than counted: the gate's whole content is a comparison of two measured shares
/// at one horizon, and a reader who cannot see which two numbers refused it cannot tell a real
/// refusal from a mis-keyed reduction - which is exactly what happened when a four-channel
/// intercept was compared against a different horizon's amplitude cost.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterceptRefusal {
    /// 1-based horizon.
    pub horizon: usize,
    /// `ȳ_close²/E[y_close²]` here: what the best constant forecast could earn.
    pub intercept: f64,
    /// What fixing the amplitude here is worth, as a share of the same persistence MSE. The
    /// comparand.
    pub amplitude_cost: f64,
}

/// A fitted amplitude calibration: two curves, their diagnostics, and the blocks they were
/// measured between.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeanCalibration {
    /// The estimator, spelled out, so a curve is attributable to how it was produced.
    pub estimator: String,
    pub blocks: Blocks,
    pub anchor: CurveFit,
    pub offset: CurveFit,
    /// `ȳ_close²/mean(y_close²)` per horizon: the whole gain any constant forecast could earn.
    pub intercept_ceiling: Vec<f64>,
    /// What the amplitude error costs per horizon, as a share of the persistence MSE - the
    /// quantity the intercept ceiling is compared against, at the SAME horizon.
    pub amplitude_cost: Vec<f64>,
    /// The horizons that comparison refused, in order, with the two shares that refused each.
    /// Empty on a block the pure-gain parameterization fits everywhere.
    pub intercept_refused: Vec<InterceptRefusal>,
}

impl MeanCalibration {
    /// Fit both curves on `moments`, which must come from the calibration block and nothing
    /// else. `blocks` is stored, not consulted: disjointness is [`Blocks::spanning`]'s job and
    /// is proven there.
    ///
    /// `Err` on every input this estimator cannot fit, structural or measured: wrong shapes,
    /// nonfinite sums, an axis too short to have curvature, a block that identifies no
    /// amplitude, a refused parameterization. There is no identity fallback - see the module
    /// docs for why a refusal cannot be allowed to look like a fitted unit gain.
    pub fn fit(blocks: Blocks, moments: &Moments) -> Result<Self> {
        moments.validate()?;
        let horizons = moments.pred_len;
        ensure!(
            horizons >= 3,
            "a curvature-penalized horizon fit needs at least three horizons, got {horizons}"
        );
        let solved: Vec<Option<Solved>> = (0..horizons)
            .map(|horizon| solve_horizon(moments, horizon))
            .collect();
        let intercept_ceiling: Vec<f64> = (0..horizons)
            .map(|horizon| moments.intercept_ceiling(horizon))
            .collect();
        let amplitude_cost: Vec<f64> = solved
            .iter()
            .map(|row| {
                row.as_ref()
                    .map_or(f64::NAN, |solved| solved.amplitude_cost)
            })
            .collect();
        let estimator = format!(
            "exact 2x2 least squares in (close anchor, intrabar offset) pooled over {} channels, \
             no intercept, and a horizon whose close-channel constant-forecast share exceeds \
             {INTERCEPT_CEILING_SHARE} of its OWN amplitude cost is refused at that horizon and \
             carried by the prior; ln g smoothed on ln h under a natural-spline roughness \
             penalty selected by GCV over {PENALTY_GRID} geometric strengths, weighted by the \
             solve's own inverse delta-method variance with the channel multiplicity charged as \
             a design effect; two-sided: shrinkage as fitted, amplification bounded by \
             exp(ln ĝ - {AMPLIFICATION_SIGMAS}·SE(ln ĝ)) floored at 1",
            moments.channels
        );
        // Unmeasured horizons are `NaN`, and `f64::max` SKIPS them, so the refusal below would
        // silently read a short axis as a clean one. Named here instead.
        let unmeasured: Vec<usize> = intercept_ceiling
            .iter()
            .enumerate()
            .filter(|(_, ceiling)| !ceiling.is_finite())
            .map(|(horizon, _)| horizon + 1)
            .collect();
        ensure!(
            unmeasured.is_empty(),
            "the calibration block carries no target bars or no close-channel persistence at {} \
             of {horizons} horizons (h={:?}), so the intercept refusal cannot be evaluated there",
            unmeasured.len(),
            &unmeasured[..unmeasured.len().min(8)]
        );
        // Distinct from the intercept refusal below and never folded into it: a block that
        // solved no horizon at all has no amplitude cost for an intercept to be compared
        // against, so a parameterization verdict there would be a verdict on a population that
        // identified nothing to parameterize.
        ensure!(
            amplitude_cost.iter().any(|cost| cost.is_finite()),
            "not one of {horizons} horizons in the calibration block identified a two-\
             coordinate amplitude, so there is nothing to fit a gain curve on. This aborts \
             rather than applying the identity: a gain of 1 here would be indistinguishable \
             from a measured unit amplitude on every chart and in every checkpoint"
        );
        // THE INTERCEPT REFUSAL. Pre-registered and unchanged in its THRESHOLD: a pure gain is
        // the wrong parameterization wherever a constant forecast could earn more than
        // `INTERCEPT_CEILING_SHARE` of what the amplitude error costs. What was wrong was the
        // aggregation, twice. The comparison is PER HORIZON, because that is what the
        // registration states and because the two quantities live at opposite ends of the axis:
        // the intercept share peaks at h = 1 and the amplitude cost past h = 170, so
        // worst-against-worst let one short horizon refuse all 192. And the intercept is the
        // CLOSE channel's, which is the population the registered margin was pinned on - see
        // `Moments::intercept_ceiling` for why the pooled quantity is the intrabar spread.
        //
        // A refused horizon keeps its measurement and loses its WEIGHT: both curves carry it
        // from its neighbours through the roughness prior exactly as they carry an unidentified
        // one, its amplification ceiling is the identity, and [`FrozenGain`] gates it out of
        // sizing under its own reason. BOTH coordinates, because one joint solve produces both
        // gains at that horizon and the columns are coupled through `ΣC·O`, so a
        // misparameterized anchor contaminates the offset estimate beside it.
        // `intercept_ceiling > 0` is not redundant with the comparison beside it. The
        // amplitude cost is `uncalibrated - residual` over the persistence, non-negative in
        // exact arithmetic - the unit gain is feasible for the same solve - so it can only go
        // negative by cancellation rounding, and a horizon whose constant forecast could earn
        // NOTHING must not be refused by the sign of a 1e-16 residue. An intercept that earns
        // nothing dominates nothing.
        let refused: Vec<bool> = (0..horizons)
            .map(|horizon| {
                amplitude_cost[horizon].is_finite()
                    && intercept_ceiling[horizon] > 0.
                    && intercept_ceiling[horizon]
                        > INTERCEPT_CEILING_SHARE * amplitude_cost[horizon]
            })
            .collect();
        let intercept_refused: Vec<InterceptRefusal> = (0..horizons)
            .filter(|horizon| refused[*horizon])
            .map(|horizon| InterceptRefusal {
                horizon: horizon + 1,
                intercept: intercept_ceiling[horizon],
                amplitude_cost: amplitude_cost[horizon],
            })
            .collect();
        ensure!(
            horizons - intercept_refused.len() >= 3,
            "the intercept refusal declined the pure-gain parameterization at {} of {horizons} \
             horizons ({}), leaving fewer than the three a curvature-penalized fit needs. Each \
             pair is that horizon's own close-channel constant-forecast share against what its \
             own amplitude error costs, and {INTERCEPT_CEILING_SHARE} of the second is the \
             registered share this estimator may leave on the table. This aborts rather than \
             applying the identity: a gain of 1 here would be indistinguishable from a measured \
             unit amplitude on every chart and in every checkpoint",
            intercept_refused.len(),
            intercept_refused
                .iter()
                .take(4)
                .map(|refusal| format!(
                    "h={} {:.3e} against {:.3e}",
                    refusal.horizon, refusal.intercept, refusal.amplitude_cost
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let curve = |coordinate: Coordinate| -> Result<CurveFit> {
            let read = |solved: &Solved| match coordinate {
                Coordinate::CloseAnchor => (solved.anchor, solved.anchor_weight),
                Coordinate::IntrabarOffset => (solved.offset, solved.offset_weight),
            };
            fit_curve(
                coordinate,
                horizons,
                intercept_refused.len(),
                // Recorded whatever its sign: a negative solve is a measurement of a horizon
                // with no usable amplitude, and it is the number a sizing consumer gates on.
                // `NaN` is the OTHER fact - this coordinate carried no energy to measure - and
                // the two must not collapse into one number. A refused horizon is recorded too:
                // the refusal is about the parameterization, not about the measurement.
                |horizon| {
                    solved[horizon]
                        .as_ref()
                        .and_then(|row| read(row).0)
                        .unwrap_or(f64::NAN)
                },
                |horizon| {
                    solved[horizon]
                        .as_ref()
                        .filter(|_| !refused[horizon])
                        .and_then(|row| {
                            let (gain, weight) = read(row);
                            gain.map(|gain| (gain, weight))
                        })
                        .filter(|(gain, weight)| {
                            gain.is_finite() && *gain > 0. && weight.is_finite() && *weight > 0.
                        })
                },
            )
        };
        Ok(Self {
            estimator,
            blocks,
            anchor: curve(Coordinate::CloseAnchor)?,
            offset: curve(Coordinate::IntrabarOffset)?,
            intercept_ceiling,
            amplitude_cost,
            intercept_refused,
        })
    }

    /// What the checkpoint carries and the model applies: the two curves and their provenance,
    /// without the diagnostics, which belong to the report that was written beside them.
    pub fn frozen(&self) -> FrozenGain {
        FrozenGain {
            estimator: self.estimator.clone(),
            blocks: self.blocks.clone(),
            anchor: self.anchor.gain.clone(),
            offset: self.offset.gain.clone(),
            measured_anchor: self
                .anchor
                .measured_gain
                .iter()
                .map(|gain| gain.is_finite().then_some(*gain))
                .collect(),
            intercept_refused: self.intercept_refused.clone(),
        }
    }
}

/// Why a horizon is gated out of position SIZING.
///
/// Three findings with three causes, and they never merge: a measured non-positive amplitude
/// (the forecast does not point the right way there), no measurement at all (the horizon is
/// carried by its neighbours through the roughness prior), and a refused parameterization (a
/// constant forecast could earn more of that horizon's MSE than a tenth of what its amplitude
/// error costs). Sizing is affine in the mean, so all three produce the same all-zero book,
/// and one sentinel for all three is how that book stops being explicable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizingGate {
    /// The calibration block identified no amplitude at this horizon.
    Unmeasured,
    /// The measured amplitude, non-positive, signed and unmodified.
    NonPositiveGain(f64),
    /// The intercept refusal, with the two shares that decided it.
    InterceptDominates { intercept: f64, amplitude_cost: f64 },
}

impl fmt::Display for SizingGate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unmeasured => write!(formatter, "unmeasured"),
            Self::NonPositiveGain(gain) => write!(formatter, "at {gain:.4}"),
            Self::InterceptDominates {
                intercept,
                amplitude_cost,
            } => write!(
                formatter,
                "intercept dominates, a constant forecast earning {intercept:.3e} against the \
                 {amplitude_cost:.3e} its amplitude error costs"
            ),
        }
    }
}

/// The frozen gain a checkpoint carries: two positive curves over the horizon axis, the signed
/// measurement they were smoothed from, and where they were fitted.
///
/// This is the whole of what is applied. It travels inside the authenticated manifest, so it
/// cannot be paired with a checkpoint it was not fitted on and needs no digest of its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrozenGain {
    pub estimator: String,
    pub blocks: Blocks,
    /// Multiplies the `√h`-scaled close mean every channel is built on.
    pub anchor: Vec<f64>,
    /// Multiplies the three intrabar offsets. Positive, so candle ordering survives.
    pub offset: Vec<f64>,
    /// The calibration block's OWN least-squares anchor gain per horizon, signed and
    /// unsmoothed: the measurement, not the applied curve. `None` where the block identified no
    /// amplitude at all, which is a different fact from a measured zero and is why this is an
    /// option rather than a sentinel - the manifest is JSON, and JSON has no NaN.
    ///
    /// Carried because the applied curve cannot answer the one question a position-sizing
    /// consumer has to ask. The fit runs on `ln g`, so every applied gain is positive by
    /// construction, and a horizon whose measured amplitude is NEGATIVE or absent is dropped
    /// from the fit and then interpolated from its neighbours - which is the right thing for a
    /// smooth curve and the wrong thing to trade, because it sizes a horizon on evidence that
    /// is not its own. [`Self::tradable`] is the gate; this vector is why it can exist without
    /// a second fit, and it is reported signed and unmodified so a gated horizon is legible.
    pub measured_anchor: Vec<Option<f64>>,
    /// The horizons whose pure-gain parameterization the calibration block refused, with the
    /// two shares that refused each. Rides in the manifest because it is a SIZING gate and
    /// nothing else downstream could reconstruct it: the applied curve is positive and smooth
    /// at a refused horizon - the roughness prior carries it from its neighbours - so the
    /// curve cannot say that the horizon's own evidence was declined.
    pub intercept_refused: Vec<InterceptRefusal>,
}

impl FrozenGain {
    /// Structural authentication, for the checkpoints a version string cannot catch: one
    /// stamped by hand, or one written by a build that changed the curve's meaning without
    /// moving the manifest format.
    pub fn validate(&self, pred_len: usize) -> Result<()> {
        ensure!(
            self.anchor.len() == pred_len
                && self.offset.len() == pred_len
                && self.measured_anchor.len() == pred_len,
            "the checkpoint's mean gain carries {} anchor, {} offset and {} measured \
             coefficients for a {pred_len}-horizon forecast",
            self.anchor.len(),
            self.offset.len(),
            self.measured_anchor.len()
        );
        ensure!(
            self.anchor
                .iter()
                .chain(&self.offset)
                .all(|gain| gain.is_finite() && *gain > 0.),
            "mean gains must be finite and strictly positive; a nonpositive gain is a sign flip \
             on every within-timestamp rank, not an amplitude correction, and a nonpositive \
             offset gain inverts the candle"
        );
        // Deliberately NOT a positivity check: a negative measured gain is a legitimate
        // measurement of a horizon with no usable amplitude, and clamping it here is exactly
        // the silent repair that left an all-zero trading result unexplained for a day.
        ensure!(
            self.measured_anchor
                .iter()
                .flatten()
                .all(|gain| gain.is_finite()),
            "a nonfinite measured anchor gain is a broken reduction, not a measurement; an \
             unmeasured horizon is null, which is a different statement from a measured zero"
        );
        ensure!(
            self.intercept_refused.iter().all(|refusal| {
                (1..=pred_len).contains(&refusal.horizon)
                    && refusal.intercept.is_finite()
                    && refusal.amplitude_cost.is_finite()
            }),
            "an intercept refusal must name a horizon of the {pred_len} forecast and both \
             shares that refused it; a refusal with no numbers gates a horizon for no stated \
             reason: {:?}",
            self.intercept_refused
        );
        Ok(())
    }

    /// Whether horizon `h` bars ahead (1-based) may be SIZED on, as opposed to merely scored.
    ///
    /// The applied curve is positive and defined everywhere; the evidence behind it is not. A
    /// horizon whose own calibration-block amplitude is non-positive or unidentified has no
    /// out-of-sample evidence that its forecast points the right way, and one whose
    /// parameterization was refused has evidence the estimator declined to use; a position
    /// taken at either is taken on its neighbours' evidence through the roughness prior.
    /// Scoring such a horizon is honest - the MSE ratio is a measurement either way - and
    /// trading it is not.
    pub fn tradable(&self, horizon: usize) -> bool {
        self.gate(horizon).is_none()
    }

    /// The one reason horizon `h` (1-based) cannot be sized, or `None` where it carries its
    /// own positive out-of-sample amplitude under a parameterization the block admitted.
    ///
    /// The intercept refusal is reported ahead of the measurement, because it is the stronger
    /// statement: the fit declined that horizon's own solve entirely, so whatever the
    /// measurement says there was not what determined the applied gain.
    pub fn gate(&self, horizon: usize) -> Option<SizingGate> {
        if let Some(refusal) = self
            .intercept_refused
            .iter()
            .find(|refusal| refusal.horizon == horizon)
        {
            return Some(SizingGate::InterceptDominates {
                intercept: refusal.intercept,
                amplitude_cost: refusal.amplitude_cost,
            });
        }
        match horizon
            .checked_sub(1)
            .and_then(|index| self.measured_anchor.get(index))
        {
            Some(Some(gain)) if *gain > 0. => None,
            Some(Some(gain)) => Some(SizingGate::NonPositiveGain(*gain)),
            _ => Some(SizingGate::Unmeasured),
        }
    }

    /// The 1-based horizons that cannot be sized, each with the reason that gated it, for the
    /// report that has to name them.
    pub fn gated(&self) -> Vec<(usize, SizingGate)> {
        (1..=self.measured_anchor.len())
            .filter_map(|horizon| self.gate(horizon).map(|gate| (horizon, gate)))
            .collect()
    }

    /// Why an account reading `horizons` (1-based) must be SIZED TO ZERO, or `None` when every
    /// horizon it reads carries its own positive out-of-sample amplitude.
    ///
    /// Prose and not a boolean, because the three gating reasons are different findings - a
    /// measured non-positive gain, no measurement at all, and a refused parameterization - and
    /// one sentinel for all three is how a gate stops being legible. Sizing is affine in the
    /// mean, so a zero mean is a zero position at every name; this string is the only thing
    /// that keeps the resulting all-zero book from being indistinguishable from a broken
    /// pipeline.
    pub fn sizing_refusal(&self, horizons: &[usize]) -> Option<String> {
        let named: Vec<String> = horizons
            .iter()
            .filter_map(|horizon| self.gate(*horizon).map(|gate| format!("h{horizon} {gate}")))
            .collect();
        (!named.is_empty()).then(|| named.join(", "))
    }
}

/// One horizon's least-squares solve over the coordinates the block actually identifies, `None`
/// where it identifies neither.
///
/// The normal equations are pooled over channels because the four channels of one bar share the
/// anchor: `f_c = C + O_c`, so the design is `[C, O_c]` with `O_close ≡ 0` and the four rows of
/// one bar constrain both coordinates jointly. That pooling is what makes the anchor gain
/// minimize the FOUR-CHANNEL squared error rather than the close channel's alone.
///
/// Either column can be EMPTY, and the solve degrades to rank 1 rather than refusing the
/// horizon. `O_close ≡ 0` by construction, so a decode whose intrabar range has collapsed - a
/// head early in training, or one scored on the close channel alone - leaves the offset column
/// identically zero at every channel and the 2x2 Gram matrix singular. The anchor amplitude is
/// still exactly measurable there, and dropping it because the SECOND coordinate is unmeasurable
/// would blank the calibration at precisely the checkpoints whose amplitude is most wrong.
/// A column with no energy has no target covariance either (`Σ O² = 0 ⇒ O ≡ 0 ⇒ Σ O·y = 0`), so
/// the residual and the amplitude cost below are exact with its term omitted.
fn solve_horizon(moments: &Moments, horizon: usize) -> Option<Solved> {
    let pooled = moments.pooled(horizon);
    let bars = moments.bars[horizon];
    let elements = bars * moments.channels as f64;
    let coordinates =
        usize::from(pooled.anchor_square > 0.) + usize::from(pooled.offset_square > 0.);
    if bars <= 0. || coordinates == 0 || elements <= coordinates as f64 || pooled.persistence <= 0.
    {
        return None;
    }
    // The Gram determinant at full rank, and the identified coordinate's own energy at rank 1.
    // Both are the quantity the delta-method cofactors below divide by, which is what lets one
    // weight expression serve both cases.
    let determinant = if coordinates == 2 {
        pooled.anchor_square * pooled.offset_square - pooled.anchor_offset * pooled.anchor_offset
    } else {
        pooled.anchor_square.max(pooled.offset_square)
    };
    if !(determinant > 0.) {
        return None;
    }
    let (anchor, offset, anchor_cofactor, offset_cofactor) = if coordinates == 2 {
        (
            Some(
                (pooled.offset_square * pooled.anchor_target
                    - pooled.anchor_offset * pooled.offset_target)
                    / determinant,
            ),
            Some(
                (pooled.anchor_square * pooled.offset_target
                    - pooled.anchor_offset * pooled.anchor_target)
                    / determinant,
            ),
            pooled.offset_square,
            pooled.anchor_square,
        )
    } else if pooled.anchor_square > 0. {
        (Some(pooled.anchor_target / determinant), None, 1., 0.)
    } else {
        (None, Some(pooled.offset_target / determinant), 0., 1.)
    };
    // `SSE(g) = Σy² - g'b` at the optimum, and the uncalibrated `SSE(1, 1)` minus it is what
    // fixing the amplitude is worth. Both in the same σ² units as the persistence they divide.
    let residual = pooled.persistence
        - anchor.unwrap_or(0.) * pooled.anchor_target
        - offset.unwrap_or(0.) * pooled.offset_target;
    let uncalibrated = pooled.anchor_square + 2. * pooled.anchor_offset + pooled.offset_square
        - 2. * (pooled.anchor_target + pooled.offset_target)
        + pooled.persistence;
    if !(residual > 0.) {
        return None;
    }
    // `Var(ĝ) = k·s²·G⁻¹` with `s² = SSE/(elements - rank)` and `k = channels`: the four candle
    // channels of one bar share one anchor error, so one BAR is one independent residual, not
    // four, and charging the multiplicity as a design effect doubles the standard error rather
    // than pretending to four times the sample. A constant factor on every weight cannot move
    // the smoother - the penalty grid is in units of the mean weight - so this only widens the
    // three-sigma amplification bound, which is the conservative direction.
    let variance = moments.channels as f64 * residual / (elements - coordinates as f64);
    let log_weight = |gain: Option<f64>, cofactor: f64| -> f64 {
        let gain_variance = variance * cofactor / determinant;
        match gain {
            Some(gain) if gain > 0. && gain_variance > 0. => gain * gain / gain_variance,
            _ => 0.,
        }
    };
    Some(Solved {
        anchor,
        offset,
        anchor_weight: log_weight(anchor, anchor_cofactor),
        offset_weight: log_weight(offset, offset_cofactor),
        amplitude_cost: (uncalibrated - residual) / pooled.persistence,
    })
}

/// The roughness-penalized WLS on `ln g` against `ln h`, plus the two-sided bound.
///
/// `measured(h)` is the block's raw signed solve at `h`, recorded whatever its sign, and
/// `identified(h)` yields `(ĝ_h, weight_h)` only where that solve is a usable positive
/// amplitude AT A HORIZON THE INTERCEPT REFUSAL ADMITS. Unidentified and refused horizons keep
/// weight zero: they are then determined by the roughness penalty alone - interpolated from
/// their neighbours, which is the only defensible thing a smoothness prior can say about
/// them - and their amplification ceiling is the identity, so a neighbour's evidence can shrink
/// them but never amplify them. Trading them is a separate question, answered by
/// [`FrozenGain::tradable`] off the measurement and the refusal list this records.
fn fit_curve(
    coordinate: Coordinate,
    horizons: usize,
    intercept_refused: usize,
    measured: impl Fn(usize) -> f64,
    identified: impl Fn(usize) -> Option<(f64, f64)>,
) -> Result<CurveFit> {
    let mut log_gain = vec![0.; horizons];
    let mut measured_gain = vec![f64::NAN; horizons];
    let mut weight = vec![0.; horizons];
    for horizon in 0..horizons {
        measured_gain[horizon] = measured(horizon);
        if let Some((gain, precision)) = identified(horizon) {
            log_gain[horizon] = gain.ln();
            weight[horizon] = precision;
        }
    }
    let usable = weight.iter().filter(|value| **value > 0.).count();
    ensure!(
        usable >= 3,
        "only {usable} of {horizons} horizons carry a calibratable {} amplitude the intercept \
         refusal admits ({intercept_refused} were declined as misparameterized), so there is \
         nothing to fit a gain curve on. This aborts rather than applying the identity: a gain \
         of 1 here would be indistinguishable from a measured unit amplitude",
        coordinate.label()
    );
    let axis: Vec<f64> = (1..=horizons).map(|h| (h as f64).ln()).collect();
    let roughness = roughness_matrix(&axis);
    let scale = weight.iter().sum::<f64>() / horizons as f64;
    let mut best: Option<(f64, f64, f64, Vec<f64>)> = None;
    for step in 0..PENALTY_GRID {
        let decade = PENALTY_DECADES.0
            + (PENALTY_DECADES.1 - PENALTY_DECADES.0) * step as f64 / (PENALTY_GRID - 1) as f64;
        let penalty = scale * 10f64.powf(decade);
        let Some((fitted, dof)) = smooth(&weight, &log_gain, &roughness, penalty) else {
            continue;
        };
        // Weighted generalized cross-validation. The residuals of neighbouring horizons are
        // positively correlated (their targets overlap in 191 of 192 bars), so this
        // under-smooths rather than over-smooths, which is the safe direction for a curve whose
        // measured second differences already sit below the noise scale.
        let residual: f64 = (0..horizons)
            .map(|i| weight[i] * (log_gain[i] - fitted[i]).powi(2))
            .sum::<f64>()
            / horizons as f64;
        let slack = 1. - dof / horizons as f64;
        if slack <= 0. {
            continue;
        }
        let score = residual / (slack * slack);
        if best.as_ref().is_none_or(|(current, ..)| score < *current) {
            best = Some((score, penalty, dof, fitted));
        }
    }
    let Some((_, penalty, effective_dof, fitted)) = best else {
        bail!(
            "no roughness penalty in the {PENALTY_GRID}-point grid produced a solvable {} fit \
             over the {usable} horizons that identified an amplitude",
            coordinate.label()
        );
    };
    let standard_error: Vec<f64> = weight
        .iter()
        .map(|value| {
            if *value > 0. {
                value.sqrt().recip()
            } else {
                f64::NAN
            }
        })
        .collect();
    let amplification_ceiling: Vec<f64> = (0..horizons)
        .map(|horizon| {
            if weight[horizon] > 0. {
                (log_gain[horizon] - AMPLIFICATION_SIGMAS * standard_error[horizon])
                    .exp()
                    .max(1.)
            } else {
                1.
            }
        })
        .collect();
    let gain: Vec<f64> = fitted
        .iter()
        .zip(&amplification_ceiling)
        .map(|(value, ceiling)| value.exp().min(*ceiling))
        .collect();
    ensure!(
        gain.iter().all(|value| value.is_finite() && *value > 0.),
        "the fitted {} gain left the positive range at {} of {horizons} horizons",
        coordinate.label(),
        gain.iter()
            .filter(|value| !(value.is_finite() && **value > 0.))
            .count()
    );
    Ok(CurveFit {
        coordinate,
        gain,
        measured_gain,
        standard_error,
        amplification_ceiling,
        weight,
        penalty,
        effective_dof,
        identified: usable,
    })
}

/// `R` such that `uᵀRu` is the natural-cubic-spline roughness `∫(u'')²` of the piecewise
/// quadratic through `(axis, u)`: divided second differences, each weighted by the interval it
/// integrates over, so a non-uniform axis is penalized in its own units rather than per index.
fn roughness_matrix(axis: &[f64]) -> Vec<f64> {
    let n = axis.len();
    let mut matrix = vec![0.; n * n];
    for index in 1..n - 1 {
        let (left, right) = (axis[index] - axis[index - 1], axis[index + 1] - axis[index]);
        let span = left + right;
        let curvature = 2. / span;
        let row = [
            (index - 1, curvature / left),
            (index, -curvature * (1. / left + 1. / right)),
            (index + 1, curvature / right),
        ];
        let measure = span / 2.;
        for (i, a) in row {
            for (j, b) in row {
                matrix[i * n + j] += measure * a * b;
            }
        }
    }
    matrix
}

/// One penalized solve: `u = (W + λR)⁻¹Wy` and `tr((W + λR)⁻¹W)`, the fit's effective degrees
/// of freedom. `None` when the system is not positive definite, which the caller treats as a
/// penalty strength to skip rather than a failure.
fn smooth(
    weight: &[f64],
    values: &[f64],
    roughness: &[f64],
    penalty: f64,
) -> Option<(Vec<f64>, f64)> {
    let n = weight.len();
    let mut system = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            system[i * n + j] = penalty * roughness[i * n + j];
        }
        system[i * n + i] += weight[i];
    }
    let factor = cholesky(system, n)?;
    let mut fitted: Vec<f64> = (0..n).map(|i| weight[i] * values[i]).collect();
    solve(&factor, n, &mut fitted);
    // `tr((W + λR)⁻¹W) = Σ_j w_j·[(W + λR)⁻¹]_{jj}`: one solve per column, which is where the
    // 192³ goes and why this is a host-side fit rather than anything on the device.
    let mut dof = 0.;
    let mut column = vec![0.; n];
    for j in 0..n {
        if weight[j] == 0. {
            continue;
        }
        column.iter_mut().for_each(|value| *value = 0.);
        column[j] = 1.;
        solve(&factor, n, &mut column);
        dof += weight[j] * column[j];
    }
    Some((fitted, dof))
}

/// In-place lower Cholesky factor of a symmetric matrix, `None` if it is not positive definite.
fn cholesky(mut matrix: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= matrix[i * n + k] * matrix[j * n + k];
            }
            if i == j {
                if !(sum > 0.) {
                    return None;
                }
                matrix[i * n + i] = sum.sqrt();
            } else {
                matrix[i * n + j] = sum / matrix[j * n + j];
            }
        }
        for j in i + 1..n {
            matrix[i * n + j] = 0.;
        }
    }
    Some(matrix)
}

/// Forward then back substitution against a lower Cholesky factor.
fn solve(factor: &[f64], n: usize, rhs: &mut [f64]) {
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= factor[i * n + k] * rhs[k];
        }
        rhs[i] = sum / factor[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for k in i + 1..n {
            sum -= factor[k * n + i] * rhs[k];
        }
        rhs[i] = sum / factor[i * n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks() -> Blocks {
        Blocks {
            calibration_first_origin_ms: 1,
            calibration_last_origin_ms: 2,
            calibration_last_target_ms: 3,
            calibration_origins: 4,
            evaluation_first_origin_ms: 5,
            evaluation_last_origin_ms: 6,
            evaluation_origins: 7,
            purge_gap_ms: 2,
        }
    }

    /// The truth this estimator has to recover: a curve that is smooth in `ln h` and passes
    /// through the three amplitudes measured on `timexer-control-4k/weights/best` (held-out
    /// full, step 3000) - `0.600` at h = 1, `0.990` at h = 11, `0.266` at h = 192. It rises
    /// before it falls, so a fit that assumed monotonicity would fail on the first eleven
    /// horizons, and it is smooth in log space, so a fit that merely passed its input through
    /// cannot match it.
    pub(super) fn measured_shape(horizon: usize) -> f64 {
        let v = (horizon as f64).ln();
        (-0.5108 + 0.5137 * v - 0.12716 * v * v).exp()
    }

    /// Moments consistent with an injected `(anchor, offset)` gain pair and a stated
    /// correlation, built so the exact 2x2 solve must return the injected pair.
    ///
    /// The construction is a population, not a sample: for each horizon it states
    /// `Σ C² = channels·varC·bars`, an offset that is orthogonal to the anchor
    /// (`Σ C·O = 0`, which is the clean case for reading the two coordinates apart), and
    /// targets whose cross moments are exactly `Σ C·y = ΣC²/anchor_gain` and
    /// `Σ O·y = ΣO²/offset_gain`. Then the least-squares solution is `1/gain` in each
    /// coordinate - i.e. the CORRECTION for a forecast that is `gain` times too large.
    pub(super) fn injected(
        horizons: usize,
        channels: usize,
        bars: f64,
        anchor_error: impl Fn(usize) -> f64,
        offset_error: impl Fn(usize) -> f64,
        rho: f64,
    ) -> Moments {
        let cells = horizons * channels;
        let mut moments = Moments {
            pred_len: horizons,
            channels,
            bars: vec![bars; horizons],
            target: vec![0.; cells],
            target_square: vec![0.; cells],
            anchor_square: vec![0.; cells],
            anchor_offset: vec![0.; cells],
            offset_square: vec![0.; cells],
            anchor_target: vec![0.; cells],
            offset_target: vec![0.; cells],
        };
        for horizon in 0..horizons {
            // Anchor variance grows with the horizon exactly as a cumulative return's does; the
            // intrabar offset does not, which is the real geometry and also what makes the two
            // coordinates' weights differ by orders of magnitude.
            let anchor_square = bars * (horizon + 1) as f64;
            let offset_square = bars * 0.25;
            let (anchor_gain, offset_gain) = (anchor_error(horizon), offset_error(horizon));
            for channel in 0..channels {
                let cell = horizon * channels + channel;
                let offset_square = if channel == CLOSE_CHANNEL {
                    0.
                } else {
                    offset_square
                };
                moments.anchor_square[cell] = anchor_square;
                moments.offset_square[cell] = offset_square;
                moments.anchor_offset[cell] = 0.;
                moments.anchor_target[cell] = anchor_square / anchor_gain;
                moments.offset_target[cell] = offset_square / offset_gain;
                // `Σy²` is set from the stated correlation: `Σfy = ρ·√(Σf²·Σy²)` with
                // `Σf² = ΣC² + ΣO²` here, so the population is a coherent (f, y) pair rather
                // than three unrelated sums.
                let forecast_square = anchor_square + offset_square;
                let forecast_target = moments.anchor_target[cell] + moments.offset_target[cell];
                moments.target_square[cell] =
                    (forecast_target / rho).powi(2) / forecast_square.max(f64::MIN_POSITIVE);
                moments.target[cell] = 0.;
            }
        }
        moments
    }

    #[test]
    fn the_exact_solve_recovers_an_injected_per_horizon_gain_in_both_coordinates() {
        let (horizons, channels) = (192, CHANNELS as usize);
        // A power-law amplitude error on the anchor - the measured shape, 3.662 at h=1 falling
        // to 0.195 at h=192 - and a constant one on the offsets.
        let anchor_error = |horizon: usize| 3.662 * ((horizon + 1) as f64).powf(-0.57);
        let moments = injected(horizons, channels, 400_000., anchor_error, |_| 1.4, 0.14);
        for horizon in 0..horizons {
            let solved = solve_horizon(&moments, horizon).expect("an identified horizon");
            let anchor = solved.anchor.expect("an identified anchor");
            let offset = solved.offset.expect("an identified offset");
            assert!(
                (anchor - 1. / anchor_error(horizon)).abs() < 1e-9,
                "anchor gain {anchor} at h={} against the injected {}",
                horizon + 1,
                1. / anchor_error(horizon)
            );
            assert!(
                (offset - 1. / 1.4).abs() < 1e-9,
                "offset gain {offset} at h={}",
                horizon + 1
            );
        }
        // A collapsed intrabar range empties the OFFSET column at every channel, so the 2x2
        // Gram matrix is singular - and the anchor amplitude is still exactly measurable. This
        // is the state a head early in training is in, which is the state whose amplitude is
        // most wrong, so refusing the horizon here would blank the calibration exactly where it
        // is needed.
        let mut flat = moments.clone();
        for cell in 0..horizons * channels {
            flat.offset_square[cell] = 0.;
            flat.offset_target[cell] = 0.;
        }
        for horizon in 0..horizons {
            let solved = solve_horizon(&flat, horizon).expect("a rank-1 design still identifies");
            let anchor = solved.anchor.expect("the anchor column carries energy");
            assert!(
                (anchor - 1. / anchor_error(horizon)).abs() < 1e-9,
                "{anchor}"
            );
            assert_eq!(solved.offset, None);
            assert!(solved.anchor_weight > 0. && solved.offset_weight == 0.);
        }
        // Both columns empty is the one case with nothing to solve.
        let mut silent = flat.clone();
        for cell in 0..horizons * channels {
            silent.anchor_square[cell] = 0.;
            silent.anchor_target[cell] = 0.;
        }
        assert!(solve_horizon(&silent, 0).is_none());
    }

    /// The fit reproduces a KNOWN gain curve of its own family, to a tolerance that leaves no
    /// room for a wrong axis, a wrong weight or a penalty that does not vanish.
    ///
    /// The injected amplitude error is the reciprocal of the curve the fit has to return, so
    /// what comes back must BE that curve - smooth in `ln h`, inside the fitted family, and with
    /// no per-horizon perturbation at all. Any deviation here is the estimator's own bias rather
    /// than sampling error, which is what makes the tolerance meaningful: the previous test
    /// proves it averages noise out, this one proves it does not distort the signal while doing
    /// so. Both coordinates, because the offset curve is fitted from the same solve and a
    /// smoother that leaked the anchor's sweep into the intrabar spread would pass the anchor
    /// half alone.
    ///
    /// Both injected curves sit strictly BELOW 1, and deliberately: the amplifying direction is
    /// bounded by `exp(ln ĝ - 3·SE)`, so a gain above 1 is reproduced only as far as its own
    /// standard error proves and a reproduction tolerance there would be a tolerance on the
    /// bound instead of on the fit. The two ceiling tests cover that half. `measured_shape`
    /// itself crosses 1 in a band around h = 5..25, which is exactly why it is scaled here.
    #[test]
    fn the_fit_reproduces_a_noiseless_injected_gain_curve_to_a_tight_tolerance() {
        let horizons = 192;
        let anchor_gain = |horizon: usize| 0.85 * measured_shape(horizon + 1);
        let offset_gain = |horizon: usize| 0.8 + 0.0005 * horizon as f64;
        let moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |h| anchor_gain(h).recip(),
            |h| offset_gain(h).recip(),
            0.06,
        );
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        assert!(fit.anchor.amplification_ceiling.iter().all(|c| *c == 1.));
        for horizon in 0..horizons {
            for (coordinate, curve, injected_gain) in [
                ("anchor", &fit.anchor, anchor_gain(horizon)),
                ("offset", &fit.offset, offset_gain(horizon)),
            ] {
                // The raw solve first: the smoother cannot be credited for a measurement it
                // never received.
                assert!(
                    (curve.measured_gain[horizon] / injected_gain - 1.).abs() < 1e-9,
                    "the measured {coordinate} gain {} at h={} is not the injected {injected_gain}",
                    curve.measured_gain[horizon],
                    horizon + 1
                );
                assert!(
                    (curve.gain[horizon] / injected_gain - 1.).abs() < 2e-3,
                    "the fitted {coordinate} gain {} at h={} left the injected {injected_gain} by \
                     more than 0.2%",
                    curve.gain[horizon],
                    horizon + 1
                );
            }
        }
        // And the recovered anchor curve carries the whole measured sweep, so the tolerance
        // above is tight against something with real dynamic range rather than against a line.
        assert!(fit.anchor.gain[0] / fit.anchor.gain[191] > 2.);
    }

    /// A horizon whose own block measured a non-positive amplitude is GATED and NAMED, and its
    /// applied gain is neither clamped to zero nor reported as a measurement it is not.
    ///
    /// Three facts have to hold at once and each one fails a different plausible bug: the
    /// APPLIED curve stays strictly positive there (the fit runs on `ln g`, and a negative
    /// applied gain would invert every position at that horizon rather than shrink it); the
    /// MEASUREMENT survives signed and unrounded, because that is the only evidence a sizing
    /// consumer has; and the sizing decision is a refusal that names the horizon and its value,
    /// because an all-zero book with no stated reason is indistinguishable from a broken
    /// pipeline. A silent clamp would satisfy the first and destroy the other two.
    #[test]
    fn a_nonpositive_measured_gain_gates_its_horizon_to_zero_size_and_is_named_not_clamped() {
        let horizons = 64;
        let mut moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |h| measured_shape(h + 1).recip(),
            |_| 1.,
            0.06,
        );
        // A sign-inverted anchor at h = 41..44: the forecast points the wrong way there, which
        // is a measurement and not a magnitude error.
        for horizon in 40..44 {
            for channel in 0..moments.channels {
                let cell = horizon * moments.channels + channel;
                moments.anchor_target[cell] = -moments.anchor_target[cell];
            }
        }
        // And h = 50 identifies nothing at all: a constant forecast, so there is no amplitude
        // to have a sign. Unmeasured and negative are different findings and the gate has to
        // keep them apart.
        for channel in 0..moments.channels {
            let cell = 50 * moments.channels + channel;
            moments.anchor_square[cell] = 0.;
            moments.anchor_target[cell] = 0.;
            moments.offset_square[cell] = 0.;
            moments.offset_target[cell] = 0.;
        }
        let frozen = MeanCalibration::fit(blocks(), &moments).unwrap().frozen();
        // A negative measurement is not a broken checkpoint: it authenticates.
        frozen.validate(horizons).unwrap();
        for horizon in 41..=44 {
            let measured = frozen.measured_anchor[horizon - 1].expect("a signed measurement");
            assert!(
                measured < 0.,
                "h={horizon} lost its negative measurement to a clamp: {measured}"
            );
            assert!(
                frozen.anchor[horizon - 1] > 0.,
                "the applied gain at h={horizon} left the positive range: {}",
                frozen.anchor[horizon - 1]
            );
            assert!(!frozen.tradable(horizon));
        }
        assert_eq!(frozen.measured_anchor[50], None);
        assert!(!frozen.tradable(51));
        // Every OTHER horizon is untouched: a gate that swallowed its neighbours would zero the
        // whole book on four bad horizons.
        for horizon in [1, 40, 45, 50, 52, 64] {
            assert!(
                frozen.tradable(horizon),
                "h={horizon} was gated by a neighbour"
            );
        }
        assert_eq!(
            frozen.gated().iter().map(|(h, _)| *h).collect::<Vec<_>>(),
            [41, 42, 43, 44, 51]
        );
        // The deployed decision, on the deployed call: named with its own value, and `None`
        // wherever nothing gates.
        let refusal = frozen
            .sizing_refusal(&[42, 1])
            .expect("a gated exit horizon must refuse to size");
        assert!(
            refusal.starts_with("h42 at -") && !refusal.contains("h1"),
            "the refusal must name the gated horizon at its measured value: {refusal}"
        );
        assert_eq!(
            frozen.sizing_refusal(&[51, 1]).as_deref(),
            Some("h51 unmeasured")
        );
        assert_eq!(frozen.sizing_refusal(&[45, 1]), None);
    }

    #[test]
    fn the_fitted_curve_recovers_a_smooth_amplitude_and_averages_out_per_horizon_noise() {
        let horizons = 192;
        // Sign-alternating perturbation: the estimator must not be able to pass by luck of a
        // seed, and averaging out an alternating error is exactly the job. The injected
        // amplitude error is the inverse of the target gain curve, so the fitted gain must come
        // back as `measured_shape`.
        let noisy = |horizon: usize| {
            (measured_shape(horizon + 1) * (1. + 0.08 * if horizon % 2 == 1 { 1. } else { -1. }))
                .recip()
        };
        let moments = injected(horizons, CHANNELS as usize, 400_000., noisy, |_| 1., 0.06);
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        let error = |curve: &[f64]| -> f64 {
            (0..horizons)
                .map(|i| (curve[i] - measured_shape(i + 1)).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let raw: Vec<f64> = (0..horizons).map(|h| noisy(h).recip()).collect();
        let (fitted, unsmoothed) = (error(&fit.anchor.gain), error(&raw));
        assert!(
            fitted < unsmoothed / 3.,
            "smoothing rms error {fitted} against raw {unsmoothed}"
        );
        assert!(fit.anchor.amplification_ceiling.iter().all(|c| *c >= 1.));
        assert!(fit
            .anchor
            .gain
            .iter()
            .zip(&fit.anchor.amplification_ceiling)
            .all(|(gain, ceiling)| *gain > 0. && gain <= ceiling));
        assert!(
            fit.anchor.gain[10] > 1.4 * fit.anchor.gain[191],
            "the fit flattened the measured decay: {} at h=11 against {} at h=192",
            fit.anchor.gain[10],
            fit.anchor.gain[191]
        );
        assert!(
            fit.anchor.gain[10] > 1.15 * fit.anchor.gain[0],
            "the fit erased the short-horizon rise: {} at h=1 against {} at h=11",
            fit.anchor.gain[0],
            fit.anchor.gain[10]
        );
        assert!(
            fit.anchor.effective_dof > 2. && fit.anchor.effective_dof < horizons as f64,
            "effective dof {} is not inside the family it selects from",
            fit.anchor.effective_dof
        );
        // The offset coordinate was injected at exactly 1, so its curve must sit on the
        // identity: a smoother that leaked the anchor's decay into the second coordinate would
        // be applying the wrong correction to the intrabar spread.
        assert!(
            fit.offset.gain.iter().all(|gain| (*gain - 1.).abs() < 0.05),
            "the offset curve drifted off the identity it was given: {:?}",
            &fit.offset.gain[..4]
        );
    }

    #[test]
    fn an_amplification_its_own_standard_error_cannot_prove_is_bounded_to_the_identity() {
        let horizons = 32;
        // A measured under-amplitude of 2.5 on 400 bars: the solve's own standard error on
        // `ln g` is far larger than `ln 2.5`, so none of it is deployable.
        let moments = injected(
            horizons,
            CHANNELS as usize,
            400.,
            |_| 1. / 2.5,
            |_| 1.,
            0.06,
        );
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        assert!(fit
            .anchor
            .measured_gain
            .iter()
            .all(|gain| (*gain - 2.5).abs() < 1e-9));
        assert!(fit.anchor.amplification_ceiling.iter().all(|c| *c == 1.));
        assert!(fit
            .anchor
            .gain
            .iter()
            .all(|gain| (*gain - 1.).abs() < 1e-12));
    }

    #[test]
    fn an_amplification_the_data_proves_is_applied_rather_than_pinned_to_one() {
        let horizons = 32;
        // The same measured gain on a thousand times the bars: now many standard errors above
        // the identity, and the shrinkage-only clamp this estimator replaced was leaving the
        // whole factor on the table.
        let moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 2.5,
            |_| 1.,
            0.06,
        );
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        for horizon in 0..horizons {
            let ceiling = fit.anchor.amplification_ceiling[horizon];
            let expected =
                (2.5f64.ln() - AMPLIFICATION_SIGMAS * fit.anchor.standard_error[horizon]).exp();
            assert!(
                (ceiling - expected).abs() < 1e-9,
                "ceiling {ceiling} at h={} is not the three-sigma lower end {expected}",
                horizon + 1
            );
            assert!(
                (fit.anchor.gain[horizon] - ceiling).abs() < 1e-9,
                "gain {} at h={} did not take the ceiling {ceiling}",
                fit.anchor.gain[horizon],
                horizon + 1
            );
            assert!(fit.anchor.gain[horizon] > 2.2);
        }
    }

    #[test]
    fn horizons_with_no_calibratable_amplitude_are_carried_by_their_neighbours() {
        let horizons = 64;
        let mut moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |h| measured_shape(h + 1).recip(),
            |_| 1.,
            0.06,
        );
        // A constant forecast at these horizons: the anchor emits nothing, so the gain is
        // undefined rather than zero. The curve must stay a smooth shrinkage across them.
        for horizon in 30..40 {
            for channel in 0..moments.channels {
                let cell = horizon * moments.channels + channel;
                moments.anchor_square[cell] = 0.;
                moments.anchor_target[cell] = 0.;
                moments.offset_square[cell] = 0.;
                moments.offset_target[cell] = 0.;
            }
        }
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        assert!(fit.anchor.weight[30..40].iter().all(|weight| *weight == 0.));
        // The unsolved horizons' amplitude cost is a GAP on the moment panel, never 0.0: a
        // zero there is a measurement - "fixing this horizon's amplitude is worth nothing" -
        // and it is also the value that would pass the intercept gate on evidence that does
        // not exist. The intercept ceiling stays measured, because the TARGET moments are
        // untouched at these horizons; only the forecast's amplitude vanished.
        assert!(
            fit.amplitude_cost[30..40].iter().all(|cost| cost.is_nan()),
            "an unmeasured amplitude cost was folded to a value: {:?}",
            &fit.amplitude_cost[30..40]
        );
        assert!(fit.amplitude_cost[..30].iter().all(|cost| *cost > 0.));
        assert!(fit.intercept_ceiling.iter().all(|share| share.is_finite()));
        for horizon in 30..40 {
            let (low, high) = (
                fit.anchor.gain[29].min(fit.anchor.gain[40]) * 0.7,
                fit.anchor.gain[29].max(fit.anchor.gain[40]) * 1.3,
            );
            assert!(
                (low..=high).contains(&fit.anchor.gain[horizon]),
                "interpolated gain {} at h={} left its neighbours' bracket {low}..={high}",
                fit.anchor.gain[horizon],
                horizon + 1
            );
        }
    }

    /// The measured shares of job 6004, as a fixture: a NEGLIGIBLE close-channel intercept
    /// under a large POOLED four-channel one.
    ///
    /// `close_share` and `intrabar_share` are each set as `bars·ȳ_c²/Σy_c²` exactly, so the two
    /// competing quantities are stated rather than approached. Only `target` is touched, and
    /// `target` enters nothing but the intercept ceiling, so the solve, the weights and the
    /// amplitude cost are bit-identical to the population the fit was already tested on: the
    /// gate is the only thing under test.
    fn with_intercept(moments: &mut Moments, close_share: f64, intrabar_share: f64) {
        for horizon in 0..moments.pred_len {
            for channel in 0..moments.channels {
                let cell = horizon * moments.channels + channel;
                let share = if channel == CLOSE_CHANNEL {
                    close_share
                } else {
                    intrabar_share
                };
                moments.target[cell] =
                    (share * moments.target_square[cell] * moments.bars[horizon]).sqrt();
            }
        }
    }

    /// `Σ_c bars·ȳ_c²/Σ_c Σy_c²` - the four-channel pooled intercept the gate used to read.
    /// Computed here from the moment fields, so what the test calls "the quantity that produced
    /// the bad run" is arithmetic in the test and not a call into the code under test.
    fn pooled_intercept(moments: &Moments, horizon: usize) -> f64 {
        let (mut earnable, mut persistence) = (0., 0.);
        for channel in 0..moments.channels {
            let cell = horizon * moments.channels + channel;
            earnable += moments.target[cell] * moments.target[cell] / moments.bars[horizon];
            persistence += moments.target_square[cell];
        }
        earnable / persistence
    }

    /// THE CASE THAT PRODUCED THE BAD RUN. A block whose close-channel intercept is negligible
    /// and whose POOLED four-channel intercept is two orders of magnitude larger must be
    /// FITTED, at every horizon.
    ///
    /// The pooled quantity is nonzero on any candle data whatsoever - a bar's high is never
    /// below its close - and it is the intrabar spread, which is what the offset COORDINATE
    /// absorbs. Job 6004 charged it to the intercept and blanked all 192 horizons in both
    /// coordinates. The fixture states both shares at the run's own measured values, and the
    /// test asserts the discriminating fact in both directions: the pooled quantity WOULD have
    /// refused, and the registered close-channel one does not.
    #[test]
    fn a_large_pooled_intercept_over_a_negligible_close_one_is_fitted_at_every_horizon() {
        let horizons = 32;
        let mut moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 0.6,
            |_| 1.,
            0.06,
        );
        // 4.63e-5 is the control checkpoint's own close-channel margin; 0.1 per intrabar
        // channel reproduces the 7.535e-2 the pooled quantity read at h = 1 on job 6004.
        with_intercept(&mut moments, 4.63e-5, 0.1);
        let fit = MeanCalibration::fit(blocks(), &moments)
            .expect("a negligible close-channel intercept must not refuse a pure gain");
        assert!(
            fit.intercept_refused.is_empty(),
            "refused {:?} on an intercept the close channel does not carry",
            fit.intercept_refused
        );
        for horizon in 0..horizons {
            let (close, pooled) = (
                fit.intercept_ceiling[horizon],
                pooled_intercept(&moments, horizon),
            );
            let cost = fit.amplitude_cost[horizon];
            assert!(
                (close - 4.63e-5).abs() < 1e-9,
                "the ceiling must be the close channel's stated share, got {close} at h={}",
                horizon + 1
            );
            assert!(
                pooled > 1e3 * close && pooled > INTERCEPT_CEILING_SHARE * cost,
                "the fixture must actually carry the defect: pooled {pooled} against close \
                 {close} and cost {cost} at h={}",
                horizon + 1
            );
            assert!(
                close <= INTERCEPT_CEILING_SHARE * cost,
                "close intercept {close} against {INTERCEPT_CEILING_SHARE} of the {cost} \
                 amplitude cost at h={}",
                horizon + 1
            );
        }
        // And the fit is a real one, not the identity a refusal used to ship: the injected
        // amplitude error is 1/0.6, so every horizon must come back at 0.6.
        assert_eq!(fit.anchor.identified, horizons);
        assert!(
            fit.anchor.gain.iter().all(|gain| (gain - 0.6).abs() < 1e-3),
            "{:?}",
            fit.anchor.gain
        );
    }

    /// A genuinely dominant close intercept at ONE horizon refuses THAT horizon and leaves the
    /// others fitted.
    ///
    /// This is the whole content of the per-horizon axis: the registered statement is "could
    /// earn at that horizon", the two quantities peak at opposite ends of the axis (h = 1 for
    /// the intercept, past h = 170 for the amplitude cost), and a max-against-max comparison
    /// let one short horizon blank the other 191.
    #[test]
    fn a_dominant_intercept_refuses_its_own_horizon_and_leaves_the_rest_fitted() {
        let horizons = 32;
        let refused_index = 10;
        let mut moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 0.6,
            |_| 1.,
            0.06,
        );
        with_intercept(&mut moments, 4.63e-5, 0.1);
        // One horizon where a constant forecast really could earn half the persistence MSE.
        let cell = refused_index * moments.channels + CLOSE_CHANNEL;
        moments.target[cell] =
            (0.5 * moments.target_square[cell] * moments.bars[refused_index]).sqrt();
        let fit = MeanCalibration::fit(blocks(), &moments)
            .expect("one refused horizon must not refuse the fit");
        assert_eq!(
            fit.intercept_refused,
            vec![InterceptRefusal {
                horizon: refused_index + 1,
                intercept: fit.intercept_ceiling[refused_index],
                amplitude_cost: fit.amplitude_cost[refused_index],
            }],
            "exactly one horizon carries a dominant intercept and it must be named with both \
             of its competing quantities"
        );
        assert!((fit.intercept_ceiling[refused_index] - 0.5).abs() < 1e-9);
        // Its own evidence is dropped from BOTH curves - one joint solve produces both gains -
        // and every other horizon keeps its weight.
        assert_eq!(fit.anchor.weight[refused_index], 0.);
        assert_eq!(fit.offset.weight[refused_index], 0.);
        assert_eq!(fit.anchor.identified, horizons - 1);
        assert_eq!(fit.offset.identified, horizons - 1);
        for horizon in 0..horizons {
            if horizon == refused_index {
                continue;
            }
            assert!(
                fit.anchor.weight[horizon] > 0.,
                "h={} lost its weight",
                horizon + 1
            );
            assert!(
                (fit.anchor.gain[horizon] - 0.6).abs() < 1e-3,
                "h={} was blanked by another horizon's refusal: {}",
                horizon + 1,
                fit.anchor.gain[horizon]
            );
        }
        // The refused horizon is still SCORED, under the gain its neighbours' evidence gives
        // it through the roughness prior and bounded above by the identity. Not 1, which is
        // what a whole-axis refusal used to ship and what `AmplitudeSplit::ratios` cannot tell
        // from a measured unit amplitude.
        assert!(
            (fit.anchor.gain[refused_index] - 0.6).abs() < 1e-3
                && (fit.anchor.gain[refused_index] - 1.).abs() > 0.1,
            "the refused horizon must be carried by its neighbours, not frozen at 1: {}",
            fit.anchor.gain[refused_index]
        );
        // Its measurement survives unmodified - the refusal is about the parameterization, not
        // about the number - and it is the SIZING gate that names it.
        let frozen = fit.frozen();
        assert!((fit.anchor.measured_gain[refused_index] - 0.6).abs() < 1e-9);
        assert_eq!(frozen.measured_anchor[refused_index], Some(0.6));
        assert!(!frozen.tradable(refused_index + 1));
        assert!(frozen.tradable(refused_index) && frozen.tradable(refused_index + 2));
        assert_eq!(frozen.gated().len(), 1);
        let refusal = frozen
            .sizing_refusal(&[refused_index + 1, 1])
            .expect("a refused exit horizon must refuse to size");
        assert!(
            refusal.starts_with(&format!("h{} intercept dominates", refused_index + 1))
                && refusal.contains("5.000e-1")
                && !refusal.contains("h1 "),
            "{refusal}"
        );
        frozen.validate(horizons).unwrap();
    }

    /// An axis on which EVERY horizon's intercept dominates has no pure-gain parameterization
    /// to fit, and that aborts - naming the count and the refusing pairs - rather than shipping
    /// a curve of ones that no downstream artifact can tell from a measurement.
    #[test]
    fn an_intercept_that_dominates_every_horizon_aborts_the_fit() {
        let horizons = 32;
        let mut moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 0.6,
            |_| 1.,
            0.06,
        );
        with_intercept(&mut moments, 0.5, 0.1);
        let error = MeanCalibration::fit(blocks(), &moments)
            .expect_err("an intercept above the pre-registered share at every horizon must refuse")
            .to_string();
        assert!(
            error.contains("declined the pure-gain parameterization at 32 of 32 horizons")
                && error.contains("h=1 5.000e-1 against")
                && error.contains("indistinguishable from a measured unit amplitude"),
            "{error}"
        );
        // And a zero-mean population leaves nothing to earn at all.
        let control = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 0.6,
            |_| 1.,
            0.06,
        );
        let fit = MeanCalibration::fit(blocks(), &control).unwrap();
        assert!(
            fit.intercept_ceiling.iter().all(|ceiling| *ceiling == 0.),
            "a zero-mean target population must leave no intercept to earn"
        );
        assert!(fit.intercept_refused.is_empty());
    }

    /// The three SIZING refusals stay THREE findings in the emitted record: a refused
    /// parameterization, a measured non-positive amplitude, and no measurement at all.
    ///
    /// All three produce the same all-zero book - sizing is affine in the mean - so one
    /// sentinel for all three is how that book stops being explicable. Each is built out of the
    /// moments rather than edited into the frozen gain afterwards, so the record under test is
    /// one the estimator actually produces.
    #[test]
    fn the_three_sizing_refusals_remain_distinguishable_in_the_frozen_record() {
        let horizons = 32;
        let (dominated, inverted, silent) = (10usize, 20usize, 25usize);
        let mut moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 0.6,
            |_| 1.,
            0.06,
        );
        with_intercept(&mut moments, 4.63e-5, 0.1);
        let close = |horizon: usize| horizon * moments.channels + CLOSE_CHANNEL;
        moments.target[close(dominated)] =
            (0.5 * moments.target_square[close(dominated)] * moments.bars[dominated]).sqrt();
        for channel in 0..moments.channels {
            let cell = inverted * moments.channels + channel;
            // A measured NEGATIVE amplitude: the forecast points the wrong way here, which is a
            // measurement and not a failure, and it may never be clamped.
            moments.anchor_target[cell] = -moments.anchor_target[cell];
            let empty = silent * moments.channels + channel;
            // No anchor energy at all: the solve degrades to rank 1 and the anchor is honestly
            // unmeasured. Its intercept goes with it - a horizon with no anchor to gain has no
            // amplitude cost either, and an intercept that earns nothing dominates nothing.
            moments.anchor_square[empty] = 0.;
            moments.anchor_target[empty] = 0.;
            moments.target[empty] = 0.;
        }
        let frozen = MeanCalibration::fit(blocks(), &moments)
            .expect("three gated horizons out of 32 leave a fittable curve")
            .frozen();
        let gated = frozen.gated();
        assert_eq!(gated.len(), 3, "{gated:?}");
        assert_eq!(gated[0].0, dominated + 1);
        assert!(
            matches!(gated[0].1, SizingGate::InterceptDominates { intercept, .. } if (intercept - 0.5).abs() < 1e-9),
            "{:?}",
            gated[0].1
        );
        assert_eq!(gated[1].0, inverted + 1);
        assert!(
            matches!(gated[1].1, SizingGate::NonPositiveGain(gain) if gain < 0.),
            "{:?}",
            gated[1].1
        );
        assert_eq!(gated[2], (silent + 1, SizingGate::Unmeasured));
        // Distinguishable as TEXT too, which is the only form a report or an account's
        // assumption list carries them in.
        let named: Vec<String> = gated
            .iter()
            .map(|(horizon, gate)| format!("h{horizon} {gate}"))
            .collect();
        assert!(named[0].contains("intercept dominates") && named[0].contains("5.000e-1"));
        assert!(named[1].contains(" at -"));
        assert!(named[2].ends_with("unmeasured"));
        assert_eq!(
            frozen.sizing_refusal(&[dominated + 1, inverted + 1, silent + 1]),
            Some(named.join(", "))
        );
        // Every OTHER horizon is untouched: three gated horizons may not zero the book.
        assert_eq!(frozen.sizing_refusal(&[1, horizons]), None);
        frozen.validate(horizons).unwrap();
    }

    /// A block with no calibratable amplitude ABORTS. It used to produce the identity with a
    /// reason attached, which is the defect this test exists to keep out: the reason lives in
    /// a struct field nothing on a chart reads, while the gain of 1 it ships is bit-identical
    /// to a measured unit amplitude - so `AmplitudeSplit::ratios` renders the `calibrated`
    /// series as an exact copy of the `uncalibrated` one and the run reports a calibration it
    /// never performed.
    #[test]
    fn a_constant_forecast_aborts_the_fit_instead_of_freezing_the_identity() {
        let horizons = 8;
        let moments = Moments {
            pred_len: horizons,
            channels: CHANNELS as usize,
            bars: vec![1000.; horizons],
            target: vec![0.; horizons * CHANNELS as usize],
            target_square: vec![1.; horizons * CHANNELS as usize],
            anchor_square: vec![0.; horizons * CHANNELS as usize],
            anchor_offset: vec![0.; horizons * CHANNELS as usize],
            offset_square: vec![0.; horizons * CHANNELS as usize],
            anchor_target: vec![0.; horizons * CHANNELS as usize],
            offset_target: vec![0.; horizons * CHANNELS as usize],
        };
        let error = MeanCalibration::fit(blocks(), &moments)
            .expect_err("a constant forecast has no amplitude to fit")
            .to_string();
        assert!(
            error.contains("nothing to fit a gain curve on")
                && error.contains("indistinguishable from a measured unit amplitude"),
            "{error}"
        );
    }

    #[test]
    fn a_frozen_gain_is_refused_for_the_wrong_horizon_count_or_a_nonpositive_curve() {
        let horizons = 16;
        let moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |h| measured_shape(h + 1).recip(),
            |_| 1.,
            0.06,
        );
        let frozen = MeanCalibration::fit(blocks(), &moments).unwrap().frozen();
        frozen.validate(horizons).unwrap();
        let refusal = frozen.validate(horizons + 1).unwrap_err().to_string();
        assert!(
            refusal.contains("coefficients for a 17-horizon forecast"),
            "{refusal}"
        );
        for mutate in [
            (|gain: &mut FrozenGain| gain.anchor[0] = 0.) as fn(&mut FrozenGain),
            |gain: &mut FrozenGain| gain.offset[0] = -1.,
            |gain: &mut FrozenGain| gain.anchor[3] = f64::NAN,
        ] {
            let mut edited = frozen.clone();
            mutate(&mut edited);
            let refusal = edited.validate(horizons).unwrap_err().to_string();
            assert!(
                refusal.contains("finite and strictly positive"),
                "{refusal}"
            );
        }
    }

    #[test]
    fn the_gained_ratio_matches_the_closed_form_and_a_gain_cannot_move_a_correlation() {
        // One horizon, one channel, an explicit population: forecast `f`, target `y`, so the
        // closed forms below are checked against arithmetic done by hand rather than against
        // the implementation's own algebra.
        let (forecast, target): (Vec<f64>, Vec<f64>) = (0..64)
            .map(|i| {
                let f = ((i % 7) as f64 - 3.) * 0.5;
                let y = 0.2 * f + ((i % 5) as f64 - 2.) * 0.3;
                (f, y)
            })
            .unzip();
        let bars = forecast.len() as f64;
        let sum = |values: &[f64]| values.iter().sum::<f64>();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let moments = Moments {
            pred_len: 3,
            channels: 1,
            bars: vec![bars; 3],
            target: vec![sum(&target); 3],
            target_square: vec![dot(&target, &target); 3],
            anchor_square: vec![dot(&forecast, &forecast); 3],
            anchor_offset: vec![0.; 3],
            offset_square: vec![0.; 3],
            anchor_target: vec![dot(&forecast, &target); 3],
            offset_target: vec![0.; 3],
        };
        let gain = 1.7;
        let sf = (dot(&forecast, &forecast) / bars).sqrt();
        let sy = (dot(&target, &target) / bars).sqrt();
        let rho = dot(&forecast, &target) / bars / (sf * sy);
        // `MSE(g·f, y)/mean(y²) = 1 - 2·g·ρ·sf/sy + g²·sf²/sy²` with sf, sy the UNCENTERED
        // second moments and ρ the uncentered correlation, which is the identity the amplitude
        // fix is derived from.
        let predicted = 1. - 2. * gain * rho * sf / sy + gain * gain * sf * sf / (sy * sy);
        let measured = moments.channel_gained_ratio(0, 0, gain, 1.);
        assert!(
            (measured - predicted).abs() < 1e-12,
            "gained ratio {measured} against the closed form {predicted}"
        );
        assert!(
            (moments.channel_gained_ratio(0, 0, 1., 1.) - moments.channel_ratio(0, 0)).abs()
                < 1e-12
        );
        // A per-horizon gain is a positive scalar within the horizon, so it multiplies the
        // uncentered correlation's numerator and its own standard deviation by the same factor:
        // ρ is invariant, exactly, and every rank statistic with it.
        let gained: Vec<f64> = forecast.iter().map(|f| gain * f).collect();
        let gained_rho =
            dot(&gained, &target) / (dot(&gained, &gained).sqrt() * dot(&target, &target).sqrt());
        let raw_rho = dot(&forecast, &target)
            / (dot(&forecast, &forecast).sqrt() * dot(&target, &target).sqrt());
        assert!(
            (gained_rho - raw_rho).abs() < 1e-15,
            "a per-horizon gain moved the correlation from {raw_rho} to {gained_rho}"
        );
        // And the MSE-optimal gain is exactly the correction the solve returns. This population
        // is close-anchor only, so the design is rank 1 and the offset coordinate is honestly
        // unmeasured rather than a fabricated zero.
        let solved = solve_horizon(&moments, 0).expect("an identified horizon");
        assert_eq!(solved.offset, None);
        assert!(
            (solved.anchor.unwrap() - dot(&forecast, &target) / dot(&forecast, &forecast)).abs()
                < 1e-12
        );
    }

    #[test]
    fn the_two_partitions_are_dated_and_proven_disjoint_in_the_bars_their_targets_read() {
        // Four calibration timestamps whose targets reach two slots ahead, then a purge band,
        // then three evaluation timestamps: the shape the corpus's reserved [70%, 80%) and
        // [80%, 90%) partitions produce.
        let step = 300_000i64;
        let base = 1_700_000_000_000i64;
        let calibration: Vec<(i64, i64)> = (0..4)
            .map(|slot| (base + slot * step, base + (slot + 2) * step))
            .collect();
        let evaluation: Vec<(i64, i64)> = (8..11)
            .map(|slot| (base + slot * step, base + (slot + 2) * step))
            .collect();
        let blocks = Blocks::spanning(&calibration, &evaluation).unwrap();
        assert_eq!(blocks.calibration_origins, 4);
        assert_eq!(blocks.evaluation_origins, 3);
        assert_eq!(blocks.calibration_first_origin_ms, base);
        assert_eq!(blocks.calibration_last_origin_ms, base + 3 * step);
        // The reach, not the last origin: the last calibration origin's targets run two slots
        // past it and that is what the gap has to clear.
        assert_eq!(blocks.calibration_last_target_ms, base + 5 * step);
        assert_eq!(blocks.evaluation_first_origin_ms, base + 8 * step);
        assert_eq!(blocks.purge_gap_ms, 3 * step);
    }

    #[test]
    fn two_partitions_whose_targets_and_origins_overlap_are_refused_rather_than_scored() {
        let step = 300_000i64;
        let base = 1_700_000_000_000i64;
        // The last calibration origin's targets reach into the evaluation block: a gain fitted
        // on the first would be scored on bars it already saw.
        let calibration: Vec<(i64, i64)> = (0..4)
            .map(|slot| (base + slot * step, base + (slot + 6) * step))
            .collect();
        let evaluation: Vec<(i64, i64)> = (8..11)
            .map(|slot| (base + slot * step, base + (slot + 2) * step))
            .collect();
        let refusal = Blocks::spanning(&calibration, &evaluation)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("share bars"), "{refusal}");
        // An empty partition is named as the population it is, not reported as a zero-origin
        // fit: this is exactly the defect the [70%, 80%) band had.
        let refusal = Blocks::spanning(&[], &evaluation).unwrap_err().to_string();
        assert!(refusal.contains("calibration partition"), "{refusal}");
        let refusal = Blocks::spanning(&calibration, &[]).unwrap_err().to_string();
        assert!(refusal.contains("evaluation partition"), "{refusal}");
    }

    /// The refusal that stopped job 5998, and the proof that the per-ticker formulation admits
    /// it while still refusing a real overlap.
    ///
    /// Two names of different history length: `LONG` trades through the whole span, `SHORT`
    /// goes quiet across the shared boundary so its own band edges land 40 slots earlier. Each
    /// name's calibration targets complete before its OWN first evaluation origin, and the two
    /// blocks nevertheless interleave on the global wall clock - `SHORT` is scored while `LONG`
    /// is still being fitted. That is the corpus's real shape, and the pooled guard cannot pass
    /// it no matter how the corpus is drawn.
    #[test]
    fn per_ticker_blocks_that_interleave_globally_are_proven_rather_than_refused() {
        let step = 300_000i64;
        let base = 1_700_000_000_000i64;
        let at = |slot: i64| base + slot * step;
        let block = |ticker: usize, first: i64, last: i64| -> Vec<DatedOrigin> {
            (first..last)
                .map(|slot| (ticker, at(slot), at(slot + 2)))
                .collect()
        };
        // LONG: fit [100, 140), scored [155, 200). SHORT: fit [60, 100), scored [110, 130).
        let calibration = [block(0, 100, 140), block(1, 60, 100)].concat();
        let evaluation = [block(0, 155, 200), block(1, 110, 130)].concat();
        // The pooled guard compares LONG's last target (slot 141) against SHORT's first scored
        // origin (slot 110) and refuses a split in which no ticker shares a bar with itself.
        let pooled = |rows: &[DatedOrigin]| -> Vec<(i64, i64)> {
            rows.iter()
                .map(|(_, origin, target)| (*origin, *target))
                .collect()
        };
        let refusal = Blocks::spanning(&pooled(&calibration), &pooled(&evaluation))
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("share bars"), "{refusal}");
        let name = |ticker: usize| ["LONG", "SHORT"][ticker].to_owned();
        let blocks = Blocks::per_ticker(&calibration, &evaluation, 2, name).unwrap();
        assert_eq!(blocks.calibration_origins, 80);
        assert_eq!(blocks.evaluation_origins, 65);
        assert_eq!(blocks.calibration_first_origin_ms, at(60));
        assert_eq!(blocks.calibration_last_target_ms, at(141));
        assert_eq!(blocks.evaluation_first_origin_ms, at(110));
        // SHORT's own separation, 110 - 101, is tighter than LONG's 155 - 141, and the
        // smallest per-ticker gap is the only number in this pair that claims anything.
        assert_eq!(blocks.purge_gap_ms, 9 * step);

        // A ticker whose own targets reach its own scored block is still refused, by name.
        let leaking = [block(0, 100, 140), block(1, 60, 110)].concat();
        let refusal = Blocks::per_ticker(&leaking, &evaluation, 2, name)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("SHORT"), "{refusal}");
        assert!(!refusal.contains("LONG"), "{refusal}");
        assert!(refusal.contains("1 of 2 tickers"), "{refusal}");
        assert!(refusal.contains("2-step target"), "{refusal}");
        // Both dates, in a unit a human can read and in the epoch milliseconds a log grep
        // needs: the old message printed two bare integers and no ticker at all.
        assert!(refusal.contains(&iso_ms(at(111))), "{refusal}");
        assert!(refusal.contains(&format!("{}", at(110))), "{refusal}");

        // A split whose two sides share no instrument is not a scored fit either.
        let refusal = Blocks::per_ticker(&block(0, 100, 140), &block(1, 150, 200), 2, name)
            .unwrap_err()
            .to_string();
        assert!(
            refusal.contains("no ticker carries origins in both"),
            "{refusal}"
        );
    }
}
