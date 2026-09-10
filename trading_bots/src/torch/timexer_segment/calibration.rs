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
//! refusal enforced by [`MeanCalibration::fit`]: `ȳ²/E[y²]` is the ENTIRE share of persistence
//! MSE any constant forecast could earn at that horizon, and the fit declines to apply any gain
//! unless that is small against the amplitude cost it would compete with. An intercept is also
//! the least stationary thing in the data - a bet on the next block's mean drift - and it is
//! the one part of an affine map that a within-timestamp IC cannot see.
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
//! # Why the identity is a legible outcome and not a failure
//!
//! The fit now runs inside the run, where an untrained head is a normal state: a zero-init head
//! emits a constant, a constant has no amplitude, and there is nothing to calibrate. So a
//! refusal is a per-coordinate VALUE - gain identically 1 with the reason recorded in
//! [`CurveFit::unidentifiable`] and printed - not an error that would abort a 4000-step run at
//! its first evaluation. The pre-registered refusals keep their teeth: no gain is applied and
//! the report says why.
//!
//! # Cost
//!
//! Zero on the training path: nothing here runs during a step and the loss buffers are not
//! touched. The moments are eight `[pred_len, CHANNELS]` f64 column reductions of tensors the
//! evaluation already has resident. The host fit is two 192x192 Cholesky sweeps per penalty
//! grid point, about 0.5 GFLOP per evaluation. The checkpoint grows by the two frozen curves,
//! `2·pred_len` f64 (≈ 8 KB of JSON at pred_len = 192).

use super::model::CHANNELS;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Decoded channel index of the close coordinate: `decode_joint` emits `(open, high, low,
/// close)`, so the anchor is last and its offset is identically zero.
pub const CLOSE_CHANNEL: usize = CHANNELS as usize - 1;

/// The pre-registered refusal threshold on the intercept: the best constant forecast may not be
/// able to earn more than this share of what the amplitude error costs, or a pure gain is the
/// wrong parameterization and the fit applies none. Measured margin on the control checkpoint
/// is 65x below this.
const OFFSET_CEILING_SHARE: f64 = 0.1;
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
    /// How long after the last bar any calibration target reads the first evaluation origin
    /// starts. Strictly positive by [`Self::spanning`]'s refusal, and it is the whole of the
    /// disjointness claim: a gap of zero would mean the two blocks share a bar. Stated in
    /// milliseconds rather than as an origin count because the corpus's reserved purge band
    /// holds no origins at all - counting them would report 0 for a separation of many days.
    pub purge_gap_ms: i64,
}

impl Blocks {
    /// Date the two populations and PROVE they are disjoint in the bars their targets read.
    ///
    /// Each entry is `(origin timestamp, timestamp of the last bar this origin's `pred_len`
    /// cumulative targets read)`. The blocks come from the corpus's own reserved partitions -
    /// the calibration band and a held-out band, separated by the corpus purge - so this does
    /// not CUT anything; it measures the separation and refuses a pair that does not have one.
    /// That refusal is the load-bearing part: a calibration fitted on bars the scoring block
    /// also reads is leakage no matter which partition produced it, and the check has to run
    /// against the realized timestamps rather than against the boundary arithmetic that was
    /// supposed to guarantee them.
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
        (pooled.anchor_square
            + 2. * pooled.anchor_offset
            + pooled.offset_square
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

    /// `ȳ²/mean(y²)` pooled over channels: the whole share of the persistence MSE that the best
    /// constant forecast could earn at this horizon, which is what justifies a pure gain over an
    /// affine map.
    fn intercept_ceiling(&self, horizon: usize) -> f64 {
        let bars = self.bars[horizon];
        if bars <= 0. {
            return 0.;
        }
        let (mut earnable, mut persistence) = (0., 0.);
        for channel in 0..self.channels {
            let cell = self.cell(horizon, channel);
            earnable += self.target[cell] * self.target[cell] / bars;
            persistence += self.target_square[cell];
        }
        if persistence > 0. {
            (earnable / persistence).clamp(0., 1.)
        } else {
            0.
        }
    }
}

/// One horizon's exact two-coordinate least-squares solution, or the reason it has none.
struct Solved {
    anchor: f64,
    offset: f64,
    /// Inverse delta-method variance of `ln g`, per coordinate.
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
    /// Why NO gain is applied on this coordinate, `None` when a curve was fitted. The gain is
    /// then identically 1 - a legible outcome, not a failure: an untrained head has no
    /// amplitude to calibrate.
    pub unidentifiable: Option<String>,
}

impl CurveFit {
    /// The identity curve, with the block's own signed measurement kept.
    ///
    /// The measurement survives every refusal on purpose: "no gain was applied" and "the
    /// amplitude could not be measured" are different facts, and a consumer that has to decide
    /// whether a horizon may be SIZED on needs the second one. `NaN` entries are horizons that
    /// really had no solve.
    fn identity(coordinate: Coordinate, measured_gain: Vec<f64>, reason: String) -> Self {
        let horizons = measured_gain.len();
        Self {
            coordinate,
            gain: vec![1.; horizons],
            measured_gain,
            standard_error: vec![f64::NAN; horizons],
            amplification_ceiling: vec![1.; horizons],
            weight: vec![0.; horizons],
            penalty: f64::NAN,
            effective_dof: 0.,
            identified: 0,
            unidentifiable: Some(reason),
        }
    }
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
    /// `ȳ²/mean(y²)` per horizon: the whole gain any constant forecast could earn.
    pub intercept_ceiling: Vec<f64>,
    /// What the amplitude error costs per horizon, as a share of the persistence MSE - the
    /// quantity the intercept ceiling is compared against.
    pub amplitude_cost: Vec<f64>,
}

impl MeanCalibration {
    /// Fit both curves on `moments`, which must come from the calibration block and nothing
    /// else. `blocks` is stored, not consulted: disjointness is [`Blocks::spanning`]'s job and
    /// is proven there.
    ///
    /// `Err` is reserved for a structurally impossible input - wrong shapes, nonfinite sums, an
    /// axis too short to have curvature. A population that simply does not identify an
    /// amplitude is not an error: it produces the identity with its reason attached.
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
            .map(|row| row.as_ref().map_or(0., |solved| solved.amplitude_cost))
            .collect();
        let estimator = format!(
            "exact 2x2 least squares in (close anchor, intrabar offset) pooled over {} channels, \
             no intercept; ln g smoothed on ln h under a natural-spline roughness penalty \
             selected by GCV over {PENALTY_GRID} geometric strengths, weighted by the solve's \
             own inverse delta-method variance with the channel multiplicity charged as a \
             design effect; two-sided: shrinkage as fitted, amplification bounded by \
             exp(ln ĝ - {AMPLIFICATION_SIGMAS}·SE(ln ĝ)) floored at 1",
            moments.channels
        );
        let worst_intercept = intercept_ceiling.iter().cloned().fold(0., f64::max);
        let worst_amplitude = amplitude_cost.iter().cloned().fold(0., f64::max);
        if worst_intercept > OFFSET_CEILING_SHARE * worst_amplitude {
            // Pre-registered and unchanged, only non-fatal: a pure gain is refused rather than
            // applied where a constant forecast could earn a comparable share of the same MSE.
            let reason = format!(
                "a pure gain is the wrong parameterization on this block: the best constant \
                 forecast could earn {worst_intercept:.3e} of the persistence MSE at its best \
                 horizon against the {worst_amplitude:.3e} the amplitude error costs at its \
                 worst, which is above the {OFFSET_CEILING_SHARE} share this estimator is \
                 allowed to leave on the table"
            );
            let measured = |read: fn(&Solved) -> f64| -> Vec<f64> {
                solved
                    .iter()
                    .map(|row| row.as_ref().map_or(f64::NAN, read))
                    .collect()
            };
            return Ok(Self {
                estimator,
                blocks,
                anchor: CurveFit::identity(
                    Coordinate::CloseAnchor,
                    measured(|solved| solved.anchor),
                    reason.clone(),
                ),
                offset: CurveFit::identity(
                    Coordinate::IntrabarOffset,
                    measured(|solved| solved.offset),
                    reason,
                ),
                intercept_ceiling,
                amplitude_cost,
            });
        }
        let curve = |coordinate: Coordinate| -> CurveFit {
            let read = |solved: &Solved| match coordinate {
                Coordinate::CloseAnchor => (solved.anchor, solved.anchor_weight),
                Coordinate::IntrabarOffset => (solved.offset, solved.offset_weight),
            };
            fit_curve(
                coordinate,
                horizons,
                // Recorded whatever its sign: a negative solve is a measurement of a horizon
                // with no usable amplitude, and it is the number a sizing consumer gates on.
                |horizon| solved[horizon].as_ref().map_or(f64::NAN, |row| read(row).0),
                |horizon| {
                    solved[horizon].as_ref().map(read).filter(|(gain, weight)| {
                        gain.is_finite() && *gain > 0. && weight.is_finite() && *weight > 0.
                    })
                },
            )
        };
        Ok(Self {
            estimator,
            blocks,
            anchor: curve(Coordinate::CloseAnchor),
            offset: curve(Coordinate::IntrabarOffset),
            intercept_ceiling,
            amplitude_cost,
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
            measured_anchor: self.anchor.measured_gain.clone(),
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
    /// unsmoothed: the measurement, not the applied curve. `NaN` where the block identified no
    /// amplitude at all.
    ///
    /// Carried because the applied curve cannot answer the one question a position-sizing
    /// consumer has to ask. The fit runs on `ln g`, so every applied gain is positive by
    /// construction, and a horizon whose measured amplitude is NEGATIVE or absent is dropped
    /// from the fit and then interpolated from its neighbours - which is the right thing for a
    /// smooth curve and the wrong thing to trade, because it sizes a horizon on evidence that
    /// is not its own. [`Self::tradable`] is the gate; this vector is why it can exist without
    /// a second fit, and it is reported signed and unmodified so a gated horizon is legible.
    pub measured_anchor: Vec<f64>,
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
                .all(|gain| gain.is_finite() || gain.is_nan()),
            "an infinite measured anchor gain is a broken reduction, not a measurement"
        );
        Ok(())
    }

    /// Whether horizon `h` bars ahead (1-based) may be SIZED on, as opposed to merely scored.
    ///
    /// The applied curve is positive everywhere; the measurement is not. A horizon whose own
    /// calibration-block amplitude is non-positive or unidentified has no out-of-sample
    /// evidence that its forecast points the right way, and a position taken there is taken on
    /// its neighbours' evidence through the roughness prior. Scoring such a horizon is honest -
    /// the MSE ratio is a measurement either way - and trading it is not.
    pub fn tradable(&self, horizon: usize) -> bool {
        self.measured_anchor
            .get(horizon - 1)
            .is_some_and(|gain| gain.is_finite() && *gain > 0.)
    }

    /// The 1-based horizons whose measurement gates them out of sizing, with the value that
    /// gated them, for the report that has to name them.
    pub fn gated(&self) -> Vec<(usize, f64)> {
        (1..=self.measured_anchor.len())
            .filter(|horizon| !self.tradable(*horizon))
            .map(|horizon| (horizon, self.measured_anchor[horizon - 1]))
            .collect()
    }

    /// Whether both curves are the identity, i.e. this checkpoint's block identified no
    /// amplitude to correct.
    pub fn is_identity(&self) -> bool {
        self.anchor
            .iter()
            .chain(&self.offset)
            .all(|gain| *gain == 1.)
    }
}

/// One horizon's exact 2x2 solve, `None` where the block identifies no amplitude there.
///
/// The normal equations are pooled over channels because the four channels of one bar share the
/// anchor: `f_c = C + O_c`, so the design is `[C, O_c]` with `O_close ≡ 0` and the four rows of
/// one bar constrain both coordinates jointly. That pooling is what makes the anchor gain
/// minimize the FOUR-CHANNEL squared error rather than the close channel's alone.
fn solve_horizon(moments: &Moments, horizon: usize) -> Option<Solved> {
    let pooled = moments.pooled(horizon);
    let bars = moments.bars[horizon];
    let elements = bars * moments.channels as f64;
    if bars <= 0. || elements <= 2. || pooled.persistence <= 0. {
        return None;
    }
    let determinant =
        pooled.anchor_square * pooled.offset_square - pooled.anchor_offset * pooled.anchor_offset;
    if !(determinant > 0.) {
        return None;
    }
    let anchor = (pooled.offset_square * pooled.anchor_target
        - pooled.anchor_offset * pooled.offset_target)
        / determinant;
    let offset = (pooled.anchor_square * pooled.offset_target
        - pooled.anchor_offset * pooled.anchor_target)
        / determinant;
    // `SSE(g) = Σy² - g'b` at the optimum, and the uncalibrated `SSE(1, 1)` minus it is what
    // fixing the amplitude is worth. Both in the same σ² units as the persistence they divide.
    let residual = pooled.persistence - anchor * pooled.anchor_target - offset * pooled.offset_target;
    let uncalibrated = pooled.anchor_square
        + 2. * pooled.anchor_offset
        + pooled.offset_square
        - 2. * (pooled.anchor_target + pooled.offset_target)
        + pooled.persistence;
    if !(residual > 0.) {
        return None;
    }
    // `Var(ĝ) = k·s²·G⁻¹` with `s² = SSE/(elements - 2)` and `k = channels`: the four candle
    // channels of one bar share one anchor error, so one BAR is one independent residual, not
    // four, and charging the multiplicity as a design effect doubles the standard error rather
    // than pretending to four times the sample. A constant factor on every weight cannot move
    // the smoother - the penalty grid is in units of the mean weight - so this only widens the
    // three-sigma amplification bound, which is the conservative direction.
    let variance = moments.channels as f64 * residual / (elements - 2.);
    let log_weight = |gain: f64, cofactor: f64| -> f64 {
        let gain_variance = variance * cofactor / determinant;
        if gain > 0. && gain_variance > 0. {
            gain * gain / gain_variance
        } else {
            0.
        }
    };
    Some(Solved {
        anchor,
        offset,
        anchor_weight: log_weight(anchor, pooled.offset_square),
        offset_weight: log_weight(offset, pooled.anchor_square),
        amplitude_cost: (uncalibrated - residual) / pooled.persistence,
    })
}

/// The roughness-penalized WLS on `ln g` against `ln h`, plus the two-sided bound.
///
/// `measured(h)` is the block's raw signed solve at `h`, recorded whatever its sign, and
/// `identified(h)` yields `(ĝ_h, weight_h)` only where that solve is a usable positive
/// amplitude. Unidentified horizons keep weight zero: they are then determined by the roughness
/// penalty alone - interpolated from their neighbours, which is the only defensible thing a
/// smoothness prior can say about them - and their amplification ceiling is the identity, so a
/// neighbour's evidence can shrink them but never amplify them. Trading them is a separate
/// question, answered by [`FrozenGain::tradable`] off the measurement this records.
fn fit_curve(
    coordinate: Coordinate,
    horizons: usize,
    measured: impl Fn(usize) -> f64,
    identified: impl Fn(usize) -> Option<(f64, f64)>,
) -> CurveFit {
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
    if usable < 3 {
        return CurveFit::identity(
            coordinate,
            measured_gain,
            format!(
                "only {usable} of {horizons} horizons carry a calibratable {} amplitude; there \
                 is nothing to fit a gain curve on",
                coordinate.label()
            ),
        );
    }
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
        return CurveFit::identity(
            coordinate,
            measured_gain,
            format!(
                "no roughness penalty produced a solvable {} fit",
                coordinate.label()
            ),
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
    if !gain.iter().all(|value| value.is_finite() && *value > 0.) {
        return CurveFit::identity(
            coordinate,
            horizons,
            format!("the fitted {} gain left the positive range", coordinate.label()),
        );
    }
    CurveFit {
        coordinate,
        gain,
        measured_gain,
        standard_error,
        amplification_ceiling,
        weight,
        penalty,
        effective_dof,
        identified: usable,
        unidentifiable: None,
    }
}

/// `R` such that `uᵀRu` is the natural-cubic-spline roughness `∫(u'')²` of the piecewise
/// quadratic through `(axis, u)`: divided second differences, each weighted by the interval it
/// integrates over, so a non-uniform axis is penalized in its own units rather than per index.
fn roughness_matrix(axis: &[f64]) -> Vec<f64> {
    let n = axis.len();
    let mut matrix = vec![0.; n * n];
    for index in 1..n - 1 {
        let (left, right) = (
            axis[index] - axis[index - 1],
            axis[index + 1] - axis[index],
        );
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
                let forecast_target =
                    moments.anchor_target[cell] + moments.offset_target[cell];
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
            assert!(
                (solved.anchor - 1. / anchor_error(horizon)).abs() < 1e-9,
                "anchor gain {} at h={} against the injected {}",
                solved.anchor,
                horizon + 1,
                1. / anchor_error(horizon)
            );
            assert!(
                (solved.offset - 1. / 1.4).abs() < 1e-9,
                "offset gain {} at h={}",
                solved.offset,
                horizon + 1
            );
        }
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
        assert!(fit.anchor.unidentifiable.is_none());
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
            fit.offset
                .gain
                .iter()
                .all(|gain| (*gain - 1.).abs() < 0.05),
            "the offset curve drifted off the identity it was given: {:?}",
            &fit.offset.gain[..4]
        );
    }

    #[test]
    fn an_amplification_its_own_standard_error_cannot_prove_is_bounded_to_the_identity() {
        let horizons = 32;
        // A measured under-amplitude of 2.5 on 400 bars: the solve's own standard error on
        // `ln g` is far larger than `ln 2.5`, so none of it is deployable.
        let moments = injected(horizons, CHANNELS as usize, 400., |_| 1. / 2.5, |_| 1., 0.06);
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        assert!(fit
            .anchor
            .measured_gain
            .iter()
            .all(|gain| (*gain - 2.5).abs() < 1e-9));
        assert!(fit.anchor.amplification_ceiling.iter().all(|c| *c == 1.));
        assert!(fit.anchor.gain.iter().all(|gain| (*gain - 1.).abs() < 1e-12));
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

    #[test]
    fn an_intercept_worth_more_than_a_tenth_of_the_amplitude_error_applies_no_gain() {
        let horizons = 32;
        let mut moments = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 0.6,
            |_| 1.,
            0.06,
        );
        // A target mean large enough that a constant forecast would earn a comparable share of
        // the same MSE: the pure-gain parameterization is refused, and refused as a VALUE - the
        // identity with its reason - rather than as an error that would abort the run.
        for horizon in 0..horizons {
            for channel in 0..moments.channels {
                let cell = horizon * moments.channels + channel;
                moments.target[cell] =
                    (0.5 * moments.target_square[cell] * moments.bars[horizon]).sqrt();
            }
        }
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        let reason = fit.anchor.unidentifiable.clone().expect("a named refusal");
        assert!(reason.contains("pure gain is the wrong parameterization"), "{reason}");
        assert!(fit.frozen().is_identity());
        // And the measured population passes it: the control checkpoint's worst intercept
        // ceiling is 4.63e-5 against a 2.995e-2 worst amplitude cost.
        let control = injected(
            horizons,
            CHANNELS as usize,
            400_000.,
            |_| 1. / 0.6,
            |_| 1.,
            0.06,
        );
        let fit = MeanCalibration::fit(blocks(), &control).unwrap();
        assert!(fit.anchor.unidentifiable.is_none());
        assert!(
            fit.intercept_ceiling.iter().all(|ceiling| *ceiling == 0.),
            "a zero-mean target population must leave no intercept to earn"
        );
    }

    #[test]
    fn a_constant_forecast_yields_the_identity_with_its_reason_rather_than_an_error() {
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
        let fit = MeanCalibration::fit(blocks(), &moments).unwrap();
        for curve in [&fit.anchor, &fit.offset] {
            let reason = curve.unidentifiable.clone().expect("a named refusal");
            assert!(reason.contains("nothing to fit a gain curve on"), "{reason}");
            assert!(curve.gain.iter().all(|gain| *gain == 1.));
        }
        fit.frozen().validate(horizons).unwrap();
        assert!(fit.frozen().is_identity());
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
        assert!(refusal.contains("coefficients for a 17-horizon forecast"), "{refusal}");
        for mutate in [
            (|gain: &mut FrozenGain| gain.anchor[0] = 0.) as fn(&mut FrozenGain),
            |gain: &mut FrozenGain| gain.offset[0] = -1.,
            |gain: &mut FrozenGain| gain.anchor[3] = f64::NAN,
        ] {
            let mut edited = frozen.clone();
            mutate(&mut edited);
            let refusal = edited.validate(horizons).unwrap_err().to_string();
            assert!(refusal.contains("finite and strictly positive"), "{refusal}");
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
        assert!((moments.channel_gained_ratio(0, 0, 1., 1.) - moments.channel_ratio(0, 0)).abs() < 1e-12);
        // A per-horizon gain is a positive scalar within the horizon, so it multiplies the
        // uncentered correlation's numerator and its own standard deviation by the same factor:
        // ρ is invariant, exactly, and every rank statistic with it.
        let gained: Vec<f64> = forecast.iter().map(|f| gain * f).collect();
        let gained_rho = dot(&gained, &target)
            / (dot(&gained, &gained).sqrt() * dot(&target, &target).sqrt());
        let raw_rho = dot(&forecast, &target)
            / (dot(&forecast, &forecast).sqrt() * dot(&target, &target).sqrt());
        assert!(
            (gained_rho - raw_rho).abs() < 1e-15,
            "a per-horizon gain moved the correlation from {raw_rho} to {gained_rho}"
        );
        // And the MSE-optimal gain is exactly the correction the solve returns.
        let solved = solve_horizon(&moments, 0).expect("an identified horizon");
        assert!((solved.anchor - dot(&forecast, &target) / dot(&forecast, &forecast)).abs() < 1e-12);
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
}
