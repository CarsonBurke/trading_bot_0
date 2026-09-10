//! The per-horizon PREDICTABLE-INFORMATION CEILING: an upper bound on the within-timestamp
//! information coefficient ANY causal forecaster can reach on this corpus at horizon `h`.
//!
//! # Why this is not a future-aware teacher
//!
//! The obvious construction - encode the realized window `t+1..t+H` through a narrow latent,
//! train it to predict the same targets, call its IC the ceiling - measures nothing, and the
//! reason is worth stating exactly once because it is the same mistake `--horizon-mean
//! basis:8:8` already paid for: **a width bottleneck is a RANK restriction, and rank is not
//! information.** A latent of width `d` that is an arbitrary function of the future window
//! can carry `y_{t,h}` itself in one coordinate; at `d = 7` it carries all seven
//! [`DECISION_HORIZONS`](super::reports::DECISION_HORIZONS) coordinates exactly. So the
//! teacher's IC is identically 1.000 at `d = 32` and at `d = 512`, by construction, before a
//! single gradient step - which is what
//! [`tests::a_future_aware_bottleneck_teacher_reaches_unit_ic_at_any_useful_width`] exhibits.
//! Narrowing the latent below 7 does not rescue it either: it only forces the teacher to
//! choose WHICH horizons to reproduce, and the 192 targets are near-duplicate cumulative sums,
//! so one coordinate already reproduces most of them.
//!
//! Bounding a future channel in BITS rather than in width would be a real construction (a
//! variational information bottleneck traces `IC(R)` against the rate `R`), but the quantity
//! that answers this project's question is `IC(R -> 0)`, which is by definition the causal
//! ceiling and is not identified by any point on the curve. It would cost a second encoder
//! pass, a rate estimator, and an extrapolation, to arrive at an estimate of exactly the
//! quantity the estimator below computes in closed form from the targets alone.
//!
//! # The construction
//!
//! Fix a horizon `h` and an evaluation timestamp `τ` with a cross-section of `n_τ` tickers.
//! The scored target is the σ-normalized market-neutral cumulative log return
//! `y_{i,h} = Σ_{j≤h} Δ_{i,j}`, where `Δ_{i,j}` is bar `τ+j`'s σ-normalized market-neutral
//! return ([`CausalPatchModel::targets`](super::model::CausalPatchModel::targets) emits the
//! cumulative form directly, so the increments are one difference along the horizon axis and
//! carry no reconstruction of their own).
//!
//! Write `Δ_{i,j} = μ_{i,j} + e_{i,j}` with `μ_{i,j} = E[Δ_{i,j} | F_τ]` the ORACLE causal
//! conditional mean under the full filtration at `τ`, so `E[e | F_τ] = 0`. Every moment below
//! is cross-sectional at fixed `τ` and then averaged over timestamps, which is exactly the
//! population the reported IC lives on. Because `μ` is `F_τ`-measurable, the law of total
//! covariance splits the observed cross-sectional covariance with no cross term:
//!
//! ```text
//! Γ(j, k) = Cov(Δ_j, Δ_k) = Cov(μ_j, μ_k) + F(j, k),   F(j, k) = E[ Cov(e_j, e_k | F_τ) ]
//! ```
//!
//! For any `F_τ`-measurable forecast `f`, `corr(f, y_h) ≤ corr(E[y_h | F_τ], y_h)`, so the
//! ceiling is `ceil_h² = Var(μ_{·,h}) / V_h` with `V_h = Var(y_h)`, and
//!
//! ```text
//! ceil_h² = 1 - ( Σ_{j≤h} F(j,j) + Σ_{j≠k≤h} F(j,k) ) / V_h.
//! ```
//!
//! That identity is assumption-free. Everything unmeasured sits in `F`. Two named inputs
//! close it:
//!
//! - **(A1) The unpredictable component is a martingale difference sequence at the bar
//!   frequency:** `F(j, k) ≥ 0` for `j ≠ k`. This is violated exactly by IN-WINDOW FEEDBACK -
//!   a shock at `τ+3` that moves the conditional mean at `τ+7`, which no forecaster standing
//!   at `τ` can know. Under momentum-shaped feedback the omitted term is positive and the
//!   estimate is a strict UPPER bound; under conditional reversal it is negative and the
//!   estimate is not a bound at all. That failure is self-announcing: it drives the estimate
//!   below zero, and this module renders a negative `ceil²` as NaN and names the horizons
//!   rather than clamping it into a number.
//! - **(A2) A stated one-bar ceiling `c₁`:** `Var(μ_j) ≤ c₁ · Γ(j,j)` at every bar. The
//!   off-diagonal structure cannot identify the diagonal - a signal that is predictable
//!   one bar ahead but carries no cross-bar structure is invisible to it - so `c₁` is supplied
//!   ([`CeilingArgs::one_bar_ceiling_ic`]) and its value is printed with the result. It must
//!   be an UPPER bound on one-bar predictability, not the measured one-bar IC.
//!
//! Under (A1) and (A2), with `D_h = Σ_{j≤h} Γ(j,j)` the summed per-bar variance and
//! `g_h = D_h / V_h` the inverse Lo-MacKinlay variance ratio,
//!
//! ```text
//! ceil_h² ≤ 1 - (1 - c₁) · g_h.
//! ```
//!
//! # What this bounds, and what it does not
//!
//! It bounds the within-timestamp IC of ANY forecast measurable with respect to information
//! available at `τ` - any architecture, any feature set, any amount of data, not merely the
//! current trunk - pooled over the whole held-out period, on the close channel. Because
//! `min MSE / Var(y) = 1 - ceil²`, it also bounds the market-neutral MSE ratio the same
//! forecaster can reach, which is the number that says how much of the MSE scoreboard is
//! reachable at all.
//!
//! It does NOT bound:
//!
//! - **A forecaster that trades a subset.** The ceiling is a pooled second moment. A policy
//!   active only in a predictable regime can exceed it on its own subsample, and nothing here
//!   says otherwise.
//! - **Anything under conditional reversal.** If in-window feedback is negative, (A1) fails in
//!   the direction that breaks the bound.
//! - **The one-bar ceiling itself.** `c₁` is an input.
//! - **Non-close channels, or rank statistics.** The construction is a Pearson second-moment
//!   argument on the close coordinate.
//! - **Anything about a specific model.** It is a property of the targets. Nothing the trunk
//!   does can move it, which is the leakage-impossibility statement and what
//!   [`tests::the_ceiling_is_a_function_of_the_targets_and_cannot_be_moved_by_any_forecast`]
//!   holds it to.
//!
//! # Leakage
//!
//! There is no fit. The estimator has no parameters, reads only the target tensor of the
//! scored population, and performs no selection, so there is no fit block whose separation
//! from the scored block could be violated: it estimates the scored population's OWN ceiling
//! rather than predicting it. The one non-target input is the scalar `c₁` from the command
//! line. The student IC it is paired against is scored on the identical retained origins in
//! the identical pass, so the difference is paired by construction rather than by alignment.

use anyhow::{ensure, Result};
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Kind, Tensor};

use super::{
    model::CHANNELS,
    runner::{FinalOrigin, CROSS_SECTION_MIN},
};

/// The report vocabulary, spelled exactly as [`super::reports`] spells it. Duplicated rather
/// than shared because those constants are private to that module and this base is written
/// here; the registry test in the TUI is what keeps the two from drifting into two different
/// families of panel.
const FULL: &str = "held-out full";

/// The axis unit. A ceiling, a realized IC and their difference are all correlation
/// coefficients, so they belong on one axis; the standard error of the paired difference is
/// in the same unit and is the only thing that says whether the gap is real.
const CEILING_UNIT: &str = "correlation coefficient (dimensionless; mean over evaluation \
                            timestamps of the within-timestamp cross-ticker correlation, the \
                            upper bound on it, and two readings of that bound in the same \
                            unit; 0 = no information, ±1 = perfect. A NaN ceiling beside a \
                            POSITIVE violation means the martingale-residual assumption broke \
                            at that horizon and nothing is stated; a NaN beside a NaN \
                            violation means the horizon was never measured. A violation of \
                            exactly 0 is the assumption HOLDING, and a break-even one-bar \
                            ceiling of exactly 0 is the student sitting below the ceiling \
                            even with zero one-bar predictability - both are measurements)";

/// The variance-ratio axis. A dimensionless ratio of two cross-sectional variances, and the
/// reading rule is the whole point: it is the SQUARE of the scale error, so a ratio of 0.35
/// is a 1.7x over-statement of the target's standard deviation, not a 0.35x one.
const VARIANCE_RATIO_UNIT: &str = "variance ratio (dimensionless; cumulative cross-sectional \
                                   target variance over what an uncorrelated accumulation of \
                                   the same bars would give; 1 = random walk and the √h shape \
                                   gain is right, below 1 = within-window mean reversion and \
                                   √h over-states the target's scale by 1/√ratio, above 1 = \
                                   momentum; NaN = no qualifying cross-section)";

/// The autocorrelation axis. Same dimensionless correlation unit as the ceiling, but a
/// different question and therefore a different panel: this one is about the TARGET's own
/// serial structure, not about how much of it any forecaster could capture.
const AUTOCORRELATION_UNIT: &str = "correlation coefficient (dimensionless; lag-1 \
                                    cross-sectional autocorrelation of the per-bar \
                                    market-neutral returns, pooled over the bars each horizon \
                                    contains and over evaluation timestamps; 0 = serially \
                                    uncorrelated, negative = reversal, ±1 = perfect; NaN at \
                                    h 1, which contains no lag)";

/// The smallest cross-section a VARIANCE ratio is worth forming on.
///
/// Deliberately far below [`CROSS_SECTION_MIN`], and the difference is not sloppiness. The
/// ceiling is a ratio of two UNBIASED cross-sectional variances pooled over timestamps, so
/// every timestamp holding two names contributes an unbiased summand and the pooled ratio is
/// consistent as the timestamp count grows. A within-timestamp CORRELATION is neither
/// unbiased nor meaningful at `n = 2`, which is why the student IC and therefore the paired
/// gap keep the project's own [`CROSS_SECTION_MIN`] floor. The two populations are reported
/// as two series, never averaged together.
const CEILING_MIN_WIDTH: f64 = 2.;

/// The DEMONSTRATED `held-out full` within-timestamp IC at `h = 1` under ANCHORED placement:
/// 0.16208, standard error 0.00839 over 245 paired cross-sections, `timexer-control-4k` step
/// 2000, weights `ccffa620…09116`, job 5516.
///
/// It is a LOWER bound on the true one-bar ceiling `c₁` - a realized forecast cannot beat the
/// bound it is measured against - and that is the whole reason it appears here rather than in
/// prose. The reported ceiling is monotone increasing in `c₁`, so wherever the break-even
/// one-bar ceiling sits BELOW this number the gap survives every admissible `c₁` and the
/// verdict stops depending on assumption (A2) at all.
///
/// The previous value, 0.0833, was the same quantity measured on the STRIDED draw, whose
/// `held-out full` cross-sections average 10.6 names. A within-timestamp correlation at
/// n ≈ 10 is attenuated by the group mean it removes, and job 5516 measured the size of that
/// attenuation at a fixed checkpoint: 2.07-2.72x across the decision horizons, 2.63x here.
/// Every number derived from 0.0833 is a lower bound through a defective draw.
pub(super) const MEASURED_ONE_BAR_IC: f64 = 0.16208;

/// `c₁`, the default assumed UPPER bound on the share of a single bar's cross-sectional
/// return variance that is predictable from information available at the origin, as an IC.
///
/// Twice [`MEASURED_ONE_BAR_IC`], i.e. a four-fold allowance in predictable VARIANCE over what
/// the shipped model already extracts at one bar. Generous on purpose: `ceil² =
/// 1 - (1 - c₁²)·g` is monotone increasing in `c₁`, so the reported curve is an upper bound
/// only if this is one.
///
/// The `h = 1` GAP under this convention is therefore `c₁ - student = student` by
/// construction, and carries no information whatsoever. It is not evidence about the one-bar
/// relationship and must never be read as such; the sensitivity sweep the run prints, and the
/// break-even one-bar ceiling per horizon, are where assumption (A2) is actually tested. The
/// h ≥ 8 gaps are unaffected by this circularity because their ceilings depend on `c₁`
/// through the measured variance ratio rather than directly.
pub(super) const DEFAULT_ONE_BAR_CEILING_IC: f64 = 0.32416;

/// Device-resident per-timestamp accumulators, `[timestamps, pred_len]` throughout.
///
/// Nine sums, all in fp64. The cross-sectional variances they form are differences of large
/// like-signed quantities at a cross-section of ten to two hundred names, and the whole result
/// is a ratio of two such differences that sits within a percent of 1; fp32 accumulation
/// across four hundred thousand origins would put the rounding error on the same order as the
/// signal the ratio is being read for.
///
/// That argument now has a number from the real corpus. `index_add_` scatters with ATOMICS on
/// CUDA, so the summation order within one timestamp bank is whatever the hardware scheduled
/// and is not reproducible even between two passes of the identical shape. Job 5556's sweep
/// scored one population twice at 256 rows and the target-only statistics these banks produce
/// differed by **4.441e-16** - two ulps of an fp64 quantity near 1, ten orders of magnitude
/// inside the five decimals anything here is ever printed to. The same atomic reordering in
/// fp32 lands near 1e-7 per bank BEFORE the cancellation described above. The nine banks are
/// held to that measured floor by `runner::FP64_SCATTER_TOLERANCE`, so a bank that silently
/// changed dtype or lost a term fails a sweep rather than a trading verdict.
pub(super) struct CeilingAccumulator {
    pred_len: i64,
    /// Dense timestamp rank of every origin, in the order the origins are scored.
    groups: Tensor,
    count: Tensor,
    sum_y: Tensor,
    sum_yy: Tensor,
    sum_d: Tensor,
    sum_dd: Tensor,
    sum_f: Tensor,
    sum_ff: Tensor,
    sum_fy: Tensor,
    /// `Σ d_j·d_{j-1}`, the lag-1 cross-product, with a structural zero in column 0. The one
    /// bank that is not needed for the ceiling: it exists so the lag-1 cross-sectional
    /// autocorrelation can be read DIRECTLY rather than inferred from an MA(1) fit to the
    /// variance ratio. A plateau level of `Q` implies `ρ₁ = (Q - 1)/2` under MA(1), and an
    /// implied `ρ₁` that the measured one does not corroborate means the reversal is not
    /// lag-1 and the MA(1) reading of the plateau is wrong.
    sum_dlag: Tensor,
    scored: i64,
    /// The retained-row tally, kept as a DEVICE scalar rather than a host `i64`.
    ///
    /// It is a sum of exact integers - `keep` is a 0/1 mask and the bank is `Int64` - so it
    /// carries the identical value the host accumulation carried. The reason it moved is not
    /// arithmetic: reading `keep.sum()` into an `i64` every batch is a device-to-host copy,
    /// which drains the launch pipeline once per batch in the middle of `accumulate` and is
    /// the one unavoidable synchronization a non-training pass used to contain. Read once, in
    /// [`Self::finish`].
    retained: Tensor,
    filled: i64,
}

impl CeilingAccumulator {
    pub(super) fn new(groups: &Tensor, group_count: usize, pred_len: i64, device: Device) -> Self {
        // `Tensor::zeros` per bank, NEVER one zero tensor cloned nine times: a
        // `shallow_clone` shares STORAGE, so every `index_add_` would land in all nine banks
        // at once and the variances would come out as differences of the same mixed sum.
        let bank = || Tensor::zeros([group_count as i64, pred_len], (Kind::Double, device));
        Self {
            pred_len,
            groups: groups.to_device(device).to_kind(Kind::Int64),
            count: Tensor::zeros([group_count as i64], (Kind::Double, device)),
            sum_y: bank(),
            sum_yy: bank(),
            sum_d: bank(),
            sum_dd: bank(),
            sum_f: bank(),
            sum_ff: bank(),
            sum_fy: bank(),
            sum_dlag: bank(),
            scored: 0,
            retained: Tensor::zeros([], (Kind::Int64, device)),
            filled: 0,
        }
    }

    /// One batch of final-origin forecasts, in the same order the group ranks were built in.
    ///
    /// A row enters only if EVERY one of its `pred_len` target bars is valid. The estimator
    /// differences the target along the horizon axis, so one missing bar corrupts two
    /// increments; and a per-horizon mask would hand each horizon a different cross-section
    /// and therefore a different denominator, which is exactly the comparison the whole
    /// per-horizon curve exists to make.
    pub(super) fn accumulate(&mut self, forecast: &FinalOrigin) {
        let rows = forecast.mask.size()[0];
        assert_eq!(forecast.mask.size()[1], self.pred_len);
        let index = self.groups.narrow(0, self.filled, rows);
        self.filled += rows;
        self.scored += rows;
        let keep = forecast.mask.prod_dim_int(1, true, Kind::Float);
        let _ = self.retained.g_add_(&keep.sum(Kind::Int64));
        let close = CHANNELS - 1;
        let y = forecast.targets.select(-1, close) * &keep;
        let f = forecast.scaled.select(-1, close) * &keep;
        // `targets` is already the CUMULATIVE σ-scaled market-neutral return from the origin,
        // so the per-bar increments are one shifted difference and never a re-derivation from
        // prices - the two would not agree bit for bit through the β and σ normalizations.
        let head = y.narrow(1, 0, 1);
        let previous = Tensor::cat(&[head.zeros_like(), y.narrow(1, 0, self.pred_len - 1)], 1);
        let d = &y - previous;
        let add = |destination: &mut Tensor, source: Tensor| {
            let _ = destination.index_add_(0, &index, &source.to_kind(Kind::Double));
        };
        add(&mut self.count, keep.squeeze_dim(1));
        add(&mut self.sum_y, y.shallow_clone());
        add(&mut self.sum_yy, y.square());
        add(&mut self.sum_d, d.shallow_clone());
        add(&mut self.sum_dd, d.square());
        add(&mut self.sum_f, f.shallow_clone());
        add(&mut self.sum_ff, f.square());
        add(&mut self.sum_fy, &f * &y);
        // The lag-1 cross-product, `d_j·d_{j-1}`, with a structural zero in column 0 because
        // there is no bar before the first. `previous_d` is one more shift of the SAME
        // difference, never a second differencing of the target.
        let previous_d = Tensor::cat(
            &[d.narrow(1, 0, 1).zeros_like(), d.narrow(1, 0, self.pred_len - 1)],
            1,
        );
        add(&mut self.sum_dlag, &d * &previous_d);
    }

    /// The curve, `one_bar_ceiling_ic` being `c₁`.
    pub(super) fn finish(&self, one_bar_ceiling_ic: f64) -> Result<CeilingCurve> {
        ensure!(
            (0. ..=1.).contains(&one_bar_ceiling_ic),
            "the one-bar ceiling is an IC and must lie in [0, 1], not {one_bar_ceiling_ic}"
        );
        let leak = 1. - one_bar_ceiling_ic * one_bar_ceiling_ic;
        let n = self.count.unsqueeze(-1);
        let unbiased = (&n - 1.).clamp_min(1.).reciprocal();
        let central = |square: &Tensor, sum: &Tensor| (square - sum * sum / &n) * &unbiased;
        let cross = |product: &Tensor, left: &Tensor, right: &Tensor| {
            (product - left * right / &n) * &unbiased
        };
        // `V_h(τ)`, `D_h(τ)` and the within-timestamp IC, one row per evaluation timestamp.
        let variance = central(&self.sum_yy, &self.sum_y);
        let forecast_variance = central(&self.sum_ff, &self.sum_f);
        let covariance = cross(&self.sum_fy, &self.sum_f, &self.sum_y);
        let per_bar = central(&self.sum_dd, &self.sum_d);
        let diagonal = per_bar.cumsum(1, Kind::Double);
        // Lag-1 cross-sectional autocovariance, per bar and then cumulated over the bars a
        // horizon contains, so it is read on exactly the population and window the variance
        // ratio is read on. Column 0 is a structural zero on both sides of the ratio.
        let shift = |value: &Tensor| {
            Tensor::cat(
                &[
                    value.narrow(1, 0, 1).zeros_like(),
                    value.narrow(1, 0, self.pred_len - 1),
                ],
                1,
            )
        };
        let lag_covariance = cross(&self.sum_dlag, &self.sum_d, &shift(&self.sum_d)).cumsum(1, Kind::Double);
        let lag_scale = (&per_bar * shift(&per_bar))
            .clamp_min(0.)
            .sqrt()
            .cumsum(1, Kind::Double);
        let ic = &covariance / (&forecast_variance * &variance).sqrt();

        let measurable = variance.gt(0.).logical_and(&diagonal.gt(0.));
        let wide = n
            .ge(CEILING_MIN_WIDTH)
            .logical_and(&measurable)
            .to_kind(Kind::Double);
        let paired = n
            .ge(CROSS_SECTION_MIN as f64)
            .logical_and(&measurable)
            .logical_and(&forecast_variance.gt(0.))
            .logical_and(&ic.isfinite())
            .to_kind(Kind::Double);

        let tally = |usable: &Tensor| usable.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        let mean = |value: &Tensor, usable: &Tensor, total: &Tensor| {
            value
                .masked_fill(&usable.eq(0.), 0.)
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
                / total
        };
        let ceiling_of = |ratio: &Tensor| {
            let square: Tensor = 1. - leak * ratio;
            square
                .clamp_min(0.)
                .sqrt()
                .masked_fill(&square.lt(0.), f64::NAN)
        };

        let paired_count = tally(&paired);
        let wide_count = tally(&wide);
        let mean_diagonal = mean(&diagonal, &paired, &paired_count);
        let mean_variance = mean(&variance, &paired, &paired_count);
        let mean_ic = mean(&ic, &paired, &paired_count);
        let ratio = &mean_diagonal / &mean_variance;
        let ceiling = ceiling_of(&ratio);
        let wide_diagonal = mean(&diagonal, &wide, &wide_count);
        let wide_variance = mean(&variance, &wide, &wide_count);
        let wide_ceiling = ceiling_of(&(&wide_diagonal / &wide_variance));
        // Measured lag-1 cross-sectional autocorrelation, pooled as a ratio of timestamp means
        // exactly as the variance ratio is, so the two are two readings of one covariance
        // structure and not two conventions.
        let lag_autocorrelation =
            mean(&lag_covariance, &wide, &wide_count) / mean(&lag_scale, &wide, &wide_count);
        // The two variance-ratio readings, both on the WIDE population because a ratio of
        // unbiased cross-sectional variances is meaningful at two names while a correlation is
        // not, so restricting them to the paired subset would discard four fifths of the
        // evidence for no reason.
        //
        // `V̄_h/D̄_h` is the Lo-MacKinlay variance ratio proper: it divides by the summed
        // per-bar variances the window actually contains, so it isolates cross-bar covariance
        // and is immune to the intraday volatility profile. `V̄_h/(h·V̄_1)` divides by the
        // FIRST bar's variance scaled by `h`, which is precisely what the decoder's
        // persistence-anchored `√h` shape gain assumes, so it is the curve that says by how
        // much that gain is wrong - and it moves with the intraday profile as well as with
        // reversal. Two questions, two series, never one number.
        let steps = Tensor::arange_start(1, self.pred_len + 1, (Kind::Double, wide_count.device()));
        let one_bar = wide_diagonal.narrow(0, 0, 1);
        let variance_ratio = &wide_variance / &wide_diagonal;
        let walk_ratio = &wide_variance / (&steps * &one_bar);

        // Delta method for the PAIRED difference, in one linear combination per horizon so the
        // ceiling's sampling error and the student IC's are correlated through the timestamps
        // they share rather than added as if they were independent draws. `θ = √(1 - leak·D̄/V̄)
        // - ρ̄` has `∂θ/∂D̄ = -leak/(2·ceil·V̄)`, `∂θ/∂V̄ = leak·D̄/(2·ceil·V̄²)` and
        // `∂θ/∂ρ̄ = -1`, and every timestamp contributes one scalar to the variance of that
        // combination.
        let scale = &ceiling * &mean_variance * 2.;
        let statistic = &diagonal * (-leak / &scale) + &variance * (leak * &mean_diagonal
            / (&scale * &mean_variance))
            - &ic;
        let gap_error = self.standard_error(&statistic, &paired, &paired_count);
        let ic_error = self.standard_error(&ic, &paired, &paired_count);

        // The two quantities that make the result readable without its documentation.
        //
        // `breakeven` inverts the construction: the one-bar ceiling `c₁` at which the student
        // would ALREADY be optimal, `c₁* = √(1 - (1 - ρ̄²)/g)`. It exists because the whole
        // curve is one monotone function of `c₁`, so a single number per horizon states the
        // entire sensitivity that a curve over `c₁` would show - and it states it in the SAME
        // unit as everything else on the axis. The reading rule is what matters: the MEASURED
        // `h = 1` IC is a LOWER bound on the true `c₁`, so wherever `c₁*` sits below it the
        // verdict "there is IC left at this horizon" holds for EVERY admissible `c₁` and is
        // no longer conditional on (A2) at all. A `c₁*` of exactly 0 is the strongest such
        // case - the student is below the ceiling even with zero one-bar predictability - and
        // is a measurement, not a gap.
        //
        // `violation` makes (A1)'s failure a number instead of a silence: `√((1 - c₁²)·g - 1)`
        // is exactly 0 wherever the martingale-residual assumption holds and positive, in the
        // same correlation unit, exactly where in-window CONDITIONAL REVERSAL has pushed the
        // construction past the point where it bounds anything. That is what separates the two
        // NaNs in the ceiling series: a NaN beside a positive violation means the assumption
        // broke at that horizon, a NaN beside a NaN violation means the horizon was never
        // measured, and this project has already been burned once by a 0 that meant "never
        // fired".
        let optimal: Tensor = 1. - (1. - mean_ic.square()) / &ratio;
        let breakeven = optimal.clamp_min(0.).sqrt();
        let excess: Tensor = &ratio * leak - 1.;
        let violation = excess.clamp_min(0.).sqrt();

        let host = |value: &Tensor| -> Result<Vec<f64>> { Ok(Vec::<f64>::try_from(value)?) };
        let martingale_ratio = host(&ratio)?;
        let variance_ratio = host(&variance_ratio)?;
        let walk_ratio = host(&walk_ratio)?;
        // Once reversal is admitted the martingale construction states nothing, and there are
        // exactly two things left to say. Both are computed here so the artifact carries them
        // beside the construction they replace rather than in a report nobody re-reads.
        //
        // (i) The ACCUMULATION bound, `ceil_h ≤ c₁·√(h·g_h)`, assumes nothing about the
        // residual's serial structure at all - only (A2). It is Cauchy-Schwarz on the
        // predictable pieces: `h` bars each carrying at most `c₁²·V̄_1` of predictable variance
        // accumulate to at most `h²·c₁²·V̄_1` when they line up perfectly. It is honest and it
        // is weak: it exceeds 1 - i.e. says nothing - for every `h > V̄_h/(c₁²·h·V̄_1)`, which
        // at `c₁ = 0.1666` and a variance ratio near 1 is every horizon past ~36.
        //
        // (ii) The PLATEAU bound, `ceil_h = √(1 - Q·(1 - c₁²)·g_h)`, restores the long end -
        // but only under a stated SECOND assumption (A1'): that the whole long-run
        // variance-ratio deficit belongs to the UNPREDICTABLE innovation, i.e. the residual's
        // own long-run variance ratio is the measured plateau `Q`. That assumption is not
        // testable from second moments of the targets - `Q_e ≥ VR_h - h·c₁²` is the only
        // data-implied bound and it goes negative by `h = 36` - so the plateau bound is
        // emitted ONLY where a plateau actually exists, and is NaN throughout when the
        // variance ratio never flattens, because then the reversal is not transient
        // microstructure and there is nothing to attribute.
        let plateau = plateau_of(&variance_ratio);
        let horizons = martingale_ratio.len();
        let accumulation_ceiling: Vec<f64> = (1..=horizons)
            .map(|horizon| {
                (one_bar_ceiling_ic * (horizon as f64 * martingale_ratio[horizon - 1]).sqrt())
                    .min(1.)
            })
            .collect();
        let plateau_ceiling: Vec<f64> = martingale_ratio
            .iter()
            .map(|g| match plateau {
                Some((quality, _)) => {
                    let square = 1. - quality * leak * g;
                    if square < 0. {
                        f64::NAN
                    } else {
                        square.sqrt()
                    }
                }
                None => f64::NAN,
            })
            .collect();
        Ok(CeilingCurve {
            ceiling: host(&ceiling)?,
            wide_ceiling: host(&wide_ceiling)?,
            student_ic: host(&mean_ic)?,
            student_ic_error: host(&ic_error)?,
            gap: host(&(&ceiling - &mean_ic))?,
            gap_error: host(&gap_error)?,
            breakeven_one_bar_ic: host(&breakeven)?,
            reversal_violation: host(&violation)?,
            accumulation_ceiling,
            plateau_ceiling,
            plateau_ratio: plateau.map(|(quality, _)| quality),
            plateau_horizon: plateau.map(|(_, horizon)| horizon),
            variance_ratio,
            walk_ratio,
            lag_autocorrelation: host(&lag_autocorrelation)?,
            martingale_ratio,
            paired_cross_sections: host(&paired_count)?,
            wide_cross_sections: host(&wide_count)?,
            one_bar_ceiling_ic,
            scored_origins: self.scored,
            retained_origins: i64::try_from(&self.retained)
                .expect("the retained tally is an Int64 scalar"),
        })
    }

    /// Standard error of the timestamp mean of `statistic`, under the same iid-timestamp
    /// approximation the scorer's own IC error uses. NaN below two contributing timestamps: a
    /// zero-width band around a mean of one draw is the strongest possible claim on the
    /// weakest possible evidence.
    fn standard_error(&self, statistic: &Tensor, usable: &Tensor, total: &Tensor) -> Tensor {
        let sum = |value: &Tensor| {
            value
                .masked_fill(&usable.eq(0.), 0.)
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
        };
        let mean = sum(statistic) / total;
        let variance = sum(&(statistic - &mean).square()) / (total - 1.).clamp_min(1.);
        (variance / total)
            .sqrt()
            .masked_fill(&total.lt(2.), f64::NAN)
    }
}

/// One ceiling measurement, per horizon, everything in correlation units except the two
/// populations and the variance ratio.
pub(super) struct CeilingCurve {
    /// `√(1 - (1 - c₁²)·D̄_h/V̄_h)` over the timestamps the paired comparison runs on.
    pub(super) ceiling: Vec<f64>,
    /// The same estimate over every timestamp holding [`CEILING_MIN_WIDTH`] names or more,
    /// which is the whole `held-out full` population rather than the subset a within-timestamp
    /// correlation is meaningful on.
    pub(super) wide_ceiling: Vec<f64>,
    pub(super) student_ic: Vec<f64>,
    pub(super) student_ic_error: Vec<f64>,
    pub(super) gap: Vec<f64>,
    pub(super) gap_error: Vec<f64>,
    /// `c₁* = √(1 - (1 - ρ̄²)/g)`: the one-bar ceiling at which the student would already sit
    /// exactly ON the ceiling at this horizon. Below the MEASURED `h = 1` IC means the gap is
    /// real for every admissible `c₁` and the verdict does not depend on (A2); exactly 0 means
    /// the student is below the ceiling even at `c₁ = 0`.
    pub(super) breakeven_one_bar_ic: Vec<f64>,
    /// `√((1 - c₁²)·g - 1)`, exactly 0 wherever the martingale-residual assumption (A1) holds
    /// and positive exactly where in-window conditional reversal has broken it. This is what
    /// distinguishes a NaN ceiling that means "the assumption failed here" from one that means
    /// "this horizon was never measured".
    pub(super) reversal_violation: Vec<f64>,
    /// `min(1, c₁·√(h·g_h))`, the ACCUMULATION bound. Assumes nothing about the residual's
    /// serial structure - only (A2) - and is therefore the only bound that survives measured
    /// in-window reversal without a further assumption. It is also weak by construction: `h`
    /// bars of at most `c₁²` predictable share accumulate to at most `h²c₁²` when they line up
    /// perfectly, so the bound reaches 1 - states nothing - at `h ≈ VR_h/c₁²`, which is about
    /// 20 bars at the MEASURED variance ratio and `c₁ = 0.1666`. Emitted so the vacuity is
    /// visible on the chart rather than asserted in prose.
    pub(super) accumulation_ceiling: Vec<f64>,
    /// `√(1 - Q·(1 - c₁²)·g_h)` under the PLATEAU assumption (A1'): the whole long-run
    /// variance-ratio deficit belongs to the unpredictable innovation, i.e. the residual's own
    /// long-run variance ratio is the fitted plateau `Q`. Inside the plateau `Q·g_h = 1`, so
    /// this bound reduces to exactly `c₁` - the long-horizon ceiling is neither larger nor
    /// smaller than the one-bar ceiling, which is the whole content of the assumption. NaN
    /// throughout when no plateau is fitted, because then the reversal is not transient and
    /// there is nothing to attribute.
    pub(super) plateau_ceiling: Vec<f64>,
    /// The FITTED plateau level `Q` - the mean of the variance ratio over the flat tail, never
    /// its last point, which is the noisiest one on the curve.
    pub(super) plateau_ratio: Option<f64>,
    /// The horizon at which the fit says the plateau is reached, `None` when none is.
    pub(super) plateau_horizon: Option<usize>,
    /// `V̄_h/D̄_h`, the Lo-MacKinlay variance ratio proper, on the WIDE population. Divides by
    /// the per-bar variances the window actually contains, so it isolates cross-bar covariance
    /// and is immune to the intraday volatility profile. 1 = random walk, below 1 = reversal.
    pub(super) variance_ratio: Vec<f64>,
    /// `V̄_h/(h·V̄_1)`, on the WIDE population. Divides by the FIRST bar's variance scaled by
    /// `h`, which is exactly what the decoder's persistence-anchored `√h` shape gain assumes,
    /// so this - and not [`Self::variance_ratio`] - is the curve that says by how much that
    /// gain is wrong. It moves with the intraday variance profile as well as with reversal,
    /// which is why both are reported and neither is called "the" variance ratio.
    pub(super) walk_ratio: Vec<f64>,
    /// Cumulative lag-1 cross-sectional autocorrelation over the bars a horizon contains,
    /// MEASURED. Its reason to exist is that a plateau at `Q` implies `ρ₁ = (Q - 1)/2` under
    /// MA(1), and an implied `ρ₁` the measured one does not corroborate means the reversal is
    /// not lag-1 and the MA(1) reading of the plateau is wrong.
    pub(super) lag_autocorrelation: Vec<f64>,
    /// `g_h = D̄_h/V̄_h`, the inverse Lo-MacKinlay variance ratio. Reported so the curve can be
    /// re-read at any other `c₁` without another pass: `ceil² = 1 - (1 - c₁²)·g`.
    pub(super) martingale_ratio: Vec<f64>,
    pub(super) paired_cross_sections: Vec<f64>,
    pub(super) wide_cross_sections: Vec<f64>,
    pub(super) one_bar_ceiling_ic: f64,
    pub(super) scored_origins: i64,
    pub(super) retained_origins: i64,
}

impl CeilingCurve {
    /// The best market-neutral MSE ratio against persistence any causal forecaster can reach,
    /// `1 - ceil²`. A different unit from the rest of the curve, so it is printed and put in
    /// the chart title rather than drawn on the correlation axis.
    pub(super) fn best_mse_ratio(&self) -> Vec<f64> {
        self.ceiling
            .iter()
            .map(|ceiling| 1. - ceiling * ceiling)
            .collect()
    }

    /// The largest absolute difference between two curves, split by whether the series passes
    /// through the MODEL, with NaN required to match NaN and a length mismatch reported as
    /// infinite rather than skipped.
    ///
    /// Returns `(targets_only, through_the_model)`.
    ///
    /// This exists for the batch-size sweep, which re-measures one population at several batch
    /// sizes, and the split is the whole point. `ceiling` and `martingale_ratio` are
    /// functionals of the TARGETS alone - no GEMM, no bf16, nothing but the nine fp64 banks
    /// receiving identical rows in a different `index_add_` grouping. Floating-point addition
    /// is not associative, so their invariance is a real claim; but it is one about fp64
    /// regrouping of ~10^5 terms, whose error is ~10^-13, so anything a reader could see means
    /// the population itself changed. **Zero is the bar and the sweep refuses a nonzero.**
    ///
    /// `student_ic` and `gap` read the model's forecast, which arrives through bf16 GEMMs
    /// whose cuBLAS kernel and split-K decomposition are chosen from the SHAPE. Requiring
    /// bit-equality there would be requiring cuBLAS to be batch-shape-invariant, which it is
    /// not and never claimed to be. Their difference is therefore reported, and judged against
    /// the curve's own standard error, rather than asserted to be zero.
    pub(super) fn max_statistic_difference(&self, other: &Self) -> (f64, f64) {
        let worst = |series: &[(&Vec<f64>, &Vec<f64>)]| {
            let mut worst = 0.0f64;
            for (left, right) in series {
                if left.len() != right.len() {
                    return f64::INFINITY;
                }
                for (a, b) in left.iter().zip(right.iter()) {
                    let difference = match (a.is_nan(), b.is_nan()) {
                        (true, true) => 0.,
                        (false, false) => (a - b).abs(),
                        _ => f64::INFINITY,
                    };
                    worst = worst.max(difference);
                }
            }
            worst
        };
        (
            worst(&[
                (&self.ceiling, &other.ceiling),
                (&self.martingale_ratio, &other.martingale_ratio),
            ]),
            worst(&[
                (&self.student_ic, &other.student_ic),
                (&self.gap, &other.gap),
            ]),
        )
    }

    /// What the horizon's ceiling actually says, as one word, so the printed line and the
    /// report never leave a NaN to be interpreted by a reader.
    pub(super) fn verdict(&self, index: usize) -> &'static str {
        if self.ceiling[index].is_finite() {
            return "measured";
        }
        if self.reversal_violation[index] > 0. {
            "in-window reversal, assumption (A1) broken, nothing stated"
        } else {
            "never measured, no qualifying cross-section"
        }
    }
}

/// The largest departure from the fitted level that still counts as flat, as a FRACTION of
/// that level.
///
/// Relative and not absolute, and the difference decides real cases. Flatness of a ratio has
/// to be judged against the ratio's own size: `1/h` is flat to 0.003 in absolute terms over
/// its last half and is nevertheless a curve still collapsing by 44% across that same tail,
/// and calling it a plateau would license attributing the whole collapse to transient
/// microstructure. At 0.15 the MEASURED tail - 0.400, 0.325, 0.347 around a level near 0.36,
/// an 11% spread - is flat, and a `1/h` decay is not flat at any onset inside half the curve.
const PLATEAU_TOLERANCE: f64 = 0.15;

/// Fit a plateau to a variance-ratio curve: the earliest onset whose whole tail sits within
/// [`PLATEAU_TOLERANCE`] of the tail's own MEAN, returning that mean and the onset horizon.
///
/// The level is the tail mean and never the last point. The measured curve reads 0.400 at
/// `h = 64`, 0.325 at 128 and 0.347 at 192 - non-monotone, i.e. the tail is already inside its
/// own noise - so a level read off `h = 192` would be an arbitrary draw from that noise and
/// would move the derived ceiling with it. A plateau must also occupy at least half the curve;
/// a "plateau" over the last few horizons of a still-falling curve is a description of the
/// noise floor, not of the process.
fn plateau_of(ratio: &[f64]) -> Option<(f64, usize)> {
    let horizons = ratio.len();
    if horizons < 4 {
        return None;
    }
    for onset in 2..=horizons / 2 {
        let tail = &ratio[onset - 1..];
        if tail.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let level = tail.iter().sum::<f64>() / tail.len() as f64;
        if level <= 0. {
            continue;
        }
        if tail
            .iter()
            .all(|value| (value - level).abs() <= PLATEAU_TOLERANCE * level)
        {
            return Some((level, onset));
        }
    }
    None
}

/// Which origins survive a subsample that drops whole TIMESTAMPS, as a mask over `stamps` in
/// the order they were given.
///
/// The policy is here rather than beside its corpus plumbing because it is the whole
/// correctness of the training draw, and it is data, not plumbing: keep every `k`-th DISTINCT
/// timestamp and then keep every origin sitting on one, so a cross-section is retained or
/// dropped ENTIRE. A strided pick over a ticker-major reference list is what this exists to
/// prevent - it keeps a scattered subset of each ticker's own timestamps, so the surviving
/// origins land on many timestamps with one name each, and every within-timestamp statistic
/// downstream then reads as blank rather than as an error. That failure has already cost this
/// project a trading verdict once.
pub(super) fn whole_timestamp_keep(stamps: &[i64], target: usize) -> Vec<bool> {
    if stamps.len() <= target {
        return vec![true; stamps.len()];
    }
    let mut distinct = stamps.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    // `distinct` is sorted, so a `step_by` over it is sorted too and membership is a binary
    // search rather than a hash set.
    let kept: Vec<i64> = distinct
        .iter()
        .step_by(stamps.len().div_ceil(target.max(1)))
        .copied()
        .collect();
    stamps
        .iter()
        .map(|stamp| kept.binary_search(stamp).is_ok())
        .collect()
}

/// A strictly NESTED ladder of whole-timestamp draws, coarsest rung first, `levels` rungs, the
/// last of which is every stamp. Rung `i` keeps every `2^(levels-1-i)`-th DISTINCT timestamp, so
/// `rung[i] ⊆ rung[i+1]` elementwise by construction: the multiples of `2s` are a subset of the
/// multiples of `s`, and both index the same sorted distinct list.
///
/// Nesting is not a convenience. A learning curve `IC(N)` is read from the DIFFERENCES between
/// its rungs, and independent draws give each rung its own sampling noise of order
/// `1/√N_timestamps` - at the coarse end that is the same size as the curvature the exponent is
/// fitted from. Nested rungs share their common mode, so the differences are determined far
/// better than the levels are, which is the whole reason the fit is possible at this budget.
///
/// The halving acts on the DISTINCT stamp set and never on the origin count. Cross-sections are
/// unequal in width, so a 2× coarser stamp set is not a 2× smaller origin count, and the
/// realized per-rung origin count - not the nominal share - is the x axis the power law must be
/// fitted against. This function does not report it; the caller counts what it received.
pub(super) fn whole_timestamp_ladder(stamps: &[i64], levels: usize) -> Vec<Vec<bool>> {
    let mut distinct = stamps.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    (0..levels)
        .map(|rung| {
            let stride = 1usize << (levels - 1 - rung);
            // `distinct` is sorted, so the strided pick is sorted too and membership is a
            // binary search rather than a hash set.
            let kept: Vec<i64> = distinct.iter().step_by(stride).copied().collect();
            stamps
                .iter()
                .map(|stamp| kept.binary_search(stamp).is_ok())
                .collect()
        })
        .collect()
}

/// The chart. Nine series, all correlation coefficients, answering one question: how much IC
/// is left at each horizon, and - since the martingale construction is now MEASURED to fail -
/// which assumption each remaining answer is bought with. Every bound carries its assumption
/// in its own label, because two ceilings on one panel with no assumption in the labels is a
/// silent lie about what is known.
pub(super) fn write_information_ceiling(
    output: &std::path::Path,
    epoch: usize,
    step: usize,
    curve: &CeilingCurve,
    tickers: usize,
) -> Result<()> {
    let horizon = curve.ceiling.len();
    ensure!(horizon > 0, "a ceiling curve needs at least one horizon");
    let line = |label: String, values: &[f64]| ReportSeries {
        label,
        values: values.iter().map(|value| *value as f32).collect(),
    };
    let last = horizon - 1;
    let series = vec![
        line(
            format!("{FULL} predictable-information ceiling IC"),
            &curve.ceiling,
        ),
        line(
            format!("{FULL} ceiling IC over every cross-section of two or more"),
            &curve.wide_ceiling,
        ),
        line(format!("{FULL} student IC"), &curve.student_ic),
        line(
            format!("{FULL} ceiling minus student IC"),
            &curve.gap,
        ),
        line(
            format!("{FULL} standard error of the paired ceiling minus student IC"),
            &curve.gap_error,
        ),
        // The (A2) sensitivity, as one number per horizon rather than a second chart over
        // `c₁`: the one-bar ceiling at which the student would already BE optimal. The whole
        // curve is monotone in `c₁`, so this single series states the entire dependence, and
        // the measured `h = 1` IC is a LOWER bound on `c₁` - so wherever this sits below it,
        // the verdict holds for every admissible `c₁` and stops being conditional at all.
        line(
            format!("{FULL} one-bar ceiling at which the student would already be optimal"),
            &curve.breakeven_one_bar_ic,
        ),
        // (A1)'s failure as a number rather than a silence, so the two NaNs are different
        // facts on the chart and not only in the documentation.
        line(
            format!("{FULL} martingale-residual assumption violation IC"),
            &curve.reversal_violation,
        ),
        // The two bounds that survive measured in-window reversal. Their assumptions are in
        // their LABELS, not in the documentation: a reader who quotes a number off this panel
        // must be unable to quote it without the condition it is true under.
        line(
            format!("{FULL} accumulation-bound ceiling IC, no serial assumption, vacuous past h 20"),
            &curve.accumulation_ceiling,
        ),
        line(
            format!(
                "{FULL} plateau-bound ceiling IC, assuming all long-run reversal is unpredictable"
            ),
            &curve.plateau_ceiling,
        ),
    ];
    write_report(
        output.join("timexer_segment_information_ceiling.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | how much within-timestamp IC is left at each horizon, and under which assumption? | {}; {} of {} {FULL} origins retained (a row needs all {horizon} target bars valid) over {:.0} paired and {:.0} wide cross-sections at h {horizon}, {tickers} tickers. THREE bounds, three assumptions: the martingale ceiling assumes serially uncorrelated unpredictable residuals and is NaN wherever the measured within-window covariance refuses it; the accumulation ceiling assumes only a one-bar ceiling of {:.4} IC and reaches 1 - states nothing - at about h 20; the plateau ceiling additionally assumes the whole long-run variance-ratio deficit is unpredictable, under which it equals the one-bar ceiling exactly inside the plateau",
                match (curve.plateau_ratio, curve.plateau_horizon) {
                    (Some(level), Some(onset)) => format!(
                        "variance ratio falls to a fitted plateau of {level:.5} from h {onset}, so the targets mean-revert inside the window and the martingale ceiling is refused"
                    ),
                    _ => "variance ratio never flattens, so no plateau is fitted and the plateau ceiling is NaN throughout".to_owned(),
                },
                curve.retained_origins,
                curve.scored_origins,
                curve.paired_cross_sections[last],
                curve.wide_cross_sections[last],
                curve.one_bar_ceiling_ic,
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(CEILING_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (1..=horizon as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// How the target's cross-sectional variance ACCUMULATES with the horizon, which is the
/// empirical replacement for the decoder's persistence-anchored `√h` shape gain.
///
/// Two curves per population and they answer different questions. `V̄_h/D̄_h` divides by the
/// per-bar variances the window actually contains, so it isolates cross-bar covariance and is
/// the reversal measurement. `V̄_h/(h·V̄_1)` divides by the first bar's variance scaled by `h`,
/// which is exactly the `√h` assumption, so it is the one that says by how much that gain is
/// wrong - and it moves with the intraday variance profile as well as with reversal. Reporting
/// only one of them would either hide the profile or attribute it to reversal.
///
/// Three populations, because the reason this curve exists is to be FROZEN into training as a
/// data-derived scale. A constant fitted on held-out data and baked into a training run leaks,
/// however mildly, so the only clean source is `training`; the licence to freeze it is the
/// three populations agreeing, not any one of them being measured well.
pub(super) fn write_variance_ratio(
    output: &std::path::Path,
    epoch: usize,
    step: usize,
    populations: &[(&str, &CeilingCurve)],
) -> Result<()> {
    let (_, first) = *populations
        .first()
        .ok_or_else(|| anyhow::anyhow!("a variance-ratio chart needs at least one population"))?;
    let horizon = first.variance_ratio.len();
    ensure!(horizon > 0, "a variance-ratio curve needs at least one horizon");
    let mut series = Vec::with_capacity(populations.len() * 2);
    for (split, curve) in populations {
        ensure!(
            curve.variance_ratio.len() == horizon,
            "every population must carry the same {horizon} horizons"
        );
        series.push(ReportSeries {
            label: format!("{split} cumulative variance over summed per-bar variance"),
            values: curve.variance_ratio.iter().map(|v| *v as f32).collect(),
        });
        series.push(ReportSeries {
            label: format!("{split} cumulative variance over the square-root-h assumption"),
            values: curve.walk_ratio.iter().map(|v| *v as f32).collect(),
        });
    }
    write_report(
        output.join("timexer_segment_variance_ratio.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | how does the market-neutral target's cross-sectional variance accumulate with the horizon? | 1 is a random walk and the decoder's √h shape gain is correct; below 1 the targets mean-revert inside the window and √h OVER-states the target's scale by the reciprocal square root. {}. Measured from the targets alone, no model fitted, so it is usable as a frozen training-time constant - fit on `training` origins, the other populations being the out-of-sample evidence that licenses freezing it",
                match (first.plateau_ratio, first.plateau_horizon) {
                    (Some(level), Some(onset)) => format!(
                        "Fitted plateau {level:.5} from h {onset} on the first population, implying the √h gain over-states the h {horizon} scale by {:.3}x",
                        level.max(f64::MIN_POSITIVE).sqrt().recip()
                    ),
                    _ => "No plateau fitted: the ratio is still moving at the longest horizon".to_owned(),
                },
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(VARIANCE_RATIO_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (1..=horizon as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// The MECHANISM behind the variance ratio, and the check on the MA(1) reading of it.
///
/// A plateau at `Q` implies `ρ₁ = (Q - 1)/2` if the reversal is pure lag-1. That implication is
/// worth nothing unless the lag-1 autocorrelation is also measured directly, because a large
/// implied `ρ₁` that the measured one does not corroborate means the reversal is spread over
/// many lags and the MA(1) reading - and every conclusion drawn from its plateau - is wrong.
/// Both series are correlation coefficients, so they share one axis by right and the
/// comparison is a glance rather than a cross-chart arithmetic exercise.
pub(super) fn write_return_autocorrelation(
    output: &std::path::Path,
    epoch: usize,
    step: usize,
    populations: &[(&str, &CeilingCurve)],
) -> Result<()> {
    let (_, first) = *populations.first().ok_or_else(|| {
        anyhow::anyhow!("an autocorrelation chart needs at least one population")
    })?;
    let horizon = first.lag_autocorrelation.len();
    ensure!(horizon > 0, "an autocorrelation curve needs at least one horizon");
    let mut series = Vec::with_capacity(populations.len() * 2);
    for (split, curve) in populations {
        series.push(ReportSeries {
            label: format!("{split} measured lag-1 cross-sectional autocorrelation"),
            values: curve.lag_autocorrelation.iter().map(|v| *v as f32).collect(),
        });
        series.push(ReportSeries {
            label: format!("{split} lag-1 autocorrelation the variance ratio implies under MA(1)"),
            values: curve
                .variance_ratio
                .iter()
                .enumerate()
                .map(|(index, ratio)| implied_lag_one(*ratio, index + 1) as f32)
                .collect(),
        });
    }
    write_report(
        output.join("timexer_segment_return_autocorrelation.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | is the measured within-window reversal a lag-1 effect? | the measured lag-1 cross-sectional autocorrelation of the per-bar market-neutral returns, cumulated over the bars each horizon contains, beside the lag-1 autocorrelation an MA(1) residual would need to produce the MEASURED variance ratio. The two agreeing means the reversal is lag-1 microstructure and the plateau may be read as one; the two diverging means it is spread over many lags and every MA(1) conclusion drawn from the plateau is void"
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(AUTOCORRELATION_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (1..=horizon as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// `ρ₁ = (VR_h - 1) / (2·(1 - 1/h))`, the lag-1 autocorrelation an MA(1) residual needs to
/// produce this variance ratio at this horizon. Undefined at `h = 1`, where the variance ratio
/// is an arithmetic identity and carries no information about any lag.
fn implied_lag_one(ratio: f64, horizon: usize) -> f64 {
    if horizon < 2 {
        return f64::NAN;
    }
    (ratio - 1.) / (2. * (1. - 1. / horizon as f64))
}

/// A future-aware encoder's latent, for the vacuity demonstration only: `width` linear
/// readouts of the REALIZED window, exactly the bottleneck a "narrow teacher" proposes.
///
/// This is not shipped machinery and never runs on the GPU path. It exists so the claim that a
/// width bottleneck bounds nothing is exhibited rather than asserted.
#[cfg(test)]
fn bottleneck_teacher(future: &Tensor, width: i64, horizons: &[i64]) -> Tensor {
    let pred_len = future.size()[1];
    // The best a width-`d` latent can do is keep `d` coordinates of the future. Keeping the
    // scored horizons themselves is optimal at those horizons and needs `horizons.len()`
    // coordinates; the rest of the width is spent on a rank-revealing projection that is
    // irrelevant to the point.
    let selector = Tensor::zeros([pred_len, width], (Kind::Float, future.device()));
    for (slot, horizon) in horizons.iter().enumerate() {
        if (slot as i64) < width {
            let _ = selector
                .get(horizon - 1)
                .get(slot as i64)
                .fill_(1.);
        }
    }
    future.matmul(&selector)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HORIZONS: i64 = 8;

    /// Within-timestamp Pearson correlation of two `[names, horizons]` blocks, per horizon.
    fn within(left: &Tensor, right: &Tensor) -> Vec<f64> {
        let centre = |t: &Tensor| t - t.mean_dim([0i64].as_slice(), true, Kind::Float);
        let (a, b) = (centre(left), centre(right));
        let numerator = (&a * &b).sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        let denominator = (a.square().sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
            * b.square().sum_dim_intlist([0i64].as_slice(), false, Kind::Double))
        .sqrt();
        Vec::<f64>::try_from(numerator / denominator).unwrap()
    }

    /// A [`FinalOrigin`] carrying nothing but the close-channel target, forecast and mask the
    /// estimator reads. The other coordinates exist because the struct has them and are never
    /// touched, so a fixture that filled them with plausible values would only assert that
    /// this module ignores them by accident rather than by construction.
    fn origin(targets: &Tensor, forecast: &Tensor, mask: &Tensor) -> FinalOrigin {
        let rows = targets.size()[0];
        let channels = |plane: &Tensor| {
            plane
                .unsqueeze(-1)
                .expand([rows, HORIZONS, CHANNELS], true)
                .contiguous()
        };
        let unused = Tensor::zeros([rows], (Kind::Float, Device::Cpu));
        FinalOrigin {
            scaled: channels(forecast),
            targets: channels(targets),
            mask: mask.shallow_clone(),
            log_scale: channels(&targets.zeros_like()),
            drift: Tensor::zeros([rows, HORIZONS, 1], (Kind::Float, Device::Cpu)),
            prices: channels(&targets.zeros_like()),
            rebased_prices: channels(&targets.zeros_like()),
            target_prices: channels(&targets.zeros_like()),
            mid: unused.shallow_clone(),
            sigma: unused,
        }
    }

    /// Build a fixture accumulator from explicit per-timestamp increment blocks, cumulating
    /// them into targets exactly as the corpus emits them.
    fn accumulate(blocks: &[(Tensor, Tensor)]) -> CeilingAccumulator {
        let names: Vec<i64> = blocks.iter().map(|(d, _)| d.size()[0]).collect();
        let ranks: Vec<i64> = names
            .iter()
            .enumerate()
            .flat_map(|(group, count)| std::iter::repeat(group as i64).take(*count as usize))
            .collect();
        let mut accumulator = CeilingAccumulator::new(
            &Tensor::from_slice(&ranks),
            blocks.len(),
            HORIZONS,
            Device::Cpu,
        );
        for (increments, forecast) in blocks {
            let targets = increments.cumsum(1, Kind::Float);
            let rows = targets.size()[0];
            accumulator.accumulate(&origin(
                &targets,
                forecast,
                &Tensor::ones([rows, HORIZONS], (Kind::Float, Device::Cpu)),
            ));
        }
        accumulator
    }

    fn noise(names: i64, seed: i64) -> Tensor {
        tch::manual_seed(seed);
        Tensor::randn([names, HORIZONS], (Kind::Float, Device::Cpu))
    }

    /// The leakage-impossibility statement, in the only direction that can matter: the ceiling
    /// is a functional of the TARGETS, so no forecast - however good, however contaminated,
    /// however much of the answer it was handed - can move it by a single bit. A ceiling a
    /// model could raise would be a ceiling the model had leaked into.
    #[test]
    fn the_ceiling_is_a_function_of_the_targets_and_cannot_be_moved_by_any_forecast() {
        let _rng = crate::torch::test_rng::exclusive();
        let names = 256;
        let increments = noise(names, 11);
        let targets = increments.cumsum(1, Kind::Float);
        // A one-bar ceiling with real slack, so the comparison is between two finite curves
        // rather than between two NaNs, which `assert_eq!` would not even see as equal.
        let one_bar = 0.9;
        let honest = accumulate(&[(increments.shallow_clone(), noise(names, 12))])
            .finish(one_bar)
            .unwrap();
        // The oracle forecast: the target itself, which is the strongest possible leak.
        let leaked = accumulate(&[(increments, targets)]).finish(one_bar).unwrap();
        assert!(honest.ceiling.iter().all(|value| value.is_finite()));
        assert_eq!(
            honest.ceiling, leaked.ceiling,
            "a forecast changed the ceiling, so the ceiling is not a property of the targets"
        );
        assert_eq!(honest.martingale_ratio, leaked.martingale_ratio);
        // And the student IC did move, so the two runs really were different measurements.
        let last = HORIZONS as usize - 1;
        assert!(leaked.student_ic[last] > 0.999);
        assert!(honest.student_ic[last].abs() < 0.5);
        // The gap against a perfect forecast is negative: the estimator bounds a CAUSAL
        // forecaster and an oracle is not one, so it is allowed to exceed the bound and the
        // series must say so rather than clamping at zero.
        assert!(leaked.gap[last] < 0.);
    }

    /// Calibration against a target with NO predictable structure beyond one bar: independent
    /// cross-sectional increments cumulated. The variance ratio is 1 in expectation, so the
    /// estimator must return exactly the one-bar ceiling it was given at every horizon - it
    /// invents nothing where nothing is there.
    ///
    /// At `h = 1` that identity is EXACT and holds on any data whatsoever, because `D_1` and
    /// `V_1` are the same number: the first target IS the first increment. A ceiling curve
    /// that does not start exactly at its own input is an arithmetic fault, not a result.
    #[test]
    fn a_martingale_target_has_a_ceiling_at_exactly_the_one_bar_input() {
        let _rng = crate::torch::test_rng::exclusive();
        // Many wide timestamps: the variance ratio is a ratio of two noisy variances, and the
        // claim is about its expectation.
        let blocks: Vec<(Tensor, Tensor)> = (0..192)
            .map(|seed| (noise(512, 100 + seed), noise(512, 900 + seed)))
            .collect();
        // 0.25 and not 0: at a one-bar ceiling of exactly zero the construction has no slack
        // at all, so half the horizons of a true martingale cross into the unidentified
        // branch on sampling noise alone - which is the estimator behaving correctly and
        // says nothing about its calibration.
        let one_bar = 0.25;
        let curve = accumulate(&blocks).finish(one_bar).unwrap();
        assert!(
            (curve.martingale_ratio[0] - 1.).abs() < 1e-9,
            "the h=1 variance ratio is 1 by construction, not {}",
            curve.martingale_ratio[0]
        );
        assert!(
            (curve.ceiling[0] - one_bar).abs() < 1e-9,
            "the h=1 ceiling is the one-bar input by construction, not {}",
            curve.ceiling[0]
        );
        for horizon in 0..HORIZONS as usize {
            assert!(
                (curve.martingale_ratio[horizon] - 1.).abs() < 0.02,
                "an independent-increment target must have a unit variance ratio, not {} at h={}",
                curve.martingale_ratio[horizon],
                horizon + 1
            );
            assert!(
                (curve.ceiling[horizon] - one_bar).abs() < 0.03,
                "ceiling {} at h={} must sit at the one-bar input {one_bar}",
                curve.ceiling[horizon],
                horizon + 1
            );
        }
    }

    /// Recovery of a PLANTED predictable component, against the closed form it is planted at.
    ///
    /// Each name carries a constant per-bar drift `s_i ~ N(0, σ_s²)` plus independent noise
    /// `N(0, σ_e²)`. The true ceiling is then `√(h·σ_s² / (h·σ_s² + σ_e²))` and the true
    /// one-bar share is `σ_s²/(σ_s² + σ_e²)`, and the estimator is algebraically exact at that
    /// input - which is the whole claim, so a fixture that only checked "it goes up with h"
    /// would prove nothing.
    #[test]
    fn a_planted_predictable_component_is_recovered_at_its_closed_form() {
        let _rng = crate::torch::test_rng::exclusive();
        let (names, signal_sd, noise_sd) = (512i64, 0.30f64, 1.0f64);
        let blocks: Vec<(Tensor, Tensor)> = (0..192)
            .map(|seed| {
                tch::manual_seed(4000 + seed);
                let drift = Tensor::randn([names, 1], (Kind::Float, Device::Cpu)) * signal_sd;
                let increments =
                    drift.expand([names, HORIZONS], true) + noise(names, 5000 + seed) * noise_sd;
                (increments, noise(names, 6000 + seed))
            })
            .collect();
        let (signal, unpredictable) = (signal_sd * signal_sd, noise_sd * noise_sd);
        let one_bar = (signal / (signal + unpredictable)).sqrt();
        let curve = accumulate(&blocks).finish(one_bar).unwrap();
        for horizon in 1..=HORIZONS as usize {
            let h = horizon as f64;
            let truth = (h * signal / (h * signal + unpredictable)).sqrt();
            assert!(
                (curve.ceiling[horizon - 1] - truth).abs() < 0.02,
                "ceiling {} at h={horizon} must recover the planted {truth}",
                curve.ceiling[horizon - 1]
            );
        }
        // And the ceiling really does climb with the horizon here, which is the shape a
        // persistent predictable component has and a martingale does not.
        assert!(curve.ceiling[HORIZONS as usize - 1] > curve.ceiling[0] + 0.1);
        // The martingale-residual assumption HOLDS on this fixture by construction, and the
        // violation series must say so with an exact 0 rather than leave it to be inferred.
        assert!(
            curve.reversal_violation.iter().all(|value| *value == 0.),
            "a fixture with no in-window feedback must report zero violation, not {:?}",
            curve.reversal_violation
        );
        assert!((0..HORIZONS as usize).all(|index| curve.verdict(index) == "measured"));
    }

    /// Conditional REVERSAL inside the window breaks the martingale-residual assumption in the
    /// direction that breaks the bound, and the estimator must refuse to state a number rather
    /// than clamp one. A ceiling of 0.0 and a ceiling that is not identified are different
    /// claims and only one of them is true here.
    #[test]
    fn in_window_reversal_reads_as_unidentified_and_never_as_a_zero_ceiling() {
        let _rng = crate::torch::test_rng::exclusive();
        let blocks: Vec<(Tensor, Tensor)> = (0..64)
            .map(|seed| {
                let base = noise(256, 7000 + seed);
                // Every bar undoes most of the previous one: strong negative lag-1
                // covariance, so the cumulative variance falls far below the summed per-bar
                // variance and `g` exceeds 1.
                let reverted = &base - base.roll([1], [1]) * 0.9;
                (reverted, noise(256, 8000 + seed))
            })
            .collect();
        let curve = accumulate(&blocks).finish(0.0833).unwrap();
        assert!(
            curve.martingale_ratio[HORIZONS as usize - 1] > 1.,
            "the fixture must actually produce a sub-unit variance ratio"
        );
        assert!(
            curve.ceiling[HORIZONS as usize - 1].is_nan(),
            "an unidentified ceiling must read NaN, not {}",
            curve.ceiling[HORIZONS as usize - 1]
        );
        assert!(curve.gap[HORIZONS as usize - 1].is_nan());
        // The (A1) failure must be a NUMBER, not a silence: a NaN ceiling beside a POSITIVE
        // violation is "the assumption broke here", and it is what distinguishes this horizon
        // from one that was simply never drawn.
        assert!(
            curve.reversal_violation[HORIZONS as usize - 1] > 0.,
            "a broken assumption must be quantified, not left as a bare NaN"
        );
        assert!(curve
            .verdict(HORIZONS as usize - 1)
            .contains("assumption (A1) broken"));
    }

    /// The break-even one-bar ceiling is the whole `c₁` sensitivity in one number, and it is
    /// exact: plant a persistent drift and hand the student that drift as its forecast. The
    /// student is then EXACTLY at the ceiling at every horizon, so `c₁*` must come back as the
    /// planted one-bar IC itself - which is what makes "the break-even sits below the measured
    /// h=1 IC, therefore the gap survives every admissible `c₁`" a reading and not a hope.
    #[test]
    fn the_break_even_one_bar_ceiling_recovers_the_input_that_would_make_the_student_optimal() {
        let _rng = crate::torch::test_rng::exclusive();
        let (names, signal_sd, noise_sd) = (512i64, 0.30f64, 1.0f64);
        let blocks: Vec<(Tensor, Tensor)> = (0..192)
            .map(|seed| {
                tch::manual_seed(4000 + seed);
                let drift = Tensor::randn([names, 1], (Kind::Float, Device::Cpu)) * signal_sd;
                let increments =
                    drift.expand([names, HORIZONS], true) + noise(names, 5000 + seed) * noise_sd;
                // The student forecasts the planted drift itself, which IS the causal
                // conditional mean up to a per-horizon positive scalar - and a within-timestamp
                // correlation is invariant to that scalar, so this student sits exactly on the
                // ceiling at every horizon.
                (increments, drift.expand([names, HORIZONS], true).contiguous())
            })
            .collect();
        let (signal, unpredictable) = (signal_sd * signal_sd, noise_sd * noise_sd);
        let one_bar = (signal / (signal + unpredictable)).sqrt();
        let curve = accumulate(&blocks).finish(one_bar).unwrap();
        for horizon in 0..HORIZONS as usize {
            assert!(
                (curve.breakeven_one_bar_ic[horizon] - one_bar).abs() < 0.02,
                "break-even {} at h={} must recover the planted one-bar IC {one_bar}",
                curve.breakeven_one_bar_ic[horizon],
                horizon + 1
            );
            // An optimal student has no gap, which is the same statement read off the other
            // series and is what makes the break-even's meaning unambiguous.
            assert!(
                curve.gap[horizon].abs() < 0.02,
                "an optimal student must show no gap, not {}",
                curve.gap[horizon]
            );
        }
    }

    /// Thin cross-sections carry the ceiling and cannot carry the paired gap, and the two
    /// populations must be visible as two counts rather than silently averaged.
    #[test]
    fn a_thin_cross_section_carries_the_ceiling_and_never_the_paired_gap() {
        let _rng = crate::torch::test_rng::exclusive();
        let thin = CROSS_SECTION_MIN - 5;
        let blocks: Vec<(Tensor, Tensor)> = (0..64)
            .map(|seed| (noise(thin, 200 + seed), noise(thin, 300 + seed)))
            .collect();
        // A one-bar ceiling with slack, so the WIDE series is finite for a reason the test
        // controls rather than by luck of the sampling noise in a unit variance ratio.
        let curve = accumulate(&blocks).finish(0.5).unwrap();
        let last = HORIZONS as usize - 1;
        assert_eq!(curve.paired_cross_sections[last], 0.);
        assert_eq!(curve.wide_cross_sections[last], 64.);
        assert!(curve.wide_ceiling[last].is_finite());
        assert!(
            curve.ceiling[last].is_nan() && curve.gap[last].is_nan(),
            "a population of zero timestamps must render NaN, never 0"
        );
        assert!(curve.gap_error[last].is_nan());
    }

    /// The reason this module measures the targets instead of training a future-aware teacher.
    ///
    /// A latent that is allowed to see the realized window reproduces the target EXACTLY at
    /// every scored horizon as soon as it is as wide as the number of scored horizons, so its
    /// IC is 1.000 and its "ceiling" is vacuous. Width is a RANK restriction, not an
    /// information one - the same confusion `--horizon-mean basis:8:8` was rejected for - and
    /// no choice of `d_model` rescues it: 512 and 32 give the identical answer, and so does 7.
    #[test]
    fn a_future_aware_bottleneck_teacher_reaches_unit_ic_at_any_useful_width() {
        let _rng = crate::torch::test_rng::exclusive();
        let scored: Vec<i64> = (1..=HORIZONS).collect();
        let future = noise(256, 77).cumsum(1, Kind::Float);
        for width in [512, 32, HORIZONS] {
            let latent = bottleneck_teacher(&future, width, &scored);
            // The teacher's forecast at horizon h is the latent coordinate that carries it.
            let forecast = latent.narrow(1, 0, HORIZONS);
            for (horizon, correlation) in within(&forecast, &future).into_iter().enumerate() {
                assert!(
                    correlation > 0.999_999,
                    "a width-{width} future-aware latent must reproduce h={} exactly, not at {correlation}",
                    horizon + 1
                );
            }
        }
    }

    /// The training draw keeps or drops whole cross-sections, and a strided pick does not.
    ///
    /// The fixture is deliberately IRREGULAR - each ticker covers a different, offset span of
    /// timestamps, exactly as real tickers do - because a regular one lets a stride pass by
    /// luck. The property asserted is the one that matters downstream: every timestamp the
    /// draw retains carries the SAME number of names it carried in the full list. The same
    /// fixture is then handed a strided pick and the assertion is that it FAILS there, so this
    /// test cannot pass by testing nothing.
    #[test]
    fn the_training_draw_keeps_whole_cross_sections_where_a_strided_pick_shreds_them() {
        let _rng = crate::torch::test_rng::exclusive();
        // Ticker-major, the layout the real reference lists use: all of ticker 0's origins,
        // then all of ticker 1's, each over its own offset span.
        let stamps: Vec<i64> = (0..11i64)
            .flat_map(|ticker| (0..40 + 7 * ticker).map(move |bar| ticker * 3 + bar))
            .collect();
        let width = |population: &[i64]| {
            let mut sorted = population.to_vec();
            sorted.sort_unstable();
            let mut widths = std::collections::BTreeMap::new();
            for stamp in sorted {
                *widths.entry(stamp).or_insert(0usize) += 1;
            }
            widths
        };
        let full = width(&stamps);
        let drawn: Vec<i64> = whole_timestamp_keep(&stamps, stamps.len() / 4)
            .into_iter()
            .zip(stamps.iter().copied())
            .filter_map(|(keep, stamp)| keep.then_some(stamp))
            .collect();
        assert!(
            !drawn.is_empty() && drawn.len() < stamps.len(),
            "the draw must actually subsample, not keep everything or nothing"
        );
        for (stamp, names) in width(&drawn) {
            assert_eq!(
                names, full[&stamp],
                "timestamp {stamp} kept {names} of its {} names, so the cross-section was cut",
                full[&stamp]
            );
        }
        // And the failure mode this exists to prevent is real on this same fixture.
        let strided: Vec<i64> = stamps.iter().copied().step_by(4).collect();
        assert!(
            width(&strided)
                .into_iter()
                .any(|(stamp, names)| names < full[&stamp]),
            "the fixture must be irregular enough that a strided pick shreds a cross-section"
        );
    }

    /// The ladder's rungs are strictly nested, coarsest first, and every rung is still a
    /// whole-timestamp draw.
    ///
    /// Nesting is the property the learning-curve fit rests on, and it is exactly the property
    /// that fails silently: a non-nested ladder produces a monotone-looking `IC(N)` whose
    /// rung-to-rung differences are noise. So it is asserted elementwise on the same irregular
    /// ticker-major fixture the draw test uses, together with the two facts that make the
    /// ladder usable - the last rung is everything, and the coarse rungs really do shrink.
    #[test]
    fn the_ladder_rungs_are_nested_coarsest_first_and_each_keeps_whole_cross_sections() {
        let _rng = crate::torch::test_rng::exclusive();
        let stamps: Vec<i64> = (0..11i64)
            .flat_map(|ticker| (0..40 + 7 * ticker).map(move |bar| ticker * 3 + bar))
            .collect();
        let mut full: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        for stamp in &stamps {
            *full.entry(*stamp).or_insert(0) += 1;
        }
        let rungs = whole_timestamp_ladder(&stamps, 5);
        assert_eq!(rungs.len(), 5);
        for (index, pair) in rungs.windows(2).enumerate() {
            for (position, (coarse, fine)) in pair[0].iter().zip(pair[1].iter()).enumerate() {
                assert!(
                    !coarse || *fine,
                    "rung {index} keeps origin {position} that rung {} drops, so the ladder is \
                     not nested",
                    index + 1
                );
            }
        }
        let counts: Vec<usize> = rungs
            .iter()
            .map(|rung| rung.iter().filter(|keep| **keep).count())
            .collect();
        assert_eq!(
            counts[4],
            stamps.len(),
            "the finest rung must be the whole population"
        );
        assert!(
            counts[0] * 4 < counts[4],
            "the coarsest rung must actually be coarse, not {counts:?}"
        );
        // Every rung is still a whole-timestamp draw, which is what a strided pick is not.
        for (index, rung) in rungs.iter().enumerate() {
            let mut kept: std::collections::BTreeMap<i64, usize> =
                std::collections::BTreeMap::new();
            for (keep, stamp) in rung.iter().zip(stamps.iter()) {
                if *keep {
                    *kept.entry(*stamp).or_insert(0) += 1;
                }
            }
            for (stamp, names) in kept {
                assert_eq!(
                    names, full[&stamp],
                    "rung {index} cut timestamp {stamp} to {names} of {}",
                    full[&stamp]
                );
            }
        }
    }

    /// The plateau is FITTED, and specifically it is not the last point of the curve.
    ///
    /// The fixture is the measured shape: a steep fall to about 0.40 by h=32 and then a flat,
    /// NON-MONOTONE tail whose last point sits above its neighbour. A level read off the last
    /// point would return 0.347; the fit must return the tail's mean and an onset inside the
    /// falling-to-flat transition, not at the end of the curve.
    #[test]
    fn the_plateau_is_the_tail_mean_and_never_the_noisiest_last_point() {
        let mut ratio = vec![1.0f64];
        for horizon in 2..=192usize {
            let bar = horizon as f64;
            // 1 + 2ρ₁(1 - 1/h) with ρ₁ = -0.325, plus a deterministic wobble in the tail so
            // the last point is not the level.
            ratio.push(1. - 0.65 * (1. - 1. / bar) + 0.03 * (bar * 0.7).sin());
        }
        let (level, onset) = plateau_of(&ratio).expect("a flattening curve must fit a plateau");
        assert!(
            (level - 0.35).abs() < 0.03,
            "the fitted level must be the tail mean near 0.35, not {level}"
        );
        assert!(
            (level - ratio[191]).abs() > 1e-6,
            "a fit that returns the last point exactly is not a fit"
        );
        assert!(
            (2..=96).contains(&onset),
            "the onset must land inside the transition, not at {onset}"
        );
        // A curve that never flattens has no plateau, and the answer is None rather than a
        // level read off wherever it happened to stop.
        let falling: Vec<f64> = (1..=192).map(|horizon| 1. / horizon as f64).collect();
        assert!(plateau_of(&falling).is_none());
    }

    /// The two reversal-admitting bounds say what they claim: the accumulation bound is exact
    /// at `h = 1`, monotone, and reaches its vacuous 1 by roughly `VR/c₁²`; the plateau bound
    /// collapses to exactly `c₁` wherever the variance ratio sits at the plateau, which is the
    /// entire content of assumption (A1') and the whole reason the long end is readable again.
    #[test]
    fn the_plateau_bound_is_exactly_the_one_bar_ceiling_inside_the_plateau() {
        let _rng = crate::torch::test_rng::exclusive();
        let (names, one_bar) = (256i64, 0.1666f64);
        // An MA(1) residual: every bar undoes a fixed share of the previous one, which is the
        // shape a variance ratio well below 1 has. The shift is zero-PADDED and never
        // `Tensor::roll`, because a circular shift makes the full-length sum telescope exactly
        // and puts an artefact in the one horizon the plateau is read at.
        let blocks: Vec<(Tensor, Tensor)> = (0..96)
            .map(|seed| {
                let base = noise(names, 21_000 + seed);
                let lagged = Tensor::cat(
                    &[
                        base.narrow(1, 0, 1).zeros_like(),
                        base.narrow(1, 0, HORIZONS - 1),
                    ],
                    1,
                );
                (&base - lagged * 0.35, noise(names, 22_000 + seed))
            })
            .collect();
        let curve = accumulate(&blocks).finish(one_bar).unwrap();
        let (level, onset) = (
            curve.plateau_ratio.expect("a reverting fixture plateaus"),
            curve.plateau_horizon.expect("a reverting fixture plateaus"),
        );
        assert!(
            level < 0.95,
            "the fixture must actually revert, not sit at {level}"
        );
        // Inside the plateau `Q·g = 1`, so the bound is `c₁` exactly. Checked at the horizons
        // whose own ratio is at the level, which is what "inside the plateau" means.
        for horizon in onset..curve.plateau_ceiling.len() {
            if (curve.variance_ratio[horizon] - level).abs() > 0.005 {
                continue;
            }
            assert!(
                (curve.plateau_ceiling[horizon] - one_bar).abs() < 0.01,
                "at h={} the plateau bound must be the one-bar ceiling {one_bar}, not {}",
                horizon + 1,
                curve.plateau_ceiling[horizon]
            );
        }
        // The accumulation bound is exact at one bar, climbs like `√h/√VR`, and reaches its
        // vacuous 1 at `h ≈ VR/c₁²`. On this eight-horizon fixture that crossing sits near
        // h 16, so vacuity is exercised where it can be: at `c₁ = 0.5` on the SAME sums, where
        // the crossing moves inside the fixture and the clamp has to fire.
        assert!((curve.accumulation_ceiling[0] - one_bar).abs() < 1e-9);
        let last = curve.accumulation_ceiling.len() - 1;
        assert!(
            curve
                .accumulation_ceiling
                .windows(2)
                .all(|pair| pair[1] > pair[0]),
            "the accumulation bound must degrade monotonically with the horizon"
        );
        assert!(
            curve.accumulation_ceiling[last] > 4. * one_bar,
            "eight bars must already burn most of the bound's range, not {}",
            curve.accumulation_ceiling[last]
        );
        let generous = accumulate(&blocks).finish(0.5).unwrap();
        assert_eq!(
            generous.accumulation_ceiling[last], 1.,
            "past its crossing the bound must read exactly 1 and state nothing"
        );
    }
}
