//! The EXPECTED-LOG-GROWTH term: the one part of the pretraining objective that is a
//! function of the quantity the strategy actually trades.
//!
//! # Why this term exists
//!
//! Run `bardist_v2` (mlq 2884) improved the traded degree of freedom's likelihood
//! monotonically for 30,000 steps while its economics decayed:
//!
//! | step  | `r` NLL  | quarter-Kelly Sharpe |
//! |-------|----------|----------------------|
//! |  7000 | -4.8616  | 5.90                 |
//! | 10364 | -4.8783  | 5.67                 |
//! | 20000 | -4.8939  | --                   |
//! | 30000 | -4.9296  | 4.96                 |
//!
//! 0.068 nats gained on `r`, 16% of the economic value lost. The arithmetic explains it.
//! Total achievable Kelly growth is `g_max = s^2 / 2` in the per-bar Sharpe `s`; at the
//! measured `s = 4.96 / sqrt(23436) = 0.0324` that is `5.25e-4` nats/bar, independently
//! confirmed twice (the cap curve peaks at +5.44 bps at 8x, and fractional-Kelly theory
//! puts quarter-Kelly at `(2c - c^2) g_max = 2.30` bps against +2.45 measured). So the
//! ENTIRE GROSS growth content of the `r` prediction is `5.25e-4` nats — 0.011% of `r`'s NLL
//! level and 0.8% of the improvement the optimizer banked. Halving the economic value
//! costs about `2e-4` nats, i.e. 0.3% of that improvement. The objective is ~10,000x
//! larger than the quantity we trade and only incidentally aligned with it, and
//! directional structure is the cheapest thing in the density, so it is learned by step
//! ~3000 and the conditional mean drifts afterwards under no meaningful constraint.
//! Corroborating: `|f*|` median rises 9.22 -> 10.69 and cap saturation 78% -> 86% while
//! the realized hit rate FALLS 0.489 -> 0.485, and the predicted tails are WIDE
//! (realized/promised 0.67x at q=0.1%), so the inflation is in the MEAN, not in sigma.
//!
//! GROSS is load-bearing in that paragraph and every figure above it is pre-cost. Read
//! against measured trading costs the strategy those numbers describe is not profitable:
//! on the SAME 256 symbol-months the bench trades, the equal-weighted one-way cost is
//! 10.620 bps impact-free and 26.351 bps all-in at 1% of ADV, against a best recalibrated
//! break-even of 4.43 bps at the 0.25x cap — a 2.4x shortfall before any impact model
//! enters. So this term is NOT justified as closing an economic gap, and nothing here
//! should be quoted as evidence that it does. It is justified as fixing a measured MODEL
//! defect: the traded conditional mean's Mincer-Zarnowitz slope falls monotonically
//! 0.4265 -> 0.3569 across the run, 33-46 standard errors below calibration and replicated
//! on a block-disjoint slice, and its cross-sectional dispersion is ~2.1-2.8x too large.
//! A better-allocated conditional mean is worth having on its own terms, and the economic
//! gap is multiplicative rather than marginal, so it will not be closed by an objective
//! term at all.
//!
//! This module adds a term whose gradient reaches that mean.
//!
//! # The term, and the derivation of its gradient
//!
//! Per bar, from the model's own predictive law over `r` conditioned on PAST BARS ONLY
//! (see [`r_probs`] for why "past only" costs nothing to obtain):
//!
//! ```text
//! mu_hat  = sum_i p_i R_i                        (R_i = exp(d_i) - 1, simple return)
//! var_hat = sum_i p_i R_i^2 - mu_hat^2
//! f_raw   = mu_hat / (var_hat + VARIANCE_FLOOR)
//! f_hat   = clamp(f_raw, -F, +F)                 (F = trade_bench::LEVERAGE_CAP)
//! L       = -log(1 + f_hat * R_realized)
//! ```
//!
//! `R_realized` is DATA and carries no gradient, so the whole derivative flows through
//! `f_hat`:
//!
//! ```text
//! dL/dtheta = -[ R / (1 + f_hat R) ] * df_hat/dtheta
//! ```
//!
//! Taking the expectation over the true law of `R` at fixed `f_hat`,
//!
//! ```text
//! d E[L] / d f = -E[ R / (1 + f R) ]
//! ```
//!
//! which is exactly the Kelly first-order condition and vanishes precisely at the true
//! growth optimum `f*`. So `-E[log(1 + f_hat R)]` is minimized over `f_hat` at `f_hat =
//! f*`, and because `f_hat` is a differentiable function of `mu_hat` with
//! `df_raw/dmu_hat = 1/(var_hat + eps) > 0`, the gradient pushes the CONDITIONAL MEAN in
//! the direction that improves realized log growth. Nothing else in the objective does
//! that: `nll` is minimized by the whole density and is 10,000x larger.
//!
//! Note what is NOT claimed. `mu/var` is the second-order (Gaussian) Kelly optimum, not
//! the exact solve [`trade_bench::kelly_fraction`] bisects for. That is deliberate: the
//! exact solve is an iterated bisection with no useful derivative, while `mu/var` is the
//! stationary point of the same second-order expansion and shares its sign and its zero.
//! The economics are reported by the bench's exact solver either way.
//!
//! # The saturation, and why the forward and backward maps differ
//!
//! `clamp` is exactly the deployed policy — [`trade_bench`] clamps its solved `f*` at
//! [`trade_bench::LEVERAGE_CAP`] — but its derivative is zero wherever it binds, and it
//! binds on 78-86% of bars at the measured `|f*|` median of 9.22-10.69. A term that is
//! gradient-dead on five bars in six would train the minority of bars where the cap is
//! slack, which is the opposite of the intent, and on a saturated bar the loss is
//! piecewise constant in `f_raw`, so not even the SIGN of the position receives signal.
//!
//! So the forward value is the hard clamp and the backward pass uses a smooth surrogate:
//!
//! ```text
//! f_soft = F * f_raw / (F + |f_raw|)             (bounded by F, f_soft = f_raw + O(f_raw^2))
//! f_hat  = f_soft + (clamp(f_raw, -F, F) - f_soft).detach()
//! ```
//!
//! The reported loss is therefore EXACTLY the deployed policy's realized log growth, and
//! the gradient is `df_soft/df_raw = (F / (F + |f_raw|))^2 > 0`: a strictly positive,
//! bar-wise down-weighting of over-confident bars that never changes the sign of the
//! Kelly gradient. The algebraic surrogate is chosen over `F*tanh(f_raw/F)` because its
//! derivative decays as `(F/|f_raw|)^2` rather than `exp(-2|f_raw|/F)`; at the measured
//! median `|f_raw|` of ~10 with `F = 4` that is 8.2% of full weight instead of 1.3%,
//! which is the difference between an attenuated signal and no signal. That median, and
//! the 78-86% bind fraction above it, were measured on runs whose bins were priced at the
//! EDGE decode; [`GrowthSupport`] now prices them at their fitted conditional means, so
//! both figures will be re-measured by the next run rather than assumed to carry over.
//!
//! # Safety of the logarithm
//!
//! The argument is `1 + f_hat R_realized`, and `R_realized` is DATA clipped to the support's
//! BOUNDS — `lower_bounds`/`upper_bounds`, never the bin decode — so this bound is invariant
//! to what the bins are priced at. On the live 300s support those bounds are
//! `[-0.088332, +0.088038]` in log space, the largest reachable simple return is
//! `exp(0.088038) - 1 = 0.092030`, and `|f_hat R| <= 4 * 0.092030 = 0.3681` leaves the log
//! argument at or above 0.6319. [`GrowthSupport::new`] ASSERTS that bound from the actual
//! fitted support rather than trusting it, and every step checks the realized minimum
//! against [`LOG_ARGUMENT_FLOOR`]. A NaN here would poison training silently, which is the
//! failure mode this repository has hit repeatedly.

use anyhow::{ensure, Context, Result};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{
    BarEmissionHead, BarSupports, MeanDecode, BAR_CHAIN, BAR_DOF, BAR_SUPPORTS_MOMENTS_VERSION,
    DOF_R, NUM_BAR_BINS,
};

use super::trade_bench::LEVERAGE_CAP;

/// `p(r|past)` is READ DIRECTLY off the head's `r` row, because `r` heads the chain and so
/// has no prefix to integrate out. A reorder that puts any factor before `r` hands it a
/// prefix, and the direct read silently becomes a teacher-forced row that would have to be
/// marginalized again. Same invariant, same reason, as the assertion at the top of
/// [`super::trade_bench`].
const _: () = assert!(
    BAR_CHAIN[0] == DOF_R,
    "the growth term reads p(r|past) off the head's r row, which is a forecast only while \
     r is BAR_CHAIN[0]"
);

/// Weight on the expected-log-growth term, applied UNCHANGED at every step.
///
/// # Measured, not guessed
///
/// The term's own MAGNITUDE is ~5e-4 nats against `nll`'s ~4.93, so weighting it by its
/// objective share would be measuring the wrong quantity: `1.0` looks inert on that chart
/// and any weight that made it look substantial would be enormous. Its GRADIENT is not
/// small — `df_raw/dmu_hat = 1/var_hat` with `var_hat ~ 1e-5` multiplies the per-bar
/// derivative by ~1e5 before it reaches a parameter — so the weight is set from a
/// GRADIENT-NORM measurement, and sweeping is forbidden by the one-seed policy.
///
/// The measurement is `||d(growth)/dtheta|| / (||d(nll)/dtheta|| + lambda
/// ||d(growth)/dtheta||)` over every trainable parameter, taken by
/// [`super::probe_growth_gradient_share`] on the real training graph and reprinted by every
/// run at [`super::GROWTH_PROBE_STEPS`].
///
/// MEASUREMENT. Three runs on the real 5-minute corpus at the deployed seed `0x5EED` and
/// `--scoring density`, ramp stage 0 (context 896), `--steps 3200 --validate-every 0
/// --checkpoint-every 0`, run dirs `growth_probe_b4`, `growth_probe_b8`, `growth_probe_b24`.
/// `lambda for 15%` is the weight that would put the growth term at 15% of the total
/// gradient norm at that probe:
///
/// ```text
/// batch  step   ||g_nll||   ||g_growth|| at lambda=1   lambda for 15%
///     4     0    6.9811e0            2.0202e-2              60.983
///     4   200    3.7375e0            3.0042e-2              21.955
///     8     0    7.6969e0            1.0841e-2             125.296
///     8   200    1.8975e0            3.4435e-3              97.243
///    24     0    5.3744e0            7.9735e-3             118.948
///    24   200    1.2800e0            4.0496e-3              55.779
/// ```
///
/// # The ratio is NOT scale-invariant, which is why the constant is derived at batch 24
///
/// The required weight moves by up to 4.4x between batch 4 and batch 24 at the same step,
/// and not monotonically. The mechanism is visible in the columns: `||g_growth||` FALLS as
/// the batch grows (3.00e-2 -> 3.44e-3 -> 4.05e-3 at step 200) while `||g_nll||` falls far
/// less steeply. The growth term's per-bar gradients substantially CANCEL — it asks
/// different bars to move their means in opposite directions, which is exactly the
/// cross-bar allocation pressure it exists to apply — whereas `nll`'s per-bar gradients
/// mostly agree, since every bar wants a sharper density. A mean over more bars therefore
/// shrinks the growth gradient much faster than it shrinks the likelihood's.
///
/// So a weight derived at a convenient small batch would be 2-4x wrong at the real one.
/// This constant is derived at `--batch-size 24`, the deployed base batch, at ramp stage 0.
///
/// # The chosen value
///
/// At batch 24 the two probes admit a weight inside the briefed 10-20% band at BOTH, which
/// neither smaller batch does: 10-20% needs `lambda` in `[74.9, 168.5]` at step 0 and in
/// `[35.1, 79.0]` at step 200, and the intersection is `[74.9, 79.0]`. `77.0` is its
/// midpoint and measures
///
/// ```text
/// step 0   : 77 * 7.9735e-3 / (5.3744 + 77 * 7.9735e-3) = 10.3%
/// step 200 : 77 * 4.0496e-3 / (1.2800 + 77 * 4.0496e-3) = 19.6%
/// ```
///
/// Hitting the band at both ends is a stronger property than hitting 15% at one, and it is
/// the reason a single constant is defensible here at all.
///
/// # Known drift, stated rather than hidden
///
/// `||g_nll||` fell 4.2x over the first 200 steps and will keep falling for the rest of a
/// run, so this share RISES with training and will leave the band later. Two things make
/// that acceptable rather than a defect to be scheduled away. The finding this term answers
/// is that the economics decay LATE — the traded mean drifts after directional structure is
/// learned around step 3000 — so a weight whose influence grows is aimed correctly, and a
/// decaying schedule would switch the term off exactly when it is needed. And the share is
/// not unobserved: the run reprints the gradient measurement, `pretrain_growth_term` charts
/// the objective share every step, and [`super::pretrain_reports::AUX_SHARE_WARN`] warns if
/// the OBJECTIVE share crosses 25%.
///
/// The batch ramp pushes the other way: stage 1 and 2 run batch 48 and 72, and by the
/// mechanism above a larger batch lowers the growth share. The two drifts are opposite in
/// sign and neither is corrected here. If a future run's reprinted probe lands outside
/// 10-20%, the fix is to renormalize the term by its own gradient norm rather than to
/// retune this constant — that is a design change and it is deliberately not smuggled in as
/// a weight.
pub const LAMBDA_GROWTH: f64 = 77.0;

/// Floor on `var_hat` in the Kelly denominator, in units of squared simple return.
///
/// It exists only to keep a degenerate belief — a point mass, `var_hat == 0` — from
/// dividing by zero; the saturation, not this constant, is what bounds `f_hat`. A
/// realistic `var_hat` is ~1e-5, so at 1e-12 the floor is seven orders of magnitude
/// below the quantity it guards and does not shrink `f_raw` anywhere it matters.
const VARIANCE_FLOOR: f64 = 1e-12;

/// Hard lower bound the log's argument must stay above, checked every step.
///
/// The structural bound is 0.6319 (see the module docs), so 0.5 is slack by a factor of
/// 1.36 in `|f_hat R|`. It is a tripwire for a broken support or a broken clamp, not a
/// working limit, and it is an error rather than a clamp because the only healthy
/// response to it firing is to stop and look.
pub const LOG_ARGUMENT_FLOOR: f64 = 0.5;

/// Bound on the UNSATURATED Kelly fraction, applied before the smooth surrogate.
///
/// Purely a finiteness guard. `mu_hat` is an expectation over the DECODED bin returns, the
/// largest of which is 0.029014 under the fitted decode on the live 300s support, so the
/// largest `f_raw` a real belief can produce is `0.029014 / VARIANCE_FLOOR = 2.9e10` and
/// this limit never binds in training. It exists because the surrogate `F f / (F + |f|)` is
/// a ratio of two quantities that both diverge
/// with `f`: at `f_raw = inf` it evaluates to `inf / inf = NaN`, and the straight-through
/// construction would then carry that NaN into the objective. At 1e12 the surrogate's
/// derivative is already 1.6e-25, so clamping there discards nothing a gradient could
/// use.
const SURROGATE_LIMIT: f64 = 1e12;

/// Per-resolution device-resident constants of the growth term.
///
/// Built once per run per bin geometry, because a `[1, NUM_BAR_BINS]` host-to-device copy
/// on every step would be pure launch overhead and because the support bound assertion
/// belongs at construction, where it can fail before a single step has run.
///
/// # The objective prices each bin at its FITTED conditional mean
///
/// [`MeanDecode::Fitted`], asked for BY NAME. [`MeanDecode::Edge`] is and remains the tree's
/// default, because every measurement in the tree — the Mincer-Zarnowitz slopes, the bench's
/// Kelly bets, the horizon frontier, the skill deciles — was taken under it. This is the one
/// OBJECTIVE-side consumer of the decode, and an objective may not pay for an outcome the
/// corpus never realized.
///
/// `r`'s two outermost bins are open-ended: `bin_of` clamps, so bin 0 catches every move
/// below its bound and bin 127 every move above it. `BarSupports::from_bins` pins their edge
/// centers ONTO those bounds — `-883.32` and `+880.38` bps on the live 300s support — while
/// the measured conditional means there are `-281.88` and `+286.01` bps. As simple returns
/// that is `-8.4543%` / `+9.2030%` against `-2.7794%` / `+2.9014%`, so the edge decode paid
/// 3.04x and 3.17x what those two bins are worth. They hold 1.4474% of the marginal mass but
/// 92.38% of its central second moment and 41.00% of its absolute first moment, so this is
/// not a rounding choice in `mu_hat = sum_b p_b R_b`: it is most of what `mu_hat` can
/// express. Under the edge decode the cheapest route to the expected log growth
/// [`LAMBDA_GROWTH`] rewards was to move mass into the two bins that overpaid threefold —
/// an in-the-loss driver of exactly the over-dispersion the Mincer-Zarnowitz mean slope
/// reports, applied unannealed at every step.
///
/// A support carrying no measured moments makes this term UNBUILDABLE and [`Self::new`] says
/// so. It does NOT fall back: a fallback restores the pricing above in full with nothing
/// anywhere saying it did, which is the failure `MeanDecode` exists to make unrepresentable.
#[derive(Debug)]
pub struct GrowthSupport {
    /// `[1, NUM_BAR_BINS]` simple return `exp(d_b) - 1` of each `r` bin at the FITTED decode
    /// `d_b = E[r | r in bin b]`.
    ///
    /// DELIBERATELY not the same convention as [`super::trade_bench::bin_returns`], which is
    /// measurement-side and stays on [`MeanDecode::Edge`] so its numbers remain comparable
    /// with every one that came before. The objective prices a bin at what it is worth; the
    /// bench keeps the convention its history was measured under.
    returns: Tensor,
    /// `[1, NUM_BAR_BINS]`, the elementwise square, i.e. `E[R|bin]^2` and not `E[R^2|bin]`.
    ///
    /// The artifact's `bin_second_moments` are LOG-space, so no exact simple-return second
    /// moment is available to read and squaring the decode is the consistent choice. It
    /// understates the marginal second moment by ~12%, against the ~5.25x OVERstatement the
    /// edge decode produced, and the residual sits in the same two catch-alls.
    returns_sq: Tensor,
    /// Inclusive support BOUNDS of `r` in LOG space, i.e. the clamp `BarSupports::bin_ids`
    /// itself applies. Read from `lower_bounds`/`upper_bounds`, which hold the bin edges and
    /// are therefore INDEPENDENT of the decode above: the realized return is clipped to
    /// these before it enters the log, so the log-argument guard stays sound whatever the
    /// bins are priced at, and re-pricing them cannot move it.
    log_lo: f64,
    log_hi: f64,
    /// Leverage cap, from [`LEVERAGE_CAP`]. Not a second constant: training and
    /// evaluation are sized at the same cap or the two numbers are not comparable.
    cap: f64,
}

impl GrowthSupport {
    /// Errors, rather than degrading to [`MeanDecode::Edge`], on a supports artifact with no
    /// fitted per-bin moments. See the type docs for what the degradation would cost.
    pub fn new(supports: &BarSupports, device: Device) -> Result<Self> {
        let decode = supports
            .mean_decode(DOF_R, MeanDecode::Fitted)
            .with_context(|| {
                format!(
                    "the expected-log-growth objective prices every r bin at its fitted \
                     conditional mean, which only a bar supports artifact of version \
                     {BAR_SUPPORTS_MOMENTS_VERSION} or later carries. The one loaded here has \
                     none, so it predates v{BAR_SUPPORTS_MOMENTS_VERSION}: measure moments onto \
                     this exact geometry with the `bar-supports-moments` subcommand and point \
                     the run at the result. Falling back to the {} decode is PROHIBITED here — \
                     it would silently pay 3.1x for the two open-ended r bins, which is the \
                     defect this decode exists to remove",
                    MeanDecode::Edge
                )
            })?;
        let returns: Vec<f32> = decode.iter().map(|d| d.exp_m1() as f32).collect();
        ensure!(
            returns.len() == NUM_BAR_BINS as usize,
            "the r support has {} bins, expected {NUM_BAR_BINS}",
            returns.len()
        );
        let log_lo = supports.lower_bounds(DOF_R)[0];
        let log_hi = supports.upper_bounds(DOF_R)[NUM_BAR_BINS as usize - 1];
        ensure!(
            log_lo.is_finite() && log_hi.is_finite() && log_lo < log_hi,
            "the r support spans [{log_lo}, {log_hi}], which is not a usable clip range"
        );
        // The structural safety of the logarithm, asserted from the ACTUAL fitted support
        // rather than from the bounds quoted in the module docs. `1 + f R` is smallest at
        // the most negative reachable return and the largest allowed long position, and
        // largest-magnitude losses can also come from a short against the top bin. Computed
        // from the BOUNDS and not from the decode on purpose: the realized return is what
        // enters the log, and it is clipped to the bounds.
        let worst = (log_lo.exp_m1().abs()).max(log_hi.exp_m1().abs());
        ensure!(
            1.0 - LEVERAGE_CAP * worst > LOG_ARGUMENT_FLOOR,
            "at cap {LEVERAGE_CAP} the r support's extreme simple return {worst:.6} drives \
             the growth term's log argument to {:.4}, at or below the {LOG_ARGUMENT_FLOOR} \
             floor. Either the support was fitted on unclipped returns or the cap moved; do \
             not lower the floor.",
            1.0 - LEVERAGE_CAP * worst
        );
        let row = Tensor::from_slice(&returns)
            .view([1, NUM_BAR_BINS])
            .to_device(device);
        Ok(Self {
            returns_sq: &row * &row,
            returns: row,
            log_lo,
            log_hi,
            cap: LEVERAGE_CAP,
        })
    }

    /// `[1, NUM_BAR_BINS]` fitted-decode simple return of each `r` bin.
    pub fn returns(&self) -> &Tensor {
        &self.returns
    }

    pub fn cap(&self) -> f64 {
        self.cap
    }
}

/// One step's growth term: the attached scalar loss and its detached diagnostics.
#[derive(Debug)]
pub struct Growth {
    /// Mean `-log(1 + f_hat R)` over every bar of the batch, in nats per bar. Attached,
    /// with the straight-through saturation described in the module docs: the VALUE is
    /// the deployed hard-clamped policy's realized log growth.
    pub loss: Tensor,
    /// `[GROWTH_STAT_COUNT]` detached diagnostics, in the order
    /// `[mean |f_hat|, clamp-bind fraction, min log argument]`. One tensor so a step
    /// pays ONE device-to-host synchronization for all three.
    pub stats: Tensor,
}

/// Entries of [`Growth::stats`].
pub const GROWTH_STAT_COUNT: usize = 3;

/// Host-side view of [`Growth::stats`].
#[derive(Clone, Copy, Debug)]
pub struct GrowthStats {
    /// Mean `|f_hat|` under the deployed hard clamp, comparable to the bench's
    /// `quarter-Kelly mean |f|` and `|f*| median` figures.
    pub mean_abs_f: f64,
    /// Fraction of bars where `|f_raw| > F`, i.e. where the deployed clamp chose the
    /// size instead of the predictive law. 0.78-0.86 on the run that motivated the term.
    pub clamp_bind: f64,
    /// Smallest `1 + f_hat R` in the batch. The guard reads this.
    pub min_log_argument: f64,
}

impl GrowthStats {
    pub fn nan() -> Self {
        Self {
            mean_abs_f: f64::NAN,
            clamp_bind: f64::NAN,
            min_log_argument: f64::NAN,
        }
    }

    /// Reads a `[GROWTH_STAT_COUNT]` tensor in ONE synchronization.
    pub fn read(stats: &Tensor) -> Self {
        let values = Vec::<f64>::try_from(stats.to_kind(Kind::Double).reshape([-1]))
            .expect("growth stats are convertible");
        assert_eq!(
            values.len(),
            GROWTH_STAT_COUNT,
            "growth stats must carry exactly {GROWTH_STAT_COUNT} entries"
        );
        Self {
            mean_abs_f: values[0],
            clamp_bind: values[1],
            min_log_argument: values[2],
        }
    }
}

/// `[rows, NUM_BAR_BINS]` `p(r | strictly past bars)`, attached.
///
/// The emission head is `Linear([h, masked prefix embeddings]) -> NUM_BAR_BINS` per DOF,
/// and the constant `prefix_mask` keeps only the slots strictly below a DOF's chain
/// position. `r` sits at chain position 0, so it sees NO slot and its logit row is a
/// function of the latent alone. That is what makes the traded law a plain read rather
/// than a mixture, and it is why `r` was put first.
///
/// The zero prefix is not a stand-in for the realized bar: the mask discards it, so
/// passing the realized bins there would return the identical row. No realized same-bar
/// value reaches this function — the signature carries the head and the causal beliefs and
/// nothing else, the same discipline [`super::trade_bench::forecast_r_probs`] is built on.
/// That one is the same law with the graph dropped; this one stays attached because it is
/// an objective term.
pub fn r_probs(head: &BarEmissionHead, beliefs: &Tensor) -> Tensor {
    let size = beliefs.size();
    assert_eq!(size.len(), 2, "beliefs must be [rows, latent_dim]");
    let rows = size[0];
    let zero_prefix = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, beliefs.device()));
    head.logits(beliefs, &zero_prefix)
        .select(1, DOF_R as i64)
        .to_kind(Kind::Float)
        .softmax(-1, Kind::Float)
}

/// First two moments of `p(r | past)`, as `(mu_hat, second_moment)`, each `[rows]`, both
/// attached.
///
/// Two inner products over the bin axis, and nothing else: `r` heads the chain, so the law
/// handed in is already the decision law and there is no same-bar factor left to integrate
/// out. `mu_hat` is a cancelling sum — ~1e-4 against per-term magnitudes of ~1e-5 — which
/// is why [`growth_loss`] runs it with autocast disabled.
pub fn r_moments(probs: &Tensor, support: &GrowthSupport) -> (Tensor, Tensor) {
    let axis = [-1i64];
    (
        (probs * &support.returns).sum_dim_intlist(axis.as_slice(), false, Kind::Float),
        (probs * &support.returns_sq).sum_dim_intlist(axis.as_slice(), false, Kind::Float),
    )
}

/// The per-bar objective: `-log(1 + f_hat R)` at the straight-through saturated Kelly
/// fraction, plus the diagnostics.
///
/// Isolated from the belief on purpose. Retargeting the economics — growth net
/// of a turnover penalty, say — is a change to this function and to nothing else.
fn per_bar_growth(
    mu_hat: &Tensor,
    second_moment: &Tensor,
    realized_return: &Tensor,
    cap: f64,
) -> Growth {
    // `E[R^2] - E[R]^2` is not a cancelling subtraction here: `E[R^2] ~ 1e-5` against
    // `mu^2 ~ 1e-8`. The clamp only catches f32 rounding on a near-degenerate belief.
    let variance = (second_moment - mu_hat * mu_hat).clamp_min(0.0);
    // `mu_hat` is an expectation over returns bounded by the support, so the true bound
    // here is `0.0781 / VARIANCE_FLOOR = 7.8e10` and [`SURROGATE_LIMIT`] cannot bind on
    // any belief this model can hold. It binds on a corrupt input, and it must: the
    // surrogate below is a ratio whose numerator and denominator both diverge, so an
    // infinite `f_raw` would give `inf/inf = NaN` in the objective.
    let f_raw = (mu_hat / (variance + VARIANCE_FLOOR)).clamp(-SURROGATE_LIMIT, SURROGATE_LIMIT);
    let f_hard = f_raw.clamp(-cap, cap);
    // `F f / (F + |f|)`: exactly `f` to first order, bounded by `F`, derivative
    // `(F / (F + |f|))^2` which is positive everywhere and decays as a square rather
    // than exponentially. See the module docs for why the backward map is not the clamp.
    let f_soft = &f_raw * cap / (f_raw.abs() + cap);
    let f_hat = &f_soft + (&f_hard - &f_soft).detach();

    let argument = &f_hat * realized_return + 1.0;
    let loss = -argument.log().mean(Kind::Float);

    let axis = [-1i64];
    let stats = tch::no_grad(|| {
        Tensor::stack(
            &[
                f_hard.detach().abs().mean(Kind::Float),
                f_raw
                    .detach()
                    .abs()
                    .gt(cap)
                    .to_kind(Kind::Float)
                    .mean(Kind::Float),
                argument.detach().amin(axis.as_slice(), false),
            ],
            0,
        )
    });
    Growth { loss, stats }
}

/// The growth term for one training batch.
///
/// `beliefs` is `[B, T, latent_dim]` where `beliefs[b, t]` is the belief formed from bars
/// up to and including `t`, and `realized_log_r` is `[B, T]` holding the LOG return of
/// the bar each belief predicts — exactly the `(beliefs, target)` alignment the
/// teacher-forced pass already produces.
///
/// Runs with autocast DISABLED. `mu_hat = sum_i p_i R_i` is a cancelling sum whose value
/// is ~1e-4 against per-term magnitudes of ~1e-5 and a spread `E|R| ~ 2e-3`, so bf16's
/// eight mantissa bits would destroy the very quantity the term exists to calibrate.
/// In f32 the same sum carries a relative error near 1e-6.
pub fn growth_loss(
    head: &BarEmissionHead,
    beliefs: &Tensor,
    realized_log_r: &Tensor,
    support: &GrowthSupport,
) -> Growth {
    tch::autocast(false, || {
        let latent = head.latent_dim();
        let flat = beliefs.reshape([-1, latent]).to_kind(Kind::Float);
        let (mu_hat, second_moment) = r_moments(&r_probs(head, &flat), support);
        // DATA: clipped to the same support the bins are clamped onto, then converted to a
        // simple return with the same `exp_m1` convention as the bin returns. Detached
        // because a gradient into the realized bar would be a gradient into the future.
        let realized = realized_log_r
            .detach()
            .reshape([-1])
            .to_kind(Kind::Float)
            .clamp(support.log_lo, support.log_hi)
            .expm1();
        assert_eq!(
            realized.size(),
            mu_hat.size(),
            "one realized return per belief"
        );
        per_bar_growth(&mu_hat, &second_moment, &realized, support.cap)
    })
}

/// Prove, on the real device and the real head, that the `r` law this term reads is the
/// head's own PREFIX-FREE row.
///
/// Called once per run, before the first step. The property is ARCHITECTURAL — `r` is
/// [`BAR_CHAIN`]`[0]`, so chain position 0's prefix mask is identically zero and no same-bar
/// factor can enter `p(r|past)` — but it is checked rather than assumed, because the
/// failure it guards is silent: a factor placed before `r` gives it a prefix, the direct
/// read becomes a teacher-forced row, and every traded number in the tree is then
/// conditioned on the bar it is betting on.
///
/// What a once-per-run check on the real device ALSO catches, and a CPU unit test does not,
/// is a precision setting: `mu_hat` is a cancelling sum with an amplification factor of
/// ~20, so a TF32-rounded f32 reduction would silently carry a percent of error.
pub fn verify_traded_law(
    head: &BarEmissionHead,
    supports: &BarSupports,
    device: Device,
) -> Result<()> {
    ensure!(
        BAR_CHAIN[0] == DOF_R,
        "the growth term reads p(r|past) straight off the head's r row, which is a forecast \
         only while r heads BAR_CHAIN"
    );
    let support = GrowthSupport::new(supports, device)?;
    let latent = head.latent_dim();
    // Deterministic probe latents at a realistic scale: the trunk is rms-normalized, so a
    // belief has unit per-component RMS.
    let rows = 16i64;
    let probe =
        Tensor::linspace(-1.0, 1.0, rows * latent, (Kind::Float, device)).view([rows, latent]);
    let (probs, drift) = tch::no_grad(|| {
        let probs = r_probs(head, &probe);
        // Every prefix slot filled with the same non-zero bin. If any of them could reach
        // the `r` row, this moves it.
        let mut drift = 0.0f64;
        for bin in [1i64, NUM_BAR_BINS / 2, NUM_BAR_BINS - 1] {
            let prefix = Tensor::full([rows, BAR_DOF as i64], bin, (Kind::Int64, device));
            let row = head
                .logits(&probe, &prefix)
                .select(1, DOF_R as i64)
                .to_kind(Kind::Float)
                .softmax(-1, Kind::Float);
            drift = drift.max((&row - &probs).abs().max().double_value(&[]));
        }
        (probs, drift)
    });

    ensure!(
        drift == 0.0,
        "the head's r row moved by {drift:.3e} when the chain prefix was filled in, so \
         p(r|past) is teacher-forced on the bar it predicts and every traded number is \
         lookahead"
    );
    let mass = (probs.sum_dim_intlist([-1i64].as_slice(), false, Kind::Double) - 1.0)
        .abs()
        .max()
        .double_value(&[]);
    ensure!(
        mass < 1e-5,
        "the traded r law does not integrate to one ({mass:.3e} off)"
    );

    let (mu, second) = tch::no_grad(|| r_moments(&probs, &support));
    // The same two reductions in f64: same maths, different precision, so a TF32-rounded
    // f32 path fails here while an honest one agrees to a few times f32 epsilon amplified
    // by the cancellation factor. 1e-4 relative is three orders looser than that and still
    // catches TF32, which loses ~2e-2 relative on `mu_hat`.
    let exact = probs.to_kind(Kind::Double);
    let want_mu = exact
        .matmul(&support.returns.to_kind(Kind::Double).transpose(0, 1))
        .reshape([-1]);
    let want_second = exact
        .matmul(&support.returns_sq.to_kind(Kind::Double).transpose(0, 1))
        .reshape([-1]);
    for (name, got, want) in [("mu_hat", &mu, &want_mu), ("E[R^2]", &second, &want_second)] {
        let scale = want.abs().max().double_value(&[]).max(1e-12);
        let error = (got.to_kind(Kind::Double) - want)
            .abs()
            .max()
            .double_value(&[])
            / scale;
        ensure!(
            error < 1e-4,
            "the growth term's {name} disagrees with an f64 reduction of the same law by \
             {error:.3e} relative on this device. That is a numerical-precision failure, not \
             a maths one — check that f32 matmul is not running in TF32 — and the term would \
             be training on a mean it cannot measure."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::{BarDof, BAR_DOF_NAMES, DOF_S, DOF_U};
    use crate::torch::test_rng;
    use tch::nn;

    /// Bars whose `r` has real dispersion and whose `s` is genuinely informative about
    /// it, so a support fitted here has non-degenerate bins on both.
    fn synthetic_supports(count: usize, seed: u64) -> BarSupports {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let samples: Vec<BarDof> = (0..count)
            .map(|_| {
                let u = next().max(1e-9);
                let s = 0.004 * (-2.0 * u.ln()).sqrt();
                BarDof {
                    r: ((next() - 0.5) * 2.0 * s) as f32,
                    s: s as f32,
                    u: next() as f32,
                    v: next() as f32,
                    w: (next() - 0.5) as f32,
                }
            })
            .collect();
        BarSupports::fit(&samples)
    }

    /// A head whose weights and prefix table are non-trivial: a zero-init head has a
    /// uniform `r` law and a zero prefix response, which makes every test below
    /// vacuous.
    fn seeded_perturbed_head(latent: i64, seed: i64) -> (nn::VarStore, BarEmissionHead) {
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), latent);
        tch::manual_seed(seed);
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.35);
            }
        });
        (vs, head)
    }

    fn probe_beliefs(rows: i64, latent: i64, seed: i64) -> Tensor {
        tch::manual_seed(seed);
        Tensor::randn([rows, latent], (Kind::Float, Device::Cpu))
    }

    /// The traded belief must be a function of the PAST alone, so no assignment of the
    /// realized same-bar `s` can move it.
    ///
    /// This now holds BY CONSTRUCTION rather than by an explicit marginalization: `r` heads
    /// the chain, so [`r_probs`] reads the head's own prefix-free row and there is nothing
    /// left to integrate out. The test is kept because that is precisely what can go wrong
    /// — a read that picked up a prefix-carrying row instead would pass every type check —
    /// and because a future reorder that hands `r` a prefix has to fail here loudly.
    ///
    /// The realized `s` is the bar's own range and it pins where the close can sit, so
    /// letting it into `f_hat` would manufacture a large, clean, entirely fake improvement.
    /// Two halves:
    ///
    /// 1. Structural. [`growth_loss`] has no parameter through which an `s` could arrive,
    ///    and [`r_probs`] builds its prefix from a constant. This test pins the value: the
    ///    loss, both moments and all three diagnostics are bit-identical under EVERY one of
    ///    the 128 candidate prefixes.
    /// 2. Non-vacuity. The head's prefix pathway is LIVE on the same fixture: the row of a
    ///    factor that DOES carry a prefix moves a wide margin under the same sweep, and the
    ///    moments taken from such a row move far more than the bit-level tolerance above.
    ///    If that ever stops holding, the invariance is free and this test guards nothing.
    ///
    /// The runtime twin, on the real head and the real device, is
    /// [`verify_traded_law`], which `pretrain` calls once before the first step.
    #[test]
    fn the_marginalized_belief_ignores_the_realized_same_bar_s() {
        let _torch_rng_guard = test_rng::exclusive();
        let latent = 20i64;
        let supports = synthetic_supports(40_000, 0x6705_0001);
        let support = GrowthSupport::new(&supports, Device::Cpu).expect("support");
        let (_vs, head) = seeded_perturbed_head(latent, 0x6705_0002);
        let (batch, steps) = (3i64, 7i64);
        let rows = batch * steps;
        let beliefs = probe_beliefs(rows, latent, 0x6705_0003).view([batch, steps, latent]);
        tch::manual_seed(0x6705_0004);
        let realized_r = Tensor::randn([batch, steps], (Kind::Float, Device::Cpu)) * 0.004;

        let baseline = growth_loss(&head, &beliefs, &realized_r, &support);
        let base_loss = baseline.loss.double_value(&[]);
        let base_stats = GrowthStats::read(&baseline.stats);
        let flat = beliefs.reshape([-1, latent]);
        let (base_mu, base_second) = r_moments(&r_probs(&head, &flat), &support);

        // The reference for the non-vacuity half: a factor whose prefix CONTAINS the realized
        // `s`, read at the zero prefix. `u` sits directly behind `s` in the chain, so its row
        // is conditioned on the range — exactly the kind of row a wrong read would pick up.
        let zero_prefix = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let conditioned = head
            .logits(&flat, &zero_prefix)
            .select(1, DOF_U as i64)
            .softmax(-1, Kind::Float);
        let (conditioned_mu, _) = r_moments(&conditioned, &support);

        let mut prefix_response = 0.0f64;
        let mut leaked_mu_drift = 0.0f64;
        for bin in 0..NUM_BAR_BINS {
            // Every row of the batch told the same lie about the same-bar range, once per
            // candidate bin, so the sweep covers the whole prefix alphabet rather than one
            // permutation of it.
            let mut values = vec![0i64; rows as usize * BAR_DOF];
            for row in 0..rows as usize {
                values[row * BAR_DOF + DOF_S] = bin;
            }
            let prefix = Tensor::from_slice(&values).view([rows, BAR_DOF as i64]);
            let swept = head
                .logits(&flat, &prefix)
                .select(1, DOF_U as i64)
                .softmax(-1, Kind::Float);
            prefix_response =
                prefix_response.max((&swept - &conditioned).abs().max().double_value(&[]));
            let (leaked_mu, _) = r_moments(&swept, &support);
            leaked_mu_drift = leaked_mu_drift
                .max((&leaked_mu - &conditioned_mu).abs().max().double_value(&[]));

            // BIT-identical, not close: a tolerance here would pass an implementation that
            // mixed a little lookahead in.
            let again = growth_loss(&head, &beliefs, &realized_r, &support);
            assert_eq!(
                again.loss.double_value(&[]),
                base_loss,
                "the growth loss moved while the realized same-bar s was swept to bin {bin}"
            );
            let (mu, second) = r_moments(&r_probs(&head, &flat), &support);
            assert_eq!((&mu - &base_mu).abs().max().double_value(&[]), 0.0);
            assert_eq!((&second - &base_second).abs().max().double_value(&[]), 0.0);
            let stats = GrowthStats::read(&again.stats);
            assert_eq!(stats.mean_abs_f, base_stats.mean_abs_f);
            assert_eq!(stats.clamp_bind, base_stats.clamp_bind);
            assert_eq!(stats.min_log_argument, base_stats.min_log_argument);
        }

        assert!(
            prefix_response > 1e-2,
            "the fixture head barely responds to its chain prefix ({prefix_response:.3e}), so \
             the r row's invariance above is free and this test cannot detect lookahead"
        );
        // The mean is the quantity the term trades on, so a wrong read has to be visible
        // THERE and not merely in the probabilities.
        assert!(
            leaked_mu_drift > 1e-5,
            "a prefix-conditioned row's mean moves by only {leaked_mu_drift:.3e} across the \
             prefix alphabet, so reading one instead of the r row would be undetectable here"
        );
    }

    /// A distribution with KNOWN mean and variance must produce the analytic
    /// `f_hat = mu / sigma^2`.
    ///
    /// Built by driving the term's own moment path with a two-point law placed exactly on
    /// two bin centers, so `mu` and `sigma^2` are closed form and the assertion is against
    /// arithmetic rather than against another implementation.
    #[test]
    fn a_known_mean_and_variance_give_the_analytic_kelly_fraction() {
        let supports = synthetic_supports(40_000, 0x6706_0001);
        let support = GrowthSupport::new(&supports, Device::Cpu).expect("support");
        let returns = Vec::<f64>::try_from(support.returns.to_kind(Kind::Double).reshape([-1]))
            .expect("bin returns");
        // Two bins well inside the support and far enough apart to give a real variance.
        let (lo_bin, hi_bin) = (32usize, 96usize);
        let (r_lo, r_hi) = (returns[lo_bin], returns[hi_bin]);
        let spread = r_hi - r_lo;
        // `p_hi` is SOLVED so the analytic fraction lands on a chosen value inside the cap,
        // rather than left wherever a round probability happens to put it. With `p` the mass
        // on the high bin, `mu = r_lo + p*spread` and `sigma^2 = p(1-p)*spread^2` are exact,
        // so `f = mu/sigma^2` is a quadratic in `p`:
        //
        //     f*spread^2 * p^2 + (spread - f*spread^2) * p + r_lo = 0.
        //
        // This is not fussiness. A fitted `r` support spans ~0.15 in simple return, so
        // `1/sigma^2` is ~1e3 and a round `p_hi = 0.4` puts the analytic fraction at -100,
        // i.e. 25x outside the leverage cap and deep inside the saturation the OTHER half of
        // the suite covers. The unclamped regime has to be reached deliberately, and that is
        // itself the finding this fixture records: at this support's width the log-optimal
        // fraction is outside the cap for all but a narrow band of beliefs, which is why the
        // bench measures 78-86% cap saturation.
        for target in [-3.0f64, -0.75, 0.75, 3.0] {
            let curvature = target * spread * spread;
            let discriminant = (spread - curvature).powi(2) - 4.0 * curvature * r_lo;
            assert!(
                discriminant > 0.0,
                "no two-point law on these bins reaches f = {target}"
            );
            let root = discriminant.sqrt();
            let p_hi = [
                (curvature - spread + root) / (2.0 * curvature),
                (curvature - spread - root) / (2.0 * curvature),
            ]
            .into_iter()
            .find(|p| *p > 0.0 && *p < 1.0)
            .expect("one root places positive mass on both bins");
            let p_lo = 1.0 - p_hi;
            let mu = p_lo * r_lo + p_hi * r_hi;
            let second = p_lo * r_lo * r_lo + p_hi * r_hi * r_hi;
            let variance = second - mu * mu;
            let analytic = mu / variance;
            assert!(
                (analytic - target).abs() < 1e-9 * target.abs(),
                "the solved law gives f = {analytic} against the requested {target}, so the \
                 quadratic above is wrong and the assertion below would be self-consistent \
                 rather than analytic"
            );
            assert!(
                analytic.abs() < LEVERAGE_CAP,
                "the fixture must stay inside the cap to test the analytic value, got \
                 {analytic}"
            );

            let mu_t = Tensor::from_slice(&[mu as f32]);
            let second_t = Tensor::from_slice(&[second as f32]);
            // A zero realized return makes the loss exactly zero and isolates `f_hat`,
            // which the mean-|f| diagnostic reports under the hard clamp.
            let zero = Tensor::from_slice(&[0.0f32]);
            let growth = per_bar_growth(&mu_t, &second_t, &zero, LEVERAGE_CAP);
            let stats = GrowthStats::read(&growth.stats);
            let error = (stats.mean_abs_f - analytic.abs()).abs() / analytic.abs();
            assert!(
                error < 1e-5,
                "at p_hi = {p_hi} the term solved f_hat = {} against the analytic \
                 mu/sigma^2 = {analytic}",
                stats.mean_abs_f
            );
            assert_eq!(
                growth.loss.double_value(&[]),
                0.0,
                "a zero realized return must cost exactly zero growth"
            );
            assert_eq!(stats.clamp_bind, 0.0, "the fixture is inside the cap");
        }
    }

    /// The log's argument must stay inside its guarded range on adversarial input, and
    /// the loss must stay finite.
    ///
    /// Adversarial means: a point mass (zero variance, so `f_raw` hits the variance
    /// floor and explodes), a mean of the wrong sign against the realized return, a
    /// realized return outside the support entirely, and a non-finite one.
    #[test]
    fn the_log_argument_never_leaves_its_guarded_range() {
        let supports = synthetic_supports(40_000, 0x6707_0001);
        let support = GrowthSupport::new(&supports, Device::Cpu).expect("support");
        let worst = support.log_lo.exp_m1().abs().max(support.log_hi.exp_m1());

        // Adversarial means: point masses (zero variance, so `f_raw` saturates the
        // variance floor), means far outside anything an expectation over the support can
        // produce, and both signs so the realized return is fought as well as followed.
        let means = [0.0f32, 1e-9, -1e-9, 0.08, -0.08, 1e30, -1e30];
        // Second moments spanning a point mass, a plausible one and an absurd one.
        let seconds = [0.0f32, 1e-18, 1e30, 6.4e-3];
        // Realized LOG returns, including two well outside the support so the clip has
        // work to do.
        let realized = [0.0f32, support.log_lo as f32, support.log_hi as f32, -30.0, 30.0];
        let clipped = Tensor::from_slice(&realized)
            .clamp(support.log_lo, support.log_hi)
            .expm1();
        let rows = clipped.size()[0];
        let spread = |value: f32| {
            Tensor::from_slice(&[value])
                .expand([rows], false)
                .contiguous()
        };
        for &mean in &means {
            for &second in &seconds {
                let growth = per_bar_growth(&spread(mean), &spread(second), &clipped, LEVERAGE_CAP);
                let stats = GrowthStats::read(&growth.stats);
                let loss = growth.loss.double_value(&[]);
                assert!(
                    loss.is_finite(),
                    "mean {mean} second {second} produced a non-finite growth loss {loss}"
                );
                assert!(
                    stats.min_log_argument > LOG_ARGUMENT_FLOOR,
                    "mean {mean} second {second}: log argument fell to {}",
                    stats.min_log_argument
                );
                assert!(
                    stats.mean_abs_f <= LEVERAGE_CAP + 1e-6,
                    "the saturated fraction exceeded the cap: {}",
                    stats.mean_abs_f
                );
            }
        }
        // The structural bound the guard is slack against, from the ACTUAL support.
        assert!(
            1.0 - LEVERAGE_CAP * worst > LOG_ARGUMENT_FLOOR,
            "the support itself violates the structural bound"
        );
        // A non-finite realized bar is the one input that legitimately propagates. It
        // means the corpus handed the trainer a broken bar, and the step's finiteness
        // check is where that must be refused — silently absorbing it would hide a data
        // fault behind a plausible loss.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let poisoned = Tensor::from_slice(&[bad])
                .clamp(support.log_lo, support.log_hi)
                .expm1();
            let growth = per_bar_growth(
                &Tensor::from_slice(&[1e-4f32]),
                &Tensor::from_slice(&[6.4e-3f32]),
                &poisoned,
                LEVERAGE_CAP,
            );
            let loss = growth.loss.double_value(&[]);
            if bad.is_nan() {
                assert!(loss.is_nan(), "a NaN bar must not be silently absorbed");
            } else {
                // `clamp` maps both infinities onto the support, so the term itself is
                // safe and the loss stays finite.
                assert!(
                    loss.is_finite(),
                    "an infinite bar clips onto the support and must give a finite loss, \
                     got {loss}"
                );
            }
        }
    }

    /// The saturation must never kill the gradient, which is the whole reason the
    /// backward map is not the clamp.
    #[test]
    fn the_saturated_fraction_still_carries_gradient() {
        for raw in [0.5f64, 4.0, 10.0, 100.0] {
            // `mu / (var + eps) = raw` with a realistic variance.
            let variance = 1e-5f64;
            let mu = raw * variance;
            let mut mean = Tensor::from_slice(&[mu as f32]).set_requires_grad(true);
            let second = Tensor::from_slice(&[(variance + mu * mu) as f32]);
            let realized = Tensor::from_slice(&[0.002f32]);
            let growth = per_bar_growth(&mean, &second, &realized, LEVERAGE_CAP);
            let value = growth.loss.double_value(&[]);
            mean.zero_grad();
            growth.loss.backward();
            let grad = mean.grad().double_value(&[]);
            assert!(
                grad.is_finite() && grad.abs() > 0.0,
                "f_raw = {raw} produced gradient {grad}, so the term is dead there"
            );
            let stats = GrowthStats::read(&growth.stats);
            // The FORWARD value is the deployed hard clamp, not the surrogate.
            let expected = -(1.0f64 + raw.min(LEVERAGE_CAP) * 0.002).ln();
            assert!(
                (value - expected).abs() < 1e-6,
                "f_raw = {raw}: the reported loss {value} is not the hard-clamped \
                 policy's growth {expected}"
            );
            assert!(
                (stats.mean_abs_f - raw.min(LEVERAGE_CAP)).abs() < 1e-5,
                "f_raw = {raw} reported |f_hat| {}",
                stats.mean_abs_f
            );
        }
    }

    /// The support-bound assertion has to be a real gate, not decoration: a support that
    /// admits returns large enough to make `1 + f R` unsafe at the cap must be refused at
    /// construction.
    #[test]
    fn a_support_that_breaks_the_log_bound_is_refused() {
        let wide: Vec<BarDof> = (0..40_000)
            .map(|i| {
                let x = (i as f32 / 40_000.0 - 0.5) * 2.0;
                BarDof {
                    // Roughly +/- 0.7 in log space: a simple return near +1.0, which at a
                    // 4x cap drives the log argument to -3.
                    r: 0.7 * x,
                    s: 0.7 * x.abs() + 1e-4,
                    u: 0.5,
                    v: 0.5,
                    w: x,
                }
            })
            .collect();
        let supports = BarSupports::fit(&wide);
        let error = GrowthSupport::new(&supports, Device::Cpu)
            .expect_err("a support this wide must be refused")
            .to_string();
        assert!(
            error.contains("log argument"),
            "the refusal must name the invariant it is protecting, got: {error}"
        );
        // And the narrow, real-shaped support must be accepted, or the gate is vacuous.
        assert!(
            GrowthSupport::new(&synthetic_supports(40_000, 0x6708_0001), Device::Cpu).is_ok(),
            "a realistically clipped support must be accepted"
        );
        assert_eq!(BAR_DOF_NAMES[DOF_R], "r", "the traded DOF must be r");
    }

    /// The objective must price every `r` bin at its FITTED conditional mean, never at the
    /// EDGE decode `centers()` returns.
    ///
    /// A pin on the ECONOMICS rather than on a literal: both conventions are read off the
    /// same support and the term is asserted to have been built on the fitted one, so
    /// rebuilding [`GrowthSupport::new`] on `centers(DOF_R)` fails the first loop. The
    /// second half keeps that from being vacuous — on a geometry where the two conventions
    /// coincided at the catch-alls the loop would pass under either.
    #[test]
    fn the_objective_prices_bins_at_their_fitted_conditional_means() {
        let supports = synthetic_supports(40_000, 0x6709_0001);
        let support = GrowthSupport::new(&supports, Device::Cpu).expect("support");
        let built = Vec::<f64>::try_from(support.returns.to_kind(Kind::Double).reshape([-1]))
            .expect("bin returns");
        let fitted = supports
            .mean_decode(DOF_R, MeanDecode::Fitted)
            .expect("a freshly fitted support carries measured moments");
        let edge = supports.centers(DOF_R);
        assert_eq!(built.len(), fitted.len(), "one priced return per bin");
        for (bin, (&got, &decode)) in built.iter().zip(fitted).enumerate() {
            let want = decode.exp_m1();
            assert!(
                (got - want).abs() <= 1e-7 * want.abs().max(1e-6),
                "bin {bin} is priced at {got:.9e}, its fitted decode is {want:.9e}"
            );
        }
        // Non-vacuity, stated on the GEOMETRY rather than on a ratio of decoded returns. How
        // far a catch-all's conditional mean sits from its bound depends on how heavy the fit
        // sample's tails are — 3.04x and 3.17x on the live 300s corpus, less on a synthetic
        // fixture — but what is STRUCTURAL, and what the defect was, is that the edge decode
        // pins these two bins onto their bounds while the mass inside them does not sit there.
        let (lo, hi) = (supports.lower_bounds(DOF_R), supports.upper_bounds(DOF_R));
        for bin in [0usize, NUM_BAR_BINS as usize - 1] {
            let bound = if bin == 0 { lo[bin] } else { hi[bin] };
            assert_eq!(
                edge[bin], bound,
                "bin {bin}: the edge decode must be pinned ONTO the support bound, or there is \
                 no mispricing here and this test guards nothing"
            );
            let width = hi[bin] - lo[bin];
            assert!(
                (fitted[bin] - bound).abs() > 0.1 * width,
                "bin {bin}: its fitted mean {:.6e} sits within 10% of the {width:.6e}-wide \
                 bin's bound {bound:.6e}, so the two decodes barely differ and the loop above \
                 proves nothing",
                fitted[bin]
            );
            assert!(
                (built[bin] - bound.exp_m1()).abs() > 1e-9,
                "bin {bin} is still priced at the EDGE decode {:.6e}",
                bound.exp_m1()
            );
        }
    }

    /// A supports artifact with no measured per-bin moments makes this term UNBUILDABLE.
    ///
    /// The prohibited alternative is `bin_means(DOF_R).unwrap_or_else(|| centers(DOF_R))`,
    /// which is what `bar_dist`'s own mean-ceiling helpers used to do: it restores the edge
    /// pricing on precisely the artifacts nobody refitted, and the objective then pays 3.1x
    /// for the two catch-alls with nothing anywhere saying so.
    #[test]
    fn a_support_without_fitted_moments_refuses_to_build_the_term() {
        let dir = std::env::temp_dir()
            .join(format!("trading_bot_0_growth_decode_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("bar_supports.300.json");
        synthetic_supports(40_000, 0x670A_0001)
            .save(&path)
            .expect("a fitted support writes a v5 artifact");

        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("parse");
        let object = raw.as_object_mut().expect("object");
        object.insert("format_version".to_owned(), serde_json::json!(4));
        object.remove("bin_means");
        object.remove("bin_second_moments");
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");

        let legacy = BarSupports::load(&path).expect("a pre-moments artifact still loads");
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            !legacy.bin_means_measured(),
            "this fixture has to be a support WITHOUT moments or it tests nothing"
        );
        let error = format!(
            "{:#}",
            GrowthSupport::new(&legacy, Device::Cpu)
                .expect_err("the growth term must refuse a support carrying no fitted moments")
        );
        assert!(
            error.contains(&format!("version {BAR_SUPPORTS_MOMENTS_VERSION}"))
                && error.contains(&format!("pre-v{BAR_SUPPORTS_MOMENTS_VERSION}")),
            "the refusal must name the required artifact version and the one it got: {error}"
        );
        assert!(
            !error.contains("log argument"),
            "the refusal must be about the missing decode, not the log bound: {error}"
        );
    }
}
