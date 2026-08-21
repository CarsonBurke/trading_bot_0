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
//! mu_hat = sum_i p_i E[R | bin_i]                 (R = expm1(r))
//! m2_hat = sum_i p_i E[R² | bin_i]
//! f_raw  = mu_hat / (m2_hat + SECOND_MOMENT_FLOOR)
//! f_hat  = clamp(f_raw, -F, +F)                  (F = trade_bench::LEVERAGE_CAP)
//! L      = bankruptcy_safe_neg_log(1 + f_hat * R_realized)
//! ```
//!
//! `R_realized` is raw DATA and carries no gradient, so the whole derivative flows through
//! `f_hat`. At wealth `>= BANKRUPTCY_BARRIER_START`, where the loss is exact log utility:
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
//! which is the exact expected-log first-order condition with respect to `f` there. Raw
//! observations in or near bankruptcy use the explicit continuation documented below. The
//! mapping from the predicted law to `f_hat` is the declared second-order approximation:
//! expanding `E[log(1 + fR)]` gives
//!
//! ```text
//! E[log(1 + fR)] = f E[R] - 0.5 f² E[R²] + O(f³ E[R³]),
//! ```
//!
//! whose stationary point is `E[R] / E[R²]`. The raw second moment — not the variance —
//! is required by that expansion. The bench uses the same moment-correct contract, so the
//! training objective and economic selection no longer size two different policies.
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
//! The forward fraction is therefore EXACTLY the deployed hard-clamped policy. Wealth at or
//! above the numerical join pays its exact realized log growth; raw-tail observations below
//! it pay the explicit continuation. The fraction's gradient is
//! `df_soft/df_raw = (F / (F + |f_raw|))^2 > 0`: a strictly positive,
//! bar-wise down-weighting of over-confident bars that never changes the sign of the
//! Kelly gradient. The algebraic surrogate is chosen over `F*tanh(f_raw/F)` because its
//! derivative decays as `(F/|f_raw|)^2` rather than `exp(-2|f_raw|/F)`; at the measured
//! median `|f_raw|` of ~10 with `F = 4` that is 8.2% of full weight instead of 1.3%,
//! which is the difference between an attenuated signal and no signal. That median, and
//! the 78-86% bind fraction above it, were measured on runs whose bins used geometric
//! representatives and squared first moments. [`GrowthSupport`] now reads both fitted
//! simple-return moments directly, so both figures will be re-measured rather than assumed.
//!
//! # Raw tails and the bankruptcy domain
//!
//! The fitted edge-bin moments include the RAW open-tail observations, so the realized
//! payoff must use the same raw simple return. Clipping `r_realized` to the finite support
//! geometry would size on one law and train against another. Every observation with
//! `1 + f_hat R >= BANKRUPTCY_BARRIER_START` therefore pays exact
//! `-log1p(f_hat R)`, including solvent wealth far below the old 0.5 support tripwire.
//!
//! Expected log utility diverges at bankruptcy, while a finite tensor objective must still
//! provide a usable gradient. Within the tiny numerical neighborhood below
//! [`BANKRUPTCY_BARRIER_START`], the loss uses the quadratic continuation of `-log(a)` about
//! that point, where `a = 1 + f_hat R`. It matches both value and slope at the join, is finite
//! for every finite raw return admitted by the encoded-data domain, and grows quadratically
//! once wealth crosses zero. This is an explicit differentiable bankruptcy penalty, not a
//! clamp of the observation or of the paid wealth law.

use anyhow::{ensure, Context, Result};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{
    BarEmissionHead, BarSupports, BAR_CHAIN, BAR_DOF, BAR_SUPPORTS_SIMPLE_RETURN_MOMENTS_VERSION,
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
/// objective share would be measuring the wrong quantity: `1.0` looks inert on that chart.
/// Its gradient is not small — holding the raw second moment fixed,
/// `df_raw/dmu_hat = 1/m2_hat` with `m2_hat ~ 1e-5` multiplies the per-bar derivative by
/// ~1e5 before it reaches a parameter — so the weight is set from a GRADIENT-NORM
/// measurement, and sweeping is forbidden by the one-seed policy.
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

/// Floor on `E[R²]` in the quadratic Kelly denominator, in squared simple-return units.
///
/// It keeps a degenerate zero-return belief from dividing by zero. A realistic raw second
/// moment is ~1e-5, so this is seven orders below the quantity it guards.
const SECOND_MOMENT_FLOOR: f64 = 1e-12;

/// Small positive wealth where the finite bankruptcy continuation joins exact log utility.
///
/// `-log(a)` is used unchanged for every `a >= 1e-4`, including deeply distressed but
/// solvent positions such as `a = 0.1`. At smaller wealth, the value- and slope-matched
/// quadratic continuation avoids evaluating `log` at zero or below. `1e-4` is well resolved
/// in f32 while leaving four orders of magnitude between the join and ordinary unit wealth;
/// under encoded log returns (`r >= -30`) and the 4x leverage cap, the continuation remains
/// finite even at the most adverse long-side raw return.
pub const BANKRUPTCY_BARRIER_START: f64 = 1e-4;

/// Bound on the UNSATURATED Kelly fraction, applied before the smooth surrogate.
///
/// Purely a finiteness guard. `mu_hat` is an expectation over fitted per-bin simple-return
/// means, so its magnitude is bounded by their largest entry and this limit never binds on a
/// valid support. It exists because the surrogate `F f / (F + |f|)` is
/// a ratio of two quantities that both diverge
/// with `f`: at `f_raw = inf` it evaluates to `inf / inf = NaN`, and the straight-through
/// construction would then carry that NaN into the objective. At 1e12 the surrogate's
/// derivative is already 1.6e-25, so clamping there discards nothing a gradient could
/// use.
const SURROGATE_LIMIT: f64 = 1e12;

/// Per-resolution device-resident constants of the growth term.
///
/// Both rows come from the support artifact's directly measured simple-return law:
/// `E[expm1(r) | bin]` and `E[expm1(r)^2 | bin]`. Neither bin centers nor nonlinear
/// transforms of log-space conditional moments participate in sizing.
#[derive(Debug)]
pub struct GrowthSupport {
    /// `[1, NUM_BAR_BINS]` fitted `E[R | bin]`, `R = expm1(r)`.
    returns: Tensor,
    /// `[1, NUM_BAR_BINS]` fitted `E[R^2 | bin]`, including within-bin dispersion.
    returns_sq: Tensor,
    cap: f64,
}
impl GrowthSupport {
    /// Refuses supports predating directly fitted simple-return moments. Reconstructing these
    /// rows from bin centers, `E[r | bin]`, or `E[r^2 | bin]` is prohibited: every such route
    /// loses either nonlinear curvature or within-bin variance. Finite support bounds are
    /// categorical routing geometry only; they never truncate the fitted or realized payoff.
    pub fn new(supports: &BarSupports, device: Device) -> Result<Self> {
        let (returns, returns_sq) =
            supports
                .simple_return_bin_moment_tensors()
                .with_context(|| {
                    format!(
                        "the expected-log-growth objective requires directly fitted E[R|bin] and \
                     E[R^2|bin] for R=expm1(r), carried only by bar supports format version \
                     {BAR_SUPPORTS_SIMPLE_RETURN_MOMENTS_VERSION} or later. The loaded artifact \
                     predates v{BAR_SUPPORTS_SIMPLE_RETURN_MOMENTS_VERSION}; rerun \
                     `bar-supports-moments` on its exact geometry. Deriving these moments from \
                     bin centers or log-return moments is prohibited"
                    )
                })?;
        ensure!(
            returns.size() == [1, NUM_BAR_BINS] && returns_sq.size() == [1, NUM_BAR_BINS],
            "the fitted r simple-return moment rows have shapes {:?} and {:?}, expected \
             [1, {NUM_BAR_BINS}]",
            returns.size(),
            returns_sq.size()
        );
        Ok(Self {
            returns: returns.to_device(device),
            returns_sq: returns_sq.to_device(device),
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
    /// Mean realized growth loss over every bar of the batch, in nats per bar. Attached,
    /// with exact `-log(1 + f_hat R)` above [`BANKRUPTCY_BARRIER_START`] and its explicit
    /// finite bankruptcy-domain continuation below it.
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
    /// Smallest raw `1 + f_hat R` in the batch, before the bankruptcy continuation.
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
/// The emission readout is `Linear([h, forecast_conditioning, masked prefix
/// embeddings]) -> NUM_BAR_BINS` per DOF. `r` sits at chain position 0, so it
/// sees no same-bar prefix; its explicit conditioning contains only the target
/// exogenous clock and the current observed market.
pub fn r_probs(head: &BarEmissionHead, beliefs: &Tensor, conditioning: &Tensor) -> Tensor {
    let size = beliefs.size();
    assert_eq!(size.len(), 2, "beliefs must be [rows, latent_dim]");
    assert_eq!(
        conditioning.size(),
        size,
        "one forecast-conditioning row per belief"
    );
    let rows = size[0];
    let zero_prefix = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, beliefs.device()));
    head.logits(beliefs, conditioning, &zero_prefix)
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

/// The per-bar expected-log objective at the straight-through saturated quadratic-Kelly
/// fraction. Wealth above the tiny numerical join pays exact log utility; the explicit
/// bankruptcy continuation handles raw-tail observations below
/// [`BANKRUPTCY_BARRIER_START`].
///
/// Isolated from the belief on purpose. Retargeting the economics — growth net
/// of a turnover penalty, say — is a change to this function and to nothing else.
fn per_bar_growth(
    mu_hat: &Tensor,
    second_moment: &Tensor,
    realized_return: &Tensor,
    cap: f64,
) -> Growth {
    // The second-order expected-log expansion is
    // `f E[R] - 0.5 f² E[R²]`; its stationary point is `E[R] / E[R²]`, not
    // `E[R] / Var(R)`. The directly fitted within-bin second moments preserve the
    // curvature that squaring conditional means would erase.
    let second_moment = second_moment.clamp_min(0.0);
    let f_raw =
        (mu_hat / (second_moment + SECOND_MOMENT_FLOOR)).clamp(-SURROGATE_LIMIT, SURROGATE_LIMIT);
    let f_hard = f_raw.clamp(-cap, cap);
    // `F f / (F + |f|)`: exactly `f` to first order, bounded by `F`, derivative
    // `(F / (F + |f|))^2` which is positive everywhere and decays as a square rather
    // than exponentially. See the module docs for why the backward map is not the clamp.
    let f_soft = &f_raw * cap / (f_raw.abs() + cap);
    let f_hat = &f_soft + (&f_hard - &f_soft).detach();

    // `-log(a)` cannot be both exact all the way to `a = 0` and finite at bankruptcy.
    // Join its second-order Taylor expansion at a small positive numerical epsilon: this is
    // value- and slope-matched, keeps distressed solvent wealth such as 0.1 exact, and
    // penalizes raw ruin observations quadratically without clamping their payoff.
    let barrier = BANKRUPTCY_BARRIER_START;
    let argument = &f_hat * realized_return + 1.0;
    let shortfall = ((barrier - &argument) / barrier).clamp_min(0.0);
    let bankruptcy_loss: Tensor = -barrier.ln() + &shortfall + 0.5 * &shortfall * &shortfall;
    // `where_self` evaluates both branches eagerly, so protect only the dormant log operand
    // below the join. The selected bankruptcy branch still receives the raw argument.
    let exact_log_operand = argument.clamp_min(barrier);
    let exact_log_loss = -exact_log_operand.log();
    let per_bar_loss = bankruptcy_loss.where_self(&argument.lt(barrier), &exact_log_loss);

    let loss = per_bar_loss.mean(Kind::Float);

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

/// Convert the observed log return into the payoff law used by both fitted moments and loss.
///
/// Open-tail observations deliberately remain raw: support bounds describe categorical
/// routing geometry, not a cap on the simple return paid by the position.
fn realized_simple_returns(realized_log_r: &Tensor) -> Tensor {
    realized_log_r.detach().to_kind(Kind::Float).expm1()
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
    conditioning: &Tensor,
    realized_log_r: &Tensor,
    support: &GrowthSupport,
) -> Growth {
    tch::autocast(false, || {
        let latent = head.latent_dim();
        let flat = beliefs.reshape([-1, latent]).to_kind(Kind::Float);
        let flat_conditioning = conditioning.reshape([-1, latent]).to_kind(Kind::Float);
        let (mu_hat, second_moment) = r_moments(&r_probs(head, &flat, &flat_conditioning), support);
        // DATA: the fitted open-tail moments retain raw observations, so the paid law must
        // retain the same raw return. Detach because a gradient into the realized bar would
        // be a gradient into the future; the numerical bankruptcy continuation, not a target
        // clip, handles wealth below the tiny safe-log join.
        let realized = realized_simple_returns(&realized_log_r.reshape([-1]));
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
    let conditioning = Tensor::zeros_like(&probe);
    let (probs, drift) = tch::no_grad(|| {
        let probs = r_probs(head, &probe, &conditioning);
        // Every prefix slot filled with the same non-zero bin. If any of them could reach
        // the `r` row, this moves it.
        let mut drift = 0.0f64;
        for bin in [1i64, NUM_BAR_BINS / 2, NUM_BAR_BINS - 1] {
            let prefix = Tensor::full([rows, BAR_DOF as i64], bin, (Kind::Int64, device));
            let row = head
                .logits(&probe, &conditioning, &prefix)
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
    use crate::torch::bar_dist::{BarDof, DOF_S, DOF_U};
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
        let conditioning = probe_beliefs(rows, latent, 0x6705_0004).view([batch, steps, latent]);
        tch::manual_seed(0x6705_0004);
        let realized_r = Tensor::randn([batch, steps], (Kind::Float, Device::Cpu)) * 0.004;

        let baseline = growth_loss(&head, &beliefs, &conditioning, &realized_r, &support);
        let base_loss = baseline.loss.double_value(&[]);
        let base_stats = GrowthStats::read(&baseline.stats);
        let flat = beliefs.reshape([-1, latent]);
        let flat_conditioning = conditioning.reshape([-1, latent]);
        let (base_mu, base_second) =
            r_moments(&r_probs(&head, &flat, &flat_conditioning), &support);

        // The reference for the non-vacuity half: a factor whose prefix CONTAINS the realized
        // `s`, read at the zero prefix. `u` sits directly behind `s` in the chain, so its row
        // is conditioned on the range — exactly the kind of row a wrong read would pick up.
        let zero_prefix = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let conditioned = head
            .logits(&flat, &flat_conditioning, &zero_prefix)
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
                .logits(&flat, &flat_conditioning, &prefix)
                .select(1, DOF_U as i64)
                .softmax(-1, Kind::Float);
            prefix_response =
                prefix_response.max((&swept - &conditioned).abs().max().double_value(&[]));
            let (leaked_mu, _) = r_moments(&swept, &support);
            leaked_mu_drift =
                leaked_mu_drift.max((&leaked_mu - &conditioned_mu).abs().max().double_value(&[]));

            // BIT-identical, not close: a tolerance here would pass an implementation that
            // mixed a little lookahead in.
            let again = growth_loss(&head, &beliefs, &conditioning, &realized_r, &support);
            assert_eq!(
                again.loss.double_value(&[]),
                base_loss,
                "the growth loss moved while the realized same-bar s was swept to bin {bin}"
            );
            let (mu, second) = r_moments(&r_probs(&head, &flat, &flat_conditioning), &support);
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

    /// Known raw moments must produce the declared analytic fraction
    /// `f_hat = E[R] / E[R²]`.
    #[test]
    fn known_raw_moments_give_the_analytic_quadratic_kelly_fraction() {
        for target in [-3.0f64, -0.75, 0.75, 3.0] {
            let second = 1.0e-3;
            let mu = target * second;
            assert!(
                second >= mu * mu,
                "fixture moments must describe a valid return law"
            );
            let analytic = mu / second;
            assert!(analytic.abs() < LEVERAGE_CAP);

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
                "the term sized {} against analytic mu/E[R²] = {analytic}",
                stats.mean_abs_f
            );
            assert_eq!(growth.loss.double_value(&[]), 0.0);
            assert_eq!(stats.clamp_bind, 0.0);
        }
    }

    /// Every finite raw observation, including returns far beyond the categorical support
    /// geometry, must produce a finite objective under the capped fraction.
    #[test]
    fn finite_raw_returns_produce_finite_growth_losses() {
        let means = [0.0f32, 1e-9, -1e-9, 0.08, -0.08, 1e30, -1e30];
        let seconds = [0.0f32, 1e-18, 1e30, 6.4e-3];
        let realized =
            realized_simple_returns(&Tensor::from_slice(&[0.0f32, -0.8, 0.8, -30.0, 30.0]));
        let rows = realized.size()[0];
        let spread = |value: f32| {
            Tensor::from_slice(&[value])
                .expand([rows], false)
                .contiguous()
        };
        for &mean in &means {
            for &second in &seconds {
                let growth =
                    per_bar_growth(&spread(mean), &spread(second), &realized, LEVERAGE_CAP);
                let stats = GrowthStats::read(&growth.stats);
                let loss = growth.loss.double_value(&[]);
                assert!(
                    loss.is_finite(),
                    "mean {mean} second {second} produced a non-finite raw-tail loss {loss}"
                );
                assert!(
                    stats.min_log_argument.is_finite(),
                    "mean {mean} second {second} produced a non-finite wealth argument"
                );
                assert!(
                    stats.mean_abs_f <= LEVERAGE_CAP + 1e-6,
                    "the saturated fraction exceeded the cap: {}",
                    stats.mean_abs_f
                );
            }
        }
    }

    /// Every wealth value above the tiny numerical join uses exact expected-log utility.
    /// The 0.1 case specifically prevents the old 0.5 tripwire from becoming a payoff
    /// approximation over valid solvent wealth.
    #[test]
    fn solvent_payoffs_including_ten_percent_wealth_are_exact_negative_log1p() {
        let second = 1.0e-3f64;
        for (target_fraction, realized) in [(0.75f64, 0.2f64), (3.0, -0.3)] {
            let mean = target_fraction * (second + SECOND_MOMENT_FLOOR);
            let growth = per_bar_growth(
                &Tensor::from_slice(&[mean as f32]),
                &Tensor::from_slice(&[second as f32]),
                &Tensor::from_slice(&[realized as f32]),
                LEVERAGE_CAP,
            );
            let wealth = 1.0 + target_fraction * realized;
            assert!(wealth >= 0.1 && wealth > BANKRUPTCY_BARRIER_START);
            let expected = -(target_fraction * realized).ln_1p();
            let actual = growth.loss.double_value(&[]);
            assert!(
                (actual - expected).abs() < 1e-5,
                "wealth {wealth}: payoff was {actual}, expected exact -log1p(fR) = {expected}"
            );
        }
    }

    /// Crossing zero wealth must not clip the target or poison the graph. The continuation
    /// pays a large finite loss and its gradient still pushes the predicted fraction away
    /// from the ruinous side.
    #[test]
    fn ruin_is_finite_strongly_adverse_and_differentiable() {
        let second = 1.0e-3f64;
        let target_fraction = 3.0f64;
        let mut mean =
            Tensor::from_slice(&[(target_fraction * (second + SECOND_MOMENT_FLOOR)) as f32])
                .set_requires_grad(true);
        let realized = realized_simple_returns(&Tensor::from_slice(&[-30.0f32]));
        let growth = per_bar_growth(
            &mean,
            &Tensor::from_slice(&[second as f32]),
            &realized,
            LEVERAGE_CAP,
        );
        let value = growth.loss.double_value(&[]);
        let stats = GrowthStats::read(&growth.stats);
        assert!(
            value.is_finite() && value > 15.0,
            "near-total-loss ruin must be finite and strongly adverse, got {value}"
        );
        assert!(
            stats.min_log_argument < -1.9,
            "the raw extreme-negative ruin argument was clipped or rewritten: {}",
            stats.min_log_argument
        );

        mean.zero_grad();
        growth.loss.backward();
        let gradient = mean.grad().double_value(&[]);
        assert!(
            gradient.is_finite() && gradient > 100.0,
            "ruin must provide a strong differentiable signal away from the position, got \
             dL/dmu={gradient}"
        );
    }

    /// The saturation must never kill the gradient, which is the whole reason the
    /// backward map is not the clamp.
    #[test]
    fn the_saturated_fraction_still_carries_gradient() {
        for raw in [0.5f64, 4.0, 10.0, 100.0] {
            // `mu / (E[R²] + eps) = raw` with a realistic raw second moment.
            let second = 1e-5f64;
            let mu = raw * second;
            let mut mean = Tensor::from_slice(&[mu as f32]).set_requires_grad(true);
            let second = Tensor::from_slice(&[second as f32]);
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

    /// The support geometry routes an extreme observation into an open tail, but both its
    /// fitted moment contribution and its realized payoff must remain the same RAW simple
    /// return rather than the finite edge bound.
    #[test]
    fn open_tail_moments_and_payoffs_share_the_raw_return_law() {
        let raw_tail_log = -0.8f32;
        let samples: Vec<BarDof> = (0..40_000)
            .map(|i| {
                let x = (i as f32 / 39_999.0 - 0.5) * 2.0;
                BarDof {
                    r: if i == 0 { raw_tail_log } else { 0.01 * x },
                    s: 0.02,
                    u: 0.5,
                    v: 0.5,
                    w: x,
                }
            })
            .collect();
        let supports = BarSupports::fit(&samples);
        assert!(
            f64::from(raw_tail_log) < supports.lower_bounds(DOF_R)[0],
            "fixture observation must lie in the fitted lower open tail"
        );
        let flat: Vec<f32> = samples
            .iter()
            .flat_map(|sample| sample.to_array())
            .collect();
        let dof = Tensor::from_slice(&flat).view([-1, BAR_DOF as i64]);
        let bins = Vec::<i64>::try_from(
            supports
                .bin_ids(&dof)
                .select(-1, DOF_R as i64)
                .reshape([-1]),
        )
        .expect("r bin ids");
        let mut tail_sum = 0.0f64;
        let mut tail_count = 0usize;
        for (sample, bin) in samples.iter().zip(bins) {
            if bin == 0 {
                tail_sum += f64::from(sample.r).exp_m1();
                tail_count += 1;
            }
        }
        let raw_tail_mean = tail_sum / tail_count as f64;
        let (fitted_means, _) = supports
            .simple_return_bin_moments()
            .expect("fresh fit carries raw simple-return moments");
        assert!(
            (fitted_means[0] - raw_tail_mean).abs() < 1e-7,
            "lower-tail fitted E[R] {} does not equal the raw routed observations \
             {raw_tail_mean}",
            fitted_means[0]
        );

        let paid = realized_simple_returns(&Tensor::from_slice(&[raw_tail_log])).double_value(&[0]);
        let raw_simple = f64::from(raw_tail_log).exp_m1();
        let clipped_simple = supports.lower_bounds(DOF_R)[0].exp_m1();
        assert!(
            (paid - raw_simple).abs() < 1e-6,
            "paid tail return {paid} differs from its raw simple return {raw_simple}"
        );
        assert!(
            (paid - clipped_simple).abs() > 0.1,
            "fixture cannot distinguish raw payoff {paid} from clipped payoff {clipped_simple}"
        );

        let second = 1.0e-3f64;
        let fraction = 0.5f64;
        let growth = per_bar_growth(
            &Tensor::from_slice(&[(fraction * (second + SECOND_MOMENT_FLOOR)) as f32]),
            &Tensor::from_slice(&[second as f32]),
            &Tensor::from_slice(&[paid as f32]),
            LEVERAGE_CAP,
        );
        let expected = -(fraction * raw_simple).ln_1p();
        let actual = growth.loss.double_value(&[]);
        assert!(
            (actual - expected).abs() < 1e-6,
            "tail payoff was {actual}, expected raw-law -log1p(fR) = {expected}"
        );
    }

    /// Growth sizing consumes both directly fitted simple-return rows. The one-hot law below
    /// isolates one bin with real within-bin variance: the persisted second moment must exceed
    /// the squared first moment, and the decoded variance must therefore be positive where the
    /// removed approximation was identically zero.
    #[test]
    fn fitted_simple_return_second_moments_change_the_decoded_variance() {
        let supports = synthetic_supports(40_000, 0x6709_0001);
        let support = GrowthSupport::new(&supports, Device::Cpu).expect("support");
        let (fitted_mean, fitted_second) = supports
            .simple_return_bin_moments()
            .expect("freshly fitted supports carry simple-return moments");
        let built_mean = Vec::<f64>::try_from(support.returns.to_kind(Kind::Double).reshape([-1]))
            .expect("bin first moments");
        let built_second =
            Vec::<f64>::try_from(support.returns_sq.to_kind(Kind::Double).reshape([-1]))
                .expect("bin second moments");
        assert_eq!(built_mean, fitted_mean);
        assert_eq!(built_second, fitted_second);

        let bin = (0..NUM_BAR_BINS as usize)
            .filter(|&i| fitted_mean[i].abs() > 1.0e-12)
            .max_by(|&a, &b| {
                let variance = |i: usize| fitted_second[i] - fitted_mean[i] * fitted_mean[i];
                variance(a).total_cmp(&variance(b))
            })
            .expect("r has a dispersed bin with nonzero mean");
        let within_bin_variance = fitted_second[bin] - fitted_mean[bin] * fitted_mean[bin];
        assert!(
            within_bin_variance > 0.0,
            "fixture bin {bin} has no within-bin simple-return variance"
        );
        assert!(
            fitted_second[bin] > fitted_mean[bin] * fitted_mean[bin],
            "E[R^2|bin] must strictly exceed E[R|bin]^2 on the dispersed fixture"
        );

        let mut probabilities = vec![0.0f32; NUM_BAR_BINS as usize];
        probabilities[bin] = 1.0;
        let probs = Tensor::from_slice(&probabilities).view([1, NUM_BAR_BINS]);
        let (decoded_mean, decoded_second) = r_moments(&probs, &support);
        let decoded_variance =
            decoded_second.double_value(&[0]) - decoded_mean.double_value(&[0]).powi(2);
        assert!(
            decoded_variance > 0.0,
            "the fitted second-moment decode must preserve within-bin variance"
        );

        let moment_correct = (fitted_mean[bin] / fitted_second[bin]).abs();
        let deterministic = (fitted_mean[bin] / (fitted_mean[bin] * fitted_mean[bin])).abs();
        assert!(
            moment_correct < deterministic,
            "within-bin E[R²] must reduce Kelly below the deterministic-bin decode: \
             {moment_correct} vs {deterministic}"
        );
    }

    /// A readable v5 support has fitted log-return moments but no fitted simple-return moments;
    /// it must make the growth term unbuildable rather than trigger a nonlinear fallback.
    #[test]
    fn a_support_without_simple_return_moments_refuses_to_build_the_term() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_growth_decode_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("bar_supports.300.json");
        synthetic_supports(40_000, 0x670A_0001)
            .save(&path)
            .expect("a fitted support writes a current artifact");

        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("parse");
        let object = raw.as_object_mut().expect("object");
        object.insert("format_version".to_owned(), serde_json::json!(5));
        object.remove("bin_simple_return_means");
        object.remove("bin_simple_return_second_moments");
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");

        let legacy = BarSupports::load(&path).expect("a v5 artifact still loads for migration");
        std::fs::remove_dir_all(&dir).ok();
        assert!(legacy.bin_means_measured());
        assert!(legacy.simple_return_bin_moments().is_none());
        let error = format!(
            "{:#}",
            GrowthSupport::new(&legacy, Device::Cpu)
                .expect_err("growth must refuse absent simple-return moments")
        );
        assert!(
            error.contains(&format!(
                "version {BAR_SUPPORTS_SIMPLE_RETURN_MOMENTS_VERSION}"
            )) && error.contains(&format!(
                "predates v{BAR_SUPPORTS_SIMPLE_RETURN_MOMENTS_VERSION}"
            )),
            "the refusal must name the required artifact version: {error}"
        );
        assert!(error.contains("Deriving these moments"));
        assert!(!error.contains("log argument"));
    }
}
