//! Detached raw-payoff growth diagnostics for categorical pretraining.
//!
//! The model is trained by the proper Hard categorical NLL plus the NextLat dynamics/KL
//! terms. Raw realized payoff is deliberately excluded from that objective: pretraining
//! calls this module under `no_grad`, so neither the trunk nor the emission head can receive
//! a growth gradient.
//!
//! The diagnostic evaluates the model's prefix-free `p(r | past)` at the moment-correct
//! quadratic fraction
//!
//! ```text
//! mu_hat = sum_i p_i E[R | bin_i]                 (R = expm1(r))
//! m2_hat = sum_i p_i E[R² | bin_i]
//! f_hat  = clamp(mu_hat / (m2_hat + eps), -F, F)  (F = trade_bench::LEVERAGE_CAP)
//! value  = bankruptcy_safe_neg_log(1 + f_hat * R_realized)
//! ```
//!
//! Both fitted moments preserve the support artifact's raw simple-return law. Open-tail
//! realized observations remain raw as well: categorical bounds route observations but do
//! not cap their payoff. Solvent wealth uses exact `-log1p(f_hat R)`; at and below the tiny
//! numerical bankruptcy join, a finite value- and slope-matched quadratic continuation
//! keeps the report defined without clipping the observation.

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
    "the growth diagnostic reads p(r|past) off the head's r row, which is a forecast only \
     while r is BAR_CHAIN[0]"
);

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

/// Bound on the raw quadratic fraction before applying the deployed leverage cap.
///
/// This is only a finiteness guard for malformed support values. It cannot affect a valid
/// fitted law, and the subsequent hard clamp makes its exact magnitude immaterial to the
/// reported policy.
const RAW_FRACTION_LIMIT: f64 = 1e12;

/// Per-resolution device-resident constants of the growth diagnostic.
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
                        "the raw-payoff growth diagnostic requires directly fitted E[R|bin] and \
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

/// One step's detached raw-payoff diagnostic value and statistics.
#[derive(Debug)]
pub struct GrowthDiagnostic {
    /// Mean realized growth diagnostic over every bar of the batch, in nats per bar, with
    /// exact `-log(1 + f_hat R)` above [`BANKRUPTCY_BARRIER_START`] and its explicit finite
    /// bankruptcy-domain continuation below it.
    pub value: Tensor,
    /// `[GROWTH_STAT_COUNT]` diagnostics, in the order
    /// `[mean |f_hat|, clamp-bind fraction, min log argument]`. One tensor so a step
    /// pays one device-to-host synchronization for all three.
    pub stats: Tensor,
}

/// Entries of [`GrowthDiagnostic::stats`].
pub const GROWTH_STAT_COUNT: usize = 3;

/// Host-side view of [`GrowthDiagnostic::stats`].
#[derive(Clone, Copy, Debug)]
pub struct GrowthStats {
    /// Mean `|f_hat|` under the deployed hard clamp, comparable to the bench's
    /// `quarter-Kelly mean |f|` and `|f*| median` figures.
    pub mean_abs_f: f64,
    /// Fraction of bars where `|f_raw| > F`, i.e. where the deployed clamp chose the
    /// size instead of the predictive law.
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

/// `[rows, NUM_BAR_BINS]` `p(r | strictly past bars)`.
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

/// First two moments of `p(r | past)`, as `(mu_hat, second_moment)`, each `[rows]`.
///
/// Two inner products over the bin axis, and nothing else: `r` heads the chain, so the law
/// handed in is already the decision law and there is no same-bar factor left to integrate
/// out. `mu_hat` is a cancelling sum — ~1e-4 against per-term magnitudes of ~1e-5 — which
/// is why [`raw_payoff_diagnostic`] runs it with autocast disabled.
pub fn r_moments(probs: &Tensor, support: &GrowthSupport) -> (Tensor, Tensor) {
    let axis = [-1i64];
    (
        (probs * &support.returns).sum_dim_intlist(axis.as_slice(), false, Kind::Float),
        (probs * &support.returns_sq).sum_dim_intlist(axis.as_slice(), false, Kind::Float),
    )
}

/// The per-bar growth diagnostic at the deployed hard-capped quadratic-Kelly fraction.
/// Wealth above the tiny numerical join pays exact log utility; the explicit bankruptcy
/// continuation handles raw-tail observations below [`BANKRUPTCY_BARRIER_START`].
fn per_bar_growth_diagnostic(
    mu_hat: &Tensor,
    second_moment: &Tensor,
    realized_return: &Tensor,
    cap: f64,
) -> GrowthDiagnostic {
    // The second-order expected-log expansion is
    // `f E[R] - 0.5 f² E[R²]`; its stationary point is `E[R] / E[R²]`, not
    // `E[R] / Var(R)`. The directly fitted within-bin second moments preserve the
    // curvature that squaring conditional means would erase.
    let second_moment = second_moment.clamp_min(0.0);
    let f_raw = (mu_hat / (second_moment + SECOND_MOMENT_FLOOR))
        .clamp(-RAW_FRACTION_LIMIT, RAW_FRACTION_LIMIT);
    let f_hat = f_raw.clamp(-cap, cap);

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

    let value = per_bar_loss.mean(Kind::Float);

    let axis = [-1i64];
    let stats = tch::no_grad(|| {
        Tensor::stack(
            &[
                f_hat.detach().abs().mean(Kind::Float),
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
    GrowthDiagnostic { value, stats }
}

/// Convert the observed log return into the payoff law used by both fitted moments and the
/// diagnostic. Open-tail observations deliberately remain raw: support bounds describe
/// categorical routing geometry, not a cap on the simple return paid by the position.
fn realized_simple_returns(realized_log_r: &Tensor) -> Tensor {
    realized_log_r.detach().to_kind(Kind::Float).expm1()
}

/// Compute the detached raw-payoff growth diagnostic for one pretraining batch.
///
/// `beliefs` is `[B, T, latent_dim]` where `beliefs[b, t]` is the belief formed from bars
/// up to and including `t`, and `realized_log_r` is `[B, T]` holding the log return of
/// the bar each belief predicts. The public boundary enforces `no_grad`; its value is a
/// batch diagnostic and never an optimized objective or held-out promotion score.
///
/// Runs with autocast disabled. `mu_hat = sum_i p_i R_i` is a cancelling sum whose value
/// is ~1e-4 against per-term magnitudes of ~1e-5 and a spread `E|R| ~ 2e-3`, so bf16's
/// eight mantissa bits would destroy the quantity being measured.
pub fn raw_payoff_diagnostic(
    head: &BarEmissionHead,
    beliefs: &Tensor,
    conditioning: &Tensor,
    realized_log_r: &Tensor,
    support: &GrowthSupport,
) -> GrowthDiagnostic {
    tch::no_grad(|| {
        tch::autocast(false, || {
            let latent = head.latent_dim();
            let flat = beliefs.reshape([-1, latent]).to_kind(Kind::Float);
            let flat_conditioning = conditioning.reshape([-1, latent]).to_kind(Kind::Float);
            let (mu_hat, second_moment) =
                r_moments(&r_probs(head, &flat, &flat_conditioning), support);
            // DATA: the fitted open-tail moments retain raw observations, so the measured
            // payoff must retain the same raw return. The numerical bankruptcy continuation,
            // not a target clip, handles wealth below the tiny safe-log join.
            let realized = realized_simple_returns(&realized_log_r.reshape([-1]));
            assert_eq!(
                realized.size(),
                mu_hat.size(),
                "one realized return per belief"
            );
            per_bar_growth_diagnostic(&mu_hat, &second_moment, &realized, support.cap)
        })
    })
}

/// Prove, on the real device and the real head, that the `r` law this diagnostic reads is the
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
        "the growth diagnostic reads p(r|past) straight off the head's r row, which is a \
         forecast only while r heads BAR_CHAIN"
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
            "the growth diagnostic's {name} disagrees with an f64 reduction of the same law by \
             {error:.3e} relative on this device. That is a numerical-precision failure, not \
             a maths one — check that f32 matmul is not running in TF32 — or the report would \
             measure a mean the model did not produce."
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

    #[test]
    fn raw_payoff_diagnostic_cannot_attach_to_model_inputs_or_parameters() {
        let _torch_rng_guard = test_rng::exclusive();
        let latent = 16;
        let supports = synthetic_supports(20_000, 0xD1A6_0001);
        let support = GrowthSupport::new(&supports, Device::Cpu).expect("support");
        let (vs, head) = seeded_perturbed_head(latent, 0xD1A6_0002);
        let beliefs = probe_beliefs(12, latent, 0xD1A6_0003)
            .view([3, 4, latent])
            .set_requires_grad(true);
        let conditioning = probe_beliefs(12, latent, 0xD1A6_0004)
            .view([3, 4, latent])
            .set_requires_grad(true);
        let realized = Tensor::full([3, 4], 30.0, (Kind::Float, Device::Cpu));

        let diagnostic = raw_payoff_diagnostic(&head, &beliefs, &conditioning, &realized, &support);

        assert!(
            !diagnostic.value.requires_grad() && !diagnostic.stats.requires_grad(),
            "raw-payoff evidence must be detached at its public API boundary"
        );
        assert!(!beliefs.grad().defined() && !conditioning.grad().defined());
        assert!(
            vs.trainable_variables()
                .iter()
                .all(|parameter| !parameter.grad().defined()),
            "reading the diagnostic must not create parameter gradients"
        );
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
    /// 1. Structural. [`raw_payoff_diagnostic`] has no parameter through which an `s` could
    ///    arrive, and [`r_probs`] builds its prefix from a constant. This test pins the value:
    ///    the diagnostic, both moments and all three statistics are bit-identical under every
    ///    one of the 128 candidate prefixes.
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

        let baseline = raw_payoff_diagnostic(&head, &beliefs, &conditioning, &realized_r, &support);
        let base_value = baseline.value.double_value(&[]);
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
            let again =
                raw_payoff_diagnostic(&head, &beliefs, &conditioning, &realized_r, &support);
            assert_eq!(
                again.value.double_value(&[]),
                base_value,
                "the growth diagnostic moved while the realized same-bar s was swept to bin {bin}"
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
        // The mean is the quantity the diagnostic measures, so a wrong read must be visible
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
            // A zero realized return makes the value exactly zero and isolates `f_hat`,
            // which the mean-|f| diagnostic reports under the hard clamp.
            let zero = Tensor::from_slice(&[0.0f32]);
            let growth = per_bar_growth_diagnostic(&mu_t, &second_t, &zero, LEVERAGE_CAP);
            let stats = GrowthStats::read(&growth.stats);
            let error = (stats.mean_abs_f - analytic.abs()).abs() / analytic.abs();
            assert!(
                error < 1e-5,
                "the diagnostic sized {} against analytic mu/E[R²] = {analytic}",
                stats.mean_abs_f
            );
            assert_eq!(growth.value.double_value(&[]), 0.0);
            assert_eq!(stats.clamp_bind, 0.0);
        }
    }

    /// Every finite raw observation, including returns far beyond the categorical support
    /// geometry, must produce a finite diagnostic under the capped fraction.
    #[test]
    fn finite_raw_returns_produce_finite_growth_diagnostics() {
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
                let growth = per_bar_growth_diagnostic(
                    &spread(mean),
                    &spread(second),
                    &realized,
                    LEVERAGE_CAP,
                );
                let stats = GrowthStats::read(&growth.stats);
                let value = growth.value.double_value(&[]);
                assert!(
                    value.is_finite(),
                    "mean {mean} second {second} produced a non-finite raw-tail diagnostic \
                     {value}"
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
            let growth = per_bar_growth_diagnostic(
                &Tensor::from_slice(&[mean as f32]),
                &Tensor::from_slice(&[second as f32]),
                &Tensor::from_slice(&[realized as f32]),
                LEVERAGE_CAP,
            );
            let wealth = 1.0 + target_fraction * realized;
            assert!(wealth >= 0.1 && wealth > BANKRUPTCY_BARRIER_START);
            let expected = -(target_fraction * realized).ln_1p();
            let actual = growth.value.double_value(&[]);
            assert!(
                (actual - expected).abs() < 1e-5,
                "wealth {wealth}: payoff was {actual}, expected exact -log1p(fR) = {expected}"
            );
        }
    }

    /// Crossing zero wealth must not clip the observation or poison the held-out report.
    #[test]
    fn ruin_is_finite_and_strongly_adverse() {
        let second = 1.0e-3f64;
        let target_fraction = 3.0f64;
        let mean = Tensor::from_slice(&[(target_fraction * (second + SECOND_MOMENT_FLOOR)) as f32]);
        let realized = realized_simple_returns(&Tensor::from_slice(&[-30.0f32]));
        let growth = per_bar_growth_diagnostic(
            &mean,
            &Tensor::from_slice(&[second as f32]),
            &realized,
            LEVERAGE_CAP,
        );
        let value = growth.value.double_value(&[]);
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
    }

    /// The reported payoff and fraction use the deployed hard leverage cap.
    #[test]
    fn saturated_fraction_uses_the_deployed_hard_cap() {
        for raw in [0.5f64, 4.0, 10.0, 100.0] {
            let second = 1e-5f64;
            let mu = raw * second;
            let mean = Tensor::from_slice(&[mu as f32]);
            let second = Tensor::from_slice(&[second as f32]);
            let realized = Tensor::from_slice(&[0.002f32]);
            let growth = per_bar_growth_diagnostic(&mean, &second, &realized, LEVERAGE_CAP);
            let value = growth.value.double_value(&[]);
            let stats = GrowthStats::read(&growth.stats);
            let expected = -(1.0f64 + raw.min(LEVERAGE_CAP) * 0.002).ln();
            assert!(
                (value - expected).abs() < 1e-6,
                "f_raw = {raw}: the reported value {value} is not the hard-clamped \
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
        let growth = per_bar_growth_diagnostic(
            &Tensor::from_slice(&[(fraction * (second + SECOND_MOMENT_FLOOR)) as f32]),
            &Tensor::from_slice(&[second as f32]),
            &Tensor::from_slice(&[paid as f32]),
            LEVERAGE_CAP,
        );
        let expected = -(fraction * raw_simple).ln_1p();
        let actual = growth.value.double_value(&[]);
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
    /// it must make the growth diagnostic unbuildable rather than trigger a nonlinear fallback.
    #[test]
    fn a_support_without_simple_return_moments_refuses_to_build_the_diagnostic() {
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
