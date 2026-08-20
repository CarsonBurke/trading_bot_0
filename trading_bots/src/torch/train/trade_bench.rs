//! What the predictive distribution is WORTH: the log-optimal (Kelly) trading bench.
//!
//! Nats are the objective, but nats do not say whether the head is economically
//! useful. `-9.29` nats/bar against a `-3.78` marginal is `+5.5` nats of code length,
//! and code length converts into money at an exchange rate nobody has measured. This
//! module measures it, by asking the only question that has a unique answer:
//! *assuming the model's predictive distribution is correct, how would one optimally
//! trade it, and what does that earn?*
//!
//! # The policy is derived, not tuned
//!
//! For a single-period bet on a return `R` with a known law, the wealth-maximizing
//! fraction of capital is the one that maximizes expected LOG growth,
//! `g(f) = E[ln(1 + f R)]` — the Kelly criterion. Nothing here is fitted: the
//! predictive law comes from the head, the expectation is an exact finite sum over the
//! 128 bins (each bin contributing its probability times the return its representative
//! `r` decodes to, atoms contributing their exact point mass), and `g` is strictly
//! concave in `f` on its domain, so the maximizer is unique and is found by bisecting
//! `g'` inside the feasible bracket. Two clamps, both stated rather than tuned:
//!
//! * [`LEVERAGE_CAP`] bounds `|f|`.
//! * A position is taken only if the expected log growth at the optimum is strictly
//!   positive. This is the "only trade high-confidence predictions" gate in its derived
//!   form. `g(0) = 0` always, so the gate binds exactly when the model sees no edge at
//!   all — a zero-edge law yields a zero position, not a coin flip.
//!
//! There is no heuristic anywhere: no "long if the mean is positive", no threshold on a
//! point forecast, no learned parameter, nothing fitted on evaluation data.
//!
//! # Only `r`, and only from the past
//!
//! **The policy trades the log return `r` and nothing else, and it may never see any
//! part of the bar it is betting on.** That constraint is not cosmetic, and the emission
//! chain is ordered to make it free: [`BAR_CHAIN`] = `r -> s -> u -> v -> w` puts the
//! traded factor FIRST, so the `r` head conditions on no same-bar factor at all and
//!
//! ```text
//! p(r | h) = softmax(head.logits(h)_r)
//! ```
//!
//! is the traded law exactly, with nothing to integrate out. The `r` row of
//! [`BarEmissionHead::forecast_log_probs`] is the same object up to a `log`/`exp` round
//! trip, which the tests check it against.
//!
//! The no-lookahead property is STRUCTURAL, not asserted at runtime:
//! [`forecast_r_probs`] takes the head and the causal beliefs, and there is no parameter
//! through which a caller could hand it the realized bar. Even the head's own
//! [`BarEmissionHead::logits`] cannot leak one into that row: the prefix mask of chain
//! position 0 is identically zero, so the row is the same whatever prefix is passed. The
//! realized bar enters this module in exactly two places, both of them
//! outcomes rather than decisions: the realized return that the position is paid on,
//! and the perfect-foresight oracle, whose entire purpose is to see it.
//!
//! # The baselines are the point
//!
//! A policy fed a fat-tailed unconditional law still posts a positive Sharpe, because
//! equities drift up and log-optimal sizing of a drifting asset is profitable without
//! any forecasting at all. So the number that matters is never the model's Sharpe; it
//! is the model's growth MINUS the growth of the identical machinery driven by the
//! fitted unconditional marginal. Six policies, identical solver, identical windows,
//! identical costs, differing only in the distribution fed in or in the multiple staked:
//!
//! * [`POLICY_MODEL`] — the conditional predictive law. The thing under test.
//! * [`POLICY_HALF`], [`POLICY_QUARTER`] — the same law at half and quarter Kelly. Not
//!   timidity: the standard remedy for MISSPECIFICATION, and under an overstated edge
//!   `g` is concave with quadratic error, so half Kelly keeps ~75% of the true growth at
//!   a quarter of the ruin exposure while a doubled `f*` can grow negatively.
//! * [`POLICY_MARGINAL`] — the train-fitted unconditional bin masses of `r`
//!   ([`BarSupports::bin_masses`]). The NULL. It depends on no model weight whatsoever,
//!   so a run that cannot beat it has bought nothing with its +5.5 nats.
//! * [`POLICY_BUY_HOLD`] — `f = 1` every bar. Not a Kelly policy; it is the market, and
//!   it is what fixes the units of [`LEVERAGE_CAP`].
//! * [`POLICY_ORACLE`] — a point mass on the REALIZED return. Perfect foresight, same
//!   solver, same cap. It is the attainable ceiling under this cap, so the model's share
//!   of it is the fraction of available edge the predictor actually captures.
//!
//! Every one of them is a clamp of ONE solved number per bar, [`WindowPaths::free`], the
//! uncapped optimum. That is exact rather than convenient: `g` is concave, so the
//! constrained optimum on `[-c, c]` is the projection of the free optimum onto it. It is
//! also what makes the whole verdict re-derivable at any cap for free.
//!
//! # The cap is a confound, so it is reported as an axis
//!
//! A single headline at [`LEVERAGE_CAP`] is uninterpretable once the cap binds: with most
//! bars clipped, "Kelly" has degenerated into maximum leverage along the predicted sign
//! and the reported edge is a property of the cap, not of the distribution. So the bench
//! reports [`TradeBench::cap_curve`] over [`CAP_GRID`] — the same verdict re-clamped at
//! eight caps, model and null alike, nothing re-solved — and
//! [`TradeBench::free_kelly`], the distribution of `|f*|` with the share of bars at the
//! cap. An edge that grows with the cap while saturation stays near one was bought with
//! leverage; an edge that is flat in the cap once the cap stops binding is the model's.
//!
//! # The tail is where a leveraged bettor actually dies
//!
//! An NLL is an average over the bulk, and a central 80% coverage band is a statement
//! about the same bulk: essentially all of the probability, and therefore all of the code
//! length, sits where nothing dangerous happens. Neither can see the 0.1% tail, and the
//! 0.1% tail is the entire risk of a leveraged position.
//! [`TradeBench::tail`] therefore tests, on held-out data, the model's OWN per-bar
//! quantiles at [`TAIL_LEVELS`]: realized exceedance rate over promised, both tails, with
//! a Wilson floor and a window-blocked interval. A ratio of four means the law promised
//! 0.1% and delivered 0.4%, which is the difference between a survivable Kelly position
//! and a wipeout — and it is invisible in every other number this repo reports.
//!
//! # Honesty
//!
//! Costs are charged on realized traded notional, and the headline output is not the
//! net growth at one assumed cost — it is [`TradeBench::model_break_even`], the cost level
//! at which the model's advantage over the marginal null disappears. A strategy whose
//! edge dies at 0.2 bps is not a strategy. The interval on the edge is a block bootstrap
//! over WINDOWS (via [`block_bootstrap`], blocked by `(symbol, calendar month)` by the
//! caller), because bars inside a window share a symbol and a regime and are not
//! independent draws. What that blocking DOES and DOES NOT buy has since been measured, and
//! the honest statement is narrower than the one this comment used to make — see
//! [`compare_clustering`], which is the check rather than the claim.
//!
//! The edge, its interval, its cost curve and its share of the perfect-foresight ceiling are
//! computed for EVERY policy against the SAME null on the SAME windows, not for the model
//! alone. Full Kelly is the ceiling of what the clamp permits and is routinely not the
//! fraction one would run, so a half-Kelly row has to be quotable as the verdict without
//! anyone re-deriving its interval by hand. [`TradeBench::model_edge`] and its siblings name
//! the model's row for the consumers that only want the headline.
//!
//! The positions are cost-BLIND — Kelly on the raw predictive law, as specified — and
//! costs are charged afterwards on the turnover that policy generated. That is
//! conservative in a known direction: a cost-aware trader would rebalance less and net
//! more, so every net number here is a lower bound on what the same predictive law is
//! worth.

use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{
    BarEmissionHead, BarSupports, BAR_CHAIN, BAR_DOF, DOF_R, NUM_BAR_BINS,
};
use crate::torch::dataset::mix64;

use super::pretrain_stats::{block_bootstrap, Dispersion, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CI_MASS};

/// `p(r | h)` is READ DIRECTLY off the head's `r` row, because `r` heads the chain and
/// therefore has no prefix to integrate out. A reorder that puts any factor before `r`
/// gives it a prefix again, at which point the direct read silently becomes a
/// teacher-forced row and the mixture that was deleted here would have to come back.
const _: () = assert!(
    BAR_CHAIN[0] == DOF_R,
    "trade_bench reads p(r|h) directly off the head's r row, which is p(r|past) only while \
     r is BAR_CHAIN[0]; a reorder that gives r a prefix must marginalize it out again"
);

/// Hard bound on `|f|`, in units of wealth.
///
/// Full Kelly is unbounded from above when a predictive law has no loss mass, and the
/// outermost bins of an equal-mass support are open-ended catch-alls whose decoded
/// center understates the true tail by construction ([`BarSupports`] clips the support
/// at the `1e-4` quantile). Both facts argue for a cap. `4.0` is chosen because it is
/// the practical portfolio-margin limit for US equities and because it does NOT bind on
/// the unconditional null: the classic Kelly leverage for an equity index is
/// `mu / sigma^2 ~ 0.08 / 0.04 ~ 2`, horizon-invariant since both scale linearly in
/// time, so the marginal baseline lands near `2` and stays a real policy rather than a
/// clamped constant. Nothing is tuned to the model, and
/// [`PolicyStats::clamped_fraction`] reports how often the cap binds, so a run in which
/// the cap is doing the deciding says so.
pub const LEVERAGE_CAP: f64 = 4.0;

/// Round-trip-free, one-way cost charged per unit of notional traded, in basis points.
///
/// A liquid US large cap quotes 1-2 bps wide, so crossing costs ~0.5-1 bp of half
/// spread, plus ~0.5 bp of fees and immediate impact for a size that does not move the
/// book. `2.0` is the defensible central estimate for trading at a 5-minute bar close.
/// It is only the DEFAULT: every net figure is also reported across [`COST_GRID_BPS`]
/// and the break-even cost is solved for exactly, so no conclusion rests on this value.
pub const DEFAULT_COST_BPS: f64 = 2.0;

/// Cost levels the reported edge curve is evaluated at, in basis points per unit traded.
/// Dense below 5 bps because that is where a 5-minute edge lives or dies.
pub const COST_GRID_BPS: [f64; 12] = [
    0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0,
];

/// Bars per trading day in this corpus. Measured, not assumed: `pretrain_stats`
/// documents ~93 bars/day for the 2048-bar windows this bench runs on.
pub const BARS_PER_TRADING_DAY: f64 = 93.0;
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;
/// Annualization factor for the per-bar Sharpe ratio.
pub const BARS_PER_YEAR: f64 = BARS_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR;

/// Held-out windows the bench trades, taken as a prefix of the pinned evaluation set.
///
/// The pinned set is drawn under `EVAL_WINDOW_SEED`, so its order is fixed across runs,
/// seeds and ablations and a prefix is as pinned as the whole. 256 windows is ~230k
/// traded bars, which is far more than a growth MEAN needs: the width of the interval on
/// the growth difference is set by the number of `(symbol, month)` BLOCKS, not by the bar
/// count. Bounding it is what keeps the bench affordable at every validation instead of
/// only at the end.
pub const TRADE_WINDOWS: usize = 256;

/// Slot of [`DEFAULT_COST_BPS`] inside [`COST_GRID_BPS`], so the charted curve passes
/// exactly through the headline net figures instead of near them.
pub const DEFAULT_COST_SLOT: usize = 5;
const _: () = assert!(
    COST_GRID_BPS[DEFAULT_COST_SLOT] == DEFAULT_COST_BPS,
    "the charted cost curve must contain the cost the headline numbers are charged at"
);

/// Rows per chunk of the traded law and the solver.
///
/// The bench runs inside a validation that shares the device with training, so the peak
/// is bounded here rather than left to scale with the evaluation batch.
pub const ROW_CHUNK: i64 = 1024;

/// Bisection steps of the Kelly solve. The bracket is at most `2 * LEVERAGE_CAP` wide,
/// so 60 halvings pin `f` to the last bits of an f64.
const SOLVER_ITERATIONS: usize = 60;
/// Fraction by which the feasible bracket is pulled inside the domain, where
/// `ln(1 + f R) -> -inf`.
const FEASIBLE_MARGIN: f64 = 1e-9;
/// Floor on a bar's wealth multiplier, so a leveraged position against a realized move
/// outside the fitted support costs a large finite number of nats instead of poisoning
/// every downstream sum with `-inf`. Every occurrence is counted in
/// [`PolicyStats::ruin_bars`].
const WEALTH_FLOOR: f64 = 1e-6;
/// Cost, in basis points, beyond which the break-even search reports "never".
///
/// Also the value the CHARTED break-even is clipped to: an infinite break-even is dropped
/// by the renderer's non-finite filter, and a dropped point looks exactly like a metric that
/// was never measured, which is the one confusion this repo's reports refuse to allow.
pub const MAX_BREAK_EVEN_BPS: f64 = 1000.0;
/// Bisection steps of the break-even cost solve, over a bracket of at most
/// [`MAX_BREAK_EVEN_BPS`]: 48 halvings resolve it to `~4e-12` bps.
const BREAK_EVEN_ITERATIONS: usize = 48;

/// Hard ceiling on `|f|` the Kelly solve will return, DECLARED rather than emergent.
///
/// # Why this constant has to exist
///
/// Until it did, the largest position this bench could ever take was set by a bin edge. The
/// solve's bracket is `[-cap, cap]` intersected with the open domain `1 + f R_b > 0`, and the
/// most negative decoded return is the outermost `r` bin's, which the support builder pins to
/// the CLIPPED BOUND rather than to a bin midpoint. On the live 300s support that bound is
/// `-883.32` bps, so `R = -0.084543` and `ln(1 + f R)` diverges at `f = 11.8283`. The measured
/// uncapped median `|f*|` is `10.49`, i.e. within 11% of it: the "uncapped" fraction was
/// largely reporting the distance to a discretization constant, and every ruin bound the bench
/// had was that same constant.
///
/// Two consequences, and the second is why this is a constant and not a comment. Re-fitting the
/// supports at a different clip quantile moves the bound. Decoding the catch-all bins at their
/// fitted conditional mean instead of their edge - which is correct for a moment, and is being
/// landed - moves it from `11.83` to `36.59`, a 3.1x LOOSENING of ruin protection that no line
/// of code would have mentioned. A maximum tolerable leverage is a risk decision, so it is
/// written down once, here, independent of the support, the clip quantile and the decode.
///
/// The value preserves today's behaviour: it sits just above the `11.8283` the support edge
/// silently enforced, so landing it moves no measured quantity, and it binds the moment the
/// decode fix would otherwise loosen the bound. Points of [`CAP_GRID`] above it cannot bind -
/// they were already truncated by the domain - and the uncapped `|f*|` histogram is what shows
/// when the ceiling, rather than the law, chose the size.
pub const MAX_LEVERAGE: f64 = 12.0;
const _: () = assert!(
    MAX_LEVERAGE > LEVERAGE_CAP,
    "the declared ruin bound must leave the headline cap free to bind on its own"
);

/// Sentinel for "solve without a cap of the caller's own", i.e. the uncapped optimum.
///
/// Distinct from [`MAX_LEVERAGE`], which bounds every solve no matter what a caller passes:
/// this says only that the CALLER imposes nothing. A large finite value keeps the bisection's
/// absolute resolution at `~1e-17` and leaves the bracket set by the distribution and the
/// declared ceiling rather than by an arbitrary caller-side number.
pub const FREE_LEVERAGE: f64 = 1.0e6;

/// Leverage caps the whole bench is re-reported at.
///
/// A single number at [`LEVERAGE_CAP`] is not interpretable once the cap binds: with 85%
/// of bars clipped the policy has degenerated into "maximum leverage along the predicted
/// sign" and is barely using the distribution it was handed, so its edge is a property of
/// the cap and not of the model. Every point of this grid re-clamps the SAME solved
/// fractions — no re-solve, no refit, nothing tuned — so the curve is pure reporting.
pub const CAP_GRID: [f64; 8] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

/// Slot of [`LEVERAGE_CAP`] inside [`CAP_GRID`], so the curve passes exactly through the
/// headline figures.
pub const CAP_GRID_DEFAULT_SLOT: usize = 4;
const _: () = assert!(
    CAP_GRID[CAP_GRID_DEFAULT_SLOT] == LEVERAGE_CAP,
    "the charted cap curve must contain the cap the headline numbers are sized at"
);

/// Bin edges of the reported histogram of `|f*|`, the UNCAPPED optimum.
///
/// Open-ended on the right: what a reader needs from this chart is the mass at and beyond
/// [`LEVERAGE_CAP`], because that mass is the fraction of bars whose size was chosen by
/// the cap rather than by the predictive law.
pub const FREE_KELLY_EDGES: [f64; 9] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, f64::INFINITY];

/// Two-sided predicted probability levels the tail calibration is tested at.
///
/// Coverage of a central 80% band says nothing about the 0.1% tail that ruins a leveraged
/// position, and neither does an NLL: both are dominated by the bulk, where essentially
/// all of the probability and all of the code length live. These are the levels at which
/// a leveraged Kelly bettor is actually exposed.
pub const TAIL_LEVELS: [f64; 4] = [0.001, 0.005, 0.01, 0.05];

/// Realized-over-promised tail ratio beyond which the traded law is called out as
/// understating its own tail.
///
/// `1.5` rather than `1.0`: at 0.1% nominal over a few hundred thousand bars the Wilson
/// half-width is itself tens of percent, so flagging every ratio above one would flag
/// noise. A 50% understatement is both outside that noise and the point where a Kelly size
/// computed from the law is materially too large.
pub const TAIL_RATIO_WARN: f64 = 1.5;

pub const POLICY_MODEL: usize = 0;
pub const POLICY_HALF: usize = 1;
pub const POLICY_QUARTER: usize = 2;
pub const POLICY_MARGINAL: usize = 3;
pub const POLICY_BUY_HOLD: usize = 4;
pub const POLICY_ORACLE: usize = 5;
pub const POLICY_COUNT: usize = 6;
/// Series names, in policy-index order.
pub const POLICY_NAMES: [&str; POLICY_COUNT] = [
    "model",
    "half kelly",
    "quarter kelly",
    "marginal null",
    "buy&hold",
    "oracle",
];
/// Kelly multiple each policy stakes, `NAN` for the ones that are not Kelly on the model.
///
/// Fractional Kelly is the standard remedy for model MISSPECIFICATION, not a timidity
/// knob, and it belongs in the reported set rather than in a comment. Full Kelly is only
/// growth-optimal when the law is exactly right and rebalancing is continuous; here the
/// law is a 128-bin estimate whose outermost bins understate the true tail by
/// construction, and rebalancing happens once per 5-minute bar across halts and gaps.
/// Under an overstated edge, `g(f)` is concave and its ERROR is quadratic in the
/// overstatement, so a bettor at `f*/2` keeps ~75% of the true growth while cutting
/// variance and ruin exposure fourfold; a bettor at a mistakenly doubled `f*` can have
/// negative growth. The asymmetry is why half Kelly is the professional default, and why
/// this bench reports it as a first-class policy beside the full-Kelly headline.
pub const POLICY_KELLY_MULTIPLE: [f64; POLICY_COUNT] =
    [1.0, 0.5, 0.25, f64::NAN, f64::NAN, f64::NAN];

// ---------------------------------------------------------------------------
// The predictive object
// ---------------------------------------------------------------------------

/// Simple return of each `r` bin: `exp(center) - 1`.
///
/// The bin's representative value is its center, which is the atom itself on a
/// zero-width bin and the midpoint of a continuous one — the same convention
/// [`BarSupports::expectation`] and the CRPS use, so a bin means the same thing to the
/// bench as it does to every other consumer of the support.
pub fn bin_returns(supports: &BarSupports) -> Vec<f64> {
    supports
        .centers(DOF_R)
        .iter()
        .map(|center| center.exp_m1())
        .collect()
}

/// `[rows, NUM_BAR_BINS]` probabilities of `p(r | strictly past bars)`.
///
/// **This is the only distribution the traded decision is ever allowed to see.** There
/// is deliberately no parameter through which the realized bar could reach it: the
/// signature carries the head and the causal beliefs and nothing else. `beliefs[i]` must
/// be the belief formed from bars up to and including the bar BEFORE the one being
/// predicted, which is exactly the alignment the pretrainer's teacher-forced pass
/// produces.
///
/// `r` is [`BAR_CHAIN`]`[0]`, so the head's `r` row IS this law. The prefix
/// [`BarEmissionHead::logits`] requires is a placeholder that chain position 0's all-zero
/// prefix mask discards; handing it the realized bar instead of zeros would return the
/// identical row.
pub fn forecast_r_probs(head: &BarEmissionHead, beliefs: &Tensor) -> Tensor {
    let size = beliefs.size();
    assert_eq!(size.len(), 2, "beliefs must be [rows, latent_dim]");
    let rows = size[0];
    tch::no_grad(|| {
        let zero_prefix = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, beliefs.device()));
        head.logits(beliefs, &zero_prefix)
            .select(1, DOF_R as i64)
            .softmax(-1, Kind::Float)
    })
}

// ---------------------------------------------------------------------------
// The Kelly solve
// ---------------------------------------------------------------------------

/// `E[ln(1 + f R)]` under a discrete law. Host-side twin of the tensor objective, used
/// by the tests and by the reported growth accounting.
pub fn expected_log_growth(probs: &[f64], returns: &[f64], fraction: f64) -> f64 {
    assert_eq!(probs.len(), returns.len());
    probs
        .iter()
        .zip(returns)
        .map(|(p, r)| {
            if *p <= 0.0 {
                0.0
            } else {
                p * (1.0 + fraction * r).max(WEALTH_FLOOR).ln()
            }
        })
        .sum()
}

/// `[rows]` log-optimal fractions for `[rows, outcomes]` probabilities and returns.
///
/// `returns` may be `[outcomes]`, `[1, outcomes]` (one law shared by every row) or
/// `[rows, outcomes]` (a per-row law, which is what the perfect-foresight oracle is).
///
/// `g(f) = sum_b p_b ln(1 + f R_b)` is strictly concave wherever it is finite, so
/// `g'(f) = sum_b p_b R_b / (1 + f R_b)` is strictly decreasing and a bisection on its
/// sign inside the feasible bracket converges to the unique maximizer — and collapses
/// onto the relevant endpoint when the maximizer is at the boundary, with no branch. The
/// bracket is `[-cap, cap]` intersected with the open domain `1 + f R_b > 0` for every
/// bin carrying mass, pulled inside by [`FEASIBLE_MARGIN`], and intersected again with the
/// DECLARED ceiling [`MAX_LEVERAGE`]; it always contains `0`, since positive-return bins only
/// bound the short side and negative ones only the long side.
///
/// The ceiling is applied here rather than left to callers on purpose: before it existed the
/// binding constraint on an "uncapped" solve was the outermost bin's decoded return, so the
/// bench's only ruin bound was a discretization constant that a support refit or a decode
/// change would move silently. See [`MAX_LEVERAGE`].
///
/// The returned fraction is exactly `0` unless the expected log growth at the optimum is
/// strictly positive.
pub fn kelly_fractions(probs: &Tensor, returns: &Tensor, cap: f64) -> Tensor {
    assert!(
        cap > 0.0 && cap.is_finite(),
        "the leverage cap must be positive and finite"
    );
    let probs = probs.to_kind(Kind::Double);
    let size = probs.size();
    assert_eq!(size.len(), 2, "probs must be [rows, outcomes]");
    let (rows, outcomes) = (size[0], size[1]);
    let returns = returns.to_kind(Kind::Double).reshape([-1, outcomes]);
    let return_rows = returns.size()[0];
    assert!(
        return_rows == rows || return_rows == 1,
        "returns must be shared ([1, outcomes]) or per-row ([rows, outcomes]), got \
         [{return_rows}, {outcomes}] against {rows} rows"
    );
    let returns = if return_rows == rows {
        returns
    } else {
        returns.expand([rows, outcomes], false)
    };

    tch::no_grad(|| {
        let mass = probs
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Double)
            .clamp_min(f64::MIN_POSITIVE);
        let probs = probs.divide(&mass);
        let live = probs.gt(0.0);
        // A ZERO-PROBABILITY bin is excluded from the bounds below, so its own
        // `1 + f R_b > 0` constraint is not enforced and the bisection may evaluate the slope
        // at an `f` where that factor is exactly zero. The term would then be `0 * R_b / 0`,
        // i.e. NaN, `slope.gt(0.0)` would be false for the whole row, and the bisection would
        // collapse silently onto its lower bracket end instead of the optimum. An f32 softmax
        // underflows to exactly zero about 103 logits below the mode, so the mass is
        // reachable; the coincidence is not, which is exactly why it must be closed
        // structurally rather than left to chance.
        //
        // Zeroing a dead bin's RETURN makes its factor identically `1` and its contribution
        // identically `0` at every `f`. Nothing else moves: `longs`/`shorts` already require
        // `live`, so the bounds are unchanged, and the growth sum's dead terms were already
        // `0 * anything`. The host twin [`expected_log_growth`] has always branched on
        // `p <= 0`; this is the tensor path acquiring the same guard, which is what the
        // "identical code path" the scalar wrapper promises actually requires.
        let returns = returns.masked_fill(&live.logical_not(), 0.0);
        // `-1/R` is the bound each outcome imposes: positive-return bins bound `f` from
        // BELOW (a short is ruined by an up move), negative ones from above.
        let bound = returns.reciprocal().neg();
        let longs = returns.gt(0.0).logical_and(&live);
        let shorts = returns.lt(0.0).logical_and(&live);
        let lower = bound
            .masked_fill(&longs.logical_not(), f64::NEG_INFINITY)
            .amax([-1i64].as_slice(), false);
        let upper = bound
            .masked_fill(&shorts.logical_not(), f64::INFINITY)
            .amin([-1i64].as_slice(), false);
        // Both bounds straddle zero, so scaling toward zero moves strictly inside the
        // open domain and leaves an infinite bound infinite.
        // `cap` is the CALLER's cap; `MAX_LEVERAGE` is the declared ceiling no caller can
        // exceed. Taking the min of the two here is what keeps the bound out of the support.
        let cap = cap.min(MAX_LEVERAGE);
        let mut lo = (lower * (1.0 - FEASIBLE_MARGIN)).clamp_min(-cap);
        let mut hi = (upper * (1.0 - FEASIBLE_MARGIN)).clamp_max(cap);

        for _ in 0..SOLVER_ITERATIONS {
            let mid = (&lo + &hi) * 0.5;
            let slope = (&probs * &returns)
                .divide(&(mid.unsqueeze(-1) * &returns + 1.0))
                .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
            let rising = slope.gt(0.0);
            lo = mid.where_self(&rising, &lo);
            hi = hi.where_self(&rising, &mid);
        }
        let fraction = (lo + hi) * 0.5;
        let growth = (&probs * (fraction.unsqueeze(-1) * &returns + 1.0).clamp_min(WEALTH_FLOOR).log())
            .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        // The derived confidence gate: no position unless the optimum strictly grows
        // wealth. `g(0) = 0`, so this fires exactly on a zero-edge law.
        fraction.where_self(&growth.gt(0.0), &growth.zeros_like())
    })
}

/// Scalar convenience over [`kelly_fractions`], on the host. Identical code path, so the
/// baseline that drives the null and the solver the tests pin cannot drift apart.
pub fn kelly_fraction(probs: &[f64], returns: &[f64], cap: f64) -> f64 {
    assert_eq!(probs.len(), returns.len(), "one return per outcome");
    let outcomes = probs.len() as i64;
    let probs = Tensor::from_slice(probs).view([1, outcomes]);
    let returns = Tensor::from_slice(returns).view([1, outcomes]);
    kelly_fractions(&probs, &returns, cap).double_value(&[0])
}

/// The NULL policy's constant position: log-optimal sizing of the train-fitted
/// unconditional law of `r`.
///
/// A function of the fitted support alone. No belief, no latent, no weight — which is
/// exactly what makes it the null, and what makes it reproducible from the artifact
/// beside the checkpoint rather than from the checkpoint.
pub fn marginal_position(supports: &BarSupports, cap: f64) -> f64 {
    kelly_fraction(supports.bin_masses(DOF_R), &bin_returns(supports), cap)
}

// ---------------------------------------------------------------------------
// Positions over held-out windows
// ---------------------------------------------------------------------------

/// One held-out window: the realized simple returns, the UNCAPPED optimum, and every
/// policy's position path at the headline cap.
#[derive(Clone, Debug)]
pub struct WindowPaths {
    /// `R_t = exp(r_t) - 1` of the bar each decision is paid on.
    pub realized: Vec<f64>,
    /// The model's uncapped log-optimal fraction, per bar.
    ///
    /// Retained because every capped and fractional variant is a CLAMP of it: the cap
    /// curve, half Kelly and quarter Kelly all fall out of this vector with no second
    /// solve, and the histogram of `|f*|` is the direct measurement of how often the cap,
    /// rather than the distribution, chose the size. Concavity is what makes the identity
    /// exact: `g` is strictly increasing below its unique maximizer, so the constrained
    /// optimum on `[-cap, cap]` is `clamp(f*, -cap, cap)`, and the strictly-positive-growth
    /// gate survives the clamp because `g(f) > g(0) = 0` for every `f` strictly between `0`
    /// and `f*`.
    pub free: Vec<f64>,
    /// Fraction of wealth held INTO bar `t`, per policy, at the headline cap.
    pub positions: [Vec<f64>; POLICY_COUNT],
    /// `E[r | strictly past bars]` per bar, in LOG-return space, under the same prefix-free
    /// law the position was solved from.
    ///
    /// This is the one number the Kelly size is almost entirely a function of, and the only
    /// one a likelihood is nearly blind to, so it is retained per bar rather than reduced:
    /// [`mean_calibration`] regresses the realized `r` on it, which is the only way to see
    /// whether the traded mean is inflated. Empty on windows built by the accounting-only
    /// constructor.
    pub predicted_mean: Vec<f64>,
    /// `Var[r | strictly past bars]` per bar, in LOG-return space, from the same law.
    ///
    /// Regressed against the realized squared residual by [`mean_calibration`]. A mean that
    /// is inflated while the variance is honest is a fixable sizing error; both wrong is a
    /// different finding, so the two are measured separately rather than pooled into one
    /// "calibration" figure.
    pub predicted_var: Vec<f64>,
    /// The uncapped log-optimal fraction under the RECALIBRATED mean, when the pass was
    /// asked for one ([`MeanShrink`]). `None` on every ordinary bench, which is why the
    /// existing policies cannot move when this is added.
    pub free_shrunk: Option<Vec<f64>>,
    /// Probability mass the law put in the two CATCH-ALL bins of `r`, per bar.
    ///
    /// The discriminator between a decode artifact and a learned error: the artifact's damage is
    /// proportional to this mass, so a flat ~1.45% across every name is the marginal's own
    /// equal-mass construction showing through, while tens of ppm on quiet names and percent on
    /// loud ones would be something the model learned. Empty when the pass did not form it.
    pub outer_mass: Vec<f64>,
    /// Upper catch-all mass minus lower, per bar. Only the NET moves `mu_hat`.
    pub outer_signed: Vec<f64>,
    /// `E[r | past]` and `Var[r | past]` with those two bins zeroed and the row renormalized.
    ///
    /// Same law, same pass, same bars - only the catch-alls removed. Refitting the calibration
    /// against these answers whether the miscalibration survives the decode convention.
    pub trimmed_mean: Vec<f64>,
    pub trimmed_var: Vec<f64>,
}

impl WindowPaths {
    pub fn bars(&self) -> usize {
        self.realized.len()
    }

    /// A window carrying positions and nothing else.
    ///
    /// For the accounting tests and fixtures, which synthesize position paths directly and
    /// have no predictive law to take a conditional mean from. Named rather than defaulted
    /// so that a caller who forgets the calibration inputs gets an empty calibration block
    /// deliberately instead of a fitted-on-zeros regression by accident.
    pub fn unmeasured(
        realized: Vec<f64>,
        free: Vec<f64>,
        positions: [Vec<f64>; POLICY_COUNT],
    ) -> Self {
        Self {
            realized,
            free,
            positions,
            predicted_mean: Vec::new(),
            predicted_var: Vec::new(),
            free_shrunk: None,
            outer_mass: Vec::new(),
            outer_signed: Vec::new(),
            trimmed_mean: Vec::new(),
            trimmed_var: Vec::new(),
        }
    }

    /// Realized `r` in LOG space, the regressand of the calibration fit. `realized` is the
    /// simple return `exp(r) - 1`, so this inverts exactly the transform that produced it.
    pub fn realized_log(&self) -> Vec<f64> {
        self.realized.iter().map(|r| r.ln_1p()).collect()
    }

    /// True when this window carries the conditional moments the calibration fit needs.
    pub fn has_moments(&self) -> bool {
        self.predicted_mean.len() == self.realized.len()
            && self.predicted_var.len() == self.realized.len()
    }

    /// True when the catch-all decomposition was formed for every bar of this window.
    pub fn has_decomposition(&self) -> bool {
        self.outer_mass.len() == self.realized.len()
            && self.outer_signed.len() == self.realized.len()
            && self.trimmed_mean.len() == self.realized.len()
            && self.trimmed_var.len() == self.realized.len()
    }
}

/// Per-window exceedance counts of the model's own far-tail quantiles.
///
/// One entry per traded window per level, so the exceedance RATE can be blocked and
/// bootstrapped like every other number here: extreme moves cluster in time, so an iid
/// binomial interval over 500k bars is a floor on the uncertainty rather than an estimate
/// of it.
#[derive(Clone, Debug)]
pub struct TailCounts {
    /// `lower[level][window]` = bars whose realized `r` fell BELOW the model's own
    /// `TAIL_LEVELS[level]` quantile for that bar.
    pub lower: [Vec<f64>; TAIL_LEVELS.len()],
    /// The upper mirror, against the `1 - TAIL_LEVELS[level]` quantile.
    pub upper: [Vec<f64>; TAIL_LEVELS.len()],
    /// Bars per window, the denominator of every rate.
    pub bars: Vec<f64>,
}

impl Default for TailCounts {
    fn default() -> Self {
        Self::empty()
    }
}

impl TailCounts {
    pub fn empty() -> Self {
        Self {
            lower: std::array::from_fn(|_| Vec::new()),
            upper: std::array::from_fn(|_| Vec::new()),
            bars: Vec::new(),
        }
    }

    pub fn windows(&self) -> usize {
        self.bars.len()
    }

    pub fn absorb(&mut self, other: TailCounts) {
        for level in 0..TAIL_LEVELS.len() {
            self.lower[level].extend_from_slice(&other.lower[level]);
            self.upper[level].extend_from_slice(&other.upper[level]);
        }
        self.bars.extend_from_slice(&other.bars);
    }

    /// Truncate to the first `windows` windows, so the tail block covers exactly the
    /// windows the bench traded and its block ids line up by the same truncation.
    pub fn truncate(&mut self, windows: usize) {
        for level in 0..TAIL_LEVELS.len() {
            self.lower[level].truncate(windows);
            self.upper[level].truncate(windows);
        }
        self.bars.truncate(windows);
    }
}

/// The parts of the bench that belong to the ARTIFACT rather than to a chunk of windows:
/// the support's bin returns and value bounds, and the unconditional null's uncapped
/// fraction. Built once per evaluation so a 170-chunk pass does not re-derive them 170
/// times, and so the null is provably one number for the whole pass.
#[derive(Debug)]
pub struct TradeSetup {
    returns: Tensor,
    /// `[1, NUM_BAR_BINS]` bin CENTERS of the `r` support, in LOG-return space.
    ///
    /// The simple returns above are `exp(center) - 1`, so this is not redundant: the
    /// conditional MEAN the calibration fit regresses against is a mean of `r` itself, and
    /// `E[exp(r) - 1]` is a different number by Jensen. Both are kept so neither consumer
    /// has to invert the other's transform.
    centers: Tensor,
    /// `[1, NUM_BAR_BINS]` value bounds of each `r` bin, in LOG-return space, for the
    /// tail quantiles. Atoms have `lo == hi`, which makes their quantile the atom itself.
    lo: Tensor,
    hi: Tensor,
    /// The null's UNCAPPED fraction, so the null can be re-clamped at every cap on the
    /// curve instead of being frozen at the headline cap.
    free_marginal: f64,
    cap: f64,
    /// The post-hoc mean recalibration to ALSO solve, when one was fitted elsewhere.
    ///
    /// `None` on every ordinary bench, and the existing policies never read it, so the
    /// headline numbers of a run are bit-identical whether or not it is set.
    shrink: Option<MeanShrink>,
}

impl TradeSetup {
    pub fn new(supports: &BarSupports, device: Device, cap: f64) -> Self {
        let returns = bin_returns(supports);
        let row = |values: &[f64]| {
            Tensor::from_slice(values)
                .view([1, NUM_BAR_BINS])
                .to_device(device)
        };
        Self {
            returns: row(&returns),
            centers: row(supports.centers(DOF_R)),
            lo: row(supports.lower_bounds(DOF_R)),
            hi: row(supports.upper_bounds(DOF_R)),
            // Solved uncapped: the headline cap is applied by clamping, so one number
            // serves every point of the cap curve.
            free_marginal: kelly_fraction(supports.bin_masses(DOF_R), &returns, FREE_LEVERAGE),
            cap,
            shrink: None,
        }
    }

    /// Also solve the log-optimal fraction under a recalibrated conditional mean.
    ///
    /// A builder rather than an argument of [`Self::new`] because the recalibration is
    /// fitted on data — on a slice DISJOINT from the one this setup will be evaluated on —
    /// and therefore cannot exist at the time the artifact-level table is built.
    pub fn with_shrink(mut self, shrink: Option<MeanShrink>) -> Self {
        self.shrink = shrink;
        self
    }

    /// The null policy's constant position at the headline cap.
    pub fn marginal_position(&self) -> f64 {
        clamp_fraction(self.free_marginal, self.cap)
    }

    /// The null's UNCAPPED fraction, which is what the cap curve re-clamps.
    pub fn free_marginal(&self) -> f64 {
        self.free_marginal
    }

    pub fn leverage_cap(&self) -> f64 {
        self.cap
    }

    /// Positions and tail exceedances over the first `windows` windows of one
    /// teacher-forced chunk.
    ///
    /// `realized_dof` is the `[windows, bars, BAR_DOF]` realized continuation the beliefs
    /// predict. Its `r` column is selected HERE rather than by the caller, so no call site
    /// can hand the bench the wrong degree of freedom, and no other column is ever read.
    pub fn paths(
        &self,
        head: &BarEmissionHead,
        beliefs: &Tensor,
        realized_dof: &Tensor,
        windows: usize,
    ) -> Result<ChunkPaths> {
        let available = beliefs.size()[0];
        let take = (windows as i64).min(available);
        if take <= 0 {
            return Ok(ChunkPaths::empty());
        }
        window_paths(
            head,
            &beliefs.narrow(0, 0, take),
            &realized_dof.narrow(0, 0, take).select(-1, DOF_R as i64),
            &TradedLaw {
                returns: &self.returns,
                centers: &self.centers,
                bounds: Some((&self.lo, &self.hi)),
                shrink: self.shrink,
            },
            self.free_marginal,
            self.cap,
        )
    }
}

/// Everything about the traded law that is a property of the ARTIFACT rather than of a
/// chunk of bars, in the form [`window_paths`] consumes.
///
/// A struct rather than positional arguments because two of the fields are `[1, 128]`
/// tensors over the same support that differ only by a transform, and a caller who swapped
/// them would get a plausible wrong answer rather than a type error.
#[derive(Debug)]
pub struct TradedLaw<'a> {
    /// `[1, NUM_BAR_BINS]` SIMPLE return of each `r` bin, `exp(center) - 1`.
    pub returns: &'a Tensor,
    /// `[1, NUM_BAR_BINS]` LOG-space center of each `r` bin.
    pub centers: &'a Tensor,
    /// `[1, NUM_BAR_BINS]` log-space value bounds, for the tail quantiles. Without them the
    /// positions are still produced and the tail block comes back empty, which is what the
    /// solver-only tests want.
    pub bounds: Option<(&'a Tensor, &'a Tensor)>,
    /// A post-hoc mean recalibration to solve a SECOND uncapped optimum under.
    pub shrink: Option<MeanShrink>,
}

impl<'a> TradedLaw<'a> {
    /// The law with no tail bounds and no recalibration: the solver-only shape.
    pub fn new(returns: &'a Tensor, centers: &'a Tensor) -> Self {
        Self {
            returns,
            centers,
            bounds: None,
            shrink: None,
        }
    }

    pub fn with_bounds(mut self, lo: &'a Tensor, hi: &'a Tensor) -> Self {
        self.bounds = Some((lo, hi));
        self
    }

    pub fn with_shrink(mut self, shrink: MeanShrink) -> Self {
        self.shrink = Some(shrink);
        self
    }
}

/// `clamp(f, -cap, cap)`, the constrained optimum of a concave `g` whose free maximizer is
/// `f`. Written once so the cap curve, the fractional policies and the headline agree by
/// construction rather than by two implementations that look the same.
pub fn clamp_fraction(free: f64, cap: f64) -> f64 {
    free.clamp(-cap, cap)
}

/// One chunk's positions plus the tail exceedances measured on the same probabilities.
///
/// Both come out of one pass because the object they are properties of — the traded `r`
/// law — is shared: computing the tail calibration separately would double the cost of the
/// bench to answer a question about the same distribution.
#[derive(Clone, Debug, Default)]
pub struct ChunkPaths {
    pub windows: Vec<WindowPaths>,
    pub tail: TailCounts,
}

impl ChunkPaths {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Append another chunk, keeping positions and tail counts in the same window order.
    ///
    /// The two vectors are extended together and never separately, which is what lets the
    /// bench truncate both by one window count and know the block ids still line up.
    pub fn absorb(&mut self, other: ChunkPaths) {
        self.windows.extend(other.windows);
        self.tail.absorb(other.tail);
    }

    /// Truncate positions and tail counts to the same first `windows` windows.
    pub fn truncate(&mut self, windows: usize) {
        self.windows.truncate(windows);
        self.tail.truncate(windows);
    }
}

/// Positions, conditional moments and tail exceedances for one chunk of pinned windows.
///
/// `beliefs` is `[windows, bars, latent_dim]` and `realized_r` is `[windows, bars]`, the
/// realized `r` of the bar each belief predicts. The traded decision is computed from
/// `beliefs` alone: `realized_r` reaches only the payoff, the perfect-foresight oracle
/// (the one policy allowed to see it) and the tail-calibration COUNT, which is an outcome
/// rather than a decision.
///
/// The Kelly solve runs ONCE per bar, uncapped. Every policy is then a clamp of that one
/// number, which is exact by concavity and is what makes the cap curve free. When
/// [`TradedLaw::shrink`] is set a SECOND uncapped optimum is solved, under the same
/// probabilities with the conditional mean recalibrated — see [`MeanShrink`] for why that is
/// a shift of the support rather than a scaling of the fraction.
///
/// The conditional mean and variance of `r` come out of the same probabilities, at the cost
/// of two reductions over an object that is already materialized. They are what
/// [`mean_calibration`] regresses, and taking them here is what makes the calibration
/// diagnostic free at every validation instead of a second pass over the corpus.
pub fn window_paths(
    head: &BarEmissionHead,
    beliefs: &Tensor,
    realized_r: &Tensor,
    law: &TradedLaw<'_>,
    free_marginal: f64,
    cap: f64,
) -> Result<ChunkPaths> {
    let shape = beliefs.size();
    ensure!(
        shape.len() == 3,
        "beliefs must be [windows, bars, latent_dim], got {shape:?}"
    );
    let (windows, bars, latent) = (shape[0], shape[1], shape[2]);
    ensure!(
        latent == head.latent_dim(),
        "beliefs carry {latent} features but the head expects {}",
        head.latent_dim()
    );
    ensure!(
        realized_r.size() == [windows, bars],
        "realized returns must align with the beliefs that predict them: {:?} vs \
         [{windows}, {bars}]",
        realized_r.size()
    );

    let rows = windows * bars;
    let flat_beliefs = beliefs.reshape([rows, latent]);
    let flat_r = realized_r.reshape([rows]).to_kind(Kind::Double);
    let realized = flat_r.expm1();
    let centers = law.centers.to_kind(Kind::Double);

    let mut free = Vec::with_capacity(rows as usize);
    let mut free_shrunk = Vec::with_capacity(if law.shrink.is_some() {
        rows as usize
    } else {
        0
    });
    let mut predicted_mean = Vec::with_capacity(rows as usize);
    let mut predicted_var = Vec::with_capacity(rows as usize);
    // CATCH-ALL DECOMPOSITION, formed in the same pass because it needs the same `rows x 128`
    // probabilities the moments come from and a second pass would be a second population.
    //
    // An equal-mass support's outermost bins are catch-alls for everything past the clip, and
    // they decode to the CLIPPED BOUND rather than to a fitted interior value - `-883.32` and
    // `+880.38` bps on the live 300s support against fitted conditional means near `+/-280`.
    // Every moment read off `centers` therefore prices roughly 1.45% of the mass three times
    // too far out. Zeroing those two bins and renormalizing removes that contribution by
    // construction, so the pair of fits (`mu` versus `mu` with the catch-alls dropped) is what
    // separates a decode artifact from a learned error in the conditional mean.
    //
    // Dropping rather than re-centring on purpose: re-centring would need the fitted per-bin
    // means, which live in the support artifact and are being added there. Dropping is
    // available today, needs no schema, and BOUNDS the artifact - it removes the whole outer
    // contribution rather than shrinking it, so a slope that does not move under it cannot be
    // rescued by a better decode either.
    let mut outer_mass = Vec::with_capacity(rows as usize);
    // SIGNED net, upper catch-all minus lower. The two are separate quantities with separate
    // consequences: the TOTAL drives the variance artifact, which is symmetric in the decode
    // error, while only the NET moves the MEAN - a symmetric pair of catch-alls at +/-880 bps
    // leaves `mu_hat` untouched no matter how much mass they hold.
    let mut outer_signed = Vec::with_capacity(rows as usize);
    let mut trimmed_mean = Vec::with_capacity(rows as usize);
    let mut trimmed_var = Vec::with_capacity(rows as usize);
    let interior = {
        let mut keep = vec![1.0f64; NUM_BAR_BINS as usize];
        keep[0] = 0.0;
        keep[NUM_BAR_BINS as usize - 1] = 0.0;
        Tensor::from_slice(&keep)
            .view([1, NUM_BAR_BINS as i64])
            .to_device(centers.device())
    };
    let mut exceed_lower: [Vec<f64>; TAIL_LEVELS.len()] = std::array::from_fn(|_| Vec::new());
    let mut exceed_upper: [Vec<f64>; TAIL_LEVELS.len()] = std::array::from_fn(|_| Vec::new());
    let mut start = 0i64;
    while start < rows {
        let len = ROW_CHUNK.min(rows - start);
        let chunk = flat_beliefs.narrow(0, start, len);
        let probs = forecast_r_probs(head, &chunk);
        free.extend(host_vec(&kelly_fractions(&probs, law.returns, FREE_LEVERAGE)));
        // The moments of `r` itself, not of the simple return: the calibration fit regresses
        // the realized LOG return, and `E[exp(r) - 1] != exp(E[r]) - 1`.
        let mass = probs
            .to_kind(Kind::Double)
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Double)
            .clamp_min(f64::MIN_POSITIVE);
        let normalized = probs.to_kind(Kind::Double).divide(&mass);
        let mu = (&normalized * &centers).sum_dim_intlist([-1i64].as_slice(), true, Kind::Double);
        let deviation = &centers - &mu;
        let var = (&normalized * &deviation * &deviation)
            .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        predicted_mean.extend(host_vec(&mu.reshape([-1])));
        predicted_var.extend(host_vec(&var));
        let lower_outer = normalized.select(-1, 0);
        let upper_outer = normalized.select(-1, NUM_BAR_BINS as i64 - 1);
        outer_mass.extend(host_vec(&(&lower_outer + &upper_outer)));
        outer_signed.extend(host_vec(&(&upper_outer - &lower_outer)));
        let interior_probs = &normalized * &interior;
        let interior_mass = interior_probs
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Double)
            .clamp_min(f64::MIN_POSITIVE);
        let interior_probs = interior_probs.divide(&interior_mass);
        let interior_mu = (&interior_probs * &centers)
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Double);
        let interior_deviation = &centers - &interior_mu;
        let interior_variance = (&interior_probs * &interior_deviation * &interior_deviation)
            .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        trimmed_mean.extend(host_vec(&interior_mu.reshape([-1])));
        trimmed_var.extend(host_vec(&interior_variance));
        if let Some(shrink) = law.shrink {
            // Shifting every bin's LOG value by `d` shifts the law's mean by exactly `d` and
            // leaves every central moment untouched, so this recalibrates the one quantity
            // the fit found miscalibrated and nothing else: `1 + R'_b = (1 + R_b) exp(d)`.
            //
            // Written as `R exp(d) + expm1(d)` rather than as `(1 + R) exp(d) - 1`, which is
            // the same identity with the cancellation removed. A 5-minute bar's `R` is of
            // order `1e-3` and `d` of order `1e-4`, so forming `1 + R` and subtracting one
            // again discards ten bits of a quantity the Kelly solve then differentiates. It
            // also makes the identity recalibration exactly the identity — `exp(0) = 1` and
            // `expm1(0) = 0` are both exact — which is what lets a run's headline numbers be
            // bit-identical whether or not a recalibration was requested.
            let d = &mu * (shrink.beta - 1.0) + shrink.alpha;
            let returns = law.returns.to_kind(Kind::Double);
            let shifted = &returns * d.exp() + d.expm1();
            free_shrunk.extend(host_vec(&kelly_fractions(&probs, &shifted, FREE_LEVERAGE)));
        }
        if let Some((lo, hi)) = law.bounds {
            let realized_chunk = flat_r.narrow(0, start, len);
            for (level, q) in TAIL_LEVELS.iter().enumerate() {
                let below = predicted_quantile(&probs, lo, hi, *q);
                let above = predicted_quantile(&probs, lo, hi, 1.0 - *q);
                exceed_lower[level].extend(host_vec(
                    &realized_chunk.lt_tensor(&below).to_kind(Kind::Double),
                ));
                exceed_upper[level].extend(host_vec(
                    &realized_chunk.gt_tensor(&above).to_kind(Kind::Double),
                ));
            }
        }
        start += len;
    }
    let realized = host_vec(&realized);

    let bars = bars as usize;
    let measured_tail = law.bounds.is_some();
    let shrunk = law.shrink.is_some();
    let mut tail = TailCounts::empty();
    let paths = (0..windows as usize)
        .map(|window| {
            let span = window * bars..(window + 1) * bars;
            let free_window = free[span.clone()].to_vec();
            let realized_window = realized[span.clone()].to_vec();
            if measured_tail {
                tail.bars.push(bars as f64);
                for level in 0..TAIL_LEVELS.len() {
                    tail.lower[level].push(exceed_lower[level][span.clone()].iter().sum());
                    tail.upper[level].push(exceed_upper[level][span.clone()].iter().sum());
                }
            }
            let staked = |multiple: f64| -> Vec<f64> {
                free_window
                    .iter()
                    .map(|f| clamp_fraction(multiple * f, cap))
                    .collect()
            };
            WindowPaths {
                realized: realized_window,
                positions: [
                    staked(POLICY_KELLY_MULTIPLE[POLICY_MODEL]),
                    staked(POLICY_KELLY_MULTIPLE[POLICY_HALF]),
                    staked(POLICY_KELLY_MULTIPLE[POLICY_QUARTER]),
                    vec![clamp_fraction(free_marginal, cap); bars],
                    vec![1.0; bars],
                    // Perfect foresight on a point mass: `g(f) = ln(1 + f R)` rises without
                    // bound along the realized sign, so the optimum is the cap itself, and
                    // a flat bar earns no position because `g` never becomes positive.
                    realized[span.clone()]
                        .iter()
                        .map(|r| cap * r.signum() * f64::from(*r != 0.0))
                        .collect(),
                ],
                free: free_window,
                predicted_mean: predicted_mean[span.clone()].to_vec(),
                predicted_var: predicted_var[span.clone()].to_vec(),
                free_shrunk: shrunk.then(|| free_shrunk[span.clone()].to_vec()),
                outer_mass: outer_mass[span.clone()].to_vec(),
                outer_signed: outer_signed[span.clone()].to_vec(),
                trimmed_mean: trimmed_mean[span.clone()].to_vec(),
                trimmed_var: trimmed_var[span].to_vec(),
            }
        })
        .collect();
    Ok(ChunkPaths { windows: paths, tail })
}

/// Per-row `q`-quantile of `r` under each row's own predictive law, in log-return space.
///
/// The predictive law is a MIXED measure: an atom bin holds its mass at a single value, a
/// continuous bin spreads it uniformly across `[lo_b, hi_b]`. The quantile therefore walks
/// the CDF to the bin that contains `q` and interpolates linearly INSIDE that bin, which is
/// exactly the uniform-within-bin assumption the `density` scoring rule already integrates
/// under — so this diagnostic tests the same object the likelihood scored, not a
/// convenient approximation of it. An atom has `lo == hi` and the interpolation collapses
/// onto the atom's own value with no special case.
pub fn predicted_quantile(probs: &Tensor, lo: &Tensor, hi: &Tensor, q: f64) -> Tensor {
    assert!(q > 0.0 && q < 1.0, "a quantile probability must be interior");
    tch::no_grad(|| {
        let probs = probs.to_kind(Kind::Double);
        let mass = probs
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Double)
            .clamp_min(f64::MIN_POSITIVE);
        let probs = probs.divide(&mass);
        let cdf = probs.cumsum(-1, Kind::Double);
        // Bins strictly below `q` IS the index of the bin that contains it.
        let index = cdf
            .lt(q)
            .to_kind(Kind::Int64)
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Int64)
            .clamp(0, NUM_BAR_BINS - 1);
        let first = index.eq(0);
        let previous = cdf
            .gather(-1, &(&index - 1).clamp_min(0), false)
            .masked_fill(&first, 0.0);
        let inside = probs.gather(-1, &index, false).clamp_min(f64::MIN_POSITIVE);
        let share = ((q - previous) / inside).clamp(0.0, 1.0);
        let rows = probs.size()[0];
        let expand = |bounds: &Tensor| {
            bounds
                .to_kind(Kind::Double)
                .expand([rows, NUM_BAR_BINS], false)
                .gather(-1, &index, false)
        };
        let low = expand(lo);
        let span = expand(hi) - &low;
        (low + share * span).squeeze_dim(-1)
    })
}

fn host_vec(tensor: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(tensor.to_kind(Kind::Double).contiguous().view([-1]))
        .expect("a 1-D f64 tensor converts to a host vector")
}

// ---------------------------------------------------------------------------
// Realized accounting
// ---------------------------------------------------------------------------

/// One window's traded notional, split by whether the MODEL chose the trade.
///
/// Both are raw sums over the window rather than per-bar means, because the consumer that
/// needs them is a turnover-weighted cost `sum(c_i tau_i) / sum(c_i)`, whose weights are
/// notional and not rates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowTurnover {
    /// Every unit of notional the ledger charged, the audit trail against [`PolicyStats`].
    pub total: f64,
    /// `total` less the entry from flat and the terminal unwind.
    ///
    /// Those two trades are a fixed `|f_first| + |f_last|` per window placed by the window
    /// SAMPLER rather than by the model - a window boundary is not a decision. They are close
    /// to uniform across names while real turnover is not, so they dilute exactly the
    /// concentration a turnover-weighted cost exists to detect and drag the composite toward
    /// the equal-weighted mean. This is the honest weight; `total` is what reconciles.
    pub interior: f64,
    pub bars: usize,
}

/// One policy's realized per-bar accounting over every traded window, flattened.
///
/// Gross growth and traded notional are cost-INDEPENDENT, which is what makes the cost
/// curve and the break-even search cheap: each cost level is one pass over these two
/// vectors rather than a re-run of the policy.
#[derive(Clone, Debug)]
struct Ledger {
    /// `ln(1 + f_t R_t)`, floored at [`WEALTH_FLOOR`].
    gross: Vec<f64>,
    /// `|f_t - f_{t-1}|`, entering flat at each window's first bar and unwinding to flat
    /// after its last, so the closing trade is paid for.
    traded: Vec<f64>,
    /// `starts[w]..starts[w + 1]` is window `w`.
    starts: Vec<usize>,
    hits: f64,
    positioned: f64,
    abs_position: f64,
    clamped: f64,
    ruin: usize,
    /// Per-window traded notional, in the same window order as `starts`.
    turnover: Vec<WindowTurnover>,
}

impl Ledger {
    fn build(windows: &[WindowPaths], policy: usize, cap: f64) -> Self {
        let bars: usize = windows.iter().map(WindowPaths::bars).sum();
        let mut ledger = Self {
            gross: Vec::with_capacity(bars),
            traded: Vec::with_capacity(bars),
            starts: Vec::with_capacity(windows.len() + 1),
            hits: 0.0,
            positioned: 0.0,
            abs_position: 0.0,
            clamped: 0.0,
            ruin: 0,
            turnover: Vec::with_capacity(windows.len()),
        };
        let clamp_edge = cap * (1.0 - 1e-6);
        for window in windows {
            ledger.starts.push(ledger.gross.len());
            let positions = &window.positions[policy];
            let mut held = 0.0f64;
            for (bar, (fraction, realized)) in positions.iter().zip(&window.realized).enumerate() {
                let multiplier = 1.0 + fraction * realized;
                if multiplier <= WEALTH_FLOOR {
                    ledger.ruin += 1;
                }
                ledger.gross.push(multiplier.max(WEALTH_FLOOR).ln());
                let unwind = if bar + 1 == positions.len() {
                    fraction.abs()
                } else {
                    0.0
                };
                let traded = (fraction - held).abs() + unwind;
                ledger.traded.push(traded);
                held = *fraction;
                ledger.abs_position += fraction.abs();
                if *fraction != 0.0 {
                    ledger.positioned += 1.0;
                    if fraction * realized > 0.0 {
                        ledger.hits += 1.0;
                    }
                }
                if fraction.abs() >= clamp_edge {
                    ledger.clamped += 1.0;
                }
            }
            // Exact rather than re-derived: `traded` at the window's first bar IS the entry
            // from flat, and `unwind` at its last IS the terminal exit, so subtracting the two
            // endpoints leaves precisely the interior rebalances the model asked for.
            let entry = positions.first().map_or(0.0, |f| f.abs());
            let exit = positions.last().map_or(0.0, |f| f.abs());
            let total: f64 = ledger.traded[*ledger.starts.last().expect("a window was pushed")..]
                .iter()
                .sum();
            ledger.turnover.push(WindowTurnover {
                total,
                interior: (total - entry - exit).max(0.0),
                bars: positions.len(),
            });
        }
        ledger.starts.push(ledger.gross.len());
        ledger
    }

    fn bars(&self) -> usize {
        self.gross.len()
    }

    /// `ln(1 - cost * traded)`, the log wealth a bar's rebalance costs. Negative.
    fn cost_of(&self, bar: usize, cost: f64) -> f64 {
        (1.0 - cost * self.traded[bar]).max(WEALTH_FLOOR).ln()
    }

    fn net_growth_per_bar(&self, cost: f64) -> f64 {
        if self.gross.is_empty() {
            return f64::NAN;
        }
        let total: f64 = (0..self.bars())
            .map(|bar| self.gross[bar] + self.cost_of(bar, cost))
            .sum();
        total / self.bars() as f64
    }

    /// Per-window net log growth per bar, the paired unit the bootstrap resamples.
    fn window_growth(&self, cost: f64) -> Vec<f64> {
        self.starts
            .windows(2)
            .map(|span| {
                let (from, to) = (span[0], span[1]);
                if to <= from {
                    return f64::NAN;
                }
                let total: f64 = (from..to)
                    .map(|bar| self.gross[bar] + self.cost_of(bar, cost))
                    .sum();
                total / (to - from) as f64
            })
            .collect()
    }

    fn stats(&self, cost: f64) -> PolicyStats {
        let bars = self.bars();
        if bars == 0 {
            return PolicyStats::nan();
        }
        let count = bars as f64;
        let mut gross_total = 0.0;
        let mut net_total = 0.0;
        let mut net_squares = 0.0;
        for bar in 0..bars {
            let net = self.gross[bar] + self.cost_of(bar, cost);
            gross_total += self.gross[bar];
            net_total += net;
            net_squares += net * net;
        }
        let mean = net_total / count;
        let variance = (net_squares / count - mean * mean).max(0.0);
        let sd = variance.sqrt();
        let sharpe = if sd > 0.0 {
            mean / sd * BARS_PER_YEAR.sqrt()
        } else {
            f64::NAN
        };

        // Drawdown is a property of a single equity path, so it is measured inside each
        // window and never across the seam between two unrelated symbols.
        let mut mean_drawdowns = 0.0;
        let mut max_drawdown = 0.0f64;
        for span in self.starts.windows(2) {
            let (from, to) = (span[0], span[1]);
            if to <= from {
                continue;
            }
            let mut equity = 0.0f64;
            let mut peak = 0.0f64;
            let mut sum = 0.0f64;
            let mut worst = 0.0f64;
            for bar in from..to {
                equity += self.gross[bar] + self.cost_of(bar, cost);
                peak = peak.max(equity);
                let drawdown = peak - equity;
                sum += drawdown;
                worst = worst.max(drawdown);
            }
            mean_drawdowns += sum / (to - from) as f64;
            max_drawdown = max_drawdown.max(worst);
        }
        let windows = (self.starts.len() - 1) as f64;

        PolicyStats {
            gross_growth: gross_total / count,
            net_growth: mean,
            sharpe,
            hit_rate: if self.positioned > 0.0 {
                self.hits / self.positioned
            } else {
                f64::NAN
            },
            turnover: self.traded.iter().sum::<f64>() / count,
            time_in_market: self.positioned / count,
            mean_abs_position: self.abs_position / count,
            clamped_fraction: self.clamped / count,
            mean_drawdown: wealth_fraction(mean_drawdowns / windows),
            max_drawdown: wealth_fraction(max_drawdown),
            ruin_bars: self.ruin,
        }
    }
}

/// A log-space loss restated as the wealth fraction it destroys.
fn wealth_fraction(nats: f64) -> f64 {
    -(-nats).exp_m1()
}

/// One policy's realized performance. Growth figures are natural log growth PER BAR;
/// drawdowns are wealth fractions.
#[derive(Clone, Copy, Debug)]
pub struct PolicyStats {
    /// Mean `ln(1 + f R)` per bar, before costs.
    pub gross_growth: f64,
    /// The same after charging the bench's cost on realized traded notional.
    pub net_growth: f64,
    /// Annualized Sharpe of the net per-bar log returns.
    pub sharpe: f64,
    /// Share of POSITIONED bars whose position had the realized move's sign.
    pub hit_rate: f64,
    /// Mean traded notional per bar, `|f_t - f_{t-1}|`.
    pub turnover: f64,
    /// Share of bars holding any position at all.
    pub time_in_market: f64,
    pub mean_abs_position: f64,
    /// Share of bars where the leverage cap, not the distribution, chose the size.
    pub clamped_fraction: f64,
    /// Mean and worst peak-to-trough loss inside a window, as a wealth fraction.
    pub mean_drawdown: f64,
    pub max_drawdown: f64,
    /// Bars whose realized move would have wiped the position out. The realized return
    /// can fall outside the fitted support, so a leveraged Kelly position sized on the
    /// model's own law is not proof against it.
    pub ruin_bars: usize,
}

impl PolicyStats {
    pub fn nan() -> Self {
        Self {
            gross_growth: f64::NAN,
            net_growth: f64::NAN,
            sharpe: f64::NAN,
            hit_rate: f64::NAN,
            turnover: f64::NAN,
            time_in_market: f64::NAN,
            mean_abs_position: f64::NAN,
            clamped_fraction: f64::NAN,
            mean_drawdown: f64::NAN,
            max_drawdown: f64::NAN,
            ruin_bars: 0,
        }
    }
}

/// One point of the leverage-cap curve: the whole verdict, re-derived at one cap.
///
/// Every field is a re-clamp of the SAME solved fractions, so nothing here is a second
/// experiment. Reading it left to right answers the question a single headline cannot: is
/// the edge a property of the predictive law, which would make it roughly stable in the cap
/// once the cap stops binding, or a property of the cap, which would make it grow with the
/// cap while `clamped_fraction` stays near one and `ruin_bars` climbs.
#[derive(Clone, Copy, Debug)]
pub struct CapPoint {
    pub cap: f64,
    /// Net growth per bar, model minus marginal null, both re-clamped at this cap.
    pub edge: f64,
    pub break_even_bps: f64,
    pub sharpe: f64,
    pub ceiling_capture: f64,
    pub mean_abs_position: f64,
    pub clamped_fraction: f64,
    /// Mean traded notional per bar at this cap.
    ///
    /// The denominator of the break-even cost, and therefore the lever that decides whether
    /// an edge is affordable at all: `break_even = gross_edge / turnover`. Reported at every
    /// cap because re-clamping changes how much the book has to rotate, and a cap that halves
    /// turnover buys twice the cost tolerance for whatever edge survives.
    pub turnover: f64,
    pub max_drawdown: f64,
    pub ruin_bars: usize,
}

impl CapPoint {
    pub fn nan() -> Self {
        Self {
            cap: f64::NAN,
            edge: f64::NAN,
            break_even_bps: f64::NAN,
            sharpe: f64::NAN,
            ceiling_capture: f64::NAN,
            mean_abs_position: f64::NAN,
            clamped_fraction: f64::NAN,
            turnover: f64::NAN,
            max_drawdown: f64::NAN,
            ruin_bars: 0,
        }
    }
}

/// The distribution of the UNCAPPED optimum `f*`, which is what the cap is hiding.
#[derive(Clone, Copy, Debug)]
pub struct FreeKelly {
    /// Share of bars in each `|f*|` bucket of [`FREE_KELLY_EDGES`].
    pub histogram: [f64; FREE_KELLY_EDGES.len() - 1],
    /// Share of bars whose uncapped optimum is at or beyond the headline cap. This is the
    /// fraction of decisions the cap, rather than the distribution, actually made.
    pub saturated: f64,
    pub median: f64,
    pub p95: f64,
    /// Mean SIGNED `f*`, so a policy that is merely long-biased is distinguishable from one
    /// that is genuinely switching sides.
    pub mean_signed: f64,
}

impl FreeKelly {
    pub fn nan() -> Self {
        Self {
            histogram: [f64::NAN; FREE_KELLY_EDGES.len() - 1],
            saturated: f64::NAN,
            median: f64::NAN,
            p95: f64::NAN,
            mean_signed: f64::NAN,
        }
    }
}

/// One tail level's calibration: what the model PROMISED against what happened.
#[derive(Clone, Copy, Debug)]
pub struct TailPoint {
    /// The predicted probability the threshold was placed at.
    pub nominal: f64,
    /// Realized exceedance frequency of the model's own per-bar threshold.
    pub realized: f64,
    /// `realized / nominal`. One is perfect; four means the model promised 0.1% and
    /// delivered 0.4%, which is the difference between a survivable position and a wipeout.
    pub ratio: f64,
    /// Wilson 95% interval on `realized` treating bars as independent. A FLOOR on the
    /// uncertainty, not an estimate of it: extreme moves cluster.
    pub wilson: (f64, f64),
    /// 95% interval from the same window blocks the growth interval uses, which is the
    /// honest one.
    pub blocked: (f64, f64),
    pub exceedances: f64,
}

impl TailPoint {
    pub fn nan() -> Self {
        Self {
            nominal: f64::NAN,
            realized: f64::NAN,
            ratio: f64::NAN,
            wilson: (f64::NAN, f64::NAN),
            blocked: (f64::NAN, f64::NAN),
            exceedances: f64::NAN,
        }
    }
}

/// Far-tail calibration of the traded law, the diagnostic no NLL and no central coverage
/// band can produce.
///
/// A likelihood is an average over the bulk: 99.9% of the probability, and therefore
/// essentially all of the code length, sits where nothing dangerous happens. A central
/// 80% coverage figure is a statement about the same bulk. Neither can see the 0.1% tail,
/// and the 0.1% tail is the entire risk of a leveraged position. This block tests, on
/// held-out data, whether the model's own far quantiles are honest.
#[derive(Clone, Copy, Debug)]
pub struct TailCalibration {
    pub lower: [TailPoint; TAIL_LEVELS.len()],
    pub upper: [TailPoint; TAIL_LEVELS.len()],
    pub bars: f64,
    pub windows: usize,
}

impl TailCalibration {
    pub fn nan() -> Self {
        Self {
            lower: [TailPoint::nan(); TAIL_LEVELS.len()],
            upper: [TailPoint::nan(); TAIL_LEVELS.len()],
            bars: 0.0,
            windows: 0,
        }
    }

    pub fn measured(&self) -> bool {
        self.windows > 0
    }

    /// The worst ratio across both tails, and whether it is the LOWER one.
    ///
    /// Reported as one number because that is the number that decides whether the policy is
    /// safe to run: an underweighted left tail is what lets a leveraged Kelly position take
    /// a loss the model believed was impossible.
    pub fn worst(&self) -> (f64, bool) {
        let mut worst = (f64::NAN, false);
        for (point, is_lower) in self
            .lower
            .iter()
            .map(|p| (p, true))
            .chain(self.upper.iter().map(|p| (p, false)))
        {
            if point.ratio.is_finite() && !(point.ratio <= worst.0) {
                worst = (point.ratio, is_lower);
            }
        }
        worst
    }
}

/// Everything the bench measured on one pinned window set.
#[derive(Clone, Copy, Debug)]
pub struct TradeBench {
    pub policies: [PolicyStats; POLICY_COUNT],
    /// Per-window paired difference `policy - marginal null` in net log growth per bar, with
    /// a block-bootstrap 95% interval, for EVERY policy. `edge[POLICY_MODEL]` is THE number;
    /// the fractional-Kelly rows exist because the moment one of them becomes the honest
    /// headline it needs a measured interval rather than a derived one.
    pub edge: [Dispersion; POLICY_COUNT],
    /// Cost, in basis points, at which each policy's `edge` reaches zero. `NAN` when there
    /// is no gross edge to lose, `INFINITY` when cost never removes it (which means the
    /// policy trades no more than the null does).
    pub break_even_bps: [f64; POLICY_COUNT],
    /// `edge / (oracle - marginal)`: each policy's share of the perfect-foresight ceiling.
    pub ceiling_capture: [f64; POLICY_COUNT],
    /// Each policy's `edge` at each level of [`COST_GRID_BPS`].
    pub cost_curve: [[f64; COST_GRID_BPS.len()]; POLICY_COUNT],
    /// The same verdict re-derived at each level of [`CAP_GRID`].
    pub cap_curve: [CapPoint; CAP_GRID.len()],
    /// The distribution of the uncapped optimum, i.e. how much of the decision the cap made.
    pub free_kelly: FreeKelly,
    /// Far-tail calibration of the traded law.
    pub tail: TailCalibration,
    /// Mincer-Zarnowitz calibration of the traded law's conditional MEAN and VARIANCE.
    ///
    /// The tail block above and the likelihood beside it are both nearly blind to the one
    /// quantity a position is a function of. This is not: it regresses the realized `r` on
    /// the predicted conditional mean, and a slope below one is a directly priced statement
    /// that the traded mean is inflated by `1 / beta`.
    pub calibration: MeanCalibration,
    /// Cost the headline net figures were charged at.
    pub cost_bps: f64,
    pub leverage_cap: f64,
    pub bars: usize,
    pub windows: usize,
    pub blocks: usize,
}

impl TradeBench {
    pub fn nan() -> Self {
        Self {
            policies: [PolicyStats::nan(); POLICY_COUNT],
            edge: [Dispersion::nan(); POLICY_COUNT],
            break_even_bps: [f64::NAN; POLICY_COUNT],
            ceiling_capture: [f64::NAN; POLICY_COUNT],
            cost_curve: [[f64::NAN; COST_GRID_BPS.len()]; POLICY_COUNT],
            cap_curve: [CapPoint::nan(); CAP_GRID.len()],
            free_kelly: FreeKelly::nan(),
            tail: TailCalibration::nan(),
            calibration: MeanCalibration::nan(),
            cost_bps: f64::NAN,
            leverage_cap: LEVERAGE_CAP,
            bars: 0,
            windows: 0,
            blocks: 0,
        }
    }

    pub fn measured(&self) -> bool {
        self.bars > 0
    }

    /// The headline edge at the bench's own cost level: the model's net growth per bar
    /// minus the unconditional null's, on identical windows.
    pub fn edge_at_default(&self) -> f64 {
        self.policies[POLICY_MODEL].net_growth - self.policies[POLICY_MARGINAL].net_growth
    }

    /// The model's own row of each per-policy quantity, named rather than indexed, because
    /// every consumer outside this module wants exactly this row.
    pub fn model_edge(&self) -> Dispersion {
        self.edge[POLICY_MODEL]
    }

    pub fn model_break_even(&self) -> f64 {
        self.break_even_bps[POLICY_MODEL]
    }

    pub fn model_capture(&self) -> f64 {
        self.ceiling_capture[POLICY_MODEL]
    }

    pub fn model_cost_curve(&self) -> &[f64; COST_GRID_BPS.len()] {
        &self.cost_curve[POLICY_MODEL]
    }

    /// Growth per bar restated as annualized log growth, which is the unit a reader can
    /// judge. Deliberately NOT exponentiated: the oracle's figure is astronomically
    /// large in wealth terms and would print as infinity.
    pub fn annualized(growth: f64) -> f64 {
        growth * BARS_PER_YEAR
    }

    /// A break-even cost as a reader should see it: the three cases are genuinely different
    /// findings and printing `NaN` or `inf` for two of them loses the distinction.
    fn break_even_text(bps: f64) -> String {
        if bps.is_nan() {
            "n/a (no gross edge)".to_owned()
        } else if bps.is_infinite() {
            "never (trades no more than the null)".to_owned()
        } else {
            format!("{bps:.2} bps")
        }
    }

    /// One console line per policy plus the verdict, in the pretrainer's house style.
    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec!["trade bench: not measured".to_owned()];
        }
        let mut lines = Vec::with_capacity(POLICY_COUNT + 2);
        lines.push(format!(
            "trade bench ({} windows / {} bars, cap {:.1}x, cost {:.2} bps, Kelly on \
             p(r|past), the head's prefix-free r row)",
            self.windows, self.bars, self.leverage_cap, self.cost_bps,
        ));
        for (policy, name) in POLICY_NAMES.iter().enumerate() {
            let stats = &self.policies[policy];
            let edge = self.edge[policy];
            lines.push(format!(
                "  {name:<13} growth {:+.4} bps/bar net ({:+.4} gross, {:+.3} nats/yr), \
                 sharpe {:+.2}, hit {:.3}, |f| {:.2} ({:.0}% capped), turnover {:.3}/bar, \
                 in market {:.3}, dd mean {:.4} max {:.4}{}",
                stats.net_growth * 1e4,
                stats.gross_growth * 1e4,
                Self::annualized(stats.net_growth),
                stats.sharpe,
                stats.hit_rate,
                stats.mean_abs_position,
                100.0 * stats.clamped_fraction,
                stats.turnover,
                stats.time_in_market,
                stats.mean_drawdown,
                stats.max_drawdown,
                if stats.ruin_bars > 0 {
                    format!(", {} RUINED bars", stats.ruin_bars)
                } else {
                    String::new()
                },
            ));
            // Every policy's own paired edge, so a fractional-Kelly row can be quoted as the
            // headline without anyone having to derive its interval by hand. The null's row
            // is a difference against itself and is omitted rather than printed as zeros.
            if policy != POLICY_MARGINAL {
                lines.push(format!(
                    "  {:<13} edge {:+.4} bps/bar (95% CI {:+.4}..{:+.4}), break-even {}, \
                     captures {:.1}% of the ceiling{}",
                    "",
                    edge.mean * 1e4,
                    edge.ci_low * 1e4,
                    edge.ci_high * 1e4,
                    Self::break_even_text(self.break_even_bps[policy]),
                    100.0 * self.ceiling_capture[policy],
                    if edge.ci_low > 0.0 {
                        ""
                    } else {
                        " — not resolvable against the null"
                    },
                ));
            }
        }
        let edge = self.model_edge();
        lines.push(format!(
            "  EDGE over the marginal null {:+.4} bps/bar (95% CI {:+.4}..{:+.4} over {} \
             blocks), break-even cost {}, captures {:.1}% of the perfect-foresight \
             ceiling{}",
            edge.mean * 1e4,
            edge.ci_low * 1e4,
            edge.ci_high * 1e4,
            edge.blocks,
            Self::break_even_text(self.model_break_even()),
            100.0 * self.model_capture(),
            if edge.ci_low > 0.0 {
                ""
            } else {
                " — NOT DISTINGUISHABLE FROM THE UNCONDITIONAL NULL"
            },
        ));
        lines.push(format!(
            "  uncapped Kelly |f*|: median {:.2}x, p95 {:.2}x, mean signed {:+.2}x, \
             {:.1}% of bars at the {:.1}x cap",
            self.free_kelly.median,
            self.free_kelly.p95,
            self.free_kelly.mean_signed,
            100.0 * self.free_kelly.saturated,
            self.leverage_cap,
        ));
        let cap_curve = self
            .cap_curve
            .iter()
            .map(|point| {
                format!(
                    "{:.2}x {:+.2}bps/bar (be {:.1}, dd {:.1}%)",
                    point.cap,
                    point.edge * 1e4,
                    point.break_even_bps,
                    100.0 * point.max_drawdown,
                )
            })
            .collect::<Vec<_>>()
            .join("  ");
        lines.push(format!("  cap curve: {cap_curve}"));
        for (level, nominal) in TAIL_LEVELS.iter().enumerate() {
            let lower = &self.tail.lower[level];
            let upper = &self.tail.upper[level];
            lines.push(format!(
                "  tail q={:<6.3}%: lower {:.3}% realized ({:.2}x promised, blocked ci \
                 {:.2}-{:.2}x)  upper {:.3}% ({:.2}x, {:.2}-{:.2}x)",
                100.0 * nominal,
                100.0 * lower.realized,
                lower.ratio,
                lower.blocked.0 / nominal,
                lower.blocked.1 / nominal,
                100.0 * upper.realized,
                upper.ratio,
                upper.blocked.0 / nominal,
                upper.blocked.1 / nominal,
            ));
        }
        let (worst, is_lower) = self.tail.worst();
        lines.push(format!(
            "  worst tail {:.2}x promised on the {} side{}",
            worst,
            if is_lower { "LOWER" } else { "upper" },
            if worst > TAIL_RATIO_WARN {
                " — the traded law understates its own far tail"
            } else {
                ""
            },
        ));
        lines.extend(self.calibration.report_lines());
        lines
    }
}

/// Everything the accounting needs that is not a window.
///
/// `free_marginal` is the null's UNCAPPED fraction. It is here rather than baked into the
/// windows because the cap curve has to re-clamp the null at every cap: freezing the null
/// at the headline cap while re-clamping the model would make the curve a comparison
/// against a moving target.
#[derive(Clone, Copy, Debug)]
pub struct BenchConfig {
    pub cost_bps: f64,
    pub cap: f64,
    pub free_marginal: f64,
}

impl BenchConfig {
    pub fn new(cost_bps: f64, cap: f64, free_marginal: f64) -> Self {
        Self {
            cost_bps,
            cap,
            free_marginal,
        }
    }
}

/// Score every policy on the same windows and pair them window by window.
///
/// `blocks[w]` is the resampling unit of window `w` — `(symbol, calendar month)` in the
/// pretrainer — because windows inside one symbol-month share a regime and are not
/// independent draws.
pub fn bench(
    windows: &[WindowPaths],
    blocks: &[u64],
    tail: &TailCounts,
    config: BenchConfig,
) -> TradeBench {
    let BenchConfig {
        cost_bps,
        cap,
        free_marginal,
    } = config;
    let mut result = TradeBench::nan();
    result.cost_bps = cost_bps;
    result.leverage_cap = cap;
    if windows.is_empty() {
        return result;
    }
    assert!(
        blocks.len() >= windows.len(),
        "every traded window needs a bootstrap block assignment: {} blocks for {} windows",
        blocks.len(),
        windows.len()
    );
    let blocks = &blocks[..windows.len()];
    let cost = cost_bps * 1e-4;

    let ledgers: Vec<Ledger> = (0..POLICY_COUNT)
        .map(|policy| Ledger::build(windows, policy, cap))
        .collect();
    for (policy, ledger) in ledgers.iter().enumerate() {
        result.policies[policy] = ledger.stats(cost);
    }
    result.bars = ledgers[POLICY_MODEL].bars();
    result.windows = windows.len();

    // Every policy is paired against the SAME null on the SAME windows, so each one gets
    // its own interval, cost curve, break-even and share of the ceiling. Doing this for the
    // model alone would have left the fractional-Kelly policies with growth figures and no
    // way to say whether their edge is resolvable or what cost kills it — and those two
    // numbers are the entire economic verdict. The null's own row is a paired difference
    // against itself: exactly zero, zero-width interval, no break-even. That is correct
    // rather than degenerate, and it is asserted.
    let null_growth = ledgers[POLICY_MARGINAL].window_growth(cost);
    let ceiling = result.policies[POLICY_ORACLE].net_growth - result.policies[POLICY_MARGINAL].net_growth;
    for policy in 0..POLICY_COUNT {
        let deltas: Vec<f64> = ledgers[policy]
            .window_growth(cost)
            .iter()
            .zip(&null_growth)
            .map(|(policy, null)| policy - null)
            .collect();
        result.edge[policy] = block_bootstrap(&deltas, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        let edge_at = |bps: f64| {
            let cost = bps * 1e-4;
            ledgers[policy].net_growth_per_bar(cost)
                - ledgers[POLICY_MARGINAL].net_growth_per_bar(cost)
        };
        for (slot, bps) in COST_GRID_BPS.iter().enumerate() {
            result.cost_curve[policy][slot] = edge_at(*bps);
        }
        result.break_even_bps[policy] = break_even_bps(&edge_at);
        result.ceiling_capture[policy] = if ceiling > 0.0 {
            (result.policies[policy].net_growth - result.policies[POLICY_MARGINAL].net_growth)
                / ceiling
        } else {
            f64::NAN
        };
    }
    result.blocks = result.edge[POLICY_MODEL].blocks;

    result.free_kelly = free_kelly(windows, cap);
    for (slot, point) in CAP_GRID.iter().enumerate() {
        result.cap_curve[slot] = cap_point(windows, *point, free_marginal, cost);
    }
    result.tail = tail_calibration(tail, blocks);
    result.calibration = mean_calibration(windows, blocks);
    result
}

/// Re-clamp every policy's position path at a different leverage cap.
///
/// The ONE definition of what a cap-curve point means, so the curve, the promotion gate's
/// paired standard error at a re-clamped cap and the shrunk policy's own curve cannot drift
/// apart. Exact rather than convenient: `g` is concave, so the constrained optimum on
/// `[-cap, cap]` is the projection of the free optimum onto it, and nothing is re-solved.
/// The oracle is re-clamped too, so a ceiling measured against it is a ceiling under the
/// SAME cap.
///
/// The conditional moments and the recalibrated fraction ride along untouched: they are
/// properties of the predictive law, not of the cap.
pub fn recap(windows: &[WindowPaths], cap: f64, free_marginal: f64) -> Vec<WindowPaths> {
    windows
        .iter()
        .map(|window| WindowPaths {
            realized: window.realized.clone(),
            free: window.free.clone(),
            positions: std::array::from_fn(|policy| {
                let multiple = POLICY_KELLY_MULTIPLE[policy];
                if multiple.is_finite() {
                    window
                        .free
                        .iter()
                        .map(|f| clamp_fraction(multiple * f, cap))
                        .collect()
                } else if policy == POLICY_MARGINAL {
                    vec![clamp_fraction(free_marginal, cap); window.bars()]
                } else if policy == POLICY_BUY_HOLD {
                    vec![1.0; window.bars()]
                } else {
                    window
                        .realized
                        .iter()
                        .map(|r| cap * r.signum() * f64::from(*r != 0.0))
                        .collect()
                }
            }),
            predicted_mean: window.predicted_mean.clone(),
            predicted_var: window.predicted_var.clone(),
            free_shrunk: window.free_shrunk.clone(),
            outer_mass: window.outer_mass.clone(),
            outer_signed: window.outer_signed.clone(),
            trimmed_mean: window.trimmed_mean.clone(),
            trimmed_var: window.trimmed_var.clone(),
        })
        .collect()
}

/// Per-window net log growth per bar of one policy, on ALREADY-recapped paths.
///
/// The paired unit the block bootstrap resamples, exposed because a selection rule that
/// gates on an economic difference needs the VECTOR rather than its mean: a standard error
/// on the difference against an incumbent is not derivable from two scalars.
pub fn window_growth_at(
    windows: &[WindowPaths],
    policy: usize,
    cap: f64,
    cost_bps: f64,
) -> Vec<f64> {
    Ledger::build(windows, policy, cap).window_growth(cost_bps * 1e-4)
}

/// The verdict re-derived at one cap, by re-clamping the already-solved fractions.
///
/// The oracle is re-clamped too, so `ceiling_capture` compares the model against what
/// perfect foresight could have earned UNDER THE SAME CAP rather than against a ceiling
/// measured at a different leverage.
fn cap_point(windows: &[WindowPaths], cap: f64, free_marginal: f64, cost: f64) -> CapPoint {
    let recapped = recap(windows, cap, free_marginal);
    let model = Ledger::build(&recapped, POLICY_MODEL, cap);
    let null = Ledger::build(&recapped, POLICY_MARGINAL, cap);
    let oracle = Ledger::build(&recapped, POLICY_ORACLE, cap);
    let stats = model.stats(cost);
    let edge_at = |bps: f64| {
        let cost = bps * 1e-4;
        model.net_growth_per_bar(cost) - null.net_growth_per_bar(cost)
    };
    let null_growth = null.stats(cost).net_growth;
    let ceiling = oracle.stats(cost).net_growth - null_growth;
    CapPoint {
        cap,
        edge: stats.net_growth - null_growth,
        break_even_bps: break_even_bps(&edge_at),
        sharpe: stats.sharpe,
        ceiling_capture: if ceiling > 0.0 {
            (stats.net_growth - null_growth) / ceiling
        } else {
            f64::NAN
        },
        mean_abs_position: stats.mean_abs_position,
        clamped_fraction: stats.clamped_fraction,
        turnover: stats.turnover,
        max_drawdown: stats.max_drawdown,
        ruin_bars: stats.ruin_bars,
    }
}

/// The distribution of `|f*|` over every traded bar, plus the saturation share.
fn free_kelly(windows: &[WindowPaths], cap: f64) -> FreeKelly {
    let mut magnitudes: Vec<f64> = windows
        .iter()
        .flat_map(|window| window.free.iter().copied())
        .collect();
    if magnitudes.is_empty() {
        return FreeKelly::nan();
    }
    let bars = magnitudes.len() as f64;
    let mean_signed = magnitudes.iter().sum::<f64>() / bars;
    let saturated = magnitudes.iter().filter(|f| f.abs() >= cap).count() as f64 / bars;
    let mut histogram = [0.0f64; FREE_KELLY_EDGES.len() - 1];
    for value in &magnitudes {
        let magnitude = value.abs();
        // `FREE_KELLY_EDGES` ends at infinity, so the last bucket is the open-ended one and
        // every finite magnitude lands somewhere.
        let bucket = FREE_KELLY_EDGES
            .windows(2)
            .position(|edge| magnitude >= edge[0] && magnitude < edge[1])
            .unwrap_or(histogram.len() - 1);
        histogram[bucket] += 1.0 / bars;
    }
    for value in magnitudes.iter_mut() {
        *value = value.abs();
    }
    magnitudes.sort_unstable_by(f64::total_cmp);
    let quantile = |q: f64| magnitudes[((magnitudes.len() as f64 - 1.0) * q).round() as usize];
    FreeKelly {
        histogram,
        saturated,
        median: quantile(0.5),
        p95: quantile(0.95),
        mean_signed,
    }
}

/// Turn per-window exceedance counts into calibration points with both intervals.
fn tail_calibration(counts: &TailCounts, blocks: &[u64]) -> TailCalibration {
    let mut result = TailCalibration::nan();
    let windows = counts.windows();
    if windows == 0 {
        return result;
    }
    let usable = windows.min(blocks.len());
    let bars: f64 = counts.bars[..usable].iter().sum();
    if bars <= 0.0 {
        return result;
    }
    result.bars = bars;
    result.windows = usable;
    let blocks = &blocks[..usable];
    let point = |exceed: &[f64], nominal: f64| -> TailPoint {
        let total: f64 = exceed[..usable].iter().sum();
        let realized = total / bars;
        // Per-window RATES, so the bootstrap resamples the same unit the growth interval
        // does. Windows share a bar count here, so the mean of the rates is the pooled rate.
        let rates: Vec<f64> = exceed[..usable]
            .iter()
            .zip(&counts.bars[..usable])
            .map(|(count, bars)| if *bars > 0.0 { count / bars } else { f64::NAN })
            .collect();
        let blocked = block_bootstrap(&rates, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        TailPoint {
            nominal,
            realized,
            ratio: realized / nominal,
            wilson: wilson_interval(total, bars),
            blocked: (blocked.ci_low, blocked.ci_high),
            exceedances: total,
        }
    };
    for (level, nominal) in TAIL_LEVELS.iter().enumerate() {
        result.lower[level] = point(&counts.lower[level], *nominal);
        result.upper[level] = point(&counts.upper[level], *nominal);
    }
    result
}

/// Wilson score interval at 95%, which is the right binomial interval at small `p`.
///
/// The normal approximation is useless here: at `p = 0.001` it produces intervals that
/// include negative probabilities, and at zero successes it produces a zero-width interval,
/// which would let "we saw no exceedances" read as "the tail is exactly right".
fn wilson_interval(successes: f64, trials: f64) -> (f64, f64) {
    if trials <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    const Z: f64 = 1.959_963_984_540_054;
    let p = successes / trials;
    let z2 = Z * Z;
    let denominator = 1.0 + z2 / trials;
    let center = (p + z2 / (2.0 * trials)) / denominator;
    let spread =
        Z * ((p * (1.0 - p) / trials) + z2 / (4.0 * trials * trials)).sqrt() / denominator;
    ((center - spread).max(0.0), (center + spread).min(1.0))
}

/// Cost level at which an edge curve crosses zero.
///
/// Bracket by doubling, then bisect. Charging cost on turnover makes the curve
/// monotonically decreasing whenever the model trades more than the null does, which is
/// the only case in which a break-even exists at all; the two degenerate cases are
/// reported as `NAN` (nothing to lose) and `INFINITY` (nothing that cost can take away)
/// rather than as a number.
fn break_even_bps(edge_at: &impl Fn(f64) -> f64) -> f64 {
    if !(edge_at(0.0) > 0.0) {
        return f64::NAN;
    }
    let mut hi = 1.0f64;
    while hi <= MAX_BREAK_EVEN_BPS && edge_at(hi) > 0.0 {
        hi *= 2.0;
    }
    if hi > MAX_BREAK_EVEN_BPS {
        return f64::INFINITY;
    }
    let mut lo = 0.0f64;
    for _ in 0..BREAK_EVEN_ITERATIONS {
        let mid = 0.5 * (lo + hi);
        if edge_at(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// Calibration of the traded conditional mean
// ---------------------------------------------------------------------------

/// A post-hoc affine recalibration of the traded conditional mean: `mu -> alpha + beta mu`.
///
/// # Why this is the growth-optimal response to a fitted slope
///
/// If the realized return satisfies `E[r | mu_hat] = alpha + beta mu_hat` with `beta < 1`,
/// then `mu_hat` is not the conditional mean — `alpha + beta mu_hat` is, and the model's own
/// number overstates the edge by `1 / beta`. Kelly sizing is monotone in the mean and its
/// error is QUADRATIC in an overstatement, so trading the inflated mean is not a harmless
/// scaling: it is the one error fractional Kelly exists to blunt. Sizing on the projection
/// instead is not a tuned haircut, it is the same log-optimal solve applied to the law whose
/// mean the data supports.
///
/// # Why it is a SHIFT of the support and not a scaling of the fraction
///
/// Halving `f*` is not the log-optimal response to halving the mean: `f*` is the root of
/// `sum_b p_b R_b / (1 + f R_b) = 0`, which depends on the whole law and not on its mean
/// alone. The exact statement is that recentering the law is a transform of its SUPPORT:
/// adding `d` to every bin's log value maps `1 + R_b` to `(1 + R_b) exp(d)`, shifts the mean
/// of `r` by exactly `d`, and leaves every central moment — including the far tail the tail
/// diagnostic says is honest — untouched. The recalibrated position is then the ordinary
/// Kelly optimum of the recentered law, solved by the same bisection, with `d` chosen so the
/// new mean is `alpha + beta mu`:
///
/// ```text
/// d = (alpha + beta mu) - mu = alpha + (beta - 1) mu
/// ```
///
/// The identity `1 + f R'_b = (1 + f(e^d - 1)) (1 + w R_b)` with
/// `w = f e^d / (1 + f(e^d - 1))` shows the shifted objective is the original growth curve
/// reparametrized plus a deterministic term, which is why no approximation appears anywhere:
/// the solve is the same solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeanShrink {
    /// Intercept of the calibration regression, in LOG-return units per bar.
    pub alpha: f64,
    /// Slope of the calibration regression. `1.0` is perfect calibration.
    pub beta: f64,
}

impl MeanShrink {
    /// The recalibration that changes nothing, which is what perfect calibration implies.
    pub fn identity() -> Self {
        Self {
            alpha: 0.0,
            beta: 1.0,
        }
    }

    /// The recalibration a fitted mean regression implies. `None` when the fit degenerated.
    pub fn from_fit(fit: &MzFit) -> Option<Self> {
        (fit.alpha.is_finite() && fit.beta.is_finite()).then_some(Self {
            alpha: fit.alpha,
            beta: fit.beta,
        })
    }

    /// The log-space shift that maps a predicted mean onto its recalibrated value.
    pub fn shift(&self, mu: f64) -> f64 {
        self.alpha + (self.beta - 1.0) * mu
    }

    pub fn is_identity(&self) -> bool {
        self.alpha == 0.0 && self.beta == 1.0
    }
}

/// One Mincer-Zarnowitz regression: `y = alpha + beta x + eps`, with a BLOCKED interval.
///
/// Perfect calibration of a forecast `x` for an outcome `y` is `alpha = 0, beta = 1`, and
/// the two failures are distinguishable: a nonzero `alpha` is a constant bias, a `beta`
/// below one says the forecast's VARIATION is too large — it moves more than the outcome it
/// predicts, which for a conditional mean is exactly the overstated-edge failure a Kelly
/// bettor pays for quadratically.
///
/// `r2` is the ordinary coefficient of determination of the same fit. On 5-minute returns it
/// is expected to be of order `1e-4`: essentially all of a bar's return is unforecastable,
/// which is precisely why the SLOPE rather than the fit quality is the quantity of interest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MzFit {
    pub alpha: f64,
    pub beta: f64,
    /// Block-bootstrap standard errors: the standard deviation of the coefficient over
    /// refits on resampled BLOCKS.
    pub alpha_se: f64,
    pub beta_se: f64,
    /// Percentile interval of the same refits, at [`CI_MASS`].
    pub alpha_ci: (f64, f64),
    pub beta_ci: (f64, f64),
    pub r2: f64,
    /// Resampling units the interval was taken over, and bars the point estimate used.
    pub blocks: usize,
    pub samples: usize,
    /// Cross-block dispersion of the SLOPE, and the part of it that is sampling noise.
    ///
    /// A pooled slope answers "is the forecast miscalibrated on average". It cannot answer
    /// "is the miscalibration COMMON across names and regimes", and that is a different
    /// question with different consequences: a decision rule that ranks by a scale-free
    /// quantity — `|mu| / sigma`, say — is exactly invariant to a miscalibration shared by
    /// every block and is distorted only by one that varies between them. So the pooled
    /// number decides the sizing and this pair decides whether a RANKING built on the same
    /// forecasts is reading what it claims to.
    ///
    /// `beta_block_sd` is the standard deviation of the per-block OLS slopes.
    /// `beta_block_noise_sd` is what that dispersion would be if every block shared one true
    /// slope and differed only by its own estimation error: the root-mean-square of the
    /// blocks' own OLS standard errors. Comparing the two is the whole point — a per-block
    /// slope over ~900 bars is noisy, so raw dispersion is not evidence of heterogeneity.
    pub beta_block_sd: f64,
    pub beta_block_noise_sd: f64,
    /// Blocks that carried enough varying bars to admit a slope AND a standard error.
    pub beta_blocks_resolved: usize,
}

impl MzFit {
    pub fn nan() -> Self {
        Self {
            alpha: f64::NAN,
            beta: f64::NAN,
            alpha_se: f64::NAN,
            beta_se: f64::NAN,
            alpha_ci: (f64::NAN, f64::NAN),
            beta_ci: (f64::NAN, f64::NAN),
            r2: f64::NAN,
            blocks: 0,
            samples: 0,
            beta_block_sd: f64::NAN,
            beta_block_noise_sd: f64::NAN,
            beta_blocks_resolved: 0,
        }
    }

    /// Standard deviations the slope sits away from perfect calibration. `beta = 1` is the
    /// null a calibration test is against, so this rather than a t-statistic against zero is
    /// the number that decides whether the miscalibration is resolvable.
    pub fn slope_t_against_one(&self) -> f64 {
        (self.beta - 1.0) / self.beta_se
    }

    /// True when the blocked interval on the slope excludes perfect calibration.
    pub fn slope_resolvable(&self) -> bool {
        self.beta_ci.0.is_finite() && self.beta_ci.1.is_finite() && !(self.beta_ci.0..=self.beta_ci.1).contains(&1.0)
    }

    /// True when the block-dispersion pair carries a real measurement.
    ///
    /// Needed because [`Self::slope_heterogeneous`] is a bool over floats and a bool cannot
    /// express three states: with `beta_block_sd` and its floor both `NaN`, the comparison
    /// `NaN > 1.25` is FALSE, so an unmeasured fit would answer "the slope is homogeneous" —
    /// a positive finding about data that does not exist. Any predicate whose inputs can be
    /// `NaN` needs a gate like this beside it, or it silently answers a question nobody asked.
    pub fn block_dispersion_measured(&self) -> bool {
        self.beta_blocks_resolved >= 8
            && self.beta_block_sd.is_finite()
            && self.beta_block_noise_sd.is_finite()
            && self.beta_block_noise_sd > 0.0
    }

    /// Cross-block slope dispersion in units of its own sampling noise.
    ///
    /// `1.0` is the reading under one shared true slope: the blocks disagree exactly as much
    /// as their own standard errors say they must. Above one is real heterogeneity, and the
    /// excess is `sqrt(observed^2 - noise^2)` rather than the difference of the two.
    pub fn block_dispersion_ratio(&self) -> f64 {
        self.beta_block_sd / self.beta_block_noise_sd
    }

    /// Standard deviation of the TRUE per-block slope, with sampling noise removed.
    ///
    /// `NaN` when unmeasured, and `0.0` — not a negative root — when the blocks disagree by
    /// less than their own noise, which is the honest reading of "no detectable variation".
    pub fn beta_block_excess_sd(&self) -> f64 {
        let observed = self.beta_block_sd * self.beta_block_sd;
        let noise = self.beta_block_noise_sd * self.beta_block_noise_sd;
        if !observed.is_finite() || !noise.is_finite() {
            return f64::NAN;
        }
        (observed - noise).max(0.0).sqrt()
    }

    /// True when the slope varies across blocks by materially more than its own noise, so a
    /// scale-free ranking built on these forecasts is NOT invariant to the miscalibration.
    ///
    /// The threshold is `1.25`, i.e. observed variance more than 1.56x the noise floor. It is
    /// a judgement rather than a test: with ~250 blocks the ratio's own standard error is
    /// about `1 / sqrt(2 * 250) = 4.5%`, so 1.25 is roughly five of those away from parity
    /// and cannot be produced by the estimator's own scatter.
    pub fn slope_heterogeneous(&self) -> bool {
        self.block_dispersion_measured() && self.block_dispersion_ratio() > 1.25
    }
}

/// Per-block sufficient statistics of a simple regression.
///
/// A block's contribution to an OLS fit is six sums, so a bootstrap refit costs one pass
/// over the BLOCKS rather than over the bars. That is what makes 1000 exact refits over
/// 230k bars free, and it is exact rather than an approximation: the normal equations of a
/// simple regression are a function of these six numbers alone.
#[derive(Clone, Copy, Debug, Default)]
struct RegressionSums {
    n: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
    yy: f64,
}

impl RegressionSums {
    fn push(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.x += x;
        self.y += y;
        self.xx += x * x;
        self.xy += x * y;
        self.yy += y * y;
    }

    fn absorb(&mut self, other: &Self) {
        self.n += other.n;
        self.x += other.x;
        self.y += other.y;
        self.xx += other.xx;
        self.xy += other.xy;
        self.yy += other.yy;
    }

    /// `(alpha, beta, r2)`, or all-NaN when the regressor does not vary.
    fn solve(&self) -> (f64, f64, f64) {
        if self.n < 2.0 {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let sxx = self.n * self.xx - self.x * self.x;
        let syy = self.n * self.yy - self.y * self.y;
        let sxy = self.n * self.xy - self.x * self.y;
        if !(sxx > 0.0) {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let beta = sxy / sxx;
        let alpha = (self.y - beta * self.x) / self.n;
        let r2 = if syy > 0.0 {
            (sxy * sxy) / (sxx * syy)
        } else {
            f64::NAN
        };
        (alpha, beta, r2)
    }

    /// Textbook OLS standard error of the slope, or `NaN` when the block cannot support one.
    ///
    /// This is the WITHIN-block noise of a single block's own slope and is deliberately not
    /// the blocked bootstrap error: the bootstrap answers how the POOLED slope would move
    /// under resampling of blocks, and what is wanted here is how much a single block's slope
    /// would move on its own bars, which is the noise floor cross-block dispersion has to
    /// clear. Bars inside one block still share a regime so this understates the true within
    /// noise, which makes the heterogeneity verdict CONSERVATIVE in the safe direction: an
    /// understated floor inflates the ratio, so a reading near one is trustworthy and a large
    /// one is the claim that needs care.
    fn slope_standard_error(&self) -> f64 {
        if self.n < 4.0 {
            return f64::NAN;
        }
        // Centred scatter, from the raw sums. `solve` uses `n` times these.
        let sxx = self.xx - self.x * self.x / self.n;
        let syy = self.yy - self.y * self.y / self.n;
        let sxy = self.xy - self.x * self.y / self.n;
        if !(sxx > 0.0) {
            return f64::NAN;
        }
        let beta = sxy / sxx;
        let residual = (syy - beta * sxy).max(0.0) / (self.n - 2.0);
        (residual / sxx).sqrt()
    }
}

/// Fit `y = alpha + beta x` and interval it by resampling BLOCKS, never bars.
///
/// The resampling scheme is [`block_bootstrap`]'s, deliberately down to the RNG: the same
/// `ChaCha12Rng` seeded with the same [`BOOTSTRAP_SEED`], the same number of draws, and
/// blocks visited in the same `BTreeMap` order, so a draw index sequence is literally the
/// same sequence of blocks the edge interval was taken over. That is what makes a slope's
/// interval comparable to an edge's interval rather than merely similar in construction.
///
/// Bars inside one `(symbol, calendar month)` block share a symbol, a regime and a level of
/// volatility, so they are not independent draws and an iid interval would be too tight.
///
/// # What this key removes, and the dependence it does NOT
///
/// This doc used to assert that an iid interval "would be roughly twenty times too tight",
/// attributing the factor to autocorrelation among the bars inside a window. THAT ATTRIBUTION IS
/// WRONG, and it was load-bearing in an argument it should not have settled. Measured on this
/// corpus's train split, MODEL-FREE:
///
/// * WITHIN-symbol SERIAL dependence, which this key DOES remove: market-factor autocorrelation
///   at lags 1-10 is `-0.015 .. +0.048`, a variance inflation of `1.1048`, i.e. about `1.05x` on a
///   standard error. That is the mechanism the old sentence named, and it is small.
/// * CROSS-SECTIONAL same-instant dependence, which this key DOES NOT remove: `rho = 0.176`
///   `[0.158, 0.201]` over the run's own unfiltered universe, giving a design effect of `327` and
///   `sqrt(327) = 18.1` on a standard error. That is numerically an excellent match to the
///   "roughly twenty" the old sentence quoted — so the NUMBER was cross-sectional while the
///   MECHANISM named was serial, and the two do not go together.
///
/// Resampling `(symbol, calendar month)` blocks INDEPENDENTLY treats two different symbols in the
/// same month as independent draws, and they are not. For a market-level functional - and
/// `beta = Cov(mu, r) / Var(mu)` is one, because same-instant residuals across symbols share the
/// common factor - this interval is therefore likely STILL TOO TIGHT, by an amount that grows with
/// how much CALENDAR OVERLAP the evaluated windows have. THAT AMOUNT IS UNRESOLVED: nobody has
/// measured the overlap, and no number is quoted here in place of one.
///
/// The remedy is a KEY CHANGE and never a second estimator, because `blocks` is an OPAQUE u64:
/// pooling every symbol inside a calendar month is a different clustering of the same rows through
/// the same code. [`compare_clustering`] runs exactly that and reports what it costs in width, so
/// the question is answered by measurement at whatever call site has the keys.
pub fn mincer_zarnowitz(
    x: &[f64],
    y: &[f64],
    blocks: &[u64],
    draws: usize,
    seed: u64,
) -> MzFit {
    mincer_zarnowitz_paired(x, y, blocks, draws, seed).fit
}

/// One fit plus every draw's slope IN DRAW ORDER, the pair a paired difference needs.
///
/// The per-draw slopes are deliberately UNSORTED and UNFILTERED - a degenerate draw contributes
/// `NAN` in its own slot rather than being dropped - because a sorted or compacted vector carries
/// no draw index and cannot be paired with anything. [`MzFit`]'s own standard error and interval
/// are computed from the finite subset, so they are bit-identical to [`mincer_zarnowitz`]'s.
///
/// `draws` is private and reachable only through [`PairedFit::delta`] and [`PairedFit::draws`],
/// so the pairing PRECONDITION is checked where the difference is taken rather than trusted at
/// every call site.
#[derive(Clone, Debug)]
pub struct PairedFit {
    pub fit: MzFit,
    draws: Vec<f64>,
}

impl PairedFit {
    /// Per-draw slopes in draw order, `NAN` where the draw was degenerate.
    pub fn draws(&self) -> &[f64] {
        &self.draws
    }

    /// True when two fits were taken over the same rows, the same blocks and the same draws, and
    /// are therefore differenceable draw by draw.
    ///
    /// `blocks` and `samples` are the observable consequence of the non-finite row filter: if one
    /// fit's regressor carried a single non-finite value the surviving key set shrank, the draw
    /// index space changed meaning, and the two draw sequences resample different populations
    /// under the same indices. Equality of these three integers is exactly the condition under
    /// which that cannot have happened.
    pub fn pairable_with(&self, other: &Self) -> bool {
        self.fit.blocks == other.fit.blocks
            && self.fit.samples == other.fit.samples
            && self.draws.len() == other.draws.len()
            && !self.draws.is_empty()
    }

    /// `self.beta - other.beta` with the difference taken INSIDE each draw, or `None` when the
    /// two fits are not pairable.
    ///
    /// `None` rather than a `NaN`-filled result on purpose: an unpairable difference is a
    /// PRECONDITION failure the caller has to see and report, and this tree's recurring defect is
    /// an absent value that reads as a measured one.
    pub fn delta(&self, other: &Self) -> Option<BlockedScalar> {
        if !self.pairable_with(other) {
            return None;
        }
        let point = self.fit.beta - other.fit.beta;
        let mut column: Vec<f64> = self
            .draws
            .iter()
            .zip(&other.draws)
            .map(|(left, right)| left - right)
            .filter(|value| value.is_finite())
            .collect();
        if column.len() < 2 {
            return Some(BlockedScalar { point, ..BlockedScalar::nan() });
        }
        column.sort_by(f64::total_cmp);
        let tail = (1.0 - CI_MASS) / 2.0;
        Some(BlockedScalar {
            point,
            se: standard_deviation(&column),
            ci: (
                sorted_percentile(&column, tail),
                sorted_percentile(&column, 1.0 - tail),
            ),
        })
    }
}

/// The same fit as [`mincer_zarnowitz`], plus every draw's slope, so two fits can be DIFFERENCED
/// WITHIN a draw instead of across two independent intervals.
///
/// # Why an unpaired difference is the wrong estimator
///
/// Two slopes measured on the same bars against the same realized `y` share almost all of their
/// sampling variance: resampling blocks moves both in the same direction and nearly the same
/// amount. Differencing two independently-bootstrapped intervals throws that common component
/// away and reports an interval several times wider than the difference actually has, which
/// buries any effect smaller than the shared noise. Recomputing both slopes on the SAME
/// resampled blocks and differencing inside the draw keeps it. This is not a different
/// bootstrap - same RNG, same seed, same draw count, same block order - it is the same one with
/// the per-draw values retained instead of reduced immediately.
///
/// # The trap that silently breaks the pairing, and the caller's obligation
///
/// **This function filters rows where `x` OR `y` is non-finite, and `x` differs between the fits
/// a caller wants to pair.** So two fits over the same `blocks` can still be taken over DIFFERENT
/// effective rows, hence different per-block sums, and the draw sequence - which is a pure
/// function of `(block count, draws, seed)` and reads no data - would then be resampling two
/// different populations under the same indices. The result looks paired and is not, with nothing
/// anywhere reporting an error.
///
/// A caller pairing several fits MUST therefore restrict every one of them to the same rows
/// first: [`finite_mask`] over all the `x` vectors plus `y` gives the intersection, and
/// [`apply_mask`] projects each vector onto it. Pairing is valid exactly when every paired call
/// receives vectors of the same length whose finite rows coincide, which the mask guarantees and
/// an identical seed does not.
///
/// A draw whose normal equations were degenerate contributes `NAN` rather than being dropped, so
/// index `k` is draw `k` in every paired call and the point estimate, standard error and interval
/// are bit-identical to [`mincer_zarnowitz`]'s.
pub fn mincer_zarnowitz_paired(
    x: &[f64],
    y: &[f64],
    blocks: &[u64],
    draws: usize,
    seed: u64,
) -> PairedFit {
    assert_eq!(x.len(), y.len(), "one outcome per forecast");
    assert_eq!(x.len(), blocks.len(), "every observation needs a block");

    let mut grouped: BTreeMap<u64, RegressionSums> = BTreeMap::new();
    let mut samples = 0usize;
    for ((block, x), y) in blocks.iter().zip(x).zip(y) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        grouped.entry(*block).or_default().push(*x, *y);
        samples += 1;
    }
    let totals: Vec<RegressionSums> = grouped.into_values().collect();
    let mut pooled = RegressionSums::default();
    for block in &totals {
        pooled.absorb(block);
    }
    let (alpha, beta, r2) = pooled.solve();
    let mut fit = MzFit {
        alpha,
        beta,
        r2,
        blocks: totals.len(),
        samples,
        ..MzFit::nan()
    };

    // Per-block slopes and their own noise, for the COMMON-versus-VARYING question the
    // pooled slope cannot answer. Blocks that cannot support both are dropped from both
    // sums together, so the dispersion and the floor are always over the same blocks.
    let mut per_block: Vec<f64> = Vec::with_capacity(totals.len());
    let mut noise_variance = 0.0f64;
    for block in &totals {
        let (_, b, _) = block.solve();
        let se = block.slope_standard_error();
        if b.is_finite() && se.is_finite() {
            per_block.push(b);
            noise_variance += se * se;
        }
    }
    if per_block.len() >= 2 {
        fit.beta_blocks_resolved = per_block.len();
        fit.beta_block_sd = standard_deviation(&per_block);
        fit.beta_block_noise_sd = (noise_variance / per_block.len() as f64).sqrt();
    }
    if totals.len() < 2 || draws == 0 {
        // One block is one observation: there is no dispersion to estimate, and a zero-width
        // interval reported as precision is the failure this refuses to commit.
        return PairedFit { fit, draws: Vec::new() };
    }

    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut alphas = Vec::with_capacity(draws);
    let mut betas = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut draw = RegressionSums::default();
        for _ in 0..totals.len() {
            draw.absorb(totals.choose(&mut rng).expect("totals is non-empty"));
        }
        let (a, b, _) = draw.solve();
        // Both or neither, so a draw never contributes to one coefficient's spread alone.
        let resolved = a.is_finite() && b.is_finite();
        alphas.push(if resolved { a } else { f64::NAN });
        betas.push(if resolved { b } else { f64::NAN });
    }
    let mut finite_alphas: Vec<f64> = alphas.iter().copied().filter(|v| v.is_finite()).collect();
    let mut finite_betas: Vec<f64> = betas.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_alphas.len() < 2 {
        return PairedFit { fit, draws: betas };
    }
    finite_alphas.sort_by(f64::total_cmp);
    finite_betas.sort_by(f64::total_cmp);
    let tail = (1.0 - CI_MASS) / 2.0;
    fit.alpha_se = standard_deviation(&finite_alphas);
    fit.beta_se = standard_deviation(&finite_betas);
    fit.alpha_ci = (
        sorted_percentile(&finite_alphas, tail),
        sorted_percentile(&finite_alphas, 1.0 - tail),
    );
    fit.beta_ci = (
        sorted_percentile(&finite_betas, tail),
        sorted_percentile(&finite_betas, 1.0 - tail),
    );
    PairedFit { fit, draws: betas }
}

/// Rows on which EVERY supplied column is finite.
///
/// The precondition of any paired bootstrap across fits whose regressors differ: see
/// [`mincer_zarnowitz_paired`] for why an identical seed does not make two fits paired. Panics on
/// ragged input rather than truncating, because a silently shortened mask would reintroduce
/// exactly the misalignment it exists to prevent.
pub fn finite_mask(columns: &[&[f64]]) -> Vec<bool> {
    let rows = columns.first().map_or(0, |column| column.len());
    for column in columns {
        assert_eq!(column.len(), rows, "every column must cover the same rows");
    }
    (0..rows)
        .map(|row| columns.iter().all(|column| column[row].is_finite()))
        .collect()
}

/// Project a column onto the rows a [`finite_mask`] kept.
pub fn apply_mask<T: Copy>(values: &[T], mask: &[bool]) -> Vec<T> {
    assert_eq!(values.len(), mask.len(), "the mask must cover every row");
    values
        .iter()
        .zip(mask)
        .filter_map(|(value, keep)| keep.then_some(*value))
        .collect()
}

/// One fit under two different CLUSTERINGS of the same rows.
///
/// # Why this is a key change and not a second estimator
///
/// `blocks` is an OPAQUE `u64`. Two different partitions of the same observations are two
/// different resampling units fed through the IDENTICAL estimator, RNG, seed and draw count - not
/// a rival implementation whose disagreement would be ambiguous. So a difference in width is
/// attributable to the clustering and to nothing else, which is what makes this decisive rather
/// than suggestive.
///
/// # What it is for
///
/// The default key is `(symbol, calendar month)`. It removes within-symbol serial dependence,
/// measured at about `1.05x` on a standard error, and it is BLIND to same-instant cross-symbol
/// dependence, measured at `rho = 0.176` with a design effect of `327`. Pooling all symbols inside
/// a calendar month removes the second at the cost of far fewer resampling units. If
/// `width_ratio` comes back near one the concern is bounded and dismissed on evidence; if it comes
/// back materially above one then every interval taken under the default key is understated and
/// the amount is now known instead of feared.
///
/// Both directions are informative and neither is the answer this is hoping for. Note the honest
/// caveat on the alternative: pooling by month leaves few blocks, so its own interval is noisy,
/// and `alternative.blocks` is reported so a reader can see the denominator the widening was
/// bought with rather than reading a wide interval as a precise measurement of width.
#[derive(Clone, Debug)]
pub struct ClusteringComparison {
    /// The fit under the caller's default key.
    pub primary: PairedFit,
    /// The same fit under the alternative key, on the same rows.
    pub alternative: PairedFit,
}

impl ClusteringComparison {
    /// Ratio of the alternative's blocked slope standard error to the primary's.
    ///
    /// Above one means the default key was understating the interval by this factor.
    pub fn width_ratio(&self) -> f64 {
        self.alternative.fit.beta_se / self.primary.fit.beta_se
    }

    pub fn report_lines(&self, label: &str) -> Vec<String> {
        vec![
            format!(
                "  {label} clustering, DEFAULT key: beta {:+.5} (se {:.5}, CI {:+.5}..{:+.5}) \
                 over {} blocks / {} bars",
                self.primary.fit.beta,
                self.primary.fit.beta_se,
                self.primary.fit.beta_ci.0,
                self.primary.fit.beta_ci.1,
                self.primary.fit.blocks,
                self.primary.fit.samples,
            ),
            format!(
                "  {label} clustering, ALTERNATIVE key: beta {:+.5} (se {:.5}, CI \
                 {:+.5}..{:+.5}) over {} blocks / {} bars",
                self.alternative.fit.beta,
                self.alternative.fit.beta_se,
                self.alternative.fit.beta_ci.0,
                self.alternative.fit.beta_ci.1,
                self.alternative.fit.blocks,
                self.alternative.fit.samples,
            ),
            format!(
                "  {label} width ratio alternative/default = {:.3}x — {}. The point estimates are \
                 IDENTICAL by construction ({:+.2e} apart); only the interval can move, because a \
                 clustering changes the resampling unit and never the pooled normal equations",
                self.width_ratio(),
                if !self.width_ratio().is_finite() {
                    "UNRESOLVED, one of the two intervals did not form"
                } else if self.width_ratio() > 1.25 {
                    "the default key UNDERSTATES the interval and every CI taken under it is too \
                     tight by about this factor"
                } else {
                    "the default key is not materially understating the interval on these rows"
                },
                self.alternative.fit.beta - self.primary.fit.beta,
            ),
        ]
    }
}

/// Fit the same rows under two clusterings. See [`ClusteringComparison`].
///
/// The two keys must cover the same rows in the same order; the point estimates are then identical
/// by construction and any difference in the reported `beta` is a bug rather than a finding, which
/// is why [`ClusteringComparison::report_lines`] prints their difference.
pub fn compare_clustering(
    x: &[f64],
    y: &[f64],
    primary: &[u64],
    alternative: &[u64],
) -> ClusteringComparison {
    assert_eq!(
        primary.len(),
        alternative.len(),
        "both clusterings must key the same rows"
    );
    ClusteringComparison {
        primary: mincer_zarnowitz_paired(x, y, primary, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        alternative: mincer_zarnowitz_paired(x, y, alternative, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
    }
}

fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Linear-interpolated percentile of an ascending slice, the convention
/// [`block_bootstrap`] reports its interval under.
fn sorted_percentile(sorted: &[f64], q: f64) -> f64 {
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

/// Number of volatility cells [`volatility_gradient`] splits the blocks into.
///
/// Quartiles rather than deciles: a cell has to hold enough blocks for a POOLED slope to be
/// worth reading, and 256 blocks over ten cells would put 25 blocks and ~22k bars in each,
/// which is thin for a squared-residual regressand with this kurtosis.
pub const VOLATILITY_CELLS: usize = 4;

/// One volatility cell: the blocks in it, and the calibration measured on just those blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolatilityCell {
    pub blocks: usize,
    pub bars: usize,
    /// RMS REALIZED residual over the cell's bars. This is the axis the cells are sorted on,
    /// and it is realized rather than predicted deliberately: sorting on `sigma_hat` would
    /// order the blocks by the very quantity under suspicion, so a block whose `sigma_hat` is
    /// inflated by noise would land in a high cell BECAUSE of the error being measured.
    pub realized_sd: f64,
    /// RMS predicted sd over the same bars. The ratio against `realized_sd` is the cell's own
    /// spread over-statement, and it is what a leakage mechanism makes vary across cells.
    pub predicted_sd: f64,
    /// Slopes POOLED over the cell's bars rather than averaged over its blocks. An average of
    /// per-block slopes weights a 40-bar block like a 900-bar one and is dominated by the
    /// noisiest blocks; pooling weights by information, which is what a cell-level statement
    /// about calibration should do.
    pub mean_slope: f64,
    pub var_slope: f64,
    /// Mean total catch-all mass of the cell's bars, or `NaN` when the pass did not form the
    /// decomposition. Reported beside the slopes because the two accounts of a heterogeneous
    /// slope differ in exactly this quantity: one shared decode convention leaves it flat across
    /// cells, a head that learned different tails does not.
    pub outer_mass: f64,
}

impl VolatilityCell {
    pub fn nan() -> Self {
        Self {
            blocks: 0,
            bars: 0,
            realized_sd: f64::NAN,
            predicted_sd: f64::NAN,
            mean_slope: f64::NAN,
            var_slope: f64::NAN,
            outer_mass: f64::NAN,
        }
    }
}

/// Calibration slopes split by the block's own REALIZED volatility, plus the gradient.
///
/// # The question this answers and [`MzFit::slope_heterogeneous`] cannot
///
/// The dispersion pair says the miscalibration varies between blocks. It cannot say ALONG
/// WHAT, and two mechanisms with opposite fixes predict the same dispersion:
///
/// * A fixed representation error - mass decoded at a bin's outer geometry rather than at its
///   conditional mean - misplaces the same MASS for every name, so it inflates a predicted
///   variance by an ABSOLUTE amount and its relative damage is largest where the true
///   variance is smallest. Predicted: the over-statement is worst in the quietest blocks, so
///   the variance slope RISES with volatility toward one.
/// * A learned error in the bulk of the law has no reason to align with volatility at all.
///   Predicted: flat, with the dispersion coming from something else entirely.
///
/// So the GRADIENT of the slope against volatility is the discriminator, and it is reported
/// with its own standard error because a monotone-looking set of four cells proves nothing
/// when each cell's slope carries its own noise. The regression is one observation per BLOCK
/// rather than per cell, so the interval reflects the blocks the cells are built from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolatilityGradient {
    /// Quietest cell first, by RMS realized residual.
    pub cells: [VolatilityCell; VOLATILITY_CELLS],
    /// OLS slope of per-block calibration slope on `log10` realized sd: the change in
    /// calibration slope per DECADE of realized volatility.
    pub mean_gradient: f64,
    pub mean_gradient_se: f64,
    pub var_gradient: f64,
    pub var_gradient_se: f64,
    /// Blocks that carried a resolvable slope and therefore entered the gradient.
    pub blocks: usize,
}

impl VolatilityGradient {
    pub fn nan() -> Self {
        Self {
            cells: [VolatilityCell::nan(); VOLATILITY_CELLS],
            mean_gradient: f64::NAN,
            mean_gradient_se: f64::NAN,
            var_gradient: f64::NAN,
            var_gradient_se: f64::NAN,
            blocks: 0,
        }
    }

    pub fn measured(&self) -> bool {
        self.blocks >= 2 * VOLATILITY_CELLS && self.var_gradient_se.is_finite()
    }

    /// True when the variance slope RISES with volatility by more than twice its own error,
    /// which is the signature a fixed absolute misplacement of mass leaves behind.
    pub fn variance_rises_with_volatility(&self) -> bool {
        self.measured() && self.var_gradient > 2.0 * self.var_gradient_se
    }

    /// True when it FALLS resolvably, which no fixed misplacement of mass can produce and
    /// which would say the quiet names are the well-calibrated ones.
    pub fn variance_falls_with_volatility(&self) -> bool {
        self.measured() && self.var_gradient < -2.0 * self.var_gradient_se
    }

    /// `tag` names the decode convention the slopes were fitted under: two arms print the same
    /// cells over the same blocks and are only distinguishable by it.
    pub fn report_lines(&self, tag: &str) -> Vec<String> {
        if !self.measured() {
            return vec![format!(
                "  {tag} volatility split: not measured (too few blocks carried a resolvable \
                 slope)"
            )];
        }
        let mut lines = Vec::with_capacity(VOLATILITY_CELLS + 1);
        for (index, cell) in self.cells.iter().enumerate() {
            lines.push(format!(
                "  {tag} vol quartile {}: realized sd {:6.2} bps/bar, predicted {:7.2} ({:.2}x \
                 too wide), mean slope {:+.4}, var slope {:+.4}, catch-all mass {:.4}% over {} \
                 blocks / {} bars",
                index,
                cell.realized_sd * 1e4,
                cell.predicted_sd * 1e4,
                cell.predicted_sd / cell.realized_sd,
                cell.mean_slope,
                cell.var_slope,
                100.0 * cell.outer_mass,
                cell.blocks,
                cell.bars,
            ));
        }
        lines.push(format!(
            "  {tag} slope gradient per decade of realized sd: var {:+.4} (se {:.4}), mean \
             {:+.4} (se {:.4}) over {} blocks{}",
            self.var_gradient,
            self.var_gradient_se,
            self.mean_gradient,
            self.mean_gradient_se,
            self.blocks,
            if self.variance_rises_with_volatility() {
                " - the spread over-statement is WORST IN THE QUIETEST names, which is what an \
                 absolute misplacement of mass does and a bulk error does not"
            } else if self.variance_falls_with_volatility() {
                " - the spread over-statement is worst in the LOUDEST names, which no fixed \
                 misplacement of mass can produce"
            } else {
                " - no resolvable alignment with volatility, so the block dispersion is not \
                 about instrument volatility"
            },
        ));
        lines
    }
}

/// Per-block calibration sums, kept together so a block's two slopes and its volatility are
/// always taken over exactly the same bars.
#[derive(Clone, Copy, Debug, Default)]
struct BlockCalibration {
    mean: RegressionSums,
    variance: RegressionSums,
    /// Sum of the PREDICTED variance over the block's bars.
    predicted: f64,
    /// Sum of the realized squared residual, the same quantity the variance fit regresses, so
    /// `sqrt(realized / bars)` is the block's model-free volatility level.
    realized: f64,
    /// Sum of the CATCH-ALL mass, or `NaN` for the whole block when the pass did not form the
    /// decomposition - NaN rather than zero, because "no mass measured" and "no mass" are
    /// different findings and only one of them is a result.
    outer: f64,
    bars: usize,
}

impl BlockCalibration {
    /// RMS realized residual: the sorting axis, and the gradient's regressor in logs.
    fn realized_sd(&self) -> f64 {
        (self.realized / self.bars.max(1) as f64).sqrt()
    }
}

/// Split the blocks into [`VOLATILITY_CELLS`] equal-count cells by realized volatility, fit
/// each cell, and regress the per-block slope on `log10` realized sd.
pub fn volatility_gradient(
    mu: &[f64],
    realized: &[f64],
    variance: &[f64],
    residual_squares: &[f64],
    outer: Option<&[f64]>,
    blocks: &[u64],
) -> VolatilityGradient {
    assert_eq!(mu.len(), realized.len(), "one outcome per forecast");
    assert_eq!(mu.len(), variance.len(), "one variance per forecast");
    assert_eq!(mu.len(), residual_squares.len(), "one residual per forecast");
    assert_eq!(mu.len(), blocks.len(), "every observation needs a block");
    if let Some(outer) = outer {
        assert_eq!(mu.len(), outer.len(), "one catch-all mass per forecast");
    }

    let mut grouped: BTreeMap<u64, BlockCalibration> = BTreeMap::new();
    for index in 0..mu.len() {
        let (m, r) = (mu[index], realized[index]);
        let (v, s) = (variance[index], residual_squares[index]);
        if !(m.is_finite() && r.is_finite() && v.is_finite() && s.is_finite() && v > 0.0) {
            continue;
        }
        let entry = grouped.entry(blocks[index]).or_default();
        entry.mean.push(m, r);
        entry.variance.push(v, s);
        entry.predicted += v;
        entry.realized += s;
        entry.outer += outer.map_or(f64::NAN, |outer| outer[index]);
        entry.bars += 1;
    }
    let mut ordered: Vec<BlockCalibration> = grouped.into_values().collect();
    if ordered.len() < VOLATILITY_CELLS {
        return VolatilityGradient::nan();
    }
    ordered.sort_by(|a, b| a.realized_sd().total_cmp(&b.realized_sd()));

    let mut result = VolatilityGradient::nan();
    for cell in 0..VOLATILITY_CELLS {
        // Contiguous, equal-count as near as integer division allows, so no block is dropped
        // and no block lands in two cells.
        let start = cell * ordered.len() / VOLATILITY_CELLS;
        let end = (cell + 1) * ordered.len() / VOLATILITY_CELLS;
        let mut mean = RegressionSums::default();
        let mut var = RegressionSums::default();
        let (mut predicted, mut realized_sum, mut bars) = (0.0f64, 0.0f64, 0usize);
        let mut outer_sum = 0.0f64;
        for block in &ordered[start..end] {
            mean.absorb(&block.mean);
            var.absorb(&block.variance);
            predicted += block.predicted;
            realized_sum += block.realized;
            bars += block.bars;
            outer_sum += block.outer;
        }
        if bars == 0 {
            continue;
        }
        result.cells[cell] = VolatilityCell {
            blocks: end - start,
            bars,
            realized_sd: (realized_sum / bars as f64).sqrt(),
            predicted_sd: (predicted / bars as f64).sqrt(),
            mean_slope: mean.solve().1,
            var_slope: var.solve().1,
            outer_mass: outer_sum / bars as f64,
        };
    }

    // The gradient itself: one observation per block, unweighted. Unweighted rather than
    // inverse-variance because a block's slope error is itself estimated from the same few
    // hundred bars, and weighting by it would let the noisiest estimate of the noise set the
    // answer. Unweighted is the conservative direction: it widens the interval.
    let mut mean_gradient = RegressionSums::default();
    let mut var_gradient = RegressionSums::default();
    for block in &ordered {
        let level = block.realized_sd();
        if !(level > 0.0) {
            continue;
        }
        let axis = level.log10();
        let (_, mean_beta, _) = block.mean.solve();
        let (_, var_beta, _) = block.variance.solve();
        if mean_beta.is_finite() && block.mean.slope_standard_error().is_finite() {
            mean_gradient.push(axis, mean_beta);
        }
        if var_beta.is_finite() && block.variance.slope_standard_error().is_finite() {
            var_gradient.push(axis, var_beta);
        }
    }
    result.blocks = var_gradient.n as usize;
    result.mean_gradient = mean_gradient.solve().1;
    result.mean_gradient_se = mean_gradient.slope_standard_error();
    result.var_gradient = var_gradient.solve().1;
    result.var_gradient_se = var_gradient.slope_standard_error();
    result
}

/// Calibration of the traded law's first two conditional moments.
///
/// Two regressions, reported separately because they answer different questions and the
/// combination is the finding:
///
/// * MEAN: realized `r` on the predicted `E[r | past]`. `beta < 1` says the traded mean is
///   inflated, which prices directly — the growth-optimal action sizes on `beta mu`.
/// * VARIANCE: realized squared residual `(r - mu)^2` on the predicted `Var[r | past]`. A
///   slope at or above one says the predicted spread is honest or conservative. The residual
///   is taken against the model's OWN mean, so an inflated mean contaminates it upward by
///   `(mu - mu_true)^2`. At the measured scale that term is negligible and cannot manufacture
///   the finding: `mu/sigma` is the per-bar Sharpe, ~0.032, so a 30%-inflated mean adds
///   `(0.3 * 0.032)^2 ~ 1e-4` of a variance to a quantity of one, four orders below the
///   effect being tested. It would matter only for a model whose mean rivalled its spread.
///
/// An inflated mean beside an honest variance is a SIZING error, fixable after the fact and
/// without retraining. An inflated mean beside a shrinking variance is a different failure
/// with a different remedy, so if the variance slope comes back below one the mean story is
/// no longer sufficient and [`Self::report_lines`] says so in as many words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeanCalibration {
    pub mean: MzFit,
    pub variance: MzFit,
    /// Root-mean-square PREDICTED standard deviation of `r` over the traded bars.
    ///
    /// Not a calibration statistic — it is the level the two slopes are relative to, and it is
    /// here because a population restriction can change it by an order of magnitude. A book of
    /// bond funds and a book of small caps can share a slope and still hold completely
    /// different positions, because the position is `mu_hat / sigma_hat^2` and a smaller
    /// denominator levers harder. Reporting the slopes without this level would let two arms
    /// look identically calibrated while one of them is trading ten times the size.
    pub mean_predicted_sd: f64,
    /// The same two slopes, split by the block's own REALIZED volatility.
    ///
    /// Carried beside the pooled fits because a pooled slope and a dispersion verdict together
    /// still cannot say whether the miscalibration is a fixed misplacement of mass or a learned
    /// error in the bulk, and those have different fixes. See [`VolatilityGradient`].
    pub gradient: VolatilityGradient,
    /// What the two catch-all bins are worth to both fits, when the pass formed it.
    pub outer: Option<OuterDecomposition>,
}

impl MeanCalibration {
    pub fn nan() -> Self {
        Self {
            mean: MzFit::nan(),
            variance: MzFit::nan(),
            mean_predicted_sd: f64::NAN,
            gradient: VolatilityGradient::nan(),
            outer: None,
        }
    }

    pub fn measured(&self) -> bool {
        self.mean.samples > 0
    }

    /// The recalibration this fit implies for the traded mean.
    pub fn shrink(&self) -> Option<MeanShrink> {
        MeanShrink::from_fit(&self.mean)
    }

    /// True when the variance slope is resolvably BELOW one, i.e. the predicted spread is too
    /// WIDE and a mean-only correction is not the whole story.
    ///
    /// The direction is worth stating because it is easy to invert. The regression is
    /// `(r - mu)^2 = a + b Var[r]`. If the head's variance is too large by a factor `k`, the
    /// realized squared residual is `1/k` of what was promised, so `b = 1/k` and an
    /// OVERSTATED spread reads as a slope BELOW one. `b > 1` would be the understated case,
    /// where the law is too confident and a Kelly bettor is over-levered for a second,
    /// independent reason.
    pub fn spread_overstated(&self) -> bool {
        self.variance.beta_ci.1.is_finite() && self.variance.beta_ci.1 < 1.0
    }

    /// The mirror case: a spread resolvably TOO NARROW, which is a different failure with a
    /// different remedy and must not be reported as the same finding.
    pub fn spread_understated(&self) -> bool {
        self.variance.beta_ci.0.is_finite() && self.variance.beta_ci.0 > 1.0
    }

    /// The model's implied Kelly size as a MULTIPLE of the growth optimum, from both slopes.
    ///
    /// A single-period log-optimal fraction is `mu / Var` to leading order. The mean
    /// regression says the head's `mu` is `1/b_mean` times the truth and the variance
    /// regression says its `Var` is `1/b_var` times the truth, so the ratio of the head's
    /// uncapped fraction to the true one is `b_var / b_mean`. Below one the head is
    /// UNDER-levered in absolute terms even while its mean is inflated, which is the
    /// configuration this run is actually in and the reason neither slope alone settles
    /// whether the sizing is too large.
    pub fn kelly_scale(&self) -> f64 {
        self.variance.beta / self.mean.beta
    }

    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec![
                "  mean calibration: not measured (the pass carried no conditional moments)"
                    .to_owned(),
            ];
        }
        let mut lines = Vec::with_capacity(4);
        lines.push(format!(
            "  MZ mean  r = a + b*mu: a {:+.4} bps/bar (se {:.4}), b {:+.4} (se {:.4}, 95% CI \
             {:+.4}..{:+.4}, {:+.2} sd from 1), R^2 {:.3e} over {} bars / {} blocks{}",
            self.mean.alpha * 1e4,
            self.mean.alpha_se * 1e4,
            self.mean.beta,
            self.mean.beta_se,
            self.mean.beta_ci.0,
            self.mean.beta_ci.1,
            self.mean.slope_t_against_one(),
            self.mean.r2,
            self.mean.samples,
            self.mean.blocks,
            if self.mean.slope_resolvable() && self.mean.beta < 1.0 {
                " — the traded MEAN is INFLATED"
            } else if self.mean.slope_resolvable() {
                " — the traded mean is DAMPED"
            } else {
                ""
            },
        ));
        lines.push(format!(
            "  MZ var  (r-mu)^2 = a + b*var: a {:+.3e} (se {:.3e}), b {:+.4} (se {:.4}, 95% CI \
             {:+.4}..{:+.4}), R^2 {:.3e}",
            self.variance.alpha,
            self.variance.alpha_se,
            self.variance.beta,
            self.variance.beta_se,
            self.variance.beta_ci.0,
            self.variance.beta_ci.1,
            self.variance.r2,
        ));
        lines.push(format!(
            "  predicted sd level: {:.2} bps/bar RMS over the traded bars — the level both \
             slopes are relative to, and the denominator the position is sized by",
            self.mean_predicted_sd * 1e4,
        ));
        if self.spread_overstated() {
            lines.push(format!(
                "  spread OVERSTATED: the predicted variance is {:.2}x the realized one. \
                 Combined with the mean slope the implied Kelly scale is b_var/b_mean = \
                 {:.2}x the growth optimum, so the two miscalibrations partly cancel in \
                 ABSOLUTE size and neither one alone describes the sizing error. What the \
                 mean recalibration corrects is the OVER-DISPERSION of the mean ACROSS bars \
                 ({:.2}x too variable), which is what decides the allocation once a cap \
                 binds and absolute scale stops mattering",
                1.0 / self.variance.beta,
                self.kelly_scale(),
                1.0 / self.mean.beta,
            ));
        }
        if self.spread_understated() {
            lines.push(format!(
                "  spread UNDERSTATED: the predicted variance is only {:.2}x the realized \
                 one, so the law is too CONFIDENT and over-levers for a second reason \
                 independent of the mean; a mean-only recalibration cannot correct it",
                1.0 / self.variance.beta,
            ));
        }
        // COMMON versus VARYING, for both slopes. A miscalibration shared by every block
        // cancels exactly out of any scale-free ranking of the same forecasts, so this pair
        // rather than the pooled slope decides whether a `|mu| / sigma` selector sorts bars by
        // signal or names by asset class. Rendered as "not measured" when it is not measured,
        // because the alternative — a bool over NaN — reports homogeneity as a finding.
        for (label, fit) in [("mean", &self.mean), ("var", &self.variance)] {
            if !fit.block_dispersion_measured() {
                lines.push(format!(
                    "  {label} slope dispersion: not measured ({} of {} blocks carried a slope \
                     and a standard error)",
                    fit.beta_blocks_resolved, fit.blocks,
                ));
                continue;
            }
            let ratio = fit.block_dispersion_ratio();
            lines.push(format!(
                "  {label} slope across blocks: sd {:.4} against a {:.4} noise floor = {:.2}x, \
                 excess sd {:.4} over {} blocks — {}",
                fit.beta_block_sd,
                fit.beta_block_noise_sd,
                ratio,
                fit.beta_block_excess_sd(),
                fit.beta_blocks_resolved,
                if fit.slope_heterogeneous() {
                    "HETEROSKEDASTIC: the miscalibration is not common, so a scale-free \
                     ranking of these forecasts sorts partly on WHICH BLOCK rather than on \
                     signal strength"
                } else {
                    "COMMON across blocks within its own noise, so a scale-free ranking of \
                     these forecasts is unaffected by it"
                },
            ));
        }
        if let Some(shrink) = self.shrink() {
            lines.push(format!(
                "  growth-optimal recalibration: mu -> {:+.5e} + {:.4} * mu (a bar at the \
                 median |mu| is repriced by {:.1}%)",
                shrink.alpha,
                shrink.beta,
                100.0 * (shrink.beta - 1.0),
            ));
        }
        lines.extend(self.gradient.report_lines("as-traded"));
        match &self.outer {
            Some(outer) => {
                lines.extend(outer.report_lines());
                // The verdict needs the AS-TRADED slope, which lives here and not on the
                // decomposition: the decomposition only carries the two re-decoded arms.
                lines.extend(outer.phi.verdict_lines(&self.mean, &outer.redecoded.mean));
            }
            // Absent because the pass did not form it, which is not the same as a law with no
            // catch-all mass, so it must not read as one.
            None => lines.push("  catch-all decomposition: not measured".to_owned()),
        }
        lines
    }
}

/// Regress the realized `r` on the traded conditional mean, and the realized squared
/// residual on the traded conditional variance, over every bar of every traded window.
///
/// `blocks[w]` is window `w`'s resampling unit; every bar of a window inherits it, which is
/// what makes the interval comparable to the edge's. Windows built without conditional
/// moments are skipped, so a fixture that synthesizes position paths reports an unmeasured
/// calibration rather than a fit on nothing.
pub fn mean_calibration(windows: &[WindowPaths], blocks: &[u64]) -> MeanCalibration {
    let usable = windows.len().min(blocks.len());
    let bars: usize = windows[..usable].iter().map(WindowPaths::bars).sum();
    let mut mu = Vec::with_capacity(bars);
    let mut realized = Vec::with_capacity(bars);
    let mut variance = Vec::with_capacity(bars);
    let mut residual_squares = Vec::with_capacity(bars);
    let mut bar_blocks = Vec::with_capacity(bars);
    let mut outer = Vec::with_capacity(bars);
    let mut decomposition = Vec::with_capacity(bars);
    for (window, block) in windows[..usable].iter().zip(blocks) {
        if !window.has_moments() {
            continue;
        }
        for ((r, m), v) in window
            .realized_log()
            .iter()
            .zip(&window.predicted_mean)
            .zip(&window.predicted_var)
        {
            mu.push(*m);
            realized.push(*r);
            variance.push(*v);
            residual_squares.push((r - m) * (r - m));
            bar_blocks.push(*block);
        }
        if window.has_decomposition() {
            outer.extend(window.outer_mass.iter().copied());
            for (((mass, signed), m), v) in window
                .outer_mass
                .iter()
                .zip(&window.outer_signed)
                .zip(&window.trimmed_mean)
                .zip(&window.trimmed_var)
            {
                decomposition.push(OuterBar {
                    mass: *mass,
                    signed: *signed,
                    interior_mean: *m,
                    interior_var: *v,
                });
            }
        }
    }
    if bar_blocks.is_empty() {
        return MeanCalibration::nan();
    }
    // The gradient's cells need one outer mass per bar or none at all; a partially decomposed
    // pass would silently pair a cell's slope with a different cell's mass.
    let outer = (outer.len() == mu.len()).then_some(outer);
    MeanCalibration {
        mean: mincer_zarnowitz(&mu, &realized, &bar_blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        variance: mincer_zarnowitz(
            &variance,
            &residual_squares,
            &bar_blocks,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        ),
        // Root-MEAN-SQUARE rather than the mean of the roots, so it is the sd of the pooled
        // predictive law rather than an average of per-bar widths. Those differ whenever
        // volatility is heteroskedastic, which on a mixed universe it emphatically is.
        mean_predicted_sd: (variance.iter().sum::<f64>() / variance.len() as f64).sqrt(),
        gradient: volatility_gradient(
            &mu,
            &realized,
            &variance,
            &residual_squares,
            outer.as_deref(),
            &bar_blocks,
        ),
        outer: (decomposition.len() == mu.len())
            .then(|| OuterDecomposition::measure(&decomposition, &realized, &bar_blocks, &mu)),
    }
}

/// One bar's catch-all sufficient statistics, which are all a re-decode needs.
#[derive(Clone, Copy, Debug)]
pub struct OuterBar {
    /// Total mass in the two catch-all bins.
    pub mass: f64,
    /// Upper catch-all mass minus lower.
    pub signed: f64,
    /// Mean of the law RESTRICTED to the interior bins and renormalized.
    pub interior_mean: f64,
    /// Variance of that same restricted law.
    pub interior_var: f64,
}

impl OuterBar {
    /// `(mu, var)` of the FULL law with the two catch-alls moved to `decode`.
    ///
    /// Exact, not an approximation, and it needs no second forward pass: a mixture of the
    /// interior law with weight `1 - mass` and two atoms carries its moments in closed form, so
    /// the interior mean and variance plus the two masses are sufficient statistics for the law
    /// under ANY choice of decode point. That is what makes the fitted-mean arm and the
    /// original-centre arm the same measurement seen twice rather than two passes.
    pub fn redecoded(&self, decode: (f64, f64)) -> (f64, f64) {
        let lower = 0.5 * (self.mass - self.signed);
        let upper = 0.5 * (self.mass + self.signed);
        let interior = 1.0 - self.mass;
        let mean = interior * self.interior_mean + lower * decode.0 + upper * decode.1;
        let second = interior * (self.interior_var + self.interior_mean * self.interior_mean)
            + lower * decode.0 * decode.0
            + upper * decode.1 * decode.1;
        (mean, (second - mean * mean).max(0.0))
    }
}

/// Fitted conditional means of the two catch-all bins of `r`, in LOG-return units.
///
/// `E[r | bin]` on training data for bins `0` and `127`, measured off the persisted 300s
/// quantile grid: `-277.12` and `+283.62` bps, against the `-883.32` and `+880.38` the decode
/// currently uses. A catch-all's mass sits near its INNER edge, so the bound is the wrong
/// representative for a moment by a factor of about 3.1 while being the right one for a sample.
///
/// Declared here because the support artifact does NOT carry fitted per-bin means. The live
/// `long_data/bars/bar_supports.300.json` and every checkpoint sidecar in this tree are
/// `format_version` 4, whose key set is exactly
/// `{dof_names, format_version, hi, lo, masses, num_bins, provenance, smoothed_marginal}`, while
/// [`BAR_SUPPORTS_MOMENTS_VERSION`] is 5 - so `BarSupports::bin_means_measured` is FALSE and
/// `BarSupports::bin_means` returns `None` on every artifact that exists. This pair is therefore
/// not a two-bin approximation to a landed object; it is the ONLY fitted-mean decode there is,
/// and it is a two-bin one: it re-prices bins `0` and `127` and leaves the other 126 at their
/// MIDPOINTS. Measured independently off the same artifact, the two marginal-decoded levels
/// agree to `0.28%` in sd - `45.450` bps midpoint interior against `45.321` centroid - because
/// only 19 interior bins move at all and the largest move is `4.50` bps at bin `126`. That is
/// why the arm is quoted for the MEAN, which is linear in each bin's law and therefore
/// insensitive at this scale, and NOT as the expected value of a three-decimal variance
/// criterion. Whoever lands a 128-bin means vector must read the same 128-vector into
/// `predicted_var` and into the comparator, or the mismatch reappears one bin inboard of where
/// this removes it.
pub const OUTER_REDECODE: (f64, f64) = (-0.027712, 0.028362);

/// One decode convention, fitted on the same bars as every other.
///
/// # The mean arm is a point estimate and the variance arm is a BOUND
///
/// A single value per catch-all bin is exactly the right representative for a MEAN - a mean is
/// linear in the bin's law, so `E[r]` under the true within-bin law equals `E[r]` under a point
/// mass at that law's mean. It is the WRONG representative for a variance, which is not linear:
/// collapsing a bin to a point discards its within-bin dispersion, measured independently at
/// `12.02%` of the true second moment with `98.6%` of that sitting in these two bins.
///
/// So `predicted_sd` and the variance slope of a re-decoded arm are one-sided: the sd is a LOWER
/// bound on what a second-moment decode reads, and the variance slope is therefore an UPPER
/// bound. The mean slope carries no such caveat. Anyone comparing the post-fix pipeline against
/// this arm must compare against a second-moment decode, not against this variance column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeArm {
    pub mean: MzFit,
    pub variance: MzFit,
    /// RMS predicted sd of `r` under this convention, so the arms are comparable as LEVELS and
    /// not only as slopes. One-sided for a re-decoded arm - see the type's own doc.
    pub predicted_sd: f64,
}

impl DecodeArm {
    /// Fit both moments of one convention. `moments[i]` is `(mu, var)` for bar `i`.
    fn measure(moments: &[(f64, f64)], realized: &[f64], blocks: &[u64]) -> Self {
        let mut mu = Vec::with_capacity(moments.len());
        let mut variance = Vec::with_capacity(moments.len());
        let mut residual_squares = Vec::with_capacity(moments.len());
        for ((m, v), r) in moments.iter().zip(realized) {
            mu.push(*m);
            variance.push(*v);
            residual_squares.push((r - m) * (r - m));
        }
        Self {
            mean: mincer_zarnowitz(&mu, realized, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
            variance: mincer_zarnowitz(
                &variance,
                &residual_squares,
                blocks,
                BOOTSTRAP_DRAWS,
                BOOTSTRAP_SEED,
            ),
            predicted_sd: (variance.iter().sum::<f64>() / variance.len().max(1) as f64).sqrt(),
        }
    }
}

// ---------------------------------------------------------------------------
// What SHARE of the forecast's variation the two catch-all bins carry
// ---------------------------------------------------------------------------

/// Index of each bootstrapped scalar inside [`PhiCensus`].
///
/// One ordering, used by the estimator that fills the array, by the accessors that read it and by
/// [`PHI_LABELS`]. A parallel field list would be two orderings a permutation could silently
/// desynchronize, and these are dimensionless numbers near zero and one that no reader could tell
/// apart if they were swapped.
const PHI_SHARE: usize = 0;
const PHI_INTERIOR_SHARE: usize = 1;
const PHI_CROSS_SHARE: usize = 2;
const PHI_INTERIOR_OUTER_CORR: usize = 3;
const PHI_BETA_MEASURED: usize = 4;
const PHI_BETA_EXACT: usize = 5;
const PHI_BETA_MODEL: usize = 6;
const PHI_BETA_REDECODED: usize = 7;
const PHI_GAP_MECHANISM: usize = 8;
const PHI_GAP_MAP: usize = 9;
const PHI_FITTED_SHARE: usize = 10;
const PHI_CHANNEL_RATIO: usize = 11;
const PHI_OUTER_CORR: usize = 12;
pub const PHI_SCALARS: usize = 13;

/// Series names of [`PhiCensus::scalars`], in index order.
///
/// Every label that names a PREDICTION also names the assumption it rests on, in the label
/// itself. A chart's title is normalized before it is drawn - emphasis in a title does not
/// survive - while series legend labels render verbatim, so a caveat is only legible if it is
/// here. The standard this is written to is not "is the number emitted" but "can a reader reach
/// the wrong conclusion from what is drawn", which is the defect a per-pass census titled as the
/// truth committed elsewhere in this tree.
pub const PHI_LABELS: [&str; PHI_SCALARS] = [
    "phi = Var(catch-all term) / Var(as-traded mean)",
    "interior share = Var(I) / Var(f)",
    "cross share 2Cov(I,D)/Var(f) - NONZERO REFUTES the phi->beta map",
    "corr(interior, fitted catch-all term) - the Cov(I,T)=0 test",
    "beta MEASURED, as-traded mean (in-draw, pairable)",
    "beta PREDICTED exact - assumes only that the fitted decode is the true mean",
    "beta PREDICTED phi-model - also assumes Cov(I,T)=0 and one channel",
    "beta MEASURED, fitted-decode arm - the map's premise is that this reads 1.0",
    "paired gap measured - exact: zero means the decode explains the slope",
    "paired gap exact - phi-model: zero means the published map is a sound summary",
    "phi under the fitted decode",
    "channel ratio Var(D_edge)/(g^2 Var(D_fitted)) - one if a single channel",
    "corr(D_edge, D_fitted)",
];

/// A pooled point estimate with a block-bootstrap standard error and percentile interval.
///
/// `point` is the estimate over EVERY bar, not the mean of the resampled draws: the draws measure
/// how far the pooled number would move under resampling of `(symbol, calendar month)` blocks,
/// which is the convention [`MzFit`] reports its slope under and therefore the only one whose
/// intervals are comparable to a slope's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockedScalar {
    pub point: f64,
    pub se: f64,
    pub ci: (f64, f64),
}

impl BlockedScalar {
    pub fn nan() -> Self {
        Self { point: f64::NAN, se: f64::NAN, ci: (f64::NAN, f64::NAN) }
    }

    pub fn measured(&self) -> bool {
        self.point.is_finite() && self.ci.0.is_finite() && self.ci.1.is_finite()
    }

    /// True when the interval EXCLUDES `value`, i.e. the estimate is resolvably away from it.
    ///
    /// Gated on the interval being measured, so an unmeasured scalar answers `false` — "not
    /// resolvably different" — rather than manufacturing a resolution out of `NaN` comparisons,
    /// which are all false and would otherwise read as "excludes nothing" by accident rather than
    /// by decision.
    pub fn excludes(&self, value: f64) -> bool {
        self.measured() && !(self.ci.0..=self.ci.1).contains(&value)
    }

    /// True when two blocked estimates' intervals overlap at all.
    ///
    /// The weakest honest reading of "these agree", and only for estimates that are NOT paired:
    /// two quantities computed from the same bars have strongly correlated errors, so this
    /// declares agreement too easily and disagreement never. Prefer a paired difference - see
    /// [`PhiCensus::mechanism_gap`] - wherever one exists.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.measured()
            && other.measured()
            && self.ci.0 <= other.ci.1
            && other.ci.0 <= self.ci.1
    }
}

/// Second-moment sufficient statistics of one block's four channels.
///
/// `f` is the as-traded (edge-decoded) conditional mean, `d` the part of it the two catch-all bins
/// contribute, `t` what those same two bins contribute under [`OUTER_REDECODE`], and `y` the
/// realized log return. Fourteen numbers per block are sufficient for every variance, covariance,
/// correlation AND SLOPE the census reports, so a bootstrap refit costs one pass over the BLOCKS
/// rather than over the bars - the same property [`RegressionSums`] gives a single slope, for the
/// same reason.
///
/// Carrying `y` here rather than pairing against a separately-bootstrapped [`MzFit`] is what makes
/// the measured slope and its prediction differenceable INSIDE a draw. Pairing across two
/// estimators would additionally require their surviving row sets to coincide, which is a
/// precondition nothing could check from the outside; pairing inside one accumulator makes it
/// true by construction.
#[derive(Clone, Copy, Debug, Default)]
struct DecodeSums {
    n: f64,
    f: f64,
    d: f64,
    t: f64,
    y: f64,
    ff: f64,
    dd: f64,
    tt: f64,
    fd: f64,
    ft: f64,
    fy: f64,
    dt: f64,
    dy: f64,
    ty: f64,
}

impl DecodeSums {
    fn push(&mut self, f: f64, d: f64, t: f64, y: f64) {
        self.n += 1.0;
        self.f += f;
        self.d += d;
        self.t += t;
        self.y += y;
        self.ff += f * f;
        self.dd += d * d;
        self.tt += t * t;
        self.fd += f * d;
        self.ft += f * t;
        self.fy += f * y;
        self.dt += d * t;
        self.dy += d * y;
        self.ty += t * y;
    }

    fn absorb(&mut self, other: &Self) {
        self.n += other.n;
        self.f += other.f;
        self.d += other.d;
        self.t += other.t;
        self.y += other.y;
        self.ff += other.ff;
        self.dd += other.dd;
        self.tt += other.tt;
        self.fd += other.fd;
        self.ft += other.ft;
        self.fy += other.fy;
        self.dt += other.dt;
        self.dy += other.dy;
        self.ty += other.ty;
    }

    /// Every reported scalar, from the fourteen sums and the measured directional gain.
    ///
    /// The interior contribution is `I = f - d` IDENTICALLY - not a separate accumulator that
    /// could drift from one - so `Var(f) = Var(I) + 2Cov(I,D) + Var(D)` holds exactly and the
    /// three shares sum to one to floating precision. That identity is why the CROSS share is
    /// reported rather than assumed away: the `phi`-to-`beta` map is derived under `Cov(I,T) = 0`,
    /// so a cross share that is not small refutes the map rather than inconveniencing it.
    fn census(&self, gain: f64) -> [f64; PHI_SCALARS] {
        let mut out = [f64::NAN; PHI_SCALARS];
        if self.n < 2.0 {
            return out;
        }
        let n = self.n;
        let var = |s: f64, ss: f64| (ss - s * s / n) / (n - 1.0);
        let cov = |a: f64, b: f64, ab: f64| (ab - a * b / n) / (n - 1.0);
        let vf = var(self.f, self.ff);
        let vd = var(self.d, self.dd);
        let vt = var(self.t, self.tt);
        let cfd = cov(self.f, self.d, self.fd);
        let cft = cov(self.f, self.t, self.ft);
        let cfy = cov(self.f, self.y, self.fy);
        let cdt = cov(self.d, self.t, self.dt);
        let cdy = cov(self.d, self.y, self.dy);
        let cty = cov(self.t, self.y, self.ty);
        if !(vf > 0.0) {
            return out;
        }
        // `I = f - d`, so every interior moment is an exact combination of the channels'.
        let vi = vf - 2.0 * cfd + vd;
        let cid = cfd - vd;
        let cit = cft - cdt;
        // The FITTED-decode mean is `I + t = f - d + t`.
        let v_fitted = vi + 2.0 * cit + vt;
        let cy_fitted = cfy - cdy + cty;
        let phi = vd / vf;
        out[PHI_SHARE] = phi;
        out[PHI_INTERIOR_SHARE] = vi / vf;
        out[PHI_CROSS_SHARE] = 2.0 * cid / vf;
        out[PHI_INTERIOR_OUTER_CORR] = if vi > 0.0 && vt > 0.0 {
            cit / (vi * vt).sqrt()
        } else {
            f64::NAN
        };
        // The OLS slope of the realized return on the as-traded mean: the same normal equation
        // `Cov(x,y)/Var(x)` that `RegressionSums::solve` uses, over the same rows, so this is the
        // calibration slope itself and not a second estimator of it.
        let measured = cfy / vf;
        // `Cov(f, I + t) / Var(f)`. Under the single assumption that the fitted decode IS the
        // conditional mean this is the calibration slope EXACTLY, with no orthogonality
        // assumption: `f` and `I + t` are both functions of the same information set, so
        // `Cov(f, y) = Cov(f, E[y | information])` identically.
        let exact = (vf - cfd + cft) / vf;
        let model = 1.0 - phi * (1.0 - 1.0 / gain);
        out[PHI_BETA_MEASURED] = measured;
        out[PHI_BETA_EXACT] = exact;
        out[PHI_BETA_MODEL] = model;
        out[PHI_BETA_REDECODED] = if v_fitted > 0.0 {
            cy_fitted / v_fitted
        } else {
            f64::NAN
        };
        // Differences formed INSIDE the draw, which is the whole point of accumulating `y` here:
        // the measured slope and its prediction share the realized return and the resampled
        // blocks, so most of their sampling variance is common and cancels. An interval on either
        // one is far wider than the interval on their difference, and the difference is what the
        // hypothesis is about.
        out[PHI_GAP_MECHANISM] = measured - exact;
        out[PHI_GAP_MAP] = exact - model;
        out[PHI_FITTED_SHARE] = if v_fitted > 0.0 { vt / v_fitted } else { f64::NAN };
        out[PHI_CHANNEL_RATIO] = if vt > 0.0 && gain.is_finite() && gain != 0.0 {
            vd / (gain * gain * vt)
        } else {
            f64::NAN
        };
        out[PHI_OUTER_CORR] = if vd > 0.0 && vt > 0.0 {
            cdt / (vd * vt).sqrt()
        } else {
            f64::NAN
        };
        out
    }
}

/// Least-squares recovery of the decode the forecasts were actually formed under, in the
/// SYMMETRIC / DIRECTIONAL basis.
///
/// # Why the decode is measured here rather than read from a constant
///
/// The catch-all contribution is `D = c_lo p_0 + c_hi p_127` with two decode values fixed across
/// bars and two probabilities varying, so a no-intercept regression of `D` on the two masses
/// recovers the decode EXACTLY and its residual is zero up to floating error. Three things
/// follow, none available from a hardcoded pair:
///
/// * The residual is a live check that `D` is what this module claims it is. A non-zero RMS says
///   the identity `D = f - (1 - mass) * interior_mean` has broken, which is the one way the whole
///   census could be silently measuring something else.
/// * The recovered decode is a GEOMETRY fingerprint. Several checkpoints scored in one pass must
///   recover the same one; if they do not they resolved different supports and no slope of one is
///   comparable to a slope of another - a failure otherwise invisible because every individual
///   number stays correct.
/// * The directional gain the amplification argument turns on stops being an inherited constant
///   and becomes a per-checkpoint measurement.
///
/// # Why `(s, a)` and not `(p_0, p_127)`
///
/// Substituting `p_0 = (s - a)/2`, `p_127 = (s + a)/2`,
///
/// ```text
/// D = s (c_lo + c_hi)/2 + a (c_hi - c_lo)/2
/// ```
///
/// so the design columns become the common catch-all LEVEL and the directional TILT, which are
/// nearly orthogonal by construction, where `p_0` and `p_127` are two similar small numbers that
/// move together across bars. Regressing on the latter is the classic ill-conditioned fit: it
/// returns a well-fitting pair whose individual coefficients are meaningless, with a SMALL
/// residual and a large condition number. `(s, a)` is also the basis the amplification is defined
/// in - it is the ratio of the two `a` coefficients - so the quantity needed comes out directly.
/// The condition number is reported beside the residual precisely because a small residual alone
/// does not establish that either coefficient is identified.
#[derive(Clone, Copy, Debug, Default)]
struct EdgeRecovery {
    n: f64,
    ss: f64,
    sa: f64,
    aa: f64,
    sd: f64,
    ad: f64,
    dd: f64,
}

impl EdgeRecovery {
    fn push(&mut self, level: f64, tilt: f64, d: f64) {
        self.n += 1.0;
        self.ss += level * level;
        self.sa += level * tilt;
        self.aa += tilt * tilt;
        self.sd += level * d;
        self.ad += tilt * d;
        self.dd += d * d;
    }

    /// `(midpoint, half-span, residual RMS, condition number)`, all `NaN` when degenerate.
    ///
    /// The condition number is the 2-norm one of the NORMAL matrix, i.e. the square of the design
    /// matrix's. Stated because the two differ by a factor of two in the exponent and quoting the
    /// wrong one understates an ill-conditioned fit.
    fn solve(&self) -> (f64, f64, f64, f64) {
        let det = self.ss * self.aa - self.sa * self.sa;
        if self.n < 2.0 || !(det.abs() > 0.0) {
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }
        let midpoint = (self.aa * self.sd - self.sa * self.ad) / det;
        let half_span = (self.ss * self.ad - self.sa * self.sd) / det;
        let residual = self.dd - midpoint * self.sd - half_span * self.ad;
        let trace = self.ss + self.aa;
        let spread = ((self.ss - self.aa) * (self.ss - self.aa) + 4.0 * self.sa * self.sa).sqrt();
        let (high, low) = (0.5 * (trace + spread), 0.5 * (trace - spread));
        (
            midpoint,
            half_span,
            (residual.max(0.0) / self.n).sqrt(),
            if low > 0.0 { high / low } else { f64::INFINITY },
        )
    }
}

/// How much of the forecast mean's VARIATION the two catch-all bins carry, and what that predicts
/// for the calibration slope.
///
/// # The prediction this exists to falsify
///
/// Write the as-traded conditional mean as `f = I + D`, where `I` is the interior bins'
/// contribution and `D = c_lo p_0 + c_hi p_127` is the two catch-alls'. In the level/tilt basis
/// `s = p_127 + p_0`, `a = p_127 - p_0`,
///
/// ```text
/// D = s (c_lo + c_hi)/2  +  a (c_hi - c_lo)/2
/// ```
///
/// so the SYMMETRIC channel is priced by the decode's midpoint and the DIRECTIONAL channel by its
/// half-span. On the live 300s geometry the edge decode's half-span is `881.85` bps against the
/// fitted decode's `280.37`, a directional amplification of `g = 3.145x`, while the midpoints are
/// `-1.47` and `+3.25` bps - the two channels are not proportional. Modelling the forecast as
/// `f = I + g T` with the fitted decode giving the true conditional mean `E[y | past] = I + T`,
/// and ASSUMING `Cov(I, T) = 0`, gives
///
/// ```text
/// beta = 1 - phi (1 - 1/g),   phi = Var(g T) / Var(f)
/// ```
///
/// which maps `beta = 0.8777` to `phi = 0.179` and `beta = 1.0058` to `phi = 0`. That map is a
/// MODEL, and this type measures every input it rests on including the two assumptions it hides:
///
/// * `Cov(I, T) = 0`. Reported as the CROSS share `2Cov(I,D)/Var(f)` - which with
///   `Var(f) = Var(I) + 2Cov(I,D) + Var(D)` makes the three shares sum to one exactly, so the
///   cross term is not a nuisance to bound but a third share to read - and as `corr(I, T)`. If it
///   is material the map is wrong, and that is a refutation of the mechanism, not a nuisance.
/// * `D = g T`, i.e. that the symmetric channel is negligible. Reported as the CHANNEL RATIO
///   `Var(D) / (g^2 Var(T))`, one when the single-channel picture holds, and as `corr(D, T)`.
///
/// Because both assumptions are avoidable, the census also reports the EXACT prediction
/// `Cov(f, I + T) / Var(f)`, which needs neither. The three numbers to read together are the
/// MEASURED slope, the EXACT prediction and the MODEL prediction, and the two gaps between them
/// are separate claims that can come apart:
///
/// * measured == exact says the catch-all decode accounts for the slope - the miscalibration is a
///   REPRESENTATION artifact rather than a learned error in the conditional mean.
/// * exact == model says the published `phi`-to-`beta` map is a sound summary of that mechanism.
///
/// Both gaps are formed INSIDE each bootstrap draw, so their intervals are paired rather than the
/// difference of two independent ones. And the whole chain rests on one PREMISE that is itself
/// measured: the fitted-decode arm's own slope must read `1.0`. An arm resolvably away from one
/// invalidates the exact prediction as well, so it is reported beside the rest instead of assumed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhiCensus {
    scalars: [BlockedScalar; PHI_SCALARS],
    /// Half-span ratio of the recovered decode to [`OUTER_REDECODE`]: the `g` above, MEASURED.
    /// Static geometry, so it is held fixed across the bootstrap draws.
    pub directional_gain: f64,
    /// The recovered decode's `(c_lo + c_hi)/2` and `(c_hi - c_lo)/2`, in LOG-return units.
    pub recovered_midpoint: f64,
    pub recovered_half_span: f64,
    /// The same thing as `(c_lo, c_hi)`, for comparison against a support artifact's `lo`/`hi`.
    pub recovered_edge: (f64, f64),
    /// RMS of `D` minus its two-term reconstruction. Zero up to floating error whenever the census
    /// is measuring what it claims to.
    pub recovery_residual_rms: f64,
    /// Condition number of the recovery's normal matrix. A small residual beside a large
    /// condition number means the fit reproduces `D` without identifying either coefficient.
    pub recovery_condition: f64,
    pub blocks: usize,
    pub samples: usize,
}

impl PhiCensus {
    pub fn nan() -> Self {
        Self {
            scalars: [BlockedScalar::nan(); PHI_SCALARS],
            directional_gain: f64::NAN,
            recovered_midpoint: f64::NAN,
            recovered_half_span: f64::NAN,
            recovered_edge: (f64::NAN, f64::NAN),
            recovery_residual_rms: f64::NAN,
            recovery_condition: f64::NAN,
            blocks: 0,
            samples: 0,
        }
    }

    pub fn measured(&self) -> bool {
        self.samples > 0
    }

    /// Every scalar in [`PHI_LABELS`] order, for a report writer that charts all of them.
    pub fn scalars(&self) -> &[BlockedScalar; PHI_SCALARS] {
        &self.scalars
    }

    /// Share of the as-traded mean's variance carried by the two catch-all bins.
    pub fn phi(&self) -> BlockedScalar {
        self.scalars[PHI_SHARE]
    }

    pub fn interior_share(&self) -> BlockedScalar {
        self.scalars[PHI_INTERIOR_SHARE]
    }

    /// `2Cov(I,D)/Var(f)`: the term the `phi`-to-`beta` map assumes away.
    pub fn cross_share(&self) -> BlockedScalar {
        self.scalars[PHI_CROSS_SHARE]
    }

    pub fn interior_outer_corr(&self) -> BlockedScalar {
        self.scalars[PHI_INTERIOR_OUTER_CORR]
    }

    /// The calibration slope itself, accumulated here so it can be paired with its predictions.
    pub fn beta_measured(&self) -> BlockedScalar {
        self.scalars[PHI_BETA_MEASURED]
    }

    /// `Cov(f, I + T)/Var(f)`: assumes only that the fitted decode is the conditional mean.
    pub fn beta_exact(&self) -> BlockedScalar {
        self.scalars[PHI_BETA_EXACT]
    }

    /// `1 - phi (1 - 1/g)`, the published map's prediction.
    pub fn beta_model(&self) -> BlockedScalar {
        self.scalars[PHI_BETA_MODEL]
    }

    /// The fitted-decode arm's own slope. The map's premise is that this reads `1.0`.
    pub fn beta_redecoded(&self) -> BlockedScalar {
        self.scalars[PHI_BETA_REDECODED]
    }

    /// Paired `measured - exact`. Zero means the amplification mechanism explains the slope.
    pub fn mechanism_gap(&self) -> BlockedScalar {
        self.scalars[PHI_GAP_MECHANISM]
    }

    /// Paired `exact - model`. Zero means the published map summarizes the mechanism soundly.
    pub fn map_gap(&self) -> BlockedScalar {
        self.scalars[PHI_GAP_MAP]
    }

    pub fn fitted_share(&self) -> BlockedScalar {
        self.scalars[PHI_FITTED_SHARE]
    }

    /// `Var(D)/(g^2 Var(T))`: one when the amplification really is a single channel.
    pub fn channel_ratio(&self) -> BlockedScalar {
        self.scalars[PHI_CHANNEL_RATIO]
    }

    pub fn outer_corr(&self) -> BlockedScalar {
        self.scalars[PHI_OUTER_CORR]
    }

    /// The three shares, which sum to one by construction. Reported so a reader can watch the
    /// identity hold rather than take it on faith; a sum away from one is a bug in this module.
    pub fn share_sum(&self) -> f64 {
        self.phi().point + self.interior_share().point + self.cross_share().point
    }

    /// `phi` the published map would need to produce a given slope, inverted from
    /// `beta = 1 - phi (1 - 1/g)`.
    ///
    /// The comparison the whole task turns on: a historical slope implies a `phi`, and the
    /// measured `phi` either is that number or is not. `NaN` when the gain is degenerate rather
    /// than a division that reads as an answer.
    pub fn phi_implied_by(&self, beta: f64) -> f64 {
        let attenuation = 1.0 - 1.0 / self.directional_gain;
        if !attenuation.is_finite() || attenuation == 0.0 {
            return f64::NAN;
        }
        (1.0 - beta) / attenuation
    }

    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec![
                "  catch-all variance census: not measured (the pass carried no decomposition)"
                    .to_owned(),
            ];
        }
        let mut lines = vec![format!(
            "  decode RECOVERED from the rows: midpoint {:+.4} bps, half-span {:+.4} bps => \
             (lo {:+.2}, hi {:+.2}) bps, against the fitted midpoint {:+.4} / half-span {:+.4}; \
             directional gain g = {:.5}x; residual {:.3e} bps/bar, normal-matrix condition \
             {:.3e}, over {} bars / {} blocks",
            self.recovered_midpoint * 1e4,
            self.recovered_half_span * 1e4,
            self.recovered_edge.0 * 1e4,
            self.recovered_edge.1 * 1e4,
            0.5 * (OUTER_REDECODE.0 + OUTER_REDECODE.1) * 1e4,
            0.5 * (OUTER_REDECODE.1 - OUTER_REDECODE.0) * 1e4,
            self.directional_gain,
            self.recovery_residual_rms * 1e4,
            self.recovery_condition,
            self.samples,
            self.blocks,
        )];
        for (label, scalar) in PHI_LABELS.iter().zip(&self.scalars) {
            lines.push(format!(
                "  {label}: {:+.5} (se {:.5}, 95% CI {:+.5}..{:+.5})",
                scalar.point, scalar.se, scalar.ci.0, scalar.ci.1,
            ));
        }
        lines.push(format!(
            "  the three shares sum to {:.6}, which is Var(f) = Var(I) + 2Cov(I,D) + Var(D) \
             holding rather than a coincidence",
            self.share_sum(),
        ));
        lines
    }

    /// The verdict on the amplification mechanism, from the PAIRED gaps.
    ///
    /// `edge` and `redecoded` are the independently-bootstrapped slopes of the same two arms, used
    /// only as a CROSS-CHECK: this census recomputes both slopes from its own accumulator, so the
    /// two routes must agree to floating error and a disagreement is a defect in one of them. The
    /// verdict itself is read off the paired gaps, which are strictly sharper.
    pub fn verdict_lines(&self, edge: &MzFit, redecoded: &MzFit) -> Vec<String> {
        if !self.measured() {
            return Vec::new();
        }
        let mut lines = vec![format!(
            "  slope CROSS-CHECK: this census reads {:+.6} for the as-traded arm and {:+.6} for \
             the fitted-decode arm; the independent regressions read {:+.6} and {:+.6} \
             (differences {:+.2e} and {:+.2e} — anything but floating error is a defect)",
            self.beta_measured().point,
            self.beta_redecoded().point,
            edge.beta,
            redecoded.beta,
            self.beta_measured().point - edge.beta,
            self.beta_redecoded().point - redecoded.beta,
        )];
        let mechanism = self.mechanism_gap();
        lines.push(format!(
            "  MECHANISM (paired): measured minus exact prediction = {:+.5} (95% CI \
             {:+.5}..{:+.5}) — {}",
            mechanism.point,
            mechanism.ci.0,
            mechanism.ci.1,
            if !mechanism.measured() {
                "UNRESOLVED, the paired interval did not form"
            } else if mechanism.excludes(0.0) {
                "REFUTED on these bars: the catch-all decode does NOT account for the slope, so \
                 the residual miscalibration is a property of the head and not of the decode"
            } else {
                "CONSISTENT: the catch-all decode accounts for the slope within the paired \
                 interval, so the miscalibration is a REPRESENTATION artifact"
            },
        ));
        let map = self.map_gap();
        lines.push(format!(
            "  MAP (paired): exact minus phi-model prediction = {:+.5} (95% CI {:+.5}..{:+.5}), \
             cross share {:+.5} moves beta by at most {:.5}, channel ratio {:+.4} — {}",
            map.point,
            map.ci.0,
            map.ci.1,
            self.cross_share().point,
            0.5 * self.cross_share().point.abs()
                * (1.0 - 1.0 / self.directional_gain).abs(),
            self.channel_ratio().point,
            if !map.measured() {
                "UNRESOLVED"
            } else if map.excludes(0.0) {
                "the published beta = 1 - phi(1 - 1/g) does NOT reproduce the exact prediction, so \
                 its Cov(I,T) = 0 and single-channel assumptions are not innocuous here"
            } else {
                "the published beta = 1 - phi(1 - 1/g) reproduces the exact prediction"
            },
        ));
        let arm = self.beta_redecoded();
        lines.push(format!(
            "  PREMISE: the fitted-decode arm's slope is {:+.4} (CI {:+.4}..{:+.4}), which {} \
             perfect calibration. The exact prediction assumes this arm IS the conditional mean, \
             so an arm resolvably away from 1.0 invalidates the prediction as well as the map",
            arm.point,
            arm.ci.0,
            arm.ci.1,
            if arm.excludes(1.0) { "EXCLUDES" } else { "contains" },
        ));
        lines.push(format!(
            "  INVERSION: this census's phi is {:+.5}; the map would need phi = {:+.5} to produce \
             the measured {:+.4}, and phi = {:+.5} to produce a slope of 1.0000",
            self.phi().point,
            self.phi_implied_by(self.beta_measured().point),
            self.beta_measured().point,
            self.phi_implied_by(1.0),
        ));
        lines
    }
}

/// Measure the catch-all variance census over every bar of a decomposed pass.
///
/// `edge_mean[i]` is bar `i`'s as-traded conditional mean - the same `mu` the calibration slope
/// regresses - and `realized[i]` the realized LOG return, the same `y`. The catch-all
/// contribution is taken as `D = mu - (1 - mass) * interior_mean`, which is
/// `c_lo p_0 + c_hi p_127` IDENTICALLY and needs no decode constant: the interior sum is exactly
/// what `trimmed_mean` renormalizes, so the subtraction cancels the 126 interior terms and
/// nothing else. It is also well conditioned - on this geometry `D` is of the same order as `mu`
/// itself, because `881.85` bps of half-span against `1.45%` of mass is a first-order
/// contribution rather than a correction - so the difference of two same-order quantities is
/// benign.
///
/// The row filter is `{mu, D, T, y}` all finite, which is a subset of the `{mu, y}` filter
/// [`mincer_zarnowitz`] applies to the same bars, so `beta_measured` is that regression's slope on
/// the same rows whenever the decomposition is finite - a property the tests pin rather than
/// assume.
fn measure_phi(
    bars: &[OuterBar],
    edge_mean: &[f64],
    realized: &[f64],
    blocks: &[u64],
) -> PhiCensus {
    assert_eq!(bars.len(), edge_mean.len(), "one forecast mean per decomposed bar");
    assert_eq!(bars.len(), realized.len(), "one realized return per decomposed bar");
    assert_eq!(bars.len(), blocks.len(), "every bar needs a block");

    let mut grouped: BTreeMap<u64, DecodeSums> = BTreeMap::new();
    let mut recovery = EdgeRecovery::default();
    let mut samples = 0usize;
    for (((bar, mean), y), block) in bars.iter().zip(edge_mean).zip(realized).zip(blocks) {
        let lower = 0.5 * (bar.mass - bar.signed);
        let upper = 0.5 * (bar.mass + bar.signed);
        let d = mean - (1.0 - bar.mass) * bar.interior_mean;
        let t = lower * OUTER_REDECODE.0 + upper * OUTER_REDECODE.1;
        if !mean.is_finite() || !y.is_finite() || !d.is_finite() || !t.is_finite() {
            continue;
        }
        grouped.entry(*block).or_default().push(*mean, d, t, *y);
        // The LEVEL and TILT columns, not the two masses: see [`EdgeRecovery`] for why the basis
        // is the difference between an identified fit and a well-fitting meaningless one.
        recovery.push(bar.mass, bar.signed, d);
        samples += 1;
    }
    let (recovered_midpoint, recovered_half_span, recovery_residual_rms, recovery_condition) =
        recovery.solve();
    // MEASURED geometry, held fixed across the draws because it is a property of the support and
    // not of the sample: resampling blocks must move the variance shares, never the bin values
    // the decode reads.
    let directional_gain =
        recovered_half_span / (0.5 * (OUTER_REDECODE.1 - OUTER_REDECODE.0));

    let totals: Vec<DecodeSums> = grouped.into_values().collect();
    let mut pooled = DecodeSums::default();
    for block in &totals {
        pooled.absorb(block);
    }
    let point = pooled.census(directional_gain);
    let mut census = PhiCensus {
        scalars: std::array::from_fn(|index| BlockedScalar {
            point: point[index],
            ..BlockedScalar::nan()
        }),
        directional_gain,
        recovered_midpoint,
        recovered_half_span,
        recovered_edge: (
            recovered_midpoint - recovered_half_span,
            recovered_midpoint + recovered_half_span,
        ),
        recovery_residual_rms,
        recovery_condition,
        blocks: totals.len(),
        samples,
    };
    if totals.len() < 2 {
        // One block is one observation: there is no dispersion to estimate, and a zero-width
        // interval reported as precision is the failure this refuses to commit.
        return census;
    }

    // The same stream as `mincer_zarnowitz` and `block_bootstrap`: same RNG, same seed, same draw
    // count, blocks visited in the same `BTreeMap` order. A `phi` interval is therefore taken over
    // the same construction as the slope it sits beside, and every scalar of a single draw comes
    // from ONE resample, which is what makes the two gaps paired rather than differences of
    // independent intervals.
    let mut rng = ChaCha12Rng::seed_from_u64(BOOTSTRAP_SEED);
    let mut columns: [Vec<f64>; PHI_SCALARS] =
        std::array::from_fn(|_| Vec::with_capacity(BOOTSTRAP_DRAWS));
    for _ in 0..BOOTSTRAP_DRAWS {
        let mut draw = DecodeSums::default();
        for _ in 0..totals.len() {
            draw.absorb(totals.choose(&mut rng).expect("totals is non-empty"));
        }
        let values = draw.census(directional_gain);
        for (column, value) in columns.iter_mut().zip(values) {
            if value.is_finite() {
                column.push(value);
            }
        }
    }
    let tail = (1.0 - CI_MASS) / 2.0;
    for (scalar, column) in census.scalars.iter_mut().zip(columns.iter_mut()) {
        if column.len() < 2 {
            continue;
        }
        column.sort_by(f64::total_cmp);
        scalar.se = standard_deviation(column);
        scalar.ci = (
            sorted_percentile(column, tail),
            sorted_percentile(column, 1.0 - tail),
        );
    }
    census
}

/// What the two CATCH-ALL bins of `r` are worth to the calibration, measured two ways.
///
/// # Why two arms and not one
///
/// The catch-alls hold real mass and the law is not wrong to put it there; what is wrong is
/// pricing it at the clipped BOUND. So there are two different questions:
///
/// * ZEROED and renormalized answers "is the decode SUFFICIENT to explain the miscalibration".
///   Zeroing cannot fail to detect an artifact that is present, so it is robust for that
///   question and an UPPER BOUND on the correction - it discards the legitimate tail
///   contribution along with the mispricing, so its slopes overshoot perfect calibration. A
///   slope that fails to move even here cannot be rescued by any decode.
/// * RE-DECODED at [`OUTER_REDECODE`] answers "what will the pipeline read after the fix". Same
///   mass, valued at its fitted conditional mean, so it is the point estimate and the
///   comparator the fix is checked against.
///
/// The two are one result, not a discrepancy: on the measured level a `0.24` variance slope maps
/// to about `1.28` re-decoded and about `2.58` zeroed, and a mean slope past one in the zeroed
/// arm is the diagnostic over-correcting BY CONSTRUCTION rather than the head being
/// under-dispersed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OuterDecomposition {
    /// Mean total catch-all mass per bar. The equal-mass construction puts the MARGINAL law's
    /// value at `2/128 = 1.5625%`, so a head at that level has learned nothing about its tails.
    pub mass: f64,
    /// Mean SIGNED net, upper minus lower. Only this moves `mu_hat`: at the current decode each
    /// unit of net mass carries `1763.7` bps of conditional mean against a `mu` near 1 bp, so
    /// `0.06%` of net mass doubles a typical forecast.
    pub signed: f64,
    pub redecoded: DecodeArm,
    pub zeroed: DecodeArm,
    /// The RE-DECODED arm split by the block's own realized volatility, with each cell's mean
    /// catch-all mass beside its slopes - the pair that separates "one shared convention error"
    /// from "a head that learned different tails for different names".
    pub gradient: VolatilityGradient,
    /// What SHARE of the forecast mean's variation those two bins carry, and what it predicts for
    /// the slope. The sharpest falsifiable form of the decode hypothesis: see [`PhiCensus`].
    pub phi: PhiCensus,
}

impl OuterDecomposition {
    fn measure(
        bars: &[OuterBar],
        realized: &[f64],
        blocks: &[u64],
        edge_mean: &[f64],
    ) -> Self {
        let count = bars.len().max(1) as f64;
        let redecoded: Vec<(f64, f64)> =
            bars.iter().map(|bar| bar.redecoded(OUTER_REDECODE)).collect();
        let zeroed: Vec<(f64, f64)> = bars
            .iter()
            .map(|bar| (bar.interior_mean, bar.interior_var))
            .collect();
        let outer: Vec<f64> = bars.iter().map(|bar| bar.mass).collect();
        let mut mu = Vec::with_capacity(bars.len());
        let mut variance = Vec::with_capacity(bars.len());
        let mut residual_squares = Vec::with_capacity(bars.len());
        for ((m, v), r) in redecoded.iter().zip(realized) {
            mu.push(*m);
            variance.push(*v);
            residual_squares.push((r - m) * (r - m));
        }
        Self {
            mass: bars.iter().map(|bar| bar.mass).sum::<f64>() / count,
            signed: bars.iter().map(|bar| bar.signed).sum::<f64>() / count,
            redecoded: DecodeArm::measure(&redecoded, realized, blocks),
            zeroed: DecodeArm::measure(&zeroed, realized, blocks),
            gradient: volatility_gradient(
                &mu,
                realized,
                &variance,
                &residual_squares,
                Some(&outer),
                blocks,
            ),
            phi: measure_phi(bars, edge_mean, realized, blocks),
        }
    }

    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "  catch-all mass: {:.4}% of the law per bar (marginal construction {:.4}%, so the \
             head has trimmed {:.1}% of the equal-mass tail), signed net {:+.4}%",
            100.0 * self.mass,
            100.0 * 2.0 / NUM_BAR_BINS as f64,
            100.0 * (1.0 - self.mass * NUM_BAR_BINS as f64 / 2.0),
            100.0 * self.signed,
        )];
        for (label, arm) in [("ZEROED    ", &self.zeroed), ("RE-DECODED", &self.redecoded)] {
            lines.push(format!(
                "  {label} catch-alls: mean slope {:+.4} (se {:.4}), var slope {:+.4} (se \
                 {:.4}), predicted sd {:7.2} bps/bar",
                arm.mean.beta,
                arm.mean.beta_se,
                arm.variance.beta,
                arm.variance.beta_se,
                arm.predicted_sd * 1e4,
            ));
        }
        lines.push(
            "  ZEROED is an UPPER BOUND on the correction - it discards the legitimate tail too, \
             so a mean slope past 1.0 there is the diagnostic over-correcting, not the head being \
             under-dispersed. RE-DECODED is the MEAN's point estimate; its VARIANCE column is \
             one-sided, because a point mass per catch-all drops the within-bin dispersion, so \
             its predicted sd is a lower bound and its var slope an upper bound"
                .to_owned(),
        );
        lines.extend(self.gradient.report_lines("re-decoded"));
        lines.extend(self.phi.report_lines());
        lines
    }
}

/// Windows usable as a CALIBRATION FIT slice, disjoint from the traded prefix in both
/// windows and BLOCKS.
///
/// # Why this split and not another
///
/// A slope fitted on the same bars it is then evaluated on manufactures its own improvement:
/// OLS chooses `beta` to minimize squared error on exactly those bars, so the recalibrated
/// policy would be reading its own answer key. The fit therefore has to come from data the
/// evaluation never sees, and there are three candidates:
///
/// * The TRAIN split. Rejected: the model was fitted on those bars, so its predictions there
///   are in-sample and their calibration is a statement about memorization rather than about
///   the held-out inflation being measured. If the inflation is an overfitting artifact — and
///   the reuse-driven decay it was found in says it may be — a train-fitted slope would come
///   back near one and the correction would be silently switched off.
/// * A calendar half of the pinned blocks. Rejected as the primary split: the pinned
///   evaluation set is drawn under one seed and its blocks are `(symbol, month)`, so a
///   calendar half is also a SYMBOL half of unequal size, and the traded prefix is not
///   calendar-ordered to begin with.
/// * Pinned windows the bench does not trade. Chosen. The pinned set holds far more windows
///   than the bench's [`TRADE_WINDOWS`] budget, all of them held out, all drawn from the same
///   distribution under the same seed, and the surplus is otherwise unused. Dropping every
///   window whose `(symbol, calendar month)` block also appears inside the traded prefix
///   makes the two slices disjoint at the level the interval is taken at, not merely at the
///   level of individual windows: no shared symbol-month regime, so no shared realization of
///   the volatility or drift a slope could be reading.
///
/// Returns at most `limit` window indices, in pinned order, all `>= traded`.
pub fn disjoint_fit_windows(blocks: &[u64], traded: usize, limit: usize) -> Vec<usize> {
    let traded = traded.min(blocks.len());
    let spoken_for: std::collections::BTreeSet<u64> = blocks[..traded].iter().copied().collect();
    blocks
        .iter()
        .enumerate()
        .skip(traded)
        .filter(|(_, block)| !spoken_for.contains(block))
        .map(|(index, _)| index)
        .take(limit)
        .collect()
}

/// True when two block assignments share no resampling unit at all.
///
/// The property a fitted-then-evaluated recalibration must satisfy, stated as a function so
/// the pass that fits and the test that polices it check the same thing.
pub fn blocks_disjoint(fit: &[u64], eval: &[u64]) -> bool {
    let fit: std::collections::BTreeSet<u64> = fit.iter().copied().collect();
    !eval.iter().any(|block| fit.contains(block))
}

// ---------------------------------------------------------------------------
// The recalibrated policy, reported beside the unrecalibrated one
// ---------------------------------------------------------------------------

/// One cap of the recalibrated-versus-unrecalibrated comparison.
///
/// Both sides are the SAME windows, the same bars, the same null and the same accounting;
/// they differ only in whether the mean the Kelly solve was handed came from the head or
/// from the head projected through a slope fitted on OTHER windows.
#[derive(Clone, Copy, Debug)]
pub struct ShrunkPoint {
    pub cap: f64,
    /// The model policy at this cap, unchanged — the number every other measurement in the
    /// session is quoted against.
    pub unshrunk: CapPoint,
    /// The same policy sized on the recalibrated mean.
    pub shrunk: CapPoint,
    /// `shrunk - unshrunk` net growth per bar, PAIRED window by window and intervalled over
    /// the same blocks.
    ///
    /// The levels cannot answer whether recalibration helped. Each side's own interval is
    /// roughly `+/-1.5` bps wide on 256 windows, because almost all of that width is the
    /// market-common regime the two policies SHARE — they trade the same bars of the same
    /// months, and a month that went badly went badly for both. Differencing window by window
    /// removes the shared term entirely, which is the same argument
    /// [`pretrain_stats::compare_runs`] makes for paired run comparison, and leaves an
    /// interval an order of magnitude tighter on the only quantity in dispute.
    pub paired: Dispersion,
}

impl ShrunkPoint {
    pub fn nan() -> Self {
        Self {
            cap: f64::NAN,
            unshrunk: CapPoint::nan(),
            shrunk: CapPoint::nan(),
            paired: Dispersion::nan(),
        }
    }

    /// Recovered edge, in the units the edge is reported in.
    ///
    /// Identical to [`Self::paired`]'s mean up to floating-point summation order: both sides
    /// face the same null, so the null cancels out of the difference.
    pub fn edge_gain(&self) -> f64 {
        self.shrunk.edge - self.unshrunk.edge
    }

    pub fn sharpe_gain(&self) -> f64 {
        self.shrunk.sharpe - self.unshrunk.sharpe
    }

    /// True when the paired interval excludes zero, i.e. the recalibration's effect at this
    /// cap is resolvable rather than a difference of two noisy levels.
    pub fn resolvable(&self) -> bool {
        self.paired.ci_low.is_finite()
            && self.paired.ci_high.is_finite()
            && (self.paired.ci_low > 0.0 || self.paired.ci_high < 0.0)
    }
}

/// The recalibrated policy's whole verdict, beside the unrecalibrated one at every cap.
///
/// This is reported as an ADDITIONAL policy and never substituted for [`POLICY_MODEL`]:
/// every other number measured this session is quoted against the unrecalibrated model, and
/// silently improving it would invalidate the comparison it exists to serve.
#[derive(Clone, Debug)]
pub struct ShrunkBench {
    /// The recalibration applied, fitted on a slice DISJOINT from these windows.
    pub shrink: MeanShrink,
    pub policy: PolicyStats,
    /// Paired edge against the same unconditional null, on the same windows.
    pub edge: Dispersion,
    pub break_even_bps: f64,
    pub ceiling_capture: f64,
    pub curve: [ShrunkPoint; CAP_GRID.len()],
    pub free_kelly: FreeKelly,
    pub cost_bps: f64,
    pub leverage_cap: f64,
    pub bars: usize,
    pub windows: usize,
    pub blocks: usize,
    /// Per-window traded notional of the RECALIBRATED book, in the traded windows' own order.
    ///
    /// The shrink is a per-bar log-space shift of the support rather than a scalar on `f`, so a
    /// turnover-weighted cost is genuinely not invariant to it and cannot be recovered by
    /// rescaling the unshrunk book's weights.
    pub turnover: Vec<WindowTurnover>,
}

impl ShrunkBench {
    pub fn measured(&self) -> bool {
        self.bars > 0
    }

    /// Console lines, quoted against the unrecalibrated bench they were measured beside.
    pub fn report_lines(&self, unshrunk: &TradeBench) -> Vec<String> {
        if !self.measured() {
            return vec!["shrunk policy: not measured".to_owned()];
        }
        let mut lines = Vec::with_capacity(CAP_GRID.len() + 4);
        lines.push(format!(
            "shrunk policy (mu -> {:+.5e} + {:.4} mu, fitted OUT OF SAMPLE; {} windows / {} \
             bars / {} blocks, cap {:.1}x, cost {:.2} bps)",
            self.shrink.alpha,
            self.shrink.beta,
            self.windows,
            self.bars,
            self.blocks,
            self.leverage_cap,
            self.cost_bps,
        ));
        let model = &unshrunk.policies[POLICY_MODEL];
        lines.push(format!(
            "  shrunk    growth {:+.4} bps/bar net, sharpe {:+.2}, hit {:.3}, |f| {:.2} \
             ({:.0}% capped), turnover {:.3}/bar, dd max {:.4}, edge {:+.4} (95% CI \
             {:+.4}..{:+.4}), break-even {}",
            self.policy.net_growth * 1e4,
            self.policy.sharpe,
            self.policy.hit_rate,
            self.policy.mean_abs_position,
            100.0 * self.policy.clamped_fraction,
            self.policy.turnover,
            self.policy.max_drawdown,
            self.edge.mean * 1e4,
            self.edge.ci_low * 1e4,
            self.edge.ci_high * 1e4,
            TradeBench::break_even_text(self.break_even_bps),
        ));
        lines.push(format!(
            "  unshrunk  growth {:+.4} bps/bar net, sharpe {:+.2}, hit {:.3}, |f| {:.2} \
             ({:.0}% capped), turnover {:.3}/bar, dd max {:.4}, edge {:+.4} (95% CI \
             {:+.4}..{:+.4}), break-even {}",
            model.net_growth * 1e4,
            model.sharpe,
            model.hit_rate,
            model.mean_abs_position,
            100.0 * model.clamped_fraction,
            model.turnover,
            model.max_drawdown,
            unshrunk.model_edge().mean * 1e4,
            unshrunk.model_edge().ci_low * 1e4,
            unshrunk.model_edge().ci_high * 1e4,
            TradeBench::break_even_text(unshrunk.model_break_even()),
        ));
        for point in &self.curve {
            lines.push(format!(
                "  cap {:>5.2}x  edge {:+.4} -> {:+.4} bps/bar, PAIRED {:+.4} (95% CI \
                 {:+.4}..{:+.4}, se {:.4}){}, sharpe {:+.2} -> {:+.2} ({:+.2}), |f| {:.2} -> \
                 {:.2}, capped {:.0}% -> {:.0}%, turnover {:.3} -> {:.3}, be {:.2} -> {:.2}, \
                 dd {:.1}% -> {:.1}%",
                point.cap,
                point.unshrunk.edge * 1e4,
                point.shrunk.edge * 1e4,
                point.paired.mean * 1e4,
                point.paired.ci_low * 1e4,
                point.paired.ci_high * 1e4,
                point.paired.se * 1e4,
                if point.resolvable() { "" } else { " NOT RESOLVABLE" },
                point.unshrunk.sharpe,
                point.shrunk.sharpe,
                point.sharpe_gain(),
                point.unshrunk.mean_abs_position,
                point.shrunk.mean_abs_position,
                100.0 * point.unshrunk.clamped_fraction,
                100.0 * point.shrunk.clamped_fraction,
                point.unshrunk.turnover,
                point.shrunk.turnover,
                point.unshrunk.break_even_bps,
                point.shrunk.break_even_bps,
                100.0 * point.unshrunk.max_drawdown,
                100.0 * point.shrunk.max_drawdown,
            ));
        }
        lines.push(format!(
            "  uncapped |f*| under the recalibrated mean: median {:.2}x, p95 {:.2}x, mean \
             signed {:+.2}x, {:.1}% at the {:.1}x cap (unrecalibrated: {:.2}x / {:.2}x / \
             {:+.2}x / {:.1}%)",
            self.free_kelly.median,
            self.free_kelly.p95,
            self.free_kelly.mean_signed,
            100.0 * self.free_kelly.saturated,
            self.leverage_cap,
            unshrunk.free_kelly.median,
            unshrunk.free_kelly.p95,
            unshrunk.free_kelly.mean_signed,
            100.0 * unshrunk.free_kelly.saturated,
        ));
        lines
    }
}

/// The recalibrated fraction promoted to THE free optimum of a parallel window set, which is
/// what lets [`recap`], [`Ledger`] and [`cap_point`] produce its whole verdict unchanged.
///
/// `None` when no window carries a recalibrated fraction. One construction, shared by the
/// bench and by the turnover emission, so a cost re-weighted by the shrunk book's turnover is
/// weighted by the same positions the shrunk bench scored.
fn promote_shrunk(windows: &[WindowPaths]) -> Option<Vec<WindowPaths>> {
    if windows.is_empty() || windows.iter().any(|window| window.free_shrunk.is_none()) {
        return None;
    }
    Some(
        windows
            .iter()
            .map(|window| WindowPaths {
                realized: window.realized.clone(),
                free: window
                    .free_shrunk
                    .clone()
                    .expect("checked above that every window carries one"),
                // Left empty deliberately: `recap` derives every policy's path from `free` and
                // the realized returns, so cloning the unrecalibrated paths here would
                // allocate six vectors per window only to overwrite them.
                positions: std::array::from_fn(|_| Vec::new()),
                predicted_mean: window.predicted_mean.clone(),
                predicted_var: window.predicted_var.clone(),
                free_shrunk: None,
                outer_mass: window.outer_mass.clone(),
                outer_signed: window.outer_signed.clone(),
                trimmed_mean: window.trimmed_mean.clone(),
                trimmed_var: window.trimmed_var.clone(),
            })
            .collect(),
    )
}


/// Score the recalibrated policy on windows that already carry a recalibrated fraction.
///
/// `None` when no window does, which is every ordinary bench: the recalibration has to be
/// fitted before the pass that evaluates it, on a slice disjoint from these windows, so its
/// absence is the normal case rather than an error.
///
/// The accounting is the SAME accounting: the recalibrated fractions are substituted into
/// [`WindowPaths::free`] and every policy, cap point and interval is then produced by the
/// identical code path the unrecalibrated bench uses. Nothing here is a second
/// implementation of the ledger, the cap curve or the bootstrap.
pub fn shrunk_bench(
    windows: &[WindowPaths],
    blocks: &[u64],
    config: BenchConfig,
    shrink: MeanShrink,
) -> Option<ShrunkBench> {
    if windows.is_empty() || windows.iter().all(|window| window.free_shrunk.is_none()) {
        return None;
    }
    assert!(
        windows.iter().all(|window| window
            .free_shrunk
            .as_ref()
            .is_some_and(|free| free.len() == window.bars())),
        "a recalibrated fraction is solved for every bar of every window or for none"
    );
    assert!(
        blocks.len() >= windows.len(),
        "every traded window needs a bootstrap block assignment: {} blocks for {} windows",
        blocks.len(),
        windows.len()
    );
    let BenchConfig {
        cost_bps,
        cap,
        free_marginal,
    } = config;
    let blocks = &blocks[..windows.len()];
    let cost = cost_bps * 1e-4;

    let promoted = recap(
        &promote_shrunk(windows).expect("checked above that every window carries one"),
        cap,
        free_marginal,
    );

    let model = Ledger::build(&promoted, POLICY_MODEL, cap);
    let null = Ledger::build(&promoted, POLICY_MARGINAL, cap);
    let oracle = Ledger::build(&promoted, POLICY_ORACLE, cap);
    let policy = model.stats(cost);
    let null_stats = null.stats(cost);
    let deltas: Vec<f64> = model
        .window_growth(cost)
        .iter()
        .zip(&null.window_growth(cost))
        .map(|(policy, null)| policy - null)
        .collect();
    let edge = block_bootstrap(&deltas, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
    let edge_at = |bps: f64| {
        let cost = bps * 1e-4;
        model.net_growth_per_bar(cost) - null.net_growth_per_bar(cost)
    };
    let ceiling = oracle.stats(cost).net_growth - null_stats.net_growth;

    let mut curve = [ShrunkPoint::nan(); CAP_GRID.len()];
    for (slot, point) in CAP_GRID.iter().enumerate() {
        // Recapped once per side, so the paired vector and the two `CapPoint`s are the same
        // clamp of the same fractions rather than two independent re-derivations.
        let plain = recap(windows, *point, free_marginal);
        let recalibrated = recap(&promoted, *point, free_marginal);
        let before = window_growth_at(&plain, POLICY_MODEL, *point, cost_bps);
        let after = window_growth_at(&recalibrated, POLICY_MODEL, *point, cost_bps);
        let gains: Vec<f64> = after
            .iter()
            .zip(&before)
            .map(|(after, before)| after - before)
            .collect();
        curve[slot] = ShrunkPoint {
            cap: *point,
            unshrunk: cap_point(windows, *point, free_marginal, cost),
            shrunk: cap_point(&promoted, *point, free_marginal, cost),
            paired: block_bootstrap(&gains, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        };
    }

    Some(ShrunkBench {
        turnover: model.turnover.clone(),
        shrink,
        policy,
        edge,
        break_even_bps: break_even_bps(&edge_at),
        ceiling_capture: if ceiling > 0.0 {
            (policy.net_growth - null_stats.net_growth) / ceiling
        } else {
            f64::NAN
        },
        curve,
        free_kelly: free_kelly(&promoted, cap),
        cost_bps,
        leverage_cap: cap,
        bars: model.bars(),
        windows: windows.len(),
        blocks: edge.blocks,
    })
}

// ---------------------------------------------------------------------------
// Cost-aware sizing: the shapes the cost-blind solve does not have
// ---------------------------------------------------------------------------

/// Knob values per sizing shape, so all three share one reported axis.
///
/// # Why this axis has to exist
///
/// [`kelly_fractions`] maximizes `E[ln(1 + f R)]`, which contains no cost term, and the
/// charge is then levied afterwards on the turnover that cost-blind solve happened to
/// generate. Under PROPORTIONAL costs that is not merely conservative, it is the wrong
/// control problem: the optimal policy has an INACTION REGION and rebalancing to the
/// frictionless optimum every bar is strictly dominated. The module header calls the gap
/// "a lower bound on what the same predictive law is worth" — this is the axis that
/// measures how loose that bound is instead of asserting it is small.
///
/// # What this axis does NOT bound
///
/// Every shape here POST-PROCESSES the cost-blind `f*`. The genuinely optimal policy under
/// proportional costs solves a different control problem whose inaction region is not
/// centred on `f*` at all, so a gain measured here is a LOWER bound on the loss from
/// cost-blindness and never the whole of it. A null result on this axis does not license
/// "cost-blind sizing costs nothing".
pub const SIZING_KNOBS: usize = 10;

/// Slot of the INCUMBENT — the cost-blind every-bar re-solve — in every shape's knob grid.
/// Every gain is paired against it.
pub const INCUMBENT_SLOT: usize = 0;

/// Slot whose knob freezes the book flat, the degenerate anchor at the far end of every
/// shape. Having it on every grid is what makes the three curves comparable end to end.
pub const FROZEN_SLOT: usize = SIZING_KNOBS - 1;

/// No-trade band widths, in MULTIPLES OF THE LEVERAGE CAP.
///
/// Stated in multiples of the cap rather than in absolute leverage so the same knob means
/// the same thing at every point of [`CAP_GRID`]: positions live in `[-cap, cap]`, so a
/// fraction of `2.0` cannot be breached by any move and freezes the book.
pub const BAND_FRACTIONS: [f64; SIZING_KNOBS] = [
    0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 2.00,
];
const _: () = assert!(
    BAND_FRACTIONS[INCUMBENT_SLOT] == 0.0,
    "a band of zero is the every-bar re-solve every gain is paired against"
);
const _: () = assert!(
    BAND_FRACTIONS[FROZEN_SLOT] >= 2.0,
    "positions span at most 2 cap, so only a band of 2 cap or more can freeze the book"
);

/// Partial-adjustment weights, DESCENDING so slot [`INCUMBENT_SLOT`] is the incumbent.
///
/// `lambda = 1` moves all the way to `f*` every bar and is therefore the identical policy a
/// band of zero is; `lambda = 0` never leaves flat. The grid runs between them so partial
/// adjustment shares the band's axis: same index, same incumbent, same frozen anchor, one
/// chart.
pub const PARTIAL_LAMBDAS: [f64; SIZING_KNOBS] =
    [1.0, 0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.12, 0.06, 0.0];
const _: () = assert!(
    PARTIAL_LAMBDAS[INCUMBENT_SLOT] == 1.0,
    "full adjustment is the every-bar re-solve every gain is paired against"
);
const _: () = assert!(
    PARTIAL_LAMBDAS[FROZEN_SLOT] == 0.0,
    "the sweep needs a frozen book at the far end of every shape"
);

/// How a cost-aware policy departs from rebalancing to `f*` every bar.
///
/// Three shapes and not one, because they make different claims and the differences are
/// measurable on this panel rather than decidable from theory alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizingShape {
    /// No-trade band, rebalancing ALL THE WAY to `f*` once it is breached.
    ///
    /// Carried as the form theory calls DOMINATED, and MEASURED TO WIN on this panel. It is
    /// what a reader assumes when told "no-trade band" and what [`super::portfolio`]'s engine
    /// implements, so having it beside [`Self::BandReflect`] was supposed to turn "reflection
    /// dominates" from a citation into a measurement. It did, and the measurement came back
    /// the other way: see [`Self::BandReflect`].
    BandToTarget,
    /// No-trade band with REFLECTION at the boundary: on breach, trade only as far as the
    /// nearest edge of the inaction region and never through it.
    ///
    /// The impulse-control form. The last increment of a move toward `f*` earns a
    /// first-order-small growth gain for a first-order-large cost, so crossing the region to
    /// its far side pays for distance that then has to be paid for again coming back.
    ///
    /// # It loses here, resolvably, and the reason is the cap
    ///
    /// On `pretrain_step_9728` at the 4x cap, reflection is worse than [`Self::BandToTarget`]
    /// at every knob and resolvably worse than the unbanded incumbent from a band of `0.100`
    /// cap onward. The mechanism is visible in the `mean |f|` column: jump-to-target holds
    /// mean exposure at `3.869` across the whole grid, while reflection walks it down to
    /// `3.492` at `0.100` and `1.964` at `0.500`.
    ///
    /// That is not churn being removed, it is LEVERAGE being removed, and it happens because
    /// the impulse-control result assumes the frictionless target is interior. Here
    /// [`LEVERAGE_CAP`] binds on 74-93% of bars, and on those bars `f*` sits ON the boundary, so
    /// the "nearest edge of the inaction region" lies on the LOW-exposure side of it and cannot
    /// lie on the high side, because there is no high side. Reflection is symmetric where the
    /// target is interior; it is one-sided exactly where the cap binds, which is nearly
    /// everywhere. Reflection therefore imposes a near-permanent exposure
    /// haircut of order the band width on a book with positive expected growth, and the growth
    /// forgone swamps the cost saved. Jump-to-target has no such bias: it either holds, or it
    /// lands exactly on `f*`.
    ///
    /// Stated here rather than only in the report because the theoretical claim is the kind a
    /// reader will re-derive from first principles and re-apply, and the condition it needs -
    /// an interior optimum - is exactly the condition this bench does not satisfy.
    BandReflect,
    /// Partial adjustment, `f_t = f_prev + lambda (f* - f_prev)`. No dead zone at all.
    ///
    /// A band is a low-pass filter WITH a dead zone, and a dead zone destroys exactly the
    /// small continuous magnitude modulation a volatility timer would live on. This shape
    /// cuts turnover while preserving that modulation, attenuated. It is also the
    /// recalibration shrink applied to the POSITION rather than to the mean, which is why
    /// [`band_shrink_overlap`] runs it: if the two are near substitutes, both switched on
    /// should buy almost nothing over either alone.
    PartialAdjust,
}

/// Every shape, in report order: the two bands, then the filter. Their ranking is a
/// measurement and it is on [`SizingShape`], not encoded in this order.
pub const SIZING_SHAPES: [SizingShape; 3] = [
    SizingShape::BandToTarget,
    SizingShape::BandReflect,
    SizingShape::PartialAdjust,
];

impl SizingShape {
    pub fn name(self) -> &'static str {
        match self {
            Self::BandToTarget => "band, to target",
            Self::BandReflect => "band, reflecting",
            Self::PartialAdjust => "partial adjustment",
        }
    }

    /// What this shape's knob MEANS, so no reader has to infer it from the numbers.
    pub fn knob_name(self) -> &'static str {
        match self {
            Self::BandToTarget | Self::BandReflect => "band / cap",
            Self::PartialAdjust => "lambda",
        }
    }

    pub fn knobs(self) -> [f64; SIZING_KNOBS] {
        match self {
            Self::BandToTarget | Self::BandReflect => BAND_FRACTIONS,
            Self::PartialAdjust => PARTIAL_LAMBDAS,
        }
    }

    /// True for the shapes that carry a dead zone, which is the property the vol-timer
    /// question turns on.
    pub fn has_dead_zone(self) -> bool {
        matches!(self, Self::BandToTarget | Self::BandReflect)
    }

    /// This shape's position path, given the already-clamped frictionless target.
    ///
    /// `target` is clamped, so every shape returns a path inside `[-cap, cap]` and nothing
    /// needs re-clamping: the band shapes return either a point of `target` or a point
    /// between the previous holding and `target`, and partial adjustment returns a convex
    /// combination of the two.
    ///
    /// The book enters FLAT, the convention [`Ledger::build`] charges the entry trade under,
    /// so every shape at every knob is paid for on the same footing as the incumbent.
    pub fn positions(self, target: &[f64], knob: f64, cap: f64) -> Vec<f64> {
        match self {
            Self::BandToTarget | Self::BandReflect => {
                let band = knob * cap;
                assert!(
                    band >= 0.0 && band.is_finite(),
                    "a no-trade band is a non-negative leverage width, got {band}"
                );
                let reflect = self == Self::BandReflect;
                let mut held = 0.0f64;
                target
                    .iter()
                    .map(|want| {
                        let wanted = want - held;
                        if wanted.abs() > band {
                            held = if reflect {
                                // Never past the target: `|wanted| > band`, so stepping back
                                // by `band` along the direction of travel lands strictly
                                // between `held` and `want`.
                                want - band * wanted.signum()
                            } else {
                                *want
                            };
                        }
                        held
                    })
                    .collect()
            }
            Self::PartialAdjust => {
                assert!(
                    (0.0..=1.0).contains(&knob),
                    "a partial-adjustment weight is a fraction of the way to the optimum, \
                     got {knob}"
                );
                let mut held = 0.0f64;
                target
                    .iter()
                    .map(|want| {
                        held += knob * (want - held);
                        held
                    })
                    .collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The cost INSIDE the objective: the myopic cost-aware solve
// ---------------------------------------------------------------------------

/// One bar's log-optimal fraction when the TRADING COST IS PART OF THE OBJECTIVE.
///
/// # The objective, and why it is a different policy rather than a better fill rule
///
/// [`kelly_fractions`] maximizes `g(f) = sum_b p_b ln(1 + f R_b)`, which contains no cost
/// term. Every shape in [`SizingShape`] then POST-PROCESSES that cost-blind `f*`, so however
/// clever the fill rule is, the point it is aiming at was chosen as if trading were free.
/// This maximizes instead
///
/// ```text
/// G(f) = sum_b p_b ln(1 + f R_b) + ln(1 - c |f - f_prev|)
/// ```
///
/// where `c` is the per-unit-notional cost [`Ledger::cost_of`] charges and `f_prev` is the
/// position already held. The second term is the same charge the ledger levies, moved from
/// after the decision to inside it. The consequence is the one no post-processing can
/// reproduce: an inaction region EMERGES from the kink at `f_prev`, and both its width and
/// its centre depend on `c`.
///
/// # Why the solve is still a bisection
///
/// `G` is concave: `g` is strictly concave, `|f - f_prev|` is convex so `1 - c|f - f_prev|`
/// is concave, it is positive on the admissible set, and `ln` of a positive concave function
/// is concave. The only new feature is a KINK at `f_prev`, where `G` is not differentiable,
/// so the smooth bisection is run on each side and the kink is tested directly:
///
/// * `f_prev` is optimal exactly when `0` lies in the subgradient interval there, i.e. when
///   `|g'(f_prev)| <= c`. That inequality IS the no-trade region, and it is stated by the
///   objective rather than chosen by a sweep.
/// * `g'(f_prev) > c` puts the optimum strictly above `f_prev`: bisect `g'(f) - c/(1 - c(f -
///   f_prev))` on `[f_prev, hi]`.
/// * `g'(f_prev) < -c` mirrors it below.
///
/// # Still MYOPIC
///
/// The objective charges this bar's trade and values this bar's growth. It does not know it
/// will trade again next bar, so it is not the Davis-Norman control problem and its inaction
/// region is narrower than the fully forward-looking one. It is nonetheless a strictly
/// stronger statement than any post-processed band: the TARGET moves, not just the fill.
///
/// # The internal check this is built to satisfy
///
/// As `c -> 0` the kink flattens and the solution must converge to [`kelly_fractions`]'s. At
/// `c == 0` it is required to agree EXACTLY, which is what
/// `the_myopic_solve_is_the_cost_blind_solve_at_zero_cost` pins, and is the check that
/// catches a sign error in the cost slope.
pub fn myopic_fractions(
    probs: &Tensor,
    returns: &Tensor,
    cap: f64,
    cost: f64,
    previous: &Tensor,
) -> Tensor {
    assert!(
        cap > 0.0 && cap.is_finite(),
        "the leverage cap must be positive and finite"
    );
    assert!(
        cost >= 0.0 && cost.is_finite(),
        "a per-unit trading cost is non-negative and finite, got {cost}"
    );
    if cost == 0.0 {
        // No kink, no cost slope, nothing to add: the objective IS the cost-blind one, so
        // this returns the identical tensor rather than a numerically-close reconstruction.
        return kelly_fractions(probs, returns, cap);
    }
    let probs = probs.to_kind(Kind::Double);
    let size = probs.size();
    assert_eq!(size.len(), 2, "probs must be [rows, outcomes]");
    let (rows, outcomes) = (size[0], size[1]);
    let returns = returns.to_kind(Kind::Double).reshape([-1, outcomes]);
    let return_rows = returns.size()[0];
    assert!(
        return_rows == rows || return_rows == 1,
        "returns must be shared ([1, outcomes]) or per-row ([rows, outcomes]), got \
         [{return_rows}, {outcomes}] against {rows} rows"
    );
    let returns = if return_rows == rows {
        returns
    } else {
        returns.expand([rows, outcomes], false)
    };
    let previous = previous.to_kind(Kind::Double).reshape([rows]);

    tch::no_grad(|| {
        let mass = probs
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Double)
            .clamp_min(f64::MIN_POSITIVE);
        let probs = probs.divide(&mass);
        let live = probs.gt(0.0);
        // Dead bins are removed from the RETURNS rather than merely from the bounds, so a
        // zero-probability bin can never contribute `0 * R / 0` to a slope. See
        // [`kelly_fractions`], which relies on the same guard.
        let returns = returns.masked_fill(&live.logical_not(), 0.0);
        let bound = returns.reciprocal().neg();
        let longs = returns.gt(0.0).logical_and(&live);
        let shorts = returns.lt(0.0).logical_and(&live);
        let lower = bound
            .masked_fill(&longs.logical_not(), f64::NEG_INFINITY)
            .amax([-1i64].as_slice(), false);
        let upper = bound
            .masked_fill(&shorts.logical_not(), f64::INFINITY)
            .amin([-1i64].as_slice(), false);
        let cap = cap.min(MAX_LEVERAGE);
        // The growth term's domain, exactly as the cost-blind solve computes it.
        let growth_lo = (&lower * (1.0 - FEASIBLE_MARGIN)).clamp_min(-cap);
        let growth_hi = (&upper * (1.0 - FEASIBLE_MARGIN)).clamp_max(cap);
        // The COST term's own domain: `1 - c|f - f_prev| > 0`, i.e. `f` within `1/c` of the
        // holding. At the bench's default cost this is thousands of times the cap and cannot
        // bind; at the top of the break-even search it is `10` and can, so it is enforced
        // rather than assumed away.
        let reach = (1.0 - FEASIBLE_MARGIN) / cost;
        let lo = growth_lo.maximum(&(&previous - reach));
        let hi = growth_hi.minimum(&(&previous + reach));
        // `f_prev` is inside both domains whenever it is a feasible position, which it is by
        // construction: it came from a previous solve under the same cap and law.
        let held = previous.clamp_tensor(Some(&lo), Some(&hi));

        // The cost-free slope at the kink decides the branch, and its magnitude against `c`
        // is the no-trade test.
        let slope_at = |f: &Tensor| -> Tensor {
            (&probs * &returns)
                .divide(&(f.unsqueeze(-1) * &returns + 1.0))
                .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double)
        };
        let kink_slope = slope_at(&held);
        let rises = kink_slope.gt(cost);
        let falls = kink_slope.lt(-cost);

        // The UP branch, bisected on `[held, hi]`. The cost slope is `-c/(1 - c(f - held))`,
        // negative throughout, which is what pulls the optimum back toward the holding.
        let mut up_lo = held.shallow_clone();
        let mut up_hi = hi.shallow_clone();
        // The DOWN branch on `[lo, held]`, cost slope `+c/(1 + c(f - held))`.
        let mut down_lo = lo.shallow_clone();
        let mut down_hi = held.shallow_clone();
        // `1 - c|f - held|` is the cost factor and `-c/factor` its slope on the UP side,
        // `+c/factor` on the DOWN side. Written as explicit tensor ops rather than as
        // `scalar - tensor`, which does not name a type here.
        let factor = |f: &Tensor| (f - &held).abs() * -cost + 1.0;
        for _ in 0..SOLVER_ITERATIONS {
            let mid = (&up_lo + &up_hi) * 0.5;
            let objective = slope_at(&mid) - factor(&mid).reciprocal() * cost;
            let rising = objective.gt(0.0);
            up_lo = mid.where_self(&rising, &up_lo);
            up_hi = up_hi.where_self(&rising, &mid);

            let mid = (&down_lo + &down_hi) * 0.5;
            let objective = slope_at(&mid) + factor(&mid).reciprocal() * cost;
            let rising = objective.gt(0.0);
            down_lo = mid.where_self(&rising, &down_lo);
            down_hi = down_hi.where_self(&rising, &mid);
        }
        let up = (up_lo + up_hi) * 0.5;
        let down = (down_lo + down_hi) * 0.5;

        // Branch selection is by the SIGN TEST at the kink rather than by comparing two
        // objective values: concavity makes the test exact, and the inactive branch's
        // bisection has converged onto `held` anyway, so a value comparison would be
        // deciding between a number and itself.
        let solved = held
            .shallow_clone()
            .where_self(&rises.logical_not(), &up)
            .where_self(&falls.logical_not(), &down);

        // The same gate the cost-blind solve applies, on the SAME objective it maximized:
        // holding nothing is only preferable if going flat is itself affordable, so the
        // comparison is against the value of unwinding to zero rather than against `0`.
        let value_at = |f: &Tensor| -> Tensor {
            let growth = (&probs * (f.unsqueeze(-1) * &returns + 1.0).clamp_min(WEALTH_FLOOR).log())
                .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
            growth + factor(f).clamp_min(WEALTH_FLOOR).log()
        };
        let flat = solved.zeros_like();
        let keep = value_at(&solved).ge_tensor(&value_at(&flat));
        solved.where_self(&keep, &flat)
    })
}

/// Which solved fraction the band is applied to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandSource {
    /// [`WindowPaths::free`] — the model's own optimum. The incumbent.
    Frictionless,
    /// [`WindowPaths::free_shrunk`] — the optimum of the recalibrated law.
    ///
    /// Both levers cut turnover, so whether they are substitutes is a measurement rather
    /// than an argument: [`band_shrink_overlap`] takes the interaction paired.
    Recalibrated,
}

impl BandSource {
    pub fn name(self) -> &'static str {
        match self {
            Self::Frictionless => "as-solved",
            Self::Recalibrated => "recalibrated",
        }
    }

    /// The fraction path this source names, or `None` when the pass did not form it.
    fn pick(self, window: &WindowPaths) -> Option<&Vec<f64>> {
        match self {
            Self::Frictionless => Some(&window.free),
            Self::Recalibrated => window.free_shrunk.as_ref(),
        }
    }
}


/// A window set whose `free` is `replacement`, so `recap`, `Ledger` and `cap_point` produce
/// an alternative sizing's whole verdict through the identical code path.
///
/// The same device [`shrunk_bench`] uses, factored out because the band sweep needs it once
/// per band per shape rule per source. `positions` is left empty deliberately: every caller
/// here runs `recap` immediately, which derives all of them from `free`.
fn rebased(windows: &[WindowPaths], replacement: Vec<Vec<f64>>) -> Vec<WindowPaths> {
    assert_eq!(
        windows.len(),
        replacement.len(),
        "one replacement fraction path per window"
    );
    windows
        .iter()
        .zip(replacement)
        .map(|(window, free)| {
            assert_eq!(
                free.len(),
                window.bars(),
                "a replacement path must cover every bar of its window"
            );
            WindowPaths {
                realized: window.realized.clone(),
                free,
                positions: std::array::from_fn(|_| Vec::new()),
                predicted_mean: window.predicted_mean.clone(),
                predicted_var: window.predicted_var.clone(),
                free_shrunk: None,
                outer_mass: window.outer_mass.clone(),
                outer_signed: window.outer_signed.clone(),
                trimmed_mean: window.trimmed_mean.clone(),
                trimmed_var: window.trimmed_var.clone(),
            }
        })
        .collect()
}

/// One point of the no-trade-band sweep.
#[derive(Clone, Copy, Debug)]
pub struct BandPoint {
    /// This shape's knob value: `band / cap` for the band shapes, `lambda` for partial
    /// adjustment. [`SizingShape::knob_name`] says which, so no row is ambiguous.
    pub knob: f64,
    /// The banded policy at this cap and this cost.
    pub policy: PolicyStats,
    /// The identical positions with the cost switched off.
    ///
    /// `stats(0.0)`, so its `sharpe` is the GROSS Sharpe — a number [`PolicyStats`] cannot
    /// otherwise express, since its own `sharpe` field is always net while `gross_growth`
    /// is always pre-cost. Carrying it makes the ceiling visible on every row.
    pub gross: PolicyStats,
    /// Paired edge against the unconditional null, same windows, same blocks.
    pub edge: Dispersion,
    pub break_even_bps: f64,
    /// PAIRED net growth against the SAME source at band zero, window by window,
    /// intervalled over the same blocks.
    ///
    /// This is the only quantity on the row that answers the question the sweep exists for.
    /// The levels share the market-common regime — identical bars of identical
    /// symbol-months — so a gap between two edge columns is not evidence; the difference is
    /// taken per window and its interval is what excludes zero or fails to. Exactly zero
    /// with a zero-width interval at [`INCUMBENT_SLOT`], which is correct rather than
    /// degenerate.
    pub gain: Dispersion,
    /// This point's traded notional as a share of the SAME source's unbanded book.
    ///
    /// Turnover, not the count of frozen bars: a band freezes the SMALLEST moves first by
    /// construction, so the share of bars it froze badly overstates the saving.
    pub turnover_share: f64,
}

impl BandPoint {
    /// True when the paired gain's interval excludes zero.
    pub fn resolvable(&self) -> bool {
        self.gain.ci_low.is_finite()
            && self.gain.ci_high.is_finite()
            && (self.gain.ci_low > 0.0 || self.gain.ci_high < 0.0)
    }

    /// Share of GROSS log growth the cost consumed at this point.
    ///
    /// `NAN` when there was no gross growth to consume, which is a different statement from
    /// "cost consumed none of it".
    pub fn cost_share_of_gross(&self) -> f64 {
        if self.gross.net_growth > 0.0 {
            (self.gross.net_growth - self.policy.net_growth) / self.gross.net_growth
        } else {
            f64::NAN
        }
    }
}

/// The no-trade band swept on one source under one shape rule, at ONE leverage cap.
///
/// One cap and not the whole [`CAP_GRID`] on purpose: the band is imposed on the CLAMPED
/// path, so re-clamping a banded path at a different cap is not the same object as banding
/// at that cap, and a cap curve built by re-clamping would silently report the first while
/// being read as the second.
#[derive(Clone, Debug)]
pub struct BandSweep {
    pub source: BandSource,
    pub shape: SizingShape,
    /// One entry per [`BAND_FRACTIONS`] slot.
    pub points: Vec<BandPoint>,
    pub cost_bps: f64,
    pub leverage_cap: f64,
    pub bars: usize,
    pub windows: usize,
    pub blocks: usize,
}

impl BandSweep {
    /// Slot with the highest break-even cost among the points that actually traded.
    ///
    /// A frozen book has no turnover and therefore an infinite or absent break-even, which
    /// is not a strategy: `is_finite` excludes it rather than letting the degenerate anchor
    /// win the column it exists to bound.
    pub fn best_break_even(&self) -> Option<usize> {
        self.points
            .iter()
            .enumerate()
            .filter(|(_, point)| point.break_even_bps.is_finite() && point.policy.turnover > 0.0)
            .max_by(|a, b| a.1.break_even_bps.total_cmp(&b.1.break_even_bps))
            .map(|(slot, _)| slot)
    }

    /// Slot with the highest PAIRED gain over the unbanded incumbent.
    ///
    /// Distinct from [`Self::best_break_even`] and routinely a different slot: break-even
    /// is gross edge over turnover and rises as the band suppresses trading, while the
    /// paired gain is measured at ONE cost and falls again once the band starts suppressing
    /// the signal. Reporting only the first would flatter any band.
    pub fn best_gain(&self) -> Option<usize> {
        self.points
            .iter()
            .enumerate()
            .filter(|(_, point)| point.gain.mean.is_finite())
            .max_by(|a, b| a.1.gain.mean.total_cmp(&b.1.gain.mean))
            .map(|(slot, _)| slot)
    }

    pub fn report_lines(&self) -> Vec<String> {
        if self.bars == 0 {
            return vec!["cost-aware sizing: not measured".to_owned()];
        }
        let incumbent = self.shape.knobs()[INCUMBENT_SLOT];
        let mut lines = Vec::with_capacity(self.points.len() + 2);
        lines.push(format!(
            "cost-aware sizing [{}] on the {} fraction ({} windows / {} bars / {} blocks, cap \
             {:.1}x, cost {:.2} bps; knob is {}, {} is the incumbent every-bar re-solve)",
            self.shape.name(),
            self.source.name(),
            self.windows,
            self.bars,
            self.blocks,
            self.leverage_cap,
            self.cost_bps,
            self.shape.knob_name(),
            incumbent,
        ));
        for point in &self.points {
            lines.push(format!(
                "  {} {:>5.3}  edge {:+.4} bps/bar, PAIRED vs incumbent {:+.4} (95% CI \
                 {:+.4}..{:+.4}, se {:.4}){}, be {:>8}, gross {:+.4}, sharpe {:+.2} net / \
                 {:+.2} gross, hit {:.3}, |f| {:.2}, turnover {:.3} ({:.3} of incumbent), cost \
                 eats {:.1}% of gross",
                self.shape.knob_name(),
                point.knob,
                point.edge.mean * 1e4,
                point.gain.mean * 1e4,
                point.gain.ci_low * 1e4,
                point.gain.ci_high * 1e4,
                point.gain.se * 1e4,
                if point.knob == incumbent {
                    " incumbent"
                } else if point.resolvable() {
                    ""
                } else {
                    " NOT RESOLVABLE"
                },
                TradeBench::break_even_text(point.break_even_bps),
                point.gross.net_growth * 1e4,
                point.policy.sharpe,
                point.gross.sharpe,
                point.policy.hit_rate,
                point.policy.mean_abs_position,
                point.policy.turnover,
                point.turnover_share,
                100.0 * point.cost_share_of_gross(),
            ));
        }
        let name = |slot: Option<usize>| match slot {
            Some(slot) => format!("{} {:.3}", self.shape.knob_name(), self.points[slot].knob),
            None => "none".to_owned(),
        };
        lines.push(format!(
            "  [{}] knob maximizing break-even {} (be {:>8}), knob maximizing the PAIRED gain \
             {} (gain {:+.4} bps/bar, {})",
            self.shape.name(),
            name(self.best_break_even()),
            self.best_break_even()
                .map_or_else(|| "n/a".to_owned(), |slot| TradeBench::break_even_text(
                    self.points[slot].break_even_bps
                )),
            name(self.best_gain()),
            self.best_gain()
                .map_or(f64::NAN, |slot| self.points[slot].gain.mean * 1e4),
            match self.best_gain() {
                Some(slot) if self.points[slot].resolvable() => "resolvable",
                Some(_) => "NOT RESOLVABLE",
                None => "unmeasured",
            },
        ));
        lines
    }
}

/// Sweep [`BAND_FRACTIONS`] on one solved fraction under one shape rule.
///
/// `None` when the requested source is absent from these windows, which is every ordinary
/// bench for [`BandSource::Recalibrated`]: the shrink has to be fitted on a block-disjoint
/// slice before the pass that evaluates it, so its absence is the normal case.
///
/// Nothing here is a second implementation of the ledger, the cost charge or the bootstrap.
/// Each band produces a position path, the path becomes the `free` of a parallel window set
/// via [`rebased`], and every figure on the row comes out of the same [`Ledger`],
/// [`break_even_bps`] and [`block_bootstrap`] the headline bench uses.
pub fn band_sweep(
    windows: &[WindowPaths],
    blocks: &[u64],
    config: BenchConfig,
    source: BandSource,
    shape: SizingShape,
) -> Option<BandSweep> {
    if windows.is_empty() || windows.iter().any(|window| source.pick(window).is_none()) {
        return None;
    }
    assert!(
        blocks.len() >= windows.len(),
        "every traded window needs a bootstrap block assignment: {} blocks for {} windows",
        blocks.len(),
        windows.len()
    );
    let BenchConfig {
        cost_bps,
        cap,
        free_marginal,
    } = config;
    let blocks = &blocks[..windows.len()];
    let cost = cost_bps * 1e-4;

    // The frictionless target is the CLAMPED path, because the band is a statement about the
    // position actually held and the cap is what decides that. Re-clamping through `recap`
    // rather than clamping here keeps the one definition of what a policy path is.
    let targets: Vec<Vec<f64>> = recap(&rebased(windows, source_paths(windows, source)), cap, free_marginal)
        .into_iter()
        .map(|window| {
            let mut positions = window.positions;
            std::mem::take(&mut positions[POLICY_MODEL])
        })
        .collect();

    // One null for the whole sweep: its position is a constant, so no band can move it, and
    // building it once is what makes every row's edge a difference against the SAME null.
    let null = Ledger::build(&recap(windows, cap, free_marginal), POLICY_MARGINAL, cap);
    let null_growth = null.window_growth(cost);

    let mut points: Vec<BandPoint> = Vec::with_capacity(BAND_FRACTIONS.len());
    let mut unbanded_turnover = f64::NAN;
    let mut incumbent: Vec<f64> = Vec::new();
    let mut bars = 0usize;
    for knob in shape.knobs() {
        let banded = rebased(
            windows,
            targets
                .iter()
                .map(|target| shape.positions(target, knob, cap))
                .collect(),
        );
        // The band path is already inside `[-cap, cap]`, so this `recap` re-derives the
        // other policies without touching the model's own path.
        let banded = recap(&banded, cap, free_marginal);
        let ledger = Ledger::build(&banded, POLICY_MODEL, cap);
        let policy = ledger.stats(cost);
        let growth = ledger.window_growth(cost);
        if knob == shape.knobs()[INCUMBENT_SLOT] {
            unbanded_turnover = policy.turnover;
            incumbent = growth.clone();
            bars = ledger.bars();
        }
        let deltas: Vec<f64> = growth
            .iter()
            .zip(&null_growth)
            .map(|(banded, null)| banded - null)
            .collect();
        let gains: Vec<f64> = growth
            .iter()
            .zip(&incumbent)
            .map(|(banded, plain)| banded - plain)
            .collect();
        let edge_at = |bps: f64| {
            let cost = bps * 1e-4;
            ledger.net_growth_per_bar(cost) - null.net_growth_per_bar(cost)
        };
        points.push(BandPoint {
            knob,
            policy,
            gross: ledger.stats(0.0),
            edge: block_bootstrap(&deltas, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
            break_even_bps: break_even_bps(&edge_at),
            gain: block_bootstrap(&gains, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
            turnover_share: if unbanded_turnover > 0.0 {
                policy.turnover / unbanded_turnover
            } else {
                f64::NAN
            },
        });
    }

    Some(BandSweep {
        source,
        shape,
        bars,
        blocks: points[INCUMBENT_SLOT].edge.blocks,
        points,
        cost_bps,
        leverage_cap: cap,
        windows: windows.len(),
    })
}

/// The requested source's fraction path per window, cloned.
fn source_paths(windows: &[WindowPaths], source: BandSource) -> Vec<Vec<f64>> {
    windows
        .iter()
        .map(|window| {
            source
                .pick(window)
                .expect("the caller checked every window carries this source")
                .clone()
        })
        .collect()
}

/// Whether the recalibration shrink and the no-trade band are SUBSTITUTES.
///
/// Both levers cut turnover — the shrink by sizing smaller, the band by rebalancing less —
/// so their gains cannot simply be added, and "they probably overlap" is an argument rather
/// than a measurement. The decisive quantity is the INTERACTION, which is a second
/// difference over the same windows and the same blocks:
///
/// ```text
/// interaction = (shrunk banded - shrunk unbanded) - (plain banded - plain unbanded)
/// ```
///
/// Negative means the band buys less once the shrink is already applied, i.e. the two are
/// substitutes; an interval covering zero means the panel cannot tell, which is a finding
/// and not a licence to add the two gains.
#[derive(Clone, Copy, Debug)]
pub struct BandShrinkOverlap {
    pub knob: f64,
    pub shape: SizingShape,
    /// Paired gain of this band over band zero on the UNRECALIBRATED fraction.
    pub gain_plain: Dispersion,
    /// The same on the RECALIBRATED fraction.
    pub gain_shrunk: Dispersion,
    pub interaction: Dispersion,
    /// `1 - gain_shrunk / gain_plain`: the share of the band's unrecalibrated gain that the
    /// shrink alone already captured.
    ///
    /// A ratio of two point estimates and therefore NOT the quantity to test — that is
    /// [`Self::interaction`]. Reported because a reader needs the magnitude in the same
    /// units the two gains are quoted in, and `NAN` when the denominator is not a gain at
    /// all, since a share of a loss is not interpretable.
    pub captured_by_shrink: f64,
}

impl BandShrinkOverlap {
    /// True when the interaction's interval excludes zero, i.e. the overlap is resolvable.
    pub fn resolvable(&self) -> bool {
        self.interaction.ci_low.is_finite()
            && self.interaction.ci_high.is_finite()
            && (self.interaction.ci_low > 0.0 || self.interaction.ci_high < 0.0)
    }

    pub fn report_line(&self) -> String {
        format!(
            "band {:.3}x filling {}: gain {:+.4} bps/bar as-solved (95% CI {:+.4}..{:+.4}) vs \
             {:+.4} recalibrated (95% CI {:+.4}..{:+.4}); INTERACTION {:+.4} (95% CI \
             {:+.4}..{:+.4}, se {:.4}) {}, shrink already captured {:.1}% of the band's gain",
            self.knob,
            self.shape.name(),
            self.gain_plain.mean * 1e4,
            self.gain_plain.ci_low * 1e4,
            self.gain_plain.ci_high * 1e4,
            self.gain_shrunk.mean * 1e4,
            self.gain_shrunk.ci_low * 1e4,
            self.gain_shrunk.ci_high * 1e4,
            self.interaction.mean * 1e4,
            self.interaction.ci_low * 1e4,
            self.interaction.ci_high * 1e4,
            self.interaction.se * 1e4,
            if self.resolvable() {
                "RESOLVABLE"
            } else {
                "not resolvable"
            },
            100.0 * self.captured_by_shrink,
        )
    }
}

/// The band-versus-shrink interaction at every band width, paired window by window.
///
/// `None` when the windows carry no recalibrated fraction, which is every ordinary bench.
pub fn band_shrink_overlap(
    windows: &[WindowPaths],
    blocks: &[u64],
    config: BenchConfig,
    shape: SizingShape,
) -> Option<Vec<BandShrinkOverlap>> {
    let plain = band_growth_paths(windows, config, BandSource::Frictionless, shape)?;
    let shrunk = band_growth_paths(windows, config, BandSource::Recalibrated, shape)?;
    let blocks = &blocks[..windows.len()];
    Some(
        BAND_FRACTIONS
            .iter()
            .enumerate()
            .map(|(slot, fraction)| {
                let gain = |arm: &[Vec<f64>]| -> Vec<f64> {
                    arm[slot]
                        .iter()
                        .zip(&arm[INCUMBENT_SLOT])
                        .map(|(banded, unbanded)| banded - unbanded)
                        .collect()
                };
                let gain_plain = gain(&plain);
                let gain_shrunk = gain(&shrunk);
                // The second difference is formed PER WINDOW before it is intervalled, so
                // the bootstrap resamples the interaction itself rather than differencing
                // two independently resampled means.
                let interaction: Vec<f64> = gain_shrunk
                    .iter()
                    .zip(&gain_plain)
                    .map(|(shrunk, plain)| shrunk - plain)
                    .collect();
                let plain_dispersion =
                    block_bootstrap(&gain_plain, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
                let shrunk_dispersion =
                    block_bootstrap(&gain_shrunk, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
                BandShrinkOverlap {
                    knob: *fraction,
                    shape,
                    captured_by_shrink: if plain_dispersion.mean > 0.0 {
                        1.0 - shrunk_dispersion.mean / plain_dispersion.mean
                    } else {
                        f64::NAN
                    },
                    gain_plain: plain_dispersion,
                    gain_shrunk: shrunk_dispersion,
                    interaction: block_bootstrap(
                        &interaction,
                        blocks,
                        BOOTSTRAP_DRAWS,
                        BOOTSTRAP_SEED,
                    ),
                }
            })
            .collect(),
    )
}

/// Per-window net growth of one source at every band width: `paths[band][window]`.
///
/// The primitive both [`band_sweep`] and [`band_shrink_overlap`] need, so the interaction
/// and the sweep cannot disagree about what a band earns.
fn band_growth_paths(
    windows: &[WindowPaths],
    config: BenchConfig,
    source: BandSource,
    shape: SizingShape,
) -> Option<Vec<Vec<f64>>> {
    if windows.is_empty() || windows.iter().any(|window| source.pick(window).is_none()) {
        return None;
    }
    let BenchConfig {
        cost_bps,
        cap,
        free_marginal,
    } = config;
    let targets: Vec<Vec<f64>> = recap(&rebased(windows, source_paths(windows, source)), cap, free_marginal)
        .into_iter()
        .map(|window| {
            let mut positions = window.positions;
            std::mem::take(&mut positions[POLICY_MODEL])
        })
        .collect();
    Some(
        shape
            .knobs()
            .iter()
            .map(|knob| {
                let banded = rebased(
                    windows,
                    targets
                        .iter()
                        .map(|target| shape.positions(target, *knob, cap))
                        .collect(),
                );
                window_growth_at(
                    &recap(&banded, cap, free_marginal),
                    POLICY_MODEL,
                    cap,
                    cost_bps,
                )
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Where the edge lives: DIRECTION or SIZE
// ---------------------------------------------------------------------------

/// The attribution arms, in index order.
///
/// A hit rate below a coin flip beside an edge whose interval excludes zero is not a
/// contradiction, but it does mean the headline cannot be read as directional skill until the
/// two halves of the decision are separated. Kelly's growth is `mu^2 / (2 sigma^2)` to second
/// order: quadratic in the SIZE of the conditional mean and only linear in getting its sign
/// right, so a forecaster that is wrong on most bars and right on the few it sizes up can earn
/// while missing more often than it hits. These arms tell that case apart from the two
/// alternatives — a genuine flat-stake direction predictor, and a book earning nothing but the
/// panel's own drift.
///
/// Every arm is the SAME bars, the SAME windows, the SAME null, the SAME accounting and the
/// SAME bootstrap blocks. They differ only in which half of the decision survives:
///
/// * `sign-only` keeps `sign(f_t)` and destroys the magnitude by staking a constant leverage
///   equal to the actual policy's own mean `|f|`, so gross exposure is matched by construction
///   and only the size information is gone.
/// * the two magnitude arms keep `|f_t|` bar for bar and destroy the sign. Two of them because
///   they answer different questions: a coin flip removes the sign's information AND any net
///   exposure to the panel's drift, while an unconditional short removes the information and
///   KEEPS the drift — which is the arm that says how much of the edge is short bias times
///   sizing rather than forecasting.
/// * `always-short` is the flat-leverage version of that same drift question, at the same mean
///   `|f|`, carrying neither sign information nor size information.
pub const ATTRIBUTION_ACTUAL: usize = 0;
pub const ATTRIBUTION_SIGN_ONLY: usize = 1;
pub const ATTRIBUTION_MAGNITUDE_RANDOM: usize = 2;
pub const ATTRIBUTION_MAGNITUDE_SHORT: usize = 3;
pub const ATTRIBUTION_SHORT_CONSTANT: usize = 4;
pub const ATTRIBUTION_MARGINAL: usize = 5;
pub const ATTRIBUTION_ARMS: usize = 6;

pub const ATTRIBUTION_NAMES: [&str; ATTRIBUTION_ARMS] = [
    "actual",
    "sign-only at mean |f|",
    "magnitude-only, coin-flip sign",
    "magnitude-only, short sign",
    "always-short at mean |f|",
    "marginal null",
];

/// Stream the coin-flip signs are drawn from, counter-based on the bar's GLOBAL index so the
/// arm is reproducible from the seed alone and does not depend on chunking or iteration order.
pub const ATTRIBUTION_SIGN_SEED: u64 = 0x5157_F11F_5EED_0001;

/// Deciles of the model's own UNCAPPED `|f*|` — its confidence — that the attribution is read
/// across.
///
/// The capped `|f|` is the wrong axis: the cap binds on most bars, so its histogram is a spike
/// and a decile split on it is a split on nothing. `|f*|` is `|mu| / sigma^2` to first order
/// and has real spread, which is what makes "is the model right where it bets big" a question
/// this panel can answer at all.
pub const ATTRIBUTION_DECILES: usize = 10;

const PANEL_CORR_ABS: usize = 0;
const PANEL_CORR_SIGNED: usize = 1;
const PANEL_HIT_SHARE: usize = 2;
const PANEL_FLAT_SHARE: usize = 3;
const PANEL_WIN_GROWTH_BPS: usize = 4;
const PANEL_LOSS_GROWTH_BPS: usize = 5;
const PANEL_WIN_LOSS_RATIO: usize = 6;
const PANEL_WIN_ABS_PNL_BPS: usize = 7;
const PANEL_LOSS_ABS_PNL_BPS: usize = 8;
const PANEL_WIN_ABS_R_BPS: usize = 9;
const PANEL_LOSS_ABS_R_BPS: usize = 10;
const PANEL_WIN_ABS_F: usize = 11;
const PANEL_LOSS_ABS_F: usize = 12;
const PANEL_MEAN_FR_BPS: usize = 13;
pub const PANEL_SCALARS: usize = 14;

pub const PANEL_LABELS: [&str; PANEL_SCALARS] = [
    "corr(|f|, |R|)",
    "corr(f, R)",
    "hit share of positioned bars",
    "flat-bar share of positioned bars",
    "mean ln(1+fR) on hits (bps)",
    "mean ln(1+fR) on misses (bps)",
    "win/loss size ratio",
    "mean |fR| on hits (bps)",
    "mean |fR| on misses (bps)",
    "mean |R| on hits (bps)",
    "mean |R| on misses (bps)",
    "mean |f| on hits",
    "mean |f| on misses",
    "mean fR per bar (bps)",
];

const CELL_SHARE: usize = 0;
const CELL_ABS_FREE: usize = 1;
const CELL_HIT: usize = 2;
const CELL_GROWTH_BPS: usize = 3;
const CELL_FR_BPS: usize = 4;
const CELL_ABS_R_BPS: usize = 5;
pub const CELL_SCALARS: usize = 6;

pub const CELL_LABELS: [&str; CELL_SCALARS] = [
    "share of bars",
    "mean |f*| (uncapped)",
    "hit rate",
    "mean ln(1+fR) (bps/bar)",
    "mean fR (bps/bar)",
    "mean |R| (bps)",
];

/// Every scalar of ONE bootstrap draw, so the panel and the confidence deciles come from one
/// resample and a comparison between two of them is paired rather than a difference of
/// independent intervals.
const PANEL_TOTAL: usize = PANEL_SCALARS + ATTRIBUTION_DECILES * CELL_SCALARS;

/// One confidence decile's running sums.
#[derive(Clone, Copy, Debug, Default)]
struct CellSums {
    bars: f64,
    positioned: f64,
    hits: f64,
    abs_free: f64,
    growth: f64,
    pnl: f64,
    abs_r: f64,
}

impl CellSums {
    fn absorb(&mut self, other: &Self) {
        self.bars += other.bars;
        self.positioned += other.positioned;
        self.hits += other.hits;
        self.abs_free += other.abs_free;
        self.growth += other.growth;
        self.pnl += other.pnl;
        self.abs_r += other.abs_r;
    }
}

/// One block's running sums over the traded panel: the two correlations, the hit/miss split
/// and the confidence deciles, all from one pass over the bars.
#[derive(Clone, Copy, Debug, Default)]
struct PanelSums {
    bars: f64,
    f: f64,
    r: f64,
    f2: f64,
    r2: f64,
    fr: f64,
    af: f64,
    ar: f64,
    af2: f64,
    ar2: f64,
    afar: f64,
    positioned: f64,
    hits: f64,
    flats: f64,
    misses: f64,
    hit_growth: f64,
    hit_pnl: f64,
    hit_abs_r: f64,
    hit_abs_f: f64,
    miss_growth: f64,
    miss_pnl: f64,
    miss_abs_r: f64,
    miss_abs_f: f64,
    cells: [CellSums; ATTRIBUTION_DECILES],
}

impl PanelSums {
    /// One bar: the position actually held, the realized simple return, the UNCAPPED fraction
    /// that sized it, and the confidence decile that fraction falls in.
    ///
    /// The hit/miss split is [`Ledger`]'s, deliberately: `positioned` counts every bar holding
    /// a position and a hit is `f R > 0`, so [`PanelSums::census`]'s hit share is the same
    /// number [`PolicyStats::hit_rate`] reports and the two cannot drift. A bar positioned into
    /// a FLAT move is neither a hit nor a miss and is counted apart, because folding it into
    /// the misses would drag the reported mean loss toward zero with bars that lost nothing.
    fn push(&mut self, f: f64, r: f64, free: f64, cell: usize) {
        let growth = (1.0 + f * r).max(WEALTH_FLOOR).ln();
        let (af, ar) = (f.abs(), r.abs());
        self.bars += 1.0;
        self.f += f;
        self.r += r;
        self.f2 += f * f;
        self.r2 += r * r;
        self.fr += f * r;
        self.af += af;
        self.ar += ar;
        self.af2 += af * af;
        self.ar2 += ar * ar;
        self.afar += af * ar;
        self.cells[cell].bars += 1.0;
        self.cells[cell].abs_free += free.abs();
        self.cells[cell].growth += growth;
        self.cells[cell].pnl += f * r;
        self.cells[cell].abs_r += ar;
        if f == 0.0 {
            return;
        }
        self.positioned += 1.0;
        self.cells[cell].positioned += 1.0;
        if f * r > 0.0 {
            self.hits += 1.0;
            self.cells[cell].hits += 1.0;
            self.hit_growth += growth;
            self.hit_pnl += af * ar;
            self.hit_abs_r += ar;
            self.hit_abs_f += af;
        } else if r == 0.0 {
            self.flats += 1.0;
        } else {
            self.misses += 1.0;
            self.miss_growth += growth;
            self.miss_pnl += af * ar;
            self.miss_abs_r += ar;
            self.miss_abs_f += af;
        }
    }

    fn absorb(&mut self, other: &Self) {
        self.bars += other.bars;
        self.f += other.f;
        self.r += other.r;
        self.f2 += other.f2;
        self.r2 += other.r2;
        self.fr += other.fr;
        self.af += other.af;
        self.ar += other.ar;
        self.af2 += other.af2;
        self.ar2 += other.ar2;
        self.afar += other.afar;
        self.positioned += other.positioned;
        self.hits += other.hits;
        self.flats += other.flats;
        self.misses += other.misses;
        self.hit_growth += other.hit_growth;
        self.hit_pnl += other.hit_pnl;
        self.hit_abs_r += other.hit_abs_r;
        self.hit_abs_f += other.hit_abs_f;
        self.miss_growth += other.miss_growth;
        self.miss_pnl += other.miss_pnl;
        self.miss_abs_r += other.miss_abs_r;
        self.miss_abs_f += other.miss_abs_f;
        for (cell, source) in self.cells.iter_mut().zip(&other.cells) {
            cell.absorb(source);
        }
    }

    fn census(&self) -> [f64; PANEL_TOTAL] {
        let mut out = [f64::NAN; PANEL_TOTAL];
        if self.bars <= 0.0 {
            return out;
        }
        let bars = self.bars;
        let correlation = |sxy: f64, sx: f64, sy: f64, sxx: f64, syy: f64| {
            let covariance = sxy / bars - (sx / bars) * (sy / bars);
            let vx = (sxx / bars - (sx / bars) * (sx / bars)).max(0.0);
            let vy = (syy / bars - (sy / bars) * (sy / bars)).max(0.0);
            let scale = (vx * vy).sqrt();
            if scale > 0.0 {
                covariance / scale
            } else {
                f64::NAN
            }
        };
        let mean = |sum: f64, count: f64| {
            if count > 0.0 {
                sum / count
            } else {
                f64::NAN
            }
        };
        out[PANEL_CORR_ABS] = correlation(self.afar, self.af, self.ar, self.af2, self.ar2);
        out[PANEL_CORR_SIGNED] = correlation(self.fr, self.f, self.r, self.f2, self.r2);
        out[PANEL_HIT_SHARE] = mean(self.hits, self.positioned);
        out[PANEL_FLAT_SHARE] = mean(self.flats, self.positioned);
        let win = mean(self.hit_growth, self.hits);
        let loss = mean(self.miss_growth, self.misses);
        out[PANEL_WIN_GROWTH_BPS] = win * 1e4;
        out[PANEL_LOSS_GROWTH_BPS] = loss * 1e4;
        out[PANEL_WIN_LOSS_RATIO] = if loss.abs() > 0.0 {
            win.abs() / loss.abs()
        } else {
            f64::NAN
        };
        out[PANEL_WIN_ABS_PNL_BPS] = mean(self.hit_pnl, self.hits) * 1e4;
        out[PANEL_LOSS_ABS_PNL_BPS] = mean(self.miss_pnl, self.misses) * 1e4;
        out[PANEL_WIN_ABS_R_BPS] = mean(self.hit_abs_r, self.hits) * 1e4;
        out[PANEL_LOSS_ABS_R_BPS] = mean(self.miss_abs_r, self.misses) * 1e4;
        out[PANEL_WIN_ABS_F] = mean(self.hit_abs_f, self.hits);
        out[PANEL_LOSS_ABS_F] = mean(self.miss_abs_f, self.misses);
        out[PANEL_MEAN_FR_BPS] = self.fr / bars * 1e4;
        for (index, cell) in self.cells.iter().enumerate() {
            let base = PANEL_SCALARS + index * CELL_SCALARS;
            out[base + CELL_SHARE] = cell.bars / bars;
            out[base + CELL_ABS_FREE] = mean(cell.abs_free, cell.bars);
            out[base + CELL_HIT] = mean(cell.hits, cell.positioned);
            out[base + CELL_GROWTH_BPS] = mean(cell.growth, cell.bars) * 1e4;
            out[base + CELL_FR_BPS] = mean(cell.pnl, cell.bars) * 1e4;
            out[base + CELL_ABS_R_BPS] = mean(cell.abs_r, cell.bars) * 1e4;
        }
        out
    }
}

/// What the traded panel looks like underneath the headline: the two correlations, the
/// asymmetry between a winning bar and a losing one, and the same picture cut by the model's
/// own confidence.
#[derive(Clone, Copy, Debug)]
pub struct TradedPanel {
    scalars: [BlockedScalar; PANEL_SCALARS],
    cells: [[BlockedScalar; CELL_SCALARS]; ATTRIBUTION_DECILES],
    /// The nine `|f*|` cut points the deciles were split at, so a cell can be named in the
    /// units the model produced rather than by its index alone.
    cuts: [f64; ATTRIBUTION_DECILES - 1],
    pub blocks: usize,
    pub samples: usize,
}

impl TradedPanel {
    pub fn nan() -> Self {
        Self {
            scalars: [BlockedScalar::nan(); PANEL_SCALARS],
            cells: [[BlockedScalar::nan(); CELL_SCALARS]; ATTRIBUTION_DECILES],
            cuts: [f64::NAN; ATTRIBUTION_DECILES - 1],
            blocks: 0,
            samples: 0,
        }
    }

    pub fn measured(&self) -> bool {
        self.samples > 0
    }

    pub fn scalars(&self) -> &[BlockedScalar; PANEL_SCALARS] {
        &self.scalars
    }

    pub fn cells(&self) -> &[[BlockedScalar; CELL_SCALARS]; ATTRIBUTION_DECILES] {
        &self.cells
    }

    pub fn cuts(&self) -> &[f64; ATTRIBUTION_DECILES - 1] {
        &self.cuts
    }

    pub fn corr_abs(&self) -> BlockedScalar {
        self.scalars[PANEL_CORR_ABS]
    }

    pub fn corr_signed(&self) -> BlockedScalar {
        self.scalars[PANEL_CORR_SIGNED]
    }

    pub fn win_loss_ratio(&self) -> BlockedScalar {
        self.scalars[PANEL_WIN_LOSS_RATIO]
    }

    /// One confidence decile's hit rate, blocked and intervalled over the shared draws.
    pub fn hit(&self, decile: usize) -> BlockedScalar {
        self.cells[decile][CELL_HIT]
    }

    /// Top confidence decile's hit rate minus the bottom's.
    ///
    /// The discriminator between a forecaster whose sign is uninformative everywhere and one
    /// whose sign is informative exactly where it bets big. A GRADIENT and not a test: the two
    /// cells carry their own intervals from the shared draws, but the difference is not itself
    /// intervalled here, so it is read as a direction and never quoted as resolvable.
    pub fn confidence_hit_gradient(&self) -> f64 {
        self.cells[ATTRIBUTION_DECILES - 1][CELL_HIT].point - self.cells[0][CELL_HIT].point
    }

    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec!["traded panel: not measured".to_owned()];
        }
        let mut lines = vec![format!(
            "traded panel over {} bars / {} blocks",
            self.samples, self.blocks
        )];
        for (label, scalar) in PANEL_LABELS.iter().zip(&self.scalars) {
            lines.push(format!(
                "  {label:<36} {:+.5} (95% CI {:+.5}..{:+.5}, se {:.5})",
                scalar.point, scalar.ci.0, scalar.ci.1, scalar.se,
            ));
        }
        lines.push(format!(
            "  confidence deciles of the UNCAPPED |f*|, cuts at [{}]",
            self.cuts
                .iter()
                .map(|cut| format!("{cut:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
        ));
        lines.push(format!(
            "  {:<8}{:>9}{:>10}{:>28}{:>16}{:>14}{:>12}",
            "decile", "share", "|f*|", "hit rate (95% CI)", "ln(1+fR) bps", "fR bps", "|R| bps",
        ));
        for (index, cell) in self.cells.iter().enumerate() {
            lines.push(format!(
                "  {:<8}{:>9.4}{:>10.3}{:>10.4} ({:.4}..{:.4}){:>+16.4}{:>+14.4}{:>12.2}",
                index,
                cell[CELL_SHARE].point,
                cell[CELL_ABS_FREE].point,
                cell[CELL_HIT].point,
                cell[CELL_HIT].ci.0,
                cell[CELL_HIT].ci.1,
                cell[CELL_GROWTH_BPS].point,
                cell[CELL_FR_BPS].point,
                cell[CELL_ABS_R_BPS].point,
            ));
        }
        lines.push(format!(
            "  hit-rate gradient across confidence: {:+.4} (top decile minus bottom)",
            self.confidence_hit_gradient(),
        ));
        lines
    }
}

/// Measure the traded panel: one pass over the bars, blocked and bootstrapped over the SAME
/// blocks and the SAME RNG stream every other interval in this module is taken over.
///
/// The confidence axis is the UNCAPPED `|f*|`, so the deciles need [`WindowPaths::free`]; a
/// window set built by the accounting-only constructor does not carry it, and the panel then
/// refuses rather than splitting on the capped position, whose histogram is a spike at the cap.
pub fn traded_panel(windows: &[WindowPaths], blocks: &[u64]) -> TradedPanel {
    if windows.is_empty() || blocks.len() < windows.len() {
        return TradedPanel::nan();
    }
    if windows.iter().any(|window| window.free.len() != window.bars()) {
        return TradedPanel::nan();
    }
    let mut magnitudes: Vec<f64> = windows
        .iter()
        .flat_map(|window| window.free.iter().map(|f| f.abs()))
        .collect();
    if magnitudes.is_empty() {
        return TradedPanel::nan();
    }
    magnitudes.sort_unstable_by(f64::total_cmp);
    let last = (magnitudes.len() - 1) as f64;
    let cuts: [f64; ATTRIBUTION_DECILES - 1] = std::array::from_fn(|index| {
        let q = (index + 1) as f64 / ATTRIBUTION_DECILES as f64;
        magnitudes[(last * q).round() as usize]
    });
    drop(magnitudes);

    let mut grouped: BTreeMap<u64, PanelSums> = BTreeMap::new();
    let mut samples = 0usize;
    for (window, block) in windows.iter().zip(blocks) {
        let sums = grouped.entry(*block).or_default();
        let positions = &window.positions[POLICY_MODEL];
        for ((f, r), free) in positions.iter().zip(&window.realized).zip(&window.free) {
            if !f.is_finite() || !r.is_finite() || !free.is_finite() {
                continue;
            }
            // `cuts` is ascending, so the count of cuts at or below the magnitude IS the cell.
            let cell = cuts
                .partition_point(|cut| *cut <= free.abs())
                .min(ATTRIBUTION_DECILES - 1);
            sums.push(*f, *r, *free, cell);
            samples += 1;
        }
    }
    let totals: Vec<PanelSums> = grouped.into_values().collect();
    let mut pooled = PanelSums::default();
    for block in &totals {
        pooled.absorb(block);
    }
    let point = pooled.census();
    let mut panel = TradedPanel {
        scalars: std::array::from_fn(|index| BlockedScalar {
            point: point[index],
            ..BlockedScalar::nan()
        }),
        cells: std::array::from_fn(|cell| {
            std::array::from_fn(|scalar| BlockedScalar {
                point: point[PANEL_SCALARS + cell * CELL_SCALARS + scalar],
                ..BlockedScalar::nan()
            })
        }),
        cuts,
        blocks: totals.len(),
        samples,
    };
    if totals.len() < 2 {
        // One block is one observation: there is no dispersion to estimate, and a zero-width
        // interval reported as precision is the failure this refuses to commit.
        return panel;
    }

    let mut rng = ChaCha12Rng::seed_from_u64(BOOTSTRAP_SEED);
    let mut columns: [Vec<f64>; PANEL_TOTAL] =
        std::array::from_fn(|_| Vec::with_capacity(BOOTSTRAP_DRAWS));
    for _ in 0..BOOTSTRAP_DRAWS {
        let mut draw = PanelSums::default();
        for _ in 0..totals.len() {
            draw.absorb(totals.choose(&mut rng).expect("totals is non-empty"));
        }
        for (column, value) in columns.iter_mut().zip(draw.census()) {
            if value.is_finite() {
                column.push(value);
            }
        }
    }
    let tail = (1.0 - CI_MASS) / 2.0;
    let finish = |scalar: &mut BlockedScalar, column: &mut Vec<f64>| {
        if column.len() < 2 {
            return;
        }
        column.sort_by(f64::total_cmp);
        scalar.se = standard_deviation(column);
        scalar.ci = (
            sorted_percentile(column, tail),
            sorted_percentile(column, 1.0 - tail),
        );
    };
    for (scalar, column) in panel.scalars.iter_mut().zip(columns.iter_mut()) {
        finish(scalar, column);
    }
    for (index, cell) in panel.cells.iter_mut().enumerate() {
        for (slot, scalar) in cell.iter_mut().enumerate() {
            finish(
                scalar,
                &mut columns[PANEL_SCALARS + index * CELL_SCALARS + slot],
            );
        }
    }
    panel
}

/// One arm's whole verdict, on the same windows and against the same null as every other.
#[derive(Clone, Copy, Debug)]
pub struct AttributionArm {
    pub policy: PolicyStats,
    /// Net growth per bar against the marginal null, PAIRED window by window.
    pub edge: Dispersion,
    /// `arm - actual`, paired window by window over the same blocks. The two levels cannot
    /// answer how much an arm gave up, because both carry the market-common regime they share;
    /// this difference removes it.
    pub paired_vs_actual: Dispersion,
    pub break_even_bps: f64,
}

impl AttributionArm {
    pub fn nan() -> Self {
        Self {
            policy: PolicyStats::nan(),
            edge: Dispersion::nan(),
            paired_vs_actual: Dispersion::nan(),
            break_even_bps: f64::NAN,
        }
    }

    /// True when the arm's own edge interval lies strictly ABOVE zero.
    pub fn carries_edge(&self) -> bool {
        self.edge.ci_low.is_finite() && self.edge.ci_low > 0.0
    }

    /// True when the interval excludes zero in either direction, i.e. the arm's edge is
    /// resolvable rather than a noisy level.
    pub fn resolvable(&self) -> bool {
        self.edge.ci_low.is_finite()
            && self.edge.ci_high.is_finite()
            && (self.edge.ci_low > 0.0 || self.edge.ci_high < 0.0)
    }
}

/// Which half of the decision the measured edge actually lived in, read off the 2x2's MAIN
/// EFFECTS rather than off the arm levels.
///
/// Both effects are measured against the always-short corner at the same mean gross exposure,
/// so a panel that merely drifts down cannot manufacture a `Direction` verdict out of a model
/// that is only short.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeSource {
    /// Nothing was measured.
    Unmeasured,
    /// The actual policy's own edge interval does not exclude zero, so there is nothing to
    /// attribute and no effect below is interpretable.
    NoEdge,
    /// The SIGN main effect is resolvably positive and the SIZE main effect is not: at a flat
    /// stake, taking the model's side beats taking a constant side. A direction predictor.
    Direction,
    /// The SIZE main effect is resolvably positive and the SIGN main effect is not: holding the
    /// side constant, varying the stake with the model's own `|f_t|` beats a flat stake of the
    /// same mean size. The conditional VARIANCE is what the model knows — a volatility timer,
    /// not a return predictor.
    Magnitude,
    /// Both main effects are resolvably positive.
    Both,
    /// The actual policy earns and NEITHER main effect resolves above zero: the edge lives in
    /// the INTERACTION — the model is right where it sizes up — and destroying either half
    /// destroys it.
    Joint,
}

/// The five-way attribution, the 2x2 factorial underneath it, and the traded panel underneath
/// that.
///
/// # Why the raw arms are not the decomposition
///
/// On a panel with a DRIFT, a constant short earns. `sign-only` therefore carries a positive
/// edge whenever the model is mostly short, whether or not its sign contains a single bit of
/// information — which is exactly the trap the 0.549 always-short hit rate sets. The four arms
/// `actual`, `sign-only`, `magnitude-only (short sign)` and `always-short` are the four corners
/// of a 2x2 in (does the SIGN come from the model, does the SIZE come from the model), all four
/// at the same mean gross exposure and all four holding the drift identically. The main effects
/// and the interaction of that design, paired window by window, are the decomposition; the raw
/// arm levels are only its margins.
#[derive(Clone, Debug)]
pub struct EdgeAttribution {
    pub arms: [AttributionArm; ATTRIBUTION_ARMS],
    /// `sign-only` minus `always-short`. Both stake a flat `matched_leverage` and both hold the
    /// drift; they differ only in where the sign came from, so this is what the model's SIGN is
    /// worth net of the short bias.
    pub sign_effect: Dispersion,
    /// `magnitude-only (short sign)` minus `always-short`. Both are unconditionally short with
    /// the same mean `|f|`; they differ only in whether the size varies with the model's own
    /// `|f_t|`, so this is what the model's SIZE is worth net of the short bias.
    pub size_effect: Dispersion,
    /// `actual - sign-only - magnitude-only(short) + always-short`, paired.
    ///
    /// What the two halves are worth TOGETHER beyond the sum of what each is worth alone. A
    /// forecaster that is right precisely where it sizes up has its entire edge here and none of
    /// it in either main effect, and that case is invisible to any single-arm comparison.
    pub interaction: Dispersion,
    /// The constant leverage the sign-only and always-short arms stake: the ACTUAL policy's
    /// mean `|f|` at this cap, so gross exposure is matched by construction rather than by
    /// choice.
    pub matched_leverage: f64,
    pub panel: TradedPanel,
    pub cost_bps: f64,
    pub leverage_cap: f64,
    pub bars: usize,
    pub windows: usize,
    pub blocks: usize,
    /// Per-window traded notional for each arm, in the traded windows' own order.
    ///
    /// A window IS a symbol, so this is what lets a cost priced per symbol be re-weighted by
    /// what the book actually rotated in that symbol. Carried per arm because an arm's
    /// break-even is a turnover number and the arms rotate differently.
    pub turnover: [Vec<WindowTurnover>; ATTRIBUTION_ARMS],
    /// Where in the conviction distribution the ACTUAL policy's sign reversals happen.
    pub flips: FlipConviction,
}

/// True when a paired interval lies strictly above zero.
fn resolvably_positive(paired: &Dispersion) -> bool {
    paired.ci_low.is_finite() && paired.ci_low > 0.0
}

impl EdgeAttribution {
    pub fn nan() -> Self {
        Self {
            arms: [AttributionArm::nan(); ATTRIBUTION_ARMS],
            sign_effect: Dispersion::nan(),
            size_effect: Dispersion::nan(),
            interaction: Dispersion::nan(),
            matched_leverage: f64::NAN,
            panel: TradedPanel::nan(),
            cost_bps: f64::NAN,
            leverage_cap: f64::NAN,
            bars: 0,
            windows: 0,
            blocks: 0,
            turnover: std::array::from_fn(|_| Vec::new()),
            flips: FlipConviction::nan(),
        }
    }

    pub fn measured(&self) -> bool {
        self.bars > 0
    }

    /// What the always-short corner earns on its own: the panel's DRIFT at the matched
    /// leverage, which is the baseline both main effects are measured against.
    pub fn drift_edge(&self) -> Dispersion {
        self.arms[ATTRIBUTION_SHORT_CONSTANT].edge
    }

    pub fn verdict(&self) -> EdgeSource {
        if !self.measured() {
            return EdgeSource::Unmeasured;
        }
        if !self.arms[ATTRIBUTION_ACTUAL].carries_edge() {
            return EdgeSource::NoEdge;
        }
        match (
            resolvably_positive(&self.sign_effect),
            resolvably_positive(&self.size_effect),
        ) {
            (true, true) => EdgeSource::Both,
            (true, false) => EdgeSource::Direction,
            (false, true) => EdgeSource::Magnitude,
            (false, false) => EdgeSource::Joint,
        }
    }

    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec!["edge attribution: not measured".to_owned()];
        }
        let mut lines = vec![format!(
            "edge attribution ({} windows / {} bars / {} blocks, cap {:.1}x, cost {:.2} bps, \
             matched leverage {:.4} = the actual policy's own mean |f|)",
            self.windows,
            self.bars,
            self.blocks,
            self.leverage_cap,
            self.cost_bps,
            self.matched_leverage,
        )];
        lines.push(format!(
            "  {:<32}{:>32}{:>12}{:>9}{:>8}{:>10}{:>9}{:>30}",
            "arm",
            "edge bps/bar (95% CI)",
            "break-even",
            "sharpe",
            "hit",
            "turnover",
            "mean|f|",
            "vs actual PAIRED (95% CI)",
        ));
        for (index, arm) in self.arms.iter().enumerate() {
            lines.push(format!(
                "  {:<32}{:>+10.4} ({:+.4}..{:+.4}){:>12}{:>+9.2}{:>8.4}{:>10.4}{:>9.3}\
                 {:>+12.4} ({:+.4}..{:+.4}){}",
                ATTRIBUTION_NAMES[index],
                arm.edge.mean * 1e4,
                arm.edge.ci_low * 1e4,
                arm.edge.ci_high * 1e4,
                TradeBench::break_even_text(arm.break_even_bps),
                arm.policy.sharpe,
                arm.policy.hit_rate,
                arm.policy.turnover,
                arm.policy.mean_abs_position,
                arm.paired_vs_actual.mean * 1e4,
                arm.paired_vs_actual.ci_low * 1e4,
                arm.paired_vs_actual.ci_high * 1e4,
                if arm.resolvable() {
                    ""
                } else {
                    "  EDGE NOT RESOLVABLE"
                },
            ));
        }
        let effect = |name: &str, paired: &Dispersion| {
            format!(
                "  {name:<44}{:>+10.4} bps/bar (95% CI {:+.4}..{:+.4}, se {:.4}){}",
                paired.mean * 1e4,
                paired.ci_low * 1e4,
                paired.ci_high * 1e4,
                paired.se * 1e4,
                if resolvably_positive(paired) {
                    ""
                } else {
                    "  NOT RESOLVABLY POSITIVE"
                },
            )
        };
        lines.push(
            "  2x2 on (sign from the model?, size from the model?), all four corners at the \
             same mean gross exposure and all four holding the panel's drift:"
                .to_owned(),
        );
        lines.push(effect(
            "SIGN effect (sign-only - always-short)",
            &self.sign_effect,
        ));
        lines.push(effect(
            "SIZE effect (magnitude-short - always-short)",
            &self.size_effect,
        ));
        lines.push(effect(
            "INTERACTION (actual - sign - size + short)",
            &self.interaction,
        ));
        lines.push(effect("DRIFT corner (always-short - null)", &self.drift_edge()));
        lines.extend(
            self.panel
                .report_lines()
                .into_iter()
                .map(|line| format!("  {line}")),
        );
        lines.extend(self.flips.report_lines());
        lines.push(
            "  turnover the MODEL chose, per arm: interior excludes the window sampler's \
             entry-from-flat and terminal unwind"
                .to_owned(),
        );
        for (index, name) in ATTRIBUTION_NAMES.iter().enumerate() {
            let rows = &self.turnover[index];
            if rows.is_empty() {
                continue;
            }
            let total: f64 = rows.iter().map(|row| row.total).sum();
            let interior: f64 = rows.iter().map(|row| row.interior).sum();
            let bars: usize = rows.iter().map(|row| row.bars).sum();
            lines.push(format!(
                "    {name:<34}{:>10.4} total/bar{:>12.4} interior/bar   interior share {:.4}",
                total / bars as f64,
                interior / bars as f64,
                interior / total,
            ));
        }
        lines.push(self.verdict_line());
        lines
    }

    pub fn verdict_line(&self) -> String {
        let actual = &self.arms[ATTRIBUTION_ACTUAL];
        let effects = format!(
            "sign {:+.4} ({:+.4}..{:+.4}), size {:+.4} ({:+.4}..{:+.4}), interaction {:+.4} \
             ({:+.4}..{:+.4}), drift corner {:+.4}, against an actual {:+.4} bps/bar",
            self.sign_effect.mean * 1e4,
            self.sign_effect.ci_low * 1e4,
            self.sign_effect.ci_high * 1e4,
            self.size_effect.mean * 1e4,
            self.size_effect.ci_low * 1e4,
            self.size_effect.ci_high * 1e4,
            self.interaction.mean * 1e4,
            self.interaction.ci_low * 1e4,
            self.interaction.ci_high * 1e4,
            self.drift_edge().mean * 1e4,
            actual.edge.mean * 1e4,
        );
        let body = match self.verdict() {
            EdgeSource::Unmeasured => "nothing was measured".to_owned(),
            EdgeSource::NoEdge => format!(
                "the actual policy's own edge interval {:+.4}..{:+.4} bps/bar does not exclude \
                 zero, so there is no edge to attribute and no effect above is interpretable",
                actual.edge.ci_low * 1e4,
                actual.edge.ci_high * 1e4,
            ),
            EdgeSource::Direction => format!(
                "DIRECTION PREDICTOR. At a flat {:.4} stake, taking the model's side resolvably \
                 beats taking a constant side, and varying the stake at a constant side does \
                 not: {effects}",
                self.matched_leverage,
            ),
            EdgeSource::Magnitude => format!(
                "VOLATILITY TIMER, NOT A RETURN PREDICTOR. Holding the side constant, varying \
                 the stake with the model's own |f_t| resolvably beats a flat stake of the same \
                 mean size, while taking the model's side at a flat stake does not beat taking \
                 a constant one. What the model knows is the conditional VARIANCE; the \
                 conditional mean's sign is not carrying the result: {effects}"
            ),
            EdgeSource::Both => format!(
                "BOTH HALVES CARRY. Each main effect resolves above zero on its own at matched \
                 gross exposure: {effects}"
            ),
            EdgeSource::Joint => format!(
                "NEITHER HALF CARRIES ALONE. The actual policy earns {:+.4} bps/bar (95% CI \
                 {:+.4}..{:+.4}), but neither main effect of the 2x2 resolves above zero. The \
                 edge is in the INTERACTION — the model is right where it sizes up — so it is \
                 neither a flat-stake direction predictor nor a pure volatility timer: \
                 {effects}",
                actual.edge.mean * 1e4,
                actual.edge.ci_low * 1e4,
                actual.edge.ci_high * 1e4,
            ),
        };
        format!("VERDICT: {body}")
    }
}

/// One arm's position path, derived from the ACTUAL policy's own path bar for bar.
///
/// `f64::signum` is not usable alone here: it returns `+1.0` at `+0.0`, so a flat bar would be
/// turned into a full-size long. The explicit `!= 0.0` factor is the guard the oracle path uses
/// for the same reason.
fn attribution_paths(
    windows: &[WindowPaths],
    arm: usize,
    leverage: f64,
    free_marginal: f64,
    cap: f64,
) -> Vec<WindowPaths> {
    let mut offset = 0u64;
    windows
        .iter()
        .map(|window| {
            let actual = &window.positions[POLICY_MODEL];
            let path: Vec<f64> = actual
                .iter()
                .enumerate()
                .map(|(bar, f)| match arm {
                    ATTRIBUTION_ACTUAL => *f,
                    ATTRIBUTION_SIGN_ONLY => {
                        clamp_fraction(leverage * f.signum() * f64::from(*f != 0.0), cap)
                    }
                    ATTRIBUTION_MAGNITUDE_RANDOM => {
                        if mix64(ATTRIBUTION_SIGN_SEED, offset + bar as u64) >> 63 == 0 {
                            f.abs()
                        } else {
                            -f.abs()
                        }
                    }
                    ATTRIBUTION_MAGNITUDE_SHORT => -f.abs(),
                    ATTRIBUTION_SHORT_CONSTANT => clamp_fraction(-leverage, cap),
                    _ => clamp_fraction(free_marginal, cap),
                })
                .collect();
            offset += actual.len() as u64;
            let mut positions: [Vec<f64>; POLICY_COUNT] = std::array::from_fn(|_| Vec::new());
            positions[POLICY_MODEL] = path;
            WindowPaths::unmeasured(window.realized.clone(), Vec::new(), positions)
        })
        .collect()
}

/// The SIGN corner's position path, per window, with the exposure matching already applied.
///
/// Exposed so a hysteresis policy can be swept against the SAME construction the attribution
/// scores rather than against a second constant-`|f|` convention: at a zero flip margin the
/// swept policy must reproduce this path bar for bar, which makes the margin-zero row a
/// cross-check of two ledgers rather than two similar numbers that nobody can reconcile.
pub fn sign_only_positions(windows: &[WindowPaths], config: BenchConfig) -> Vec<Vec<f64>> {
    if windows.is_empty() {
        return Vec::new();
    }
    let recapped = recap(windows, config.cap, config.free_marginal);
    let matched = Ledger::build(&recapped, POLICY_MODEL, config.cap)
        .stats(config.cost_bps * 1e-4)
        .mean_abs_position;
    attribution_paths(
        &recapped,
        ATTRIBUTION_SIGN_ONLY,
        matched,
        config.free_marginal,
        config.cap,
    )
    .into_iter()
    .map(|mut window| std::mem::take(&mut window.positions[POLICY_MODEL]))
    .collect()
}

/// Where in the conviction distribution the model's sign REVERSALS happen.
///
/// Hysteresis on the sign is only cheap if flips concentrate where the predicted mean is near
/// zero: suppressing a coin-flip reversal costs almost no edge, while suppressing a reversal
/// the model was confident about costs the edge that reversal was going to earn. This is the
/// one diagnostic that says which of those two worlds the panel is in, and it is a histogram
/// over bars already in memory rather than another pass.
#[derive(Clone, Copy, Debug)]
pub struct FlipConviction {
    /// Deciles of `|mu_hat|` over POSITIONED bars, in log-return space.
    pub cuts: [f64; ATTRIBUTION_DECILES - 1],
    /// Share of the flips falling in each decile of that axis. Uniform means flips happen at
    /// every conviction; front-loaded means they are concentrated in the noise.
    pub flip_share: [f64; ATTRIBUTION_DECILES],
    pub flips: usize,
    pub positioned: usize,
    pub mean_abs_mu_positioned: f64,
    pub mean_abs_mu_flip: f64,
}

impl FlipConviction {
    pub fn nan() -> Self {
        Self {
            cuts: [f64::NAN; ATTRIBUTION_DECILES - 1],
            flip_share: [f64::NAN; ATTRIBUTION_DECILES],
            flips: 0,
            positioned: 0,
            mean_abs_mu_positioned: f64::NAN,
            mean_abs_mu_flip: f64::NAN,
        }
    }

    pub fn measured(&self) -> bool {
        self.positioned > 0
    }

    /// Share of flips in the bottom two conviction deciles - the part hysteresis gets nearly
    /// free. `0.2` is what an uninformative flip axis would produce.
    pub fn low_conviction_share(&self) -> f64 {
        self.flip_share[0] + self.flip_share[1]
    }

    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec!["flip conviction: not measured".to_owned()];
        }
        let mut lines = vec![format!(
            "  flip conviction ({} flips over {} positioned bars, {:.4} flips/bar)",
            self.flips,
            self.positioned,
            self.flips as f64 / self.positioned as f64,
        )];
        lines.push(format!(
            "    mean |mu_hat| bps: {:.4} at flips against {:.4} on all positioned bars \
             (ratio {:.4})",
            self.mean_abs_mu_flip * 1e4,
            self.mean_abs_mu_positioned * 1e4,
            self.mean_abs_mu_flip / self.mean_abs_mu_positioned,
        ));
        lines.push(format!(
            "    flip share by |mu_hat| decile: [{}] - uniform is 0.1000 everywhere",
            self.flip_share
                .iter()
                .map(|share| format!("{share:.4}"))
                .collect::<Vec<_>>()
                .join(", "),
        ));
        lines.push(format!(
            "    bottom two conviction deciles carry {:.4} of flips (0.2000 is uninformative)",
            self.low_conviction_share(),
        ));
        lines
    }
}

/// Measure [`FlipConviction`] on the ACTUAL policy's own path.
///
/// Refuses rather than guessing when the windows carry no predicted mean, because the
/// accounting-only constructor leaves it empty and a zero-filled conviction axis would report
/// every flip as maximally unconvinced - the most favourable possible answer, invented.
fn flip_conviction(windows: &[WindowPaths]) -> FlipConviction {
    if windows
        .iter()
        .any(|window| window.predicted_mean.len() != window.bars())
    {
        return FlipConviction::nan();
    }
    let mut positioned: Vec<f64> = Vec::new();
    let mut flipped: Vec<f64> = Vec::new();
    for window in windows {
        let path = &window.positions[POLICY_MODEL];
        for (bar, fraction) in path.iter().enumerate() {
            if *fraction == 0.0 {
                continue;
            }
            let conviction = window.predicted_mean[bar].abs();
            if !conviction.is_finite() {
                continue;
            }
            positioned.push(conviction);
            let held = if bar == 0 { 0.0 } else { path[bar - 1] };
            if held != 0.0 && held * fraction < 0.0 {
                flipped.push(conviction);
            }
        }
    }
    if positioned.is_empty() {
        return FlipConviction::nan();
    }
    let mut sorted = positioned.clone();
    sorted.sort_unstable_by(f64::total_cmp);
    let last = (sorted.len() - 1) as f64;
    let cuts: [f64; ATTRIBUTION_DECILES - 1] = std::array::from_fn(|index| {
        let q = (index + 1) as f64 / ATTRIBUTION_DECILES as f64;
        sorted[(last * q).round() as usize]
    });
    let bucket = |value: f64| cuts.iter().filter(|cut| value > **cut).count();
    let mut counts = [0usize; ATTRIBUTION_DECILES];
    for conviction in &flipped {
        counts[bucket(*conviction)] += 1;
    }
    let flips = flipped.len();
    let denominator = flips.max(1) as f64;
    FlipConviction {
        cuts,
        flip_share: std::array::from_fn(|index| counts[index] as f64 / denominator),
        flips,
        positioned: positioned.len(),
        mean_abs_mu_positioned: positioned.iter().sum::<f64>() / positioned.len() as f64,
        mean_abs_mu_flip: flipped.iter().sum::<f64>() / denominator,
    }
}

/// Split the measured edge into what the model's SIGN is worth and what its SIZE is worth.
///
/// Every arm is scored by the SAME [`Ledger`], against the SAME marginal null, on the SAME
/// windows, and intervalled by the SAME [`block_bootstrap`] over the SAME blocks at the SAME
/// seed — so the resampled block sequence is literally identical across arms, and the paired
/// differences are paired in the strong sense rather than merely differenced.
pub fn edge_attribution(
    windows: &[WindowPaths],
    blocks: &[u64],
    config: BenchConfig,
) -> EdgeAttribution {
    let mut result = EdgeAttribution::nan();
    result.cost_bps = config.cost_bps;
    result.leverage_cap = config.cap;
    if windows.is_empty() {
        return result;
    }
    assert!(
        blocks.len() >= windows.len(),
        "every traded window needs a bootstrap block assignment: {} blocks for {} windows",
        blocks.len(),
        windows.len()
    );
    let BenchConfig {
        cost_bps,
        cap,
        free_marginal,
    } = config;
    let blocks = &blocks[..windows.len()];
    let cost = cost_bps * 1e-4;

    // Re-clamped once, so every arm is derived from the identical position path the headline
    // bench scored rather than from whatever cap the windows happened to be solved at.
    let recapped = recap(windows, cap, free_marginal);
    let matched_leverage = Ledger::build(&recapped, POLICY_MODEL, cap)
        .stats(cost)
        .mean_abs_position;

    // One arm at a time, so only one arm's synthesized paths are resident at once.
    let ledgers: Vec<Ledger> = (0..ATTRIBUTION_ARMS)
        .map(|arm| {
            let paths = attribution_paths(&recapped, arm, matched_leverage, free_marginal, cap);
            Ledger::build(&paths, POLICY_MODEL, cap)
        })
        .collect();
    let growth: Vec<Vec<f64>> = ledgers.iter().map(|arm| arm.window_growth(cost)).collect();

    for arm in 0..ATTRIBUTION_ARMS {
        let deltas: Vec<f64> = growth[arm]
            .iter()
            .zip(&growth[ATTRIBUTION_MARGINAL])
            .map(|(arm, null)| arm - null)
            .collect();
        let against_actual: Vec<f64> = growth[arm]
            .iter()
            .zip(&growth[ATTRIBUTION_ACTUAL])
            .map(|(arm, actual)| arm - actual)
            .collect();
        let edge_at = |bps: f64| {
            let cost = bps * 1e-4;
            ledgers[arm].net_growth_per_bar(cost)
                - ledgers[ATTRIBUTION_MARGINAL].net_growth_per_bar(cost)
        };
        result.arms[arm] = AttributionArm {
            policy: ledgers[arm].stats(cost),
            edge: block_bootstrap(&deltas, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
            paired_vs_actual: block_bootstrap(
                &against_actual,
                blocks,
                BOOTSTRAP_DRAWS,
                BOOTSTRAP_SEED,
            ),
            break_even_bps: break_even_bps(&edge_at),
        };
        result.turnover[arm] = ledgers[arm].turnover.clone();
    }

    // The 2x2's main effects and interaction, from the SAME per-window growth vectors the arm
    // intervals came from, so every contrast on this page is a contrast of the same numbers.
    let paired = |left: usize, right: usize| -> Dispersion {
        let deltas: Vec<f64> = growth[left]
            .iter()
            .zip(&growth[right])
            .map(|(left, right)| left - right)
            .collect();
        block_bootstrap(&deltas, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    };
    result.sign_effect = paired(ATTRIBUTION_SIGN_ONLY, ATTRIBUTION_SHORT_CONSTANT);
    result.size_effect = paired(ATTRIBUTION_MAGNITUDE_SHORT, ATTRIBUTION_SHORT_CONSTANT);
    let interaction: Vec<f64> = (0..growth[ATTRIBUTION_ACTUAL].len())
        .map(|window| {
            growth[ATTRIBUTION_ACTUAL][window] - growth[ATTRIBUTION_SIGN_ONLY][window]
                - growth[ATTRIBUTION_MAGNITUDE_SHORT][window]
                + growth[ATTRIBUTION_SHORT_CONSTANT][window]
        })
        .collect();
    result.interaction = block_bootstrap(&interaction, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
    result.matched_leverage = matched_leverage;
    result.panel = traded_panel(&recapped, blocks);
    result.flips = flip_conviction(&recapped);
    result.bars = ledgers[ATTRIBUTION_ACTUAL].bars();
    result.windows = windows.len();
    result.blocks = result.arms[ATTRIBUTION_ACTUAL].edge.blocks;
    result
}

pub const HYSTERESIS_MARGINS: [f64; 13] = [
    0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, f64::INFINITY,
];

/// Standardized-conviction thresholds, in units of the head's own predicted SD.
///
/// The same doubling structure and the same length as [`HYSTERESIS_MARGINS`], so the two axes
/// share a grid INDEX and can be charted and tested against each other row for row.
pub const HYSTERESIS_SD_MARGINS: [f64; 13] = [
    0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, f64::INFINITY,
];

/// What a flip margin is compared AGAINST.
///
/// # Why this is an axis and not a constant
///
/// A threshold on raw `|mu_hat|` is not a threshold on signal quality. `|mu_hat|` is large partly
/// because the name is VOLATILE, so filtering on it preferentially retains high-sigma names -
/// which are thinner, dearer and higher-impact. Measured on this panel, the book's
/// turnover-weighted ADV percentile falls from 0.5954 at margin zero to 0.4742 at margin 32 and
/// twenty of 256 names stop trading altogether: a conviction filter is a COVERT LIQUIDITY
/// FILTER, and it makes the per-unit cost RISE as the book slows, which is the opposite of what
/// a turnover cut is supposed to buy.
///
/// Dividing by the head's own predicted SD thresholds the standardized signal instead, which is
/// scale-free across names. Whether that actually removes the tilt is a question about the ADV
/// percentile of the surviving turnover - a column with no P&L in it - so it is decided by
/// measurement on a different quantity than the one it is meant to improve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvictionAxis {
    /// `|mu_hat|` in bps. The incumbent construction.
    Raw,
    /// `|mu_hat| / sd_hat`, in units of the head's own predicted SD.
    Standardized,
}

impl ConvictionAxis {
    pub fn margins(self) -> &'static [f64; 13] {
        match self {
            Self::Raw => &HYSTERESIS_MARGINS,
            Self::Standardized => &HYSTERESIS_SD_MARGINS,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw |mu|",
            Self::Standardized => "|mu|/sd",
        }
    }

    /// The margin in the same units as [`conviction`](Self::conviction): bps convert, standard
    /// deviations are already dimensionless.
    fn threshold(self, margin: f64) -> f64 {
        match self {
            Self::Raw => margin * 1e-4,
            Self::Standardized => margin,
        }
    }

    /// Units the margin is quoted in, so a table never mixes bps with standard deviations.
    pub fn units(self) -> &'static str {
        match self {
            Self::Raw => "bps",
            Self::Standardized => "sd",
        }
    }

    /// Decimals needed to render this axis's grid without two margins colliding.
    ///
    /// The sd grid's two tightest knobs are 0.005 and 0.01, which BOTH render as "0.01" at two
    /// decimals - and a collision here is silent and destructive rather than cosmetic: two
    /// different books get written under one policy string, a downstream join pools them, and the
    /// absorbed arm vanishes from every table while the surviving label describes a book nobody
    /// ran. This shipped once and cost a grid arm.
    fn decimals(self) -> usize {
        match self {
            Self::Raw => 2,
            Self::Standardized => 3,
        }
    }

    /// This axis's margin, in its own units and at enough precision to be unique on its grid.
    pub fn margin_label(self, margin: f64) -> String {
        if margin.is_infinite() {
            "never".to_owned()
        } else {
            format!("{margin:.*}", self.decimals())
        }
    }

    /// The policy string a per-window turnover row is keyed by, naming BOTH the axis and the
    /// margin: two axes' margins are numerically overlapping and mean different things.
    pub fn policy_label(self, margin: f64) -> String {
        format!(
            "sign hysteresis grid [{}] margin {} {}",
            self.name(),
            self.margin_label(margin),
            self.units()
        )
    }

    /// The conviction a bar's reversal is judged on. `NAN` when the axis is unavailable, which
    /// makes the margin comparison fail closed rather than treating an absent SD as certainty.
    fn conviction(self, window: &WindowPaths, bar: usize) -> f64 {
        let mean = window.predicted_mean[bar].abs();
        match self {
            Self::Raw => mean,
            Self::Standardized => {
                let variance = window.predicted_var.get(bar).copied().unwrap_or(f64::NAN);
                if variance > 0.0 {
                    mean / variance.sqrt()
                } else {
                    f64::NAN
                }
            }
        }
    }
}

pub const CONVICTION_AXES: [ConvictionAxis; 2] =
    [ConvictionAxis::Raw, ConvictionAxis::Standardized];

/// The measured cost levels the frontier's net growth is evaluated at, cheapest first.
///
/// The first four are WEIGHTINGS of the same impact-free measurement rather than four different
/// costs. A constant-`|f|` book's own composite was expected to be pinned between the sign-only
/// weighting and the published equal weighting under any retention monotone in a name's flip
/// count; quoting both reports the width of that expectation instead of picking a point inside
/// it. The actual book's weighting leads because it is what the incumbent is quoted against, and
/// the shrunk book's is the dearest weighting on this panel and prices the shrink cells of the
/// composition 2x2 on their own weights.
///
/// The last two are the fitted books' OWN MEASURED composites, and they are here because they
/// broke that expectation by a factor of 1.9 - see [`horizon::MATCHED_HYSTERESIS_PRIMARY_BPS`].
/// They are the only cost levels at which the two headline rows can be read honestly, so the
/// table carries them rather than leaving the correction to arithmetic downstream.
pub const HYSTERESIS_NET_COSTS: [(&str, f64); 6] = [
    ("actual-book", super::horizon::MATCHED_ACTUAL_BOOK_BPS),
    ("sign-only", super::horizon::MATCHED_SIGN_ONLY_BPS),
    ("equal-weighted", super::horizon::MATCHED_MEASURED_BPS),
    ("shrunk-book", super::horizon::MATCHED_SHRUNK_BOOK_BPS),
    (
        "fitted-book primary",
        super::horizon::MATCHED_HYSTERESIS_PRIMARY_BPS,
    ),
    (
        "fitted-book secondary",
        super::horizon::MATCHED_HYSTERESIS_SECONDARY_BPS,
    ),
];

/// Slot of the cost the margin is SELECTED on.
///
/// The equal-weighted constant, which is the dearest weighting a CONSTANT-`|f|` book can carry:
/// selection sits at the worst end of that bound deliberately, so a margin that wins does not
/// depend on the book's composite landing at the favourable end of a range nobody has measured
/// yet. Deliberately NOT the shrunk book's dearer weighting - no constant-`|f|` book can price
/// there, since that weighting is produced by the magnitude turnover this construction removes.
pub const HYSTERESIS_SELECTION_COST: usize = 2;
const _: () = assert!(
    HYSTERESIS_NET_COSTS[HYSTERESIS_SELECTION_COST].1 == super::horizon::MATCHED_MEASURED_BPS,
    "the selection cost is the equal-weighted published constant"
);

/// `never` for the unreachable margin, which is a knob setting and not a number.
pub fn margin_label(margin_bps: f64) -> String {
    if margin_bps.is_infinite() {
        "never".to_owned()
    } else {
        format!("{margin_bps:.2}")
    }
}

/// One flip margin's realized accounting.
#[derive(Clone, Debug)]
pub struct HysteresisPoint {
    /// Flip margin in bps of predicted mean. `INFINITY` never reverses.
    pub margin_bps: f64,
    pub policy: PolicyStats,
    /// Growth over the marginal null, paired window by window - the SAME baseline and the same
    /// blocks every arm in [`EdgeAttribution`] is quoted against. Scored at the bench's assumed
    /// cost, which is NOT the measured one.
    pub edge: Dispersion,
    /// Paired against the sign-only arm, which is this sweep at margin zero.
    pub vs_sign_only: Dispersion,
    /// Net growth over the null at [`horizon::MATCHED_MEASURED_BPS`], paired window by window.
    ///
    /// THE OBJECTIVE. `break_even_bps` is monotone increasing in the margin, so maximizing it
    /// selects the widest knob by construction and runs to a book that never trades; it is the
    /// right quantity to COMPARE against a cost and the wrong one to OPTIMIZE. This is growth
    /// actually realized once the measured cost is charged, which peaks and then falls.
    pub net_at_measured: Dispersion,
    /// `net_at_measured` paired against the sign-only incumbent.
    pub net_vs_sign_only: Dispersion,
    pub break_even_bps: f64,
    /// Mean length in bars of a maximal run of constant nonzero position.
    pub mean_hold_bars: f64,
    /// Per-window traded notional, so this book can be priced on ITS OWN weights rather than
    /// inheriting the incumbent's - an 82% turnover cut can arrive with a worse per-unit cost.
    pub turnover: Vec<WindowTurnover>,
    /// Net growth over the null at each of [`HYSTERESIS_NET_COSTS`], in bps/bar, computed
    /// THROUGH THE LEDGER at that cost rather than reconstructed as `gross - c * turnover`.
    ///
    /// The linear reconstruction is not exact, and this book is where that bites hardest. A
    /// bar's rebalance is charged `ln(1 - c * traded)` ([`Ledger::cost_of`]), so net growth is
    /// concave in traded notional and NONLINEAR in `c`: the linear form drops a
    /// `c^2 / 2 * mean(traded^2)` term, and on a constant-`|f|` book `traded` is zero on most
    /// bars and twice the stake on a flip, which is the worst case for a second moment. The
    /// reconstruction always understates the cost and so overstates net.
    ///
    /// What survives exactly is the SIGN: `break_even_bps` is the bisection root of this same
    /// ledger's `edge_at`, monotone decreasing in `c`, so `sign(net(c)) == sign(be - c)` holds
    /// however curved the level is. Verdicts are safe; magnitudes need the ledger.
    /// The same levels computed POOLED - per-bar over the whole panel via [`Ledger::edge_at`] -
    /// rather than paired window by window.
    ///
    /// Both are kept because their agreement is a cross-check of two independent accountings,
    /// asserted at every cost level by a test. The paired form carries the CI; the pooled form is
    /// what `break_even_bps` is the root of, so it is the one the sign guarantee attaches to.
    pub net_at_cost_pooled: [f64; HYSTERESIS_NET_COSTS.len()],
    /// Each carries its own CI, because the whole point of the column is whether the row clears
    /// that cost RESOLVABLY: a level of `+0.029` against per-window dispersion of order a bps is
    /// a different claim from the same level with a CI that excludes zero, and quoting the level
    /// alone invites the reader to make that call by eye.
    pub net_at_cost: [Dispersion; HYSTERESIS_NET_COSTS.len()],
    /// The linear reconstruction `edge + (assumed - c) * turnover` at the selection cost, so the
    /// gap against `net_at_cost` is REPORTED rather than argued about.
    pub net_reconstructed_bps: f64,
    /// All-in cost at this row's own participation, in bps. [INFERENCE]
    ///
    /// Impact scales as the square root of participation, so a book trading a fraction of the
    /// incumbent's notional pays a fraction of its impact: the impact component of
    /// [`horizon::MATCHED_ALL_IN_BPS`] is scaled by `sqrt(turnover / incumbent turnover)`. This
    /// rides the UNFITTED `IMPACT_K` and the incumbent's weighting rather than this book's, so
    /// the ORDERING across margins is robust - the coefficient scales every row alike - while
    /// the LEVEL against break-even is not. A robustness check, never the headline.
    pub all_in_cost_bps: f64,
    /// Net growth over the null at [`all_in_cost_bps`](Self::all_in_cost_bps), through the
    /// ledger. [INFERENCE], inheriting that constant's status.
    pub net_all_in_bps: f64,
}

/// The sign-hysteresis frontier: what holding the model's sign longer costs and buys.
///
/// Every lever measured on this panel that produced a resolvable economic gain worked by
/// TRADING LESS, and on a book whose turnover is almost entirely sign flips the only remaining
/// way to trade less is to flip less. This sweeps that directly rather than inferring it.
#[derive(Clone, Debug)]
pub struct HysteresisSweep {
    pub points: Vec<HysteresisPoint>,
    pub matched_leverage: f64,
    pub cost_bps: f64,
    pub leverage_cap: f64,
    pub windows: usize,
    pub blocks: usize,
    /// The null's own traded notional per bar.
    ///
    /// The break-even solve is monotone in the cost, and therefore interpretable as a price the
    /// row has to clear, ONLY where the row outtrades this. Below it the edge curve turns
    /// INCREASING - a dearer world helps a book that trades less than its benchmark - and
    /// `break_even > c` would invert to mean the row wins only ABOVE that cost. Driving turnover
    /// down a wide grid heads straight at that boundary, so the boundary is carried as a number
    /// rather than left as a caveat.
    pub null_turnover: f64,
    /// Which conviction the margins threshold. The two axes have different UNITS, so a reader
    /// who sees only the numbers cannot tell them apart.
    pub axis: ConvictionAxis,
}

impl HysteresisSweep {
    pub fn measured(&self) -> bool {
        !self.points.is_empty()
    }

    /// The margin with the highest NET GROWTH at the measured cost among the rows that still
    /// trade.
    ///
    /// Deliberately not break-even: that is monotone increasing in the margin over this whole
    /// grid, so maximizing it selects the widest knob by construction and runs to a book that
    /// never trades. Net growth at the cost actually charged peaks and then falls, which is the
    /// only ranking that can name an interior optimum.
    pub fn best_net(&self) -> Option<&HysteresisPoint> {
        self.points
            .iter()
            .filter(|point| point.policy.turnover > 0.0 && point.net_at_measured.mean.is_finite())
            .max_by(|left, right| {
                left.net_at_measured
                    .mean
                    .total_cmp(&right.net_at_measured.mean)
            })
    }

    pub fn at_margin(&self, margin_bps: f64) -> Option<&HysteresisPoint> {
        self.points
            .iter()
            .find(|point| point.margin_bps.total_cmp(&margin_bps).is_eq())
    }

    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec!["sign hysteresis: not measured".to_owned()];
        }
        let mut lines = vec![
            format!(
                "sign hysteresis ({} windows / {} blocks, cap {:.1}x, cost {:.2} bps, flat stake \
                 {:.4}): margin 0 IS the sign-only arm",
                self.windows, self.blocks, self.leverage_cap, self.cost_bps, self.matched_leverage,
            ),
            format!(
                "  {:<12}{:>28}{:>12}{:>26}{:>9}{:>10}{:>8}{:>18}",
                "margin bps",
                "edge bps/bar (95% CI)",
                "break-even",
                "NET at measured (95% CI)",
                "sharpe",
                "turnover",
                "hold",
                "net vs incumbent",
            ),
        ];
        for point in &self.points {
            lines.push(format!(
                "  {:<12}{:>+12.4} ({:+.4}..{:+.4}){:>12}{:>+13.4} ({:+.4}..{:+.4}){:>+9.2}\
                 {:>10.4}{:>8.2}{:>+11.4}",
                self.axis.margin_label(point.margin_bps),
                point.edge.mean * 1e4,
                point.edge.ci_low * 1e4,
                point.edge.ci_high * 1e4,
                if point.break_even_bps.is_finite() {
                    format!("{:.2} bps", point.break_even_bps)
                } else {
                    "none".to_owned()
                },
                point.net_at_measured.mean * 1e4,
                point.net_at_measured.ci_low * 1e4,
                point.net_at_measured.ci_high * 1e4,
                point.policy.sharpe,
                point.policy.turnover,
                point.mean_hold_bars,
                point.net_vs_sign_only.mean * 1e4,
            ));
        }
        // Three WEIGHTINGS of one measured impact-free cost, each charged through the ledger's
        // own log accounting rather than reconstructed as `gross - c * turnover`. The linear
        // gap column reports what that distinction is worth per row, so nobody has to take it
        // on argument. The all-in column scales impact by sqrt(participation) and is
        // [INFERENCE] on an UNFITTED coefficient: its ordering across margins is robust because
        // the coefficient scales every row alike, its level against break-even is not.
        lines.push(
            "    NET growth at each measured cost weighting, charged through the ledger:"
                .to_owned(),
        );
        let mut header = format!("      {:<12}", "margin bps");
        for (name, bps) in HYSTERESIS_NET_COSTS {
            header.push_str(&format!("{:>18}", format!("{name} @{bps:.3}")));
        }
        header.push_str(&format!(
            "{:>12}{:>15}{:>13}",
            "all-in bps", "net all-in", "linear gap"
        ));
        lines.push(header);
        for point in &self.points {
            let mut row = format!("      {:<12}", self.axis.margin_label(point.margin_bps));
            for net in &point.net_at_cost {
                row.push_str(&format!("{:>+18.4}", net.mean * 1e4));
            }
            row.push_str(&format!(
                "{:>12.2}{:>+12.4}{:>+12.4}",
                point.all_in_cost_bps,
                point.net_all_in_bps,
                point.net_at_cost[HYSTERESIS_SELECTION_COST].mean * 1e4
                    - point.net_reconstructed_bps,
            ));
            lines.push(row);
        }
        // The break-even column is a price only while the row outtrades the null. Below that a
        // dearer world HELPS the row and the comparison inverts, so the boundary is printed and
        // any row at or under it is named rather than quietly tabulated.
        let inverted: Vec<String> = self
            .points
            .iter()
            .filter(|point| point.policy.turnover <= self.null_turnover)
            .map(|point| self.axis.margin_label(point.margin_bps))
            .collect();
        lines.push(format!(
            "    the null trades {:.6}/bar; break-even reads as a price only above that{}",
            self.null_turnover,
            if inverted.is_empty() {
                ", which every row clears".to_owned()
            } else {
                format!(
                    " - MARGINS {} DO NOT OUTTRADE THE NULL, their break-even has changed \
                     character and is not a cost they clear",
                    inverted.join(", ")
                )
            }
        ));
        // Where each column turns over. Break-even rises monotonically across this whole grid,
        // so it cannot name an interior optimum and only these columns can.
        for (slot, (name, bps)) in HYSTERESIS_NET_COSTS.iter().enumerate() {
            lines.push(self.peak_line(
                &format!("{name} @{bps:.3}"),
                &|point: &HysteresisPoint| point.net_at_cost[slot].mean * 1e4,
            ));
        }
        lines.push(self.peak_line("all-in [INFERENCE]", &|point| point.net_all_in_bps));
        lines.push(self.verdict_line());
        lines
    }

    /// Where a net-growth column turns over, with a peak at the grid's edge named as such.
    ///
    /// A maximum at the widest finite margin is NOT an optimum: it means the grid does not
    /// contain one, and the two have to read differently or a truncated sweep gets published as
    /// a frontier.
    fn peak_line(&self, label: &str, of: &dyn Fn(&HysteresisPoint) -> f64) -> String {
        let best = self
            .points
            .iter()
            .filter(|point| point.policy.turnover > 0.0 && of(point).is_finite())
            .max_by(|left, right| of(left).total_cmp(&of(right)));
        let Some(best) = best else {
            return format!("      {label}: no row trades, UNMEASURED");
        };
        let edge_margin = self
            .points
            .iter()
            .filter(|point| point.margin_bps.is_finite())
            .next_back()
            .map_or(f64::NAN, |point| point.margin_bps);
        format!(
            "      {label}: peaks at margin {} with net {:+.4} bps/bar{}",
            self.axis.margin_label(best.margin_bps),
            of(best),
            if best.margin_bps == edge_margin {
                " - STILL RISING AT THE GRID EDGE, this grid does not contain the optimum"
            } else {
                ""
            }
        )
    }

    pub fn verdict_line(&self) -> String {
        let incumbent = match self.points.first() {
            Some(point) => point,
            None => return "VERDICT: sign hysteresis not measured".to_owned(),
        };
        let gained = self
            .points
            .iter()
            .filter(|point| resolvably_positive(&point.net_vs_sign_only))
            .max_by(|left, right| {
                left.net_at_measured
                    .mean
                    .total_cmp(&right.net_at_measured.mean)
            });
        match gained {
            Some(point) => format!(
                "VERDICT: HOLDING LONGER PAYS. Margin {:.2} bps beats the sign-only incumbent by \
                 {:+.4} bps/bar (95% CI {:+.4}..{:+.4}), lifting break-even {:.2} -> {:.2} bps and \
                 the mean hold {:.2} -> {:.2} bars",
                point.margin_bps,
                point.vs_sign_only.mean * 1e4,
                point.vs_sign_only.ci_low * 1e4,
                point.vs_sign_only.ci_high * 1e4,
                incumbent.break_even_bps,
                point.break_even_bps,
                incumbent.mean_hold_bars,
                point.mean_hold_bars,
            ),
            None => format!(
                "VERDICT: HOLDING LONGER BUYS NOTHING. No flip margin resolvably beats the \
                 sign-only incumbent; the best break-even on the frontier is {:.2} bps against \
                 the incumbent's {:.2}, so suppressing a reversal costs at least the edge that \
                 reversal was going to earn. The panel's flips are spread across conviction, \
                 which is the mechanism.",
                self.best_net()
                    .map(|point| point.break_even_bps)
                    .unwrap_or(f64::NAN),
                incumbent.break_even_bps,
            ),
        }
    }
}

/// The sign-only path with reversals suppressed until the conviction clears `margin`.
///
/// The state machine runs on the SIGN-ONLY target rather than on the raw predicted mean, so
/// margin zero is the sign-only arm exactly - including the bars where the growth gate holds
/// the book flat, which a rule keyed on `sign(mu_hat)` alone would trade through. That identity
/// holds on EITHER conviction axis, because at margin zero the comparison is skipped entirely
/// and an unavailable conviction cannot change the path.
fn hysteresis_paths(
    windows: &[WindowPaths],
    margin: f64,
    axis: ConvictionAxis,
    leverage: f64,
    cap: f64,
) -> Vec<WindowPaths> {
    let threshold = axis.threshold(margin);
    windows
        .iter()
        .map(|window| {
            let actual = &window.positions[POLICY_MODEL];
            let mut state = 0.0f64;
            let path: Vec<f64> = actual
                .iter()
                .enumerate()
                .map(|(bar, fraction)| {
                    let target = if *fraction > 0.0 {
                        1.0
                    } else if *fraction < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let conviction = axis.conviction(window, bar);
                    if target == 0.0 || state == 0.0 || target == state {
                        // Entering, holding, and the growth gate are all margin-independent:
                        // a margin governs REVERSALS and nothing else.
                        state = target;
                    } else if threshold <= 0.0 || conviction > threshold {
                        state = target;
                    }
                    clamp_fraction(leverage * state, cap)
                })
                .collect();
            let mut positions: [Vec<f64>; POLICY_COUNT] = std::array::from_fn(|_| Vec::new());
            positions[POLICY_MODEL] = path;
            WindowPaths::unmeasured(window.realized.clone(), Vec::new(), positions)
        })
        .collect()
}

/// Mean length of a maximal run of constant nonzero position, over every window.
fn mean_hold_bars(windows: &[WindowPaths]) -> f64 {
    let mut positioned = 0usize;
    let mut runs = 0usize;
    for window in windows {
        let mut held = 0.0f64;
        for fraction in &window.positions[POLICY_MODEL] {
            if *fraction != 0.0 {
                positioned += 1;
                if *fraction != held {
                    runs += 1;
                }
            }
            held = *fraction;
        }
    }
    if runs == 0 {
        return f64::NAN;
    }
    positioned as f64 / runs as f64
}

/// Sweep the sign-hysteresis frontier on one conviction axis.
///
/// Refuses when the windows carry no predicted mean, because the margin is a comparison
/// against it and a zero-filled conviction axis would make every margin behave as `never`. The
/// standardized axis additionally needs a predicted SD per bar and refuses without one, rather
/// than silently degrading to a raw-mean threshold under a standardized label.
pub fn hysteresis_sweep(
    windows: &[WindowPaths],
    blocks: &[u64],
    config: BenchConfig,
    axis: ConvictionAxis,
) -> Option<HysteresisSweep> {
    if windows.is_empty() || blocks.len() < windows.len() {
        return None;
    }
    if windows
        .iter()
        .any(|window| window.predicted_mean.len() != window.bars())
    {
        return None;
    }
    if axis == ConvictionAxis::Standardized
        && windows
            .iter()
            .any(|window| window.predicted_var.len() != window.bars())
    {
        return None;
    }
    let BenchConfig {
        cost_bps,
        cap,
        free_marginal,
    } = config;
    let blocks = &blocks[..windows.len()];
    let cost = cost_bps * 1e-4;
    let recapped = recap(windows, cap, free_marginal);
    let matched = Ledger::build(&recapped, POLICY_MODEL, cap)
        .stats(cost)
        .mean_abs_position;

    // The same null every attribution arm is quoted against, so a row of this sweep and a row
    // of that table are the same kind of number.
    let null = Ledger::build(
        &attribution_paths(&recapped, ATTRIBUTION_MARGINAL, matched, free_marginal, cap),
        POLICY_MODEL,
        cap,
    );
    let null_growth = null.window_growth(cost);
    // The cost the book is actually charged in the world, not the bench's assumed 2 bps. The
    // equal-weighted published constant leads because it is the DEAREST of the three measured
    // weightings, so selecting on it never depends on this book's own composite - measured
    // downstream from the turnover this sweep emits - landing at the favourable end.
    let measured = HYSTERESIS_NET_COSTS[HYSTERESIS_SELECTION_COST].1 * 1e-4;
    let null_measured = null.window_growth(measured);
    // Every cost column is paired against the null on the SAME windows, so each carries its own
    // CI rather than a level whose resolvability a reader has to guess at. Precomputed per cost
    // outside the margin loop: the null does not depend on the margin.
    let null_per_cost: Vec<Vec<f64>> = HYSTERESIS_NET_COSTS
        .iter()
        .map(|(_, bps)| null.window_growth(bps * 1e-4))
        .collect();
    let mut sign_only_growth: Vec<f64> = Vec::new();
    let mut sign_only_measured: Vec<f64> = Vec::new();
    // Margin zero is the first row, so the incumbent's participation is in hand before any
    // later row needs it to scale impact.
    let mut incumbent_turnover = f64::NAN;

    let points = axis
        .margins()
        .iter()
        .map(|margin_bps| {
            let paths = hysteresis_paths(&recapped, *margin_bps, axis, matched, cap);
            let ledger = Ledger::build(&paths, POLICY_MODEL, cap);
            let growth = ledger.window_growth(cost);
            let deltas: Vec<f64> = growth
                .iter()
                .zip(&null_growth)
                .map(|(arm, null)| arm - null)
                .collect();
            if sign_only_growth.is_empty() {
                sign_only_growth = growth.clone();
            }
            let against_sign: Vec<f64> = growth
                .iter()
                .zip(&sign_only_growth)
                .map(|(arm, sign)| arm - sign)
                .collect();
            let at_measured = ledger.window_growth(measured);
            let net_deltas: Vec<f64> = at_measured
                .iter()
                .zip(&null_measured)
                .map(|(arm, null)| arm - null)
                .collect();
            if sign_only_measured.is_empty() {
                sign_only_measured = at_measured.clone();
            }
            let net_against_sign: Vec<f64> = at_measured
                .iter()
                .zip(&sign_only_measured)
                .map(|(arm, sign)| arm - sign)
                .collect();
            let edge_at = |bps: f64| {
                let cost = bps * 1e-4;
                ledger.net_growth_per_bar(cost) - null.net_growth_per_bar(cost)
            };
            let policy = ledger.stats(cost);
            if incumbent_turnover.is_nan() {
                incumbent_turnover = policy.turnover;
            }
            let edge = block_bootstrap(&deltas, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
            // Impact is the ONLY component that scales with participation; the fixed component
            // does not, so only the difference is scaled.
            let impact =
                super::horizon::MATCHED_ALL_IN_BPS - super::horizon::MATCHED_MEASURED_BPS;
            let all_in_cost_bps = super::horizon::MATCHED_MEASURED_BPS
                + impact * (policy.turnover / incumbent_turnover).sqrt();
            HysteresisPoint {
                margin_bps: *margin_bps,
                net_at_cost: std::array::from_fn(|slot| {
                    let arm = ledger.window_growth(HYSTERESIS_NET_COSTS[slot].1 * 1e-4);
                    let paired: Vec<f64> = arm
                        .iter()
                        .zip(&null_per_cost[slot])
                        .map(|(arm, null)| arm - null)
                        .collect();
                    block_bootstrap(&paired, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
                }),
                net_at_cost_pooled: std::array::from_fn(|slot| {
                    edge_at(HYSTERESIS_NET_COSTS[slot].1) * 1e4
                }),
                // Deliberately the arithmetic a reader would do from the printed table: the
                // reported edge at the assumed cost, re-charged linearly at the measured one.
                net_reconstructed_bps: edge.mean * 1e4
                    + (cost_bps - HYSTERESIS_NET_COSTS[HYSTERESIS_SELECTION_COST].1)
                        * policy.turnover,
                net_all_in_bps: edge_at(all_in_cost_bps) * 1e4,
                all_in_cost_bps,
                policy,
                edge,
                vs_sign_only: block_bootstrap(
                    &against_sign,
                    blocks,
                    BOOTSTRAP_DRAWS,
                    BOOTSTRAP_SEED,
                ),
                net_at_measured: block_bootstrap(
                    &net_deltas,
                    blocks,
                    BOOTSTRAP_DRAWS,
                    BOOTSTRAP_SEED,
                ),
                net_vs_sign_only: block_bootstrap(
                    &net_against_sign,
                    blocks,
                    BOOTSTRAP_DRAWS,
                    BOOTSTRAP_SEED,
                ),
                turnover: ledger.turnover.clone(),
                break_even_bps: break_even_bps(&edge_at),
                mean_hold_bars: mean_hold_bars(&paths),
            }
        })
        .collect::<Vec<_>>();

    let blocks_used = points.first().map_or(0, |point| point.edge.blocks);
    Some(HysteresisSweep {
        points,
        matched_leverage: matched,
        cost_bps,
        leverage_cap: cap,
        windows: windows.len(),
        blocks: blocks_used,
        null_turnover: null.stats(cost).turnover,
        axis,
    })
}

/// A flip margin CHOSEN on one slice and scored on another.
#[derive(Clone, Debug)]
pub struct HysteresisOos {
    /// Which conviction the margin thresholds, so a table never mixes bps with SDs.
    pub axis: ConvictionAxis,
    /// The margin the fit slice selected, on net growth at the measured cost.
    pub fitted_margin_bps: f64,
    /// What that margin was worth on the FIT slice - the number selection saw.
    pub fit_net_bps: f64,
    /// The same margin's row on the traded slice, which selection never saw.
    pub evaluated: HysteresisPoint,
    /// The traded slice's own in-sample argmax, quoted only to size the selection bias.
    pub in_sample_best_margin_bps: f64,
    pub in_sample_best_net_bps: f64,
    /// The FIT slice's whole frontier.
    ///
    /// Carried so the selection can be REDONE at a cost measured per margin. The cost this
    /// gate selects on is a constant, and the book's true cost is not - it rises with the margin,
    /// because a conviction threshold is a covert liquidity filter. Correcting that requires a
    /// cost per margin measured on the FIT slice's own turnover; pricing the corrected argmax on
    /// the traded slice's turnover would be in-sample selection one level up.
    pub fit_frontier: HysteresisSweep,
    /// The traded slice's whole frontier, so the corrected margin can be read off it ONCE.
    pub traded_frontier: HysteresisSweep,
}

impl HysteresisOos {
    /// True when the out-of-sample gain over the incumbent excludes zero.
    pub fn earned(&self) -> bool {
        resolvably_positive(&self.evaluated.net_vs_sign_only)
    }

    pub fn report_lines(&self) -> Vec<String> {
        let point = &self.evaluated;
        vec![
            format!(
                "  OUT-OF-SAMPLE GATE [{} axis]: margin {} {} fitted on the block-disjoint fit \
                 slice (fit net {:+.4} bps/bar at the selection cost), then scored on the traded \
                 slice it never saw",
                self.axis.name(),
                self.axis.margin_label(self.fitted_margin_bps),
                self.axis.units(),
                self.fit_net_bps,
            ),
            format!(
                "    traded slice at that margin: net {:+.4} bps/bar (95% CI {:+.4}..{:+.4}), \
                 vs the sign-only incumbent {:+.4} ({:+.4}..{:+.4}), break-even {:.2} bps, \
                 turnover {:.4}, hold {:.2} bars",
                point.net_at_measured.mean * 1e4,
                point.net_at_measured.ci_low * 1e4,
                point.net_at_measured.ci_high * 1e4,
                point.net_vs_sign_only.mean * 1e4,
                point.net_vs_sign_only.ci_low * 1e4,
                point.net_vs_sign_only.ci_high * 1e4,
                point.break_even_bps,
                point.policy.turnover,
                point.mean_hold_bars,
            ),
            format!(
                "    same row, the rest of it: edge at the assumed cost {:+.4} bps/bar (95% CI \
                 {:+.4}..{:+.4}), sharpe {:+.2}, hit {:.4}, all-in cost {:.2} bps -> net {:+.4} \
                 [INFERENCE]",
                point.edge.mean * 1e4,
                point.edge.ci_low * 1e4,
                point.edge.ci_high * 1e4,
                point.policy.sharpe,
                point.policy.hit_rate,
                point.all_in_cost_bps,
                point.net_all_in_bps,
            ),
            // Every cost level with its own CI, because the fitted-book anchors are where this
            // row is decided and a level near zero there is a different claim from a level near
            // zero with a CI that excludes it.
            format!(
                "    net at each measured cost, with CI: {}",
                HYSTERESIS_NET_COSTS
                    .iter()
                    .enumerate()
                    .map(|(slot, (name, bps))| {
                        let net = &point.net_at_cost[slot];
                        format!(
                            "{name} @{bps:.3} {:+.4} ({:+.4}..{:+.4}){}",
                            net.mean * 1e4,
                            net.ci_low * 1e4,
                            net.ci_high * 1e4,
                            if resolvably_positive(net) {
                                " RESOLVABLE"
                            } else {
                                " straddles zero"
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
            format!(
                "    selection bias: the traded slice's OWN argmax is margin {:.2} at {:+.4} \
                 bps/bar, so choosing out of sample cost {:+.4}",
                self.in_sample_best_margin_bps,
                self.in_sample_best_net_bps,
                point.net_at_measured.mean * 1e4 - self.in_sample_best_net_bps,
            ),
            format!(
                "    VERDICT: {}",
                if self.earned() {
                    "EARNED - the gain over the incumbent survives out-of-sample selection"
                } else {
                    "NOT EARNED - the in-sample frontier does not survive honest selection"
                }
            ),
        ]
    }
}

/// Fit the flip margin on one slice and score it on another.
///
/// The whole frontier was swept on the panel it is measured on, so its argmax is a knob chosen
/// knowing the answer. This is the same discipline the mean recalibration already uses: fit on
/// a block-disjoint slice, evaluate on the traded prefix, and report the paired gain there.
pub fn hysteresis_out_of_sample(
    fit: &[WindowPaths],
    fit_blocks: &[u64],
    traded: &[WindowPaths],
    traded_blocks: &[u64],
    config: BenchConfig,
    axis: ConvictionAxis,
) -> Option<HysteresisOos> {
    let fit_frontier = hysteresis_sweep(fit, fit_blocks, config, axis)?;
    let chosen = fit_frontier.best_net()?;
    let fitted_margin_bps = chosen.margin_bps;
    let fit_net_bps = chosen.net_at_measured.mean * 1e4;
    let traded_frontier = hysteresis_sweep(traded, traded_blocks, config, axis)?;
    let evaluated = traded_frontier.at_margin(fitted_margin_bps)?.clone();
    let in_sample = traded_frontier.best_net()?;
    Some(HysteresisOos {
        axis,
        fitted_margin_bps,
        fit_net_bps,
        evaluated,
        in_sample_best_margin_bps: in_sample.margin_bps,
        in_sample_best_net_bps: in_sample.net_at_measured.mean * 1e4,
        fit_frontier,
        traded_frontier,
    })
}

pub const COMPOSITION_INCUMBENT: usize = 0;
pub const COMPOSITION_HYSTERESIS: usize = 1;
pub const COMPOSITION_SHRINK: usize = 2;
pub const COMPOSITION_BOTH: usize = 3;
pub const COMPOSITION_CELLS: usize = 4;

pub const COMPOSITION_NAMES: [&str; COMPOSITION_CELLS] = [
    "incumbent",
    "hysteresis only",
    "shrink only",
    "shrink + hysteresis",
];

/// One cell of the 2x2, on the same windows, the same blocks and the same null as the other
/// three.
#[derive(Clone, Debug)]
pub struct CompositionCell {
    pub policy: PolicyStats,
    /// Net growth over the null at the selection cost, paired window by window.
    pub net: Dispersion,
    pub break_even_bps: f64,
    pub mean_hold_bars: f64,
}

/// The recalibration shrink crossed with sign hysteresis, paired on identical blocks.
///
/// Both levers cut the cost of the same book and neither is a rescaling of the other, so their
/// gains cannot be added without measuring the second difference. There is a mechanism that
/// says they must fight - hysteresis works while the leverage cap stays BOUND, which is what
/// makes every trade a sign flip, whereas the shrink works by sizing smaller and UNBINDING the
/// cap, switching the expensive magnitude channel back on - and there is a mechanism that says
/// they must compose, since one acts on the sign path and the other on the mean. Both are
/// arguments. [`interaction`](Self::interaction) is the measurement, and it is reported whatever
/// it says.
#[derive(Clone, Debug)]
pub struct HysteresisComposition {
    /// The margin both hysteresis cells use. Fitted OUT OF SAMPLE and passed in, never swept
    /// here: a 2x2 that re-picked the margin on these windows would be comparing an argmax
    /// against three fixed policies.
    pub margin_bps: f64,
    pub cells: [CompositionCell; COMPOSITION_CELLS],
    /// `hysteresis only - incumbent`.
    pub hysteresis_effect: Dispersion,
    /// `shrink only - incumbent`.
    pub shrink_effect: Dispersion,
    /// The second difference: `(both - shrink) - (hysteresis - incumbent)`. Zero is additivity,
    /// negative is antagonism, positive is complementarity.
    pub interaction: Dispersion,
    /// `both - hysteresis only`: what adding the shrink to the stronger single lever is worth.
    /// The decision-relevant number, because shipping both is a choice against shipping one.
    pub both_vs_hysteresis: Dispersion,
    /// The cost the net columns are charged at, in bps.
    pub cost_bps: f64,
}

impl HysteresisComposition {
    /// True when the second difference resolvably excludes additivity in either direction.
    pub fn resolvable_interaction(&self) -> bool {
        self.interaction.ci_low.is_finite()
            && (self.interaction.ci_low > 0.0 || self.interaction.ci_high < 0.0)
    }

    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "  shrink x hysteresis at margin {} bps, paired on identical blocks, net at \
                 {:.3} bps:",
                margin_label(self.margin_bps),
                self.cost_bps,
            ),
            format!(
                "    {:<22}{:>26}{:>12}{:>10}{:>8}",
                "cell", "NET bps/bar (95% CI)", "break-even", "turnover", "hold",
            ),
        ];
        for (name, cell) in COMPOSITION_NAMES.iter().zip(&self.cells) {
            lines.push(format!(
                "    {:<22}{:>+13.4} ({:+.4}..{:+.4}){:>12}{:>10.4}{:>8.2}",
                name,
                cell.net.mean * 1e4,
                cell.net.ci_low * 1e4,
                cell.net.ci_high * 1e4,
                if cell.break_even_bps.is_finite() {
                    format!("{:.2} bps", cell.break_even_bps)
                } else {
                    "none".to_owned()
                },
                cell.policy.turnover,
                cell.mean_hold_bars,
            ));
        }
        for (label, paired) in [
            ("hysteresis alone", &self.hysteresis_effect),
            ("shrink alone", &self.shrink_effect),
            ("INTERACTION", &self.interaction),
            ("both vs hysteresis alone", &self.both_vs_hysteresis),
        ] {
            lines.push(format!(
                "      {:<26}{:>+10.4} bps/bar (95% CI {:+.4}..{:+.4})",
                label,
                paired.mean * 1e4,
                paired.ci_low * 1e4,
                paired.ci_high * 1e4,
            ));
        }
        lines.push(format!("    VERDICT: {}", self.verdict()));
        lines
    }

    fn verdict(&self) -> String {
        let sum = (self.hysteresis_effect.mean + self.shrink_effect.mean) * 1e4;
        let both = (self.cells[COMPOSITION_BOTH].net.mean
            - self.cells[COMPOSITION_INCUMBENT].net.mean)
            * 1e4;
        if !self.resolvable_interaction() {
            return format!(
                "ADDITIVITY NOT RESOLVED. The two levers sum to {sum:+.4} bps/bar and together \
                 deliver {both:+.4}, but the second difference's interval straddles zero, so \
                 this panel cannot say whether they compose or fight. The gains still must NOT \
                 be added - an unresolved interaction is not a zero one."
            );
        }
        if self.interaction.ci_high < 0.0 {
            format!(
                "ANTAGONISTIC, RESOLVABLY. Adding the shrink to hysteresis is worth {:+.4} \
                 bps/bar (95% CI {:+.4}..{:+.4}); the levers sum to {sum:+.4} but together \
                 deliver only {both:+.4}. The two gains cannot be added.",
                self.both_vs_hysteresis.mean * 1e4,
                self.both_vs_hysteresis.ci_low * 1e4,
                self.both_vs_hysteresis.ci_high * 1e4,
            )
        } else {
            format!(
                "COMPLEMENTARY, RESOLVABLY. Together the levers deliver {both:+.4} bps/bar \
                 against a sum of {sum:+.4}, so the composition beats additivity."
            )
        }
    }
}

/// Cross the recalibration shrink with sign hysteresis at ONE margin, paired per window.
///
/// `None` unless every window carries a recalibrated fraction, which is the normal case for any
/// bench whose shrink was not fitted on a disjoint slice first.
pub fn hysteresis_composition(
    windows: &[WindowPaths],
    blocks: &[u64],
    config: BenchConfig,
    margin_bps: f64,
    axis: ConvictionAxis,
) -> Option<HysteresisComposition> {
    if windows.is_empty() || blocks.len() < windows.len() {
        return None;
    }
    if windows
        .iter()
        .any(|window| window.predicted_mean.len() != window.bars())
    {
        return None;
    }
    let BenchConfig {
        cost_bps,
        cap,
        free_marginal,
    } = config;
    let blocks = &blocks[..windows.len()];
    let cost = cost_bps * 1e-4;
    let cost_at = HYSTERESIS_NET_COSTS[HYSTERESIS_SELECTION_COST].1;
    let measured = cost_at * 1e-4;

    let plain = recap(windows, cap, free_marginal);
    let shrunk = recap(&promote_shrunk(windows)?, cap, free_marginal);
    // Each hysteresis cell holds `|f|` at ITS OWN book's mean, so the constant-stake
    // construction is stake-matched to the book it derives from. Matching both cells to the
    // unshrunk stake would smuggle a leverage change into the shrink's row.
    let stake = |paths: &[WindowPaths]| {
        Ledger::build(paths, POLICY_MODEL, cap)
            .stats(cost)
            .mean_abs_position
    };
    let plain_stake = stake(&plain);
    let cell_paths = [
        plain.clone(),
        hysteresis_paths(&plain, margin_bps, axis, plain_stake, cap),
        shrunk.clone(),
        hysteresis_paths(&shrunk, margin_bps, axis, stake(&shrunk), cap),
    ];
    // ONE null for all four cells - the same marginal null every other table on this panel is
    // quoted against - so all three differences cancel it exactly.
    let null = Ledger::build(
        &attribution_paths(
            &plain,
            ATTRIBUTION_MARGINAL,
            plain_stake,
            free_marginal,
            cap,
        ),
        POLICY_MODEL,
        cap,
    );
    let null_measured = null.window_growth(measured);

    let mut nets: Vec<Vec<f64>> = Vec::with_capacity(COMPOSITION_CELLS);
    let mut cells: Vec<CompositionCell> = Vec::with_capacity(COMPOSITION_CELLS);
    for paths in &cell_paths {
        let ledger = Ledger::build(paths, POLICY_MODEL, cap);
        let net: Vec<f64> = ledger
            .window_growth(measured)
            .iter()
            .zip(&null_measured)
            .map(|(arm, null)| arm - null)
            .collect();
        let edge_at = |bps: f64| {
            let cost = bps * 1e-4;
            ledger.net_growth_per_bar(cost) - null.net_growth_per_bar(cost)
        };
        cells.push(CompositionCell {
            policy: ledger.stats(cost),
            net: block_bootstrap(&net, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
            break_even_bps: break_even_bps(&edge_at),
            mean_hold_bars: mean_hold_bars(paths),
        });
        nets.push(net);
    }

    let paired = |left: usize, right: usize| {
        let deltas: Vec<f64> = nets[left]
            .iter()
            .zip(&nets[right])
            .map(|(left, right)| left - right)
            .collect();
        block_bootstrap(&deltas, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    };
    let interaction: Vec<f64> = (0..windows.len())
        .map(|window| {
            (nets[COMPOSITION_BOTH][window] - nets[COMPOSITION_SHRINK][window])
                - (nets[COMPOSITION_HYSTERESIS][window] - nets[COMPOSITION_INCUMBENT][window])
        })
        .collect();

    Some(HysteresisComposition {
        margin_bps,
        cells: cells
            .try_into()
            .expect("one cell was pushed for each of COMPOSITION_CELLS"),
        hysteresis_effect: paired(COMPOSITION_HYSTERESIS, COMPOSITION_INCUMBENT),
        shrink_effect: paired(COMPOSITION_SHRINK, COMPOSITION_INCUMBENT),
        interaction: block_bootstrap(&interaction, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        both_vs_hysteresis: paired(COMPOSITION_BOTH, COMPOSITION_HYSTERESIS),
        cost_bps: cost_at,
    })
}

/// Horizons the signal's directional content is measured at, in bars.
pub const DECAY_HORIZONS: [usize; 5] = [1, 2, 6, 12, 39];

/// How fast the CURRENT signal's directional content decays with holding horizon.
///
/// Every lever this panel has tested operates a signal FITTED to the one-bar target at some
/// longer horizon, and a decaying signal operated past its horizon loses by construction. This
/// separates the two: it measures the signal itself against k-bar-ahead returns, with no policy
/// and no cost, so a failure of the levers can be told apart from a failure of the signal.
/// It does NOT speak to a model TRAINED on a k-bar target, whose predictable component and
/// noise floor are different quantities.
#[derive(Clone, Copy, Debug)]
pub struct DecayPoint {
    pub horizon: usize,
    /// Share of bars where `sign(mu_hat_t)` matches the sign of the k-bar forward log return.
    pub hit_rate: Dispersion,
    /// `sign(mu_hat_t) * forward_k / k`, in log-return units PER BAR so horizons compare.
    pub edge_per_bar: Dispersion,
    /// Pooled Pearson correlation of `mu_hat_t` against the k-bar forward log return.
    pub correlation: f64,
    pub samples: usize,
}

#[derive(Clone, Debug)]
pub struct SignalDecay {
    pub points: Vec<DecayPoint>,
    pub blocks: usize,
}

impl SignalDecay {
    pub fn measured(&self) -> bool {
        !self.points.is_empty()
    }

    pub fn report_lines(&self) -> Vec<String> {
        if !self.measured() {
            return vec!["signal decay: not measured".to_owned()];
        }
        let mut lines = vec![
            "signal decay: the CURRENT one-bar signal scored against k-bar-ahead returns, no \
             policy and no cost - this bounds a one-bar signal HELD longer, never a k-bar model"
                .to_owned(),
            format!(
                "  {:<8}{:>28}{:>30}{:>12}{:>12}",
                "k bars", "hit rate (95% CI)", "edge bps/bar (95% CI)", "corr", "samples",
            ),
        ];
        for point in &self.points {
            lines.push(format!(
                "  {:<8}{:>12.4} ({:.4}..{:.4}){:>+14.4} ({:+.4}..{:+.4}){:>+12.5}{:>12}",
                point.horizon,
                point.hit_rate.mean,
                point.hit_rate.ci_low,
                point.hit_rate.ci_high,
                point.edge_per_bar.mean * 1e4,
                point.edge_per_bar.ci_low * 1e4,
                point.edge_per_bar.ci_high * 1e4,
                point.correlation,
                point.samples,
            ));
        }
        lines
    }
}

/// Measure [`SignalDecay`] over the traded windows.
pub fn signal_decay(windows: &[WindowPaths], blocks: &[u64]) -> SignalDecay {
    if windows.is_empty()
        || blocks.len() < windows.len()
        || windows
            .iter()
            .any(|window| window.predicted_mean.len() != window.bars())
    {
        return SignalDecay {
            points: Vec::new(),
            blocks: 0,
        };
    }
    let blocks = &blocks[..windows.len()];
    // `ln(1 + R)` per bar, prefix-summed per window, so a k-bar forward return is one
    // subtraction rather than a k-long inner loop at every bar and every horizon.
    let prefixes: Vec<Vec<f64>> = windows
        .iter()
        .map(|window| {
            let mut prefix = Vec::with_capacity(window.realized.len() + 1);
            let mut total = 0.0f64;
            prefix.push(0.0);
            for realized in &window.realized {
                total += (1.0 + realized).max(WEALTH_FLOOR).ln();
                prefix.push(total);
            }
            prefix
        })
        .collect();

    let points = DECAY_HORIZONS
        .iter()
        .map(|horizon| {
            let k = *horizon;
            let mut hit_by_window = Vec::with_capacity(windows.len());
            let mut edge_by_window = Vec::with_capacity(windows.len());
            let (mut sum_x, mut sum_y, mut sum_xx, mut sum_yy, mut sum_xy) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut samples = 0usize;
            for (window, prefix) in windows.iter().zip(&prefixes) {
                let bars = window.bars();
                let (mut hits, mut edge, mut count) = (0.0f64, 0.0f64, 0usize);
                for bar in 0..bars.saturating_sub(k - 1) {
                    let mu = window.predicted_mean[bar];
                    if mu == 0.0 || !mu.is_finite() {
                        continue;
                    }
                    let forward = prefix[bar + k] - prefix[bar];
                    if !forward.is_finite() {
                        continue;
                    }
                    let side = if mu > 0.0 { 1.0 } else { -1.0 };
                    if side * forward > 0.0 {
                        hits += 1.0;
                    }
                    edge += side * forward / k as f64;
                    count += 1;
                    sum_x += mu;
                    sum_y += forward;
                    sum_xx += mu * mu;
                    sum_yy += forward * forward;
                    sum_xy += mu * forward;
                }
                samples += count;
                let denominator = count.max(1) as f64;
                hit_by_window.push(hits / denominator);
                edge_by_window.push(edge / denominator);
            }
            let n = samples as f64;
            let covariance = sum_xy - sum_x * sum_y / n;
            let spread = ((sum_xx - sum_x * sum_x / n) * (sum_yy - sum_y * sum_y / n)).sqrt();
            DecayPoint {
                horizon: k,
                hit_rate: block_bootstrap(&hit_by_window, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
                edge_per_bar: block_bootstrap(
                    &edge_by_window,
                    blocks,
                    BOOTSTRAP_DRAWS,
                    BOOTSTRAP_SEED,
                ),
                correlation: if spread > 0.0 {
                    covariance / spread
                } else {
                    f64::NAN
                },
                samples,
            }
        })
        .collect::<Vec<_>>();

    let blocks_used = points.first().map_or(0, |point| point.hit_rate.blocks);
    SignalDecay {
        points,
        blocks: blocks_used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::{encode_dof, BarDof, VolumeEma, DOF_S};
    use crate::torch::dataset::mix64;
    use crate::torch::test_rng;
    use tch::nn;

    /// Counter-based uniforms, so every fixture is reproducible from its seed alone.
    fn uniform(seed: u64, index: u64) -> f64 {
        (mix64(seed, index) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn synthetic_supports(count: usize, seed: u64) -> BarSupports {
        let samples: Vec<BarDof> = (0..count)
            .map(|i| {
                let i = i as u64;
                let u1 = uniform(seed, 3 * i).max(1e-12);
                let u2 = uniform(seed, 3 * i + 1);
                let gauss = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let r = 0.002 * gauss;
                let s = (0.003 * uniform(seed, 3 * i + 2)).max(0.0);
                BarDof {
                    r: r as f32,
                    s: s as f32,
                    u: uniform(seed, 3 * i + 2) as f32,
                    v: uniform(seed, 3 * i) as f32,
                    w: (0.5 * gauss) as f32,
                }
            })
            .collect();
        BarSupports::fit(&samples)
    }

    /// A head whose weights are not all zero, so the prefix table and the chain
    /// conditioning actually do something.
    fn perturbed_head(latent: i64, seed: u64) -> (nn::VarStore, BarEmissionHead) {
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), latent);
        tch::no_grad(|| {
            for (index, mut tensor) in vs.trainable_variables().into_iter().enumerate() {
                let numel = tensor.numel() as u64;
                let values: Vec<f32> = (0..numel)
                    .map(|slot| {
                        (2.0 * uniform(mix64(seed, index as u64), slot) - 1.0) as f32 * 0.4
                    })
                    .collect();
                let replacement = Tensor::from_slice(&values).reshape(tensor.size());
                tensor.copy_(&replacement);
            }
        });
        (vs, head)
    }

    fn beliefs(rows: i64, latent: i64, seed: u64) -> Tensor {
        let values: Vec<f32> = (0..rows * latent)
            .map(|slot| (2.0 * uniform(seed, slot as u64) - 1.0) as f32)
            .collect();
        Tensor::from_slice(&values).view([rows, latent])
    }

    // -----------------------------------------------------------------------
    // The solver
    // -----------------------------------------------------------------------

    #[test]
    fn solver_recovers_the_analytic_kelly_fraction() {
        // A `b`-to-1 bet won with probability `p` has the textbook optimum
        // `f = p - (1 - p) / b`.
        for (p, b, expected) in [(0.6, 1.0, 0.2), (0.55, 2.0, 0.325), (0.51, 1.0, 0.02)] {
            let probs = [p, 1.0 - p];
            let returns = [b, -1.0];
            let solved = kelly_fraction(&probs, &returns, LEVERAGE_CAP);
            assert!(
                (solved - expected).abs() < 1e-9,
                "kelly on a {b}:1 bet at p={p} solved to {solved}, analytic {expected}"
            );
            // And it really is the maximizer: perturbing it either way loses growth.
            let best = expected_log_growth(&probs, &returns, solved);
            for step in [-1e-4, 1e-4] {
                assert!(
                    expected_log_growth(&probs, &returns, solved + step) < best,
                    "{solved} is not a local maximum of the expected log growth"
                );
            }
        }
        // Two-sided asymmetric bet with a closed form: 0.5/(1 + 0.5f) = 0.4/(1 - 0.4f)
        // solves at f = 0.25.
        let solved = kelly_fraction(&[0.5, 0.5], &[0.5, -0.4], LEVERAGE_CAP);
        assert!(
            (solved - 0.25).abs() < 1e-9,
            "asymmetric two-point kelly solved to {solved}, analytic 0.25"
        );
    }

    #[test]
    fn a_symmetric_zero_edge_law_takes_exactly_no_position() {
        for magnitude in [0.002, 0.02, 0.2] {
            let solved = kelly_fraction(&[0.5, 0.5], &[magnitude, -magnitude], LEVERAGE_CAP);
            assert_eq!(
                solved, 0.0,
                "a symmetric +/-{magnitude} bet must take exactly zero position, got {solved}"
            );
        }
        // Symmetric in SIMPLE returns is the zero-edge condition; a law symmetric in log
        // space has a positive expected simple return and must NOT be flattened.
        let log_symmetric = [(0.01f64).exp_m1(), (-0.01f64).exp_m1()];
        assert!(
            kelly_fraction(&[0.5, 0.5], &log_symmetric, LEVERAGE_CAP) > 0.0,
            "a log-symmetric law has a positive expected return and is tradeable"
        );
    }

    #[test]
    fn the_solver_respects_the_cap_and_the_ruin_boundary() {
        // No loss mass at all: growth is unbounded in `f`, so the cap must bind.
        let capped = kelly_fraction(&[0.5, 0.5], &[0.01, 0.02], LEVERAGE_CAP);
        assert!(
            (capped - LEVERAGE_CAP).abs() < 1e-9,
            "a law with no loss mass must saturate the cap, got {capped}"
        );
        // A 50% down bin makes any position at or above 2x ruinous, so the feasible
        // bound binds strictly inside the cap.
        let bounded = kelly_fraction(&[0.999, 0.001], &[0.01, -0.5], 10.0);
        assert!(
            bounded > 0.0 && bounded < 2.0,
            "the feasible bracket must keep 1 + f R positive, got {bounded}"
        );
        // Mirror image on the short side.
        let short = kelly_fraction(&[0.5, 0.5], &[-0.01, -0.02], LEVERAGE_CAP);
        assert!(
            (short + LEVERAGE_CAP).abs() < 1e-9,
            "a law with no gain mass must saturate the SHORT cap, got {short}"
        );
    }

    #[test]
    fn the_solver_is_row_wise_and_matches_the_scalar_path() {
        let returns = [0.02, -0.01, 0.0];
        let rows = [[0.4, 0.4, 0.2], [0.2, 0.6, 0.2], [1.0 / 3.0; 3]];
        let probs = Tensor::from_slice(&rows.concat()).view([3, 3]);
        let batched = host_vec(&kelly_fractions(
            &probs,
            &Tensor::from_slice(&returns).view([1, 3]),
            LEVERAGE_CAP,
        ));
        for (row, expected) in rows.iter().zip(&batched) {
            let scalar = kelly_fraction(row, &returns, LEVERAGE_CAP);
            assert_eq!(
                scalar, *expected,
                "the batched solver must agree bit for bit with the scalar path"
            );
        }
    }

    // -----------------------------------------------------------------------
    // The predictive object
    // -----------------------------------------------------------------------

    /// The traded `r` law must be the head's OWN prefix-free row: normalized, invariant to
    /// whatever prefix the head is handed, and the same object the head's own forecast
    /// reports.
    ///
    /// `r` is [`BAR_CHAIN`]`[0]`, so [`forecast_r_probs`] reads that row directly. The
    /// failure this guards is reading a row that is NOT prefix-free — a teacher-forced row
    /// of a later factor, or a first factor that a reorder has handed a prefix.
    #[test]
    fn the_traded_r_law_is_the_heads_prefix_free_row() {
        let _torch_rng_guard = test_rng::shared();
        let latent = 24;
        let (_vs, head) = perturbed_head(latent, 0x7EA5_0001);
        let h = beliefs(7, latent, 0x7EA5_0002);
        let probs = forecast_r_probs(&head, &h);
        let rows = h.size()[0];

        let mass = host_vec(&probs.sum_dim_intlist([-1i64].as_slice(), false, Kind::Double));
        for total in mass {
            assert!(
                (total - 1.0).abs() < 1e-5,
                "the traded law must integrate to 1, got {total}"
            );
        }

        // No prefix assignment can move it. BIT-identical, not close: a tolerance would
        // pass a read that had picked up a teacher-forced row.
        for bin in [0i64, 1, 37, 64, NUM_BAR_BINS - 1] {
            let prefix = Tensor::full([rows, BAR_DOF as i64], bin, (Kind::Int64, Device::Cpu));
            let row = head
                .logits(&h, &prefix)
                .select(1, DOF_R as i64)
                .softmax(-1, Kind::Float);
            assert_eq!(
                (&row - &probs).abs().max().double_value(&[]),
                0.0,
                "the r row moved when every prefix slot was set to bin {bin}, so it is not \
                 the prefix-free row this module trades"
            );
        }

        // And it is the head's own forecast row, which is a separate implementation: an
        // ancestral-draw mixture over the chain, whose first factor is drawn from no prefix.
        let forecast = head
            .forecast_log_probs(&h, 4, 0x7EA5_0003)
            .select(1, DOF_R as i64)
            .exp();
        let error = (&probs - &forecast).abs().max().double_value(&[]);
        assert!(
            error < 1e-6,
            "the traded law differs from the head's own forecast row by {error:.3e}, which \
             on a prefix-free factor can only be the log/exp round trip"
        );

        // Non-vacuity: the head's prefix pathway is LIVE, so the invariance above is the
        // chain position doing its job rather than a dead prefix embedding. A factor that
        // does have a prefix must move when that prefix does.
        let zero = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let filled = Tensor::full([rows, BAR_DOF as i64], 64i64, (Kind::Int64, Device::Cpu));
        let response = (head.logits(&h, &zero).select(1, DOF_S as i64)
            - head.logits(&h, &filled).select(1, DOF_S as i64))
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            response > 1e-3,
            "the fixture head has no prefix response at all ({response:.3e} logits), so the \
             invariance of the r row proves nothing"
        );
    }

    // -----------------------------------------------------------------------
    // Lookahead
    // -----------------------------------------------------------------------

    #[test]
    fn the_traded_decision_cannot_reach_the_same_bar_dof() {
        let _torch_rng_guard = test_rng::shared();
        let latent = 20;
        let (_vs, head) = perturbed_head(latent, 0xA001);
        let supports = synthetic_supports(30_000, 0xA002);
        let returns = Tensor::from_slice(&bin_returns(&supports)).view([1, NUM_BAR_BINS]);
        let centers = Tensor::from_slice(supports.centers(DOF_R)).view([1, NUM_BAR_BINS]);
        let free_null = marginal_position(&supports, FREE_LEVERAGE);
        let (windows, bars) = (3i64, 16i64);
        let h = beliefs(windows * bars, latent, 0xA003).view([windows, bars, latent]);

        // A monotone realized path, so reversing it provably reverses the sign pattern
        // the oracle keys on rather than relying on a random draw to differ.
        let realized = Tensor::from_slice(
            &(0..windows * bars)
                .map(|slot| (0.001 * (slot % bars - bars / 2) as f64) as f32)
                .collect::<Vec<f32>>(),
        )
        .view([windows, bars]);
        let shuffled = realized.flip([1i64]);

        let paths = |realized: &Tensor| {
            window_paths(
                &head,
                &h,
                realized,
                &TradedLaw::new(&returns, &centers),
                free_null,
                LEVERAGE_CAP,
            )
            .expect("paths")
        };
        let honest = paths(&realized);
        let permuted = paths(&shuffled);
        for (a, b) in honest.windows.iter().zip(&permuted.windows) {
            assert_eq!(
                a.free, b.free,
                "the uncapped optimum moved when only the realized bar changed, which is \
                 lookahead at the source"
            );
            for policy in 0..POLICY_COUNT {
                if policy == POLICY_ORACLE {
                    assert_ne!(
                        a.positions[policy], b.positions[policy],
                        "the oracle is the one policy that must see the realized bar"
                    );
                } else {
                    assert_eq!(
                        a.positions[policy], b.positions[policy],
                        "{} moved when only the realized bar changed, which is lookahead",
                        POLICY_NAMES[policy]
                    );
                }
            }
        }
        assert_eq!(
            honest.tail.windows(),
            0,
            "without support bounds there is no tail block to count"
        );
        // The positive statement: the traded path IS the Kelly solve of a distribution
        // whose own signature admits no realized bar, and every non-oracle policy is a
        // clamp of that one solve rather than a second decision.
        let flat = h.reshape([windows * bars, latent]);
        let independent = host_vec(&kelly_fractions(
            &forecast_r_probs(&head, &flat),
            &returns,
            FREE_LEVERAGE,
        ));
        for (window, paths) in honest.windows.iter().enumerate() {
            for (bar, free) in paths.free.iter().enumerate() {
                let expected = independent[window * bars as usize + bar];
                assert_eq!(*free, expected);
                for policy in 0..POLICY_COUNT {
                    let multiple = POLICY_KELLY_MULTIPLE[policy];
                    if multiple.is_finite() {
                        assert_eq!(
                            paths.positions[policy][bar],
                            clamp_fraction(multiple * expected, LEVERAGE_CAP),
                            "{} must be a clamp of the one solve",
                            POLICY_NAMES[policy]
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Baselines
    // -----------------------------------------------------------------------

    #[test]
    fn the_marginal_null_is_reproducible_and_weight_independent() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(40_000, 0xB001);
        let first = marginal_position(&supports, LEVERAGE_CAP);
        let second = marginal_position(&supports, LEVERAGE_CAP);
        assert_eq!(first, second, "the null must be bit-reproducible");
        assert!(
            first.is_finite() && first != 0.0,
            "a drifting equity law has a nonzero log-optimal position, got {first}"
        );
        // Solving under the cap and clamping the uncapped solve are the same policy, because
        // `g` is concave: this is the identity the whole cap curve rests on, and it is
        // checked here rather than assumed. To SOLVER tolerance, not bitwise — the two
        // bisections start from different brackets, so their last bits differ by design.
        let free = marginal_position(&supports, FREE_LEVERAGE);
        let clamped = clamp_fraction(free, LEVERAGE_CAP);
        assert!(
            (clamped - first).abs() < 1e-6,
            "clamping the uncapped null ({free} -> {clamped}) disagrees with solving it under \
             the cap ({first}), so the cap curve is not re-clamping the same policy"
        );

        // Two entirely different heads, same null.
        let latent = 12;
        let (_a, head_a) = perturbed_head(latent, 0xB002);
        let (_b, head_b) = perturbed_head(latent, 0xB003);
        let returns = Tensor::from_slice(&bin_returns(&supports)).view([1, NUM_BAR_BINS]);
        let centers = Tensor::from_slice(supports.centers(DOF_R)).view([1, NUM_BAR_BINS]);
        let h = beliefs(8, latent, 0xB004).view([2, 4, latent]);
        let realized = Tensor::zeros([2, 4], (Kind::Float, Device::Cpu));
        let paths_of = |head: &BarEmissionHead| {
            window_paths(
                head,
                &h,
                &realized,
                &TradedLaw::new(&returns, &centers),
                marginal_position(&supports, FREE_LEVERAGE),
                LEVERAGE_CAP,
            )
            .expect("paths")
        };
        let paths_a = paths_of(&head_a);
        let paths_b = paths_of(&head_b);
        for (a, b) in paths_a.windows.iter().zip(&paths_b.windows) {
            assert_eq!(
                a.positions[POLICY_MARGINAL], b.positions[POLICY_MARGINAL],
                "the null moved with the model weights"
            );
            assert!(
                a.positions[POLICY_MARGINAL].iter().all(|f| *f == clamped),
                "the null is a constant position, by construction"
            );
        }
        // Two different heads must produce different predictive laws, or this test proves
        // nothing. Compared on the law rather than the position, because a saturated cap
        // would hide a genuine disagreement behind two identical clamps.
        let law_a = forecast_r_probs(&head_a, &h.reshape([8, latent]));
        let law_b = forecast_r_probs(&head_b, &h.reshape([8, latent]));
        assert!(
            (&law_a - &law_b).abs().max().double_value(&[]) > 1e-4,
            "two independently perturbed heads must disagree about p(r|past)"
        );
    }

    /// Windows with a known realized path and an arbitrary uncapped optimum, for the
    /// accounting invariants.
    ///
    /// The `free` path deliberately exceeds the cap on some bars, so `clamped_fraction`
    /// and the cap curve have something to measure, and every policy is derived from it by
    /// the same clamp the production path uses.
    fn fixture_windows(count: usize, bars: usize, seed: u64) -> Vec<WindowPaths> {
        (0..count)
            .map(|window| {
                let realized: Vec<f64> = (0..bars)
                    .map(|bar| {
                        0.006 * (2.0 * uniform(seed, (window * bars + bar) as u64) - 1.0)
                    })
                    .collect();
                let free: Vec<f64> = (0..bars)
                    .map(|bar| {
                        1.6 * LEVERAGE_CAP
                            * (2.0 * uniform(mix64(seed, 1), (window * bars + bar) as u64) - 1.0)
                    })
                    .collect();
                let positions = std::array::from_fn(|policy| {
                    let multiple = POLICY_KELLY_MULTIPLE[policy];
                    if multiple.is_finite() {
                        free.iter()
                            .map(|f| clamp_fraction(multiple * f, LEVERAGE_CAP))
                            .collect()
                    } else if policy == POLICY_MARGINAL {
                        vec![1.7; bars]
                    } else if policy == POLICY_BUY_HOLD {
                        vec![1.0; bars]
                    } else {
                        realized
                            .iter()
                            .map(|r| kelly_fraction(&[1.0], &[*r], LEVERAGE_CAP))
                            .collect()
                    }
                });
                WindowPaths::unmeasured(realized, free, positions)
            })
            .collect()
    }

    #[test]
    fn perfect_foresight_upper_bounds_every_policy_on_every_window() {
        let windows = fixture_windows(12, 40, 0xC001);
        let oracle = Ledger::build(&windows, POLICY_ORACLE, LEVERAGE_CAP);
        let oracle_growth = oracle.window_growth(0.0);
        for policy in 0..POLICY_COUNT {
            if policy == POLICY_ORACLE {
                continue;
            }
            let other = Ledger::build(&windows, policy, LEVERAGE_CAP).window_growth(0.0);
            for (window, (ceiling, growth)) in oracle_growth.iter().zip(&other).enumerate() {
                assert!(
                    *ceiling >= growth - 1e-12,
                    "policy {} beat perfect foresight on window {window}: {growth} > \
                     {ceiling}",
                    POLICY_NAMES[policy]
                );
            }
        }
        // Bar by bar, not merely on average: the oracle maximizes each term.
        for window in &windows {
            for (bar, realized) in window.realized.iter().enumerate() {
                let ceiling = (1.0 + window.positions[POLICY_ORACLE][bar] * realized).ln();
                for policy in 0..POLICY_COUNT {
                    if policy == POLICY_ORACLE {
                        continue;
                    }
                    let growth = (1.0 + window.positions[policy][bar] * realized).ln();
                    assert!(ceiling >= growth - 1e-12);
                }
            }
        }
    }

    #[test]
    fn costs_monotonically_reduce_realized_growth() {
        let windows = fixture_windows(8, 32, 0xD001);
        for policy in 0..POLICY_COUNT {
            let ledger = Ledger::build(&windows, policy, LEVERAGE_CAP);
            let mut previous = f64::INFINITY;
            for bps in COST_GRID_BPS {
                let growth = ledger.net_growth_per_bar(bps * 1e-4);
                assert!(
                    growth < previous,
                    "{} growth did not fall from {previous} at {bps} bps: {growth}",
                    POLICY_NAMES[policy]
                );
                previous = growth;
            }
            assert!(
                ledger.stats(0.0).turnover > 0.0,
                "every policy in this fixture trades at least its entry and unwind"
            );
        }
    }

    fn fixture_config() -> BenchConfig {
        BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 1.7)
    }

    #[test]
    fn the_bench_pairs_windows_and_solves_the_break_even() {
        let windows = fixture_windows(24, 48, 0xE001);
        let blocks: Vec<u64> = (0..windows.len() as u64).map(|w| w / 4).collect();
        let measured = bench(&windows, &blocks, &TailCounts::empty(), fixture_config());
        assert_eq!(measured.bars, 24 * 48);
        assert_eq!(measured.windows, 24);
        assert_eq!(measured.blocks, 6);
        assert_eq!(measured.model_cost_curve().len(), COST_GRID_BPS.len());
        // The curve is the same object the break-even is solved on.
        assert!(
            (measured.model_cost_curve()[DEFAULT_COST_SLOT] - measured.edge_at_default()).abs()
                < 1e-12
        );
        assert!(
            measured
                .model_cost_curve()
                .windows(2)
                .all(|pair| pair[0] > pair[1]),
            "the edge curve must fall with cost: {:?}",
            measured.model_cost_curve()
        );
        // Random positions cannot beat the null, so this fixture has no break-even and
        // says so rather than inventing one.
        assert!(
            measured.model_break_even().is_nan() || measured.model_break_even() > 0.0,
            "break-even {} is not a cost",
            measured.model_break_even()
        );
        // Every policy carries its OWN paired verdict on the SAME windows, and the null's own
        // row is the difference of the null against itself: exactly zero, with a zero-width
        // interval and no break-even. If that row is ever non-zero the pairing is broken and
        // every other policy's edge is measured against something other than the null.
        assert_eq!(measured.edge[POLICY_MARGINAL].mean, 0.0);
        assert_eq!(measured.edge[POLICY_MARGINAL].ci_low, 0.0);
        assert_eq!(measured.edge[POLICY_MARGINAL].ci_high, 0.0);
        assert!(
            measured.edge[POLICY_MARGINAL].blocks == measured.blocks,
            "the null's own row must be resampled on the same blocks as the model's"
        );
        for policy in 0..POLICY_COUNT {
            assert!(
                (measured.cost_curve[policy][DEFAULT_COST_SLOT]
                    - (measured.policies[policy].net_growth
                        - measured.policies[POLICY_MARGINAL].net_growth))
                    .abs()
                    < 1e-12,
                "{}'s cost curve must pass through its own headline edge",
                POLICY_NAMES[policy]
            );
            assert!(
                measured.ceiling_capture[policy].is_nan()
                    || measured.ceiling_capture[policy] <= 1.0 + 1e-12,
                "{} captured more than the perfect-foresight ceiling: {}",
                POLICY_NAMES[policy],
                measured.ceiling_capture[policy]
            );
        }
        // Printed so `cargo test -- --nocapture` shows the console panel this bench emits,
        // which is the same text the pretrainer logs at every validation.
        for line in measured.report_lines() {
            println!("{line}");
            assert!(!line.is_empty());
        }
    }

    #[test]
    fn the_cap_curve_re_derives_the_headline_at_the_headline_cap() {
        let windows = fixture_windows(16, 32, 0xE002);
        let blocks: Vec<u64> = (0..windows.len() as u64).map(|w| w / 4).collect();
        let measured = bench(&windows, &blocks, &TailCounts::empty(), fixture_config());
        let slot = CAP_GRID
            .iter()
            .position(|cap| *cap == LEVERAGE_CAP)
            .expect("the headline cap is a point of the cap grid");
        let point = measured.cap_curve[slot];
        // The curve is a re-clamp of the same solved fractions, so at the headline cap it
        // must reproduce the headline exactly rather than approximately. If this drifts, the
        // curve is measuring a different policy than the one being reported.
        assert!(
            (point.edge - measured.edge_at_default()).abs() < 1e-12,
            "cap curve edge {} != headline {}",
            point.edge,
            measured.edge_at_default()
        );
        assert!((point.sharpe - measured.policies[POLICY_MODEL].sharpe).abs() < 1e-12);
        assert!(
            (point.max_drawdown - measured.policies[POLICY_MODEL].max_drawdown).abs() < 1e-12
        );
        assert!(
            (point.clamped_fraction - measured.policies[POLICY_MODEL].clamped_fraction).abs()
                < 1e-12
        );

        // Exposure has to RISE with the cap, since every position is a clamp of the same
        // free fraction. A curve that did not would mean the re-clamp is not a clamp.
        for pair in measured.cap_curve.windows(2) {
            assert!(
                pair[1].mean_abs_position >= pair[0].mean_abs_position - 1e-12,
                "exposure fell as the cap rose: {:?} then {:?}",
                pair[0].mean_abs_position,
                pair[1].mean_abs_position
            );
            assert!(
                pair[1].clamped_fraction <= pair[0].clamped_fraction + 1e-12,
                "a looser cap bound MORE bars"
            );
        }
        // The fixture's free path exceeds the cap by construction, so the saturation figure
        // is a measurement rather than a structural zero.
        assert!(
            measured.free_kelly.saturated > 0.0,
            "this fixture saturates the cap on purpose"
        );
        let mass: f64 = measured.free_kelly.histogram.iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-9,
            "the |f*| histogram is a distribution, got mass {mass}"
        );
        assert!(measured.free_kelly.median <= measured.free_kelly.p95);
    }

    #[test]
    fn the_tail_block_turns_counts_into_calibration() {
        // Four windows of 1000 bars. The lower 1% threshold was breached 40 times, i.e.
        // exactly 4x what the model promised, and the upper 1% exactly once per 100 bars.
        let mut counts = TailCounts::empty();
        counts.bars = vec![1000.0; 4];
        let one_percent = TAIL_LEVELS
            .iter()
            .position(|q| *q == 0.01)
            .expect("1% is a reported tail level");
        for level in 0..TAIL_LEVELS.len() {
            counts.lower[level] = vec![0.0; 4];
            counts.upper[level] = vec![0.0; 4];
        }
        counts.lower[one_percent] = vec![10.0; 4];
        counts.upper[one_percent] = vec![10.0, 10.0, 10.0, 10.0];
        counts.upper[one_percent][3] = 10.0;
        let blocks: Vec<u64> = vec![0, 0, 1, 1];
        let calibration = tail_calibration(&counts, &blocks);
        assert_eq!(calibration.windows, 4);
        assert_eq!(calibration.bars, 4000.0);
        let lower = calibration.lower[one_percent];
        assert!((lower.realized - 0.01).abs() < 1e-12);
        assert!((lower.ratio - 1.0).abs() < 1e-12);
        assert!(
            lower.wilson.0 < lower.realized && lower.realized < lower.wilson.1,
            "the Wilson interval must bracket the point estimate: {:?}",
            lower.wilson
        );
        assert!(
            lower.blocked.0 <= lower.realized && lower.realized <= lower.blocked.1,
            "the blocked interval must bracket the pooled rate: {:?}",
            lower.blocked
        );
        // Every window has the identical rate, so resampling windows cannot move it: the
        // blocked interval collapses onto the point estimate. That is the correct answer
        // and it is what proves the interval is driven by BETWEEN-window variation.
        assert!((lower.blocked.1 - lower.blocked.0).abs() < 1e-12);

        // Zero exceedances must not read as perfect calibration.
        let empty_level = TAIL_LEVELS
            .iter()
            .position(|q| *q == 0.001)
            .expect("0.1% is a reported tail level");
        let never = calibration.lower[empty_level];
        assert_eq!(never.realized, 0.0);
        assert!(
            never.wilson.1 > 0.0,
            "an unobserved tail is uncertain, not exact: {:?}",
            never.wilson
        );

        // A four-fold understatement has to surface as the worst ratio, on the lower side.
        counts.lower[one_percent] = vec![40.0; 4];
        let understated = tail_calibration(&counts, &blocks);
        let (worst, is_lower) = understated.worst();
        assert!((worst - 4.0).abs() < 1e-12, "worst ratio {worst} is not 4x");
        assert!(is_lower, "the understated side is the lower one");
        assert!(worst > TAIL_RATIO_WARN, "4x must trip the warning threshold");
    }

    #[test]
    fn the_predicted_quantile_inverts_a_known_cdf_exactly() {
        // A uniform law over 128 unit-width bins spanning `[0, 128)`: the `q`-quantile in
        // value space is exactly `128 q`, which makes this an analytic check on the
        // interpolation rather than a self-consistency check.
        let bins = NUM_BAR_BINS as f64;
        let probs = Tensor::full([1, NUM_BAR_BINS], 1.0 / bins, (Kind::Double, Device::Cpu));
        let lo = Tensor::from_slice(&(0..NUM_BAR_BINS).map(|i| i as f64).collect::<Vec<_>>())
            .view([1, NUM_BAR_BINS]);
        let hi = &lo + 1.0;
        for q in [0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.999] {
            let value = predicted_quantile(&probs, &lo, &hi, q).double_value(&[0]);
            assert!(
                (value - bins * q).abs() < 1e-9,
                "q={q} inverted to {value}, not {}",
                bins * q
            );
        }
        // Nested levels give nested thresholds, which is what makes an exceedance count at
        // one level a subset of the count at a looser one.
        let mut previous = f64::NEG_INFINITY;
        for q in TAIL_LEVELS {
            let value = predicted_quantile(&probs, &lo, &hi, q).double_value(&[0]);
            assert!(value > previous, "tail thresholds must be nested");
            previous = value;
        }
    }

    #[test]
    fn tail_counts_are_nested_across_levels_on_a_real_head() {
        let _torch_rng_guard = test_rng::shared();
        let latent = 12;
        let (_vs, head) = perturbed_head(latent, 0x1A01);
        let supports = synthetic_supports(30_000, 0x1A02);
        let returns = Tensor::from_slice(&bin_returns(&supports)).view([1, NUM_BAR_BINS]);
        let centers = Tensor::from_slice(supports.centers(DOF_R)).view([1, NUM_BAR_BINS]);
        let lo = Tensor::from_slice(supports.lower_bounds(DOF_R)).view([1, NUM_BAR_BINS]);
        let hi = Tensor::from_slice(supports.upper_bounds(DOF_R)).view([1, NUM_BAR_BINS]);
        let (windows, bars) = (3i64, 64i64);
        let h = beliefs(windows * bars, latent, 0x1A03).view([windows, bars, latent]);
        // A realized path with genuine outliers, so both tails are actually reached.
        let realized = Tensor::from_slice(
            &(0..windows * bars)
                .map(|slot| {
                    let u = uniform(0x1A04, slot as u64);
                    (0.004 * (2.0 * u - 1.0) + if slot % 17 == 0 { 0.09 } else { 0.0 }) as f32
                })
                .collect::<Vec<f32>>(),
        )
        .view([windows, bars]);
        let chunk = window_paths(
            &head,
            &h,
            &realized,
            &TradedLaw::new(&returns, &centers).with_bounds(&lo, &hi),
            marginal_position(&supports, FREE_LEVERAGE),
            LEVERAGE_CAP,
        )
        .expect("paths");
        assert_eq!(chunk.tail.windows(), windows as usize);
        assert!(chunk.tail.bars.iter().all(|b| *b == bars as f64));
        for window in 0..windows as usize {
            for side in [&chunk.tail.lower, &chunk.tail.upper] {
                for level in 0..TAIL_LEVELS.len() {
                    assert!(
                        side[level][window] <= bars as f64,
                        "more exceedances than bars"
                    );
                    if level > 0 {
                        assert!(
                            side[level][window] >= side[level - 1][window],
                            "a looser tail level must count at least as many exceedances: \
                             level {level} counted {} against {}",
                            side[level][window],
                            side[level - 1][window]
                        );
                    }
                }
            }
        }
        // The upper tail must actually fire on a path with +9% bars in it, or the nesting
        // assertions above are vacuous.
        let loosest = TAIL_LEVELS.len() - 1;
        assert!(
            chunk.tail.upper[loosest].iter().sum::<f64>() > 0.0,
            "a path with 9% bars must breach the model's upper tail somewhere"
        );
    }

    #[test]
    fn the_break_even_is_where_the_edge_actually_vanishes() {
        // A synthetic edge curve: the model out-grows the null by 1 bp/bar gross and
        // trades 2.0 of notional per bar against the null's 0.
        let edge_at = |bps: f64| 1e-4 - 2.0 * bps * 1e-4;
        let crossing = break_even_bps(&edge_at);
        assert!(
            (crossing - 0.5).abs() < 1e-9,
            "an edge of 1 bp/bar at 2.0 turnover breaks even at 0.5 bps, got {crossing}"
        );
        assert!(break_even_bps(&|_| -1e-6).is_nan(), "no edge, no break-even");
        assert!(
            break_even_bps(&|_| 1e-6).is_infinite(),
            "an edge cost cannot touch never breaks even"
        );
    }

    #[test]
    fn the_bin_returns_are_the_supports_own_geometry() {
        let supports = synthetic_supports(20_000, 0xF001);
        let returns = bin_returns(&supports);
        assert_eq!(returns.len(), NUM_BAR_BINS as usize);
        assert!(
            returns.windows(2).all(|pair| pair[0] <= pair[1]),
            "bin returns inherit the support's value order"
        );
        assert!(
            returns.iter().all(|r| *r > -1.0 && r.is_finite()),
            "a simple return derived from a finite log return is above -100%"
        );
        for (bin, center) in supports.centers(DOF_R).iter().enumerate() {
            assert!((returns[bin] - center.exp_m1()).abs() < 1e-15);
        }
        // An atom's bin reproduces the atom's own return exactly.
        for atom in supports.atoms(DOF_R) {
            let bin = supports.bin_of(DOF_R, atom.value as f64);
            assert!((returns[bin] - (atom.value as f64).exp_m1()).abs() < 1e-9);
        }
    }

    #[test]
    fn encoded_bars_round_trip_into_the_returns_the_bench_pays() {
        // The bench's `R` must be the bar's actual close-to-close simple return, not a
        // reinterpretation of it.
        let bar = shared::bars::PackedBar {
            ts_ms: 0,
            open: 100.0,
            high: 101.5,
            low: 99.5,
            close: 101.0,
            volume: 1000.0,
            vwap: 100.5,
            trades: 10,
        };
        let ema = VolumeEma::default();
        let dof = encode_dof(100.0, &bar, ema.reference_for(bar.volume));
        let simple = (dof.r as f64).exp_m1();
        assert!(
            (simple - 0.01).abs() < 1e-6,
            "a 100 -> 101 bar is a 1% return, got {simple}"
        );
    }

    // -----------------------------------------------------------------------
    // Mean calibration
    // -----------------------------------------------------------------------

    /// A synthetic panel whose calibration slope is KNOWN by construction.
    ///
    /// The truth is `r = mu + eps` with `mu` a genuine per-bar conditional mean and `eps`
    /// zero-mean noise that is independent of it. The FORECAST handed to the regression is
    /// `mu / beta`, so `E[r | forecast] = beta * forecast` exactly and the population slope is
    /// `beta` whatever the noise scale is. The noise is deterministic — an antithetic pair per
    /// block, so it sums to zero inside every block and cannot smuggle a correlation with `mu`
    /// in through a finite sample.
    fn calibrated_panel(
        blocks: usize,
        bars_per_block: usize,
        beta: f64,
        noise: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let mut forecast = Vec::with_capacity(blocks * bars_per_block);
        let mut realized = Vec::with_capacity(blocks * bars_per_block);
        let mut ids = Vec::with_capacity(blocks * bars_per_block);
        for block in 0..blocks {
            // A block-level level shift on top of a per-bar signal, so blocks are genuinely
            // clustered and the blocked interval has something to be wider than.
            let level = 0.0004 * (2.0 * uniform(seed, block as u64) - 1.0);
            for bar in 0..bars_per_block {
                let slot = (block * bars_per_block + bar) as u64;
                let mu = level + 0.001 * (2.0 * uniform(mix64(seed, 1), slot) - 1.0);
                let sign = if bar % 2 == 0 { 1.0 } else { -1.0 };
                let eps = sign * noise * (0.5 + uniform(mix64(seed, 2), slot / 2));
                forecast.push(mu / beta);
                realized.push(mu + eps);
                ids.push(block as u64);
            }
        }
        (forecast, realized, ids)
    }

    #[test]
    fn a_known_calibration_slope_is_recovered_within_its_own_bootstrap_error() {
        // Predictions inflated by exactly 1 / 0.7: the recoverable slope is 0.7.
        let (forecast, realized, blocks) = calibrated_panel(64, 32, 0.7, 0.004, 0xCA11_0001);
        let fit = mincer_zarnowitz(
            &forecast,
            &realized,
            &blocks,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        );

        assert_eq!(fit.samples, 64 * 32);
        assert_eq!(fit.blocks, 64);
        assert!(
            (fit.beta - 0.7).abs() < 2.0 * fit.beta_se,
            "the slope of a panel built with beta = 0.7 came back {:.5} +/- {:.5}, which is \
             more than two of its own standard errors away",
            fit.beta,
            fit.beta_se
        );
        assert!(
            fit.beta_ci.0 < 0.7 && 0.7 < fit.beta_ci.1,
            "the blocked interval {:.4}..{:.4} misses the slope it was built with",
            fit.beta_ci.0,
            fit.beta_ci.1
        );
        assert!(
            fit.slope_resolvable(),
            "an inflation of 1/0.7 over 2048 bars has to be resolvable from perfect \
             calibration, or the diagnostic cannot see the effect it exists for"
        );
        assert!(
            fit.alpha.abs() < 3.0 * fit.alpha_se.max(1e-12),
            "the panel has no intercept, so the fitted one must be inside its own noise: \
             {:+.3e} +/- {:.3e}",
            fit.alpha,
            fit.alpha_se
        );
        // A perfectly calibrated panel must come back at one, or the estimator is biased
        // rather than the model miscalibrated.
        let (honest_x, honest_y, honest_blocks) =
            calibrated_panel(64, 32, 1.0, 0.004, 0xCA11_0002);
        let honest = mincer_zarnowitz(
            &honest_x,
            &honest_y,
            &honest_blocks,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        );
        assert!(
            (honest.beta - 1.0).abs() < 2.0 * honest.beta_se,
            "a calibrated panel fitted to beta {:.5} +/- {:.5}",
            honest.beta,
            honest.beta_se
        );
        assert!(
            !honest.slope_resolvable(),
            "a calibrated panel must NOT be flagged as miscalibrated"
        );
    }

    /// A panel whose blocks share ONE slope must read as common; one whose blocks genuinely
    /// disagree must read as heteroskedastic. Both directions are asserted, so the predicate
    /// cannot pass by always answering the same way, and the unmeasured case is asserted to be
    /// a THIRD state rather than the homogeneous branch.
    #[test]
    fn slope_dispersion_separates_a_shared_slope_from_genuinely_varying_ones() {
        // 512 bars per block at a tenth of the earlier noise. The size is not decoration: at
        // 64 bars and eps_sd 0.004 a single block's slope carries a standard error near 0.71,
        // which swamps any dispersion this fixture could build, and the estimator correctly
        // reports "common" for a panel it cannot resolve. A test that ran there would be
        // asserting the noise floor rather than the statistic.
        let (forecast_h, realized_h, blocks_h) = calibrated_panel(64, 512, 0.7, 0.0005, 0xCA11_0301);
        let homogeneous = mincer_zarnowitz(
            &forecast_h,
            &realized_h,
            &blocks_h,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        );
        assert!(
            homogeneous.block_dispersion_measured(),
            "a 64-block panel with 512 bars each must resolve a dispersion"
        );
        assert!(
            !homogeneous.slope_heterogeneous(),
            "blocks built from ONE slope came back heteroskedastic at {:.2}x its own noise \
             floor (sd {:.4} against {:.4})",
            homogeneous.block_dispersion_ratio(),
            homogeneous.beta_block_sd,
            homogeneous.beta_block_noise_sd,
        );

        // Alternating blocks at 0.35 and 1.05: the same pooled slope of 0.7 on average, so
        // only the DISPERSION can tell the two panels apart.
        let mut forecast = Vec::new();
        let mut realized = Vec::new();
        let mut blocks = Vec::new();
        for block in 0..64u64 {
            let beta = if block % 2 == 0 { 0.35 } else { 1.05 };
            let (f, r, _) = calibrated_panel(1, 512, beta, 0.0005, 0xCA11_0400 + block);
            forecast.extend(f);
            realized.extend(r);
            blocks.extend(std::iter::repeat_n(block, 512));
        }
        let varying = mincer_zarnowitz(&forecast, &realized, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        assert!(
            varying.slope_heterogeneous(),
            "blocks built at 0.35 and 1.05 came back COMMON at {:.2}x (sd {:.4} against {:.4})",
            varying.block_dispersion_ratio(),
            varying.beta_block_sd,
            varying.beta_block_noise_sd,
        );
        assert!(
            varying.beta_block_excess_sd() > 0.1,
            "a 0.35/1.05 split has a true slope sd near 0.35 and reported excess {:.4}",
            varying.beta_block_excess_sd(),
        );

        // The third state. `NaN > 1.25` is false, so without the gate an unmeasured fit would
        // answer "homogeneous" — a positive finding about data that does not exist.
        let absent = MzFit::nan();
        assert!(!absent.block_dispersion_measured());
        assert!(
            !absent.slope_heterogeneous(),
            "an unmeasured fit must not claim heterogeneity"
        );
        assert!(
            absent.block_dispersion_ratio().is_nan(),
            "and the ratio itself must stay NaN rather than becoming a number"
        );
    }

    /// A panel whose CLUSTERING is the point: most of the forecast's variation is a
    /// block-level level, and the realized return carries a block-level shock on top of the
    /// per-bar noise.
    ///
    /// Both halves are needed and neither alone is enough, which is a fact about slopes rather
    /// than about this fixture. A blocked interval on a MEAN is wider than an iid one whenever
    /// the outcome is clustered at all. A blocked interval on a SLOPE widens only through the
    /// covariance between a block's mean regressor and that block's common error: if the
    /// regressor has no block-level component, the within-block variation identifies the slope
    /// and the common shock cancels out of the normal equations. Measured on this fixture the
    /// ratio is about 3; with the regressor's block level removed it collapses to 1.0, and
    /// with the shock removed to 1.0 as well.
    ///
    /// This is what a `(symbol, calendar month)` block actually looks like: one name in one
    /// month has a persistent forecast level and a realized regime return, and both persist
    /// for the whole block.
    fn clustered_panel(
        blocks: usize,
        bars_per_block: usize,
        beta: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let mut forecast = Vec::with_capacity(blocks * bars_per_block);
        let mut realized = Vec::with_capacity(blocks * bars_per_block);
        let mut ids = Vec::with_capacity(blocks * bars_per_block);
        for block in 0..blocks {
            let level = 0.0010 * (2.0 * uniform(seed, block as u64) - 1.0);
            let shock = 0.0040 * (2.0 * uniform(mix64(seed, 3), block as u64) - 1.0);
            for bar in 0..bars_per_block {
                let slot = (block * bars_per_block + bar) as u64;
                let mu = level + 0.0004 * (2.0 * uniform(mix64(seed, 1), slot) - 1.0);
                let eps = 0.004 * (2.0 * uniform(mix64(seed, 2), slot) - 1.0);
                forecast.push(mu / beta);
                realized.push(mu + shock + eps);
                ids.push(block as u64);
            }
        }
        (forecast, realized, ids)
    }

    #[test]
    fn the_blocked_slope_interval_is_wider_than_an_unclustered_one() {
        let (forecast, realized, blocks) = clustered_panel(64, 32, 0.7, 0xCA11_0003);
        let se_of = |ids: &[u64]| {
            mincer_zarnowitz(&forecast, &realized, ids, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED).beta_se
        };
        let blocked = se_of(&blocks);
        // One block per bar: the interval a naive iid bootstrap would report.
        let singletons: Vec<u64> = (0..blocks.len() as u64).collect();
        let iid = se_of(&singletons);
        assert!(
            blocked > 1.5 * iid,
            "blocking by regime must widen the slope's interval ({blocked:.5} against \
             {iid:.5}), or the clustering the bars actually have is being divided away"
        );

        // And the mechanism is the one documented, not an accident of the seed: strip the
        // block-level component out of the REGRESSOR and the same blocked resampling over the
        // same block shocks stops widening anything, because a slope is then identified
        // entirely within blocks.
        let within: Vec<f64> = forecast
            .iter()
            .enumerate()
            .map(|(slot, f)| {
                let block = slot / 32;
                let mean: f64 =
                    forecast[block * 32..(block + 1) * 32].iter().sum::<f64>() / 32.0;
                f - mean
            })
            .collect();
        let demeaned = |ids: &[u64]| {
            mincer_zarnowitz(&within, &realized, ids, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED).beta_se
        };
        let ratio = demeaned(&blocks) / demeaned(&singletons);
        assert!(
            ratio < 1.2,
            "with no block-level regressor there is nothing for the block shock to correlate \
             with, so the blocked interval must stop widening; ratio {ratio:.3}"
        );
    }

    #[test]
    fn the_calibration_fit_is_deterministic_and_survives_non_finite_rows() {
        let (mut forecast, mut realized, mut blocks) =
            calibrated_panel(16, 16, 0.7, 0.004, 0xCA11_0004);
        let clean = mincer_zarnowitz(&forecast, &realized, &blocks, 256, BOOTSTRAP_SEED);
        let again = mincer_zarnowitz(&forecast, &realized, &blocks, 256, BOOTSTRAP_SEED);
        assert_eq!(clean, again, "the same panel must fit to the same numbers");

        forecast.push(f64::NAN);
        realized.push(0.001);
        blocks.push(999);
        forecast.push(0.001);
        realized.push(f64::INFINITY);
        blocks.push(999);
        let filtered = mincer_zarnowitz(&forecast, &realized, &blocks, 256, BOOTSTRAP_SEED);
        assert_eq!(
            filtered, clean,
            "non-finite rows must be dropped, not propagated into the fit"
        );
    }

    #[test]
    fn the_fit_slice_is_disjoint_from_the_traded_slice_in_windows_and_blocks() {
        // Windows 0..4 are traded and sit in blocks 10, 10, 11, 12. Window 5 REPEATS block
        // 11, so it must be refused even though it is outside the traded prefix.
        let blocks = vec![10, 10, 11, 12, 20, 11, 21, 20, 22, 23];
        let traded = 4usize;
        let fit = disjoint_fit_windows(&blocks, traded, 100);

        assert!(
            fit.iter().all(|index| *index >= traded),
            "a traded window can never be part of the fit slice: {fit:?}"
        );
        assert!(
            !fit.contains(&5),
            "window 5 shares block 11 with the traded prefix and must be refused: {fit:?}"
        );
        assert_eq!(fit, vec![4, 6, 7, 8, 9]);

        let fit_blocks: Vec<u64> = fit.iter().map(|index| blocks[*index]).collect();
        let eval_blocks = blocks[..traded].to_vec();
        assert!(
            blocks_disjoint(&fit_blocks, &eval_blocks),
            "the two slices share a resampling unit, so a slope fitted on one is partly \
             fitted on the other: {fit_blocks:?} against {eval_blocks:?}"
        );
        // The budget is respected, and it takes a PREFIX so the slice is reproducible.
        assert_eq!(disjoint_fit_windows(&blocks, traded, 2), vec![4, 6]);
        // Everything shares a block with the traded prefix: no fit slice exists, and saying
        // so is the only honest answer.
        assert!(disjoint_fit_windows(&[7, 7, 7, 7], 2, 100).is_empty());
    }

    #[test]
    fn a_shared_block_is_detected_however_it_is_ordered() {
        assert!(blocks_disjoint(&[1, 2, 3], &[4, 5, 6]));
        assert!(!blocks_disjoint(&[1, 2, 3], &[6, 5, 3]));
        assert!(!blocks_disjoint(&[3], &[3]));
        assert!(blocks_disjoint(&[], &[1, 2]));
    }

    /// A discrete law over `r`, its bin returns, and the exact optimum of the TRUE law.
    ///
    /// Deterministic, and built so the model's law differs from the truth by a pure LOCATION
    /// shift: identical masses, identical spread, mean inflated by `1 / beta`. That isolates
    /// exactly the failure the recalibration is for.
    fn inflated_law(beta: f64, spread: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let bins = 41usize;
        let mut masses = Vec::with_capacity(bins);
        let mut centers = Vec::with_capacity(bins);
        for bin in 0..bins {
            let z = (bin as f64 - (bins as f64 - 1.0) / 2.0) / 6.0;
            masses.push((-0.5 * z * z).exp());
            centers.push(spread * z);
        }
        let total: f64 = masses.iter().sum();
        for mass in &mut masses {
            *mass /= total;
        }
        // A genuine edge: shift the whole support so the true conditional mean is positive.
        let truth: Vec<f64> = centers.iter().map(|c| c + 0.0004).collect();
        let mean_true: f64 = masses.iter().zip(&truth).map(|(p, c)| p * c).sum();
        // The model reports the same shape with the mean divided by beta.
        let inflation = mean_true / beta - mean_true;
        let model: Vec<f64> = truth.iter().map(|c| c + inflation).collect();
        (masses, truth, model)
    }

    #[test]
    fn shrinking_by_the_true_slope_increases_expected_log_growth() {
        let beta = 0.7;
        // `0.012` of sd against `0.0004` of drift is a per-bar Sharpe of `0.033`, the level the
        // real bench measures, and it puts BOTH optima strictly inside [`MAX_LEVERAGE`]: the
        // model asks about `4x` and the truth about `3x`. A narrower law puts both against the
        // declared ruin bound, where they are equal by construction and prove nothing.
        let (masses, truth, model) = inflated_law(beta, 0.012);
        let true_returns: Vec<f64> = truth.iter().map(|c| c.exp() - 1.0).collect();
        let model_returns: Vec<f64> = model.iter().map(|c| c.exp() - 1.0).collect();
        let mean_model: f64 = masses.iter().zip(&model).map(|(p, c)| p * c).sum();

        // The recalibration the fitted slope prescribes, applied exactly as `window_paths`
        // applies it: shift every bin's LOG value so the mean becomes `beta * mu`.
        let shrink = MeanShrink {
            alpha: 0.0,
            beta,
        };
        let shift = shrink.shift(mean_model);
        let shrunk_returns: Vec<f64> = model_returns
            .iter()
            .map(|r| (1.0 + r) * shift.exp() - 1.0)
            .collect();
        let shrunk_mean: f64 = masses
            .iter()
            .zip(&model)
            .map(|(p, c)| p * (c + shift))
            .sum();
        let mean_true: f64 = masses.iter().zip(&truth).map(|(p, c)| p * c).sum();
        assert!(
            (shrunk_mean - mean_true).abs() < 1e-15,
            "recalibrating by the TRUE slope has to land exactly on the true mean: {shrunk_mean:.3e} \
             against {mean_true:.3e}"
        );

        let unshrunk_f = kelly_fraction(&masses, &model_returns, FREE_LEVERAGE);
        let shrunk_f = kelly_fraction(&masses, &shrunk_returns, FREE_LEVERAGE);
        assert!(
            shrunk_f > 0.0 && shrunk_f < unshrunk_f,
            "an inflated mean has to ask for a strictly larger position: {unshrunk_f:.4} \
             against {shrunk_f:.4}"
        );

        // Both positions are PAID under the true law, which is what makes this a statement
        // about realized growth rather than about the model's own opinion of itself.
        let unshrunk_growth = expected_log_growth(&masses, &true_returns, unshrunk_f);
        let shrunk_growth = expected_log_growth(&masses, &true_returns, shrunk_f);
        assert!(
            shrunk_growth > unshrunk_growth,
            "sizing on the recalibrated mean must earn more under the truth: {shrunk_growth:.6e} \
             against {unshrunk_growth:.6e}"
        );
        // And it must be the OPTIMUM of the true law, not merely an improvement: the true
        // law's own Kelly fraction is what the recalibration reconstructs.
        let oracle_f = kelly_fraction(&masses, &true_returns, FREE_LEVERAGE);
        assert!(
            (shrunk_f - oracle_f).abs() < 1e-6 * oracle_f.abs().max(1.0),
            "recalibrating by the true slope reconstructs the true law, so its position must \
             be the true law's own optimum: {shrunk_f:.6} against {oracle_f:.6}"
        );
    }

    /// The clipped support bounds of `r` on the live 300s grid: the decode convention the
    /// pipeline reads today, and the thing the decomposition is asked to undo.
    const BOUND: (f64, f64) = (-0.088332, 0.088038);

    #[test]
    fn redecoding_the_catch_alls_reproduces_a_direct_sum_over_the_bins() {
        // An explicit law: 128 bins, the outer two carrying deliberately lopsided mass so the
        // signed net is not zero and a symmetric bug cannot pass.
        let bins = NUM_BAR_BINS as usize;
        let mut probs = vec![0.0f64; bins];
        probs[0] = 0.004;
        probs[bins - 1] = 0.018;
        let interior: f64 = 1.0 - probs[0] - probs[bins - 1];
        let centers: Vec<f64> = (0..bins)
            .map(|bin| 0.0004 * (bin as f64 - (bins as f64 - 1.0) / 2.0))
            .collect();
        // A non-uniform interior, so `interior_mean` is not the midpoint by accident.
        let weights: Vec<f64> = (1..bins - 1).map(|bin| 1.0 + (bin % 7) as f64).collect();
        let total: f64 = weights.iter().sum();
        for (bin, weight) in weights.iter().enumerate() {
            probs[bin + 1] = interior * weight / total;
        }

        // The interior law, renormalized - exactly what `window_paths` forms with the mask.
        let interior_mean: f64 =
            (1..bins - 1).map(|b| probs[b] * centers[b]).sum::<f64>() / interior;
        let interior_second: f64 =
            (1..bins - 1).map(|b| probs[b] * centers[b] * centers[b]).sum::<f64>() / interior;
        let bar = OuterBar {
            mass: probs[0] + probs[bins - 1],
            signed: probs[bins - 1] - probs[0],
            interior_mean,
            interior_var: interior_second - interior_mean * interior_mean,
        };

        // Against a direct sum over all 128 bins with the two catch-alls valued at `decode`.
        for decode in [BOUND, OUTER_REDECODE, (-0.01, 0.02)] {
            let mut values = centers.clone();
            values[0] = decode.0;
            values[bins - 1] = decode.1;
            let direct_mean: f64 = (0..bins).map(|b| probs[b] * values[b]).sum();
            let direct_second: f64 = (0..bins).map(|b| probs[b] * values[b] * values[b]).sum();
            let direct_var = direct_second - direct_mean * direct_mean;
            let (mean, var) = bar.redecoded(decode);
            assert!(
                (mean - direct_mean).abs() < 1e-15,
                "the closed form has to be the same number as the sum: {mean:.6e} against \
                 {direct_mean:.6e} at decode {decode:?}"
            );
            assert!(
                (var - direct_var).abs() < 1e-15,
                "and so does the variance: {var:.6e} against {direct_var:.6e} at decode \
                 {decode:?}"
            );
        }
    }

    /// Windows whose INTERIOR law is perfectly calibrated and whose reported mean is inflated
    /// purely by pricing catch-all mass at the clipped bound.
    ///
    /// The whole point of the fixture: there is no learned error anywhere in it. The conditional
    /// mean of the interior law IS the true conditional mean, so a decomposition that works must
    /// read `beta = 1` off the zeroed arm while the as-traded slope reads `beta` - and one that
    /// merely rescales something cannot, because the inflation is carried by a signed mass whose
    /// contribution depends on the decode point rather than by a factor on `mu`.
    fn decode_artifact_windows(beta: f64, windows: usize, bars: usize) -> Vec<WindowPaths> {
        // Enough mass to carry the inflation at the bound without ever exceeding the total, and
        // near the equal-mass construction's own `2/128 = 1.5625%`.
        let mass = 0.02;
        let spread = BOUND.1 + BOUND.0;
        let lever = BOUND.1 - BOUND.0;
        (0..windows)
            .map(|window| {
                let mut realized = Vec::with_capacity(bars);
                let mut predicted_mean = Vec::with_capacity(bars);
                let mut predicted_var = Vec::with_capacity(bars);
                let mut outer_mass = Vec::with_capacity(bars);
                let mut outer_signed = Vec::with_capacity(bars);
                let mut trimmed_mean = Vec::with_capacity(bars);
                let mut trimmed_var = Vec::with_capacity(bars);
                for bar in 0..bars {
                    let slot = (window * bars + bar) as u64;
                    // A per-bar true mean that changes sign, plus a block-level level shift so
                    // the blocked interval has clustering to be wider than.
                    let mu = 0.0004 * (2.0 * uniform(0xCA11_0700, window as u64) - 1.0)
                        + 0.0008 * (2.0 * uniform(mix64(0xCA11_0700, 1), slot) - 1.0);
                    // Antithetic noise: sums to zero inside the window, so no finite-sample
                    // correlation with `mu` can manufacture a slope.
                    let sign = if bar % 2 == 0 { 1.0 } else { -1.0 };
                    let eps = sign * 0.004 * (0.5 + uniform(mix64(0xCA11_0700, 2), slot / 2));
                    realized.push((mu + eps).exp() - 1.0);
                    // Solve the signed net that makes the FULL law's mean `mu / beta` when the
                    // catch-alls are decoded at the bound.
                    let signed =
                        2.0 * (mu / beta - (1.0 - mass) * mu - 0.5 * mass * spread) / lever;
                    assert!(
                        signed.abs() <= mass,
                        "the fixture must stay a probability: {signed:.4e} of net against \
                         {mass:.4e} of mass"
                    );
                    let interior_var = 0.012 * 0.012;
                    let bar_stats = OuterBar {
                        mass,
                        signed,
                        interior_mean: mu,
                        interior_var,
                    };
                    let (full_mean, full_var) = bar_stats.redecoded(BOUND);
                    predicted_mean.push(full_mean);
                    predicted_var.push(full_var);
                    outer_mass.push(mass);
                    outer_signed.push(signed);
                    trimmed_mean.push(mu);
                    trimmed_var.push(interior_var);
                }
                let mut paths = WindowPaths::unmeasured(
                    realized.clone(),
                    vec![0.0; bars],
                    std::array::from_fn(|_| vec![0.0; bars]),
                );
                paths.predicted_mean = predicted_mean;
                paths.predicted_var = predicted_var;
                paths.outer_mass = outer_mass;
                paths.outer_signed = outer_signed;
                paths.trimmed_mean = trimmed_mean;
                paths.trimmed_var = trimmed_var;
                paths
            })
            .collect()
    }

    #[test]
    fn zeroing_the_catch_alls_recovers_a_slope_the_decode_convention_destroyed() {
        let beta = 0.7;
        let windows = decode_artifact_windows(beta, 64, 64);
        let blocks: Vec<u64> = (0..windows.len() as u64).map(|w| w / 2).collect();
        let calibration = mean_calibration(&windows, &blocks);

        // As traded, the slope is the artifact and nothing else.
        assert!(
            (calibration.mean.beta - beta).abs() < 2.0 * calibration.mean.beta_se.max(1e-9),
            "the as-traded slope must read the inflation: {:.4} against {beta:.4} (se {:.4})",
            calibration.mean.beta,
            calibration.mean.beta_se
        );
        let outer = calibration
            .outer
            .as_ref()
            .expect("the fixture carries a catch-all decomposition for every bar");

        // ZEROED removes the whole outer contribution, and since the interior law IS the truth
        // here, it has to land on perfect calibration.
        assert!(
            (outer.zeroed.mean.beta - 1.0).abs() < 2.0 * outer.zeroed.mean.beta_se.max(1e-9),
            "zeroing an artifact that is the ONLY error must recover perfect calibration: \
             {:.4} (se {:.4})",
            outer.zeroed.mean.beta,
            outer.zeroed.mean.beta_se
        );

        // RE-DECODED moves the same mass to a nearer point, so it recovers PART of the slope -
        // strictly between the two, which is what makes it a point estimate rather than a bound.
        assert!(
            outer.redecoded.mean.beta > calibration.mean.beta
                && outer.redecoded.mean.beta < outer.zeroed.mean.beta,
            "the re-decoded arm has to sit strictly between as-traded and zeroed: {:.4} \
             against {:.4} and {:.4}",
            outer.redecoded.mean.beta,
            calibration.mean.beta,
            outer.zeroed.mean.beta
        );

        // And the masses are reported, not inferred: a decomposition that lost them would still
        // pass every slope assertion above.
        assert!(
            (outer.mass - 0.02).abs() < 1e-12,
            "the mean catch-all mass has to be the fixture's own: {:.6}",
            outer.mass
        );
    }

    #[test]
    fn a_pass_without_the_decomposition_reports_it_absent_rather_than_zero() {
        // Same windows with the four decomposition vectors dropped: every slope still fits, and
        // the arms must be ABSENT. A zero here would read as "the law holds no catch-all mass",
        // which is a finding, and this is the lack of one.
        let mut windows = decode_artifact_windows(0.7, 8, 32);
        for window in &mut windows {
            window.outer_mass.clear();
            window.outer_signed.clear();
            window.trimmed_mean.clear();
            window.trimmed_var.clear();
        }
        let blocks: Vec<u64> = (0..windows.len() as u64).collect();
        let calibration = mean_calibration(&windows, &blocks);
        assert!(calibration.mean.beta.is_finite(), "the mean fit still stands on its own");
        assert!(calibration.outer.is_none(), "an unformed decomposition must not read as zero");
        assert!(
            calibration
                .report_lines()
                .iter()
                .any(|line| line.contains("catch-all decomposition: not measured")),
            "and the console has to say so: {:?}",
            calibration.report_lines()
        );
    }

    /// Windows carrying a per-bar recalibrated fraction, for the accounting of the shrunk
    /// policy and for the calibration fit.
    ///
    /// The TRUE conditional mean varies per bar, so the calibration regression has a regressor
    /// that actually moves; the model's law is the same shape with that mean divided by `beta`,
    /// so the population slope is `beta` by construction. The realized return of a bar is its
    /// own true law's inverse CDF at a low-discrepancy quantile, taken from a sequence
    /// independent of the mean sweep so the residual carries no information about the forecast.
    fn shrunk_fixture(beta: f64, windows: usize, bars: usize) -> Vec<WindowPaths> {
        // Narrower than the growth test's law on purpose: this fixture needs a MIX of bars
        // whose optimum clears the headline cap and bars whose optimum does not, so the
        // recalibration has somewhere to bite. At this width the sweep spans both.
        let (masses, base, _) = inflated_law(beta, 0.0035);
        let bins = masses.len();
        // Centered shape: every bar's law is this shifted onto its own mean.
        let base_mean: f64 = masses.iter().zip(&base).map(|(p, c)| p * c).sum();
        let shape: Vec<f64> = base.iter().map(|c| c - base_mean).collect();
        let variance: f64 = masses
            .iter()
            .zip(&shape)
            .map(|(p, c)| p * c * c)
            .sum();
        let mut cumulative = Vec::with_capacity(bins);
        let mut running = 0.0;
        for mass in &masses {
            running += mass;
            cumulative.push(running);
        }
        // The golden-ratio sequence: equidistributed, and its correlation with a smooth mean
        // sweep over the same index vanishes rather than being merely small.
        const PHI: f64 = 0.618_033_988_749_895;

        let mass_row = Tensor::from_slice(&masses).view([1, bins as i64]);
        (0..windows)
            .map(|window| {
                let mut realized = Vec::with_capacity(bars);
                let mut predicted_mean = Vec::with_capacity(bars);
                let mut model_rows = Vec::with_capacity(bars * bins);
                let mut shrunk_rows = Vec::with_capacity(bars * bins);
                for bar in 0..bars {
                    let slot = (window * bars + bar) as f64;
                    // A true edge that changes sign and magnitude across bars.
                    let mu_true = 0.0006 * ((slot * 0.11).sin() + 0.35);
                    let mu_model = mu_true / beta;
                    let shift = MeanShrink { alpha: 0.0, beta }.shift(mu_model);
                    predicted_mean.push(mu_model);
                    for centered in &shape {
                        model_rows.push((centered + mu_model).exp() - 1.0);
                        shrunk_rows.push((centered + mu_model + shift).exp() - 1.0);
                    }
                    let q = (slot * PHI).fract();
                    let bin = cumulative
                        .iter()
                        .position(|c| *c >= q)
                        .unwrap_or(bins - 1);
                    realized.push((shape[bin] + mu_true).exp() - 1.0);
                }
                let row = |values: &[f64]| {
                    Tensor::from_slice(values).view([bars as i64, bins as i64])
                };
                let probs = mass_row.expand([bars as i64, bins as i64], false);
                let free = host_vec(&kelly_fractions(&probs, &row(&model_rows), FREE_LEVERAGE));
                let shrunk =
                    host_vec(&kelly_fractions(&probs, &row(&shrunk_rows), FREE_LEVERAGE));
                let mut paths = WindowPaths::unmeasured(
                    realized.clone(),
                    free.clone(),
                    std::array::from_fn(|policy| {
                        let multiple = POLICY_KELLY_MULTIPLE[policy];
                        if multiple.is_finite() {
                            free.iter()
                                .map(|f| clamp_fraction(multiple * f, LEVERAGE_CAP))
                                .collect()
                        } else if policy == POLICY_MARGINAL {
                            vec![0.5; bars]
                        } else if policy == POLICY_BUY_HOLD {
                            vec![1.0; bars]
                        } else {
                            realized
                                .iter()
                                .map(|r| LEVERAGE_CAP * r.signum())
                                .collect()
                        }
                    }),
                );
                paths.predicted_mean = predicted_mean;
                paths.predicted_var = vec![variance; bars];
                paths.free_shrunk = Some(shrunk);
                paths
            })
            .collect()
    }

    #[test]
    fn the_shrunk_policy_out_earns_the_inflated_one_through_the_real_accounting() {
        let beta = 0.7;
        // 40 windows in 20 blocks: the paired interval is a real interval only if there are
        // enough resampling units to have one, and 4 blocks cannot resolve a 2 bps effect no
        // matter how genuine it is. The per-window effect here is the same at any size; what
        // the extra blocks buy is the ability to say so.
        let windows = shrunk_fixture(beta, 40, 64);
        let blocks: Vec<u64> = (0..windows.len() as u64).map(|w| w / 2).collect();
        let config = BenchConfig::new(0.0, LEVERAGE_CAP, 0.5);
        let plain = bench(&windows, &blocks, &TailCounts::empty(), config);
        let shrunk = shrunk_bench(&windows, &blocks, config, MeanShrink { alpha: 0.0, beta })
            .expect("the fixture carries a recalibrated fraction for every bar");

        assert_eq!(shrunk.bars, plain.bars);
        assert_eq!(shrunk.windows, plain.windows);
        assert!(
            shrunk.policy.gross_growth > plain.policies[POLICY_MODEL].gross_growth,
            "the recalibrated policy has to realize more log growth on the same bars: \
             {:.6e} against {:.6e}",
            shrunk.policy.gross_growth,
            plain.policies[POLICY_MODEL].gross_growth
        );
        assert!(
            shrunk.policy.mean_abs_position < plain.policies[POLICY_MODEL].mean_abs_position,
            "correcting an inflated mean must take a SMALLER position"
        );
        assert!(
            shrunk.edge.mean > plain.model_edge().mean,
            "the recalibrated policy's edge over the same null has to be larger"
        );
        // Every cap on the curve is the same comparison re-clamped, and the unshrunk column
        // must reproduce the ordinary cap curve exactly rather than approximately.
        for (slot, point) in shrunk.curve.iter().enumerate() {
            assert_eq!(point.cap, CAP_GRID[slot]);
            assert_eq!(
                point.unshrunk.edge, plain.cap_curve[slot].edge,
                "the unshrunk column of the comparison is not the bench's own cap curve at \
                 cap {}",
                CAP_GRID[slot]
            );
            assert!(
                point.shrunk.mean_abs_position <= point.unshrunk.mean_abs_position + 1e-12,
                "a shrunk mean cannot ask for more leverage at cap {}",
                CAP_GRID[slot]
            );
        }
        // The PAIRED difference is the number the comparison is decided on, so it has to be
        // the difference of the two columns it sits beside rather than a third quantity, and
        // it has to be resolvable on a fixture built with a real effect.
        for (slot, point) in shrunk.curve.iter().enumerate() {
            assert!(
                (point.paired.mean - point.edge_gain()).abs() < 1e-12,
                "the paired mean {:.6e} is not the gap between the two columns {:.6e} at cap {}",
                point.paired.mean,
                point.edge_gain(),
                CAP_GRID[slot]
            );
            assert_eq!(
                point.paired.samples,
                shrunk.windows,
                "the paired unit is the WINDOW, so there is one observation per window"
            );
        }
        let headline = shrunk
            .curve
            .iter()
            .find(|point| point.cap == LEVERAGE_CAP)
            .expect("the headline cap is on the grid");
        assert!(
            headline.paired.mean > 0.0,
            "a fixture whose mean is inflated by a known 1/{beta} must gain from correcting \
             it: {:+.4e} (CI {:+.4e}..{:+.4e})",
            headline.paired.mean,
            headline.paired.ci_low,
            headline.paired.ci_high
        );
        // NOT asserted here: that the interval excludes zero. It does not, and saying so is
        // the point. 20 synthetic blocks cannot resolve a 1.6 bps effect whose per-window
        // dispersion is what this fixture builds, and a test tuned until it could would be a
        // test of the seed. `resolvable()` is pinned directly below instead, and on the real
        // panel the same 256-block interval either resolves or reports that it does not.
        assert!(
            !headline.resolvable(),
            "20 blocks resolving a 1.6 bps gain would mean the interval is too tight, which \
             is the failure mode a blocked bootstrap exists to prevent"
        );

        // Pairing is not cosmetic: it exists because the two policies share the regime. On
        // this fixture the shared term is the realized path itself, so the unpaired interval
        // on either level has to be far wider than the interval on the difference.
        assert!(
            plain.model_edge().se > 3.0 * headline.paired.se,
            "if the level's interval ({:.4e}) is not much wider than the paired one \
             ({:.4e}), pairing is buying nothing and the claim it rests on is wrong",
            plain.model_edge().se,
            headline.paired.se
        );
    }

    /// `resolvable()` is the gate the whole verdict hangs on, so its three cases are pinned
    /// directly rather than only through a fixture that happens to land in one of them.
    #[test]
    fn a_paired_gain_is_resolvable_only_when_its_interval_excludes_zero() {
        let point = |mean: f64, lo: f64, hi: f64| ShrunkPoint {
            paired: Dispersion {
                mean,
                se: (hi - lo) / 4.0,
                ci_low: lo,
                ci_high: hi,
                blocks: 256,
                samples: 256,
            },
            ..ShrunkPoint::nan()
        };
        assert!(point(2.0, 0.5, 3.5).resolvable(), "a positive band excluding zero resolves");
        assert!(point(-2.0, -3.5, -0.5).resolvable(), "so does a negative one");
        assert!(
            !point(2.0, -0.5, 4.5).resolvable(),
            "a band straddling zero does NOT resolve, however large its point estimate"
        );
        assert!(
            !ShrunkPoint::nan().resolvable(),
            "an unmeasured point cannot resolve anything"
        );
    }

    /// The two slopes describe two different failures and their COMBINATION decides the sizing.
    ///
    /// Reported directions are easy to invert, so they are pinned here: a variance slope below
    /// one is an OVERSTATED spread, above one an understated one, and the implied Kelly scale
    /// is the ratio of the two slopes.
    #[test]
    fn the_two_slopes_name_their_own_directions_and_compose_into_a_kelly_scale() {
        let fit = |beta: f64, se: f64| MzFit {
            alpha: 0.0,
            beta,
            alpha_se: se,
            beta_se: se,
            alpha_ci: (-se, se),
            beta_ci: (beta - 2.0 * se, beta + 2.0 * se),
            r2: 0.01,
            blocks: 64,
            samples: 2048,
            // Not `..Default::default()`: a zeroed `beta_block_sd` and noise floor would make
            // `slope_heterogeneous` read 0.0/0.0 as a POSITIVE finding of homogeneity on a
            // fixture that carries no block dispersion at all. Unmeasured must stay
            // unrepresentable as measured.
            ..MzFit::nan()
        };

        // The configuration this campaign actually measured: mean slope 0.36, variance slope
        // 0.24. The spread is overstated, NOT understated, and the two together leave the head
        // asking for 2/3 of the growth-optimal size despite an inflated mean.
        let measured = MeanCalibration {
            mean: fit(0.36, 0.014),
            variance: fit(0.24, 0.014),
            mean_predicted_sd: 0.003,
            gradient: VolatilityGradient::nan(),
            outer: None,
        };
        assert!(measured.spread_overstated() && !measured.spread_understated());
        assert!(
            (measured.kelly_scale() - 0.24 / 0.36).abs() < 1e-12,
            "kelly scale {}",
            measured.kelly_scale()
        );
        assert!(
            measured.kelly_scale() < 1.0,
            "an inflated mean beside a spread inflated MORE is a net under-size, and calling \
             it an over-size is the inversion this test exists to catch"
        );
        let lines = measured.report_lines();
        assert!(
            lines.iter().any(|line| line.contains("spread OVERSTATED")),
            "the console must name the direction: {lines:?}"
        );

        // The mirror case must report the other failure, and must not be reported as this one.
        let confident = MeanCalibration {
            mean: fit(0.36, 0.014),
            variance: fit(1.60, 0.014),
            mean_predicted_sd: 0.003,
            gradient: VolatilityGradient::nan(),
            outer: None,
        };
        assert!(confident.spread_understated() && !confident.spread_overstated());
        assert!(confident.kelly_scale() > 1.0);
        assert!(confident
            .report_lines()
            .iter()
            .any(|line| line.contains("spread UNDERSTATED")));

        // A slope whose interval straddles one is not a finding in either direction.
        let honest = MeanCalibration {
            mean: fit(0.36, 0.014),
            variance: fit(1.00, 0.20),
            mean_predicted_sd: 0.003,
            gradient: VolatilityGradient::nan(),
            outer: None,
        };
        assert!(!honest.spread_overstated() && !honest.spread_understated());
    }

    #[test]
    fn the_identity_recalibration_reproduces_the_untouched_policy_exactly() {
        let _torch_rng_guard = test_rng::shared();
        let latent = 12;
        let (_vs, head) = perturbed_head(latent, 0xCA11_0101);
        let supports = synthetic_supports(30_000, 0xCA11_0102);
        let returns = Tensor::from_slice(&bin_returns(&supports)).view([1, NUM_BAR_BINS]);
        let centers = Tensor::from_slice(supports.centers(DOF_R)).view([1, NUM_BAR_BINS]);
        let (windows, bars) = (2i64, 24i64);
        let h = beliefs(windows * bars, latent, 0xCA11_0103).view([windows, bars, latent]);
        let realized = Tensor::from_slice(
            &(0..windows * bars)
                .map(|slot| (0.002 * (2.0 * uniform(0xCA11_0104, slot as u64) - 1.0)) as f32)
                .collect::<Vec<f32>>(),
        )
        .view([windows, bars]);

        let law = TradedLaw::new(&returns, &centers).with_shrink(MeanShrink::identity());
        let chunk = window_paths(
            &head,
            &h,
            &realized,
            &law,
            marginal_position(&supports, FREE_LEVERAGE),
            LEVERAGE_CAP,
        )
        .expect("paths");
        for window in &chunk.windows {
            let shrunk = window
                .free_shrunk
                .as_ref()
                .expect("a shrink was requested, so every window carries one");
            assert_eq!(
                *shrunk, window.free,
                "the identity recalibration shifts nothing, so it must reproduce the \
                 untouched solve bit for bit"
            );
            assert!(
                window.has_moments(),
                "the pass must report one conditional mean and variance per bar"
            );
            assert!(
                window.predicted_var.iter().all(|v| *v > 0.0),
                "a 128-bin predictive law has strictly positive variance"
            );
            // The conditional mean lives inside the support, which is the cheapest available
            // check that it is a mean of `r` and not of the simple return.
            let (lo, hi) = (
                supports.centers(DOF_R)[0],
                supports.centers(DOF_R)[NUM_BAR_BINS as usize - 1],
            );
            assert!(
                window.predicted_mean.iter().all(|m| *m >= lo && *m <= hi),
                "a conditional mean outside the bin centers it averages is not a mean"
            );
        }
    }

    #[test]
    fn a_measured_bench_reports_its_calibration_and_a_synthetic_one_refuses_to() {
        // `fixture_windows` carries no conditional moments, so the calibration block has to
        // come back unmeasured rather than fitted on nothing.
        let windows = fixture_windows(6, 32, 0xCA11_0201);
        let blocks: Vec<u64> = (0..windows.len() as u64).map(|w| w / 2).collect();
        let plain = bench(
            &windows,
            &blocks,
            &TailCounts::empty(),
            BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 1.7),
        );
        assert!(!plain.calibration.measured());
        assert!(plain.calibration.shrink().is_none());
        assert!(plain
            .calibration
            .report_lines()
            .iter()
            .any(|line| line.contains("not measured")));

        // A fixture that DOES carry them reports a fit, and the fit sees the inflation the
        // fixture was built with. The interval is wide here on purpose: 512 bars in 4 blocks
        // is a small panel, and the point of this test is that the pipeline from a window's
        // conditional moments through to a slope is wired up. The estimator's accuracy is
        // pinned separately, on a panel large enough to pin it.
        let measured = shrunk_fixture(0.7, 8, 64);
        let calibration = mean_calibration(&measured, &blocks_for(&measured));
        assert!(calibration.measured());
        assert_eq!(calibration.mean.samples, 8 * 64);
        assert_eq!(calibration.variance.samples, 8 * 64);
        assert!(
            calibration.mean.beta.is_finite() && calibration.mean.beta_se.is_finite(),
            "the fixture's conditional mean varies per bar, so the regression is well posed"
        );
        assert!(
            (calibration.mean.beta - 0.7).abs() < 4.0 * calibration.mean.beta_se,
            "the end-to-end fit came back at {:.4} +/- {:.4} on a panel built with 0.7",
            calibration.mean.beta,
            calibration.mean.beta_se
        );
        assert!(
            calibration.shrink().is_some(),
            "a finite fit has to yield a recalibration"
        );
    }

    fn blocks_for(windows: &[WindowPaths]) -> Vec<u64> {
        (0..windows.len() as u64).map(|w| w / 2).collect()
    }

    // -----------------------------------------------------------------------
    // The no-trade band
    // -----------------------------------------------------------------------

    /// A target path that churns: iid uniform over the whole capped range, so the
    /// bar-to-bar move is spread across `[0, 2 cap]` and a band in the middle of that
    /// distribution has something to freeze. An alternating `+/-cap` path would have every
    /// move at exactly `2 cap` and no band below the freezing width would bite, which is
    /// correct behaviour but pins nothing.
    fn churning_targets(bars: usize, seed: u64) -> Vec<f64> {
        (0..bars)
            .map(|bar| LEVERAGE_CAP * (2.0 * uniform(seed, bar as u64) - 1.0))
            .collect()
    }

    #[test]
    fn every_shape_at_its_incumbent_knob_is_the_identity() {
        let target = churning_targets(500, 0xBA0D);
        for shape in SIZING_SHAPES {
            assert_eq!(
                shape.positions(&target, shape.knobs()[INCUMBENT_SLOT], LEVERAGE_CAP),
                target,
                "[{}] at its incumbent knob must return the every-bar re-solve untouched",
                shape.name()
            );
        }
    }

    #[test]
    fn every_shape_at_its_frozen_knob_never_leaves_flat() {
        let target = churning_targets(500, 0xBA0E);
        for shape in SIZING_SHAPES {
            let held = shape.positions(&target, shape.knobs()[FROZEN_SLOT], LEVERAGE_CAP);
            assert!(
                held.iter().all(|f| *f == 0.0),
                "[{}] at knob {} still left flat",
                shape.name(),
                shape.knobs()[FROZEN_SLOT]
            );
        }
    }

    /// The MECHANISM behind reflection losing on this panel: when the target is pinned at the
    /// cap, reflection de-levers and jump-to-target does not.
    ///
    /// Measured on `pretrain_step_9728` at the 4x cap, mean `|f|` holds at `3.869` across the
    /// whole jump-to-target grid while reflection walks it down to `3.492` at a band of `0.100`
    /// cap and `1.964` at `0.500`. That is leverage removed rather than churn removed, and on a
    /// book with positive expected growth the growth forgone swamps the cost saved.
    ///
    /// Pinned here structurally, because the theoretical claim that reflection dominates is the
    /// kind a future reader re-derives and re-applies, and the condition it needs - an INTERIOR
    /// frictionless optimum - is exactly the condition a binding [`LEVERAGE_CAP`] destroys.
    #[test]
    fn reflection_de_levers_a_capped_book_and_jumping_to_target_does_not() {
        // The regime that matters: the target alternates between the two cap boundaries, which
        // is what a book whose cap binds on 74-93% of bars actually does.
        let target: Vec<f64> = (0..400)
            .map(|bar| {
                if (bar / 7) % 2 == 0 {
                    LEVERAGE_CAP
                } else {
                    -LEVERAGE_CAP
                }
            })
            .collect();
        let mean_abs = |path: &[f64]| path.iter().map(|f| f.abs()).sum::<f64>() / path.len() as f64;
        let reference = mean_abs(&target);
        for slot in 1..FROZEN_SLOT {
            let jump = SizingShape::BandToTarget.positions(
                &target,
                SizingShape::BandToTarget.knobs()[slot],
                LEVERAGE_CAP,
            );
            let reflect = SizingShape::BandReflect.positions(
                &target,
                SizingShape::BandReflect.knobs()[slot],
                LEVERAGE_CAP,
            );
            // Jump-to-target either holds or lands exactly on the target, and the target is at
            // the cap on every bar, so its exposure is the target's exposure whenever it moved.
            assert!(
                jump.iter().all(|f| f.abs() == LEVERAGE_CAP || *f == 0.0),
                "[slot {slot}] jump-to-target left a position strictly inside the cap, so it is \
                 no longer landing on the target"
            );
            assert!(
                mean_abs(&reflect) < reference,
                "[slot {slot}] reflection failed to de-lever a cap-pinned book: {} against the \
                 target's {reference}",
                mean_abs(&reflect)
            );
            assert!(
                mean_abs(&reflect) <= mean_abs(&jump) + 1e-12,
                "[slot {slot}] reflection held MORE exposure than jump-to-target: {} against {}",
                mean_abs(&reflect),
                mean_abs(&jump)
            );
        }
    }

    /// Reflection is the whole content of the impulse-control form, so it is asserted
    /// directly: on every breach the move is exactly `|wanted| - band`, and the landing point
    /// is between the previous holding and the target rather than through it.
    #[test]
    fn the_band_shapes_respect_the_cap_and_reflection_never_overshoots() {
        let target = churning_targets(500, 0xBA0F);
        for shape in [SizingShape::BandToTarget, SizingShape::BandReflect] {
            for fraction in shape.knobs() {
                let band = fraction * LEVERAGE_CAP;
                let held = shape.positions(&target, fraction, LEVERAGE_CAP);
                let mut previous = 0.0f64;
                for (bar, (position, want)) in held.iter().zip(&target).enumerate() {
                    assert!(
                        position.abs() <= LEVERAGE_CAP,
                        "[{}] band {fraction} left the cap at bar {bar}: {position}",
                        shape.name()
                    );
                    let wanted = want - previous;
                    if wanted.abs() <= band {
                        assert_eq!(
                            *position, previous,
                            "[{}] band {fraction} traded a move of {wanted} from inside the \
                             dead zone at bar {bar}",
                            shape.name()
                        );
                    } else {
                        let moved = position - previous;
                        assert!(
                            moved * wanted > 0.0,
                            "[{}] band {fraction} moved {moved} against a wanted {wanted} at \
                             bar {bar}",
                            shape.name()
                        );
                        match shape {
                            SizingShape::BandToTarget => assert_eq!(*position, *want),
                            SizingShape::BandReflect => {
                                assert!(
                                    (moved.abs() - (wanted.abs() - band)).abs() < 1e-12,
                                    "reflection at band {fraction} moved {moved} where {} was \
                                     due at bar {bar}",
                                    wanted.abs() - band
                                );
                                assert!(
                                    (position - want).abs() <= band + 1e-12,
                                    "reflection landed {position} further than one band from \
                                     {want} at bar {bar}"
                                );
                            }
                            SizingShape::PartialAdjust => unreachable!("band shapes only"),
                        }
                    }
                    previous = *position;
                }
            }
        }
    }

    /// Partial adjustment has NO dead zone: it moves on every bar the target moved at all,
    /// by exactly `lambda` of the distance. That is the property that separates it from a
    /// band, and it is the reason it can preserve a continuous magnitude signal a band
    /// destroys.
    #[test]
    fn partial_adjustment_moves_a_fixed_fraction_of_the_way_with_no_dead_zone() {
        let target = churning_targets(500, 0xBA12);
        let shape = SizingShape::PartialAdjust;
        assert!(!shape.has_dead_zone());
        for lambda in shape.knobs() {
            let held = shape.positions(&target, lambda, LEVERAGE_CAP);
            let mut previous = 0.0f64;
            for (bar, (position, want)) in held.iter().zip(&target).enumerate() {
                assert!(
                    (position - (previous + lambda * (want - previous))).abs() < 1e-12,
                    "lambda {lambda} landed {position} where {} was due at bar {bar}",
                    previous + lambda * (want - previous)
                );
                assert!(
                    position.abs() <= LEVERAGE_CAP + 1e-12,
                    "lambda {lambda} left the cap at bar {bar}: {position}"
                );
                if lambda > 0.0 && (want - previous).abs() > 0.0 {
                    assert_ne!(
                        *position, previous,
                        "partial adjustment must have no dead zone, but lambda {lambda} froze \
                         a wanted move of {} at bar {bar}",
                        want - previous
                    );
                }
                previous = *position;
            }
        }
    }

    /// A knob further from the incumbent must not trade MORE, which is the entire economic
    /// premise of the sweep.
    ///
    /// A fixture property rather than a theorem: freezing or lagging a position can leave a
    /// larger move due later, so monotonicity is not provable for an arbitrary target path.
    /// It is asserted on a deterministic churning fixture, the same footing
    /// [`super::super::portfolio`]'s band test stands on.
    #[test]
    fn a_knob_further_from_the_incumbent_does_not_trade_more() {
        let target = churning_targets(2000, 0xBA10);
        for shape in SIZING_SHAPES {
            let mut previous = f64::INFINITY;
            for knob in shape.knobs() {
                let held = shape.positions(&target, knob, LEVERAGE_CAP);
                let traded: f64 = std::iter::once(held[0].abs())
                    .chain(held.windows(2).map(|pair| (pair[1] - pair[0]).abs()))
                    .sum();
                assert!(
                    traded <= previous + 1e-9,
                    "[{}] knob {knob} traded {traded} against the previous knob's {previous}",
                    shape.name()
                );
                previous = traded;
            }
            assert!(
                previous < 1e-12,
                "[{}] the frozen knob should have traded nothing, but it traded {previous}",
                shape.name()
            );
        }
    }

    // -----------------------------------------------------------------------
    // The myopic cost-aware FOC
    // -----------------------------------------------------------------------

    /// A three-point law whose cost-free optimum lands STRICTLY INSIDE the cap.
    ///
    /// That is the whole requirement, and the first version of this fixture failed it: with
    /// `f*` above the cap every solve returns the cap, the inaction region is the entire
    /// range at any cost, and the branch tests pass while testing nothing. Here `g'` is
    /// `+1.6e-3` at flat and crosses zero near `3.8`, so a bench-scale cost carves an
    /// inaction region with the range straddling it on both sides.
    fn myopic_law() -> (Vec<f64>, Vec<f64>) {
        (vec![0.52, 0.28, 0.20], vec![0.02, -0.01, -0.03])
    }

    fn myopic_scalar(probs: &[f64], returns: &[f64], cap: f64, cost: f64, previous: f64) -> f64 {
        let outcomes = probs.len() as i64;
        myopic_fractions(
            &Tensor::from_slice(probs).view([1, outcomes]),
            &Tensor::from_slice(returns).view([1, outcomes]),
            cap,
            cost,
            &Tensor::from_slice(&[previous]),
        )
        .double_value(&[0])
    }

    /// The check the whole construction is built to satisfy: with no cost there is no kink and
    /// no cost slope, so the myopic solve must be the cost-blind solve. Bit for bit, not
    /// nearly — a sign error in the cost slope cannot survive this.
    #[test]
    fn the_myopic_solve_is_the_cost_blind_solve_at_zero_cost() {
        let (probs, returns) = myopic_law();
        let blind = kelly_fraction(&probs, &returns, LEVERAGE_CAP);
        for previous in [-LEVERAGE_CAP, -1.0, 0.0, 1.0, 2.5, LEVERAGE_CAP] {
            assert_eq!(
                myopic_scalar(&probs, &returns, LEVERAGE_CAP, 0.0, previous),
                blind,
                "at zero cost the held position cannot matter, but holding {previous} moved it"
            );
        }
    }

    /// The limit approached from above, and the RATE at which it is approached.
    ///
    /// The first version of this test demanded the gap be under `1e-6` at `0.001` bps and it
    /// failed at `2.3e-4` - correctly. The myopic optimum sits where `g'(f) = c/(1 - c(f-held))`,
    /// so to first order it is displaced from the cost-blind root by `c / |g''(f*)|`, and this
    /// law's `g''` is about `-4.3e-4`: a cost of `1e-7` moves the optimum by `2.3e-4` and no
    /// tolerance on the displacement alone is meaningful. The displacement LAW is, and it is a
    /// far stronger statement than any tolerance, so the test asserts that instead: halving the
    /// cost must halve the gap.
    ///
    /// It also says something the panel needs. `|g''|` this small means the objective is nearly
    /// FLAT near its optimum, so the sizing is weakly determined by the law and strongly
    /// determined by whatever else touches it - the cap, the cost, the recalibration. That is
    /// the same fact as "the cap binds on 74-93% of bars", seen from the solver's side.
    #[test]
    fn the_myopic_gap_to_the_cost_blind_solve_is_first_order_in_the_cost() {
        let (probs, returns) = myopic_law();
        let blind = kelly_fraction(&probs, &returns, LEVERAGE_CAP);
        let gap_at = |bps: f64| (myopic_scalar(&probs, &returns, LEVERAGE_CAP, bps * 1e-4, 0.0) - blind).abs();
        let mut previous = (f64::INFINITY, f64::INFINITY);
        for bps in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let gap = gap_at(bps);
            assert!(
                gap < previous.1,
                "the gap grew from {} to {gap} as the cost fell to {bps} bps",
                previous.1
            );
            if previous.0.is_finite() {
                // Each step halves the cost, so each step must halve the gap. The tolerance is
                // on the RATIO, which is what the first-order law predicts; second-order terms
                // are what keep it from being exact.
                let ratio = gap / previous.1;
                assert!(
                    (ratio - 0.5).abs() < 0.05,
                    "halving the cost from {} to {bps} bps changed the gap by {ratio}, not by \
                     the first-order 0.5",
                    previous.0
                );
            }
            previous = (bps, gap);
        }
        // And the displacement is the predicted `c / |g''|` rather than an arbitrary number.
        let slope = |f: f64| -> f64 {
            probs
                .iter()
                .zip(&returns)
                .map(|(p, r)| p * r / (1.0 + f * r))
                .sum::<f64>()
        };
        let curvature = (slope(blind + 1e-3) - slope(blind - 1e-3)) / 2e-3;
        let predicted = 1.0e-4 / curvature.abs();
        let measured = gap_at(1.0);
        assert!(
            (measured / predicted - 1.0).abs() < 0.05,
            "the displacement at 1 bp was {measured}, against the first-order prediction \
             c/|g''| = {predicted}"
        );
    }

    /// The INACTION REGION, which is the property no post-processing of a cost-blind `f*` can
    /// have: it emerges from the kink, and its width is set by the cost.
    ///
    /// `|g'(f_prev)| <= c` is the exact condition, so the test states it in those terms rather
    /// than by eyeballing a width: at a holding where the cost-free slope is small the solve
    /// must return the holding UNCHANGED, and at one where it is large it must move.
    #[test]
    fn the_myopic_solve_holds_still_exactly_inside_the_subgradient_interval() {
        let (probs, returns) = myopic_law();
        // The bench's own default, so the region under test is the one the pass actually solves.
        let cost = DEFAULT_COST_BPS * 1e-4;
        let slope = |f: f64| -> f64 {
            probs
                .iter()
                .zip(&returns)
                .map(|(p, r)| p * r / (1.0 + f * r))
                .sum()
        };
        let mut held_still = 0usize;
        let mut moved = 0usize;
        for step in 0..=40 {
            let previous = -LEVERAGE_CAP + 2.0 * LEVERAGE_CAP * step as f64 / 40.0;
            let solved = myopic_scalar(&probs, &returns, LEVERAGE_CAP, cost, previous);
            if slope(previous).abs() <= cost {
                assert!(
                    (solved - previous).abs() < 1e-9,
                    "the cost-free slope at {previous} is {:.3e}, inside +/-{cost}, so the \
                     myopic optimum is the holding itself, but it solved to {solved}",
                    slope(previous)
                );
                held_still += 1;
            } else {
                // Outside the interval it must move, and it must move TOWARD the cost-free
                // optimum: the cost term only ever pulls the target back, never past it.
                assert!(
                    (solved - previous).abs() > 0.0,
                    "the cost-free slope at {previous} is {:.3e}, outside +/-{cost}, so the \
                     solve had to move",
                    slope(previous)
                );
                assert!(
                    (solved - previous) * slope(previous) > 0.0,
                    "moved from {previous} to {solved} against a slope of {:.3e}",
                    slope(previous)
                );
                moved += 1;
            }
        }
        assert!(
            held_still > 0 && moved > 0,
            "the sweep of holdings must straddle the inaction region to test anything: {} \
             still, {} moved",
            held_still,
            moved
        );
    }

    /// A wider cost must widen the inaction region, which is the statement "the region's width
    /// is cost-dependent" made falsifiable.
    #[test]
    fn a_larger_cost_widens_the_emergent_inaction_region() {
        let (probs, returns) = myopic_law();
        let holdings: Vec<f64> = (0..=80)
            .map(|step| -LEVERAGE_CAP + 2.0 * LEVERAGE_CAP * step as f64 / 80.0)
            .collect();
        let mut previous_still = 0usize;
        for bps in [1.0, 5.0, 20.0, 80.0, 300.0] {
            let still = holdings
                .iter()
                .filter(|previous| {
                    let solved =
                        myopic_scalar(&probs, &returns, LEVERAGE_CAP, bps * 1e-4, **previous);
                    (solved - **previous).abs() < 1e-9
                })
                .count();
            assert!(
                still >= previous_still,
                "the inaction region shrank from {previous_still} to {still} holdings when the \
                 cost ROSE to {bps} bps"
            );
            previous_still = still;
        }
        assert!(
            previous_still > holdings.len() / 2,
            "a 300 bps cost should freeze most of the range, froze {previous_still} of {}",
            holdings.len()
        );
    }

    /// The cost term's own domain, `|f - f_prev| < 1/c`, is enforced rather than assumed away.
    ///
    /// At the bench's default cost it is thousands of times the cap and cannot bind, but the
    /// break-even search drives the cost to [`MAX_BREAK_EVEN_BPS`], where `1/c` is `10` and
    /// sits INSIDE [`MAX_LEVERAGE`]. A solve that ignored it would take the log of a negative
    /// number.
    #[test]
    fn the_cost_domain_binds_before_the_ruin_ceiling_at_extreme_costs() {
        let (probs, returns) = myopic_law();
        let cost = MAX_BREAK_EVEN_BPS * 1e-4;
        assert!(
            1.0 / cost < MAX_LEVERAGE,
            "this test is only meaningful when the cost domain is tighter than the ceiling"
        );
        for previous in [-8.0, 0.0, 8.0] {
            let solved = myopic_scalar(&probs, &returns, FREE_LEVERAGE, cost, previous);
            assert!(
                solved.is_finite(),
                "an extreme cost produced a non-finite fraction from a holding of {previous}"
            );
            assert!(
                (solved - previous).abs() <= 1.0 / cost,
                "the solve moved {} from {previous}, further than the cost domain's {}",
                (solved - previous).abs(),
                1.0 / cost
            );
        }
    }

    /// The batched path and the scalar path are one implementation, so a row's answer cannot
    /// depend on what it was batched with.
    #[test]
    fn the_batched_myopic_solve_agrees_with_the_scalar_one() {
        let (probs, returns) = myopic_law();
        let holdings = [-3.0, -0.5, 0.0, 0.75, 2.0, 4.0];
        let cost = 15.0e-4;
        let rows = holdings.len() as i64;
        let outcomes = probs.len() as i64;
        let batched = host_vec(&myopic_fractions(
            &Tensor::from_slice(&probs)
                .view([1, outcomes])
                .expand([rows, outcomes], false),
            &Tensor::from_slice(&returns).view([1, outcomes]),
            LEVERAGE_CAP,
            cost,
            &Tensor::from_slice(&holdings),
        ));
        for (previous, expected) in holdings.iter().zip(&batched) {
            assert_eq!(
                myopic_scalar(&probs, &returns, LEVERAGE_CAP, cost, *previous),
                *expected,
                "the batched myopic solve disagreed with the scalar one at {previous}"
            );
        }
    }

    /// The myopic solve attains a HIGHER value of its own objective than the cost-blind
    /// target does, which is the whole claim: the cost-blind fraction is not optimal for an
    /// objective that charges for reaching it.
    #[test]
    fn the_myopic_solve_beats_the_cost_blind_target_on_the_cost_aware_objective() {
        let (probs, returns) = myopic_law();
        let blind = kelly_fraction(&probs, &returns, LEVERAGE_CAP);
        for bps in [2.0, 10.0, 40.0] {
            let cost = bps * 1e-4;
            for previous in [-2.0, -0.25, 0.0, 1.0, 3.0] {
                let value = |f: f64| {
                    expected_log_growth(&probs, &returns, f)
                        + (1.0 - cost * (f - previous).abs()).max(WEALTH_FLOOR).ln()
                };
                let solved = myopic_scalar(&probs, &returns, LEVERAGE_CAP, cost, previous);
                assert!(
                    value(solved) >= value(blind) - 1e-12,
                    "at {bps} bps from a holding of {previous} the myopic solve scored {:.6e} \
                     against the cost-blind target's {:.6e}",
                    value(solved),
                    value(blind)
                );
                // And it never asks for more leverage than the cost-blind solve: the cost
                // term is a penalty on distance travelled, so it can only pull the target in.
                assert!(
                    solved.abs() <= blind.abs().max(previous.abs()) + 1e-9,
                    "the cost-aware solve asked for {solved} where cost-blind wanted {blind} \
                     from a holding of {previous}"
                );
            }
        }
    }

    /// A ZERO-PROBABILITY bin whose `-1/R` lands exactly on a bisection midpoint used to make
    /// the cost-blind slope `0 * R / 0`, i.e. NaN, which turned the whole row's sign test
    /// false and collapsed the bisection onto its lower bracket end.
    ///
    /// Constructed rather than hoped for: the live bins bound nothing inside the cap, so the
    /// bracket is exactly `[-cap, cap]` and the second midpoint is `+cap/2 = 2.0`; a dead bin
    /// at `R = -0.5` has `1 + 2 R = 0` there. The uncapped optimum of the live pair is `25`,
    /// so the correct answer is the cap and the pre-guard answer was `2.0` — a 2x sizing
    /// error, silent, on any bar whose law underflowed a bin to zero.
    #[test]
    fn a_zero_probability_bin_cannot_poison_the_slope() {
        let probs = [0.5, 0.5, 0.0];
        let returns = [0.02, -0.01, -0.5];
        let solved = kelly_fraction(&probs, &returns, LEVERAGE_CAP);
        assert!(
            solved.is_finite(),
            "a zero-mass bin produced a non-finite fraction: {solved}"
        );
        assert!(
            (solved - LEVERAGE_CAP).abs() < 1e-9,
            "the live pair's optimum is 25x, so the cap must bind; got {solved}"
        );
        // The dead bin carries no mass, so deleting it entirely must change nothing.
        assert_eq!(
            solved,
            kelly_fraction(&probs[..2], &returns[..2], LEVERAGE_CAP),
            "a zero-mass bin changed the answer, so it was contributing to the sum"
        );
        // Same guard on the myopic path, which shares the hazard.
        let myopic = myopic_scalar(&probs, &returns, LEVERAGE_CAP, 5.0e-4, 0.0);
        assert!(
            myopic.is_finite() && myopic > 0.0,
            "the myopic solve was poisoned by the same zero-mass bin: {myopic}"
        );
    }

    /// The band-zero row of the sweep IS the headline bench's model row, bit for bit.
    ///
    /// The one assertion that makes the sweep a measurement of the incumbent policy rather
    /// than of a re-implementation of it: every figure on that row has to come out of the
    /// same ledger, the same cost charge and the same null the headline is quoted from, or
    /// the paired gains below it are differences between two different accountings.
    #[test]
    fn the_unbanded_row_reproduces_the_headline_bench_exactly() {
        let windows = shrunk_fixture(0.7, 40, 64);
        let blocks = blocks_for(&windows);
        let config = BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 0.5);
        let plain = bench(&windows, &blocks, &TailCounts::empty(), config);
        for shape in SIZING_SHAPES {
            let sweep = band_sweep(&windows, &blocks, config, BandSource::Frictionless, shape)
                .expect("the fixture carries a solved fraction for every bar");
            let row = &sweep.points[INCUMBENT_SLOT];
            let model = &plain.policies[POLICY_MODEL];
            assert_eq!(row.knob, shape.knobs()[INCUMBENT_SLOT]);
            assert_eq!(sweep.bars, plain.bars, "the sweep must trade the same bars");
            assert_eq!(row.policy.net_growth, model.net_growth);
            assert_eq!(row.policy.gross_growth, model.gross_growth);
            assert_eq!(row.policy.turnover, model.turnover);
            assert_eq!(row.policy.hit_rate, model.hit_rate);
            assert_eq!(row.policy.sharpe, model.sharpe);
            assert_eq!(row.edge.mean, plain.model_edge().mean);
            assert_eq!(row.break_even_bps, plain.model_break_even());
            // Paired against itself, so exactly zero rather than nearly zero.
            assert_eq!(row.gain.mean, 0.0);
            assert_eq!(row.turnover_share, 1.0);
        }
    }

    /// `gross` is the identical positions with the charge switched off, so its growth must be
    /// the net row's own pre-cost column and its Sharpe must be the number `PolicyStats`
    /// cannot otherwise state.
    #[test]
    fn the_gross_column_is_the_same_positions_at_zero_cost() {
        let windows = shrunk_fixture(0.7, 20, 64);
        let blocks = blocks_for(&windows);
        let config = BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 0.5);
        let sweep = band_sweep(
            &windows,
            &blocks,
            config,
            BandSource::Frictionless,
            SizingShape::BandToTarget,
        )
        .expect("fixture");
        for point in &sweep.points {
            assert_eq!(
                point.gross.net_growth, point.policy.gross_growth,
                "the zero-cost ledger and the gross column must be one number"
            );
            assert_eq!(point.gross.turnover, point.policy.turnover);
            assert!(
                point.policy.net_growth <= point.gross.net_growth + 1e-15,
                "charging a cost cannot raise net growth: {} against {}",
                point.policy.net_growth,
                point.gross.net_growth
            );
            if point.policy.turnover > 0.0 {
                assert!(
                    point.policy.net_growth < point.gross.net_growth,
                    "a book that traded must have paid something"
                );
            }
        }
    }

    /// `break_even_bps` is the cost at which the policy's edge over the null CROSSES ZERO, so
    /// re-running the sweep at that cost has to produce an edge of zero on the same row.
    ///
    /// # What this test used to assert, and why that was wrong
    ///
    /// It asserted the intuition the whole cost-aware axis was built on: break-even is gross
    /// edge over turnover, so a shape that removes turnover while keeping any gross edge must
    /// tolerate MORE cost. It failed, and it failed for a real reason, not a fixture artifact:
    /// on this fixture the `0.500x` band cut turnover to `0.949` of unbanded and LOWERED
    /// break-even from `37.5403` to `36.8038` bps. The band destroyed gross edge faster than it
    /// destroyed turnover.
    ///
    /// That is a structural property of a dead zone, not a coincidence. A dead zone freezes the
    /// SMALLEST target moves by construction. If the edge lives in the magnitude of the
    /// position rather than in its sign - which is exactly this checkpoint's situation - then
    /// the smallest moves are not churn to be suppressed, they are the signal at low amplitude,
    /// and suppressing them first is the worst available order. So "trade less to raise
    /// break-even" is not free and is not even signed. It is a measurement, which is what the
    /// swept panel is for.
    ///
    /// The invariant that IS true is the one below, and it is the one worth defending: the
    /// reported break-even must be the root of the reported edge curve, or the column is not a
    /// break-even at all.
    #[test]
    fn the_reported_break_even_is_the_root_of_the_edge_curve() {
        let windows = shrunk_fixture(0.7, 40, 64);
        let blocks = blocks_for(&windows);
        let sweep_at = |bps: f64| {
            band_sweep(
                &windows,
                &blocks,
                BenchConfig::new(bps, LEVERAGE_CAP, 0.5),
                BandSource::Frictionless,
                SizingShape::BandToTarget,
            )
            .expect("fixture")
        };
        let sweep = sweep_at(DEFAULT_COST_BPS);
        let crossing = sweep.points[INCUMBENT_SLOT].break_even_bps;
        assert!(
            crossing.is_finite() && crossing > 0.0,
            "the incumbent must have a finite break-even for this test to have a subject"
        );
        let edge_at =
            |bps: f64| sweep_at(bps).points[INCUMBENT_SLOT].edge.mean;
        let below = edge_at(0.5 * crossing);
        let at = edge_at(crossing);
        let above = edge_at(2.0 * crossing);
        assert!(
            below > 0.0,
            "at half the break-even cost the policy must still be ahead of the null, got {below}"
        );
        assert!(
            above < 0.0,
            "at twice the break-even cost the policy must be behind the null, got {above}"
        );
        // The bootstrap means windows and the root search means bars; the fixture's windows are
        // all the same length, so the two coincide and the residual is solver tolerance only.
        assert!(
            at.abs() < 1e-9,
            "the edge at the reported break-even of {crossing} bps was {at}, not zero"
        );
        assert!(
            sweep.best_break_even().is_some(),
            "a sweep with a finite break-even somewhere must name the knob that maximizes it"
        );
        assert!(
            sweep
                .best_break_even()
                .is_some_and(|slot| sweep.points[slot].policy.turnover > 0.0),
            "the frozen book must not win the break-even column"
        );
    }

    /// A book that never trades pays nothing, so its growth cannot depend on the cost.
    ///
    /// The frozen anchor exists at the far end of every knob grid precisely so this is
    /// checkable, and it is the one row on the panel whose cost sensitivity is known a priori.
    #[test]
    fn the_frozen_book_is_cost_independent() {
        let windows = shrunk_fixture(0.7, 40, 64);
        let blocks = blocks_for(&windows);
        for shape in SIZING_SHAPES {
            let sweep = |bps: f64| {
                band_sweep(
                    &windows,
                    &blocks,
                    BenchConfig::new(bps, LEVERAGE_CAP, 0.5),
                    BandSource::Frictionless,
                    shape,
                )
                .expect("fixture")
                .points[FROZEN_SLOT]
                .policy
            };
            let cheap = sweep(0.0);
            let dear = sweep(MAX_BREAK_EVEN_BPS);
            assert_eq!(
                cheap.turnover, 0.0,
                "[{}] the frozen knob traded {}",
                shape.name(),
                cheap.turnover
            );
            assert_eq!(
                cheap.net_growth,
                dear.net_growth,
                "[{}] a book that never trades changed its growth between 0 and {} bps",
                shape.name(),
                MAX_BREAK_EVEN_BPS
            );
        }
    }

    /// The interaction is the second difference of the SAME per-window growths the two sweeps
    /// report, so it cannot disagree with them.
    #[test]
    fn the_band_shrink_interaction_is_the_difference_of_the_two_gains() {
        let windows = shrunk_fixture(0.7, 40, 64);
        let blocks = blocks_for(&windows);
        let config = BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 0.5);
        let overlap = band_shrink_overlap(&windows, &blocks, config, SizingShape::BandToTarget)
            .expect("the fixture carries a recalibrated fraction for every bar");
        assert_eq!(overlap.len(), BAND_FRACTIONS.len());

        let plain = band_sweep(&windows, &blocks, config, BandSource::Frictionless, SizingShape::BandToTarget)
            .expect("fixture");
        let shrunk = band_sweep(&windows, &blocks, config, BandSource::Recalibrated, SizingShape::BandToTarget)
            .expect("fixture");
        for (slot, point) in overlap.iter().enumerate() {
            assert_eq!(point.knob, BAND_FRACTIONS[slot]);
            assert_eq!(
                point.gain_plain.mean, plain.points[slot].gain.mean,
                "the overlap's as-solved gain must be the sweep's own at band {}",
                point.knob
            );
            assert_eq!(
                point.gain_shrunk.mean, shrunk.points[slot].gain.mean,
                "the overlap's recalibrated gain must be the sweep's own at band {}",
                point.knob
            );
            // The bootstrap is linear in the resampled vector and both arms share a seed and
            // a block map, so the interaction's MEAN is exactly the difference of the means.
            assert!(
                (point.interaction.mean - (point.gain_shrunk.mean - point.gain_plain.mean)).abs()
                    < 1e-18,
                "the interaction {:.6e} is not the second difference {:.6e} at band {}",
                point.interaction.mean,
                point.gain_shrunk.mean - point.gain_plain.mean,
                point.knob
            );
        }
        let zero = &overlap[INCUMBENT_SLOT];
        assert_eq!(zero.gain_plain.mean, 0.0);
        assert_eq!(zero.gain_shrunk.mean, 0.0);
        assert_eq!(zero.interaction.mean, 0.0);
        assert!(
            zero.captured_by_shrink.is_nan(),
            "a share of a zero gain is not interpretable and must not print as a number"
        );
    }

    /// A window set with no recalibrated fraction yields no recalibrated sweep and no
    /// overlap, which is the normal case rather than an error.
    #[test]
    fn a_pass_without_a_recalibration_reports_no_recalibrated_band() {
        let windows = fixture_windows(8, 32, 0xBA11);
        let blocks = blocks_for(&windows);
        let config = BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 1.7);
        assert!(
            band_sweep(&windows, &blocks, config, BandSource::Frictionless, SizingShape::BandToTarget)
                .is_some(),
            "every window carries its own solved fraction"
        );
        assert!(
            band_sweep(&windows, &blocks, config, BandSource::Recalibrated, SizingShape::BandToTarget)
                .is_none()
        );
        assert!(band_shrink_overlap(&windows, &blocks, config, SizingShape::BandToTarget).is_none());
        assert!(band_sweep(&[], &blocks, config, BandSource::Frictionless, SizingShape::BandToTarget)
            .is_none());
    }

    /// Six of these blocks land in one log - three shapes on two sources - so every row has to
    /// name which shape and which fraction it belongs to. A block that does not is unreadable
    /// beside the other five, and this project's retraction history is mostly numbers that were
    /// read against the wrong construction.
    #[test]
    fn the_band_report_names_every_row_and_its_verdict() {
        let windows = shrunk_fixture(0.7, 40, 64);
        let blocks = blocks_for(&windows);
        let config = BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 0.5);
        for shape in SIZING_SHAPES {
            for source in [BandSource::Frictionless, BandSource::Recalibrated] {
                let sweep = band_sweep(&windows, &blocks, config, source, shape)
                    .expect("the fixture carries both fractions");
                let lines = sweep.report_lines();
                assert_eq!(lines.len(), SIZING_KNOBS + 2);
                assert!(
                    lines[0].contains(shape.name())
                        && lines[0].contains(source.name())
                        && lines[0].contains(shape.knob_name()),
                    "the header must name the shape, the fraction and what its knob means: {}",
                    lines[0]
                );
                assert!(
                    lines[1].contains("incumbent"),
                    "the incumbent row must say so rather than claim resolvability: {}",
                    lines[1]
                );
                assert!(
                    lines
                        .last()
                        .expect("verdict")
                        .contains("maximizing break-even"),
                    "{:?}",
                    lines.last()
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Where the edge lives
    // -----------------------------------------------------------------------

    fn attribution_config() -> BenchConfig {
        BenchConfig::new(DEFAULT_COST_BPS, LEVERAGE_CAP, 1.7)
    }

    /// The two undamaged arms must be the SAME objects the headline bench scores, or the
    /// attribution is a second implementation of the ledger and its arms are not comparable to
    /// the number the session quotes.
    #[test]
    fn the_undamaged_arms_reproduce_the_bench_they_are_quoted_against() {
        let windows = fixture_windows(16, 48, 0xED9E);
        let blocks = blocks_for(&windows);
        let config = attribution_config();
        let bench = bench(&windows, &blocks, &TailCounts::empty(), config);
        let split = edge_attribution(&windows, &blocks, config);

        let actual = &split.arms[ATTRIBUTION_ACTUAL];
        assert_eq!(actual.policy.net_growth, bench.policies[POLICY_MODEL].net_growth);
        assert_eq!(actual.policy.hit_rate, bench.policies[POLICY_MODEL].hit_rate);
        assert_eq!(actual.policy.turnover, bench.policies[POLICY_MODEL].turnover);
        assert_eq!(actual.edge, bench.model_edge());
        // The fixture's fractions are noise, so both break-evens are `NAN` — "there was never
        // an edge for a cost to remove" — and `NAN != NAN`. What has to match is the bit
        // pattern, not the ordering.
        assert_eq!(
            actual.break_even_bps.to_bits(),
            bench.model_break_even().to_bits()
        );

        let null = &split.arms[ATTRIBUTION_MARGINAL];
        assert_eq!(
            null.policy.net_growth,
            bench.policies[POLICY_MARGINAL].net_growth
        );
        assert_eq!(
            null.edge.mean, 0.0,
            "the null arm is a paired difference against itself and must be exactly zero"
        );
        assert_eq!(
            actual.paired_vs_actual.mean, 0.0,
            "the actual arm differenced against itself must be exactly zero"
        );
        assert_eq!(split.bars, bench.bars);
        assert_eq!(split.blocks, bench.blocks);
    }

    /// The per-window turnover CostAudit re-weights costs by must be the SAME notional the
    /// arm's own break-even divided by, or the composite is a weighted mean of one book's costs
    /// against another book's weights - the exact cross-construction error this measurement
    /// exists to remove.
    #[test]
    fn the_emitted_turnover_reconciles_with_the_turnover_each_arm_is_scored_on() {
        let windows = fixture_windows(16, 48, 0xED9E);
        let blocks = blocks_for(&windows);
        let split = edge_attribution(&windows, &blocks, attribution_config());

        for (arm, rows) in split.turnover.iter().enumerate() {
            assert_eq!(
                rows.len(),
                windows.len(),
                "{} must emit one row per traded window",
                ATTRIBUTION_NAMES[arm]
            );
            let bars: usize = rows.iter().map(|row| row.bars).sum();
            let total: f64 = rows.iter().map(|row| row.total).sum();
            let reported = split.arms[arm].policy.turnover;
            assert!(
                (total / bars as f64 - reported).abs() <= 1e-12 * reported.max(1.0),
                "{} emits {} per bar but is scored on {reported}",
                ATTRIBUTION_NAMES[arm],
                total / bars as f64
            );
            for row in rows {
                assert!(
                    row.interior <= row.total + 1e-12,
                    "interior turnover cannot exceed the total it is carved out of"
                );
                assert!(row.interior >= 0.0, "turnover is a notional and never negative");
            }
        }

        // The always-short arm holds a CONSTANT position, so every interior rebalance is
        // exactly zero and its whole turnover is the sampler's entry and unwind. That makes it
        // the sharpest available check that the split is carving at the right joint rather
        // than merely subtracting two plausible numbers.
        let short = &split.turnover[ATTRIBUTION_SHORT_CONSTANT];
        for row in short {
            assert_eq!(
                row.interior, 0.0,
                "a constant position rebalances never, so its interior turnover is zero"
            );
        }
    }

    /// A panel whose sign reverses ONLY on near-zero predicted means must report its flips as
    /// low-conviction, because that is the measurement hysteresis is sized from: suppressing a
    /// reversal the model was unconvinced about is nearly free, and suppressing one it was
    /// certain about costs the edge that reversal was going to earn.
    #[test]
    fn flips_that_happen_at_no_conviction_are_reported_as_low_conviction() {
        const BARS: usize = 40;
        let windows: Vec<WindowPaths> = (0..4)
            .map(|_| {
                let mut held = 2.0f64;
                let mut path = Vec::with_capacity(BARS);
                let mut mu = Vec::with_capacity(BARS);
                for bar in 0..BARS {
                    // The sign reverses exactly on the bars the model is least sure about.
                    if bar % 10 == 9 {
                        held = -held;
                        mu.push(1e-6);
                    } else {
                        mu.push(1e-3);
                    }
                    path.push(held);
                }
                let mut positions: [Vec<f64>; POLICY_COUNT] = std::array::from_fn(|_| Vec::new());
                positions[POLICY_MODEL] = path;
                let mut window =
                    WindowPaths::unmeasured(vec![0.0; BARS], Vec::new(), positions);
                window.predicted_mean = mu;
                window
            })
            .collect();

        let flips = flip_conviction(&windows);
        assert!(flips.measured());
        assert_eq!(flips.positioned, 4 * BARS);
        assert_eq!(flips.flips, 4 * 4, "one reversal every ten bars per window");
        let mass: f64 = flips.flip_share.iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-12,
            "the flip histogram is a distribution and must sum to one: {mass}"
        );
        assert_eq!(
            flips.flip_share[0], 1.0,
            "every reversal happened at the bottom of the conviction axis"
        );
        assert!(
            flips.mean_abs_mu_flip < flips.mean_abs_mu_positioned,
            "{} against {}",
            flips.mean_abs_mu_flip,
            flips.mean_abs_mu_positioned
        );

        // An accounting-only window set carries no conviction axis, and inventing one would
        // report every flip as maximally unconvinced - the most favourable possible answer.
        let blind: Vec<WindowPaths> = windows
            .iter()
            .map(|window| {
                WindowPaths::unmeasured(
                    window.realized.clone(),
                    Vec::new(),
                    window.positions.clone(),
                )
            })
            .collect();
        assert!(
            !flip_conviction(&blind).measured(),
            "a panel with no conviction axis must refuse rather than report zeros"
        );
    }

    /// Each damaged arm destroys exactly one half of the decision and nothing else.
    #[test]
    fn each_arm_destroys_exactly_the_half_it_names() {
        let windows = fixture_windows(8, 32, 0x51A9);
        let config = attribution_config();
        let recapped = recap(&windows, config.cap, config.free_marginal);
        let leverage = Ledger::build(&recapped, POLICY_MODEL, config.cap)
            .stats(config.cost_bps * 1e-4)
            .mean_abs_position;

        let arm = |index: usize| {
            attribution_paths(&recapped, index, leverage, config.free_marginal, config.cap)
        };
        let sign_only = arm(ATTRIBUTION_SIGN_ONLY);
        let random = arm(ATTRIBUTION_MAGNITUDE_RANDOM);
        let short = arm(ATTRIBUTION_MAGNITUDE_SHORT);
        let flat_short = arm(ATTRIBUTION_SHORT_CONSTANT);

        let mut coin_flipped = 0usize;
        let mut bars = 0usize;
        for window in 0..recapped.len() {
            let actual = &recapped[window].positions[POLICY_MODEL];
            for bar in 0..actual.len() {
                let f = actual[bar];
                bars += 1;
                // SIGN-ONLY keeps the sign and only the sign.
                let staked = sign_only[window].positions[POLICY_MODEL][bar];
                assert_eq!(staked.signum() * f64::from(staked != 0.0), f.signum() * f64::from(f != 0.0));
                assert!(
                    (staked.abs() - leverage).abs() < 1e-12 || staked == 0.0,
                    "sign-only stakes the matched leverage, got {staked}"
                );
                // Both magnitude arms keep |f| bar for bar and only change the sign.
                let coin = random[window].positions[POLICY_MODEL][bar];
                let bear = short[window].positions[POLICY_MODEL][bar];
                assert_eq!(coin.abs(), f.abs());
                assert_eq!(bear.abs(), f.abs());
                assert!(bear <= 0.0, "the short-sign arm never goes long: {bear}");
                if coin != 0.0 && coin.signum() != f.signum() {
                    coin_flipped += 1;
                }
                // ALWAYS-SHORT is flat at the matched leverage, carrying neither half.
                let constant = flat_short[window].positions[POLICY_MODEL][bar];
                assert!((constant + leverage).abs() < 1e-12, "got {constant}");
            }
        }
        assert!(
            coin_flipped > bars / 4 && coin_flipped < 3 * bars / 4,
            "the coin flip must actually flip roughly half the bars: {coin_flipped} of {bars}"
        );
    }

    /// The panel's hit share IS [`PolicyStats::hit_rate`], measured a second way over the same
    /// bars. Two hit rates on one page that disagree is how a session retracts a verdict.
    #[test]
    fn the_panel_hit_share_is_the_policys_own_hit_rate() {
        let windows = fixture_windows(10, 36, 0x4177);
        let blocks = blocks_for(&windows);
        let config = attribution_config();
        let split = edge_attribution(&windows, &blocks, config);

        let pooled = split.panel.scalars()[PANEL_HIT_SHARE].point;
        let policy = split.arms[ATTRIBUTION_ACTUAL].policy.hit_rate;
        assert!(
            (pooled - policy).abs() < 1e-12,
            "panel {pooled} against policy {policy}"
        );
        // Every positioned bar is a hit, a miss or a flat move, and nothing else.
        let flat = split.panel.scalars()[PANEL_FLAT_SHARE].point;
        let hits = split.panel.scalars()[PANEL_HIT_SHARE].point;
        assert!(hits + flat <= 1.0 + 1e-12);
        // The deciles partition the bars.
        let share: f64 = (0..ATTRIBUTION_DECILES)
            .map(|cell| split.panel.cells()[cell][CELL_SHARE].point)
            .sum();
        assert!((share - 1.0).abs() < 1e-9, "decile shares sum to {share}");
    }

    /// A panel whose SIGN is informative at constant size must be called a direction
    /// predictor, and the sign-only arm must keep essentially all of the edge.
    #[test]
    fn a_constant_size_informative_sign_is_called_a_direction_predictor() {
        // `f*` is the same magnitude on every bar and points the right way 70% of the time, so
        // there is no size information to find and the sign is the whole signal.
        let (bars, count) = (64usize, 24usize);
        let windows: Vec<WindowPaths> = (0..count)
            .map(|window| {
                let realized: Vec<f64> = (0..bars)
                    .map(|bar| 0.004 * (2.0 * uniform(0xD152, (window * bars + bar) as u64) - 1.0))
                    .collect();
                let free: Vec<f64> = realized
                    .iter()
                    .enumerate()
                    .map(|(bar, r)| {
                        let right = uniform(0xD153, (window * bars + bar) as u64) < 0.7;
                        2.0 * r.signum() * if right { 1.0 } else { -1.0 }
                    })
                    .collect();
                let positions = std::array::from_fn(|_| Vec::new());
                WindowPaths::unmeasured(realized, free, positions)
            })
            .collect();
        let blocks = blocks_for(&windows);
        let config = BenchConfig::new(0.0, LEVERAGE_CAP, 0.0);
        let split = edge_attribution(&windows, &blocks, config);

        assert_eq!(split.verdict(), EdgeSource::Direction, "{}", split.verdict_line());
        // The size half is exactly dead by construction: `|f*|` is constant, so the
        // magnitude-short corner IS the always-short corner and the size effect is identically
        // zero. That is what makes the SIGN effect the whole result rather than the larger of
        // two noisy numbers.
        assert_eq!(split.size_effect.mean, 0.0);
        assert!(
            resolvably_positive(&split.sign_effect),
            "the sign effect must resolve above zero: {:?}",
            split.sign_effect
        );
        // With `|f*|` constant the sign-only corner IS the actual policy and the
        // magnitude-short corner IS the always-short corner, so the interaction is identically
        // zero and the whole edge is the sign main effect plus the drift.
        assert_eq!(split.interaction.mean, 0.0);
        let reconstructed = split.sign_effect.mean + split.drift_edge().mean;
        let actual_edge = split.arms[ATTRIBUTION_ACTUAL].edge.mean;
        assert!(
            (reconstructed - actual_edge).abs() <= 1e-12 * actual_edge.abs(),
            "{reconstructed} against {actual_edge}"
        );
        assert!(
            split.panel.corr_signed().point > 0.0,
            "an informative sign shows up as a positive corr(f, R)"
        );
    }

    /// A panel whose SIZE tracks a persistent drift, with a sign that carries nothing, must be
    /// attributed to the magnitude and never to the direction.
    #[test]
    fn a_drift_tracking_size_with_a_dead_sign_is_attributed_to_the_magnitude() {
        // Every bar drifts DOWN and the model always shorts, but it shorts harder on the bars
        // whose drift is larger: the sign is a constant and carries no information at all, so
        // whatever edge exists is the size's.
        let (bars, count) = (64usize, 24usize);
        let windows: Vec<WindowPaths> = (0..count)
            .map(|window| {
                let strength: Vec<f64> = (0..bars)
                    .map(|bar| uniform(0xB1A5, (window * bars + bar) as u64))
                    .collect();
                let realized: Vec<f64> = strength
                    .iter()
                    .enumerate()
                    .map(|(bar, s)| {
                        -0.0016 * s
                            + 0.0008 * (2.0 * uniform(0xB1A6, (window * bars + bar) as u64) - 1.0)
                    })
                    .collect();
                let free: Vec<f64> = strength.iter().map(|s| -0.4 - 3.2 * s).collect();
                let positions = std::array::from_fn(|_| Vec::new());
                WindowPaths::unmeasured(realized, free, positions)
            })
            .collect();
        let blocks = blocks_for(&windows);
        let config = BenchConfig::new(0.0, LEVERAGE_CAP, 0.0);
        let split = edge_attribution(&windows, &blocks, config);

        assert_eq!(
            split.verdict(),
            EdgeSource::Magnitude,
            "a constant sign carries no direction information: {}",
            split.verdict_line()
        );
        // The sign half is exactly dead by construction: the model is always short, so the
        // sign-only corner IS the always-short corner.
        assert_eq!(split.sign_effect.mean, 0.0);
        assert!(
            resolvably_positive(&split.size_effect),
            "the size effect must resolve above zero: {:?}",
            split.size_effect
        );
        assert!(
            split.drift_edge().mean > 0.0,
            "the always-short corner earns the panel's drift, which is what the main effects \
             are measured net of"
        );
    }

    /// The 2x2 is a DECOMPOSITION, not four loosely related contrasts: the two main effects,
    /// the interaction and the drift corner have to add up to the actual policy's own edge. If
    /// they do not, the arms are not four corners of one design and none of them means what its
    /// label says.
    #[test]
    fn the_two_main_effects_the_interaction_and_the_drift_reconstruct_the_edge() {
        let windows = fixture_windows(14, 44, 0xFAC7);
        let blocks = blocks_for(&windows);
        let split = edge_attribution(&windows, &blocks, attribution_config());

        let reconstructed = split.sign_effect.mean
            + split.size_effect.mean
            + split.interaction.mean
            + split.drift_edge().mean;
        let actual = split.arms[ATTRIBUTION_ACTUAL].edge.mean;
        assert!(
            (reconstructed - actual).abs() <= 1e-12 * actual.abs().max(1e-12),
            "the design does not close: {reconstructed} against {actual}"
        );
        // Every contrast is intervalled over the same blocks as the edge it decomposes, or the
        // widths are not comparable to each other.
        for paired in [
            split.sign_effect,
            split.size_effect,
            split.interaction,
            split.drift_edge(),
        ] {
            assert_eq!(paired.blocks, split.blocks);
            assert!(paired.ci_low.is_finite() && paired.ci_high.is_finite());
        }
    }

    /// The two correlations are the ordinary Pearson correlations of the traded panel,
    /// recovered to full precision against a direct pass over the same bars.
    #[test]
    fn the_panel_correlations_are_the_pooled_pearson_correlations() {
        let windows = fixture_windows(9, 33, 0x0C09);
        let blocks = blocks_for(&windows);
        let config = attribution_config();
        let recapped = recap(&windows, config.cap, config.free_marginal);
        let panel = traded_panel(&recapped, &blocks);

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for window in &recapped {
            for (f, r) in window.positions[POLICY_MODEL].iter().zip(&window.realized) {
                xs.push(*f);
                ys.push(*r);
            }
        }
        let pearson = |x: &[f64], y: &[f64]| {
            let n = x.len() as f64;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / n;
            let vx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum::<f64>() / n;
            let vy: f64 = y.iter().map(|b| (b - my) * (b - my)).sum::<f64>() / n;
            cov / (vx * vy).sqrt()
        };
        let signed = pearson(&xs, &ys);
        let absolute = pearson(
            &xs.iter().map(|f| f.abs()).collect::<Vec<_>>(),
            &ys.iter().map(|r| r.abs()).collect::<Vec<_>>(),
        );
        assert!(
            (panel.corr_signed().point - signed).abs() < 1e-9,
            "{} vs {signed}",
            panel.corr_signed().point
        );
        assert!(
            (panel.corr_abs().point - absolute).abs() < 1e-9,
            "{} vs {absolute}",
            panel.corr_abs().point
        );
        assert!(panel.corr_signed().ci.0 < panel.corr_signed().point);
        assert!(panel.corr_signed().ci.1 > panel.corr_signed().point);
    }

    /// Same windows, same blocks, same numbers — including the coin-flip arm, whose signs come
    /// from a counter keyed on the bar's global index and never from ambient entropy.
    #[test]
    fn the_attribution_is_reproducible_to_the_last_digit() {
        let windows = fixture_windows(7, 29, 0x2EED);
        let blocks = blocks_for(&windows);
        let config = attribution_config();
        let first = edge_attribution(&windows, &blocks, config);
        let second = edge_attribution(&windows, &blocks, config);
        for arm in 0..ATTRIBUTION_ARMS {
            assert_eq!(first.arms[arm].edge, second.arms[arm].edge, "arm {arm}");
            assert_eq!(
                first.arms[arm].paired_vs_actual,
                second.arms[arm].paired_vs_actual,
                "arm {arm}"
            );
            assert_eq!(
                first.arms[arm].policy.net_growth,
                second.arms[arm].policy.net_growth
            );
        }
        for scalar in 0..PANEL_SCALARS {
            assert_eq!(
                first.panel.scalars()[scalar].point,
                second.panel.scalars()[scalar].point
            );
            assert_eq!(
                first.panel.scalars()[scalar].ci,
                second.panel.scalars()[scalar].ci
            );
        }
    }

    /// A window set with no uncapped fraction cannot be split by confidence, and the panel says
    /// so instead of splitting on the capped position — whose histogram is a spike at the cap.
    #[test]
    fn a_panel_without_the_uncapped_fraction_refuses_rather_than_splitting_on_the_cap() {
        let realized = vec![0.001, -0.002, 0.003];
        let positions = std::array::from_fn(|_| vec![1.0; 3]);
        let windows = vec![WindowPaths::unmeasured(realized, Vec::new(), positions)];
        let panel = traded_panel(&windows, &[0]);
        assert!(!panel.measured());
        assert_eq!(panel.samples, 0);
        assert_eq!(panel.report_lines(), vec!["traded panel: not measured".to_owned()]);
    }

    /// A conviction-carrying fixture: the sign reverses on a fixed cadence and the predicted
    /// mean says how sure the model was about each reversal.
    fn conviction_windows(count: usize, bars: usize, cadence: usize) -> Vec<WindowPaths> {
        (0..count)
            .map(|window| {
                let mut held = 1.0f64;
                let mut free = Vec::with_capacity(bars);
                let mut mu = Vec::with_capacity(bars);
                let mut realized = Vec::with_capacity(bars);
                for bar in 0..bars {
                    if bar % cadence == cadence - 1 {
                        held = -held;
                    }
                    free.push(2.0 * held);
                    // Conviction rises through each run, so a margin bites the EARLY reversals
                    // first and the frontier has somewhere to move.
                    mu.push(held * 1e-4 * (1 + bar % cadence) as f64);
                    realized.push(held * 0.0004 * uniform(0xC0FFEE, (window * bars + bar) as u64));
                }
                let mut positions: [Vec<f64>; POLICY_COUNT] = std::array::from_fn(|_| Vec::new());
                positions[POLICY_MODEL] = free.clone();
                let mut paths = WindowPaths::unmeasured(realized, free, positions);
                paths.predicted_mean = mu;
                paths
            })
            .collect()
    }

    /// Margin zero must be the sign-only arm bar for bar. Without that identity the sweep is a
    /// second exposure-matching convention and its margin-zero row cannot be reconciled with
    /// the attribution the session quotes.
    #[test]
    fn a_zero_flip_margin_reproduces_the_sign_only_arm_exactly() {
        let windows = conviction_windows(6, 60, 5);
        let blocks = blocks_for(&windows);
        let config = attribution_config();
        let sign_only = sign_only_positions(&windows, config);
        let recapped = recap(&windows, config.cap, config.free_marginal);
        let matched = Ledger::build(&recapped, POLICY_MODEL, config.cap)
            .stats(config.cost_bps * 1e-4)
            .mean_abs_position;
        let held = hysteresis_paths(&recapped, 0.0, ConvictionAxis::Raw, matched, config.cap);

        assert_eq!(held.len(), sign_only.len());
        for (window, target) in held.iter().zip(&sign_only) {
            assert_eq!(
                &window.positions[POLICY_MODEL], target,
                "margin zero must not move a single bar off the sign-only path"
            );
        }

        let sweep = hysteresis_sweep(&windows, &blocks, config, ConvictionAxis::Raw)
            .expect("fixture carries a mean");
        let incumbent = &sweep.points[0];
        assert_eq!(incumbent.margin_bps, 0.0);
        assert_eq!(
            incumbent.vs_sign_only.mean, 0.0,
            "the incumbent is paired against itself and must be exactly zero"
        );
    }

    /// Raising the margin can only ever REMOVE reversals, so turnover must fall monotonically
    /// and the hold must lengthen. A sweep that violates this is not measuring hysteresis.
    #[test]
    fn a_wider_flip_margin_trades_strictly_less_and_holds_strictly_longer() {
        let windows = conviction_windows(6, 60, 5);
        let blocks = blocks_for(&windows);
        let sweep = hysteresis_sweep(&windows, &blocks, attribution_config(), ConvictionAxis::Raw)
            .expect("fixture");

        for pair in sweep.points.windows(2) {
            let (looser, tighter) = (&pair[0], &pair[1]);
            assert!(
                tighter.policy.turnover <= looser.policy.turnover + 1e-12,
                "margin {} trades {} against margin {}'s {}",
                tighter.margin_bps,
                tighter.policy.turnover,
                looser.margin_bps,
                looser.policy.turnover
            );
            assert!(
                tighter.mean_hold_bars >= looser.mean_hold_bars - 1e-12,
                "margin {} holds {:.4} against margin {}'s {:.4}",
                tighter.margin_bps,
                tighter.mean_hold_bars,
                looser.margin_bps,
                looser.mean_hold_bars
            );
        }

        // The frozen anchor never reverses, so each window is one unbroken run.
        let frozen = sweep.points.last().expect("a frozen anchor");
        assert!(frozen.margin_bps.is_infinite());
        assert_eq!(
            frozen.mean_hold_bars, 60.0,
            "an unreachable margin holds its first side to the end of the window"
        );
    }

    /// The net column must be the LEDGER's own accounting, not `gross - c * turnover`.
    ///
    /// Two independent routes to net at the selection cost have to agree: the bootstrap's mean
    /// of per-window deltas, and the pooled per-bar `edge_at` the break-even solve uses. They
    /// coincide exactly only when both are the same accounting over equal-length windows, so
    /// this catches a column silently reconstructed instead of measured.
    ///
    /// It then pins the reason the reconstruction cannot be used for magnitudes: a bar's
    /// rebalance is charged `ln(1 - c * traded)`, which is strictly dearer than `c * traded`
    /// whenever the book trades at all.
    #[test]
    fn the_net_column_is_the_ledgers_accounting_and_not_a_linear_reconstruction() {
        let windows = conviction_windows(6, 60, 5);
        let blocks = blocks_for(&windows);
        let config = attribution_config();
        let sweep = hysteresis_sweep(&windows, &blocks, config, ConvictionAxis::Raw)
            .expect("fixture carries a mean");

        for point in &sweep.points {
            assert!(
                (point.net_at_cost_pooled[HYSTERESIS_SELECTION_COST]
                    - point.net_at_measured.mean * 1e4)
                    .abs()
                    < 1e-9,
                "margin {} reports net {:+.6} pooled against {:+.6} paired: the two are not the \
                 same accounting",
                point.margin_bps,
                point.net_at_cost_pooled[HYSTERESIS_SELECTION_COST],
                point.net_at_measured.mean * 1e4,
            );
        }

        // The finding, straight off a ledger rather than off the identity: the log-space charge
        // strictly exceeds the linear one, so a linear reconstruction always understates cost
        // and therefore always overstates net.
        let recapped = recap(&windows, config.cap, config.free_marginal);
        let matched = Ledger::build(&recapped, POLICY_MODEL, config.cap)
            .stats(config.cost_bps * 1e-4)
            .mean_abs_position;
        let ledger = Ledger::build(
            &hysteresis_paths(&recapped, 1.0, ConvictionAxis::Raw, matched, config.cap),
            POLICY_MODEL,
            config.cap,
        );
        let cost = HYSTERESIS_NET_COSTS[HYSTERESIS_SELECTION_COST].1 * 1e-4;
        let turnover = ledger.stats(cost).turnover;
        assert!(turnover > 0.0, "this test needs a book that trades");
        let charged = ledger.net_growth_per_bar(0.0) - ledger.net_growth_per_bar(cost);
        assert!(
            charged > cost * turnover,
            "the ledger charged {charged:.3e} where the linear form charges {:.3e}: the log \
             cost must be strictly dearer",
            cost * turnover,
        );

        assert!(
            sweep.points.iter().any(|point| (point.net_at_cost_pooled
                [HYSTERESIS_SELECTION_COST]
                - point.net_reconstructed_bps)
                .abs()
                > 1e-6),
            "no row's reported linear gap is nonzero, so the gap column is not measuring anything"
        );
    }

    /// The standardized axis thresholds `|mu|/sd`, so it needs a per-bar SD and must REFUSE
    /// without one rather than degrade to a raw-mean threshold under a standardized label. Where
    /// the SD does exist, margin zero still has to reproduce the sign-only arm exactly, because
    /// at zero the conviction comparison is skipped on either axis - which is what makes the two
    /// frontiers comparable row for row instead of two unrelated sweeps.
    #[test]
    fn the_standardized_axis_refuses_without_a_predicted_sd_and_agrees_at_margin_zero() {
        let bare = conviction_windows(6, 60, 5);
        let bare_blocks = blocks_for(&bare);
        assert!(
            bare.iter().any(|window| window.predicted_var.is_empty()),
            "this fixture is the one that carries no SD"
        );
        assert!(
            hysteresis_sweep(
                &bare,
                &bare_blocks,
                attribution_config(),
                ConvictionAxis::Standardized
            )
            .is_none(),
            "a standardized sweep without an SD must refuse, not fall back to raw |mu|"
        );

        // A constant SD makes the standardized conviction a rescaling of `|mu|`, which is the
        // only case where the two axes are directly comparable above zero - exactly what makes
        // the margin-zero identity a real cross-check rather than a coincidence of empty grids.
        let windows: Vec<WindowPaths> = conviction_windows(6, 60, 5)
            .into_iter()
            .map(|mut window| {
                let bars = window.bars();
                window.predicted_var = vec![1e-8; bars];
                window
            })
            .collect();
        let blocks = blocks_for(&windows);
        let config = attribution_config();
        let raw = hysteresis_sweep(&windows, &blocks, config, ConvictionAxis::Raw)
            .expect("the raw axis sweeps");
        let standardized = hysteresis_sweep(&windows, &blocks, config, ConvictionAxis::Standardized)
            .expect("this fixture carries a per-bar SD");

        assert_eq!(raw.points[0].margin_bps, 0.0);
        assert_eq!(standardized.points[0].margin_bps, 0.0);
        assert!(
            (raw.points[0].policy.turnover - standardized.points[0].policy.turnover).abs() < 1e-12,
            "margin zero turnover differs across axes: {} against {}",
            raw.points[0].policy.turnover,
            standardized.points[0].policy.turnover
        );
        assert!(
            (raw.points[0].net_at_measured.mean - standardized.points[0].net_at_measured.mean)
                .abs()
                < 1e-12,
            "margin zero is the sign-only arm on both axes, so its net cannot differ"
        );
        // And the axes are genuinely different constructions above zero: a standardized
        // threshold in SD units cannot reproduce a bps threshold's book.
        assert!(
            standardized
                .points
                .iter()
                .any(|point| point.policy.turnover > 0.0 && point.margin_bps > 0.0),
            "the standardized grid never trades, so it is measuring nothing"
        );
    }

    /// Every grid arm across BOTH axes must render to a DISTINCT policy string.
    ///
    /// This shipped broken once: `{:.2}` rendered the sd axis's 0.005 and 0.01 knobs identically,
    /// so two different books were written under one label, a downstream join pooled them, and
    /// one grid arm vanished from every table while the surviving label described a book nobody
    /// ran. Nothing downstream can detect that - the rows are well-formed and the count is
    /// plausible - so the guard has to live here, at the point the string is made.
    #[test]
    fn every_grid_arm_gets_its_own_policy_label() {
        let mut seen: std::collections::BTreeMap<String, (ConvictionAxis, f64)> =
            std::collections::BTreeMap::new();
        for axis in CONVICTION_AXES {
            for margin in *axis.margins() {
                let label = axis.policy_label(margin);
                if let Some((prior_axis, prior)) = seen.insert(label.clone(), (axis, margin)) {
                    panic!(
                        "policy label {label:?} is produced by both {:?} margin {prior} and \
                         {axis:?} margin {margin}: two books would be written under one key",
                        prior_axis
                    );
                }
            }
        }
        assert_eq!(
            seen.len(),
            HYSTERESIS_MARGINS.len() + HYSTERESIS_SD_MARGINS.len(),
            "one or more grid arms collided into a shared label"
        );
    }

    /// `sign(net(c)) == sign(break_even - c)` is the one part of the linear identity that
    /// survives the log-space charge, because break-even is the bisection root of the same
    /// `edge_at` the net column is read off and that curve is monotone in the cost. Verdicts may
    /// be drawn from a break-even comparison; magnitudes may not.
    #[test]
    fn the_sign_of_net_growth_agrees_with_the_break_even_comparison_at_every_cost() {
        let windows = conviction_windows(6, 60, 5);
        let blocks = blocks_for(&windows);
        let sweep = hysteresis_sweep(&windows, &blocks, attribution_config(), ConvictionAxis::Raw)
            .expect("fixture");

        for point in &sweep.points {
            if !point.break_even_bps.is_finite() {
                continue;
            }
            for (slot, (name, bps)) in HYSTERESIS_NET_COSTS.iter().enumerate() {
                let margin = point.break_even_bps - bps;
                if margin.abs() < 1e-6 {
                    continue;
                }
                assert_eq!(
                    point.net_at_cost_pooled[slot] > 0.0,
                    margin > 0.0,
                    "margin {} at the {name} weighting: net {:+.6} against break-even {:.4} \
                     versus cost {bps:.4}",
                    point.margin_bps,
                    point.net_at_cost_pooled[slot],
                    point.break_even_bps,
                );
            }
        }
    }

    /// A break-even is a PRICE only while the row outtrades the null. A wide grid drives
    /// turnover toward the null's, and past it the edge curve turns increasing - a dearer world
    /// helps a book that trades less than its benchmark - so `break_even > c` would invert to
    /// mean the row wins only ABOVE that cost.
    ///
    /// The solve is safe rather than silently wrong: an increasing curve never crosses down, the
    /// doubling bracket exhausts and it returns infinity. This pins that, so a finite break-even
    /// anywhere on the grid is itself proof the row is still on the decreasing branch and can be
    /// read as a cost.
    #[test]
    fn a_finite_break_even_is_only_ever_reported_where_the_row_outtrades_the_null() {
        let windows = conviction_windows(6, 60, 5);
        let blocks = blocks_for(&windows);
        let sweep = hysteresis_sweep(&windows, &blocks, attribution_config(), ConvictionAxis::Raw)
            .expect("fixture");

        assert!(
            sweep.null_turnover > 0.0,
            "the null trades its own entry and unwind, so its turnover is never zero"
        );
        for point in &sweep.points {
            if !point.break_even_bps.is_finite() {
                continue;
            }
            assert!(
                point.policy.turnover > sweep.null_turnover,
                "margin {} reports a finite break-even of {:.4} bps while trading {:.6}/bar \
                 against the null's {:.6}: the edge curve is increasing there and that number is \
                 not a price",
                point.margin_bps,
                point.break_even_bps,
                point.policy.turnover,
                sweep.null_turnover,
            );
        }
    }

    /// A recalibrated fixture whose shrunk book sizes SMALLER, which is what a fitted slope
    /// below one produces and what unbinds the leverage cap.
    fn shrinkable_conviction_windows(count: usize, bars: usize, cadence: usize) -> Vec<WindowPaths> {
        conviction_windows(count, bars, cadence)
            .into_iter()
            .map(|mut window| {
                window.free_shrunk = Some(window.free.iter().map(|free| free * 0.4).collect());
                window
            })
            .collect()
    }

    /// The 2x2 must DECOMPOSE: the two single-lever effects and the second difference have to
    /// sum to what both levers together deliver, on the same windows. Without that identity the
    /// interaction is an unrelated third number and cannot license or refuse adding the gains.
    #[test]
    fn the_shrink_hysteresis_second_difference_closes_against_its_own_cells() {
        let windows = shrinkable_conviction_windows(6, 60, 5);
        let blocks = blocks_for(&windows);
        let composition =
            hysteresis_composition(&windows, &blocks, attribution_config(), 1.0, ConvictionAxis::Raw)
                .expect("every window carries a recalibrated fraction");

        let cell = |slot: usize| composition.cells[slot].net.mean;
        let both = cell(COMPOSITION_BOTH) - cell(COMPOSITION_INCUMBENT);
        let decomposed = composition.hysteresis_effect.mean
            + composition.shrink_effect.mean
            + composition.interaction.mean;
        assert!(
            (both - decomposed).abs() < 1e-15,
            "both levers deliver {both:.6e} against a decomposition of {decomposed:.6e}"
        );
        assert!(
            (composition.hysteresis_effect.mean
                - (cell(COMPOSITION_HYSTERESIS) - cell(COMPOSITION_INCUMBENT)))
                .abs()
                < 1e-15,
            "the hysteresis effect is not its own two cells' difference"
        );
        assert!(
            (composition.both_vs_hysteresis.mean
                - (cell(COMPOSITION_BOTH) - cell(COMPOSITION_HYSTERESIS)))
                .abs()
                < 1e-15,
            "the decision-relevant gain is not its own two cells' difference"
        );
        assert_eq!(composition.cells.len(), COMPOSITION_NAMES.len());
    }

    /// The shrink has to be FITTED on a disjoint slice before it can be crossed with anything,
    /// so windows that carry no recalibrated fraction must refuse rather than quietly scoring
    /// the unshrunk book twice and reporting a zero interaction.
    #[test]
    fn a_composition_without_a_recalibrated_fraction_refuses_rather_than_scoring_twice() {
        let windows = conviction_windows(6, 60, 5);
        let blocks = blocks_for(&windows);
        assert!(hysteresis_composition(
            &windows,
            &blocks,
            attribution_config(),
            1.0,
            ConvictionAxis::Raw
        )
        .is_none());
    }

    /// A panel with no conviction axis cannot be swept, because the margin is a comparison
    /// against it and a zero-filled axis would silently freeze every row.
    #[test]
    fn a_sweep_without_a_conviction_axis_refuses_rather_than_freezing() {
        let windows = fixture_windows(8, 32, 0x51D3);
        let blocks = blocks_for(&windows);
        assert!(
            hysteresis_sweep(&windows, &blocks, attribution_config(), ConvictionAxis::Raw)
                .is_none()
        );
        assert!(!signal_decay(&windows, &blocks).measured());
    }

    /// A signal that predicts a PERSISTENT drift keeps its directional content as the horizon
    /// grows, and one that predicts a single bar loses it. The decay curve has to tell them
    /// apart or it cannot bound what holding longer is worth.
    #[test]
    fn a_persistent_signal_keeps_its_directional_content_as_the_horizon_grows() {
        let windows = conviction_windows(8, 80, 40);
        let decay = signal_decay(&windows, &blocks_for(&windows));
        assert!(decay.measured());
        assert_eq!(decay.points.len(), DECAY_HORIZONS.len());

        // The fixture's sign is constant for 40 bars at a time and the realized return carries
        // it, so every horizon inside a run must beat a coin flip.
        for point in &decay.points {
            assert!(point.samples > 0, "k={} measured nothing", point.horizon);
            assert!(
                point.hit_rate.mean > 0.5,
                "k={} hit {:.4} on a panel whose drift persists {} bars",
                point.horizon,
                point.hit_rate.mean,
                40
            );
        }
    }
}
