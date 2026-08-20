//! Does this predictor have a forecast horizon it can afford to trade?
//!
//! # The question, and why it is the last one worth asking about this model
//!
//! Break-even cost is gross edge divided by turnover. At the 5-minute rebalance the one-book
//! backtest ([`super::portfolio`]) measures a break-even of 1.39 bps, and the best
//! recalibrated variant anyone has produced this session reaches 4.43 bps, against a MATCHED,
//! MEASURED, impact-free one-way cost of [`MATCHED_MEASURED_BPS`] on exactly the symbol-months
//! that break-even was measured on — a 2.4x shortfall that needs no impact model to hold.
//! Restricted to the 43 traded names in the deepest liquidity decile the same measurement gives
//! [`MATCHED_DEEPEST_DECILE_BPS`], a 1.12x shortfall, so the failure is universe-wide but NARROW
//! at the liquid end — with the caveat that only the COST was restricted there and not the edge.
//! The perfect-foresight oracle on the same panel breaks even at 48.91 bps, so market structure
//! is not the obstacle. The one lever the arithmetic leaves is trading less often.
//!
//! [`super::portfolio`]'s no-trade band pulled that lever the cheap way — it keeps a ONE-BAR
//! forecast and refuses to act on it — and found the crossing at a band where the
//! equal-weight and marginal-null books also turn positive, i.e. at a corner that is
//! buy-and-hold wearing a band. That experiment freezes stale positions. It is NOT the
//! experiment of forecasting `k` bars ahead, and the two differ in exactly the way that
//! decides whether this model is finished as a trading object:
//!
//! * **STALE** ([`Construction::Stale`], [`Construction::StaleExact`]) sizes on the ONE-BAR
//!   law and then holds for `k` bars. Turnover falls like `1/k` and the position is stale for
//!   `k-1` of every `k` bars. This is the control, and it is what banding measures.
//! * **HORIZON** ([`Construction::Horizon`]) sizes on the law of the `k`-BAR AGGREGATE log
//!   return and rebalances on the same `k` clock. Turnover falls identically, and the
//!   forecast is of the quantity actually held. This is the experiment.
//!
//! If HORIZON is not materially better than STALE, the model has no long-horizon skill and
//! its cross-sectional signal is a one-bar object that cannot be traded at any affordable
//! frequency. That is a finding, not a failure, and this module is built to be able to report
//! it.
//!
//! # The `k`-bar predictive law, and why it is not a product of marginals
//!
//! The aggregate log return over a holding period is the SUM of `k` per-bar log returns, so
//! its law is the `k`-fold convolution of laws that are not independent — each bar's law is
//! conditioned on a belief the previous bar moved. The chain does not admit a closed form, so
//! it is sampled: `BarDynamics::step` advances the belief and `BarEmissionHead::sample` draws
//! the bar, which is exactly `RolloutMode::Dynamics`'s mechanism, called
//! component-by-component rather than through `BarWorldModel::imagine` because `imagine`
//! needs a `BarWorldModelSession` and a session needs a prefill: one belief per rebalance
//! would cost a 1024-bar trunk pass each, while the block pass in [`scan_panel`] already
//! produces every belief the panel has at `1/1024` of that. The tests pin the two against
//! each other on the same seed, so this is a cheaper route to the same numbers rather than a
//! second implementation of them.
//!
//! ## Rao-Blackwellization, which is what makes the sample count affordable
//!
//! The naive estimator averages the SAMPLED aggregate `sum_j r_j`. That cannot work here, and
//! the arithmetic says so before any code runs: the per-bar conditional mean is ~1 bps
//! against a per-bar volatility of ~31 bps, so estimating a `k`-bar drift of ~`k` bps from
//! sampled returns whose spread is `sqrt(k) * 31` bps needs `N >> 961/k` samples PER NAME PER
//! REBALANCE — about 500 at `k = 2` — or the cross-sectional pattern of the weights is
//! Monte-Carlo noise and the measured edge is biased to zero by the estimator rather than by
//! the model. So the drift is estimated by the tower property instead:
//!
//! ```text
//! E[sum_j r_j | past] = sum_j E[ m(h_j) ],   m(h) = sum_i p(r = i | h, past only) * c_i
//! ```
//!
//! `m(h)` is the EXACT conditional mean of one bar's log return given a belief, conditioned
//! on strictly past bars alone; `h_0` is the real belief, so the `j = 0` term carries
//! no Monte-Carlo error at all. The estimator is unbiased for every `k` — it is the same
//! expectation, evaluated one conditioning layer deeper — and its spread is the spread of a
//! CONDITIONAL MEAN (~1 bps) rather than of a return (~31 bps), which is a ~30x variance
//! reduction and turns 500 samples into 96. Both estimators are computed and reported side by
//! side ([`HorizonMechanism::rb_mu_bps`] against [`HorizonMechanism::plain_mu_bps`], each
//! with its own standard error) so the reduction is a measurement in the output rather than a
//! claim in a comment.
//!
//! The second moment is NOT Rao-Blackwellized: `Var(sum_j r_j)` is estimated by the plain
//! sample variance of the sampled aggregate, which is unbiased with no closure, and its
//! ~`sqrt(2/N)` relative error lands on a quantity that is nearly common across names and
//! enters the sizing as a denominator, where a multiplicative error does not distort the
//! cross-sectional pattern the book actually trades.
//!
//! ## The sizing closure, and why it does not decide the answer
//!
//! Sizing from two moments rather than from a solved distribution is a choice, and it is
//! forced by the variance reduction: the moment estimator is what is affordable, so the Kelly
//! fraction is the second-order optimum of the buy-and-hold payoff under a lognormal
//! aggregate,
//!
//! ```text
//! M1 = exp(mu_L + var_L / 2),  M2 = exp(2 mu_L + 2 var_L)
//! E[R] = M1 - 1,  E[R^2] = M2 - 2 M1 + 1,  f = E[R] / E[R^2]
//! ```
//!
//! which is the `mu / var` second-order Kelly [`super::growth`] optimizes against, for the
//! same stated reason: it shares the exact solve's sign and its zero.
//!
//! Two facts keep this from deciding the result. First, the gross constraint makes the SCALE
//! of `f` irrelevant: the raw Kelly vector has gross ~1000 against a cap of 2, so
//! [`project_gross`] binds at every rebalance ([`HorizonMetrics::bound_fraction`] reports
//! that it does) and the book trades the normalized DIRECTION of the vector. Only the
//! cross-sectional pattern survives, and the closure is monotone in `mu_L` name by name.
//! Second, at `k = 1` the closure and [`kelly_fractions`]'s exact 128-bin solve are both
//! available on the SAME law, so the whole effect of the closure is a measured row of the
//! table: [`Construction::Stale`] against [`Construction::StaleExact`] at `k = 1`.
//!
//! # No lookahead
//!
//! Two mechanisms, both of which this repository has previously got wrong.
//!
//! * **Same-bar.** A traded decision must not condition on any part of the bar it is betting
//!   on. The emission chain is ordered to make that free: `r` heads
//!   [`BAR_CHAIN`](crate::torch::bar_dist::BAR_CHAIN), so its
//!   head row sees no same-bar factor and IS `p(r | past)`. Every decision moment here — the
//!   one-bar law and every `m(h_j)` inside the rollout — comes from [`forecast_r_probs`], the
//!   same prefix-free row [`super::growth`] takes the objective's mean from and
//!   [`super::portfolio`] trades, called rather than reimplemented. The test derives `r`'s
//!   prefix set from [`BAR_CHAIN`](crate::torch::bar_dist::BAR_CHAIN) rather than assuming
//!   it, pins the extracted moments
//!   against a direct [`BarEmissionHead::logits`] read, asserts the row is bit-invariant
//!   across a sweep of prefix assignments, and separately asserts that a factor which DOES
//!   carry a prefix responds to it — so it cannot pass on a head whose prefix pathway is dead
//!   or on a reorder that hands `r` a prefix.
//! * **Multi-bar.** A `k`-bar forecast must not see the `k` bars it forecasts. The rollout is
//!   generative: it consumes its OWN sampled bars, and no realized future bar is an argument
//!   of any function on the path. The test proves it the only way that can fail loudly — it
//!   permutes the realized future bars in the corpus ON DISK, rebuilds everything and asserts
//!   bit-identity — and then permutes the CAUSAL history instead and asserts the forecast
//!   moves, so it cannot pass vacuously.
//!
//! One thing does leak, and it is stated rather than hidden: `BarDynamics::step` needs the
//! calendar of the bar it advances over, so the rollout reads the TIMESTAMPS of the symbol's
//! next `k` real bars off the corpus. That leaks which bars printed — a halt, a late open —
//! but no price, no volume and no return. Synthesizing the grid instead would need a session
//! calendar model, and assuming a dense grid would be a fabricated bar.
//!
//! # The book: buy-and-hold between rebalances, which is what makes the payoff exact
//!
//! [`super::portfolio`]'s loop re-establishes its target weights at every instant, so "hold
//! for `k` bars" cannot be expressed in it, and its private accounting cannot be reused from
//! here. This module runs its own book on the REBALANCE clock, and the arithmetic is exact
//! rather than approximate: a position held without trading is buy-and-hold, whose payoff
//! over the window is exactly `w * (exp(L) - 1)` in the aggregate LOG return
//! `L = ln(close_exit / close_entry)`, and the weight that comes out the other end has
//! drifted to exactly `w (1 + R) / multiplier`. So the period multiplier is
//! `1 + sum_i w_i (exp(L_i) - 1) - cost` and nothing is linearized.
//!
//! Duplicating an accounting loop is a liability, so it is pinned: at `k = 1` and ZERO cost
//! this book is run against [`super::portfolio::backtest`] on the same panel and the same
//! sizing, and the whole equity path has to agree to `1e-12` relative. Not bit-identity, and
//! for a stated reason: [`super::portfolio`] recomputes its realized simple return from the
//! panel's `f32` log return while this book recomputes it in `f64` from the corpus's closes, so
//! the two payoffs differ in the last bits of the mantissa. At any nonzero cost they diverge
//! further and legitimately: `portfolio` charges turnover against its previous TARGET, this
//! book charges it against the weight the hold actually drifted to, and only the second is what
//! a trader who held for a bar would pay.
//!
//! What differs from [`super::portfolio`], deliberately:
//!
//! * Drawdown and Sharpe are measured on the PERIOD clock, so a drawdown that opens and
//!   closes inside one holding period is invisible to [`HorizonMetrics::max_drawdown`]. The
//!   book cannot trade on it either, but it is a real understatement of risk and it is the
//!   one number here that a bar-clock mark would report differently.
//! * Break-even cost is solved by RE-RUNNING the book at each trial cost rather than by
//!   replaying a fixed payoff and turnover path. It has to be: cost enters the drift through
//!   the multiplier, so a replay — which is exact in
//!   [`super::portfolio::PortfolioRun::log_growth_at_cost`], where weights never drift —
//!   would be a linearization here. Same bisection, no approximation.
//! * Gross exposure is imposed at rebalances only. Between them it drifts with prices,
//!   because re-imposing it would be a trade the policy does not make.
//!   [`HorizonMetrics::max_gross`] reports how far it drifted.
//!
//! # Diversification degrades with the horizon, so the book reports it at every `k`
//!
//! Realized cross-sectional correlation RISES with aggregation on this panel — measured
//! `rho = 0.1914` at 1 bar over 1,756 gap-free blocks and `0.2750` at 12 bars over 38, with
//! NO measurement available past 12 because the val panel's contiguous runs are shorter than
//! 39 bars. That cuts against the lever this module tests: a longer-holding book is more
//! correlated, so per-name Kelly over-levers it further and its diversification is worth
//! less exactly where its turnover is cheapest. This module does not extrapolate `rho`. It
//! measures the consequence directly and per `k`:
//! [`HorizonMetrics::mean_first_factor_exposure`] is how much of the book is one bet, and
//! [`HorizonMetrics::leverage_error`] is realized book volatility over the volatility the
//! per-name laws imply under independence — the factor by which the book is over-levered,
//! measured rather than modelled. Both are columns of the table and series of the chart.
//!
//! # What this module does NOT measure
//!
//! * Intra-period drawdown (above).
//! * Cross-sectional correlation at `k >= 39`. Not extrapolated, not reported: unmeasurable
//!   on this panel. [`HorizonMetrics::leverage_error`] is the realized substitute and at
//!   large `k` it is itself estimated from few periods.
//! * The dynamics head beyond the horizon it was diagnosed at. `pretrain`'s rollout
//!   diagnostics stop at 100 bars, so `k = 195` and `k = 390` extrapolate a belief-advance
//!   mechanism 2-4x past anything checked against the exact trunk. Those rows are flagged
//!   `DYN-EXTRAP` in the table.
//! * Statistical power at long `k`. A five-month held-out panel holds ~20 non-overlapping
//!   390-bar periods, so the long-`k` Sharpe rests on ~20 observations.
//!   [`MIN_CREDIBLE_PERIODS`] flags those rows `FEW-PERIODS`.
//! * Overlapping-window estimates of anything. Every row uses non-overlapping periods on a
//!   single fixed phase (`t = 0, k, 2k, ...`). Averaging over phases would tighten the
//!   estimates and is not done.
//! * Any cost model other than the flat one the sweep is run with, unless the operator
//!   supplies [`super::portfolio::PanelCost`]. Break-even is a flat-cost equivalent by
//!   construction, so the headline column is cost-model independent; only the net-growth
//!   column moves.
//! * A turnover-weighted per-symbol cost to compare break-even against. The break-even column
//!   is a TURNOVER-weighted flat equivalent and [`MATCHED_MEASURED_BPS`] is an EQUAL-weighted
//!   mean over names, so the comparison is exact for the equal-weight baseline and approximate
//!   for a book whose weights vary. Closing it means running the sweep against per-symbol
//!   costs, which [`CostModel`] already admits and which was not run here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use shared::report::{read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{DOF_R, NUM_BAR_BINS};
use crate::torch::dataset::{
    future_conditioning_ids, BarCorpus, BarEndpoint, BAR_TIME_FEATURES,
};
use crate::torch::world_model::{world_model_metadata_path, BarWorldModel, BAR_MODEL_DIM};

use super::portfolio::{
    marginal_forecasts, CostModel, FlatCost, Panel, PanelConfig, PanelForecast, Policy,
    ADV_TRAILING_BARS, BELIEF_EMIT, BELIEF_PRE_CONTEXT, DEFAULT_COST_BPS, DEFAULT_GROSS_CAP,
    MAX_BREAK_EVEN_BPS, POLICIES,
};
use super::trade_bench::{
    bin_returns, forecast_r_probs, kelly_fractions, FREE_LEVERAGE, ROW_CHUNK,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Holding periods, in bars of the deployment resolution.
///
/// `39` is half a regular session, `78` is one, `195` is two and a half days and `390` is a
/// trading week at 5-minute RTH bars. The grid is geometric below a day because that is where
/// the turnover-versus-edge trade-off is decided, and it stops at a week because a five-month
/// held-out panel cannot carry enough non-overlapping weeks to say anything beyond it (see
/// [`MIN_CREDIBLE_PERIODS`]).
///
/// This is the axis of the experiment, not a tuned parameter: the deliverable is the SHAPE of
/// break-even against `k`, and a single favourable `k` would be a selected number.
pub const HOLD_HORIZONS: [usize; 9] = [1, 2, 4, 8, 16, 39, 78, 195, 390];

/// Deepest horizon `pretrain`'s rollout diagnostics reach, and therefore the deepest at which
/// `RolloutMode::Dynamics`'s drift against the exact trunk has ever been measured.
///
/// `pretrain_reports::ROLLOUT_HORIZONS` ends at 100 and `pretrain::SNAPSHOT_HORIZON` holds
/// exactly that many bars out, so a sampled row above this extrapolates the belief-advance
/// mechanism and says so.
pub const DYNAMICS_DIAGNOSED_HORIZON: usize = 100;

/// Fewest non-overlapping holding periods a row needs before its Sharpe and drawdown are
/// treated as measurements rather than as noise.
///
/// Twenty is where the standard error of an annualized Sharpe falls to roughly a quarter of
/// its own value; below it the ordering of two rows is not information.
pub const MIN_CREDIBLE_PERIODS: usize = 20;

/// Monte-Carlo paths drawn per (name, rebalance) at every sampled row.
///
/// Set from the variance-reduction arithmetic in the module docs, not by taste: the
/// Rao-Blackwellized drift has a per-path spread of the order of the conditional mean itself,
/// so `sqrt(N) / 2` is the signal-to-noise of one name's drift estimate and 96 puts that near
/// 5. The plain estimator would need ~500 at `k = 2` for the same, and both are reported so
/// the claim is checkable.
pub const DEFAULT_SAMPLES: usize = 96;

/// Independent replicate sample sets per sampled row.
///
/// Break-even cost is a bisection over a compounded book, so no closed form propagates path
/// noise into it. The only honest error bar is the spread of the whole pipeline across
/// independent sample sets, which costs a factor of this constant and is why it is 3 rather
/// than 10.
pub const DEFAULT_REPLICATES: usize = 3;

/// Sampling temperature of the rollout. Exactly `1.0`: any other value measures a different
/// predictive law than the one the checkpoint was selected on.
pub const ROLLOUT_TEMPERATURE: f64 = 1.0;

/// Rows (`pairs * samples`) held on the device at once inside the rollout.
///
/// `BarDynamics`'s hidden layer is 1664 wide, so a chunk holds `8192 * 1664` f32 of
/// activation — 54 MiB — and the whole pass stays inside the ~2 GiB inference budget a shared
/// GPU allows.
const SAMPLE_ROW_LIMIT: usize = 8192;

/// Floor on the aggregate log-return variance in the Kelly denominator, in nats squared.
///
/// A realistic one-bar value is ~1e-5, so this is seven orders below the quantity it guards
/// and exists only to keep a degenerate sample set from dividing by zero.
const VARIANCE_FLOOR: f64 = 1e-12;

/// Bisection steps of the break-even solve over `[0, MAX_BREAK_EVEN_BPS]`.
///
/// Fewer than [`super::portfolio`]'s 48 because each step re-runs the whole book rather than
/// replaying a stored path; 40 halvings still resolve the cost to `~1e-9` bps, eleven orders
/// below the distinction anyone acts on.
const BREAK_EVEN_ITERATIONS: usize = 40;

/// The MATCHED, MEASURED one-way cost the verdict is stated against, in bps: half-spread plus
/// commission plus regulatory fee, with NO impact model and no free parameter, equal-weighted
/// over exactly the 256 traded symbol-months a break-even was last measured on.
///
/// Three properties, each of which was got wrong at least once this session before it was got
/// right, and all three matter for reading a break-even against it.
///
/// * **Matched.** `super::portfolio_cost`'s restriction run priced the SAME 256 symbol-months
///   the policy traded, not a universe-wide decile. Decile occupancy came out
///   `[8, 24, 18, 18, 24, 27, 29, 38, 27, 43]` over thinnest-to-deepest, i.e. a draw that spans
///   the universe with a mild tilt to the deep end. The earlier practice of quoting the
///   deepest UNIVERSE decile's median — a different symbol set from any break-even — against a
///   break-even understated the cost by 2.6x, and that comparison is retired. Its same-symbols
///   replacement is [`MATCHED_DEEPEST_DECILE_BPS`].
/// * **Measured.** No impact term. The matched ALL-IN figure at 1% of ADV is
///   [`MATCHED_ALL_IN_BPS`], but most of the gap is square-root impact at the literature
///   default `IMPACT_K = 0.5` that nobody fitted to this corpus, so a conclusion drawn against
///   this column survives the impact coefficient being wrong by any factor. That is the
///   stronger claim, which is why it is the headline.
/// * **Equal-weighted, not median.** Each traded name contributes exactly one window, so the
///   book holds all 256 equally; per-name cost is heavily right-skewed, so the mean (10.620)
///   sits well above the median (7.230) and the mean is the dimensionally matched figure.
///
/// The one mismatch that remains, stated because it cannot be closed from here: a break-even
/// in this module is a TURNOVER-weighted flat-cost equivalent, while this constant is an
/// EQUAL-weighted mean over names. They coincide exactly for the equal-weight baseline and
/// only approximately for a book whose weights vary. Closing it needs per-symbol costs inside
/// the sweep, which is what [`CostModel`] is the seam for; [`UNIVERSE_MEASURED_BPS`] is
/// carried as the second comparator so the sensitivity to that choice is visible.
pub const MATCHED_MEASURED_BPS: f64 = 10.620;

/// Matched all-in one-way cost at 1% of ADV, impact included at `IMPACT_K = 0.5`, span-pooled.
/// Anchor-month pricing came out 1.19 bps CHEAPER (25.165), so no conclusion here turns on the
/// pooling. See [`MATCHED_MEASURED_BPS`] for why the verdict does not lead with this number.
pub const MATCHED_ALL_IN_BPS: f64 = 26.351;

/// Turnover-weighted matched measured one-way cost of the ACTUAL book, in bps, over the
/// INTERIOR turnover of the same 256 traded symbol-months.
///
/// This closes the weighting mismatch [`MATCHED_MEASURED_BPS`]'s docs name as open and cannot
/// close from there: a break-even is a TURNOVER-weighted flat-cost equivalent, so the
/// dimensionally matched comparator weights each name by what the book actually rotated in it
/// rather than equally. Interior turnover is the weight because a window's entry-from-flat and
/// terminal unwind are placed by the window SAMPLER and not by the model, and being near
/// uniform across names they dilute exactly the concentration a turnover weighting exists to
/// detect.
pub const MATCHED_ACTUAL_BOOK_BPS: f64 = 10.501;

/// The same, weighted by the SIGN-ONLY arm's interior turnover.
///
/// This is the anchor for any constant-`|f|` book, the whole sign-hysteresis frontier included:
/// such a book holds `|f|` fixed, so every unit of its turnover is a sign flip and its
/// composite is a REWEIGHTING of this arm's own flip cost over the same names. A wider flip
/// margin only ever REMOVES flips, so the frontier's composite is pinned between this weighting
/// and the equal-weighted [`MATCHED_MEASURED_BPS`] under any retention monotone in a name's
/// flip count - a 0.081 bps window. That bound is CONDITIONAL on the monotonicity;
/// unconditionally the arithmetic admits `[2.2, 27.4]`, because retention is a per-name
/// fraction and filling the budget from the cheapest or dearest names is not excluded by
/// anything except the assumption. Only the book's own measured weights close it.
pub const MATCHED_SIGN_ONLY_BPS: f64 = 10.539;

/// The same, weighted by the RECALIBRATED (shrunk-mean) book's interior turnover.
///
/// Dearer than every constant-`|f|` weighting, and that is the finding rather than an aside: the
/// shrink sizes smaller, which unbinds the leverage cap and switches magnitude modulation back
/// on as a turnover source, and magnitude-driven turnover concentrates in names that are
/// thinner, costlier and more volatile. Carried here because the shrink appears as a cell of the
/// shrink-by-hysteresis 2x2 and a cell has to be priced on its own weights.
pub const MATCHED_SHRUNK_BOOK_BPS: f64 = 12.379;

/// Measured impact-free one-way cost of the PRIMARY checkpoint's out-of-sample-fitted hysteresis
/// book: `epoch_0_ctx2048@10817` at flip margin 16 bps of raw `|mu_hat|`.
///
/// # Why this is a constant and not a bound
///
/// The conditional bound `[MATCHED_SIGN_ONLY_BPS, MATCHED_MEASURED_BPS]` documented above is
/// BROKEN for this book, by a factor of 1.9, and the reason is the mechanism rather than the
/// arithmetic. The bound assumed retention monotone in a name's flip count and uncorrelated with
/// the name's own cost. Retention is strongly correlated with it: a threshold on raw `|mu_hat|`
/// retains names whose predicted mean is large, which is partly a statement that the name is
/// VOLATILE, and volatile names are thin and dear. The book's turnover-weighted ADV percentile
/// falls from 0.5954 at margin zero to 0.4930 here, and twenty of 256 names stop trading. A
/// conviction filter is a covert liquidity filter, so a constant-`|f|` book's composite must be
/// MEASURED on its own turnover and can never be bounded from the sign-only arm's weights.
pub const MATCHED_HYSTERESIS_PRIMARY_BPS: f64 = 20.096;

/// The same for the SECONDARY checkpoint's fitted book: `pretrain_step_9728@9728` at margin 32.
///
/// Costs within 0.27 bps of the primary's despite 1.7x less turnover at twice the margin, which
/// is the only cross-checkpoint evidence available on the SHAPE of the cost-versus-margin curve
/// and points at steep-then-flat rather than continued climbing. Weak evidence, two points on
/// two different checkpoints; the per-margin grid settles it.
pub const MATCHED_HYSTERESIS_SECONDARY_BPS: f64 = 20.368;

/// Equal-weighted measured impact-free one-way cost over the WHOLE 5,297-symbol universe, as
/// the comparator for a book that is not restricted to the traded 256.
pub const UNIVERSE_MEASURED_BPS: f64 = 12.325;

/// Equal-weighted measured impact-free one-way cost of the 43 TRADED names that occupy the
/// deepest liquidity decile of the universe: the cheapest cell that exists on the same symbol
/// set any break-even was measured on, and therefore the FLOOR the verdict is stated against.
///
/// This replaces a retracted constant, and the reason is the whole lesson. The number carried
/// here before was 4.150, the MEDIAN of the deepest decile of the 5,297-symbol universe. It is
/// a real measurement and it was retracted as a comparator because it prices a DIFFERENT SYMBOL
/// SET from the one the break-even came from. `super::portfolio_cost` then ran the decile
/// restriction inside the traded 256, which is the matched question, and the answer is 19%
/// higher. The matched per-decile means, thinnest first, are
/// `[28.587, 22.019, 15.289, 10.004, 12.907, 9.098, 8.405, 8.099, 6.904, 4.955]` — NOT monotone
/// at deciles 3 and 4, so nothing here may assert that cost falls with liquidity.
///
/// The mismatch that remains, carried because it cannot be closed from this module: the
/// break-evens this floor is compared against are measured on ALL the traded names, so
/// restricting the book to these 43 would change the EDGE too and nobody has measured the edge
/// on that subset. A comparison against this floor is therefore cost-restricted against
/// edge-unrestricted: indicative, not matched. It is quoted as a floor precisely because a
/// break-even BELOW it fails even the most favourable matched cell, which is a conclusion that
/// survives the edge being remeasured; a break-even ABOVE it would prove nothing until the edge
/// on those 43 names is measured.
pub const MATCHED_DEEPEST_DECILE_BPS: f64 = 4.955;

/// Worst-case effect on [`MATCHED_DEEPEST_DECILE_BPS`] of the decile's membership moving by one
/// name, in bps: `max_i |x_i - mean| / n` over all 43 members, measured by
/// `super::portfolio_cost` on the real corpus.
///
/// It is carried because a reference line with a known sensitivity is harder to misread than a
/// bare constant, and because a sibling constant WAS retracted this session when a
/// floor-versus-round choice on a tail width moved a headline by 2x. That failure cannot reach
/// this number: `portfolio_cost::decile_of_symbol` cuts `lo = decile * count / DECILES` with
/// each decile's `hi` the next one's `lo` over the whole 5,297-symbol calibration universe, an
/// EXACT PARTITION with no name dropped, none double-counted and no rounding choice available.
/// One member is also only `1/43` of this statistic rather than all of it.
///
/// 5.21% of the mean is not negligible - a 43-name mean of a right-skewed quantity gives one
/// expensive name real weight - so the bound is stated rather than waved away. What it settles
/// is that it cannot cross the verdict: the shortfall of the best credible horizon break-even
/// against this floor is `4.955 / 2.337 = 2.120x`, and the perturbation in the MODEL'S FAVOUR
/// (floor down the full amount, `4.697`) still leaves `2.010x`. The sign of the risk is
/// favourable anyway, because the name at the boundary is by construction the LEAST liquid
/// member and therefore the most expensive, so losing it LOWERS the floor.
pub const MATCHED_DEEPEST_DECILE_BOUNDARY_BPS: f64 = 0.258;

/// Which side of a cost threshold a break-even falls on, with UNMEASURED as its own answer.
///
/// A bool over floats cannot carry three states, and this is the site where that bites: `bps`
/// is `NaN` for a policy whose net growth never crossed zero inside the bracket, and
/// `NaN > threshold` is `false`, so a plain comparison reports the ABSENCE of a measurement as
/// the POSITIVE finding "BELOW the cost" — the verdict this module exists to produce, asserted
/// from nothing. Four separate instances of this shape cost this session a campaign, two charts
/// and three retracted numbers, so it gets a third branch rather than a comment.
fn side(bps: f64, threshold: f64) -> &'static str {
    if !bps.is_finite() {
        "NOT MEASURED against"
    } else if bps > threshold {
        "ABOVE"
    } else {
        "BELOW"
    }
}

/// A break-even as a chart value and a table cell, clamped but never invented.
///
/// [`MAX_BREAK_EVEN_BPS`] exists because a book with no turnover has an unbounded break-even,
/// which is a real statement and clamps honestly. `f64::min` however IGNORES `NaN`, so
/// `f64::NAN.min(MAX_BREAK_EVEN_BPS)` is `1000.0` exactly: an UNMEASURED row would render as the
/// most profitable row on the chart, on a panel beside real cost reference lines at 4.955 and
/// 10.620. `NaN` is therefore passed through, which renders as a gap in the series and as `NaN`
/// in the table, and only a genuinely infinite break-even reaches the cap.
fn displayed_break_even(bps: f64) -> f64 {
    if bps.is_nan() { f64::NAN } else { bps.min(MAX_BREAK_EVEN_BPS) }
}

pub const HORIZON_FRONTIER_BASE: &str = "pretrain_horizon_frontier";

// ---------------------------------------------------------------------------
// Constructions
// ---------------------------------------------------------------------------

/// How a position's size is derived, which is the whole experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Construction {
    /// One-bar law, sized by the [`Construction::Horizon`] moment closure, held `k` bars. The
    /// control, matched to the experiment in functional form so the only difference between
    /// them is the horizon of the law.
    Stale,
    /// One-bar law, sized by [`kelly_fractions`]'s exact 128-bin solve, held `k` bars.
    /// [`super::portfolio`]'s own sizing, carried so the closure's contribution is visible.
    StaleExact,
    /// The `k`-bar AGGREGATE law, sampled, sized by the moment closure. The experiment.
    Horizon,
}

/// Every construction, in report order.
pub const CONSTRUCTIONS: [Construction; 3] = [
    Construction::Stale,
    Construction::StaleExact,
    Construction::Horizon,
];

impl Construction {
    pub fn name(self) -> &'static str {
        match self {
            Construction::Stale => "stale-1bar",
            Construction::StaleExact => "stale-1bar-exact",
            Construction::Horizon => "horizon-k",
        }
    }

    /// Whether this construction reads sampled rollouts, and therefore carries Monte-Carlo
    /// error. The two stale constructions are exact functions of the checkpoint and the
    /// panel; their reported standard errors are `0.0` because they ARE zero, not because
    /// they were not measured.
    pub fn is_sampled(self) -> bool {
        matches!(self, Construction::Horizon)
    }
}

// ---------------------------------------------------------------------------
// The one-bar scan
// ---------------------------------------------------------------------------

/// Every belief the panel has, plus the exact one-bar law at each of them.
///
/// The beliefs live on the HOST. At the default panel that is `instants * breadth * 512` f32,
/// about 0.7 GiB, affordable in RAM and not on a shared GPU; chunks go to the device as the
/// rollout needs them. The size is asserted against [`BELIEF_CACHE_LIMIT_BYTES`] before
/// anything is allocated, so an over-large panel fails with the knob to turn rather than with
/// an allocation error.
pub struct PanelBeliefs {
    /// Row-major `[rows, BAR_MODEL_DIM]`.
    beliefs: Vec<f32>,
    /// `row_of[t][slot]` indexes [`Self::beliefs`].
    row_of: Vec<Vec<u32>>,
    /// The exact one-bar law in [`super::portfolio`]'s own reduction: Kelly from
    /// [`kelly_fractions`], mean and variance of the SIMPLE return.
    pub one_bar: Vec<PanelForecast>,
    /// Mean of the same law's LOG return, per panel entry, in nats.
    pub mu_log: Vec<Vec<f64>>,
    /// Variance of the same law's LOG return, per panel entry, in nats squared.
    pub var_log: Vec<Vec<f64>>,
}

/// Largest belief cache this module will allocate, in bytes. Six gibibytes is generous for a
/// host allocation and small enough that the failure is a message rather than an OOM.
const BELIEF_CACHE_LIMIT_BYTES: usize = 6 << 30;

impl PanelBeliefs {
    pub fn bytes(&self) -> usize {
        self.beliefs.len() * std::mem::size_of::<f32>()
    }

    pub fn entries(&self) -> usize {
        self.beliefs.len() / BAR_MODEL_DIM as usize
    }

    fn belief_row(&self, row: u32) -> &[f32] {
        let dim = BAR_MODEL_DIM as usize;
        let at = row as usize * dim;
        &self.beliefs[at..at + dim]
    }
}

/// One block pass over every panel bar: the belief, the prefix-free one-bar law of `r`,
/// and both of that law's moment reductions.
///
/// Structurally [`super::portfolio::model_forecasts`], and deliberately so — the `k = 1` row
/// of this sweep has to be the portfolio bench's own number or the sweep is measuring a
/// different model. It differs only in keeping the belief (which the rollout needs) and the
/// LOG moments (which the aggregate closure needs) beside the simple-return reduction.
pub fn scan_panel(
    model: &BarWorldModel,
    corpus: &BarCorpus,
    panel: &Panel,
    res_secs: u32,
) -> Result<PanelBeliefs> {
    let supports = model
        .supports_for(res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {res_secs}s"))?;
    let device = model.device();
    let returns_host = bin_returns(supports);
    let returns = Tensor::from_slice(&returns_host)
        .view([1, NUM_BAR_BINS])
        .to_device(device);
    let centers_host: Vec<f64> = supports.centers(DOF_R).to_vec();
    let centers = Tensor::from_slice(&centers_host)
        .view([NUM_BAR_BINS, 1])
        .to_device(device)
        .to_kind(Kind::Double);
    let centers_sq = &centers * &centers;

    let entries: usize = panel.slices().iter().map(|s| s.symbols.len()).sum();
    let want = entries * BAR_MODEL_DIM as usize * std::mem::size_of::<f32>();
    ensure!(
        want <= BELIEF_CACHE_LIMIT_BYTES,
        "the belief cache for {entries} panel entries would need {:.1} GiB of host memory, \
         over the {:.1} GiB limit; lower --max-symbols or --max-instants",
        want as f64 / (1u64 << 30) as f64,
        BELIEF_CACHE_LIMIT_BYTES as f64 / (1u64 << 30) as f64
    );

    let mut out = PanelBeliefs {
        beliefs: vec![f32::NAN; entries * BAR_MODEL_DIM as usize],
        row_of: panel
            .slices()
            .iter()
            .map(|s| vec![u32::MAX; s.symbols.len()])
            .collect(),
        one_bar: panel
            .slices()
            .iter()
            .map(|s| PanelForecast {
                kelly_f: vec![f32::NAN; s.symbols.len()],
                mean_r: vec![f32::NAN; s.symbols.len()],
                var_r: vec![f32::NAN; s.symbols.len()],
            })
            .collect(),
        mu_log: panel
            .slices()
            .iter()
            .map(|s| vec![f64::NAN; s.symbols.len()])
            .collect(),
        var_log: panel
            .slices()
            .iter()
            .map(|s| vec![f64::NAN; s.symbols.len()])
            .collect(),
    };

    // Where each (symbol, bar) lands, grouped by symbol and in bar order, and the belief-cache
    // row it owns. The cache is filled in this order, so it is contiguous per symbol and a
    // rollout batch spanning rebalances gathers scattered rows on the host once.
    let mut wanted: Vec<Vec<(u32, usize, usize)>> = vec![Vec::new(); panel.symbols().len()];
    for (t, slice) in panel.slices().iter().enumerate() {
        for (k, &id) in slice.symbols.iter().enumerate() {
            wanted[id as usize].push((panel.bar_index(t, k), t, k));
        }
    }
    let mut next_row = 0u32;
    for targets in &wanted {
        for &(_, t, k) in targets {
            out.row_of[t][k] = next_row;
            next_row += 1;
        }
    }
    debug_assert_eq!(next_row as usize, entries);

    for (id, targets) in wanted.iter().enumerate() {
        if targets.is_empty() {
            continue;
        }
        let series = panel.series_of(id as u32);
        let (first, last) = (targets[0].0 as usize, targets[targets.len() - 1].0 as usize);
        ensure!(
            first as i64 >= BELIEF_PRE_CONTEXT + 1,
            "{} is tradeable from bar {first}, which cannot carry a belief with \
             {BELIEF_PRE_CONTEXT} bars of causal history plus the predecessor close the \
             encoder needs; build the panel with a larger `min_history`",
            panel.symbols()[id]
        );
        let slot: BTreeMap<usize, usize> = targets
            .iter()
            .enumerate()
            .map(|(index, &(bar, _, _))| (bar as usize, index))
            .collect();

        let mut cursor = first;
        while cursor <= last {
            let emit = BELIEF_EMIT.min((last - cursor + 1) as i64);
            let end = cursor + emit as usize - 2;
            let len = emit + BELIEF_PRE_CONTEXT;
            let batch = corpus
                .dof_window(&[BarEndpoint { series, bar: end }], &[0], len, device)
                .with_context(|| {
                    format!(
                        "belief block of {len} bars ending at {end} for {}",
                        panel.symbols()[id]
                    )
                })?;
            let beliefs = model.beliefs(&batch.dof, &batch.time_ids);
            let latent = *beliefs.size().last().expect("beliefs carry a feature dim");
            ensure!(
                latent == BAR_MODEL_DIM,
                "the checkpoint's belief width is {latent}, not {BAR_MODEL_DIM}"
            );
            let block = beliefs
                .narrow(1, len - emit, emit)
                .reshape([emit, latent])
                .contiguous();

            let mut start = 0i64;
            while start < emit {
                let rows = ROW_CHUNK.min(emit - start);
                let chunk = block.narrow(0, start, rows);
                // The decision law and its moments with autocast OFF: `mu = sum_i p_i c_i` is
                // a cancelling sum whose value is ~1e-4 against a term spread of ~1e-3, and
                // bf16's eight mantissa bits would destroy exactly the quantity this sweep is
                // about. The BELIEF above is computed under the ambient autocast on purpose —
                // that is the regime the checkpoint was trained and selected under.
                let (kelly, mean, var, mu_l, var_l) = tch::autocast(false, || {
                    let probs = forecast_r_probs(model.head(), &chunk);
                    let kelly = host_f64(&kelly_fractions(&probs, &returns, FREE_LEVERAGE));
                    let probs = probs.to_kind(Kind::Double);
                    let mean = probs
                        .matmul(&returns.reshape([NUM_BAR_BINS, 1]).to_kind(Kind::Double))
                        .reshape([-1]);
                    let second = probs
                        .matmul(
                            &(&returns * &returns)
                                .reshape([NUM_BAR_BINS, 1])
                                .to_kind(Kind::Double),
                        )
                        .reshape([-1]);
                    let var = (&second - &mean * &mean).clamp_min(0.0);
                    let mu_l = probs.matmul(&centers).reshape([-1]);
                    let second_l = probs.matmul(&centers_sq).reshape([-1]);
                    let var_l = (&second_l - &mu_l * &mu_l).clamp_min(0.0);
                    (
                        kelly,
                        host_f64(&mean),
                        host_f64(&var),
                        host_f64(&mu_l),
                        host_f64(&var_l),
                    )
                });
                let flat = host_f32(&chunk);
                let dim = BAR_MODEL_DIM as usize;
                for row in 0..rows as usize {
                    let bar = cursor + (start as usize) + row;
                    let Some(&index) = slot.get(&bar) else {
                        continue;
                    };
                    let (_, t, k) = targets[index];
                    out.one_bar[t].kelly_f[k] = kelly[row] as f32;
                    out.one_bar[t].mean_r[k] = mean[row] as f32;
                    out.one_bar[t].var_r[k] = var[row] as f32;
                    out.mu_log[t][k] = mu_l[row];
                    out.var_log[t][k] = var_l[row];
                    let at = out.row_of[t][k] as usize * dim;
                    out.beliefs[at..at + dim].copy_from_slice(&flat[row * dim..(row + 1) * dim]);
                }
                start += rows;
            }
            cursor += emit as usize;
        }
    }

    for (t, forecast) in out.one_bar.iter().enumerate() {
        ensure!(
            forecast.kelly_f.iter().all(|f| f.is_finite())
                && out.mu_log[t].iter().all(|m| m.is_finite())
                && out.var_log[t].iter().all(|v| v.is_finite()),
            "instant {t} has a symbol the belief pass never reached, so its position would be \
             sized from a NaN"
        );
    }
    Ok(out)
}

fn host_f32(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(tensor.to_kind(Kind::Float).reshape([-1]).to(Device::Cpu))
        .expect("a float tensor converts to a host vector")
}

fn host_f64(tensor: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(tensor.to_kind(Kind::Double).reshape([-1]).to(Device::Cpu))
        .expect("a double tensor converts to a host vector")
}

// ---------------------------------------------------------------------------
// The rebalance schedule and the realized aggregate
// ---------------------------------------------------------------------------

/// One name's entry in one holding period.
#[derive(Clone, Copy, Debug)]
pub struct Leg {
    /// Panel symbol id.
    pub id: u32,
    /// Slot of the name in the rebalance instant's slice.
    pub slot: usize,
    /// Row of [`PanelBeliefs`] holding the decision-time belief.
    pub row: u32,
    /// Bars of THIS symbol inside the holding window, at least one. Fewer than `k` when the
    /// symbol stopped printing or the panel ended.
    pub steps: usize,
    /// Realized aggregate LOG return of a position held across the window:
    /// `ln(close(exit) / close(entry))`, where entry is the close at the panel instant BEFORE
    /// the rebalance and exit is the symbol's last close inside the window.
    pub realized_log: f64,
    /// Dollar ADV at the rebalance, trailing and strictly causal.
    pub adv_usd: f64,
}

/// One non-overlapping holding period.
#[derive(Clone, Debug)]
pub struct Period {
    /// Panel instant the rebalance happens at, and whose slice defines who is tradeable.
    pub instant: usize,
    pub ts_ms: i64,
    pub legs: Vec<Leg>,
}

/// Cut the panel into non-overlapping `k`-bar holding periods on the phase `t = 0, k, 2k...`
///
/// The exit mark is read off the corpus rather than accumulated from the panel's per-instant
/// returns, because a symbol that skips a panel instant has no panel return there while its
/// HELD position still earned the move: `ln(close(b + m - 1) / close(b - 1))` over the
/// symbol's own consecutive bars is the truth, and it needs no forward fill.
pub fn schedule(
    corpus: &BarCorpus,
    panel: &Panel,
    beliefs: &PanelBeliefs,
    k: usize,
) -> Result<Vec<Period>> {
    ensure!(k >= 1, "a holding period is at least one bar");
    let instants = panel.instants();
    let mut periods = Vec::with_capacity(instants.div_ceil(k));
    for start in (0..instants).step_by(k) {
        let slice = &panel.slices()[start];
        let end_instant = (start + k - 1).min(instants - 1);
        let window_end_ts = panel.slices()[end_instant].ts_ms;
        let mut legs = Vec::with_capacity(slice.symbols.len());
        for (slot, &id) in slice.symbols.iter().enumerate() {
            let series = panel.series_of(id);
            let bars = corpus.bars(series);
            let entry_bar = panel.bar_index(start, slot) as usize;
            ensure!(
                entry_bar >= 1,
                "{} is tradeable at its first bar, which has no entry close",
                panel.symbols()[id as usize]
            );
            let entry_close = f64::from(bars[entry_bar - 1].close);
            // Walk the symbol's OWN consecutive bars while they stay inside the window and
            // carry a usable close. The first is guaranteed by the panel.
            let mut steps = 0usize;
            let mut exit_close = entry_close;
            while steps < k && entry_bar + steps < bars.len() {
                let bar = bars[entry_bar + steps];
                if bar.ts() > window_end_ts {
                    break;
                }
                let close = f64::from(bar.close);
                if !(close > 0.0) {
                    break;
                }
                steps += 1;
                exit_close = close;
            }
            ensure!(
                steps >= 1 && entry_close > 0.0,
                "{} is in the panel at instant {start} but has no usable held bar there",
                panel.symbols()[id as usize]
            );
            let realized_log = (exit_close / entry_close).ln();
            ensure!(
                realized_log.is_finite(),
                "{} has a non-finite aggregate return over instant {start}",
                panel.symbols()[id as usize]
            );
            legs.push(Leg {
                id,
                slot,
                row: beliefs.row_of[start][slot],
                steps,
                realized_log,
                adv_usd: f64::from(panel.adv_usd(start, slot)),
            });
        }
        ensure!(
            !legs.is_empty(),
            "instant {start} is a rebalance with nothing tradeable"
        );
        periods.push(Period {
            instant: start,
            ts_ms: slice.ts_ms,
            legs,
        });
    }
    ensure!(
        periods.len() >= 2,
        "a {k}-bar holding period leaves {} periods in a {instants}-instant panel, which is \
         not a book",
        periods.len()
    );
    Ok(periods)
}

// ---------------------------------------------------------------------------
// The k-bar predictive law
// ---------------------------------------------------------------------------

/// The `k`-bar aggregate law of one leg, as the two moments the closure needs plus the
/// evidence that the estimator worked.
#[derive(Clone, Copy, Debug, Default)]
pub struct AggregateLaw {
    /// Rao-Blackwellized `E[sum_j r_j | past]`, in nats.
    pub mu_log: f64,
    /// Plain sampled-mean estimate of the same quantity, in nats. Reported beside `mu_log`
    /// because the ratio of their standard errors is the variance reduction the module docs
    /// claim, measured on the real panel.
    pub plain_mu_log: f64,
    /// Sample variance of the sampled aggregate, in nats squared.
    pub var_log: f64,
    /// Monte-Carlo standard error of [`Self::mu_log`], in nats.
    pub mu_se: f64,
    /// Monte-Carlo standard error of [`Self::plain_mu_log`], in nats.
    pub plain_mu_se: f64,
}

impl AggregateLaw {
    /// The one-bar law lifted into the same shape, exactly and with zero Monte-Carlo error.
    fn exact_one_bar(mu_log: f64, var_log: f64) -> Self {
        Self {
            mu_log,
            plain_mu_log: mu_log,
            var_log,
            mu_se: 0.0,
            plain_mu_se: 0.0,
        }
    }
}

/// The second-order Kelly fraction of a buy-and-hold over an aggregate whose log return has
/// mean `mu_log` and variance `var_log`, under the lognormal closure derived in the module
/// docs. Clamped at [`FREE_LEVERAGE`], the same effectively-uncapped bound
/// [`super::portfolio`] solves its exact fractions at.
pub fn closure_kelly(mu_log: f64, var_log: f64) -> f64 {
    let var = var_log.max(VARIANCE_FLOOR);
    let m1 = (mu_log + 0.5 * var).exp();
    let m2 = (2.0 * mu_log + 2.0 * var).exp();
    let mean = m1 - 1.0;
    let second = m2 - 2.0 * m1 + 1.0;
    if !(second > 0.0) || !mean.is_finite() {
        return 0.0;
    }
    (mean / second).clamp(-FREE_LEVERAGE, FREE_LEVERAGE)
}

/// Variance of the SIMPLE aggregate return under the same lognormal closure, so the sizing
/// and the independence diagnostic describe one law.
fn closure_simple_var(mu_log: f64, var_log: f64) -> f64 {
    let var = var_log.max(0.0);
    ((2.0 * mu_log + 2.0 * var).exp() - (2.0 * mu_log + var).exp()).max(0.0)
}

/// Sample the `k`-bar aggregate law of every leg of every period.
///
/// One entry per period, aligned with [`Period::legs`]. The rollout is
/// `RolloutMode::Dynamics`'s mechanism run directly on cached beliefs; see the module docs for
/// why `imagine` is not called, and the tests for the proof that it is the same mechanism.
///
/// Batching is across (period, leg) pairs, which are independent, so the sequential depth is
/// the number of steps and not the number of rebalances.
pub fn horizon_laws(
    model: &BarWorldModel,
    corpus: &BarCorpus,
    panel: &Panel,
    beliefs: &PanelBeliefs,
    periods: &[Period],
    res_secs: u32,
    samples: usize,
) -> Result<Vec<Vec<AggregateLaw>>> {
    ensure!(samples >= 2, "a sampled law needs at least two paths");
    let supports = model
        .supports_for(res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {res_secs}s"))?;
    let device = model.device();
    let head = model.head();
    let dynamics = model.dynamics();
    let centers_host: Vec<f64> = supports.centers(DOF_R).to_vec();
    let centers = Tensor::from_slice(&centers_host)
        .view([NUM_BAR_BINS, 1])
        .to_device(device)
        .to_kind(Kind::Double);

    let mut out: Vec<Vec<AggregateLaw>> = periods
        .iter()
        .map(|p| vec![AggregateLaw::default(); p.legs.len()])
        .collect();

    // Flatten to (period, leg) pairs so one chunk can span rebalances.
    let pairs: Vec<(usize, usize)> = periods
        .iter()
        .enumerate()
        .flat_map(|(p, period)| (0..period.legs.len()).map(move |l| (p, l)))
        .collect();
    let chunk_pairs = (SAMPLE_ROW_LIMIT / samples).max(1);
    let samples_i = samples as i64;

    for chunk in pairs.chunks(chunk_pairs) {
        let count = chunk.len() as i64;
        let deepest = chunk
            .iter()
            .map(|&(p, l)| periods[p].legs[l].steps)
            .max()
            .expect("a chunk has at least one pair");
        // The exogenous conditioning of each leg's own next `deepest` bars, padded past its
        // last held bar with a repeat of that bar's ids. Padded steps are masked out of both
        // accumulators, so the padding cannot reach a reported number; it exists only so one
        // chunk can hold legs of different lengths.
        //
        // `future_conditioning_ids`, not `bar_time_ids`: these are the bars of an IMAGINED
        // continuation, so the market proxy's state at them is not knowable at the decision and
        // must arrive as MISSING. The function has no parameter through which a market channel
        // could be supplied, which is what makes that structural rather than remembered.
        let mut clock = vec![0i64; chunk.len() * deepest * BAR_TIME_FEATURES];
        let mut active = vec![0.0f64; chunk.len() * deepest];
        let mut flat = vec![0.0f32; chunk.len() * BAR_MODEL_DIM as usize];
        for (index, &(p, l)) in chunk.iter().enumerate() {
            let leg = periods[p].legs[l];
            let series = panel.series_of(leg.id);
            let bars = corpus.bars(series);
            let entry_bar = panel.bar_index(periods[p].instant, leg.slot) as usize;
            for step in 0..deepest {
                active[index * deepest + step] = f64::from(u8::from(step < leg.steps));
                let bar = entry_bar + step.min(leg.steps - 1);
                let ids = future_conditioning_ids(
                    bars[bar].ts(),
                    bar.checked_sub(1).map(|prev| bars[prev].ts()),
                    res_secs,
                );
                let at = (index * deepest + step) * BAR_TIME_FEATURES;
                clock[at..at + BAR_TIME_FEATURES].copy_from_slice(&ids);
            }
            let dim = BAR_MODEL_DIM as usize;
            flat[index * dim..(index + 1) * dim].copy_from_slice(beliefs.belief_row(leg.row));
        }
        let clock = Tensor::from_slice(&clock)
            .view([count, deepest as i64, BAR_TIME_FEATURES as i64])
            .to_device(device);
        let active = Tensor::from_slice(&active)
            .view([count, deepest as i64])
            .to_device(device);
        let seed = Tensor::from_slice(&flat)
            .view([count, BAR_MODEL_DIM])
            .to_device(device);

        let (rb, plain) = tch::no_grad(|| {
            // One row per (pair, path), interleaved so row `pair * samples + n` is path `n`.
            let mut h = seed.repeat_interleave_self_int(samples_i, 0, None);
            let total = count * samples_i;
            let mut sum_m = Tensor::zeros([total], (Kind::Double, device));
            let mut sum_r = Tensor::zeros([total], (Kind::Double, device));
            for step in 0..deepest as i64 {
                let live = active
                    .select(1, step)
                    .repeat_interleave_self_int(samples_i, 0, None);
                // The DECISION moment, BEFORE this bar is drawn: the conditional mean of its
                // log return under the head's prefix-free `r` row. Autocast off, for the
                // reason stated in `scan_panel`, and row-chunked so the peak stays bounded by
                // `ROW_CHUNK` rather than by the sample count.
                let m = tch::autocast(false, || {
                    let mut parts = Vec::with_capacity((total / ROW_CHUNK + 1) as usize);
                    let mut at = 0i64;
                    while at < total {
                        let rows = ROW_CHUNK.min(total - at);
                        let probs = forecast_r_probs(head, &h.narrow(0, at, rows))
                            .to_kind(Kind::Double);
                        parts.push(probs.matmul(&centers).reshape([-1]));
                        at += rows;
                    }
                    Tensor::cat(&parts, 0)
                });
                sum_m += &m * &live;
                let ids = clock
                    .select(1, step)
                    .repeat_interleave_self_int(samples_i, 0, None);
                let dof = head.sample(&h, supports, ROLLOUT_TEMPERATURE);
                let drawn = dof
                    .select(1, DOF_R as i64)
                    .to_kind(Kind::Double)
                    .reshape([-1]);
                sum_r += &drawn * &live;
                h = dynamics.step(&h, &dof, &ids);
            }
            (
                sum_m.view([count, samples_i]),
                sum_r.view([count, samples_i]),
            )
        });

        let root = (samples as f64).sqrt();
        let rb_mu = host_f64(&rb.mean_dim([1i64].as_slice(), false, Kind::Double));
        let rb_sd = host_f64(&rb.std_dim([1i64].as_slice(), true, false));
        let plain_mu = host_f64(&plain.mean_dim([1i64].as_slice(), false, Kind::Double));
        let plain_sd = host_f64(&plain.std_dim([1i64].as_slice(), true, false));
        for (index, &(p, l)) in chunk.iter().enumerate() {
            let var = plain_sd[index] * plain_sd[index];
            ensure!(
                rb_mu[index].is_finite() && var.is_finite(),
                "the {}-step rollout of {} at instant {} produced a non-finite law",
                periods[p].legs[l].steps,
                panel.symbols()[periods[p].legs[l].id as usize],
                periods[p].instant
            );
            out[p][l] = AggregateLaw {
                mu_log: rb_mu[index],
                plain_mu_log: plain_mu[index],
                var_log: var,
                mu_se: rb_sd[index] / root,
                plain_mu_se: plain_sd[index] / root,
            };
        }
    }
    Ok(out)
}

/// The law every policy is sized from at one `k` under one construction, aligned with
/// [`Period::legs`], plus the null's own fraction.
#[derive(Clone, Debug)]
pub struct HorizonInputs {
    pub construction: Construction,
    /// `kelly[p][l]` is the uncapped preference of the model policies.
    pub kelly: Vec<Vec<f64>>,
    /// Predicted variance of the held aggregate's SIMPLE return, for the independence
    /// diagnostic behind [`HorizonMetrics::leverage_error`].
    pub pred_var: Vec<Vec<f64>>,
    /// The unconditional-marginal null's fraction. ONE number: every present name shares it,
    /// which is why its value cannot move the null's book at all — the gross projection is
    /// scale-free on a constant vector, and a test pins that.
    pub marginal_kelly: f64,
    /// Per-leg law summaries, for the mechanism columns.
    pub laws: Vec<Vec<AggregateLaw>>,
}

/// Assemble the sizing inputs for one construction.
pub fn build_inputs(
    construction: Construction,
    beliefs: &PanelBeliefs,
    periods: &[Period],
    sampled: Option<&[Vec<AggregateLaw>]>,
    marginal: &PanelForecast,
) -> Result<HorizonInputs> {
    let mut kelly = Vec::with_capacity(periods.len());
    let mut pred_var = Vec::with_capacity(periods.len());
    let mut laws = Vec::with_capacity(periods.len());
    for (p, period) in periods.iter().enumerate() {
        let mut k_row = Vec::with_capacity(period.legs.len());
        let mut v_row = Vec::with_capacity(period.legs.len());
        let mut l_row = Vec::with_capacity(period.legs.len());
        for (l, leg) in period.legs.iter().enumerate() {
            let law = match construction {
                Construction::Stale | Construction::StaleExact => AggregateLaw::exact_one_bar(
                    beliefs.mu_log[period.instant][leg.slot],
                    beliefs.var_log[period.instant][leg.slot],
                ),
                Construction::Horizon => {
                    let sampled = sampled.context(
                        "the horizon construction needs sampled aggregate laws and got none",
                    )?;
                    ensure!(
                        sampled.len() == periods.len() && sampled[p].len() == period.legs.len(),
                        "the sampled laws do not align with the schedule"
                    );
                    sampled[p][l]
                }
            };
            let f = match construction {
                Construction::StaleExact => {
                    f64::from(beliefs.one_bar[period.instant].kelly_f[leg.slot])
                }
                Construction::Stale | Construction::Horizon => {
                    closure_kelly(law.mu_log, law.var_log)
                }
            };
            k_row.push(f);
            v_row.push(closure_simple_var(law.mu_log, law.var_log));
            l_row.push(law);
        }
        kelly.push(k_row);
        pred_var.push(v_row);
        laws.push(l_row);
    }
    let marginal_kelly = marginal.kelly_f.first().map_or(0.0, |f| f64::from(*f));
    Ok(HorizonInputs {
        construction,
        kelly,
        pred_var,
        marginal_kelly,
        laws,
    })
}

// ---------------------------------------------------------------------------
// The book
// ---------------------------------------------------------------------------

/// Scale `raw` onto the L1 ball of radius `budget`, in place. Returns whether it bound.
///
/// Proportional, never truncating, for the same reason [`super::portfolio`]'s own projection
/// is: a leverage limit scales a book rather than dropping its smallest names.
fn project_gross(raw: &mut [f64], budget: f64) -> bool {
    let gross: f64 = raw.iter().map(|w| w.abs()).sum();
    if !(gross > budget) || !gross.is_finite() {
        return false;
    }
    let scale = budget / gross;
    for w in raw.iter_mut() {
        *w *= scale;
    }
    true
}

/// The raw preference vector of one policy over one period's legs.
///
/// Mirrors [`super::portfolio::Policy`]'s private `raw_weights` with the payoff read at the
/// HOLDING horizon rather than at the next bar: the oracle's perfect foresight is of the
/// aggregate a held position actually earns, which is the only ceiling that means anything
/// once a position is held for more than one bar.
fn raw_weights(
    policy: Policy,
    period: &Period,
    inputs: &HorizonInputs,
    index: usize,
    budget: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    let n = period.legs.len();
    match policy {
        Policy::Model => out.extend(inputs.kelly[index].iter().copied()),
        Policy::MarketNeutral => {
            if n == 0 {
                return;
            }
            let mean = inputs.kelly[index].iter().sum::<f64>() / n as f64;
            out.extend(inputs.kelly[index].iter().map(|f| f - mean));
        }
        Policy::Marginal => out.extend(std::iter::repeat_n(inputs.marginal_kelly, n)),
        Policy::EqualWeight => out.extend(std::iter::repeat_n(1.0, n)),
        Policy::Oracle => {
            out.extend(std::iter::repeat_n(0.0, n));
            let best = period
                .legs
                .iter()
                .enumerate()
                .map(|(l, leg)| (l, leg.realized_log.exp_m1()))
                .filter(|(_, payoff)| payoff.is_finite() && *payoff != 0.0)
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()));
            if let Some((l, payoff)) = best {
                out[l] = budget * payoff.signum();
            }
        }
    }
}

/// One policy's realized path through the holding periods.
#[derive(Clone, Debug)]
pub struct HorizonBook {
    pub policy: Policy,
    pub construction: Construction,
    pub k: usize,
    pub gross_cap: f64,
    /// Natural log of wealth, one entry per period boundary, `[0] == 0.0`.
    pub log_equity: Vec<f64>,
    /// `sum_i w_i (exp(L_i) - 1)` over the period, BEFORE cost.
    pub payoff: Vec<f64>,
    /// Simple return of the period, net of cost.
    pub returns: Vec<f64>,
    pub gross: Vec<f64>,
    pub net: Vec<f64>,
    /// Gross exposure at the END of the period, after the hold drifted it.
    pub drifted_gross: Vec<f64>,
    pub turnover: Vec<f64>,
    pub cost: Vec<f64>,
    pub factor: Vec<f64>,
    pub pred_var: Vec<f64>,
    pub bound: Vec<bool>,
    pub ruined_at: Option<usize>,
}

/// Compound one policy through the holding periods at one gross cap.
///
/// Positions are established at the rebalance and then HELD: nothing trades until the next
/// rebalance, weights drift with prices, and the drifted vector is what the next target is
/// charged against. That is the whole difference from [`super::portfolio::backtest`], and at
/// `k = 1` it collapses onto it up to the single bar of drift inside each instant.
#[allow(clippy::too_many_arguments)]
pub fn run_book(
    panel: &Panel,
    periods: &[Period],
    inputs: &HorizonInputs,
    policy: Policy,
    k: usize,
    gross_cap: f64,
    cost: &dyn CostModel,
    capital_usd: f64,
) -> Result<HorizonBook> {
    ensure!(
        gross_cap > 0.0 && gross_cap.is_finite(),
        "the gross cap must be positive and finite, got {gross_cap}"
    );
    ensure!(
        inputs.kelly.len() == periods.len(),
        "the sizing inputs cover {} periods against {} rebalances",
        inputs.kelly.len(),
        periods.len()
    );
    let budget = policy.gross_budget(gross_cap);
    let names = panel.symbols().len();
    let loading = panel.first_factor();

    let mut log_wealth = 0.0f64;
    let mut held = vec![0.0f64; names];
    let mut target = vec![0.0f64; names];
    let mut realized = vec![0.0f64; names];
    let mut last_adv = vec![0.0f64; names];
    let mut raw: Vec<f64> = Vec::with_capacity(names);

    let mut book = HorizonBook {
        policy,
        construction: inputs.construction,
        k,
        gross_cap,
        log_equity: Vec::with_capacity(periods.len() + 1),
        payoff: Vec::with_capacity(periods.len()),
        returns: Vec::with_capacity(periods.len()),
        gross: Vec::with_capacity(periods.len()),
        net: Vec::with_capacity(periods.len()),
        drifted_gross: Vec::with_capacity(periods.len()),
        turnover: Vec::with_capacity(periods.len()),
        cost: Vec::with_capacity(periods.len()),
        factor: Vec::with_capacity(periods.len()),
        pred_var: Vec::with_capacity(periods.len()),
        bound: Vec::with_capacity(periods.len()),
        ruined_at: None,
    };
    book.log_equity.push(0.0);

    for (index, period) in periods.iter().enumerate() {
        for leg in &period.legs {
            if leg.adv_usd.is_finite() && leg.adv_usd > 0.0 {
                last_adv[leg.id as usize] = leg.adv_usd;
            }
        }
        if book.ruined_at.is_some() {
            // A dead book holds nothing, trades nothing and earns nothing, forever. It is
            // still recorded at every period so the curve keeps the rebalance clock.
            book.log_equity.push(f64::NEG_INFINITY);
            book.payoff.push(0.0);
            book.returns.push(0.0);
            book.gross.push(0.0);
            book.net.push(0.0);
            book.drifted_gross.push(0.0);
            book.turnover.push(0.0);
            book.cost.push(0.0);
            book.factor.push(0.0);
            book.pred_var.push(0.0);
            book.bound.push(false);
            continue;
        }

        raw_weights(policy, period, inputs, index, budget, &mut raw);
        ensure!(
            raw.len() == period.legs.len(),
            "policy {} produced {} weights for {} legs at period {index}",
            policy.name(),
            raw.len(),
            period.legs.len()
        );
        for w in raw.iter_mut() {
            if !w.is_finite() {
                *w = 0.0;
            }
        }
        let bound = project_gross(&mut raw, budget);

        // Absence is a zero target and an unwind, exactly as in `portfolio`: a name that is
        // not in this rebalance's slice cannot be held through the coming window, whatever it
        // was worth at the last one, and that unwind is charged like any other trade.
        target[..].fill(0.0);
        realized[..].fill(0.0);
        for (l, leg) in period.legs.iter().enumerate() {
            target[leg.id as usize] = raw[l];
            realized[leg.id as usize] = leg.realized_log.exp_m1();
        }

        let mut payoff = 0.0f64;
        let mut gross = 0.0f64;
        let mut net = 0.0f64;
        let mut factor = 0.0f64;
        let mut pred_var = 0.0f64;
        for (l, leg) in period.legs.iter().enumerate() {
            let w = target[leg.id as usize];
            gross += w.abs();
            net += w;
            factor += w * f64::from(loading[leg.id as usize]);
            let var = inputs.pred_var[index][l];
            if var.is_finite() && var > 0.0 {
                pred_var += w * w * var;
            }
            payoff += w * realized[leg.id as usize];
        }
        ensure!(
            gross <= budget * (1.0 + 1e-9) + 1e-12,
            "policy {} used gross {gross} against a budget of {budget} at period {index}",
            policy.name()
        );

        let mut turnover = 0.0f64;
        let mut cost_frac = 0.0f64;
        let wealth_usd = (log_wealth.exp() * capital_usd).min(f64::from(f32::MAX));
        for id in 0..names {
            let delta = (target[id] - held[id]).abs();
            if delta == 0.0 {
                continue;
            }
            turnover += delta;
            let adv = last_adv[id];
            // No observed liquidity means the size is unpriceable, not free: charge it at a
            // full-ADV clip, the worst bucket any sane cost curve carries.
            let frac = if adv > 0.0 {
                (((delta * wealth_usd) / adv) as f32).min(f32::MAX)
            } else {
                1.0
            };
            let bps = f64::from(cost.cost_bps(id as u32, period.ts_ms, frac));
            ensure!(
                bps.is_finite() && bps >= 0.0,
                "the cost model returned {bps} bps for symbol {id} at {}",
                period.ts_ms
            );
            cost_frac += delta * bps * 1e-4;
        }

        let multiplier = 1.0 + payoff - cost_frac;
        let (realized_return, drifted) = if multiplier > 0.0 {
            log_wealth += multiplier.ln();
            // Buy-and-hold drift: the position is worth `w (1 + R)` of the pre-cost wealth and
            // the book is worth `multiplier` of it, so this is the weight the next rebalance is
            // charged against. Exact, not an approximation.
            let mut drifted_gross = 0.0f64;
            for id in 0..names {
                held[id] = target[id] * (1.0 + realized[id]) / multiplier;
                drifted_gross += held[id].abs();
            }
            (multiplier - 1.0, drifted_gross)
        } else {
            log_wealth = f64::NEG_INFINITY;
            book.ruined_at = Some(index);
            held.fill(0.0);
            (-1.0, 0.0)
        };

        book.log_equity.push(log_wealth);
        book.payoff.push(payoff);
        book.returns.push(realized_return);
        book.gross.push(gross);
        book.net.push(net);
        book.drifted_gross.push(drifted);
        book.turnover.push(turnover);
        book.cost.push(cost_frac);
        book.factor.push(factor);
        book.pred_var.push(pred_var);
        book.bound.push(bound);
    }
    Ok(book)
}

/// What a trader would quote for one `(k, construction, policy)`, annualized from the panel's
/// own measured calendar.
#[derive(Clone, Copy, Debug)]
pub struct HorizonMetrics {
    pub periods: usize,
    pub span_years: f64,
    pub periods_per_year: f64,
    pub final_log_wealth: f64,
    /// Net log growth per year at the cost the sweep was run with.
    pub log_growth_per_year: f64,
    /// Log growth per year at EXACTLY zero cost: what the law was worth before paying to act
    /// on it. Measured by re-running the book at zero cost, not by subtraction.
    pub gross_log_growth_per_year: f64,
    /// Flat one-way cost, in bps, at which net log growth crosses zero.
    pub break_even_cost_bps: f64,
    pub cagr: f64,
    pub sharpe: f64,
    pub vol: f64,
    /// Measured on the PERIOD clock; intra-period drawdown is invisible to it.
    pub max_drawdown: f64,
    pub mean_gross: f64,
    /// Largest gross the book ever carried, INCLUDING the drift inside a holding period.
    pub max_gross: f64,
    pub mean_net: f64,
    /// Absolute weight traded per period, summed and divided by the panel's trading days.
    pub turnover_per_day: f64,
    /// Turnover per period as a multiple of the gross the book actually held.
    pub rotation_per_period: f64,
    pub bound_fraction: f64,
    /// Mean absolute projection of the book onto the panel's leading eigenvector, as a
    /// fraction of its gross. `0` is factor-neutral; `1` is a book that is one bet on the
    /// market wearing the costume of many.
    pub mean_first_factor_exposure: f64,
    pub first_factor_share: f64,
    /// Realized book volatility divided by the volatility the per-name laws imply under
    /// INDEPENDENCE. Per-name Kelly sizes as if this were `1.0`; whatever it is, is the factor
    /// by which the book is over-levered. This is the measured stand-in for a cross-sectional
    /// correlation that cannot be estimated past 12 bars on this panel.
    pub leverage_error: f64,
    pub ruined_at_period: f64,
}

impl HorizonMetrics {
    fn of(book: &HorizonBook, gross: &HorizonBook, panel: &Panel) -> Self {
        let n = book.returns.len();
        let years = panel.span_years();
        let per_year = if years > 0.0 {
            n as f64 / years
        } else {
            f64::NAN
        };
        let final_log_wealth = *book.log_equity.last().expect("the curve starts at 0.0");
        let log_growth_per_year = if years > 0.0 {
            final_log_wealth / years
        } else {
            f64::NAN
        };
        let mean = book.returns.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            book.returns
                .iter()
                .map(|r| (r - mean) * (r - mean))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            f64::NAN
        };
        let sd = variance.sqrt();
        let mut peak = f64::NEG_INFINITY;
        let mut max_drawdown = 0.0f64;
        for &log_w in &book.log_equity {
            peak = peak.max(log_w);
            if peak > f64::NEG_INFINITY {
                max_drawdown = max_drawdown.max(1.0 - (log_w - peak).exp());
            }
        }
        let mut exposure_sum = 0.0f64;
        let mut exposure_count = 0usize;
        for (f, g) in book.factor.iter().zip(&book.gross) {
            if *g > 0.0 {
                exposure_sum += (f / g).abs();
                exposure_count += 1;
            }
        }
        let payoff_mean = book.payoff.iter().sum::<f64>() / n as f64;
        let payoff_var = if n > 1 {
            book.payoff
                .iter()
                .map(|p| (p - payoff_mean) * (p - payoff_mean))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            f64::NAN
        };
        let independence_var = book.pred_var.iter().sum::<f64>() / n as f64;
        let days = panel.trading_days().max(1) as f64;
        let mean_of = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
        let gross_final = *gross.log_equity.last().expect("the curve starts at 0.0");
        Self {
            periods: n,
            span_years: years,
            periods_per_year: per_year,
            final_log_wealth,
            log_growth_per_year,
            gross_log_growth_per_year: if years > 0.0 {
                gross_final / years
            } else {
                f64::NAN
            },
            break_even_cost_bps: f64::NAN,
            cagr: if final_log_wealth == f64::NEG_INFINITY {
                -1.0
            } else {
                log_growth_per_year.exp_m1()
            },
            sharpe: if sd > 0.0 {
                mean / sd * per_year.sqrt()
            } else {
                f64::NAN
            },
            vol: sd * per_year.sqrt(),
            max_drawdown,
            mean_gross: mean_of(&book.gross),
            max_gross: book
                .gross
                .iter()
                .chain(&book.drifted_gross)
                .copied()
                .fold(0.0, f64::max),
            mean_net: mean_of(&book.net),
            turnover_per_day: book.turnover.iter().sum::<f64>() / days,
            rotation_per_period: {
                let g = mean_of(&book.gross);
                if g > 0.0 {
                    mean_of(&book.turnover) / g
                } else {
                    f64::NAN
                }
            },
            bound_fraction: book.bound.iter().filter(|b| **b).count() as f64 / n as f64,
            mean_first_factor_exposure: if exposure_count > 0 {
                exposure_sum / exposure_count as f64
            } else {
                f64::NAN
            },
            first_factor_share: panel.first_factor_share(),
            leverage_error: if independence_var > 0.0 && payoff_var.is_finite() {
                (payoff_var / independence_var).sqrt()
            } else {
                f64::NAN
            },
            ruined_at_period: book.ruined_at.map_or(f64::NAN, |p| p as f64),
        }
    }
}

/// Run one policy at one `k` and measure it, including the zero-cost re-run and the break-even
/// solve.
///
/// The break-even bisection RE-RUNS the book at every trial cost. It has to: cost enters the
/// buy-and-hold drift through the period multiplier, so replaying a stored payoff and turnover
/// path at a different cost would be a linearization here. The book loop is `periods *
/// breadth` arithmetic, so 40 re-runs cost far less than the forecast that produced its
/// inputs.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    panel: &Panel,
    periods: &[Period],
    inputs: &HorizonInputs,
    policy: Policy,
    k: usize,
    gross_cap: f64,
    cost: &dyn CostModel,
    capital_usd: f64,
) -> Result<HorizonMetrics> {
    let book = run_book(
        panel, periods, inputs, policy, k, gross_cap, cost, capital_usd,
    )?;
    let free = FlatCost::new(0.0);
    let gross = run_book(
        panel, periods, inputs, policy, k, gross_cap, &free, capital_usd,
    )?;
    let mut metrics = HorizonMetrics::of(&book, &gross, panel);

    let at = |bps: f64| -> Result<f64> {
        let flat = FlatCost::new(bps as f32);
        let run = run_book(
            panel, periods, inputs, policy, k, gross_cap, &flat, capital_usd,
        )?;
        Ok(*run.log_equity.last().expect("the curve starts at 0.0"))
    };
    metrics.break_even_cost_bps = if !(at(0.0)? > 0.0) {
        0.0
    } else if at(MAX_BREAK_EVEN_BPS)? > 0.0 {
        f64::INFINITY
    } else {
        let (mut lo, mut hi) = (0.0f64, MAX_BREAK_EVEN_BPS);
        for _ in 0..BREAK_EVEN_ITERATIONS {
            let mid = 0.5 * (lo + hi);
            if at(mid)? > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    Ok(metrics)
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

/// The mechanism behind one row: the numbers that say WHY break-even moved, not just that it
/// did. Everything in basis points of the held aggregate, averaged over every leg of every
/// period.
#[derive(Clone, Copy, Debug, Default)]
pub struct HorizonMechanism {
    /// Mean predicted conditional drift of the held aggregate, signed.
    pub rb_mu_bps: f64,
    /// The same drift from the plain sampled estimator, for the variance-reduction check.
    pub plain_mu_bps: f64,
    /// Mean Monte-Carlo standard error of the Rao-Blackwellized drift.
    pub rb_mu_se_bps: f64,
    /// Mean Monte-Carlo standard error of the plain drift. Its ratio to the line above is the
    /// measured variance reduction.
    pub plain_mu_se_bps: f64,
    /// Mean predicted volatility of the held aggregate.
    pub pred_sigma_bps: f64,
    /// Realized volatility of the held aggregate over the same legs.
    pub realized_sigma_bps: f64,
    /// Fraction of legs whose predicted drift has the sign of the realized aggregate. The only
    /// one of the three "win rates" that is a directional-skill statistic.
    pub sign_agreement: f64,
    /// Fraction of legs whose realized held aggregate was positive. A property of the market
    /// over the window, not of the model.
    pub realized_up_fraction: f64,
    pub legs: usize,
}

impl HorizonMechanism {
    fn of(periods: &[Period], inputs: &HorizonInputs) -> Self {
        let mut out = Self::default();
        let mut realized_sum = 0.0f64;
        let mut realized_sq = 0.0f64;
        for (p, period) in periods.iter().enumerate() {
            for (l, leg) in period.legs.iter().enumerate() {
                let law = inputs.laws[p][l];
                out.rb_mu_bps += law.mu_log;
                out.plain_mu_bps += law.plain_mu_log;
                out.rb_mu_se_bps += law.mu_se;
                out.plain_mu_se_bps += law.plain_mu_se;
                out.pred_sigma_bps += law.var_log.max(0.0).sqrt();
                realized_sum += leg.realized_log;
                realized_sq += leg.realized_log * leg.realized_log;
                if leg.realized_log != 0.0 && law.mu_log.signum() == leg.realized_log.signum() {
                    out.sign_agreement += 1.0;
                }
                if leg.realized_log > 0.0 {
                    out.realized_up_fraction += 1.0;
                }
                out.legs += 1;
            }
        }
        if out.legs == 0 {
            return out;
        }
        let n = out.legs as f64;
        let scale = 1.0e4 / n;
        out.rb_mu_bps *= scale;
        out.plain_mu_bps *= scale;
        out.rb_mu_se_bps *= scale;
        out.plain_mu_se_bps *= scale;
        out.pred_sigma_bps *= scale;
        let mean = realized_sum / n;
        out.realized_sigma_bps = (realized_sq / n - mean * mean).max(0.0).sqrt() * 1.0e4;
        out.sign_agreement /= n;
        out.realized_up_fraction /= n;
        out
    }
}

/// One `(k, construction, policy)` row of the frontier, with its replicate spread.
#[derive(Clone, Copy, Debug)]
pub struct HorizonRow {
    pub k: usize,
    pub construction: Construction,
    pub policy: Policy,
    pub periods: usize,
    /// Mean over replicates. Identical to the single value for the unsampled constructions.
    pub metrics: HorizonMetrics,
    /// Standard error ACROSS replicate sample sets of the four headline numbers. Exactly `0.0`
    /// where the construction is not sampled, because there the error is zero.
    pub break_even_se: f64,
    pub gross_growth_se: f64,
    pub net_growth_se: f64,
    pub sharpe_se: f64,
    pub replicates: usize,
    pub mechanism: HorizonMechanism,
}

impl HorizonRow {
    /// Whether the row has enough non-overlapping periods for its risk statistics to be
    /// information rather than noise.
    pub fn credible(&self) -> bool {
        self.periods >= MIN_CREDIBLE_PERIODS
    }

    /// Whether the row's `k` is past the horizon the belief-advance mechanism has ever been
    /// diagnosed at. Only the sampled construction advances a belief, so only it extrapolates.
    pub fn extrapolates_dynamics(&self) -> bool {
        self.construction.is_sampled() && self.k > DYNAMICS_DIAGNOSED_HORIZON
    }

    pub fn flags(&self) -> &'static str {
        match (self.credible(), self.extrapolates_dynamics()) {
            (true, false) => "",
            (true, true) => "DYN-EXTRAP",
            (false, false) => "FEW-PERIODS",
            (false, true) => "FEW-PERIODS DYN-EXTRAP",
        }
    }
}

/// Everything the sweep measured, plus the panel it measured on.
#[derive(Clone, Debug)]
pub struct HorizonFrontier {
    pub rows: Vec<HorizonRow>,
    pub gross_cap: f64,
    pub cost_bps: f64,
    pub cost_label: String,
    pub samples: usize,
    pub replicates: usize,
    pub instants: usize,
    pub symbols: usize,
    pub mean_breadth: f64,
    pub trading_days: usize,
    pub span_years: f64,
    pub first_ts_ms: i64,
    pub last_ts_ms: i64,
    pub checkpoint: String,
    pub lineage_sha256: String,
}

impl HorizonFrontier {
    pub fn row(&self, k: usize, construction: Construction, policy: Policy) -> Option<&HorizonRow> {
        self.rows
            .iter()
            .find(|r| r.k == k && r.construction == construction && r.policy == policy)
    }

    /// The largest break-even the MODEL achieves at any `k` under the HORIZON construction.
    /// The verdict, in one call.
    pub fn best_model_horizon(&self) -> Option<&HorizonRow> {
        self.rows
            .iter()
            .filter(|r| r.construction == Construction::Horizon && r.policy == Policy::Model)
            .max_by(|a, b| {
                a.metrics
                    .break_even_cost_bps
                    .total_cmp(&b.metrics.break_even_cost_bps)
            })
    }

    /// The best break-even any BASELINE achieves at the same `k` and construction as `row`. A
    /// model row that does not beat this is not a model result.
    pub fn best_baseline_at(&self, row: &HorizonRow) -> Option<&HorizonRow> {
        self.rows
            .iter()
            .filter(|r| {
                r.k == row.k
                    && r.construction == row.construction
                    && matches!(
                        r.policy,
                        Policy::EqualWeight | Policy::Marginal | Policy::Oracle
                    )
            })
            .max_by(|a, b| {
                a.metrics
                    .break_even_cost_bps
                    .total_cmp(&b.metrics.break_even_cost_bps)
            })
    }

    /// The one-line verdict this module exists to produce.
    ///
    /// Stated against [`MATCHED_MEASURED_BPS`], the matched impact-free cost over all the traded
    /// names, and against [`MATCHED_DEEPEST_DECILE_BPS`] as the floor: a break-even under the
    /// floor fails even the CHEAPEST matched cell that exists, under no impact model at all,
    /// which is a stronger statement than failing the headline figure and the one conclusion here
    /// that survives the edge being remeasured on a liquidity-restricted book.
    pub fn verdict(&self) -> String {
        let Some(best) = self.best_model_horizon() else {
            return "no horizon row was measured".to_owned();
        };
        let baseline = self.best_baseline_at(best);
        let beaten = baseline
            .is_some_and(|b| b.metrics.break_even_cost_bps >= best.metrics.break_even_cost_bps);
        let bps = best.metrics.break_even_cost_bps;
        format!(
            "model under the horizon construction: best break-even {bps:.4} +/- {:.4} bps at \
             k={} ({} periods{}). Against the MATCHED measured impact-free cost of \
             {MATCHED_MEASURED_BPS:.3} bps it is {}; against the \
             {MATCHED_DEEPEST_DECILE_BPS:.3} bps MATCHED deepest-decile FLOOR (equal-weighted \
             mean over 43 of the traded names, cost-restricted against edge-unrestricted) it is \
             {}; against the \
             {MATCHED_ALL_IN_BPS:.3} bps matched \
             all-in figure it is {}. Best baseline at that k is {} at {:.4} bps, which {} the \
             model.",
            best.break_even_se,
            best.k,
            best.periods,
            if best.flags().is_empty() {
                String::new()
            } else {
                format!(", {}", best.flags())
            },
            side(bps, MATCHED_MEASURED_BPS),
            side(bps, MATCHED_DEEPEST_DECILE_BPS),
            side(bps, MATCHED_ALL_IN_BPS),
            baseline.map_or("none", |b| b.policy.name()),
            baseline.map_or(f64::NAN, |b| b.metrics.break_even_cost_bps),
            if beaten { "BEATS" } else { "does not beat" },
        )
    }

    /// A printable table of the whole sweep.
    pub fn table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "horizon sweep: {} symbols, {} instants, mean breadth {:.1}, {} trading days, \
             {:.3} years, gross {:.1}x, cost {}, {} samples x {} replicates\n\
             panel [{}, {}]  checkpoint {}  lineage {}\n",
            self.symbols,
            self.instants,
            self.mean_breadth,
            self.trading_days,
            self.span_years,
            self.gross_cap,
            self.cost_label,
            self.samples,
            self.replicates,
            self.first_ts_ms,
            self.last_ts_ms,
            self.checkpoint,
            &self.lineage_sha256[..self.lineage_sha256.len().min(12)],
        ));
        out.push_str(
            "construction        policy                  k  periods    break-even   +/-SE  \
             gross/yr    net/yr  Sharpe  turn/day  rot/per  bind   f1exp  lev-err   mu bps  \
             sig bps   sign  flags\n",
        );
        for row in &self.rows {
            out.push_str(&format!(
                "{:<18}  {:<20}  {:>4}  {:>7}  {:>11.4}  {:>6.4}  {:>8.3}  {:>8.3}  {:>6.2}  \
                 {:>8.2}  {:>7.3}  {:>4.2}  {:>6.3}  {:>7.3}  {:>7.3}  {:>7.2}  {:>5.3}  {}\n",
                row.construction.name(),
                row.policy.name(),
                row.k,
                row.periods,
                displayed_break_even(row.metrics.break_even_cost_bps),
                row.break_even_se,
                row.metrics.gross_log_growth_per_year,
                row.metrics.log_growth_per_year,
                row.metrics.sharpe,
                row.metrics.turnover_per_day,
                row.metrics.rotation_per_period,
                row.metrics.bound_fraction,
                row.metrics.mean_first_factor_exposure,
                row.metrics.leverage_error,
                row.mechanism.rb_mu_bps,
                row.mechanism.pred_sigma_bps,
                row.mechanism.sign_agreement,
                row.flags(),
            ));
        }
        out.push_str(&format!(
            "\nthe correlation term structure is measurable only to 12 bars on this panel, so \
             every k >= 39 row has NO rho input; `lev-err` is the realized substitute\n\
             VERDICT: {}\n",
            self.verdict()
        ));
        out
    }
}

/// Mean and standard error of a replicate set. A single value has zero error, which is the
/// truth for an unsampled construction rather than a missing measurement.
fn replicate_stats(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (values[0], 0.0);
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    if finite.len() < 2 {
        return (mean, 0.0);
    }
    let var =
        finite.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (finite.len() - 1) as f64;
    (mean, (var / finite.len() as f64).sqrt())
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HorizonArgs {
    /// Directory of `<SYM>.<res>.bars` files.
    pub bars_dir: PathBuf,
    /// The checkpoint to trade. Its metadata and supports sidecars resolve beside it.
    pub checkpoint: PathBuf,
    /// Generation directory the chart lands in.
    pub gens_dir: PathBuf,
    pub res_secs: u32,
    pub device: Device,
    /// The PINNED global split, so the panel is held out by construction.
    pub split_bounds: (i64, i64),
    pub max_symbols: usize,
    pub max_instants: usize,
    pub cost_bps: f32,
    pub capital_usd: f64,
    pub gross_cap: f64,
    pub samples: usize,
    pub replicates: usize,
    /// Seed of the first replicate. Replicate `r` uses `seed + r`.
    pub seed: i64,
    pub label: String,
}

impl HorizonArgs {
    /// The configuration the quoted numbers are measured at.
    pub fn defaults(bars_dir: PathBuf, checkpoint: PathBuf, gens_dir: PathBuf) -> Self {
        Self {
            bars_dir,
            checkpoint,
            gens_dir,
            res_secs: 300,
            device: Device::cuda_if_available(),
            split_bounds: crate::data::ingest::PINNED_SPLIT_BOUNDS,
            max_symbols: 48,
            max_instants: 7_800,
            cost_bps: DEFAULT_COST_BPS,
            capital_usd: 1.0e7,
            gross_cap: DEFAULT_GROSS_CAP,
            samples: DEFAULT_SAMPLES,
            replicates: DEFAULT_REPLICATES,
            seed: 0x5EED,
            label: "horizon".to_owned(),
        }
    }
}

/// Build the held-out panel, scan it once, then sweep every holding period under every
/// construction and write the chart.
pub fn run_horizon_sweep(args: &HorizonArgs) -> Result<HorizonFrontier> {
    // The world model asserts a bf16 autocast on CUDA and it is right to: a book measured
    // under a different numeric regime than the one the weights were selected under is
    // measuring a different model. The moment extractions opt out locally.
    crate::torch::cuda::cfg::configure_cuda();
    ensure!(
        args.samples >= 2,
        "a sampled aggregate law needs at least two paths, got {}",
        args.samples
    );
    ensure!(args.replicates >= 1, "the sweep needs at least one replicate");
    let (val_start, val_end) = args.split_bounds;
    let config = PanelConfig::new((val_start, val_end), args.max_symbols, args.max_instants);
    let corpus = BarCorpus::load_with_bounds(
        &args.bars_dir,
        args.res_secs,
        config.min_history + ADV_TRAILING_BARS,
        (val_start, val_end),
    )?;
    ensure!(
        corpus.split_bounds() == (val_start, val_end),
        "the corpus did not take the pinned split bounds"
    );
    let panel = Panel::build(&corpus, &config)?;

    let metadata = world_model_metadata_path(&args.checkpoint);
    let model = BarWorldModel::load(&args.checkpoint, &metadata, args.device)
        .with_context(|| format!("loading {}", args.checkpoint.display()))?;
    ensure!(
        model.all_parameters_frozen(),
        "the checkpoint loaded for a horizon sweep is still trainable"
    );
    let supports = model
        .supports_for(args.res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {}s", args.res_secs))?;

    let started = std::time::Instant::now();
    let beliefs = scan_panel(&model, &corpus, &panel, args.res_secs)?;
    println!(
        "[horizon] scanned {} panel entries in {:.1}s, belief cache {:.2} GiB",
        beliefs.entries(),
        started.elapsed().as_secs_f64(),
        beliefs.bytes() as f64 / (1u64 << 30) as f64
    );
    let marginal_panel = marginal_forecasts(&panel, supports);
    let marginal = marginal_panel
        .first()
        .cloned()
        .context("the panel has no instants")?;

    let cost = FlatCost::new(args.cost_bps);
    let mut rows = Vec::new();
    for &k in &HOLD_HORIZONS {
        let periods = schedule(&corpus, &panel, &beliefs, k)?;
        for &construction in &CONSTRUCTIONS {
            // At k = 1 the aggregate IS the one-bar law, so the sampled construction's DRIFT is
            // exactly the stale construction's (its only term is the real belief's own
            // conditional mean) and the k = 1 rows sitting on top of each other is the
            // validation of the sampling path. Its VARIANCE is not identical and should not be:
            // the exact figure is the variance of the bin CENTERS, while a draw is uniform
            // WITHIN its bin, so the sampled figure carries the mean within-bin dispersion on
            // top. That gap belongs to the 128-bin discretization, not to the estimator.
            let replicates = if construction.is_sampled() {
                args.replicates
            } else {
                1
            };
            let mut per_policy: BTreeMap<usize, Vec<HorizonMetrics>> = BTreeMap::new();
            let mut mechanism = HorizonMechanism::default();
            for replicate in 0..replicates {
                let sampled = if construction.is_sampled() {
                    tch::manual_seed(args.seed + replicate as i64);
                    Some(horizon_laws(
                        &model,
                        &corpus,
                        &panel,
                        &beliefs,
                        &periods,
                        args.res_secs,
                        args.samples,
                    )?)
                } else {
                    None
                };
                let inputs = build_inputs(
                    construction,
                    &beliefs,
                    &periods,
                    sampled.as_deref(),
                    &marginal,
                )?;
                if replicate == 0 {
                    mechanism = HorizonMechanism::of(&periods, &inputs);
                }
                for (p, &policy) in POLICIES.iter().enumerate() {
                    let metrics = measure(
                        &panel,
                        &periods,
                        &inputs,
                        policy,
                        k,
                        args.gross_cap,
                        &cost,
                        args.capital_usd,
                    )?;
                    per_policy.entry(p).or_default().push(metrics);
                }
            }
            for (p, &policy) in POLICIES.iter().enumerate() {
                let set = &per_policy[&p];
                let pick = |f: fn(&HorizonMetrics) -> f64| -> (f64, f64) {
                    replicate_stats(&set.iter().map(f).collect::<Vec<_>>())
                };
                let (break_even, break_even_se) =
                    pick(|m| displayed_break_even(m.break_even_cost_bps));
                let (gross_growth, gross_growth_se) = pick(|m| m.gross_log_growth_per_year);
                let (net_growth, net_growth_se) = pick(|m| m.log_growth_per_year);
                let (sharpe, sharpe_se) = pick(|m| m.sharpe);
                let mut metrics = set[0];
                metrics.break_even_cost_bps = break_even;
                metrics.gross_log_growth_per_year = gross_growth;
                metrics.log_growth_per_year = net_growth;
                metrics.sharpe = sharpe;
                rows.push(HorizonRow {
                    k,
                    construction,
                    policy,
                    periods: periods.len(),
                    metrics,
                    break_even_se,
                    gross_growth_se,
                    net_growth_se,
                    sharpe_se,
                    replicates,
                    mechanism,
                });
            }
        }
        println!(
            "[horizon] k={k} done at {:.1}s ({} periods)",
            started.elapsed().as_secs_f64(),
            periods.len()
        );
    }

    let breadth = panel.breadth();
    let frontier = HorizonFrontier {
        rows,
        gross_cap: args.gross_cap,
        cost_bps: f64::from(args.cost_bps),
        cost_label: format!("flat {:.2} bps one-way", args.cost_bps),
        samples: args.samples,
        replicates: args.replicates,
        instants: panel.instants(),
        symbols: panel.symbols().len(),
        mean_breadth: breadth.mean,
        trading_days: panel.trading_days(),
        span_years: panel.span_years(),
        first_ts_ms: panel.slices().first().map_or(0, |s| s.ts_ms),
        last_ts_ms: panel.slices().last().map_or(0, |s| s.ts_ms),
        checkpoint: args.checkpoint.display().to_string(),
        lineage_sha256: model.lineage_sha256().to_owned(),
    };
    write_horizon_frontier(&args.gens_dir, &args.label, &frontier)?;
    Ok(frontier)
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// Write [`HORIZON_FRONTIER_BASE`]: break-even, growth, turnover, Sharpe, factor exposure and
/// leverage error against the holding horizon, for every policy under every construction,
/// plus the replicate standard errors and the mechanism columns behind the model's curve.
///
/// The x-axis is the INDEX into [`HOLD_HORIZONS`], with `k` itself carried as a series, for
/// the same reason [`super::portfolio`]'s frontier carries its band: a `MultiLine` report has
/// no independent x values, and an axis running 1..390 on a linear scale hides everything
/// below a day.
pub fn write_horizon_frontier(dir: &Path, label: &str, frontier: &HorizonFrontier) -> Result<()> {
    ensure!(
        !frontier.rows.is_empty(),
        "the horizon sweep measured nothing, so there is nothing to write"
    );
    let ks: Vec<usize> = HOLD_HORIZONS.to_vec();
    let mut series = vec![
        ReportSeries {
            label: "k (bars held)".to_owned(),
            values: ks.iter().map(|k| *k as f32).collect(),
        },
        ReportSeries {
            label: "periods".to_owned(),
            values: ks
                .iter()
                .map(|k| {
                    frontier
                        .rows
                        .iter()
                        .find(|r| r.k == *k)
                        .map_or(f32::NAN, |r| r.periods as f32)
                })
                .collect(),
        },
        ReportSeries {
            label: format!(
                "matched measured cost {MATCHED_MEASURED_BPS:.3} bps (equal-weighted, 256 traded)"
            ),
            values: vec![MATCHED_MEASURED_BPS as f32; ks.len()],
        },
        ReportSeries {
            label: format!(
                "universe measured cost {UNIVERSE_MEASURED_BPS:.3} bps (equal-weighted, 5,297 \
                 symbols)"
            ),
            values: vec![UNIVERSE_MEASURED_BPS as f32; ks.len()],
        },
        ReportSeries {
            label: format!(
                "matched deepest-decile measured cost {MATCHED_DEEPEST_DECILE_BPS:.3} bps \
                 (equal-weighted mean, 43 of 256 traded names, \
                 +/-{MATCHED_DEEPEST_DECILE_BOUNDARY_BPS:.3} worst-case one-name boundary)"
            ),
            values: vec![MATCHED_DEEPEST_DECILE_BPS as f32; ks.len()],
        },
        ReportSeries {
            label: format!("matched all-in cost {MATCHED_ALL_IN_BPS:.3} bps"),
            values: vec![MATCHED_ALL_IN_BPS as f32; ks.len()],
        },
    ];

    let at = |k: usize, c: Construction, p: Policy, f: &dyn Fn(&HorizonRow) -> f64| -> f32 {
        frontier.row(k, c, p).map_or(f32::NAN, |r| f(r) as f32)
    };
    type Pick = &'static dyn Fn(&HorizonRow) -> f64;
    let per_policy: [(&str, Pick); 7] = [
        (
            "break-even bps",
            &|r: &HorizonRow| displayed_break_even(r.metrics.break_even_cost_bps),
        ),
        ("gross log growth/yr", &|r: &HorizonRow| {
            r.metrics.gross_log_growth_per_year
        }),
        ("net log growth/yr", &|r: &HorizonRow| {
            r.metrics.log_growth_per_year
        }),
        ("turnover/day", &|r: &HorizonRow| {
            r.metrics.turnover_per_day
        }),
        ("Sharpe", &|r: &HorizonRow| r.metrics.sharpe),
        ("first-factor exposure", &|r: &HorizonRow| {
            r.metrics.mean_first_factor_exposure
        }),
        ("leverage error", &|r: &HorizonRow| {
            r.metrics.leverage_error
        }),
    ];
    for &construction in &CONSTRUCTIONS {
        for &policy in &POLICIES {
            for (name, pick) in &per_policy {
                series.push(ReportSeries {
                    label: format!("{} {} {name}", construction.name(), policy.name()),
                    values: ks
                        .iter()
                        .map(|k| at(*k, construction, policy, *pick))
                        .collect(),
                });
            }
        }
    }
    // The model's own error bars and the mechanism behind its curve. Only the model rows carry
    // these: a baseline has no forecast, so it has neither a drift nor a standard error of one.
    let model_only: [(&str, Pick); 11] = [
        ("break-even SE", &|r: &HorizonRow| r.break_even_se),
        ("gross growth SE", &|r: &HorizonRow| r.gross_growth_se),
        ("net growth SE", &|r: &HorizonRow| r.net_growth_se),
        ("Sharpe SE", &|r: &HorizonRow| r.sharpe_se),
        ("predicted drift bps", &|r: &HorizonRow| {
            r.mechanism.rb_mu_bps
        }),
        ("plain-estimator drift bps", &|r: &HorizonRow| {
            r.mechanism.plain_mu_bps
        }),
        ("drift MC SE bps", &|r: &HorizonRow| r.mechanism.rb_mu_se_bps),
        ("plain drift MC SE bps", &|r: &HorizonRow| {
            r.mechanism.plain_mu_se_bps
        }),
        ("predicted sigma bps", &|r: &HorizonRow| {
            r.mechanism.pred_sigma_bps
        }),
        ("realized sigma bps", &|r: &HorizonRow| {
            r.mechanism.realized_sigma_bps
        }),
        ("sign agreement", &|r: &HorizonRow| {
            r.mechanism.sign_agreement
        }),
    ];
    for (name, pick) in &model_only {
        for &construction in &CONSTRUCTIONS {
            series.push(ReportSeries {
                label: format!("{} model {name}", construction.name()),
                values: ks
                    .iter()
                    .map(|k| at(*k, construction, Policy::Model, *pick))
                    .collect(),
            });
        }
    }
    series.push(ReportSeries {
        label: "realized up fraction".to_owned(),
        values: ks
            .iter()
            .map(|k| {
                at(*k, Construction::Stale, Policy::Model, &|r: &HorizonRow| {
                    r.mechanism.realized_up_fraction
                })
            })
            .collect(),
    });

    let path = dir.join(format!("{HORIZON_FRONTIER_BASE}.report.bin"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    write_report(
        &path,
        &Report {
            title: format!(
                "Break-even vs Holding Horizon at {:.1}x Gross - {label}",
                frontier.gross_cap
            ),
            x_label: Some("holding-horizon index (see the `k (bars held)` series)".to_owned()),
            y_label: Some("annualized; break-even and drift in bps".to_owned()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::MultiLine { series },
        },
    )
    .with_context(|| format!("writing {}", path.display()))?;
    // Reading it back is what turns "the writer ran" into "the chart exists": a truncated or
    // all-NaN series renders as a blank panel and nothing else notices.
    let report = read_report(&path).with_context(|| format!("reading back {}", path.display()))?;
    match report.kind {
        ReportKind::MultiLine { series } => ensure!(
            series.iter().any(|s| s.values.iter().any(|v| v.is_finite())),
            "{HORIZON_FRONTIER_BASE} holds no finite value"
        ),
        other => bail!("{HORIZON_FRONTIER_BASE} came back as {other:?}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::test_rng;
    use crate::torch::bar_dist::{BAR_CHAIN, BAR_DOF};
    use crate::torch::train::portfolio::{backtest, BacktestConfig, PolicyInputs, GROSS_CAPS};
    use crate::torch::world_model::{world_model_supports_path, BarModules, BarWorldModelMetadata};
    use shared::bars::{write_bar_file, PackedBar, FILE_EXTENSION};
    use std::sync::atomic::{AtomicU64, Ordering};
    use tch::nn;

    static SCRATCH: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("horizon_{name}_{}_{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn mix(seed: u64, index: u64) -> f64 {
        crate::torch::dataset::mix64(seed, index) as f64 / u64::MAX as f64
    }

    const RES: u32 = 300;
    const STEP_MS: i64 = RES as i64 * 1_000;
    /// Enough history that the first tradeable bar clears [`BELIEF_PRE_CONTEXT`], and no more:
    /// every extra bar is a trunk token this test pays for on the CPU.
    const HISTORY_BARS: usize = BELIEF_PRE_CONTEXT as usize + 40;
    const VAL_BARS: usize = 48;
    const FIXTURE_SYMBOLS: usize = 3;

    /// A synthetic bar series with a stated random walk. Deterministic in `seed` alone, so two
    /// fixtures built from the same seed hold byte-identical files.
    fn fixture_bars(seed: u64) -> Vec<PackedBar> {
        let base = 1_600_000_000_000i64 / STEP_MS * STEP_MS;
        let mut close = 100.0f32;
        (0..(HISTORY_BARS + VAL_BARS) as u64)
            .map(|slot| {
                let open = close;
                close = (close * (1.0 + 0.01 * (2.0 * mix(seed, 4 * slot) - 1.0) as f32)).max(1.0);
                let spread = (0.004 * mix(seed, 4 * slot + 1)) as f32 * open;
                PackedBar {
                    ts_ms: base + slot as i64 * STEP_MS,
                    open,
                    high: open.max(close) + spread,
                    low: (open.min(close) - spread).max(0.5),
                    close,
                    volume: (1_000.0 + 49_000.0 * mix(seed, 4 * slot + 2)) as f32,
                    vwap: 0.5 * (open + close),
                    trades: 1 + (499.0 * mix(seed, 4 * slot + 3)) as u32,
                }
            })
            .collect()
    }

    /// Rotate the OHLCV payload of `bars[range]` by one slot while leaving every `ts_ms` where
    /// it was.
    ///
    /// This is the permutation the no-lookahead test needs, and the timestamps have to survive
    /// it: the rollout legitimately reads the CALENDAR of the bars it steps over (a halt, a late
    /// open), so moving a timestamp would break an invariance the code never claimed. Moving
    /// prices and volumes is exactly the claim.
    fn rotate_payload(bars: &mut [PackedBar], from: usize, len: usize) {
        assert!(len >= 2, "a rotation of fewer than two bars is the identity");
        let stamps: Vec<i64> = bars[from..from + len].iter().map(|b| b.ts_ms).collect();
        bars[from..from + len].rotate_left(1);
        for (bar, ts) in bars[from..from + len].iter_mut().zip(&stamps) {
            bar.ts_ms = *ts;
        }
    }

    /// A corpus, a randomized frozen checkpoint and the panel over the held-out span.
    ///
    /// The checkpoint's weights are RANDOMIZED rather than left at `BarModules::new`'s
    /// initialization, and that is load-bearing rather than decorative: an untrained
    /// [`crate::torch::world_model::BarDynamics`] is documented to be the identity and an
    /// untrained head emits one law for every belief, so on a zero-init checkpoint every
    /// forecast in this module would be the same number and every equality assertion below
    /// would hold no matter how badly the code leaked. Each test that asserts an equality also
    /// asserts that the laws it compared are spread out, which is what turns that from an
    /// intention into a check.
    struct Fixture {
        dir: PathBuf,
        corpus: BarCorpus,
        panel: Panel,
        model: BarWorldModel,
        bounds: (i64, i64),
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    impl Fixture {
        /// `mutate` sees each symbol's bars before they are written, so a test can move the
        /// realized future out from under a forecast.
        ///
        /// # The seeding order is load-bearing, and it is a trap
        ///
        /// [`BarModules::new`] (`torch::world_model`) builds every projection through
        /// `uniform_init`, which draws from the GLOBAL torch generator. So the seed must be set
        /// BEFORE the constructor runs, not before the perturbation that follows it. Seeding
        /// after `new` makes only the perturbation reproducible and leaves the base weights
        /// dependent on whatever the harness happened to draw earlier in the process.
        ///
        /// The failure that produces is worth naming, because it cost a debug cycle here and it
        /// does not look like what it is: two fixtures built by identical code hold DIFFERENT
        /// checkpoints, so a differential test over them - "permute the realized future, the
        /// forecast must not move" - sees two completely unrelated belief vectors and reports a
        /// LOOKAHEAD LEAK. The signature is that the two beliefs are unrelated rather than
        /// slightly displaced: a real one-bar leak perturbs a belief, it does not replace it.
        ///
        /// Any test in this repository that builds two model instances and compares them is
        /// exposed to this, which is several. `RngIsolation` found the sibling defect in
        /// production code the same day (`build_trainer` reseeding the global generator), so
        /// this is the second instance of one class: a global generator consumed by a
        /// constructor that does not advertise it.
        fn new(label: &str, mutate: impl Fn(usize, &mut Vec<PackedBar>)) -> Self {
            let dir = scratch_dir(label);
            let mut first_val_ts = 0i64;
            for index in 0..FIXTURE_SYMBOLS {
                let mut bars = fixture_bars(11 + index as u64);
                first_val_ts = bars[HISTORY_BARS].ts_ms;
                mutate(index, &mut bars);
                let symbol = format!("S{index}");
                write_bar_file(
                    &dir.join(format!("{symbol}.{RES}.{FILE_EXTENSION}")),
                    &symbol,
                    RES,
                    &bars,
                )
                .expect("write fixture bars");
            }
            let last_ts = first_val_ts + (VAL_BARS as i64) * STEP_MS;
            let bounds = (first_val_ts, last_ts);
            let corpus = BarCorpus::load_with_bounds(&dir, RES, HISTORY_BARS, bounds)
                .expect("fixture corpus");
            assert_eq!(corpus.split_bounds(), bounds);

            // The supports are fitted on TRAIN bars only, which the mutations below never
            // touch, so two fixtures that differ by a val-span permutation share bin geometry
            // exactly and a law comparison between them is a comparison of the same quantity.
            let weights = dir.join("world_model.ot");
            corpus
                .fit_supports(4_096, 5)
                .save(&world_model_supports_path(&weights, RES))
                .expect("save supports");
            // The seed goes BEFORE `BarModules::new`, not after: `new` runs `uniform_init` on
            // every projection and therefore consumes the GLOBAL torch RNG. Seeding after it
            // makes only the perturbation reproducible, leaves the base weights dependent on
            // whatever the harness drew before this fixture, and produces two fixtures with
            // different checkpoints — which reads exactly like a lookahead failure and is not
            // one.
            let mut vs = nn::VarStore::new(Device::Cpu);
            tch::manual_seed(4_242);
            let _modules = BarModules::new(&vs.root());
            tch::no_grad(|| {
                // One draw per variable in NAME order, so the perturbation does not depend on
                // the var store's hash iteration order either.
                let mut named: Vec<(String, Tensor)> = vs.variables().into_iter().collect();
                named.sort_by(|a, b| a.0.cmp(&b.0));
                for (_, mut tensor) in named {
                    let perturbed = &tensor + Tensor::randn_like(&tensor) * 0.05;
                    tensor.copy_(&perturbed);
                }
            });
            vs.freeze();
            vs.save(&weights).expect("save weights");
            let metadata = BarWorldModelMetadata::save_for_checkpoint(&weights, &[RES], RES)
                .expect("save metadata");
            let model =
                BarWorldModel::load(&weights, &metadata, Device::Cpu).expect("load fixture model");

            let config = PanelConfig {
                start_ts_ms: bounds.0,
                end_ts_ms: bounds.1,
                max_symbols: FIXTURE_SYMBOLS,
                min_history: BELIEF_PRE_CONTEXT as usize + 1,
                max_instants: VAL_BARS,
            };
            let panel = Panel::build(&corpus, &config).expect("fixture panel");
            assert!(
                panel.instants() >= 16 && panel.symbols().len() == FIXTURE_SYMBOLS,
                "the fixture panel is {} instants over {} symbols, too small to schedule",
                panel.instants(),
                panel.symbols().len()
            );
            Self {
                dir,
                corpus,
                panel,
                model,
                bounds,
            }
        }

        fn beliefs(&self) -> PanelBeliefs {
            scan_panel(&self.model, &self.corpus, &self.panel, RES).expect("panel scan")
        }

        fn marginal(&self) -> PanelForecast {
            let supports = self.model.supports_for(RES).expect("fixture supports");
            marginal_forecasts(&self.panel, supports)
                .first()
                .cloned()
                .expect("a marginal forecast")
        }
    }

    /// Spread of a set of numbers, as the non-vacuity witness every equality test below carries.
    fn spread(values: &[f64]) -> f64 {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() < 2 {
            return 0.0;
        }
        let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
        max - min
    }

    // -----------------------------------------------------------------------
    // Arithmetic, no model
    // -----------------------------------------------------------------------

    /// The lognormal closure has to share the EXACT solve's sign and its zero, because those are
    /// the only two properties that survive the gross projection into the book: the projection
    /// destroys the scale, so a book trades the sign and the cross-sectional ordering of the
    /// Kelly vector and nothing else.
    ///
    /// Pinned against [`kelly_fractions`] — the bisection [`super::portfolio`] sizes with — on a
    /// discretized lognormal whose two moments are the closure's own inputs. A second-order
    /// closure cannot reproduce the exact magnitude and is not asked to; it is asked to agree
    /// on which side of zero it is and on where zero is.
    #[test]
    fn the_moment_closure_shares_the_exact_solves_sign_and_zero() {
        let var = 1.0e-5f64;
        let sd = var.sqrt();
        // A 128-bin discretization of `N(mu, var)` in LOG space, on the same +/- 6 sigma grid for
        // every `mu`, converted to simple returns exactly as `bin_returns` does.
        let exact = |mu: f64| -> f64 {
            let bins = NUM_BAR_BINS as usize;
            let (lo, hi) = (-6.0 * sd, 6.0 * sd);
            let width = (hi - lo) / bins as f64;
            let mut probs = Vec::with_capacity(bins);
            let mut returns = Vec::with_capacity(bins);
            for bin in 0..bins {
                let center = lo + width * (bin as f64 + 0.5);
                let z = (center - mu) / sd;
                probs.push((-0.5 * z * z).exp());
                returns.push(center.exp_m1());
            }
            let mass: f64 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= mass;
            }
            let probs = Tensor::from_slice(&probs).view([1, NUM_BAR_BINS]);
            let returns = Tensor::from_slice(&returns).view([1, NUM_BAR_BINS]);
            kelly_fractions(&probs, &returns, FREE_LEVERAGE).double_value(&[0])
        };

        // At `mu_log = 0` the SIMPLE return still has a positive mean, `exp(var/2) - 1`, so the
        // log-optimal fraction is positive rather than zero. Both must say so.
        assert!(
            closure_kelly(0.0, var) > 0.0 && exact(0.0) > 0.0,
            "at mu_log = 0 both fractions must be positive: closure {} exact {}",
            closure_kelly(0.0, var),
            exact(0.0)
        );
        // The zero sits where the simple mean vanishes, at `mu_log = -var/2`, for both.
        let at_zero = closure_kelly(-0.5 * var, var);
        assert!(
            at_zero.abs() < 1.0e-4 * closure_kelly(1.0e-3, var).abs(),
            "the closure's zero should sit at mu_log = -var/2, got f = {at_zero}"
        );
        assert!(
            exact(-0.5 * var).abs() < 0.05 * exact(1.0e-3).abs(),
            "the exact solve's zero is not at mu_log = -var/2 either, so the grid is too coarse \
             to be testing anything: f = {}",
            exact(-0.5 * var)
        );
        // Sign agreement across the range a 5-minute bar actually spans.
        let mut nonzero = 0usize;
        for step in 0..32 {
            let mu = -2.0e-3 + 4.0e-3 * f64::from(step) / 31.0;
            let (closed, solved) = (closure_kelly(mu, var), exact(mu));
            assert_eq!(
                closed > 0.0,
                solved > 0.0,
                "the closure and the exact solve disagree on the SIGN at mu_log = {mu}: \
                 {closed} vs {solved}"
            );
            if solved.abs() > 1.0e-6 {
                nonzero += 1;
            }
        }
        assert!(
            nonzero >= 24,
            "only {nonzero} of 32 probe drifts produced a nonzero exact fraction, so the sign \
             agreement above is mostly the agreement of two zeros"
        );
        // Strictly increasing in the drift, which is what makes the cross-section an ordering.
        let mut previous = f64::NEG_INFINITY;
        for step in 0..64 {
            let mu = -2.0e-3 + 4.0e-3 * f64::from(step) / 63.0;
            let f = closure_kelly(mu, var);
            assert!(
                f > previous,
                "the closure is not monotone in the drift at mu_log = {mu}: {f} <= {previous}"
            );
            previous = f;
        }
        // A degenerate law cannot divide by zero, and a huge drift cannot escape the bound the
        // exact solve is run at.
        assert!(closure_kelly(1.0e-4, 0.0).is_finite());
        assert!(closure_kelly(50.0, 1.0e-5).abs() <= FREE_LEVERAGE);
    }

    /// The report base has to exist on disk and hold finite values for every registered series,
    /// because a `MultiLine` report of all-NaN renders as a blank panel and nothing notices.
    ///
    /// This is the test named in `pretrain_reports::tests::CYCLE_EXEMPT` for
    /// `pretrain_horizon_frontier`.
    #[test]
    fn the_horizon_frontier_base_is_written_and_read_back() {
        let dir = scratch_dir("frontier");
        let mut rows = Vec::new();
        for (index, &k) in HOLD_HORIZONS.iter().enumerate() {
            for &construction in &CONSTRUCTIONS {
                for &policy in &POLICIES {
                    let base = 1.0 + index as f64;
                    rows.push(HorizonRow {
                        k,
                        construction,
                        policy,
                        periods: 400 / k.max(1),
                        metrics: HorizonMetrics {
                            periods: 400 / k.max(1),
                            span_years: 0.42,
                            periods_per_year: 1.0,
                            final_log_wealth: base,
                            log_growth_per_year: -base,
                            gross_log_growth_per_year: base,
                            break_even_cost_bps: base,
                            cagr: 0.1,
                            sharpe: base,
                            vol: 0.2,
                            max_drawdown: 0.3,
                            mean_gross: 2.0,
                            max_gross: 2.1,
                            mean_net: 0.0,
                            turnover_per_day: base,
                            rotation_per_period: 0.5,
                            bound_fraction: 1.0,
                            mean_first_factor_exposure: 0.27,
                            first_factor_share: 0.27,
                            leverage_error: 1.9,
                            ruined_at_period: f64::NAN,
                        },
                        break_even_se: 0.01,
                        gross_growth_se: 0.02,
                        net_growth_se: 0.03,
                        sharpe_se: 0.04,
                        replicates: 3,
                        mechanism: HorizonMechanism {
                            rb_mu_bps: 1.0,
                            plain_mu_bps: 1.1,
                            rb_mu_se_bps: 0.1,
                            plain_mu_se_bps: 3.0,
                            pred_sigma_bps: 31.0,
                            realized_sigma_bps: 30.0,
                            sign_agreement: 0.5,
                            realized_up_fraction: 0.49,
                            legs: 1_000,
                        },
                    });
                }
            }
        }
        let frontier = HorizonFrontier {
            rows,
            gross_cap: DEFAULT_GROSS_CAP,
            cost_bps: 2.0,
            cost_label: "flat 2.00 bps one-way".to_owned(),
            samples: DEFAULT_SAMPLES,
            replicates: DEFAULT_REPLICATES,
            instants: 400,
            symbols: 48,
            mean_breadth: 47.0,
            trading_days: 100,
            span_years: 0.42,
            first_ts_ms: 1,
            last_ts_ms: 2,
            checkpoint: "weights/pretrain_best.ot".to_owned(),
            lineage_sha256: "0".repeat(64),
        };
        write_horizon_frontier(&dir, "unit", &frontier).expect("write the frontier");

        let path = dir.join(format!("{HORIZON_FRONTIER_BASE}.report.bin"));
        let report = read_report(&path).expect("read the frontier back");
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("the frontier must be a MultiLine report");
        };
        // Every construction x policy x column, plus the model-only columns, plus the axis and
        // the four cost reference lines. Each has one value per holding horizon.
        let expected = 4 + 2 + CONSTRUCTIONS.len() * POLICIES.len() * 7 + CONSTRUCTIONS.len() * 11 + 1;
        assert_eq!(series.len(), expected, "the frontier's series count moved");
        for line in &series {
            assert_eq!(
                line.values.len(),
                HOLD_HORIZONS.len(),
                "series `{}` is not one value per holding horizon",
                line.label
            );
            assert!(
                line.values.iter().any(|v| v.is_finite()),
                "series `{}` came back with no finite value at all",
                line.label
            );
        }
        for construction in CONSTRUCTIONS {
            for policy in POLICIES {
                let label = format!("{} {} break-even bps", construction.name(), policy.name());
                assert!(
                    series.iter().any(|s| s.label == label),
                    "the frontier is missing the `{label}` series, so a policy is unreadable \
                     at some horizon"
                );
            }
        }
        assert!(
            series
                .iter()
                .any(|s| s.label.contains(&format!("{MATCHED_MEASURED_BPS:.3}"))),
            "the frontier must carry the matched measured cost as a reference line"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every field unmeasured. Only the field a test is about is then set, so a test cannot
    /// accidentally depend on a plausible-looking default.
    fn nan_metrics() -> HorizonMetrics {
        HorizonMetrics {
            periods: 0,
            span_years: f64::NAN,
            periods_per_year: f64::NAN,
            final_log_wealth: f64::NAN,
            log_growth_per_year: f64::NAN,
            gross_log_growth_per_year: f64::NAN,
            break_even_cost_bps: f64::NAN,
            cagr: f64::NAN,
            sharpe: f64::NAN,
            vol: f64::NAN,
            max_drawdown: f64::NAN,
            mean_gross: f64::NAN,
            max_gross: f64::NAN,
            mean_net: f64::NAN,
            turnover_per_day: f64::NAN,
            rotation_per_period: f64::NAN,
            bound_fraction: f64::NAN,
            mean_first_factor_exposure: f64::NAN,
            first_factor_share: f64::NAN,
            leverage_error: f64::NAN,
            ruined_at_period: f64::NAN,
        }
    }

    /// The verdict has to name the baseline when a baseline wins, because a corner where
    /// equal-weight beats the model is the exact failure the band frontier already walked into.
    #[test]
    fn the_verdict_says_so_when_a_baseline_beats_the_model() {
        let row = |policy: Policy, bps: f64| HorizonRow {
            k: 39,
            construction: Construction::Horizon,
            policy,
            periods: 200,
            metrics: HorizonMetrics {
                break_even_cost_bps: bps,
                ..nan_metrics()
            },
            break_even_se: 0.5,
            gross_growth_se: 0.0,
            net_growth_se: 0.0,
            sharpe_se: 0.0,
            replicates: 3,
            mechanism: HorizonMechanism::default(),
        };
        let mut frontier = HorizonFrontier {
            rows: vec![row(Policy::Model, 6.0), row(Policy::EqualWeight, 20.0)],
            gross_cap: 2.0,
            cost_bps: 2.0,
            cost_label: "flat".to_owned(),
            samples: 8,
            replicates: 3,
            instants: 200,
            symbols: 3,
            mean_breadth: 3.0,
            trading_days: 10,
            span_years: 0.1,
            first_ts_ms: 0,
            last_ts_ms: 1,
            checkpoint: String::new(),
            lineage_sha256: "0".repeat(64),
        };
        let verdict = frontier.verdict();
        assert!(
            verdict.contains("BEATS") && verdict.contains("equal weight"),
            "a losing model must be named as losing: {verdict}"
        );
        assert!(
            verdict.contains(&format!("{MATCHED_MEASURED_BPS:.3}")) && verdict.contains("BELOW"),
            "the verdict must state the matched threshold and which side it falls on: {verdict}"
        );
        // And the other way: a model above every baseline and above the matched cost.
        frontier.rows = vec![row(Policy::Model, 40.0), row(Policy::EqualWeight, 2.0)];
        let verdict = frontier.verdict();
        assert!(
            verdict.contains("does not beat") && verdict.contains("ABOVE"),
            "a winning model must be reported as winning: {verdict}"
        );
    }

    // -----------------------------------------------------------------------
    // No lookahead
    // -----------------------------------------------------------------------

    /// THE test this module exists to make trustworthy: a `k`-bar forecast must not see the `k`
    /// bars it forecasts.
    ///
    /// Two directions, both required, because either one alone is worthless.
    ///
    /// * **Invariance.** Rotating the PRICES and VOLUMES of the bars inside holding window 0,
    ///   on disk, leaves period 0's `k`-bar law BIT-IDENTICAL. Those bars are period 0's future,
    ///   the timestamps are untouched (the rollout legitimately reads the calendar), and nothing
    ///   else about the corpus moved.
    /// * **Non-vacuity.** The SAME rotation is period 1's causal history, and period 1's law
    ///   must therefore MOVE. Without this half, a forecaster that returned a constant would
    ///   pass the invariance half perfectly.
    ///
    /// The spread assertions are the third guard: they prove the compared laws differ across
    /// legs at all, so the bit-identity is not the identity of one repeated number.
    #[test]
    fn the_k_bar_law_ignores_the_realized_bars_inside_its_own_window() {
        let _torch_rng_guard = test_rng::exclusive();
        const K: usize = 4;

        let base = Fixture::new("nolook_base", |_, _| {});
        let base_beliefs = base.beliefs();
        let base_periods =
            schedule(&base.corpus, &base.panel, &base_beliefs, K).expect("base schedule");
        assert!(
            base_periods.len() >= 3,
            "this test needs a period 0 and a period 1, got {}",
            base_periods.len()
        );
        let first_bar = base.panel.bar_index(base_periods[0].instant, 0) as usize;
        tch::manual_seed(7);
        let base_laws = horizon_laws(
            &base.model,
            &base.corpus,
            &base.panel,
            &base_beliefs,
            &base_periods,
            RES,
            8,
        )
        .expect("base laws");

        // The mutation: rotate holding window 0's payload for every symbol. Window 0 spans the
        // panel's first K instants, which for a dense fixture is bars `first_bar .. first_bar + K`.
        let moved = Fixture::new("nolook_moved", |_, bars| {
            rotate_payload(bars, first_bar, K);
        });
        assert_eq!(
            moved.bounds, base.bounds,
            "the two fixtures must be scored against the same split"
        );
        let moved_beliefs = moved.beliefs();
        let moved_periods =
            schedule(&moved.corpus, &moved.panel, &moved_beliefs, K).expect("moved schedule");
        assert_eq!(
            moved_periods.len(),
            base_periods.len(),
            "the rotation must not change the panel's clock"
        );
        tch::manual_seed(7);
        let moved_laws = horizon_laws(
            &moved.model,
            &moved.corpus,
            &moved.panel,
            &moved_beliefs,
            &moved_periods,
            RES,
            8,
        )
        .expect("moved laws");

        // Localize before comparing laws, so a failure names the layer that leaked rather than
        // the layer that reported. Each of these is a strictly stronger claim than the one
        // below it.
        let base_supports = base.model.supports_for(RES).expect("base supports");
        let moved_supports = moved.model.supports_for(RES).expect("moved supports");
        assert_eq!(
            base_supports.centers(DOF_R),
            moved_supports.centers(DOF_R),
            "the two fixtures fitted DIFFERENT bin geometry, so their laws are not comparable \
             at all; the permutation must be confined to the val span"
        );
        for (slot, (&a, &b)) in base_beliefs.row_of[0]
            .iter()
            .zip(&moved_beliefs.row_of[0])
            .enumerate()
        {
            assert_eq!(a, b, "slot {slot} of instant 0 moved in the belief cache");
            assert_eq!(
                base_beliefs.belief_row(a),
                moved_beliefs.belief_row(b),
                "the BELIEF at instant 0 slot {slot} changed when bars at and after instant 0 \
                 were permuted, so `scan_panel` is reading the bar it is about to predict"
            );
            assert_eq!(
                base_beliefs.mu_log[0][slot].to_bits(),
                moved_beliefs.mu_log[0][slot].to_bits(),
                "the one-bar drift at instant 0 slot {slot} changed under the same permutation"
            );
        }

        // Non-vacuity, first pass: the laws being compared are not all the same number.
        let mus: Vec<f64> = base_laws[0].iter().map(|l| l.mu_log).collect();
        assert!(
            spread(&mus) > 1.0e-9,
            "period 0's legs all carry the same drift ({mus:?}), so bit-identity below would \
             prove nothing; the fixture checkpoint is degenerate"
        );

        // INVARIANCE. Period 0's decision stands on bars strictly before `first_bar`, and its
        // rollout reads only its own samples plus the untouched calendar.
        for (leg, (want, got)) in base_laws[0].iter().zip(&moved_laws[0]).enumerate() {
            assert_eq!(
                want.mu_log.to_bits(),
                got.mu_log.to_bits(),
                "leg {leg} of period 0 changed its k-bar drift when the realized bars INSIDE \
                 its own holding window were permuted: {} -> {}. That is lookahead.",
                want.mu_log,
                got.mu_log
            );
            assert_eq!(
                want.var_log.to_bits(),
                got.var_log.to_bits(),
                "leg {leg} of period 0 changed its k-bar variance under the same permutation"
            );
            assert_eq!(
                want.plain_mu_log.to_bits(),
                got.plain_mu_log.to_bits(),
                "leg {leg} of period 0 changed its plain-estimator drift under the same \
                 permutation"
            );
        }

        // NON-VACUITY. The same bars are period 1's causal history, so period 1 must move. A
        // forecaster that ignored its inputs entirely would fail exactly here.
        let changed = base_laws[1]
            .iter()
            .zip(&moved_laws[1])
            .filter(|(want, got)| want.mu_log.to_bits() != got.mu_log.to_bits())
            .count();
        assert!(
            changed > 0,
            "period 1's laws are unchanged after its CAUSAL HISTORY was permuted, so the \
             invariance above is vacuous: the forecast is not reading its history at all"
        );
    }

    /// The one-bar law that sizes every stale row, and every step of every rollout, must be
    /// `p(r | strictly past bars)` — never a law that knows any part of the bar it predicts.
    ///
    /// `r`'s prefix set is DERIVED from [`BAR_CHAIN`] here rather than assumed, so this test
    /// stays correct under any factorization order. `r` currently HEADS the chain, so that set
    /// is empty, `p(r | past)` is the head's `r` row at ANY prefix, and the property asserted
    /// is exactly that: the row is bit-identical across a sweep of prefix assignments and the
    /// panel's drift is its mean. A reorder that hands `r` a prefix breaks the sweep and fails
    /// here rather than silently certifying a teacher-forced law.
    ///
    /// The reference is independent of the path under test: it calls
    /// [`BarEmissionHead::logits`] directly and reduces in `f64`, never touching
    /// [`forecast_r_probs`].
    #[test]
    fn the_one_bar_drift_uses_no_part_of_the_bar_it_predicts() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("marginal", |_, _| {});
        let beliefs = fixture.beliefs();
        let supports = fixture.model.supports_for(RES).expect("fixture supports");
        let centers = supports.centers(DOF_R).to_vec();
        let head = fixture.model.head();

        // `r`'s prefix set, derived rather than assumed: every factor ahead of it in the
        // chain. Empty while `r` heads the chain, which is what makes a direct read a
        // forecast; anything in it would have to be marginalized out instead.
        let r_position = BAR_CHAIN
            .iter()
            .position(|&dof| dof == DOF_R)
            .expect("r is a chain factor");
        assert!(
            BAR_CHAIN[..r_position].is_empty(),
            "r has chain prefix {:?}, so the panel's one-bar law is teacher-forced on it and \
             the direct reference below is the wrong law",
            &BAR_CHAIN[..r_position]
        );
        // The last chain factor carries every prefix slot, so it is the control that proves
        // the head's prefix pathway is live at all.
        let deepest = BAR_CHAIN[BAR_DOF - 1];

        // Prefix assignments the `r` row must be blind to. Bin 0 is the unvisited seed, so the
        // others are the ones that could leak.
        let prefixes: Vec<Tensor> = [0i64, 1, NUM_BAR_BINS / 2, NUM_BAR_BINS - 1]
            .iter()
            .map(|bin| Tensor::full([1, BAR_DOF as i64], *bin, (Kind::Int64, Device::Cpu)))
            .collect();

        let mut checked = 0usize;
        let mut worst = 0.0f64;
        let mut traded_response = 0.0f64;
        let mut deepest_response = 0.0f64;
        let mut seen = Vec::new();
        for (t, row) in beliefs.row_of.iter().enumerate().take(4) {
            for (slot, &belief_row) in row.iter().enumerate() {
                let h =
                    Tensor::from_slice(beliefs.belief_row(belief_row)).view([1, BAR_MODEL_DIM]);
                let (want, traded_drift, deepest_drift) = tch::no_grad(|| {
                    let row_at = |prefix: &Tensor, dof: usize| -> Vec<f64> {
                        Vec::<f64>::try_from(
                            head.logits(&h, prefix)
                                .select(1, dof as i64)
                                .softmax(-1, Kind::Double)
                                .reshape([-1]),
                        )
                        .expect("a probability row")
                    };
                    let spread_of = |a: &[f64], b: &[f64]| -> f64 {
                        a.iter()
                            .zip(b)
                            .map(|(x, y)| (x - y).abs())
                            .fold(0.0f64, f64::max)
                    };
                    let base = row_at(&prefixes[0], DOF_R);
                    let base_deepest = row_at(&prefixes[0], deepest);
                    let mut traded_drift = 0.0f64;
                    let mut deepest_drift = 0.0f64;
                    for prefix in &prefixes[1..] {
                        traded_drift = traded_drift.max(spread_of(&base, &row_at(prefix, DOF_R)));
                        deepest_drift =
                            deepest_drift.max(spread_of(&base_deepest, &row_at(prefix, deepest)));
                    }
                    let mass: f64 = base.iter().sum();
                    assert!(
                        (mass - 1.0).abs() < 1.0e-9,
                        "the reference row at ({t}, {slot}) has mass {mass}"
                    );
                    let mean: f64 = base.iter().zip(&centers).map(|(p, c)| p * c).sum();
                    (mean, traded_drift, deepest_drift)
                });
                let got = beliefs.mu_log[t][slot];
                worst = worst.max((got - want).abs());
                traded_response = traded_response.max(traded_drift);
                deepest_response = deepest_response.max(deepest_drift);
                seen.push(want);
                checked += 1;
            }
        }
        assert!(checked >= 8, "only {checked} beliefs were compared");
        // Both sides are `f32` head logits reduced in `f64`; the quantity is ~1e-4, so 1e-7
        // absolute is three to four significant figures of agreement while still being four
        // orders tighter than the ~1e-3 shift a teacher-forced law would introduce.
        assert!(
            worst < 1.0e-7,
            "the panel's one-bar drift disagrees with a direct read of the head's r row by \
             {worst:.3e} absolute nats, so it is not the law it claims to be"
        );
        // The property. Exactly zero, because a prefix-free row is the SAME arithmetic under
        // every prefix — a tolerance here would pass a law that leaked a little.
        assert_eq!(
            traded_response, 0.0,
            "the head's r row moved across a prefix sweep, so the one-bar law is conditioned \
             on the bar it predicts"
        );
        // Non-vacuity: the prefix pathway is live on this fixture, so the invariance above is
        // the chain position doing its job rather than a dead prefix embedding.
        assert!(
            deepest_response > 1.0e-6,
            "no row of this fixture's head responds to its prefix ({deepest_response:.3e}), so \
             the invariance above proves nothing"
        );
        assert!(
            spread(&seen) > 1.0e-8,
            "every reference drift came out the same number, so the agreement above is the \
             agreement of two constants"
        );
    }

    // -----------------------------------------------------------------------
    // The book
    // -----------------------------------------------------------------------

    /// At `k = 1` and zero cost this book IS [`super::portfolio::backtest`], and it has to be:
    /// a second accounting loop that quietly disagrees with the first would make every row of
    /// the sweep incomparable to the 5-minute number it exists to be compared against.
    ///
    /// Zero cost, because the two loops legitimately differ on what turnover is charged
    /// against — see the module docs — and `1e-12` relative rather than bit-identity because
    /// `portfolio` recomputes the realized simple return from the panel's `f32` log return
    /// while this book recomputes it in `f64` from the corpus closes.
    #[test]
    fn the_one_bar_book_reproduces_the_portfolio_backtest_at_zero_cost() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("onebar", |_, _| {});
        let beliefs = fixture.beliefs();
        let periods = schedule(&fixture.corpus, &fixture.panel, &beliefs, 1).expect("schedule");
        assert_eq!(
            periods.len(),
            fixture.panel.instants(),
            "a one-bar holding period is one period per instant"
        );
        let marginal = fixture.marginal();
        let inputs = build_inputs(
            Construction::StaleExact,
            &beliefs,
            &periods,
            None,
            &marginal,
        )
        .expect("stale-exact inputs");
        let marginal_panel =
            marginal_forecasts(&fixture.panel, fixture.model.supports_for(RES).unwrap());
        let free = FlatCost::new(0.0);

        for &cap in &GROSS_CAPS {
            for &policy in &POLICIES {
                let mine = run_book(
                    &fixture.panel,
                    &periods,
                    &inputs,
                    policy,
                    1,
                    cap,
                    &free,
                    1.0e7,
                )
                .expect("horizon book");
                let theirs = backtest(
                    &fixture.panel,
                    &PolicyInputs {
                        model: &beliefs.one_bar,
                        marginal: &marginal_panel,
                    },
                    policy,
                    cap,
                    &free,
                    &BacktestConfig {
                        capital_usd: 1.0e7,
                        band: 0.0,
                    },
                )
                .expect("portfolio book");
                assert_eq!(
                    mine.log_equity.len(),
                    theirs.log_equity.len(),
                    "{} at {cap}x: the two books have different clocks",
                    policy.name()
                );
                // Absolute, in nats of cumulative log wealth, not relative: log wealth passes
                // through zero, so a relative bound on it is a bound on nothing. `portfolio`
                // reads its realized return from the panel's `f32` log return while this book
                // recomputes it in `f64` from the corpus closes, which is a ~6e-8 relative
                // difference per bar and measured at 9e-11 nats per period here. 1e-6 nats over
                // the whole path is four orders of headroom above that and five orders below any
                // difference a reader would act on.
                for (index, (a, b)) in mine.log_equity.iter().zip(&theirs.log_equity).enumerate() {
                    assert!(
                        (a - b).abs() < 1.0e-6,
                        "{} at {cap}x diverges from the portfolio bench at period {index}: \
                         {a} vs {b}",
                        policy.name()
                    );
                }
                // TURNOVER is where the two books legitimately part, and the gap is bounded
                // rather than tolerated. `portfolio` charges the move from its previous TARGET;
                // this book charges the move from the weight the hold drifted to,
                // `held_i = w_i (1 + R_i) / M` with `M = 1 + sum_j w_j R_j` at zero cost. So
                // `|held_i - w_i| = |w_i| |R_i - (M - 1)| / M` and the two turnovers can differ
                // by at most `gross * (max_i |R_i| + |payoff|) / M`, evaluated on the PREVIOUS
                // period because that is the hold that drifted. Anything larger is not drift.
                assert_eq!(
                    mine.turnover[0], theirs.turnover[0],
                    "{} at {cap}x: both books start flat, so the FIRST rebalance has no drift to \
                     disagree about",
                    policy.name()
                );
                for index in 1..mine.turnover.len() {
                    let previous = index - 1;
                    let max_abs_r = fixture.panel.slices()[previous]
                        .realized_r
                        .iter()
                        .map(|r| f64::from(*r).exp_m1().abs())
                        .fold(0.0, f64::max);
                    let payoff = mine.payoff[previous];
                    // The `1e-6` relative slack is not a fudge: the oracle holds ONE name, so
                    // for it the inequality is an EQUALITY (`max_i |R_i|` is that name's own
                    // return and `payoff` is `w R`), and it was measured to be attained to 2e-10
                    // absolute. The slack covers rounding of a bound assembled from the panel's
                    // `f32` returns while the book compounds in `f64`.
                    let bound = mine.gross[previous] * (max_abs_r + payoff.abs())
                        / (1.0 + payoff).max(1.0e-9)
                        * (1.0 + 1.0e-6)
                        + 1.0e-12;
                    let gap = (mine.turnover[index] - theirs.turnover[index]).abs();
                    assert!(
                        gap <= bound,
                        "{} at {cap}x differs from the portfolio bench by {gap} of turnover at \
                         period {index}, above the {bound} that one bar of buy-and-hold drift \
                         can explain",
                        policy.name()
                    );
                }
                // And the gap must actually BE the drift somewhere, or this book is silently
                // re-imposing its target weights and the k > 1 rows mean nothing.
                if policy == Policy::Model {
                    let moved = (1..mine.turnover.len())
                        .filter(|&i| (mine.turnover[i] - theirs.turnover[i]).abs() > 1.0e-9)
                        .count();
                    assert!(
                        moved > 0,
                        "at {cap}x this book's turnover is identical to the portfolio bench's at \
                         every period, so its positions are not drifting with prices at all"
                    );
                }
            }
        }
        // Non-vacuity: an equality between two flat lines proves nothing.
        let model = run_book(
            &fixture.panel,
            &periods,
            &inputs,
            Policy::Model,
            1,
            DEFAULT_GROSS_CAP,
            &free,
            1.0e7,
        )
        .expect("horizon book");
        assert!(
            spread(&model.log_equity) > 1.0e-6 && model.turnover.iter().sum::<f64>() > 0.0,
            "the compared book never moved, so the agreement above is trivial"
        );
    }

    /// The break-even solve has to be a real bracket of the zero of net growth, not a number
    /// the bisection happened to land on: growth must be positive just below it and negative
    /// just above.
    #[test]
    fn the_break_even_cost_brackets_the_zero_of_net_growth() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("breakeven", |_, _| {});
        let beliefs = fixture.beliefs();
        let periods = schedule(&fixture.corpus, &fixture.panel, &beliefs, 2).expect("schedule");
        let marginal = fixture.marginal();
        let inputs = build_inputs(Construction::Stale, &beliefs, &periods, None, &marginal)
            .expect("stale inputs");

        // The oracle is the one policy guaranteed to have a positive gross edge on any panel,
        // so it is the one whose break-even must be a finite interior crossing.
        let metrics = measure(
            &fixture.panel,
            &periods,
            &inputs,
            Policy::Oracle,
            2,
            DEFAULT_GROSS_CAP,
            &FlatCost::new(0.0),
            1.0e7,
        )
        .expect("oracle metrics");
        let bps = metrics.break_even_cost_bps;
        assert!(
            bps > 0.0 && bps < MAX_BREAK_EVEN_BPS,
            "the oracle's break-even should be a finite interior crossing, got {bps}"
        );
        let growth_at = |cost: f64| -> f64 {
            let book = run_book(
                &fixture.panel,
                &periods,
                &inputs,
                Policy::Oracle,
                2,
                DEFAULT_GROSS_CAP,
                &FlatCost::new(cost as f32),
                1.0e7,
            )
            .expect("book at a trial cost");
            *book.log_equity.last().expect("a curve")
        };
        assert!(
            growth_at(bps * 0.98) > 0.0,
            "net growth is not positive just BELOW the reported break-even of {bps} bps"
        );
        assert!(
            growth_at(bps * 1.02) < 0.0,
            "net growth is not negative just ABOVE the reported break-even of {bps} bps"
        );
        // And a policy with no edge at all must report exactly zero rather than a bisection
        // artifact.
        let flat = HorizonInputs {
            construction: Construction::Stale,
            kelly: periods.iter().map(|p| vec![0.0; p.legs.len()]).collect(),
            pred_var: periods.iter().map(|p| vec![1.0e-5; p.legs.len()]).collect(),
            marginal_kelly: 0.0,
            laws: periods
                .iter()
                .map(|p| vec![AggregateLaw::default(); p.legs.len()])
                .collect(),
        };
        let idle = measure(
            &fixture.panel,
            &periods,
            &flat,
            Policy::Model,
            2,
            DEFAULT_GROSS_CAP,
            &FlatCost::new(0.0),
            1.0e7,
        )
        .expect("idle metrics");
        assert_eq!(
            idle.break_even_cost_bps, 0.0,
            "a book that never trades and never earns must break even at exactly 0 bps"
        );
    }

    /// A held position earns the AGGREGATE move, so the schedule's realized log return has to
    /// be the sum of the per-bar log returns over the window, and the payoff has to be
    /// `exp` of it minus one rather than a sum of per-bar simple returns.
    ///
    /// This is the arithmetic that makes a `k > 1` row mean anything, and getting it wrong is
    /// invisible: a sum of simple returns is within `O(k * sigma^2)` of the truth and would
    /// simply bias every long-horizon row.
    #[test]
    fn the_schedule_aggregates_the_held_move_multiplicatively() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("aggregate", |_, _| {});
        let beliefs = fixture.beliefs();
        for k in [1usize, 3, 8] {
            let periods = schedule(&fixture.corpus, &fixture.panel, &beliefs, k).expect("sched");
            for period in &periods {
                for leg in &period.legs {
                    let bars = fixture.corpus.bars(fixture.panel.series_of(leg.id));
                    let entry = fixture.panel.bar_index(period.instant, leg.slot) as usize;
                    let mut want = 0.0f64;
                    for step in 0..leg.steps {
                        let now = f64::from(bars[entry + step].close);
                        let previous = f64::from(bars[entry + step - 1].close);
                        want += (now / previous).ln();
                    }
                    assert!(
                        (leg.realized_log - want).abs() <= 1.0e-12 * want.abs().max(1.0e-6),
                        "k={k}: the held aggregate of {} over instant {} is {} but the sum of \
                         its {} per-bar log returns is {want}",
                        fixture.panel.symbols()[leg.id as usize],
                        period.instant,
                        leg.realized_log,
                        leg.steps
                    );
                    assert!(
                        leg.steps >= 1 && leg.steps <= k,
                        "k={k}: a leg held {} bars",
                        leg.steps
                    );
                }
            }
            // Non-overlapping, and covering the panel.
            let instants: usize = periods.iter().map(|_| k).sum();
            assert!(
                instants >= fixture.panel.instants(),
                "k={k}: {} periods of {k} bars do not cover {} instants",
                periods.len(),
                fixture.panel.instants()
            );
        }
    }

    /// The Rao-Blackwellized drift estimator has to actually be the promised variance
    /// reduction, because the whole sample budget is set from that claim.
    ///
    /// Two checks, one of which cannot be argued with: at `k = 1` the estimator is EXACT (the
    /// only term is the real belief's own conditional mean, so its Monte-Carlo error is zero
    /// by construction and must be reported as zero), and at `k > 1` its standard error must
    /// be materially below the plain sampled-aggregate estimator's on the same draws.
    #[test]
    fn the_rao_blackwellized_drift_beats_the_plain_estimator() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("variance", |_, _| {});
        let beliefs = fixture.beliefs();

        let one = schedule(&fixture.corpus, &fixture.panel, &beliefs, 1).expect("k=1 schedule");
        tch::manual_seed(19);
        let laws =
            horizon_laws(&fixture.model, &fixture.corpus, &fixture.panel, &beliefs, &one, RES, 8)
                .expect("k=1 laws");
        for (p, row) in laws.iter().enumerate() {
            for (l, law) in row.iter().enumerate() {
                assert_eq!(
                    law.mu_se, 0.0,
                    "the k=1 Rao-Blackwellized drift at ({p}, {l}) reports a Monte-Carlo error \
                     of {}, but its only term is the real belief's exact conditional mean",
                    law.mu_se
                );
                // And it must be that exact mean, not a sampled approximation to it.
                let want = beliefs.mu_log[one[p].instant][one[p].legs[l].slot];
                assert!(
                    (law.mu_log - want).abs() < 1.0e-12,
                    "the k=1 sampled law's drift {} is not the exact one-bar drift {want}",
                    law.mu_log
                );
            }
        }

        let four = schedule(&fixture.corpus, &fixture.panel, &beliefs, 4).expect("k=4 schedule");
        tch::manual_seed(19);
        let laws = horizon_laws(
            &fixture.model,
            &fixture.corpus,
            &fixture.panel,
            &beliefs,
            &four,
            RES,
            32,
        )
        .expect("k=4 laws");
        let mut rb = 0.0f64;
        let mut plain = 0.0f64;
        let mut legs = 0usize;
        for row in &laws {
            for law in row {
                rb += law.mu_se;
                plain += law.plain_mu_se;
                legs += 1;
            }
        }
        assert!(legs > 0, "no legs were sampled");
        let (rb, plain) = (rb / legs as f64, plain / legs as f64);
        assert!(
            plain > 0.0 && rb > 0.0,
            "both estimators must carry a real error bar at k > 1, got rb {rb} plain {plain}"
        );
        assert!(
            rb < 0.5 * plain,
            "the Rao-Blackwellized drift's mean standard error is {rb:.3e} against the plain \
             estimator's {plain:.3e}. The sample budget is sized on this reduction being \
             large, so if it is not, DEFAULT_SAMPLES is wrong."
        );
    }

    /// The horizon construction has to be measuring the `k`-BAR aggregate, and has to be a
    /// different measurement from holding a one-bar forecast. Both halves matter: if it is not
    /// accumulating `k` bars it is not the experiment, and if it does not differ from the
    /// control the sweep's central comparison is comparing a thing with itself.
    ///
    /// The sharp check on accumulation is the VARIANCE. Under any chain, the variance of a sum
    /// of `k` roughly-uncorrelated per-bar returns grows about linearly in `k`, so the sampled
    /// aggregate variance at `k = 4` must be several times the one-bar variance and land near
    /// `4x` it. A rollout that silently dropped its steps, or double-counted them, moves that
    /// ratio out of any plausible band.
    ///
    /// Note what is deliberately NOT asserted: the sampled variance at `k = 1` does NOT equal
    /// the exact one-bar variance, and should not. The exact figure is the variance of the bin
    /// CENTERS under the law; a draw is uniform WITHIN its bin, so the sampled figure carries
    /// the mean within-bin dispersion on top. That gap is a property of the 128-bin
    /// discretization rather than an error, so the test bounds it instead of demanding zero.
    #[test]
    fn the_horizon_construction_accumulates_k_bars_and_differs_from_holding_one() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("constructions", |_, _| {});
        let beliefs = fixture.beliefs();
        let marginal = fixture.marginal();
        let free = FlatCost::new(0.0);

        let sized = |periods: &[Period], laws: Option<&[Vec<AggregateLaw>]>, c: Construction| {
            let inputs = build_inputs(c, &beliefs, periods, laws, &marginal).expect("inputs");
            inputs.kelly.clone()
        };
        let mean_var = |laws: &[Vec<AggregateLaw>]| -> f64 {
            let (sum, count) = laws.iter().flatten().fold((0.0, 0usize), |(s, n), law| {
                (s + law.var_log, n + 1)
            });
            sum / count.max(1) as f64
        };

        // k = 1: the sampled DRIFT is exactly the one-bar drift, which is pinned in
        // `the_rao_blackwellized_drift_beats_the_plain_estimator`. Here the claim is about the
        // variance's SCALE: the sampled figure must sit within a factor of two of the exact one,
        // which is loose enough to admit the within-bin term and tight enough to reject a
        // sampler that is drawing from the wrong support or the wrong DOF.
        let one = schedule(&fixture.corpus, &fixture.panel, &beliefs, 1).expect("k=1");
        tch::manual_seed(3);
        let one_laws = horizon_laws(
            &fixture.model,
            &fixture.corpus,
            &fixture.panel,
            &beliefs,
            &one,
            RES,
            256,
        )
        .expect("k=1 laws");
        let sampled_one = mean_var(&one_laws);
        let exact_one = {
            let (sum, count) = one.iter().fold((0.0, 0usize), |(s, n), period| {
                let row: f64 = period
                    .legs
                    .iter()
                    .map(|leg| beliefs.var_log[period.instant][leg.slot])
                    .sum();
                (s + row, n + period.legs.len())
            });
            sum / count.max(1) as f64
        };
        assert!(
            exact_one > 0.0 && sampled_one > 0.0,
            "both one-bar variances must be positive: sampled {sampled_one} exact {exact_one}"
        );
        let ratio = sampled_one / exact_one;
        assert!(
            (0.5..2.0).contains(&ratio),
            "the sampled one-bar variance is {ratio:.3}x the exact bin-center variance \
             (sampled {sampled_one:.3e}, exact {exact_one:.3e}). Within-bin dispersion explains \
             a modest excess; a factor outside [0.5, 2] means the rollout is drawing from the \
             wrong law."
        );
        // And at k = 1 the two constructions must still SIZE in the same direction, because the
        // drift they share is what decides the sign and the gross projection destroys the rest.
        let stale_one = sized(&one, None, Construction::Stale);
        let horizon_one = sized(&one, Some(&one_laws), Construction::Horizon);
        let mut agreeing = 0usize;
        let mut total = 0usize;
        for (a, b) in stale_one.iter().zip(&horizon_one) {
            for (x, y) in a.iter().zip(b) {
                total += 1;
                if (*x > 0.0) == (*y > 0.0) {
                    agreeing += 1;
                }
            }
        }
        assert_eq!(
            agreeing, total,
            "at k=1 the two constructions share their drift exactly, so they cannot disagree on \
             the SIGN of a single position, yet {} of {total} disagree",
            total - agreeing
        );

        let four = schedule(&fixture.corpus, &fixture.panel, &beliefs, 4).expect("k=4");
        tch::manual_seed(3);
        let four_laws = horizon_laws(
            &fixture.model,
            &fixture.corpus,
            &fixture.panel,
            &beliefs,
            &four,
            RES,
            256,
        )
        .expect("k=4 laws");
        // ACCUMULATION. Four bars of a roughly-uncorrelated chain carry about four times the
        // variance of one, so the ratio has to land in a band around 4. A rollout that ran one
        // step, or ran four but summed only the last, or double-counted the padded steps, all
        // land outside [2, 8].
        let sampled_four = mean_var(&four_laws);
        let growth = sampled_four / sampled_one;
        assert!(
            (2.0..8.0).contains(&growth),
            "the k=4 aggregate variance is {growth:.3}x the k=1 figure (k=4 {sampled_four:.3e}, \
             k=1 {sampled_one:.3e}). A four-bar sum should be near 4x, so the rollout is not \
             accumulating four bars."
        );
        // And the drift accumulates too: the mean ABSOLUTE drift must grow with the horizon,
        // because it is a sum of k conditional means rather than one of them.
        let abs_drift = |laws: &[Vec<AggregateLaw>]| -> f64 {
            let (sum, count) = laws
                .iter()
                .flatten()
                .fold((0.0, 0usize), |(s, n), law| (s + law.mu_log.abs(), n + 1));
            sum / count.max(1) as f64
        };
        let (one_drift, four_drift) = (abs_drift(&one_laws), abs_drift(&four_laws));
        assert!(
            four_drift > 1.3 * one_drift,
            "the mean absolute k=4 drift ({four_drift:.3e}) is not materially above the k=1 \
             figure ({one_drift:.3e}), so the rollout's later steps are contributing nothing"
        );
        let stale = sized(&four, None, Construction::Stale);
        let horizon = sized(&four, Some(&four_laws), Construction::Horizon);
        let differing = stale
            .iter()
            .flatten()
            .zip(horizon.iter().flatten())
            .filter(|(a, b)| (*a - *b).abs() > 1.0e-9 * a.abs().max(1.0))
            .count();
        assert!(
            differing > 0,
            "at k=4 the one-bar-and-hold control and the k-bar-aggregate experiment size \
             identically, so the sweep's central comparison is vacuous"
        );

        // And the books they produce are distinguishable, which is the thing the report claims.
        let of = |inputs: &HorizonInputs| {
            measure(
                &fixture.panel,
                &four,
                inputs,
                Policy::Model,
                4,
                DEFAULT_GROSS_CAP,
                &free,
                1.0e7,
            )
            .expect("metrics")
            .gross_log_growth_per_year
        };
        let a = of(&build_inputs(Construction::Stale, &beliefs, &four, None, &marginal).unwrap());
        let b = of(
            &build_inputs(
                Construction::Horizon,
                &beliefs,
                &four,
                Some(&four_laws),
                &marginal,
            )
            .unwrap(),
        );
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() > 0.0,
            "the two constructions produce the same gross growth at k=4 ({a} vs {b})"
        );
    }
}
