//! What it COSTS to trade the predictor, how much AUM that cost supports, and how
//! correlated the book the sizing implicitly assumes away actually is.
//!
//! # Why a constant was the answer, and therefore the bug
//!
//! [`super::trade_bench`] charges [`super::trade_bench::DEFAULT_COST_BPS`] — a flat `2.00`
//! bps one-way — and reports a break-even of `3.29` bps at step 20000. The whole verdict
//! therefore sits `1.29` bps from a sign change, which means the assumed constant, not the
//! model, is deciding whether the strategy exists. That is not a cost model; it is a coin
//! flip with a citation.
//!
//! Meanwhile every [`PackedBar`] in the corpus carries `high`, `low`, `close`, `volume`,
//! `vwap` and `trades`, and the model reads none of them beyond OHLC. Those fields are
//! exactly what a spread and an ADV are made of. So this module measures cost from the data
//! instead of asserting it:
//!
//! * **Spread** — [`corwin_schultz_alpha`], the Corwin & Schultz (2012) high-low estimator,
//!   cross-checked against [`abdi_ranaldo_moment`], the Abdi & Ranaldo (2017)
//!   close-high-low estimator. Two independent estimators of the same quantity, because one
//!   cannot distinguish "this name is genuinely wide" from "my estimator broke at this
//!   sampling frequency".
//! * **Fees** — commission per SHARE divided by the symbol's own price, so a `$4` name
//!   correctly pays a hundred times the commission in bps that a `$400` name pays. That is
//!   a real, large, entirely data-driven cross-sectional effect which any flat bps
//!   assumption erases: at `$4`, commission alone is `8.75` bps, nearly three times the
//!   entire break-even the trade bench reports.
//! * **Impact** — the square-root law, `k * sigma_daily * sqrt(notional / ADV)`, with `k` a
//!   stated LITERATURE DEFAULT ([`IMPACT_K`]) and never a fitted parameter, swept over
//!   [`IMPACT_K_GRID`] so no conclusion rests on one coefficient.
//!
//! Nothing here is calibrated to the model's returns. Every input is a property of the bar
//! corpus, measured per symbol per calendar month, and the whole table is reported by
//! liquidity decile so it is visible which part of a 5,297-symbol universe is tradeable at
//! all rather than merely present.
//!
//! # Negative spread estimates are reported, not clamped
//!
//! Both estimators are moment estimators of a small quantity buried in a much larger
//! volatility, and at 5-minute sampling a large minority of two-bar windows implies a
//! NEGATIVE spread. That is not a bug to be hidden behind `max(0, .)`: clamping each window
//! before averaging converts symmetric estimation noise into an upward bias, which is
//! precisely how a spread estimate silently becomes a cost assumption again.
//!
//! So the primary figure pools the estimator's own statistic across the month and transforms
//! ONCE ([`LiquidityBucket::cs_spread_bps`]), letting negative windows cancel positive ones
//! as they should; the clamp-then-average recipe Corwin & Schultz actually recommend is
//! reported BESIDE it ([`LiquidityBucket::cs_spread_bps_clamped`]) so the bias is visible as
//! a gap rather than baked in; and the share of negative windows is a reported number
//! ([`LiquidityBucket::cs_negative_share`]). A symbol whose POOLED estimate is still
//! negative is not clamped to zero — that would price it as free, the most dangerous number
//! available. It is marked unmeasured, priced at the cross-sectional median, and counted in
//! [`CostCalibration::unmeasured`].
//!
//! # The correlation diagnostic, which is the actual finding
//!
//! Portfolio Kelly is `f* = Sigma^-1 mu`. The predictor emits PER-SYMBOL MARGINALS and no
//! cross-sectional joint at all, so any book built by stacking per-name Kelly bets is
//! implicitly asserting `Sigma` is diagonal. Equities are not: they load on a common factor
//! intraday, and summing independent bets therefore understates portfolio variance by a
//! factor that grows with breadth.
//!
//! [`cross_correlation`] measures it rather than asserting it. The exact identity it is
//! built to expose, for equal per-name variance `sigma^2` and uniform pairwise correlation
//! `rho`, is
//!
//! ```text
//! realized portfolio variance     (1 - rho) * sum_i w_i^2  +  rho * (sum_i w_i)^2
//! ---------------------------- =  -----------------------------------------------
//! independence-implied variance                 sum_i w_i^2
//! ```
//!
//! which is `1 + (N-1) * rho` for an equal-weight long book of `N` names and `1 - rho` for a
//! dollar-neutral one. The square root of that ratio is the multiple by which per-name Kelly
//! over-levers, and it is reported for a long-only, a signed and a dollar-neutral book with
//! the realized first-factor loading of each ([`BookVolRatio::factor_exposure`]) beside it —
//! because dollar neutrality only collapses the factor term when the loadings are
//! homogeneous, and if it fails to collapse, THAT is the finding.
//!
//! The ratio is decomposed rather than quoted whole, because a single
//! predicted-over-realized number conflates two different errors:
//!
//! * [`BookVolRatio::marginal_factor`] — the model's own per-name variance forecast against
//!   the realized per-name variance. A property of the head's calibration.
//! * [`BookVolRatio::correlation_factor`] — the realized portfolio variance against the
//!   independence-implied variance built from the SAME realized per-name variances. A
//!   property of the cross-section alone, and the part no per-name recalibration can fix.
//!
//! Their product is [`BookVolRatio::total_factor`]. Reporting only the product would let a
//! well-calibrated head with a catastrophic correlation blind spot look like a mildly
//! miscalibrated one.

use anyhow::{ensure, Context, Result};
use chrono::{DateTime, Datelike, Utc};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use shared::bars::PackedBar;
use shared::report::{ReportSeries, ScaleKind};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use tch::{Kind, Tensor};

use super::portfolio::{CostModel, PanelForecast, PanelSlice};
use super::pretrain_reports::{point_series, write_chart};

// ---------------------------------------------------------------------------
// Stated constants: every one of these is a citation, not a knob
// ---------------------------------------------------------------------------

/// `3 - 2*sqrt(2)`, the constant of the Corwin-Schultz two-period range identity.
pub const CS_K1: f64 = 3.0 - 2.0 * std::f64::consts::SQRT_2;

/// Coefficient of the square-root impact law, `impact = k * sigma_daily * sqrt(Q/ADV)`.
///
/// **A LITERATURE DEFAULT, NOT A FITTED PARAMETER.** Nothing in this repository estimates it,
/// and it cannot be estimated from bar data at all — impact is a property of one's own order
/// flow, and this corpus contains none. The three anchors that bracket it:
///
/// * Almgren, Thum, Hauptmann & Li (2005), "Direct estimation of equity market impact",
///   *Risk* 18(7), 57-62, fitted on ~700k institutional orders: permanent impact coefficient
///   `0.314` at a size exponent of `0.891`, temporary coefficient `0.142` at a size exponent
///   of `0.600`. The reduced square-root form sits between them at `O(0.3)`.
/// * Tóth, Lemperiere, Deremble, de Lataillade, Kockelkoren & Bouchaud (2011), "Anomalous
///   price impact and the critical nature of liquidity in financial markets", *Phys. Rev. X*
///   1:021006: the square-root law is universal with a prefactor of order `0.5` to `1` for
///   metaorders.
/// * Grinold & Kahn, *Active Portfolio Management* (2000): the practitioner rule that one
///   day's volume costs about one daily volatility, i.e. `k = 1`.
///
/// `0.5` is the centre of that range and is what the headline uses. Because the choice is
/// unfittable rather than merely uncertain, the capacity curve is reported at every point of
/// [`IMPACT_K_GRID`] and the AUM at which net return crosses zero is quoted at each — so the
/// sensitivity to this constant is a reported axis rather than a hidden assumption.
pub const IMPACT_K: f64 = 0.5;

/// The impact coefficients the capacity curve is re-derived at. Spans the literature.
pub const IMPACT_K_GRID: [f64; 3] = [0.25, 0.5, 1.0];

/// Slot of [`IMPACT_K`] in [`IMPACT_K_GRID`], so the headline curve is one of the charted ones.
pub const IMPACT_K_DEFAULT_SLOT: usize = 1;
const _: () = assert!(IMPACT_K_GRID[IMPACT_K_DEFAULT_SLOT] == IMPACT_K);

/// Commission per share, in dollars: IBKR Pro tiered US equities, `$0.0035`/share.
///
/// Charged per SHARE, which is why it enters as `1e4 * rate / price` bps and why a low-priced
/// name is structurally expensive. Over the 256 traded names it is `2.514` bps equal-weighted
/// against `0.528` bps weighted by dollar ADV, so the units choice is worth a factor of `4.8`
/// and evaluating `rate / price` once at the book's mean price would understate the
/// equal-weighted truth by `4.19x`. That is why [`LiquidityBucket::harmonic_price`] is a
/// harmonic mean: `1/price` is what has to be averaged, and it is convex.
///
/// # The price basis, which is NOT the price shares traded at
///
/// The corpus is Polygon data pulled with `adjusted=true` (see `crate::data::historical`), so
/// every `close` is SPLIT- AND DIVIDEND-ADJUSTED. A per-share fee divided by an adjusted price is
/// a units mismatch: a forward split makes the historical adjusted price too LOW and overstates
/// the commission by the split factor, a reverse split does the reverse. The corpus carries no
/// split history, so the absolute size of that rescaling is UNMEASURABLE from here. What is
/// measured is its effect on the pooling: the span-pooled figure is `10.620` bps against
/// `10.539` bps priced at each window's own anchor month, a gap of `0.081` bps, and the
/// pooled-over-anchor price ratio of the traded names runs `0.479` to `1.577` across its deciles
/// with 0 of 256 beyond `10x` or `0.1x`. So no traded name is rescaled by a reverse split of the
/// kind this corpus is known to contain elsewhere, and the sub-dollar case does not arise at all:
/// the traded set's cheapest price decile is `$5.19` and none of it is under `$1`.
pub const COMMISSION_PER_SHARE_USD: f64 = 0.0035;

/// IBKR's stated cap on the per-share commission: 1% of trade value. Binds below `$0.35`.
///
/// It binds on 0 of the 256 traded names, so it is a guard rather than an active term here.
/// IBKR's other stated bound, the `$0.35` per-ORDER minimum, is deliberately NOT modelled: it
/// binds only below 100 shares, and a per-order floor cannot be expressed in a per-notional cost
/// model that never sees an order size. Every figure here therefore understates the cost of a
/// book small enough to trade odd lots.
pub const COMMISSION_CAP_FRACTION: f64 = 0.01;

/// Regulatory fees, in bps of notional, charged on every leg as a ROUND-TRIP AVERAGE.
///
/// Both statutory fees are SELL-side only, so the quantity that belongs on a one-way leg of a
/// book whose buys and sells balance is HALF the sell-side rate. Charged that way, a full round
/// trip pays the sell-side rate exactly once, which is what a trader pays. At the FY2025 rates
/// this constant was written against:
///
/// * SEC Section 31, `$27.80` per `$1,000,000` of proceeds on SELLS: `0.278` bps of a sale,
///   `0.139` bps per leg.
/// * FINRA Trading Activity Fee, `$0.000166`/share on covered equity SALES: `0.0237` bps of a
///   sale at a `$70` price, `0.0119` bps per leg.
///
/// `0.139 + 0.0119 = 0.1509`, which is this constant. The earlier derivation of the same number
/// halved the SEC leg and did NOT halve the TAF leg, then rounded the resulting `0.163` down to
/// `0.15`; the two errors cancelled and the constant is right, but the sidedness convention has
/// to be one convention or the audit trail is worthless.
///
/// # Two things this constant is known to get wrong, both MEASURED
///
/// **The TAF is per SHARE and this is flat bps.** That is the same units error the commission
/// term exists to avoid, and it is here only because the TAF is small. Priced per share at each
/// name's own price and halved, the honest figure is `0.1986` bps/leg over the 256 traded names
/// and `0.2530` bps/leg over the universe, against the `0.150` charged: an understatement of
/// `0.049` and `0.103` bps/leg, `0.46%` and `0.84%` of the respective impact-free headline costs.
/// It is left flat because correcting it moves four published constants for less than one percent
/// of a figure whose spread term carries `75%` of the total, and a change with that blast radius
/// should be made when the rates below are refreshed rather than on its own.
///
/// **The rates are stale.** SEC Fee Rate Advisory 2026-2 sets Section 31 to `$20.60` per million
/// from 2026-04-04, and `$0.00` per million on charge dates through 2026-04-03; FINRA raised the
/// TAF on covered equity sales to `$0.000195`/share with a `$9.79`/trade cap on 2026-01-01. At
/// those rates, priced per share and halved, the honest figure is `0.1730` bps/leg matched and
/// `0.2369` universe. The per-order TAF cap is not modelled at all, which makes every figure here
/// an upper bound for a single very large order.
///
/// Every number in this comment is printed by
/// `tests::the_traded_window_set_is_priced_against_the_deciles_it_occupies`.
pub const REGULATORY_BPS: f64 = 0.15;

/// Participation levels the decile cost table is quoted at, as a fraction of ADV.
///
/// `0.0` first, deliberately: it is the impact-free floor, so the table shows what the spread
/// and the fees alone cost before any size argument enters.
pub const PARTICIPATION_GRID: [f64; 4] = [0.0, 0.001, 0.01, 0.05];

/// Slot of the `1%` of ADV column, the participation the headline decile figure is quoted at.
pub const PARTICIPATION_HEADLINE_SLOT: usize = 2;
const _: () = assert!(PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT] == 0.01);

/// Liquidity buckets the universe is reported in. Decile `0` is the THINNEST.
pub const DECILES: usize = 10;

/// AUM levels, in dollars, the capacity curve is evaluated at. Log-spaced over six decades,
/// which is what it takes to bracket the zero crossing of a 5-minute equity book without
/// assuming where it is.
pub const AUM_GRID: [f64; 13] = [
    1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11, 3e11, 1e12,
];

/// GROSS edge of the existing per-window Kelly bench at step 20000 of run `bardist_v2`, in bps
/// per bar, before any transaction cost.
///
/// A MEASUREMENT taken from that bench, imported so capacity can be quoted against a real edge.
/// It is not this module's number and this module does not endorse the framing that produced it:
/// that bench trades 256 independent windows as 256 separate books, so its gross is a per-window
/// average and not a portfolio return. It is used here for exactly one purpose — to answer "at the
/// largest edge anybody has measured on this model, what AUM does the MEASURED cost of trading
/// support?" — and that answer is an upper bound on capacity for the same reason.
pub const TRADE_BENCH_GROSS_BPS: f64 = 11.0170;

/// The bench's NET edge at the same checkpoint under its flat 2.00 bps assumption, for the
/// comparison this module exists to make: net at the assumed cost against net at the measured one.
pub const TRADE_BENCH_NET_BPS: f64 = 4.3202;

/// Gross edges, in bps per bar, the capacity crossing is additionally reported at.
///
/// Spans an order of magnitude around [`TRADE_BENCH_GROSS_BPS`], because the honest capacity
/// statement is a CURVE over the assumed edge: the cost side is measured from the corpus, the edge
/// side belongs to whichever checkpoint is being traded, and pretending one number for the latter
/// is what produced a 24,900x-per-year headline in the first place.
pub const GROSS_EDGE_GRID: [f64; 6] = [2.0, 4.0, 6.0, 11.0170, 20.0, 40.0];

/// Aggregation horizons, in bars, the cross-sectional correlation term structure is measured at.
///
/// `1` is the sampling frequency the model trades. `78` is the bar COUNT of one regular US
/// session, but it is NOT a one-day return and must not be read as one: the panel only carries
/// returns whose two bars are exactly one stride apart, so every overnight and weekend move is
/// absent from the tape entirely, and a 78-bar block therefore spans slightly more than a session
/// while omitting the single most cross-sectionally correlated return equities produce. A genuine
/// one-day figure would be higher - daily US equity cross-correlation is typically 0.3 to 0.5
/// against the 0.2284 measured here at 78 bars.
///
/// The curve is not decoration. The Epps (1979) effect makes measured correlation RISE with the
/// sampling interval, so a correlation read at 5 minutes is a LOWER bound on the co-movement a
/// position held across bars is exposed to. Both effects point the same way, which is why the
/// horizon-1 correlation - and any diversification shortfall computed from it - is a floor rather
/// than an estimate. Blocks are built only from gap-free runs, and the block count travels with
/// every point, because at the long end the estimate is read off few enough blocks to be noise.
pub const CORR_HORIZONS: [usize; 6] = [1, 3, 6, 12, 39, 78];

/// Largest correlation matrix that is formed explicitly.
///
/// The `O(N^2)` objects — the median pairwise correlation, the eigenvalue spectrum, the
/// participation ratio — are computed on a deterministic stride subsample at this width, and
/// the width used is reported. Everything computable in `O(T*N)` (the MEAN pairwise
/// correlation, every volatility ratio, every factor exposure) runs on the FULL panel, so no
/// headline number is subsampled.
pub const MAX_EIGEN_DIM: usize = 1024;

/// Eigenvalue shares reported, largest first.
pub const REPORTED_FACTORS: usize = 10;

/// Minimum symbols a correlation panel needs before its cross-section means anything.
pub const MIN_PANEL_SYMBOLS: usize = 8;

/// Minimum two-bar windows a spread estimate needs before it is treated as a measurement.
pub const MIN_SPREAD_PAIRS: u64 = 64;

/// Report base names, all three registered in [`shared::report::PRETRAIN_REPORT_BASES`] and
/// exempted from the in-run cycle walk with this module's battery test named as their executor.
pub const COST_DECILE_BASE: &str = "pretrain_cost_deciles";
pub const CAPACITY_CURVE_BASE: &str = "pretrain_capacity_curve";
pub const CROSS_CORRELATION_BASE: &str = "pretrain_cross_correlation";

// ---------------------------------------------------------------------------
// The two spread estimators
// ---------------------------------------------------------------------------

/// Corwin-Schultz `alpha` for one pair of ADJACENT bars, or `None` if either bar is unusable.
///
/// Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low
/// Prices", *Journal of Finance* 67(2), 719-760, equations (14)-(18). The identity: a period's
/// observed high is almost surely a buy at the ask and its low a sell at the bid, so the
/// observed range overstates the true range by the spread. Two adjacent periods give two
/// readings of that overstatement (`beta`) and one of the two-period range (`gamma`), and the
/// difference identifies the spread separately from the volatility.
///
/// Returns `alpha`, not the spread, because pooling happens in `alpha` space: the transform to
/// a proportional spread, `S = 2*tanh(alpha/2)`, is nonlinear, and averaging AFTER it while
/// discarding negatives is exactly the clamp-induced upward bias this module refuses to hide.
#[inline]
pub fn corwin_schultz_alpha(a: PackedBar, b: PackedBar) -> Option<f64> {
    if !usable(a) || !usable(b) {
        return None;
    }
    let (h1, l1) = (a.high as f64, a.low as f64);
    let (h2, l2) = (b.high as f64, b.low as f64);
    let r1 = (h1 / l1).ln();
    let r2 = (h2 / l2).ln();
    let beta = r1 * r1 + r2 * r2;
    let two_period = (h1.max(h2) / l1.min(l2)).ln();
    let gamma = two_period * two_period;
    let alpha = ((2.0 * beta).sqrt() - beta.sqrt()) / CS_K1 - (gamma / CS_K1).sqrt();
    alpha.is_finite().then_some(alpha)
}

/// Proportional spread implied by a Corwin-Schultz `alpha`.
///
/// `2*(e^a - 1)/(1 + e^a)` written as `2*tanh(a/2)`, the same function without the overflow.
/// Negative for negative `alpha`, on purpose.
#[inline]
pub fn cs_spread(alpha: f64) -> f64 {
    2.0 * (0.5 * alpha).tanh()
}

/// Abdi-Ranaldo squared-spread moment for one pair of ADJACENT bars.
///
/// Abdi & Ranaldo (2017), "A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and
/// Low Prices", *Review of Financial Studies* 30(12), 4437-4480, equation (8):
/// `E[4 * (c_t - eta_t) * (c_t - eta_{t+1})] = S^2`, where `c` is the log close and
/// `eta = (log high + log low)/2` is the log mid-range, a proxy for the efficient price.
///
/// It is here as an INDEPENDENT check on Corwin-Schultz, not as a second opinion to average
/// in. The two use different moments of the same three prices, so agreement is evidence the
/// estimate is a spread and disagreement is evidence at least one estimator has broken at
/// 5-minute sampling — a distinction no single estimator can make about itself.
#[inline]
pub fn abdi_ranaldo_moment(a: PackedBar, b: PackedBar) -> Option<f64> {
    if !usable(a) || !usable(b) {
        return None;
    }
    let eta_a = 0.5 * ((a.high as f64).ln() + (a.low as f64).ln());
    let eta_b = 0.5 * ((b.high as f64).ln() + (b.low as f64).ln());
    let close = (a.close as f64).ln();
    let moment = 4.0 * (close - eta_a) * (close - eta_b);
    moment.is_finite().then_some(moment)
}

/// A bar whose prices can carry a spread estimate at all.
#[inline]
fn usable(bar: PackedBar) -> bool {
    let (h, l, c) = (bar.high as f64, bar.low as f64, bar.close as f64);
    h.is_finite() && l.is_finite() && c.is_finite() && l > 0.0 && h >= l && c > 0.0
}

// ---------------------------------------------------------------------------
// Per-symbol, per-month liquidity measurement
// ---------------------------------------------------------------------------

/// One calendar month of one symbol's measured trading conditions.
///
/// Monthly rather than span-pooled because that is the frequency Corwin & Schultz estimate at,
/// and because ADV over this corpus moves by an order of magnitude across five years for a
/// large share of the universe: a span-average ADV would price 2021 trades at 2026 liquidity
/// and vice versa. It is also what makes the `ts_ms` argument of [`CostModel::cost_bps`]
/// load-bearing instead of decorative.
#[derive(Clone, Copy, Debug)]
pub struct LiquidityBucket {
    /// `year * 12 + (month - 1)`.
    pub month: i32,
    /// Distinct UTC dates the bucket saw bars on.
    pub sessions: u32,
    pub bars: u64,
    /// Adjacent-bar windows the RANGE estimators could use.
    pub pairs: u64,
    /// PRIMARY spread estimate: `2*tanh(mean(alpha)/2)` in bps. May be NEGATIVE, which means
    /// the estimator failed on this month and is reported as such rather than clamped.
    pub cs_spread_bps: f64,
    /// Corwin & Schultz's own recipe: per-window spread with negatives set to zero, then
    /// averaged. Always non-negative and therefore upward biased; carried beside
    /// [`Self::cs_spread_bps`] so the size of that bias is visible.
    pub cs_spread_bps_clamped: f64,
    /// Share of adjacent-bar windows whose `alpha` implied a negative spread.
    pub cs_negative_share: f64,
    /// Abdi-Ranaldo cross-check in bps: `sqrt(mean moment)`. NaN when the pooled moment is
    /// negative, the same failure mode reported the same way.
    pub ar_spread_bps: f64,
    pub ar_negative_share: f64,
    /// Roll (1984) effective spread in bps, `2*sqrt(-cov(r_t, r_{t+1}))`, from the pooled
    /// first-order autocovariance of contiguous log close-to-close returns.
    ///
    /// **This is the estimator that survives 5-minute sampling, and it is the one the cost model
    /// uses.** Corwin-Schultz and Abdi-Ranaldo are both DAILY estimators: they read the spread out
    /// of the high-low RANGE, and at 5-minute sampling the range is dominated by volatility, so
    /// `alpha` becomes a difference of two nearly equal quantities and the estimate collapses into
    /// its own cancellation error. Roll reads the spread out of the bid-ask BOUNCE instead, which
    /// is a first-order effect at high frequency and gets stronger, not weaker, as the sampling
    /// interval shrinks. See [`SymbolCost::measure`] for the measured comparison.
    ///
    /// NaN when the pooled autocovariance is POSITIVE, which is a real failure mode — genuine
    /// return momentum at the bar scale swamps the bounce — reported rather than clamped.
    pub roll_spread_bps: f64,
    /// Share of return pairs whose product was negative, i.e. individually bounce-like. A pooled
    /// estimate is what the spread is read from; this says how consistently the sign held.
    pub roll_negative_share: f64,
    /// Adjacent RETURN pairs the Roll estimator could use. Distinct from [`Self::pairs`] and it
    /// is what gates the primary: a symbol whose high/low are unusable but whose closes are clean
    /// has `pairs == 0` and a perfectly good Roll estimate, and gating the primary on a range
    /// estimator's sample size would discard it.
    pub roll_pairs: u64,
    /// Realized volatility of the log close-to-close return, scaled to one session by the
    /// bucket's OWN measured bars per session rather than by an assumed constant.
    pub sigma_daily: f64,
    /// MEDIAN dollar volume per session, `sum(volume * vwap)` per UTC date. Median, not mean:
    /// one earnings day can be twenty times a normal session, and a mean ADV would price every
    /// other day off it.
    pub adv_usd: f64,
    /// Harmonic mean price, `1 / mean(1/close)`. The exact average to use here: commission
    /// enters as `rate / price`, so the quantity that must be averaged is `1/price`, and its
    /// reciprocal is the harmonic mean by definition.
    pub harmonic_price: f64,
}

impl LiquidityBucket {
    /// A bucket that measured nothing, used as the pooled value of an empty symbol.
    fn empty(month: i32) -> Self {
        Self {
            month,
            sessions: 0,
            bars: 0,
            pairs: 0,
            cs_spread_bps: f64::NAN,
            cs_spread_bps_clamped: f64::NAN,
            cs_negative_share: f64::NAN,
            ar_spread_bps: f64::NAN,
            ar_negative_share: f64::NAN,
            roll_spread_bps: f64::NAN,
            roll_negative_share: f64::NAN,
            roll_pairs: 0,
            sigma_daily: f64::NAN,
            adv_usd: f64::NAN,
            harmonic_price: f64::NAN,
        }
    }

    /// The primary spread if it is a spread, else `None`.
    ///
    /// [`Self::roll_spread_bps`] is the primary because it is the only one of the three that
    /// SURVIVES this corpus: measured over all 5,297 symbols at 5-minute sampling, pooled
    /// Corwin-Schultz is NEGATIVE in every one of the ten liquidity deciles (median `-15.4` bps in
    /// the thinnest, `-1.5` bps in the deepest, with 43-49% of windows negative), because a
    /// range-based estimator at this frequency is a difference of two nearly equal quantities.
    /// Both range estimators are retained and reported so that failure stays visible instead of
    /// being quietly replaced.
    ///
    /// A non-positive estimate is a measurement failure, and a failure priced at zero is the most
    /// dangerous number in a cost model, so it never reaches a caller as one.
    #[inline]
    pub fn measured_spread_bps(&self) -> Option<f64> {
        (self.roll_spread_bps.is_finite()
            && self.roll_spread_bps > 0.0
            && self.roll_pairs >= MIN_SPREAD_PAIRS)
            .then_some(self.roll_spread_bps)
    }

    /// What Corwin-Schultz would have priced this bucket at, for the comparison that justifies
    /// not using it.
    #[inline]
    pub fn cs_measured_spread_bps(&self) -> Option<f64> {
        (self.cs_spread_bps.is_finite()
            && self.cs_spread_bps > 0.0
            && self.pairs >= MIN_SPREAD_PAIRS)
            .then_some(self.cs_spread_bps)
    }
}

/// Streaming accumulator for one calendar month. Retains nothing per bar beyond the
/// per-session dollar volumes, of which a month holds about twenty.
#[derive(Clone, Debug)]
struct BucketAccum {
    month: i32,
    bars: u64,
    inv_price_sum: f64,
    inv_price_n: u64,
    alpha_sum: f64,
    alpha_n: u64,
    alpha_negative: u64,
    clamped_sum: f64,
    ar_sum: f64,
    ar_n: u64,
    ar_negative: u64,
    roll_cross_sum: f64,
    roll_cross_n: u64,
    roll_cross_negative: u64,
    last_ret: Option<f64>,
    ret_sum: f64,
    ret_sq_sum: f64,
    ret_n: u64,
    session_volume: Vec<f64>,
    last_day: i64,
}

impl BucketAccum {
    fn new(month: i32) -> Self {
        Self {
            month,
            bars: 0,
            inv_price_sum: 0.0,
            inv_price_n: 0,
            alpha_sum: 0.0,
            alpha_n: 0,
            alpha_negative: 0,
            clamped_sum: 0.0,
            ar_sum: 0.0,
            ar_n: 0,
            ar_negative: 0,
            roll_cross_sum: 0.0,
            roll_cross_n: 0,
            roll_cross_negative: 0,
            last_ret: None,
            ret_sum: 0.0,
            ret_sq_sum: 0.0,
            ret_n: 0,
            session_volume: Vec::new(),
            last_day: i64::MIN,
        }
    }

    fn push_bar(&mut self, bar: PackedBar, day: i64) {
        self.bars += 1;
        let close = bar.close as f64;
        if close.is_finite() && close > 0.0 {
            self.inv_price_sum += 1.0 / close;
            self.inv_price_n += 1;
        }
        if day != self.last_day {
            self.session_volume.push(0.0);
            self.last_day = day;
        }
        let vwap = bar.vwap as f64;
        let price = if vwap.is_finite() && vwap > 0.0 {
            vwap
        } else {
            close
        };
        let volume = bar.volume as f64;
        if volume.is_finite() && volume > 0.0 && price.is_finite() && price > 0.0 {
            if let Some(last) = self.session_volume.last_mut() {
                *last += volume * price;
            }
        }
    }

    fn push_pair(&mut self, a: PackedBar, b: PackedBar) {
        if let Some(alpha) = corwin_schultz_alpha(a, b) {
            self.alpha_sum += alpha;
            self.alpha_n += 1;
            let spread = cs_spread(alpha);
            if spread < 0.0 {
                self.alpha_negative += 1;
            } else {
                self.clamped_sum += spread;
            }
        }
        if let Some(moment) = abdi_ranaldo_moment(a, b) {
            self.ar_sum += moment;
            self.ar_n += 1;
            if moment < 0.0 {
                self.ar_negative += 1;
            }
        }
        let (pa, pb) = (a.close as f64, b.close as f64);
        if pa > 0.0 && pb > 0.0 {
            let ret = (pb / pa).ln();
            if ret.is_finite() {
                self.ret_sum += ret;
                self.ret_sq_sum += ret * ret;
                self.ret_n += 1;
                // Roll's moment needs two ADJACENT returns, so it consumes the previous one and
                // this one. `last_ret` is cleared by `break_chain` at every session gap: a product
                // straddling an overnight would pair the last return of one day with the first of
                // the next, which is not a bounce and would corrupt the pooled covariance.
                if let Some(previous) = self.last_ret {
                    let product = previous * ret;
                    self.roll_cross_sum += product;
                    self.roll_cross_n += 1;
                    if product < 0.0 {
                        self.roll_cross_negative += 1;
                    }
                }
                self.last_ret = Some(ret);
            } else {
                self.last_ret = None;
            }
        } else {
            self.last_ret = None;
        }
    }

    /// Break the return chain, so no Roll product spans a gap in the tape.
    fn break_chain(&mut self) {
        self.last_ret = None;
    }

    fn absorb(&mut self, other: &BucketAccum) {
        self.bars += other.bars;
        self.inv_price_sum += other.inv_price_sum;
        self.inv_price_n += other.inv_price_n;
        self.alpha_sum += other.alpha_sum;
        self.alpha_n += other.alpha_n;
        self.alpha_negative += other.alpha_negative;
        self.clamped_sum += other.clamped_sum;
        self.ar_sum += other.ar_sum;
        self.ar_n += other.ar_n;
        self.ar_negative += other.ar_negative;
        self.ret_sum += other.ret_sum;
        self.ret_sq_sum += other.ret_sq_sum;
        self.ret_n += other.ret_n;
        self.roll_cross_sum += other.roll_cross_sum;
        self.roll_cross_n += other.roll_cross_n;
        self.roll_cross_negative += other.roll_cross_negative;
        self.session_volume.extend_from_slice(&other.session_volume);
    }

    fn finish(&self) -> LiquidityBucket {
        let sessions = self.session_volume.len() as u32;
        let mut volumes = self.session_volume.clone();
        let adv_usd = median(&mut volumes);
        let sigma_daily = if self.ret_n >= 2 && sessions > 0 {
            let n = self.ret_n as f64;
            let mean = self.ret_sum / n;
            let variance = ((self.ret_sq_sum / n) - mean * mean).max(0.0) * n / (n - 1.0);
            let per_session = self.bars as f64 / sessions as f64;
            variance.sqrt() * per_session.sqrt()
        } else {
            f64::NAN
        };
        let (cs_spread_bps, cs_clamped, cs_negative) = if self.alpha_n > 0 {
            let n = self.alpha_n as f64;
            (
                1.0e4 * cs_spread(self.alpha_sum / n),
                1.0e4 * self.clamped_sum / n,
                self.alpha_negative as f64 / n,
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        let (ar_spread_bps, ar_negative) = if self.ar_n > 0 {
            let n = self.ar_n as f64;
            let pooled = self.ar_sum / n;
            (
                if pooled >= 0.0 {
                    1.0e4 * pooled.sqrt()
                } else {
                    f64::NAN
                },
                self.ar_negative as f64 / n,
            )
        } else {
            (f64::NAN, f64::NAN)
        };
        let (roll_spread_bps, roll_negative_share) = if self.roll_cross_n > 0 && self.ret_n > 0 {
            let pairs = self.roll_cross_n as f64;
            // `cov(r_t, r_{t+1})`, mean-corrected. The correction is tiny at 5-minute sampling
            // (drift squared is ~1e-12 against a covariance of ~1e-8) but it is free and it keeps
            // the estimator the covariance it claims to be rather than a raw cross moment.
            let mean = self.ret_sum / self.ret_n as f64;
            let covariance = self.roll_cross_sum / pairs - mean * mean;
            (
                if covariance < 0.0 {
                    1.0e4 * 2.0 * (-covariance).sqrt()
                } else {
                    f64::NAN
                },
                self.roll_cross_negative as f64 / pairs,
            )
        } else {
            (f64::NAN, f64::NAN)
        };
        LiquidityBucket {
            month: self.month,
            sessions,
            bars: self.bars,
            pairs: self.alpha_n,
            cs_spread_bps,
            cs_spread_bps_clamped: cs_clamped,
            cs_negative_share: cs_negative,
            ar_spread_bps,
            ar_negative_share: ar_negative,
            roll_spread_bps,
            roll_negative_share,
            roll_pairs: self.roll_cross_n,
            sigma_daily,
            adv_usd,
            harmonic_price: if self.inv_price_sum > 0.0 {
                self.inv_price_n as f64 / self.inv_price_sum
            } else {
                f64::NAN
            },
        }
    }
}

/// Every measured month of one symbol, plus the span-pooled figures.
#[derive(Clone, Debug)]
pub struct SymbolCost {
    pub symbol: String,
    /// Ascending in `month`, so a lookup is a binary search.
    pub buckets: Vec<LiquidityBucket>,
    /// The same statistics accumulated over the WHOLE span, not a re-average of the monthly
    /// figures: pooling `alpha` across five years is a different and better-conditioned
    /// estimator than averaging sixty noisy monthly transforms.
    pub pooled: LiquidityBucket,
}

impl SymbolCost {
    /// Measure one symbol from its bars. Pure function of the corpus, which is what makes it
    /// testable on synthetic bars carrying a KNOWN injected spread.
    ///
    /// `res_secs` gates which adjacent pairs are usable: a pair straddling an overnight or
    /// weekend gap violates every one of the three estimators' contiguity assumption and is
    /// skipped, not adjusted. Corwin & Schultz devote a section to overnight adjustments precisely
    /// because getting this wrong inflates `gamma` and destroys the estimate; Roll's product is
    /// broken across the same gaps by [`BucketAccum::break_chain`], since an overnight return
    /// paired with the next morning's is not a bid-ask bounce.
    ///
    /// # Which estimator survives 5-minute sampling
    ///
    /// Measured over the whole 5,297-symbol corpus, by liquidity decile, thinnest to deepest:
    ///
    /// ```text
    /// median ADV      $1.1M   $2.0M   $3.3M   $5.2M   $8.6M  $14.2M  $24.3M  $44.3M  $94.7M  $301.4M
    /// Corwin-Schultz -15.39  -11.79  -10.46   -8.95   -7.01   -5.89   -4.76   -3.79   -2.51    -1.53
    /// Abdi-Ranaldo    25.35   15.47   14.03   12.09   11.87   10.44    9.37    8.18    6.61     5.90
    /// ```
    ///
    /// Corwin-Schultz is NEGATIVE in all ten deciles, with 43-49% of windows negative. That is not
    /// noise to be averaged away, it is the estimator being out of its regime: `alpha` is
    /// `(sqrt(2b) - sqrt(b))/k1 - sqrt(g/k1)`, and at 5-minute sampling the two terms agree to
    /// within a fraction of a percent, so the estimate IS the cancellation error. Both range
    /// estimators are kept and reported anyway, because a cost model whose primary estimator
    /// silently replaced a failing one would hide exactly this.
    pub fn measure(symbol: &str, bars: &[PackedBar], res_secs: u32) -> Self {
        let stride = res_secs as i64 * 1000;
        let mut buckets: Vec<BucketAccum> = Vec::new();
        for (index, bar) in bars.iter().enumerate() {
            let bar = *bar;
            let ts = bar.ts_ms;
            let Some(month) = month_index(ts) else {
                continue;
            };
            if buckets.last().map(|b| b.month) != Some(month) {
                buckets.push(BucketAccum::new(month));
            }
            let accum = buckets.last_mut().expect("just pushed");
            accum.push_bar(bar, ts.div_euclid(86_400_000));
            match bars.get(index + 1) {
                Some(next) if next.ts_ms - ts == stride => accum.push_pair(bar, *next),
                _ => accum.break_chain(),
            }
        }
        let mut pooled = BucketAccum::new(buckets.first().map_or(0, |b| b.month));
        for bucket in &buckets {
            pooled.absorb(bucket);
        }
        Self {
            symbol: symbol.to_owned(),
            buckets: buckets.iter().map(BucketAccum::finish).collect(),
            pooled: if buckets.is_empty() {
                LiquidityBucket::empty(0)
            } else {
                pooled.finish()
            },
        }
    }

    /// The bucket covering `ts_ms`, else the NEAREST measured month.
    ///
    /// Nearest rather than pooled: a query one month past the corpus end should be priced at
    /// the last liquidity actually observed, not at a five-year average that includes 2021.
    pub fn bucket_at(&self, ts_ms: i64) -> Option<&LiquidityBucket> {
        if self.buckets.is_empty() {
            return None;
        }
        let month = month_index(ts_ms)?;
        match self.buckets.binary_search_by_key(&month, |b| b.month) {
            Ok(hit) => Some(&self.buckets[hit]),
            Err(insert) => {
                let before = insert.checked_sub(1).map(|i| &self.buckets[i]);
                let after = self.buckets.get(insert);
                match (before, after) {
                    (Some(b), Some(a)) => Some(if month - b.month <= a.month - month { b } else { a }),
                    (Some(b), None) => Some(b),
                    (None, Some(a)) => Some(a),
                    (None, None) => None,
                }
            }
        }
    }
}

/// `year * 12 + (month - 1)` of an epoch-millis instant.
#[inline]
fn month_index(ts_ms: i64) -> Option<i32> {
    let stamp = DateTime::<Utc>::from_timestamp_millis(ts_ms)?;
    Some(stamp.year() * 12 + stamp.month0() as i32)
}

// ---------------------------------------------------------------------------
// The calibration and the cost model
// ---------------------------------------------------------------------------

/// The whole universe's measured trading conditions, keyed by the panel's `u32` symbol id —
/// the index into the run's symbol table, which is what the panel contract carries.
#[derive(Debug)]
pub struct CostCalibration {
    pub res_secs: u32,
    pub symbols: Vec<SymbolCost>,
    /// Cross-sectional median of the measurable pooled spreads, in bps. Neutral by
    /// construction, which is why it and not zero is what an unmeasured symbol is priced at.
    pub fallback_spread_bps: f64,
    pub fallback_sigma_daily: f64,
    pub fallback_harmonic_price: f64,
    pub fallback_adv_usd: f64,
    /// Symbol ids whose SPAN-POOLED spread could not be measured and are therefore priced at
    /// the cross-sectional median. Reported, never silently clamped.
    pub unmeasured: Vec<u32>,
}

impl CostCalibration {
    /// Measure a set of `(symbol, bars)` series. The seam every test uses, and what
    /// [`Self::from_corpus`] reduces to.
    pub fn from_series(series: &[(String, &[PackedBar])], res_secs: u32) -> Result<Self> {
        ensure!(!series.is_empty(), "a cost calibration needs a universe");
        let symbols: Vec<SymbolCost> = series
            .iter()
            .map(|(symbol, bars)| SymbolCost::measure(symbol, bars, res_secs))
            .collect();
        Ok(Self::from_measured(symbols, res_secs))
    }

    /// Measure the real packed corpus, in parallel over symbols.
    ///
    /// `threads` is bounded by the caller rather than left to rayon's default: this pass
    /// touches every one of the corpus's ~451M bars, and it runs on a box that is also
    /// training. Each worker holds one symbol's accumulators and nothing else, so peak memory
    /// is `threads * O(months)`, not `O(corpus)`.
    pub fn from_corpus(corpus: &crate::torch::dataset::BarCorpus, threads: usize) -> Result<Self> {
        use rayon::prelude::*;

        let count = corpus.series_count();
        ensure!(count > 0, "the corpus holds no series to calibrate against");
        let res_secs = corpus.res_secs();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build()
            .context("building the calibration thread pool")?;
        let measured = pool.install(|| {
            (0..count)
                .into_par_iter()
                .map(|series| {
                    SymbolCost::measure(corpus.symbol(series), corpus.bars(series), res_secs)
                })
                .collect::<Vec<_>>()
        });
        Ok(Self::from_measured(measured, res_secs))
    }

    fn from_measured(symbols: Vec<SymbolCost>, res_secs: u32) -> Self {
        let mut spreads: Vec<f64> = symbols
            .iter()
            .filter_map(|s| s.pooled.measured_spread_bps())
            .collect();
        let mut sigmas: Vec<f64> = symbols.iter().map(|s| s.pooled.sigma_daily).collect();
        let mut prices: Vec<f64> = symbols.iter().map(|s| s.pooled.harmonic_price).collect();
        let mut advs: Vec<f64> = symbols.iter().map(|s| s.pooled.adv_usd).collect();
        let unmeasured = symbols
            .iter()
            .enumerate()
            .filter(|(_, s)| s.pooled.measured_spread_bps().is_none())
            .map(|(index, _)| index as u32)
            .collect();
        Self {
            res_secs,
            fallback_spread_bps: median(&mut spreads),
            fallback_sigma_daily: median(&mut sigmas),
            fallback_harmonic_price: median(&mut prices),
            fallback_adv_usd: median(&mut advs),
            unmeasured,
            symbols,
        }
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Span-pooled dollar ADV per symbol id, for the liquidity ranking.
    pub fn pooled_adv_usd(&self, symbol: u32) -> f64 {
        self.symbols
            .get(symbol as usize)
            .map_or(f64::NAN, |s| s.pooled.adv_usd)
    }
}

/// Everything the cost of one `(symbol, month)` is made of, resolved once.
///
/// Split out because the size-INDEPENDENT parts dominate the work — a bucket lookup is a
/// binary search — while the size-dependent part is one multiply and one square root. A
/// capacity sweep over thirteen AUM levels resolves once and re-evaluates thirteen times, and
/// [`BarCostModel::cost_bps`] is defined in terms of the same two functions, so there is one
/// code path rather than a fast one and a correct one.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedCost {
    /// Half the estimated proportional spread: the cost of crossing it once.
    pub half_spread_bps: f64,
    /// `1e4 * min(commission_per_share / price, cap)`.
    pub commission_bps: f64,
    pub regulatory_bps: f64,
    /// `1e4 * k * sigma_daily`. Multiply by `sqrt(notional/ADV)` for impact in bps.
    pub impact_coefficient_bps: f64,
    /// Dollar ADV of the resolved month. NaN when the symbol has no measurable volume at all,
    /// which callers must treat as "unpriceable", never as "infinite capacity".
    pub adv_usd: f64,
    /// True when this symbol's own spread was unmeasurable and the cross-sectional median
    /// stood in for it.
    pub spread_fallback: bool,
}

impl ResolvedCost {
    /// Cost that does not depend on how much is traded.
    #[inline]
    pub fn fixed_bps(&self) -> f64 {
        self.half_spread_bps + self.commission_bps + self.regulatory_bps
    }

    /// Square-root impact of trading `notional_frac` of ADV.
    ///
    /// Trading NOTHING costs nothing, even in a symbol whose impact coefficient could not be
    /// measured: `0.0` short-circuits before the coefficient is touched, so the impact-free column
    /// of the decile table stays the fixed floor it is defined to be instead of going NaN for every
    /// symbol-month with no measurable volatility. At any positive size an unmeasurable coefficient
    /// propagates as NaN, which is the whole point - a cost that cannot be measured must not read
    /// as zero.
    #[inline]
    pub fn impact_bps(&self, notional_frac: f64) -> f64 {
        if notional_frac <= 0.0 {
            return 0.0;
        }
        self.impact_coefficient_bps * notional_frac.sqrt()
    }

    #[inline]
    pub fn total_bps(&self, notional_frac: f64) -> f64 {
        self.fixed_bps() + self.impact_bps(notional_frac)
    }
}

/// The data-driven [`CostModel`].
///
/// Holds the calibration behind an [`Arc`] so [`Self::with_impact_k`] is a pointer copy: the
/// capacity curve is reported at three impact coefficients, and re-measuring a 5,297-symbol
/// universe three times to change one scalar would be absurd.
#[derive(Clone, Debug)]
pub struct BarCostModel {
    calibration: Arc<CostCalibration>,
    impact_k: f64,
}

impl BarCostModel {
    pub fn new(calibration: Arc<CostCalibration>) -> Self {
        Self {
            calibration,
            impact_k: IMPACT_K,
        }
    }

    pub fn with_impact_k(&self, impact_k: f64) -> Self {
        Self {
            calibration: Arc::clone(&self.calibration),
            impact_k,
        }
    }

    pub fn impact_k(&self) -> f64 {
        self.impact_k
    }

    pub fn calibration(&self) -> &CostCalibration {
        &self.calibration
    }

    /// Whether this symbol's spread was measured from its own bars.
    pub fn is_measured(&self, symbol: u32) -> bool {
        self.calibration
            .symbols
            .get(symbol as usize)
            .and_then(|s| s.pooled.measured_spread_bps())
            .is_some()
    }

    /// Resolve the `(symbol, month)` cost inputs.
    ///
    /// Three tiers, in order, each strictly better than the next: the symbol's OWN month; the
    /// symbol's own SPAN when that month was too noisy to estimate; the cross-sectional median
    /// when the symbol's whole span was. Only the third counts as a fallback in
    /// [`CostCalibration::unmeasured`], because falling back from one noisy month to the same
    /// symbol's five years is still that symbol's own liquidity.
    pub fn resolve(&self, symbol: u32, ts_ms: i64) -> ResolvedCost {
        let entry = self.calibration.symbols.get(symbol as usize);
        self.resolve_from(entry.and_then(|s| s.bucket_at(ts_ms)), entry.map(|s| &s.pooled))
    }

    /// The same resolution against the symbol's SPAN-POOLED liquidity, which is what the
    /// decile table is a statement about: a per-month table would be sixty tables.
    pub fn resolve_pooled(&self, symbol: u32) -> ResolvedCost {
        let pooled = self
            .calibration
            .symbols
            .get(symbol as usize)
            .map(|s| &s.pooled);
        self.resolve_from(pooled, pooled)
    }

    fn resolve_from(
        &self,
        bucket: Option<&LiquidityBucket>,
        pooled: Option<&LiquidityBucket>,
    ) -> ResolvedCost {
        let (spread_bps, spread_fallback) =
            match bucket.and_then(LiquidityBucket::measured_spread_bps) {
                Some(spread) => (spread, false),
                None => match pooled.and_then(LiquidityBucket::measured_spread_bps) {
                    Some(spread) => (spread, false),
                    None => (self.calibration.fallback_spread_bps, true),
                },
            };
        // POSITIVE, not merely finite. A symbol-month whose close never moved has a perfectly
        // finite `sigma_daily == 0.0`, and accepting it short-circuits both fallbacks and sets
        // `impact_coefficient_bps` to exactly zero - free impact at any size, for the least liquid
        // names in the universe, silently inflating every capacity number. Zero volatility is a
        // measurement failure and is propagated as NaN so the leg is COUNTED as unpriceable.
        let sigma_daily = first_finite_positive([
            bucket.map(|b| b.sigma_daily),
            pooled.map(|p| p.sigma_daily),
            Some(self.calibration.fallback_sigma_daily),
        ])
        .unwrap_or(f64::NAN);
        // NAN, not INFINITY, and the difference is a real defect this sentinel used to have. An
        // unmeasurable price with `INFINITY` makes `COMMISSION_PER_SHARE_USD / price` exactly `0.0`
        // and the `min` against the cap keeps it there, so a symbol whose price could not be
        // measured reported FREE commission - an absent measurement rendering as the most
        // favourable one available. A bool or a clamp cannot express three states, so the absent
        // value has to be the one that propagates.
        let price = first_finite_positive([
            bucket.map(|b| b.harmonic_price),
            pooled.map(|p| p.harmonic_price),
            Some(self.calibration.fallback_harmonic_price),
        ])
        .unwrap_or(f64::NAN);
        let adv_usd = first_finite_positive([
            bucket.map(|b| b.adv_usd),
            pooled.map(|p| p.adv_usd),
            Some(self.calibration.fallback_adv_usd),
        ])
        .unwrap_or(f64::NAN);
        ResolvedCost {
            half_spread_bps: 0.5 * spread_bps,
            // Guarded, because `f64::min` IGNORES NaN and returns the other operand: an unguarded
            // `(x / NAN).min(COMMISSION_CAP_FRACTION)` is the CAP, so the absent measurement would
            // still render as a number - merely a conservative one instead of a free one. The clamp
            // has to be unreachable when its input is absent, not merely pointed the other way.
            commission_bps: if price.is_finite() {
                1.0e4 * (COMMISSION_PER_SHARE_USD / price).min(COMMISSION_CAP_FRACTION)
            } else {
                f64::NAN
            },
            regulatory_bps: REGULATORY_BPS,
            impact_coefficient_bps: 1.0e4 * self.impact_k * sigma_daily,
            adv_usd,
            spread_fallback,
        }
    }

    /// Symbols ordered from THINNEST to deepest by span-pooled dollar ADV, with the id as a
    /// deterministic tiebreak.
    ///
    /// An UNMEASURABLE ADV must sort to the bottom decile, which is where an untradeable name
    /// belongs. `total_cmp` alone would put it in the TOP one: IEEE total order ranks a positive
    /// NaN above `+inf`, so the thinnest names in the universe would be reported as its most liquid
    /// tenth. Non-finite maps to `-inf` for the SORT KEY only - every median still reads the real
    /// value, so an unmeasurable ADV stays unmeasurable in the table.
    fn liquidity_ranking(&self) -> Vec<(f64, u32)> {
        let mut ranked: Vec<(f64, u32)> = (0..self.calibration.len() as u32)
            .map(|symbol| (self.calibration.pooled_adv_usd(symbol), symbol))
            .collect();
        let rank_key = |adv: f64| if adv.is_finite() { adv } else { f64::NEG_INFINITY };
        ranked.sort_by(|a, b| rank_key(a.0).total_cmp(&rank_key(b.0)).then(a.1.cmp(&b.1)));
        ranked
    }

    /// Which liquidity decile each symbol id falls in, indexed by symbol id.
    ///
    /// Shares [`Self::liquidity_ranking`] and the same `lo..hi` slicing with [`Self::deciles`], so
    /// the membership and the table can never disagree about which tenth a name belongs to. Exists
    /// so a TRADED SUBSET can be priced against the deciles it actually occupies: a break-even
    /// measured on 256 windows compared against a universe-wide median is a category error, and
    /// closing it needs the decile of each traded name rather than another median.
    pub fn decile_of_symbol(&self) -> Vec<usize> {
        let ranked = self.liquidity_ranking();
        let count = ranked.len();
        let mut out = vec![0usize; count];
        for decile in 0..DECILES {
            let lo = decile * count / DECILES;
            let hi = (decile + 1) * count / DECILES;
            for &(_, symbol) in &ranked[lo..hi] {
                out[symbol as usize] = decile;
            }
        }
        out
    }

    /// Median all-in one-way cost per liquidity decile, at every [`PARTICIPATION_GRID`] level.
    ///
    /// This table, not the headline, answers "which part of the universe is tradeable at all".
    /// A decile whose median all-in cost at a realistic participation exceeds the bench's
    /// `3.29` bps break-even cannot host the strategy at any size.
    pub fn deciles(&self) -> Vec<CostDecile> {
        let ranked = self.liquidity_ranking();
        let count = ranked.len();
        (0..DECILES)
            .map(|decile| {
                let lo = decile * count / DECILES;
                let hi = (decile + 1) * count / DECILES;
                let members = &ranked[lo..hi];
                let mut advs = Vec::with_capacity(members.len());
                let mut prices = Vec::with_capacity(members.len());
                let mut cs = Vec::with_capacity(members.len());
                let mut cs_clamped = Vec::with_capacity(members.len());
                let mut ar = Vec::with_capacity(members.len());
                let mut roll = Vec::with_capacity(members.len());
                let mut roll_negatives = Vec::with_capacity(members.len());
                let mut cs_unmeasured = 0usize;
                let mut negatives = Vec::with_capacity(members.len());
                let mut sigmas = Vec::with_capacity(members.len());
                let mut fees = Vec::with_capacity(members.len());
                let mut all_in: Vec<Vec<f64>> = PARTICIPATION_GRID
                    .iter()
                    .map(|_| Vec::with_capacity(members.len()))
                    .collect();
                let mut unmeasured = 0usize;
                let mut impact_unpriceable = 0usize;
                let mut fixed_unmeasurable = 0usize;
                for &(_, symbol) in members {
                    let pooled = &self.calibration.symbols[symbol as usize].pooled;
                    advs.push(pooled.adv_usd);
                    prices.push(pooled.harmonic_price);
                    cs.push(pooled.cs_spread_bps);
                    cs_clamped.push(pooled.cs_spread_bps_clamped);
                    ar.push(pooled.ar_spread_bps);
                    roll.push(pooled.roll_spread_bps);
                    roll_negatives.push(pooled.roll_negative_share);
                    if pooled.cs_measured_spread_bps().is_none() {
                        cs_unmeasured += 1;
                    }
                    negatives.push(pooled.cs_negative_share);
                    sigmas.push(pooled.sigma_daily);
                    let resolved = self.resolve_pooled(symbol);
                    if resolved.spread_fallback {
                        unmeasured += 1;
                    }
                    if !(resolved.impact_coefficient_bps.is_finite()
                        && resolved.impact_coefficient_bps > 0.0)
                    {
                        impact_unpriceable += 1;
                    }
                    // The FIXED floor can now be absent too, since an unmeasurable price makes the
                    // commission NaN rather than free. `median` drops it, so without this counter
                    // the impact-free column would be a survivor median exactly like the sized ones.
                    if !resolved.fixed_bps().is_finite() {
                        fixed_unmeasurable += 1;
                    }
                    fees.push(resolved.commission_bps + resolved.regulatory_bps);
                    for (slot, participation) in PARTICIPATION_GRID.iter().enumerate() {
                        all_in[slot].push(resolved.total_bps(*participation));
                    }
                }
                CostDecile {
                    decile,
                    symbols: members.len(),
                    unmeasured,
                    median_adv_usd: median(&mut advs),
                    median_harmonic_price: median(&mut prices),
                    median_cs_spread_bps: median(&mut cs),
                    median_cs_spread_bps_clamped: median(&mut cs_clamped),
                    median_ar_spread_bps: median(&mut ar),
                    median_roll_spread_bps: median(&mut roll),
                    median_roll_negative_share: median(&mut roll_negatives),
                    cs_unmeasured,
                    median_cs_negative_share: median(&mut negatives),
                    median_sigma_daily: median(&mut sigmas),
                    median_fee_bps: median(&mut fees),
                    impact_unpriceable,
                    fixed_unmeasurable,
                    median_all_in_bps: all_in.iter_mut().map(median).collect(),
                }
            })
            .collect()
    }
}

impl CostModel for BarCostModel {
    fn cost_bps(&self, symbol: u32, ts_ms: i64, notional_frac: f32) -> f32 {
        self.resolve(symbol, ts_ms).total_bps(notional_frac as f64) as f32
    }
}

/// One liquidity decile's measured cost. Decile `0` is the THINNEST tenth of the universe.
#[derive(Clone, Debug)]
pub struct CostDecile {
    pub decile: usize,
    pub symbols: usize,
    /// Members priced at the cross-sectional median because their own spread was unmeasurable.
    pub unmeasured: usize,
    pub median_adv_usd: f64,
    pub median_harmonic_price: f64,
    pub median_cs_spread_bps: f64,
    pub median_cs_spread_bps_clamped: f64,
    pub median_ar_spread_bps: f64,
    /// PRIMARY: the Roll (1984) spread the cost model actually charges.
    pub median_roll_spread_bps: f64,
    pub median_roll_negative_share: f64,
    /// Members whose CORWIN-SCHULTZ estimate was unusable. Reported beside
    /// [`Self::unmeasured`], which counts the primary estimator's failures, so the gap between
    /// the two IS the evidence for the choice of primary.
    pub cs_unmeasured: usize,
    pub median_cs_negative_share: f64,
    pub median_sigma_daily: f64,
    pub median_fee_bps: f64,
    /// Members whose IMPACT could not be priced, because their volatility measured zero.
    ///
    /// Their sized costs are NaN and [`median`] drops non-finite entries, so without this counter
    /// every sized column of the table would be a SURVIVOR median: the thinnest deciles are where
    /// zero-volatility months concentrate, so dropping them moves the median by SELECTION and not
    /// only by re-measurement, and a break-even table quoting the survivor figure would understate
    /// the cost of exactly the names it is trying to rule out. The impact-FREE column is unaffected
    /// - see [`ResolvedCost::impact_bps`].
    pub impact_unpriceable: usize,
    /// Members whose FIXED floor could not be measured, because their price was unmeasurable and
    /// the commission therefore is not a number. Distinct from [`Self::impact_unpriceable`], which
    /// is about size: these members are absent from EVERY column including the impact-free one, so
    /// without this count that column is a survivor median too.
    pub fixed_unmeasurable: usize,
    /// One entry per [`PARTICIPATION_GRID`] level. The `0.0` entry is the fixed floor and is
    /// measured on every member; the sized entries omit the [`Self::impact_unpriceable`] members.
    pub median_all_in_bps: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Books: turning per-name Kelly into a shared-capital allocation
// ---------------------------------------------------------------------------

/// How per-name Kelly fractions become weights of ONE book.
///
/// The bug this exists to make unrepresentable: the trade bench gives each of 256 windows its
/// own book at up to `4x` of its OWN wealth, which is `1024x` gross exposure of nothing in
/// particular. Every style here normalizes to a FIXED gross exposure, so adding breadth
/// diversifies instead of levering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookStyle {
    /// Long legs only, shorts discarded. The most factor-exposed book the signal admits, and
    /// therefore the upper bound on the over-levering factor.
    LongOnly,
    /// Both legs at their signed Kelly sign. The book the sizing actually implies.
    Signed,
    /// Signed, then cross-sectionally demeaned so the book is dollar neutral. Collapses the
    /// common factor only when the loadings are homogeneous, which is exactly what
    /// [`BookVolRatio::factor_exposure`] is measured to check.
    DollarNeutral,
}

impl BookStyle {
    pub fn label(self) -> &'static str {
        match self {
            BookStyle::LongOnly => "long-only",
            BookStyle::Signed => "signed",
            BookStyle::DollarNeutral => "dollar-neutral",
        }
    }
}

/// The three books every diagnostic is reported for.
pub const BOOK_STYLES: [BookStyle; 3] = [
    BookStyle::LongOnly,
    BookStyle::Signed,
    BookStyle::DollarNeutral,
];

/// A book's style, its gross exposure `sum_i |w_i|`, and its no-trade band.
#[derive(Clone, Copy, Debug)]
pub struct BookSpec {
    pub style: BookStyle,
    pub gross_leverage: f64,
    /// Per-symbol NO-TRADE BAND, in the same weight units as `gross_leverage`: a leg whose target
    /// differs from the weight already held by less than this is NOT traded, and the held weight
    /// carries forward.
    ///
    /// This is the one lever the cost arithmetic leaves. Break-even is `gross_edge / turnover`, and
    /// the existing bench turns over `3.346` per BAR — the book rotates completely about every
    /// third bar — because Kelly is re-solved every bar with zero inertia and no cost awareness.
    /// That turnover is not a fact about the market; it is a property of the policy. Cutting it 10x
    /// moves break-even from `3.29` bps to about `33` bps, which clears even the thin deciles. What
    /// it costs is edge, since some of the edge is genuinely high frequency and a band forgoes it.
    /// Where those two curves cross is the whole remaining question, and it is why banding lives
    /// here rather than in a caller: the gross return is earned on the HELD weights, so a band's
    /// drag on the edge is MEASURED by the same pass that measures its saving on cost, and neither
    /// can be quoted without the other.
    ///
    /// `0.0` reproduces the unbanded book exactly.
    pub no_trade_band: f64,
}

impl BookSpec {
    pub fn new(style: BookStyle, gross_leverage: f64) -> Self {
        Self {
            style,
            gross_leverage,
            no_trade_band: 0.0,
        }
    }

    pub fn with_band(self, no_trade_band: f64) -> Self {
        Self {
            no_trade_band,
            ..self
        }
    }
}

/// Weights of one bar's book, `sum |w| == gross_leverage` unless the signal is degenerate.
///
/// Writes into `out` rather than allocating, because the capacity sweep calls this once per bar
/// per book and a 20,000-bar panel does not need 20,000 vectors.
///
/// A degenerate signal — every `f` zero, every `f` non-positive for [`BookStyle::LongOnly`],
/// every finite `f` identical for [`BookStyle::DollarNeutral`] — yields all zeros. That is a real
/// answer, not a failure: it is what "this bar carried no allocatable information" looks like,
/// and every downstream statistic drops a zero-weight bar.
///
/// A NON-FINITE forecast is EXCLUDED, not neutralized to zero. The distinction only shows up in
/// [`BookStyle::DollarNeutral`] and it matters there: substituting `0` would make the name a
/// participant in the cross-sectional demeaning, so a symbol the model said nothing about would
/// come out with a real SHORT against the rest of the book. Excluding it instead means the
/// demeaning is taken over the names that carry a forecast, the book is dollar neutral across
/// exactly those, and a nameless symbol carries no position in any style.
pub fn book_weights(kelly_f: &[f32], spec: BookSpec, out: &mut Vec<f64>) {
    out.clear();
    out.reserve(kelly_f.len());
    let mean = if spec.style == BookStyle::DollarNeutral {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for value in kelly_f.iter().map(|f| *f as f64).filter(|f| f.is_finite()) {
            sum += value;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    } else {
        0.0
    };
    out.extend(kelly_f.iter().map(|f| {
        let value = *f as f64;
        if !value.is_finite() {
            return 0.0;
        }
        match spec.style {
            BookStyle::LongOnly => value.max(0.0),
            BookStyle::Signed => value,
            BookStyle::DollarNeutral => value - mean,
        }
    }));
    let gross: f64 = out.iter().map(|w| w.abs()).sum();
    if gross > 0.0 && gross.is_finite() {
        let scale = spec.gross_leverage / gross;
        for value in out.iter_mut() {
            *value *= scale;
        }
    } else {
        out.iter_mut().for_each(|value| *value = 0.0);
    }
}

// ---------------------------------------------------------------------------
// Capacity: net return against AUM
// ---------------------------------------------------------------------------

/// One AUM level of the capacity curve.
#[derive(Clone, Copy, Debug)]
pub struct CapacityPoint {
    pub aum_usd: f64,
    /// Cost-free arithmetic portfolio return, bps per bar. Independent of AUM, so it is the
    /// same number at every point and is charted as the ceiling the costs eat into.
    pub gross_bps: f64,
    /// Half-spread, commission and regulatory cost on realized turnover. Also independent of
    /// AUM, which is why the capacity curve is a statement about IMPACT alone.
    pub fixed_cost_bps: f64,
    pub impact_cost_bps: f64,
    pub net_bps: f64,
    /// Mean and 99th-percentile participation across traded symbol-bars, as a fraction of that
    /// symbol's dollar ADV. Reported because a net return computed where the book trades 30% of
    /// ADV per bar is arithmetic, not a forecast: the square-root law was never fitted there
    /// and no equity book executes there.
    pub mean_participation: f64,
    pub p99_participation: f64,
}

/// Net return against AUM for one book at one impact coefficient.
#[derive(Clone, Debug)]
pub struct CapacityCurve {
    pub style: BookStyle,
    pub gross_leverage: f64,
    pub impact_k: f64,
    pub points: Vec<CapacityPoint>,
    /// Cost-free arithmetic portfolio return, bps/bar. Constant across the grid.
    pub gross_bps: f64,
    /// Half-spread, commission and regulatory cost on realized turnover, bps/bar. Also constant
    /// across the grid, which is why the capacity curve is a statement about IMPACT alone.
    pub fixed_cost_bps: f64,
    /// Realized turnover, `mean_bar sum_i |dw_i|`, in ABSOLUTE weight units.
    ///
    /// Not a fraction of anything: at gross leverage `4.0` a turnover of `3.35` is 0.84 of the
    /// book replaced per bar, i.e. a full rotation every ~1.2 bars, and reading it as "rotates
    /// every third bar" understates the trading rate by the gross multiple. Divide by
    /// [`Self::gross_exposure_per_bar`] - which is the exposure the book ACTUALLY carried, not the
    /// nominal target - for the rotation rate.
    pub turnover_per_bar: f64,
    /// Realized gross exposure, `mean_bar sum_i |w_i|`, in the same units.
    ///
    /// Measured rather than assumed equal to [`BookSpec::gross_leverage`], because a bar whose
    /// signal was degenerate carries no exposure and a banded book holds yesterday's weights
    /// rather than today's normalized ones.
    pub gross_exposure_per_bar: f64,
    /// Turnover-weighted total, `mean_bar sum_i |dw_i| * (1e4 k sigma_i) * sqrt(|dw_i| / ADV_i)`.
    /// Impact at AUM `A` is exactly `sqrt(A)` times this, which is what makes the zero crossing
    /// closed form rather than bisected.
    pub impact_per_sqrt_aum: f64,
    /// AUM, in dollars, at which net return crosses zero.
    ///
    /// `0.0` when the AUM-independent part is already non-positive — the book does not pay for
    /// its own spread and shrinking does not help. `INFINITY` when the book generates no
    /// turnover at all and impact can never bite. Otherwise the exact solution of
    /// `gross - fixed = sqrt(A) * impact_per_sqrt_aum`.
    pub zero_crossing_usd: f64,
    /// Bars the book was evaluated over.
    pub traded_bars: usize,
    /// Traded legs whose symbol had no measurable dollar ADV. They still pay the fixed cost,
    /// and they are excluded from the impact sum and COUNTED, because one unpriceable name must
    /// not turn the whole curve into NaN and must not silently read as free capacity either.
    pub unpriced_impact_legs: usize,
    /// Per-symbol no-trade band the book was run under, in the same absolute weight units as
    /// [`Self::turnover_per_bar`]. `0.0` is the unbanded book.
    pub no_trade_band: f64,
    /// Share of rebalance legs the band FROZE, counting only legs that wanted to move.
    ///
    /// The band exists to cut turnover, so this is what says whether a given band did anything: a
    /// frontier point whose banded share is zero is the unbanded book wearing a different label,
    /// and one near one is a book that has stopped trading. Legs already sitting on their target
    /// are excluded, because counting them would report an unbanded book as heavily banded.
    pub banded_leg_share: f64,
}

impl CapacityCurve {
    /// The AUM at which net return crosses zero for an ASSUMED gross edge, in bps per bar.
    ///
    /// This, not [`Self::zero_crossing_usd`], is the usable capacity statement, because the cost
    /// side of the curve is a property of the BOOK and the CORPUS while the gross side is a
    /// property of the MODEL. The two are measured by different things on different hardware: this
    /// module runs on CPU over stored bars and has no model forecasts, so a capacity number
    /// conditioned on an edge is honest where a capacity number computed from a fabricated edge
    /// would not be. Feed it [`TRADE_BENCH_GROSS_BPS`] to get the capacity of the edge the trade
    /// bench actually measured.
    ///
    /// Same three branches as [`Self::zero_crossing_usd`]: `0.0` when the assumed edge cannot even
    /// pay the spread, `INFINITY` when the book generates no impact-bearing turnover, else the
    /// exact solution of `gross - fixed = sqrt(A) * impact_per_sqrt_aum`.
    pub fn zero_crossing_at_gross(&self, gross_bps: f64) -> f64 {
        let headroom = gross_bps - self.fixed_cost_bps;
        if !(headroom.is_finite() && self.impact_per_sqrt_aum.is_finite()) {
            f64::NAN
        } else if headroom <= 0.0 {
            0.0
        } else if self.impact_per_sqrt_aum <= 0.0 {
            f64::INFINITY
        } else {
            let root = headroom / self.impact_per_sqrt_aum;
            root * root
        }
    }

    /// Net return, bps/bar, at an assumed gross edge and a given AUM.
    pub fn net_at(&self, gross_bps: f64, aum_usd: f64) -> f64 {
        gross_bps - self.fixed_cost_bps - aum_usd.max(0.0).sqrt() * self.impact_per_sqrt_aum
    }

    /// Turnover as a fraction of the exposure actually carried: `1.0` is a full rotation per bar.
    pub fn rotation_per_bar(&self) -> f64 {
        self.turnover_per_bar / self.gross_exposure_per_bar
    }

    /// Cost per traded dollar, in bps, at which an assumed gross edge is exactly consumed.
    ///
    /// The same definition the per-window trade bench reports its `3.29` bps break-even under -
    /// edge divided by turnover - but on THIS book's turnover, so the two are comparable. It is
    /// the number the decile table must be read against: a book whose break-even is below the
    /// median all-in cost of the decile it trades in cannot be run at any AUM, which is a
    /// statement impact and capacity cannot rescue.
    pub fn break_even_cost_bps(&self, gross_bps: f64) -> f64 {
        if self.turnover_per_bar > 0.0 {
            gross_bps / self.turnover_per_bar
        } else {
            f64::INFINITY
        }
    }
}

/// AUM-independent per-bar accounting of one book, computed once so a thirteen-point sweep is
/// thirteen multiplies rather than thirteen passes over the panel.
#[derive(Clone, Debug)]
struct BookLedger {
    gross_sum: f64,
    fixed_cost_sum: f64,
    impact_root_sum: f64,
    turnover_sum: f64,
    gross_exposure_sum: f64,
    /// Mean and p99 of `|dw| / ADV` over traded legs. Participation at AUM `A` is exactly `A`
    /// times either, since both are positively homogeneous of degree one in `A`.
    participation_mean_per_usd: f64,
    participation_p99_per_usd: f64,
    bars: usize,
    unpriced_impact_legs: usize,
    banded_legs: usize,
    total_legs: usize,
}

/// Per-bar turnover and gross return of a book over a panel.
///
/// The turnover bookkeeping is the part that is easy to get wrong, so the invariant is stated
/// once: `held[i]` is the weight the book ACTUALLY carries, and it is the only thing the book
/// earns on. Each bar, every symbol in the slice is moved to its target unless the move is inside
/// [`BookSpec::no_trade_band`], and every symbol that was held but is NO LONGER IN THE PANEL is
/// unwound regardless of the band — a departing name cannot be held, so charging only the
/// still-present names would make a rotating universe look free.
///
/// Returns are earned on the POST-REBALANCE held weights, which is what makes the band's cost
/// measurable: skipping a trade saves the spread and simultaneously leaves the book on a stale
/// position, and both effects land in the same pass. Computing the gross return on the TARGET
/// weights instead would credit the book with an edge it never held and make every band look free.
fn book_ledger(
    slices: &[PanelSlice],
    forecasts: &[PanelForecast],
    model: &BarCostModel,
    spec: BookSpec,
) -> Result<BookLedger> {
    ensure!(
        slices.len() == forecasts.len(),
        "the panel has {} slices but {} forecasts",
        slices.len(),
        forecasts.len()
    );
    ensure!(
        spec.no_trade_band >= 0.0 && spec.no_trade_band.is_finite(),
        "a no-trade band must be a finite non-negative weight, got {}",
        spec.no_trade_band
    );
    let width = slices
        .iter()
        .flat_map(|slice| slice.symbols.iter().copied())
        .max()
        .map_or(0, |max| max as usize + 1);
    let mut held = vec![0.0f64; width];
    let mut target = vec![0.0f64; width];
    let mut in_slice = vec![false; width];
    let mut held_flag = vec![false; width];
    let mut held_ids: Vec<u32> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut participation: Vec<f64> = Vec::new();
    let mut gross_sum = 0.0f64;
    let mut fixed_cost_sum = 0.0f64;
    let mut impact_root_sum = 0.0f64;
    let mut turnover_sum = 0.0f64;
    let mut gross_exposure_sum = 0.0f64;
    let mut unpriced_impact_legs = 0usize;
    let mut banded_legs = 0usize;
    let mut total_legs = 0usize;

    for (slice, forecast) in slices.iter().zip(forecasts) {
        ensure!(
            slice.symbols.len() == slice.realized_r.len()
                && slice.symbols.len() == forecast.kelly_f.len(),
            "slice at {} has {} symbols, {} realized returns and {} forecasts",
            slice.ts_ms,
            slice.symbols.len(),
            slice.realized_r.len(),
            forecast.kelly_f.len()
        );
        book_weights(&forecast.kelly_f, spec, &mut weights);
        for (position, &symbol) in slice.symbols.iter().enumerate() {
            target[symbol as usize] = weights[position];
            in_slice[symbol as usize] = true;
        }
        let mut charge = |symbol: u32, delta: f64| {
            if !delta.is_finite() || delta <= 0.0 {
                return;
            }
            turnover_sum += delta;
            let resolved = model.resolve(symbol, slice.ts_ms);
            fixed_cost_sum += delta * resolved.fixed_bps();
            // A leg is priceable only if BOTH its ADV and its volatility were measured. A
            // zero-or-NaN impact coefficient is a failed volatility measurement, not free
            // trading, and it is exactly as dangerous as an unmeasurable ADV - so it is counted
            // the same way rather than adding zero to `impact_root_sum`.
            let priceable = resolved.adv_usd.is_finite()
                && resolved.adv_usd > 0.0
                && resolved.impact_coefficient_bps.is_finite()
                && resolved.impact_coefficient_bps > 0.0;
            if priceable {
                let fraction_per_usd = delta / resolved.adv_usd;
                impact_root_sum +=
                    delta * resolved.impact_coefficient_bps * fraction_per_usd.sqrt();
                participation.push(fraction_per_usd);
            } else {
                unpriced_impact_legs += 1;
            }
        };
        // Rebalance the names in the panel, subject to the band.
        for &symbol in &slice.symbols {
            let index = symbol as usize;
            let delta = target[index] - held[index];
            total_legs += 1;
            if delta.abs() <= spec.no_trade_band {
                // Only a leg that WANTED to move counts as frozen; a leg already on its target
                // was never a trade and must not inflate the banded share of an unbanded book.
                if delta != 0.0 {
                    banded_legs += 1;
                }
                continue;
            }
            charge(symbol, delta.abs());
            held[index] = target[index];
        }
        // Unwind anything held that the panel no longer carries. The band does NOT apply: the
        // alternative is holding a position in a name that is not in the book's universe.
        for &symbol in &held_ids {
            let index = symbol as usize;
            if held[index] != 0.0 && !in_slice[index] {
                charge(symbol, held[index].abs());
                held[index] = 0.0;
            }
        }
        // Earn on what is HELD after rebalancing. `realized_r` is a LOG return, so the arithmetic
        // contribution of a weight is `w * expm1(r)`.
        for (position, &symbol) in slice.symbols.iter().enumerate() {
            let weight = held[symbol as usize];
            gross_sum += weight * (slice.realized_r[position] as f64).exp_m1();
            gross_exposure_sum += weight.abs();
            target[symbol as usize] = 0.0;
            in_slice[symbol as usize] = false;
        }
        // Carry forward every symbol still holding a position, whether it was in this slice or
        // survived from an earlier one, so nothing can be stranded un-unwound. `held_flag` mirrors
        // membership in `held_ids` so this stays O(N) per bar: a linear `contains` here would make
        // the pass O(T*N^2), which at 1,024 names over 8,192 instants is billions of comparisons
        // per curve and this function is called once per book per impact coefficient per band.
        held_ids.retain(|&symbol| {
            let live = held[symbol as usize] != 0.0;
            if !live {
                held_flag[symbol as usize] = false;
            }
            live
        });
        for &symbol in &slice.symbols {
            let index = symbol as usize;
            if held[index] != 0.0 && !held_flag[index] {
                held_flag[index] = true;
                held_ids.push(symbol);
            }
        }
    }

    let legs = participation.len() as f64;
    let participation_mean_per_usd = if legs > 0.0 {
        participation.iter().sum::<f64>() / legs
    } else {
        f64::NAN
    };
    let participation_p99_per_usd = quantile(&mut participation, 0.99);
    Ok(BookLedger {
        gross_sum,
        fixed_cost_sum,
        turnover_sum,
        gross_exposure_sum,
        impact_root_sum,
        participation_mean_per_usd,
        participation_p99_per_usd,
        bars: slices.len(),
        unpriced_impact_legs,
        banded_legs,
        total_legs,
    })
}

/// Net portfolio return against AUM, from impact-free up to where impact eats the edge.
///
/// Takes the concrete [`BarCostModel`] rather than `&dyn CostModel` because capacity is a
/// question about ADV as well as cost, and ADV is not on the trait. Splitting one calibration
/// across two arguments to preserve an abstraction nothing else needs would be worse than
/// naming the dependency.
pub fn capacity_curve(
    slices: &[PanelSlice],
    forecasts: &[PanelForecast],
    model: &BarCostModel,
    spec: BookSpec,
    aum_grid: &[f64],
) -> Result<CapacityCurve> {
    let ledger = book_ledger(slices, forecasts, model, spec)?;
    ensure!(ledger.bars > 0, "a capacity curve needs at least one bar");
    let bars = ledger.bars as f64;
    let gross_bps = 1.0e4 * ledger.gross_sum / bars;
    let fixed_cost_bps = ledger.fixed_cost_sum / bars;
    let impact_per_sqrt_aum = ledger.impact_root_sum / bars;

    let points = aum_grid
        .iter()
        .map(|&aum_usd| {
            let impact_cost_bps = aum_usd.max(0.0).sqrt() * impact_per_sqrt_aum;
            CapacityPoint {
                aum_usd,
                gross_bps,
                fixed_cost_bps,
                impact_cost_bps,
                net_bps: gross_bps - fixed_cost_bps - impact_cost_bps,
                mean_participation: aum_usd * ledger.participation_mean_per_usd,
                p99_participation: aum_usd * ledger.participation_p99_per_usd,
            }
        })
        .collect();

    // Closed form, not a bisection: every bar's impact is exactly `sqrt(AUM)` times a
    // constant, so the mean is too, and `gross - fixed = sqrt(A) * J` inverts exactly. The
    // battery test re-runs the WHOLE accounting at the crossing and asserts the net is zero
    // there, so the algebra is checked against the same code that produced the grid.
    let headroom = gross_bps - fixed_cost_bps;
    let zero_crossing_usd = if !(headroom.is_finite() && impact_per_sqrt_aum.is_finite()) {
        f64::NAN
    } else if headroom <= 0.0 {
        0.0
    } else if impact_per_sqrt_aum <= 0.0 {
        f64::INFINITY
    } else {
        let root = headroom / impact_per_sqrt_aum;
        root * root
    };

    Ok(CapacityCurve {
        style: spec.style,
        gross_leverage: spec.gross_leverage,
        impact_k: model.impact_k(),
        points,
        gross_bps,
        fixed_cost_bps,
        turnover_per_bar: ledger.turnover_sum / bars,
        gross_exposure_per_bar: ledger.gross_exposure_sum / bars,
        impact_per_sqrt_aum,
        zero_crossing_usd,
        traded_bars: ledger.bars,
        unpriced_impact_legs: ledger.unpriced_impact_legs,
        no_trade_band: spec.no_trade_band,
        banded_leg_share: if ledger.total_legs > 0 {
            ledger.banded_legs as f64 / ledger.total_legs as f64
        } else {
            f64::NAN
        },
    })
}

// ---------------------------------------------------------------------------
// The correlation diagnostic
// ---------------------------------------------------------------------------

/// One book's predicted-against-realized volatility, decomposed.
///
/// All three volatilities are ROOT MEAN SQUARE per-bar returns in bps of AUM, measured around
/// the book's own predicted mean, which is what a variance forecast is a forecast of.
#[derive(Clone, Copy, Debug)]
pub struct BookVolRatio {
    pub style: BookStyle,
    /// `sqrt(mean_t sum_i w_it^2 * var_model_it)`: the volatility the sizing implicitly
    /// predicts, since a diagonal `Sigma` is what per-name marginals amount to.
    pub predicted_bps: f64,
    /// The same independence assumption with the model's marginals replaced by the REALIZED
    /// per-name variances. The bridge that separates a marginal error from a correlation error.
    pub independent_bps: f64,
    /// `sqrt(mean_t (p_t - pbar)^2)`: the realized portfolio VOLATILITY, centered on the SAMPLE
    /// mean.
    ///
    /// Sample-centered, not predicted-mean-centered, and the distinction is the whole point of
    /// splitting this from [`Self::forecast_rms_bps`]. `predicted_bps` and `independent_bps` are
    /// built from variances, which carry no mean at all; centering the realized side on the
    /// model's predicted mean instead would fold forecast BIAS into a ratio that is supposed to
    /// measure CORRELATION. That is inert only while the forecast mean is zero, which is exactly
    /// the scenario case, and goes live the instant real model forecasts arrive - the case the
    /// number would be trusted most.
    pub realized_bps: f64,
    /// `sqrt(mean_t (p_t - mu_t)^2)`: the RMS error of the book's predicted mean.
    ///
    /// Reported beside the volatility rather than instead of it, because it is the honest scale
    /// of what the book actually got wrong: it contains both the volatility and the squared bias,
    /// and `forecast_rms_bps > realized_bps` is a directional forecast error rather than a
    /// correlation problem. Never a denominator of any factor.
    pub forecast_rms_bps: f64,
    /// `realized / independent`. Pure cross-section: the multiple by which stacking per-name
    /// Kelly bets over-levers a correlated book. **The headline.**
    pub correlation_factor: f64,
    /// `independent / predicted`. Pure per-name calibration of the head's variance.
    pub marginal_factor: f64,
    /// `realized / predicted` = `correlation_factor * marginal_factor`.
    pub total_factor: f64,
    /// `sum_i w_i / sum_i |w_i|`, averaged over bars: how directional the book is.
    pub net_gross_ratio: f64,
    /// `rms_t(sum_i w_it b_i) / rms_t(sum_i |w_it b_i|)` on the leading eigenvector `b` of the
    /// realized correlation matrix, in `[0, 1]`.
    ///
    /// Zero means the book carries no first-factor exposure and the factor term genuinely
    /// cancelled. A dollar-neutral book with a non-zero value here has NOT hedged the factor,
    /// only the dollars, and that is a finding rather than an implementation detail.
    pub factor_exposure: f64,
    /// Bars the book actually held a non-zero position on. `0` means the signal admitted no book
    /// of this style at all — a uniform per-name Kelly has no dollar-neutral allocation — and is
    /// why every other field is NaN when it happens.
    pub held_bars: usize,
}

/// Realized cross-sectional structure of the panel, and what it does to per-name sizing.
#[derive(Clone, Debug)]
pub struct CrossCorrelation {
    /// Symbols present in EVERY slice. Correlations need a common panel; a symbol with holes is
    /// dropped rather than zero-filled, because filling a missing return with zero deflates
    /// every correlation it participates in.
    pub panel_symbols: usize,
    pub dropped_symbols: usize,
    pub slices: usize,
    /// Exact over the full common panel, via `sum_ij c_ij = (1/T) sum_t (sum_i z_it)^2`, so the
    /// headline correlation is never subsampled.
    pub mean_pairwise_corr: f64,
    /// Symbols in the explicit correlation matrix, capped at [`MAX_EIGEN_DIM`].
    pub eigen_symbols: usize,
    /// Of those, the columns that actually VARIED over the span.
    ///
    /// A column whose return never moves is zeroed by standardization, so its diagonal is `0` and
    /// every pair it belongs to is an injected zero rather than a measured correlation. Those pairs
    /// are excluded from [`Self::median_pairwise_corr`], and this is how many columns were left: a
    /// gap between it and `eigen_symbols` says the panel carries dead names, which is itself a
    /// statement about the tradeable universe.
    pub live_eigen_symbols: usize,
    pub median_pairwise_corr: f64,
    /// `lambda_1 / N`: the share of total realized cross-sectional variance one common factor
    /// carries. Equals the mean pairwise correlation exactly under a one-factor structure, so
    /// the gap between the two measures how one-factor the panel is.
    pub first_factor_share: f64,
    /// Descending, up to [`REPORTED_FACTORS`] entries.
    pub factor_shares: Vec<f64>,
    /// `trace(C)^2 / ||C||_F^2`: the number of independent directions the cross-section behaves
    /// like. `N` for an uncorrelated panel, `1` for a single factor.
    pub effective_rank: f64,
    /// `(horizon in bars, mean pairwise correlation, non-overlapping blocks measured)`.
    ///
    /// Blocks are built only from runs of bars that are genuinely adjacent on the panel's own
    /// stride, so a block never straddles an overnight gap - see
    /// [`MatrixPanel::horizon_correlation`]. The block count travels with the number because at a
    /// session-length horizon it collapses to roughly one block per session, and a correlation
    /// read off a handful of blocks is noise wearing a term structure's clothes. Rising WITH an
    /// adequate block count is the Epps effect and means the 5-minute figure understates the
    /// co-movement a held position faces; flat is equally a finding, and means the diversification
    /// shortfall is not a sampling artifact a longer holding period would wash out.
    pub horizon_corr: Vec<(usize, f64, usize)>,
    /// `(year*12 + month0, mean pairwise correlation)` across the span.
    pub monthly_corr: Vec<(i32, f64)>,
    /// One entry per [`BOOK_STYLES`] style.
    pub books: Vec<BookVolRatio>,
}

impl CrossCorrelation {
    pub fn book(&self, style: BookStyle) -> Option<&BookVolRatio> {
        self.books.iter().find(|book| book.style == style)
    }

    /// The equal-weight over-levering factor a panel of `symbols` names would carry at this
    /// panel's measured mean pairwise correlation, `sqrt(1 + (N-1) * rho)`.
    ///
    /// An EXTRAPOLATION, labelled as one wherever it is reported. It is exact only for equal
    /// weights and uniform correlation, and it exists because the measured factor is a property
    /// of the panel's breadth: quoting a factor measured on 500 names as if it applied to 5,297
    /// would understate the problem by more than a factor of three.
    pub fn breadth_extrapolated_factor(&self, symbols: usize) -> f64 {
        if symbols == 0 || !self.mean_pairwise_corr.is_finite() {
            return f64::NAN;
        }
        (1.0 + (symbols as f64 - 1.0) * self.mean_pairwise_corr)
            .max(0.0)
            .sqrt()
    }
}

/// How much of the panel the `O(N^2)` diagnostics are allowed to form explicitly.
#[derive(Clone, Copy, Debug)]
pub struct CorrelationConfig {
    pub max_eigen_symbols: usize,
    /// Slices retained for the matrix diagnostics, taken as the MOST RECENT contiguous run
    /// rather than strided, because horizon aggregation needs adjacency.
    pub max_matrix_slices: usize,
    pub gross_leverage: f64,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            max_eigen_symbols: MAX_EIGEN_DIM,
            max_matrix_slices: 1 << 15,
            gross_leverage: 1.0,
        }
    }
}

/// Measure the panel's realized cross-section and what per-name sizing does to it.
pub fn cross_correlation(
    slices: &[PanelSlice],
    forecasts: &[PanelForecast],
    config: CorrelationConfig,
) -> Result<CrossCorrelation> {
    ensure!(
        slices.len() == forecasts.len(),
        "the panel has {} slices but {} forecasts",
        slices.len(),
        forecasts.len()
    );
    ensure!(slices.len() >= 2, "a correlation panel needs two slices");

    let (common, dropped) = common_symbols(slices)?;
    ensure!(
        common.len() >= MIN_PANEL_SYMBOLS,
        "only {} symbols are present in all {} slices; a cross-section needs at least \
         {MIN_PANEL_SYMBOLS}, and zero-filling the holes would deflate every correlation \
         reported",
        common.len(),
        slices.len()
    );

    let width = common.len();
    let position_of = index_map(&common);
    // Dense `T x N` log returns of the common panel: the one matrix this function retains, f32
    // because a correlation is scale free and four bytes buys twice the panel.
    let mut returns = Vec::<f32>::with_capacity(slices.len() * width);
    let mut timestamps = Vec::<i64>::with_capacity(slices.len());
    for slice in slices {
        let base = returns.len();
        returns.resize(base + width, 0.0);
        for (position, &symbol) in slice.symbols.iter().enumerate() {
            if let Some(&column) = position_of.get(&symbol) {
                returns[base + column] = slice.realized_r[position];
            }
        }
        timestamps.push(slice.ts_ms);
    }

    // Realized per-name moments of the SIMPLE return, the quantity a portfolio variance is
    // built from.
    let mut name_var = vec![0.0f64; width];
    let rows = slices.len();
    for column in 0..width {
        let mut sum = 0.0;
        let mut sq = 0.0;
        for row in 0..rows {
            let simple = (returns[row * width + column] as f64).exp_m1();
            sum += simple;
            sq += simple * simple;
        }
        let n = rows as f64;
        let mean = sum / n;
        name_var[column] = ((sq / n) - mean * mean).max(0.0) * n / (n - 1.0).max(1.0);
    }

    let mean_pairwise_corr = mean_pairwise_correlation(&returns, width, rows);

    // The `O(N^2)` half, on a capped sub-panel whose columns map back to real symbol ids.
    let matrix = MatrixPanel::sample(&returns, width, rows, &common, &timestamps, config);
    let stats = matrix.stats()?;
    let horizon_corr = CORR_HORIZONS
        .iter()
        .map(|&horizon| {
            let (corr, blocks) = matrix.horizon_correlation(horizon);
            (horizon, corr, blocks)
        })
        .collect();
    let monthly_corr = matrix.monthly_correlation();
    let loadings: Vec<(u32, f64)> = matrix
        .symbols
        .iter()
        .copied()
        .zip(stats.loadings.iter().copied())
        .collect();

    let books = BOOK_STYLES
        .iter()
        .map(|&style| {
            let spec = BookSpec::new(style, config.gross_leverage);
            let mut book = book_vol_ratio(slices, forecasts, &position_of, &name_var, spec)?;
            book.factor_exposure = factor_exposure(slices, forecasts, &loadings, spec);
            Ok(book)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(CrossCorrelation {
        panel_symbols: width,
        dropped_symbols: dropped,
        slices: rows,
        mean_pairwise_corr,
        eigen_symbols: matrix.width,
        live_eigen_symbols: stats.live_width,
        median_pairwise_corr: stats.median_pairwise,
        first_factor_share: stats.shares.first().copied().unwrap_or(f64::NAN),
        factor_shares: stats.shares,
        effective_rank: stats.effective_rank,
        horizon_corr,
        monthly_corr,
        books,
    })
}

/// Symbols present in every slice, and how many were dropped for having holes.
///
/// A symbol listed twice in one slice is an error rather than a duplicate to tolerate: it would
/// be double-counted in every weight normalization and every variance sum, silently.
fn common_symbols(slices: &[PanelSlice]) -> Result<(Vec<u32>, usize)> {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for slice in slices {
        seen.clear();
        for &symbol in &slice.symbols {
            ensure!(
                seen.insert(symbol),
                "the slice at {} lists symbol {symbol} twice; the panel is not a \
                 cross-section",
                slice.ts_ms
            );
            *counts.entry(symbol).or_insert(0) += 1;
        }
    }
    let mut common: Vec<u32> = counts
        .iter()
        .filter(|(_, &count)| count == slices.len())
        .map(|(&symbol, _)| symbol)
        .collect();
    common.sort_unstable();
    let dropped = counts.len() - common.len();
    Ok((common, dropped))
}

fn index_map(symbols: &[u32]) -> HashMap<u32, usize> {
    symbols
        .iter()
        .enumerate()
        .map(|(index, &symbol)| (symbol, index))
        .collect()
}

/// `(sum_ij c_ij - N) / (N^2 - N)` computed in `O(T*N)` without forming `C`.
///
/// `sum_ij c_ij = (1/T) sum_t (sum_i z_it)^2` for column-standardized `z`, which is the whole
/// trick: the mean pairwise correlation is a property of the cross-sectional SUM of
/// standardized returns and never needs a pairwise object at all.
fn mean_pairwise_correlation(returns: &[f32], width: usize, rows: usize) -> f64 {
    if width < 2 || rows < 2 {
        return f64::NAN;
    }
    let mut mean = vec![0.0f64; width];
    let mut sd = vec![0.0f64; width];
    for column in 0..width {
        let mut sum = 0.0;
        let mut sq = 0.0;
        for row in 0..rows {
            let value = returns[row * width + column] as f64;
            sum += value;
            sq += value * value;
        }
        let n = rows as f64;
        mean[column] = sum / n;
        sd[column] = ((sq / n) - mean[column] * mean[column]).max(0.0).sqrt();
    }
    let live: Vec<usize> = (0..width).filter(|&column| sd[column] > 0.0).collect();
    if live.len() < 2 {
        return f64::NAN;
    }
    let n = live.len() as f64;
    let mut total = 0.0;
    for row in 0..rows {
        let mut sum = 0.0;
        for &column in &live {
            sum += (returns[row * width + column] as f64 - mean[column]) / sd[column];
        }
        total += sum * sum;
    }
    let grand = total / rows as f64;
    (grand - n) / (n * n - n)
}

/// One book's realized-against-predicted volatility, streamed over the whole panel.
fn book_vol_ratio(
    slices: &[PanelSlice],
    forecasts: &[PanelForecast],
    position_of: &HashMap<u32, usize>,
    name_var: &[f64],
    spec: BookSpec,
) -> Result<BookVolRatio> {
    let mut weights: Vec<f64> = Vec::new();
    let mut realized_sum = 0.0f64;
    let mut realized_sq = 0.0f64;
    let mut error_sq = 0.0f64;
    let mut predicted = 0.0f64;
    let mut independent = 0.0f64;
    let mut net_gross = 0.0f64;
    let mut bars = 0usize;
    for (slice, forecast) in slices.iter().zip(forecasts) {
        ensure!(
            slice.symbols.len() == forecast.kelly_f.len()
                && slice.symbols.len() == forecast.var_r.len()
                && slice.symbols.len() == forecast.mean_r.len()
                && slice.symbols.len() == slice.realized_r.len(),
            "the slice at {} is not shaped like its forecast",
            slice.ts_ms
        );
        // Weights are formed on the FULL slice — the book the sizing would actually build — and
        // the panel-common subset is what the variance identity is evaluated on, so a dropped
        // symbol shrinks the book rather than silently re-normalizing it upward.
        book_weights(&forecast.kelly_f, spec, &mut weights);
        let mut realized = 0.0f64;
        let mut mean = 0.0f64;
        let mut predicted_var = 0.0f64;
        let mut independent_var = 0.0f64;
        let mut gross = 0.0f64;
        let mut net = 0.0f64;
        for (position, &symbol) in slice.symbols.iter().enumerate() {
            let Some(&column) = position_of.get(&symbol) else {
                continue;
            };
            let w = weights[position];
            if w == 0.0 {
                continue;
            }
            realized += w * (slice.realized_r[position] as f64).exp_m1();
            // `mean_r` and `var_r` are already SIMPLE-return moments, `E[R]` and `Var[R]` over
            // the 128 bins with per-bin payoff `exp(center) - 1`, the same convention the Kelly
            // solve is defined on. Only `realized_r` is a log return.
            mean += w * forecast.mean_r[position] as f64;
            predicted_var += w * w * (forecast.var_r[position] as f64).max(0.0);
            independent_var += w * w * name_var[column];
            gross += w.abs();
            net += w;
        }
        if gross <= 0.0 {
            continue;
        }
        // TWO accumulations, deliberately. `realized_sum`/`realized_sq` give the SAMPLE-centered
        // volatility, the only quantity commensurable with a variance forecast; `error_sq` gives
        // the forecast RMS, which carries the bias and is reported separately rather than divided
        // by anything.
        realized_sum += realized;
        realized_sq += realized * realized;
        let error = realized - mean;
        error_sq += error * error;
        predicted += predicted_var;
        independent += independent_var;
        net_gross += net / gross;
        bars += 1;
    }
    if bars == 0 {
        // A degenerate book is a property of the SIGNAL, not a failure of the measurement: a
        // uniform per-name Kelly has no dollar-neutral allocation at all. It reports NaN and the
        // other two books still report, because erroring here would let one degenerate style
        // delete the entire cross-sectional diagnostic. `held_bars == 0` says which case it is.
        return Ok(BookVolRatio {
            style: spec.style,
            predicted_bps: f64::NAN,
            independent_bps: f64::NAN,
            realized_bps: f64::NAN,
            forecast_rms_bps: f64::NAN,
            correlation_factor: f64::NAN,
            marginal_factor: f64::NAN,
            total_factor: f64::NAN,
            net_gross_ratio: f64::NAN,
            factor_exposure: f64::NAN,
            held_bars: 0,
        });
    }
    let n = bars as f64;
    // Sample variance with the `n - 1` correction, because the mean is estimated from the same
    // sample. `predicted` and `independent` are means of per-bar variances and need no correction.
    let realized_mean = realized_sum / n;
    let realized_var = if bars > 1 {
        ((realized_sq / n) - realized_mean * realized_mean).max(0.0) * n / (n - 1.0)
    } else {
        f64::NAN
    };
    let realized_bps = 1.0e4 * realized_var.sqrt();
    let forecast_rms_bps = 1.0e4 * (error_sq / n).sqrt();
    let predicted_bps = 1.0e4 * (predicted / n).sqrt();
    let independent_bps = 1.0e4 * (independent / n).sqrt();
    Ok(BookVolRatio {
        style: spec.style,
        predicted_bps,
        independent_bps,
        realized_bps,
        forecast_rms_bps,
        correlation_factor: realized_bps / independent_bps,
        marginal_factor: independent_bps / predicted_bps,
        total_factor: realized_bps / predicted_bps,
        net_gross_ratio: net_gross / n,
        factor_exposure: f64::NAN,
        held_bars: bars,
    })
}

/// `rms_t(sum_i w_it b_i) / rms_t(sum_i |w_it b_i|)`, the share of the book's gross factor bet
/// that does not cancel.
///
/// `b` is the SIGNED leading eigenvector - see [`MatrixStats::loadings`]. The global sign cancels
/// because both numerator and denominator are homogeneous of degree one in `b`, but the relative
/// signs are what let a book that is long high-beta and short low-beta report the small net factor
/// bet it actually carries instead of the gross one.
fn factor_exposure(
    slices: &[PanelSlice],
    forecasts: &[PanelForecast],
    loadings: &[(u32, f64)],
    spec: BookSpec,
) -> f64 {
    if loadings.is_empty() {
        return f64::NAN;
    }
    let loading_of: HashMap<u32, f64> = loadings.iter().copied().collect();
    let mut weights: Vec<f64> = Vec::new();
    let mut signed_sq = 0.0f64;
    let mut gross_sq = 0.0f64;
    let mut bars = 0usize;
    for (slice, forecast) in slices.iter().zip(forecasts) {
        if slice.symbols.len() != forecast.kelly_f.len() {
            return f64::NAN;
        }
        book_weights(&forecast.kelly_f, spec, &mut weights);
        let mut signed = 0.0f64;
        let mut gross = 0.0f64;
        for (position, symbol) in slice.symbols.iter().enumerate() {
            let Some(&loading) = loading_of.get(symbol) else {
                continue;
            };
            let bet = weights[position] * loading;
            signed += bet;
            gross += bet.abs();
        }
        if gross <= 0.0 {
            continue;
        }
        signed_sq += signed * signed;
        gross_sq += gross * gross;
        bars += 1;
    }
    if bars == 0 || gross_sq <= 0.0 {
        return f64::NAN;
    }
    (signed_sq / gross_sq).sqrt()
}

/// The capped sub-panel every `O(N^2)` diagnostic runs on.
struct MatrixPanel {
    /// Row-major `rows x width` LOG returns.
    returns: Vec<f32>,
    width: usize,
    rows: usize,
    /// Real panel symbol ids, one per column, so a loading can be attributed to a symbol.
    symbols: Vec<u32>,
    timestamps: Vec<i64>,
}

/// Everything read off the explicit correlation matrix.
struct MatrixStats {
    /// `lambda_k / trace(C)`, descending, at most [`REPORTED_FACTORS`] entries.
    shares: Vec<f64>,
    effective_rank: f64,
    median_pairwise: f64,
    /// The leading eigenvector, SIGNED, one entry per column.
    ///
    /// Signed, not absolute. The global sign is arbitrary and cancels in
    /// [`BookVolRatio::factor_exposure`], which is a ratio of squares — but the RELATIVE signs are
    /// the whole content: a book long two names whose loadings have opposite sign genuinely
    /// carries no factor bet, and absolute loadings would report it as fully exposed. It never
    /// bites on a long-only equity panel, where every first-factor loading shares a sign, and
    /// bites hardest on the dollar-neutral construction that is the only correctly-levered one
    /// available.
    loadings: Vec<f64>,
    /// Columns with non-zero variance. A dead column is zeroed by [`MatrixPanel::standardized`],
    /// so its diagonal is `0` and every pair it takes part in is `0`; those pairs are excluded
    /// from `median_pairwise` rather than counted as evidence of zero correlation.
    live_width: usize,
}

impl MatrixPanel {
    /// Deterministic stride over symbols, and the MOST RECENT contiguous run of slices.
    ///
    /// Striding the symbols keeps the sample spread across the whole ticker ordering rather
    /// than truncating to an alphabetic prefix; taking slices contiguously is required because
    /// [`Self::horizon_correlation`] aggregates adjacent bars and a strided time axis would
    /// aggregate across gaps.
    fn sample(
        returns: &[f32],
        width: usize,
        rows: usize,
        common: &[u32],
        timestamps: &[i64],
        config: CorrelationConfig,
    ) -> Self {
        let stride = width.div_ceil(config.max_eigen_symbols.max(1)).max(1);
        let columns: Vec<usize> = (0..width).step_by(stride).collect();
        let keep = rows.min(config.max_matrix_slices.max(2));
        let first = rows - keep;
        let mut sampled = Vec::with_capacity(keep * columns.len());
        for row in first..rows {
            for &column in &columns {
                sampled.push(returns[row * width + column]);
            }
        }
        Self {
            width: columns.len(),
            rows: keep,
            symbols: columns.iter().map(|&column| common[column]).collect(),
            returns: sampled,
            timestamps: timestamps[first..].to_vec(),
        }
    }

    /// Column-standardized returns, with dead columns zeroed rather than divided by zero.
    fn standardized(&self) -> Option<Tensor> {
        if self.width < 2 || self.rows < 2 {
            return None;
        }
        let raw = Tensor::from_slice(&self.returns)
            .reshape([self.rows as i64, self.width as i64])
            .to_kind(Kind::Double);
        let mean = raw.mean_dim([0i64].as_slice(), true, Kind::Double);
        let centered = &raw - &mean;
        let sd = centered
            .pow_tensor_scalar(2.0)
            .mean_dim([0i64].as_slice(), true, Kind::Double)
            .sqrt();
        // A column with no variation carries no correlation; standardizing it would divide by
        // zero, so it is zeroed and drops out of every inner product instead. Its diagonal is
        // then `0` rather than `1`, which is why `trace` and not `width` is the denominator of
        // every variance share below.
        let live = sd.gt(0.0);
        let ones = Tensor::from(1.0f64);
        let safe = sd.where_self(&live, &ones);
        Some((&centered / &safe) * live.to_kind(Kind::Double))
    }

    /// Eigenvalue shares, participation ratio, median off-diagonal and leading eigenvector.
    ///
    /// The correlation matrix is formed explicitly — `width` is capped at [`MAX_EIGEN_DIM`], so
    /// this is at most a 1024x1024 symmetric eigendecomposition, milliseconds of LAPACK — and
    /// every statistic is read off the SAME matrix. Deriving the median from one object and the
    /// spectrum from another is how the two quietly stop describing the same panel.
    fn stats(&self) -> Result<MatrixStats> {
        let Some(z) = self.standardized() else {
            return Ok(MatrixStats {
                shares: Vec::new(),
                effective_rank: f64::NAN,
                median_pairwise: f64::NAN,
                loadings: Vec::new(),
                live_width: 0,
            });
        };
        tch::no_grad(|| {
            let corr = z.transpose(0, 1).matmul(&z) / self.rows as f64;
            let (values, vectors) = corr.linalg_eigh("L");
            let mut spectrum = Vec::<f64>::try_from(values.to_kind(Kind::Double))
                .context("reading the correlation spectrum")?;
            spectrum.retain(|value| value.is_finite());
            spectrum.sort_by(|a, b| b.total_cmp(a));
            let trace: f64 = spectrum.iter().sum();
            let frobenius: f64 = spectrum.iter().map(|value| value * value).sum();
            // `linalg_eigh` returns eigenvalues ASCENDING, so the last column is the leading
            // eigenvector. Kept signed - see `MatrixStats::loadings`.
            let loadings = Vec::<f64>::try_from(
                vectors
                    .select(1, self.width as i64 - 1)
                    .to_kind(Kind::Double),
            )
            .context("reading the leading factor loadings")?;
            let flat = Vec::<f64>::try_from(corr.reshape([-1]).to_kind(Kind::Double))
                .context("reading the correlation matrix")?;
            // A dead column's diagonal is 0 rather than 1, which is exactly how to identify it,
            // and every off-diagonal it participates in is an injected zero rather than a measured
            // correlation. Including those would drag the MEDIAN toward zero in proportion to the
            // dead share: five dead columns beside three perfectly correlated live ones make 25 of
            // 28 pairs zero, reporting a median of 0 for a panel whose live pairs are all 1.
            let live: Vec<usize> = (0..self.width)
                .filter(|&column| flat[column * self.width + column] > 0.0)
                .collect();
            let mut off_diagonal = Vec::with_capacity(live.len().saturating_sub(1) * live.len() / 2);
            for (offset, &row) in live.iter().enumerate() {
                for &column in &live[offset + 1..] {
                    off_diagonal.push(flat[row * self.width + column]);
                }
            }
            Ok(MatrixStats {
                shares: if trace > 0.0 {
                    spectrum
                        .iter()
                        .take(REPORTED_FACTORS)
                        .map(|value| value / trace)
                        .collect()
                } else {
                    Vec::new()
                },
                effective_rank: if frobenius > 0.0 {
                    trace * trace / frobenius
                } else {
                    f64::NAN
                },
                median_pairwise: median(&mut off_diagonal),
                loadings,
                live_width: live.len(),
            })
        })
    }

    /// The sampling interval of the retained slices, as the smallest positive gap between
    /// consecutive timestamps. `None` when the panel has no two distinct instants.
    fn stride_ms(&self) -> Option<i64> {
        self.timestamps
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .filter(|delta| *delta > 0)
            .min()
    }

    /// Mean pairwise correlation of returns aggregated over `horizon` NON-OVERLAPPING bars.
    ///
    /// Aggregation is a sum of LOG returns, which is exactly the multi-bar log return — but ONLY
    /// if the bars summed are genuinely adjacent in time. The panel's grid is contiguous inside a
    /// session and has an overnight gap between sessions, so a block that straddles a gap is not a
    /// multi-bar return at all: it is a sum over a hole, and summing it produced a "term structure"
    /// that was arithmetically fine and semantically empty. Blocks are therefore built only from
    /// runs whose consecutive timestamps differ by exactly the panel's stride, and a run shorter
    /// than `horizon` contributes nothing.
    ///
    /// A consequence worth stating: at `horizon` equal to a full session the number of usable
    /// blocks collapses to roughly one per session, so the estimate gets noisy exactly where the
    /// Epps effect is supposed to show up. `blocks` is what says whether the number means anything.
    fn horizon_correlation(&self, horizon: usize) -> (f64, usize) {
        if horizon == 0 {
            return (f64::NAN, 0);
        }
        if horizon == 1 {
            return (
                mean_pairwise_correlation(&self.returns, self.width, self.rows),
                self.rows,
            );
        }
        let Some(stride) = self.stride_ms() else {
            return (f64::NAN, 0);
        };
        let mut aggregated: Vec<f32> = Vec::new();
        let mut blocks = 0usize;
        let mut start = 0usize;
        while start < self.rows {
            // Extend a contiguous run as far as the stride holds.
            let mut end = start + 1;
            while end < self.rows && self.timestamps[end] - self.timestamps[end - 1] == stride {
                end += 1;
            }
            let run = end - start;
            for block in 0..run / horizon {
                let base = start + block * horizon;
                let out = aggregated.len();
                aggregated.resize(out + self.width, 0.0);
                for step in 0..horizon {
                    let row = base + step;
                    for column in 0..self.width {
                        aggregated[out + column] += self.returns[row * self.width + column];
                    }
                }
                blocks += 1;
            }
            start = end;
        }
        if blocks < 2 {
            return (f64::NAN, blocks);
        }
        (
            mean_pairwise_correlation(&aggregated, self.width, blocks),
            blocks,
        )
    }

    /// Mean pairwise correlation within each calendar month of the sub-panel's span.
    fn monthly_correlation(&self) -> Vec<(i32, f64)> {
        let mut out: Vec<(i32, f64)> = Vec::new();
        let mut start = 0usize;
        while start < self.rows {
            let month = month_index(self.timestamps[start]);
            let mut end = start + 1;
            while end < self.rows && month_index(self.timestamps[end]) == month {
                end += 1;
            }
            if let Some(month) = month {
                let rows = end - start;
                if rows >= 2 {
                    let block = &self.returns[start * self.width..end * self.width];
                    out.push((month, mean_pairwise_correlation(block, self.width, rows)));
                }
            }
            start = end;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// Everything the three charts are drawn from, so a caller measures once and reports once.
#[derive(Clone, Debug)]
pub struct CostCapacityReport {
    pub deciles: Vec<CostDecile>,
    /// One curve per [`IMPACT_K_GRID`] entry, all for the same book.
    pub capacity: Vec<CapacityCurve>,
    pub correlation: CrossCorrelation,
    /// Universe size the breadth extrapolation is quoted at.
    pub universe: usize,
    pub unmeasured_symbols: usize,
}

/// Write [`COST_DECILE_BASE`], [`CAPACITY_CURVE_BASE`] and [`CROSS_CORRELATION_BASE`].
///
/// All three are properties of the CORPUS — a spread, an ADV and a realized cross-sectional
/// covariance are measured from stored bars and do not move when a training step does — so they
/// are written by this battery rather than by the in-run reporter cycle, and are exempted from
/// that cycle's walk with this module's battery test named as their executor.
pub fn write_cost_capacity_reports(
    dir: &Path,
    report: &CostCapacityReport,
    suffix: &str,
) -> Result<()> {
    write_chart(
        dir,
        COST_DECILE_BASE,
        format!(
            "Pretrain Measured Trading Cost by Liquidity Decile (Roll spread PRIMARY, sqrt \
             impact k={IMPACT_K}, {} of {} symbols unmeasured; Corwin-Schultz failed on {}) \
             - {suffix}",
            report.unmeasured_symbols,
            report.universe,
            report
                .deciles
                .iter()
                .map(|decile| decile.cs_unmeasured)
                .sum::<usize>(),
        ),
        "liquidity decile (0 = thinnest tenth of the universe)",
        "bps one-way, and the liquidity that sets it",
        ScaleKind::Symlog,
        decile_series(&report.deciles),
    )?;

    write_chart(
        dir,
        CAPACITY_CURVE_BASE,
        format!(
            "Pretrain Capacity: Net Return vs AUM ({} book, gross {:.2}x held, turnover \
             {:.3}/bar = {:.2} rotations/bar, spread+fees {:.3} bps/bar, needs cost below \
             {:.2} bps/traded-$ at the bench's {TRADE_BENCH_GROSS_BPS:.4} gross, {}) - {suffix}",
            report
                .capacity
                .first()
                .map_or("no", |curve| curve.style.label()),
            report
                .capacity
                .first()
                .map_or(f64::NAN, |curve| curve.gross_exposure_per_bar),
            report
                .capacity
                .first()
                .map_or(f64::NAN, |curve| curve.turnover_per_bar),
            report
                .capacity
                .first()
                .map_or(f64::NAN, |curve| curve.rotation_per_bar()),
            report
                .capacity
                .first()
                .map_or(f64::NAN, |curve| curve.fixed_cost_bps),
            report.capacity.first().map_or(f64::NAN, |curve| curve
                .break_even_cost_bps(TRADE_BENCH_GROSS_BPS)),
            crossing_label(&report.capacity)
        ),
        "AUM grid index (see the `AUM ($M)` series)",
        "bps per bar of AUM (do NOT annualize naively: 23,436 bars/year compounds absurdly)",
        ScaleKind::Symlog,
        capacity_series(&report.capacity),
    )?;

    write_chart(
        dir,
        CROSS_CORRELATION_BASE,
        format!(
            "Pretrain Cross-Sectional Correlation and the Kelly Over-Levering Factor \
             (rho {:.4}, first factor {:.1}% of variance, long-only over-lever {:.2}x on \
             {} names) - {suffix}",
            report.correlation.mean_pairwise_corr,
            100.0 * report.correlation.first_factor_share,
            report
                .correlation
                .book(BookStyle::LongOnly)
                .map_or(f64::NAN, |book| book.correlation_factor),
            report.correlation.panel_symbols,
        ),
        "index (horizon, month or factor, per series)",
        "correlation, variance share, or volatility multiple",
        ScaleKind::Symlog,
        correlation_series(&report.correlation, report.universe),
    )?;
    Ok(())
}

fn decile_series(deciles: &[CostDecile]) -> Vec<ReportSeries> {
    let column = |label: &str, pick: &dyn Fn(&CostDecile) -> f64| ReportSeries {
        label: label.to_owned(),
        values: deciles.iter().map(|d| pick(d) as f32).collect(),
    };
    let mut series = vec![
        column("median ADV ($M)", &|d| d.median_adv_usd / 1.0e6),
        column("median price ($)", &|d| d.median_harmonic_price),
        column("Roll spread, PRIMARY (bps)", &|d| {
            d.median_roll_spread_bps
        }),
        column("CS spread (bps)", &|d| d.median_cs_spread_bps),
        column("CS spread, clamped (bps)", &|d| {
            d.median_cs_spread_bps_clamped
        }),
        column("AR spread (bps)", &|d| d.median_ar_spread_bps),
        column("Roll negative-product share", &|d| {
            d.median_roll_negative_share
        }),
        column("CS negative-window share", &|d| d.median_cs_negative_share),
        column("daily sigma", &|d| d.median_sigma_daily),
        column("fees (bps)", &|d| d.median_fee_bps),
        column("bench break-even (bps)", &|_| {
            super::trade_bench::DEFAULT_COST_BPS
        }),
    ];
    for (slot, participation) in PARTICIPATION_GRID.iter().enumerate() {
        series.push(ReportSeries {
            label: format!("all-in @{:.1}% ADV (bps)", 100.0 * participation),
            values: deciles
                .iter()
                .map(|d| d.median_all_in_bps.get(slot).copied().unwrap_or(f64::NAN) as f32)
                .collect(),
        });
    }
    series.push(column("symbols", &|d| d.symbols as f64));
    series.push(column("unmeasured symbols", &|d| d.unmeasured as f64));
    series.push(column("CS-unmeasured symbols", &|d| d.cs_unmeasured as f64));
    // Without this column every SIZED column above is a survivor median: unpriceable members are
    // dropped by `median`, they concentrate in the thin deciles, and that moves the median by
    // selection rather than by measurement.
    series.push(column("impact-unpriceable symbols", &|d| {
        d.impact_unpriceable as f64
    }));
    // And the same argument one column further left: an unmeasurable PRICE makes the commission a
    // NaN, so such a member is dropped from the impact-FREE column too. Without this counter the
    // `0.0` slot - the one quoted as the floor, the one every break-even comparison is made
    // against - is a survivor median as well, and the safest-looking number in the table is the one
    // carrying the silent selection.
    series.push(column("fixed-unmeasurable symbols", &|d| {
        d.fixed_unmeasurable as f64
    }));
    series
}

fn capacity_series(curves: &[CapacityCurve]) -> Vec<ReportSeries> {
    let mut series = Vec::new();
    if let Some(first) = curves.first() {
        series.push(ReportSeries {
            label: "AUM ($M)".to_owned(),
            values: first
                .points
                .iter()
                .map(|p| (p.aum_usd / 1.0e6) as f32)
                .collect(),
        });
        series.push(ReportSeries {
            label: "gross (bps/bar)".to_owned(),
            values: first.points.iter().map(|p| p.gross_bps as f32).collect(),
        });
        series.push(ReportSeries {
            label: "spread+fees (bps/bar)".to_owned(),
            values: first
                .points
                .iter()
                .map(|p| p.fixed_cost_bps as f32)
                .collect(),
        });
    }
    for curve in curves {
        series.push(ReportSeries {
            label: format!("net, k={} (bps/bar)", curve.impact_k),
            values: curve.points.iter().map(|p| p.net_bps as f32).collect(),
        });
        series.push(ReportSeries {
            label: format!("impact, k={} (bps/bar)", curve.impact_k),
            values: curve
                .points
                .iter()
                .map(|p| p.impact_cost_bps as f32)
                .collect(),
        });
    }
    if let Some(first) = curves.first() {
        series.push(ReportSeries {
            label: "mean participation (%ADV)".to_owned(),
            values: first
                .points
                .iter()
                .map(|p| (100.0 * p.mean_participation) as f32)
                .collect(),
        });
        series.push(ReportSeries {
            label: "p99 participation (%ADV)".to_owned(),
            values: first
                .points
                .iter()
                .map(|p| (100.0 * p.p99_participation) as f32)
                .collect(),
        });
    }
    for curve in curves {
        series.push(point_series(
            &format!("zero-crossing AUM ($M), k={}", curve.impact_k),
            curve.zero_crossing_usd / 1.0e6,
        ));
        series.push(point_series(
            &format!("turnover/bar, k={}", curve.impact_k),
            curve.turnover_per_bar,
        ));
        series.push(point_series(
            &format!("gross held/bar, k={}", curve.impact_k),
            curve.gross_exposure_per_bar,
        ));
        series.push(point_series(
            &format!("rotations/bar, k={}", curve.impact_k),
            curve.rotation_per_bar(),
        ));
        // What the decile table must be read against: the cost per traded dollar this book can
        // afford at the bench's own gross edge.
        series.push(point_series(
            &format!("affordable cost (bps/traded-$), k={}", curve.impact_k),
            curve.break_even_cost_bps(TRADE_BENCH_GROSS_BPS),
        ));
        series.push(point_series(
            &format!("no-trade band, k={}", curve.impact_k),
            curve.no_trade_band,
        ));
        series.push(point_series(
            &format!("banded leg share, k={}", curve.impact_k),
            curve.banded_leg_share,
        ));
    }
    // The usable capacity statement: cost is measured from the corpus, the edge belongs to a
    // checkpoint, so the crossing is reported as a curve over the assumed gross edge with the
    // trade bench's own measured +11.0170 bps/bar marked on it.
    if let Some(first) = curves.first() {
        series.push(ReportSeries {
            label: "assumed gross edge (bps/bar)".to_owned(),
            values: GROSS_EDGE_GRID.iter().map(|edge| *edge as f32).collect(),
        });
        for curve in curves {
            series.push(ReportSeries {
                label: format!("crossing AUM ($M) vs edge, k={}", curve.impact_k),
                values: GROSS_EDGE_GRID
                    .iter()
                    .map(|&edge| (curve.zero_crossing_at_gross(edge) / 1.0e6) as f32)
                    .collect(),
            });
        }
        series.push(point_series(
            "crossing AUM ($M) at the bench's measured gross",
            first.zero_crossing_at_gross(TRADE_BENCH_GROSS_BPS) / 1.0e6,
        ));
        series.push(point_series(
            "net at the bench's gross, zero AUM (bps/bar)",
            first.net_at(TRADE_BENCH_GROSS_BPS, 0.0),
        ));
        series.push(point_series(
            "the bench's own net under its flat 2.00 bps (bps/bar)",
            TRADE_BENCH_NET_BPS,
        ));
    }
    series
}

fn correlation_series(corr: &CrossCorrelation, universe: usize) -> Vec<ReportSeries> {
    let mut series = vec![
        ReportSeries {
            label: "horizon (bars)".to_owned(),
            values: corr
                .horizon_corr
                .iter()
                .map(|(horizon, _, _)| *horizon as f32)
                .collect(),
        },
        ReportSeries {
            label: "mean pairwise corr by horizon".to_owned(),
            values: corr
                .horizon_corr
                .iter()
                .map(|(_, value, _)| *value as f32)
                .collect(),
        },
        // The sample size of every point above. At a session-length horizon the gap-respecting
        // blocking leaves about one block per session, so this is what says whether the last
        // point of the term structure is a measurement or a rumour.
        ReportSeries {
            label: "non-overlapping blocks by horizon".to_owned(),
            values: corr
                .horizon_corr
                .iter()
                .map(|(_, _, blocks)| *blocks as f32)
                .collect(),
        },
        ReportSeries {
            label: "mean pairwise corr by month".to_owned(),
            values: corr
                .monthly_corr
                .iter()
                .map(|(_, value)| *value as f32)
                .collect(),
        },
        ReportSeries {
            label: "eigenvalue share (descending)".to_owned(),
            values: corr.factor_shares.iter().map(|s| *s as f32).collect(),
        },
        point_series("mean pairwise corr", corr.mean_pairwise_corr),
        point_series("median pairwise corr", corr.median_pairwise_corr),
        point_series("first-factor variance share", corr.first_factor_share),
        point_series("effective rank", corr.effective_rank),
        point_series("panel symbols", corr.panel_symbols as f64),
        point_series("eigen symbols", corr.eigen_symbols as f64),
        point_series("live eigen symbols", corr.live_eigen_symbols as f64),
        point_series("dropped symbols", corr.dropped_symbols as f64),
        point_series(
            "equal-weight over-lever at full universe",
            corr.breadth_extrapolated_factor(universe),
        ),
    ];
    for book in &corr.books {
        let name = book.style.label();
        series.push(point_series(
            &format!("{name}: correlation over-lever"),
            book.correlation_factor,
        ));
        series.push(point_series(
            &format!("{name}: marginal vol factor"),
            book.marginal_factor,
        ));
        series.push(point_series(
            &format!("{name}: total vol factor"),
            book.total_factor,
        ));
        series.push(point_series(
            &format!("{name}: first-factor exposure"),
            book.factor_exposure,
        ));
        series.push(point_series(
            &format!("{name}: net/gross"),
            book.net_gross_ratio,
        ));
        series.push(point_series(
            &format!("{name}: realized vol (bps/bar)"),
            book.realized_bps,
        ));
        series.push(point_series(
            &format!("{name}: forecast RMS error (bps/bar)"),
            book.forecast_rms_bps,
        ));
        series.push(point_series(
            &format!("{name}: independence-implied vol (bps/bar)"),
            book.independent_bps,
        ));
        series.push(point_series(
            &format!("{name}: bars held"),
            book.held_bars as f64,
        ));
    }
    series
}

fn crossing_label(curves: &[CapacityCurve]) -> String {
    if curves.is_empty() {
        return "no crossing measured".to_owned();
    }
    let body = curves
        .iter()
        .map(|curve| {
            let crossing = curve.zero_crossing_usd;
            if !crossing.is_finite() {
                format!("k={}: never", curve.impact_k)
            } else if crossing >= 1.0e9 {
                format!("k={}: ${:.2}B", curve.impact_k, crossing / 1.0e9)
            } else {
                format!("k={}: ${:.1}M", curve.impact_k, crossing / 1.0e6)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("zero-crossing {body}")
}

// ---------------------------------------------------------------------------
// Panels straight from the corpus, for the parts that need no model at all
// ---------------------------------------------------------------------------

/// Build a calendar-aligned panel of realized log returns from the corpus.
///
/// The correlation diagnostic needs no model: a realized cross-sectional covariance is a
/// property of the data. So this exists to answer the correlation question on real bars at full
/// breadth without touching the GPU, while [`cross_correlation`] itself takes whatever panel
/// the portfolio engine hands it.
///
/// Two passes over the mmap, on purpose. Retaining every symbol's return series to decide the
/// grid afterwards would cost hundreds of megabytes for a five-month span; the first pass keeps
/// only a timestamp histogram, and the second materializes exactly the retained grid. A symbol
/// is admitted only if it has a return at EVERY retained instant, which is what makes the panel
/// a cross-section rather than a ragged join.
pub fn corpus_panel(
    corpus: &crate::torch::dataset::BarCorpus,
    from_ms: i64,
    to_ms: i64,
    max_symbols: usize,
    max_slices: usize,
) -> Result<(Vec<PanelSlice>, Vec<u32>)> {
    ensure!(to_ms > from_ms, "an empty span carries no panel");
    ensure!(max_symbols >= MIN_PANEL_SYMBOLS, "the panel cap is too small");
    let stride = corpus.res_secs() as i64 * 1000;

    let mut coverage: HashMap<i64, usize> = HashMap::new();
    for series in 0..corpus.series_count() {
        for pair in span_slice(corpus.bars(series), from_ms, to_ms).windows(2) {
            if let Some(ts) = contiguous_return(pair[0], pair[1], stride, from_ms, to_ms) {
                *coverage.entry(ts.0).or_insert(0) += 1;
            }
        }
    }
    ensure!(!coverage.is_empty(), "no symbol has a return in the span");

    // Keep the instants the most symbols share, in time order and contiguous from the END of
    // the span, because the horizon term structure aggregates adjacent bars.
    let best = coverage.values().copied().max().unwrap_or(0);
    let floor = (best as f64 * 0.9) as usize;
    let mut grid: Vec<i64> = coverage
        .iter()
        .filter(|(_, &count)| count >= floor)
        .map(|(&ts, _)| ts)
        .collect();
    grid.sort_unstable();
    if grid.len() > max_slices {
        grid.drain(..grid.len() - max_slices);
    }
    ensure!(grid.len() >= 2, "the retained grid holds fewer than 2 instants");
    let slot_of: HashMap<i64, usize> = grid
        .iter()
        .enumerate()
        .map(|(index, &ts)| (ts, index))
        .collect();

    // Second pass: materialize only complete rows, ranked by traded dollars so a capped panel
    // is the tradeable end of the universe rather than an alphabetic prefix.
    let mut candidates: Vec<(f64, u32, Vec<f32>)> = Vec::new();
    for series in 0..corpus.series_count() {
        let bars = span_slice(corpus.bars(series), from_ms, to_ms);
        let mut row = vec![f32::NAN; grid.len()];
        let mut filled = 0usize;
        let mut dollars = 0.0f64;
        for pair in bars.windows(2) {
            let Some((ts, log_return)) = contiguous_return(pair[0], pair[1], stride, from_ms, to_ms)
            else {
                continue;
            };
            if let Some(&slot) = slot_of.get(&ts) {
                if row[slot].is_nan() {
                    filled += 1;
                }
                row[slot] = log_return;
                dollars += dollar_volume(pair[1]);
            }
        }
        if filled == grid.len() {
            candidates.push((dollars, series as u32, row));
        }
    }
    ensure!(
        candidates.len() >= MIN_PANEL_SYMBOLS,
        "only {} of {} symbols are complete on a {}-instant grid",
        candidates.len(),
        corpus.series_count(),
        grid.len()
    );
    if candidates.len() > max_symbols {
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        candidates.truncate(max_symbols);
    }
    candidates.sort_by_key(|(_, symbol, _)| *symbol);

    let symbols: Vec<u32> = candidates.iter().map(|(_, symbol, _)| *symbol).collect();
    let slices = grid
        .iter()
        .enumerate()
        .map(|(index, &ts_ms)| PanelSlice {
            ts_ms,
            symbols: symbols.clone(),
            realized_r: candidates.iter().map(|(_, _, row)| row[index]).collect(),
        })
        .collect();
    Ok((slices, symbols))
}

/// The records that can produce a return timestamped inside `[from_ms, to_ms)`.
///
/// Two binary searches instead of a full scan. Over the pinned val span that is roughly 8% of
/// the corpus's 451M bars, so it turns a 16 GB mmap walk per pass into a proportional one — which
/// matters because [`corpus_panel`] makes TWO passes and this runs on a box that is training.
/// The record BEFORE the span is retained: without it the first in-span bar has no predecessor
/// and its return would be silently missing, which is exactly the kind of hole that would make a
/// symbol look incomplete.
#[inline]
fn span_slice(bars: &[PackedBar], from_ms: i64, to_ms: i64) -> &[PackedBar] {
    let start = bars
        .partition_point(|bar| bar.ts_ms < from_ms)
        .saturating_sub(1);
    let end = bars.partition_point(|bar| bar.ts_ms < to_ms);
    &bars[start..end.max(start)]
}

/// `(timestamp of b, log return into b)` when `b` immediately follows `a` inside the span.
#[inline]
fn contiguous_return(
    a: PackedBar,
    b: PackedBar,
    stride: i64,
    from_ms: i64,
    to_ms: i64,
) -> Option<(i64, f32)> {
    let ts = b.ts_ms;
    if ts < from_ms || ts >= to_ms || ts - a.ts_ms != stride {
        return None;
    }
    let (pa, pb) = (a.close as f64, b.close as f64);
    if pa <= 0.0 || pb <= 0.0 {
        return None;
    }
    let log_return = (pb / pa).ln();
    log_return.is_finite().then_some((ts, log_return as f32))
}

#[inline]
fn dollar_volume(bar: PackedBar) -> f64 {
    let vwap = bar.vwap as f64;
    let price = if vwap.is_finite() && vwap > 0.0 {
        vwap
    } else {
        bar.close as f64
    };
    let volume = bar.volume as f64;
    if volume.is_finite() && volume > 0.0 && price.is_finite() && price > 0.0 {
        volume * price
    } else {
        0.0
    }
}

/// A book whose per-name MOMENTS are the panel's own realized moments and whose per-bar SIGNS
/// are a fixed, documented pattern rather than a model output.
///
/// **This is a scenario, not a forecast, and every number derived from it says so.** The
/// correlation over-levering factor is a function of the weights and the realized returns alone,
/// so it can be measured exactly without a model — but the WEIGHTS are the model's, and they are
/// not available on CPU. What IS available is the shape the trade bench measured at step 20000:
/// 83.8% of bars pinned at the `4x` cap and a mean signed position of `+0.76x`, so
/// `(0.76 + 4)/8 = 59.5%` of positions long. `long_share` instantiates exactly that, and the
/// long-only and dollar-neutral books bracket it.
///
/// The signs are drawn INDEPENDENTLY across names and bars, which is the optimistic end of the
/// range: the model's real signs are driven by one shared market state, so its per-bar net
/// exposure will be far more concentrated than this and its over-levering factor
/// correspondingly closer to the long-only bound.
///
/// `var_r` is set to the panel's realized per-name variance, which makes
/// [`BookVolRatio::marginal_factor`] exactly `1` by construction: the marginal-calibration
/// multiplier needs the head's own variances and belongs to whoever runs the model. The
/// CORRELATION multiplier does not, and is what this measures.
pub fn scenario_forecasts(slices: &[PanelSlice], long_share: f64, seed: u64) -> Vec<PanelForecast> {
    let width = slices.first().map_or(0, |slice| slice.symbols.len());
    let uniform = slices.iter().all(|slice| slice.symbols.len() == width);
    let mut var_r = vec![0.0f32; width];
    if uniform && slices.len() >= 2 {
        for column in 0..width {
            let mut sum = 0.0f64;
            let mut sq = 0.0f64;
            for slice in slices {
                let simple = (slice.realized_r[column] as f64).exp_m1();
                sum += simple;
                sq += simple * simple;
            }
            let n = slices.len() as f64;
            let mean = sum / n;
            var_r[column] = (((sq / n) - mean * mean).max(0.0) * n / (n - 1.0)) as f32;
        }
    }
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    slices
        .iter()
        .map(|slice| {
            let count = slice.symbols.len();
            PanelForecast {
                kelly_f: (0..count)
                    .map(|_| {
                        if rng.random::<f64>() < long_share {
                            1.0f32
                        } else {
                            -1.0f32
                        }
                    })
                    .collect(),
                mean_r: vec![0.0f32; count],
                var_r: if uniform {
                    var_r.clone()
                } else {
                    vec![0.0f32; count]
                },
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

/// Median of the finite entries, NaN when there are none. Mutates `values` by sorting it.
fn median(values: &mut Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Nearest-rank quantile of the finite entries. Mutates `values` by sorting it.
fn quantile(values: &mut Vec<f64>, probability: f64) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((values.len() as f64 - 1.0) * probability).round() as usize;
    values[index.min(values.len() - 1)]
}

fn first_finite_positive<const N: usize>(candidates: [Option<f64>; N]) -> Option<f64> {
    candidates
        .into_iter()
        .flatten()
        .find(|value| value.is_finite() && *value > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{read_report, ReportKind};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::{fs, path::PathBuf};

    static SCRATCH: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "portfolio_cost_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    const RES_SECS: u32 = 300;
    const STRIDE_MS: i64 = RES_SECS as i64 * 1000;
    /// Bars per synthetic session, one regular US 6.5-hour session of 5-minute bars.
    const SESSION_BARS: usize = 78;
    /// 2024-01-02T14:40:00Z, so a whole synthetic session lands inside one UTC day.
    const EPOCH_MS: i64 = 1_704_206_700_000;

    /// Bars generated from a random walk in the EFFICIENT price, observed through a fixed
    /// proportional spread: every tick prints at `mid * (1 +/- S/2)`, the bar's high is the
    /// largest print and its low the smallest.
    ///
    /// This is the data-generating process both estimators are derived under, which is what
    /// makes "the estimator recovers the spread it was given" a test of the implementation
    /// rather than of the estimator's literature. The overnight gap is left as a real gap so the
    /// contiguity filter is exercised.
    fn synthetic_bars(
        count: usize,
        spread: f64,
        sigma_bar: f64,
        start_price: f64,
        ticks_per_bar: usize,
        seed: u64,
    ) -> Vec<PackedBar> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut log_mid = start_price.ln();
        let per_tick = sigma_bar / (ticks_per_bar as f64).sqrt();
        let mut bars = Vec::with_capacity(count);
        for index in 0..count {
            let session = index / SESSION_BARS;
            let within = index % SESSION_BARS;
            let ts_ms = EPOCH_MS + session as i64 * 86_400_000 + within as i64 * STRIDE_MS;
            let open_mid = log_mid.exp();
            let mut high = f64::NEG_INFINITY;
            let mut low = f64::INFINITY;
            let mut last = open_mid;
            for _ in 0..ticks_per_bar {
                log_mid += per_tick * normal(&mut rng);
                let mid = log_mid.exp();
                let side = if rng.random::<bool>() { 1.0 } else { -1.0 };
                let print = mid * (1.0 + side * 0.5 * spread);
                high = high.max(print);
                low = low.min(print);
                last = print;
            }
            bars.push(PackedBar {
                ts_ms,
                open: open_mid as f32,
                high: high as f32,
                low: low as f32,
                close: last as f32,
                volume: 10_000.0,
                vwap: (0.5 * (high + low)) as f32,
                trades: ticks_per_bar as u32,
            });
        }
        bars
    }

    /// Box-Muller from a uniform stream: deterministic, and it keeps libtorch's process-global
    /// generator out of this module's tests entirely.
    fn normal(rng: &mut ChaCha12Rng) -> f64 {
        let u1: f64 = rng.random::<f64>().max(1.0e-12);
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A panel whose cross-section is exactly one factor plus idiosyncratic noise, so every
    /// diagnostic has a closed-form target.
    ///
    /// `r_it = beta_i * F_t + eps_it` with `var(F) = rho` and `var(eps_i) = 1 - rho`, scaled by
    /// `sigma`, so uniform `beta` gives uniform pairwise correlation `rho`.
    fn factor_panel(
        symbols: usize,
        rows: usize,
        rho: f64,
        sigma: f64,
        betas: &[f64],
        seed: u64,
    ) -> Vec<PanelSlice> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let ids: Vec<u32> = (0..symbols as u32).collect();
        (0..rows)
            .map(|row| {
                let factor = rho.sqrt() * normal(&mut rng);
                let realized_r = (0..symbols)
                    .map(|symbol| {
                        let idiosyncratic = (1.0 - rho).sqrt() * normal(&mut rng);
                        (sigma * (betas[symbol] * factor + idiosyncratic)) as f32
                    })
                    .collect();
                PanelSlice {
                    ts_ms: EPOCH_MS + row as i64 * STRIDE_MS,
                    symbols: ids.clone(),
                    realized_r,
                }
            })
            .collect()
    }

    /// Forecasts with a FIXED per-name Kelly pattern, so the weights are known exactly and the
    /// closed-form variance identity can be evaluated against them.
    fn fixed_forecasts(slices: &[PanelSlice], kelly: &[f32], var_r: &[f32]) -> Vec<PanelForecast> {
        slices
            .iter()
            .map(|slice| PanelForecast {
                kelly_f: kelly.to_vec(),
                mean_r: vec![0.0f32; slice.symbols.len()],
                var_r: var_r.to_vec(),
            })
            .collect()
    }

    fn calibration_of(series: &[(String, Vec<PackedBar>)]) -> Arc<CostCalibration> {
        let borrowed: Vec<(String, &[PackedBar])> = series
            .iter()
            .map(|(symbol, bars)| (symbol.clone(), bars.as_slice()))
            .collect();
        Arc::new(
            CostCalibration::from_series(&borrowed, RES_SECS).expect("the calibration measures"),
        )
    }

    /// The estimators' own data-generating process, run backwards.
    ///
    /// A 20 bps spread on bars whose own volatility is 10 bps is where Corwin-Schultz is meant
    /// to work, and both estimators land inside 25% of the injected value. The tolerance is wide
    /// on purpose: these are moment estimators of a second-order quantity, and pinning them
    /// tighter than their own sampling error would make the test a record of one seed.
    #[test]
    fn both_spread_estimators_recover_a_known_injected_spread() {
        let spread = 0.0020;
        let bars = synthetic_bars(20_000, spread, 0.0010, 50.0, 64, 0xC0FFEE);
        let measured = SymbolCost::measure("WIDE", &bars, RES_SECS);
        let cs = measured.pooled.cs_spread_bps;
        let ar = measured.pooled.ar_spread_bps;
        let target = 1.0e4 * spread;
        assert!(
            (cs - target).abs() / target < 0.25,
            "Corwin-Schultz recovered {cs:.3} bps from an injected {target:.3} bps"
        );
        assert!(
            (ar - target).abs() / target < 0.25,
            "Abdi-Ranaldo recovered {ar:.3} bps from an injected {target:.3} bps"
        );
        // 20,000 bars is 256 whole sessions plus a remainder: one window is lost at each
        // session boundary and one at the end of the series.
        assert_eq!(measured.pooled.pairs, 20_000 - 1 - 20_000 / SESSION_BARS as u64);
        // A wide spread against a quiet tape is the easy regime, so few windows should come out
        // negative here. This is the control for the next test.
        assert!(
            measured.pooled.cs_negative_share < 0.35,
            "negative-window share {} is too high for a 2x spread-to-volatility ratio",
            measured.pooled.cs_negative_share
        );
    }

    /// The failure mode the module refuses to clamp, demonstrated.
    ///
    /// A 1 bps spread buried in 40 bps of per-bar volatility makes a large minority of windows
    /// imply a negative spread. Two things must hold: the rate is REPORTED, and the
    /// clamp-then-average recipe is materially more biased than pooling — which is the entire
    /// argument for pooling, and is checked rather than asserted in prose.
    #[test]
    fn negative_spread_windows_are_reported_and_the_clamp_is_biased_upward() {
        let bars = synthetic_bars(20_000, 0.0001, 0.0040, 50.0, 64, 0xBADBEEF);
        let measured = SymbolCost::measure("THIN", &bars, RES_SECS);
        let negative = measured.pooled.cs_negative_share;
        assert!(
            negative > 0.15 && negative < 1.0,
            "a 1 bps spread under 40 bps of volatility should produce many negative windows, \
             got {negative}"
        );
        let pooled = measured.pooled.cs_spread_bps;
        let clamped = measured.pooled.cs_spread_bps_clamped;
        assert!(
            clamped > pooled + 1.0,
            "clamp-then-average ({clamped:.3} bps) must be visibly more biased than pooling \
             ({pooled:.3} bps); if it is not, the pooling argument is wrong"
        );
    }

    /// A symbol nobody can measure is priced at the cross-sectional median and COUNTED, never
    /// clamped to zero. Zero is the one number that would make an untradeable name look like the
    /// cheapest thing in the book.
    #[test]
    fn an_unmeasurable_symbol_falls_back_to_the_median_and_is_counted() {
        // A degenerate tape: every bar a single print, so high == low == close and both
        // estimators see no range at all.
        let flat: Vec<PackedBar> = (0..4_000)
            .map(|index| PackedBar {
                ts_ms: EPOCH_MS
                    + (index / SESSION_BARS) as i64 * 86_400_000
                    + (index % SESSION_BARS) as i64 * STRIDE_MS,
                open: 10.0,
                high: 10.0,
                low: 10.0,
                close: 10.0,
                volume: 100.0,
                vwap: 10.0,
                trades: 1,
            })
            .collect();
        let series = vec![
            (
                "WIDE".to_owned(),
                synthetic_bars(8_000, 0.0030, 0.0010, 50.0, 64, 1),
            ),
            (
                "MID".to_owned(),
                synthetic_bars(8_000, 0.0020, 0.0010, 50.0, 64, 2),
            ),
            ("FLAT".to_owned(), flat),
        ];
        let calibration = calibration_of(&series);
        assert_eq!(
            calibration.unmeasured,
            vec![2],
            "only the degenerate tape should be unmeasurable"
        );
        assert!(calibration.fallback_spread_bps > 0.0);
        let model = BarCostModel::new(Arc::clone(&calibration));
        let resolved = model.resolve(2, EPOCH_MS);
        assert!(resolved.spread_fallback, "the fallback must be flagged");
        assert!(
            (resolved.half_spread_bps - 0.5 * calibration.fallback_spread_bps).abs() < 1.0e-9,
            "the fallback must be the cross-sectional median, not zero"
        );
        assert!(!model.is_measured(2) && model.is_measured(0));
    }

    /// Impact is the square-root law, so quadrupling size doubles it and the fixed part does not
    /// move. Both halves matter: a cost model whose fixed part scaled with size would pass a
    /// ratio test on the total and still be wrong.
    #[test]
    fn impact_scales_as_the_square_root_of_size() {
        let series = vec![(
            "SYM".to_owned(),
            synthetic_bars(8_000, 0.0020, 0.0010, 50.0, 64, 7),
        )];
        let model = BarCostModel::new(calibration_of(&series));
        let resolved = model.resolve(0, EPOCH_MS);
        assert!(resolved.impact_coefficient_bps > 0.0);
        let base = resolved.impact_bps(0.0025);
        assert!(
            (resolved.impact_bps(0.01) - 2.0 * base).abs() < 1.0e-9,
            "4x the size must be exactly 2x the impact"
        );
        assert!(
            (resolved.impact_bps(0.0225) - 3.0 * base).abs() < 1.0e-9,
            "9x the size must be exactly 3x the impact"
        );
        assert_eq!(resolved.impact_bps(0.0), 0.0);
        assert!((resolved.total_bps(0.0) - resolved.fixed_bps()).abs() < 1.0e-12);
        // The trait sees the same number the internals do.
        let via_trait = CostModel::cost_bps(&model, 0, EPOCH_MS, 0.01) as f64;
        assert!((via_trait - resolved.total_bps(0.01)).abs() < 1.0e-3);
        // `k` is a stated literature default, and rescaling it rescales impact exactly.
        let doubled = model.with_impact_k(2.0 * model.impact_k());
        assert!(
            (doubled.resolve(0, EPOCH_MS).impact_bps(0.01) - 2.0 * resolved.impact_bps(0.01)).abs()
                < 1.0e-9
        );
    }

    /// The PRIMARY estimator, on the process it is derived under.
    ///
    /// `synthetic_bars` is exactly Roll's model - an efficient random walk observed through a
    /// fixed proportional spread with independent 50/50 sides - so `2*sqrt(-cov(r_t, r_{t+1}))`
    /// must return the injected spread. The tolerance is 5% rather than the range estimators' 25%
    /// because the serial covariance of 20,000 returns is a far better-conditioned moment than a
    /// difference of two nearly equal ranges, which is the whole reason this one is the primary.
    #[test]
    fn the_primary_roll_spread_recovers_a_known_injected_spread() {
        let spread = 0.0020;
        let bars = synthetic_bars(20_000, spread, 0.0010, 50.0, 64, 0xC0FFEE);
        let measured = SymbolCost::measure("WIDE", &bars, RES_SECS);
        let target = 1.0e4 * spread;
        let roll = measured.pooled.roll_spread_bps;
        assert!(
            (roll - target).abs() / target < 0.05,
            "Roll recovered {roll:.3} bps from an injected {target:.3} bps"
        );
        assert!(
            (measured
                .pooled
                .measured_spread_bps()
                .expect("a clean tape is measurable")
                - roll)
                .abs()
                < 1.0e-12,
            "the primary the cost model charges must BE the Roll estimate"
        );
        // Each estimator loses observations at every chain boundary, and Roll loses TWICE as many
        // because it consumes a pair of adjacent RETURNS where the range estimators consume a pair
        // of adjacent bars. 20,000 bars is 256 whole sessions plus a 32-bar remainder, so there
        // are 257 contiguous chains.
        let chains = 20_000 / SESSION_BARS as u64 + 1;
        assert_eq!(measured.pooled.pairs, 20_000 - chains);
        assert_eq!(measured.pooled.roll_pairs, 20_000 - 2 * chains);
        // `roll_negative_share` counts products that came out NEGATIVE, which for a bid-ask bounce
        // is the sign the estimator lives on: an efficient price plus an independent 50/50 side
        // makes consecutive returns anticorrelated. A majority is what a real spread looks like,
        // and a share at or below one half would say the bounce is not there.
        assert!(
            measured.pooled.roll_negative_share > 0.55,
            "a real spread must make the majority of adjacent return products negative, got {}",
            measured.pooled.roll_negative_share
        );
    }

    /// A tape whose RANGES are broken but whose CLOSES are clean is priced, not discarded.
    ///
    /// The two estimators have different sample sizes and the primary must be gated on its own.
    /// Gating Roll on the Corwin-Schultz window count discards every symbol with a bad `low`
    /// field - a zero, a `high < low` inversion - even though Roll measured it perfectly from the
    /// closes, and the discard lands silently in `CostCalibration::unmeasured`, which is the
    /// evidence the Roll promotion itself rests on.
    #[test]
    fn a_tape_with_broken_ranges_is_still_priced_from_its_closes() {
        let clean = synthetic_bars(8_000, 0.0020, 0.0010, 50.0, 64, 0x5EED);
        // Same closes, ranges destroyed: `usable` rejects every bar, so no Corwin-Schultz or
        // Abdi-Ranaldo window survives while every return pair does.
        let broken: Vec<PackedBar> = clean
            .iter()
            .map(|bar| PackedBar {
                high: 0.0,
                low: 0.0,
                ..*bar
            })
            .collect();
        let measured = SymbolCost::measure("BROKEN", &broken, RES_SECS);
        assert_eq!(
            measured.pooled.pairs, 0,
            "no range window can survive a destroyed high/low"
        );
        assert!(measured.pooled.roll_pairs > 7_000);
        assert!(
            measured.pooled.cs_measured_spread_bps().is_none(),
            "Corwin-Schultz must report failure on this tape"
        );
        let roll = measured
            .pooled
            .measured_spread_bps()
            .expect("clean closes are enough for the primary");
        let reference = SymbolCost::measure("CLEAN", &clean, RES_SECS)
            .pooled
            .measured_spread_bps()
            .expect("the clean tape measures");
        assert!(
            (roll - reference).abs() < 1.0e-9,
            "the ranges are not an input to Roll, so the estimate must be identical: \
             {roll} vs {reference}"
        );
    }

    /// A volatility of exactly zero is a measurement FAILURE, not free trading.
    ///
    /// A halted or pinned name gives a finite `sigma_daily == 0.0`, which is the most dangerous
    /// number in the module: it makes impact identically zero at every size, so the symbol has
    /// unbounded capacity, and unless it is counted the one safety counter designed to catch this
    /// class - `unpriced_impact_legs` - misses it entirely.
    #[test]
    fn a_symbol_with_no_volatility_is_unpriceable_rather_than_free() {
        let flat: Vec<PackedBar> = (0..4_000)
            .map(|index| PackedBar {
                ts_ms: EPOCH_MS
                    + (index / SESSION_BARS) as i64 * 86_400_000
                    + (index % SESSION_BARS) as i64 * STRIDE_MS,
                open: 10.0,
                high: 10.05,
                low: 9.95,
                close: 10.0,
                volume: 100_000.0,
                vwap: 10.0,
                trades: 40,
            })
            .collect();
        // The tape has a range every bar, so ADV and price are measurable and only the
        // close-to-close volatility is degenerate. That isolates the sigma path from the ADV path.
        //
        // A universe of ONE is also load-bearing: every fallback here is a CROSS-SECTIONAL median,
        // so a lone unmeasurable name has nothing to impute from and reaches the terminal
        // unpriceable branch. The companion case - where a measurable peer exists and the sigma IS
        // imputed - is a different branch, tested separately below.
        let series = vec![("PINNED".to_owned(), flat)];
        let calibration = calibration_of(&series);
        assert!(
            calibration.symbols[0].pooled.adv_usd > 0.0,
            "the fixture must have a measurable ADV, or it tests the wrong branch"
        );
        assert_eq!(calibration.symbols[0].pooled.sigma_daily, 0.0);
        let model = BarCostModel::new(Arc::clone(&calibration));
        let resolved = model.resolve(0, EPOCH_MS);
        assert!(
            !resolved.impact_coefficient_bps.is_finite(),
            "a zero volatility must propagate as unmeasurable, got {}",
            resolved.impact_coefficient_bps
        );
        // Zero SIZE costs zero impact even when the coefficient is unmeasurable, so the
        // impact-free column of the decile table is the fixed floor by construction rather than by
        // luck. It cannot be asserted finite HERE - a lone name has no cross-sectional spread to
        // fall back on either - so the finiteness half lives in the companion test below.
        assert_eq!(resolved.impact_bps(0.0), 0.0);
        assert!(
            resolved.total_bps(0.01).is_nan(),
            "at positive size an unmeasurable coefficient must propagate, not read as zero"
        );
        // And the decile table must COUNT it, because `median` silently drops the NaN.
        let counted: usize = model
            .deciles()
            .iter()
            .map(|d| d.impact_unpriceable)
            .sum();
        assert_eq!(
            counted, 1,
            "the one unpriceable symbol must be counted in exactly one decile"
        );
        // And the ledger must COUNT the leg rather than silently adding zero impact.
        let slices: Vec<PanelSlice> = (0..8)
            .map(|row| PanelSlice {
                ts_ms: EPOCH_MS + row as i64 * STRIDE_MS,
                symbols: vec![0],
                realized_r: vec![if row % 2 == 0 { 0.001 } else { -0.001 }],
            })
            .collect();
        let forecasts: Vec<PanelForecast> = (0..8)
            .map(|row| PanelForecast {
                kelly_f: vec![if row % 2 == 0 { 1.0 } else { -1.0 }],
                mean_r: vec![0.0],
                var_r: vec![1.0e-6],
            })
            .collect();
        let curve = capacity_curve(
            &slices,
            &forecasts,
            &model,
            BookSpec::new(BookStyle::Signed, 1.0),
            &AUM_GRID,
        )
        .expect("the curve computes");
        assert!(
            curve.unpriced_impact_legs > 0,
            "an unmeasurable volatility must be counted, not read as free capacity"
        );
        assert_eq!(
            curve.impact_per_sqrt_aum, 0.0,
            "no priceable leg means no impact SUM, which the leg count is what qualifies"
        );
    }

    /// With a measurable peer the zero-volatility name is IMPUTED, and never free.
    ///
    /// The companion of the test above. Every fallback in `resolve_from` is a cross-sectional
    /// median, so the presence of one measurable name changes the branch a degenerate one takes:
    /// unpriceable becomes imputed. The invariant that has to hold across BOTH branches is the same
    /// one, and it is the only one that matters for capacity - the coefficient is never `0.0`, so a
    /// name whose volatility could not be measured never reads as free to trade at size.
    ///
    /// This is also where the impact-free column is pinned FINITE: with a real cross-sectional
    /// spread the fixed floor is a number, so `median` keeps every member and the thinnest decile's
    /// floor is not a survivor median of the names that happened to have a measurable sigma.
    #[test]
    fn a_zero_volatility_symbol_beside_a_measurable_peer_is_imputed_not_free() {
        let flat: Vec<PackedBar> = (0..4_000)
            .map(|index| PackedBar {
                ts_ms: EPOCH_MS
                    + (index / SESSION_BARS) as i64 * 86_400_000
                    + (index % SESSION_BARS) as i64 * STRIDE_MS,
                open: 10.0,
                high: 10.05,
                low: 9.95,
                close: 10.0,
                volume: 100_000.0,
                vwap: 10.0,
                trades: 40,
            })
            .collect();
        let series = vec![
            ("PINNED".to_owned(), flat),
            (
                "LIQUID".to_owned(),
                synthetic_bars(4_000, 0.0020, 0.0010, 50.0, 64, 0xA11D),
            ),
        ];
        let calibration = calibration_of(&series);
        assert_eq!(calibration.symbols[0].pooled.sigma_daily, 0.0);
        assert!(
            calibration.fallback_sigma_daily > 0.0,
            "the peer must supply a cross-sectional sigma, or this is the lone-name branch"
        );
        let model = BarCostModel::new(Arc::clone(&calibration));
        let resolved = model.resolve(0, EPOCH_MS);
        assert!(
            resolved.impact_coefficient_bps > 0.0,
            "an imputed coefficient must be POSITIVE - zero is free trading at any size, got {}",
            resolved.impact_coefficient_bps
        );
        assert!(
            (resolved.impact_coefficient_bps - 1.0e4 * IMPACT_K * calibration.fallback_sigma_daily)
                .abs()
                < 1.0e-9,
            "the imputation must be the cross-sectional sigma, not an invented constant"
        );
        assert!(
            resolved.total_bps(0.01) > resolved.fixed_bps(),
            "at positive size the imputed impact must cost something"
        );
        for decile in model.deciles() {
            if decile.symbols == 0 {
                continue;
            }
            assert!(
                decile.median_all_in_bps[0].is_finite(),
                "decile {} lost its fixed floor, so its impact-free median is a survivor median",
                decile.decile
            );
            assert_eq!(
                decile.impact_unpriceable, 0,
                "every member is priceable once the cross-section can impute"
            );
        }
    }

    /// An unmeasurable PRICE must not make trading free, and must not make it exactly the cap.
    ///
    /// Commission is `per_share / price` in bps, capped. Two ways to get this wrong, and the second
    /// is the one that looks like the fix:
    ///
    /// - `price` falling back to `INFINITY` makes the quotient exactly `0.0` and the cap never
    ///   binds, so a symbol whose price could not be measured reports FREE commission. An absent
    ///   measurement rendering as the most favourable one available.
    /// - `price` falling back to `NAN` looks correct and is not: `f64::min` IGNORES NaN and returns
    ///   the other operand, so `(x / NAN).min(COMMISSION_CAP_FRACTION)` is the CAP. Still a
    ///   fabricated number, merely conservative rather than free, and "conservative" is not
    ///   "measured".
    ///
    /// So the clamp must be UNREACHABLE when its input is absent, which is what the `is_finite`
    /// guard in `resolve_from` does. This test pins both halves and the decile counter, because a
    /// NaN fixed floor is dropped by `median` and would otherwise turn the impact-free column into a
    /// survivor median.
    #[test]
    fn an_unmeasurable_price_makes_the_fee_unmeasurable_not_free_and_not_the_cap() {
        // Non-positive closes, so `SymbolCost::measure`'s `close > 0.0` gate never accepts a price.
        // A universe of ONE, so the cross-sectional fallback has nothing to impute from either.
        let priceless: Vec<PackedBar> = (0..4_000)
            .map(|index| PackedBar {
                ts_ms: EPOCH_MS
                    + (index / SESSION_BARS) as i64 * 86_400_000
                    + (index % SESSION_BARS) as i64 * STRIDE_MS,
                open: 0.0,
                high: 0.0,
                low: 0.0,
                close: 0.0,
                volume: 100_000.0,
                vwap: 0.0,
                trades: 40,
            })
            .collect();
        let series = vec![("VOID".to_owned(), priceless)];
        let calibration = calibration_of(&series);
        assert!(
            !calibration.symbols[0].pooled.harmonic_price.is_finite()
                || calibration.symbols[0].pooled.harmonic_price <= 0.0,
            "the fixture must have no measurable price, or it tests the wrong branch"
        );
        let model = BarCostModel::new(Arc::clone(&calibration));
        let resolved = model.resolve(0, EPOCH_MS);
        assert!(
            resolved.commission_bps.is_nan(),
            "an unmeasurable price must leave the fee unmeasurable, got {} bps",
            resolved.commission_bps
        );
        assert_ne!(
            resolved.commission_bps, 0.0,
            "a free fee is the favourable direction and the original defect"
        );
        let capped = 1.0e4 * COMMISSION_CAP_FRACTION;
        assert!(
            !(resolved.commission_bps == capped),
            "the cap is what an unguarded `min` over NaN returns, and it is not a measurement"
        );
        assert!(
            !resolved.fixed_bps().is_finite(),
            "the fixed floor must carry the absence, not absorb it"
        );
        // And the table must COUNT it, in the column that is otherwise measured on every member.
        let counted: usize = model
            .deciles()
            .iter()
            .map(|d| d.fixed_unmeasurable)
            .sum();
        assert_eq!(
            counted, 1,
            "the one unmeasurable fixed floor must be counted in exactly one decile"
        );
    }

    /// A horizon block never spans a tape gap, and the block count says how many survived.
    ///
    /// Summing log returns across an overnight hole is arithmetically fine and semantically
    /// empty: the panel carries no overnight return at all, so the "multi-bar return" would be a
    /// sum over a discontinuity. A term structure built that way reads as a clean scientific
    /// result and is not one.
    #[test]
    fn a_horizon_block_never_straddles_a_tape_gap() {
        let symbols = 16usize;
        let per_session = 12usize;
        let sessions = 5usize;
        let betas = vec![1.0f64; symbols];
        let contiguous = factor_panel(symbols, per_session * sessions, 0.3, 0.002, &betas, 0x11CE);
        // Same returns, re-stamped onto a session grid with a real overnight gap between blocks.
        let gapped: Vec<PanelSlice> = contiguous
            .iter()
            .enumerate()
            .map(|(row, slice)| PanelSlice {
                ts_ms: EPOCH_MS
                    + (row / per_session) as i64 * 86_400_000
                    + (row % per_session) as i64 * STRIDE_MS,
                symbols: slice.symbols.clone(),
                realized_r: slice.realized_r.clone(),
            })
            .collect();
        let config = CorrelationConfig::default();
        let kelly = vec![1.0f32; symbols];
        let var_r = vec![4.0e-6f32; symbols];
        let forecasts = fixed_forecasts(&gapped, &kelly, &var_r);
        let measured =
            cross_correlation(&gapped, &forecasts, config).expect("the gapped panel measures");
        let blocks_at = |horizon: usize| {
            measured
                .horizon_corr
                .iter()
                .find(|(h, _, _)| *h == horizon)
                .map(|(_, _, blocks)| *blocks)
                .expect("the horizon is on the grid")
        };
        assert_eq!(blocks_at(1), per_session * sessions);
        // 12 bars per session: horizon 6 fits twice per session, horizon 12 exactly once, and
        // horizon 39 not at all - which is a NaN with its block count, not a number.
        assert_eq!(blocks_at(6), 2 * sessions);
        assert_eq!(blocks_at(12), sessions);
        assert_eq!(blocks_at(39), 0);
        let long = measured
            .horizon_corr
            .iter()
            .find(|(h, _, _)| *h == 39)
            .map(|(_, corr, _)| *corr)
            .expect("39 is on the grid");
        assert!(
            long.is_nan(),
            "a horizon no gap-free run can hold must report NaN, not a cross-gap sum, got {long}"
        );
        // The blocking is the ONLY difference: on a panel with no gaps the same returns give the
        // same horizon-12 correlation, since there every 12-bar block is contiguous anyway.
        let flat_forecasts = fixed_forecasts(&contiguous, &kelly, &var_r);
        let flat = cross_correlation(&contiguous, &flat_forecasts, config)
            .expect("the contiguous panel measures");
        let gapped_12 = measured
            .horizon_corr
            .iter()
            .find(|(h, _, _)| *h == 12)
            .map(|(_, corr, _)| *corr)
            .expect("12 is on the grid");
        let flat_12 = flat
            .horizon_corr
            .iter()
            .find(|(h, _, _)| *h == 12)
            .map(|(_, corr, _)| *corr)
            .expect("12 is on the grid");
        assert!(
            (gapped_12 - flat_12).abs() < 1.0e-9,
            "session-aligned blocks must reproduce the gap-free answer exactly: {gapped_12} vs \
             {flat_12}"
        );
    }

    /// Factor exposure is measured on SIGNED loadings, so a book that is long both sides of a
    /// two-sided factor reports the small net bet it carries rather than its gross one.
    ///
    /// Two equal halves with `beta = +1` and `beta = -1` give a leading eigenvector whose signs
    /// split the same way. An equal-weight LONG-ONLY book over both halves is then almost
    /// factor-neutral by construction, and absolute loadings would report it as fully exposed:
    /// the exact failure that would certify a hedged book as unhedged, and the dollar-neutral
    /// construction is the only correctly-levered one available.
    #[test]
    fn factor_exposure_reads_the_sign_of_the_loading_not_its_magnitude() {
        let symbols = 64usize;
        let betas: Vec<f64> = (0..symbols)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let slices = factor_panel(symbols, 2_048, 0.5, 0.002, &betas, 0xB1A5);
        let kelly = vec![1.0f32; symbols];
        let var_r = vec![4.0e-6f32; symbols];
        let forecasts = fixed_forecasts(&slices, &kelly, &var_r);
        let measured = cross_correlation(&slices, &forecasts, CorrelationConfig::default())
            .expect("the two-sided panel measures");
        let long_only = measured
            .book(BookStyle::LongOnly)
            .expect("the long-only book is measured");
        assert!(
            long_only.factor_exposure < 0.25,
            "an equal-weight book over a two-sided factor carries almost no net factor bet; \
             absolute loadings would report ~1.0, got {}",
            long_only.factor_exposure
        );
        // And the panel really is one strong factor, so the low exposure is the BOOK cancelling
        // rather than the factor being absent.
        assert!(
            measured.first_factor_share > 0.4,
            "the fixture must carry a dominant factor, got {}",
            measured.first_factor_share
        );
    }

    /// A biased forecast mean must not move the correlation measurement.
    ///
    /// `predicted_bps` and `independent_bps` are built from variances and carry no mean, so
    /// centering the realized side on the model's predicted mean would fold forecast BIAS into a
    /// ratio whose entire job is to measure correlation. It is inert only while the forecast mean
    /// is zero - which every scenario here pins - and goes live the moment real model forecasts
    /// arrive, i.e. exactly when the number would be trusted most.
    #[test]
    fn a_biased_forecast_mean_moves_the_rms_and_not_the_correlation_factor() {
        let symbols = 32usize;
        let betas = vec![1.0f64; symbols];
        let slices = factor_panel(symbols, 2_048, 0.35, 0.002, &betas, 0xD1A5);
        let kelly: Vec<f32> = (0..symbols)
            .map(|index| 1.0 + 0.5 * (index as f32 / symbols as f32))
            .collect();
        let var_r = vec![4.0e-6f32; symbols];
        let unbiased = fixed_forecasts(&slices, &kelly, &var_r);
        // A large constant per-name mean: 50 bps per bar, which is enormous next to a 20 bps
        // per-bar volatility, so a mean-polluted variance would be dominated by it.
        let biased: Vec<PanelForecast> = unbiased
            .iter()
            .map(|forecast| PanelForecast {
                kelly_f: forecast.kelly_f.clone(),
                mean_r: vec![0.005f32; symbols],
                var_r: forecast.var_r.clone(),
            })
            .collect();
        let config = CorrelationConfig::default();
        let clean = cross_correlation(&slices, &unbiased, config).expect("the panel measures");
        let dirty = cross_correlation(&slices, &biased, config).expect("the panel measures");
        for style in BOOK_STYLES {
            let a = clean.book(style).expect("book measured");
            let b = dirty.book(style).expect("book measured");
            if a.held_bars == 0 {
                assert_eq!(b.held_bars, 0);
                continue;
            }
            assert!(
                (a.realized_bps - b.realized_bps).abs() < 1.0e-9,
                "{}: realized VOLATILITY must not depend on the forecast mean, {} vs {}",
                style.label(),
                a.realized_bps,
                b.realized_bps
            );
            assert!(
                (a.correlation_factor - b.correlation_factor).abs() < 1.0e-9,
                "{}: the correlation factor must not depend on the forecast mean, {} vs {}",
                style.label(),
                a.correlation_factor,
                b.correlation_factor
            );
            // The bias is not discarded - it is reported where it belongs. A dollar-neutral book
            // nets a constant per-name mean to zero, so only the directional books see it.
            if style == BookStyle::DollarNeutral {
                assert!(
                    (a.forecast_rms_bps - b.forecast_rms_bps).abs() < 1.0e-6,
                    "a dollar-neutral book cancels a constant mean"
                );
            } else {
                // Closed form rather than a ratio: the RMS is the Pythagorean sum of the
                // volatility and the bias, so `sqrt(rms^2 - vol^2)` must return the injected mean
                // error - 50 bps per unit of gross exposure, all of it long here - and that is a
                // number the test can name instead of a threshold it has to guess.
                let injected = 1.0e4 * 0.005 * config.gross_leverage;
                let recovered = (b.forecast_rms_bps * b.forecast_rms_bps
                    - a.forecast_rms_bps * a.forecast_rms_bps)
                    .sqrt();
                assert!(
                    (recovered - injected).abs() / injected < 0.05,
                    "{}: the RMS must carry exactly the injected {injected:.2} bps of mean \
                     error, recovered {recovered:.2} from {} vs {}",
                    style.label(),
                    a.forecast_rms_bps,
                    b.forecast_rms_bps
                );
            }
        }
    }

    /// Commission is charged per SHARE, so a cheap name is structurally expensive in bps. This is
    /// the effect a flat bps cost assumption erases, and it is large.
    #[test]
    fn a_low_priced_symbol_pays_more_in_fee_bps_than_a_high_priced_one() {
        let series = vec![
            (
                "CHEAP".to_owned(),
                synthetic_bars(8_000, 0.0020, 0.0010, 4.0, 64, 11),
            ),
            (
                "RICH".to_owned(),
                synthetic_bars(8_000, 0.0020, 0.0010, 400.0, 64, 12),
            ),
        ];
        let model = BarCostModel::new(calibration_of(&series));
        let cheap = model.resolve(0, EPOCH_MS).commission_bps;
        let rich = model.resolve(1, EPOCH_MS).commission_bps;
        assert!(
            cheap > 20.0 * rich,
            "a 100x price difference must show up as ~100x the commission bps, got {cheap:.3} \
             vs {rich:.3}"
        );
        assert!(
            cheap > 5.0,
            "$0.0035/share on a $4 name is 8.75 bps and must dwarf the bench's 3.29 bps \
             break-even, got {cheap:.3}"
        );
    }

    /// The liquidity table must actually order by liquidity: a wide, thin, cheap name has to land
    /// in a costlier decile than a tight, deep, expensive one.
    #[test]
    fn the_decile_table_orders_cost_by_liquidity() {
        let mut series = Vec::new();
        for index in 0..40u64 {
            let fraction = index as f64 / 39.0;
            let spread = 0.0002 + 0.0060 * (1.0 - fraction);
            let price = 5.0 + 200.0 * fraction;
            let mut bars = synthetic_bars(4_000, spread, 0.0010, price, 64, 100 + index);
            let volume = (1.0e3 + 1.0e6 * fraction) as f32;
            for bar in bars.iter_mut() {
                bar.volume = volume;
            }
            series.push((format!("S{index:02}"), bars));
        }
        let model = BarCostModel::new(calibration_of(&series));
        let deciles = model.deciles();
        assert_eq!(deciles.len(), DECILES);
        let thin = &deciles[0];
        let deep = &deciles[DECILES - 1];
        assert!(
            deep.median_adv_usd > 100.0 * thin.median_adv_usd,
            "the deciles are not ordered by ADV: {} vs {}",
            thin.median_adv_usd,
            deep.median_adv_usd
        );
        assert!(
            thin.median_cs_spread_bps > 5.0 * deep.median_cs_spread_bps,
            "the thin decile must be visibly wider: {:.3} vs {:.3} bps",
            thin.median_cs_spread_bps,
            deep.median_cs_spread_bps
        );
        let slot = PARTICIPATION_HEADLINE_SLOT;
        assert!(
            thin.median_all_in_bps[slot] > deep.median_all_in_bps[slot],
            "all-in cost at 1% of ADV must fall with liquidity"
        );
        // The impact-free column is the fixed floor, so it can never exceed a sized one.
        for decile in &deciles {
            assert!(decile.median_all_in_bps[0] <= decile.median_all_in_bps[slot] + 1.0e-9);
        }
    }

    /// The headline diagnostic against its closed form.
    ///
    /// For equal per-name variance and uniform pairwise correlation `rho`, the realized portfolio
    /// variance over the independence-implied one is exactly
    /// `(1 - rho) + rho * (sum w)^2 / sum w^2` for ANY weights. That identity is what makes this
    /// a test rather than a regression baseline: it predicts `sqrt(1 + (N-1) rho)` for an
    /// equal-weight long book and `sqrt(1 - rho)` for a dollar-neutral one, and both are checked
    /// against the measured number.
    #[test]
    fn a_known_factor_panel_recovers_its_eigenvalue_share_and_over_levering_factor() {
        let symbols = 64usize;
        let rho = 0.30;
        let sigma = 0.002;
        let betas = vec![1.0f64; symbols];
        let slices = factor_panel(symbols, 8_192, rho, sigma, &betas, 0x5EED);
        // Dispersed but strictly positive Kelly, so the long-only and dollar-neutral books are
        // both non-degenerate and the GENERAL identity is exercised rather than its equal-weight
        // special case.
        let kelly: Vec<f32> = (0..symbols)
            .map(|index| 1.0 + 0.5 * (index as f32 / symbols as f32))
            .collect();
        let var_r = vec![(sigma * sigma) as f32; symbols];
        let forecasts = fixed_forecasts(&slices, &kelly, &var_r);
        let measured = cross_correlation(&slices, &forecasts, CorrelationConfig::default())
            .expect("the panel measures");

        assert_eq!(measured.panel_symbols, symbols);
        assert_eq!(measured.dropped_symbols, 0);
        assert_eq!(measured.eigen_symbols, symbols);
        assert!(
            (measured.mean_pairwise_corr - rho).abs() < 0.02,
            "mean pairwise correlation {} should recover the injected {rho}",
            measured.mean_pairwise_corr
        );
        assert!(
            (measured.median_pairwise_corr - rho).abs() < 0.05,
            "median pairwise correlation {} should recover the injected {rho}",
            measured.median_pairwise_corr
        );
        let expected_share = (1.0 + (symbols as f64 - 1.0) * rho) / symbols as f64;
        assert!(
            (measured.first_factor_share - expected_share).abs() < 0.02,
            "first-factor share {} should recover (1 + (N-1) rho)/N = {expected_share}",
            measured.first_factor_share
        );
        // One factor plus isotropic noise: the participation ratio sits far below N and the
        // leading share far above 1/N.
        assert!(
            measured.effective_rank < symbols as f64
                && measured.effective_rank > 1.0
                && measured.first_factor_share > 5.0 / symbols as f64,
            "effective rank {} / share {} do not describe a one-factor panel",
            measured.effective_rank,
            measured.first_factor_share
        );

        let expected_factor = |weights: &[f64]| {
            let sum: f64 = weights.iter().sum();
            let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
            ((1.0 - rho) + rho * sum * sum / sum_sq).sqrt()
        };
        let mut weights = Vec::new();

        book_weights(&kelly, BookSpec::new(BookStyle::LongOnly, 1.0), &mut weights);
        let long_only = measured
            .book(BookStyle::LongOnly)
            .expect("the long-only book is measured");
        let target = expected_factor(&weights);
        assert!(
            (long_only.correlation_factor - target).abs() / target < 0.06,
            "long-only over-levering factor {} should be {target}",
            long_only.correlation_factor
        );
        // The identity's equal-weight reading, which is the number the report quotes.
        let breadth = (1.0 + (symbols as f64 - 1.0) * rho).sqrt();
        assert!(
            (long_only.correlation_factor - breadth).abs() / breadth < 0.10,
            "with near-uniform weights the factor {} should be sqrt(1 + (N-1) rho) = {breadth}",
            long_only.correlation_factor
        );
        // `var_r` IS the realized per-name variance here, so the marginal leg is exactly 1 and
        // the whole of the total factor is correlation. That is the decomposition working.
        assert!(
            (long_only.marginal_factor - 1.0).abs() < 0.08,
            "marginal factor {} should be 1 when the forecast variance is the realized one",
            long_only.marginal_factor
        );
        assert!(
            (long_only.total_factor - long_only.correlation_factor * long_only.marginal_factor)
                .abs()
                < 1.0e-9,
            "the decomposition must multiply back to the total"
        );
        assert!(
            long_only.factor_exposure > 0.9,
            "a long-only book on homogeneous loadings is almost pure factor, got {}",
            long_only.factor_exposure
        );

        book_weights(
            &kelly,
            BookSpec::new(BookStyle::DollarNeutral, 1.0),
            &mut weights,
        );
        let neutral = measured
            .book(BookStyle::DollarNeutral)
            .expect("the neutral book is measured");
        let neutral_target = (1.0 - rho).sqrt();
        assert!(
            (neutral.correlation_factor - neutral_target).abs() < 0.06,
            "dollar neutrality on homogeneous loadings must collapse the factor term to \
             sqrt(1 - rho) = {neutral_target}, got {}",
            neutral.correlation_factor
        );
        assert!(
            neutral.factor_exposure < 0.10,
            "a dollar-neutral book on homogeneous loadings carries no factor bet, got {}",
            neutral.factor_exposure
        );
        assert!(neutral.net_gross_ratio.abs() < 1.0e-6);
        assert!(neutral.held_bars == measured.slices && long_only.held_bars == measured.slices);
        // The closed form of this ratio is sqrt(1 + (N-1) rho) / sqrt(1 - rho) = 4.432/0.837 =
        // 5.30x, so a 4x floor is a real statement about the mechanism rather than a fitted
        // threshold: per-name Kelly over-levers a long book by a large multiple and a
        // homogeneous-loading neutral book not at all.
        let closed_form = breadth / (1.0 - rho).sqrt();
        assert!(
            long_only.correlation_factor / neutral.correlation_factor > 4.0
                && (long_only.correlation_factor / neutral.correlation_factor - closed_form).abs()
                    / closed_form
                    < 0.10,
            "the whole finding is that the long book over-levers and the neutral one does not: \
             {} vs {}, ratio should be {closed_form}",
            long_only.correlation_factor,
            neutral.correlation_factor
        );
        // Breadth extrapolation is monotone in breadth and agrees with the measured panel.
        let at_panel = measured.breadth_extrapolated_factor(symbols);
        assert!((at_panel - breadth).abs() / breadth < 0.05);
        assert!(measured.breadth_extrapolated_factor(5_297) > at_panel);
    }

    /// The case the brief says is itself a finding: dollar neutrality does NOT neutralize the
    /// factor when the loadings are heterogeneous, because zero net DOLLARS is not zero net BETA.
    #[test]
    fn dollar_neutrality_fails_to_hedge_heterogeneous_loadings() {
        let symbols = 64usize;
        let rho = 0.30;
        let sigma = 0.002;
        // Half the panel loads at 0.25, half at 1.75: same average beta, wildly different
        // cross-section.
        let betas: Vec<f64> = (0..symbols)
            .map(|index| if index % 2 == 0 { 0.25 } else { 1.75 })
            .collect();
        let slices = factor_panel(symbols, 8_192, rho, sigma, &betas, 0xFEED);
        // Long the high-beta half, short the low-beta half: dollar neutral, emphatically not
        // beta neutral.
        let kelly: Vec<f32> = (0..symbols)
            .map(|index| if index % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let var_r: Vec<f32> = betas
            .iter()
            .map(|beta| (sigma * sigma * (rho * beta * beta + (1.0 - rho))) as f32)
            .collect();
        let forecasts = fixed_forecasts(&slices, &kelly, &var_r);
        let measured = cross_correlation(&slices, &forecasts, CorrelationConfig::default())
            .expect("the panel measures");
        let neutral = measured
            .book(BookStyle::DollarNeutral)
            .expect("the neutral book is measured");
        assert!(
            neutral.net_gross_ratio.abs() < 1.0e-6,
            "the book must be dollar neutral for the finding to mean anything"
        );
        assert!(
            neutral.correlation_factor > 1.3,
            "a dollar-neutral book with a residual beta bet still over-levers; measured {}",
            neutral.correlation_factor
        );
        assert!(
            neutral.factor_exposure > 0.3,
            "the residual first-factor exposure is the mechanism and must be visible, got {}",
            neutral.factor_exposure
        );
    }

    /// The Epps effect: correlation measured at 5 minutes is a LOWER bound on the co-movement a
    /// position held across bars faces. Injected here as a factor that only shows up at the
    /// aggregate horizon, which is exactly how microstructure noise behaves.
    #[test]
    fn the_correlation_term_structure_rises_with_the_aggregation_horizon() {
        let symbols = 32usize;
        let rows = 4_096usize;
        let sigma = 0.002;
        let mut rng = ChaCha12Rng::seed_from_u64(0x0EF5);
        let ids: Vec<u32> = (0..symbols as u32).collect();
        // A common factor that persists over six bars against idiosyncratic noise that does not:
        // at horizon 1 the shared part is a small share of the variance, at horizon 6+ it is most
        // of it.
        let mut slices = Vec::with_capacity(rows);
        let mut factor = 0.0f64;
        for row in 0..rows {
            if row % 6 == 0 {
                factor = normal(&mut rng);
            }
            let realized_r = (0..symbols)
                .map(|_| (sigma * (0.4 * factor + normal(&mut rng))) as f32)
                .collect();
            slices.push(PanelSlice {
                ts_ms: EPOCH_MS + row as i64 * STRIDE_MS,
                symbols: ids.clone(),
                realized_r,
            });
        }
        // Dispersed rather than uniform, so all THREE books are non-degenerate: a uniform Kelly
        // has no dollar-neutral allocation at all, which the next test covers on purpose.
        let kelly: Vec<f32> = (0..symbols)
            .map(|index| 1.0 + 0.5 * (index as f32 / symbols as f32))
            .collect();
        let var_r = vec![(sigma * sigma) as f32; symbols];
        let forecasts = fixed_forecasts(&slices, &kelly, &var_r);
        let measured = cross_correlation(&slices, &forecasts, CorrelationConfig::default())
            .expect("the panel measures");
        let short = measured.horizon_corr[0].1;
        let long = measured
            .horizon_corr
            .iter()
            .find(|(horizon, _, _)| *horizon == 12)
            .map(|(_, corr, _)| *corr)
            .expect("horizon 12 is on the grid");
        assert!(
            long > short + 0.05,
            "aggregating a persistent common factor must raise measured correlation: {short} at \
             1 bar vs {long} at 12"
        );
        assert!(!measured.monthly_corr.is_empty());
    }

    /// A uniform per-name Kelly admits no dollar-neutral book. That is a property of the signal,
    /// so the measurement reports it as NaN with `held_bars == 0` and the other two books still
    /// report — rather than erroring and deleting the whole cross-sectional diagnostic.
    #[test]
    fn a_uniform_signal_reports_a_degenerate_dollar_neutral_book_rather_than_failing() {
        let symbols = 32usize;
        let betas = vec![1.0f64; symbols];
        let slices = factor_panel(symbols, 1_024, 0.25, 0.002, &betas, 0x51F);
        let kelly = vec![1.0f32; symbols];
        let var_r = vec![4.0e-6f32; symbols];
        let forecasts = fixed_forecasts(&slices, &kelly, &var_r);
        let measured = cross_correlation(&slices, &forecasts, CorrelationConfig::default())
            .expect("a degenerate book must not fail the whole measurement");
        let neutral = measured
            .book(BookStyle::DollarNeutral)
            .expect("the neutral book is still reported");
        assert_eq!(neutral.held_bars, 0);
        assert!(
            neutral.correlation_factor.is_nan() && neutral.realized_bps.is_nan(),
            "a book that never held a position must report NaN, not a number"
        );
        for style in [BookStyle::LongOnly, BookStyle::Signed] {
            let book = measured.book(style).expect("the book is reported");
            assert_eq!(book.held_bars, measured.slices);
            assert!(
                book.correlation_factor.is_finite() && book.correlation_factor > 1.0,
                "{} must still measure a real over-levering factor, got {}",
                style.label(),
                book.correlation_factor
            );
        }
    }

    /// A ragged panel is reported as ragged, not silently zero-filled: filling a missing return
    /// with zero deflates every correlation the symbol takes part in.
    #[test]
    fn a_symbol_with_holes_is_dropped_and_counted() {
        let symbols = 16usize;
        let betas = vec![1.0f64; symbols];
        let mut slices = factor_panel(symbols, 512, 0.2, 0.002, &betas, 0xA11);
        slices[100].symbols.pop();
        slices[100].realized_r.pop();
        let kelly = vec![1.0f32; symbols];
        let var_r = vec![4.0e-6f32; symbols];
        let mut forecasts = fixed_forecasts(&slices, &kelly, &var_r);
        forecasts[100].kelly_f.pop();
        forecasts[100].var_r.pop();
        let measured = cross_correlation(&slices, &forecasts, CorrelationConfig::default())
            .expect("the panel measures");
        assert_eq!(measured.panel_symbols, symbols - 1);
        assert_eq!(measured.dropped_symbols, 1);
    }

    /// A slice listing one symbol twice is rejected rather than double counted.
    #[test]
    fn a_duplicated_symbol_is_rejected() {
        let symbols = 16usize;
        let betas = vec![1.0f64; symbols];
        let mut slices = factor_panel(symbols, 64, 0.2, 0.002, &betas, 0xD0DE);
        slices[3].symbols[5] = slices[3].symbols[4];
        let kelly = vec![1.0f32; symbols];
        let var_r = vec![4.0e-6f32; symbols];
        let forecasts = fixed_forecasts(&slices, &kelly, &var_r);
        let error = cross_correlation(&slices, &forecasts, CorrelationConfig::default())
            .expect_err("a duplicated symbol must be rejected");
        assert!(format!("{error}").contains("twice"), "unexpected: {error}");
    }

    /// The capacity curve is monotone in AUM and crosses zero exactly where the closed form says,
    /// checked by re-running the SAME accounting at the crossing rather than by trusting algebra.
    #[test]
    fn the_capacity_curve_crosses_zero_where_the_accounting_says_it_does() {
        let symbols = 32usize;
        let mut series = Vec::new();
        for index in 0..symbols {
            let mut bars = synthetic_bars(8_000, 0.0004, 0.0010, 40.0, 64, 500 + index as u64);
            for bar in bars.iter_mut() {
                bar.volume = 200_000.0;
            }
            series.push((format!("C{index:02}"), bars));
        }
        let model = BarCostModel::new(calibration_of(&series));
        // A panel with a real, positive per-name edge and a book that genuinely rebalances, so
        // impact has turnover to bite on and the crossing is finite and interior.
        let mut rng = ChaCha12Rng::seed_from_u64(0xCAFE);
        let ids: Vec<u32> = (0..symbols as u32).collect();
        let slices: Vec<PanelSlice> = (0..2_000)
            .map(|row| PanelSlice {
                ts_ms: EPOCH_MS + row as i64 * STRIDE_MS,
                symbols: ids.clone(),
                realized_r: (0..symbols)
                    .map(|_| (0.0004 + 0.0010 * normal(&mut rng)) as f32)
                    .collect(),
            })
            .collect();
        let forecasts: Vec<PanelForecast> = slices
            .iter()
            .enumerate()
            .map(|(row, slice)| PanelForecast {
                kelly_f: (0..slice.symbols.len())
                    .map(|symbol| 1.0 + ((row * 7 + symbol * 13) % 11) as f32)
                    .collect(),
                mean_r: vec![0.0004; slice.symbols.len()],
                var_r: vec![1.0e-6; slice.symbols.len()],
            })
            .collect();
        let spec = BookSpec::new(BookStyle::LongOnly, 1.0);
        let curve = capacity_curve(&slices, &forecasts, &model, spec, &AUM_GRID)
            .expect("the capacity curve computes");

        assert_eq!(curve.traded_bars, slices.len());
        assert_eq!(curve.unpriced_impact_legs, 0);
        assert_eq!(curve.impact_k, IMPACT_K);
        for pair in curve.points.windows(2) {
            assert!(
                pair[1].net_bps <= pair[0].net_bps + 1.0e-12,
                "net return must be non-increasing in AUM: {:?} then {:?}",
                pair[0],
                pair[1]
            );
            assert!(
                (pair[0].gross_bps - pair[1].gross_bps).abs() < 1.0e-12
                    && (pair[0].fixed_cost_bps - pair[1].fixed_cost_bps).abs() < 1.0e-12,
                "gross return and spread cost do not depend on AUM"
            );
            assert!(pair[1].mean_participation > pair[0].mean_participation);
        }
        assert!(
            curve.points[0].net_bps > 0.0,
            "the fixture must be profitable at small size or the crossing is vacuous"
        );
        let crossing = curve.zero_crossing_usd;
        assert!(
            crossing.is_finite() && crossing > AUM_GRID[0] && crossing < *AUM_GRID.last().unwrap(),
            "the crossing {crossing} must be interior to the grid for this test to bite"
        );
        // Re-run the whole accounting at the crossing: the net there must be zero.
        let at_crossing = capacity_curve(&slices, &forecasts, &model, spec, &[crossing])
            .expect("the crossing evaluates");
        assert!(
            at_crossing.points[0].net_bps.abs() < 1.0e-9,
            "net at the closed-form crossing is {} bps, not zero",
            at_crossing.points[0].net_bps
        );
        // A larger impact coefficient must bring the crossing in, quadratically.
        let doubled = capacity_curve(
            &slices,
            &forecasts,
            &model.with_impact_k(2.0 * IMPACT_K),
            spec,
            &AUM_GRID,
        )
        .expect("the doubled curve computes");
        assert!(
            (doubled.zero_crossing_usd - crossing / 4.0).abs() / crossing < 1.0e-6,
            "doubling k must quarter the capacity: {} vs {}",
            doubled.zero_crossing_usd,
            crossing / 4.0
        );
    }

    /// A book that cannot pay for its own spread has zero capacity, not a large one. The
    /// degenerate direction matters: reporting NaN or a huge number here would read as "no
    /// capacity constraint found".
    #[test]
    fn a_book_that_cannot_pay_its_spread_has_zero_capacity() {
        let symbols = 8usize;
        let series: Vec<(String, Vec<PackedBar>)> = (0..symbols)
            .map(|index| {
                (
                    format!("W{index}"),
                    synthetic_bars(4_000, 0.0060, 0.0010, 20.0, 64, 900 + index as u64),
                )
            })
            .collect();
        let model = BarCostModel::new(calibration_of(&series));
        let ids: Vec<u32> = (0..symbols as u32).collect();
        let mut rng = ChaCha12Rng::seed_from_u64(0xDEAD);
        let slices: Vec<PanelSlice> = (0..500)
            .map(|row| PanelSlice {
                ts_ms: EPOCH_MS + row as i64 * STRIDE_MS,
                symbols: ids.clone(),
                realized_r: (0..symbols)
                    .map(|_| (0.00001 * normal(&mut rng)) as f32)
                    .collect(),
            })
            .collect();
        // Alternating sign forces full turnover every bar against a 60 bps spread.
        let forecasts: Vec<PanelForecast> = slices
            .iter()
            .enumerate()
            .map(|(row, slice)| PanelForecast {
                kelly_f: (0..slice.symbols.len())
                    .map(|symbol| if (row + symbol) % 2 == 0 { 1.0 } else { -1.0 })
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-8; slice.symbols.len()],
            })
            .collect();
        let curve = capacity_curve(
            &slices,
            &forecasts,
            &model,
            BookSpec::new(BookStyle::Signed, 1.0),
            &AUM_GRID,
        )
        .expect("the capacity curve computes");
        assert!(curve.points[0].fixed_cost_bps > curve.points[0].gross_bps);
        assert_eq!(
            curve.zero_crossing_usd, 0.0,
            "a book that cannot pay its spread has zero capacity, not an unbounded one"
        );
    }

    /// A symbol that leaves the panel is UNWOUND and charged for it. Charging only the still
    /// present names would make a rotating universe look free.
    #[test]
    fn a_symbol_leaving_the_panel_is_charged_its_unwind() {
        let series: Vec<(String, Vec<PackedBar>)> = (0..2)
            .map(|index| {
                (
                    format!("U{index}"),
                    synthetic_bars(4_000, 0.0020, 0.0010, 50.0, 64, 300 + index as u64),
                )
            })
            .collect();
        let model = BarCostModel::new(calibration_of(&series));
        // Bar 0 holds both names; bar 1 holds only the first, so the second must be unwound.
        let slices = vec![
            PanelSlice {
                ts_ms: EPOCH_MS,
                symbols: vec![0, 1],
                realized_r: vec![0.0, 0.0],
            },
            PanelSlice {
                ts_ms: EPOCH_MS + STRIDE_MS,
                symbols: vec![0],
                realized_r: vec![0.0],
            },
        ];
        let forecasts = vec![
            PanelForecast {
                kelly_f: vec![1.0, 1.0],
                mean_r: vec![0.0, 0.0],
                var_r: vec![1.0e-8, 1.0e-8],
            },
            PanelForecast {
                kelly_f: vec![1.0],
                mean_r: vec![0.0],
                var_r: vec![1.0e-8],
            },
        ];
        let curve = capacity_curve(
            &slices,
            &forecasts,
            &model,
            BookSpec::new(BookStyle::LongOnly, 1.0),
            &[0.0],
        )
        .expect("the capacity curve computes");
        // Turnover: 1.0 to build the two-name book, then 0.5 of unwind on the departing name plus
        // 0.5 of top-up on the survivor as it goes from half to the whole book. Total 2.0.
        // Both names carry their own measured spread, so the expected charge is the sum of the
        // two fixed costs: one full leg of each over two bars.
        let expected = (model.resolve(0, EPOCH_MS).fixed_bps()
            + model.resolve(1, EPOCH_MS).fixed_bps())
            / 2.0;
        assert!(
            (curve.points[0].fixed_cost_bps - expected).abs() < 1.0e-6,
            "expected {expected} bps/bar of fixed cost, got {}",
            curve.points[0].fixed_cost_bps
        );
    }

    /// Gross exposure is FIXED at the book's leverage regardless of breadth. This is the property
    /// whose absence produced a 1,024x gross book and a 24,900x-per-year headline.
    #[test]
    fn book_weights_hold_gross_exposure_fixed_as_breadth_grows() {
        let mut weights = Vec::new();
        for symbols in [4usize, 64, 1_024] {
            let kelly: Vec<f32> = (0..symbols)
                .map(|index| if index % 3 == 0 { -4.0 } else { 4.0 })
                .collect();
            for style in BOOK_STYLES {
                book_weights(&kelly, BookSpec::new(style, 4.0), &mut weights);
                let gross: f64 = weights.iter().map(|w| w.abs()).sum();
                assert!(
                    (gross - 4.0).abs() < 1.0e-9,
                    "{} book at {symbols} names has gross {gross}, not 4",
                    style.label()
                );
            }
            book_weights(
                &kelly,
                BookSpec::new(BookStyle::DollarNeutral, 4.0),
                &mut weights,
            );
            assert!(
                weights.iter().sum::<f64>().abs() < 1.0e-9,
                "the dollar-neutral book must be dollar neutral"
            );
        }
        // Degenerate signals produce a zero book, which is a real answer.
        book_weights(
            &[0.0, 0.0, 0.0],
            BookSpec::new(BookStyle::Signed, 4.0),
            &mut weights,
        );
        assert!(weights.iter().all(|w| *w == 0.0));
        book_weights(
            &[-1.0, -2.0],
            BookSpec::new(BookStyle::LongOnly, 4.0),
            &mut weights,
        );
        assert!(weights.iter().all(|w| *w == 0.0));
        book_weights(
            &[3.0, 3.0, 3.0],
            BookSpec::new(BookStyle::DollarNeutral, 4.0),
            &mut weights,
        );
        assert!(weights.iter().all(|w| *w == 0.0));
        // A non-finite forecast is EXCLUDED, not substituted with zero: it takes no position, the
        // remaining names keep the full gross exposure, and the dollar-neutral book stays neutral
        // across exactly the names that carried a forecast.
        for style in BOOK_STYLES {
            book_weights(
                &[f32::NAN, 1.0, 3.0],
                BookSpec::new(style, 2.0),
                &mut weights,
            );
            assert_eq!(weights[0], 0.0, "{} gave a NaN forecast a position", style.label());
            assert!(
                (weights.iter().map(|w| w.abs()).sum::<f64>() - 2.0).abs() < 1.0e-9,
                "{} lost its gross exposure to a NaN",
                style.label()
            );
        }
        book_weights(
            &[f32::NAN, 1.0, 3.0],
            BookSpec::new(BookStyle::DollarNeutral, 2.0),
            &mut weights,
        );
        assert!(
            weights.iter().sum::<f64>().abs() < 1.0e-9,
            "excluding the NaN must leave the book dollar neutral over the named symbols, got \
             {weights:?}"
        );
    }

    /// Adjacent-bar windows straddling an overnight gap are SKIPPED, not adjusted. Corwin &
    /// Schultz devote a section to why: a gap inflates the two-period range and destroys the
    /// estimate.
    #[test]
    fn windows_straddling_a_session_gap_are_skipped() {
        let bars = synthetic_bars(SESSION_BARS * 20, 0.0020, 0.0010, 50.0, 64, 0x6A9);
        let measured = SymbolCost::measure("GAP", &bars, RES_SECS);
        // 20 sessions of 78 bars: 77 contiguous windows each, so 1540 rather than 1559.
        assert_eq!(measured.pooled.pairs, 20 * (SESSION_BARS as u64 - 1));
        assert_eq!(measured.pooled.sessions, 20);
        assert_eq!(measured.pooled.bars, SESSION_BARS as u64 * 20);

        // Bars per session drives the volatility scaling, so it has to be MEASURED, not assumed.
        // Checked on a bounce-free tape, where the close is the efficient price exactly and the
        // session scaling is the only thing left to get wrong.
        let per_session = 0.0010 * (SESSION_BARS as f64).sqrt();
        let clean = SymbolCost::measure(
            "CLEAN",
            &synthetic_bars(SESSION_BARS * 20, 0.0, 0.0010, 50.0, 64, 0x6A9),
            RES_SECS,
        );
        assert!(
            (clean.pooled.sigma_daily - per_session).abs() / per_session < 0.10,
            "daily sigma {} should scale a 10 bps bar to a {SESSION_BARS}-bar session \
             ({per_session})",
            clean.pooled.sigma_daily
        );

        // On the spread-bearing tape the SAME estimator reads high, and that is correct, not a
        // bug: close-to-close returns carry the bid-ask bounce, so their variance is
        // `sigma^2 + 2*(S/2)^2` — here `sigma = S/2 = 10` bps, giving an inflation of exactly
        // `sqrt(3) = 1.73x`. It is asserted rather than hidden because `sigma_daily` feeds the
        // impact coefficient, so this is a real upward bias in the impact estimate and therefore
        // a CONSERVATIVE one for a capacity claim. Roll (1984) is the same identity read the
        // other way: the bounce is what a spread estimator extracts.
        let inflation = measured.pooled.sigma_daily / clean.pooled.sigma_daily;
        let bounce = 3.0f64.sqrt();
        assert!(
            (inflation - bounce).abs() / bounce < 0.15,
            "the bid-ask bounce should inflate close-to-close volatility by sqrt(3) = {bounce}, \
             measured {inflation}"
        );
    }

    /// Liquidity is measured PER MONTH, so a symbol whose ADV collapses is priced at the liquidity
    /// of the month being traded rather than at a five-year average. This is what makes `ts_ms` on
    /// [`CostModel::cost_bps`] load-bearing rather than decorative.
    #[test]
    fn cost_tracks_the_month_being_traded_not_the_span_average() {
        // 40 sessions from 2024-01-02: sessions 0..29 are January, 30..39 February. The volume
        // regime changes exactly at the month boundary so the test is about the bucketing and not
        // about where the split landed.
        let mut bars = synthetic_bars(SESSION_BARS * 40, 0.0020, 0.0010, 50.0, 64, 0x3F0);
        for (index, bar) in bars.iter_mut().enumerate() {
            bar.volume = if index / SESSION_BARS < 30 { 1.0e6 } else { 1.0e4 };
        }
        let measured = SymbolCost::measure("FADE", &bars, RES_SECS);
        assert_eq!(
            measured.buckets.len(),
            2,
            "the fixture must span exactly two calendar months"
        );
        let last_ts = bars[bars.len() - 1].ts_ms;
        let series = vec![("FADE".to_owned(), bars.clone())];
        let model = BarCostModel::new(calibration_of(&series));
        let early = model.resolve(0, bars[0].ts_ms);
        let late = model.resolve(0, last_ts);
        assert!(
            early.adv_usd > 10.0 * late.adv_usd,
            "the month-resolved ADV must follow the tape: {} then {}",
            early.adv_usd,
            late.adv_usd
        );
        // Same DOLLAR order, different month, materially different cost — which is the whole
        // point of resolving liquidity per month.
        let dollars = 1.0e6;
        let early_bps = early.total_bps(dollars / early.adv_usd);
        let late_bps = late.total_bps(dollars / late.adv_usd);
        assert!(
            late_bps > early_bps + 1.0,
            "a $1M order in the thin month must cost visibly more: {late_bps:.3} vs \
             {early_bps:.3} bps"
        );
        // A query past the corpus end resolves to the NEAREST measured month, not the span.
        let beyond = model.resolve(0, last_ts + 400 * 86_400_000);
        assert!((beyond.adv_usd - late.adv_usd).abs() / late.adv_usd < 1.0e-9);
    }

    /// The corpus-facing path, on a synthetic on-disk corpus rather than the real 451M-bar one.
    ///
    /// [`corpus_panel`] and [`CostCalibration::from_corpus`] are what produce every real-data
    /// number this module reports, and neither is exercised by any in-memory fixture. Three claims
    /// are checked: the parallel corpus pass agrees EXACTLY with the single-threaded
    /// [`CostCalibration::from_series`] on the same bars, a symbol with a hole in the span is
    /// excluded from the panel rather than zero-filled, and a capped panel keeps the most heavily
    /// TRADED names rather than an alphabetic prefix.
    #[test]
    fn a_corpus_on_disk_calibrates_and_yields_a_complete_panel() {
        use shared::bars::write_bar_file;

        let dir = scratch_dir("corpus");
        let symbols = 12usize;
        let sessions = 8usize;
        let from_ms = EPOCH_MS + 3 * 86_400_000;
        let to_ms = EPOCH_MS + 6 * 86_400_000;
        let mut written: Vec<(String, Vec<PackedBar>)> = Vec::new();
        for index in 0..symbols {
            let name = format!("S{index:02}");
            let mut bars = synthetic_bars(
                SESSION_BARS * sessions,
                0.0010,
                0.0010,
                20.0 + 10.0 * index as f64,
                32,
                4_000 + index as u64,
            );
            for bar in bars.iter_mut() {
                bar.volume = 1.0e4 * (index as f32 + 1.0);
            }
            // Symbol 0 loses the whole of session 4, which sits inside the span, so it cannot be
            // complete on the retained grid.
            if index == 0 {
                bars.retain(|bar| {
                    let session = (bar.ts_ms - EPOCH_MS) / 86_400_000;
                    session != 4
                });
            }
            write_bar_file(&dir.join(format!("{name}.{RES_SECS}.bars")), &name, RES_SECS, &bars)
                .expect("the bar file writes");
            written.push((name, bars));
        }

        let corpus =
            crate::torch::dataset::BarCorpus::load_with_bounds(&dir, RES_SECS, 200, (from_ms, to_ms))
                .expect("the synthetic corpus loads");
        assert_eq!(corpus.series_count(), symbols);

        // The parallel corpus pass and the single-threaded series pass are the same measurement.
        let parallel = CostCalibration::from_corpus(&corpus, 3).expect("the corpus calibrates");
        let ordered: Vec<(String, &[PackedBar])> = (0..corpus.series_count())
            .map(|series| {
                (
                    corpus.symbol(series).to_owned(),
                    corpus.bars(series),
                )
            })
            .collect();
        let serial = CostCalibration::from_series(&ordered, RES_SECS).expect("the series calibrate");
        assert_eq!(parallel.len(), serial.len());
        assert_eq!(parallel.unmeasured, serial.unmeasured);
        for (index, (a, b)) in parallel.symbols.iter().zip(&serial.symbols).enumerate() {
            assert_eq!(a.symbol, b.symbol, "symbol {index} is out of order");
            assert_eq!(
                a.pooled.cs_spread_bps.to_bits(),
                b.pooled.cs_spread_bps.to_bits(),
                "{}: the parallel pass measured a different spread",
                a.symbol
            );
            assert_eq!(a.pooled.adv_usd.to_bits(), b.pooled.adv_usd.to_bits());
            assert_eq!(a.buckets.len(), b.buckets.len());
        }

        let cap = 8usize;
        let (slices, panel) =
            corpus_panel(&corpus, from_ms, to_ms, cap, 64).expect("the panel builds");
        assert!(slices.len() <= 64 && slices.len() >= 2);
        assert_eq!(panel.len(), cap);
        assert!(
            !panel.contains(&0),
            "the symbol with a hole inside the span must be excluded, got {panel:?}"
        );
        // Ranked by traded dollars, so the cap keeps the top `cap` volumes: symbols 4..11.
        assert_eq!(panel, (symbols as u32 - cap as u32..symbols as u32).collect::<Vec<u32>>());
        for slice in &slices {
            assert_eq!(slice.symbols, panel);
            assert_eq!(slice.realized_r.len(), panel.len());
            assert!(slice.realized_r.iter().all(|r| r.is_finite()));
            assert!(slice.ts_ms >= from_ms && slice.ts_ms < to_ms);
        }
        for pair in slices.windows(2) {
            assert!(pair[1].ts_ms > pair[0].ts_ms, "the grid must be time ordered");
        }

        // And the whole diagnostic runs on it, which is what the ignored real-corpus test does at
        // scale.
        let model = BarCostModel::new(Arc::new(parallel));
        let forecasts = scenario_forecasts(&slices, 0.6, 0xC0DE);
        let correlation = cross_correlation(
            &slices,
            &forecasts,
            CorrelationConfig {
                gross_leverage: 4.0,
                ..CorrelationConfig::default()
            },
        )
        .expect("the cross-section measures");
        assert_eq!(correlation.panel_symbols, cap);
        let curve = capacity_curve(
            &slices,
            &forecasts,
            &model,
            BookSpec::new(BookStyle::LongOnly, 4.0),
            &AUM_GRID,
        )
        .expect("the capacity curve computes");
        assert_eq!(curve.unpriced_impact_legs, 0);
        assert!(curve.points.iter().all(|p| p.net_bps.is_finite()));
        let _ = fs::remove_dir_all(&dir);
    }

    /// The writer named in `pretrain_reports::CYCLE_EXEMPT` for all three of this module's bases.
    ///
    /// Executes the whole path — measure a synthetic universe, build the book, sweep capacity at
    /// every impact coefficient, measure the cross-section, write the charts — and reads each
    /// artifact back. An exemption whose writer is never executed is what let a registered base
    /// with no writer at all ship, so this test is the exemption's entire justification.
    #[test]
    fn the_cost_capacity_battery_writes_all_three_registered_bases() {
        let symbols = 24usize;
        let series: Vec<(String, Vec<PackedBar>)> = (0..symbols)
            .map(|index| {
                let fraction = index as f64 / (symbols - 1) as f64;
                let mut bars = synthetic_bars(
                    4_000,
                    0.0004 + 0.0030 * (1.0 - fraction),
                    0.0010,
                    10.0 + 100.0 * fraction,
                    64,
                    700 + index as u64,
                );
                for bar in bars.iter_mut() {
                    bar.volume = (1.0e4 + 1.0e6 * fraction) as f32;
                }
                (format!("B{index:02}"), bars)
            })
            .collect();
        let calibration = calibration_of(&series);
        let unmeasured_symbols = calibration.unmeasured.len();
        let universe = calibration.len();
        let model = BarCostModel::new(calibration);

        let betas = vec![1.0f64; symbols];
        let slices = factor_panel(symbols, 2_048, 0.35, 0.002, &betas, 0xB007);
        let forecasts = scenario_forecasts(&slices, 0.595, 0xB008);
        let spec = BookSpec::new(BookStyle::LongOnly, 4.0);
        let capacity = IMPACT_K_GRID
            .iter()
            .map(|&k| {
                capacity_curve(&slices, &forecasts, &model.with_impact_k(k), spec, &AUM_GRID)
                    .expect("the capacity curve computes")
            })
            .collect::<Vec<_>>();
        let correlation = cross_correlation(
            &slices,
            &forecasts,
            CorrelationConfig {
                gross_leverage: 4.0,
                ..CorrelationConfig::default()
            },
        )
        .expect("the cross-section measures");
        let report = CostCapacityReport {
            deciles: model.deciles(),
            capacity,
            correlation,
            universe,
            unmeasured_symbols,
        };

        let dir = scratch_dir("battery");
        write_cost_capacity_reports(&dir, &report, "fixture").expect("the battery writes");
        for base in [COST_DECILE_BASE, CAPACITY_CURVE_BASE, CROSS_CORRELATION_BASE] {
            assert!(
                shared::report::PRETRAIN_REPORT_BASES.contains(&base),
                "{base} must be registered in shared::report::PRETRAIN_REPORT_BASES or the TUI \
                 never scans for it"
            );
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was not written");
            let read = read_report(&path).expect("the report reads back");
            let lines = read.kind.to_lines();
            assert!(!lines.is_empty(), "{base} rendered no rows");
            assert!(
                lines.iter().any(|line| line.contains('=')),
                "{base} rendered no labelled values"
            );
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

    /// The MATCHED comparison: cost of exactly the symbols a break-even was measured on.
    ///
    /// Whether a liquidity restriction and a quietness restriction are ONE restriction.
    ///
    /// A capped policy cannot express confidence through size, so the only lever it has is trading
    /// fewer bars, and two candidate filters were proposed: the deep end of the liquidity ranking
    /// (cost side, this module) and the top of a `mu_hat / sigma_hat` ranking (skill side). Their
    /// product is only a product if they select different names. Both failure directions are real:
    /// heavy overlap double-counts one selection, and heavy anti-overlap starves the intersection
    /// until it cannot resolve.
    ///
    /// Rank correlation rather than a comparison of per-decile means, because the selection happens
    /// name by name and a monotone table of means is compatible with a nearly random name ordering.
    /// The intersection count is reported beside its independence expectation AND that
    /// expectation's own sampling sd, because the correlation alone does not say how many names
    /// survive both filters and a bare count does not say whether it could have resolved. Drawing
    /// `deep_n` of `pairs` names and asking how many fall in a tail of `tail_n` is exactly
    /// HYPERGEOMETRIC, so the sd is closed-form rather than assumed - and at a tenth-sized tail
    /// against a 43-name draw even a count of ZERO reaches only p = 0.008, which is why the tail
    /// width is a parameter and both a tenth and a quartile are reported.
    ///
    /// Non-finite members are DROPPED rather than ordered, and `pairs` reports how many remained: a
    /// name with no measurable volume or no measurable volatility has no position in either ranking,
    /// and giving it one would invent the very ordering being measured.
    struct Overlap {
        pairs: usize,
        rho: f64,
        /// Width of EACH volatility tail, in names. A parameter rather than a fixed tenth, because
        /// a tenth-sized cut cannot resolve depletion on a 256-name panel at ANY effect size.
        tail_n: usize,
        deep_n: usize,
        /// Deepest-liquidity names that are also in the QUIETEST tail.
        intersection: usize,
        /// Deepest-liquidity names that are also in the LOUDEST tail. Both tails share
        /// [`Self::expected`], which is symmetric in the tail by construction, so whichever tail a
        /// candidate selector turns out to occupy the comparison is already here.
        loud_intersection: usize,
        expected: f64,
        /// Hypergeometric sd of [`Self::expected`]: `sqrt(n k (1 - k) (N - n) / (N - 1))` with
        /// `k = tail_n / N`. Reported beside every count so a cut that COULD NOT have resolved is
        /// never read as one that did - the failure this field exists to prevent is a bare `1`
        /// against an expected `4.2` reading as depletion when even `0` would not have reached
        /// two sigma.
        sd: f64,
        /// EXACT one-sided hypergeometric `P(X <= observed)` for the quiet and loud counts. Not a
        /// normal approximation of the z beside them: at the counts this panel produces the normal
        /// tail is wrong by ~30% in the direction that MANUFACTURES significance - the loud quartile
        /// cell reads -2.22 sigma, which a normal table converts to 0.013 while the exact figure is
        /// 0.017. A z is for eyeballing; these are the numbers that may be quoted.
        p_quiet: f64,
        p_loud: f64,
    }

    /// EXACT one-sided `P(X <= observed)` for drawing `draw` of `total` names and counting how many
    /// land in a tail of `tail`.
    ///
    /// Summed in LOG space over a table of log-factorials, because the binomial coefficients here
    /// reach `C(256, 43) ~ 1e48` and the ratio of two overflowing terms is not recoverable once
    /// either has overflowed. The table is built once per call at `total + 1` entries, which is
    /// nothing beside the corpus pass that produces the counts.
    ///
    /// A normal approximation was the alternative and is not adequate: the counts this panel yields
    /// sit 1-2 sd below a mean near 10, far from where the continuous tail is accurate, and it errs
    /// by ~30% in the direction that OVERSTATES significance.
    fn hypergeometric_at_most(total: usize, draw: usize, tail: usize, observed: usize) -> f64 {
        if total == 0 || draw > total || tail > total {
            return f64::NAN;
        }
        let mut ln_fact = vec![0.0f64; total + 1];
        for k in 1..=total {
            ln_fact[k] = ln_fact[k - 1] + (k as f64).ln();
        }
        let ln_choose = |n: usize, k: usize| ln_fact[n] - ln_fact[k] - ln_fact[n - k];
        let ln_total = ln_choose(total, draw);
        let mut sum = 0.0f64;
        // `i` ranges over the counts that are actually reachable: at most the tail's own size, and
        // no fewer than the draw leaves outside it.
        let lowest = draw.saturating_sub(total - tail);
        for i in lowest..=observed.min(tail).min(draw) {
            sum += (ln_choose(tail, i) + ln_choose(total - tail, draw - i) - ln_total).exp();
        }
        sum.min(1.0)
    }

    /// Midranks, so tied volumes or tied volatilities cannot manufacture an ordering.
    fn midranks(values: &[f64]) -> Vec<f64> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .expect("callers drop non-finite values before ranking")
        });
        let mut out = vec![0.0f64; values.len()];
        let mut start = 0usize;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && values[order[end]] == values[order[start]] {
                end += 1;
            }
            // 1-based average rank of the tied run.
            let shared = (start + end - 1) as f64 / 2.0 + 1.0;
            for &index in &order[start..end] {
                out[index] = shared;
            }
            start = end;
        }
        out
    }

    /// `pairs` are `(dollar ADV, realized sigma)`; `deep_n` is the size of the deepest-liquidity
    /// decile of the same set, so the intersection is taken against exactly the sub-book a peer
    /// would restrict to rather than against a boundary re-derived here. `tail_fraction` sets each
    /// volatility tail's width - call it at several widths and print them together, because the
    /// narrow cut is the question originally asked and the wide cut is the one that can be answered.
    fn liquidity_volatility_overlap(
        pairs: &[(f64, f64)],
        deep_n: usize,
        tail_fraction: f64,
    ) -> Overlap {
        let kept: Vec<(f64, f64)> = pairs
            .iter()
            .copied()
            .filter(|&(adv, sigma)| adv.is_finite() && sigma.is_finite())
            .collect();
        let n = kept.len();
        if n < 2 {
            return Overlap {
                pairs: n,
                rho: f64::NAN,
                tail_n: 0,
                deep_n: 0,
                intersection: 0,
                loud_intersection: 0,
                expected: f64::NAN,
                sd: f64::NAN,
                p_quiet: f64::NAN,
                p_loud: f64::NAN,
            };
        }
        let advs: Vec<f64> = kept.iter().map(|&(adv, _)| adv).collect();
        let sigmas: Vec<f64> = kept.iter().map(|&(_, sigma)| sigma).collect();
        let adv_rank = midranks(&advs);
        let sigma_rank = midranks(&sigmas);
        let mean = (n as f64 + 1.0) / 2.0;
        let mut cov = 0.0f64;
        let mut var_a = 0.0f64;
        let mut var_s = 0.0f64;
        for index in 0..n {
            let a = adv_rank[index] - mean;
            let s = sigma_rank[index] - mean;
            cov += a * s;
            var_a += a * a;
            var_s += s * s;
        }
        // A degenerate ranking - every volume or every volatility identical - has no correlation
        // rather than a zero one, and must not read as evidence of independence.
        let rho = if var_a > 0.0 && var_s > 0.0 {
            cov / (var_a * var_s).sqrt()
        } else {
            f64::NAN
        };
        // The deepest-liquidity members ARE the highest-ADV members: decile membership is assigned
        // by dollar-ADV rank, so taking the top `deep_n` here reproduces that set exactly, clamped
        // in case dropped members shrank the ranking below it.
        let deep_n = deep_n.min(n);
        let tail_n = ((n as f64 * tail_fraction).round() as usize).clamp(1, n);
        let deep_cut = {
            let mut sorted = advs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            sorted[n - deep_n]
        };
        let mut by_sigma = sigmas.clone();
        by_sigma.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let quiet_cut = by_sigma[tail_n - 1];
        // The LOUD tail as well, because which tail a confidence selector occupies is a measured
        // fact and not ours to assume. `SkillAudit` measured the `|mu_hat| / sigma_hat` top decile
        // at 1.46x the sigma_hat of its bottom decile - it lands on the LOUD names, because
        // `|mu_hat|` alone separates 14.9x and swamps the ratio's denominator. Reporting only the
        // quiet side would answer a question nobody asked, on the wrong population.
        let loud_cut = by_sigma[n - tail_n];
        let intersection = kept
            .iter()
            .filter(|&&(adv, sigma)| adv >= deep_cut && sigma <= quiet_cut)
            .count();
        let loud_intersection = kept
            .iter()
            .filter(|&&(adv, sigma)| adv >= deep_cut && sigma >= loud_cut)
            .count();
        // Drawing `deep_n` of `n` names and counting how many land in a tail of `tail_n` is exactly
        // a hypergeometric draw, so both moments are closed-form. The sd is what says whether a
        // count BELOW expectation is depletion or is the cut's own noise.
        let share = tail_n as f64 / n as f64;
        let expected = deep_n as f64 * share;
        let sd = if n > 1 {
            (expected * (1.0 - share) * (n - deep_n) as f64 / (n as f64 - 1.0)).sqrt()
        } else {
            f64::NAN
        };
        Overlap {
            pairs: n,
            rho,
            tail_n,
            deep_n,
            intersection,
            loud_intersection,
            expected,
            sd,
            p_quiet: hypergeometric_at_most(n, deep_n, tail_n, intersection),
            p_loud: hypergeometric_at_most(n, deep_n, tail_n, loud_intersection),
        }
    }

    /// The overlap statistic recovers a KNOWN ordering, a known intersection, and admits ties.
    ///
    /// Every number this reports is used to decide whether two proposed trade filters are one
    /// filter, and both failure directions look like a plausible result rather than a bug: a sign
    /// error turns "the liquid names are the quiet ones" into its opposite, and an off-by-one in a
    /// tail cut turns an independence verdict into a double-counting one. So the recovery is
    /// asserted against constructions whose answers are known by hand rather than by this code.
    #[test]
    fn the_overlap_statistic_recovers_a_known_ordering_and_intersection() {
        // Perfectly anti-monotone: the highest volume is the quietest name. Spearman is exactly -1,
        // and the quietest tenth then sits ENTIRELY inside the deepest decile.
        let anti: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, (100 - i) as f64)).collect();
        let measured = liquidity_volatility_overlap(&anti, 10, 0.1);
        assert!(
            (measured.rho + 1.0).abs() < 1.0e-12,
            "a perfectly anti-monotone panel must read rho = -1, got {}",
            measured.rho
        );
        assert_eq!(measured.tail_n, 10, "a hundred names give a tenth of ten");
        assert_eq!(
            measured.intersection, 10,
            "under perfect anti-correlation the quietest tenth IS the deepest decile"
        );
        // The two tails must be reported independently and must not be the same number. On this
        // construction the loud tail is the THIN end, so its intersection with the deep decile is
        // empty - and a helper that computed one tail and reported it twice would pass every
        // assertion above while answering the wrong question, which is the failure this pins.
        assert_eq!(
            measured.loud_intersection, 0,
            "under perfect anti-correlation the loudest tenth is disjoint from the deepest decile"
        );
        assert!(
            (measured.expected - 1.0).abs() < 1.0e-12,
            "ten of a hundred crossed with ten expects one, got {}",
            measured.expected
        );
        // The sd of that expectation, against the closed form worked by hand: drawing 10 of 100
        // names into a tail of 10 gives sqrt(10 * 0.1 * 0.9 * 90 / 99) = 0.904534. Without this the
        // counts above are unreadable - a `1` against an expected `4.2` on the real panel is
        // depletion or noise depending ENTIRELY on this number, and it read as depletion for forty
        // minutes because nothing reported it.
        assert!(
            (measured.sd - 0.904534).abs() < 1.0e-6,
            "hypergeometric sd must match the closed form, got {}",
            measured.sd
        );

        // Perfectly monotone: the highest volume is the loudest name, so the two tails are
        // DISJOINT. Independence expects one name and the measurement finds none - the starvation
        // direction, which has to be distinguishable from the double-counting one.
        let monotone: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, i as f64)).collect();
        let measured = liquidity_volatility_overlap(&monotone, 10, 0.1);
        assert!(
            (measured.rho - 1.0).abs() < 1.0e-12,
            "a perfectly monotone panel must read rho = +1, got {}",
            measured.rho
        );
        assert_eq!(
            measured.intersection, 0,
            "under perfect correlation the quietest tenth and the deepest decile are disjoint"
        );
        // And the mirror: the loud tail is now FULLY inside the deep decile. Together with the
        // anti-monotone case this proves the two counters move in opposite directions, so neither
        // can be silently reading the other's cut.
        assert_eq!(
            measured.loud_intersection, 10,
            "under perfect correlation the loudest tenth IS the deepest decile"
        );
        // WIDENING THE TAIL CHANGES THE ESTIMAND AND MUST CHANGE THE ARITHMETIC WITH IT. The same
        // panel cut at a quartile still contains all ten deep names in the loud tail, but expects
        // 2.5 of them rather than 1.0 - so the RATIO to independence falls from 10.0x to 4.0x while
        // the count is unchanged. A widened cut that forgot to widen its expectation would inflate
        // every ratio it reported, which is the specific way a resolution fix can manufacture the
        // effect it was introduced to measure honestly.
        let wide = liquidity_volatility_overlap(&monotone, 10, 0.25);
        assert_eq!(wide.tail_n, 25, "a quartile of a hundred names is twenty-five");
        assert_eq!(
            wide.loud_intersection, 10,
            "all ten deep names stay inside a widened loud tail"
        );
        assert!(
            (wide.expected - 2.5).abs() < 1.0e-12,
            "widening the tail must widen the expectation, got {}",
            wide.expected
        );
        // And the sd rises with it: sqrt(10 * 0.25 * 0.75 * 90 / 99) = 1.3055824. The whole point of
        // the wider cut is that an effect of fixed RELATIVE size clears more sigma, which is only
        // true if both moments track the width.
        assert!(
            (wide.sd - 1.3055824).abs() < 1.0e-6,
            "the sd must track the tail width, got {}",
            wide.sd
        );

        // THE EXACT TAIL, against values computed independently of this code. The panel's own
        // geometry: 256 names, a 43-name draw, and a tail of 25, 26, 43 or 64. These four are the
        // cells the real measurement lands in, so an error in the log-factorial sum would be
        // invisible in every synthetic fixture and would surface only as a mis-quoted p.
        for (tail, observed, expect) in [
            (25usize, 1usize, 0.052_3f64),
            (26, 2, 0.149_2),
            (43, 4, 0.107_7),
            (64, 5, 0.017_0),
        ] {
            let p = hypergeometric_at_most(256, 43, tail, observed);
            assert!(
                (p - expect).abs() < 5.0e-4,
                "exact hypergeometric P(X<={observed}) for a {tail}-name tail must be {expect}, got {p}"
            );
        }
        // AND THE POWER CEILING, WHICH IS THE REASON THE TAIL WIDTH IS A PARAMETER AT ALL: at a
        // tenth-sized tail even an observed count of ZERO reaches only p = 0.0078, so no effect size
        // whatsoever resolves at two sigma there. Widening to a quartile drops that floor to
        // 1.095e-6, nearly four orders down. This asserts the CEILING rather than any measured
        // value, so it stands even if every count on the real panel changes.
        let floor_tenth = hypergeometric_at_most(256, 43, 25, 0);
        assert!(
            floor_tenth > 0.0078 && floor_tenth < 0.0079,
            "a tenth-sized tail cannot resolve depletion at any effect size, floor was {floor_tenth}"
        );
        let floor_quartile = hypergeometric_at_most(256, 43, 64, 0);
        assert!(
            (floor_quartile - 1.0955e-6).abs() < 1.0e-9,
            "a quartile-sized tail must resolve total depletion, floor was {floor_quartile}"
        );
        // A total that IS the draw has no randomness left: every name is drawn, so the count is the
        // tail size with probability one and `P(X <= tail)` is exactly 1. The reachable-range floor
        // exists for this case - summing from zero would count impossible outcomes.
        assert!(
            (hypergeometric_at_most(10, 10, 4, 4) - 1.0).abs() < 1.0e-12,
            "drawing every name puts the whole tail in the sample with certainty"
        );
        assert!(
            hypergeometric_at_most(10, 10, 4, 3) < 1.0e-12,
            "drawing every name cannot yield fewer tail members than the tail holds"
        );

        // A DEGENERATE axis has no correlation rather than a zero one. Reading NaN as 0.0 here would
        // report "independent" for a panel carrying no volatility information at all, which is the
        // unmeasured-as-measured error that has cost this session repeatedly.
        let flat: Vec<(f64, f64)> = (0..40).map(|i| (i as f64, 7.0)).collect();
        let measured = liquidity_volatility_overlap(&flat, 4, 0.1);
        assert!(
            measured.rho.is_nan(),
            "a panel with one volatility value has no rank correlation, got {}",
            measured.rho
        );
        assert_eq!(measured.pairs, 40, "a tied axis is not a dropped axis");

        // Non-finite members are dropped from the ranking rather than ordered into it, and `pairs`
        // says so, because an unmeasurable volume has no place in a volume ranking.
        let mut holed = anti.clone();
        holed.push((f64::NAN, 1.0));
        holed.push((5.0, f64::NAN));
        let measured = liquidity_volatility_overlap(&holed, 10, 0.1);
        assert_eq!(
            measured.pairs, 100,
            "two unmeasurable members must leave the ranking at its measured size"
        );
        assert!(
            (measured.rho + 1.0).abs() < 1.0e-12,
            "dropping unmeasurable members must not perturb the recovered ordering, got {}",
            measured.rho
        );

        // Fewer than two measured pairs is not a correlation of zero.
        let single = [(1.0f64, 2.0f64)];
        assert!(
            liquidity_volatility_overlap(&single, 1, 0.1).rho.is_nan(),
            "one name cannot carry a rank correlation"
        );
    }

    /// A break-even measured on 256 pinned val windows and a cost measured over all 5,297 symbols
    /// are different universes, and pairing them is a category error in an unknown direction: a
    /// traded subset selected on anything liquidity-correlated sits in the DEEPER deciles, so a
    /// universe-wide median overstates its cost, while quoting the deepest decile understates what
    /// a wider deployment would pay. This prices the traded set against the deciles it actually
    /// occupies, which is the only version of the comparison that means anything.
    ///
    /// Reads `pretrain_calibration_windows.json` - written by `pretrain-calibration`, listing the
    /// traded and the block-disjoint fit slices row by row. The path is overridable with
    /// `TB_CALIBRATION_WINDOWS` because the artifact belongs to a RUN rather than to the corpus;
    /// the default is the run this session measured. `#[ignore]`d for the same reason as the
    /// battery above: it calibrates the whole corpus. Run it with the thread-pinned invocation
    /// documented on `the_real_corpus_prices_itself_and_reports_its_own_over_levering_factor` -
    /// `OMP_NUM_THREADS=1` is mandatory, not advisory.
    #[test]
    #[ignore = "reads the real corpus and a run artifact: minutes of CPU"]
    fn the_traded_window_set_is_priced_against_the_deciles_it_occupies() {
        use crate::data::ingest::{bars_dir, PINNED_SPLIT_BOUNDS};
        use crate::torch::dataset::{BarCorpus, DEFAULT_MIN_BARS};
        use std::collections::{BTreeMap, BTreeSet};
        // The published constants this measurement is the provenance of, re-derived below so a
        // drift between the literal and the corpus prints as a disagreement instead of silently
        // staying authoritative in a doc comment.
        use crate::torch::train::horizon::{
            MATCHED_ALL_IN_BPS, MATCHED_DEEPEST_DECILE_BPS, MATCHED_MEASURED_BPS,
            UNIVERSE_MEASURED_BPS,
        };

        const DEFAULT_WINDOWS: &str =
            "training/runs/bardist_v2/gens/2/pretrain_calibration_windows.json";
        let path = std::env::var("TB_CALIBRATION_WINDOWS").unwrap_or_else(|_| {
            std::env::var("CARGO_MANIFEST_DIR")
                .map(|dir| {
                    Path::new(&dir)
                        .parent()
                        .unwrap_or(Path::new("."))
                        .join(DEFAULT_WINDOWS)
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|_| DEFAULT_WINDOWS.to_owned())
        });
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading the window artifact {path}: {error}"));
        // Deliberately not a serde type: the artifact belongs to another module and pinning its
        // full schema here would couple two agents' files over a debugging aid. Only the two fields
        // this measurement needs are extracted.
        let parsed: serde_json::Value =
            serde_json::from_str(&text).expect("the window artifact is JSON");
        let traded = parsed["traded"]
            .as_array()
            .expect("the artifact has a `traded` array");
        let wanted: HashSet<String> = traded
            .iter()
            .map(|row| {
                row["symbol"]
                    .as_str()
                    .expect("every traded row names a symbol")
                    .to_owned()
            })
            .collect();
        let anchor_of: HashMap<String, i64> = traded
            .iter()
            .map(|row| {
                (
                    row["symbol"].as_str().expect("symbol").to_owned(),
                    row["ts_ms"].as_i64().expect("ts_ms"),
                )
            })
            .collect();
        // The FIT slice draws a DIFFERENT 256 names than the traded slice, so a fit-slice argmax
        // has to be priced on the fit slice's own symbols. `wanted` stays the TRADED set because
        // the four published constants are equal-weighted means over exactly those names; `priced`
        // is the union and is what the join artifact covers. Absent `fit`, the two are equal.
        let fit = parsed["fit"].as_array();
        let priced: HashSet<String> = wanted
            .iter()
            .cloned()
            .chain(fit.into_iter().flatten().filter_map(|row| {
                row["symbol"].as_str().map(str::to_owned)
            }))
            .collect();
        // Anchor month per name, traded taking precedence where a name is in both slices: the
        // traded windows are the ones the published constants were measured on.
        let anchor_of: HashMap<String, i64> = fit
            .into_iter()
            .flatten()
            .filter_map(|row| {
                Some((row["symbol"].as_str()?.to_owned(), row["ts_ms"].as_i64()?))
            })
            .chain(anchor_of)
            .collect();
        println!(
            "== matched traded set: {} rows, {} distinct symbols, context {} ==",
            traded.len(),
            wanted.len(),
            parsed["context"]
        );

        let corpus =
            BarCorpus::load_with_bounds(&bars_dir(), RES_SECS, DEFAULT_MIN_BARS, PINNED_SPLIT_BOUNDS)
                .expect("the 300s corpus loads");
        let calibration =
            Arc::new(CostCalibration::from_corpus(&corpus, 4).expect("the corpus calibrates"));
        let model = BarCostModel::new(Arc::clone(&calibration));
        // The ranking is over the FULL universe, deliberately: the question is which tenth of the
        // tradeable world these names occupy, not how they rank among themselves.
        let decile_of = model.decile_of_symbol();
        // ADV PERCENTILE over the FULL universe, so a traded name's depth is a position in the
        // tradeable world rather than among its 255 peers. Needed because the composite below has
        // to answer "is the book trading the thin end", and a decile index is too coarse to weight.
        let mut sorted_adv: Vec<f64> = (0..calibration.len() as u32)
            .map(|symbol| model.resolve_pooled(symbol).adv_usd)
            .filter(|adv| adv.is_finite() && *adv > 0.0)
            .collect();
        sorted_adv.sort_unstable_by(f64::total_cmp);
        let adv_percentile = |adv: f64| -> f64 {
            if adv.is_finite() && !sorted_adv.is_empty() {
                sorted_adv.partition_point(|entry| *entry < adv) as f64 / sorted_adv.len() as f64
            } else {
                f64::NAN
            }
        };
        // One row per traded name, written out so the TURNOVER-WEIGHTED composite is a JOIN on
        // `symbol` rather than a re-derivation of anybody's cost. These are exactly the `c_i` that
        // `MATCHED_MEASURED_BPS` is the equal-weighted mean of.
        let mut cost_rows: Vec<serde_json::Value> = Vec::with_capacity(wanted.len());

        let mut found = 0usize;
        let mut histogram = [0usize; DECILES];
        let mut fixed: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut sized: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut month_sized: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut month_fixed: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut month_commission: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut month_half: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut month_price: Vec<f64> = Vec::with_capacity(wanted.len());
        // Span-pooled harmonic price over anchor-month harmonic price, per name. A cumulative
        // split adjustment shows up here as a large ratio and nothing else in this corpus does.
        let mut price_ratio: Vec<f64> = Vec::with_capacity(wanted.len());
        // VOLATILITY per decile, because the cost axis and a Sharpe-selected confidence axis may be
        // the SAME axis. A `mu_hat / sigma_hat` selector lifts quiet instruments into its top decile
        // regardless of forecast quality, and if the most liquid tenth is also the quiet tenth then
        // crossing the two selections double-counts one. Measuring sigma per liquidity decile is the
        // cost side of settling that, and it needs no model forecasts.
        let mut decile_sigma: Vec<Vec<f64>> = (0..DECILES).map(|_| Vec::new()).collect();
        // The matched set BROKEN DOWN by the decile each name occupies. Without this the only
        // liquidity-restricted comparator available is a universe-wide decile median, which is a
        // different set of symbols from the one the break-even was measured on and therefore not a
        // comparator at all. This answers the question that actually decides the strategy: is there
        // a liquidity-restricted sub-book of the traded names whose own cost it could clear?
        let mut decile_fixed: Vec<Vec<f64>> = (0..DECILES).map(|_| Vec::new()).collect();
        let mut decile_sized: Vec<Vec<f64>> = (0..DECILES).map(|_| Vec::new()).collect();
        // The TICKERS, not just the counts: a peer re-measuring the edge on a liquidity-restricted
        // sub-book has to restrict to exactly the names priced here, or the two halves of the
        // comparison are again different populations. Re-deriving the boundary downstream is the
        // failure this list exists to prevent.
        let mut decile_names: Vec<Vec<&str>> = (0..DECILES).map(|_| Vec::new()).collect();
        let mut ranked: Vec<(f64, f64)> = Vec::with_capacity(wanted.len());
        let mut unpriceable = 0usize;
        // COMPONENT DECOMPOSITION of the impact-free figure, name by name, so the headline scalar
        // can be audited term by term rather than trusted whole. `half_spread + commission +
        // regulatory` is `fixed` exactly, by construction of `ResolvedCost::fixed_bps`, so the
        // three vectors below are a partition of the same number over the same population at the
        // same resolution tier and no reweighting hides inside the split.
        let mut half_spread: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut commission: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut regulatory: Vec<f64> = Vec::with_capacity(wanted.len());
        // The ZERO-COMMISSION counterfactual: the same measured figure with the per-share
        // commission set to exactly zero and every other term untouched. Commission is the one
        // component that is a broker's schedule rather than a property of the market, so it is the
        // one whose removal is a question somebody can actually act on.
        let mut fixed_free_commission: Vec<f64> = Vec::with_capacity(wanted.len());
        // Harmonic price per traded name. Commission enters as `rate / price`, so the traded set's
        // PRICE distribution is the whole cross-sectional structure of the commission term and a
        // commission mean quoted without it is uninterpretable.
        let mut prices: Vec<f64> = Vec::with_capacity(wanted.len());
        // What the regulatory leg costs when the FINRA Trading Activity Fee is priced per SHARE at
        // the name's own price instead of at the assumed $70 baked into `REGULATORY_BPS`. Same
        // sidedness convention as that constant - both statutory fees are SELL-side, so both are
        // halved to state them per unit of ONE-WAY turnover.
        let mut regulatory_per_share: Vec<f64> = Vec::with_capacity(wanted.len());
        let mut decile_free_commission: Vec<Vec<f64>> = (0..DECILES).map(|_| Vec::new()).collect();
        let mut decile_commission: Vec<Vec<f64>> = (0..DECILES).map(|_| Vec::new()).collect();
        let mut decile_half_spread: Vec<Vec<f64>> = (0..DECILES).map(|_| Vec::new()).collect();
        let mut decile_price: Vec<Vec<f64>> = (0..DECILES).map(|_| Vec::new()).collect();
        // SEC Section 31: `$27.80` per `$1,000,000` of proceeds, SELL side only, so `0.278` bps of
        // a sale and half that per unit of one-way turnover in a book whose legs balance.
        const SEC_SELL_BPS: f64 = 0.278;
        // FINRA Trading Activity Fee: per SHARE, SELL side only. A per-order cap of `$8.30` exists
        // and is NOT modelled here, so this figure is an upper bound for very large single orders.
        const TAF_SELL_PER_SHARE_USD: f64 = 0.000166;
        let regulatory_priced_per_share = |price: f64| -> f64 {
            if price.is_finite() && price > 0.0 {
                0.5 * (SEC_SELL_BPS + 1.0e4 * TAF_SELL_PER_SHARE_USD / price)
            } else {
                f64::NAN
            }
        };
        let mut fallback = 0usize;
        for symbol in 0..calibration.len() as u32 {
            let name = &calibration.symbols[symbol as usize].symbol;
            if !priced.contains(name) {
                continue;
            }
            let pooled = model.resolve_pooled(symbol);
            let price = calibration.symbols[symbol as usize].pooled.harmonic_price;
            let anchor_fixed = anchor_of
                .get(name)
                .map_or(f64::NAN, |&anchor| model.resolve(symbol, anchor).fixed_bps());
            cost_rows.push(serde_json::json!({
                "symbol": name,
                "half_spread_bps": pooled.half_spread_bps,
                "commission_bps": pooled.commission_bps,
                "regulatory_bps": pooled.regulatory_bps,
                "fixed_bps": pooled.fixed_bps(),
                "fixed_bps_anchor_month": anchor_fixed,
                "all_in_bps_at_1pct_adv": pooled.total_bps(
                    PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT],
                ),
                "harmonic_price_usd": price,
                "adv_usd": pooled.adv_usd,
                "adv_percentile_universe": adv_percentile(pooled.adv_usd),
                "sigma_daily": calibration.symbols[symbol as usize].pooled.sigma_daily,
                "liquidity_decile": decile_of[symbol as usize],
                "spread_fallback": pooled.spread_fallback,
            }));
            // The HEADLINE population is the TRADED set alone: the four published constants are
            // equal-weighted means over exactly those names, so a fit-only name reaches the join
            // without entering any published statistic.
            if !wanted.contains(name) {
                continue;
            }
            found += 1;
            histogram[decile_of[symbol as usize]] += 1;
            if pooled.spread_fallback {
                fallback += 1;
            }
            if !(pooled.impact_coefficient_bps.is_finite() && pooled.impact_coefficient_bps > 0.0) {
                unpriceable += 1;
            }
            fixed.push(pooled.fixed_bps());
            sized.push(pooled.total_bps(PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT]));
            half_spread.push(pooled.half_spread_bps);
            commission.push(pooled.commission_bps);
            regulatory.push(pooled.regulatory_bps);
            fixed_free_commission.push(pooled.half_spread_bps + pooled.regulatory_bps);
            prices.push(price);
            regulatory_per_share.push(regulatory_priced_per_share(price));
            decile_free_commission[decile_of[symbol as usize]]
                .push(pooled.half_spread_bps + pooled.regulatory_bps);
            decile_commission[decile_of[symbol as usize]].push(pooled.commission_bps);
            decile_half_spread[decile_of[symbol as usize]].push(pooled.half_spread_bps);
            decile_price[decile_of[symbol as usize]].push(price);
            decile_fixed[decile_of[symbol as usize]].push(pooled.fixed_bps());
            decile_sized[decile_of[symbol as usize]]
                .push(pooled.total_bps(PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT]));
            decile_names[decile_of[symbol as usize]].push(name.as_str());
            decile_sigma[decile_of[symbol as usize]]
                .push(calibration.symbols[symbol as usize].pooled.sigma_daily);
            // The PAIR, per name, for the overlap question: does selecting on liquidity also select
            // on quietness? A decile table of means can look monotone while the name-level ranking
            // is nearly random, and it is the name-level ranking that decides whether a liquidity
            // restriction and a volatility-flavoured restriction are one restriction or two.
            ranked.push((
                pooled.adv_usd,
                calibration.symbols[symbol as usize].pooled.sigma_daily,
            ));
            // Priced at the window's ANCHOR month rather than the five-year pool, which is the
            // liquidity the trade would actually have met.
            //
            // The IMPACT-FREE figure is resolved here too, and that closes a real gap: the
            // published constant is span-pooled, so it prices five years of liquidity against an
            // edge measured inside a five-month window. It also bounds a units exposure that the
            // sized figure cannot show. The corpus is Polygon data pulled with `adjusted=true`
            // (see `data::historical`), so `close` - and therefore `harmonic_price` - is SPLIT- AND
            // DIVIDEND-ADJUSTED and is NOT the price at which shares changed hands. Commission is
            // charged PER SHARE against that price, so every cumulative split between a bar and the
            // download date rescales the commission of that bar: a forward split OVERSTATES it and
            // a reverse split UNDERSTATES it, by the split factor. The anchor month sits inside the
            // pinned validation span, so its adjustment exposure is months rather than years, and
            // the pooled-against-anchor price RATIO is what makes the size of the effect visible
            // instead of arguable.
            if let Some(&anchor) = anchor_of.get(name) {
                let at_anchor = model.resolve(symbol, anchor);
                month_sized.push(
                    at_anchor.total_bps(PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT]),
                );
                month_fixed.push(at_anchor.fixed_bps());
                month_commission.push(at_anchor.commission_bps);
                month_half.push(at_anchor.half_spread_bps);
                let anchor_price = calibration.symbols[symbol as usize]
                    .bucket_at(anchor)
                    .map_or(f64::NAN, |bucket| bucket.harmonic_price);
                month_price.push(anchor_price);
                if anchor_price.is_finite() && anchor_price > 0.0 && price.is_finite() {
                    price_ratio.push(price / anchor_price);
                }
            }
        }
        assert!(
            found > 0,
            "no traded symbol matched the corpus; the artifact and the corpus disagree"
        );

        // The EQUAL-WEIGHTED MEAN, not the median, is the statistic dimensionally matched to a
        // bar-pooled break-even: the edge is pooled over 256 windows x 896 bars with every bar
        // counting once, and each name contributes exactly one window, so the book holds all 256
        // equally. Cost across this universe is heavily right-skewed - the decile spread runs 19.3
        // to 4.2 bps measured - so the mean sits above the median and it is the mean the book pays.
        // The median is reported beside it as the robustness figure, which inverts the usual
        // convention deliberately: a median summarises a UNIVERSE, a mean summarises a BOOK.
        let mean_of = |values: &[f64]| -> (f64, usize) {
            let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
            let dropped = values.len() - finite.len();
            if finite.is_empty() {
                return (f64::NAN, dropped);
            }
            (finite.iter().sum::<f64>() / finite.len() as f64, dropped)
        };
        // Universe-wide equal weighting, the number neither side had quoted: the decile table is
        // ten medians and cannot be averaged into one.
        let mut universe_fixed: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_sized: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_half: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_commission: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_free_commission: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_price: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_regulatory_per_share: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_adv: Vec<f64> = Vec::with_capacity(calibration.len());
        let mut universe_impact: Vec<f64> = Vec::with_capacity(calibration.len());
        for symbol in 0..calibration.len() as u32 {
            let pooled = model.resolve_pooled(symbol);
            universe_fixed.push(pooled.fixed_bps());
            universe_sized.push(pooled.total_bps(PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT]));
            let price = calibration.symbols[symbol as usize].pooled.harmonic_price;
            universe_half.push(pooled.half_spread_bps);
            universe_commission.push(pooled.commission_bps);
            universe_free_commission.push(pooled.half_spread_bps + pooled.regulatory_bps);
            universe_price.push(price);
            universe_regulatory_per_share.push(regulatory_priced_per_share(price));
            universe_adv.push(pooled.adv_usd);
            universe_impact
                .push(pooled.impact_bps(PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT]));
        }

        let universe = model.deciles();
        println!(
            "matched on {found} of {} named symbols ({} spread-fallback, {} impact-unpriceable)",
            wanted.len(),
            fallback,
            unpriceable
        );
        println!("decile occupancy (0 = thinnest): {histogram:?}");
        println!("== matched cost WITHIN each occupied decile (0 = thinnest) ==");
        for decile in 0..DECILES {
            if decile_fixed[decile].is_empty() {
                println!("decile {decile}: no traded name occupies it");
                continue;
            }
            let (dfixed, dfixed_dropped) = mean_of(&decile_fixed[decile]);
            let (dsized, dsized_dropped) = mean_of(&decile_sized[decile]);
            let (dsigma, dsigma_dropped) = mean_of(&decile_sigma[decile]);
            println!(
                "decile {decile}: n={:3} measured impact-free MEAN {dfixed:8.3} bps (median \
                 {:8.3}, {dfixed_dropped} dropped) | all-in @1% MEAN {dsized:8.3} bps \
                 ({dsized_dropped} dropped) | daily sigma MEAN {dsigma:8.5} (median {:8.5}, \
                 {dsigma_dropped} dropped)",
                decile_fixed[decile].len(),
                median(&mut decile_fixed[decile].clone()),
                median(&mut decile_sigma[decile].clone()),
            );
            let (dhalf, _) = mean_of(&decile_half_spread[decile]);
            let (dcomm, _) = mean_of(&decile_commission[decile]);
            let (dfree, dfree_dropped) = mean_of(&decile_free_commission[decile]);
            let (dprice, dprice_dropped) = mean_of(&decile_price[decile]);
            println!(
                "         {decile}: components MEAN half-spread {dhalf:8.3} + commission \
                 {dcomm:8.3} + regulatory {REGULATORY_BPS:.3} = {dfixed:8.3} | \
                 ZERO-COMMISSION {dfree:8.3} bps ({dfree_dropped} dropped) | harmonic price MEAN \
                 {dprice:9.2} median {:9.2} ({dprice_dropped} dropped)",
                median(&mut decile_price[decile].clone()),
            );
            println!(
                "         {decile}: components MEDIAN half-spread {:8.3} + commission {:8.3} + \
                 regulatory {REGULATORY_BPS:.3} = {:8.3} (a sum of medians, which is the median \
                 of nothing) | median of the SUM {:8.3}",
                median(&mut decile_half_spread[decile].clone()),
                median(&mut decile_commission[decile].clone()),
                median(&mut decile_half_spread[decile].clone())
                    + median(&mut decile_commission[decile].clone())
                    + REGULATORY_BPS,
                median(&mut decile_fixed[decile].clone()),
            );
        }
        // THE OVERLAP NUMBER. A cost-side liquidity restriction and a `mu_hat / sigma_hat`
        // confidence restriction are the same restriction to the extent that dollar ADV predicts
        // quietness. Rank correlation rather than a comparison of decile means, because the
        // decision is made name by name, and the intersection count beside it, because a moderate
        // correlation still leaves the joint cell either double-counted or starved.
        //
        // Both inputs are measured from stored bars with no model in the path: `adv_usd` is
        // span-pooled dollar volume and `sigma_daily` is realized close-to-close volatility. That
        // matters, because every model-derived `sigma_hat` is under suspicion of a representation
        // artifact and this pair cannot carry one.
        //
        // THREE CUTS, PRINTED TOGETHER, AND THE NARROW ONE IS NOT DELETED. A tenth-sized tail is the
        // question originally asked and it CANNOT BE ANSWERED on this panel: 25 of 256 names against
        // a 43-name draw expects 4.20 with sd 1.78, so even an observed ZERO reaches only p = 0.008
        // and a strong depletion is indistinguishable from noise at any effect size. That is a
        // property of the PANEL, not of a bad result, and every tail-versus-tail intersection taken
        // here inherits it. Widening to a sixth and a quartile is a genuine change of estimand - the
        // loudest QUARTER is not the loudest TENTH - so the narrow cut stays printed beside the wide
        // ones and the reader can see which claim rests on which. `sd` is the hypergeometric sd of
        // each expectation, so no count on this line can be read as resolved without its own noise
        // level sitting next to it.
        for tail_fraction in [0.1f64, 1.0 / 6.0, 0.25] {
            let overlap =
                liquidity_volatility_overlap(&ranked, decile_names[DECILES - 1].len(), tail_fraction);
            let z = |count: usize| (count as f64 - overlap.expected) / overlap.sd.max(f64::MIN_POSITIVE);
            println!(
                "LIQUIDITY vs VOLATILITY over {} traded names: Spearman rho(dollar ADV, realized \
                 sigma) = {:+.4} | deepest-liquidity decile ({} names) vs volatility tails of {} \
                 names ({:.3} of the panel): QUIETEST {} ({:+.2} sigma, {:.2}x, exact P(X<=obs) \
                 {:.4}) LOUDEST {} ({:+.2} sigma, {:.2}x, exact P(X<=obs) {:.4}), {:.2} expected \
                 for either if independent, sd {:.3}",
                overlap.pairs,
                overlap.rho,
                overlap.deep_n,
                overlap.tail_n,
                tail_fraction,
                overlap.intersection,
                z(overlap.intersection),
                overlap.intersection as f64 / overlap.expected.max(f64::MIN_POSITIVE),
                overlap.p_quiet,
                overlap.loud_intersection,
                z(overlap.loud_intersection),
                overlap.loud_intersection as f64 / overlap.expected.max(f64::MIN_POSITIVE),
                overlap.p_loud,
                overlap.expected,
                overlap.sd,
            );
        }
        // The deepest decile's MEMBERS, printed so a peer restricting an edge measurement to this
        // sub-book restricts to exactly the names priced above rather than re-deriving a boundary.
        let mut deepest: Vec<&str> = decile_names[DECILES - 1].clone();
        deepest.sort_unstable();
        println!(
            "deepest-decile traded symbols ({}): {}",
            deepest.len(),
            deepest.join(" ")
        );
        // BOUNDARY SENSITIVITY OF THE DEEPEST-DECILE MEAN, because that mean is a PERMANENTLY
        // CHARTED reference line in `horizon.rs` and its membership comes from a rank cut.
        //
        // The cut is `decile * count / DECILES` over the FULL 5,297-symbol universe, an exact
        // partition where each decile's `hi` is the next one's `lo` - so no name is dropped or
        // double-counted and there is no floor-versus-round choice to make. That is structurally
        // different from the tail cut whose `(n / DECILES).max(1)` on a 256-long ranking I retracted:
        // there one boundary name was 4% of the tail AND the statistic was a count of 1-2, so a
        // single name was 100% of the effect.
        //
        // What remains is a name sitting at universe rank 4767, which would join or leave the decile
        // under a one-rank perturbation. Reported as the WORST CASE over all members rather than by
        // identifying which one it is: no single member can move an equal-weighted mean of `n` by
        // more than `|x_i - mean| / n`, so the max of that over the decile bounds the exposure
        // whichever name is at the boundary.
        let deep_costs = &decile_fixed[DECILES - 1];
        let (deep_mean, _) = mean_of(deep_costs);
        let worst_leverage = deep_costs
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .map(|v| (v - deep_mean).abs() / deep_costs.len() as f64)
            .fold(0.0f64, f64::max);
        println!(
            "deepest-decile MEAN {deep_mean:.3} bps over {} names: worst single-member leverage \
             {worst_leverage:.4} bps ({:.2}% of the mean), so a one-rank boundary perturbation \
             cannot move the charted floor by more than that",
            deep_costs.len(),
            100.0 * worst_leverage / deep_mean.max(f64::MIN_POSITIVE),
        );
        let (fixed_mean, fixed_dropped) = mean_of(&fixed);
        let (sized_mean, sized_dropped) = mean_of(&sized);
        let (month_mean, month_dropped) = mean_of(&month_sized);
        let (universe_fixed_mean, universe_fixed_dropped) = mean_of(&universe_fixed);
        let (universe_sized_mean, universe_sized_dropped) = mean_of(&universe_sized);
        // THE UNIVERSE'S EXTREME COST TAIL, which is what decides whether widening the fit slice
        // dilutes the expensive names or simply draws more of them.
        //
        // Measured because the alternative is an exponent. A turnover-weighted cost over a small
        // draw is wide when extreme-cost names land in the high-turnover positions, and whether a
        // larger draw fixes that is a property of how many such names the universe HOLDS, not of
        // any scaling law. Printed as a count above each threshold so a later draw's top-weighted
        // costs can be read against the population they came from rather than against a prediction.
        {
            let mut tail = universe_fixed.clone();
            let (p50, p90, p99, p999) = (
                quantile(&mut tail, 0.50),
                quantile(&mut tail, 0.90),
                quantile(&mut tail, 0.99),
                quantile(&mut tail, 0.999),
            );
            let finite = universe_fixed.iter().filter(|bps| bps.is_finite()).count();
            let above = |threshold: f64| {
                universe_fixed
                    .iter()
                    .filter(|bps| bps.is_finite() && **bps >= threshold)
                    .count()
            };
            println!(
                "UNIVERSE impact-free cost TAIL over {finite} priceable names: median {p50:.3} \
                 p90 {p90:.3} p99 {p99:.3} p99.9 {p999:.3} bps | names at or above 25 bps: {} \
                 ({:.2}%), 50 bps: {} ({:.2}%), 85 bps: {} ({:.2}%), 100 bps: {} ({:.2}%)",
                above(25.0),
                100.0 * above(25.0) as f64 / finite as f64,
                above(50.0),
                100.0 * above(50.0) as f64 / finite as f64,
                above(85.0),
                100.0 * above(85.0) as f64 / finite as f64,
                above(100.0),
                100.0 * above(100.0) as f64 / finite as f64,
            );
        }
        println!(
            "MATCHED measured impact-free: MEAN {fixed_mean:.3} bps ({fixed_dropped} dropped \
             non-finite) | median {:.3} bps",
            median(&mut fixed.clone()),
        );
        println!(
            "MATCHED all-in @1% ADV (k={IMPACT_K}): MEAN {sized_mean:.3} bps span-pooled \
             ({sized_dropped} dropped), MEAN {month_mean:.3} bps anchor-month ({month_dropped} \
             dropped) | median {:.3} span-pooled, {:.3} anchor-month",
            median(&mut sized.clone()),
            median(&mut month_sized.clone()),
        );
        println!(
            "UNIVERSE ({} symbols): measured impact-free MEAN {universe_fixed_mean:.3} bps \
             ({universe_fixed_dropped} dropped), all-in @1% MEAN {universe_sized_mean:.3} bps \
             ({universe_sized_dropped} dropped)",
            calibration.len(),
        );
        println!(
            "UNIVERSE decile medians, impact-free: thinnest {:.3} deepest {:.3} | all-in @1%: \
             thinnest {:.3} deepest {:.3}",
            universe[0].median_all_in_bps[0],
            universe[DECILES - 1].median_all_in_bps[0],
            universe[0].median_all_in_bps[PARTICIPATION_HEADLINE_SLOT],
            universe[DECILES - 1].median_all_in_bps[PARTICIPATION_HEADLINE_SLOT],
        );

        // =====================================================================
        // COST COMPONENT AUDIT
        // =====================================================================
        //
        // Everything below is the SAME resolution as `fixed_mean` on the SAME 256 names: the
        // pooled tier, impact-free, one leg. The point of splitting it is that a single scalar
        // cannot be checked, and three of the four questions this audit answers - is commission
        // per share, is the regulatory leg sided correctly, what does zero commission buy - are
        // questions about ONE TERM that the scalar cannot be interrogated about.
        let (half_mean, half_dropped) = mean_of(&half_spread);
        let (comm_mean, comm_dropped) = mean_of(&commission);
        let (reg_mean, _) = mean_of(&regulatory);
        let (free_comm_mean, free_comm_dropped) = mean_of(&fixed_free_commission);
        let (price_mean, price_dropped) = mean_of(&prices);
        let (reg_share_mean, reg_share_dropped) = mean_of(&regulatory_per_share);
        println!(
            "== MATCHED component decomposition, {found} names, impact-free, one-way, span-pooled =="
        );
        println!(
            "half-spread  MEAN {half_mean:8.3} bps (median {:8.3}, {half_dropped} dropped) = \
             {:5.1}% of {fixed_mean:.3}",
            median(&mut half_spread.clone()),
            100.0 * half_mean / fixed_mean,
        );
        println!(
            "commission   MEAN {comm_mean:8.3} bps (median {:8.3}, {comm_dropped} dropped) = \
             {:5.1}% of {fixed_mean:.3}",
            median(&mut commission.clone()),
            100.0 * comm_mean / fixed_mean,
        );
        println!(
            "regulatory   MEAN {reg_mean:8.3} bps (a CONSTANT, {REGULATORY_BPS}) = {:5.1}% of \
             {fixed_mean:.3}",
            100.0 * reg_mean / fixed_mean,
        );
        println!(
            "SUM of the three: {:8.3} bps against `fixed` MEAN {fixed_mean:.3} bps (residual \
             {:.3e}, which must be zero: `fixed_bps` IS this sum)",
            half_mean + comm_mean + reg_mean,
            half_mean + comm_mean + reg_mean - fixed_mean,
        );

        // THE ZERO-COMMISSION COUNTERFACTUAL. Commission is a broker's schedule, so unlike the
        // spread it is the one term a different account could genuinely remove.
        let deep_free = &decile_free_commission[DECILES - 1];
        let (deep_free_mean, deep_free_dropped) = mean_of(deep_free);
        let (universe_free_mean, universe_free_dropped) = mean_of(&universe_free_commission);
        println!(
            "ZERO-COMMISSION counterfactual (commission set to exactly 0, every other term \
             untouched): MATCHED {free_comm_mean:.3} bps against {fixed_mean:.3} ({:+.3}, \
             {:.1}% cheaper, {free_comm_dropped} dropped) | DEEPEST-DECILE {deep_free_mean:.3} \
             bps against {deep_mean:.3} ({:+.3}, {:.1}% cheaper, {deep_free_dropped} dropped) \
             | UNIVERSE {universe_free_mean:.3} bps against {universe_fixed_mean:.3} \
             ({universe_free_dropped} dropped)",
            free_comm_mean - fixed_mean,
            100.0 * (fixed_mean - free_comm_mean) / fixed_mean,
            deep_free_mean - deep_mean,
            100.0 * (deep_mean - deep_free_mean) / deep_mean,
        );

        // THE PRICE DISTRIBUTION, which IS the cross-sectional structure of the commission term:
        // commission is `1e4 * rate / price`, so a commission mean quoted without the price
        // deciles beside it cannot be checked against any broker schedule.
        let price_deciles: Vec<f64> = (1..DECILES)
            .map(|step| quantile(&mut prices.clone(), step as f64 / DECILES as f64))
            .collect();
        println!(
            "TRADED harmonic-price deciles (d1..d9): {} | MEAN {price_mean:.2} ({price_dropped} \
             dropped) | commission at each decile: {}",
            price_deciles
                .iter()
                .map(|p| format!("{p:.2}"))
                .collect::<Vec<_>>()
                .join(" "),
            price_deciles
                .iter()
                .map(|p| format!(
                    "{:.3}",
                    1.0e4 * (COMMISSION_PER_SHARE_USD / p).min(COMMISSION_CAP_FRACTION)
                ))
                .collect::<Vec<_>>()
                .join(" "),
        );
        let sub_dollar = prices.iter().filter(|p| p.is_finite() && **p < 1.0).count();
        let sub_five = prices.iter().filter(|p| p.is_finite() && **p < 5.0).count();
        let capped = prices
            .iter()
            .filter(|p| {
                p.is_finite() && **p > 0.0 && COMMISSION_PER_SHARE_USD / **p > COMMISSION_CAP_FRACTION
            })
            .count();
        println!(
            "TRADED price tail: {sub_dollar} names under $1, {sub_five} under $5, {capped} where \
             the 1%-of-value commission CAP binds (below ${:.2})",
            COMMISSION_PER_SHARE_USD / COMMISSION_CAP_FRACTION,
        );

        // THE FLAT-BPS ERROR, stated as the arithmetic a flat-bps commission model performs:
        // evaluate `rate / price` ONCE at the book's average price instead of per name and
        // average. The gap is pure Jensen on a convex function of price and is the entire reason
        // this module prices commission per share.
        let flat_equivalent =
            1.0e4 * (COMMISSION_PER_SHARE_USD / price_mean).min(COMMISSION_CAP_FRACTION);
        // ADV-WEIGHTED, named as such and NOT as turnover-weighted: this is what a book that
        // traded each name in proportion to its dollar ADV would pay, which is a property of the
        // corpus alone. The model's own turnover weighting needs the model's positions and is not
        // available here; conflating the two is exactly the substitution this campaign keeps
        // retracting.
        let weighted_mean = |values: &[f64]| -> (f64, usize) {
            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;
            let mut dropped = 0usize;
            for (value, (adv, _)) in values.iter().zip(&ranked) {
                if value.is_finite() && adv.is_finite() && *adv > 0.0 {
                    numerator += adv * value;
                    denominator += adv;
                } else {
                    dropped += 1;
                }
            }
            if denominator > 0.0 {
                (numerator / denominator, dropped)
            } else {
                (f64::NAN, dropped)
            }
        };
        let (comm_adv_weighted, comm_adv_dropped) = weighted_mean(&commission);
        let (fixed_adv_weighted, fixed_adv_dropped) = weighted_mean(&fixed);
        let (half_adv_weighted, _) = weighted_mean(&half_spread);
        println!(
            "COMMISSION weighting: per-share EQUAL-weighted {comm_mean:.3} bps | per-share \
             ADV-NOTIONAL-weighted {comm_adv_weighted:.3} bps ({comm_adv_dropped} dropped) | \
             flat-bps model evaluated once at the MEAN price ${price_mean:.2} gives \
             {flat_equivalent:.3} bps, understating the equal-weighted per-share truth by \
             {:.2}x",
            comm_mean / flat_equivalent,
        );
        println!(
            "ADV-NOTIONAL-weighted MATCHED impact-free {fixed_adv_weighted:.3} bps \
             ({fixed_adv_dropped} dropped) against the EQUAL-weighted {fixed_mean:.3} bps, \
             half-spread {half_adv_weighted:.3} against {half_mean:.3}. Construction, not a \
             substitute: ADV weighting is a corpus property and NOT the model's turnover."
        );
        // THE RECONCILIATION. A one-way decomposition of `half-spread 3.196 + commission 0.338 +
        // regulatory 0.150 = 3.684` bps has been quoted against `MATCHED_MEASURED_BPS`, which is
        // 10.620. Two numbers claiming to be the same measurement cannot both stand, so every
        // construction this cost path admits is priced here and the one that lands on 3.684 is
        // named. The candidates differ in exactly two ways: POPULATION (the traded 256 against the
        // whole 5,297) and WEIGHTING (equal over names against dollar-ADV over notional).
        let universe_weighted_mean = |values: &[f64]| -> (f64, usize) {
            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;
            let mut dropped = 0usize;
            for (value, adv) in values.iter().zip(&universe_adv) {
                if value.is_finite() && adv.is_finite() && *adv > 0.0 {
                    numerator += adv * value;
                    denominator += adv;
                } else {
                    dropped += 1;
                }
            }
            if denominator > 0.0 {
                (numerator / denominator, dropped)
            } else {
                (f64::NAN, dropped)
            }
        };
        let (universe_half_adv, universe_half_adv_dropped) = universe_weighted_mean(&universe_half);
        let (universe_comm_adv, _) = universe_weighted_mean(&universe_commission);
        let (universe_fixed_adv, _) = universe_weighted_mean(&universe_fixed);
        let (universe_impact_adv, _) = universe_weighted_mean(&universe_impact);
        let (sized_adv_weighted, _) = weighted_mean(&sized);
        println!(
            "== RECONCILIATION of the 3.684 bps decomposition against MATCHED_MEASURED_BPS \
             {MATCHED_MEASURED_BPS:.3} =="
        );
        println!(
            "UNIVERSE, ADV-NOTIONAL-weighted: half-spread {universe_half_adv:.3} + commission \
             {universe_comm_adv:.3} + regulatory {REGULATORY_BPS:.3} = {universe_fixed_adv:.3} bps \
             ({universe_half_adv_dropped} dropped) | impact @1% ADV {universe_impact_adv:.3} bps",
        );
        println!(
            "MATCHED,  ADV-NOTIONAL-weighted: half-spread {half_adv_weighted:.3} + commission \
             {comm_adv_weighted:.3} + regulatory {REGULATORY_BPS:.3} = {fixed_adv_weighted:.3} bps \
             | all-in @1% ADV {sized_adv_weighted:.3} bps",
        );
        println!(
            "UNIVERSE, EQUAL-weighted:        {:.3} bps | MATCHED, EQUAL-weighted: \
             {fixed_mean:.3} bps (this is MATCHED_MEASURED_BPS) | MATCHED deepest decile, \
             EQUAL-weighted: {deep_mean:.3} bps",
            universe_fixed_mean,
        );
        // SUM OF COMPONENT MEDIANS, over every decile of the UNIVERSE. It is not the median of
        // anything - the median of a sum is not the sum of medians for correlated terms - but it is
        // the construction a decile table invites, because `CostDecile` carries a median spread and
        // a median fee side by side and adding them is one keystroke. Printed for all ten deciles
        // so an orphan decomposition can be located rather than guessed at.
        println!("== UNIVERSE deciles, SUM OF COMPONENT MEDIANS (not a median of anything) ==");
        for row in &universe {
            let half = 0.5 * row.median_roll_spread_bps;
            let impact = 1.0e4
                * IMPACT_K
                * row.median_sigma_daily
                * PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT].sqrt();
            println!(
                "universe decile {}: half of median Roll spread {half:8.3} + median fee \
                 {:8.3} = {:8.3} bps | median of the SUM {:8.3} bps | impact at median sigma \
                 {impact:8.3} bps | median price ${:9.2}",
                row.decile,
                row.median_fee_bps,
                half + row.median_fee_bps,
                row.median_all_in_bps[0],
                row.median_harmonic_price,
            );
        }

        // ANCHOR-MONTH pricing of the IMPACT-FREE figure, which the published constant does not
        // carry. `MATCHED_MEASURED_BPS` is span-pooled: it prices five years of liquidity against
        // an edge measured inside a five-month pinned window. The anchor month is the month each
        // window actually traded in, so this is the same 256 names at the liquidity and the PRICE
        // LEVEL they were actually traded at.
        let (month_fixed_mean, month_fixed_dropped) = mean_of(&month_fixed);
        let (month_comm_mean, _) = mean_of(&month_commission);
        let (month_half_mean, _) = mean_of(&month_half);
        let (month_price_mean, month_price_dropped) = mean_of(&month_price);
        println!(
            "ANCHOR-MONTH matched impact-free: half-spread {month_half_mean:.3} + commission \
             {month_comm_mean:.3} + regulatory {REGULATORY_BPS:.3} = {month_fixed_mean:.3} bps \
             ({month_fixed_dropped} dropped) against the span-pooled {fixed_mean:.3} bps that is \
             MATCHED_MEASURED_BPS ({:+.3}) | anchor-month harmonic price MEAN \
             ${month_price_mean:.2} ({month_price_dropped} dropped) against span-pooled \
             ${price_mean:.2}",
            month_fixed_mean - fixed_mean,
        );
        // SPLIT-ADJUSTMENT EXPOSURE of the per-share commission term. `close` is adjusted, so the
        // price the commission is divided by is not the price shares traded at, and the ratio of
        // the five-year pooled price to the anchor-month price is the only in-corpus observable
        // that scales with the cumulative adjustment. Reported as a distribution, because the
        // question is not the average name but how many names are badly rescaled.
        let ratio_deciles: Vec<f64> = (1..DECILES)
            .map(|step| quantile(&mut price_ratio.clone(), step as f64 / DECILES as f64))
            .collect();
        let wide = price_ratio
            .iter()
            .filter(|r| r.is_finite() && (**r > 2.0 || **r < 0.5))
            .count();
        let very_wide = price_ratio
            .iter()
            .filter(|r| r.is_finite() && (**r > 10.0 || **r < 0.1))
            .count();
        println!(
            "SPLIT/ADJUSTMENT exposure of the per-share commission: pooled-over-anchor harmonic \
             price ratio over {} names, deciles (d1..d9) {} | {wide} names beyond 2x or 0.5x, \
             {very_wide} beyond 10x or 0.1x. A ratio of `x` means the span-pooled commission of \
             that name is charged at a price level `x` times its traded-month level.",
            price_ratio.len(),
            ratio_deciles
                .iter()
                .map(|r| format!("{r:.3}"))
                .collect::<Vec<_>>()
                .join(" "),
        );

        // THE STATUTORY RATES HAVE MOVED SINCE `REGULATORY_BPS` WAS WRITTEN, so the constant is
        // priced at both vintages rather than only at the one its doc comment cites. Sources:
        // SEC Fee Rate Advisory 2026-2 sets Section 31 at `$20.60` per million from 2026-04-04
        // (and `$0.00` per million on charge dates through 2026-04-03); FINRA raised the TAF on
        // covered equity sales to `$0.000195`/share on 2026-01-01, cap `$9.79`/trade.
        const SEC_SELL_BPS_FY2026: f64 = 0.206;
        const TAF_SELL_PER_SHARE_USD_2026: f64 = 0.000195;
        let regulatory_current = |price: f64| -> f64 {
            if price.is_finite() && price > 0.0 {
                0.5 * (SEC_SELL_BPS_FY2026 + 1.0e4 * TAF_SELL_PER_SHARE_USD_2026 / price)
            } else {
                f64::NAN
            }
        };
        let matched_current: Vec<f64> = prices.iter().copied().map(regulatory_current).collect();
        let universe_current: Vec<f64> =
            universe_price.iter().copied().map(regulatory_current).collect();
        let (matched_current_mean, _) = mean_of(&matched_current);
        let (universe_current_mean, _) = mean_of(&universe_current);
        println!(
            "REGULATORY rate vintage: charged {REGULATORY_BPS:.3} bps/leg from SEC \
             {SEC_SELL_BPS} bps + TAF ${TAF_SELL_PER_SHARE_USD}/share (FY2025 rates) | at CURRENT \
             rates (SEC {SEC_SELL_BPS_FY2026} bps sell-side, TAF \
             ${TAF_SELL_PER_SHARE_USD_2026}/share) priced per share: MATCHED MEAN \
             {matched_current_mean:.4} bps/leg, UNIVERSE MEAN {universe_current_mean:.4} bps/leg. \
             Error of the charged constant at current rates: MATCHED {:+.4}, UNIVERSE {:+.4} \
             bps/leg",
            matched_current_mean - REGULATORY_BPS,
            universe_current_mean - REGULATORY_BPS,
        );

        // THE REGULATORY LEG. `REGULATORY_BPS` is a flat constant whose own doc derives it at an
        // assumed $70 price, but the FINRA TAF inside it is charged PER SHARE, so its bps value
        // moves with price exactly as commission does. This prices it per share at each name's own
        // price under the same SELL-side-halved convention and reports the gap.
        let (universe_reg_share_mean, universe_reg_share_dropped) =
            mean_of(&universe_regulatory_per_share);
        println!(
            "REGULATORY sidedness/units: constant charged {REGULATORY_BPS:.3} bps/leg | \
             SEC-31 {SEC_SELL_BPS} bps SELL-side -> {:.4} bps/leg | TAF \
             ${TAF_SELL_PER_SHARE_USD}/share SELL-side priced at each name's own price -> \
             per-share-correct MATCHED MEAN {reg_share_mean:.4} bps/leg (median {:.4}, \
             {reg_share_dropped} dropped), UNIVERSE MEAN {universe_reg_share_mean:.4} bps/leg \
             ({universe_reg_share_dropped} dropped). Understatement of the charged constant: \
             MATCHED {:+.4} bps/leg, UNIVERSE {:+.4} bps/leg",
            0.5 * SEC_SELL_BPS,
            median(&mut regulatory_per_share.clone()),
            reg_share_mean - REGULATORY_BPS,
            universe_reg_share_mean - REGULATORY_BPS,
        );

        // THE UNIVERSE decomposition, so the matched figure can be read against the population it
        // was drawn from term by term rather than only in total.
        let (universe_half_mean, _) = mean_of(&universe_half);
        let (universe_comm_mean, universe_comm_dropped) = mean_of(&universe_commission);
        let (universe_price_mean, _) = mean_of(&universe_price);
        println!(
            "UNIVERSE component decomposition: half-spread {universe_half_mean:.3} + commission \
             {universe_comm_mean:.3} ({universe_comm_dropped} dropped) + regulatory \
             {REGULATORY_BPS:.3} = {universe_fixed_mean:.3} bps | harmonic price MEAN \
             ${universe_price_mean:.2} median ${:.2}",
            median(&mut universe_price.clone()),
        );
        // The two published constants re-derived in this run, so a drift shows up as a printed
        // disagreement rather than as a stale literal.
        println!(
            "PUBLISHED constants against this run: MATCHED_MEASURED_BPS \
             {MATCHED_MEASURED_BPS:.3} vs measured {fixed_mean:.3} ({:+.3}) | \
             MATCHED_DEEPEST_DECILE_BPS {MATCHED_DEEPEST_DECILE_BPS:.3} vs measured \
             {deep_mean:.3} ({:+.3}) | MATCHED_ALL_IN_BPS {MATCHED_ALL_IN_BPS:.3} vs measured \
             {sized_mean:.3} ({:+.3}) | UNIVERSE_MEASURED_BPS {UNIVERSE_MEASURED_BPS:.3} vs \
             measured {universe_fixed_mean:.3} ({:+.3})",
            fixed_mean - MATCHED_MEASURED_BPS,
            deep_mean - MATCHED_DEEPEST_DECILE_BPS,
            sized_mean - MATCHED_ALL_IN_BPS,
            universe_fixed_mean - UNIVERSE_MEASURED_BPS,
        );

        // THE JOIN ARTIFACT. Written beside the window artifact this test reads, in the same shape
        // and for the same reason: the per-name costs belong to a CORPUS measurement while the
        // per-name turnover belongs to a RUN, and the turnover-weighted composite is a join of the
        // two. Emitting `c_i` here means whoever measures turnover never re-derives a cost, which
        // is the substitution that has produced every retraction in this campaign.
        let cost_artifact = Path::new(&path).with_file_name("pretrain_cost_per_symbol.json");
        let payload = serde_json::json!({
            "source": "portfolio_cost::tests::the_traded_window_set_is_priced_against_the_deciles_it_occupies",
            "windows_artifact": path,
            "universe_symbols": calibration.len(),
            "construction": "span-pooled ResolvedCost, impact-free unless named, ONE leg",
            "matched_equal_weighted_fixed_bps": fixed_mean,
            "matched_adv_notional_weighted_fixed_bps": fixed_adv_weighted,
            "rows": cost_rows,
        });
        fs::write(&cost_artifact, serde_json::to_vec_pretty(&payload).expect("the payload encodes"))
            .unwrap_or_else(|error| panic!("writing {}: {error}", cost_artifact.display()));
        println!(
            "wrote {} rows of per-symbol cost to {}",
            cost_rows.len(),
            cost_artifact.display()
        );

        // THE COMPOSITE, when a turnover artifact exists. `break_even = gross_edge / turnover` and
        // the bench charges ONE uniform scalar on `Ledger::traded`, so the flat equivalent that
        // kills the edge is `sum_i c_i tau_i / sum_i tau_i` - the TURNOVER-weighted mean, exactly.
        // Equal-weighted is the right comparator only for an equal-weight book, and ADV-notional
        // weighting is a corpus property standing in for a book property. This is the real thing,
        // and it is arithmetic on two files with no GPU and no model.
        //
        // Absent the file the test still passes: a composite nobody has measured must not be
        // fabricated, and a skipped join says so on stdout rather than producing a number.
        match std::env::var("TB_WINDOW_TURNOVER") {
            Err(_) => println!(
                "TURNOVER-WEIGHTED composite SKIPPED: set TB_WINDOW_TURNOVER to a JSON file of \
                 {{\"rows\":[{{\"symbol\",\"turnover\",\"turnover_interior\",\"bars\",\"policy\"?}}]}}. \
                 `turnover` is the per-window sum of |f_t - f_{{t-1}}| INCLUDING the entry from \
                 flat and the terminal unwind; `turnover_interior` EXCLUDES both. Interior is \
                 primary: the two boundary trades are a fixed 2 units per window allocated by the \
                 window SAMPLER rather than by the model, so they dilute exactly the concentration \
                 the composite exists to detect. Equal-weighted {fixed_mean:.3} bps and \
                 ADV-notional-weighted {fixed_adv_weighted:.3} bps stand alone until it lands."
            ),
            Ok(turnover_path) => {
                let text = fs::read_to_string(&turnover_path)
                    .unwrap_or_else(|error| panic!("reading {turnover_path}: {error}"));
                let parsed: serde_json::Value =
                    serde_json::from_str(&text).expect("the turnover artifact is JSON");
                let default_policy = parsed["policy"].as_str().unwrap_or("model").to_owned();
                // `turnover` is the TRADED slice and `turnover_fit` the FIT slice;
                // `rows` is the standalone shape. Whichever exist are priced, each tagged with its
                // slice, because an argmax taken on fit-slice turnover must be priced with
                // fit-slice cost weights - pricing a fit-slice choice with traded-slice weights is
                // in-sample selection one level up.
                let rows: Vec<(&'static str, &serde_json::Value)> =
                    [("traded", "turnover"), ("fit", "turnover_fit"), ("traded", "rows")]
                        .iter()
                        .filter_map(|(slice, key)| {
                            parsed[*key].as_array().map(|array| (*slice, array))
                        })
                        .flat_map(|(slice, array)| array.iter().map(move |row| (slice, row)))
                        .collect();
                assert!(
                    !rows.is_empty(),
                    "the turnover artifact has no `turnover`, `turnover_fit` or `rows` array"
                );
                // The TRAIN-REGION rank, as a CROSS-CHECK and never as the primary. It is a median
                // over 58 sessions of 2021-09-22..2025-10-06 from `universe.json`, which is the
                // quantity universe MEMBERSHIP was selected on, while this module's own percentile
                // is a span-pooled bar-measured ADV rank, which is the quantity the cost model
                // resolves against. Two rankings of the same names; if they disagree on the traded
                // book that is itself a finding, and it can only be seen by carrying both. Absent
                // file or absent name yields NaN and drops out of the weighted mean rather than
                // being imputed. Nine corpus symbols have no train-region rank AT ALL and their
                // absence is not a join failure.
                let train_region: HashMap<String, f64> = fs::read_to_string(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .expect("the package lives inside the workspace")
                        .join("long_data/adv_percentile_train_region.json"),
                )
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| value["rows"].as_array().cloned())
                .map(|rows| {
                    rows.iter()
                        .filter_map(|row| {
                            Some((
                                row["symbol"].as_str()?.to_owned(),
                                // That file states the rank on 0..100; this module's own rank is
                                // 0..1. Rescaled at the boundary so the two are comparable, which
                                // is the only reason to carry the second rank at all.
                                row["adv_percentile_train_region"].as_f64()? / 100.0,
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default();
                let cost_of: HashMap<&str, (f64, f64, f64, f64)> = cost_rows
                    .iter()
                    .map(|row| {
                        let symbol = row["symbol"].as_str().expect("symbol");
                        let fixed = row["fixed_bps"].as_f64().unwrap_or(f64::NAN);
                        (
                            symbol,
                            (
                                fixed,
                                row["adv_percentile_universe"].as_f64().unwrap_or(f64::NAN),
                                train_region.get(symbol).copied().unwrap_or(f64::NAN),
                                // Impact at the headline 1%-of-ADV participation slot. Carried so
                                // the ALL-IN composite is weighted by the same turnover as the
                                // fixed one; weighting the two terms differently is the defect
                                // this whole block exists to eliminate.
                                row["all_in_bps_at_1pct_adv"].as_f64().unwrap_or(f64::NAN) - fixed,
                            ),
                        )
                    })
                    .collect();
                println!(
                    "train-region ADV percentiles available for {} of {} traded names ({} corpus \
                     rows loaded)",
                    cost_rows
                        .iter()
                        .filter(|row| row["symbol"]
                            .as_str()
                            .is_some_and(|symbol| train_region.contains_key(symbol)))
                        .count(),
                    cost_rows.len(),
                    train_region.len(),
                );
                // A SILENT LEFT-JOIN DROP IS THE DEFECT THIS PANICS ON. Dropping an unmatched
                // symbol removes a candidate from the EXPENSIVE end of the cross-section, so the
                // composite would come out low and look favourable for the same reason the
                // deepest-decile comparison did. An unmatched symbol is a population disagreement
                // between the artifact and the corpus, which is a wiring error, not a pricing
                // question - so it names every offender and stops.
                let orphans: Vec<&str> = rows
                    .iter()
                    .filter_map(|(_, row)| row["symbol"].as_str())
                    .filter(|symbol| !cost_of.contains_key(*symbol))
                    .collect();
                assert!(
                    orphans.is_empty(),
                    "{} turnover rows name symbols with no cost row, so the join would drop them \
                     and bias the composite toward the cheap end: {}",
                    orphans.len(),
                    orphans.join(" ")
                );
                // THE SHARED-NAME CHANNEL, priced rather than argued.
                //
                // At 256 fit windows the two slices were NAME-disjoint by accident, so a margin
                // fitted on one and read on the other could not transfer through a shared name. The
                // design's actual guarantee is BLOCK-disjointness on (symbol, calendar-month), which
                // survives any fit width; name-disjointness was a free bonus that a wider fit slice
                // spends. What matters is not the intersection COUNT but the share of fit-slice
                // turnover WEIGHT those names carry, because a name with negligible weight cannot
                // move the selection whatever it shares.
                {
                    let fit_rows = rows.iter().filter(|(slice, _)| *slice == "fit");
                    let mut shared_names: BTreeSet<&str> = BTreeSet::new();
                    let mut fit_names: BTreeSet<&str> = BTreeSet::new();
                    let (mut shared_turnover, mut total_turnover) = (0.0f64, 0.0f64);
                    for (_, row) in fit_rows {
                        let Some(symbol) = row["symbol"].as_str() else { continue };
                        fit_names.insert(symbol);
                        let turnover = row["turnover_interior"]
                            .as_f64()
                            .or_else(|| row["turnover"].as_f64())
                            .filter(|value| value.is_finite() && *value > 0.0)
                            .unwrap_or(0.0);
                        total_turnover += turnover;
                        if wanted.contains(symbol) {
                            shared_names.insert(symbol);
                            shared_turnover += turnover;
                        }
                    }
                    if !fit_names.is_empty() {
                        println!(
                            "FIT/TRADED NAME OVERLAP: {} of {} fit names also appear in the traded \
                             {} ({:.2}% of names), carrying {:.3}% of fit-slice turnover weight. \
                             Block-disjointness on (symbol, calendar-month) is the enforced \
                             guarantee and is unaffected; name-disjointness is what a wider fit \
                             slice spends",
                            shared_names.len(),
                            fit_names.len(),
                            wanted.len(),
                            100.0 * shared_names.len() as f64 / fit_names.len() as f64,
                            100.0 * shared_turnover / total_turnover.max(f64::MIN_POSITIVE),
                        );
                    }
                }
                // A symbol must appear ONCE per (slice, policy, checkpoint). Two rows for one name
                // under one policy string mean two DIFFERENT books were formatted to the same label,
                // so one grid arm is missing from the table and the label that absorbed it would
                // describe a book nobody ran.
                //
                // Handled by REFUSAL rather than by the orphan guard's panic, and the asymmetry is
                // the point: a dropped symbol biases EVERY arm's composite toward the cheap end,
                // while a label collision biases only its own arm. So an orphan stops the pass and a
                // collision forfeits one row and is named, leaving the rest of the grid priced. The
                // arm is never reported as a pooled average, which is the one outcome that would be
                // wrong in a way a reader could not see.
                let mut seen: BTreeMap<(&str, &str, i64), usize> = BTreeMap::new();
                for (slice, row) in &rows {
                    let key = (
                        *slice,
                        row["policy"].as_str().unwrap_or(&default_policy),
                        row["step"].as_i64().unwrap_or(-1),
                    );
                    *seen.entry(key).or_insert(0) += 1;
                }
                let collided: BTreeSet<(&str, &str, i64)> = seen
                    .iter()
                    .filter_map(|((slice, policy, step), count)| {
                        let names = rows
                            .iter()
                            .filter(|(row_slice, row)| {
                                row_slice == slice
                                    && row["policy"].as_str().unwrap_or(&default_policy) == *policy
                                    && row["step"].as_i64().unwrap_or(-1) == *step
                            })
                            .filter_map(|(_, row)| row["symbol"].as_str())
                            .collect::<BTreeSet<_>>();
                        (names.len() < *count).then_some((*slice, *policy, *step))
                    })
                    .collect();
                for (slice, policy, step) in &collided {
                    println!(
                        "arm [{slice}] {policy} @{step} REFUSED: {} rows over fewer distinct \
                         symbols, so two books share this label and a composite for it would be a \
                         pooled average of two different books. The arm is forfeited, not pooled, \
                         and the grid arm it collided with is MISSING from this table",
                        seen[&(*slice, *policy, *step)],
                    );
                }
                // Per policy per turnover column. A struct rather than a widening tuple because it
                // now carries the three second-moment sums the interval needs and the top-weight
                // names the dilution question is about.
                #[derive(Clone, Copy)]
                struct Arm {
                    cost_turnover: f64,
                    turnover: f64,
                    pct_turnover: f64,
                    train_pct_turnover: f64,
                    train_turnover: f64,
                    impact_turnover: f64,
                    sq_turnover: f64,
                    sq_cost_turnover: f64,
                    sq_cost_sq_turnover: f64,
                    rows: usize,
                    top: [(f64, f64); 3],
                }
                const EMPTY_ARM: Arm = Arm {
                    cost_turnover: 0.0,
                    turnover: 0.0,
                    pct_turnover: 0.0,
                    train_pct_turnover: 0.0,
                    train_turnover: 0.0,
                    impact_turnover: 0.0,
                    sq_turnover: 0.0,
                    sq_cost_turnover: 0.0,
                    sq_cost_sq_turnover: 0.0,
                    rows: 0,
                    top: [(0.0, f64::NAN); 3],
                };
                impl Arm {
                    fn weighted(&self) -> f64 {
                        self.cost_turnover / self.turnover
                    }
                    /// 95% interval WIDTH of the turnover-weighted mean, by the Hajek
                    /// ratio-estimator linearization.
                    ///
                    /// Deterministic on purpose. Validated against a 4,000-resample bootstrap on
                    /// eight arms spanning both slices and widths 3.2 to 31.4 bps, agreeing to
                    /// within 3.2% everywhere, so a seed would buy nothing and would put an RNG
                    /// inside a number that scores a pre-registered prediction.
                    fn interval_width(&self) -> f64 {
                        let mean = self.weighted();
                        let variance = (self.sq_cost_sq_turnover
                            - 2.0 * mean * self.sq_cost_turnover
                            + mean * mean * self.sq_turnover)
                            / (self.turnover * self.turnover);
                        2.0 * 1.96 * variance.max(0.0).sqrt()
                    }
                    /// `1 / HHI` over the turnover weights: how many names the estimator effectively
                    /// rests on. This is the quantity a wider draw has to raise, and it is what
                    /// distinguishes dilution from simply drawing more company for the same tail.
                    fn effective_names(&self) -> f64 {
                        self.turnover * self.turnover / self.sq_turnover
                    }
                }
                let mut arms: BTreeMap<String, (Arm, Arm, usize)> = BTreeMap::new();
                for (slice, row) in &rows {
                    if collided.contains(&(
                        *slice,
                        row["policy"].as_str().unwrap_or(&default_policy),
                        row["step"].as_i64().unwrap_or(-1),
                    )) {
                        continue;
                    }
                    // Keyed by policy AND checkpoint AND slice. Pooling two checkpoints would
                    // produce a composite for a book no checkpoint traded; pooling two slices
                    // would price a fit-slice choice with traded-slice weights.
                    let policy = match (row["policy"].as_str(), row["step"].as_i64()) {
                        (Some(policy), Some(step)) => format!("[{slice}] {policy} @{step}"),
                        (Some(policy), None) => format!("[{slice}] {policy}"),
                        (None, Some(step)) => format!("[{slice}] {default_policy} @{step}"),
                        (None, None) => format!("[{slice}] {default_policy}"),
                    };
                    let entry = arms.entry(policy).or_insert((EMPTY_ARM, EMPTY_ARM, 0));
                    let symbol = row["symbol"].as_str().expect("every turnover row names a symbol");
                    let &(cost, percentile, train_percentile, impact) =
                        cost_of.get(symbol).expect("checked above");
                    // An absent `turnover_interior` is absent, not zero: the total column still
                    // accumulates and the interior arm reports a smaller row count, so a producer
                    // that emitted only one column cannot silently be read as the other.
                    for (column, accumulator) in [
                        (row["turnover"].as_f64(), &mut entry.0),
                        (row["turnover_interior"].as_f64(), &mut entry.1),
                    ] {
                        match column {
                            Some(turnover)
                                if turnover.is_finite() && turnover > 0.0 && cost.is_finite() =>
                            {
                                accumulator.cost_turnover += cost * turnover;
                                accumulator.turnover += turnover;
                                accumulator.pct_turnover += percentile * turnover;
                                // Separate denominator: a name with no train-region rank must not
                                // contribute zero to the numerator of a weighted mean.
                                if train_percentile.is_finite() {
                                    accumulator.train_pct_turnover += train_percentile * turnover;
                                    accumulator.train_turnover += turnover;
                                }
                                accumulator.impact_turnover += impact * turnover;
                                let square = turnover * turnover;
                                accumulator.sq_turnover += square;
                                accumulator.sq_cost_turnover += square * cost;
                                accumulator.sq_cost_sq_turnover += square * cost * cost;
                                accumulator.rows += 1;
                                // The three heaviest names by turnover weight, kept because the
                                // dilution question is about WHICH names the estimator rests on and
                                // what they cost, not about how many rows there are.
                                if turnover > accumulator.top[2].0 {
                                    accumulator.top[2] = (turnover, cost);
                                    accumulator.top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                                }
                            }
                            _ => {}
                        }
                    }
                    entry.2 += 1;
                }
                let equal_weighted_percentile = cost_rows
                    .iter()
                    .filter_map(|row| row["adv_percentile_universe"].as_f64())
                    .filter(|value| value.is_finite())
                    .sum::<f64>()
                    / cost_rows.len() as f64;
                println!(
                    "== TURNOVER-WEIGHTED composite, joined on symbol, no GPU. INTERIOR is \
                     primary; TOTAL carries the sampler's fixed two boundary trades per window =="
                );
                for (policy, (total, interior, rows_seen)) in &arms {
                    for (label, arm) in [("INTERIOR", interior), ("TOTAL", total)] {
                        if arm.turnover <= 0.0 {
                            println!(
                                "arm {policy} {label}: no usable turnover in {rows_seen} rows, so \
                                 no composite - the producer did not emit this column"
                            );
                            continue;
                        }
                        let weighted = arm.weighted();
                        println!(
                            "arm {policy} {label}: TURNOVER-weighted matched impact-free \
                             {weighted:.3} bps over {} rows | against EQUAL-weighted \
                             {fixed_mean:.3} ({:+.3}, {:.2}x) and ADV-NOTIONAL-weighted \
                             {fixed_adv_weighted:.3} ({:+.3}, {:.2}x) | turnover-weighted mean ADV \
                             PERCENTILE of the book {:.4} against {equal_weighted_percentile:.4} \
                             equal-weighted over the traded names (0 = thinnest of the 5,297, \
                             1 = deepest; this rank is SPAN-POOLED BAR-MEASURED ADV, not the \
                             train-region 58-session median in `universe.json`, which is a third \
                             construction)",
                            arm.rows,
                            weighted - fixed_mean,
                            weighted / fixed_mean,
                            weighted - fixed_adv_weighted,
                            weighted / fixed_adv_weighted,
                            arm.pct_turnover / arm.turnover,
                        );
                        if arm.train_turnover > 0.0 {
                            println!(
                                "  ^ cross-check, SAME weights, DIFFERENT rank: turnover-weighted \
                                 mean TRAIN-REGION ADV percentile {:.4} over the {:.1}% of this \
                                 arm's turnover whose name has one. That grid is a 58-session \
                                 median over the TRAIN region and this book trades the VAL split, \
                                 so it is a third construction and is reported, not substituted",
                                arm.train_pct_turnover / arm.train_turnover,
                                100.0 * arm.train_turnover / arm.turnover,
                            );
                        }
                        // THE DETERMINATION OF THIS ARM'S COST, which is what decides whether a
                        // selection made on it is resolvable. Reported beside the point estimate
                        // because a margin chosen on a cost with a 20-bps interval is not resolvably
                        // better than its neighbours, and that is invisible from the estimate alone.
                        println!(
                            "  ^ DETERMINATION: 95% interval width {:.2} bps (Hajek ratio \
                             linearization, deterministic - no seed) | effective names 1/HHI {:.1} \
                             of {} rows | top-3 by turnover weight cost {:.1}, {:.1}, {:.1} bps \
                             against this arm's weighted mean {weighted:.3}. A wider draw must raise \
                             the effective count and dilute those three; if they reappear at the top \
                             with more company, the interval will not have narrowed",
                            arm.interval_width(),
                            arm.effective_names(),
                            arm.rows,
                            arm.top[0].1,
                            arm.top[1].1,
                            arm.top[2].1,
                        );
                        // ALL-IN, weighted by the SAME turnover as the fixed term. Impact scales as
                        // sqrt(participation) (`total_bps`, :973), and participation is proportional
                        // to turnover per bar at fixed AUM, so an arm trading less pays less impact
                        // per unit traded. Referenced to the incumbent book's own turnover so the
                        // ratio is against a measured baseline rather than an assumed AUM. Every
                        // figure on this line rides the UNFITTED `IMPACT_K`, so it is a robustness
                        // check and never a headline.
                        // The reference is the `actual` arm OF THE SAME SLICE: participation is a
                        // property of the book on those windows, so a fit-slice arm scaled against
                        // a traded-slice incumbent would mix the two populations.
                        let slice_tag = policy.split(']').next().map(|head| format!("{head}]"));
                        let incumbent = arms
                            .iter()
                            .find(|(name, _)| {
                                slice_tag
                                    .as_deref()
                                    .is_some_and(|tag| name.starts_with(tag) && name.contains("actual"))
                            })
                            .map(|(_, (_, interior, _))| interior.turnover);
                        if let Some(reference) = incumbent.filter(|value| *value > 0.0) {
                            let scale = (arm.turnover / reference).sqrt();
                            let impact = arm.impact_turnover / arm.turnover * scale;
                            println!(
                                "  ^ ALL-IN [INFERENCE, unfitted IMPACT_K]: impact at 1% ADV \
                                 {:.3} bps scaled by sqrt(turnover / incumbent turnover) = \
                                 {scale:.4} gives {impact:.3}, so all-in {:.3} bps against \
                                 impact-free {weighted:.3}. Both terms carry the SAME turnover \
                                 weights; mixing an equal-weighted impact into a turnover-weighted \
                                 fixed cost is the defect this line exists to avoid",
                                arm.impact_turnover / arm.turnover,
                                weighted + impact,
                            );
                        }
                    }
                }
            }
        }
        // A mean over a subset with dropped members is not a mean over the subset, so the count is
        // part of the number rather than a footnote.
        assert!(
            fixed_mean.is_finite() && fixed_mean > 0.0,
            "the matched impact-free cost must be a measured positive number, got {fixed_mean}"
        );
    }

    /// The headline scalars of the whole exercise, measured on the REAL corpus.
    ///
    /// `#[ignore]`d because it reads all 5,297 intraday files and computes a full cross-sectional
    /// spectrum, which is minutes of CPU and gigabytes of page cache. It is also the only thing
    /// that can answer the actual question, because every synthetic fixture answers a question
    /// about the fixture.
    ///
    /// Run it exactly like this and read the printed block:
    ///
    /// ```text
    /// OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= ./torch-env.sh \
    ///     cargo test -j 4 --lib -- portfolio_cost::tests::the_real_corpus \
    ///     --ignored --nocapture --test-threads=1
    /// ```
    ///
    /// The env vars are house rule on every `cargo test` in this crate and cost nothing here, but
    /// they are NOT what bounds this test, and the difference is worth stating because assuming
    /// otherwise has already burned an afternoon. MEASURED on the matched sibling test, fully
    /// pinned exactly as above: 46.2 s wall, 170.1 s user + 5.7 s sys, so 3.81 CORES - not one.
    /// The same test with no thread env vars at all ran 42.7 s on a warm cache, i.e. pinning moved
    /// neither the wall time nor the core count.
    ///
    /// What actually sets the core count is an ARGUMENT, not the environment:
    /// [`CostCalibration::from_corpus`] builds an EXPLICIT rayon pool with
    /// `ThreadPoolBuilder::num_threads(threads)` and `install`s the pass into it, and the tests here
    /// pass `4`. Lower that argument to make this fixture cheaper; no environment variable will do
    /// it. `RAYON_NUM_THREADS` will not either, and that is the distinction worth keeping: it sizes
    /// the GLOBAL pool, which an explicit `num_threads` bypasses. Dataset code that calls
    /// `par_iter` with no pool of its own DOES answer to `RAYON_NUM_THREADS`, so "which pool" is the
    /// question, not "which variable".
    ///
    /// libtorch's intra-op pool is a third, separate budget and a real cause of core burn in tests
    /// that run tensor ops - `TORCH_NUM_THREADS` is a no-op for it in this binary because a pre-main
    /// `.init_array` constructor has already pinned the pool by the time it would be read, leaving
    /// `OMP_NUM_THREADS` as the lever - but this module runs no model, so that budget is not the one
    /// operating here.
    ///
    /// A pool size is a CEILING, never a prediction. This fixture reaches 3.81 of its 4 because it
    /// is genuinely CPU-parallel - 170 s of user time over 46 s of wall - but the same reasoning
    /// applied to a pass that is dominated by serial GPU submission gives the wrong answer by more
    /// than a factor of two: `pretrain-calibration` was MEASURED at 1.85 cores with rayon's global
    /// pool unbounded at twelve, because its parallel batch build is a small share of the work. So
    /// read `user / wall` once per command shape and write the number down; do not infer a core
    /// count from a pool size, which is the error this comment previously made in the other
    /// direction.
    #[test]
    #[ignore = "reads the real long_data/bars corpus: minutes of CPU"]
    fn the_real_corpus_prices_itself_and_reports_its_own_over_levering_factor() {
        use crate::data::ingest::{bars_dir, PINNED_SPLIT_BOUNDS};
        use crate::torch::dataset::{BarCorpus, DEFAULT_MIN_BARS};

        let corpus =
            BarCorpus::load_with_bounds(&bars_dir(), RES_SECS, DEFAULT_MIN_BARS, PINNED_SPLIT_BOUNDS)
                .expect("the 300s corpus loads");
        let calibration =
            Arc::new(CostCalibration::from_corpus(&corpus, 4).expect("the corpus calibrates"));
        let universe = calibration.len();
        let unmeasured_symbols = calibration.unmeasured.len();
        let model = BarCostModel::new(calibration);
        let deciles = model.deciles();
        println!("== measured cost by liquidity decile ({universe} symbols) ==");
        for decile in &deciles {
            println!(
                "decile {} n={:4} adv=${:11.3}M px=${:8.2} CS={:9.3}bps (clamped {:9.3}, neg \
                 {:5.1}%) AR={:9.3}bps fees={:7.3}bps all-in@1%ADV={:9.3}bps unmeasured={}",
                decile.decile,
                decile.symbols,
                decile.median_adv_usd / 1.0e6,
                decile.median_harmonic_price,
                decile.median_cs_spread_bps,
                decile.median_cs_spread_bps_clamped,
                100.0 * decile.median_cs_negative_share,
                decile.median_ar_spread_bps,
                decile.median_fee_bps,
                decile.median_all_in_bps[PARTICIPATION_HEADLINE_SLOT],
                decile.unmeasured,
            );
        }

        // Held-out span only, so the cross-section is measured where the model is evaluated.
        let (slices, symbols) = corpus_panel(
            &corpus,
            PINNED_SPLIT_BOUNDS.0,
            PINNED_SPLIT_BOUNDS.1,
            1_024,
            8_192,
        )
        .expect("the held-out panel builds");
        println!(
            "== panel: {} instants x {} symbols ==",
            slices.len(),
            symbols.len()
        );
        let forecasts = scenario_forecasts(&slices, 0.595, 0xF00D);
        let correlation = cross_correlation(
            &slices,
            &forecasts,
            CorrelationConfig {
                gross_leverage: 4.0,
                ..CorrelationConfig::default()
            },
        )
        .expect("the cross-section measures");
        println!(
            "rho={:.4} median={:.4} first-factor share={:.4} effective rank={:.1} (on {} of {} \
             names)",
            correlation.mean_pairwise_corr,
            correlation.median_pairwise_corr,
            correlation.first_factor_share,
            correlation.effective_rank,
            correlation.eigen_symbols,
            correlation.panel_symbols,
        );
        for (horizon, corr, blocks) in &correlation.horizon_corr {
            println!("  horizon {horizon:3} bars: rho={corr:.4} on {blocks} gap-free blocks");
        }
        for book in &correlation.books {
            println!(
                "  {:>14}: realized={:9.4}bps independent={:9.4}bps over-lever={:7.3}x \
                 factor-exposure={:.3} net/gross={:+.3}",
                book.style.label(),
                book.realized_bps,
                book.independent_bps,
                book.correlation_factor,
                book.factor_exposure,
                book.net_gross_ratio,
            );
        }
        println!(
            "  equal-weight over-lever extrapolated to {universe} names: {:.2}x",
            correlation.breadth_extrapolated_factor(universe)
        );

        let spec = BookSpec::new(BookStyle::LongOnly, 4.0);
        let capacity: Vec<CapacityCurve> = IMPACT_K_GRID
            .iter()
            .map(|&k| {
                capacity_curve(&slices, &forecasts, &model.with_impact_k(k), spec, &AUM_GRID)
                    .expect("the capacity curve computes")
            })
            .collect();
        for curve in &capacity {
            println!(
                "k={:4}: gross={:+9.4}bps/bar fixed={:9.4}bps/bar crossing=${:.4}M unpriced \
                 legs={}",
                curve.impact_k,
                curve.points[0].gross_bps,
                curve.points[0].fixed_cost_bps,
                curve.zero_crossing_usd / 1.0e6,
                curve.unpriced_impact_legs,
            );
        }

        let dir = scratch_dir("real");
        write_cost_capacity_reports(
            &dir,
            &CostCapacityReport {
                deciles,
                capacity,
                correlation,
                universe,
                unmeasured_symbols,
            },
            "real corpus",
        )
        .expect("the battery writes");
        println!("reports written to {}", dir.display());
    }
}
