//! ONE book, ONE equity curve, calendar time: what the predictive law is worth to a
//! portfolio that actually exists.
//!
//! # The defect this replaces
//!
//! [`super::trade_bench`] answers "how would a log-optimal bettor size a single name?" and
//! answers it correctly. It then reports the MEAN of that answer over 256 independent
//! `(symbol, segment)` windows, each betting up to [`super::trade_bench::LEVERAGE_CAP`] of
//! its OWN wealth. There is no shared capital, no cross-sectional allocation and no
//! calendar: 256 books at 4x is 1,024x of gross exposure held simultaneously, and the mean
//! of 256 separate log-growth rates is not the growth rate of anything a trader can own.
//! At step 20000 of `bardist_v2` that framing produced `+4.3202` bps/bar net which, at the
//! bench's own hardcoded `93 * 252` bars per year, annualizes to `exp(10.125)` — about
//! 24,900x per year. That number is not a profit estimate. It is a proof that the framing
//! is broken.
//!
//! This module fixes the framing, and nothing else. The per-name predictive law and the
//! Kelly solve are [`super::trade_bench`]'s, called rather than copied — including the
//! prefix-free read of `p(r | past)` that keeps the
//! decision free of lookahead. What changes is everything above the single name:
//!
//! * **A panel, in calendar time.** Bars are grouped by `ts_ms` across symbols. Symbols do
//!   not share a calendar — halts, listings, delistings, thin pre-market prints — so
//!   absence is explicit and no price is ever forward-filled.
//! * **One capital constraint.** `sum_i |w_i| <= GROSS_CAP` over the WHOLE book. That is
//!   what binds a real trader: a prime broker limits gross, not per-name leverage.
//! * **One equity curve.** `W_{t+1} = W_t * (1 + sum_i w_i r_i - cost_t)`, compounded
//!   through the panel's own clock. A book that reaches zero is dead and stays dead.
//! * **Annualization from the MEASURED span.** The panel knows its own first and last
//!   instant and its own instant count, so bars-per-year is divided out of the data rather
//!   than asserted by a constant.
//!
//! # Absence, precisely
//!
//! A symbol is tradeable at instant `t` only when it has a bar at `t` AND a bar at the
//! instant immediately before it in the panel. Both are required because the payoff is a
//! close-to-close return: without a bar at `t-1` there is no price at which the position
//! could have been established, and inventing one is exactly the forward fill this refuses
//! to do. An absent symbol therefore has target weight zero, contributes nothing to the
//! payoff, and contributes only the turnover of unwinding whatever was held. That unwind is
//! charged at `t`; its cost model needs a liquidity estimate rather than a price, so it uses
//! the symbol's last observed dollar volume. No return and no price is ever fabricated.
//!
//! # The policies
//!
//! Weights come from ONE raw vector per policy, projected onto the gross ball. The
//! projection is proportional because the raw vector is a preference ordering with
//! magnitudes: scaling it preserves the relative bets, which is what a leverage limit does
//! to a book, whereas truncating names would silently change the portfolio.
//!
//! * [`Policy::Model`] — raw `w_i = f*_i`, the uncapped per-name Kelly fraction.
//! * [`Policy::MarketNeutral`] — the same vector with its mean subtracted, so the book
//!   carries no net factor exposure and its P&L is cross-sectional selection alone.
//! * [`Policy::Marginal`] — the unconditional-marginal NULL lifted to portfolio form: every
//!   present name gets the same Kelly fraction of the train-fitted law of `r`, which after
//!   the gross projection is the equal-weighted long book levered to the cap. It depends on
//!   no model weight, so a run that cannot beat it has bought nothing.
//! * [`Policy::EqualWeight`] — the same book UNLEVERED, gross exactly `1.0` at every cap.
//!   It is the market, and it is what fixes the units. Note that it coincides with
//!   [`Policy::Marginal`] at `GROSS_CAP = 1`; that is a fact about the null, not a bug.
//! * [`Policy::Oracle`] — perfect foresight under the SAME gross constraint. Maximizing
//!   `sum_i w_i r_i` over the L1 ball of radius `G` is a linear program whose solution is
//!   the whole budget on the single largest `|r_i|`, so the ceiling is degenerate and
//!   enormous. That is the honest ceiling of this constraint, and the model's share of it
//!   is the fraction of available cross-sectional edge the predictor captures.
//!
//! # What is deliberately absent
//!
//! No risk model, no covariance shrinkage, no optimizer. A Kelly vector projected onto a
//! gross ball is the smallest object that answers the question "does the predictive law
//! make money in one book?", and every additional layer is a place for a fitted parameter
//! to hide. The answer this produces is a LOWER bound in a known direction: a real book
//! would net the positions against a factor model and rebalance less often.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use shared::report::{read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{BarSupports, DOF_R, NUM_BAR_BINS};
use crate::torch::dataset::{BarCorpus, BarEndpoint};
use crate::torch::world_model::{world_model_metadata_path, BarWorldModel, BAR_MAX_CONTEXT};

use super::portfolio_cost::{
    BarCostModel, CostCalibration, DECILES, IMPACT_K, IMPACT_K_DEFAULT_SLOT, IMPACT_K_GRID,
    PARTICIPATION_GRID, PARTICIPATION_HEADLINE_SLOT,
};
use super::pretrain_stats::{BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CI_MASS};
use super::trade_bench::{
    bin_returns, forecast_r_probs, kelly_fraction, kelly_fractions, FREE_LEVERAGE, ROW_CHUNK,
};

// ---------------------------------------------------------------------------
// The contract shared with `portfolio_cost`
// ---------------------------------------------------------------------------

/// Everything that is TRADEABLE at one instant of calendar time.
///
/// `symbols[k]` indexes the panel's own symbol table ([`Panel::symbols`]) and
/// `realized_r[k]` is that symbol's realized LOG return over the bar ending at `ts_ms`,
/// measured against its close at the immediately preceding panel instant. A symbol that
/// lacks either bar is simply not in the vectors: absence is a shorter vector, never a
/// filled-forward price.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PanelSlice {
    pub ts_ms: i64,
    pub symbols: Vec<u32>,
    pub realized_r: Vec<f32>,
}

/// The predictive law of one slice, reduced to what a sizer needs.
///
/// Entries align positionally with [`PanelSlice::symbols`]. `kelly_f` is the UNCAPPED
/// log-optimal fraction of `p(r_i | past_i)`; the gross constraint is applied by the
/// portfolio engine, never here, so one forecast serves every point of the gross curve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PanelForecast {
    pub kelly_f: Vec<f32>,
    pub mean_r: Vec<f32>,
    pub var_r: Vec<f32>,
}

/// One leg's one-way cost in bps, and how much of it came from that leg's OWN measured
/// liquidity rather than from a cross-sectional stand-in.
///
/// The distinction is the whole reason this is not a bare `f32`. A cost model built from
/// measured bars cannot price every symbol-month: a month whose close never moved has no
/// volatility to put in the square-root impact law, and a month whose contiguous returns
/// show momentum rather than bid-ask bounce has no Roll spread. [`super::portfolio_cost`]
/// propagates both as NaN, and a NaN that reaches a wealth multiplier is silently either
/// free trading or a dead book. So every substitution is made explicit here, charged at the
/// cross-sectional median, and COUNTED — a frontier row whose net growth rests on stand-ins
/// is not a measurement, and it says so on its own row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegCost {
    pub bps: f32,
    /// The traded MONTH carried no usable spread or no positive volatility of its own, so
    /// the symbol's whole-span measurement stood in. Counted separately from the two below
    /// because it is still that symbol's own liquidity, five years of it instead of one
    /// month: a degradation of resolution, not of provenance.
    pub month_substituted: bool,
    /// The symbol's own spread was unmeasurable over its WHOLE span, so the cross-sectional
    /// median spread stood in. This leg's spread is not a measurement of this symbol.
    pub spread_substituted: bool,
    /// The symbol's own daily volatility was unmeasurable over its whole span, so the
    /// cross-sectional median volatility stood in and the square-root impact this leg paid
    /// is not a measurement of this symbol either.
    pub impact_substituted: bool,
}

impl LegCost {
    /// A cost priced entirely from the leg's own measured liquidity.
    #[inline]
    pub const fn own(bps: f32) -> Self {
        Self {
            bps,
            month_substituted: false,
            spread_substituted: false,
            impact_substituted: false,
        }
    }

    /// Whether any part of this leg's cost came from the cross-section rather than from the
    /// symbol. The month tier is deliberately NOT in this predicate; see the field.
    #[inline]
    pub const fn substituted(&self) -> bool {
        self.spread_substituted || self.impact_substituted
    }
}

pub trait CostModel {
    /// Per-symbol one-way cost in BPS for trading `notional_frac` of that symbol's ADV.
    fn cost_bps(&self, symbol: u32, ts_ms: i64, notional_frac: f32) -> f32;

    /// The same cost with its provenance attached, for the models that have one.
    ///
    /// Defaulted, so a model whose cost is a stated constant ([`FlatCost`]) needs no second
    /// implementation and widening this trait cannot change what any existing model charges.
    /// The default claims own-liquidity provenance because a constant is exactly as true of
    /// one symbol as of another; only a MEASURED model can have a stand-in to declare.
    fn leg_cost(&self, symbol: u32, ts_ms: i64, notional_frac: f32) -> LegCost {
        LegCost::own(self.cost_bps(symbol, ts_ms, notional_frac))
    }
}

/// A constant one-way cost, independent of size and symbol.
///
/// The defensible central estimate for a liquid US large cap at a 5-minute close is ~2 bps
/// (half a 1-2 bp spread, plus fees and immediate impact). It is the FLOOR of what trading
/// costs, so a strategy that dies here dies under any real model; [`PanelCost`] wraps
/// `portfolio_cost`'s size- and liquidity-aware measurement, which this one bounds from
/// below, and both are reported side by side so the gap is visible rather than substituted.
#[derive(Clone, Copy, Debug)]
pub struct FlatCost {
    pub bps: f32,
}

impl FlatCost {
    pub const fn new(bps: f32) -> Self {
        Self { bps }
    }
}

impl CostModel for FlatCost {
    fn cost_bps(&self, _symbol: u32, _ts_ms: i64, _notional_frac: f32) -> f32 {
        self.bps
    }
}

/// One-way cost charged when nothing better is supplied. See [`FlatCost`].
pub const DEFAULT_COST_BPS: f32 = 2.0;

/// Which parts of the measured cost an arm charges.
///
/// These are not three brokers; they are the cost DECOMPOSITION asked as three backtests,
/// because the only way to know which component decides the verdict is to remove it and re-run
/// the same book.
///
/// * [`Self::All`] is what a trader pays.
/// * [`Self::NoFees`] removes the commission and the regulatory fee. Commission enters as
///   `rate / price`, so it is the one component a different venue or a different price level
///   could genuinely remove.
/// * [`Self::NoImpact`] removes the square-root impact term and NOTHING else, leaving the
///   half-spread, the commission and the regulatory fee: every term measured from the bars and
///   none from an unfitted coefficient. This is the LOAD-BEARING arm. A conclusion that holds
///   here cannot be argued away by disputing [`IMPACT_K`], because no impact model enters it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostParts {
    All,
    NoFees,
    NoImpact,
}

impl CostParts {
    pub const fn label(self) -> &'static str {
        match self {
            Self::All => "all-in",
            Self::NoFees => "no fees",
            Self::NoImpact => "measured, impact-free",
        }
    }

    /// Whether this arm charges the square-root impact term, which is the only part of the
    /// measured cost that rests on an assumed coefficient.
    pub const fn charges_impact(self) -> bool {
        !matches!(self, Self::NoImpact)
    }

    pub const fn charges_fees(self) -> bool {
        !matches!(self, Self::NoFees)
    }
}

/// [`BarCostModel`] addressed by PANEL symbol id.
///
/// # The translation, which is the defect this fixes
///
/// `portfolio_cost` keys its calibration by the [`BarCorpus`] SERIES index — position in the
/// corpus's own symbol table — while everything in this module carries a panel-local id, the
/// position in [`Panel::symbols`]. The panel is a liquidity-ranked subset of the corpus, so
/// the two agree only by accident, and handing a `BarCostModel` straight to [`backtest`]
/// prices each name at some unrelated symbol's spread. That is why the whole session's
/// frontier ran under [`FlatCost`]: the real model was there and could not be wired up
/// without this one `Vec<u32>`, built once from [`Panel::series_of`].
///
/// The mapping is valid ONLY for the exact corpus load the panel was built from: series
/// indices are positions in a filtered symbol table, so changing the bars directory or the
/// `min_bars` admission shifts them. [`Self::new`] therefore takes the panel itself rather
/// than a caller-supplied index vector, and the calibration it wraps must have been measured
/// on that same corpus.
///
/// # What it charges
///
/// `half_spread + commission + regulatory + k * sigma_daily * sqrt(notional / ADV)`, every
/// term measured per symbol per calendar month, with `k` a stated literature default swept
/// over [`IMPACT_K_GRID`] rather than fitted, and with [`CostParts`] deciding which of the
/// four terms this instance charges at all. The participation the impact term is evaluated at
/// comes from [`Panel::adv_usd`] — a strictly trailing 20-day dollar volume — not from the
/// calibration's own ADV, so the size argument is causal at the bar being traded.
pub struct PanelCost {
    model: BarCostModel,
    /// `series[panel id]` is the corpus series index the calibration is keyed by.
    series: Vec<u32>,
    parts: CostParts,
}

impl PanelCost {
    /// Wrap `model` for `panel`. The calibration behind `model` must have been measured on
    /// the same corpus the panel was built from; see the type's own documentation.
    pub fn new(panel: &Panel, model: BarCostModel, parts: CostParts) -> Self {
        let series = (0..panel.symbols().len() as u32)
            .map(|id| panel.series_of(id) as u32)
            .collect();
        Self {
            model,
            series,
            parts,
        }
    }

    pub fn impact_k(&self) -> f64 {
        self.model.impact_k()
    }

    pub fn parts(&self) -> CostParts {
        self.parts
    }

    pub fn model(&self) -> &BarCostModel {
        &self.model
    }

    /// The same wrapper at another impact coefficient. A pointer copy of the calibration.
    pub fn with_impact_k(&self, impact_k: f64) -> Self {
        Self {
            model: self.model.with_impact_k(impact_k),
            series: self.series.clone(),
            parts: self.parts,
        }
    }

    /// The same wrapper charging a different subset of the cost. A pointer copy.
    pub fn with_parts(&self, parts: CostParts) -> Self {
        Self {
            model: self.model.clone(),
            series: self.series.clone(),
            parts,
        }
    }

    /// One leg's cost, with every stand-in declared.
    ///
    /// The three tiers are `portfolio_cost`'s own: the traded month, the symbol's whole span,
    /// the cross-sectional median. This reads them back out of the calibration rather than
    /// inferring them from the resolved number, because a substituted median is a perfectly
    /// finite cost and there is nothing in the arithmetic to notice.
    fn priced(&self, symbol: u32, ts_ms: i64, notional_frac: f32) -> LegCost {
        let Some(&series) = self.series.get(symbol as usize) else {
            // A panel id with no series behind it cannot happen: the vector is built from the
            // panel's own symbol table. If it ever does, it is a wiring error rather than a
            // pricing question, and a cost of NaN trips `backtest`'s finite-cost assertion
            // with the symbol and the instant in the message.
            return LegCost {
                bps: f32::NAN,
                month_substituted: true,
                spread_substituted: true,
                impact_substituted: true,
            };
        };
        let entry = self.model.calibration().symbols.get(series as usize);
        let bucket = entry.and_then(|s| s.bucket_at(ts_ms));
        let pooled = entry.map(|s| &s.pooled);
        let month_measured = bucket.is_some_and(|b| {
            b.measured_spread_bps().is_some() && b.sigma_daily.is_finite() && b.sigma_daily > 0.0
        });
        let own_sigma = [bucket, pooled]
            .into_iter()
            .flatten()
            .any(|b| b.sigma_daily.is_finite() && b.sigma_daily > 0.0);

        let resolved = self.model.resolve(series, ts_ms);
        let fees = if self.parts.charges_fees() {
            resolved.commission_bps + resolved.regulatory_bps
        } else {
            0.0
        };
        let impact = if !self.parts.charges_impact() {
            0.0
        } else if own_sigma {
            resolved.impact_bps(f64::from(notional_frac))
        } else {
            // The symbol's own volatility is unmeasurable at every tier, so the impact this
            // leg pays is the cross-sectional median's. That is a LOWER bound in a known
            // direction — a name with no measurable volatility is at the thin end of the
            // universe, not the middle of it — which is why the leg is counted and the count
            // travels with every net figure derived from it.
            1.0e4
                * self.model.impact_k()
                * self.model.calibration().fallback_sigma_daily
                * f64::from(notional_frac).max(0.0).sqrt()
        };
        LegCost {
            bps: (resolved.half_spread_bps + fees + impact) as f32,
            month_substituted: !month_measured,
            spread_substituted: resolved.spread_fallback,
            // An arm that charges no impact at all has no volatility stand-in to declare: the
            // measurement it could not make is one it never used.
            impact_substituted: self.parts.charges_impact() && !own_sigma,
        }
    }
}

impl CostModel for PanelCost {
    fn cost_bps(&self, symbol: u32, ts_ms: i64, notional_frac: f32) -> f32 {
        self.priced(symbol, ts_ms, notional_frac).bps
    }

    fn leg_cost(&self, symbol: u32, ts_ms: i64, notional_frac: f32) -> LegCost {
        self.priced(symbol, ts_ms, notional_frac)
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Gross-exposure caps the whole backtest is reported at, in units of book equity.
///
/// `1.0` is a fully invested unlevered book, `2.0` is Reg-T, `4.0` is the practical
/// portfolio-margin limit. A single headline at one cap is uninterpretable once the
/// constraint binds every bar, which is why [`PortfolioMetrics::bound_fraction`] is
/// reported beside every point.
pub const GROSS_CAPS: [f64; 3] = [1.0, 2.0, 4.0];

/// Cap the single-number headline is quoted at: Reg-T, the leverage a book can actually
/// hold overnight without portfolio margin.
pub const DEFAULT_GROSS_CAP: f64 = 2.0;

/// Slot of [`DEFAULT_GROSS_CAP`] in [`GROSS_CAPS`], so the curve passes exactly through the
/// headline rather than near it.
pub const DEFAULT_GROSS_SLOT: usize = 1;

const _: () = assert!(GROSS_CAPS[DEFAULT_GROSS_SLOT] == DEFAULT_GROSS_CAP);

/// Bars of own history a belief must stand on before it is allowed to size a position.
///
/// Half of [`BAR_MAX_CONTEXT`]: the trunk is trained at contexts up to 2048 and the
/// remaining half of each block is what the block emits, so every emitted belief has at
/// least this much causal history and no belief is ever produced from a short prefix.
pub const BELIEF_PRE_CONTEXT: i64 = BAR_MAX_CONTEXT / 2;

/// Beliefs emitted per trunk block. `BELIEF_PRE_CONTEXT + BELIEF_EMIT == BAR_MAX_CONTEXT`,
/// so a block is exactly one full-context forward pass and positions never run past the
/// range the PoPE phases were trained on.
pub const BELIEF_EMIT: i64 = BAR_MAX_CONTEXT - BELIEF_PRE_CONTEXT;

/// Trailing bars the per-symbol dollar volume is averaged over, for the ADV a cost model
/// prices size against. Twenty trading days of 5-minute bars, the standard ADV window.
pub const ADV_TRAILING_BARS: usize = 20 * 93;

/// Milliseconds in a Julian year, for turning a measured panel span into years.
const MS_PER_YEAR: f64 = 365.25 * 86_400_000.0;
const MS_PER_DAY: i64 = 86_400_000;

/// Relative slack allowed when asserting the gross constraint, absorbing the f64 rounding of
/// the projection itself and nothing else.
const GROSS_TOLERANCE: f64 = 1e-9;

/// Log10 wealth a DEAD book is drawn at, for the picture alone.
///
/// `-9` is one cent of a ten-million-dollar book: below it the book does not exist in any
/// sense a broker recognizes. A dead book's true log wealth is `-inf`, which every renderer
/// drops as non-finite — and a dropped point looks exactly like a metric that was never
/// measured, which is the one confusion this repo's reports refuse to allow. The FACT lives
/// in [`PortfolioMetrics::ruined_at_instant`], never in the floor.
pub const RUIN_FLOOR_LOG10: f64 = -9.0;

// ---------------------------------------------------------------------------
// The panel
// ---------------------------------------------------------------------------

/// How much of the held-out calendar to trade, and which names.
#[derive(Clone, Debug)]
pub struct PanelConfig {
    /// First instant the book may trade. Bars before it are history only.
    pub start_ts_ms: i64,
    /// Exclusive last instant.
    pub end_ts_ms: i64,
    /// At most this many symbols, ranked by dollar volume measured STRICTLY BEFORE
    /// `start_ts_ms`. Ranking on the traded span itself would be lookahead: survivorship
    /// and volume spikes both correlate with the returns being scored.
    pub max_symbols: usize,
    /// Bars of own history a symbol needs before its first tradeable instant.
    ///
    /// One is the arithmetic minimum — a close-to-close return needs a predecessor close —
    /// and that is all the panel itself enforces, because a panel traded by the
    /// unconditional null needs no belief at all. A panel that will be handed to
    /// [`model_forecasts`] needs [`BELIEF_PRE_CONTEXT`] more, which is what
    /// [`PanelConfig::new`] sets and what the forecaster re-checks against the bars it is
    /// actually asked for.
    pub min_history: usize,
    /// At most this many instants, taken as a contiguous prefix of the span.
    pub max_instants: usize,
}

impl PanelConfig {
    /// A panel over the whole held-out span, bounded in both directions so a backtest is
    /// affordable at every validation.
    pub fn new(bounds: (i64, i64), max_symbols: usize, max_instants: usize) -> Self {
        Self {
            start_ts_ms: bounds.0,
            end_ts_ms: bounds.1,
            max_symbols,
            min_history: (BELIEF_PRE_CONTEXT + 1) as usize,
            max_instants,
        }
    }
}

/// How many names were actually tradeable, which is the first thing that decides whether a
/// cross-sectional number means anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Breadth {
    pub mean: f64,
    pub min: usize,
    pub max: usize,
}

/// A calendar-aligned cross-section of held-out bars.
///
/// The panel owns the symbol table, so the `u32` in [`PanelSlice::symbols`] is a stable
/// index into [`Panel::symbols`] for every consumer, including the cost model.
#[derive(Clone, Debug)]
pub struct Panel {
    symbols: Vec<String>,
    /// Corpus series index of each panel symbol, so a forecaster can find its bars.
    series: Vec<usize>,
    slices: Vec<PanelSlice>,
    /// `bar_index[t][k]` is the corpus bar index of `slices[t].symbols[k]`.
    bar_index: Vec<Vec<u32>>,
    /// `dollar_volume[t][k]` is the trailing mean dollar volume PER BAR of that symbol,
    /// measured strictly before the bar being traded. Multiplied by the panel's measured
    /// bars-per-day to become an ADV.
    dollar_volume: Vec<Vec<f32>>,
    /// Leading eigenvector of the panel's own realized return correlation, unit L2 norm.
    ///
    /// A book sized on per-name Kelly is sized as if the names were independent. They are
    /// not, and this is the direction in which they are least independent, so the book's
    /// projection onto it is the single number that says how much of the "diversified"
    /// position is one bet.
    first_factor: Vec<f32>,
    /// Share of the panel's total cross-sectional return variance the leading eigenvector
    /// carries. One name's worth over the whole panel would be `1 / breadth`.
    first_factor_share: f64,
    /// Distinct UTC dates covered, for turnover-per-day and bars-per-day.
    trading_days: usize,
}

impl Panel {
    /// Group the corpus into calendar instants over `config`'s span.
    ///
    /// Two-pass: rank the universe on liquidity measured strictly before the span, then walk
    /// the union of timestamps of the survivors. The union is the panel's clock; a symbol
    /// missing from an instant is missing from that instant's vectors.
    pub fn build(corpus: &BarCorpus, config: &PanelConfig) -> Result<Self> {
        ensure!(
            config.start_ts_ms < config.end_ts_ms,
            "panel span is empty: [{}, {})",
            config.start_ts_ms,
            config.end_ts_ms
        );
        ensure!(
            config.min_history >= 1,
            "a tradeable bar needs a predecessor close, so min_history must be at least 1"
        );
        ensure!(config.max_symbols > 0, "a panel needs at least one symbol");
        ensure!(config.max_instants > 0, "a panel needs at least one instant");

        let ranked = rank_by_prior_liquidity(corpus, config);
        ensure!(
            !ranked.is_empty(),
            "no symbol has {} bars before {} in a corpus of {} series",
            config.min_history,
            config.start_ts_ms,
            corpus.series_count()
        );
        let chosen = &ranked[..ranked.len().min(config.max_symbols)];

        // The panel CLOCK: every instant at which any chosen symbol printed. Its first
        // entry can never be traded — nothing has a predecessor close there — so it is a
        // reference instant and the tradeable slices are `clock[1..]`. Keeping it as an
        // empty slice would put a permanent breadth-zero point into every average.
        let mut ticks = BTreeSet::new();
        for &(series, _) in chosen {
            for bar in span_bars(corpus, series, config) {
                ticks.insert(corpus.ts_ms(series, bar));
            }
        }
        let clock: Vec<i64> = ticks.into_iter().take(config.max_instants + 1).collect();
        ensure!(
            clock.len() >= 3,
            "a clock of {} instants leaves fewer than two tradeable ones after the \
             reference instant",
            clock.len()
        );
        let slot: BTreeMap<i64, usize> = clock
            .iter()
            .enumerate()
            .map(|(index, &ts)| (ts, index))
            .collect();
        let tradeable = clock.len() - 1;

        let mut symbols = Vec::with_capacity(chosen.len());
        let mut series_of = Vec::with_capacity(chosen.len());
        let mut slices: Vec<PanelSlice> = clock[1..]
            .iter()
            .map(|&ts_ms| PanelSlice {
                ts_ms,
                ..PanelSlice::default()
            })
            .collect();
        let mut bar_index = vec![Vec::new(); tradeable];
        let mut dollar_volume = vec![Vec::new(); tradeable];

        for &(series, _) in chosen {
            let id = symbols.len() as u32;
            let bars = corpus.bars(series);
            let mut trailing = TrailingDollarVolume::new(bars, config.start_ts_ms);
            let mut placed = false;
            for bar in span_bars(corpus, series, config) {
                let Some(&tick) = slot.get(&bars[bar].ts()) else {
                    // Past `max_instants`; the clock stops here for every symbol at once.
                    break;
                };
                trailing.advance_to(bars, bar);
                if tick == 0 || bar < config.min_history {
                    continue;
                }
                // Absence, in its two forms: no bar at this instant (the symbol never
                // reaches this branch) and no bar at the one before it (the close the
                // position would have been established at does not exist). Filling either
                // in is the defect this refuses.
                if bars[bar - 1].ts() != clock[tick - 1] {
                    continue;
                }
                let (prev, close) = (bars[bar - 1].close, bars[bar].close);
                if !(prev > 0.0 && close > 0.0) {
                    continue;
                }
                let realized = (f64::from(close) / f64::from(prev)).ln();
                if !realized.is_finite() {
                    continue;
                }
                let t = tick - 1;
                slices[t].symbols.push(id);
                slices[t].realized_r.push(realized as f32);
                bar_index[t].push(bar as u32);
                dollar_volume[t].push(trailing.mean_per_bar());
                placed = true;
            }
            if placed {
                symbols.push(corpus.symbol(series).to_string());
                series_of.push(series);
            }
        }
        ensure!(
            !symbols.is_empty(),
            "every candidate symbol was absent at every instant of [{}, {}); the span may \
             predate the corpus",
            config.start_ts_ms,
            config.end_ts_ms
        );

        let trading_days = slices
            .iter()
            .map(|s| s.ts_ms.div_euclid(MS_PER_DAY))
            .collect::<BTreeSet<_>>()
            .len();
        let (first_factor, first_factor_share) = estimate_first_factor(&slices, symbols.len());
        Ok(Self {
            symbols,
            series: series_of,
            slices,
            bar_index,
            dollar_volume,
            first_factor,
            first_factor_share,
            trading_days,
        })
    }

    /// Assemble a panel from explicit parts, for fixtures that need a known answer.
    ///
    /// Validates the same invariants [`Self::build`] establishes, so a fixture cannot test
    /// the engine against a panel the builder could never produce.
    pub fn from_parts(
        symbols: Vec<String>,
        slices: Vec<PanelSlice>,
        dollar_volume: Vec<Vec<f32>>,
    ) -> Result<Self> {
        ensure!(!symbols.is_empty(), "a panel needs at least one symbol");
        ensure!(slices.len() >= 2, "a panel needs at least two instants");
        ensure!(
            slices.len() == dollar_volume.len(),
            "dollar volume must align with the slices"
        );
        let mut bar_index = Vec::with_capacity(slices.len());
        for (t, slice) in slices.iter().enumerate() {
            ensure!(
                slice.symbols.len() == slice.realized_r.len()
                    && slice.symbols.len() == dollar_volume[t].len(),
                "slice {t} is ragged: {} symbols, {} returns, {} volumes",
                slice.symbols.len(),
                slice.realized_r.len(),
                dollar_volume[t].len()
            );
            ensure!(
                slice.symbols.iter().all(|&s| (s as usize) < symbols.len()),
                "slice {t} names a symbol outside the table of {}",
                symbols.len()
            );
            ensure!(
                t == 0 || slices[t - 1].ts_ms < slice.ts_ms,
                "panel instants must strictly increase"
            );
            bar_index.push(vec![0u32; slice.symbols.len()]);
        }
        let trading_days = slices
            .iter()
            .map(|s| s.ts_ms.div_euclid(MS_PER_DAY))
            .collect::<BTreeSet<_>>()
            .len();
        let series = (0..symbols.len()).collect();
        let (first_factor, first_factor_share) = estimate_first_factor(&slices, symbols.len());
        Ok(Self {
            symbols,
            series,
            slices,
            bar_index,
            dollar_volume,
            first_factor,
            first_factor_share,
            trading_days,
        })
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    pub fn slices(&self) -> &[PanelSlice] {
        &self.slices
    }

    /// Leading eigenvector of the realized return correlation, indexed by panel symbol id.
    pub fn first_factor(&self) -> &[f32] {
        &self.first_factor
    }

    /// Variance share of [`Self::first_factor`]. `1.0` would be a panel of one asset.
    pub fn first_factor_share(&self) -> f64 {
        self.first_factor_share
    }

    pub fn instants(&self) -> usize {
        self.slices.len()
    }

    /// Corpus series index of panel symbol `id`.
    pub fn series_of(&self, id: u32) -> usize {
        self.series[id as usize]
    }

    /// Corpus bar index of `slices[t].symbols[k]`.
    pub fn bar_index(&self, t: usize, k: usize) -> u32 {
        self.bar_index[t][k]
    }

    /// Wall-clock span actually covered, in milliseconds.
    pub fn span_ms(&self) -> i64 {
        self.slices
            .last()
            .zip(self.slices.first())
            .map_or(0, |(last, first)| last.ts_ms - first.ts_ms)
    }

    /// The panel's own span in years. Every annualization divides by THIS, never by a
    /// bars-per-year constant.
    pub fn span_years(&self) -> f64 {
        self.span_ms() as f64 / MS_PER_YEAR
    }

    /// Instants per year, measured: `instants / span_years`. A panel that trades only the
    /// regular session reports a smaller number than one that trades pre-market, which is
    /// exactly right and is what a hardcoded `93 * 252` cannot express.
    pub fn instants_per_year(&self) -> f64 {
        let years = self.span_years();
        if years > 0.0 {
            self.slices.len() as f64 / years
        } else {
            f64::NAN
        }
    }

    /// Distinct UTC dates the panel touches. US session bars all open inside one UTC date
    /// (04:00 ET is 08:00/09:00 UTC and the last post-market open is 19:55 ET), so a UTC
    /// date is a trading date here.
    pub fn trading_days(&self) -> usize {
        self.trading_days
    }

    pub fn instants_per_day(&self) -> f64 {
        if self.trading_days > 0 {
            self.slices.len() as f64 / self.trading_days as f64
        } else {
            f64::NAN
        }
    }

    pub fn breadth(&self) -> Breadth {
        let counts = self.slices.iter().map(|s| s.symbols.len());
        let total: usize = counts.clone().sum();
        Breadth {
            mean: total as f64 / self.slices.len() as f64,
            min: counts.clone().min().unwrap_or(0),
            max: counts.max().unwrap_or(0),
        }
    }

    /// Dollar ADV of `slices[t].symbols[k]`: trailing mean dollar volume per bar, scaled to
    /// the panel's own measured instants per day.
    pub fn adv_usd(&self, t: usize, k: usize) -> f32 {
        (f64::from(self.dollar_volume[t][k]) * self.instants_per_day()) as f32
    }
}

/// Power iterations used to extract the leading eigenvector. The gap between the first and
/// second eigenvalue of an equity correlation matrix is large - the market factor is not a
/// close call - so convergence is fast and 128 is generous.
const FACTOR_ITERATIONS: usize = 128;

/// Leading eigenvector of the panel's realized return correlation, and its variance share.
///
/// The panel is RAGGED: a symbol has no return at an instant where it did not print. The
/// estimator standardizes each name over its OWN present instants and treats an absent
/// entry as a zero deviation, which is the pairwise-complete covariance shrunk toward zero
/// for names that are often missing. That shrinkage is the honest direction: a name we
/// rarely observe together with the rest genuinely offers less evidence of comovement, and
/// the alternative - imputing a return where no trade happened - is the forward-fill this
/// whole module exists to refuse.
///
/// Returns a unit-L2 loading vector indexed by panel symbol id, and `lambda_1 / trace`,
/// where the trace is the summed present-fraction of the names. A panel whose returns are
/// pure independent noise puts `~1 / breadth` of the variance on any one direction; the
/// measured value is how far from that the market actually is.
fn estimate_first_factor(slices: &[PanelSlice], names: usize) -> (Vec<f32>, f64) {
    if names == 0 || slices.is_empty() {
        return (vec![0.0; names], f64::NAN);
    }
    let instants = slices.len();
    // Per-name mean and standard deviation over the instants where the name printed.
    let mut count = vec![0u32; names];
    let mut sum = vec![0.0f64; names];
    let mut sum_sq = vec![0.0f64; names];
    for slice in slices {
        for (k, &id) in slice.symbols.iter().enumerate() {
            let r = f64::from(slice.realized_r[k]);
            if !r.is_finite() {
                continue;
            }
            let id = id as usize;
            count[id] += 1;
            sum[id] += r;
            sum_sq[id] += r * r;
        }
    }
    let mut mean = vec![0.0f64; names];
    let mut inv_sd = vec![0.0f64; names];
    for id in 0..names {
        // Two observations is the minimum at which a deviation is defined at all; a name
        // seen once or never contributes a zero column and cannot load on any factor.
        if count[id] < 2 {
            continue;
        }
        let n = f64::from(count[id]);
        let m = sum[id] / n;
        let var = (sum_sq[id] / n - m * m).max(0.0);
        if var <= 0.0 {
            continue;
        }
        mean[id] = m;
        inv_sd[id] = var.sqrt().recip();
    }

    // The standardized panel, kept sparse: one row per instant, only the names that printed.
    let rows: Vec<Vec<(u32, f64)>> = slices
        .iter()
        .map(|slice| {
            slice
                .symbols
                .iter()
                .enumerate()
                .filter_map(|(k, &id)| {
                    let scale = inv_sd[id as usize];
                    let r = f64::from(slice.realized_r[k]);
                    if scale == 0.0 || !r.is_finite() {
                        return None;
                    }
                    Some((id, (r - mean[id as usize]) * scale))
                })
                .collect()
        })
        .collect();
    let trace: f64 = count
        .iter()
        .zip(&inv_sd)
        .filter(|(_, &s)| s != 0.0)
        .map(|(&c, _)| f64::from(c) / instants as f64)
        .sum();
    if trace <= 0.0 {
        return (vec![0.0; names], f64::NAN);
    }

    // Power iteration on `C = Z^T Z / instants`, never forming `C`. Starting from the
    // all-ones direction rather than a random one keeps this deterministic and starts it
    // pointed at the market, which is what it converges to anyway.
    let mut v = vec![1.0f64 / (names as f64).sqrt(); names];
    let mut scratch = vec![0.0f64; names];
    let mut lambda = 0.0f64;
    for _ in 0..FACTOR_ITERATIONS {
        scratch.iter_mut().for_each(|x| *x = 0.0);
        for row in &rows {
            let mut dot = 0.0f64;
            for &(id, z) in row {
                dot += z * v[id as usize];
            }
            if dot == 0.0 {
                continue;
            }
            for &(id, z) in row {
                scratch[id as usize] += z * dot;
            }
        }
        for x in scratch.iter_mut() {
            *x /= instants as f64;
        }
        let norm = scratch.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            return (vec![0.0; names], f64::NAN);
        }
        lambda = norm;
        for (dst, src) in v.iter_mut().zip(&scratch) {
            *dst = src / norm;
        }
    }
    // Sign is arbitrary in an eigenvector; pin it so the market factor reads as long.
    if v.iter().sum::<f64>() < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
    (
        v.iter().map(|&x| x as f32).collect(),
        (lambda / trace).clamp(0.0, 1.0),
    )
}

/// Bars of `series` inside the panel span, as corpus indices.
fn span_bars(corpus: &BarCorpus, series: usize, config: &PanelConfig) -> std::ops::Range<usize> {
    let bars = corpus.bars(series);
    let lo = bars.partition_point(|b| b.ts() < config.start_ts_ms);
    let hi = bars.partition_point(|b| b.ts() < config.end_ts_ms);
    lo..hi
}

/// Rank the universe by dollar volume over the [`ADV_TRAILING_BARS`] bars STRICTLY BEFORE
/// the panel span, dropping symbols too short to carry a belief.
fn rank_by_prior_liquidity(corpus: &BarCorpus, config: &PanelConfig) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = (0..corpus.series_count())
        .filter_map(|series| {
            let bars = corpus.bars(series);
            let cut = bars.partition_point(|b| b.ts() < config.start_ts_ms);
            if cut <= config.min_history {
                return None;
            }
            let from = cut.saturating_sub(ADV_TRAILING_BARS);
            let dollars: f64 = bars[from..cut]
                .iter()
                .map(|bar| f64::from(bar.volume) * f64::from(bar.vwap))
                .filter(|v| v.is_finite() && *v > 0.0)
                .sum();
            (dollars > 0.0).then_some((series, dollars / (cut - from) as f64))
        })
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
}

/// Causal rolling mean of dollar volume over the trailing [`ADV_TRAILING_BARS`] bars,
/// advanced bar by bar. Strictly past: the bar being traded is never in its own average.
struct TrailingDollarVolume {
    sum: f64,
    count: usize,
    next: usize,
}

impl TrailingDollarVolume {
    fn new(bars: &[shared::bars::PackedBar], start_ts_ms: i64) -> Self {
        let cut = bars.partition_point(|b| b.ts() < start_ts_ms);
        let from = cut.saturating_sub(ADV_TRAILING_BARS);
        let mut sum = 0.0;
        let mut count = 0usize;
        for bar in &bars[from..cut] {
            let dollars = f64::from(bar.volume) * f64::from(bar.vwap);
            if dollars.is_finite() && dollars > 0.0 {
                sum += dollars;
                count += 1;
            }
        }
        Self {
            sum,
            count,
            next: cut,
        }
    }

    /// Fold in every bar strictly before `bar` that has not been folded in yet.
    fn advance_to(&mut self, bars: &[shared::bars::PackedBar], bar: usize) {
        while self.next < bar {
            let dollars = f64::from(bars[self.next].volume) * f64::from(bars[self.next].vwap);
            if dollars.is_finite() && dollars > 0.0 {
                self.sum += dollars;
                self.count += 1;
            }
            self.next += 1;
        }
    }

    fn mean_per_bar(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum / self.count as f64) as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Forecasts
// ---------------------------------------------------------------------------

/// The unconditional-marginal NULL, lifted to the panel.
///
/// One number — the log-optimal fraction of the train-fitted unconditional law of `r` —
/// broadcast to every present symbol at every instant. It reads no model weight and no
/// belief, which is exactly what makes it the null.
pub fn marginal_forecasts(panel: &Panel, supports: &BarSupports) -> Vec<PanelForecast> {
    let returns = bin_returns(supports);
    let masses = supports.bin_masses(DOF_R);
    let free = kelly_fraction(masses, &returns, FREE_LEVERAGE) as f32;
    let mean: f64 = masses.iter().zip(&returns).map(|(p, r)| p * r).sum();
    let second: f64 = masses.iter().zip(&returns).map(|(p, r)| p * r * r).sum();
    let var = (second - mean * mean).max(0.0) as f32;
    panel
        .slices()
        .iter()
        .map(|slice| {
            let n = slice.symbols.len();
            PanelForecast {
                kelly_f: vec![free; n],
                mean_r: vec![mean as f32; n],
                var_r: vec![var; n],
            }
        })
        .collect()
}

/// Per-name `p(r_i | past_i)` for every entry of the panel, reduced to Kelly, mean and
/// variance.
///
/// The distribution is [`forecast_r_probs`]'s — the head's own `r` row, which is
/// `p(r | past)` outright because `r` heads the emission chain — and the fraction is
/// [`kelly_fractions`]'s, solved uncapped. Both are called, not reimplemented: the
/// no-lookahead property is a
/// property of those functions' signatures and survives being called from here.
///
/// Beliefs are produced in blocks of [`BAR_MAX_CONTEXT`] bars that emit only their last
/// [`BELIEF_EMIT`] positions, so every belief that sizes a position stands on at least
/// [`BELIEF_PRE_CONTEXT`] bars of its own causal history.
pub fn model_forecasts(
    model: &BarWorldModel,
    corpus: &BarCorpus,
    panel: &Panel,
    res_secs: u32,
) -> Result<Vec<PanelForecast>> {
    let supports = model
        .supports_for(res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {res_secs}s"))?;
    let device = model.device();
    let returns_host = bin_returns(supports);
    let returns = Tensor::from_slice(&returns_host)
        .view([1, NUM_BAR_BINS])
        .to_device(device);

    let mut out: Vec<PanelForecast> = panel
        .slices()
        .iter()
        .map(|slice| {
            let n = slice.symbols.len();
            PanelForecast {
                kelly_f: vec![f32::NAN; n],
                mean_r: vec![f32::NAN; n],
                var_r: vec![f32::NAN; n],
            }
        })
        .collect();

    // Where each (symbol, bar) lands in the output, grouped by symbol and in bar order.
    let mut wanted: Vec<Vec<(u32, usize, usize)>> = vec![Vec::new(); panel.symbols().len()];
    for (t, slice) in panel.slices().iter().enumerate() {
        for (k, &id) in slice.symbols.iter().enumerate() {
            wanted[id as usize].push((panel.bar_index(t, k), t, k));
        }
    }

    for (id, targets) in wanted.iter().enumerate() {
        if targets.is_empty() {
            continue;
        }
        let series = panel.series_of(id as u32);
        let (first, last) = (targets[0].0 as usize, targets[targets.len() - 1].0 as usize);
        // The panel only guarantees a predecessor close. A BELIEF additionally needs its
        // causal history to exist, and the failure without this check is a `dof_window`
        // error deep inside the block loop rather than a statement about the panel.
        ensure!(
            first as i64 >= BELIEF_PRE_CONTEXT + 1,
            "{} is tradeable from bar {first}, which cannot carry a belief with \
             {BELIEF_PRE_CONTEXT} bars of causal history plus the predecessor close the \
             encoder needs; build the panel with a larger `min_history`",
            panel.symbols()[id]
        );
        // Bar index -> slot in this symbol's target list, so a block can scatter its
        // contiguous belief run back onto the sparse instants the symbol was present at.
        let slot: BTreeMap<usize, usize> = targets
            .iter()
            .enumerate()
            .map(|(index, &(bar, _, _))| (bar as usize, index))
            .collect();

        let mut cursor = first;
        while cursor <= last {
            let emit = BELIEF_EMIT.min((last - cursor + 1) as i64);
            // Inputs are the bars that PRECEDE the emitted predictions, plus the causal
            // history each belief must stand on.
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
            let block = beliefs
                .narrow(1, len - emit, emit)
                .reshape([emit, latent])
                .contiguous();

            let mut start = 0i64;
            while start < emit {
                let rows = ROW_CHUNK.min(emit - start);
                let probs = forecast_r_probs(model.head(), &block.narrow(0, start, rows));
                let kelly = host_f32(&kelly_fractions(&probs, &returns, FREE_LEVERAGE));
                let probs = probs.to_kind(Kind::Double);
                let mean = probs.matmul(&returns.reshape([NUM_BAR_BINS, 1])).squeeze();
                let second = probs
                    .matmul(&(&returns * &returns).reshape([NUM_BAR_BINS, 1]))
                    .squeeze();
                let var = (second - &mean * &mean).clamp_min(0.0);
                let (mean, var) = (host_f32(&mean.reshape([-1])), host_f32(&var.reshape([-1])));
                for row in 0..rows as usize {
                    let bar = cursor + (start as usize) + row;
                    let Some(&index) = slot.get(&bar) else {
                        continue;
                    };
                    let (_, t, k) = targets[index];
                    out[t].kelly_f[k] = kelly[row];
                    out[t].mean_r[k] = mean[row];
                    out[t].var_r[k] = var[row];
                }
                start += rows;
            }
            cursor += emit as usize;
        }
    }

    for (t, forecast) in out.iter().enumerate() {
        ensure!(
            forecast.kelly_f.iter().all(|f| f.is_finite()),
            "instant {t} has a symbol the belief pass never reached, so its position would \
             be sized from a NaN"
        );
    }
    Ok(out)
}

fn host_f32(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(tensor.to_kind(Kind::Float).reshape([-1]).to(Device::Cpu))
        .expect("a 1-D float tensor converts to a host vector")
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Model,
    MarketNeutral,
    Marginal,
    EqualWeight,
    Oracle,
}

pub const POLICIES: [Policy; 5] = [
    Policy::Model,
    Policy::MarketNeutral,
    Policy::Marginal,
    Policy::EqualWeight,
    Policy::Oracle,
];

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Policy::Model => "model",
            Policy::MarketNeutral => "model, market-neutral",
            Policy::Marginal => "marginal null",
            Policy::EqualWeight => "equal weight",
            Policy::Oracle => "oracle",
        }
    }

    /// Gross budget this policy may use at `cap`.
    ///
    /// Everything is capped at `cap` except [`Policy::EqualWeight`], which is the UNLEVERED
    /// market by definition and stays at gross `1.0` so the same benchmark line runs through
    /// every point of the gross curve.
    pub fn gross_budget(self, cap: f64) -> f64 {
        match self {
            Policy::EqualWeight => 1.0f64.min(cap),
            _ => cap,
        }
    }

    /// The raw, unconstrained preference vector for one instant.
    ///
    /// `budget` is the gross the projection will allow. Only the oracle reads it: every
    /// other policy states a preference and lets the projection scale it DOWN, which is
    /// what a leverage limit does, while the oracle's answer to "what is the most this
    /// constraint could have earned" is the constraint itself.
    fn raw_weights(
        self,
        slice: &PanelSlice,
        model: &PanelForecast,
        marginal: &PanelForecast,
        budget: f64,
        out: &mut Vec<f64>,
    ) {
        out.clear();
        match self {
            Policy::Model => out.extend(model.kelly_f.iter().map(|f| f64::from(*f))),
            Policy::MarketNeutral => {
                let n = model.kelly_f.len();
                if n == 0 {
                    return;
                }
                // Breadth-weighted mean, which for an equal-count cross-section is the plain
                // mean of the present names: the book carries no net exposure by
                // construction rather than by an optimizer's constraint.
                let mean =
                    model.kelly_f.iter().map(|f| f64::from(*f)).sum::<f64>() / n as f64;
                out.extend(model.kelly_f.iter().map(|f| f64::from(*f) - mean));
            }
            Policy::Marginal => out.extend(marginal.kelly_f.iter().map(|f| f64::from(*f))),
            Policy::EqualWeight => out.extend(std::iter::repeat_n(1.0, slice.symbols.len())),
            Policy::Oracle => {
                // The payoff is LINEAR in the weights, so maximizing it over the L1 ball of
                // radius `G` puts the whole budget on one name. The name is the largest
                // SIMPLE return, not the largest log return: `expm1` is monotone but not
                // odd, so a `+2%` log move pays more than a `-2%` one pays short and the two
                // orderings genuinely differ. Degenerate, uninvestable, and exactly the
                // ceiling of this constraint.
                out.extend(std::iter::repeat_n(0.0, slice.realized_r.len()));
                let best = slice
                    .realized_r
                    .iter()
                    .enumerate()
                    .map(|(index, r)| (index, f64::from(*r).exp_m1()))
                    .filter(|(_, payoff)| payoff.is_finite() && *payoff != 0.0)
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()));
                if let Some((index, payoff)) = best {
                    out[index] = budget * payoff.signum();
                }
            }
        }
    }
}

/// Both forecast streams a policy set is scored against, on the SAME panel.
#[derive(Clone, Copy, Debug)]
pub struct PolicyInputs<'a> {
    pub model: &'a [PanelForecast],
    pub marginal: &'a [PanelForecast],
}

/// Scale `raw` onto the L1 ball of radius `budget`, in place. Returns whether it bound.
///
/// Proportional, never truncating: the raw vector is a preference ordering WITH magnitudes,
/// and a leverage limit scales a book rather than dropping its smallest names.
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

// ---------------------------------------------------------------------------
// The book
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct BacktestConfig {
    /// Book equity at the first instant, in dollars. It is what turns a weight into a
    /// notional and therefore what makes a size-aware cost model mean anything.
    pub capital_usd: f64,
    /// No-trade band, in ABSOLUTE WEIGHT units: a name whose target moves by `<= band` is
    /// left where it is rather than refined at a cost.
    ///
    /// Zero is the every-bar re-solve that produced the original bench's turnover of 3.35
    /// per bar. The band is the ONE lever the break-even arithmetic leaves - break-even is
    /// gross edge over turnover, so a band that cuts turnover tenfold raises the cost the
    /// book can survive tenfold, and the only question is how much of the edge is itself
    /// high-frequency and dies with the trading. That trade-off is a curve, not a number,
    /// which is why this is swept as an axis and never tuned.
    pub band: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        // Ten million dollars: large enough that a 5-minute cross-section of large caps can
        // absorb it, small enough that impact is not the whole answer.
        Self {
            capital_usd: 1.0e7,
            band: 0.0,
        }
    }
}

/// What the cost model could and could not measure over one book's realized trading, and at
/// what participation it was asked to price.
///
/// Every field is a COUNT or a SUM over the legs the book actually traded, so a row of the
/// frontier can state how much of its net figure rests on a cross-sectional stand-in instead
/// of leaving that invisible. A leg is one `(name, instant)` whose weight moved; a leg that
/// did not move is not charged and is not counted here.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LegAudit {
    /// Legs charged at all.
    pub legs: u64,
    /// Legs whose traded MONTH carried no usable spread or volatility, priced at the same
    /// symbol's whole-span liquidity. Still that symbol's own measurement.
    pub month_substituted: u64,
    /// Legs whose symbol had no measurable spread at any tier, priced at the cross-sectional
    /// median spread.
    pub spread_substituted: u64,
    /// Legs whose symbol had no measurable volatility at any tier, so their square-root
    /// impact is the cross-sectional median's rather than their own.
    pub impact_substituted: u64,
    /// Traded weight carried by legs with a spread or impact stand-in. Turnover rather than
    /// leg count, because the cost a stand-in decided is proportional to the notional it
    /// priced, not to the number of names it priced.
    pub substituted_turnover: f64,
    /// Legs whose symbol had NO observed dollar volume at all, charged at a full-ADV clip by
    /// [`backtest`]. Unpriceable size is never free size.
    pub no_liquidity_legs: u64,
    /// Total traded weight, which is `sum(PortfolioRun::turnover)`.
    pub turnover: f64,
    /// Total cost paid as a fraction of equity, which is `sum(PortfolioRun::cost)`.
    pub cost: f64,
    /// `sum_legs |delta_w| * (traded notional / ADV)`, for the notional-weighted mean
    /// participation the impact term was actually evaluated at.
    pub weighted_participation: f64,
    /// Largest participation of ADV any single leg was priced at.
    pub max_participation: f64,
}

impl LegAudit {
    /// Share of charged legs whose spread or impact came from the cross-section.
    pub fn substituted_leg_share(&self) -> f64 {
        if self.legs > 0 {
            (self.spread_substituted + self.impact_substituted).min(self.legs) as f64
                / self.legs as f64
        } else {
            f64::NAN
        }
    }

    /// Share of traded weight priced with a stand-in.
    pub fn substituted_turnover_share(&self) -> f64 {
        if self.turnover > 0.0 {
            self.substituted_turnover / self.turnover
        } else {
            f64::NAN
        }
    }

    /// The one-way cost the book ACTUALLY paid, in bps per dollar traded.
    ///
    /// This is the number that makes a per-symbol cost model comparable to the flat constant
    /// it replaces: `cost / turnover` is the effective flat rate the measured model charged
    /// this particular book, and the break-even bps beside it is stated in the same units.
    pub fn realized_cost_bps(&self) -> f64 {
        if self.turnover > 0.0 {
            1.0e4 * self.cost / self.turnover
        } else {
            f64::NAN
        }
    }

    /// Notional-weighted mean participation of ADV, the argument the impact term was priced at.
    pub fn mean_participation(&self) -> f64 {
        if self.turnover > 0.0 {
            self.weighted_participation / self.turnover
        } else {
            f64::NAN
        }
    }
}

/// Round-trip position lifecycles: the second of the three things a "win rate" can mean.
///
/// A lifecycle is one name held with ONE sign, from the instant its weight left zero to the
/// instant it returned to zero or flipped. Its P&L is the sum of the payoffs it earned while
/// held; its NET P&L subtracts the cost of every trade inside it, with the cost of an exit
/// charged to the lifecycle being exited. A position still open at the last instant is closed
/// there and counted, because dropping it would keep only the trades that happened to end and
/// that selection is not neutral.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TradeAudit {
    pub trades: u64,
    /// Lifecycles whose payoff was positive before cost.
    pub gross_wins: u64,
    /// Lifecycles whose payoff was positive after the cost of the trades inside them.
    pub net_wins: u64,
    /// Instants summed over lifecycles: the mean holding period is this over [`Self::trades`].
    pub bars_held: u64,
    /// Name-instants holding a non-zero weight in a name that printed.
    pub positioned_legs: u64,
    /// The subset whose HELD weight had the sign of the realized return. This is
    /// [`super::trade_bench`]'s own `hit_rate` definition — sign agreement conditional on a
    /// position — lifted to a portfolio, where it is no longer the same statistic as the share
    /// of profitable bars, because a book can be right on most names and lose on the bar.
    pub sign_agreements: u64,
}

impl TradeAudit {
    /// Share of round-trip lifecycles that made money net of the cost of trading them.
    pub fn net_win_rate(&self) -> f64 {
        if self.trades > 0 {
            self.net_wins as f64 / self.trades as f64
        } else {
            f64::NAN
        }
    }

    pub fn gross_win_rate(&self) -> f64 {
        if self.trades > 0 {
            self.gross_wins as f64 / self.trades as f64
        } else {
            f64::NAN
        }
    }

    pub fn mean_hold_bars(&self) -> f64 {
        if self.trades > 0 {
            self.bars_held as f64 / self.trades as f64
        } else {
            f64::NAN
        }
    }

    /// Sign agreement between the held weight and the realized return, over positioned legs.
    pub fn position_sign_agreement(&self) -> f64 {
        if self.positioned_legs > 0 {
            self.sign_agreements as f64 / self.positioned_legs as f64
        } else {
            f64::NAN
        }
    }
}

/// One book's realized path through the panel.
#[derive(Clone, Debug)]
pub struct PortfolioRun {
    pub policy: Policy,
    pub gross_cap: f64,
    /// No-trade band this book ran under, in absolute weight units. Zero is the every-bar
    /// re-solve; the frontier sweeps it.
    pub band: f64,
    /// Natural log of wealth, `log_equity[0] == 0.0`, `NEG_INFINITY` once the book is dead.
    ///
    /// The book compounds in LOG space and this is the primitive: a perfect-foresight
    /// ceiling at 4x gross over five months multiplies wealth by more than `e^700`, which is
    /// not representable, and a bench that silently reports `inf` has stopped measuring.
    pub log_equity: Vec<f64>,
    /// `exp(log_equity)`, for readers who want wealth rather than its logarithm. It
    /// saturates to `+inf` exactly where `log_equity` says the ceiling has run past f64.
    pub equity: Vec<f64>,
    /// `sum_i w_i r_i` at instant `t`, BEFORE cost. Cost-independent, because the weights
    /// are Kelly on the raw predictive law and the projection is a leverage limit: nothing
    /// in the sizing consults the cost model. That is what makes
    /// [`PortfolioRun::log_growth_at_cost`] exact rather than an approximation.
    pub payoff: Vec<f64>,
    /// Simple return of instant `t`, net of cost. Floored at `-1` on the ruin bar.
    pub returns: Vec<f64>,
    pub gross: Vec<f64>,
    pub net: Vec<f64>,
    pub turnover: Vec<f64>,
    pub cost: Vec<f64>,
    /// `sum_i w_i v_i` at instant `t`, where `v` is the panel's leading eigenvector. How
    /// much of a nominally diversified book is one bet on the market.
    pub factor: Vec<f64>,
    /// `sum_i w_i^2 var(r_i)` at instant `t`: the variance the book would have if the names
    /// were independent, which is exactly what per-name Kelly sizing assumes.
    pub pred_var: Vec<f64>,
    pub bound: Vec<bool>,
    /// The instant at which wealth first reached zero. A dead book stays dead.
    pub ruined_at: Option<usize>,
    /// What the cost model measured and what it substituted, over this book's own trading.
    pub legs: LegAudit,
    /// Round-trip lifecycles and per-leg sign agreement, for the win-rate question.
    pub trades: TradeAudit,
    pub metrics: PortfolioMetrics,
}

/// What a trader would quote, all of it annualized from the panel's OWN calendar.
#[derive(Clone, Copy, Debug)]
pub struct PortfolioMetrics {
    pub policy: &'static str,
    pub gross_cap: f64,
    pub instants: usize,
    pub span_years: f64,
    pub instants_per_year: f64,
    pub trading_days: usize,
    /// `ln(final wealth)`, `NEG_INFINITY` for a dead book. Always defined.
    pub final_log_wealth: f64,
    pub final_wealth: f64,
    /// `ln(final wealth) / span_years`: the compounding rate, in nats per year. This is the
    /// number that stays finite and comparable no matter how absurd the ceiling gets.
    pub log_growth_per_year: f64,
    /// Log growth per year BEFORE any transaction cost: what the predictive law was worth
    /// to this book before paying to act on it.
    pub gross_log_growth_per_year: f64,
    /// Flat one-way cost, in bps, at which the net log growth crosses zero.
    ///
    /// `+inf` for a book that no cost can push under (it never trades, or it grows without
    /// turnover), `0` for a book already under water gross of cost. This is the headline: a
    /// strategy whose break-even is 0.8 bps is not a strategy, whatever its gross number.
    pub break_even_cost_bps: f64,
    /// `exp(log_growth_per_year) - 1`, the figure a trader quotes. `+inf` when the ceiling
    /// overflows, which is itself the finding.
    pub cagr: f64,
    pub sharpe: f64,
    pub vol: f64,
    pub max_drawdown: f64,
    pub calmar: f64,
    pub mean_gross: f64,
    pub max_gross: f64,
    pub mean_net: f64,
    pub max_abs_net: f64,
    pub turnover_per_day: f64,
    pub mean_breadth: f64,
    pub bound_fraction: f64,
    /// Mean cost drag per instant, in basis points of equity.
    pub cost_bps_per_instant: f64,
    /// Turnover per instant as a multiple of the book's own REALIZED gross exposure: how
    /// many times the book turns itself over per bar.
    ///
    /// Absolute turnover is not comparable across leverage caps and, once a band freezes
    /// stale residuals, is not comparable across bands either - realized gross drifts away
    /// from the nominal cap. This divides by what the book actually held.
    pub rotation_per_instant: f64,
    /// Mean absolute projection of the book onto the panel's leading eigenvector, as a
    /// FRACTION of the book's gross. `0` is factor-neutral; `1` is a book that is entirely
    /// one bet on the market wearing the costume of many.
    pub mean_first_factor_exposure: f64,
    /// Share of the panel's cross-sectional return variance the leading eigenvector holds.
    /// A property of the market, identical across policies; carried here so the exposure
    /// above can be read against the thing it is an exposure to.
    pub first_factor_share: f64,
    /// Realized book volatility divided by the volatility the per-name predictive laws
    /// imply under INDEPENDENCE. Per-name Kelly sizes as if this were `1.0`; whatever it
    /// actually is, is the factor by which the book is over-levered.
    pub leverage_error: f64,
    pub ruined_at_instant: f64,
    /// Mean per-instant book payoff BEFORE cost, in bps of equity, and its per-instant
    /// standard deviation. The pair the whole strategy question reduces to: a book whose
    /// conditional edge is a fraction of one round trip's cost cannot be rescued by any
    /// hit rate, and these two numbers are what make that comparison rather than infer it.
    pub payoff_bps_per_instant: f64,
    pub payoff_sd_bps_per_instant: f64,
    /// `payoff_bps_per_instant / payoff_sd_bps_per_instant`: the per-instant information
    /// ratio, unannualized, so it can be read against the per-bar arithmetic directly.
    pub payoff_ratio_per_instant: f64,
    /// Share of instants whose NET return was positive. The FIRST of the three things a "win
    /// rate" can mean, and the only one that is a property of the book's own equity curve.
    pub bar_win_rate: f64,
    /// Share of round-trip lifecycles that made money net of the cost of trading them, and
    /// how many there were. The SECOND meaning.
    pub trade_win_rate: f64,
    pub trade_win_rate_gross: f64,
    pub trades: f64,
    pub mean_hold_bars: f64,
    /// Share of positioned name-instants whose held weight had the realized move's sign. The
    /// THIRD meaning, and the one [`super::trade_bench`]'s `hit_rate` field computes.
    pub position_sign_agreement: f64,
    /// The one-way cost the book actually paid, in bps per dollar traded. Under [`FlatCost`]
    /// this is the constant itself; under [`PanelCost`] it is the measured per-symbol model
    /// reduced to the single number [`Self::break_even_cost_bps`] must be compared against.
    pub realized_cost_bps: f64,
    /// Notional-weighted mean, and worst, participation of ADV the impact term was priced at.
    pub mean_participation_of_adv: f64,
    pub max_participation_of_adv: f64,
    /// Share of charged legs, and of traded weight, whose spread or impact came from the
    /// cross-sectional median instead of from the symbol's own bars. A net figure with a
    /// large share here is a statement about the median, not about the book.
    pub substituted_leg_share: f64,
    pub substituted_turnover_share: f64,
    /// Legs charged at a full-ADV clip for having no observed dollar volume at all.
    pub no_liquidity_legs: f64,
}

/// One name's open position, for the round-trip accounting in [`backtest`].
#[derive(Clone, Copy, Debug, Default)]
struct OpenTrade {
    /// Sign of the held weight; `0` when the name is flat and no lifecycle is open.
    sign: i8,
    /// Payoff earned while held, as a fraction of book equity at each instant.
    gross: f64,
    /// Cost of every trade charged to this lifecycle, same units.
    cost: f64,
    bars: u32,
}

#[inline]
fn sign_of(weight: f64) -> i8 {
    if weight > 0.0 {
        1
    } else if weight < 0.0 {
        -1
    } else {
        0
    }
}

/// Retire one lifecycle into the audit and reset it. A flat name has nothing to retire.
#[inline]
fn close_trade(open: &mut OpenTrade, audit: &mut TradeAudit) {
    if open.sign == 0 {
        return;
    }
    audit.trades += 1;
    audit.bars_held += u64::from(open.bars);
    if open.gross > 0.0 {
        audit.gross_wins += 1;
    }
    if open.gross - open.cost > 0.0 {
        audit.net_wins += 1;
    }
    *open = OpenTrade::default();
}

/// Compound one policy through the panel at one gross cap.
///
/// The whole loop is four lines of economics wrapped in bookkeeping: form the raw vector,
/// project it onto the gross ball, pay `sum_i w_i r_i` minus the cost of getting there, and
/// multiply. Everything else is the accounting that makes the result quotable.
pub fn backtest(
    panel: &Panel,
    inputs: &PolicyInputs<'_>,
    policy: Policy,
    gross_cap: f64,
    cost: &dyn CostModel,
    config: &BacktestConfig,
) -> Result<PortfolioRun> {
    let band = config.band;
    ensure!(
        band >= 0.0 && band.is_finite(),
        "the no-trade band must be a non-negative weight, got {band}"
    );
    ensure!(
        gross_cap > 0.0 && gross_cap.is_finite(),
        "the gross cap must be positive and finite, got {gross_cap}"
    );
    ensure!(
        inputs.model.len() == panel.instants() && inputs.marginal.len() == panel.instants(),
        "forecasts must cover every instant: {} model / {} marginal against {} instants",
        inputs.model.len(),
        inputs.marginal.len(),
        panel.instants()
    );
    let budget = policy.gross_budget(gross_cap);
    let names = panel.symbols().len();

    let mut log_wealth = 0.0f64;
    let mut held = vec![0.0f64; names];
    let mut target = vec![0.0f64; names];
    let mut last_adv = vec![0.0f32; names];
    let mut raw: Vec<f64> = Vec::with_capacity(names);
    // One open lifecycle per name, for the round-trip win rate. Flat names carry `sign == 0`
    // and cost nothing to keep here.
    let mut open = vec![OpenTrade::default(); names];

    let instants = panel.instants();
    let mut run = PortfolioRun {
        policy,
        gross_cap,
        band,
        log_equity: Vec::with_capacity(instants + 1),
        equity: Vec::with_capacity(instants + 1),
        payoff: Vec::with_capacity(instants),
        returns: Vec::with_capacity(instants),
        gross: Vec::with_capacity(instants),
        net: Vec::with_capacity(instants),
        turnover: Vec::with_capacity(instants),
        cost: Vec::with_capacity(instants),
        factor: Vec::with_capacity(instants),
        pred_var: Vec::with_capacity(instants),
        bound: Vec::with_capacity(instants),
        ruined_at: None,
        legs: LegAudit::default(),
        trades: TradeAudit::default(),
        metrics: PortfolioMetrics::empty(policy, gross_cap),
    };
    run.log_equity.push(log_wealth);
    run.equity.push(1.0);

    for (t, slice) in panel.slices().iter().enumerate() {
        for (k, &id) in slice.symbols.iter().enumerate() {
            let adv = panel.adv_usd(t, k);
            if adv.is_finite() && adv > 0.0 {
                last_adv[id as usize] = adv;
            }
        }
        if run.ruined_at.is_some() {
            // A dead book holds nothing, trades nothing and earns nothing, forever. It is
            // still recorded at every instant so the curve keeps the panel's clock.
            run.log_equity.push(f64::NEG_INFINITY);
            run.equity.push(0.0);
            run.payoff.push(0.0);
            run.returns.push(0.0);
            run.gross.push(0.0);
            run.net.push(0.0);
            run.turnover.push(0.0);
            run.cost.push(0.0);
            run.factor.push(0.0);
            run.pred_var.push(0.0);
            run.bound.push(false);
            continue;
        }

        policy.raw_weights(
            slice,
            &inputs.model[t],
            &inputs.marginal[t],
            budget,
            &mut raw,
        );
        ensure!(
            raw.len() == slice.symbols.len(),
            "policy {} produced {} weights for {} tradeable names at instant {t}",
            policy.name(),
            raw.len(),
            slice.symbols.len()
        );
        for w in raw.iter_mut() {
            if !w.is_finite() {
                *w = 0.0;
            }
        }
        let bound = project_gross(&mut raw, budget);

        target[..].fill(0.0);
        for (k, &id) in slice.symbols.iter().enumerate() {
            target[id as usize] = raw[k];
        }

        // The no-trade band. A move smaller than `band` is not worth what it costs to make,
        // so the book KEEPS its stale weight instead of paying to refine it. Only names that
        // print at this instant may be held: an absent name cannot be traded at all, so it
        // goes to zero whatever the band says, and that unwind is charged like any other.
        //
        // Holding a stale weight can push the book back over its budget, so the gross ball
        // is re-imposed afterwards. That deleverage is a real trade and is charged as one;
        // it is also why the constraint is asserted on the FINAL vector rather than on the
        // projection, which is the only thing that makes the assertion worth anything.
        if band > 0.0 {
            for &id in &slice.symbols {
                let id = id as usize;
                if (target[id] - held[id]).abs() <= band {
                    target[id] = held[id];
                }
            }
            let held_gross: f64 = target.iter().map(|w| w.abs()).sum();
            if held_gross > budget {
                let shrink = budget / held_gross;
                for w in target.iter_mut() {
                    *w *= shrink;
                }
            }
        }

        // The rebalance is charged FIRST, then the bar's payoff is attributed. The order is
        // load-bearing for the lifecycle bookkeeping alone: a name that opens at this instant
        // has no lifecycle to earn into until the trade that opened it has been recorded.
        // Nothing about the arithmetic depends on it - the cost loop reads `target` and
        // `held`, the payoff loop reads `target` and the slice, and neither writes what the
        // other reads.
        let mut turnover = 0.0f64;
        let mut cost_frac = 0.0f64;
        // The notional a weight is worth. A ceiling that has compounded past f64 makes this
        // infinite; the clip keeps the cost model inside its own domain, and a book that big
        // is a diagnostic rather than a thing whose transaction costs matter.
        let wealth_usd = (log_wealth.exp() * config.capital_usd).min(f64::from(f32::MAX));
        for id in 0..names {
            let (previous, now) = (held[id], target[id]);
            let delta = (now - previous).abs();
            if delta == 0.0 {
                continue;
            }
            turnover += delta;
            let adv = last_adv[id];
            // No observed liquidity means the size is unpriceable, not free: charge the
            // model at a full-ADV clip, the worst bucket any sane cost curve carries.
            let frac = if adv > 0.0 {
                (((delta * wealth_usd) / f64::from(adv)) as f32).min(f32::MAX)
            } else {
                run.legs.no_liquidity_legs += 1;
                1.0
            };
            let leg = cost.leg_cost(id as u32, slice.ts_ms, frac);
            let bps = f64::from(leg.bps);
            ensure!(
                bps.is_finite() && bps >= 0.0,
                "the cost model returned {bps} bps for symbol {id} at {}",
                slice.ts_ms
            );
            let paid = delta * bps * 1e-4;
            cost_frac += paid;

            run.legs.legs += 1;
            run.legs.turnover += delta;
            run.legs.cost += paid;
            run.legs.month_substituted += u64::from(leg.month_substituted);
            run.legs.spread_substituted += u64::from(leg.spread_substituted);
            run.legs.impact_substituted += u64::from(leg.impact_substituted);
            if leg.substituted() {
                run.legs.substituted_turnover += delta;
            }
            run.legs.weighted_participation += delta * f64::from(frac);
            run.legs.max_participation = run.legs.max_participation.max(f64::from(frac));

            // The lifecycle. An exit pays for the whole move that ended it, including the
            // half of a sign flip that belongs to the position being opened: attributing part
            // of one trade's cost to the next one would let a book of flips report round trips
            // that each paid for half of themselves.
            let closing = previous != 0.0 && (now == 0.0 || (previous > 0.0) != (now > 0.0));
            if closing {
                open[id].cost += paid;
                close_trade(&mut open[id], &mut run.trades);
                open[id].sign = sign_of(now);
            } else {
                if previous == 0.0 {
                    open[id].sign = sign_of(now);
                }
                open[id].cost += paid;
            }
        }
        held.copy_from_slice(&target);

        let mut payoff = 0.0f64;
        let mut gross = 0.0f64;
        let mut net = 0.0f64;
        // The book's projection onto the panel's leading eigenvector, and the variance the
        // per-name predictive laws imply IF the names were independent. Kelly assumes they
        // are; the ratio of realized to independence volatility is the price of that
        // assumption, measured rather than argued.
        let mut factor = 0.0f64;
        let mut pred_var = 0.0f64;
        let loading = panel.first_factor();
        let forecast = &inputs.model[t];
        for (k, &id) in slice.symbols.iter().enumerate() {
            let w = target[id as usize];
            gross += w.abs();
            net += w;
            factor += w * f64::from(loading[id as usize]);
            let var = f64::from(forecast.var_r[k]);
            if var.is_finite() && var > 0.0 {
                pred_var += w * w * var;
            }
            // `realized_r` is a LOG return; a position is paid the simple one.
            let realized = f64::from(slice.realized_r[k]);
            let earned = w * realized.exp_m1();
            payoff += earned;
            if w != 0.0 {
                let lifecycle = &mut open[id as usize];
                lifecycle.bars += 1;
                lifecycle.gross += earned;
                run.trades.positioned_legs += 1;
                if w * realized > 0.0 {
                    run.trades.sign_agreements += 1;
                }
            }
        }
        ensure!(
            gross <= budget * (1.0 + GROSS_TOLERANCE) + GROSS_TOLERANCE,
            "policy {} used gross {gross} against a budget of {budget} at instant {t}",
            policy.name()
        );

        let multiplier = 1.0 + payoff - cost_frac;
        let realized = if multiplier > 0.0 {
            log_wealth += multiplier.ln();
            multiplier - 1.0
        } else {
            log_wealth = f64::NEG_INFINITY;
            run.ruined_at = Some(t);
            held.fill(0.0);
            // The book is dead, so every position it held ended here. Counting them is what
            // keeps the round-trip win rate from silently dropping the worst trades in the
            // run - which are exactly the ones that killed it.
            for lifecycle in open.iter_mut() {
                close_trade(lifecycle, &mut run.trades);
            }
            -1.0
        };
        run.factor.push(factor);
        run.pred_var.push(pred_var);
        run.log_equity.push(log_wealth);
        run.equity.push(log_wealth.exp());
        run.payoff.push(payoff);
        run.returns.push(realized);
        run.gross.push(gross);
        run.net.push(net);
        run.turnover.push(turnover);
        run.cost.push(cost_frac);
        run.bound.push(bound);
    }
    // A position still open at the last instant is closed there and counted. Dropping it
    // would keep only the lifecycles that happened to end inside the panel, and a book whose
    // losers are the ones it is still holding would report the win rate of its winners.
    for lifecycle in open.iter_mut() {
        close_trade(lifecycle, &mut run.trades);
    }

    run.metrics = PortfolioMetrics::of(&run, panel);
    Ok(run)
}

/// Bisection steps of the break-even solve over `[0, MAX_BREAK_EVEN_BPS]`: 48 halvings
/// resolve it to `~4e-12` bps.
const BREAK_EVEN_ITERATIONS: usize = 48;

/// Flat one-way cost, in bps, beyond which the break-even search reports "never".
///
/// A break-even above this is not a distinction a trader can act on, and an infinite one is
/// dropped by the renderer's non-finite filter — which looks exactly like a metric that was
/// never measured.
pub const MAX_BREAK_EVEN_BPS: f64 = 1000.0;

impl PortfolioRun {
    /// Total net log growth over the panel at a FLAT one-way cost of `bps`.
    ///
    /// Exact, not a linearization: the weights are cost-blind, so the realized payoff and
    /// turnover paths this replays are the same ones at every cost level, and the only thing
    /// that changes is the multiplier. Ruin is respected — a cost level that kills the book
    /// kills it here too, and the growth is `-inf` from that instant on.
    pub fn log_growth_at_cost(&self, bps: f64) -> f64 {
        let rate = bps * 1e-4;
        let mut total = 0.0f64;
        for (payoff, turnover) in self.payoff.iter().zip(&self.turnover) {
            let multiplier = 1.0 + payoff - rate * turnover;
            if multiplier <= 0.0 {
                return f64::NEG_INFINITY;
            }
            total += multiplier.ln();
        }
        total
    }

    /// Flat one-way cost at which the net log growth crosses zero.
    ///
    /// `log_growth_at_cost` is monotonically decreasing in the cost whenever the book trades
    /// at all, so a bisection on `[0, MAX_BREAK_EVEN_BPS]` is exact up to its own tolerance.
    /// The two degenerate cases are reported as themselves: a book already losing before any
    /// cost breaks even at `0`, and a book no cost can sink never breaks even.
    pub fn break_even_cost_bps(&self) -> f64 {
        if !(self.log_growth_at_cost(0.0) > 0.0) {
            return 0.0;
        }
        if self.log_growth_at_cost(MAX_BREAK_EVEN_BPS) > 0.0 {
            return f64::INFINITY;
        }
        let (mut lo, mut hi) = (0.0f64, MAX_BREAK_EVEN_BPS);
        for _ in 0..BREAK_EVEN_ITERATIONS {
            let mid = 0.5 * (lo + hi);
            if self.log_growth_at_cost(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// The break-even a CHART may carry: the clip that keeps `+inf` visible without letting an
/// UNMEASURED book borrow the bound.
///
/// `f64::min` returns the non-NaN operand, so `f64::NAN.min(MAX_BREAK_EVEN_BPS)` is
/// `MAX_BREAK_EVEN_BPS` - a book that never traded would plot at the most favourable
/// break-even on the axis, beside real ones, and nothing would say it was never measured. The
/// three states are distinct and only two of them belong on a curve: a finite break-even is
/// itself, `+inf` clips to the bound the bisection stops at (a MEASURED "no cost sinks it"),
/// and NaN stays NaN so the renderer's non-finite filter drops the point.
#[inline]
pub fn charted_break_even_bps(bps: f64) -> f64 {
    if bps.is_nan() {
        f64::NAN
    } else {
        bps.min(MAX_BREAK_EVEN_BPS)
    }
}

impl PortfolioMetrics {
    fn empty(policy: Policy, gross_cap: f64) -> Self {
        Self {
            policy: policy.name(),
            gross_cap,
            instants: 0,
            span_years: f64::NAN,
            instants_per_year: f64::NAN,
            trading_days: 0,
            final_log_wealth: f64::NAN,
            final_wealth: f64::NAN,
            log_growth_per_year: f64::NAN,
            gross_log_growth_per_year: f64::NAN,
            break_even_cost_bps: f64::NAN,
            cagr: f64::NAN,
            sharpe: f64::NAN,
            vol: f64::NAN,
            max_drawdown: f64::NAN,
            calmar: f64::NAN,
            mean_gross: f64::NAN,
            max_gross: f64::NAN,
            mean_net: f64::NAN,
            max_abs_net: f64::NAN,
            turnover_per_day: f64::NAN,
            mean_breadth: f64::NAN,
            bound_fraction: f64::NAN,
            cost_bps_per_instant: f64::NAN,
            rotation_per_instant: f64::NAN,
            mean_first_factor_exposure: f64::NAN,
            first_factor_share: f64::NAN,
            leverage_error: f64::NAN,
            ruined_at_instant: f64::NAN,
            payoff_bps_per_instant: f64::NAN,
            payoff_sd_bps_per_instant: f64::NAN,
            payoff_ratio_per_instant: f64::NAN,
            bar_win_rate: f64::NAN,
            trade_win_rate: f64::NAN,
            trade_win_rate_gross: f64::NAN,
            trades: f64::NAN,
            mean_hold_bars: f64::NAN,
            position_sign_agreement: f64::NAN,
            realized_cost_bps: f64::NAN,
            mean_participation_of_adv: f64::NAN,
            max_participation_of_adv: f64::NAN,
            substituted_leg_share: f64::NAN,
            substituted_turnover_share: f64::NAN,
            no_liquidity_legs: f64::NAN,
        }
    }

    fn of(run: &PortfolioRun, panel: &Panel) -> Self {
        let n = run.returns.len();
        let years = panel.span_years();
        let per_year = panel.instants_per_year();
        let final_log_wealth = *run.log_equity.last().expect("the curve starts at 0.0");

        // Everything about compounding is computed in LOG space and only then exponentiated.
        // A ruined book is `-inf` nats, which is `-100%` a year however long the span; the
        // perfect-foresight ceiling can be several hundred nats a year, whose CAGR is not
        // representable and is reported as the infinity it is rather than as a wrapped
        // number.
        let log_growth_per_year = if years > 0.0 {
            final_log_wealth / years
        } else {
            f64::NAN
        };
        let cagr = if final_log_wealth == f64::NEG_INFINITY {
            -1.0
        } else {
            log_growth_per_year.exp_m1()
        };

        let mean = run.returns.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            run.returns
                .iter()
                .map(|r| (r - mean) * (r - mean))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            f64::NAN
        };
        let sd = variance.sqrt();
        let vol = sd * per_year.sqrt();
        // Excess over a zero risk-free rate, stated rather than assumed: over a five-month
        // held-out span at a 5-minute horizon the cash rate moves the ratio by less than the
        // width of its own interval.
        let sharpe = if sd > 0.0 { mean / sd * per_year.sqrt() } else { f64::NAN };

        // Drawdown from the log curve: `1 - exp(log w - log peak)` is exact and stays inside
        // `[0, 1]` even where the linear curve has overflowed.
        let mut peak = f64::NEG_INFINITY;
        let mut max_drawdown: f64 = 0.0;
        for &log_w in &run.log_equity {
            peak = peak.max(log_w);
            if peak > f64::NEG_INFINITY {
                max_drawdown = max_drawdown.max(1.0 - (log_w - peak).exp());
            }
        }

        // How much of the book is one bet, and by how much per-name Kelly mis-sized it.
        // The exposure is normalized by gross so it is a share of the book rather than a
        // number that grows with the leverage cap; instants where the book is flat carry no
        // exposure and are excluded rather than counted as neutral.
        let mut exposure_sum = 0.0f64;
        let mut exposure_count = 0usize;
        for (f, g) in run.factor.iter().zip(&run.gross) {
            if *g > 0.0 {
                exposure_sum += (f / g).abs();
                exposure_count += 1;
            }
        }
        // Realized against independence-implied volatility, both of the COST-FREE payoff:
        // the predictive laws say nothing about transaction costs, so charging the realized
        // side for them would compare two different objects.
        let payoff_mean = run.payoff.iter().sum::<f64>() / n as f64;
        let payoff_var = if n > 1 {
            run.payoff
                .iter()
                .map(|p| (p - payoff_mean) * (p - payoff_mean))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            f64::NAN
        };
        let independence_var = run.pred_var.iter().sum::<f64>() / n as f64;

        let days = panel.trading_days().max(1) as f64;
        let mean_of = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
        Self {
            policy: run.policy.name(),
            gross_cap: run.gross_cap,
            instants: n,
            span_years: years,
            instants_per_year: per_year,
            trading_days: panel.trading_days(),
            final_log_wealth,
            final_wealth: final_log_wealth.exp(),
            log_growth_per_year,
            gross_log_growth_per_year: if years > 0.0 {
                run.log_growth_at_cost(0.0) / years
            } else {
                f64::NAN
            },
            break_even_cost_bps: run.break_even_cost_bps(),
            cagr,
            sharpe,
            vol,
            max_drawdown,
            calmar: if max_drawdown > 0.0 {
                cagr / max_drawdown
            } else {
                f64::NAN
            },
            mean_gross: mean_of(&run.gross),
            max_gross: run.gross.iter().copied().fold(0.0, f64::max),
            mean_net: mean_of(&run.net),
            max_abs_net: run.net.iter().map(|n| n.abs()).fold(0.0, f64::max),
            turnover_per_day: run.turnover.iter().sum::<f64>() / days,
            mean_breadth: panel.breadth().mean,
            bound_fraction: run.bound.iter().filter(|b| **b).count() as f64 / n as f64,
            cost_bps_per_instant: mean_of(&run.cost) * 1e4,
            rotation_per_instant: {
                let held = mean_of(&run.gross);
                if held > 0.0 {
                    mean_of(&run.turnover) / held
                } else {
                    f64::NAN
                }
            },
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
            ruined_at_instant: run.ruined_at.map_or(f64::NAN, |t| t as f64),
            // The per-bar edge and the per-bar volatility, in bps of equity, from the
            // COST-FREE payoff. Cost belongs on the other side of the comparison these two
            // exist to make: "is the conditional mean of a bar worth one round trip".
            payoff_bps_per_instant: 1.0e4 * payoff_mean,
            payoff_sd_bps_per_instant: 1.0e4 * payoff_var.sqrt(),
            payoff_ratio_per_instant: if payoff_var > 0.0 {
                payoff_mean / payoff_var.sqrt()
            } else {
                f64::NAN
            },
            bar_win_rate: run.returns.iter().filter(|r| **r > 0.0).count() as f64 / n as f64,
            trade_win_rate: run.trades.net_win_rate(),
            trade_win_rate_gross: run.trades.gross_win_rate(),
            trades: run.trades.trades as f64,
            mean_hold_bars: run.trades.mean_hold_bars(),
            position_sign_agreement: run.trades.position_sign_agreement(),
            realized_cost_bps: run.legs.realized_cost_bps(),
            mean_participation_of_adv: run.legs.mean_participation(),
            max_participation_of_adv: run.legs.max_participation,
            substituted_leg_share: run.legs.substituted_leg_share(),
            substituted_turnover_share: run.legs.substituted_turnover_share(),
            no_liquidity_legs: run.legs.no_liquidity_legs as f64,
        }
    }
}

// ---------------------------------------------------------------------------
// The whole verdict
// ---------------------------------------------------------------------------

/// No-trade band grid, in MULTIPLES OF A TYPICAL POSITION.
///
/// A typical position is `gross_cap / mean_breadth`, so a fraction of `1.0` means "only
/// rebalance a name when the move is at least a whole normal-sized position". The grid is
/// stated in these units rather than in absolute weight because absolute weight is not
/// scale-free: `0.05` is a rounding error on a 4-name book and freezes an 80-name one.
///
/// This is the axis, not a tuned parameter. Break-even cost is gross edge over turnover, so
/// banding buys break-even at the price of edge, and the deliverable is where those two
/// curves cross - not a band that flatters the book.
pub const BAND_FRACTIONS: [f64; 7] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

/// One point of the turnover-edge frontier: what a book earns once it stops trading so much.
#[derive(Clone, Copy, Debug)]
pub struct FrontierPoint {
    pub policy: &'static str,
    /// Band in multiples of a typical position, from [`BAND_FRACTIONS`].
    pub band_fraction: f64,
    /// The same band in ABSOLUTE WEIGHT units, which is what the engine actually applied.
    pub band: f64,
    pub turnover_per_day: f64,
    /// Log growth per year before cost: the edge that survives the reduced trading.
    pub gross_log_growth_per_year: f64,
    /// Log growth per year after the cost model: the number that decides it.
    pub log_growth_per_year: f64,
    /// Flat one-way cost, in bps, at which this point's net growth reaches zero.
    pub break_even_cost_bps: f64,
    pub mean_gross: f64,
    pub sharpe: f64,
    /// Mean realized cost drag per instant, in bps of equity, under the supplied model.
    pub cost_bps_per_instant: f64,
    /// Turnover per instant as a multiple of realized gross: the book's rotation rate, and
    /// the denominator of `break_even = gross_edge / turnover`.
    pub rotation_per_instant: f64,
    /// This point's total traded notional as a share of the SAME book at band zero.
    ///
    /// Turnover, not legs. A band freezes the smallest deltas by construction, so the
    /// fraction of name-bars it froze badly overstates the saving; this is the saving.
    pub turnover_share_of_unbanded: f64,
    /// The one-way cost the book actually PAID under this arm's model, in bps per dollar
    /// traded. Read against [`Self::break_even_cost_bps`] this is the whole verdict of the
    /// row: the point is tradeable exactly when the break-even exceeds this.
    pub realized_cost_bps: f64,
    /// Legs charged, and how many of them rest on a cross-sectional stand-in rather than on
    /// the symbol's own measured spread or volatility.
    pub legs: LegAudit,
    /// Share of instants with a positive net return, share of round-trip lifecycles that made
    /// money net of cost, and sign agreement between the held weight and the realized move.
    /// The three distinct things a "win rate" can mean, on every row.
    pub bar_win_rate: f64,
    pub trade_win_rate: f64,
    pub position_sign_agreement: f64,
    /// Per-instant book payoff before cost, in bps of equity, and its standard deviation.
    pub payoff_bps_per_instant: f64,
    pub payoff_sd_bps_per_instant: f64,
}

/// Which cost model one arm of the frontier ran under.
///
/// The frontier is reported under EVERY arm rather than under the best one, because the
/// session's whole conclusion — "break-even 2.15 bps against a real cost of 10.99 to 20.6" —
/// was a comparison between two different cost models, and a comparison is not a measurement
/// until both sides are computed on the same book, the same panel and the same weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CostArm {
    /// A stated constant. Kept so every number quoted before the per-symbol model existed
    /// stays reproducible on the same panel rather than being silently superseded.
    Flat { bps: f32 },
    /// The measured per-symbol model: realized spread, per-share commission, regulatory fee
    /// and square-root impact at the traded participation of that symbol's own ADV.
    ///
    /// `impact_k` is an ASSUMPTION, not a fit — see [`IMPACT_K`] — which is why it is a field
    /// of the arm and why the impact-charging arms appear three times, once per
    /// [`IMPACT_K_GRID`] entry, so no net figure is a point when the literature only supports a
    /// band. The [`CostParts::NoImpact`] arm carries no coefficient at all and is therefore
    /// stated once, which is exactly what makes it the load-bearing one.
    Measured { impact_k: f64, parts: CostParts },
}

impl CostArm {
    /// The label every table, chart series and headline claim carries. States the assumption
    /// inline, so an impact coefficient can never be read as a measurement.
    pub fn label(self) -> String {
        match self {
            Self::Flat { bps } => format!("flat {bps:.2} bps"),
            Self::Measured { parts, .. } if !parts.charges_impact() => {
                format!("measured {} (no assumed coefficient)", parts.label())
            }
            Self::Measured { impact_k, parts } => {
                format!("measured {}, k={impact_k:.2} ASSUMED", parts.label())
            }
        }
    }

    /// Whether this arm's cost came from the corpus rather than from a constant.
    pub fn is_measured(self) -> bool {
        matches!(self, Self::Measured { .. })
    }

    /// Whether every term this arm charges was measured from the bars, with no assumed
    /// coefficient anywhere in it. Main's ruling: this is the column an argument may rest on.
    pub fn is_assumption_free(self) -> bool {
        matches!(self, Self::Measured { parts, .. } if !parts.charges_impact())
    }
}

/// One cost model's whole turnover-edge frontier: `points[policy][band]`.
#[derive(Clone, Debug)]
pub struct FrontierArm {
    pub arm: CostArm,
    pub points: Vec<Vec<FrontierPoint>>,
}

impl FrontierArm {
    /// The row of one policy, by [`POLICIES`] index.
    pub fn policy(&self, policy: usize) -> &[FrontierPoint] {
        &self.points[policy]
    }

    /// The band grid's index of the first point whose NET growth is positive, if any.
    ///
    /// The deliverable of the whole frontier, stated as an index rather than as a verdict: a
    /// crossing that only the equal-weight and marginal-null policies also achieve is a
    /// property of the band, not of the model, and reading that off requires both rows.
    ///
    /// `> 0.0` is FALSE for a NaN, so a band whose growth was never measured is skipped rather
    /// than crossed, and `None` means "no measured band crossed" rather than "every band was
    /// measured and none crossed". The two are different claims:
    /// [`PortfolioBench::crossing_table`] prints the unmeasured count beside every cell so the
    /// distinction is visible, and callers reading this index alone must not restate it as the
    /// latter.
    pub fn crossing(&self, policy: usize) -> Option<usize> {
        self.points[policy]
            .iter()
            .position(|p| p.log_growth_per_year > 0.0)
    }
}

/// One liquidity decile of the TRADED panel: what one bar of one name is worth, and what one
/// one-way trade in it costs, on the same row.
///
/// # Why the two halves belong on one axis
///
/// Every conclusion this module has produced is a comparison between an edge and a cost, and
/// the two have until now been measured on different objects — the edge on the model's panel,
/// the cost on the whole 5,297-symbol universe by decile. Putting both on the traded panel's
/// OWN liquidity deciles makes the comparison a subtraction instead of an inference, and it
/// answers the one question a break-even number cannot: whether the edge lives where the cost
/// is low. A predictor whose edge is concentrated in the thinnest decile has no strategy
/// however good its pooled break-even looks.
///
/// # Units, stated once
///
/// Edge figures are per NAME-BAR in bps of that name's own notional, not per book instant:
/// they are properties of the predictive law, and a book-level number would fold in the
/// leverage and the breadth. Cost figures are ONE-WAY in bps of traded notional at
/// [`PARTICIPATION_HEADLINE_SLOT`] of the symbol's own ADV; a round trip is twice them.
///
/// The four cost components are cross-sectional MEANS over the decile's symbols, so they add
/// up to [`Self::all_in_bps`] exactly. [`Self::median_all_in_bps`] is carried beside them
/// because the median is what `portfolio_cost`'s universe-wide decile table reports, and a
/// mean and a median of the same quantity must be comparable rather than confusable.
#[derive(Clone, Debug)]
pub struct EdgeVsCost {
    /// `0` is the THINNEST decile of the traded panel. `usize::MAX` marks the pooled row.
    pub decile: usize,
    pub symbols: usize,
    pub name_bars: u64,
    /// Median over the decile's symbols of the panel's own trailing dollar ADV.
    pub median_adv_usd: f64,
    // ---- what a bar is worth -------------------------------------------------------------
    /// Mean realized LOG return per name-bar, in bps. The unconditional drift of the decile.
    pub mean_r_bps: f64,
    /// Standard deviation of the same, in bps: the per-bar sigma every Kelly fraction and
    /// every Sharpe in this repository is implicitly divided by.
    pub sd_r_bps: f64,
    /// `mean(sign(f*) * r)` in bps over name-bars carrying a non-zero forecast: the
    /// DIRECTIONAL edge, the quantity a hit rate is a hit rate of.
    ///
    /// The denominator is every POSITIONED bar, flat ones included, because a bar that did not
    /// move still pays to be traded. That makes this the right number to divide a round-trip
    /// cost into and the WRONG number to call directional accuracy - see
    /// [`Self::signed_edge_per_moving_bar_bps`], which strips the attenuation.
    pub signed_edge_bps: f64,
    /// Standard error of [`Self::signed_edge_bps`], computed ACROSS INSTANTS from the
    /// per-instant cross-sectional mean rather than across name-bars.
    ///
    /// Name-bars are not independent draws — the decile's names load on one market factor at
    /// every instant — so `sd / sqrt(name_bars)` would overstate the precision by roughly the
    /// square root of the breadth. Averaging within the instant first and taking the standard
    /// error over instants is the Fama-MacBeth estimator, and it is the honest one here.
    pub signed_edge_se_bps: f64,
    /// `sum(f* r) / sum(|f*|)` in bps: the edge per unit of position actually taken, which is
    /// what a Kelly-sized book earns rather than what an equally-weighted sign bet earns.
    pub kelly_weighted_edge_bps: f64,
    /// Share of name-bars with a non-zero forecast whose forecast sign matched the realized
    /// move. This is the FORECAST's directional accuracy, free of any position sizing, band or
    /// gross constraint.
    ///
    /// A bar with `r == 0` has no sign to agree with, and it is counted here as a
    /// DISAGREEMENT because the denominator is every positioned bar. On a 5-minute panel that
    /// is not a rounding detail: read this beside [`Self::flat_share`] and
    /// [`Self::sign_agreement_on_moving_bars`] or the attenuation is invisible.
    pub forecast_sign_agreement: f64,
    /// Name-bars this decile carried a non-zero, finite Kelly fraction on: the denominator of
    /// [`Self::signed_edge_bps`] and [`Self::forecast_sign_agreement`].
    pub positioned_bars: u64,
    /// Positioned bars whose realized return was EXACTLY zero - no direction to be right
    /// about, and a zero contribution to the edge numerator.
    ///
    /// Surfaced rather than absorbed because both statistics above are attenuated by exactly
    /// `1 - flat_share`, and each decile carries its own flat share: the quiet decile holds
    /// more of them, so a top-minus-bottom difference is a difference of differently
    /// attenuated quantities and its RATIO is inflated by the ratio of the two shares.
    pub flat_positioned_bars: u64,
    /// Median `|f*|`, the uncapped Kelly fraction: `mu / sigma^2` of the per-name law.
    pub median_abs_kelly: f64,
    /// Mean of the head's OWN predicted mean return, in bps, for the calibration comparison
    /// against [`Self::mean_r_bps`].
    pub mean_forecast_bps: f64,
    // ---- what a trade costs --------------------------------------------------------------
    /// Half the measured proportional spread: the cost of crossing it once.
    pub half_spread_bps: f64,
    /// Per-SHARE commission divided by the symbol's own price. See
    /// [`super::portfolio_cost::COMMISSION_PER_SHARE_USD`].
    pub commission_bps: f64,
    pub regulatory_bps: f64,
    /// Square-root impact at [`PARTICIPATION_HEADLINE_SLOT`] of ADV, one entry per
    /// [`IMPACT_K_GRID`] coefficient. The coefficient is an ASSUMPTION, never a fit.
    pub impact_bps: [f64; IMPACT_K_GRID.len()],
    /// `half_spread + commission + regulatory + impact_bps[k]`, exactly.
    pub all_in_bps: [f64; IMPACT_K_GRID.len()],
    /// The same all-in cost as a cross-sectional MEDIAN, for comparison with
    /// `portfolio_cost`'s universe-wide decile table.
    pub median_all_in_bps: [f64; IMPACT_K_GRID.len()],
    /// Symbols in this decile whose own spread, or own volatility, was unmeasurable at every
    /// tier and priced at the cross-sectional median instead.
    pub spread_substituted: usize,
    pub impact_substituted: usize,
}

impl EdgeVsCost {
    fn empty(decile: usize) -> Self {
        Self {
            decile,
            symbols: 0,
            name_bars: 0,
            median_adv_usd: f64::NAN,
            mean_r_bps: f64::NAN,
            sd_r_bps: f64::NAN,
            signed_edge_bps: f64::NAN,
            signed_edge_se_bps: f64::NAN,
            kelly_weighted_edge_bps: f64::NAN,
            forecast_sign_agreement: f64::NAN,
            positioned_bars: 0,
            flat_positioned_bars: 0,
            median_abs_kelly: f64::NAN,
            mean_forecast_bps: f64::NAN,
            half_spread_bps: f64::NAN,
            commission_bps: f64::NAN,
            regulatory_bps: f64::NAN,
            impact_bps: [f64::NAN; IMPACT_K_GRID.len()],
            all_in_bps: [f64::NAN; IMPACT_K_GRID.len()],
            median_all_in_bps: [f64::NAN; IMPACT_K_GRID.len()],
            spread_substituted: 0,
            impact_substituted: 0,
        }
    }

    /// One-way all-in cost at the headline impact coefficient.
    pub fn headline_all_in_bps(&self) -> f64 {
        self.all_in_bps[IMPACT_K_DEFAULT_SLOT]
    }

    /// One-way cost with NO impact term: half-spread, commission and regulatory fee only.
    ///
    /// The most favourable arm that is still a cost, and therefore the one an IMPOSSIBILITY
    /// claim has to be made against: a strategy that cannot pay this cannot be rescued by
    /// trading smaller, because impact is the only component participation moves.
    pub fn impact_free_bps(&self) -> f64 {
        self.half_spread_bps + self.commission_bps + self.regulatory_bps
    }

    /// Round trips per year the decile's directional edge would pay for at the headline cost:
    /// `signed_edge / (2 * all_in)`. Below `1` the edge does not survive being acted on once.
    pub fn edge_over_round_trip(&self) -> f64 {
        self.signed_edge_bps / (2.0 * self.headline_all_in_bps())
    }

    /// Positioned bars that actually moved: the denominator a DIRECTIONAL claim needs.
    pub fn moving_positioned_bars(&self) -> u64 {
        self.positioned_bars
            .saturating_sub(self.flat_positioned_bars)
    }

    /// Share of positioned bars with `r == 0`, NaN when nothing was positioned.
    ///
    /// Both directional statistics on this row are multiplied by `1 - flat_share` relative to
    /// their moving-bar versions.
    pub fn flat_share(&self) -> f64 {
        if self.positioned_bars == 0 {
            return f64::NAN;
        }
        self.flat_positioned_bars as f64 / self.positioned_bars as f64
    }

    /// [`Self::forecast_sign_agreement`] over bars that HAD a sign. A coin flip reads as `0.5`
    /// here; on the attenuated version it reads as `0.5 * (1 - flat_share)`, which is not a
    /// coin flip and not anything else either.
    pub fn sign_agreement_on_moving_bars(&self) -> f64 {
        let moving = self.moving_positioned_bars();
        if moving == 0 {
            return f64::NAN;
        }
        self.forecast_sign_agreement * self.positioned_bars as f64 / moving as f64
    }

    /// [`Self::signed_edge_bps`] with the flat bars removed from the denominator.
    ///
    /// This is the number a per-decile RATIO must be taken over. It is NOT the number to
    /// compare against a round-trip cost: a flat bar still pays to be traded, which is why
    /// [`Self::edge_over_round_trip`] keeps the attenuated one.
    pub fn signed_edge_per_moving_bar_bps(&self) -> f64 {
        let moving = self.moving_positioned_bars();
        if moving == 0 {
            return f64::NAN;
        }
        self.signed_edge_bps * self.positioned_bars as f64 / moving as f64
    }
}

/// The traded panel's liquidity deciles, plus the same measurement pooled over all of it.
#[derive(Clone, Debug)]
pub struct EdgeVsCostTable {
    /// Thinnest first. Fewer than [`DECILES`] entries on a panel with fewer names than that.
    pub deciles: Vec<EdgeVsCost>,
    pub pooled: EdgeVsCost,
    /// Whether the cost half of the table was measured at all. `false` leaves every cost
    /// column NaN rather than filling it with a constant, because a cost nobody measured must
    /// not be renderable as a number.
    pub measured: bool,
    /// Block-bootstrap intervals on the EDGE half of the table.
    ///
    /// Every edge figure above is a point estimate over 11 trading days of one panel, and the
    /// session's principal economic conclusion - that the deepest liquidity decile is where the
    /// edge is WEAKEST, so restricting to cheap names makes the strategy worse rather than
    /// better - is a ratio of two of them. A ratio of two point estimates cannot be called
    /// negative until it is resolvably below one.
    pub intervals: EdgeIntervals,
}

// ---------------------------------------------------------------------------
// The interval on the edge half of the table
// ---------------------------------------------------------------------------

/// One edge statistic's block-bootstrap interval.
///
/// Deliberately not [`super::pretrain_stats::Dispersion`]: every statistic here is a RATIO of
/// two resampled sums - an edge over a bar count, an edge over a cost, one row's edge over
/// another's - rather than the mean of a resampled vector, so `mean` would be the wrong name
/// for [`Self::point`] and a caller could reasonably average two of them.
///
/// The resampling scheme IS [`super::pretrain_stats::block_bootstrap`]'s, down to the RNG: the
/// same `ChaCha12Rng` seeded once with [`BOOTSTRAP_SEED`], the same [`BOOTSTRAP_DRAWS`] draws,
/// as many blocks drawn with replacement per draw as there are blocks, and the same
/// linear-interpolated percentiles at [`CI_MASS`]. That is what makes an interval here
/// comparable to one from the trade bench rather than merely similar in construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeCi {
    /// The FULL-sample statistic, never the mean of the draws: the bootstrap sizes the
    /// interval and is not allowed to move the number it is an interval on.
    pub point: f64,
    /// Standard deviation of the draws.
    pub se: f64,
    pub lo: f64,
    pub hi: f64,
    /// Resampling units the interval was taken over. This, not the name-bar count, is what
    /// governs the width, so it is reported everywhere the interval is.
    pub blocks: usize,
    /// Draws that produced a finite statistic. Below half of [`BOOTSTRAP_DRAWS`] the interval
    /// is REFUSED rather than reported over the survivors.
    pub draws: usize,
}

impl EdgeCi {
    /// A point estimate with no interval around it: fewer than two blocks, or a statistic that
    /// is NaN because no cost model was supplied. The point is kept so the table can still
    /// print it, and [`Self::is_measured`] is what every verdict gates on.
    fn point_only(point: f64, blocks: usize, draws: usize) -> Self {
        Self {
            point,
            se: f64::NAN,
            lo: f64::NAN,
            hi: f64::NAN,
            blocks,
            draws,
        }
    }

    /// Whether an interval was measured at all. A point estimate alone is not one.
    pub fn is_measured(&self) -> bool {
        self.draws > 0 && self.lo.is_finite() && self.hi.is_finite()
    }

    /// `Some(true)` when the whole interval sits on one side of `value`, `Some(false)` when it
    /// straddles it, `None` when nothing was measured.
    ///
    /// Three states rather than a `bool`, because an UNMEASURED interval "does not exclude 1.0"
    /// in exactly the same way a wide one does not, and rendering those two identically is the
    /// defect class this session has found five times in four files.
    pub fn excludes(&self, value: f64) -> Option<bool> {
        if !self.is_measured() {
            return None;
        }
        Some(self.lo > value || self.hi < value)
    }

    /// Which side of `value` the interval is on, as a word rather than a sign.
    pub fn verdict(&self, value: f64) -> &'static str {
        if !self.is_measured() {
            return "n/a";
        }
        if self.lo > value {
            return "clears";
        }
        if self.hi < value {
            return "short";
        }
        "unresolved"
    }

    /// `point (lo..hi over N blocks)`, or an explicit refusal.
    pub fn text(&self) -> String {
        if !self.is_measured() {
            return format!("{:.4} unmeasured/{} blocks", self.point, self.blocks);
        }
        format!(
            "{:.4} ({:.4}..{:.4} /{})",
            self.point, self.lo, self.hi, self.blocks
        )
    }
}

/// One instant's contribution to one row's edge: the numerator and both denominators.
///
/// This is the resampling atom. Keeping it as three sums rather than as a per-instant MEAN is
/// what makes the bootstrap a ratio-of-sums estimator like the point estimate it intervals: an
/// average of per-instant means would weight a 1-name instant like a 99-name one, and this
/// panel's breadth runs from 1 to 99.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EdgeInstant {
    signed_sum: f64,
    signed_bars: u64,
    flat_bars: u64,
}

impl EdgeInstant {
    fn push(&mut self, signed: f64, flat: u64) {
        self.signed_sum += signed;
        self.signed_bars += 1;
        self.flat_bars += flat;
    }

    fn add(&mut self, other: &Self) {
        self.signed_sum += other.signed_sum;
        self.signed_bars += other.signed_bars;
        self.flat_bars += other.flat_bars;
    }

    /// Cross-sectional mean of `sign(f*) r` here, `None` when nothing was positioned. The
    /// Fama-MacBeth standard error's own unit, read off the same sums the bootstrap resamples.
    fn cross_sectional_mean(&self) -> Option<f64> {
        (self.signed_bars > 0).then(|| self.signed_sum / self.signed_bars as f64)
    }

    /// Identical arithmetic to [`EdgeVsCost::signed_edge_bps`]: flat bars in the denominator,
    /// which is what a cost comparison needs.
    fn signed_edge_bps(&self) -> f64 {
        if self.signed_bars == 0 {
            return f64::NAN;
        }
        1.0e4 * self.signed_sum / self.signed_bars as f64
    }

    /// Identical arithmetic to [`EdgeVsCost::signed_edge_per_moving_bar_bps`]: the version a
    /// RATIO between two rows must be taken over, since each row carries its own flat share.
    fn moving_edge_bps(&self) -> f64 {
        let moving = self.signed_bars.saturating_sub(self.flat_bars);
        if moving == 0 {
            return f64::NAN;
        }
        1.0e4 * self.signed_sum / moving as f64
    }
}

/// Every interval the edge table carries, under ONE blocking.
#[derive(Clone, Debug)]
pub struct EdgeCiSet {
    /// What a resampling unit IS. Printed beside every number taken over it, because the width
    /// is a property of this choice and of nothing else.
    pub blocking: &'static str,
    pub blocks: usize,
    pub instants: usize,
    /// Thinnest decile first, matching [`EdgeVsCostTable::deciles`].
    pub signed_edge_bps: Vec<EdgeCi>,
    pub pooled_signed_edge_bps: EdgeCi,
    /// Round trips the decile's edge pays for at the headline all-in cost, i.e.
    /// [`EdgeVsCost::edge_over_round_trip`]. `1.0` is the break-even a strategy must clear to
    /// exist at all.
    pub edge_over_round_trip: Vec<EdgeCi>,
    pub pooled_edge_over_round_trip: EdgeCi,
    /// The same at ZERO impact - see [`EdgeVsCost::impact_free_bps`].
    pub impact_free_over_round_trip: Vec<EdgeCi>,
    pub pooled_impact_free_over_round_trip: EdgeCi,
    /// Deepest decile's MOVING-bar edge over the pooled one, recomputed inside every draw so
    /// numerator and denominator move together. An unpaired interval on a ratio of two
    /// positively correlated quantities is too WIDE, which would understate a negative result.
    pub deepest_over_pooled: EdgeCi,
}

impl EdgeCiSet {
    /// The decile whose impact-free edge pays for the most round trips, by POINT estimate, with
    /// its interval.
    ///
    /// An argmax over ten correlated cells, so the point is biased UP and the interval is NOT
    /// selection-adjusted. That is the safe direction for the only claim made from it: if even
    /// the winner's unadjusted upper bound is below `1.0`, no selection adjustment can lift it
    /// above `1.0`, so "no decile clears one round trip" is conservative. The pooled row beside
    /// it is selection-free and is what a level should be quoted from.
    pub fn best_impact_free(&self) -> Option<(usize, EdgeCi)> {
        let mut best: Option<(usize, EdgeCi)> = None;
        for (decile, ci) in self.impact_free_over_round_trip.iter().enumerate() {
            if !ci.point.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, held)| ci.point > held.point) {
                best = Some((decile, *ci));
            }
        }
        best
    }
}

/// The edge table's intervals under both blockings this panel admits.
#[derive(Clone, Debug)]
pub struct EdgeIntervals {
    /// One resampling unit per TRADING DAY. The honest one: a 5-minute bar's edge is correlated
    /// with the next bar's and with every other name's at the same instant, and the day is the
    /// regime unit that owns both. It also needs no tuning parameter, which a fixed contiguous
    /// block length would, and this panel is 11 days long - so the number of blocks is small
    /// and is REPORTED rather than buried, because it is what the width is governed by.
    pub by_day: EdgeCiSet,
    /// One resampling unit per INSTANT.
    ///
    /// Carried as the control on the day blocking, not as a second estimate. Resampling instants
    /// is only a FLOOR on the width when the per-instant statistic is POSITIVELY serially
    /// correlated inside a day - then it divides the variance by a sample size the panel does
    /// not have. `the_day_blocked_edge_interval_is_wider_than_the_instant_blocked_one` builds a
    /// panel whose regime IS the day and pins that direction, so the machinery does widen when
    /// there is something to widen for.
    ///
    /// On the real panel there is not. The pooled level reads `0.4988..1.3092` day-blocked
    /// against `0.4416..1.2615` instant-blocked - widths `0.810` and `0.820`, agreeing to 1.2%,
    /// with the instant one marginally the WIDER of the two rather than the floor. So the
    /// cross-sectional mean edge of a 5-minute instant carries no day-level regime this panel
    /// can detect, and neither verdict below depends on which blocking a reader trusts. The
    /// paired ratio spreads a little further (`0.7396` day against `0.8013` instant) in the same
    /// direction: day blocking keeps a day's numerator and denominator inside one draw, so the
    /// factor common to both cancels there and not under instant blocking.
    pub by_instant: EdgeCiSet,
}

/// Instants grouped into resampling units by `ids[t]`, ascending by id.
fn group_instants(ids: &[u64]) -> Vec<Vec<usize>> {
    let mut grouped: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, id) in ids.iter().enumerate() {
        grouped.entry(*id).or_default().push(index);
    }
    grouped.into_values().collect()
}

/// Linear-interpolated percentile of an ascending slice, [`block_bootstrap`]'s own convention.
///
/// [`block_bootstrap`]: super::pretrain_stats::block_bootstrap
fn percentile_of(sorted: &[f64], q: f64) -> f64 {
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

/// Resample `blocks` with replacement and recompute EVERY edge statistic inside each draw.
///
/// `rows[d][t]` is decile `d`'s partial sums at instant `t`, `pooled[t]` the same over the whole
/// panel. One index sequence per draw drives every statistic, which is what makes the ratios
/// paired rather than merely simultaneous.
///
/// The COST arguments are held FIXED across draws, and that is a scope statement rather than an
/// approximation: `all_in` and `impact_free` are cross-sectional means and medians over
/// symbol-MONTHS measured off stored bars, not time-series averages over the traded instants,
/// so resampling instants cannot move them. Every interval here is therefore sampling error in
/// the EDGE and in nothing else; widening it for cost uncertainty needs a second resampling of
/// the calibration and is a different measurement.
#[allow(clippy::too_many_arguments)]
fn bootstrap_edge(
    rows: &[Vec<EdgeInstant>],
    pooled: &[EdgeInstant],
    all_in: &[f64],
    impact_free: &[f64],
    pooled_all_in: f64,
    pooled_impact_free: f64,
    blocks: &[Vec<usize>],
    blocking: &'static str,
) -> EdgeCiSet {
    let buckets = rows.len();
    let block_count = blocks.len();
    // Per-(row, block) partial sums, so one draw is a pass over the BLOCKS rather than over the
    // instants. Without this, 1000 paired draws over 2000 instants and 11 rows would be 22M row
    // updates; with it, a day-blocked draw touches 11 blocks.
    let mut block_rows: Vec<Vec<EdgeInstant>> =
        vec![vec![EdgeInstant::default(); block_count]; buckets];
    let mut block_pooled: Vec<EdgeInstant> = vec![EdgeInstant::default(); block_count];
    for (block, members) in blocks.iter().enumerate() {
        for &t in members {
            for (decile, row) in rows.iter().enumerate() {
                block_rows[decile][block].add(&row[t]);
            }
            block_pooled[block].add(&pooled[t]);
        }
    }
    let sum_all = |cells: &[EdgeInstant]| {
        let mut total = EdgeInstant::default();
        for cell in cells {
            total.add(cell);
        }
        total
    };
    let full_rows: Vec<EdgeInstant> = block_rows.iter().map(|row| sum_all(row)).collect();
    let full_pooled = sum_all(&block_pooled);
    let trip = |edge: f64, cost: f64| edge / (2.0 * cost);

    let mut edge_draws: Vec<Vec<f64>> = vec![Vec::with_capacity(BOOTSTRAP_DRAWS); buckets];
    let mut trip_draws: Vec<Vec<f64>> = vec![Vec::with_capacity(BOOTSTRAP_DRAWS); buckets];
    let mut free_draws: Vec<Vec<f64>> = vec![Vec::with_capacity(BOOTSTRAP_DRAWS); buckets];
    let mut pooled_edge_draws: Vec<f64> = Vec::with_capacity(BOOTSTRAP_DRAWS);
    let mut pooled_trip_draws: Vec<f64> = Vec::with_capacity(BOOTSTRAP_DRAWS);
    let mut pooled_free_draws: Vec<f64> = Vec::with_capacity(BOOTSTRAP_DRAWS);
    let mut ratio_draws: Vec<f64> = Vec::with_capacity(BOOTSTRAP_DRAWS);
    // One block is one observation: there is no dispersion to estimate and pretending otherwise
    // would report a zero-width interval as though it were precision. Same rule, same reason as
    // `block_bootstrap`'s own guard.
    if block_count >= 2 {
        let indices: Vec<usize> = (0..block_count).collect();
        let mut rng = ChaCha12Rng::seed_from_u64(BOOTSTRAP_SEED);
        let mut totals = vec![EdgeInstant::default(); buckets];
        for _ in 0..BOOTSTRAP_DRAWS {
            totals.iter_mut().for_each(|cell| *cell = EdgeInstant::default());
            let mut pooled_total = EdgeInstant::default();
            for _ in 0..block_count {
                let block = *indices.choose(&mut rng).expect("a block index exists");
                for (decile, total) in totals.iter_mut().enumerate() {
                    total.add(&block_rows[decile][block]);
                }
                pooled_total.add(&block_pooled[block]);
            }
            let pooled_edge = pooled_total.signed_edge_bps();
            pooled_edge_draws.push(pooled_edge);
            pooled_trip_draws.push(trip(pooled_edge, pooled_all_in));
            pooled_free_draws.push(trip(pooled_edge, pooled_impact_free));
            for (decile, total) in totals.iter().enumerate() {
                let edge = total.signed_edge_bps();
                edge_draws[decile].push(edge);
                trip_draws[decile].push(trip(edge, all_in[decile]));
                free_draws[decile].push(trip(edge, impact_free[decile]));
            }
            if let Some(deepest) = totals.last() {
                ratio_draws.push(deepest.moving_edge_bps() / pooled_total.moving_edge_bps());
            }
        }
    }

    let finish = |point: f64, draws: &mut Vec<f64>| -> EdgeCi {
        draws.retain(|value| value.is_finite());
        if !point.is_finite() || 2 * draws.len() < BOOTSTRAP_DRAWS {
            return EdgeCi::point_only(point, block_count, draws.len());
        }
        draws.sort_by(f64::total_cmp);
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance = draws
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / (draws.len() - 1) as f64;
        let tail = 0.5 * (1.0 - CI_MASS);
        EdgeCi {
            point,
            se: variance.sqrt(),
            lo: percentile_of(draws, tail),
            hi: percentile_of(draws, 1.0 - tail),
            blocks: block_count,
            draws: draws.len(),
        }
    };
    let deepest_point = full_rows
        .last()
        .map_or(f64::NAN, |deepest| {
            deepest.moving_edge_bps() / full_pooled.moving_edge_bps()
        });
    EdgeCiSet {
        blocking,
        blocks: block_count,
        instants: pooled.len(),
        signed_edge_bps: (0..buckets)
            .map(|decile| finish(full_rows[decile].signed_edge_bps(), &mut edge_draws[decile]))
            .collect(),
        pooled_signed_edge_bps: finish(full_pooled.signed_edge_bps(), &mut pooled_edge_draws),
        edge_over_round_trip: (0..buckets)
            .map(|decile| {
                finish(
                    trip(full_rows[decile].signed_edge_bps(), all_in[decile]),
                    &mut trip_draws[decile],
                )
            })
            .collect(),
        pooled_edge_over_round_trip: finish(
            trip(full_pooled.signed_edge_bps(), pooled_all_in),
            &mut pooled_trip_draws,
        ),
        impact_free_over_round_trip: (0..buckets)
            .map(|decile| {
                finish(
                    trip(full_rows[decile].signed_edge_bps(), impact_free[decile]),
                    &mut free_draws[decile],
                )
            })
            .collect(),
        pooled_impact_free_over_round_trip: finish(
            trip(full_pooled.signed_edge_bps(), pooled_impact_free),
            &mut pooled_free_draws,
        ),
        deepest_over_pooled: finish(deepest_point, &mut ratio_draws),
    }
}

/// Accumulator for one decile's name-bars.
#[derive(Clone, Debug, Default)]
struct EdgeAccum {
    bars: u64,
    r_sum: f64,
    r_squares: f64,
    signed_sum: f64,
    signed_bars: u64,
    agreements: u64,
    /// Positioned bars with `realized == 0.0`: no sign to agree with, zero edge contribution,
    /// and a full unit of both denominators. Counted so the attenuation is reportable.
    flat_bars: u64,
    kelly_r_sum: f64,
    abs_kelly_sum: f64,
    forecast_sum: f64,
    forecast_bars: u64,
    abs_kelly: Vec<f64>,
    /// Cross-sectional mean of `sign(f*) r` at each instant that had any, for the standard
    /// error that respects the cross-section.
    instant_means: Vec<f64>,
}

impl EdgeAccum {
    fn push(&mut self, realized: f64, kelly: f64, forecast: f64) {
        self.bars += 1;
        self.r_sum += realized;
        self.r_squares += realized * realized;
        if kelly.is_finite() && kelly != 0.0 {
            self.signed_bars += 1;
            // `sign(f*) * r`, which for a CORRECT short is a positive contribution. Writing
            // this as `realized.copysign(kelly)` would instead be `sign(f*) * |r|` - it would
            // pay a right-side short a negative edge and report roughly zero on any panel whose
            // book is two-sided, which is a directional statistic that cannot see direction.
            self.signed_sum += if kelly > 0.0 { realized } else { -realized };
            if kelly * realized > 0.0 {
                self.agreements += 1;
            } else if realized == 0.0 {
                self.flat_bars += 1;
            }
            self.kelly_r_sum += kelly * realized;
            self.abs_kelly_sum += kelly.abs();
            self.abs_kelly.push(kelly.abs());
        }
        if forecast.is_finite() {
            self.forecast_bars += 1;
            self.forecast_sum += forecast;
        }
    }

    fn finish(mut self, decile: usize) -> EdgeVsCost {
        let mut out = EdgeVsCost::empty(decile);
        out.name_bars = self.bars;
        if self.bars == 0 {
            return out;
        }
        let n = self.bars as f64;
        let mean = self.r_sum / n;
        let variance = (self.r_squares / n - mean * mean).max(0.0);
        out.mean_r_bps = 1.0e4 * mean;
        out.sd_r_bps = 1.0e4 * variance.sqrt();
        out.positioned_bars = self.signed_bars;
        out.flat_positioned_bars = self.flat_bars;
        if self.signed_bars > 0 {
            out.signed_edge_bps = 1.0e4 * self.signed_sum / self.signed_bars as f64;
            out.forecast_sign_agreement = self.agreements as f64 / self.signed_bars as f64;
            out.median_abs_kelly = median_of(&mut self.abs_kelly);
        }
        if self.abs_kelly_sum > 0.0 {
            out.kelly_weighted_edge_bps = 1.0e4 * self.kelly_r_sum / self.abs_kelly_sum;
        }
        if self.forecast_bars > 0 {
            out.mean_forecast_bps = 1.0e4 * self.forecast_sum / self.forecast_bars as f64;
        }
        let instants = self.instant_means.len();
        if instants > 1 {
            let mean_of_means = self.instant_means.iter().sum::<f64>() / instants as f64;
            let variance = self
                .instant_means
                .iter()
                .map(|m| (m - mean_of_means) * (m - mean_of_means))
                .sum::<f64>()
                / (instants - 1) as f64;
            out.signed_edge_se_bps = 1.0e4 * (variance / instants as f64).sqrt();
        }
        out
    }
}

impl EdgeVsCostTable {
    /// Measure the panel's own liquidity deciles: the edge from `inputs.model`, the cost from
    /// `model` when one is supplied.
    ///
    /// Deciles are cut on the panel's OWN trailing dollar volume — the same causal quantity
    /// [`backtest`] prices size against — rather than on the calibration's span-pooled ADV, so
    /// the ranking exists even with no cost model and never looks forward.
    pub fn measure(panel: &Panel, inputs: &PolicyInputs<'_>, model: Option<&BarCostModel>) -> Self {
        let names = panel.symbols().len();
        let mut volume = vec![0.0f64; names];
        let mut counted = vec![0u64; names];
        for (t, slice) in panel.slices().iter().enumerate() {
            for (k, &id) in slice.symbols.iter().enumerate() {
                let adv = f64::from(panel.adv_usd(t, k));
                if adv.is_finite() && adv > 0.0 {
                    volume[id as usize] += adv;
                    counted[id as usize] += 1;
                }
            }
        }
        let mean_adv: Vec<f64> = (0..names)
            .map(|id| {
                if counted[id] > 0 {
                    volume[id] / counted[id] as f64
                } else {
                    f64::NAN
                }
            })
            .collect();
        // An unmeasurable ADV sorts to the THINNEST decile, where an untradeable name belongs:
        // IEEE total order would rank a NaN above `+inf` and put it in the deepest.
        let mut ranked: Vec<usize> = (0..names).collect();
        let key = |adv: f64| if adv.is_finite() { adv } else { f64::NEG_INFINITY };
        ranked.sort_by(|a, b| key(mean_adv[*a]).total_cmp(&key(mean_adv[*b])).then(a.cmp(b)));

        let buckets = DECILES.min(names.max(1));
        let mut decile_of = vec![0usize; names];
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); buckets];
        for (rank, &id) in ranked.iter().enumerate() {
            let decile = (rank * buckets / names).min(buckets - 1);
            decile_of[id] = decile;
            members[decile].push(id);
        }

        let mut accum = vec![EdgeAccum::default(); buckets];
        let mut pooled = EdgeAccum::default();
        // Per-instant partial sums, kept for the block bootstrap and REDUCED to the Fama-MacBeth
        // cross-sectional means below rather than accumulated a second time: the standard error
        // and the interval must be two readings of one quantity, not two implementations of it.
        let instants = panel.slices().len();
        let mut row_instants: Vec<Vec<EdgeInstant>> = vec![Vec::with_capacity(instants); buckets];
        let mut pooled_instants: Vec<EdgeInstant> = Vec::with_capacity(instants);
        let mut day_of: Vec<u64> = Vec::with_capacity(instants);
        let mut cells = vec![EdgeInstant::default(); buckets];
        for (t, slice) in panel.slices().iter().enumerate() {
            cells.iter_mut().for_each(|cell| *cell = EdgeInstant::default());
            let mut pooled_cell = EdgeInstant::default();
            let forecast = &inputs.model[t];
            for (k, &id) in slice.symbols.iter().enumerate() {
                let realized = f64::from(slice.realized_r[k]);
                let kelly = f64::from(forecast.kelly_f[k]);
                let mean_r = f64::from(forecast.mean_r[k]);
                let decile = decile_of[id as usize];
                accum[decile].push(realized, kelly, mean_r);
                pooled.push(realized, kelly, mean_r);
                if kelly.is_finite() && kelly != 0.0 {
                    let signed = if kelly > 0.0 { realized } else { -realized };
                    // Exactly `EdgeAccum::push`'s flat branch: a positioned bar whose realized
                    // return is zero has no sign to be right about and contributes no edge.
                    let flat = u64::from(realized == 0.0);
                    cells[decile].push(signed, flat);
                    pooled_cell.push(signed, flat);
                }
            }
            for (decile, cell) in cells.iter().enumerate() {
                if let Some(mean) = cell.cross_sectional_mean() {
                    accum[decile].instant_means.push(mean);
                }
                row_instants[decile].push(*cell);
            }
            if let Some(mean) = pooled_cell.cross_sectional_mean() {
                pooled.instant_means.push(mean);
            }
            pooled_instants.push(pooled_cell);
            // A UTC day is a trading day on this corpus: US regular hours are 13:30-20:00 UTC,
            // so no session straddles midnight and no block splits one in half. This is the
            // panel's OWN trading-day definition, the one `Panel::trading_days` counts.
            day_of.push(slice.ts_ms.div_euclid(MS_PER_DAY) as u64);
        }

        let mut deciles: Vec<EdgeVsCost> = accum
            .into_iter()
            .enumerate()
            .map(|(decile, a)| a.finish(decile))
            .collect();
        let mut pooled = pooled.finish(usize::MAX);
        for (decile, row) in deciles.iter_mut().enumerate() {
            row.symbols = members[decile].len();
            let mut advs: Vec<f64> = members[decile].iter().map(|id| mean_adv[*id]).collect();
            row.median_adv_usd = median_of(&mut advs);
        }
        pooled.symbols = names;
        let mut all_advs = mean_adv.clone();
        pooled.median_adv_usd = median_of(&mut all_advs);

        if let Some(model) = model {
            for (decile, row) in deciles.iter_mut().enumerate() {
                price_decile(row, panel, model, &members[decile]);
            }
            price_decile(&mut pooled, panel, model, &ranked);
        }
        let all_in: Vec<f64> = deciles.iter().map(EdgeVsCost::headline_all_in_bps).collect();
        let impact_free: Vec<f64> = deciles.iter().map(EdgeVsCost::impact_free_bps).collect();
        let pooled_all_in = pooled.headline_all_in_bps();
        let pooled_impact_free = pooled.impact_free_bps();
        let intervals = EdgeIntervals {
            by_day: bootstrap_edge(
                &row_instants,
                &pooled_instants,
                &all_in,
                &impact_free,
                pooled_all_in,
                pooled_impact_free,
                &group_instants(&day_of),
                "trading day",
            ),
            by_instant: bootstrap_edge(
                &row_instants,
                &pooled_instants,
                &all_in,
                &impact_free,
                pooled_all_in,
                pooled_impact_free,
                &(0..instants).map(|t| vec![t]).collect::<Vec<Vec<usize>>>(),
                "instant",
            ),
        };
        Self {
            deciles,
            pooled,
            measured: model.is_some(),
            intervals,
        }
    }

    /// The interval half of the table: what the edge figures above are worth as MEASUREMENTS.
    ///
    /// Two verdicts live here and nothing else in this module can supply them. Whether any
    /// decile's edge resolvably pays for one round trip, and whether the deepest decile's edge
    /// is resolvably below the pooled one - the negative-liquidity result, which is a ratio of
    /// two point estimates until this says otherwise.
    pub fn interval_table(&self) -> String {
        let day = &self.intervals.by_day;
        let instant = &self.intervals.by_instant;
        let mut out = format!(
            "  INTERVALS, EDGE ONLY: block bootstrap over {} {} blocks ({} instants), {} draws, \
             every ratio recomputed inside each draw so numerator and denominator are PAIRED. \
             The cost denominator is held FIXED - it is a cross-sectional mean and median over \
             symbol-months off stored bars, not a time-series average over the traded instants \
             - so this is sampling error in the EDGE and in nothing else.\n",
            day.blocks, day.blocking, day.instants, BOOTSTRAP_DRAWS,
        );
        out.push_str(&format!(
            "{:<8}{:>30}{:>30}{:>12}\n",
            "decile", "round trips @ all-in", "round trips @ zero impact", "clears 1?",
        ));
        let row = |label: String, all_in: &EdgeCi, free: &EdgeCi| {
            format!(
                "{label:<8}{:>30}{:>30}{:>12}\n",
                all_in.text(),
                free.text(),
                free.verdict(1.0),
            )
        };
        for (decile, (all_in, free)) in day
            .edge_over_round_trip
            .iter()
            .zip(&day.impact_free_over_round_trip)
            .enumerate()
        {
            out.push_str(&row(format!("{decile}"), all_in, free));
        }
        out.push_str(&row(
            "pooled".to_owned(),
            &day.pooled_edge_over_round_trip,
            &day.pooled_impact_free_over_round_trip,
        ));
        // The LEVEL under both blockings. The trips columns above are this number over a cost
        // held fixed, so their widths scale with it exactly - which makes this the one line
        // where a reader can see whether the choice of blocking is doing any work at all. What
        // gets printed is the two widths, not a claimed direction between them.
        let width = |ci: &EdgeCi| if ci.is_measured() { ci.hi - ci.lo } else { f64::NAN };
        out.push_str(&format!(
            "  pooled signed edge per positioned bar: {} bps by {} (width {:.4}); {} bps by {} \
             (width {:.4})\n",
            day.pooled_signed_edge_bps.text(),
            day.blocking,
            width(&day.pooled_signed_edge_bps),
            instant.pooled_signed_edge_bps.text(),
            instant.blocking,
            width(&instant.pooled_signed_edge_bps),
        ));
        let excludes = |state: Option<bool>| match state {
            None => "never measured against",
            Some(true) => "EXCLUDES",
            Some(false) => "straddles",
        };
        out.push_str(&format!(
            "  deepest over pooled, per MOVING bar: {} by {} - {} 1.0; {} by {} - {} 1.0 (widths \
             {:.4} and {:.4}). Instant blocking returns the narrower interval only where the \
             per-instant statistic is serially correlated inside the day; where the two widths \
             agree, the verdict does not rest on which one a reader trusts.\n",
            day.deepest_over_pooled.text(),
            day.blocking,
            excludes(day.deepest_over_pooled.excludes(1.0)),
            instant.deepest_over_pooled.text(),
            instant.blocking,
            excludes(instant.deepest_over_pooled.excludes(1.0)),
            width(&day.deepest_over_pooled),
            width(&instant.deepest_over_pooled),
        ));
        if let Some((decile, ci)) = day.best_impact_free() {
            // The reciprocal only where the interval is entirely on the positive side: with a
            // lower bound at or below zero the shortfall is unbounded above, and printing
            // `1 / lo` there would render an infinite shortfall as a finite one.
            let shortfall = if ci.is_measured() && ci.lo > 0.0 {
                format!("shortfall {:.1}x ({:.1}x..{:.1}x)", 1.0 / ci.point, 1.0 / ci.hi, 1.0 / ci.lo)
            } else if ci.point > 0.0 {
                format!("shortfall {:.1}x, unbounded above (the edge interval reaches zero)", 1.0 / ci.point)
            } else {
                "no shortfall is definable: the edge itself is not positive".to_owned()
            };
            out.push_str(&format!(
                "  best decile at ZERO impact, by point estimate: decile {decile}, {} round \
                 trips, {shortfall} - an ARGMAX over {} correlated cells, so the point is biased \
                 UP and the interval is NOT selection-adjusted. That is conservative for the \
                 only claim made from it: an upper bound below 1.0 cannot be lifted above 1.0 by \
                 any selection adjustment. The pooled row is the selection-free one.\n",
                ci.text(),
                day.impact_free_over_round_trip.len(),
            ));
        }
        out
    }
}

/// Fill one row's cost columns from the measured model, over `members` (panel symbol ids).
///
/// Span-POOLED liquidity rather than one month's: a decile is a statement about the symbols in
/// it over the whole traded span, and a per-month table would be one table per month.
fn price_decile(row: &mut EdgeVsCost, panel: &Panel, model: &BarCostModel, members: &[usize]) {
    if members.is_empty() {
        return;
    }
    let participation = PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT];
    let mut half = 0.0f64;
    let mut commission = 0.0f64;
    let mut regulatory = 0.0f64;
    let mut impact = [0.0f64; IMPACT_K_GRID.len()];
    let mut all_in: Vec<Vec<f64>> = IMPACT_K_GRID.iter().map(|_| Vec::new()).collect();
    let (mut spread_substituted, mut impact_substituted) = (0usize, 0usize);
    for &id in members {
        let series = panel.series_of(id as u32) as u32;
        let resolved = model.resolve_pooled(series);
        half += resolved.half_spread_bps;
        commission += resolved.commission_bps;
        regulatory += resolved.regulatory_bps;
        if resolved.spread_fallback {
            spread_substituted += 1;
        }
        if !resolved.impact_coefficient_bps.is_finite() {
            impact_substituted += 1;
        }
        for (slot, k) in IMPACT_K_GRID.iter().enumerate() {
            // The resolved coefficient carries the model's own `impact_k`; rescaling it is
            // exact and avoids re-resolving the same symbol three times.
            let scaled = resolved.impact_coefficient_bps * k / model.impact_k();
            let leg = scaled * participation.sqrt();
            impact[slot] += leg;
            all_in[slot].push(resolved.fixed_bps() + leg);
        }
    }
    let count = members.len() as f64;
    row.half_spread_bps = half / count;
    row.commission_bps = commission / count;
    row.regulatory_bps = regulatory / count;
    row.spread_substituted = spread_substituted;
    row.impact_substituted = impact_substituted;
    for slot in 0..IMPACT_K_GRID.len() {
        row.impact_bps[slot] = impact[slot] / count;
        row.all_in_bps[slot] =
            row.half_spread_bps + row.commission_bps + row.regulatory_bps + row.impact_bps[slot];
        row.median_all_in_bps[slot] = median_of(&mut all_in[slot]);
    }
}

/// Median of the finite entries, NaN when there are none. Sorts `values` in place.
fn median_of(values: &mut Vec<f64>) -> f64 {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// Sweep [`BAND_FRACTIONS`] for every policy at [`DEFAULT_GROSS_CAP`] under one cost model.
///
/// A typical position is the gross budget spread over the mean number of names actually
/// tradeable, so the band grid is stated in multiples of one and converted here: the same
/// fraction means the same thing on a 4-name fixture and on a 100-name panel, which an
/// absolute weight would not.
fn frontier_arm(
    panel: &Panel,
    inputs: &PolicyInputs<'_>,
    cost: &dyn CostModel,
    arm: CostArm,
    config: &BacktestConfig,
) -> Result<FrontierArm> {
    let breadth = panel.breadth().mean.max(1.0);
    let mut points = Vec::with_capacity(POLICIES.len());
    for policy in POLICIES {
        // The band-zero book of the SAME arm is the denominator of the turnover share, so a
        // measured arm's saving is measured against its own unbanded trading rather than
        // against the flat arm's - the two hold identical positions, but reading one arm's
        // turnover against another's would only be right by that coincidence.
        let mut row: Vec<FrontierPoint> = Vec::with_capacity(BAND_FRACTIONS.len());
        let mut unbanded = f64::NAN;
        for fraction in BAND_FRACTIONS {
            let band = fraction * DEFAULT_GROSS_CAP / breadth;
            let banded = backtest(
                panel,
                inputs,
                policy,
                DEFAULT_GROSS_CAP,
                cost,
                &BacktestConfig { band, ..*config },
            )?;
            let traded: f64 = banded.turnover.iter().sum();
            if fraction == 0.0 {
                unbanded = traded;
            }
            let m = &banded.metrics;
            row.push(FrontierPoint {
                policy: policy.name(),
                band_fraction: fraction,
                band,
                turnover_per_day: m.turnover_per_day,
                gross_log_growth_per_year: m.gross_log_growth_per_year,
                log_growth_per_year: m.log_growth_per_year,
                break_even_cost_bps: m.break_even_cost_bps,
                mean_gross: m.mean_gross,
                sharpe: m.sharpe,
                cost_bps_per_instant: m.cost_bps_per_instant,
                rotation_per_instant: m.rotation_per_instant,
                turnover_share_of_unbanded: if unbanded > 0.0 {
                    traded / unbanded
                } else {
                    f64::NAN
                },
                realized_cost_bps: m.realized_cost_bps,
                legs: banded.legs,
                bar_win_rate: m.bar_win_rate,
                trade_win_rate: m.trade_win_rate,
                position_sign_agreement: m.position_sign_agreement,
                payoff_bps_per_instant: m.payoff_bps_per_instant,
                payoff_sd_bps_per_instant: m.payoff_sd_bps_per_instant,
            });
        }
        points.push(row);
    }
    Ok(FrontierArm { arm, points })
}

/// Every policy at every gross cap on ONE panel, plus the panel's own description.
#[derive(Clone, Debug)]
pub struct PortfolioBench {
    pub instants: usize,
    pub symbols: usize,
    pub first_ts_ms: i64,
    pub last_ts_ms: i64,
    pub span_years: f64,
    pub instants_per_year: f64,
    pub trading_days: usize,
    pub breadth: Breadth,
    pub cost_bps: f32,
    /// `runs[cap][policy]`, in [`GROSS_CAPS`] and [`POLICIES`] order. Band zero throughout.
    pub runs: Vec<Vec<PortfolioRun>>,
    /// One entry per COST ARM, `arms[0]` always being the flat reference at
    /// [`Self::cost_bps`]. Each arm holds `points[policy][band]` in [`POLICIES`] and
    /// [`BAND_FRACTIONS`] order, all at [`DEFAULT_GROSS_CAP`].
    ///
    /// The same weights, the same panel and the same turnover under every cost model: nothing
    /// in the sizing consults the cost, so the arms differ by exactly the thing being varied.
    pub arms: Vec<FrontierArm>,
    /// The traded panel's own liquidity deciles: edge per name-bar against cost per trade.
    pub edge: EdgeVsCostTable,
}

impl PortfolioBench {
    /// Run every policy at every cap on one panel, then sweep the band under every cost arm.
    ///
    /// `cost` is the reference arm — the flat constant every number quoted before the measured
    /// model existed was computed under — and it is the one the `runs` grid and the equity
    /// curves use, so those stay exactly comparable. `measured`, when supplied, adds the
    /// per-symbol spread, commission, regulatory fee and square-root impact of [`PanelCost`]:
    /// once per [`IMPACT_K_GRID`] coefficient because that coefficient is a stated literature
    /// default and not a fit, once more per coefficient with the commission and the regulatory
    /// fee set to exactly zero because "would a cheaper broker fix this" is a question about
    /// the decomposition that only a second backtest can answer, and once with the impact term
    /// removed entirely — the arm in which every charged term was measured from the bars.
    ///
    /// `measured` must have been calibrated on the same corpus `panel` was built from; the id
    /// translation is [`PanelCost`]'s and is the reason this takes the panel.
    pub fn run(
        panel: &Panel,
        inputs: &PolicyInputs<'_>,
        cost: &dyn CostModel,
        cost_bps: f32,
        measured: Option<&BarCostModel>,
        config: &BacktestConfig,
    ) -> Result<Self> {
        let mut runs = Vec::with_capacity(GROSS_CAPS.len());
        for cap in GROSS_CAPS {
            let mut row = Vec::with_capacity(POLICIES.len());
            for policy in POLICIES {
                row.push(backtest(panel, inputs, policy, cap, cost, config)?);
            }
            runs.push(row);
        }

        let mut arms = vec![frontier_arm(
            panel,
            inputs,
            cost,
            CostArm::Flat { bps: cost_bps },
            config,
        )?];
        if let Some(measured) = measured {
            // The IMPACT-FREE arm first, because it is the one an argument may rest on: no
            // assumed coefficient enters it, so a verdict here cannot be moved by disputing
            // `IMPACT_K`. `impact_k` is carried for shape only and charges nothing.
            arms.push(frontier_arm(
                panel,
                inputs,
                &PanelCost::new(panel, measured.clone(), CostParts::NoImpact),
                CostArm::Measured {
                    impact_k: IMPACT_K,
                    parts: CostParts::NoImpact,
                },
                config,
            )?);
            for parts in [CostParts::All, CostParts::NoFees] {
                for impact_k in IMPACT_K_GRID {
                    let priced = PanelCost::new(panel, measured.with_impact_k(impact_k), parts);
                    arms.push(frontier_arm(
                        panel,
                        inputs,
                        &priced,
                        CostArm::Measured { impact_k, parts },
                        config,
                    )?);
                }
            }
        }
        Ok(Self {
            instants: panel.instants(),
            symbols: panel.symbols().len(),
            first_ts_ms: panel.slices().first().map_or(0, |s| s.ts_ms),
            last_ts_ms: panel.slices().last().map_or(0, |s| s.ts_ms),
            span_years: panel.span_years(),
            instants_per_year: panel.instants_per_year(),
            trading_days: panel.trading_days(),
            breadth: panel.breadth(),
            cost_bps,
            runs,
            edge: EdgeVsCostTable::measure(panel, inputs, measured),
            arms,
        })
    }

    /// The FLAT reference arm, which is `arms[0]` by construction.
    pub fn flat_arm(&self) -> &FrontierArm {
        &self.arms[0]
    }

    /// The measured arm at the headline impact coefficient charging every component, if one was
    /// run. What a trader would actually pay.
    pub fn headline_measured_arm(&self) -> Option<&FrontierArm> {
        self.arms.iter().find(|a| {
            a.arm
                == CostArm::Measured {
                    impact_k: IMPACT_K,
                    parts: CostParts::All,
                }
        })
    }

    /// The arm every charged term of which was MEASURED, with no assumed coefficient in it.
    pub fn assumption_free_arm(&self) -> Option<&FrontierArm> {
        self.arms.iter().find(|a| a.arm.is_assumption_free())
    }

    pub fn metrics(&self, cap: usize, policy: usize) -> &PortfolioMetrics {
        &self.runs[cap][policy].metrics
    }

    /// The headline: the model's book at [`DEFAULT_GROSS_CAP`].
    pub fn headline(&self) -> &PortfolioMetrics {
        self.metrics(DEFAULT_GROSS_SLOT, 0)
    }

    /// A plain-text table, the form a trader reads.
    pub fn table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "panel: {} instants x {} symbols, {} .. {} ({:.4} y, {:.0} instants/y, {} \
             trading days), breadth mean {:.1} min {} max {}, cost {:.2} bps\n",
            self.instants,
            self.symbols,
            self.first_ts_ms,
            self.last_ts_ms,
            self.span_years,
            self.instants_per_year,
            self.trading_days,
            self.breadth.mean,
            self.breadth.min,
            self.breadth.max,
            self.cost_bps,
        ));
        out.push_str(&format!(
            "{:<24}{:>6}{:>10}{:>10}{:>9}{:>12}{:>9}{:>9}{:>9}{:>9}{:>10}{:>7}{:>8}{:>8}\n",
            "policy",
            "gross",
            "grossLnG",
            "netLnG",
            "b/e bps",
            "CAGR",
            "Sharpe",
            "vol",
            "maxDD",
            "meanNet",
            "turn/day",
            "bound",
            "factor",
            "levErr",
        ));
        for (c, cap) in GROSS_CAPS.iter().enumerate() {
            for (p, _) in POLICIES.iter().enumerate() {
                let m = self.metrics(c, p);
                // CAGR past a few hundred percent is unreadable and past `e^709` is not
                // representable at all, so the log growth is printed FIRST and the CAGR is
                // elided where it has stopped being a number a reader can use.
                let cagr = if m.cagr.abs() < 1.0e6 {
                    format!("{:>11.2}%", 100.0 * m.cagr)
                } else {
                    format!("{:>12}", "off scale")
                };
                let break_even = if m.break_even_cost_bps.is_finite() {
                    format!("{:>9.2}", m.break_even_cost_bps)
                } else {
                    format!("{:>9}", "never")
                };
                out.push_str(&format!(
                    "{:<24}{:>6.1}{:>10.3}{:>10.3}{break_even}{cagr}{:>9.2}{:>8.1}%{:>8.1}%\
                     {:>9.2}{:>10.2}{:>6.0}%{:>8.3}{:>8.2}\n",
                    m.policy,
                    cap,
                    m.gross_log_growth_per_year,
                    m.log_growth_per_year,
                    m.sharpe,
                    100.0 * m.vol,
                    100.0 * m.max_drawdown,
                    m.mean_net,
                    m.turnover_per_day,
                    100.0 * m.bound_fraction,
                    m.mean_first_factor_exposure,
                    m.leverage_error,
                ));
            }
        }
        out
    }

    /// The turnover-edge frontier of ONE arm: what the book earns once it stops trading so
    /// much, and whether that is enough.
    ///
    /// Break-even cost is gross edge divided by turnover. Banding buys break-even by cutting
    /// the denominator and pays for it out of the numerator, because part of any edge at a
    /// 5-minute horizon IS the high-frequency part. This table is where the two curves meet,
    /// and the column that decides everything is `b/e bps` against the real cost of trading.
    ///
    /// `cost_floor_bps` is the cost the `pays` column tests against for arms that have no cost
    /// of their own to quote; a measured arm quotes what it actually PAID per dollar traded and
    /// tests against that instead, which is the comparison that decides the strategy.
    pub fn frontier_table(&self, arm: &FrontierArm, cost_floor_bps: f64) -> String {
        let mut out = format!(
            "turnover-edge frontier at {DEFAULT_GROSS_CAP:.1}x gross under [{}], band in \
             multiples of a typical position ({:.1} names); a point is tradeable when b/e \
             exceeds the one-way cost it paid ({} bps when the arm is a constant)\n",
            arm.arm.label(),
            self.breadth.mean,
            format_args!("{cost_floor_bps:.2}"),
        );
        out.push_str(&format!(
            "{:<22}{:>6}{:>9}{:>8}{:>8}{:>9}{:>9}{:>8}{:>8}{:>5}{:>8}{:>8}{:>8}{:>7}{:>7}{:>7}\
             {:>7}\n",
            "policy",
            "band",
            "turn/day",
            "rot/bar",
            "turnShr",
            "grossLnG",
            "netLnG",
            "b/e bps",
            "paid bps",
            "pays",
            "Sharpe",
            "mu bps",
            "sd bps",
            "barWin",
            "trdWin",
            "sgnAgr",
            "subLeg",
        ));
        for row in &arm.points {
            for point in row {
                // Three states, three labels. `is_finite()` alone would print a NEVER-TRADED
                // book's NaN as "never", which is a verdict about a measurement that does not
                // exist; `+inf` is the measured "no cost sinks it".
                let break_even = if point.break_even_cost_bps.is_finite() {
                    format!("{:>8.2}", point.break_even_cost_bps)
                } else if point.break_even_cost_bps.is_nan() {
                    format!("{:>8}", "n/a")
                } else {
                    format!("{:>8}", "unkill")
                };
                // The cost this row is judged against: its OWN measured one when it has one,
                // the arm's constant when the arm IS a constant. A measured arm with no
                // measurable cost on this row is not silently judged against the flat
                // reference - it has no verdict at all.
                let floor = if point.realized_cost_bps.is_finite() {
                    Some(point.realized_cost_bps)
                } else if arm.arm.is_measured() {
                    None
                } else {
                    Some(cost_floor_bps)
                };
                // A verdict needs BOTH sides measured. Either one absent is "n/a", never the
                // false branch of a comparison against NaN.
                let pays = match (point.break_even_cost_bps.is_nan(), floor) {
                    (false, Some(floor)) if point.break_even_cost_bps > floor => "yes",
                    (false, Some(_)) => "no",
                    _ => "n/a",
                };
                out.push_str(&format!(
                    "{:<22}{:>6.2}{:>9.2}{:>8.3}{:>8.3}{:>9.3}{:>9.3}{break_even}{:>8.2}{:>5}\
                     {:>8.2}{:>8.3}{:>8.2}{:>7.3}{:>7.3}{:>7.3}{:>7.4}\n",
                    point.policy,
                    point.band_fraction,
                    point.turnover_per_day,
                    point.rotation_per_instant,
                    point.turnover_share_of_unbanded,
                    point.gross_log_growth_per_year,
                    point.log_growth_per_year,
                    point.realized_cost_bps,
                    pays,
                    point.sharpe,
                    point.payoff_bps_per_instant,
                    point.payoff_sd_bps_per_instant,
                    point.bar_win_rate,
                    point.trade_win_rate,
                    point.position_sign_agreement,
                    point.legs.substituted_leg_share(),
                ));
            }
        }
        out
    }

    /// Every cost arm's frontier, one after the other, with the unpriceable-leg counts.
    ///
    /// The IMPACT_K band is three of these arms and the zero-fee arm is three more. Reading
    /// them together is the point: a conclusion that survives `k = 0.25` and dies at `k = 1.0`
    /// is a conclusion about an assumed coefficient, and one that dies with the fees included
    /// and lives without them is a conclusion about a broker.
    pub fn cost_arm_table(&self) -> String {
        let mut out = String::new();
        for arm in &self.arms {
            out.push_str(&self.frontier_table(arm, f64::from(self.cost_bps)));
            out.push_str(&self.leg_provenance_line(arm));
            out.push('\n');
        }
        out
    }

    /// One line stating how much of an arm's pricing was a cross-sectional stand-in.
    ///
    /// Read off the MODEL policy's unbanded book, which is the one every headline is quoted
    /// from. `subLeg` on each row of the frontier carries the same statistic per row.
    fn leg_provenance_line(&self, arm: &FrontierArm) -> String {
        let point = &arm.points[0][0];
        let legs = point.legs;
        format!(
            "  provenance [{}] model @ band 0: {} legs charged, {} priced at the \
             cross-sectional median spread, {} at the median volatility, {} at the same \
             symbol's other months, {} with no observed volume (charged at a full-ADV clip); \
             {:.4} of traded weight substituted; participation of ADV mean {:.3e} max {:.3e}\n",
            arm.arm.label(),
            legs.legs,
            legs.spread_substituted,
            legs.impact_substituted,
            legs.month_substituted,
            legs.no_liquidity_legs,
            legs.substituted_turnover_share(),
            legs.mean_participation(),
            legs.max_participation,
        )
    }

    /// Edge per name-bar against cost per trade, by the traded panel's own liquidity decile.
    ///
    /// The table that answers "what would have to go away for this to work": the four cost
    /// components are additive and stated separately, and the directional edge is beside them
    /// in the same units, so the shortfall is a subtraction rather than an argument.
    pub fn edge_table(&self) -> String {
        let mut out = format!(
            "edge per name-bar against ONE-WAY cost at {:.1}% of ADV, by liquidity decile of \
             the traded panel (decile 0 is thinnest); a round trip is TWICE the all-in \
             column{}\n",
            100.0 * PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT],
            if self.edge.measured {
                ""
            } else {
                " - NO COST MODEL WAS SUPPLIED, so every cost column is unmeasured"
            },
        );
        out.push_str(
            "  `sgnEdge` and `sgnAgr` are per POSITIONED bar, flat ones included, which is the \
             denominator a\n  cost comparison needs and NOT the one a directional claim needs: \
             both are attenuated by\n  `1 - flat`, and each decile carries its own flat share. \
             `mvEdge` and `mvAgr` strip it.\n",
        );
        out.push_str(&format!(
            "{:<8}{:>5}{:>10}{:>12}{:>9}{:>9}{:>9}{:>9}{:>8}{:>8}{:>8}{:>8}{:>8}{:>9}{:>8}\
             {:>8}{:>8}{:>9}{:>9}\n",
            "decile",
            "syms",
            "nameBars",
            "medADV$",
            "mu bps",
            "sd bps",
            "sgnEdge",
            "se",
            "sgnAgr",
            "flat",
            "mvEdge",
            "mvAgr",
            "med|f|",
            "predMu",
            "halfSpr",
            "comm",
            "reg",
            "impact",
            "allIn",
        ));
        let line = |row: &EdgeVsCost| {
            let label = if row.decile == usize::MAX {
                "pooled".to_owned()
            } else {
                format!("{}", row.decile)
            };
            format!(
                "{label:<8}{:>5}{:>10}{:>12.3e}{:>9.4}{:>9.3}{:>9.4}{:>9.4}{:>8.4}{:>8.4}\
                 {:>8.4}{:>8.4}{:>8.2}{:>9.4}{:>8.3}{:>8.3}{:>8.3}{:>9.3}{:>9.3}\n",
                row.symbols,
                row.name_bars,
                row.median_adv_usd,
                row.mean_r_bps,
                row.sd_r_bps,
                row.signed_edge_bps,
                row.signed_edge_se_bps,
                row.forecast_sign_agreement,
                row.flat_share(),
                row.signed_edge_per_moving_bar_bps(),
                row.sign_agreement_on_moving_bars(),
                row.median_abs_kelly,
                row.mean_forecast_bps,
                row.half_spread_bps,
                row.commission_bps,
                row.regulatory_bps,
                row.impact_bps[IMPACT_K_DEFAULT_SLOT],
                row.all_in_bps[IMPACT_K_DEFAULT_SLOT],
            )
        };
        for row in &self.edge.deciles {
            out.push_str(&line(row));
        }
        out.push_str(&line(&self.edge.pooled));
        out.push_str(&format!(
            "  impact is an ASSUMPTION at k = {IMPACT_K:.2}; the same column at k = {:?} is \
             {:?} bps pooled, and the all-in becomes {:?} bps\n",
            IMPACT_K_GRID,
            self.edge.pooled.impact_bps,
            self.edge.pooled.all_in_bps,
        ));
        out.push_str(&format!(
            "  {} of {} panel symbols priced at the cross-sectional median spread, {} at the \
             median volatility\n",
            self.edge.pooled.spread_substituted,
            self.edge.pooled.symbols,
            self.edge.pooled.impact_substituted,
        ));
        // The ratio a selective strategy would multiply against a cost saving. It must be taken
        // over the MOVING-bar edge: the attenuated one carries each decile's own flat share, so
        // its ratio is inflated by `(1 - flat_deep) / (1 - flat_pooled)` and the inflation does
        // not cancel.
        if let Some(deepest) = self.edge.deciles.last() {
            out.push_str(&format!(
                "  deepest-decile edge over pooled edge: {:.4} per POSITIONED bar (attenuated), \
                 {:.4} per MOVING bar; flat share {:.4} deepest vs {:.4} pooled\n",
                deepest.signed_edge_bps / self.edge.pooled.signed_edge_bps,
                deepest.signed_edge_per_moving_bar_bps()
                    / self.edge.pooled.signed_edge_per_moving_bar_bps(),
                deepest.flat_share(),
                self.edge.pooled.flat_share(),
            ));
        }
        out.push_str(&self.edge.interval_table());
        out
    }

    /// Where net growth crosses zero under every arm, for the model AND for the two baselines
    /// that have no model in them at all.
    ///
    /// The single most retractable claim this module can make is "the frontier crosses zero at
    /// band X", because under a flat cost the equal-weight and marginal-null books cross there
    /// too, which makes the corner buy-and-hold wearing a band rather than model edge. So the
    /// crossing is never reported alone: every arm's row carries the model beside both nulls,
    /// and the verdict column says which of the three it belongs to.
    pub fn crossing_table(&self) -> String {
        let watched = [
            (Policy::Model, "model"),
            (Policy::EqualWeight, "equal-weight"),
            (Policy::Marginal, "marginal-null"),
        ];
        let mut out = String::from(
            "net-growth crossing by cost arm: the first band whose NET log growth per year is \
             positive, for the model and for the two policies that read no model weight\n",
        );
        out.push_str(&format!(
            "{:<34}{:>16}{:>16}{:>16}{:>26}\n",
            "cost arm", "model", "equal-weight", "marginal-null", "verdict",
        ));
        for arm in &self.arms {
            let mut cells = Vec::with_capacity(watched.len());
            for (policy, _) in watched {
                let index = POLICIES
                    .iter()
                    .position(|p| *p == policy)
                    .expect("every watched policy is in POLICIES");
                let crossing = arm.crossing(index).map(|band| {
                    (
                        BAND_FRACTIONS[band],
                        arm.points[index][band].log_growth_per_year,
                    )
                });
                // `crossing` is a `>` over floats and therefore blind to a third state: a band
                // whose book never traded carries NaN, which is not a band that failed to cross.
                // Count them, so "never" cannot absorb "never measured".
                let unmeasured = arm.points[index]
                    .iter()
                    .filter(|p| p.log_growth_per_year.is_nan())
                    .count();
                cells.push((crossing, unmeasured, arm.points[index].len()));
            }
            let show = |cell: &(Option<(f64, f64)>, usize, usize)| match cell {
                (Some((band, growth)), _, _) => {
                    format!("{:>16}", format!("{band:.2} ({growth:+.3})"))
                }
                (None, unmeasured, bands) if unmeasured == bands => format!("{:>16}", "unmeasured"),
                (None, 0, _) => format!("{:>16}", "never"),
                (None, unmeasured, _) => format!("{:>16}", format!("never ({unmeasured} n/a)")),
            };
            // The model's verdict is only a verdict when the model's own row was measured; a
            // wholly unmeasured model row is not "no crossing", it is nothing.
            let verdict = match (cells[0].0, cells[1].0, cells[2].0) {
                (None, _, _) if cells[0].1 == cells[0].2 => "model row UNMEASURED",
                (None, _, _) => "no crossing at all",
                (Some(_), None, None) => "MODEL result",
                (Some((m, _)), Some((e, _)), _) if m < e => "model crosses first",
                (Some((m, _)), _, Some((n, _))) if m < n => "model crosses first",
                _ => "baseline result",
            };
            out.push_str(&format!(
                "{:<34}{}{}{}{:>26}\n",
                arm.arm.label(),
                show(&cells[0]),
                show(&cells[1]),
                show(&cells[2]),
                verdict,
            ));
        }
        out
    }

    /// The three distinct statistics a "win rate" can name, on the model's unbanded book.
    ///
    /// They are three different numbers and only one of them is a directional-skill statistic.
    /// [`super::trade_bench`]'s `hit_rate` field is the THIRD: `trade_bench.rs` counts a bar as
    /// a hit when `fraction * realized > 0` over bars carrying a non-zero position, which is
    /// sign agreement conditional on a position and not the share of profitable bars — for a
    /// single name those coincide, for a portfolio they do not.
    pub fn win_rate_table(&self) -> String {
        let run = &self.runs[DEFAULT_GROSS_SLOT][0];
        let m = &run.metrics;
        let mut out = format!(
            "win rates of the {} book at {DEFAULT_GROSS_CAP:.1}x gross, band 0, under [{}]\n",
            m.policy,
            self.flat_arm().arm.label(),
        );
        out.push_str(&format!(
            "  (a) bars with a positive NET return          {:.4} of {} instants\n",
            m.bar_win_rate, m.instants,
        ));
        out.push_str(&format!(
            "  (b) round-trip lifecycles in profit          {:.4} net, {:.4} gross of cost, \
             over {:.0} lifecycles averaging {:.2} bars held\n",
            m.trade_win_rate, m.trade_win_rate_gross, m.trades, m.mean_hold_bars,
        ));
        out.push_str(&format!(
            "  (c) sign agreement, held weight vs realized  {:.4} of {} positioned name-bars \
             (this is `trade_bench`'s `hit`)\n",
            m.position_sign_agreement, run.trades.positioned_legs,
        ));
        out.push_str(&format!(
            "      sign agreement, FORECAST vs realized     {:.4} over the whole panel, free \
             of sizing, band and gross cap\n",
            self.edge.pooled.forecast_sign_agreement,
        ));
        // A bar with `r == 0` has no sign to agree with and is counted as a disagreement above,
        // so the line a reader compares against `0.5` is attenuated by the flat share. Both
        // denominators, on adjacent lines, or the comparison is wrong by that factor.
        out.push_str(&format!(
            "      the same over MOVING bars only           {:.4}, with {:.4} of positioned \
             name-bars flat (r exactly 0, no sign to agree with)\n",
            self.edge.pooled.sign_agreement_on_moving_bars(),
            self.edge.pooled.flat_share(),
        ));
        out.push_str(&format!(
            "  per-instant book payoff before cost          {:+.4} bps, sd {:.3} bps, ratio \
             {:+.5} per instant ({:+.3} annualized at {:.0} instants/y)\n",
            m.payoff_bps_per_instant,
            m.payoff_sd_bps_per_instant,
            m.payoff_ratio_per_instant,
            m.payoff_ratio_per_instant * self.instants_per_year.sqrt(),
            self.instants_per_year,
        ));
        out.push_str(&format!(
            "  per name-bar realized return                {:+.4} bps mean, {:.3} bps sd; \
             directional edge {:+.4} +- {:.4} bps, Kelly-weighted {:+.4} bps, median |f*| \
             {:.2}\n",
            self.edge.pooled.mean_r_bps,
            self.edge.pooled.sd_r_bps,
            self.edge.pooled.signed_edge_bps,
            self.edge.pooled.signed_edge_se_bps,
            self.edge.pooled.kelly_weighted_edge_bps,
            self.edge.pooled.median_abs_kelly,
        ));
        out.push_str(&format!(
            "  one round trip at the pooled all-in cost     {:.3} bps against a directional \
             edge of {:+.4} bps: the edge is {:.4} round trips\n",
            2.0 * self.edge.pooled.headline_all_in_bps(),
            self.edge.pooled.signed_edge_bps,
            self.edge.pooled.edge_over_round_trip(),
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

pub const PORTFOLIO_EQUITY_BASE: &str = "pretrain_portfolio_equity";
pub const PORTFOLIO_METRICS_BASE: &str = "pretrain_portfolio_metrics";
pub const PORTFOLIO_GROSS_CURVE_BASE: &str = "pretrain_portfolio_gross_curve";
/// The turnover-edge frontier. Break-even is gross edge over turnover, so the ONE lever the
/// arithmetic leaves is trading less; this base is that lever swept, under every cost arm.
pub const PORTFOLIO_FRONTIER_BASE: &str = "pretrain_portfolio_frontier";
/// Edge per name-bar against one-way cost per trade, by the traded panel's liquidity decile.
///
/// A separate base from `pretrain_cost_deciles` and not derivable from it: that one is a
/// property of the whole 5,297-symbol corpus and carries no edge at all, this one is the
/// hundred names a book actually traded with the model's own directional edge on the same
/// x-axis. The comparison the whole strategy question reduces to is a subtraction between two
/// series of this chart.
pub const PORTFOLIO_EDGE_BASE: &str = "pretrain_portfolio_edge_vs_cost";

/// Metric rows of [`PORTFOLIO_METRICS_BASE`], in series order. The x-axis is the policy
/// index, so one column of the chart is one book.
pub const METRIC_ROWS: [&str; 30] = [
    "log growth per year, gross of cost",
    "log growth per year, net",
    "break-even cost (bps)",
    "CAGR",
    "Sharpe",
    "annualized vol",
    "max drawdown",
    "Calmar",
    "mean gross",
    "max gross",
    "mean net",
    "turnover/day",
    "mean breadth",
    "gross bound fraction",
    "first-factor exposure (share of gross)",
    "leverage error (realized / independence vol)",
    "payoff per instant, gross of cost (bps)",
    "payoff sd per instant (bps)",
    "payoff ratio per instant",
    "win rate (a): bars with positive net return",
    "win rate (b): round trips in profit, net",
    "win rate (b): round trips in profit, gross",
    "round-trip lifecycles",
    "mean bars held per lifecycle",
    "win rate (c): sign agreement, held weight vs realized",
    "realized one-way cost paid (bps per dollar traded)",
    "mean participation of ADV",
    "max participation of ADV",
    "substituted-leg share (cross-sectional stand-in)",
    "legs with no observed volume",
];

/// Write the three portfolio charts into a generation directory.
pub fn write_portfolio_bench(dir: &Path, label: &str, bench: &PortfolioBench) -> Result<()> {
    ensure!(
        bench.instants > 0 && !bench.runs.is_empty(),
        "the portfolio bench traded no instants, so there is nothing to write"
    );
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let suffix = format!(
        "{label} - {} instants x {} symbols over {:.3}y, breadth {:.1} (min {}), {:.2} bps",
        bench.instants,
        bench.symbols,
        bench.span_years,
        bench.breadth.mean,
        bench.breadth.min,
        bench.cost_bps,
    );

    // The equity curves themselves, in LOG10 of wealth. Every policy at every cap, because
    // the whole point of the gross axis is that the shape of the curve changes with it.
    //
    // Log10 rather than wealth: the perfect-foresight ceiling multiplies wealth by more than
    // `e^700` over a held-out span and every linear plot of it is either `inf` or a flat line
    // beside a spike. A dead book has no logarithm at all, so it is drawn at
    // [`RUIN_FLOOR_LOG10`] — a stated floor for the PICTURE only; the fact is
    // `PortfolioMetrics::ruined_at_instant`, and the metrics chart carries it.
    let mut equity = Vec::with_capacity(GROSS_CAPS.len() * POLICIES.len());
    for (c, cap) in GROSS_CAPS.iter().enumerate() {
        for (p, policy) in POLICIES.iter().enumerate() {
            let run = &bench.runs[c][p];
            equity.push(ReportSeries {
                label: format!("{} @ {cap:.1}x", policy.name()),
                values: run
                    .log_equity
                    .iter()
                    .map(|w| (w / std::f64::consts::LN_10).max(RUIN_FLOOR_LOG10) as f32)
                    .collect(),
            });
        }
    }
    write_chart(
        dir,
        PORTFOLIO_EQUITY_BASE,
        format!("Portfolio Equity, One Book on One Calendar - {suffix}"),
        "panel instant",
        "log10 wealth, starting from 0.0 (a dead book is floored at -9)",
        ScaleKind::Linear,
        equity,
    )?;

    // The table, transposed into series: x is the policy, one line per metric.
    let axis = POLICIES
        .iter()
        .map(|p| p.name())
        .collect::<Vec<_>>()
        .join(" | ");
    let metrics = (0..POLICIES.len())
        .map(|p| *bench.metrics(DEFAULT_GROSS_SLOT, p))
        .collect::<Vec<_>>();
    let row = |pick: fn(&PortfolioMetrics) -> f64| -> Vec<f32> {
        metrics.iter().map(|m| pick(m) as f32).collect()
    };
    let picks: [fn(&PortfolioMetrics) -> f64; METRIC_ROWS.len()] = [
        |m| m.gross_log_growth_per_year,
        |m| m.log_growth_per_year,
        // An infinite break-even would be dropped by the renderer, which reads as unmeasured
        // rather than as unkillable; clip it to the bound the search itself stops at. A NaN one
        // must NOT borrow that bound - see `charted_break_even_bps`.
        |m| charted_break_even_bps(m.break_even_cost_bps),
        |m| m.cagr,
        |m| m.sharpe,
        |m| m.vol,
        |m| m.max_drawdown,
        |m| m.calmar,
        |m| m.mean_gross,
        |m| m.max_gross,
        |m| m.mean_net,
        |m| m.turnover_per_day,
        |m| m.mean_breadth,
        |m| m.bound_fraction,
        |m| m.mean_first_factor_exposure,
        |m| m.leverage_error,
        |m| m.payoff_bps_per_instant,
        |m| m.payoff_sd_bps_per_instant,
        |m| m.payoff_ratio_per_instant,
        |m| m.bar_win_rate,
        |m| m.trade_win_rate,
        |m| m.trade_win_rate_gross,
        |m| m.trades,
        |m| m.mean_hold_bars,
        |m| m.position_sign_agreement,
        |m| m.realized_cost_bps,
        |m| m.mean_participation_of_adv,
        |m| m.max_participation_of_adv,
        |m| m.substituted_leg_share,
        |m| m.no_liquidity_legs,
    ];
    write_chart(
        dir,
        PORTFOLIO_METRICS_BASE,
        format!("Portfolio Metrics at {DEFAULT_GROSS_CAP:.1}x Gross - {suffix}"),
        format!("policy index: {axis}").as_str(),
        "annualized from the panel's measured span",
        ScaleKind::Symlog,
        METRIC_ROWS
            .iter()
            .zip(picks)
            .map(|(label, pick)| ReportSeries {
                label: (*label).to_owned(),
                values: row(pick),
            })
            .collect(),
    )?;

    // The gross axis. A CAGR that scales with the cap while the bound fraction stays at one
    // was bought with leverage, not with prediction.
    let mut curve = vec![ReportSeries {
        label: "gross cap".to_owned(),
        values: GROSS_CAPS.iter().map(|c| *c as f32).collect(),
    }];
    for (p, policy) in POLICIES.iter().enumerate() {
        for (label, pick) in [
            (
                "net log growth/yr",
                (|m: &PortfolioMetrics| m.log_growth_per_year) as fn(&PortfolioMetrics) -> f64,
            ),
            ("gross log growth/yr", |m: &PortfolioMetrics| {
                m.gross_log_growth_per_year
            }),
            ("break-even bps", |m: &PortfolioMetrics| {
                charted_break_even_bps(m.break_even_cost_bps)
            }),
            ("Sharpe", |m: &PortfolioMetrics| m.sharpe),
            ("max drawdown", |m: &PortfolioMetrics| m.max_drawdown),
        ] {
            curve.push(ReportSeries {
                label: format!("{} {label}", policy.name()),
                values: (0..GROSS_CAPS.len())
                    .map(|c| pick(bench.metrics(c, p)) as f32)
                    .collect(),
            });
        }
    }
    write_chart(
        dir,
        PORTFOLIO_GROSS_CURVE_BASE,
        format!("Portfolio Verdict vs the Gross Constraint - {suffix}"),
        "gross cap index (see the `gross cap` series)",
        "annualized",
        ScaleKind::Symlog,
        curve,
    )?;

    // The turnover axis, under EVERY cost arm. Break-even rises as the band suppresses trading
    // and the gross edge falls as the band suppresses the signal; whether they cross above the
    // real cost of trading is the only remaining question about this strategy, and it is this
    // chart. The flat arm's five series keep their original labels so the curves quoted before
    // the measured model existed stay findable; every measured arm's series names its own arm,
    // including the impact coefficient it ASSUMED.
    let mut frontier = vec![ReportSeries {
        label: "band (typical positions)".to_owned(),
        values: BAND_FRACTIONS.iter().map(|b| *b as f32).collect(),
    }];
    for (a, arm) in bench.arms.iter().enumerate() {
        for row in &arm.points {
            let policy = row.first().map_or("", |p| p.policy);
            for (label, pick) in [
                (
                    "turnover/day",
                    (|p: &FrontierPoint| p.turnover_per_day) as fn(&FrontierPoint) -> f64,
                ),
                ("gross log growth/yr", |p: &FrontierPoint| {
                    p.gross_log_growth_per_year
                }),
                ("net log growth/yr", |p: &FrontierPoint| {
                    p.log_growth_per_year
                }),
                ("break-even bps", |p: &FrontierPoint| {
                    charted_break_even_bps(p.break_even_cost_bps)
                }),
                ("Sharpe", |p: &FrontierPoint| p.sharpe),
                ("one-way cost paid bps", |p: &FrontierPoint| {
                    p.realized_cost_bps
                }),
                ("substituted-leg share", |p: &FrontierPoint| {
                    p.legs.substituted_leg_share()
                }),
            ] {
                // The flat arm carries neither a paid-cost nor a substitution curve worth
                // drawing: the first is its own constant and the second is zero by definition.
                if a == 0 && matches!(label, "one-way cost paid bps" | "substituted-leg share") {
                    continue;
                }
                frontier.push(ReportSeries {
                    label: if a == 0 {
                        format!("{policy} {label}")
                    } else {
                        format!("{policy} {label} [{}]", arm.arm.label())
                    },
                    values: row.iter().map(|p| pick(p) as f32).collect(),
                });
            }
        }
    }
    write_chart(
        dir,
        PORTFOLIO_FRONTIER_BASE,
        format!("Turnover-Edge Frontier at {DEFAULT_GROSS_CAP:.1}x Gross - {suffix}"),
        "no-trade band index (see the `band` series)",
        "annualized; break-even and cost in bps",
        ScaleKind::Symlog,
        frontier,
    )?;

    // Edge against cost on ONE axis. Both halves are per name-bar or per one-way trade in bps,
    // so the vertical distance between the `directional edge` series and the `all-in` series IS
    // the shortfall, decile by decile, with no arithmetic left to the reader.
    let deciles = &bench.edge.deciles;
    let mut edge = vec![ReportSeries {
        label: "median dollar ADV of the decile".to_owned(),
        values: deciles.iter().map(|d| d.median_adv_usd as f32).collect(),
    }];
    let mut push = |label: String, pick: &dyn Fn(&EdgeVsCost) -> f64| {
        edge.push(ReportSeries {
            label,
            values: deciles.iter().map(|d| pick(d) as f32).collect(),
        });
    };
    push("mean realized return (bps/name-bar)".to_owned(), &|d| {
        d.mean_r_bps
    });
    push("realized sd (bps/name-bar)".to_owned(), &|d| d.sd_r_bps);
    push("directional edge (bps/name-bar)".to_owned(), &|d| {
        d.signed_edge_bps
    });
    push("directional edge standard error (bps)".to_owned(), &|d| {
        d.signed_edge_se_bps
    });
    push("Kelly-weighted edge (bps/name-bar)".to_owned(), &|d| {
        d.kelly_weighted_edge_bps
    });
    push("forecast sign agreement".to_owned(), &|d| {
        d.forecast_sign_agreement
    });
    // The same two statistics with the flat bars out of the denominator, plus the share itself,
    // because the attenuation is per decile and a reader comparing deciles on the attenuated
    // series is comparing differently-scaled quantities. See `EdgeVsCost::flat_share`.
    push("flat share of positioned bars".to_owned(), &|d| {
        d.flat_share()
    });
    push("signed edge per MOVING bar (bps)".to_owned(), &|d| {
        d.signed_edge_per_moving_bar_bps()
    });
    push("forecast sign agreement, MOVING bars".to_owned(), &|d| {
        d.sign_agreement_on_moving_bars()
    });
    push("median |f*|".to_owned(), &|d| d.median_abs_kelly);
    push("predicted mean return (bps/name-bar)".to_owned(), &|d| {
        d.mean_forecast_bps
    });
    push("half-spread (bps, one way)".to_owned(), &|d| {
        d.half_spread_bps
    });
    push("commission (bps, one way)".to_owned(), &|d| d.commission_bps);
    push("regulatory fee (bps, one way)".to_owned(), &|d| {
        d.regulatory_bps
    });
    push(
        "measured cost, impact-free (bps, one way)".to_owned(),
        &|d| d.impact_free_bps(),
    );
    for (slot, k) in IMPACT_K_GRID.iter().enumerate() {
        push(
            format!(
                "impact at {:.0}% of ADV, k={k:.2} ASSUMED (bps, one way)",
                100.0 * PARTICIPATION_GRID[PARTICIPATION_HEADLINE_SLOT]
            ),
            &|d| d.impact_bps[slot],
        );
        push(
            format!("all-in cost, k={k:.2} ASSUMED (bps, one way)"),
            &|d| d.all_in_bps[slot],
        );
    }
    push("symbols priced at the median spread".to_owned(), &|d| {
        d.spread_substituted as f64
    });
    push("symbols priced at the median volatility".to_owned(), &|d| {
        d.impact_substituted as f64
    });
    push("name-bars".to_owned(), &|d| d.name_bars as f64);
    // The interval on the edge half, day-blocked, so a reader of the chart can see whether a
    // decile's round-trip coverage is resolvably below one rather than merely below it. The cost
    // series above carry no interval by construction: they are medians and means over
    // symbol-months, not time-series averages, and this bootstrap does not resample them.
    let by_day = &bench.edge.intervals.by_day;
    for (label, cis) in [
        (
            "round trips the edge pays for, all-in",
            &by_day.edge_over_round_trip,
        ),
        (
            "round trips the edge pays for, zero impact",
            &by_day.impact_free_over_round_trip,
        ),
    ] {
        edge.push(ReportSeries {
            label: label.to_owned(),
            values: cis.iter().map(|ci| ci.point as f32).collect(),
        });
        edge.push(ReportSeries {
            label: format!("{label}, 95% lo ({} blocked)", by_day.blocking),
            values: cis.iter().map(|ci| ci.lo as f32).collect(),
        });
        edge.push(ReportSeries {
            label: format!("{label}, 95% hi ({} blocked)", by_day.blocking),
            values: cis.iter().map(|ci| ci.hi as f32).collect(),
        });
    }
    write_chart(
        dir,
        PORTFOLIO_EDGE_BASE,
        format!("Edge per Name-Bar vs One-Way Cost, by Traded-Panel Liquidity Decile - {suffix}"),
        "liquidity decile of the traded panel (0 is thinnest; see the median ADV series)",
        "bps per name-bar (edge) and bps per one-way trade (cost)",
        ScaleKind::Symlog,
        edge,
    )
}

fn write_chart(
    dir: &Path,
    base: &str,
    title: String,
    x_label: &str,
    y_label: &str,
    scale: ScaleKind,
    series: Vec<ReportSeries>,
) -> Result<()> {
    ensure!(!series.is_empty(), "{base} would be an empty chart");
    let path = dir.join(format!("{base}.report.bin"));
    write_report(
        &path,
        &Report {
            title,
            x_label: Some(x_label.to_owned()),
            y_label: Some(y_label.to_owned()),
            scale,
            kind: ReportKind::MultiLine { series },
        },
    )
    .with_context(|| format!("writing {}", path.display()))?;
    // Reading it straight back is what turns "the writer ran" into "the chart exists": a
    // truncated or non-finite series renders as a blank panel and nothing else notices.
    let report = read_report(&path).with_context(|| format!("reading back {}", path.display()))?;
    match report.kind {
        ReportKind::MultiLine { series } => ensure!(
            series
                .iter()
                .any(|s| s.values.iter().any(|v| v.is_finite())),
            "{base} holds no finite value"
        ),
        other => bail!("{base} came back as {other:?}"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PortfolioArgs {
    /// Directory of `<SYM>.<res>.bars` files.
    pub bars_dir: PathBuf,
    /// The checkpoint to trade. Its metadata and supports sidecars are resolved beside it.
    pub checkpoint: PathBuf,
    /// Generation directory the three charts land in.
    pub gens_dir: PathBuf,
    pub res_secs: u32,
    pub device: Device,
    /// The PINNED global split, so the panel is held out by construction.
    pub split_bounds: (i64, i64),
    pub max_symbols: usize,
    pub max_instants: usize,
    /// One-way cost of the FLAT reference arm, in bps. Kept beside the measured model rather
    /// than replaced by it: every number quoted before the measurement existed was computed
    /// under this constant, and a comparison needs both sides.
    pub cost_bps: f32,
    /// Threads the per-symbol cost calibration is measured on, or `0` to skip the measurement
    /// entirely and report the flat arm alone.
    ///
    /// The calibration walks every bar of every series in the corpus, which is minutes of CPU
    /// on the real corpus, so it is a stated cost rather than a hidden one. Skipping it is
    /// honest and visible: [`PortfolioBench::arms`] then holds one arm and
    /// [`EdgeVsCostTable::measured`] is `false`, so no cost column can be read as measured.
    pub cost_threads: usize,
    pub capital_usd: f64,
    pub label: String,
}

/// Build the panel, forecast it, run every policy at every cap and write the charts.
pub fn run_portfolio_backtest(args: &PortfolioArgs) -> Result<PortfolioBench> {
    // The world model asserts a bf16 autocast on CUDA, and it is right to: a portfolio
    // measured under a different numeric regime than the one the weights were trained and
    // validated under is measuring something else. Harmless on CPU.
    crate::torch::cuda::cfg::configure_cuda();
    let (val_start, val_end) = args.split_bounds;
    let config = PanelConfig::new((val_start, val_end), args.max_symbols, args.max_instants);
    // The corpus is opened with the PINNED bounds so its own split derivation can never
    // move under a growing corpus, and `min_bars` admits exactly the symbols long enough to
    // carry a belief plus the trailing ADV window the cost model prices against.
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
    let model = BarWorldModel::load(&args.checkpoint, &metadata, args.device).with_context(|| {
        format!(
            "loading the traded checkpoint {}",
            args.checkpoint.display()
        )
    })?;
    let supports = model
        .supports_for(args.res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {}s", args.res_secs))?;

    let marginal = marginal_forecasts(&panel, supports);
    let model_forecast = model_forecasts(&model, &corpus, &panel, args.res_secs)?;
    let inputs = PolicyInputs {
        model: &model_forecast,
        marginal: &marginal,
    };
    // The measured per-symbol cost model, calibrated on THIS corpus so `Panel::series_of` is
    // the right translation, and handed to the bench as the concrete type: the arms it builds
    // wrap it in `PanelCost` per impact coefficient and per fee arm.
    let calibration = if args.cost_threads > 0 {
        Some(Arc::new(
            CostCalibration::from_corpus(&corpus, args.cost_threads)
                .context("measuring the per-symbol cost calibration")?,
        ))
    } else {
        None
    };
    let measured = calibration.as_ref().map(|c| BarCostModel::new(Arc::clone(c)));
    if let Some(measured) = measured.as_ref() {
        let calibration = measured.calibration();
        ensure!(
            calibration.len() == corpus.series_count(),
            "the calibration measured {} series against the corpus's {}, so the panel's \
             `series_of` translation would price the wrong symbols",
            calibration.len(),
            corpus.series_count()
        );
    }
    let cost = FlatCost::new(args.cost_bps);
    let bench = PortfolioBench::run(
        &panel,
        &inputs,
        &cost,
        args.cost_bps,
        measured.as_ref(),
        &BacktestConfig {
            capital_usd: args.capital_usd,
            ..BacktestConfig::default()
        },
    )?;
    write_portfolio_bench(&args.gens_dir, &args.label, &bench)?;
    Ok(bench)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::{BarDof, BAR_VOLUME_EMA_SPAN};
    use crate::torch::dataset::mix64;
    use shared::bars::{write_bar_file, PackedBar};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SCRATCH: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "portfolio_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn uniform(seed: u64, index: u64) -> f64 {
        (mix64(seed, index) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A panel with the returns and volumes stated outright, so an expected equity curve can
    /// be written down by hand.
    fn fixture_panel(rows: &[(i64, Vec<(u32, f32)>)], names: usize) -> Panel {
        let symbols = (0..names).map(|i| format!("S{i}")).collect();
        let mut slices = Vec::with_capacity(rows.len());
        let mut volumes = Vec::with_capacity(rows.len());
        for (ts_ms, entries) in rows {
            slices.push(PanelSlice {
                ts_ms: *ts_ms,
                symbols: entries.iter().map(|(id, _)| *id).collect(),
                realized_r: entries.iter().map(|(_, r)| *r).collect(),
            });
            volumes.push(vec![1.0e9f32; entries.len()]);
        }
        Panel::from_parts(symbols, slices, volumes).expect("fixture panel")
    }

    fn constant_forecast(panel: &Panel, kelly: &[f32]) -> Vec<PanelForecast> {
        panel
            .slices()
            .iter()
            .map(|slice| PanelForecast {
                kelly_f: slice
                    .symbols
                    .iter()
                    .map(|id| kelly[*id as usize])
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect()
    }

    const FIVE_MIN: i64 = 300_000;

    // -----------------------------------------------------------------------
    // The equity curve
    // -----------------------------------------------------------------------

    /// Known returns and known weights against an equity curve computed by hand.
    ///
    /// Two names, uncapped Kelly `3` and `1`, gross cap `1.0`: the projection is exactly
    /// `0.75 / 0.25` at every instant, and with a zero cost model the multiplier of instant
    /// `t` is `1 + 0.75 * (e^{r0} - 1) + 0.25 * (e^{r1} - 1)` with nothing else in it.
    #[test]
    fn a_known_panel_reproduces_an_analytic_equity_curve() {
        let log_returns = [[0.01f32, -0.02], [-0.005, 0.03], [0.002, 0.004]];
        let rows: Vec<(i64, Vec<(u32, f32)>)> = log_returns
            .iter()
            .enumerate()
            .map(|(t, r)| (t as i64 * FIVE_MIN, vec![(0u32, r[0]), (1u32, r[1])]))
            .collect();
        let panel = fixture_panel(&rows, 2);
        let model = constant_forecast(&panel, &[3.0, 1.0]);
        let marginal = constant_forecast(&panel, &[1.0, 1.0]);
        let run = backtest(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &marginal,
            },
            Policy::Model,
            1.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("the fixture backtests");

        let mut wealth = 1.0f64;
        for (t, r) in log_returns.iter().enumerate() {
            let payoff = 0.75 * f64::from(r[0]).exp_m1() + 0.25 * f64::from(r[1]).exp_m1();
            wealth *= 1.0 + payoff;
            assert!(
                (run.equity[t + 1] - wealth).abs() < 1e-12,
                "instant {t}: engine {} vs analytic {wealth}",
                run.equity[t + 1]
            );
            assert!((run.returns[t] - payoff).abs() < 1e-12);
            assert!((run.gross[t] - 1.0).abs() < 1e-12, "the cap must bind exactly");
            assert!(run.bound[t], "raw gross 4.0 against a budget of 1.0 binds");
        }
        assert!(run.ruined_at.is_none());
        assert!((run.metrics.final_wealth - wealth).abs() < 1e-12);
        assert_eq!(run.metrics.turnover_per_day, 1.0, "one entry, no rebalance");
    }

    /// Costs are charged on realized traded notional, at the stated rate, and only on the
    /// weight that actually moved.
    #[test]
    fn cost_is_charged_on_the_weight_that_moved() {
        let rows = vec![
            (0, vec![(0u32, 0.0f32)]),
            (FIVE_MIN, vec![(0u32, 0.0f32)]),
        ];
        let panel = fixture_panel(&rows, 1);
        let model = constant_forecast(&panel, &[1.0]);
        let run = backtest(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &model,
            },
            Policy::Model,
            2.0,
            &FlatCost::new(10.0),
            &BacktestConfig::default(),
        )
        .expect("backtest");
        // Raw gross 1.0 is inside a budget of 2.0, so the weight is exactly 1.0: the first
        // instant pays 10 bps on a full unit, the second pays nothing at all.
        assert!((run.cost[0] - 1.0e-3).abs() < 1e-15, "{}", run.cost[0]);
        assert_eq!(run.cost[1], 0.0);
        assert_eq!(run.turnover[1], 0.0);
        assert!(!run.bound[0], "a raw gross below the budget must not bind");
    }

    /// A book that reaches zero is dead, and every number downstream says so.
    #[test]
    fn a_book_driven_to_zero_stays_dead() {
        // -60% on a 2x gross long is a 120% loss: the multiplier is negative, not merely
        // small, which is the case an averaging bench silently keeps trading through.
        let rows = vec![
            (0, vec![(0u32, 0.01f32)]),
            (FIVE_MIN, vec![(0u32, (0.4f32).ln())]),
            (2 * FIVE_MIN, vec![(0u32, 1.0f32)]),
            (3 * FIVE_MIN, vec![(0u32, 1.0f32)]),
        ];
        let panel = fixture_panel(&rows, 1);
        let model = constant_forecast(&panel, &[8.0]);
        let run = backtest(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &model,
            },
            Policy::Model,
            2.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("backtest");

        assert_eq!(run.ruined_at, Some(1));
        assert_eq!(run.returns[1], -1.0);
        for t in 1..rows.len() {
            assert_eq!(run.equity[t + 1], 0.0, "a dead book stays dead at {t}");
            assert_eq!(
                run.log_equity[t + 1],
                f64::NEG_INFINITY,
                "a dead book has no logarithm at {t}"
            );
        }
        // The two enormous up moves after the ruin must not resurrect anything.
        assert_eq!(run.returns[2], 0.0);
        assert_eq!(run.returns[3], 0.0);
        assert_eq!(run.gross[3], 0.0);
        assert_eq!(run.metrics.final_wealth, 0.0);
        assert_eq!(run.metrics.cagr, -1.0, "total loss is -100% a year, not NaN");
        assert!((run.metrics.max_drawdown - 1.0).abs() < 1e-12);
        assert_eq!(run.metrics.ruined_at_instant, 1.0);
    }

    /// A book that compounds past `f64` still reports a NUMBER.
    ///
    /// The perfect-foresight ceiling at 4x gross over a five-month held-out span multiplies
    /// wealth by more than `e^700`, which is exactly where a linear accumulator silently
    /// becomes `inf` and every metric derived from it becomes garbage. The log curve is the
    /// primitive, so the growth RATE stays finite and only the human-readable CAGR overflows.
    #[test]
    fn a_ceiling_that_overflows_wealth_still_reports_a_growth_rate() {
        // 5% a bar for 20,000 bars is `e^976`, comfortably past `f64::MAX`.
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..20_000)
            .map(|t| (t as i64 * FIVE_MIN, vec![(0u32, 0.05f32)]))
            .collect();
        let panel = fixture_panel(&rows, 1);
        let model = constant_forecast(&panel, &[1.0]);
        let run = backtest(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &model,
            },
            Policy::Model,
            1.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("backtest");

        assert_eq!(run.metrics.final_wealth, f64::INFINITY, "wealth overflows");
        assert_eq!(run.metrics.cagr, f64::INFINITY, "and so does its CAGR");
        // The rate does not: 20,000 instants of `ln(1 + expm1(0.05))` nats over the panel's
        // own measured span.
        let expected = 20_000.0 * f64::from(0.05f32) / panel.span_years();
        assert!(
            run.metrics.log_growth_per_year.is_finite()
                && (run.metrics.log_growth_per_year / expected - 1.0).abs() < 1e-9,
            "log growth {} against {expected}",
            run.metrics.log_growth_per_year
        );
        // Monotone up, so no drawdown at all, and a constant return series has essentially
        // no volatility to annualize.
        assert_eq!(run.metrics.max_drawdown, 0.0);
        assert!(run.metrics.vol < 1e-9, "vol {}", run.metrics.vol);
    }

    // -----------------------------------------------------------------------
    // The constraint
    // -----------------------------------------------------------------------

    /// The gross constraint holds at every instant, for every policy, at every cap — on a
    /// panel whose raw Kelly vectors are wild in both sign and magnitude.
    #[test]
    fn the_gross_constraint_is_never_violated() {
        let names = 7usize;
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..40)
            .map(|t| {
                let present: Vec<(u32, f32)> = (0..names as u32)
                    .filter(|id| uniform(0xA11CE, u64::from(*id) * 97 + t) > 0.25)
                    .map(|id| {
                        let r = 0.02 * (2.0 * uniform(0xBEEF, u64::from(id) * 31 + t) - 1.0);
                        (id, r as f32)
                    })
                    .collect();
                (t as i64 * FIVE_MIN, present)
            })
            .filter(|(_, present)| !present.is_empty())
            .collect();
        let panel = fixture_panel(&rows, names);
        let model: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(t, slice)| PanelForecast {
                kelly_f: slice
                    .symbols
                    .iter()
                    .map(|id| {
                        (40.0 * (uniform(0xF00D, u64::from(*id) * 13 + t as u64) - 0.5)) as f32
                    })
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let marginal = marginal_like(&panel, 2.0);
        let inputs = PolicyInputs {
            model: &model,
            marginal: &marginal,
        };

        for cap in GROSS_CAPS {
            for policy in POLICIES {
                let run = backtest(
                    &panel,
                    &inputs,
                    policy,
                    cap,
                    &FlatCost::new(DEFAULT_COST_BPS),
                    &BacktestConfig::default(),
                )
                .unwrap_or_else(|e| panic!("{} at {cap}x: {e}", policy.name()));
                let budget = policy.gross_budget(cap);
                for (t, gross) in run.gross.iter().enumerate() {
                    assert!(
                        *gross <= budget * (1.0 + 1e-9) + 1e-12,
                        "{} used gross {gross} against {budget} at instant {t}",
                        policy.name()
                    );
                }
                assert!(
                    run.gross.iter().copied().fold(0.0, f64::max) > 0.0,
                    "{} never took a position at {cap}x",
                    policy.name()
                );
            }
        }
    }

    fn marginal_like(panel: &Panel, kelly: f32) -> Vec<PanelForecast> {
        panel
            .slices()
            .iter()
            .map(|slice| PanelForecast {
                kelly_f: vec![kelly; slice.symbols.len()],
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect()
    }

    /// The market-neutral variant carries no net exposure, at every instant.
    #[test]
    fn the_market_neutral_book_carries_no_net_exposure() {
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..12)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..5u32)
                        .map(|id| (id, (0.01 * (uniform(7, u64::from(id) + t) - 0.5)) as f32))
                        .collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, 5);
        let model: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(t, slice)| PanelForecast {
                kelly_f: slice
                    .symbols
                    .iter()
                    .map(|id| (10.0 * uniform(99, u64::from(*id) * 7 + t as u64)) as f32)
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let run = backtest(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &model,
            },
            Policy::MarketNeutral,
            2.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("backtest");
        for (t, net) in run.net.iter().enumerate() {
            assert!(net.abs() < 1e-12, "net exposure {net} at instant {t}");
        }
        assert!(run.metrics.max_abs_net < 1e-12);
        // All long by construction, so the long-only book is emphatically not neutral.
        let plain = backtest(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &model,
            },
            Policy::Model,
            2.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("backtest");
        assert!(plain.metrics.mean_net > 1.0);
    }

    /// Perfect foresight under the SAME gross constraint is an upper bound on the payoff of
    /// every other policy, instant by instant. That is what makes it the ceiling.
    #[test]
    fn the_oracle_is_the_ceiling_under_the_same_gross_cap() {
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..25)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..6u32)
                        .filter(|id| uniform(3, u64::from(*id) * 5 + t) > 0.2)
                        .map(|id| (id, (0.03 * (uniform(5, u64::from(id) * 11 + t) - 0.5)) as f32))
                        .collect::<Vec<_>>(),
                )
            })
            .filter(|(_, p)| !p.is_empty())
            .collect();
        let panel = fixture_panel(&rows, 6);
        let model: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(t, slice)| PanelForecast {
                kelly_f: slice
                    .symbols
                    .iter()
                    .map(|id| (12.0 * (uniform(11, u64::from(*id) + t as u64) - 0.5)) as f32)
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let marginal = marginal_like(&panel, 2.0);
        let inputs = PolicyInputs {
            model: &model,
            marginal: &marginal,
        };
        let free = FlatCost::new(0.0);
        let oracle = backtest(&panel, &inputs, Policy::Oracle, 2.0, &free, &BacktestConfig::default())
            .expect("oracle");
        for policy in POLICIES {
            let run = backtest(&panel, &inputs, policy, 2.0, &free, &BacktestConfig::default())
                .expect("policy");
            for t in 0..panel.instants() {
                assert!(
                    oracle.returns[t] >= run.returns[t] - 1e-12,
                    "{} earned {} at instant {t}, above the ceiling {}",
                    policy.name(),
                    run.returns[t],
                    oracle.returns[t]
                );
            }
        }
        assert!(oracle.metrics.cagr > 0.0, "the ceiling must be profitable");
    }

    // -----------------------------------------------------------------------
    // Absence
    // -----------------------------------------------------------------------

    /// A symbol absent from an instant contributes exactly nothing to that instant's payoff,
    /// and its position is unwound rather than carried at a made-up price.
    #[test]
    fn absence_contributes_no_payoff_and_unwinds_the_position() {
        // Symbol 1 prints at instants 0 and 2 but not at 1. Whatever it does between them is
        // unobserved, so it must not reach the book.
        let rows = vec![
            (0i64, vec![(0u32, 0.0f32), (1u32, 0.0f32)]),
            (FIVE_MIN, vec![(0u32, 0.01f32)]),
            (2 * FIVE_MIN, vec![(0u32, 0.0f32), (1u32, 0.5f32)]),
        ];
        let panel = fixture_panel(&rows, 2);
        let model = constant_forecast(&panel, &[1.0, 1.0]);
        let inputs = PolicyInputs {
            model: &model,
            marginal: &model,
        };
        let run = backtest(
            &panel,
            &inputs,
            Policy::Model,
            2.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("backtest");

        // Instant 1 holds symbol 0 alone at weight 1.0 and earns exactly its return.
        assert!((run.gross[1] - 1.0).abs() < 1e-12);
        // Through f32: the panel stores returns at the corpus's own precision.
        assert!((run.returns[1] - f64::from(0.01f32).exp_m1()).abs() < 1e-12);
        // The unwind of symbol 1 (weight 1.0 -> 0.0) is real turnover, on top of nothing
        // moving in symbol 0.
        assert!((run.turnover[1] - 1.0).abs() < 1e-12);
        // And symbol 1's absence at instant 1 did not let its +50% at instant 2 leak
        // backwards. Both names carry Kelly 1.0, raw gross 2.0 sits exactly on the budget,
        // so each holds a full unit and the payoff is symbol 1's move alone.
        assert!((run.returns[2] - f64::from(0.5f32).exp_m1()).abs() < 1e-12);

        // The same panel with symbol 1 deleted from instant 0 as well must leave instant 1
        // byte-identical: absence carries no state.
        let stripped = vec![
            (0i64, vec![(0u32, 0.0f32)]),
            (FIVE_MIN, vec![(0u32, 0.01f32)]),
            (2 * FIVE_MIN, vec![(0u32, 0.0f32), (1u32, 0.5f32)]),
        ];
        let other = fixture_panel(&stripped, 2);
        let other_model = constant_forecast(&other, &[1.0, 1.0]);
        let alt = backtest(
            &other,
            &PolicyInputs {
                model: &other_model,
                marginal: &other_model,
            },
            Policy::Model,
            2.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("backtest");
        assert!((alt.returns[1] - run.returns[1]).abs() < 1e-12);
    }

    /// The panel builder itself refuses a bar whose predecessor instant is missing, so the
    /// no-forward-fill rule is established where the data is read, not patched downstream.
    #[test]
    fn the_panel_never_forward_fills_a_missing_instant() {
        let dir = scratch_dir("gap");
        let res = 300u32;
        let history = 40usize;
        let start = 10_000_000_000_000i64;
        // A dense symbol and a symbol with a hole punched in the middle of the panel span.
        let dense = synthetic_bars(history + 6, start - history as i64 * FIVE_MIN, 1.0, 11);
        let mut holed = synthetic_bars(history + 6, start - history as i64 * FIVE_MIN, 2.0, 12);
        let hole_ts = start + 2 * FIVE_MIN;
        holed.retain(|b| b.ts() != hole_ts);
        write_bar_file(&dir.join(format!("AAA.{res}.bars")), "AAA", res, &dense).expect("write");
        write_bar_file(&dir.join(format!("BBB.{res}.bars")), "BBB", res, &holed).expect("write");

        let corpus = BarCorpus::load(&dir, res, 1).expect("corpus");
        let config = PanelConfig {
            start_ts_ms: start,
            end_ts_ms: start + 6 * FIVE_MIN,
            max_symbols: 8,
            min_history: 3,
            max_instants: 16,
        };
        let panel = Panel::build(&corpus, &config).expect("panel");
        let holed_id = panel
            .symbols()
            .iter()
            .position(|s| s == "BBB")
            .expect("the holed symbol is in the table") as u32;

        let at = |ts: i64| {
            panel
                .slices()
                .iter()
                .find(|s| s.ts_ms == ts)
                .unwrap_or_else(|| panic!("instant {ts} is in the panel"))
        };
        assert!(
            !at(hole_ts).symbols.contains(&holed_id),
            "a symbol with no bar at an instant cannot be tradeable at it"
        );
        assert!(
            !at(hole_ts + FIVE_MIN).symbols.contains(&holed_id),
            "the instant AFTER a hole has no predecessor close, so it is not tradeable \
             either; admitting it would be a close-to-close return across a gap the book \
             was not positioned over"
        );
        assert!(
            at(hole_ts + 2 * FIVE_MIN).symbols.contains(&holed_id),
            "once both closes exist again the symbol is tradeable"
        );
        // Breadth reports the hole rather than hiding it.
        assert_eq!(panel.breadth().min, 1);
        assert!(panel.breadth().mean < 2.0 && panel.breadth().mean > 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Panel-level bookkeeping: the span, the clock and the ADV all come from the data.
    #[test]
    fn the_panel_measures_its_own_calendar() {
        let dir = scratch_dir("calendar");
        let res = 300u32;
        let history = 40usize;
        let start = 10_000_000_000_000i64;
        let bars = synthetic_bars(history + 5, start - history as i64 * FIVE_MIN, 1.0, 21);
        write_bar_file(&dir.join(format!("AAA.{res}.bars")), "AAA", res, &bars).expect("write");
        let corpus = BarCorpus::load(&dir, res, 1).expect("corpus");
        let panel = Panel::build(
            &corpus,
            &PanelConfig {
                start_ts_ms: start,
                end_ts_ms: start + 5 * FIVE_MIN,
                max_symbols: 4,
                min_history: 3,
                max_instants: 16,
            },
        )
        .expect("panel");

        // Five bars land in the span; the first is the reference instant that fixes the
        // predecessor close, so four are tradeable and the span is measured across those.
        assert_eq!(panel.instants(), 4);
        assert_eq!(panel.span_ms(), 3 * FIVE_MIN);
        let expected_years = (3 * FIVE_MIN) as f64 / MS_PER_YEAR;
        assert!((panel.span_years() - expected_years).abs() < 1e-15);
        assert!((panel.instants_per_year() - 4.0 / expected_years).abs() < 1e-6);
        assert_eq!(panel.trading_days(), 1, "the fixture spans one UTC date");
        // ADV is the trailing per-bar dollar volume scaled by the measured instants/day,
        // never a hardcoded bar count.
        let slice0 = &panel.slices()[0];
        assert!(!slice0.symbols.is_empty());
        let adv = panel.adv_usd(0, 0);
        assert!(adv > 0.0 && adv.is_finite(), "adv {adv}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Annualization divides by the panel's measured span. Two panels with the same returns
    /// and different clocks must annualize differently, which a `93 * 252` constant cannot
    /// express.
    #[test]
    fn annualization_follows_the_measured_span_not_a_constant() {
        let make = |stride: i64| {
            let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..20)
                .map(|t| (t * stride, vec![(0u32, 0.001f32)]))
                .collect();
            let panel = fixture_panel(&rows, 1);
            let model = constant_forecast(&panel, &[1.0]);
            let run = backtest(
                &panel,
                &PolicyInputs {
                    model: &model,
                    marginal: &model,
                },
                Policy::Model,
                1.0,
                &FlatCost::new(0.0),
                &BacktestConfig::default(),
            )
            .expect("backtest");
            (panel.span_years(), run.metrics)
        };
        let (fast_years, fast) = make(FIVE_MIN);
        let (slow_years, slow) = make(60 * FIVE_MIN);

        assert!((slow_years / fast_years - 60.0).abs() < 1e-9);
        // Identical wealth, sixty times the wall clock: the slower panel's CAGR must be the
        // 60th root of the faster one's growth factor.
        assert!((fast.final_wealth - slow.final_wealth).abs() < 1e-15);
        let fast_growth: f64 = 1.0 + fast.cagr;
        let slow_growth: f64 = 1.0 + slow.cagr;
        assert!(
            (fast_growth.powf(1.0 / 60.0) - slow_growth).abs() / slow_growth < 1e-9,
            "{fast_growth} vs {slow_growth}"
        );
        // And the Sharpe annualizer is the measured instants per year, not 23,436.
        assert!((fast.instants_per_year / slow.instants_per_year - 60.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // The traded law
    // -----------------------------------------------------------------------

    /// The marginal null is the bench's own `kelly_fraction` of the fitted unconditional law
    /// of `r`, broadcast. If the two ever diverge the "null" would be a second, unrelated
    /// policy and every edge measured against it would be meaningless.
    #[test]
    fn the_marginal_null_is_the_benchs_own_unconditional_solve() {
        let supports = synthetic_supports(4096, 0xC0FFEE);
        let rows = vec![
            (0i64, vec![(0u32, 0.0f32), (1u32, 0.0f32)]),
            (FIVE_MIN, vec![(0u32, 0.0f32)]),
        ];
        let panel = fixture_panel(&rows, 2);
        let forecasts = marginal_forecasts(&panel, &supports);
        let expected =
            kelly_fraction(supports.bin_masses(DOF_R), &bin_returns(&supports), FREE_LEVERAGE)
                as f32;
        assert_eq!(forecasts.len(), 2);
        assert_eq!(forecasts[0].kelly_f, vec![expected; 2]);
        assert_eq!(forecasts[1].kelly_f, vec![expected; 1]);
        assert!(
            forecasts[0].var_r[0] > 0.0,
            "the unconditional law of a real support has variance"
        );
    }

    fn synthetic_supports(count: usize, seed: u64) -> BarSupports {
        let samples: Vec<BarDof> = (0..count)
            .map(|i| {
                let i = i as u64;
                let u1 = uniform(seed, 3 * i).max(1e-12);
                let u2 = uniform(seed, 3 * i + 1);
                let gauss = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                BarDof {
                    r: (0.002 * gauss) as f32,
                    s: (0.003 * uniform(seed, 3 * i + 2)).max(0.0) as f32,
                    u: uniform(seed, 3 * i + 2) as f32,
                    v: uniform(seed, 3 * i) as f32,
                    w: (0.5 * gauss) as f32,
                }
            })
            .collect();
        BarSupports::fit(&samples)
    }

    // -----------------------------------------------------------------------
    // The reports
    // -----------------------------------------------------------------------

    /// Executes the writer of all FIVE registered bases and reads each one back, under a
    /// measured cost model so the arm sweep and the edge-versus-cost chart are exercised too.
    ///
    /// This is the test named in `pretrain_reports::tests::CYCLE_EXEMPT`: the portfolio bench
    /// is not part of a pretraining cycle, so the cycle walk cannot cover it and only an
    /// explicit execution can prove the bases are not blank panels.
    #[test]
    fn the_five_portfolio_bases_are_written_and_read_back() {
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..30)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..4u32)
                        .map(|id| (id, (0.004 * (uniform(2, u64::from(id) * 3 + t) - 0.5)) as f32))
                        .collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, 4);
        let model: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(t, slice)| PanelForecast {
                kelly_f: slice
                    .symbols
                    .iter()
                    .map(|id| (8.0 * (uniform(4, u64::from(*id) + t as u64) - 0.5)) as f32)
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let marginal = marginal_like(&panel, 2.0);
        let measured = fixture_cost_model(4);
        let bench = PortfolioBench::run(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &marginal,
            },
            &FlatCost::new(DEFAULT_COST_BPS),
            DEFAULT_COST_BPS,
            Some(&measured),
            &BacktestConfig::default(),
        )
        .expect("bench");

        let dir = scratch_dir("reports");
        write_portfolio_bench(&dir, "fixture", &bench).expect("charts land");
        for base in [
            PORTFOLIO_EQUITY_BASE,
            PORTFOLIO_METRICS_BASE,
            PORTFOLIO_GROSS_CURVE_BASE,
            PORTFOLIO_FRONTIER_BASE,
            PORTFOLIO_EDGE_BASE,
        ] {
            assert!(
                shared::report::PRETRAIN_REPORT_BASES.contains(&base),
                "{base} is written but not registered, so nothing can render it"
            );
            let report = read_report(&dir.join(format!("{base}.report.bin")))
                .unwrap_or_else(|e| panic!("{base} reads back: {e}"));
            let ReportKind::MultiLine { series } = report.kind else {
                panic!("{base} is not a MultiLine chart");
            };
            assert!(
                series.iter().any(|s| s.values.iter().any(|v| v.is_finite())),
                "{base} holds no finite value"
            );
        }
        // The equity chart carries one series per (policy, cap) and one point per instant
        // plus the starting wealth.
        let report = read_report(&dir.join(format!("{PORTFOLIO_EQUITY_BASE}.report.bin")))
            .expect("equity");
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("equity is not a MultiLine chart");
        };
        assert_eq!(series.len(), GROSS_CAPS.len() * POLICIES.len());
        assert!(series
            .iter()
            .all(|s| s.values.len() == panel.instants() + 1));
        assert!(!bench.table().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Alignment, the band, and the factor
    // -----------------------------------------------------------------------

    /// A panel whose forecast IS the realized return must earn exactly the oracle's payoff.
    ///
    /// This is the test for the one defect that would produce "cost survives, edge vanishes"
    /// without looking like a bug anywhere: a forecast scored against the wrong bar. A
    /// one-instant shift leaves every position, every turnover and every cost untouched and
    /// destroys only the payoff, which is indistinguishable from a model with no edge. So
    /// the alignment is pinned in both directions - the aligned forecast reproduces the
    /// ceiling to the last bit, and the shifted one provably does not.
    #[test]
    fn a_forecast_that_knows_the_realized_return_earns_the_oracle_payoff() {
        let names = 4usize;
        let instants = 24usize;
        // Every instant has one decisive name and three that barely move, so the oracle's
        // single-name answer is unambiguous and a shift changes which name it picks.
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..instants)
            .map(|t| {
                let winner = (t % names) as u32;
                let sign = if t % 2 == 0 { 1.0f32 } else { -1.0 };
                (
                    t as i64 * FIVE_MIN,
                    (0..names as u32)
                        .map(|id| {
                            let r = if id == winner {
                                sign * 0.01
                            } else {
                                0.0002 * (uniform(11, u64::from(id) * 31 + t as u64) - 0.5) as f32
                            };
                            (id, r)
                        })
                        .collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, names);

        // A forecast that is right about the sign and enormous about the size: after the
        // gross projection it is the oracle's own vector, budget on the single best name.
        let omniscient: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .map(|slice| {
                let best = slice
                    .realized_r
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .map(|(k, _)| k)
                    .expect("a non-empty slice");
                PanelForecast {
                    kelly_f: (0..slice.symbols.len())
                        .map(|k| {
                            if k == best {
                                100.0 * slice.realized_r[k].signum()
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                    mean_r: slice.realized_r.clone(),
                    var_r: vec![1.0e-4; slice.symbols.len()],
                }
            })
            .collect();
        let marginal = constant_forecast(&panel, &[0.0; 4]);
        let free = FlatCost::new(0.0);
        let inputs = PolicyInputs {
            model: &omniscient,
            marginal: &marginal,
        };
        let aligned = backtest(
            &panel,
            &inputs,
            Policy::Model,
            2.0,
            &free,
            &BacktestConfig::default(),
        )
        .expect("aligned");
        let oracle = backtest(
            &panel,
            &inputs,
            Policy::Oracle,
            2.0,
            &free,
            &BacktestConfig::default(),
        )
        .expect("oracle");
        for t in 0..panel.instants() {
            assert!(
                (aligned.payoff[t] - oracle.payoff[t]).abs() < 1e-12,
                "instant {t}: a forecast that knows the answer earned {} against the \
                 ceiling's {}; the realized bar and the forecast bar are not the same bar",
                aligned.payoff[t],
                oracle.payoff[t]
            );
        }

        // Now break it exactly one instant, the way an off-by-one would. Same positions,
        // same turnover, same cost, different bar - and the payoff must collapse.
        let mut shifted = omniscient.clone();
        shifted.rotate_left(1);
        let broken = backtest(
            &panel,
            &PolicyInputs {
                model: &shifted,
                marginal: &marginal,
            },
            Policy::Model,
            2.0,
            &free,
            &BacktestConfig::default(),
        )
        .expect("shifted");
        let aligned_total: f64 = aligned.payoff.iter().sum();
        let broken_total: f64 = broken.payoff.iter().sum();
        assert!(
            broken_total < 0.5 * aligned_total,
            "a one-instant shift earned {broken_total} against the aligned {aligned_total}; \
             this test cannot detect the misalignment it exists to detect"
        );
    }

    /// The band freezes trades, and the gross constraint survives the stale weights.
    ///
    /// Holding a position the projection wanted smaller is the one way banding can breach
    /// the budget, which is why the engine re-projects and why the assertion is on the FINAL
    /// vector. Turnover must fall, because that is the entire point of the band.
    #[test]
    fn the_band_cuts_turnover_without_breaching_the_gross_budget() {
        let names = 6usize;
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..80)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..names as u32)
                        .map(|id| {
                            (
                                id,
                                (0.006 * (uniform(7, u64::from(id) * 17 + t) - 0.5)) as f32,
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, names);
        // A forecast that churns: the sign flips on its own schedule per name, so an
        // unbanded book rebalances every instant and a banded one cannot.
        let churning: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(t, slice)| PanelForecast {
                kelly_f: slice
                    .symbols
                    .iter()
                    .map(|id| (6.0 * (uniform(23, u64::from(*id) * 13 + t as u64) - 0.5)) as f32)
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let marginal = constant_forecast(&panel, &[0.0; 6]);
        let inputs = PolicyInputs {
            model: &churning,
            marginal: &marginal,
        };
        let cap = 2.0f64;
        let mut previous = f64::INFINITY;
        for fraction in BAND_FRACTIONS {
            let band = fraction * cap / names as f64;
            for policy in POLICIES {
                let run = backtest(
                    &panel,
                    &inputs,
                    policy,
                    cap,
                    &FlatCost::new(0.0),
                    &BacktestConfig {
                        band,
                        ..BacktestConfig::default()
                    },
                )
                .expect("banded run");
                let budget = policy.gross_budget(cap);
                for (t, gross) in run.gross.iter().enumerate() {
                    assert!(
                        *gross <= budget * (1.0 + GROSS_TOLERANCE) + GROSS_TOLERANCE,
                        "{} at band {band} held gross {gross} against {budget} at {t}",
                        policy.name()
                    );
                }
            }
            let model = backtest(
                &panel,
                &inputs,
                Policy::Model,
                cap,
                &FlatCost::new(0.0),
                &BacktestConfig {
                    band,
                    ..BacktestConfig::default()
                },
            )
            .expect("model run");
            let turnover: f64 = model.turnover.iter().sum();
            assert!(
                turnover <= previous + 1e-9,
                "band {fraction} traded {turnover} against the tighter band's {previous}"
            );
            previous = turnover;
        }
        assert!(
            previous < 1e-6,
            "a band of a whole book should freeze it entirely, but it still traded {previous}"
        );
    }

    /// A panel where every name is the same bet must say so, and a neutral book must escape.
    ///
    /// Per-name Kelly sizes as if the names were independent. With four names moving as one,
    /// an equal-weight book's realized volatility is `sqrt(4)` times what independence
    /// implies, and that ratio is exactly the over-leverage a correlation-blind sizer buys.
    /// The dollar-neutral book, on the same panel, must carry almost none of the factor.
    #[test]
    fn the_leading_factor_and_the_leverage_error_are_measured_not_assumed() {
        let names = 4usize;
        let sd = 0.01f64;
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..400)
            .map(|t| {
                // One shared shock, no idiosyncratic part at all: the panel is rank one.
                let shock = (sd * (2.0 * uniform(31, t) - 1.0) * 3.0f64.sqrt()) as f32;
                (
                    t as i64 * FIVE_MIN,
                    (0..names as u32).map(|id| (id, shock)).collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, names);
        assert!(
            panel.first_factor_share() > 0.99,
            "a rank-one panel puts nearly all of its variance on one direction, got {}",
            panel.first_factor_share()
        );
        assert!(
            panel.first_factor().iter().all(|v| *v > 0.4),
            "every name loads on a shock every name shares: {:?}",
            panel.first_factor()
        );

        // The realized variance of the shared shock, handed to the sizer as each name's own
        // predictive variance. Independence would then imply `var / 4` for an equal book.
        let realized_var = {
            let all: Vec<f64> = panel
                .slices()
                .iter()
                .map(|s| f64::from(s.realized_r[0]).exp_m1())
                .collect();
            let mean = all.iter().sum::<f64>() / all.len() as f64;
            all.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (all.len() - 1) as f64
        };
        let forecast: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .map(|slice| PanelForecast {
                kelly_f: vec![1.0; slice.symbols.len()],
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![realized_var as f32; slice.symbols.len()],
            })
            .collect();
        let inputs = PolicyInputs {
            model: &forecast,
            marginal: &forecast,
        };
        let long_only = backtest(
            &panel,
            &inputs,
            Policy::EqualWeight,
            1.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("equal weight");
        let error = long_only.metrics.leverage_error;
        assert!(
            (error - (names as f64).sqrt()).abs() < 0.05,
            "four names moving as one are one name at four times the size: expected a \
             leverage error near {}, measured {error}",
            (names as f64).sqrt()
        );
        assert!(
            long_only.metrics.mean_first_factor_exposure > 0.4,
            "an equal-weight book on a rank-one panel IS the factor, measured {}",
            long_only.metrics.mean_first_factor_exposure
        );

        // A uniform forecast makes the neutral book identically flat, which is trivially
        // factor-free and proves nothing. Disagreeing per-name Kelly gives it a real
        // cross-section to hold, and neutrality then has to be earned: the panel is rank
        // one with a uniform loading, so a zero-sum weight vector projects to exactly zero.
        let spread: Vec<PanelForecast> = forecast
            .iter()
            .map(|f| PanelForecast {
                kelly_f: (0..f.kelly_f.len()).map(|k| 1.0 + k as f32).collect(),
                mean_r: f.mean_r.clone(),
                var_r: f.var_r.clone(),
            })
            .collect();
        let neutral = backtest(
            &panel,
            &PolicyInputs {
                model: &spread,
                marginal: &spread,
            },
            Policy::MarketNeutral,
            1.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("market neutral");
        assert!(
            neutral.metrics.mean_gross > 0.5,
            "the neutral book must actually hold something, gross {}",
            neutral.metrics.mean_gross
        );
        assert!(
            neutral.metrics.mean_first_factor_exposure < 1e-6,
            "subtracting the breadth-weighted mean must leave no factor behind, measured {}",
            neutral.metrics.mean_first_factor_exposure
        );
    }

    /// An UNMEASURED break-even may not borrow the clip bound, and an unmeasured row may not
    /// render as a verdict.
    ///
    /// `f64::min` returns the non-NaN operand, so the obvious `nan.min(MAX_BREAK_EVEN_BPS)`
    /// plots a book that never traded at exactly the most favourable break-even on the axis,
    /// beside real ones, with nothing to mark it. The same shape appears in every `>` over a
    /// float that can be absent: `NaN > cost` is false, so "not measured" renders as the
    /// confident "no". Three states, three renderings, and the language gives two for free.
    #[test]
    fn an_unmeasured_break_even_never_renders_as_a_measured_one() {
        // The language semantics this whole test exists because of. `black_box` because the
        // lint that flags a literal NaN comparison is itself the point being demonstrated.
        let absent = std::hint::black_box(f64::NAN);
        assert_eq!(absent.min(MAX_BREAK_EVEN_BPS), MAX_BREAK_EVEN_BPS);
        assert!(!(absent > 0.0) && !(absent <= 0.0));

        // The charted value keeps all three states apart.
        assert!(charted_break_even_bps(f64::NAN).is_nan());
        assert_eq!(
            charted_break_even_bps(f64::INFINITY),
            MAX_BREAK_EVEN_BPS,
            "an unkillable book is MEASURED and must stay on the axis at the search's bound"
        );
        assert_eq!(charted_break_even_bps(3.5), 3.5);
        assert_eq!(
            charted_break_even_bps(4.0 * MAX_BREAK_EVEN_BPS),
            MAX_BREAK_EVEN_BPS
        );

        // And the tables say so in words. A book whose Kelly fraction is exactly zero never
        // opens a position, so its cost per dollar TRADED is 0/0 - unmeasurable - which is
        // exactly the row that used to print a confident verdict.
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..2)
            .map(|t| (t as i64 * FIVE_MIN, vec![(0u32, 0.001f32), (1u32, -0.001)]))
            .collect();
        let panel = fixture_panel(&rows, 2);
        let forecasts = constant_forecast(&panel, &[0.0, 0.0]);
        let measured = fixture_cost_model(2);
        let bench = PortfolioBench::run(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            &FlatCost::new(DEFAULT_COST_BPS),
            DEFAULT_COST_BPS,
            Some(&measured),
            &BacktestConfig::default(),
        )
        .expect("bench");

        let flat = bench.flat_arm();
        let unmeasured_rows = flat
            .points
            .iter()
            .flatten()
            .filter(|p| p.realized_cost_bps.is_nan())
            .count();
        assert!(
            unmeasured_rows > 0,
            "the fixture must actually produce a zero-turnover row or this test proves nothing"
        );

        // The FLAT arm keeps its verdict on such a row, and that is correct rather than sloppy:
        // its cost is a stated constant that would apply to any trade, so "this book cannot bear
        // 2.00 bps" is a claim about the book, not about a missing measurement. Only the
        // realized-cost column is unmeasurable there, and it prints as NaN.
        let flat_table = bench.frontier_table(flat, f64::from(DEFAULT_COST_BPS));
        assert!(flat_table.contains("NaN"), "{flat_table}");

        // The MEASURED arm has no constant to fall back on, so the same row gets no verdict.
        let arm = bench
            .headline_measured_arm()
            .expect("the measured arm ran beside the flat one");
        let table = bench.frontier_table(arm, f64::from(DEFAULT_COST_BPS));
        assert!(
            table.contains("n/a"),
            "a measured arm with no measurable cost on a row must say n/a, table was:\n{table}"
        );
        // The old behaviour: the flat constant silently standing in for a measured arm's own
        // cost, and a comparison against it rendering as a confident yes or no.
        for line in table.lines().filter(|l| l.contains("NaN")) {
            assert!(
                line.contains("n/a"),
                "a measured row with an unmeasurable cost rendered a verdict: {line}"
            );
        }
    }

    /// Bars with a deterministic random walk, a positive volume and a VWAP, so a corpus
    /// fixture has a liquidity ranking to sort on.
    fn synthetic_bars(count: usize, first_ts: i64, base: f32, seed: u64) -> Vec<PackedBar> {
        let mut price = f64::from(base);
        (0..count)
            .map(|i| {
                let step = 0.004 * (uniform(seed, i as u64) - 0.5);
                price *= (1.0 + step).max(0.5);
                let close = price as f32;
                PackedBar {
                    ts_ms: first_ts + i as i64 * FIVE_MIN,
                    open: close,
                    high: close * 1.001,
                    low: close * 0.999,
                    close,
                    volume: 1000.0 + (i as f32),
                    vwap: close,
                    trades: 10,
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // The measured cost model, wired to panel ids
    // -----------------------------------------------------------------------

    /// Bars from a random walk in the EFFICIENT price observed through a fixed proportional
    /// spread: every bar's close prints at `mid * (1 +/- S/2)` on an independently drawn side,
    /// which is the data-generating process the Roll estimator is derived under and therefore
    /// the only kind of fixture whose MEASURED spread is a known number.
    ///
    /// Contiguous at the stride on purpose: `SymbolCost::measure` breaks the return chain across
    /// a gap, and a fixture full of gaps would have too few Roll pairs to measure anything.
    fn bounce_bars(
        count: usize,
        first_ts: i64,
        spread: f64,
        base: f64,
        volume: f32,
        seed: u64,
    ) -> Vec<PackedBar> {
        let mut log_mid = base.ln();
        (0..count)
            .map(|i| {
                let i = i as u64;
                log_mid += 0.001 * (uniform(seed, 2 * i) - 0.5);
                let mid = log_mid.exp();
                let side = if uniform(seed, 2 * i + 1) < 0.5 { 1.0 } else { -1.0 };
                let close = mid * (1.0 + side * 0.5 * spread);
                PackedBar {
                    ts_ms: first_ts + i as i64 * FIVE_MIN,
                    open: mid as f32,
                    high: (mid * (1.0 + 0.5 * spread)) as f32,
                    low: (mid * (1.0 - 0.5 * spread)) as f32,
                    close: close as f32,
                    volume,
                    vwap: mid as f32,
                    trades: 64,
                }
            })
            .collect()
    }

    /// A calibration over `names` series whose spreads, prices and volumes all differ, so a
    /// per-symbol cost model is distinguishable from a constant. Series `i` is symbol `i`, which
    /// is what [`fixture_panel`] produces.
    fn fixture_cost_series(names: usize) -> Vec<(String, Vec<PackedBar>)> {
        (0..names)
            .map(|i| {
                let spread = 0.0004 * (1.0 + i as f64);
                let price = 20.0 * (1.0 + i as f64);
                let volume = 2_000.0 * (1.0 + i as f32);
                (
                    format!("S{i}"),
                    bounce_bars(800, 0, spread, price, volume, 0x5EED + i as u64),
                )
            })
            .collect()
    }

    fn cost_model_of(series: &[(String, Vec<PackedBar>)]) -> BarCostModel {
        let borrowed: Vec<(String, &[PackedBar])> = series
            .iter()
            .map(|(symbol, bars)| (symbol.clone(), bars.as_slice()))
            .collect();
        BarCostModel::new(Arc::new(
            CostCalibration::from_series(&borrowed, 300).expect("the calibration measures"),
        ))
    }

    fn fixture_cost_model(names: usize) -> BarCostModel {
        cost_model_of(&fixture_cost_series(names))
    }

    /// The defect this whole batch exists to fix: `portfolio_cost` keys its calibration by
    /// CORPUS SERIES index while the panel carries a panel-local id, so a `BarCostModel` used
    /// directly prices each name at some unrelated symbol's liquidity.
    ///
    /// The fixture is built so the two orders genuinely disagree — the panel ranks by dollar
    /// volume measured before the span, so its id `0` is the corpus's most liquid series, not
    /// its first — and then asserts that [`PanelCost`] charges panel id `k` exactly what the
    /// calibration charges `series_of(k)`, and that the untranslated model charges something
    /// else. Without the second half the test would pass on an identity mapping.
    #[test]
    fn the_panel_cost_translates_panel_ids_into_corpus_series() {
        let dir = scratch_dir("cost_ids");
        let res = 300u32;
        let history = 40usize;
        let names = 3usize;
        // Thin first, deep last: the panel's liquidity ranking must therefore REVERSE the
        // corpus order, which is exactly the disagreement the translation exists for.
        let series = (0..names)
            .map(|i| {
                let spread = 0.0010 * (1.0 + 3.0 * i as f64);
                let volume = 1_000.0 * (1.0 + 10.0 * i as f32);
                (
                    format!("S{i}"),
                    bounce_bars(history + 60, 0, spread, 50.0, volume, 0xAB + i as u64),
                )
            })
            .collect::<Vec<_>>();
        for (symbol, bars) in &series {
            write_bar_file(&dir.join(format!("{symbol}.{res}.bars")), symbol, res, bars)
                .expect("bars land");
        }
        let start = history as i64 * FIVE_MIN;
        let end = (history + 60) as i64 * FIVE_MIN;
        let corpus = BarCorpus::load_with_bounds(&dir, res, 1, (start, end)).expect("corpus");
        let config = PanelConfig {
            start_ts_ms: start,
            end_ts_ms: end,
            max_symbols: names,
            min_history: 1,
            max_instants: 50,
        };
        let panel = Panel::build(&corpus, &config).expect("panel");
        let model = cost_model_of(&series);
        assert_eq!(
            model.calibration().len(),
            corpus.series_count(),
            "the calibration must cover the corpus it is keyed by"
        );
        let translated = PanelCost::new(&panel, model.clone(), CostParts::All);

        let mut disagreements = 0usize;
        for id in 0..panel.symbols().len() as u32 {
            let series_index = panel.series_of(id) as u32;
            let expected = model.resolve(series_index, start).total_bps(0.0);
            let charged = f64::from(translated.cost_bps(id, start, 0.0));
            assert!(
                (charged - expected).abs() < 1e-6,
                "panel id {id} (series {series_index}) was charged {charged} bps against its \
                 own symbol's {expected}"
            );
            // What the defect did: the untranslated model reads the calibration at the PANEL
            // id, which is a different symbol whenever the two orders disagree.
            let untranslated = f64::from(model.cost_bps(id, start, 0.0));
            if (untranslated - expected).abs() > 1e-6 {
                disagreements += 1;
            }
        }
        assert!(
            disagreements > 0,
            "the fixture must make the panel order and the corpus order disagree, or the \
             translation is untested"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A leg whose symbol has NO measurable volatility is counted, not given free impact.
    ///
    /// The flat-close series has `sigma_daily == 0`, which `portfolio_cost` propagates as an
    /// unpriceable impact coefficient. Charging it at zero would be the most dangerous number in
    /// a cost model, so [`PanelCost`] substitutes the cross-sectional median and declares it,
    /// and the count travels on the row.
    #[test]
    fn an_unpriceable_impact_is_counted_and_never_free() {
        let mut series = fixture_cost_series(2);
        // A symbol whose close never moves: finite, positive, and completely unmeasurable.
        let flat: Vec<PackedBar> = (0..800)
            .map(|i| PackedBar {
                ts_ms: i as i64 * FIVE_MIN,
                open: 40.0,
                high: 40.0,
                low: 40.0,
                close: 40.0,
                volume: 5_000.0,
                vwap: 40.0,
                trades: 64,
            })
            .collect();
        series.push(("FLAT".to_owned(), flat));
        let model = cost_model_of(&series);
        let dead = (series.len() - 1) as u32;
        // Its OWN volatility is unmeasurable at every tier...
        let pooled = &model.calibration().symbols[dead as usize].pooled;
        assert!(
            !(pooled.sigma_daily.is_finite() && pooled.sigma_daily > 0.0),
            "a flat close must leave no measurable volatility, got {}",
            pooled.sigma_daily
        );
        // ...and `resolve` nonetheless returns a perfectly finite coefficient, because it
        // substitutes the cross-sectional median. THAT is why the substitution has to be read
        // out of the buckets: there is nothing in the resolved number to notice.
        assert!(
            model.resolve(dead, 0).impact_coefficient_bps.is_finite(),
            "the median stand-in is a number, which is exactly the trap"
        );

        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..6)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..series.len() as u32).map(|id| (id, 0.001f32)).collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, series.len());
        let priced = PanelCost::new(&panel, model, CostParts::All);
        let leg = priced.leg_cost(dead, 0, 0.01);
        assert!(
            leg.impact_substituted,
            "the unpriceable leg must declare its stand-in"
        );
        assert!(
            leg.bps.is_finite() && leg.bps > 0.0,
            "an unpriceable leg is charged the cross-sectional median, never zero: {}",
            leg.bps
        );
        // And a symbol that CAN be priced declares nothing.
        assert!(!priced.leg_cost(0, 0, 0.01).impact_substituted);

        // The count reaches the run, and the turnover it priced with it.
        let forecasts = constant_forecast(&panel, &vec![3.0; series.len()]);
        let run = backtest(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            Policy::Model,
            2.0,
            &priced,
            &BacktestConfig::default(),
        )
        .expect("the run prices every leg");
        assert!(
            run.legs.impact_substituted > 0 && run.legs.substituted_turnover > 0.0,
            "the unpriceable legs must be counted on the run: {:?}",
            run.legs
        );
        assert!(run.legs.legs >= run.legs.impact_substituted);
        assert!(run.metrics.substituted_leg_share > 0.0);
    }

    /// The three arms differ by EXACTLY the components they name, on the same leg.
    ///
    /// `all-in` minus `no fees` must be the commission plus the regulatory fee, and `all-in`
    /// minus `impact-free` must be the square-root impact term. If either identity fails, an
    /// arm is answering a different question than its label claims.
    #[test]
    fn each_cost_arm_removes_exactly_the_component_it_names() {
        let series = fixture_cost_series(3);
        let model = cost_model_of(&series);
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..4)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..3u32).map(|id| (id, 0.0f32)).collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, 3);
        let all = PanelCost::new(&panel, model.clone(), CostParts::All);
        let no_fees = all.with_parts(CostParts::NoFees);
        let no_impact = all.with_parts(CostParts::NoImpact);
        let participation = 0.02f32;
        for id in 0..3u32 {
            let resolved = model.resolve(panel.series_of(id) as u32, 0);
            let full = f64::from(all.cost_bps(id, 0, participation));
            let fee_free = f64::from(no_fees.cost_bps(id, 0, participation));
            let impact_free = f64::from(no_impact.cost_bps(id, 0, participation));
            assert!(
                (full - fee_free - (resolved.commission_bps + resolved.regulatory_bps)).abs()
                    < 1e-5,
                "the fee arm must remove exactly the commission and the regulatory fee"
            );
            assert!(
                (full - impact_free - resolved.impact_bps(f64::from(participation))).abs() < 1e-5,
                "the impact-free arm must remove exactly the square-root impact"
            );
            // And the impact-free arm cannot depend on the coefficient it does not charge.
            for k in IMPACT_K_GRID {
                let other = PanelCost::new(&panel, model.with_impact_k(k), CostParts::NoImpact);
                assert!(
                    (f64::from(other.cost_bps(id, 0, participation)) - impact_free).abs() < 1e-9,
                    "the impact-free arm moved with k = {k}, so it is not assumption-free"
                );
            }
        }
    }

    /// The impact term is an ASSUMPTION, so every measured arm must exist once per coefficient
    /// and the net figures must be ordered by it: more impact can only cost more.
    #[test]
    fn the_impact_coefficient_is_swept_and_orders_the_net_growth() {
        let names = 4usize;
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..40)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..names as u32)
                        .map(|id| {
                            (
                                id,
                                (0.004 * (uniform(11, u64::from(id) * 7 + t) - 0.5)) as f32,
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, names);
        let model: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(t, slice)| PanelForecast {
                kelly_f: slice
                    .symbols
                    .iter()
                    .map(|id| (6.0 * (uniform(13, u64::from(*id) * 5 + t as u64) - 0.5)) as f32)
                    .collect(),
                mean_r: vec![0.0; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let marginal = marginal_like(&panel, 2.0);
        let measured = fixture_cost_model(names);
        // A book big enough that the square-root impact is not a rounding error, so the sweep
        // has something to order.
        let bench = PortfolioBench::run(
            &panel,
            &PolicyInputs {
                model: &model,
                marginal: &marginal,
            },
            &FlatCost::new(DEFAULT_COST_BPS),
            DEFAULT_COST_BPS,
            Some(&measured),
            &BacktestConfig {
                capital_usd: 1.0e9,
                band: 0.0,
            },
        )
        .expect("bench");

        assert_eq!(bench.arms.len(), 1 + 1 + 2 * IMPACT_K_GRID.len());
        assert!(bench.flat_arm().arm == CostArm::Flat { bps: DEFAULT_COST_BPS });
        assert!(bench.headline_measured_arm().is_some());
        let free = bench
            .assumption_free_arm()
            .expect("the impact-free arm is always run beside the measured ones");
        assert!(free.arm.is_assumption_free());

        // Net growth is monotone DOWN in the assumed coefficient, and the impact-free arm is
        // the ceiling of the all-in family: it charges strictly less than any of them.
        let mut previous = f64::INFINITY;
        for k in IMPACT_K_GRID {
            let arm = bench
                .arms
                .iter()
                .find(|a| {
                    a.arm
                        == CostArm::Measured {
                            impact_k: k,
                            parts: CostParts::All,
                        }
                })
                .expect("every grid coefficient has an arm");
            let net = arm.points[0][0].log_growth_per_year;
            assert!(
                net <= previous + 1e-12,
                "k = {k} netted {net} against the smaller coefficient's {previous}"
            );
            previous = net;
            assert!(
                free.points[0][0].realized_cost_bps < arm.points[0][0].realized_cost_bps,
                "the impact-free arm must be the cheapest of the measured family: {} vs {}",
                free.points[0][0].realized_cost_bps,
                arm.points[0][0].realized_cost_bps
            );
            // The zero-fee arm of the same coefficient charges strictly less than the all-in.
            let no_fees = bench
                .arms
                .iter()
                .find(|a| {
                    a.arm
                        == CostArm::Measured {
                            impact_k: k,
                            parts: CostParts::NoFees,
                        }
                })
                .expect("every grid coefficient has a fee-free arm");
            assert!(
                no_fees.points[0][0].realized_cost_bps < arm.points[0][0].realized_cost_bps,
                "removing the commission must reduce what the book paid"
            );
        }
        // Every arm traded the same book, so the gross growth cannot move with the cost.
        for arm in &bench.arms {
            for (p, row) in arm.points.iter().enumerate() {
                for (b, point) in row.iter().enumerate() {
                    assert!(
                        (point.gross_log_growth_per_year
                            - bench.flat_arm().points[p][b].gross_log_growth_per_year)
                            .abs()
                            < 1e-12,
                        "[{}] moved the GROSS growth of {} at band {}",
                        arm.arm.label(),
                        point.policy,
                        point.band_fraction
                    );
                }
            }
        }
        // The decomposition adds up, per decile and pooled, at every coefficient.
        assert!(bench.edge.measured);
        for row in bench.edge.deciles.iter().chain([&bench.edge.pooled]) {
            if row.symbols == 0 {
                continue;
            }
            for slot in 0..IMPACT_K_GRID.len() {
                let sum = row.half_spread_bps
                    + row.commission_bps
                    + row.regulatory_bps
                    + row.impact_bps[slot];
                assert!(
                    (sum - row.all_in_bps[slot]).abs() < 1e-9,
                    "decile {}'s four components must sum to its all-in cost: {sum} vs {}",
                    row.decile as i64,
                    row.all_in_bps[slot]
                );
            }
            assert!(row.commission_bps > 0.0 && row.half_spread_bps > 0.0);
            assert!(row.sd_r_bps > 0.0 && row.name_bars > 0);
        }
        assert!(!bench.cost_arm_table().is_empty());
        assert!(!bench.crossing_table().is_empty());
        assert!(!bench.edge_table().is_empty());
        assert!(!bench.win_rate_table().is_empty());
    }

    /// The three things a "win rate" can mean are three different numbers, and each is what it
    /// says it is on a panel whose answer is written down by hand.
    ///
    /// One name, four instants, a constant long position of the whole budget. The realized log
    /// returns are `+, -, +, +`: three of four bars are up, so the bar win rate is `0.75` and so
    /// is the sign agreement — with one name and one sign they must coincide, which is exactly
    /// why the portfolio case needs all three reported. The position never leaves the book, so
    /// there is exactly ONE round-trip lifecycle, closed at the last instant, and its P&L is the
    /// sum of the four bars.
    #[test]
    fn the_three_win_rates_are_three_different_statistics() {
        let log_returns = [0.01f32, -0.004, 0.002, 0.003];
        let rows: Vec<(i64, Vec<(u32, f32)>)> = log_returns
            .iter()
            .enumerate()
            .map(|(t, r)| (t as i64 * FIVE_MIN, vec![(0u32, *r)]))
            .collect();
        let panel = fixture_panel(&rows, 1);
        let forecasts = constant_forecast(&panel, &[4.0]);
        let run = backtest(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            Policy::Model,
            1.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("run");
        assert_eq!(run.trades.trades, 1, "one position, opened once, closed once");
        assert_eq!(run.trades.bars_held, 4);
        assert_eq!(run.trades.positioned_legs, 4);
        assert_eq!(run.trades.sign_agreements, 3);
        assert_eq!(run.trades.gross_wins, 1, "the lifecycle made money overall");
        assert_eq!(run.trades.net_wins, 1, "at a zero cost, net equals gross");
        assert!((run.metrics.bar_win_rate - 0.75).abs() < 1e-12);
        assert!((run.metrics.position_sign_agreement - 0.75).abs() < 1e-12);
        assert!((run.metrics.trade_win_rate - 1.0).abs() < 1e-12);
        assert!((run.metrics.mean_hold_bars - 4.0).abs() < 1e-12);

        // A cost large enough to sink the lifecycle flips (b) without touching (a)'s gross
        // arithmetic or (c) at all: the three are genuinely independent statistics.
        let expensive = backtest(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            Policy::Model,
            1.0,
            &FlatCost::new(400.0),
            &BacktestConfig::default(),
        )
        .expect("run");
        assert_eq!(expensive.trades.gross_wins, 1);
        assert_eq!(expensive.trades.net_wins, 0, "the round trip paid to lose");
        assert!((expensive.metrics.position_sign_agreement - 0.75).abs() < 1e-12);
        assert!(expensive.metrics.bar_win_rate < run.metrics.bar_win_rate);

        // A position that FLIPS is two lifecycles, not one, and the exit pays for the flip.
        let flipping: Vec<PanelForecast> = (0..log_returns.len())
            .map(|t| PanelForecast {
                kelly_f: vec![if t % 2 == 0 { 4.0 } else { -4.0 }],
                mean_r: vec![0.0],
                var_r: vec![1.0e-4],
            })
            .collect();
        let flips = backtest(
            &panel,
            &PolicyInputs {
                model: &flipping,
                marginal: &forecasts,
            },
            Policy::Model,
            1.0,
            &FlatCost::new(0.0),
            &BacktestConfig::default(),
        )
        .expect("run");
        assert_eq!(
            flips.trades.trades, 4,
            "four bars of alternating sign are four lifecycles"
        );
        assert_eq!(flips.trades.bars_held, 4);
    }

    /// The panel's own liquidity deciles carry BOTH halves of the comparison, and the edge half
    /// is a property of the forecast rather than of the book.
    ///
    /// The fixture gives every name-bar the same realized return and a forecast whose sign is
    /// always right, so the directional edge must be exactly that return in bps and the sign
    /// agreement exactly `1.0`. A sign-blind estimator would report zero edge on the same panel.
    #[test]
    fn the_edge_table_measures_the_forecast_not_the_book() {
        let names = 10usize;
        let realized = 0.0005f32;
        // An EVEN number of instants, so the alternating return has an exactly zero
        // unconditional mean rather than a nearly zero one. `fixture_panel` makes every row a
        // tradeable slice, unlike `Panel::build`, which spends the first one as the reference.
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..20)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..names as u32)
                        .map(|id| (id, if t % 2 == 0 { realized } else { -realized }))
                        .collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, names);
        // The forecast knows the sign of every bar, and nothing else. Its sign is read off the
        // panel's OWN slice rather than off a row parity, so the test cannot pass by accident on
        // a panel whose slices are offset from its rows.
        let forecasts: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .map(|slice| PanelForecast {
                kelly_f: slice
                    .realized_r
                    .iter()
                    .map(|r| if *r >= 0.0 { 2.0 } else { -2.0 })
                    .collect(),
                mean_r: slice.realized_r.clone(),
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let table = EdgeVsCostTable::measure(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            None,
        );
        assert!(!table.measured, "no cost model was supplied");
        assert!(table.pooled.half_spread_bps.is_nan(), "and none is invented");
        assert_eq!(table.deciles.len(), DECILES);
        assert!((table.pooled.forecast_sign_agreement - 1.0).abs() < 1e-12);
        assert!(
            (table.pooled.signed_edge_bps - 1.0e4 * f64::from(realized)).abs() < 1e-6,
            "the directional edge must be the realized move itself: {}",
            table.pooled.signed_edge_bps
        );
        // The unconditional mean is zero on this panel, which is the entire point: a predictor
        // with no drift to ride can still carry a directional edge, and only the signed
        // statistic can see it.
        assert!(table.pooled.mean_r_bps.abs() < 1e-9);
        assert!((table.pooled.sd_r_bps - 1.0e4 * f64::from(realized)).abs() < 1e-6);
        assert_eq!(table.pooled.name_bars, (names * panel.instants()) as u64);
        // Every name is identical here, so the cross-sectional mean is the same at every
        // instant and the Fama-MacBeth standard error is zero rather than merely small.
        assert!(table.pooled.signed_edge_se_bps < 1e-9);
    }

    /// A panel whose REGIME is a property of the day: every name and every instant inside one
    /// day carries the same return, so the panel holds `days` independent draws however many
    /// name-bars it has. Exactly the structure that makes an instant-resampled interval wrong.
    fn day_regime_panel(names: usize, days: usize, per_day: usize, seed: u64) -> Panel {
        let mut rows: Vec<(i64, Vec<(u32, f32)>)> = Vec::with_capacity(days * per_day);
        for day in 0..days {
            let shock = if uniform(seed, day as u64) > 0.5 {
                1.0f32
            } else {
                -1.0f32
            };
            for step in 0..per_day {
                let ts = day as i64 * MS_PER_DAY + step as i64 * FIVE_MIN;
                rows.push((
                    ts,
                    (0..names as u32).map(|id| (id, 0.002 * shock)).collect(),
                ));
            }
        }
        fixture_panel(&rows, names)
    }

    /// Always long, so `sign(f*) r` is the realized return itself and the edge inherits whatever
    /// structure the panel has.
    fn always_long(panel: &Panel) -> Vec<PanelForecast> {
        panel
            .slices()
            .iter()
            .map(|slice| PanelForecast {
                kelly_f: vec![1.0; slice.symbols.len()],
                mean_r: slice.realized_r.clone(),
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect()
    }

    /// The interval on the edge is governed by the number of BLOCKS, not by the number of
    /// name-bars, and on a panel whose regime is the day the instant-blocked interval is a floor
    /// that is far too tight. Both are asserted, because a bootstrap that ignored its blocking
    /// would pass a one-sided version of this test.
    ///
    /// Also pins the two properties that make the interval readable at all: the point estimate
    /// is the FULL-sample statistic and cannot depend on the blocking, and a FIXED cost
    /// denominator can only rescale the edge's interval rather than add uncertainty of its own.
    #[test]
    fn the_day_blocked_edge_interval_is_wider_than_the_instant_blocked_one() {
        let (names, days, per_day) = (4usize, 12usize, 20usize);
        let panel = day_regime_panel(names, days, per_day, 0xD0D0_1234);
        let forecasts = always_long(&panel);
        let model = fixture_cost_model(names);
        let inputs = PolicyInputs {
            model: &forecasts,
            marginal: &forecasts,
        };
        let table = EdgeVsCostTable::measure(&panel, &inputs, Some(&model));
        let day = &table.intervals.by_day;
        let instant = &table.intervals.by_instant;
        assert_eq!(day.blocks, days, "one resampling unit per trading day");
        assert_eq!(instant.blocks, panel.instants(), "one per instant");
        assert_eq!(day.instants, panel.instants());

        let pooled_edge = table.pooled.signed_edge_bps;
        for set in [day, instant] {
            let point = set.pooled_signed_edge_bps.point;
            assert!(
                (point - pooled_edge).abs() < 1e-9 * pooled_edge.abs().max(1.0),
                "the {} bootstrap moved the point estimate: {point} against the table's \
                 {pooled_edge}",
                set.blocking,
            );
        }
        let (wide, tight) = (
            day.pooled_signed_edge_bps.se,
            instant.pooled_signed_edge_bps.se,
        );
        assert!(
            wide > 3.0 * tight,
            "a day-level regime must widen the interval: {wide:.4} over {days} day blocks \
             against {tight:.4} over {} instant blocks",
            panel.instants(),
        );

        // A fixed denominator rescales and nothing more, which is what "sampling error in the
        // edge only" MEANS rather than merely claims.
        let trip = &day.pooled_edge_over_round_trip;
        let cost = 2.0 * table.pooled.headline_all_in_bps();
        assert!(cost.is_finite() && cost > 0.0, "the fixture prices a cost");
        assert!(
            (trip.point - pooled_edge / cost).abs() < 1e-9 * trip.point.abs().max(1.0),
            "round trips per edge must be the edge over TWICE the one-way cost: {}",
            trip.point,
        );
        assert!(
            (trip.se - wide / cost).abs() < 1e-9 * trip.se.abs().max(1.0),
            "a fixed cost can only rescale the interval: {} against {}",
            trip.se,
            wide / cost,
        );
        assert!(trip.is_measured(), "and the rescaled interval is measured");

        // The interval is a property of the data, not of the run.
        let again = EdgeVsCostTable::measure(&panel, &inputs, Some(&model));
        assert_eq!(
            again.intervals.by_day.pooled_signed_edge_bps, day.pooled_signed_edge_bps,
            "two bootstraps of one panel must agree to the last digit",
        );
        assert_eq!(
            again.intervals.by_day.deepest_over_pooled, day.deepest_over_pooled,
            "including the paired ratio",
        );
    }

    /// One block is one observation. A panel inside a single trading day admits a point estimate
    /// and NO interval, and it must not render a zero-width one as precision - which is exactly
    /// what averaging one block over and over would produce.
    #[test]
    fn a_single_block_reports_a_point_and_refuses_an_interval() {
        let names = 4usize;
        let panel = day_regime_panel(names, 1, 20, 0xD0D0_5678);
        let forecasts = always_long(&panel);
        let table = EdgeVsCostTable::measure(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            None,
        );
        let day = &table.intervals.by_day;
        assert_eq!(day.blocks, 1, "twenty five-minute bars are one UTC day");
        let ci = day.pooled_signed_edge_bps;
        assert!(ci.point.is_finite(), "the point estimate still exists");
        assert!(!ci.is_measured(), "one block cannot carry an interval");
        assert_eq!(ci.excludes(0.0), None, "and no verdict reads off it");
        assert_eq!(ci.verdict(0.0), "n/a");
        assert!(
            ci.text().contains("unmeasured"),
            "rendered as a refusal rather than a number: {}",
            ci.text(),
        );
        // Twenty instant blocks over the same bars DO resolve, so the refusal above is a
        // property of the blocking rather than of the panel.
        assert!(table.intervals.by_instant.pooled_signed_edge_bps.is_measured());
        assert_eq!(table.intervals.by_instant.blocks, panel.instants());
    }

    /// An unmeasured COST must leave the round-trip interval unmeasured while the EDGE interval
    /// stands. The two halves of that ratio come from different machinery and only one of them
    /// can be missing, so a blanket "no cost model, no intervals" would throw away a measurement
    /// that exists.
    #[test]
    fn an_unpriced_panel_keeps_its_edge_interval_and_refuses_the_round_trip_one() {
        let names = 4usize;
        let panel = day_regime_panel(names, 12, 20, 0xD0D0_9ABC);
        let forecasts = always_long(&panel);
        let table = EdgeVsCostTable::measure(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            None,
        );
        assert!(!table.measured, "no cost model was supplied");
        let day = &table.intervals.by_day;
        assert!(
            day.pooled_signed_edge_bps.is_measured(),
            "the edge needs no cost model and its interval must survive",
        );
        let trip = day.pooled_edge_over_round_trip;
        assert!(trip.point.is_nan(), "an unpriced round trip is not a number");
        assert!(!trip.is_measured());
        assert_eq!(trip.verdict(1.0), "n/a", "and it carries no verdict");
        assert_eq!(trip.excludes(1.0), None);
        let rendered = table.interval_table();
        assert!(
            rendered.contains("n/a"),
            "the table must print the refusal: {rendered}",
        );
    }

    /// A bar that did not move has no sign to be right about, and both directional statistics
    /// are attenuated by exactly the share of such bars.
    ///
    /// `sign(f*) * r` contributes zero from a flat bar while the denominator counts it, so a
    /// forecast that is right on EVERY moving bar reports an agreement of `1 - flat_share` and
    /// an edge scaled by the same factor. That is the correct denominator for dividing a
    /// round-trip cost into - a flat bar still pays to be traded - and the wrong one for a
    /// directional claim or for any RATIO between deciles, because each decile carries its own
    /// flat share and the factor does not cancel.
    #[test]
    fn a_flat_bar_attenuates_the_directional_statistics_by_exactly_its_share() {
        let names = 4usize;
        let realized = 0.0005f32;
        // Every SECOND instant is flat for every name: half the positioned bars have no sign.
        let rows: Vec<(i64, Vec<(u32, f32)>)> = (0..20)
            .map(|t| {
                (
                    t as i64 * FIVE_MIN,
                    (0..names as u32)
                        .map(|id| (id, if t % 2 == 0 { realized } else { 0.0 }))
                        .collect(),
                )
            })
            .collect();
        let panel = fixture_panel(&rows, names);
        // Always long, always the same size: the forecast is right on every bar that moved.
        let forecasts: Vec<PanelForecast> = panel
            .slices()
            .iter()
            .map(|slice| PanelForecast {
                kelly_f: vec![2.0; slice.symbols.len()],
                mean_r: vec![realized; slice.symbols.len()],
                var_r: vec![1.0e-4; slice.symbols.len()],
            })
            .collect();
        let table = EdgeVsCostTable::measure(
            &panel,
            &PolicyInputs {
                model: &forecasts,
                marginal: &forecasts,
            },
            None,
        );
        let pooled = &table.pooled;
        assert_eq!(pooled.positioned_bars, (names * panel.instants()) as u64);
        assert_eq!(pooled.flat_positioned_bars, pooled.positioned_bars / 2);
        assert!((pooled.flat_share() - 0.5).abs() < 1e-12);

        // The attenuated pair: a perfect forecast reading as a coin flip.
        assert!(
            (pooled.forecast_sign_agreement - 0.5).abs() < 1e-12,
            "attenuated agreement {}",
            pooled.forecast_sign_agreement
        );
        assert!(
            (pooled.signed_edge_bps - 0.5 * 1.0e4 * f64::from(realized)).abs() < 1e-6,
            "attenuated edge {}",
            pooled.signed_edge_bps
        );

        // The moving-bar pair: the forecast as it actually is.
        assert!(
            (pooled.sign_agreement_on_moving_bars() - 1.0).abs() < 1e-12,
            "moving-bar agreement {}",
            pooled.sign_agreement_on_moving_bars()
        );
        assert!(
            (pooled.signed_edge_per_moving_bar_bps() - 1.0e4 * f64::from(realized)).abs() < 1e-6,
            "moving-bar edge {}",
            pooled.signed_edge_per_moving_bar_bps()
        );
        // A book that never positioned has no denominator at all, and the accessors must say
        // "not measured" rather than pick one.
        let unpositioned = EdgeVsCost::empty(0);
        assert!(unpositioned.flat_share().is_nan());
        assert!(unpositioned.sign_agreement_on_moving_bars().is_nan());
        assert!(unpositioned.signed_edge_per_moving_bar_bps().is_nan());
    }

    /// The volume EMA span the encoder warms up over is a corpus-level constant; this module
    /// must not have grown its own copy.
    #[test]
    fn the_adv_window_is_stated_in_bars_not_derived_from_the_encoder() {
        assert!(ADV_TRAILING_BARS > BAR_VOLUME_EMA_SPAN as usize);
        assert_eq!(BELIEF_PRE_CONTEXT + BELIEF_EMIT, BAR_MAX_CONTEXT);
    }

    /// The real verdict: the promoted checkpoint traded as ONE book over a calendar-aligned
    /// panel of the PINNED held-out split.
    ///
    /// Ignored by default because it needs the 451M-bar corpus, a 156 MB checkpoint and a
    /// belief pass over every panel bar. Run it with
    ///
    /// ```text
    /// OMP_NUM_THREADS=8 TORCH_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    ///     ./torch-env.sh cargo test --release --lib -j 4 \
    ///     the_promoted_checkpoint_trades_the_held_out_panel -- --ignored --nocapture
    /// ```
    ///
    /// # What this costs, MEASURED, per command shape
    ///
    /// Read `(user + sys) / wall` once per shape and write it down; a core count INFERRED from a
    /// pool size has been wrong by more than 2x elsewhere in this tree. Both figures below were
    /// taken on the release test binary directly rather than through `cargo test`, because
    /// cargo's freshness check adds about 34 s of wall and its rustc jobs land in the same
    /// totals. A GPU-bound sweep measured at ~1 core was running throughout, so if anything
    /// these over-count slightly.
    ///
    /// * WHOLE BOX - the configuration whose numbers are quoted. `OMP_NUM_THREADS=8`,
    ///   `TORCH_NUM_THREADS=8`, rayon's global pool unbounded, `PORTFOLIO_COST_THREADS=8`,
    ///   100 symbols x 2000 instants: **244 s wall, 8.85 cores** (1794 s user + 364 s sys).
    ///   This is a lane-holding job - queue behind it, not beside it.
    /// * CHEAP SMOKE - everything pinned to `1` including `RAYON_NUM_THREADS`, plus
    ///   `PORTFOLIO_COST_THREADS=0` so only the flat reference arm runs, 20 symbols x 150
    ///   instants: **37 s wall, 1.56 cores** (47 s user + 10 s sys). Runs beside anything.
    ///
    /// The residual 0.56 cores in the fully pinned shape is corpus mmap page-in, not a thread
    /// pool: `sys` is a fifth of that total. Pinning cost nothing in wall time in either shape.
    ///
    /// # Three thread budgets, one lever each
    ///
    /// The trap is that no single variable bounds this test. The belief pass runs tensor ops and
    /// answers to `OMP_NUM_THREADS` (`--test-threads` caps concurrent TESTS, not the threads
    /// inside one; `TORCH_NUM_THREADS` is reported by other agents in this tree to be read too
    /// late to bind, a claim measured by them and not by this file). `BarCorpus` opening and
    /// batch assembly use rayon's GLOBAL pool via bare `par_iter`, so they answer to
    /// `RAYON_NUM_THREADS`. The cost calibration answers to NEITHER: it runs no tensor ops and
    /// [`CostCalibration::from_corpus`] builds an EXPLICIT pool with
    /// `ThreadPoolBuilder::num_threads`, fed here by `PORTFOLIO_COST_THREADS`, which is a
    /// function argument no environment variable can reach.
    ///
    /// The panel size is an env knob rather than a constant because the belief pass is the
    /// whole cost and the affordable size depends entirely on whether a GPU is free; the
    /// DEFAULTS are the configuration whose numbers are quoted, and every run prints the
    /// panel it actually measured beside the table so a smaller run can never be mistaken
    /// for the headline. No measured number depends on any of the three thread counts: the
    /// calibration is per symbol and every cost arm is charged on one fixed set of cost-blind
    /// weights.
    #[test]
    #[ignore = "needs the long_data corpus and a trained checkpoint"]
    fn the_promoted_checkpoint_trades_the_held_out_panel() {
        let env_usize = |name: &str, fallback: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(fallback)
        };
        // Cargo runs a unit test with the PACKAGE as its working directory, while the
        // corpus and the run directories live at the workspace root one level up.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the package sits inside the workspace")
            .to_path_buf();
        let at = |name: &str, fallback: &str| -> PathBuf {
            std::env::var(name).map_or_else(|_| root.join(fallback), PathBuf::from)
        };
        let checkpoint = at(
            "PORTFOLIO_CHECKPOINT",
            "training/runs/bardist_v2/weights/pretrain_best.ot",
        );
        let run = std::env::var("PORTFOLIO_RUN")
            .unwrap_or_else(|_| "bardist_v2_portfolio".to_owned());
        let gens = root.join("training/runs").join(&run).join("gens/0");

        let args = PortfolioArgs {
            bars_dir: at("PORTFOLIO_BARS", "long_data/bars"),
            checkpoint,
            gens_dir: gens.clone(),
            res_secs: 300,
            device: Device::cuda_if_available(),
            split_bounds: crate::data::ingest::PINNED_SPLIT_BOUNDS,
            max_symbols: env_usize("PORTFOLIO_SYMBOLS", 100),
            max_instants: env_usize("PORTFOLIO_INSTANTS", 2000),
            cost_bps: std::env::var("PORTFOLIO_COST_BPS")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(DEFAULT_COST_BPS),
            // The per-symbol calibration walks every bar of the corpus, so its thread count is
            // an env knob for the same reason the panel size is. `PORTFOLIO_COST_THREADS=0`
            // reports the flat arm alone and says so in every table.
            cost_threads: env_usize("PORTFOLIO_COST_THREADS", 8),
            capital_usd: 1.0e7,
            label: run.clone(),
        };
        let started = std::time::Instant::now();
        let bench = run_portfolio_backtest(&args).expect("the held-out panel backtests");
        println!("{}", bench.table());
        println!("{}", bench.cost_arm_table());
        println!("{}", bench.crossing_table());
        println!("{}", bench.edge_table());
        println!("{}", bench.win_rate_table());
        println!(
            "[portfolio] {:.1}s, charts in {}",
            started.elapsed().as_secs_f64(),
            gens.display()
        );

        // The panel is held out by construction and the book is one book.
        assert!(bench.first_ts_ms >= crate::data::ingest::PINNED_SPLIT_BOUNDS.0);
        assert!(bench.last_ts_ms < crate::data::ingest::PINNED_SPLIT_BOUNDS.1);
        assert!(bench.breadth.min >= 1 && bench.breadth.mean > 1.0);
        for (c, cap) in GROSS_CAPS.iter().enumerate() {
            for (p, policy) in POLICIES.iter().enumerate() {
                let m = bench.metrics(c, p);
                assert!(
                    m.max_gross <= policy.gross_budget(*cap) * (1.0 + 1e-9) + 1e-12,
                    "{} breached the gross constraint at {cap}x",
                    policy.name()
                );
                // CAGR overflows for the perfect-foresight ceiling by construction; the log
                // growth is the number that must always be a number.
                assert!(
                    m.log_growth_per_year.is_finite(),
                    "{} has no log growth at {cap}x",
                    policy.name()
                );
            }
        }

        // Every arm ran the same book: the weights are cost-blind, so the GROSS growth of a
        // point cannot depend on which cost model was charged. This is the invariant that makes
        // the arms a comparison rather than seven unrelated backtests.
        for arm in &bench.arms {
            for (p, row) in arm.points.iter().enumerate() {
                for (b, point) in row.iter().enumerate() {
                    let reference = bench.flat_arm().points[p][b].gross_log_growth_per_year;
                    assert!(
                        (point.gross_log_growth_per_year - reference).abs() <= 1e-9,
                        "[{}] {} at band {} earned gross {} against the flat arm's {reference}",
                        arm.arm.label(),
                        point.policy,
                        point.band_fraction,
                        point.gross_log_growth_per_year,
                    );
                    assert!(
                        point.realized_cost_bps.is_nan() || point.realized_cost_bps >= 0.0,
                        "[{}] paid a negative cost",
                        arm.arm.label()
                    );
                }
            }
        }
        if args.cost_threads > 0 {
            assert!(
                bench.edge.measured,
                "a calibration was measured, so the cost columns must be measured too"
            );
            assert_eq!(
                bench.arms.len(),
                1 + 1 + 2 * IMPACT_K_GRID.len(),
                "flat, impact-free, and one arm per (impact k, fee arm)"
            );
        }
    }
}
