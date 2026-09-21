//! Causal, integer-share account replay. Forecasts are residual log returns, not price returns.
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use chrono::{Datelike, TimeZone, Utc};
use clap::Args;
use serde::{Deserialize, Serialize};

use crate::torch::train::portfolio_cost::{BarCostModel, CostCalibration};

const BAR_MS: i64 = 300_000;
const DAY_MS: f64 = 86_400_000.0;
const YEAR_MS: f64 = 365.25 * DAY_MS;
const EPS: f64 = 1e-8;

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub struct PortfolioConfig {
    #[arg(long, default_value_t = 10_000.0)]
    pub initial_cash: f64,
    #[arg(long, default_value_t = 8)]
    pub horizon: usize,
    #[arg(long, default_value_t = 4)]
    pub rebalance_bars: usize,
    #[arg(long, default_value_t = 30)]
    pub max_names: usize,
    #[arg(long, default_value_t = 0.05)]
    pub max_weight: f64,
    #[arg(long, default_value_t = 1.0)]
    pub max_gross: f64,
    #[arg(long, default_value_t = 0.02)]
    pub max_net: f64,
    #[arg(long, default_value_t = 10.0)]
    pub risk_aversion: f64,
    /// One-way traded notional / equity at each decision (including reductions).
    #[arg(long, default_value_t = 0.5)]
    pub max_turnover: f64,
    #[arg(long, default_value_t = 25.0)]
    pub min_trade: f64,
    /// Full quoted spread; each execution pays half.
    #[arg(long, default_value_t = 2.0)]
    pub spread_bps: f64,
    #[arg(long, default_value_t = 2.0)]
    pub slippage_bps: f64,
    /// Flat assumed annual short borrow rate; not observed locate availability.
    #[arg(long, default_value_t = 300.0)]
    pub borrow_bps: f64,
    #[arg(long, default_value_t = 0.01)]
    pub participation: f64,
    #[arg(long, default_value_t = false)]
    pub allow_assumed_short: bool,
    #[arg(long, default_value_t = 0.25)]
    pub max_group_weight: f64,
    #[arg(long, default_value_t = 0.25)]
    pub max_sector_weight: f64,
    /// Absolute beta-weighted net exposure / equity.
    #[arg(long, default_value_t = 0.05)]
    pub max_beta: f64,
    /// Additional collateral on top of segregated short proceeds. Longs are cash funded.
    #[arg(long, default_value_t = 0.5)]
    pub initial_margin: f64,
    #[arg(long, default_value_t = 0.3)]
    pub maintenance_margin: f64,
    /// Cancel entries after this many elapsed five-minute intervals, including overnight.
    #[arg(long, default_value_t = 1)]
    pub max_order_age_bars: usize,
    /// Conservative mark reserve on unobserved inventory, per elapsed calendar day.
    #[arg(long, default_value_t = 100.0)]
    pub stale_haircut_bps_per_day: f64,
    /// Fixed-tier per-share commission. Only [`CommissionTier::Fixed`] charges it.
    #[arg(long, default_value_t = 0.005)]
    pub commission_per_share: f64,
    /// Fixed-tier per-order minimum.
    #[arg(long, default_value_t = 1.0)]
    pub commission_min: f64,
    /// Fixed-tier notional cap, all-inclusive of exchange and regulatory fees.
    #[arg(long, default_value_t = 0.01)]
    pub commission_cap_fraction: f64,
    /// Which IBKR Pro US-equity schedule to charge.
    #[arg(long, value_enum, default_value_t = CommissionTier::TieredRemove)]
    pub commission_tier: CommissionTier,
    /// Tiered execution commission, before clearing and venue fees, at our share volume.
    #[arg(long, default_value_t = 0.0035)]
    pub tiered_per_share: f64,
    /// Tiered per-order minimum, kept separate from the fixed tier's dollar minimum.
    #[arg(long, default_value_t = 0.35)]
    pub tiered_min: f64,
    #[arg(long, default_value_t = 0.005)]
    pub tiered_cap_fraction: f64,
    /// NSCC/DTC clearing, charged on both sides of a tiered fill.
    #[arg(long, default_value_t = 0.0002)]
    pub clearing_per_share: f64,
    #[arg(long, default_value_t = 0.0030)]
    pub exchange_per_share_remove: f64,
    #[arg(long, default_value_t = 0.0025)]
    pub exchange_rebate_per_share_add: f64,
    /// SEC Section 31, USD per million of covered sale proceeds.
    #[arg(long, default_value_t = 27.80)]
    pub sec_fee_per_million: f64,
    /// The Section 31 rate was cut to zero on 2025-05-14; sales at or after this instant pay
    /// nothing. A single scenario rate would misprice one half of a multi-year tape.
    #[arg(long, default_value_t = 1_747_195_200_000)]
    pub sec_fee_zero_from_ms: i64,
    #[arg(long, default_value_t = 0.000166)]
    pub taf_per_share: f64,
    #[arg(long, default_value_t = 8.30)]
    pub taf_cap: f64,
    #[arg(long, default_value_t = 0.000003)]
    pub cat_per_share: f64,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            initial_cash: 10_000.0,
            horizon: 8,
            rebalance_bars: 4,
            max_names: 30,
            max_weight: 0.05,
            max_gross: 1.0,
            max_net: 0.02,
            risk_aversion: 10.0,
            max_turnover: 0.5,
            min_trade: 25.0,
            spread_bps: 2.0,
            slippage_bps: 2.0,
            borrow_bps: 300.0,
            participation: 0.01,
            allow_assumed_short: false,
            max_group_weight: 0.25,
            max_sector_weight: 0.25,
            max_beta: 0.05,
            initial_margin: 0.5,
            maintenance_margin: 0.3,
            max_order_age_bars: 1,
            stale_haircut_bps_per_day: 100.0,
            commission_per_share: 0.005,
            commission_min: 1.0,
            commission_cap_fraction: 0.01,
            commission_tier: CommissionTier::TieredRemove,
            tiered_per_share: 0.0035,
            tiered_min: 0.35,
            tiered_cap_fraction: 0.005,
            clearing_per_share: 0.0002,
            exchange_per_share_remove: 0.0030,
            exchange_rebate_per_share_add: 0.0025,
            sec_fee_per_million: 27.80,
            sec_fee_zero_from_ms: 1_747_195_200_000,
            taf_per_share: 0.000166,
            taf_cap: 8.30,
            cat_per_share: 0.000003,
        }
    }
}

impl PortfolioConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.initial_cash.is_finite() && self.initial_cash > 0.0,
            "initial cash must be positive"
        );
        ensure!(
            self.horizon > 0 && self.rebalance_bars > 0 && self.max_names > 0,
            "horizon, rebalance bars and name limit must be positive"
        );
        ensure!(
            self.max_order_age_bars > 0
                && self.max_order_age_bars <= i64::MAX as usize / BAR_MS as usize,
            "invalid order lifetime"
        );
        ensure!(
            self.horizon <= i64::MAX as usize / BAR_MS as usize,
            "invalid forecast horizon"
        );
        for (name, value) in [
            ("max weight", self.max_weight),
            ("max gross", self.max_gross),
            ("max net", self.max_net),
            ("risk aversion", self.risk_aversion),
            ("max turnover", self.max_turnover),
            ("min trade", self.min_trade),
            ("spread bps", self.spread_bps),
            ("slippage bps", self.slippage_bps),
            ("borrow bps", self.borrow_bps),
            ("participation", self.participation),
            ("group weight", self.max_group_weight),
            ("sector weight", self.max_sector_weight),
            ("max beta", self.max_beta),
            ("initial margin", self.initial_margin),
            ("maintenance margin", self.maintenance_margin),
            ("stale haircut", self.stale_haircut_bps_per_day),
            ("commission per share", self.commission_per_share),
            ("commission min", self.commission_min),
            ("commission cap", self.commission_cap_fraction),
            ("tiered per share", self.tiered_per_share),
            ("tiered min", self.tiered_min),
            ("tiered cap", self.tiered_cap_fraction),
            ("clearing per share", self.clearing_per_share),
            ("exchange per share", self.exchange_per_share_remove),
            (
                "exchange rebate per share",
                self.exchange_rebate_per_share_add,
            ),
            ("SEC fee", self.sec_fee_per_million),
            ("TAF fee", self.taf_per_share),
            ("TAF cap", self.taf_cap),
            ("CAT fee", self.cat_per_share),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{name} must be finite and nonnegative"
            );
        }
        ensure!(
            self.max_weight <= 1.0
                && self.participation <= 1.0
                && self.commission_cap_fraction <= 1.0
                && self.tiered_cap_fraction <= 1.0,
            "weight, participation and commission caps cannot exceed one"
        );
        // Reg-T permits 4:1 intraday on a margin account. Between 2:1 and 4:1 the book is
        // intraday-only financing, which `simulate` records as an explicit assumption rather
        // than quietly asserting that overnight portfolio margin exists.
        ensure!(
            self.max_gross <= 4.0,
            "gross leverage above four exceeds the Reg-T intraday limit"
        );
        ensure!(
            self.max_gross >= self.max_net,
            "gross cap below the net cap is unsatisfiable"
        );
        ensure!(
            self.maintenance_margin <= self.initial_margin,
            "maintenance margin exceeds initial margin"
        );
        ensure!(
            (self.spread_bps * 0.5 + self.slippage_bps) < 10_000.0,
            "execution friction must be less than price"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub symbol: String,
    /// Panel symbol id, for the measured per-symbol cost lookup.
    pub symbol_index: u32,
    pub sector: Option<String>,
    pub shortable: bool,
    pub beta: f64,
    pub risk_group: usize,
}
/// Expected residual log return over the holding period and its standard deviation, both in
/// RAW return units: a fraction of price, never a multiple of the name's own sigma.
///
/// [`AllocationContext::utility`] is a DOLLAR objective. `value_usd * mean` is dollar alpha,
/// `(value_usd * std)^2` is dollar variance, and both are differenced against a dollar
/// execution cost. Sigma-unit quantities are therefore not admissible here: dividing by a
/// five-minute sigma near 2.5e-3 inflates alpha by 4e2 and the risk penalty by 1.6e5, so the
/// penalty outruns the alpha by two and a half orders of magnitude and the objective
/// liquidates almost the whole cross-section. `simulate` refuses forecasts too large to be
/// log returns rather than trusting the comment.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Forecast {
    pub mean: f64,
    pub std: f64,
    pub expires_ms: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quote {
    pub asset: usize,
    pub open: f64,
    pub close: f64,
    pub volume: f64,
    pub forecast: Option<Forecast>,
    /// Externally supplied signed target weight of equity. Present means an outside policy
    /// has already chosen the cross-section and `allocate` only enforces feasibility.
    pub target: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub timestamp_ms: i64,
    pub quotes: Vec<Quote>,
    pub decision: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tape {
    pub assets: Vec<Asset>,
    pub frames: Vec<Frame>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountPoint {
    pub timestamp_ms: i64,
    pub values: BTreeMap<String, f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountEvaluation {
    /// One entry per SIMULATED bar, in tape order, each carrying `return`: the equity-relative
    /// net return realized over THAT bar. The product of `1 + return` over the whole series is
    /// the run's total net return, and that reconciliation is the series' whole contract.
    ///
    /// The cadence is the TAPE'S, not a regular session's. The tape retains every observed
    /// mark, extended hours included, so a US-equity year is roughly 48 000 of these bars
    /// rather than the 19 656 a 09:30-16:00 session would imply. Anything annualizing this
    /// series must therefore take its bars-per-year from [`AccountEvaluation::span_years`] and
    /// the series length, never from a regular-session constant: that substitution is exactly
    /// how a 9.6% year gets reported as 3.8%.
    pub points: Vec<AccountPoint>,
    pub daily: Vec<AccountPoint>,
    pub monthly: Vec<AccountPoint>,
    pub summary: BTreeMap<String, f64>,
    pub assumptions: Vec<String>,
}

impl AccountEvaluation {
    /// Wall-clock extent of the simulated window in years, from the points' own timestamps.
    ///
    /// The one measurement every annualization on this evaluation derives from, so the
    /// headline table and the report panels cannot disagree about how long the window was.
    /// NaN when the window has no extent, which no annualization can use.
    pub fn span_years(&self) -> f64 {
        match (self.points.first(), self.points.last()) {
            (Some(first), Some(last)) if last.timestamp_ms > first.timestamp_ms => {
                (last.timestamp_ms - first.timestamp_ms) as f64 / YEAR_MS
            }
            _ => f64::NAN,
        }
    }
}

/// Which IBKR Pro US-equity commission schedule a fill is charged under.
///
/// The tiered schedule unbundles clearing and venue fees, so the same order costs a different
/// amount depending on whether it removed or added liquidity. Both are modelled because the
/// difference, 55 mils per share, is the same order of magnitude as the whole measured edge
/// at our horizon and therefore decides whether the book is tradeable at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[value(rename_all = "kebab-case")]
pub enum CommissionTier {
    TieredRemove,
    TieredAdd,
    Fixed,
}

/// Where the spread and impact charged on a fill come from.
///
/// `Flat` is the scenario assumption the existing account evaluation reproduces bit for bit.
/// `Measured` charges every symbol its own Roll half-spread and square-root impact out of a
/// calibration measured from the same bars, which is the only form in which a capacity claim
/// at a given AUM means anything. The calibration sits behind an [`Arc`] because one AUM
/// sweep replays the same tape at several account sizes.
///
/// Unlike the rest of the configuration this is deliberately not serialisable: the
/// calibration is a measured artefact of the bar corpus, not a frozen knob, and it is
/// reproduced by re-measuring rather than by being written into a report title.
#[derive(Clone, Debug)]
pub enum CostSource {
    Flat,
    Measured(Arc<CostCalibration>),
}

#[derive(Clone, Copy, Default)]
struct Costs {
    commission: f64,
    regulatory: f64,
    spread: f64,
    slippage: f64,
    impact: f64,
    borrow: f64,
}
impl Costs {
    fn total(self) -> f64 {
        self.commission + self.regulatory + self.spread + self.slippage + self.impact + self.borrow
    }
    fn add(&mut self, other: Self) {
        self.commission += other.commission;
        self.regulatory += other.regulatory;
        self.spread += other.spread;
        self.slippage += other.slippage;
        self.impact += other.impact;
        self.borrow += other.borrow;
    }
}

/// Cost of one fill of `shares` at `mid`, under the configured commission tier and the
/// supplied spread/impact source.
///
/// The per-ORDER quantities - the commission minimum and the TAF cap - are charged per FILL
/// here, as they were before. An order that filled in several pieces would pay the minimum
/// several times, which can only overstate cost; the simulator's orders fill at most once per
/// bar, so the overstatement is bounded by the number of bars an entry survives.
fn trade_cost(
    shares: i64,
    mid: f64,
    c: &PortfolioConfig,
    source: &CostSource,
    symbol_index: u32,
    ts_ms: i64,
) -> Costs {
    if shares == 0 {
        return Costs::default();
    }
    let quantity = shares.unsigned_abs() as f64;
    let notional = quantity * mid;
    let (spread, slippage, impact, friction_bps) = match source {
        CostSource::Flat => (
            notional * c.spread_bps * 0.5 / 10_000.0,
            notional * c.slippage_bps / 10_000.0,
            0.0,
            c.spread_bps * 0.5 + c.slippage_bps,
        ),
        CostSource::Measured(calibration) => {
            let resolved = BarCostModel::new(Arc::clone(calibration)).resolve(symbol_index, ts_ms);
            let half_spread_bps = if resolved.half_spread_bps.is_finite() {
                resolved.half_spread_bps
            } else {
                c.spread_bps * 0.5
            };
            // An unmeasurable dollar ADV is not infinite capacity: price the fill as a whole
            // day's volume. An unmeasurable impact coefficient falls back to the flat
            // slippage assumption rather than propagating NaN into the account.
            let participation = if resolved.adv_usd.is_finite() && resolved.adv_usd > 0.0 {
                notional / resolved.adv_usd
            } else {
                1.0
            };
            let impact_bps = match resolved.impact_bps(participation) {
                bps if bps.is_finite() => bps,
                _ => c.slippage_bps,
            };
            (
                notional * half_spread_bps / 10_000.0,
                0.0,
                notional * impact_bps / 10_000.0,
                half_spread_bps + impact_bps,
            )
        }
    };
    let execution_notional = notional * (1.0 + shares.signum() as f64 * friction_bps / 10_000.0);
    let commission = match c.commission_tier {
        CommissionTier::Fixed => (quantity * c.commission_per_share)
            .max(c.commission_min)
            .min(execution_notional * c.commission_cap_fraction),
        CommissionTier::TieredRemove => (quantity
            * (c.tiered_per_share + c.clearing_per_share + c.exchange_per_share_remove))
            .max(c.tiered_min)
            .min(execution_notional * c.tiered_cap_fraction),
        // A maker rebate can exceed execution plus clearing; the broker keeps the difference
        // rather than paying us, so the schedule is floored before the order minimum applies.
        CommissionTier::TieredAdd => (quantity
            * (c.tiered_per_share + c.clearing_per_share - c.exchange_rebate_per_share_add))
            .max(0.0)
            .max(c.tiered_min)
            .min(execution_notional * c.tiered_cap_fraction),
    };
    Costs {
        commission,
        regulatory: if shares < 0 {
            let sec = if ts_ms >= c.sec_fee_zero_from_ms {
                0.0
            } else {
                execution_notional * c.sec_fee_per_million / 1_000_000.0
            };
            sec + (quantity * c.taf_per_share).min(c.taf_cap)
        } else {
            0.0
        } + quantity * c.cat_per_share,
        spread,
        slippage,
        impact,
        borrow: 0.0,
    }
}

#[derive(Clone)]
struct Pending {
    target: Vec<i64>,
    made_ms: i64,
    volume_limit: Vec<i64>,
}
struct Account {
    shares: Vec<i64>,
    marks: Vec<f64>,
    marked_ms: Vec<Option<i64>>,
    volumes: Vec<f64>,
    forecasts: Vec<Option<(Forecast, i64)>>,
    /// Externally supplied target weight per name, with the instant it was supplied. Only a
    /// weight supplied at the decision's own instant is acted on; a stale one is an exit.
    targets: Vec<Option<(f64, i64)>>,
    cash: f64,
    costs: Costs,
    turnover: f64,
    fills: usize,
    rejected: usize,
    margin_breaches: usize,
    insolvent: bool,
    pending: Option<Pending>,
}

struct Exposure {
    equity: f64,
    long: f64,
    short: f64,
    beta: f64,
    reserve: f64,
    stale_gross: f64,
    stale_count: usize,
    names: usize,
    max_name: f64,
    max_group: f64,
    max_sector: f64,
}
impl Exposure {
    fn gross(&self) -> f64 {
        self.long + self.short
    }
}

fn exposure(
    shares: &[i64],
    marks: &[f64],
    cash: f64,
    marked_ms: &[Option<i64>],
    now: i64,
    assets: &[Asset],
    c: &PortfolioConfig,
) -> Exposure {
    let mut e = Exposure {
        equity: cash,
        long: 0.0,
        short: 0.0,
        beta: 0.0,
        reserve: 0.0,
        stale_gross: 0.0,
        stale_count: 0,
        names: 0,
        max_name: 0.0,
        max_group: 0.0,
        max_sector: 0.0,
    };
    let mut groups = BTreeMap::<usize, f64>::new();
    let mut sectors = BTreeMap::<&str, f64>::new();
    for (i, &q) in shares.iter().enumerate() {
        if q == 0 {
            continue;
        }
        let value = q as f64 * marks[i];
        let gross = value.abs();
        e.names += 1;
        e.equity += value;
        if q > 0 {
            e.long += gross;
        } else {
            e.short += gross;
        }
        e.beta += value * assets[i].beta;
        e.max_name = e.max_name.max(gross);
        *groups.entry(assets[i].risk_group).or_default() += gross;
        if let Some(sector) = assets[i].sector.as_deref() {
            *sectors.entry(sector).or_default() += gross;
        }
        let age = marked_ms[i].map_or(0, |t| now.saturating_sub(t));
        if age >= BAR_MS {
            e.stale_count += 1;
            e.stale_gross += gross;
            let haircut = age as f64 / DAY_MS * c.stale_haircut_bps_per_day / 10_000.0;
            e.reserve += gross * if q > 0 { haircut.min(1.0) } else { haircut };
        }
    }
    e.max_group = groups.values().copied().fold(0.0, f64::max);
    e.max_sector = sectors.values().copied().fold(0.0, f64::max);
    e.equity -= e.reserve;
    e
}

/// The first constraint a proposed book violates, in the order [`constrained`] tests them.
///
/// Named so the ramp can say WHY it could not carry the whole supplied cross-section. A
/// count of names lost between two stages is only half a diagnosis; without the binding
/// constraint the reader cannot tell a concentration budget from a financing wall.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Participation,
    Shortable,
    MinTrade,
    Turnover,
    Equity,
    Names,
    Weight,
    Gross,
    Net,
    Beta,
    Group,
    Sector,
    Funding,
    Maintenance,
}

impl Bound {
    /// The `_count` summary key this bound is censused under.
    fn key(self) -> &'static str {
        match self {
            Self::Participation => "book_ramp_bound_participation_count",
            Self::Shortable => "book_ramp_bound_shortable_count",
            Self::MinTrade => "book_ramp_bound_min_trade_count",
            Self::Turnover => "book_ramp_bound_turnover_count",
            Self::Equity => "book_ramp_bound_equity_count",
            Self::Names => "book_ramp_bound_names_count",
            Self::Weight => "book_ramp_bound_weight_count",
            Self::Gross => "book_ramp_bound_gross_count",
            Self::Net => "book_ramp_bound_net_count",
            Self::Beta => "book_ramp_bound_beta_count",
            Self::Group => "book_ramp_bound_group_count",
            Self::Sector => "book_ramp_bound_sector_count",
            Self::Funding => "book_ramp_bound_funding_count",
            Self::Maintenance => "book_ramp_bound_maintenance_count",
        }
    }
    const ALL: [Self; 14] = [
        Self::Participation,
        Self::Shortable,
        Self::MinTrade,
        Self::Turnover,
        Self::Equity,
        Self::Names,
        Self::Weight,
        Self::Gross,
        Self::Net,
        Self::Beta,
        Self::Group,
        Self::Sector,
        Self::Funding,
        Self::Maintenance,
    ];
}

fn bound(e: &Exposure, cash: f64, c: &PortfolioConfig) -> Option<Bound> {
    if e.equity <= 0.0 {
        Some(Bound::Equity)
    } else if e.names > c.max_names {
        Some(Bound::Names)
    } else if e.max_name > c.max_weight * e.equity + EPS {
        Some(Bound::Weight)
    } else if e.gross() > c.max_gross * e.equity + EPS {
        Some(Bound::Gross)
    } else if (e.long - e.short).abs() > c.max_net * e.equity + EPS {
        Some(Bound::Net)
    } else if e.beta.abs() > c.max_beta * e.equity + EPS {
        Some(Bound::Beta)
    } else if e.max_group > c.max_group_weight * e.equity + EPS {
        Some(Bound::Group)
    } else if e.max_sector > c.max_sector_weight * e.equity + EPS {
        Some(Bound::Sector)
    } else if cash + EPS < (1.0 + c.initial_margin) * e.short {
        Some(Bound::Funding)
    } else if e.equity + EPS < c.maintenance_margin * e.gross() {
        Some(Bound::Maintenance)
    } else {
        None
    }
}

fn constrained(e: &Exposure, cash: f64, c: &PortfolioConfig) -> bool {
    bound(e, cash, c).is_none()
}

impl Account {
    fn new(n: usize, cash: f64) -> Self {
        Self {
            shares: vec![0; n],
            marks: vec![0.0; n],
            marked_ms: vec![None; n],
            volumes: vec![0.0; n],
            forecasts: vec![None; n],
            targets: vec![None; n],
            cash,
            costs: Costs::default(),
            turnover: 0.0,
            fills: 0,
            rejected: 0,
            margin_breaches: 0,
            insolvent: false,
            pending: None,
        }
    }
    fn exposure(&self, now: i64, assets: &[Asset], c: &PortfolioConfig) -> Exposure {
        exposure(
            &self.shares,
            &self.marks,
            self.cash,
            &self.marked_ms,
            now,
            assets,
            c,
        )
    }
    fn accrue(&mut self, elapsed: i64, c: &PortfolioConfig) {
        if elapsed <= 0 {
            return;
        }
        let short: f64 = self
            .shares
            .iter()
            .zip(&self.marks)
            .filter(|(q, _)| **q < 0)
            .map(|(q, p)| -(*q as f64) * p)
            .sum();
        let fee = short * c.borrow_bps / 10_000.0 * elapsed as f64 / YEAR_MS;
        self.cash -= fee;
        self.costs.borrow += fee;
    }
    /// Applies an already priced fill, so the caller owns the cost inputs - symbol, timestamp
    /// and cost source - rather than this function growing a parameter for each of them.
    fn fill(&mut self, asset: usize, delta: i64, open: f64, costs: Costs) {
        if delta == 0 {
            return;
        }
        self.cash -= delta as f64 * open + costs.total();
        self.shares[asset] += delta;
        self.costs.add(costs);
        self.turnover += delta.unsigned_abs() as f64 * open;
        self.fills += 1;
    }
}

fn share_limit(value: f64) -> i64 {
    // Keep additions/subtractions exact and safely inside the signed integer range.
    value.floor().max(0.0).min((i64::MAX / 4) as f64) as i64
}

fn live_forecast(a: &Account, i: usize, now: i64) -> Option<Forecast> {
    a.forecasts[i].map(|(f, timestamp)| {
        let duration = (f.expires_ms - timestamp) as f64;
        Forecast {
            mean: f.mean * ((f.expires_ms - now).max(0) as f64 / duration),
            // Expiry removes alpha, not the learned risk of already-held inventory.
            std: f.std,
            expires_ms: f.expires_ms,
        }
    })
}

struct AllocationContext<'a> {
    account: &'a Account,
    assets: &'a [Asset],
    config: &'a PortfolioConfig,
    source: &'a CostSource,
    now: i64,
    equity: f64,
    limits: Vec<i64>,
    signals: Vec<Option<Forecast>>,
    /// Supplied weights live at this instant, `None` where the tape supplied none.
    targets: Vec<Option<f64>>,
    group_index: Vec<usize>,
    group_risk: RefCell<Vec<f64>>,
}
impl<'a> AllocationContext<'a> {
    fn new(
        a: &'a Account,
        assets: &'a [Asset],
        now: i64,
        c: &'a PortfolioConfig,
        equity: f64,
        source: &'a CostSource,
    ) -> Self {
        let mut groups = BTreeMap::new();
        let group_index = assets
            .iter()
            .map(|asset| {
                let next = groups.len();
                *groups.entry(asset.risk_group).or_insert(next)
            })
            .collect();
        Self {
            account: a,
            assets,
            config: c,
            source,
            now,
            equity,
            limits: a
                .volumes
                .iter()
                .map(|v| share_limit(v * c.participation))
                .collect(),
            signals: (0..assets.len())
                .map(|i| live_forecast(a, i, now))
                .collect(),
            targets: a
                .targets
                .iter()
                .map(|t| t.and_then(|(weight, supplied)| (supplied == now).then_some(weight)))
                .collect(),
            group_index,
            group_risk: RefCell::new(vec![0.0; groups.len()]),
        }
    }
    /// Whether an outside policy chose this decision's cross-section.
    fn supplied(&self) -> bool {
        self.targets.iter().any(Option::is_some)
    }
    /// Shares one supplied weight asks for, clamped to what the name may legally hold.
    ///
    /// A quoted name without a live weight is a weight of zero: a book states its whole
    /// cross-section every decision, so silence is an exit, not a hold.
    fn desired_share(&self, i: usize) -> i64 {
        let a = self.account;
        let Some(weight) = self.targets[i] else {
            return 0;
        };
        if a.marks[i] <= 0.0 {
            return a.shares[i];
        }
        let shares = weight * self.equity / a.marks[i];
        if !shares.is_finite() {
            return 0;
        }
        if shares < 0.0 && !self.assets[i].shortable && !self.config.allow_assumed_short {
            return 0;
        }
        let magnitude = share_limit(shares.abs().round()).min(share_limit(
            self.config.max_weight * self.equity / a.marks[i],
        ));
        if shares < 0.0 {
            -magnitude
        } else {
            magnitude
        }
    }
    /// The supplied vector shrunk `scale` of the way from the held book toward `desired`.
    ///
    /// Three clamps are per-name rather than portfolio-wide, because a portfolio-wide veto
    /// would let one illiquid or one sub-minimum name suppress the entire cross-section: a
    /// trade larger than this bar's participation allowance is truncated to it, a trade too
    /// small to be worth its minimum commission is not placed at all, and a trade smaller
    /// than half a share cannot be placed because shares are integers.
    ///
    /// The returned [`Clamp`] says which one fired. The last two both leave the name exactly
    /// where it was, so without the distinction a small account's cross-section evaporates
    /// with no reason attached - and the rounding clamp fires FIRST, which is why a
    /// minimum-trade census alone reads zero while names are disappearing.
    fn scaled_desired(&self, i: usize, desired: i64, scale: f64) -> (i64, Clamp) {
        let a = self.account;
        let held = a.shares[i];
        let scaled = (desired - held) as f64 * scale;
        let magnitude = share_limit(scaled.abs().round()).min(self.limits[i]);
        let moved = held + if scaled < 0.0 { -magnitude } else { magnitude };
        if moved == held {
            return (
                held,
                if scaled == 0.0 {
                    Clamp::None
                } else if self.limits[i] == 0 {
                    Clamp::Participation
                } else {
                    Clamp::Rounded
                },
            );
        }
        if moved != 0 && (moved - held).unsigned_abs() as f64 * a.marks[i] < self.config.min_trade {
            return (held, Clamp::MinTrade);
        }
        (moved, Clamp::None)
    }
    /// Mean-variance utility of one target book, in DOLLARS.
    ///
    /// Every term is a dollar amount at this instant: `value * mean` is expected dollar P&L
    /// over the forecast's remaining life, `risk_aversion * variance / (2 * equity)` is the
    /// dollar charge for `variance` dollars-squared of forecast variance, and `costs` is the
    /// round trip priced by [`trade_cost`]. That commensurability is the whole reason the
    /// objective can decide whether a name's edge pays for its own execution, and it holds
    /// only while [`Forecast`] carries raw log returns.
    ///
    /// Dividing both sides by equity gives the textbook per-unit-equity form
    /// `sum(w*mu) - (risk_aversion/2)*sum((w*sigma)^2)`, so `risk_aversion` is risk aversion
    /// against RAW return variance over the holding period: an unconstrained single name is
    /// sized at `mu / (risk_aversion * sigma^2)`, which at `risk_aversion = 10`, a 64-bar
    /// residual sigma near 2e-2 and a 4 bp edge asks for about a tenth of equity, i.e. the
    /// per-name and gross caps bind before the penalty does. That is the intended regime.
    fn utility(&self, target: &[i64]) -> f64 {
        let c = self.config;
        let a = self.account;
        let mut alpha = 0.0;
        let mut variance = 0.0;
        let mut costs = 0.0;
        let mut groups = self.group_risk.borrow_mut();
        groups.fill(0.0);
        for (i, &q) in target.iter().enumerate() {
            if q == 0 && a.shares[i] == 0 {
                continue;
            }
            let value = q as f64 * a.marks[i];
            let signal = self.signals[i].unwrap_or(Forecast {
                mean: 0.0,
                std: 0.0,
                expires_ms: self.now,
            });
            alpha += value * signal.mean;
            variance += (value * signal.std).powi(2);
            groups[self.group_index[i]] += value * signal.std;
            let delta = q - a.shares[i];
            costs += trade_cost(
                delta,
                a.marks[i],
                c,
                self.source,
                self.assets[i].symbol_index,
                self.now,
            )
            .total();
            // New risk must pay its own future exit, including minimum commission. Existing
            // risk pays only the incremental rebalance cost; do not fabricate a fresh budget.
            let added = if q.signum() == a.shares[i].signum() {
                (q.unsigned_abs() as i64 - a.shares[i].unsigned_abs() as i64).max(0) * q.signum()
            } else {
                q
            };
            costs += trade_cost(
                -added,
                a.marks[i],
                c,
                self.source,
                self.assets[i].symbol_index,
                self.now,
            )
            .total();
            if q < 0 {
                costs += -value * c.borrow_bps / 10_000.0
                    * (signal.expires_ms - self.now).max(0) as f64
                    / YEAR_MS;
            }
        }
        // Diagonal residual risk plus perfectly correlated within-group stress risk. This
        // is a conservative surrogate, not a fitted future covariance matrix.
        variance += groups.iter().map(|v| v * v).sum::<f64>();
        alpha - c.risk_aversion * variance / (2.0 * self.equity) - costs
    }
    fn feasible(&self, target: &[i64]) -> bool {
        self.feasible_bounds(target, true, true)
    }

    fn feasible_bounds(&self, target: &[i64], minimum_trade: bool, balance: bool) -> bool {
        self.bound(target, minimum_trade, balance).is_none()
    }

    /// The first constraint `target` violates, or `None` when the book is admissible.
    fn bound(&self, target: &[i64], minimum_trade: bool, balance: bool) -> Option<Bound> {
        let a = self.account;
        let c = self.config;
        let mut cash = a.cash;
        let mut turnover = 0.0;
        for (i, &q) in target.iter().enumerate() {
            let delta = q - a.shares[i];
            if delta.unsigned_abs() > self.limits[i] as u64 {
                return Some(Bound::Participation);
            }
            let increased = q < 0 && (a.shares[i] >= 0 || q < a.shares[i]);
            if increased && !self.assets[i].shortable && !c.allow_assumed_short {
                return Some(Bound::Shortable);
            }
            if minimum_trade
                && delta != 0
                && q != 0
                && delta.unsigned_abs() as f64 * a.marks[i] < c.min_trade
            {
                return Some(Bound::MinTrade);
            }
            turnover += delta.unsigned_abs() as f64 * a.marks[i];
            cash -= delta as f64 * a.marks[i]
                + trade_cost(
                    delta,
                    a.marks[i],
                    c,
                    self.source,
                    self.assets[i].symbol_index,
                    self.now,
                )
                .total();
        }
        if turnover > self.equity * c.max_turnover + EPS {
            return Some(Bound::Turnover);
        }
        let e = exposure(
            target,
            &a.marks,
            cash,
            &a.marked_ms,
            self.now,
            self.assets,
            c,
        );
        let violated = bound(&e, cash, c);
        if balance {
            violated
        } else {
            // A counterpart may repair net, beta and funding, but not add room to a
            // saturated gross concentration budget when both directions add risk.
            violated.filter(|b| {
                matches!(
                    b,
                    Bound::Equity
                        | Bound::Names
                        | Bound::Weight
                        | Bound::Gross
                        | Bound::Group
                        | Bound::Sector
                )
            })
        }
    }

    fn sized_coordinate(&self, scratch: &mut [i64], i: usize, desired: i64) -> i64 {
        let current = scratch[i];
        scratch[i] = desired;
        if self.feasible_bounds(scratch, false, true) {
            scratch[i] = current;
            return desired;
        }
        let delta = desired - current;
        let (mut low, mut high) = (0, delta.unsigned_abs() as i64);
        while low < high {
            let middle = low + (high - low + 1) / 2;
            scratch[i] = current + delta.signum() * middle;
            if self.feasible_bounds(scratch, false, true) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        scratch[i] = current;
        current + delta.signum() * low
    }
}

/// Which per-name clamp held a name away from its scaled target.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Clamp {
    None,
    Participation,
    Rounded,
    MinTrade,
}

/// Where one decision's cross-section loses names.
///
/// The counts are one attribution chain, so a 256-name decile book that reaches the account
/// and comes out holding three names is legible as exactly one stage failing rather than as
/// an unexplained empty book. Only the first stage of the chain, the book's own selection
/// before its no-trade band, is invisible from here; the book counts that one and injects it
/// under the same `breadth_*_count` key family so all of them chart together.
#[derive(Clone, Copy, Default)]
struct Breadth {
    /// Names carrying a nonzero supplied weight at this instant: the cross-section the book
    /// handed over, already past its own no-trade band.
    banded: usize,
    /// Names with a nonzero desired share count entering the feasibility search.
    requested: usize,
    /// Names still nonzero after the largest uniformly feasible scaling of that vector, i.e.
    /// before the ascent runs at all.
    ramped: usize,
    /// Names with a nonzero position after the ascent.
    held: usize,
    /// Names the ramp had nonzero and the ascent took back to zero.
    ascent_trimmed: usize,
    /// Requested names whose FULL-scale move would be suppressed by `min_trade` alone: the
    /// capacity floor of a small account, asked as "would the minimum trade bite if this
    /// account tried to place the whole book", independent of how far the ramp actually got.
    min_trade_suppressed: usize,
    /// Names the ramp left untouched at the ACCEPTED scale because a fraction of a share
    /// cannot be placed. This fires BEFORE the minimum-trade floor and is the clamp that
    /// actually empties a small book, so the two are counted separately or the floor reads
    /// zero while the cross-section disappears.
    rounding_deleted: usize,
    /// The constraint that refused the whole supplied cross-section at full scale. `None`
    /// means the account carried every name the book asked for and the ramp cost nothing.
    ramp_bound: Option<Bound>,
    /// Largest fraction of the requested MOVE the ramp could carry. Near zero when the book
    /// is already in place and the remaining deltas are small, not only when it is starved.
    ramp_scale: f64,
    /// Whether an outside cross-section was supplied at all, so the stage means are taken
    /// over one comparable population instead of mixing in forced deleveraging frames.
    supplied: bool,
}

/// One decision's target book, plus the diagnostics the caller cannot recover from it.
struct Allocation {
    target: Vec<i64>,
    /// The turnover budget bound this decision: either the weights asked for more one-way
    /// notional than `max_turnover` allows, or the basket placed consumed almost all of it.
    turnover_capped: bool,
    /// One-way notional the full-scale requested book needed, over equity, measured after the
    /// per-name participation truncation and minimum-trade floor and therefore on exactly the
    /// scale `max_turnover` bounds. A reading well above the cap says the book genuinely asked
    /// for that much rotation; a reading below it with `turnover_capped` set says the placed
    /// basket merely consumed the budget.
    requested_turnover_fraction: f64,
    breadth: Breadth,
}

impl Allocation {
    /// A decision no cross-section was requested at: a forced deleveraging or an insolvent
    /// account. No ascent runs, so the held book is its own ramp.
    fn forced(target: Vec<i64>) -> Self {
        let held = target.iter().filter(|q| **q != 0).count();
        Self {
            target,
            turnover_capped: false,
            requested_turnover_fraction: 0.0,
            breadth: Breadth {
                ramped: held,
                held,
                ..Breadth::default()
            },
        }
    }
}

/// Target share counts for one decision.
///
/// Two branches, selected by whether the tape supplied weights at this instant.
///
/// * SUPPLIED ([`Quote::target`], the cross-sectional book). The outside vector is the sizing
///   authority - the amplitude rule below never runs - so the ascent is seeded with the
///   largest uniform scaling of it every cap admits and the candidate set is built from it.
///   Whether cash is also a candidate depends on whether the tape carries a [`Forecast`]: with
///   one, `utility` has genuine alpha and its cost term is a real no-trade band, so the ascent
///   may trim or drop a name whose edge does not pay for its own execution. Without one alpha
///   is identically zero, `utility` then prefers cash to every position, and a zero candidate
///   would liquidate a book whose edge it cannot see; so in that case no candidate ever points
///   away from the supplied vector and the ascent can only close the remaining gap.
/// * AMPLITUDE (no weights). Unchanged mean-variance sizing from each forecast's own mean and
///   std, with cash as a valid candidate, then the same ascent.
fn allocate(
    a: &Account,
    assets: &[Asset],
    now: i64,
    c: &PortfolioConfig,
    source: &CostSource,
) -> Allocation {
    let e = a.exposure(now, assets, c);
    if e.equity <= 0.0 {
        return Allocation::forced(vec![0; assets.len()]);
    }
    let ctx = AllocationContext::new(a, assets, now, c, e.equity, source);
    let mut target = a.shares.clone();
    // Drift breaches are reduced, never financed with another purchase. The next-open
    // handler repeats feasibility checks against actual fill prices and fees.
    if !ctx.feasible(&target) {
        // Prefer the smallest proportional deleveraging that restores all caps. If
        // liquidity/turnover precludes any feasible basket, queue emergency reductions.
        for step in (0..100).rev() {
            for (i, q) in target.iter_mut().enumerate() {
                *q = a.shares[i].signum()
                    * share_limit(a.shares[i].unsigned_abs() as f64 * step as f64 / 100.0);
            }
            if ctx.feasible(&target) {
                return Allocation::forced(target);
            }
        }
        for q in &mut target {
            *q = 0;
        }
        return Allocation::forced(target);
    }
    let mut candidates = Vec::<(usize, i64)>::new();
    let mut breadth = Breadth {
        banded: ctx
            .targets
            .iter()
            .filter(|t| t.is_some_and(|weight| weight != 0.0))
            .count(),
        ..Breadth::default()
    };
    let budget = e.equity * c.max_turnover;
    let mut requested_turnover = 0.0;
    let one_way = |target: &[i64]| -> f64 {
        target
            .iter()
            .zip(&a.shares)
            .zip(&a.marks)
            .map(|((q, held), price)| (q - held).unsigned_abs() as f64 * price)
            .sum()
    };
    if ctx.supplied() {
        let desired: Vec<i64> = (0..assets.len()).map(|i| ctx.desired_share(i)).collect();
        breadth.supplied = true;
        breadth.requested = desired.iter().filter(|q| **q != 0).count();
        let mut scaled = a.shares.clone();
        // Scale zero is the current book, which is feasible here, so the search always
        // terminates. The largest feasible scale is the most of the supplied cross-section
        // this account's caps, liquidity and financing can actually carry.
        for step in (0..=100).rev() {
            let scale = step as f64 / 100.0;
            let mut floored = 0;
            let mut rounded = 0;
            for i in 0..assets.len() {
                let (moved, clamp) = ctx.scaled_desired(i, desired[i], scale);
                scaled[i] = moved;
                match clamp {
                    Clamp::MinTrade => floored += 1,
                    Clamp::Rounded | Clamp::Participation => rounded += 1,
                    Clamp::None => {}
                }
            }
            if step == 100 {
                requested_turnover = one_way(&scaled);
                breadth.min_trade_suppressed = floored;
                breadth.ramp_bound = ctx.bound(&scaled, true, true);
            }
            if ctx.feasible(&scaled) {
                breadth.ramp_scale = scale;
                breadth.rounding_deleted = rounded;
                target.copy_from_slice(&scaled);
                break;
            }
        }
        let priced = ctx.signals.iter().any(Option::is_some);
        for (i, &q) in desired.iter().enumerate() {
            if q != target[i] {
                candidates.push((i, q));
            }
            if priced && target[i] != 0 {
                candidates.push((i, 0));
            }
        }
    } else {
        // No supplied vector to ramp: the amplitude rule sizes each name against the caps
        // directly, so nothing is uniformly scaled down and the ramp stage is a no-op.
        breadth.ramp_scale = 1.0;
        breadth.supplied = true;
        for i in 0..assets.len() {
            if a.marks[i] <= 0.0 || a.marked_ms[i] != Some(now) {
                continue;
            }
            if a.shares[i] != 0 {
                candidates.push((i, 0));
            }
            let Some(signal) = ctx.signals[i] else {
                continue;
            };
            let sign = if signal.mean > 0.0 {
                1
            } else if signal.mean < 0.0 {
                -1
            } else {
                continue;
            };
            if sign < 0 && !assets[i].shortable && !c.allow_assumed_short {
                continue;
            }
            let risk_weight = if c.risk_aversion > 0.0 {
                signal.mean.abs() / (2.0 * c.risk_aversion * signal.std.powi(2))
            } else {
                c.max_weight
            };
            // Leave room for fees in constraints, rather than allocate up to an infeasible cap.
            let budget_weight = c
                .max_weight
                .min(risk_weight)
                .min(c.max_group_weight)
                .min(c.max_gross)
                .min(c.max_turnover)
                .min(if assets[i].sector.is_some() {
                    c.max_sector_weight
                } else {
                    f64::INFINITY
                });
            let cap = share_limit(e.equity * budget_weight * 0.99 / a.marks[i]);
            let volume_cap = if a.shares[i].signum() == sign {
                a.shares[i].unsigned_abs() as i64 + ctx.limits[i]
            } else {
                ctx.limits[i].saturating_sub(a.shares[i].unsigned_abs() as i64)
            };
            let q = cap.min(volume_cap) * sign;
            if q != 0 {
                candidates.push((i, q));
                breadth.requested += 1;
            }
        }
    }
    breadth.ramped = target.iter().filter(|q| **q != 0).count();
    let ramped = target.clone();
    let mut scratch = target.clone();
    // Discrete coordinate ascent with paired long/short moves lets a tight neutral mandate
    // enter a basket from cash. A deterministic signal-ranked shortlist bounds pair work.
    for _ in 0..assets
        .len()
        .min(c.max_names.saturating_mul(2))
        .saturating_add(8)
    {
        let current = ctx.utility(&target);
        let mut best = current;
        let mut chosen = None;
        let mut longs = Vec::new();
        let mut shorts = Vec::new();
        scratch.copy_from_slice(&target);
        for &(i, q) in &candidates {
            if q == target[i] {
                continue;
            }
            let sized = ctx.sized_coordinate(&mut scratch, i, q);
            scratch[i] = sized;
            let score = ctx.utility(&scratch);
            if score > best + EPS && ctx.feasible(&scratch) {
                best = score;
                chosen = Some(((i, sized), None));
            }
            scratch[i] = q;
            let score = ctx.utility(&scratch);
            // Prefer moves with concentration headroom; retain saturated directions as
            // fallback so a paired sale can fund a same-group replacement.
            let headroom = q.unsigned_abs() < target[i].unsigned_abs()
                || ctx.feasible_bounds(&scratch, false, false);
            if q > target[i] {
                longs.push((headroom, score - current, i, q));
            }
            if q < target[i] {
                shorts.push((headroom, score - current, i, q));
            }
            scratch[i] = target[i];
        }
        let rank = |x: &(bool, f64, usize, i64), y: &(bool, f64, usize, i64)| {
            y.0.cmp(&x.0).then(y.1.total_cmp(&x.1)).then(x.2.cmp(&y.2))
        };
        longs.sort_by(rank);
        shorts.sort_by(rank);
        longs.truncate(12);
        shorts.truncate(12);
        let used_turnover: f64 = target
            .iter()
            .zip(&a.shares)
            .zip(&a.marks)
            .map(|((q, held), price)| (q - held).unsigned_abs() as f64 * price)
            .sum();
        let pair_budget = ((e.equity * c.max_turnover - used_turnover) * 0.5).max(0.0);
        for &(_, _, i, qi) in &longs {
            for &(_, _, j, qj) in &shorts {
                if i == j {
                    continue;
                }
                let dollars = ((qi - target[i]) as f64 * a.marks[i])
                    .min((target[j] - qj) as f64 * a.marks[j])
                    .min(pair_budget);
                let pair_i = target[i] + share_limit(dollars / a.marks[i]);
                let pair_j = target[j] - share_limit(dollars / a.marks[j]);
                scratch[i] = pair_i;
                scratch[j] = pair_j;
                let score = ctx.utility(&scratch);
                if score > best + EPS && ctx.feasible(&scratch) {
                    best = score;
                    chosen = Some(((i, pair_i), Some((j, pair_j))));
                }
                scratch[i] = target[i];
                scratch[j] = target[j];
            }
        }
        let Some(((i, q), second)) = chosen else {
            break;
        };
        target[i] = q;
        if let Some((j, q)) = second {
            target[j] = q;
        }
    }
    let placed = one_way(&target);
    breadth.held = target.iter().filter(|q| **q != 0).count();
    breadth.ascent_trimmed = ramped
        .iter()
        .zip(&target)
        .filter(|(before, after)| **before != 0 && **after == 0)
        .count();
    Allocation {
        target,
        turnover_capped: requested_turnover > budget + EPS
            || (budget > 0.0 && placed >= budget * 0.99 - EPS),
        requested_turnover_fraction: requested_turnover / e.equity,
        breadth,
    }
}

fn reduction(current: i64, target: i64) -> i64 {
    if current == 0 {
        0
    } else if current.signum() != target.signum() {
        -current
    } else if target.unsigned_abs() < current.unsigned_abs() {
        target - current
    } else {
        0
    }
}

fn execute_pending(
    a: &mut Account,
    frame: &Frame,
    assets: &[Asset],
    c: &PortfolioConfig,
    source: &CostSource,
    force: bool,
) {
    let now = frame.timestamp_ms;
    let Some(mut pending) = a.pending.take() else {
        return;
    };
    if now < pending.made_ms {
        a.pending = Some(pending);
        return;
    }
    let expired = now.saturating_sub(pending.made_ms) >= c.max_order_age_bars as i64 * BAR_MS;
    let starting_turnover = a.turnover;
    let starting_equity = a.exposure(now, assets, c).equity;
    let mut opens = vec![None; assets.len()];
    for q in &frame.quotes {
        opens[q.asset] = Some(q.open);
    }
    // Missing bars never fill. Reductions have priority, and can remain pending beyond the
    // entry lifetime; current known participation limits are refreshed at later decisions.
    for i in 0..assets.len() {
        let desired = if force { 0 } else { pending.target[i] };
        let delta = reduction(a.shares[i], desired);
        if let Some(open) = opens[i] {
            let delta = delta.signum() * (delta.unsigned_abs() as i64).min(pending.volume_limit[i]);
            let costs = trade_cost(delta, open, c, source, assets[i].symbol_index, now);
            a.fill(i, delta, open, costs);
            pending.volume_limit[i] -= delta.unsigned_abs() as i64;
        }
        if force || expired {
            if expired && pending.target[i] != a.shares[i] && reduction(a.shares[i], desired) == 0 {
                a.rejected += 1;
            }
            pending.target[i] = if reduction(a.shares[i], desired) != 0 {
                desired
            } else {
                a.shares[i]
            };
        }
    }
    let mut increases = vec![0i64; assets.len()];
    let mut all_available = true;
    if !force && !expired && !a.insolvent {
        for i in 0..assets.len() {
            if reduction(a.shares[i], pending.target[i]) != 0 {
                all_available = false;
                continue;
            }
            let delta = pending.target[i] - a.shares[i];
            if delta != 0 {
                all_available &= opens[i].is_some();
                increases[i] =
                    delta.signum() * (delta.unsigned_abs() as i64).min(pending.volume_limit[i]);
            }
        }
    }
    if all_available && increases.iter().any(|&q| q != 0) {
        let mut proposed = a.shares.clone();
        let mut fill_deltas = vec![0; assets.len()];
        let mut accepted = false;
        let ctx = AllocationContext::new(a, assets, now, c, starting_equity, source);
        let current_utility = ctx.utility(&a.shares);
        // Basket scaling uses only this open's observed prices. Do not leg into an
        // unhedged portfolio if a counterpart is absent or fees make it unfundable.
        for step in (1..=64).rev() {
            let scale = step as f64 / 64.0;
            let mut cash = a.cash;
            let mut turnover = 0.0;
            for i in 0..assets.len() {
                let delta =
                    increases[i].signum() * share_limit(increases[i].unsigned_abs() as f64 * scale);
                fill_deltas[i] = delta;
                proposed[i] = a.shares[i] + delta;
                if delta != 0 {
                    let open = opens[i].expect("available basket");
                    cash -= delta as f64 * open
                        + trade_cost(delta, open, c, source, assets[i].symbol_index, now).total();
                    turnover += delta.unsigned_abs() as f64 * open;
                }
            }
            let e = exposure(&proposed, &a.marks, cash, &a.marked_ms, now, assets, c);
            let funded = constrained(&e, cash, c)
                && turnover + a.turnover - starting_turnover
                    <= starting_equity * c.max_turnover + EPS;
            let minimum = fill_deltas
                .iter()
                .enumerate()
                .all(|(i, q)| *q == 0 || q.unsigned_abs() as f64 * a.marks[i] >= c.min_trade);
            if funded
                && minimum
                && fill_deltas.iter().any(|&q| q != 0)
                && ctx.utility(&proposed) > current_utility + EPS
            {
                accepted = true;
                break;
            }
        }
        drop(ctx);
        if accepted {
            for i in 0..assets.len() {
                if fill_deltas[i] != 0 {
                    let open = opens[i].expect("available basket");
                    let costs =
                        trade_cost(fill_deltas[i], open, c, source, assets[i].symbol_index, now);
                    a.fill(i, fill_deltas[i], open, costs);
                }
            }
        } else {
            a.rejected += increases.iter().filter(|&&q| q != 0).count();
        }
        // A scaled fill is final, not a fresh order for the missing amount next bar.
        for i in 0..assets.len() {
            if increases[i] != 0 {
                pending.target[i] = a.shares[i];
            }
        }
    }
    if !force && pending.target != a.shares {
        a.pending = Some(pending);
    }
}

fn make_pending(a: &Account, target: Vec<i64>, now: i64, c: &PortfolioConfig) -> Pending {
    Pending {
        target,
        made_ms: now,
        volume_limit: a
            .volumes
            .iter()
            .map(|v| share_limit(v * c.participation))
            .collect(),
    }
}

fn point(
    a: &Account,
    now: i64,
    assets: &[Asset],
    c: &PortfolioConfig,
    peak: &mut f64,
    previous: f64,
) -> AccountPoint {
    let e = a.exposure(now, assets, c);
    *peak = peak.max(e.equity);
    let mut values = BTreeMap::new();
    for (key, value) in [
        ("equity_usd", e.equity),
        ("cash_usd", a.cash),
        ("cash_baseline_usd", c.initial_cash),
        ("pnl_usd", e.equity - c.initial_cash),
        ("drawdown_fraction", (e.equity - *peak) / *peak),
        ("gross_usd", e.gross()),
        ("net_usd", e.long - e.short),
        ("long_usd", e.long),
        ("short_usd", e.short),
        (
            "available_cash_usd",
            a.cash - (1.0 + c.initial_margin) * e.short,
        ),
        ("margin_requirement_usd", c.maintenance_margin * e.gross()),
        ("commission_usd", a.costs.commission),
        ("regulatory_usd", a.costs.regulatory),
        ("spread_usd", a.costs.spread),
        ("slippage_usd", a.costs.slippage),
        ("impact_usd", a.costs.impact),
        ("borrow_usd", a.costs.borrow),
        ("costs_usd", a.costs.total()),
        ("turnover_usd", a.turnover),
        ("turnover_fraction", a.turnover / c.initial_cash),
        ("held_count", e.names as f64),
        (
            "pending_count",
            a.pending.as_ref().map_or(0, |p| {
                p.target
                    .iter()
                    .zip(&a.shares)
                    .filter(|(x, y)| x != y)
                    .count()
            }) as f64,
        ),
        ("stale_count", e.stale_count as f64),
        ("stale_gross_usd", e.stale_gross),
        ("stale_reserve_usd", e.reserve),
        ("fill_count", a.fills as f64),
        ("rejected_count", a.rejected as f64),
        ("margin_breach_count", a.margin_breaches as f64),
        ("insolvent_count", usize::from(a.insolvent) as f64),
    ] {
        if value.is_finite() {
            values.insert(key.to_owned(), value);
        }
    }
    if previous > 0.0 {
        let realized = e.equity / previous - 1.0;
        values.insert("return_fraction".into(), realized);
        // The same quantity under the bare key the book evaluation aggregates over arbitrary
        // intervals, so interval P&L never has to be recovered from two equity levels.
        values.insert("return".into(), realized);
    }
    if e.equity > 0.0 {
        for (key, value) in [
            ("gross_fraction", e.gross() / e.equity),
            ("net_fraction", (e.long - e.short) / e.equity),
            ("beta_fraction", e.beta / e.equity),
            ("max_name_fraction", e.max_name / e.equity),
            ("max_group_fraction", e.max_group / e.equity),
            ("max_sector_fraction", e.max_sector / e.equity),
        ] {
            if value.is_finite() {
                values.insert(key.into(), value);
            }
        }
    }
    values.retain(|_, value| value.is_finite());
    AccountPoint {
        timestamp_ms: now,
        values,
    }
}

fn aggregate(points: &[AccountPoint], initial: f64, monthly: bool) -> Vec<AccountPoint> {
    let mut last_by_period = BTreeMap::new();
    for p in points {
        let date = Utc
            .timestamp_millis_opt(p.timestamp_ms)
            .single()
            .expect("validated timestamp")
            .with_timezone(&chrono_tz::America::New_York);
        let key = (
            date.year(),
            date.month(),
            if monthly { 0 } else { date.day() },
        );
        last_by_period.insert(key, p);
    }
    let mut previous = initial;
    let mut costs = 0.0;
    let mut turnover = 0.0;
    last_by_period
        .into_values()
        .map(|p| {
            let equity = p.values["equity_usd"];
            let total_cost = p.values["costs_usd"];
            let total_turnover = p.values["turnover_usd"];
            let mut values = BTreeMap::from([
                ("equity_usd".into(), equity),
                ("pnl_usd".into(), equity - previous),
                ("costs_usd".into(), total_cost - costs),
                ("turnover_usd".into(), total_turnover - turnover),
                ("cash_baseline_usd".into(), initial),
            ]);
            if previous > 0.0 {
                values.insert("return_fraction".into(), equity / previous - 1.0);
            }
            previous = equity;
            costs = total_cost;
            turnover = total_turnover;
            values.retain(|_, value| value.is_finite());
            AccountPoint {
                timestamp_ms: p.timestamp_ms,
                values,
            }
        })
        .collect()
}

pub fn simulate(
    tape: &Tape,
    config: &PortfolioConfig,
    costs: &CostSource,
) -> Result<AccountEvaluation> {
    config.validate()?;
    ensure!(
        !tape.assets.is_empty() && !tape.frames.is_empty(),
        "portfolio tape must contain assets and frames"
    );
    let mut symbols = BTreeSet::new();
    for asset in &tape.assets {
        ensure!(
            !asset.symbol.is_empty() && symbols.insert(&asset.symbol),
            "empty or duplicate asset symbol"
        );
        ensure!(
            asset.beta.is_finite(),
            "nonfinite asset beta for {}",
            asset.symbol
        );
    }
    let mut previous_timestamp: Option<i64> = None;
    for frame in &tape.frames {
        let close = frame
            .timestamp_ms
            .checked_add(BAR_MS)
            .ok_or_else(|| anyhow::anyhow!("timestamp overflow"))?;
        ensure!(
            Utc.timestamp_millis_opt(close).single().is_some(),
            "invalid calendar timestamp"
        );
        if let Some(previous) = previous_timestamp {
            ensure!(
                frame.timestamp_ms >= previous + BAR_MS,
                "frames overlap or are not chronological"
            );
        }
        previous_timestamp = Some(frame.timestamp_ms);
        let mut seen = BTreeSet::new();
        for q in &frame.quotes {
            ensure!(
                q.asset < tape.assets.len() && seen.insert(q.asset),
                "unknown or duplicate quote asset"
            );
            ensure!(
                q.open.is_finite()
                    && q.open > 0.0
                    && q.close.is_finite()
                    && q.close > 0.0
                    && q.volume.is_finite()
                    && q.volume >= 0.0,
                "invalid price or volume"
            );
            if let Some(f) = q.forecast {
                ensure!(
                    f.mean.is_finite() && f.std.is_finite() && f.std > 0.0 && f.expires_ms > close,
                    "invalid forecast or nonfuture expiry"
                );
                // [`Forecast`] is raw log returns. A residual log return over any horizon this
                // account trades is at most a few percent, so a magnitude at or above one is a
                // sigma-unit quantity that was never multiplied back by the name's own sigma -
                // the one mistake that silently inverts the sign of the whole objective, since
                // it inflates alpha by 1/sigma and the risk penalty by 1/sigma squared.
                ensure!(
                    f.mean.abs() < 1.0 && f.std < 1.0,
                    "forecast mean {} dispersion {} is not a raw log return; a sigma-unit \
                     forecast must be multiplied by the name's own sigma before it reaches the \
                     account, or the risk penalty outruns the alpha by 1/sigma",
                    f.mean,
                    f.std
                );
            }
            if let Some(target) = q.target {
                ensure!(
                    target.is_finite() && target.abs() <= config.max_gross,
                    "invalid supplied target weight"
                );
            }
        }
    }
    let mut a = Account::new(tape.assets.len(), config.initial_cash);
    let mut points = Vec::with_capacity(tape.frames.len());
    let mut peak = config.initial_cash;
    let mut previous_equity = config.initial_cash;
    let mut accounted_ms = tape.frames[0].timestamp_ms;
    let mut decisions = 0usize;
    let mut rebalances = 0usize;
    let mut no_trade = 0usize;
    let mut turnover_capped = 0usize;
    let mut breadth_sum = Breadth::default();
    let mut requested_turnover_sum = 0.0;
    let mut ramp_scale_sum = 0.0;
    let mut breadth_decisions = 0usize;
    let mut forced_decisions = 0usize;
    let mut ramp_bounds = [0usize; Bound::ALL.len()];
    let mut gross_sum = 0.0;
    let mut net_sum = 0.0;
    let mut names_sum = 0.0;
    let mut exposure_bars = 0.0;
    let mut worst_drawdown: f64 = 0.0;
    let mut liquidating = false;
    for (index, frame) in tape.frames.iter().enumerate() {
        a.accrue(frame.timestamp_ms.saturating_sub(accounted_ms), config);
        let mut decided: Option<(Breadth, f64)> = None;
        // Only the open is visible to execution; this bar's volume, close and signal enter
        // state AFTER orders have filled. Absent symbols retain their last observed mark.
        for q in &frame.quotes {
            a.marks[q.asset] = q.open;
            a.marked_ms[q.asset] = Some(frame.timestamp_ms);
        }
        let previously_liquidating = liquidating;
        let e = a.exposure(frame.timestamp_ms, &tape.assets, config);
        ensure!(
            e.equity.is_finite()
                && e.gross().is_finite()
                && a.cash.is_finite()
                && a.costs.total().is_finite(),
            "account overflow at open {}",
            frame.timestamp_ms
        );
        let breach =
            e.equity <= 0.0 || e.equity < config.maintenance_margin * e.gross() || a.cash < e.short;
        if breach {
            a.margin_breaches += 1;
            liquidating = true;
        }
        if e.equity <= 0.0 {
            a.insolvent = true;
        }
        let final_frame = index + 1 == tape.frames.len();
        if final_frame || previously_liquidating {
            a.pending = Some(make_pending(
                &a,
                vec![0; tape.assets.len()],
                frame.timestamp_ms,
                config,
            ));
        } else if breach {
            // A newly observed opening gap may reject entry orders, but cannot invent
            // an already-submitted exit at that same auction price.
            if let Some(pending) = &mut a.pending {
                for (i, target) in pending.target.iter_mut().enumerate() {
                    *target = a.shares[i] + reduction(a.shares[i], *target);
                }
            }
        }
        execute_pending(
            &mut a,
            frame,
            &tape.assets,
            config,
            costs,
            final_frame || previously_liquidating,
        );
        a.accrue(BAR_MS, config);
        let close_ms = frame.timestamp_ms + BAR_MS;
        for q in &frame.quotes {
            a.marks[q.asset] = q.close;
            a.marked_ms[q.asset] = Some(close_ms);
            a.volumes[q.asset] = q.volume;
            if let Some(forecast) = q.forecast {
                a.forecasts[q.asset] = Some((forecast, close_ms));
            }
            if let Some(target) = q.target {
                a.targets[q.asset] = Some((target, close_ms));
            }
        }
        accounted_ms = close_ms;
        let e = a.exposure(close_ms, &tape.assets, config);
        ensure!(
            e.equity.is_finite()
                && e.gross().is_finite()
                && a.cash.is_finite()
                && a.costs.total().is_finite(),
            "account overflow at {close_ms}"
        );
        if e.equity <= 0.0 {
            a.insolvent = true;
            liquidating = true;
        }
        if e.equity < config.maintenance_margin * e.gross() || a.cash < e.short {
            a.margin_breaches += 1;
            liquidating = true;
        }
        if liquidating && !final_frame {
            a.pending = Some(make_pending(
                &a,
                vec![0; tape.assets.len()],
                close_ms,
                config,
            ));
        } else if frame.decision && !final_frame && !a.insolvent {
            decisions += 1;
            let allocation = allocate(&a, &tape.assets, close_ms, config, costs);
            turnover_capped += usize::from(allocation.turnover_capped);
            let b = allocation.breadth;
            if b.supplied {
                breadth_decisions += 1;
                breadth_sum.banded += b.banded;
                breadth_sum.requested += b.requested;
                breadth_sum.ramped += b.ramped;
                breadth_sum.held += b.held;
                breadth_sum.ascent_trimmed += b.ascent_trimmed;
                breadth_sum.min_trade_suppressed += b.min_trade_suppressed;
                breadth_sum.rounding_deleted += b.rounding_deleted;
                requested_turnover_sum += allocation.requested_turnover_fraction;
                ramp_scale_sum += b.ramp_scale;
                if let Some(violated) = b.ramp_bound {
                    let slot = Bound::ALL
                        .iter()
                        .position(|candidate| *candidate == violated)
                        .expect("every bound is censused");
                    ramp_bounds[slot] += 1;
                }
            } else {
                forced_decisions += 1;
            }
            decided = Some((b, allocation.requested_turnover_fraction));
            if allocation.target != a.shares {
                rebalances += 1;
                // Replace, never add to, unfilled orders at a new information set.
                a.pending = Some(make_pending(&a, allocation.target, close_ms, config));
            } else {
                no_trade += 1;
                a.pending = None;
            }
        }
        let mut p = point(
            &a,
            close_ms,
            &tape.assets,
            config,
            &mut peak,
            previous_equity,
        );
        // Only decision frames carry these, so the series read as gaps between decisions
        // rather than as a zero book on every intervening bar.
        if let Some((b, requested_turnover)) = decided.filter(|(b, _)| b.supplied) {
            for (key, value) in [
                ("breadth_banded_count", b.banded as f64),
                ("breadth_requested_count", b.requested as f64),
                ("breadth_ramped_count", b.ramped as f64),
                ("breadth_held_count", b.held as f64),
                ("breadth_ascent_trimmed_count", b.ascent_trimmed as f64),
                (
                    "breadth_min_trade_suppressed_count",
                    b.min_trade_suppressed as f64,
                ),
                ("breadth_rounding_deleted_count", b.rounding_deleted as f64),
                ("requested_turnover_fraction", requested_turnover),
                ("ramp_scale_fraction", b.ramp_scale),
            ] {
                p.values.insert(key.into(), value);
            }
            if b.ramped > 0 {
                p.values.insert(
                    "ascent_trim_fraction".into(),
                    b.ascent_trimmed as f64 / b.ramped as f64,
                );
            }
            if b.requested > 0 {
                for (key, value) in [
                    (
                        "min_trade_suppressed_fraction",
                        b.min_trade_suppressed as f64,
                    ),
                    ("rounding_deleted_fraction", b.rounding_deleted as f64),
                ] {
                    p.values.insert(key.into(), value / b.requested as f64);
                }
            }
        }
        previous_equity = e.equity;
        if let Some(drawdown) = p.values.get("drawdown_fraction") {
            worst_drawdown = worst_drawdown.min(*drawdown);
        }
        if let Some(gross) = p.values.get("gross_fraction") {
            gross_sum += gross;
            net_sum += p.values.get("net_fraction").copied().unwrap_or(0.0);
            exposure_bars += 1.0;
        }
        names_sum += p.values.get("held_count").copied().unwrap_or(0.0);
        points.push(p);
    }
    let daily = aggregate(&points, config.initial_cash, false);
    let monthly = aggregate(&points, config.initial_cash, true);
    let mut summary = points.last().expect("nonempty tape").values.clone();
    let last = a.exposure(accounted_ms, &tape.assets, config);
    for (key, value) in [
        ("initial_cash_usd", config.initial_cash),
        (
            "total_return_fraction",
            last.equity / config.initial_cash - 1.0,
        ),
        ("max_drawdown_fraction", worst_drawdown),
        ("remaining_count", last.names as f64),
        ("remaining_gross_usd", last.gross()),
        ("decision_count", decisions as f64),
        ("rebalance_count", rebalances as f64),
        ("no_trade_count", no_trade as f64),
        ("period_count", points.len() as f64),
        ("daily_count", daily.len() as f64),
        ("monthly_count", monthly.len() as f64),
        ("universe_count", tape.assets.len() as f64),
        (
            "shortable_count",
            tape.assets
                .iter()
                .filter(|a| a.shortable || config.allow_assumed_short)
                .count() as f64,
        ),
    ] {
        if value.is_finite() {
            summary.insert(key.into(), value);
        }
    }
    // `spread_fallback` trips only when a symbol's whole pooled calibration span was
    // unmeasurable, never for a single month, so it is time invariant: one resolve per symbol
    // is the entire census of how much of the charged cost is assumed rather than measured.
    // Reported under two names because the reporting layer routes `_count` keys by unit.
    let spread_fallback = match costs {
        CostSource::Flat => 0,
        CostSource::Measured(calibration) => {
            let model = BarCostModel::new(Arc::clone(calibration));
            let at = tape.frames[0].timestamp_ms;
            tape.assets
                .iter()
                .filter(|asset| model.resolve(asset.symbol_index, at).spread_fallback)
                .count()
        }
    };
    let spread_measured = match costs {
        CostSource::Flat => 0,
        CostSource::Measured(_) => tape.assets.len() - spread_fallback,
    };
    let span_ms = (accounted_ms - tape.frames[0].timestamp_ms).max(BAR_MS) as f64;
    let total_costs = a.costs.total();
    for (key, value) in [
        (
            "gross_pnl_usd",
            last.equity - config.initial_cash + total_costs,
        ),
        ("net_pnl_usd", last.equity - config.initial_cash),
        ("commission_usd", a.costs.commission),
        ("regulatory_usd", a.costs.regulatory),
        ("spread_usd", a.costs.spread),
        ("slippage_usd", a.costs.slippage),
        ("impact_usd", a.costs.impact),
        ("borrow_usd", a.costs.borrow),
        ("traded_notional_usd", a.turnover),
        (
            "cost_bps_of_traded_notional",
            if a.turnover > 0.0 {
                total_costs / a.turnover * 10_000.0
            } else {
                0.0
            },
        ),
        (
            "turnover_annualized",
            a.turnover / config.initial_cash * YEAR_MS / span_ms,
        ),
        (
            "mean_gross_exposure",
            if exposure_bars > 0.0 {
                gross_sum / exposure_bars
            } else {
                0.0
            },
        ),
        (
            "mean_net_exposure",
            if exposure_bars > 0.0 {
                net_sum / exposure_bars
            } else {
                0.0
            },
        ),
        ("mean_active_names", names_sum / points.len() as f64),
        ("turnover_capped_count", turnover_capped as f64),
        ("spread_fallback_symbols", spread_fallback as f64),
        ("spread_fallback_count", spread_fallback as f64),
        ("spread_measured_count", spread_measured as f64),
        ("bars_simulated", points.len() as f64),
        ("sessions_simulated", daily.len() as f64),
    ] {
        if value.is_finite() {
            summary.insert(key.into(), value);
        }
    }
    // Breadth attribution, averaged over the decisions an outside cross-section was actually
    // supplied at, so the chain compares stage to stage over one population. Forced
    // deleveraging frames are counted separately instead of diluting the stages with zeros.
    // `book_breadth_selected` is the book's own pre-band count, the one link not visible here.
    // Longs consume cash and shorts credit it, so a dollar-neutral book of side s leaves cash
    // at equity and must still post (1 + initial_margin) * s against the short leg:
    // s <= equity / (1 + initial_margin), i.e. gross <= 2 / (1 + initial_margin). With the
    // default half-margin that ceiling is 1.33, so a 2.0 gross request is unreachable by
    // arithmetic and the ramp is permanently scaling down. Reported beside the realized gross
    // so the gap reads as a financing limit and not as a sizing failure.
    summary.insert(
        "book_funding_gross_ceiling".into(),
        2.0 / (1.0 + config.initial_margin),
    );
    summary.insert("book_forced_decision_count".into(), forced_decisions as f64);
    if breadth_decisions > 0 {
        let per_decision = breadth_decisions as f64;
        for (key, value) in [
            ("book_breadth_banded", breadth_sum.banded as f64),
            ("book_breadth_requested", breadth_sum.requested as f64),
            ("book_breadth_ramped", breadth_sum.ramped as f64),
            ("book_breadth_held", breadth_sum.held as f64),
            (
                "book_breadth_ascent_trimmed",
                breadth_sum.ascent_trimmed as f64,
            ),
            (
                "book_breadth_rounding_deleted",
                breadth_sum.rounding_deleted as f64,
            ),
            ("book_requested_turnover_fraction", requested_turnover_sum),
            ("book_ramp_scale_fraction", ramp_scale_sum),
        ] {
            summary.insert(key.into(), value / per_decision);
        }
        for (slot, violated) in Bound::ALL.into_iter().enumerate() {
            summary.insert(violated.key().into(), ramp_bounds[slot] as f64);
        }
        if breadth_sum.ramped > 0 {
            summary.insert(
                "book_ascent_trim_fraction".into(),
                breadth_sum.ascent_trimmed as f64 / breadth_sum.ramped as f64,
            );
        }
        if breadth_sum.requested > 0 {
            for (key, value) in [
                (
                    "book_min_trade_suppressed_fraction",
                    breadth_sum.min_trade_suppressed as f64,
                ),
                (
                    "book_rounding_deleted_fraction",
                    breadth_sum.rounding_deleted as f64,
                ),
            ] {
                summary.insert(key.into(), value / breadth_sum.requested as f64);
            }
        }
    }
    let returns: Vec<f64> = daily
        .iter()
        .filter_map(|p| p.values.get("return_fraction").copied())
        .collect();
    if returns.len() > 1 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        if variance > 0.0 {
            let sharpe = mean / variance.sqrt() * 252.0_f64.sqrt();
            if sharpe.is_finite() {
                summary.insert("daily_sharpe_ratio".into(), sharpe);
            }
        }
    }
    if !monthly.is_empty() {
        summary.insert(
            "monthly_positive_fraction".into(),
            monthly.iter().filter(|p| p.values["pnl_usd"] > 0.0).count() as f64
                / monthly.len() as f64,
        );
    }
    if summary.values().any(|x| !x.is_finite()) {
        bail!("nonfinite account summary");
    }
    let mut assumptions = vec![
        "Forecasts are calibrated cumulative residual log returns in RAW units - a fraction of price, never a multiple of the name's own sigma - used as small-return dollar alpha, not full price-return predictions. The objective is a dollar quantity (value*mean alpha against risk_aversion*sum((value*std)^2)/(2*equity) and a dollar round-trip cost), so a sigma-unit forecast would inflate alpha by 1/sigma and the risk penalty by 1/sigma^2 and liquidate the cross-section; forecasts at or above unit magnitude are refused rather than sized. risk_aversion is therefore aversion to raw return variance over the holding period: an unconstrained name is sized at mean/(risk_aversion*std^2). Risk uses supplied causal beta/groups, diagonal forecast variance plus within-group correlated stress; no future covariance fitting.".into(),
        "Discrete cost-aware coordinate allocation is deterministic and heuristic, not a claim of globally optimal integer programming. Candidate pairing is signal-ranked, never outcome-ranked; cash is a valid target. Held positions incur only incremental rebalance costs; new risk must cover entry and expected exit minimum commissions.".into(),
        "Five-minute timestamps denote bar starts. Close-known orders fill only at a later observed bar open; no own-bar open, synthetic missing-bar fills or future-volume sizing. Entry baskets require all counterpart opens. Realized prices, integer sizes, costs and funding constraints are rechecked before increases.".into(),
        "Cross-sectional weights supplied on the tape are treated as the alpha model itself: sizing is round(weight * equity / price), clamped per name by the weight cap, shortability and participation, then scaled uniformly to the largest multiple every cap, minimum-trade and financing constraint admits. turnover_capped_count counts decisions whose requested weights needed more one-way notional than max_turnover allows, or whose placed basket consumed at least 99% of that budget. Tapes supplying no weights use the amplitude mean-variance rule unchanged.".into(),
        "Reported cost ratios are definitional, not measured spreads: cost_bps_of_traded_notional divides every charged cost, borrow included, by one-way traded notional, and turnover_annualized divides that notional by initial cash and scales by the tape's calendar span, so a tape covering less than a year extrapolates. mean_gross_exposure, mean_net_exposure and mean_active_names are equal-weighted over simulated bars, including bars holding nothing.".into(),
        "Breadth attribution is reported per decision frame and as a mean over the decisions an outside cross-section was actually supplied at, forced deleveraging frames being counted separately as book_forced_decision_count rather than diluting the stages with zeros, so a cross-section that arrives wide and is held narrow is legible at the stage that narrowed it: book_breadth_banded is the supplied weights the book handed over (already past its own no-trade band), book_breadth_requested the names whose weight survives shortability, marks and integer rounding into a nonzero share count, book_breadth_ramped the names still nonzero after the largest uniformly feasible scaling, and book_breadth_held the names the coordinate ascent leaves open. book_ascent_trim_fraction is the share of ramped names the ascent zeroes because their edge does not pay for their execution, and book_requested_turnover_fraction is the full-scale requested one-way notional over equity, on the same scale as the max_turnover ceiling it is compared against.".into(),
        "The requested-to-ramped loss is attributed rather than left as a count. book_ramp_scale_fraction is the mean largest uniform scaling of the requested MOVE that every cap, liquidity, minimum-trade and financing check admits, so it is small both when the account is starved and when the book is already in place; the book_ramp_bound_*_count census counts decisions whose FULL-scale request was refused, keyed by the first constraint to refuse it in the order participation, shortability, minimum trade, turnover, then equity, name limit, per-name weight, gross, net, beta, empirical group, short funding and maintenance margin. It is a first-violation census, not an attribution of a simultaneously binding set. Two distinct per-name floors are separated because the account is INTEGER-share: book_rounding_deleted_fraction is the share of requested names the ramp leaves untouched at the accepted scale because the scaled move is under half a share, and book_min_trade_suppressed_fraction the share whose FULL-scale move would fall under min_trade. Rounding fires first, so a minimum-trade census alone reads zero while the cross-section is disappearing, and at small account sizes the binding floor is neither: it is the per-ORDER commission minimum inside the objective, which a position of a few hundred dollars cannot cover out of a ten basis point edge.".into(),
        match costs {
            CostSource::Flat => format!("Execution assumption: flat full spread {} bps (half each side), additional slippage {} bps each side; participation <= {} of last completed bar volume. Volume is not known at execution; this is a conservative historical proxy, not an assertion of actual available liquidity.", config.spread_bps, config.slippage_bps, config.participation),
            CostSource::Measured(_) => format!("Execution cost is MEASURED per symbol from the same five-minute bars: Roll half-spread plus square-root impact on the fill's share of that symbol-month's dollar ADV, replacing both the flat spread and the flat slippage assumption. {} of {} universe symbols have no usable Roll estimate anywhere in their span and fall back to the cross-sectional median spread; a symbol with no measurable ADV is priced as if the fill were a whole day's volume. Participation <= {} of last completed bar volume. The impact coefficient is calibrated, not an observed fill.", spread_fallback, tape.assets.len(), config.participation),
        },
        format!("2025 IBKR Pro US-equity fee scenario from https://www.interactivebrokers.com/en/pricing/commissions-stocks.php: {}. Sales pay ${}/million SEC Section 31 before epoch millisecond {} and nothing at or after it, the rate having been cut to zero on 2025-05-14, plus ${}/share FINRA TAF capped at ${}; both sides pay ${}/share CAT. Per-order minimums and the TAF cap are charged per fill, which can only overstate cost. Configurable scenario rates, not authenticated historical execution metadata, and no unobserved taxes or venue-specific fees are asserted.",
            match config.commission_tier {
                CommissionTier::TieredRemove => format!("tiered ${}/share execution, ${}/share clearing and ${}/share liquidity-removing venue fee, ${} order minimum, {} notional cap", config.tiered_per_share, config.clearing_per_share, config.exchange_per_share_remove, config.tiered_min, config.tiered_cap_fraction),
                CommissionTier::TieredAdd => format!("tiered ${}/share execution and ${}/share clearing less a ${}/share liquidity-adding rebate, floored at zero, ${} order minimum, {} notional cap; EVERY fill is assumed to add liquidity, which no order type guarantees and which this tape cannot verify", config.tiered_per_share, config.clearing_per_share, config.exchange_rebate_per_share_add, config.tiered_min, config.tiered_cap_fraction),
                CommissionTier::Fixed => format!("fixed ${}/share all-inclusive of exchange and regulatory fees, ${} order minimum, {} notional cap", config.commission_per_share, config.commission_min, config.commission_cap_fraction),
            },
            config.sec_fee_per_million, config.sec_fee_zero_from_ms, config.taf_per_share, config.taf_cap, config.cat_per_share),
        format!("Short eligibility uses supplied metadata{}; annual borrow assumption {} bps charged on marked short notional over all elapsed calendar time, including overnight/weekends. No borrow availability/rate history or cash interest is invented.", if config.allow_assumed_short { " OR explicit allow-assumed-short override" } else { " only" }, config.borrow_bps),
        "Longs are cash funded; short proceeds are segregated with configured initial collateral, plus gross maintenance margin. Breaches halt new entries and trigger next-observed-open liquidation, subject to participation. Insolvency is absorbing; negative equity is reported rather than reset.".into(),
        "Name, gross, absolute net, beta, empirical group and supplied-sector caps apply to planned portfolios and entry fills. Missing sector metadata creates no invented sector; sector constraints cover only supplied sectors. Market drift can breach caps before a causal reduction fills; forced risk reductions may exceed ordinary turnover/minimum-trade thresholds.".into(),
        format!("Final scheduled bar cancels entries and attempts liquidation of previously held shares at its open using prior-known participation. Absent or unfillable shares remain inventory. Marks unobserved for at least one bar reserve {} bps/calendar-day of exposure (long reserve capped at its marked value, short uncapped); this is an explicit conservative reserve, not a realized fill.", config.stale_haircut_bps_per_day),
        "Account equity includes cash plus marked long/short inventory less stale reserve; all costs debit cash exactly once. Turnover is one-way absolute traded notional. Daily/monthly returns use successive New York calendar period-end equity, include partial first/last periods and calendar borrow, and have no external cash flows. Cash baseline has zero interest/cost. Daily Sharpe uses sample standard deviation and sqrt(252), omitted when undefined.".into(),
        format!("Generic margin-account scenario conditional on actual permissions, not broker jurisdiction/PDT eligibility validation or a claim of unsettled cash-account reuse. A $10k US account may be prohibited from this day-trading schedule. Gross exposure is capped at {} and refused above 4.0, the Reg-T intraday limit; interest on borrowed cash is not modeled.", config.max_gross),
        "Corpus prices are split/dividend-adjusted. Integer shares are adjusted-price share units; commissions and rounding are scenario approximations, not broker-exact historical fills. No point-in-time corporate-action share ledger or dividend cashflow metadata is supplied, and no separate dividends are credited (which would double count adjustment).".into(),
        format!("Frozen account configuration: {}", serde_json::to_string(config)?),
    ];
    if config.max_gross > 2.0 {
        assumptions.push(format!("Gross exposure cap {} exceeds 2.0: Reg-T permits 4:1 intraday but only 2:1 overnight, so carrying this book across a session boundary requires a portfolio-margin (risk-based) account rather than Reg-T. No overnight haircut, portfolio-margin stress requirement or concentration add-on is modeled.", config.max_gross));
    }
    Ok(AccountEvaluation {
        points,
        daily,
        monthly,
        summary,
        assumptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-existing invariant tests are statements about the account, not about a fee
    /// schedule or a cost source, so they keep calling two-argument `simulate` and
    /// three-argument `allocate`: these shadow the real ones with the flat cost source.
    fn simulate(tape: &Tape, config: &PortfolioConfig) -> anyhow::Result<AccountEvaluation> {
        super::simulate(tape, config, &CostSource::Flat)
    }
    fn allocate(a: &Account, assets: &[Asset], now: i64, c: &PortfolioConfig) -> Vec<i64> {
        super::allocate(a, assets, now, c, &CostSource::Flat).target
    }

    fn config() -> PortfolioConfig {
        PortfolioConfig {
            max_weight: 0.45,
            max_gross: 1.0,
            max_net: 1.0,
            max_beta: 1.0,
            max_group_weight: 1.0,
            max_sector_weight: 1.0,
            max_turnover: 2.0,
            participation: 1.0,
            risk_aversion: 1.0,
            ..PortfolioConfig::default()
        }
    }
    fn asset(name: &str, group: usize, shortable: bool) -> Asset {
        Asset {
            symbol: name.into(),
            symbol_index: 0,
            sector: None,
            shortable,
            beta: 1.0,
            risk_group: group,
        }
    }
    #[test]
    fn forecast_and_expected_borrow_use_projected_calendar_expiry() {
        let mut a = Account::new(1, 10_000.0);
        let now = 1_735_833_600_000;
        let expiry = now + 3 * 86_400_000;
        a.forecasts[0] = Some((
            Forecast {
                mean: -0.02,
                std: 0.01,
                expires_ms: expiry,
            },
            now,
        ));
        assert!(live_forecast(&a, 0, now + 86_400_000).unwrap().mean < 0.0);
        assert_eq!(live_forecast(&a, 0, expiry).unwrap().mean, 0.0);
        assert_eq!(live_forecast(&a, 0, expiry).unwrap().std, 0.01);
    }

    fn quote(asset: usize, open: f64, close: f64, mean: Option<f64>) -> Quote {
        Quote {
            asset,
            open,
            close,
            volume: 10_000.0,
            forecast: mean.map(|mean| Forecast {
                mean,
                std: 0.01,
                expires_ms: 0,
            }),
            target: None,
        }
    }
    fn frame(i: i64, mut quotes: Vec<Quote>, decision: bool) -> Frame {
        let timestamp_ms = 1_735_833_600_000 + i * BAR_MS;
        for quote in &mut quotes {
            if let Some(f) = &mut quote.forecast {
                if f.expires_ms == 0 {
                    f.expires_ms = timestamp_ms + 9 * BAR_MS;
                }
            }
        }
        Frame {
            timestamp_ms,
            quotes,
            decision,
        }
    }

    #[test]
    fn orders_do_not_use_signal_bar_open_or_future_volume() {
        let c = config();
        let tape = Tape {
            assets: vec![asset("A", 0, false)],
            frames: vec![
                frame(0, vec![quote(0, 1.0, 100.0, Some(0.1))], true),
                frame(1, vec![quote(0, 110.0, 110.0, None)], false),
                frame(2, vec![quote(0, 110.0, 110.0, None)], false),
            ],
        };
        let result = simulate(&tape, &c).unwrap();
        assert_eq!(result.points[0].values["fill_count"], 0.0);
        assert!(result.points[1].values["fill_count"] > 0.0);
        assert!(result.summary["pnl_usd"] < 0.0); // Cannot capture the signal bar's 100x move.
        let mut no_liquidity = tape.clone();
        no_liquidity.frames[0].quotes[0].volume = 0.0;
        assert_eq!(
            simulate(&no_liquidity, &c).unwrap().summary["fill_count"],
            0.0
        );
    }

    #[test]
    fn missing_next_bar_expires_entry_and_missing_final_bar_preserves_inventory() {
        let c = config();
        let mut tape = Tape {
            assets: vec![asset("A", 0, false)],
            frames: vec![
                frame(0, vec![quote(0, 100.0, 100.0, Some(0.1))], true),
                frame(1, vec![], false),
                frame(2, vec![quote(0, 100.0, 100.0, None)], false),
                frame(3, vec![], false),
            ],
        };
        assert_eq!(simulate(&tape, &c).unwrap().summary["fill_count"], 0.0);
        tape.frames[1].quotes.push(quote(0, 100.0, 100.0, None));
        tape.frames[2].quotes.clear();
        let result = simulate(&tape, &c).unwrap();
        assert_eq!(result.summary["remaining_count"], 1.0);
        assert_eq!(result.summary["fill_count"], 1.0);
        assert!(result.summary["stale_reserve_usd"] > 0.0);
    }

    #[test]
    fn flat_round_trip_loses_exactly_all_execution_costs() {
        let tape = Tape {
            assets: vec![asset("A", 0, false)],
            frames: vec![
                frame(0, vec![quote(0, 100.0, 100.0, Some(0.1))], true),
                frame(1, vec![quote(0, 100.0, 100.0, None)], false),
                frame(2, vec![quote(0, 100.0, 100.0, None)], false),
            ],
        };
        let result = simulate(&tape, &config()).unwrap();
        assert_eq!(result.summary["fill_count"], 2.0);
        assert_eq!(result.summary["remaining_count"], 0.0);
        assert!((result.summary["pnl_usd"] + result.summary["costs_usd"]).abs() < 1e-8);
        assert!(result.summary["regulatory_usd"] > 0.0);
        assert_eq!(result.summary["cash_usd"], result.summary["equity_usd"]);
    }

    #[test]
    fn diversified_neutral_basket_obeys_group_sector_beta_and_cash_limits() {
        let mut c = config();
        c.max_net = 0.02;
        c.max_beta = 0.02;
        c.max_weight = 0.2;
        c.max_group_weight = 0.3;
        c.max_sector_weight = 0.3;
        let mut assets = vec![
            asset("L1", 0, false),
            asset("L2", 0, false),
            asset("S1", 1, true),
            asset("S2", 1, true),
        ];
        assets[0].sector = Some("sector".into());
        assets[1].sector = Some("sector".into());
        let quotes = |signal| {
            vec![
                quote(0, 100.0, 100.0, signal),
                quote(1, 100.0, 100.0, signal),
                quote(2, 100.0, 100.0, signal.map(|x| -x)),
                quote(3, 100.0, 100.0, signal.map(|x| -x)),
            ]
        };
        let tape = Tape {
            assets,
            frames: vec![
                frame(0, quotes(Some(0.1)), true),
                frame(1, quotes(None), false),
                frame(2, quotes(None), false),
            ],
        };
        let result = simulate(&tape, &c).unwrap();
        let p = &result.points[1].values;
        assert!(p["gross_usd"] > 0.0);
        assert!(p["net_fraction"].abs() <= c.max_net);
        assert!(p["beta_fraction"].abs() <= c.max_beta);
        assert!(p["max_group_fraction"] <= c.max_group_weight);
        assert!(p["max_sector_fraction"] <= c.max_sector_weight);
        assert!(p["max_name_fraction"] <= c.max_weight);
        assert!(p["available_cash_usd"] >= 0.0);
    }

    #[test]
    fn same_signal_carries_without_repaying_minimum_commissions() {
        let tape = Tape {
            assets: vec![asset("A", 0, false)],
            frames: (0..5)
                .map(|i| frame(i, vec![quote(0, 100.0, 100.0, Some(0.1))], true))
                .collect(),
        };
        let result = simulate(&tape, &config()).unwrap();
        assert_eq!(result.summary["fill_count"], 2.0);
        assert_eq!(result.points[3].values["fill_count"], 1.0);
    }

    #[test]
    fn zero_and_subcommission_edge_preserve_cash_and_no_unlocated_shorts() {
        for mean in [0.0, 0.00001, -0.1] {
            let tape = Tape {
                assets: vec![asset("A", 0, false)],
                frames: (0..4)
                    .map(|i| frame(i, vec![quote(0, 100.0, 100.0, Some(mean))], true))
                    .collect(),
            };
            let result = simulate(&tape, &config()).unwrap();
            assert_eq!(result.summary["fill_count"], 0.0);
            assert_eq!(result.summary["equity_usd"], config().initial_cash);
            assert_eq!(result.daily[0].values["return_fraction"], 0.0);
        }
    }

    #[test]
    fn positive_long_edge_can_use_net_allowance_without_short_permission() {
        let mut c = config();
        c.max_net = 0.02;
        let tape = Tape {
            assets: vec![asset("A", 0, false)],
            frames: (0..3)
                .map(|i| frame(i, vec![quote(0, 10.0, 10.0, Some(0.1))], i == 0))
                .collect(),
        };
        let result = simulate(&tape, &c).unwrap();
        assert!(result.points[1].values["long_usd"] > 0.0);
        assert!(result.points[1].values["net_fraction"] <= c.max_net);
        assert_eq!(result.points[1].values["short_usd"], 0.0);
    }

    #[test]
    fn opening_gap_margin_call_cannot_liquidate_at_already_observed_open() {
        let tape = Tape {
            assets: vec![asset("S", 0, true)],
            frames: vec![
                frame(0, vec![quote(0, 100.0, 100.0, Some(-0.1))], true),
                frame(1, vec![quote(0, 100.0, 100.0, None)], false),
                frame(2, vec![quote(0, 1_000.0, 1_200.0, None)], false),
                frame(3, vec![quote(0, 1_300.0, 1_300.0, None)], false),
            ],
        };
        let result = simulate(&tape, &config()).unwrap();
        assert_eq!(result.points[2].values["fill_count"], 1.0);
        assert_eq!(result.points[2].values["insolvent_count"], 1.0);
        assert_eq!(result.summary["fill_count"], 2.0);
        assert!(result.summary["equity_usd"] < 0.0);
        assert_eq!(result.summary["remaining_count"], 0.0);
    }

    #[test]
    fn final_liquidation_cannot_use_final_bar_volume() {
        let mut entry_bar = quote(0, 100.0, 100.0, None);
        entry_bar.volume = 1.0;
        let tape = Tape {
            assets: vec![asset("A", 0, false)],
            frames: vec![
                frame(0, vec![quote(0, 100.0, 100.0, Some(0.1))], true),
                frame(1, vec![entry_bar], false),
                frame(2, vec![quote(0, 100.0, 100.0, None)], false),
            ],
        };
        let result = simulate(&tape, &config()).unwrap();
        assert!(result.summary["remaining_gross_usd"] > 0.0);
        assert!(
            (result.points[1].values["gross_usd"] - result.summary["remaining_gross_usd"] - 100.0)
                .abs()
                < 1e-8
        );
    }

    #[test]
    fn beta_headroom_sizes_profitable_integer_order_instead_of_false_cash() {
        let mut assets = vec![asset("HIGH_BETA", 0, false)];
        assets[0].beta = 4.0;
        let tape = Tape {
            assets,
            frames: vec![
                frame(0, vec![quote(0, 50.0, 50.0, Some(0.1))], true),
                frame(1, vec![quote(0, 50.0, 50.0, None)], false),
                frame(2, vec![quote(0, 50.0, 50.0, None)], false),
            ],
        };
        let result = simulate(&tape, &PortfolioConfig::default()).unwrap();
        assert_eq!(result.points[1].values["long_usd"], 100.0);
        assert!(result.points[1].values["beta_fraction"] <= 0.05);
    }

    #[test]
    fn saturated_high_alpha_group_does_not_block_profitable_diversification() {
        let assets: Vec<_> = (0..42)
            .map(|i| asset(&format!("A{i}"), usize::from(i >= 40), true))
            .collect();
        let quotes: Vec<_> = (0..42)
            .map(|i| {
                let mean = if i % 2 == 0 { 1.0 } else { -1.0 } * if i < 40 { 0.1 } else { 0.01 };
                quote(i, 100.0, 100.0, Some(mean))
            })
            .collect();
        let now = frame(0, Vec::new(), false).timestamp_ms + BAR_MS;
        let mut account = Account::new(assets.len(), 10_000.0);
        for q in quotes {
            account.marks[q.asset] = q.close;
            account.marked_ms[q.asset] = Some(now);
            account.volumes[q.asset] = q.volume;
            account.forecasts[q.asset] = q.forecast.map(|mut f| {
                f.expires_ms = now + 8 * BAR_MS;
                (f, now)
            });
        }
        let target = allocate(&account, &assets, now, &PortfolioConfig::default());
        assert!(target[40] > 0 && target[41] < 0);
    }

    #[test]
    fn concentration_limit_permits_profitable_same_group_replacement() {
        let assets = vec![
            asset("OLD", 0, true),
            asset("HEDGE", 0, true),
            asset("NEW", 0, true),
        ];
        let now = frame(0, Vec::new(), false).timestamp_ms + BAR_MS;
        let mut account = Account::new(3, 10_000.0);
        account.shares = vec![4, -4, 0];
        for (i, mean) in [0.005, -0.1, 0.1].into_iter().enumerate() {
            account.marks[i] = 100.0;
            account.marked_ms[i] = Some(now);
            account.volumes[i] = 10_000.0;
            account.forecasts[i] = Some((
                Forecast {
                    mean,
                    std: 0.01,
                    expires_ms: now + 8 * BAR_MS,
                },
                now,
            ));
        }
        let c = PortfolioConfig {
            max_group_weight: 0.081,
            ..PortfolioConfig::default()
        };
        let target = allocate(&account, &assets, now, &c);
        assert_eq!(target, vec![0, -4, 4]);
    }

    #[test]
    fn short_borrow_accrues_across_calendar_gap() {
        let c = config();
        let tape = Tape {
            assets: vec![asset("S", 0, true)],
            frames: vec![
                frame(0, vec![quote(0, 100.0, 100.0, Some(-0.1))], true),
                frame(1, vec![quote(0, 100.0, 100.0, None)], false),
                frame(865, vec![quote(0, 100.0, 100.0, None)], false),
            ],
        };
        let result = simulate(&tape, &c).unwrap();
        let short = result.points[1].values["short_usd"];
        let expected = short * c.borrow_bps / 10_000.0 * (864 * BAR_MS) as f64 / YEAR_MS;
        assert!(short > 0.0);
        assert!((result.summary["borrow_usd"] - expected).abs() < 1e-8);
        assert!((result.summary["pnl_usd"] + result.summary["costs_usd"]).abs() < 1e-8);
    }

    #[test]
    fn tiered_remove_charges_unbundled_per_share_fees_under_minimum_and_cap() {
        let c = PortfolioConfig::default();
        let before_repeal = c.sec_fee_zero_from_ms - 1;
        // 100 * (0.0035 execution + 0.0002 clearing + 0.0030 liquidity removal) = $0.67:
        // above the $0.35 order minimum, far below 0.5% of the $2,500 traded.
        let charged = trade_cost(100, 25.0, &c, &CostSource::Flat, 0, before_repeal);
        assert!((charged.commission - 0.67).abs() < 1e-12);
        // Ten shares is $0.067 of per-share fees, so the order minimum binds instead.
        let minimum = trade_cost(10, 25.0, &c, &CostSource::Flat, 0, before_repeal);
        assert_eq!(minimum.commission, c.tiered_min);
        // One share of a dollar stock: 0.5% of the executed notional is half a cent, which
        // is below even the order minimum, so the cap is what is charged.
        let capped = trade_cost(1, 1.0, &c, &CostSource::Flat, 0, before_repeal);
        let executed = 1.0 + (c.spread_bps * 0.5 + c.slippage_bps) / 10_000.0;
        assert!((capped.commission - executed * c.tiered_cap_fraction).abs() < 1e-12);
        assert!(capped.commission < 0.0051);
    }

    #[test]
    fn adding_liquidity_is_strictly_cheaper_than_removing_it() {
        let remove = PortfolioConfig::default();
        let add = PortfolioConfig {
            commission_tier: CommissionTier::TieredAdd,
            ..PortfolioConfig::default()
        };
        let ts = remove.sec_fee_zero_from_ms - 1;
        let removing = trade_cost(1_000, 25.0, &remove, &CostSource::Flat, 0, ts);
        let adding = trade_cost(1_000, 25.0, &add, &CostSource::Flat, 0, ts);
        assert!(adding.commission < removing.commission);
        assert!(adding.total() < removing.total());
    }

    #[test]
    fn section_31_applies_only_to_sales_before_the_2025_repeal() {
        let c = PortfolioConfig::default();
        let quantity = 1_000.0;
        let taf_and_cat = (quantity * c.taf_per_share).min(c.taf_cap) + quantity * c.cat_per_share;
        let before = trade_cost(
            -1_000,
            50.0,
            &c,
            &CostSource::Flat,
            0,
            c.sec_fee_zero_from_ms - 1,
        );
        let after = trade_cost(
            -1_000,
            50.0,
            &c,
            &CostSource::Flat,
            0,
            c.sec_fee_zero_from_ms,
        );
        assert!(before.regulatory > after.regulatory);
        assert!((after.regulatory - taf_and_cat).abs() < 1e-12);
        for ts in [c.sec_fee_zero_from_ms - 1, c.sec_fee_zero_from_ms] {
            let buy = trade_cost(1_000, 50.0, &c, &CostSource::Flat, 0, ts);
            assert!((buy.regulatory - quantity * c.cat_per_share).abs() < 1e-12);
        }
    }

    #[test]
    fn taf_cap_binds_above_fifty_thousand_shares_sold() {
        let c = PortfolioConfig::default();
        let ts = c.sec_fee_zero_from_ms;
        // $8.30 / $0.000166 is exactly 50,000 shares, so anything larger pays the cap.
        let at_cap = trade_cost(-50_000, 10.0, &c, &CostSource::Flat, 0, ts);
        let above = trade_cost(-60_000, 10.0, &c, &CostSource::Flat, 0, ts);
        assert!((at_cap.regulatory - c.taf_cap - 50_000.0 * c.cat_per_share).abs() < 1e-9);
        assert!((above.regulatory - c.taf_cap - 60_000.0 * c.cat_per_share).abs() < 1e-9);
    }

    #[test]
    fn flat_fixed_tier_reproduces_the_premeasurement_cost_exactly() {
        let c = PortfolioConfig {
            commission_tier: CommissionTier::Fixed,
            ..PortfolioConfig::default()
        };
        for shares in [-137i64, 240] {
            let mid = 43.17;
            let quantity = shares.unsigned_abs() as f64;
            let notional = quantity * mid;
            let execution_notional = notional
                * (1.0 + shares.signum() as f64 * (c.spread_bps * 0.5 + c.slippage_bps) / 10_000.0);
            let costs = trade_cost(
                shares,
                mid,
                &c,
                &CostSource::Flat,
                0,
                c.sec_fee_zero_from_ms - 1,
            );
            assert_eq!(
                costs.commission,
                (quantity * c.commission_per_share)
                    .max(c.commission_min)
                    .min(execution_notional * c.commission_cap_fraction)
            );
            assert_eq!(
                costs.regulatory,
                if shares < 0 {
                    execution_notional * c.sec_fee_per_million / 1_000_000.0
                        + (quantity * c.taf_per_share).min(c.taf_cap)
                } else {
                    0.0
                } + quantity * c.cat_per_share
            );
            assert_eq!(costs.spread, notional * c.spread_bps * 0.5 / 10_000.0);
            assert_eq!(costs.slippage, notional * c.slippage_bps / 10_000.0);
            assert_eq!(costs.impact, 0.0);
        }
    }

    #[test]
    fn supplied_weights_are_reproduced_and_infeasible_ones_are_deleveraged() {
        let assets = vec![asset("L", 0, false), asset("S", 1, true)];
        let now = frame(0, Vec::new(), false).timestamp_ms + BAR_MS;
        let c = PortfolioConfig {
            max_weight: 1.0,
            max_gross: 1.0,
            max_net: 0.02,
            max_beta: 1.0,
            max_group_weight: 1.0,
            max_sector_weight: 1.0,
            max_turnover: 2.0,
            participation: 1.0,
            ..PortfolioConfig::default()
        };
        let book = |long: f64, short: f64| {
            let mut account = Account::new(2, 10_000.0);
            for i in 0..2 {
                account.marks[i] = 100.0;
                account.marked_ms[i] = Some(now);
                account.volumes[i] = 10_000.0;
            }
            account.targets[0] = Some((long, now));
            account.targets[1] = Some((short, now));
            allocate(&account, &assets, now, &c)
        };
        assert_eq!(book(0.4, -0.4), vec![40, -40]);
        let deleveraged = book(0.9, -0.9);
        assert_eq!(deleveraged[0], -deleveraged[1]);
        assert!(deleveraged[0] > 0 && deleveraged[0] < 90);
        assert!(deleveraged[0] as f64 * 2.0 * 100.0 <= c.max_gross * 10_000.0);
    }

    #[test]
    fn leveraged_gross_is_permitted_up_to_the_reg_t_intraday_limit() {
        let mut c = config();
        for gross in [1.0, 2.0, 4.0] {
            c.max_gross = gross;
            c.validate().unwrap();
        }
        c.max_gross = 4.5;
        assert!(c.validate().is_err());
        c.max_gross = 0.5;
        c.max_net = 1.0;
        assert!(c.validate().is_err());
        let tape = Tape {
            assets: vec![asset("A", 0, false)],
            frames: vec![frame(0, vec![quote(0, 100.0, 100.0, None)], false)],
        };
        let levered = PortfolioConfig {
            max_gross: 2.5,
            ..config()
        };
        let margin = |c: &PortfolioConfig| {
            simulate(&tape, c)
                .unwrap()
                .assumptions
                .iter()
                .any(|line| line.contains("portfolio-margin"))
        };
        assert!(margin(&levered));
        assert!(!margin(&config()));
    }

    /// The unit contract between the tape and the objective, not the arithmetic of `utility`.
    ///
    /// A sigma-unit forecast is numerically well formed and passes every finiteness check, so
    /// only a test that states the economics in RAW units and then watches the account act on
    /// them can tell the two apart. The same edge - 40 bp expected over 64 bars against a
    /// round trip under 8 bp - is held when it arrives raw and, once the identical numbers are
    /// divided by the name's 25 bp five-minute sigma, inflates the quadratic penalty by
    /// 1/sigma^2 until the ascent prefers cash: the book empties with no invalid number in it.
    #[test]
    fn sigma_unit_forecast_empties_the_book_and_is_refused_at_the_tape() {
        const SIGMA: f64 = 0.0025;
        let raw = Forecast {
            mean: 0.004,
            std: SIGMA * 64.0_f64.sqrt(),
            expires_ms: 0,
        };
        let sigma_units = Forecast {
            mean: raw.mean / SIGMA,
            std: raw.std / SIGMA,
            expires_ms: 0,
        };
        let assets = vec![asset("A", 0, false)];
        let c = config();
        let now = frame(0, Vec::new(), false).timestamp_ms + BAR_MS;
        let book = |forecast: Forecast| {
            let mut account = Account::new(1, 10_000.0);
            account.marks[0] = 100.0;
            account.marked_ms[0] = Some(now);
            account.volumes[0] = 10_000.0;
            account.targets[0] = Some((0.40, now));
            account.forecasts[0] = Some((
                Forecast {
                    expires_ms: now + 64 * BAR_MS,
                    ..forecast
                },
                now,
            ));
            super::allocate(&account, &assets, now, &c, &CostSource::Flat)
        };
        let taken = book(raw);
        assert_eq!(
            taken.target[0], 40,
            "a 40 bp raw edge against a sub-8 bp round trip must survive the ascent"
        );
        assert_eq!(taken.breadth.banded, 1);
        assert_eq!(taken.breadth.requested, 1);
        assert_eq!(taken.breadth.ramped, 1);
        assert_eq!(taken.breadth.held, 1);
        assert_eq!(taken.breadth.ascent_trimmed, 0);
        assert_eq!(taken.breadth.min_trade_suppressed, 0);
        let emptied = book(sigma_units);
        assert_eq!(
            emptied.target[0], 0,
            "sigma units must be shown to zero the same position, not merely be disliked"
        );
        assert_eq!(emptied.breadth.requested, 1);
        assert_eq!(emptied.breadth.ramped, 1);
        assert_eq!(emptied.breadth.held, 0);
        assert_eq!(emptied.breadth.ascent_trimmed, 1);

        let tape = |forecast: Forecast| Tape {
            assets: assets.clone(),
            frames: vec![
                frame(
                    0,
                    vec![Quote {
                        forecast: Some(forecast),
                        target: Some(0.40),
                        ..quote(0, 100.0, 100.0, None)
                    }],
                    true,
                ),
                frame(1, vec![quote(0, 100.0, 100.0, None)], false),
            ],
        };
        let admitted = simulate(&tape(raw), &c).expect("a raw forecast is admissible");
        for key in [
            "book_breadth_banded",
            "book_breadth_requested",
            "book_breadth_ramped",
            "book_breadth_held",
        ] {
            assert_eq!(admitted.summary[key], 1.0, "{key}");
        }
        assert_eq!(admitted.summary["book_ascent_trim_fraction"], 0.0);
        assert_eq!(admitted.summary["book_min_trade_suppressed_fraction"], 0.0);
        let refused = simulate(&tape(sigma_units), &c)
            .expect_err("a sigma-unit forecast must be refused, not silently sized");
        assert!(
            format!("{refused:#}").contains("not a raw log return"),
            "{refused:#}"
        );
    }

    /// The minimum-trade floor is a capacity limit, so it has to be readable as one.
    ///
    /// $600 of equity over ten names is $24 a name, a dollar under the $25 floor, and every
    /// name is therefore suppressed. Without the count this is indistinguishable from a book
    /// that supplied no cross-section at all.
    #[test]
    fn minimum_trade_floor_is_counted_rather_than_emptying_the_book_silently() {
        let assets: Vec<_> = (0..10).map(|i| asset(&format!("A{i}"), 0, true)).collect();
        let c = PortfolioConfig {
            max_weight: 0.4,
            max_net: 1.0,
            max_beta: 1.0,
            max_group_weight: 1.0,
            max_sector_weight: 1.0,
            max_turnover: 2.0,
            participation: 1.0,
            ..PortfolioConfig::default()
        };
        let now = frame(0, Vec::new(), false).timestamp_ms + BAR_MS;
        let mut account = Account::new(10, 600.0);
        for i in 0..10 {
            // Dollar shares, so integer rounding is never the binding clamp.
            account.marks[i] = 1.0;
            account.marked_ms[i] = Some(now);
            account.volumes[i] = 1_000_000.0;
            account.targets[i] = Some((if i % 2 == 0 { 0.04 } else { -0.04 }, now));
        }
        let allocation = super::allocate(&account, &assets, now, &c, &CostSource::Flat);
        assert_eq!(allocation.breadth.banded, 10);
        assert_eq!(allocation.breadth.requested, 10);
        assert_eq!(allocation.breadth.min_trade_suppressed, 10);
        assert_eq!(allocation.breadth.held, 0);
        assert_eq!(allocation.breadth.ascent_trimmed, 0);
    }

    /// A dollar-neutral book cannot reach a gross cap above `2 / (1 + initial_margin)`.
    ///
    /// Longs consume cash and shorts credit it, so building long = short = s leaves cash at
    /// equity while the short leg still has to post `(1 + initial_margin) * s`. The ceiling is
    /// arithmetic and raising `max_gross` cannot move it. Pinning it is what stops the next
    /// reader who finds gross stuck at two thirds of the cap from blaming the ramp.
    #[test]
    fn dollar_neutral_gross_is_capped_by_short_collateral_not_by_the_gross_cap() {
        const N: usize = 8;
        let assets: Vec<_> = (0..N).map(|i| asset(&format!("A{i}"), i, true)).collect();
        let c = PortfolioConfig {
            initial_cash: 1_000_000.0,
            max_names: N,
            max_weight: 0.3,
            max_gross: 2.0,
            max_net: 0.02,
            max_beta: 0.05,
            max_group_weight: 1.0,
            max_sector_weight: 1.0,
            max_turnover: 2.0,
            participation: 1.0,
            min_trade: 1.0,
            ..PortfolioConfig::default()
        };
        let now = frame(0, Vec::new(), false).timestamp_ms + BAR_MS;
        let mut account = Account::new(N, c.initial_cash);
        for i in 0..N {
            account.marks[i] = 100.0;
            account.marked_ms[i] = Some(now);
            account.volumes[i] = 1_000_000.0;
            account.targets[i] = Some((if i < N / 2 { 0.25 } else { -0.25 }, now));
        }
        let allocation = super::allocate(&account, &assets, now, &c, &CostSource::Flat);
        let gross = allocation
            .target
            .iter()
            .map(|q| q.unsigned_abs() as f64 * 100.0)
            .sum::<f64>()
            / c.initial_cash;
        let ceiling = 2.0 / (1.0 + c.initial_margin);
        assert!(gross <= ceiling + 1e-3, "gross {gross} exceeds {ceiling}");
        assert!(
            gross > ceiling - 0.05,
            "short collateral, not another cap, must be the binding limit: {gross}"
        );
        assert_eq!(allocation.breadth.requested, N);
        assert_eq!(allocation.breadth.held, N);
    }
}
