//! Causal, integer-share account replay. Forecasts are residual log returns, not price returns.
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail, ensure};
use chrono::{Datelike, TimeZone, Utc};
use clap::Args;
use serde::{Deserialize, Serialize};

const BAR_MS: i64 = 300_000;
const DAY_MS: f64 = 86_400_000.0;
const YEAR_MS: f64 = 365.25 * DAY_MS;
const EPS: f64 = 1e-8;

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub(super) struct PortfolioConfig {
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
    #[arg(long, default_value_t = 0.005)]
    pub commission_per_share: f64,
    #[arg(long, default_value_t = 1.0)]
    pub commission_min: f64,
    #[arg(long, default_value_t = 0.01)]
    pub commission_cap_fraction: f64,
    /// Explicit conservative sell-fee assumption, USD per million sold; not a broker quote.
    #[arg(long, default_value_t = 20.60)]
    pub sec_fee_per_million: f64,
    #[arg(long, default_value_t = 0.000195)]
    pub taf_per_share: f64,
    #[arg(long, default_value_t = 9.79)]
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
            sec_fee_per_million: 20.60,
            taf_per_share: 0.000195,
            taf_cap: 9.79,
            cat_per_share: 0.000003,
        }
    }
}

impl PortfolioConfig {
    pub(super) fn validate(&self) -> Result<()> {
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
                && self.commission_cap_fraction <= 1.0,
            "weight, participation and commission cap cannot exceed one"
        );
        ensure!(
            self.max_gross <= 1.0,
            "gross leverage above one requires an unsupported financing model"
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
pub(super) struct Asset {
    pub symbol: String,
    pub sector: Option<String>,
    pub shortable: bool,
    pub beta: f64,
    pub risk_group: usize,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(super) struct Forecast {
    pub mean: f64,
    pub std: f64,
    pub expires_ms: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Quote {
    pub asset: usize,
    pub open: f64,
    pub close: f64,
    pub volume: f64,
    pub forecast: Option<Forecast>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Frame {
    pub timestamp_ms: i64,
    pub quotes: Vec<Quote>,
    pub decision: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Tape {
    pub assets: Vec<Asset>,
    pub frames: Vec<Frame>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct AccountPoint {
    pub timestamp_ms: i64,
    pub values: BTreeMap<String, f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct AccountEvaluation {
    pub points: Vec<AccountPoint>,
    pub daily: Vec<AccountPoint>,
    pub monthly: Vec<AccountPoint>,
    pub summary: BTreeMap<String, f64>,
    pub assumptions: Vec<String>,
}

#[derive(Clone, Copy, Default)]
struct Costs {
    commission: f64,
    regulatory: f64,
    spread: f64,
    slippage: f64,
    borrow: f64,
}
impl Costs {
    fn total(self) -> f64 {
        self.commission + self.regulatory + self.spread + self.slippage + self.borrow
    }
    fn add(&mut self, other: Self) {
        self.commission += other.commission;
        self.regulatory += other.regulatory;
        self.spread += other.spread;
        self.slippage += other.slippage;
        self.borrow += other.borrow;
    }
}

fn trade_cost(shares: i64, mid: f64, c: &PortfolioConfig) -> Costs {
    if shares == 0 {
        return Costs::default();
    }
    let quantity = shares.unsigned_abs() as f64;
    let notional = quantity * mid;
    let execution_notional = notional
        * (1.0 + shares.signum() as f64 * (c.spread_bps * 0.5 + c.slippage_bps) / 10_000.0);
    Costs {
        commission: (quantity * c.commission_per_share)
            .max(c.commission_min)
            .min(execution_notional * c.commission_cap_fraction),
        regulatory: if shares < 0 {
            execution_notional * c.sec_fee_per_million / 1_000_000.0
                + (quantity * c.taf_per_share).min(c.taf_cap)
        } else {
            0.0
        } + quantity * c.cat_per_share,
        spread: notional * c.spread_bps * 0.5 / 10_000.0,
        slippage: notional * c.slippage_bps / 10_000.0,
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

fn constrained(e: &Exposure, cash: f64, c: &PortfolioConfig) -> bool {
    e.equity > 0.0
        && e.names <= c.max_names
        && e.max_name <= c.max_weight * e.equity + EPS
        && e.gross() <= c.max_gross * e.equity + EPS
        && (e.long - e.short).abs() <= c.max_net * e.equity + EPS
        && e.beta.abs() <= c.max_beta * e.equity + EPS
        && e.max_group <= c.max_group_weight * e.equity + EPS
        && e.max_sector <= c.max_sector_weight * e.equity + EPS
        && cash + EPS >= (1.0 + c.initial_margin) * e.short
        && e.equity + EPS >= c.maintenance_margin * e.gross()
}

impl Account {
    fn new(n: usize, cash: f64) -> Self {
        Self {
            shares: vec![0; n],
            marks: vec![0.0; n],
            marked_ms: vec![None; n],
            volumes: vec![0.0; n],
            forecasts: vec![None; n],
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
    fn fill(&mut self, asset: usize, delta: i64, open: f64, c: &PortfolioConfig) {
        if delta == 0 {
            return;
        }
        let costs = trade_cost(delta, open, c);
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
    now: i64,
    equity: f64,
    limits: Vec<i64>,
    signals: Vec<Option<Forecast>>,
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
            group_index,
            group_risk: RefCell::new(vec![0.0; groups.len()]),
        }
    }
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
            costs += trade_cost(delta, a.marks[i], c).total();
            // New risk must pay its own future exit, including minimum commission. Existing
            // risk pays only the incremental rebalance cost; do not fabricate a fresh budget.
            let added = if q.signum() == a.shares[i].signum() {
                (q.unsigned_abs() as i64 - a.shares[i].unsigned_abs() as i64).max(0) * q.signum()
            } else {
                q
            };
            costs += trade_cost(-added, a.marks[i], c).total();
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
        let a = self.account;
        let c = self.config;
        let mut cash = a.cash;
        let mut turnover = 0.0;
        for (i, &q) in target.iter().enumerate() {
            let delta = q - a.shares[i];
            if delta.unsigned_abs() > self.limits[i] as u64 {
                return false;
            }
            let increased = q < 0 && (a.shares[i] >= 0 || q < a.shares[i]);
            if increased && !self.assets[i].shortable && !c.allow_assumed_short {
                return false;
            }
            if minimum_trade
                && delta != 0
                && q != 0
                && delta.unsigned_abs() as f64 * a.marks[i] < c.min_trade
            {
                return false;
            }
            turnover += delta.unsigned_abs() as f64 * a.marks[i];
            cash -= delta as f64 * a.marks[i] + trade_cost(delta, a.marks[i], c).total();
        }
        if turnover > self.equity * c.max_turnover + EPS {
            return false;
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
        if balance {
            constrained(&e, cash, c)
        } else {
            // A counterpart may repair net, beta and funding, but not add room to a
            // saturated gross concentration budget when both directions add risk.
            e.equity > 0.0
                && e.names <= c.max_names
                && e.max_name <= c.max_weight * e.equity + EPS
                && e.gross() <= c.max_gross * e.equity + EPS
                && e.max_group <= c.max_group_weight * e.equity + EPS
                && e.max_sector <= c.max_sector_weight * e.equity + EPS
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

fn allocate(a: &Account, assets: &[Asset], now: i64, c: &PortfolioConfig) -> Vec<i64> {
    let e = a.exposure(now, assets, c);
    if e.equity <= 0.0 {
        return vec![0; assets.len()];
    }
    let ctx = AllocationContext::new(a, assets, now, c, e.equity);
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
                return target;
            }
        }
        for q in &mut target {
            *q = 0;
        }
        return target;
    }
    let mut candidates = Vec::<(usize, i64)>::new();
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
        }
    }
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
    target
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
            a.fill(i, delta, open, c);
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
        let ctx = AllocationContext::new(a, assets, now, c, starting_equity);
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
                    cash -= delta as f64 * open + trade_cost(delta, open, c).total();
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
                    a.fill(i, fill_deltas[i], opens[i].expect("available basket"), c);
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
        values.insert("return_fraction".into(), e.equity / previous - 1.0);
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

pub(super) fn simulate(tape: &Tape, config: &PortfolioConfig) -> Result<AccountEvaluation> {
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
    let mut worst_drawdown: f64 = 0.0;
    let mut liquidating = false;
    for (index, frame) in tape.frames.iter().enumerate() {
        a.accrue(frame.timestamp_ms.saturating_sub(accounted_ms), config);
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
            let target = allocate(&a, &tape.assets, close_ms, config);
            if target != a.shares {
                rebalances += 1;
                // Replace, never add to, unfilled orders at a new information set.
                a.pending = Some(make_pending(&a, target, close_ms, config));
            } else {
                no_trade += 1;
                a.pending = None;
            }
        }
        let p = point(
            &a,
            close_ms,
            &tape.assets,
            config,
            &mut peak,
            previous_equity,
        );
        previous_equity = e.equity;
        if let Some(drawdown) = p.values.get("drawdown_fraction") {
            worst_drawdown = worst_drawdown.min(*drawdown);
        }
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
    let assumptions = vec![
        "Forecasts are calibrated cumulative residual log returns, used as small-return dollar alpha, not full price-return predictions. Risk uses supplied causal beta/groups, diagonal forecast variance plus within-group correlated stress; no future covariance fitting.".into(),
        "Discrete cost-aware coordinate allocation is deterministic and heuristic, not a claim of globally optimal integer programming. Candidate pairing is signal-ranked, never outcome-ranked; cash is a valid target. Held positions incur only incremental rebalance costs; new risk must cover entry and expected exit minimum commissions.".into(),
        "Five-minute timestamps denote bar starts. Close-known orders fill only at a later observed bar open; no own-bar open, synthetic missing-bar fills or future-volume sizing. Entry baskets require all counterpart opens. Realized prices, integer sizes, costs and funding constraints are rechecked before increases.".into(),
        format!("Execution assumption: full spread {} bps (half each side), additional slippage {} bps each side; participation <= {} of last completed bar volume. Volume is not known at execution; this is a conservative historical proxy, not an assertion of actual available liquidity.", config.spread_bps, config.slippage_bps, config.participation),
        format!("Static 2026-09-07 fee scenario from https://www.interactivebrokers.com/en/pricing/commissions-stocks.php: fixed ${}/share, ${} minimum, {} notional cap; sells ${}/million SEC plus ${}/share TAF capped at ${}; both sides ${}/share CAT. Configurable scenario rates, not historical authenticated execution metadata. No unobserved taxes, rebates or exchange-specific fees are asserted.", config.commission_per_share, config.commission_min, config.commission_cap_fraction, config.sec_fee_per_million, config.taf_per_share, config.taf_cap, config.cat_per_share),
        format!("Short eligibility uses supplied metadata{}; annual borrow assumption {} bps charged on marked short notional over all elapsed calendar time, including overnight/weekends. No borrow availability/rate history or cash interest is invented.", if config.allow_assumed_short { " OR explicit allow-assumed-short override" } else { " only" }, config.borrow_bps),
        "Longs are cash funded; short proceeds are segregated with configured initial collateral, plus gross maintenance margin. Breaches halt new entries and trigger next-observed-open liquidation, subject to participation. Insolvency is absorbing; negative equity is reported rather than reset.".into(),
        "Name, gross, absolute net, beta, empirical group and supplied-sector caps apply to planned portfolios and entry fills. Missing sector metadata creates no invented sector; sector constraints cover only supplied sectors. Market drift can breach caps before a causal reduction fills; forced risk reductions may exceed ordinary turnover/minimum-trade thresholds.".into(),
        format!("Final scheduled bar cancels entries and attempts liquidation of previously held shares at its open using prior-known participation. Absent or unfillable shares remain inventory. Marks unobserved for at least one bar reserve {} bps/calendar-day of exposure (long reserve capped at its marked value, short uncapped); this is an explicit conservative reserve, not a realized fill.", config.stale_haircut_bps_per_day),
        "Account equity includes cash plus marked long/short inventory less stale reserve; all costs debit cash exactly once. Turnover is one-way absolute traded notional. Daily/monthly returns use successive New York calendar period-end equity, include partial first/last periods and calendar borrow, and have no external cash flows. Cash baseline has zero interest/cost. Daily Sharpe uses sample standard deviation and sqrt(252), omitted when undefined.".into(),
        "Generic margin-account scenario conditional on actual permissions, not broker jurisdiction/PDT eligibility validation or a claim of unsettled cash-account reuse. A $10k US account may be prohibited from this day-trading schedule. Gross leverage above one is rejected; no interest-funded leverage is modeled.".into(),
        "Corpus prices are split/dividend-adjusted. Integer shares are adjusted-price share units; commissions and rounding are scenario approximations, not broker-exact historical fills. No point-in-time corporate-action share ledger or dividend cashflow metadata is supplied, and no separate dividends are credited (which would double count adjustment).".into(),
        format!("Frozen account configuration: {}", serde_json::to_string(config)?),
    ];
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
}
