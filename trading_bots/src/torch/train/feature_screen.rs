//! CPU-first rolling-origin screen for causal information families.
//!
//! This command deliberately fits small streaming linear models rather than a neural model.  Its
//! purpose is to decide which context families justify neural capacity.  Every target comes from
//! `BarCorpus::direct_return_targets`, every fold comes from `rolling_origin_plan`, and no terminal
//! split is representable here.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use shared::bars::PackedBar;
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};

use crate::torch::adjusted_daily::ADJUSTED_DAILY_CONTEXT_FEATURES;
use crate::torch::bar_dist::encode_dof;
use crate::torch::dataset::{
    bar_time_ids, mix64, BarCorpus, RollingOriginFold, TimeRange, BAR_TIME_CARDINALITY,
    DIRECT_RETURN_HORIZONS,
};

use super::portfolio_cost::{BarCostModel, CostCalibration};
use super::pretrain::{load_corpus, CorpusFlags};
use super::pretrain_stats::{equal_count_buckets, mid_ranks, ProductMoments, RidgeFit, RidgeSums};

pub const FEATURE_SCREEN_IC_BASE: &str = "pretrain_feature_screen_ic";
pub const FEATURE_SCREEN_SPREAD_BASE: &str = "pretrain_feature_screen_spread";
pub const FEATURE_SCREEN_CONTROLS_BASE: &str = "pretrain_feature_screen_controls";

const MILLIS_PER_DAY: i64 = 86_400_000;
const BASELINE_FEATURES: usize = 6;
const INNER_TAIL_DIVISOR: usize = 5;
const LAMBDAS: [f64; 6] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0];
const DECILES: usize = 10;

#[derive(Clone, Debug)]
pub struct FeatureScreenArgs {
    pub output: String,
    pub folds: usize,
    pub max_symbols: usize,
    pub max_instants: usize,
    pub cost_threads: usize,
    pub capital_usd: f64,
    pub seed: u64,
    pub corpus: CorpusFlags,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Clock,
    OwnBars,
    MarketResidual,
    CrossSection,
    TrailingLiquidity,
    AdjustedDaily,
    SlowEarningsFundamentals,
    Macro,
    Sector,
    PreciseEarningsEvent,
}

impl Family {
    const ALL: [Self; 10] = [
        Self::Clock,
        Self::OwnBars,
        Self::MarketResidual,
        Self::CrossSection,
        Self::TrailingLiquidity,
        Self::AdjustedDaily,
        Self::SlowEarningsFundamentals,
        Self::Macro,
        Self::Sector,
        Self::PreciseEarningsEvent,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Clock => "clock_baseline",
            Self::OwnBars => "own_ohlcv_rolling",
            Self::MarketResidual => "spy_exact_timestamp_residual",
            Self::CrossSection => "synchronized_cross_section",
            Self::TrailingLiquidity => "strictly_trailing_liquidity_adv_cost",
            Self::AdjustedDaily => "last_fully_completed_adjusted_daily",
            Self::SlowEarningsFundamentals => "slow_earnings_fundamentals_plus_90d_unavailable",
            Self::Macro => "macro_release_vintage_unavailable",
            Self::Sector => "effective_dated_sector_unavailable",
            Self::PreciseEarningsEvent => "exact_earnings_announcement_timestamp_unavailable",
        }
    }

    fn unavailable_prerequisite(self) -> Option<&'static str> {
        match self {
            Self::SlowEarningsFundamentals => Some(
                "existing filing source synthesizes availability as report-date plus 90 calendar days; arm not initialized in this exact-as-of screen",
            ),
            Self::Macro => Some(
                "no release-vintage macro source is initialized for this command; revised/current values are not substituted",
            ),
            Self::Sector => Some(
                "effective-dated industry/sector membership does not exist; current labels forbidden",
            ),
            Self::PreciseEarningsEvent => Some(
                "actual earnings announcement timestamps do not exist; report-date heuristics forbidden",
            ),
            _ => None,
        }
    }

    fn feature_count(self) -> usize {
        match self {
            Self::Clock => 0,
            Self::OwnBars => 17,
            Self::MarketResidual => 5,
            Self::CrossSection => 6,
            Self::TrailingLiquidity => 7,
            Self::AdjustedDaily => ADJUSTED_DAILY_CONTEXT_FEATURES,
            Self::SlowEarningsFundamentals
            | Self::Macro
            | Self::Sector
            | Self::PreciseEarningsEvent => 0,
        }
    }
}

#[derive(Clone, Debug)]
struct Row {
    symbol: u32,
    ts_ms: i64,
    baseline: [f64; BASELINE_FEATURES],
    target: [f64; DIRECT_RETURN_HORIZONS.len()],
    own: Vec<f64>,
    market: Vec<f64>,
    cross: Vec<f64>,
    liquidity: Vec<f64>,
    daily: Vec<f64>,
    market_present: bool,
    cross_present: bool,
    liquidity_present: bool,
    daily_present: bool,
}

impl Row {
    fn family(&self, family: Family) -> &[f64] {
        match family {
            Family::Clock => &[],
            Family::OwnBars => &self.own,
            Family::MarketResidual => &self.market,
            Family::CrossSection => &self.cross,
            Family::TrailingLiquidity => &self.liquidity,
            Family::AdjustedDaily => &self.daily,
            Family::SlowEarningsFundamentals
            | Family::Macro
            | Family::Sector
            | Family::PreciseEarningsEvent => &[],
        }
    }

    fn family_present(&self, family: Family) -> bool {
        match family {
            Family::Clock | Family::OwnBars => true,
            Family::MarketResidual => self.market_present,
            Family::CrossSection => self.cross_present,
            Family::TrailingLiquidity => self.liquidity_present,
            Family::AdjustedDaily => self.daily_present,
            Family::SlowEarningsFundamentals
            | Family::Macro
            | Family::Sector
            | Family::PreciseEarningsEvent => false,
        }
    }
}

#[derive(Clone, Debug)]
struct InstantRows {
    ts_ms: i64,
    rows: Vec<Row>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    Real,
    Masked,
    Shuffled,
}

#[derive(Clone, Debug)]
struct Standardizer {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl Standardizer {
    fn fit(groups: &[InstantRows], family: Family) -> Result<Self> {
        let dimensions = BASELINE_FEATURES + family.feature_count();
        let mut sum = vec![0.0; dimensions];
        let mut square = vec![0.0; dimensions];
        let mut count = 0.0;
        for group in groups {
            for row in &group.rows {
                for (slot, value) in row
                    .baseline
                    .iter()
                    .copied()
                    .chain(row.family(family).iter().copied())
                    .enumerate()
                {
                    ensure!(
                        value.is_finite(),
                        "non-finite feature before standardization"
                    );
                    sum[slot] += value;
                    square[slot] += value * value;
                }
                count += 1.0;
            }
        }
        ensure!(count > 0.0, "cannot standardize an empty fit prefix");
        let mut mean = vec![0.0; dimensions];
        let mut scale = vec![1.0; dimensions];
        for feature in 0..dimensions {
            mean[feature] = sum[feature] / count;
            let variance = (square[feature] / count - mean[feature] * mean[feature]).max(0.0);
            if variance > f64::EPSILON {
                scale[feature] = variance.sqrt();
            }
        }
        Ok(Self { mean, scale })
    }

    fn transform(
        &self,
        baseline: &[f64; BASELINE_FEATURES],
        family: &[f64],
        masked: bool,
    ) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.mean.len());
        for (slot, value) in baseline
            .iter()
            .copied()
            .chain(family.iter().copied())
            .enumerate()
        {
            if masked && slot >= BASELINE_FEATURES {
                // A fit-standardized zero is the mean in raw space, hence exactly zero here.
                out.push(0.0);
            } else {
                out.push((value - self.mean[slot]) / self.scale[slot]);
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
struct FittedModel {
    standardizer: Standardizer,
    fit: RidgeFit,
}

#[derive(Clone, Debug, Default)]
struct FoldMetrics {
    pearson_ic: f64,
    rank_ic: f64,
    gross_spread_bps: f64,
    impact_free_net_bps: f64,
    all_in_net_bps: f64,
    breadth: f64,
    coverage: f64,
    missingness: f64,
    lambda: f64,
}

#[derive(Clone, Debug)]
struct ArmFold {
    family: Family,
    horizon: usize,
    real: FoldMetrics,
    masked: FoldMetrics,
    shuffled: FoldMetrics,
}

fn positive(value: f32) -> Option<f64> {
    let value = f64::from(value);
    (value.is_finite() && value > 0.0).then_some(value)
}

fn dollar_volume(bar: PackedBar) -> f64 {
    let price = positive(bar.vwap).or_else(|| positive(bar.close));
    let volume = f64::from(bar.volume);
    match price {
        Some(price) if volume.is_finite() && volume > 0.0 => price * volume,
        _ => 0.0,
    }
}

fn baseline_features(bar: PackedBar, previous_ts: Option<i64>, res_secs: u32) -> [f64; 6] {
    let ids = bar_time_ids(bar.ts(), previous_ts, res_secs, None);
    let mut out = [0.0; 6];
    for slot in 0..6 {
        out[slot] = ids[slot] as f64 / (BAR_TIME_CARDINALITY[slot] - 1).max(1) as f64;
    }
    out
}

fn own_features(bars: &[PackedBar], index: usize) -> Vec<f64> {
    let current = bars[index];
    let prior = &bars[index.saturating_sub(78)..index];
    let ema_volume = if prior.is_empty() {
        current.volume
    } else {
        prior
            .iter()
            .map(|bar| f64::from(bar.volume).max(0.0))
            .sum::<f64>() as f32
            / prior.len() as f32
    };
    let dof = encode_dof(bars[index - 1].close, &current, ema_volume).to_array();
    let close = positive(current.close).unwrap_or(1.0);
    let rolling_return = |lookback: usize| {
        let anchor = index.saturating_sub(lookback).max(1);
        positive(bars[anchor].close).map_or(0.0, |value| (close / value).ln())
    };
    let rolling_vol = |lookback: usize| {
        let from = index.saturating_sub(lookback).max(1);
        let mut moments = ProductMoments::default();
        for slot in from..=index {
            let Some(a) = positive(bars[slot - 1].close) else {
                continue;
            };
            let Some(b) = positive(bars[slot].close) else {
                continue;
            };
            moments.push((b / a).ln(), (b / a).ln());
        }
        let count = moments.count();
        if count < 2.0 {
            0.0
        } else {
            let mean = moments.mean_x();
            let mut sum = 0.0;
            let mut n = 0.0f64;
            for slot in from..=index {
                if let (Some(a), Some(b)) =
                    (positive(bars[slot - 1].close), positive(bars[slot].close))
                {
                    let value = (b / a).ln();
                    sum += (value - mean) * (value - mean);
                    n += 1.0;
                }
            }
            (sum / (n - 1.0).max(1.0)).sqrt()
        }
    };
    let vwap_deviation = positive(current.vwap).map_or(0.0, |vwap| (close / vwap).ln());
    let mut out: Vec<f64> = dof.into_iter().map(f64::from).collect();
    out.extend([
        rolling_return(1),
        rolling_return(4),
        rolling_return(16),
        rolling_return(78),
        rolling_vol(16),
        rolling_vol(78),
        vwap_deviation,
        dollar_volume(current).max(1.0).ln(),
        f64::from(current.trades).max(0.0).ln_1p(),
        f64::from(current.volume).max(0.0).ln_1p(),
        (prior.len() as f64).ln_1p(),
        1.0,
    ]);
    debug_assert_eq!(out.len(), Family::OwnBars.feature_count());
    out
}

fn trailing_liquidity_features(bars: &[PackedBar], index: usize) -> (Vec<f64>, bool) {
    let decision = bars[index].ts();
    let cutoff = decision.saturating_sub(30 * MILLIS_PER_DAY);
    let start = bars[..index].partition_point(|bar| bar.ts() < cutoff);
    let prior = &bars[start..index]; // strictly before the decision bar
    if prior.len() < 8 {
        return (vec![0.0; Family::TrailingLiquidity.feature_count()], false);
    }
    let mut dollars = 0.0;
    let mut volume = 0.0;
    let mut trades = 0.0;
    let mut days = BTreeSet::new();
    let mut ranges = 0.0;
    for bar in prior {
        dollars += dollar_volume(*bar);
        volume += f64::from(bar.volume).max(0.0);
        trades += f64::from(bar.trades).max(0.0);
        days.insert(bar.ts().div_euclid(MILLIS_PER_DAY));
        if let (Some(high), Some(low)) = (positive(bar.high), positive(bar.low)) {
            ranges += (high / low).ln();
        }
    }
    let day_count = days.len().max(1) as f64;
    (
        vec![
            (dollars / day_count).max(1.0).ln(),
            (dollars / prior.len() as f64).max(1.0).ln(),
            (volume / prior.len() as f64).max(0.0).ln_1p(),
            (trades / prior.len() as f64).max(0.0).ln_1p(),
            ranges / prior.len() as f64,
            prior.len() as f64 / day_count,
            1.0,
        ],
        true,
    )
}

fn spy_features(corpus: &BarCorpus) -> Option<HashMap<i64, [f64; 3]>> {
    let symbol = corpus.symbols().iter().position(|symbol| symbol == "SPY")?;
    let bars = corpus.bars(symbol);
    let mut out = HashMap::with_capacity(bars.len());
    for index in 1..bars.len() {
        let prior = &bars[index.saturating_sub(78)..index];
        let ema = prior
            .iter()
            .map(|bar| f64::from(bar.volume).max(0.0))
            .sum::<f64>()
            / prior.len().max(1) as f64;
        let dof = encode_dof(bars[index - 1].close, &bars[index], ema as f32);
        out.insert(
            bars[index].ts(),
            [f64::from(dof.r), f64::from(dof.s), f64::from(dof.w)],
        );
    }
    Some(out)
}

fn ranked_symbols(
    corpus: &BarCorpus,
    calibration: &CostCalibration,
    max_symbols: usize,
) -> Vec<u32> {
    let mut symbols: Vec<u32> = (0..corpus.series_count() as u32)
        .filter(|symbol| {
            let adv = calibration.pooled_adv_usd(*symbol);
            adv.is_finite() && adv > 0.0
        })
        .collect();
    symbols.sort_by(|left, right| {
        calibration
            .pooled_adv_usd(*right)
            .total_cmp(&calibration.pooled_adv_usd(*left))
            .then_with(|| {
                corpus
                    .symbol(*left as usize)
                    .cmp(corpus.symbol(*right as usize))
            })
    });
    symbols.truncate(max_symbols.min(symbols.len()));
    symbols
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstantSelection {
    Trailing,
    Leading,
}

fn panel_row(
    corpus: &BarCorpus,
    symbol: u32,
    index: usize,
    range: TimeRange,
    spy: Option<&HashMap<i64, [f64; 3]>>,
) -> Result<Option<Row>> {
    let bars = corpus.bars(symbol as usize);
    let Some(target) = corpus.direct_return_targets(symbol as usize, index, range)? else {
        return Ok(None); // complete H100 evidence is mandatory for every retained row
    };
    let own = own_features(bars, index);
    let own_r = own[0];
    let (market, market_present) = match spy.and_then(|rows| rows.get(&bars[index].ts())) {
        Some(proxy) => (
            vec![proxy[0], proxy[1], proxy[2], own_r - proxy[0], 1.0],
            true,
        ),
        None => (vec![0.0; Family::MarketResidual.feature_count()], false),
    };
    let (liquidity, liquidity_present) = trailing_liquidity_features(bars, index);
    let daily = corpus.adjusted_daily_features(symbol as usize, bars[index].ts());
    let daily_present = daily.is_available();
    let daily_features = daily.values.to_vec();
    Ok(Some(Row {
        symbol,
        ts_ms: bars[index].ts(),
        baseline: baseline_features(bars[index], Some(bars[index - 1].ts()), corpus.res_secs()),
        target: target.map(f64::from),
        own,
        market,
        cross: vec![0.0; Family::CrossSection.feature_count()],
        liquidity,
        daily: daily_features,
        market_present,
        cross_present: false,
        liquidity_present,
        daily_present,
    }))
}

fn build_panel(
    corpus: &BarCorpus,
    selected: &[u32],
    range: TimeRange,
    max_instants: usize,
    selection: InstantSelection,
    spy: Option<&HashMap<i64, [f64; 3]>>,
) -> Result<Vec<InstantRows>> {
    let mut instants: BTreeMap<i64, Vec<Row>> = BTreeMap::new();
    for &symbol in selected {
        let bars = corpus.bars(symbol as usize);
        let start = bars.partition_point(|bar| bar.ts() < range.start_ms).max(1);
        let end = bars.partition_point(|bar| bar.ts() < range.end_ms);
        match selection {
            InstantSelection::Trailing => {
                let mut retained_for_symbol = 0usize;
                for index in (start..end).rev() {
                    let Some(row) = panel_row(corpus, symbol, index, range, spy)? else {
                        continue;
                    };
                    instants.entry(row.ts_ms).or_default().push(row);
                    while instants.len() > max_instants {
                        instants.pop_first();
                    }
                    retained_for_symbol += 1;
                    if retained_for_symbol == max_instants {
                        break;
                    }
                }
            }
            InstantSelection::Leading => {
                for index in start..end {
                    if instants.len() == max_instants {
                        let last = *instants
                            .last_key_value()
                            .expect("bounded panel is non-empty")
                            .0;
                        if bars[index].ts() > last {
                            break;
                        }
                    }
                    let Some(row) = panel_row(corpus, symbol, index, range, spy)? else {
                        continue;
                    };
                    instants.entry(row.ts_ms).or_default().push(row);
                    while instants.len() > max_instants {
                        instants.pop_last();
                    }
                }
            }
        }
    }
    let mut groups: Vec<InstantRows> = instants
        .into_iter()
        .map(|(ts_ms, mut rows)| {
            rows.sort_by_key(|row| row.symbol);
            InstantRows { ts_ms, rows }
        })
        .collect();
    debug_assert!(groups.len() <= max_instants);
    for group in &mut groups {
        let returns: Vec<f64> = group.rows.iter().map(|row| row.own[0]).collect();
        let dollars: Vec<f64> = group.rows.iter().map(|row| row.own[12]).collect();
        if returns.len() < 2 {
            continue;
        }
        let return_ranks = mid_ranks(&returns);
        let dollar_ranks = mid_ranks(&dollars);
        let breadth =
            returns.iter().filter(|value| **value > 0.0).count() as f64 / returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let dispersion = (returns
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64)
            .sqrt();
        for (slot, row) in group.rows.iter_mut().enumerate() {
            row.cross = vec![
                2.0 * (return_ranks[slot] - 1.0) / (returns.len() - 1) as f64 - 1.0,
                2.0 * (dollar_ranks[slot] - 1.0) / (returns.len() - 1) as f64 - 1.0,
                breadth,
                dispersion,
                returns.len() as f64,
                1.0,
            ];
            row.cross_present = true;
        }
    }
    Ok(groups)
}

fn inner_split(groups: &[InstantRows]) -> usize {
    let tail = (groups.len() / INNER_TAIL_DIVISOR).max(1);
    groups
        .len()
        .saturating_sub(tail)
        .max(1)
        .min(groups.len().saturating_sub(1))
}

fn rotated_source(group: &InstantRows, row: usize, seed: u64) -> usize {
    if group.rows.len() < 2 {
        return row;
    }
    let offset = 1 + mix64(seed, group.ts_ms as u64) as usize % (group.rows.len() - 1);
    (row + offset) % group.rows.len()
}

fn raw_features<'a>(
    groups: &'a [InstantRows],
    group_index: usize,
    row_index: usize,
    family: Family,
    control: Control,
    seed: u64,
) -> (&'a [f64; BASELINE_FEATURES], &'a [f64]) {
    let row = &groups[group_index].rows[row_index];
    match control {
        Control::Real | Control::Masked => (&row.baseline, row.family(family)),
        Control::Shuffled => {
            let source = rotated_source(&groups[group_index], row_index, seed);
            (
                &row.baseline,
                groups[group_index].rows[source].family(family),
            )
        }
    }
}

fn fit_with_lambda(
    groups: &[InstantRows],
    family: Family,
    horizon_slot: usize,
    control: Control,
    seed: u64,
    standardizer: Standardizer,
    lambda: f64,
) -> Result<FittedModel> {
    let mut sums = RidgeSums::new(BASELINE_FEATURES + family.feature_count())?;
    for group_index in 0..groups.len() {
        for row_index in 0..groups[group_index].rows.len() {
            let (baseline, feature) =
                raw_features(groups, group_index, row_index, family, control, seed);
            let standardized =
                standardizer.transform(baseline, feature, control == Control::Masked);
            sums.push(
                &standardized,
                groups[group_index].rows[row_index].target[horizon_slot],
            )?;
        }
    }
    Ok(FittedModel {
        fit: sums.fit(lambda)?,
        standardizer,
    })
}

fn choose_lambda(
    groups: &[InstantRows],
    family: Family,
    horizon_slot: usize,
    control: Control,
    seed: u64,
) -> Result<f64> {
    ensure!(
        groups.len() >= 2,
        "lambda selection needs two synchronized fit instants"
    );
    let split = inner_split(groups);
    let standardizer = Standardizer::fit(&groups[..split], family)?;
    let mut best = (f64::INFINITY, LAMBDAS[0]);
    for lambda in LAMBDAS {
        let model = fit_with_lambda(
            &groups[..split],
            family,
            horizon_slot,
            control,
            seed,
            standardizer.clone(),
            lambda,
        )?;
        let mut instant_mse = 0.0;
        let mut instants = 0usize;
        for (local_group, group) in groups[split..].iter().enumerate() {
            let group_index = split + local_group;
            let mut mse = 0.0;
            let mut rows = 0usize;
            for row_index in 0..group.rows.len() {
                let (baseline, feature) =
                    raw_features(groups, group_index, row_index, family, control, seed);
                let x = model
                    .standardizer
                    .transform(baseline, feature, control == Control::Masked);
                let error = model.fit.predict(&x)? - group.rows[row_index].target[horizon_slot];
                mse += error * error;
                rows += 1;
            }
            if rows > 0 {
                instant_mse += mse / rows as f64;
                instants += 1;
            }
        }
        let score = instant_mse / instants.max(1) as f64;
        if score < best.0 {
            best = (score, lambda);
        }
    }
    Ok(best.1)
}

fn fit_model(
    groups: &[InstantRows],
    family: Family,
    horizon_slot: usize,
    control: Control,
    seed: u64,
) -> Result<FittedModel> {
    let lambda = choose_lambda(groups, family, horizon_slot, control, seed)?;
    let standardizer = Standardizer::fit(groups, family)?;
    fit_with_lambda(
        groups,
        family,
        horizon_slot,
        control,
        seed,
        standardizer,
        lambda,
    )
}

fn predict(
    model: &FittedModel,
    groups: &[InstantRows],
    group_index: usize,
    row_index: usize,
    family: Family,
    control: Control,
    seed: u64,
) -> Result<f64> {
    let (baseline, feature) = raw_features(groups, group_index, row_index, family, control, seed);
    let x = model
        .standardizer
        .transform(baseline, feature, control == Control::Masked);
    model.fit.predict(&x)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn evaluate(
    validation: &[InstantRows],
    family: Family,
    horizon_slot: usize,
    control: Control,
    seed: u64,
    model: &FittedModel,
    baseline_model: &FittedModel,
    incremental_reference: Option<(&FittedModel, Family, Control)>,
    cost: &BarCostModel,
    capital_usd: f64,
) -> Result<FoldMetrics> {
    let mut pearson = Vec::new();
    let mut rank = Vec::new();
    let mut gross = Vec::new();
    let mut fixed_net = Vec::new();
    let mut all_in_net = Vec::new();
    let mut breadth = Vec::new();
    let mut covered = 0usize;
    let mut total = 0usize;
    for group_index in 0..validation.len() {
        let group = &validation[group_index];
        if group.rows.is_empty() {
            continue;
        }
        let mut predictions = Vec::with_capacity(group.rows.len());
        let mut baseline_predictions = Vec::with_capacity(group.rows.len());
        let mut reference_predictions = Vec::with_capacity(group.rows.len());
        let mut targets = Vec::with_capacity(group.rows.len());
        for row_index in 0..group.rows.len() {
            predictions.push(predict(
                model,
                validation,
                group_index,
                row_index,
                family,
                control,
                seed,
            )?);
            baseline_predictions.push(predict(
                baseline_model,
                validation,
                group_index,
                row_index,
                Family::Clock,
                Control::Real,
                seed,
            )?);
            if let Some((reference_model, reference_family, reference_control)) =
                incremental_reference
            {
                reference_predictions.push(predict(
                    reference_model,
                    validation,
                    group_index,
                    row_index,
                    reference_family,
                    reference_control,
                    seed,
                )?);
            }
            targets.push(group.rows[row_index].target[horizon_slot]);
            covered += usize::from(group.rows[row_index].family_present(family));
            total += 1;
        }
        let (incremental, residual): (Vec<f64>, Vec<f64>) = if incremental_reference.is_some() {
            (
                predictions
                    .iter()
                    .zip(&reference_predictions)
                    .map(|(real, reference)| real - reference)
                    .collect(),
                targets
                    .iter()
                    .zip(&baseline_predictions)
                    .map(|(target, base)| target - base)
                    .collect(),
            )
        } else {
            (predictions.clone(), targets.clone())
        };
        let mut moments = ProductMoments::default();
        for (&x, &y) in incremental.iter().zip(&residual) {
            moments.push(x, y);
        }
        if moments.corr().is_finite() {
            pearson.push(moments.corr());
        }
        let prediction_ranks = mid_ranks(&incremental);
        let target_ranks = mid_ranks(&residual);
        let mut rank_moments = ProductMoments::default();
        for (&x, &y) in prediction_ranks.iter().zip(&target_ranks) {
            rank_moments.push(x, y);
        }
        if rank_moments.corr().is_finite() {
            rank.push(rank_moments.corr());
        }
        let Ok(deciles) = equal_count_buckets(&predictions, DECILES) else {
            // A tied boundary is unmeasured. Never let symbol id manufacture a decile.
            continue;
        };
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for (row_index, bucket) in deciles.into_iter().enumerate() {
            if bucket == 0 {
                bottom.push(row_index);
            } else if bucket + 1 == DECILES {
                top.push(row_index);
            }
        }
        let gross_return = mean(
            &top.iter()
                .map(|&slot| targets[slot].exp_m1())
                .collect::<Vec<_>>(),
        ) - mean(
            &bottom
                .iter()
                .map(|&slot| targets[slot].exp_m1())
                .collect::<Vec<_>>(),
        );
        let selected = top.len() + bottom.len();
        let notional = capital_usd / selected.max(1) as f64;
        let leg_cost = |slots: &[usize], impact: bool| {
            mean(
                &slots
                    .iter()
                    .map(|&slot| {
                        let resolved = cost.resolve(group.rows[slot].symbol, group.ts_ms);
                        let one_way = if impact {
                            resolved.total_bps(notional / resolved.adv_usd)
                        } else {
                            resolved.fixed_bps()
                        };
                        2.0 * one_way // enter and unwind the horizon position
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let gross_bps = gross_return * 10_000.0;
        gross.push(gross_bps);
        fixed_net.push(gross_bps - leg_cost(&top, false) - leg_cost(&bottom, false));
        all_in_net.push(gross_bps - leg_cost(&top, true) - leg_cost(&bottom, true));
        breadth.push(selected as f64);
    }
    Ok(FoldMetrics {
        pearson_ic: mean(&pearson),
        rank_ic: mean(&rank),
        gross_spread_bps: mean(&gross),
        impact_free_net_bps: mean(&fixed_net),
        all_in_net_bps: mean(&all_in_net),
        breadth: mean(&breadth),
        coverage: covered as f64 / total.max(1) as f64,
        missingness: 1.0 - covered as f64 / total.max(1) as f64,
        lambda: model.fit.lambda,
    })
}

fn unavailable_metrics() -> FoldMetrics {
    FoldMetrics {
        pearson_ic: f64::NAN,
        rank_ic: f64::NAN,
        gross_spread_bps: f64::NAN,
        impact_free_net_bps: f64::NAN,
        all_in_net_bps: f64::NAN,
        breadth: f64::NAN,
        coverage: 0.0,
        missingness: 1.0,
        lambda: f64::NAN,
    }
}

fn screen_fold(
    fold: &RollingOriginFold,
    corpus: &BarCorpus,
    selected: &[u32],
    max_instants: usize,
    spy: Option<&HashMap<i64, [f64; 3]>>,
    cost: &BarCostModel,
    capital_usd: f64,
    seed: u64,
) -> Result<Vec<ArmFold>> {
    let fit_rows = build_panel(
        corpus,
        selected,
        fold.fit,
        max_instants,
        InstantSelection::Trailing,
        spy,
    )
    .with_context(|| format!("building fold {} fit panel", fold.id))?;
    let validation_rows = build_panel(
        corpus,
        selected,
        fold.validation,
        max_instants,
        InstantSelection::Leading,
        spy,
    )
    .with_context(|| format!("building fold {} validation panel", fold.id))?;
    ensure!(
        fit_rows.len() >= 2,
        "fold {} has fewer than two fit instants",
        fold.id
    );
    ensure!(
        !validation_rows.is_empty(),
        "fold {} has no validation instants",
        fold.id
    );
    let mut out = Vec::new();
    for (horizon_slot, &horizon) in DIRECT_RETURN_HORIZONS.iter().enumerate() {
        let baseline = fit_model(
            &fit_rows,
            Family::Clock,
            horizon_slot,
            Control::Real,
            mix64(seed, u64::from(fold.id) << 32 | horizon as u64),
        )?;
        for family in Family::ALL {
            let dynamically_unavailable = match family {
                Family::MarketResidual => spy.is_none(),
                Family::AdjustedDaily => corpus.adjusted_daily_initialization_error().is_some(),
                _ => false,
            };
            if family.unavailable_prerequisite().is_some() || dynamically_unavailable {
                let missing = unavailable_metrics();
                out.push(ArmFold {
                    family,
                    horizon,
                    real: missing.clone(),
                    masked: missing.clone(),
                    shuffled: missing,
                });
                continue;
            }
            if family == Family::Clock {
                let metrics = evaluate(
                    &validation_rows,
                    family,
                    horizon_slot,
                    Control::Real,
                    seed,
                    &baseline,
                    &baseline,
                    None,
                    cost,
                    capital_usd,
                )?;
                out.push(ArmFold {
                    family,
                    horizon,
                    real: metrics.clone(),
                    masked: metrics.clone(),
                    shuffled: metrics,
                });
                continue;
            }
            let fit_seed = mix64(
                seed ^ 0xF17F_17F1,
                u64::from(fold.id) << 32 | horizon as u64,
            );
            let validation_seed = mix64(
                seed ^ 0xA11D_A7E5,
                u64::from(fold.id) << 32 | horizon as u64,
            );
            let real_model = fit_model(&fit_rows, family, horizon_slot, Control::Real, fit_seed)?;
            let masked_model =
                fit_model(&fit_rows, family, horizon_slot, Control::Masked, fit_seed)?;
            let shuffled_model =
                fit_model(&fit_rows, family, horizon_slot, Control::Shuffled, fit_seed)?;
            let real = evaluate(
                &validation_rows,
                family,
                horizon_slot,
                Control::Real,
                validation_seed,
                &real_model,
                &baseline,
                Some((&masked_model, family, Control::Masked)),
                cost,
                capital_usd,
            )?;
            let masked = evaluate(
                &validation_rows,
                family,
                horizon_slot,
                Control::Masked,
                validation_seed,
                &masked_model,
                &baseline,
                Some((&baseline, Family::Clock, Control::Real)),
                cost,
                capital_usd,
            )?;
            let shuffled = evaluate(
                &validation_rows,
                family,
                horizon_slot,
                Control::Shuffled,
                validation_seed,
                &shuffled_model,
                &baseline,
                Some((&masked_model, family, Control::Masked)),
                cost,
                capital_usd,
            )?;
            out.push(ArmFold {
                family,
                horizon,
                real,
                masked,
                shuffled,
            });
        }
    }
    Ok(out)
}

fn arm_series(
    rows: &[ArmFold],
    suffix: &str,
    value: impl Fn(&ArmFold) -> f64 + Copy,
) -> Vec<ReportSeries> {
    let mut out = Vec::with_capacity(Family::ALL.len() * DIRECT_RETURN_HORIZONS.len());
    for family in Family::ALL {
        for horizon in DIRECT_RETURN_HORIZONS {
            out.push(ReportSeries {
                label: format!("{}:H{horizon}:{suffix}", family.label()),
                values: rows
                    .iter()
                    .filter(|row| row.family == family && row.horizon == horizon)
                    .map(|row| value(row) as f32)
                    .collect(),
            });
        }
    }
    out
}

fn fold_agreement(rows: &[ArmFold], row: &ArmFold, metric: impl Fn(&FoldMetrics) -> f64) -> f64 {
    let peers: Vec<f64> = rows
        .iter()
        .filter(|peer| peer.family == row.family && peer.horizon == row.horizon)
        .map(|peer| metric(&peer.real))
        .filter(|value| value.is_finite())
        .collect();
    let consensus = mean(&peers).signum();
    if peers.is_empty() || consensus == 0.0 {
        f64::NAN
    } else {
        peers
            .iter()
            .filter(|value| value.signum() == consensus)
            .count() as f64
            / peers.len() as f64
    }
}

fn report_title(
    noun: &str,
    plan_hash: &str,
    universe: &str,
    bounds: (i64, i64),
    dynamic_unavailable: &[String],
) -> String {
    let static_unavailable = Family::ALL
        .into_iter()
        .filter_map(|family| {
            family
                .unavailable_prerequisite()
                .map(|why| format!("{}: {why}", family.label()))
        })
        .chain(dynamic_unavailable.iter().cloned())
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "Feature Screen {noun} | fold_plan_sha256={plan_hash} | universe_sha256={universe} | bounds=[{}, {}) | cost=canonical-causal-pre-origin; round-trip=2x-one-way-per-leg; equal-capital-selected-names | unavailable={static_unavailable}",
        bounds.0, bounds.1
    )
}

fn write_reports(
    output: &Path,
    rows: &[ArmFold],
    plan_hash: &str,
    universe: &str,
    bounds: (i64, i64),
    dynamic_unavailable: &[String],
) -> Result<()> {
    fs::create_dir_all(output).with_context(|| format!("creating {}", output.display()))?;

    let mut ic_series = arm_series(rows, "baseline_residual_pearson_ic", |row| {
        row.real.pearson_ic
    });
    ic_series.extend(arm_series(
        rows,
        "within_instant_tie_aware_rank_ic",
        |row| row.real.rank_ic,
    ));
    ic_series.extend(arm_series(rows, "coverage", |row| row.real.coverage));
    ic_series.extend(arm_series(rows, "missingness", |row| row.real.missingness));
    ic_series.extend(arm_series(rows, "selected_lambda", |row| row.real.lambda));
    ic_series.extend(arm_series(rows, "fold_sign_agreement", |row| {
        fold_agreement(rows, row, |metrics| metrics.pearson_ic)
    }));
    let ic_report = Report {
        title: report_title("IC", plan_hash, universe, bounds, dynamic_unavailable),
        x_label: Some("rolling-origin fold id".into()),
        y_label: Some("equal-weighted synchronized validation instant".into()),
        scale: ScaleKind::Linear,
        kind: ReportKind::MultiLine { series: ic_series },
    };

    let mut spread_series = arm_series(rows, "exact_decile_gross_bps", |row| {
        row.real.gross_spread_bps
    });
    spread_series.extend(arm_series(rows, "matched_impact_free_net_bps", |row| {
        row.real.impact_free_net_bps
    }));
    spread_series.extend(arm_series(rows, "matched_all_in_net_bps", |row| {
        row.real.all_in_net_bps
    }));
    spread_series.extend(arm_series(rows, "selected_name_breadth", |row| {
        row.real.breadth
    }));
    spread_series.extend(arm_series(rows, "coverage", |row| row.real.coverage));
    spread_series.extend(arm_series(rows, "fold_sign_agreement", |row| {
        fold_agreement(rows, row, |metrics| metrics.gross_spread_bps)
    }));
    let spread_report = Report {
        title: report_title("Spread", plan_hash, universe, bounds, dynamic_unavailable),
        x_label: Some("rolling-origin fold id".into()),
        y_label: Some("bps per synchronized instant; exact untied top/bottom deciles only".into()),
        scale: ScaleKind::Linear,
        kind: ReportKind::MultiLine {
            series: spread_series,
        },
    };

    let mut control_series = arm_series(rows, "pearson_real_minus_masked", |row| {
        row.real.pearson_ic - row.masked.pearson_ic
    });
    control_series.extend(arm_series(rows, "pearson_real_minus_shuffled", |row| {
        row.real.pearson_ic - row.shuffled.pearson_ic
    }));
    control_series.extend(arm_series(rows, "rank_real_minus_masked", |row| {
        row.real.rank_ic - row.masked.rank_ic
    }));
    control_series.extend(arm_series(rows, "rank_real_minus_shuffled", |row| {
        row.real.rank_ic - row.shuffled.rank_ic
    }));
    control_series.extend(arm_series(
        rows,
        "gross_spread_real_minus_masked_bps",
        |row| row.real.gross_spread_bps - row.masked.gross_spread_bps,
    ));
    control_series.extend(arm_series(
        rows,
        "gross_spread_real_minus_shuffled_bps",
        |row| row.real.gross_spread_bps - row.shuffled.gross_spread_bps,
    ));
    control_series.extend(arm_series(rows, "all_in_real_minus_masked_bps", |row| {
        row.real.all_in_net_bps - row.masked.all_in_net_bps
    }));
    control_series.extend(arm_series(rows, "all_in_real_minus_shuffled_bps", |row| {
        row.real.all_in_net_bps - row.shuffled.all_in_net_bps
    }));
    control_series.extend(arm_series(rows, "coverage", |row| row.real.coverage));
    let controls_report = Report {
        title: report_title(
            "Matched Controls",
            plan_hash,
            universe,
            bounds,
            dynamic_unavailable,
        ),
        x_label: Some("rolling-origin fold id".into()),
        y_label: Some("paired real-minus-control on identical validation rows".into()),
        scale: ScaleKind::Linear,
        kind: ReportKind::MultiLine {
            series: control_series,
        },
    };

    for (base, report) in [
        (FEATURE_SCREEN_IC_BASE, ic_report),
        (FEATURE_SCREEN_SPREAD_BASE, spread_report),
        (FEATURE_SCREEN_CONTROLS_BASE, controls_report),
    ] {
        write_report(output.join(format!("{base}.report.bin")), &report)
            .with_context(|| format!("writing registered {base} report"))?;
    }
    Ok(())
}

/// Run the CPU-only information-value screen. No torch device, checkpoint or neural weight is
/// opened anywhere in this path.
pub fn pretrain_feature_screen(args: FeatureScreenArgs) -> Result<()> {
    ensure!(args.folds > 0, "--folds must be positive");
    ensure!(
        args.max_symbols >= DECILES,
        "--max-symbols must be at least {DECILES}"
    );
    ensure!(args.max_instants >= 2, "--max-instants must be at least 2");
    ensure!(args.cost_threads > 0, "--cost-threads must be positive");
    ensure!(
        args.capital_usd.is_finite() && args.capital_usd > 0.0,
        "--capital must be positive and finite"
    );
    ensure!(
        args.corpus.split_bounds.is_some(),
        "pretrain-feature-screen requires pinned --split-bounds"
    );
    ensure!(
        !args.corpus.derive_split_bounds,
        "pretrain-feature-screen cannot derive mutable split bounds"
    );

    let corpus = load_corpus(&args.corpus)?;
    let plan = corpus.rolling_origin_plan(args.folds, args.seed, args.max_instants)?;
    let cutoff = plan.earliest_origin_universe_cutoff_ms;
    let calibration = Arc::new(CostCalibration::from_corpus(
        &corpus,
        args.cost_threads,
        cutoff,
    )?);
    let selected = ranked_symbols(&corpus, &calibration, args.max_symbols);
    ensure!(
        selected.len() >= DECILES,
        "earliest-origin admitted universe has fewer than {DECILES} names"
    );
    let spy = spy_features(&corpus);
    let cost = BarCostModel::new(calibration);
    let mut rows = Vec::new();
    for fold in &plan.folds {
        rows.extend(screen_fold(
            fold,
            &corpus,
            &selected,
            args.max_instants,
            spy.as_ref(),
            &cost,
            args.capital_usd,
            args.seed,
        )?);
    }
    let mut dynamic_unavailable = Vec::new();
    if spy.is_none() {
        dynamic_unavailable.push(
            "spy_exact_timestamp_residual: SPY is not in the earliest-origin admitted corpus; no proxy substituted"
                .to_string(),
        );
    }
    if let Some(error) = corpus.adjusted_daily_initialization_error() {
        dynamic_unavailable.push(format!(
            "last_fully_completed_adjusted_daily: {error}; no intraday reconstruction substituted"
        ));
    }
    write_reports(
        &PathBuf::from(&args.output),
        &rows,
        &plan.canonical_sha256(),
        &plan.earliest_origin_universe_digest,
        corpus.split_bounds(),
        &dynamic_unavailable,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{pretrain_report_owner, read_report, PretrainReportOwner};

    fn row(ts_ms: i64, symbol: u32, x: f64, target: f64) -> Row {
        Row {
            symbol,
            ts_ms,
            baseline: [x, 0.0, 0.0, 0.0, 0.0, 0.0],
            target: [target; 6],
            own: vec![x; Family::OwnBars.feature_count()],
            market: vec![x; Family::MarketResidual.feature_count()],
            cross: vec![x; Family::CrossSection.feature_count()],
            liquidity: vec![x; Family::TrailingLiquidity.feature_count()],
            daily: vec![x; Family::AdjustedDaily.feature_count()],
            market_present: true,
            cross_present: true,
            liquidity_present: true,
            daily_present: true,
        }
    }

    fn groups(count: usize) -> Vec<InstantRows> {
        (0..count)
            .map(|instant| InstantRows {
                ts_ms: 1_700_000_000_000 + instant as i64 * 300_000,
                rows: (0..10)
                    .map(|symbol| {
                        let x = instant as f64 + symbol as f64 / 10.0;
                        row(
                            1_700_000_000_000 + instant as i64 * 300_000,
                            symbol,
                            x,
                            2.0 * x + 3.0,
                        )
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn normalization_and_lambda_use_only_the_fold_fit_rows() {
        let fit_rows = groups(10);
        let mut external_validation = groups(3);
        for group in &mut external_validation {
            for row in &mut group.rows {
                row.own.fill(1e9);
                row.target.fill(-1e9);
            }
        }
        let before = Standardizer::fit(&fit_rows, Family::OwnBars).unwrap();
        let lambda_before = choose_lambda(&fit_rows, Family::OwnBars, 0, Control::Real, 7).unwrap();
        let after = Standardizer::fit(&fit_rows, Family::OwnBars).unwrap();
        let lambda_after = choose_lambda(&fit_rows, Family::OwnBars, 0, Control::Real, 7).unwrap();
        assert_eq!(before.mean, after.mean);
        assert_eq!(lambda_before, lambda_after);
        assert_eq!(external_validation[0].rows[0].target[0], -1e9);
    }

    #[test]
    fn streaming_ridge_recovers_a_known_line() {
        let fit = fit_model(&groups(12), Family::Clock, 0, Control::Real, 9).unwrap();
        let x = fit
            .standardizer
            .transform(&[4.25, 0.0, 0.0, 0.0, 0.0, 0.0], &[], false);
        assert!((fit.fit.predict(&x).unwrap() - 11.5).abs() < 0.1);
    }

    #[test]
    fn own_features_do_not_read_appended_future_bars() {
        let make = |slot: usize| PackedBar {
            ts_ms: slot as i64 * 300_000,
            open: 100.0 + slot as f32,
            high: 101.0 + slot as f32,
            low: 99.0 + slot as f32,
            close: 100.5 + slot as f32,
            volume: 1_000.0 + slot as f32,
            vwap: 100.25 + slot as f32,
            trades: 10 + slot as u32,
        };
        let mut bars: Vec<_> = (0..100).map(make).collect();
        let before = own_features(&bars, 90);
        bars.extend((100..140).map(make));
        assert_eq!(before, own_features(&bars, 90));
    }

    #[test]
    fn whole_bundle_rotation_is_deterministic_and_non_vacuous() {
        let groups = groups(2);
        for row_index in 0..groups[0].rows.len() {
            let source = rotated_source(&groups[0], row_index, 13);
            assert_ne!(source, row_index);
            assert_eq!(source, rotated_source(&groups[0], row_index, 13));
            assert_eq!(
                raw_features(
                    &groups,
                    0,
                    row_index,
                    Family::OwnBars,
                    Control::Shuffled,
                    13
                )
                .1,
                groups[0].rows[source].own
            );
        }
    }

    #[test]
    fn tied_decile_boundaries_are_unmeasured() {
        assert!(equal_count_buckets(&[1.0; 10], DECILES).is_err());
        let values = [0.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert!(equal_count_buckets(&values, DECILES).is_err());
    }

    #[test]
    fn cross_section_is_aggregated_only_within_one_instant() {
        let mut rows = groups(2);
        let first: Vec<f64> = rows[0].rows.iter().map(|row| row.own[0]).collect();
        let second: Vec<f64> = rows[1].rows.iter().map(|row| row.own[0]).collect();
        assert_ne!(mean(&first), mean(&second));
        rows[0].rows[0].own[0] = -1000.0;
        assert_eq!(
            rows[1]
                .rows
                .iter()
                .map(|row| row.own[0])
                .collect::<Vec<_>>(),
            second
        );
    }

    #[test]
    fn report_registry_owns_and_reads_all_feature_screen_bases() {
        for base in [
            FEATURE_SCREEN_IC_BASE,
            FEATURE_SCREEN_SPREAD_BASE,
            FEATURE_SCREEN_CONTROLS_BASE,
        ] {
            assert_eq!(
                pretrain_report_owner(base),
                Some(PretrainReportOwner::FeatureScreen)
            );
        }
        let dir =
            std::env::temp_dir().join(format!("feature-screen-report-{}", std::process::id()));
        let rows = vec![ArmFold {
            family: Family::Sector,
            horizon: 1,
            real: unavailable_metrics(),
            masked: unavailable_metrics(),
            shuffled: unavailable_metrics(),
        }];
        write_reports(&dir, &rows, &"a".repeat(64), &"b".repeat(64), (1, 2), &[]).unwrap();
        for base in [
            FEATURE_SCREEN_IC_BASE,
            FEATURE_SCREEN_SPREAD_BASE,
            FEATURE_SCREEN_CONTROLS_BASE,
        ] {
            let report = read_report(&dir.join(format!("{base}.report.bin"))).unwrap();
            assert!(report.title.contains("effective_dated_sector_unavailable"));
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unavailable_arms_emit_nan_with_zero_coverage() {
        let metrics = unavailable_metrics();
        assert!(metrics.pearson_ic.is_nan());
        assert!(metrics.gross_spread_bps.is_nan());
        assert_eq!(metrics.coverage, 0.0);
        assert_eq!(metrics.missingness, 1.0);
    }

    #[test]
    fn cost_rows_keep_original_corpus_symbol_ids() {
        let rows = groups(1);
        for (expected, row) in rows[0].rows.iter().enumerate() {
            assert_eq!(row.symbol, expected as u32);
        }
    }
}
