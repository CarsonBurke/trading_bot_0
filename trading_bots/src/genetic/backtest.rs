use std::path::Path;

use hashbrown::HashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::constants::COMMISSION_RATE;

use crate::{
    charts::{assets_chart, buy_sell_chart},
    history::report::{write_report, Report, ReportKind, ScaleKind, TradePoint},
    types::{Account, MappedHistorical},
    utils::{
        convert_historical, ema, get_rsi_values, is_min_transaction, round_to_stock_fractional,
    },
};

use super::{
    family::{DecisionContext, StrategyFamilySpec},
    metrics::{BacktestMetricAccumulator, BacktestMetrics, TradeSummary, STARTING_CASH},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketDataset {
    pub split_name: String,
    pub tickers: Vec<String>,
    pub bars: MappedHistorical,
    pub close_prices: Vec<Vec<f64>>,
    pub benchmark_assets: Vec<f64>,
}

impl MarketDataset {
    pub fn new(
        split_name: impl Into<String>,
        tickers: Vec<String>,
        bars: MappedHistorical,
    ) -> Self {
        let close_prices = bars.iter().map(convert_historical).collect::<Vec<_>>();
        let benchmark_assets = benchmark_curve_from_prices(&close_prices);
        Self {
            split_name: split_name.into(),
            tickers,
            bars,
            close_prices,
            benchmark_assets,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestTrace {
    pub total_assets: Vec<f64>,
    pub cash: Vec<f64>,
    pub benchmark_assets: Vec<f64>,
    pub prices_by_ticker: Vec<Vec<f64>>,
    pub positioned_by_ticker: Vec<Vec<f64>>,
    pub buys_by_ticker: Vec<Vec<u32>>,
    pub sells_by_ticker: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestOutcome {
    pub metrics: BacktestMetrics,
    pub trace: Option<BacktestTrace>,
}

pub fn evaluate_family<F: StrategyFamilySpec>(
    family: &F,
    genome: &F::Genome,
    market: &MarketDataset,
    capture_trace: bool,
) -> BacktestOutcome {
    let ticker_count = market.bars.len();
    let indexes = market
        .close_prices
        .first()
        .map(|prices| prices.len())
        .unwrap_or(0);
    let indicator_cfg = family.indicator_config(genome);

    let decider_rsi_by_ticker =
        compute_rsi_map(&market.close_prices, indicator_cfg.decider_rsi_alpha);
    let amount_rsi_by_ticker =
        compute_rsi_map(&market.close_prices, indicator_cfg.amount_rsi_alpha);
    let price_ema_by_ticker = compute_ema_map(&market.close_prices, indicator_cfg.price_ema_alpha);
    let fast_ema_by_ticker = indicator_cfg
        .fast_ema_alpha
        .map(|alpha| compute_ema_map(&market.close_prices, alpha));
    let slow_ema_by_ticker = indicator_cfg
        .slow_ema_alpha
        .map(|alpha| compute_ema_map(&market.close_prices, alpha));

    let mut account = Account::new(STARTING_CASH, ticker_count);
    let mut metric_accumulator = BacktestMetricAccumulator::default();
    let mut total_assets = capture_trace.then(|| Vec::with_capacity(indexes));
    let mut cash_curve = capture_trace.then(|| Vec::with_capacity(indexes));
    let mut positioned_by_ticker =
        capture_trace.then(|| vec![Vec::with_capacity(indexes); ticker_count]);
    let mut buys_by_ticker = capture_trace.then(|| vec![Vec::new(); ticker_count]);
    let mut sells_by_ticker = capture_trace.then(|| vec![Vec::new(); ticker_count]);
    let mut states = market
        .close_prices
        .iter()
        .map(|prices| TickerRuntimeState::new(prices[0]))
        .collect::<Vec<_>>();
    let mut trade_summary = TradeSummary::default();

    for index in 0..indexes {
        let prices_at_index = (0..ticker_count)
            .map(|ticker_idx| market.close_prices[ticker_idx][index])
            .collect::<Vec<_>>();
        let position_values = account
            .positions
            .iter()
            .enumerate()
            .map(|(ticker_idx, position)| position.value_with_price(prices_at_index[ticker_idx]))
            .collect::<Vec<_>>();
        let assets = account.cash + position_values.iter().sum::<f64>();

        let mut contexts = Vec::with_capacity(ticker_count);
        for ticker_idx in 0..ticker_count {
            let price = prices_at_index[ticker_idx];
            let position = account.positions[ticker_idx];
            let state = &mut states[ticker_idx];
            state.observe(
                price,
                decider_rsi_by_ticker[ticker_idx][index],
                position.quantity > 0.0,
            );
            let benchmark_return_since_entry_pct =
                state.benchmark_return_since_entry_pct(market.benchmark_assets[index]);
            let excess_return_since_entry_pct = if position.quantity > 0.0 {
                Some(((price / position.avg_price.max(1e-6)) - 1.0) * 100.0)
                    .zip(benchmark_return_since_entry_pct)
                    .map(|(ticker_return_pct, benchmark_return_pct)| {
                        ticker_return_pct - benchmark_return_pct
                    })
            } else {
                None
            };
            state.observe_relative_performance(index, excess_return_since_entry_pct);

            contexts.push(DecisionContext {
                index,
                ticker_count: market.bars.len(),
                ticker_idx,
                price,
                decider_rsi: decider_rsi_by_ticker[ticker_idx][index],
                amount_rsi: amount_rsi_by_ticker[ticker_idx][index],
                price_ema: price_ema_by_ticker[ticker_idx][index],
                fast_ema: fast_ema_by_ticker
                    .as_ref()
                    .map(|series| series[ticker_idx][index]),
                slow_ema: slow_ema_by_ticker
                    .as_ref()
                    .map(|series| series[ticker_idx][index]),
                local_minimum: state.local_minimum,
                local_maximum: state.local_maximum,
                last_buy_price: state.last_buy_price,
                last_sell_price: state.last_sell_price,
                lowest_rsi_since_flat: state.lowest_rsi_since_flat,
                highest_rsi_since_long: state.highest_rsi_since_long,
                position_value: position_values[ticker_idx],
                position_avg_price: position.avg_price,
                position_quantity: position.quantity,
                bars_since_entry: state.bars_since_entry(index),
                benchmark_return_since_entry_pct,
                excess_return_since_entry_pct,
                excess_return_peak_pct: state.peak_excess_return_since_entry_pct,
                excess_return_delta_pct: state.excess_return_delta_pct,
                underperformance_streak: Some(state.underperformance_streak),
                assets,
                cash: account.cash,
            });
        }

        rebalance_to_targets(
            family,
            genome,
            index,
            assets,
            &prices_at_index,
            market.benchmark_assets[index],
            &contexts,
            &mut account,
            &mut states,
            buys_by_ticker.as_mut(),
            sells_by_ticker.as_mut(),
            &mut trade_summary,
        );

        let post_rebalance_values = account
            .positions
            .iter()
            .enumerate()
            .map(|(ticker_idx, position)| position.value_with_price(prices_at_index[ticker_idx]))
            .collect::<Vec<_>>();
        let post_assets = account.cash + post_rebalance_values.iter().sum::<f64>();

        metric_accumulator.observe(post_assets);
        if let Some(cash_curve) = cash_curve.as_mut() {
            cash_curve.push(account.cash);
        }
        if let Some(total_assets) = total_assets.as_mut() {
            total_assets.push(post_assets);
        }
        if let Some(positioned_by_ticker) = positioned_by_ticker.as_mut() {
            for ticker_idx in 0..ticker_count {
                positioned_by_ticker[ticker_idx].push(post_rebalance_values[ticker_idx]);
            }
        }
    }

    let metrics = metric_accumulator.finish(
        market
            .benchmark_assets
            .last()
            .copied()
            .unwrap_or(STARTING_CASH),
        trade_summary,
    );
    let trace = if capture_trace {
        Some(BacktestTrace {
            total_assets: total_assets.unwrap_or_default(),
            cash: cash_curve.unwrap_or_default(),
            benchmark_assets: market.benchmark_assets.clone(),
            prices_by_ticker: market.close_prices.clone(),
            positioned_by_ticker: positioned_by_ticker.unwrap_or_default(),
            buys_by_ticker: buys_by_ticker.unwrap_or_default(),
            sells_by_ticker: sells_by_ticker.unwrap_or_default(),
        })
    } else {
        None
    };

    BacktestOutcome { metrics, trace }
}

pub fn write_trace_reports(
    output_dir: &Path,
    split_name: &str,
    tickers: &[String],
    trace: &BacktestTrace,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let assets_report = Report {
        title: format!("{split_name} Assets"),
        x_label: Some("Step".to_string()),
        y_label: Some("Assets".to_string()),
        scale: ScaleKind::Linear,
        kind: ReportKind::Assets {
            total: to_f32(&trace.total_assets),
            cash: to_f32(&trace.cash),
            positioned: None,
            benchmark: Some(to_f32(&trace.benchmark_assets)),
        },
    };
    write_report(
        output_dir
            .join(format!("ga_{split_name}_assets.report.bin"))
            .to_string_lossy()
            .as_ref(),
        &assets_report,
    )?;

    let asset_dir = output_dir.to_string_lossy().to_string();
    let _ = assets_chart(
        &asset_dir,
        &trace.total_assets,
        &trace.cash,
        None,
        Some(&trace.benchmark_assets),
    );

    tickers
        .par_iter()
        .enumerate()
        .try_for_each(|(ticker_idx, ticker)| {
            write_ticker_trace_reports(output_dir, split_name, ticker, trace, ticker_idx)
        })?;

    Ok(())
}

fn write_ticker_trace_reports(
    output_dir: &Path,
    split_name: &str,
    ticker: &str,
    trace: &BacktestTrace,
    ticker_idx: usize,
) -> anyhow::Result<()> {
    let ticker_dir = output_dir.join(ticker);
    std::fs::create_dir_all(&ticker_dir)?;
    let position_value = trace.positioned_by_ticker[ticker_idx].clone();
    let zero_cash = vec![0.0; position_value.len()];

    let buy_sell_report = Report {
        title: format!("{split_name} {ticker} Buy/Sell"),
        x_label: Some("Step".to_string()),
        y_label: Some("Price".to_string()),
        scale: ScaleKind::Linear,
        kind: ReportKind::BuySell {
            prices: to_f32(&trace.prices_by_ticker[ticker_idx]),
            buys: trace.buys_by_ticker[ticker_idx]
                .iter()
                .map(|index| TradePoint { index: *index })
                .collect(),
            sells: trace.sells_by_ticker[ticker_idx]
                .iter()
                .map(|index| TradePoint { index: *index })
                .collect(),
        },
    };
    write_report(
        ticker_dir
            .join("buy_sell.report.bin")
            .to_string_lossy()
            .as_ref(),
        &buy_sell_report,
    )?;

    let assets_report = Report {
        title: format!("{split_name} {ticker} Position Value"),
        x_label: Some("Step".to_string()),
        y_label: Some("Assets".to_string()),
        scale: ScaleKind::Linear,
        kind: ReportKind::Assets {
            total: to_f32(&position_value),
            cash: to_f32(&zero_cash),
            positioned: Some(to_f32(&position_value)),
            benchmark: None,
        },
    };
    write_report(
        ticker_dir
            .join("assets.report.bin")
            .to_string_lossy()
            .as_ref(),
        &assets_report,
    )?;

    let chart_dir = ticker_dir.to_string_lossy().to_string();
    let buy_indexes: HashMap<usize, (f64, f64)> = trace.buys_by_ticker[ticker_idx]
        .iter()
        .map(|index| {
            (
                *index as usize,
                (trace.prices_by_ticker[ticker_idx][*index as usize], 1.0),
            )
        })
        .collect();
    let sell_indexes: HashMap<usize, (f64, f64)> = trace.sells_by_ticker[ticker_idx]
        .iter()
        .map(|index| {
            (
                *index as usize,
                (trace.prices_by_ticker[ticker_idx][*index as usize], 1.0),
            )
        })
        .collect();
    let _ = buy_sell_chart(
        &chart_dir,
        &trace.prices_by_ticker[ticker_idx],
        &buy_indexes,
        &sell_indexes,
    );
    let _ = assets_chart(
        &chart_dir,
        &position_value,
        &zero_cash,
        Some(&position_value),
        None,
    );
    Ok(())
}

fn compute_rsi_map(data: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    data.iter()
        .map(|ticker_data| get_rsi_values(ticker_data, alpha))
        .collect()
}

fn compute_ema_map(data: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    data.iter()
        .map(|ticker_data| ema(ticker_data, alpha))
        .collect()
}

fn benchmark_curve_from_prices(prices_by_ticker: &[Vec<f64>]) -> Vec<f64> {
    if prices_by_ticker.is_empty() {
        return vec![STARTING_CASH];
    }
    let allocation = STARTING_CASH / prices_by_ticker.len() as f64;
    let quantities: Vec<f64> = prices_by_ticker
        .iter()
        .map(|ticker_prices| allocation / ticker_prices[0])
        .collect();

    (0..prices_by_ticker[0].len())
        .map(|index| {
            quantities
                .iter()
                .enumerate()
                .map(|(ticker_idx, quantity)| quantity * prices_by_ticker[ticker_idx][index])
                .sum::<f64>()
        })
        .collect()
}

fn rebalance_to_targets<F: StrategyFamilySpec>(
    family: &F,
    genome: &F::Genome,
    index: usize,
    assets: f64,
    prices: &[f64],
    benchmark_assets: f64,
    contexts: &[DecisionContext],
    account: &mut Account,
    states: &mut [TickerRuntimeState],
    mut buy_markers: Option<&mut Vec<Vec<u32>>>,
    mut sell_markers: Option<&mut Vec<Vec<u32>>>,
    trade_summary: &mut TradeSummary,
) {
    if assets <= f64::EPSILON {
        return;
    }

    let asset_scores = contexts
        .iter()
        .map(|ctx| family.asset_desirability(genome, ctx).max(0.0))
        .collect::<Vec<_>>();
    let cash_score = family.cash_desirability(genome, contexts).max(0.0);
    let total_score = cash_score + asset_scores.iter().sum::<f64>();
    if total_score <= f64::EPSILON {
        return;
    }

    let target_weights = asset_scores
        .iter()
        .zip(contexts.iter())
        .map(|(score, ctx)| {
            let raw_weight = score / total_score;
            let min_weight = family.min_target_weight(genome, ctx).clamp(0.0, 1.0);
            let max_weight = family.max_target_weight(genome, ctx).clamp(0.0, 1.0);
            raw_weight.min(max_weight).max(min_weight.min(max_weight))
        })
        .collect::<Vec<_>>();
    let target_values = target_weights
        .iter()
        .map(|weight| assets * weight)
        .collect::<Vec<_>>();
    let buy_deadband = assets * 0.0025;
    let sell_deadband = assets * 0.006;

    let current_values = account
        .positions
        .iter()
        .enumerate()
        .map(|(ticker_idx, position)| position.value_with_price(prices[ticker_idx]))
        .collect::<Vec<_>>();

    for ticker_idx in 0..prices.len() {
        let excess = (current_values[ticker_idx] - target_values[ticker_idx]).max(0.0);
        if excess <= sell_deadband {
            continue;
        }
        if let Some(sell_markers) = sell_markers.as_deref_mut() {
            let _ = try_sell(
                ticker_idx,
                index,
                prices[ticker_idx],
                excess,
                account,
                states,
                Some(&mut sell_markers[ticker_idx]),
                trade_summary,
            );
        } else {
            let _ = try_sell(
                ticker_idx,
                index,
                prices[ticker_idx],
                excess,
                account,
                states,
                None,
                trade_summary,
            );
        }
    }

    let refreshed_values = account
        .positions
        .iter()
        .enumerate()
        .map(|(ticker_idx, position)| position.value_with_price(prices[ticker_idx]))
        .collect::<Vec<_>>();

    let mut deficits = refreshed_values
        .iter()
        .enumerate()
        .map(|(ticker_idx, value)| (ticker_idx, (target_values[ticker_idx] - value).max(0.0)))
        .filter(|(_, deficit)| *deficit > buy_deadband)
        .collect::<Vec<_>>();
    deficits.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (ticker_idx, deficit) in deficits {
        let budget = deficit.min(account.cash);
        if let Some(buy_markers) = buy_markers.as_deref_mut() {
            let _ = try_buy(
                ticker_idx,
                index,
                prices[ticker_idx],
                budget,
                assets,
                benchmark_assets,
                account,
                states,
                Some(&mut buy_markers[ticker_idx]),
                trade_summary,
            );
        } else {
            let _ = try_buy(
                ticker_idx,
                index,
                prices[ticker_idx],
                budget,
                assets,
                benchmark_assets,
                account,
                states,
                None,
                trade_summary,
            );
        }
    }
}

fn try_buy(
    ticker_idx: usize,
    index: usize,
    price: f64,
    buy_budget: f64,
    assets: f64,
    benchmark_assets: f64,
    account: &mut Account,
    states: &mut [TickerRuntimeState],
    buy_markers: Option<&mut Vec<u32>>,
    trade_summary: &mut TradeSummary,
) -> bool {
    if !is_min_transaction(assets, buy_budget) {
        return false;
    }
    let quantity = buy_budget / (price + COMMISSION_RATE);
    if quantity <= 0.0 {
        return false;
    }
    let total_buy = price * quantity;
    let commission = quantity * COMMISSION_RATE;
    let total_cost = total_buy + commission;
    if total_cost > account.cash {
        return false;
    }

    let was_flat = account.positions[ticker_idx].quantity <= 0.0;
    account.positions[ticker_idx].add(price, quantity);
    account.cash -= total_cost;
    states[ticker_idx].record_buy(price, benchmark_assets, was_flat);
    if let Some(buy_markers) = buy_markers {
        buy_markers.push(index as u32);
    }
    trade_summary.turnover += total_buy;
    trade_summary.buy_count += 1;
    trade_summary.total_commissions += commission;
    true
}

fn try_sell(
    ticker_idx: usize,
    index: usize,
    price: f64,
    sell_budget: f64,
    account: &mut Account,
    states: &mut [TickerRuntimeState],
    sell_markers: Option<&mut Vec<u32>>,
    trade_summary: &mut TradeSummary,
) -> bool {
    let position = &mut account.positions[ticker_idx];
    if position.quantity <= 0.0 {
        return false;
    }
    let position_value = position.value_with_price(price);
    let (total_sell, quantity) = round_to_stock_fractional(price, sell_budget.min(position_value));
    if quantity <= 0.0 {
        return false;
    }
    let commission = quantity * COMMISSION_RATE;

    let profitable = price > position.avg_price;
    position.quantity -= quantity;
    if position.quantity <= 1e-9 {
        position.quantity = 0.0;
        position.avg_price = 0.0;
    }
    account.cash += total_sell - commission;
    states[ticker_idx].record_sell(price, position.quantity <= 0.0);
    if let Some(sell_markers) = sell_markers {
        sell_markers.push(index as u32);
    }
    trade_summary.turnover += total_sell;
    trade_summary.sell_count += 1;
    trade_summary.total_commissions += commission;
    if profitable {
        trade_summary.profitable_sells += 1;
    }
    true
}

fn to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

#[cfg(test)]
mod tests {
    use ibapi::market_data::historical::Bar;
    use rand::rngs::StdRng;
    use serde::{Deserialize, Serialize};
    use time::{Duration, OffsetDateTime};

    use super::*;
    use crate::genetic::family::{GeneticFamily, IndicatorConfig};

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct ConcentratedGenome;

    struct ConcentratedFamily;

    impl StrategyFamilySpec for ConcentratedFamily {
        type Genome = ConcentratedGenome;

        fn kind(&self) -> GeneticFamily {
            GeneticFamily::TrendBreakout
        }

        fn seed_genome(&self, _rng: &mut StdRng) -> Self::Genome {
            ConcentratedGenome
        }

        fn mutate(&self, _genome: &mut Self::Genome, _rng: &mut StdRng, _entropy: f64) {}

        fn crossover(
            &self,
            _left: &Self::Genome,
            _right: &Self::Genome,
            _rng: &mut StdRng,
        ) -> Self::Genome {
            ConcentratedGenome
        }

        fn indicator_config(&self, _genome: &Self::Genome) -> IndicatorConfig {
            IndicatorConfig {
                decider_rsi_alpha: 0.05,
                amount_rsi_alpha: 0.05,
                price_ema_alpha: 0.05,
                fast_ema_alpha: None,
                slow_ema_alpha: None,
            }
        }

        fn asset_desirability(&self, _genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
            if ctx.index == 0 && ctx.ticker_idx == 0 {
                4.0
            } else {
                0.0
            }
        }

        fn cash_desirability(&self, _genome: &Self::Genome, _contexts: &[DecisionContext]) -> f64 {
            0.0
        }
    }

    fn bars(base: f64) -> Vec<Bar> {
        (0..8)
            .map(|index| Bar {
                date: OffsetDateTime::UNIX_EPOCH + Duration::minutes(index as i64 * 5),
                open: base + index as f64,
                high: base + index as f64 + 1.0,
                low: base + index as f64 - 1.0,
                close: base + index as f64,
                volume: 1_000.0,
                wap: base + index as f64,
                count: 1,
            })
            .collect()
    }

    #[test]
    fn family_target_simplex_can_exceed_old_equal_weight_cap() {
        let market = MarketDataset::new(
            "train",
            vec!["AAA".to_string(), "BBB".to_string()],
            vec![bars(100.0), bars(80.0)],
        );
        let outcome = evaluate_family(&ConcentratedFamily, &ConcentratedGenome, &market, true);
        let trace = outcome.trace.expect("expected trace");
        let first_ticker_position = trace.positioned_by_ticker[0][1];
        let old_equal_weight_cap = STARTING_CASH / market.tickers.len() as f64;

        assert!(
            first_ticker_position > old_equal_weight_cap,
            "expected concentrated allocation above old equal-weight cap: {first_ticker_position} <= {old_equal_weight_cap}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct TickerRuntimeState {
    local_minimum: f64,
    local_maximum: f64,
    last_buy_price: Option<f64>,
    last_sell_price: Option<f64>,
    lowest_rsi_since_flat: Option<f64>,
    highest_rsi_since_long: Option<f64>,
    position_open_benchmark_assets: Option<f64>,
    position_entry_index: Option<usize>,
    last_excess_return_since_entry_pct: Option<f64>,
    peak_excess_return_since_entry_pct: Option<f64>,
    excess_return_delta_pct: Option<f64>,
    underperformance_streak: usize,
}

impl TickerRuntimeState {
    fn new(initial_price: f64) -> Self {
        Self {
            local_minimum: initial_price,
            local_maximum: initial_price,
            last_buy_price: None,
            last_sell_price: None,
            lowest_rsi_since_flat: None,
            highest_rsi_since_long: None,
            position_open_benchmark_assets: None,
            position_entry_index: None,
            last_excess_return_since_entry_pct: None,
            peak_excess_return_since_entry_pct: None,
            excess_return_delta_pct: None,
            underperformance_streak: 0,
        }
    }

    fn observe(&mut self, price: f64, rsi: f64, in_position: bool) {
        self.local_minimum = self.local_minimum.min(price);
        self.local_maximum = self.local_maximum.max(price);
        if in_position {
            self.highest_rsi_since_long = Some(
                self.highest_rsi_since_long
                    .map(|current| current.max(rsi))
                    .unwrap_or(rsi),
            );
        } else {
            self.lowest_rsi_since_flat = Some(
                self.lowest_rsi_since_flat
                    .map(|current| current.min(rsi))
                    .unwrap_or(rsi),
            );
        }
    }

    fn record_buy(&mut self, price: f64, benchmark_assets: f64, was_flat: bool) {
        self.last_buy_price = Some(price);
        self.last_sell_price = None;
        self.lowest_rsi_since_flat = None;
        self.local_minimum = price;
        self.local_maximum = price;
        if was_flat {
            self.position_open_benchmark_assets = Some(benchmark_assets);
            self.position_entry_index = None;
            self.last_excess_return_since_entry_pct = None;
            self.peak_excess_return_since_entry_pct = None;
            self.excess_return_delta_pct = None;
            self.underperformance_streak = 0;
        }
    }

    fn record_sell(&mut self, price: f64, fully_closed: bool) {
        self.last_sell_price = Some(price);
        self.last_buy_price = None;
        self.highest_rsi_since_long = None;
        self.local_minimum = price;
        self.local_maximum = price;
        if fully_closed {
            self.position_open_benchmark_assets = None;
            self.position_entry_index = None;
            self.last_excess_return_since_entry_pct = None;
            self.peak_excess_return_since_entry_pct = None;
            self.excess_return_delta_pct = None;
            self.underperformance_streak = 0;
        }
    }

    fn benchmark_return_since_entry_pct(&self, benchmark_assets: f64) -> Option<f64> {
        self.position_open_benchmark_assets.map(|entry_benchmark_assets| {
            ((benchmark_assets / entry_benchmark_assets.max(f64::EPSILON)) - 1.0) * 100.0
        })
    }

    fn observe_relative_performance(&mut self, index: usize, excess_return_since_entry_pct: Option<f64>) {
        match excess_return_since_entry_pct {
            Some(excess_return_pct) => {
                if self.position_entry_index.is_none() {
                    self.position_entry_index = Some(index);
                }
                self.excess_return_delta_pct = Some(
                    excess_return_pct
                        - self
                            .last_excess_return_since_entry_pct
                            .unwrap_or(excess_return_pct),
                );
                self.peak_excess_return_since_entry_pct = Some(
                    self.peak_excess_return_since_entry_pct
                        .map(|current| current.max(excess_return_pct))
                        .unwrap_or(excess_return_pct),
                );
                self.last_excess_return_since_entry_pct = Some(excess_return_pct);
                if excess_return_pct < 0.0 {
                    self.underperformance_streak += 1;
                } else {
                    self.underperformance_streak = 0;
                }
            }
            None => {
                self.last_excess_return_since_entry_pct = None;
                self.excess_return_delta_pct = None;
                self.peak_excess_return_since_entry_pct = None;
                self.underperformance_streak = 0;
            }
        }
    }

    fn bars_since_entry(&self, index: usize) -> Option<usize> {
        self.position_entry_index.map(|entry_index| index.saturating_sub(entry_index))
    }
}
