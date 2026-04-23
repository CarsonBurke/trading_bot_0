use std::path::Path;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

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
        let mut positioned_total = 0.0;
        for ticker_idx in 0..ticker_count {
            let price = market.close_prices[ticker_idx][index];
            let positioned = account.positions[ticker_idx].value_with_price(price);
            if let Some(positioned_by_ticker) = positioned_by_ticker.as_mut() {
                positioned_by_ticker[ticker_idx].push(positioned);
            }
            positioned_total += positioned;
        }

        let assets = account.cash + positioned_total;
        metric_accumulator.observe(assets);
        if let Some(cash_curve) = cash_curve.as_mut() {
            cash_curve.push(account.cash);
        }
        if let Some(total_assets) = total_assets.as_mut() {
            total_assets.push(assets);
        }

        for ticker_idx in 0..ticker_count {
            let price = market.close_prices[ticker_idx][index];
            let position = account.positions[ticker_idx];
            let state = &mut states[ticker_idx];
            state.observe(
                price,
                decider_rsi_by_ticker[ticker_idx][index],
                position.quantity > 0.0,
            );

            let ctx = DecisionContext {
                index,
                ticker_count: market.bars.len(),
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
                position_value: position.value_with_price(price),
                position_avg_price: position.avg_price,
                position_quantity: position.quantity,
                assets,
                cash: account.cash,
            };

            if family.allow_sell(genome, &ctx) {
                let sell_budget = family
                    .sell_budget(genome, &ctx)
                    .clamp(0.0, ctx.position_value);
                let sold = if let Some(sells_by_ticker) = sells_by_ticker.as_mut() {
                    try_sell(
                        ticker_idx,
                        index,
                        price,
                        sell_budget,
                        &mut account,
                        &mut states,
                        Some(&mut sells_by_ticker[ticker_idx]),
                        &mut trade_summary,
                    )
                } else {
                    try_sell(
                        ticker_idx,
                        index,
                        price,
                        sell_budget,
                        &mut account,
                        &mut states,
                        None,
                        &mut trade_summary,
                    )
                };
                if sold {
                    continue;
                }
            }

            if family.allow_buy(genome, &ctx) {
                let buy_budget = family.buy_budget(genome, &ctx).clamp(0.0, ctx.cash);
                let bought = if let Some(buys_by_ticker) = buys_by_ticker.as_mut() {
                    try_buy(
                        ticker_idx,
                        index,
                        price,
                        buy_budget,
                        assets,
                        &mut account,
                        &mut states,
                        Some(&mut buys_by_ticker[ticker_idx]),
                        &mut trade_summary,
                    )
                } else {
                    try_buy(
                        ticker_idx,
                        index,
                        price,
                        buy_budget,
                        assets,
                        &mut account,
                        &mut states,
                        None,
                        &mut trade_summary,
                    )
                };
                if bought {
                    continue;
                }
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

    for (ticker_idx, ticker) in tickers.iter().enumerate() {
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
    }

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

fn try_buy(
    ticker_idx: usize,
    index: usize,
    price: f64,
    buy_budget: f64,
    assets: f64,
    account: &mut Account,
    states: &mut [TickerRuntimeState],
    buy_markers: Option<&mut Vec<u32>>,
    trade_summary: &mut TradeSummary,
) -> bool {
    if !is_min_transaction(assets, buy_budget) {
        return false;
    }
    let (total_buy, quantity) = round_to_stock_fractional(price, buy_budget);
    if quantity <= 0.0 {
        return false;
    }

    account.positions[ticker_idx].add(price, quantity);
    account.cash -= total_buy;
    states[ticker_idx].record_buy(price);
    if let Some(buy_markers) = buy_markers {
        buy_markers.push(index as u32);
    }
    trade_summary.turnover += total_buy;
    trade_summary.buy_count += 1;
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
    if position.quantity <= 0.0 || position.avg_price >= price {
        return false;
    }
    let position_value = position.value_with_price(price);
    let (total_sell, quantity) = round_to_stock_fractional(price, sell_budget.min(position_value));
    if quantity <= 0.0 {
        return false;
    }

    let profitable = price > position.avg_price;
    position.quantity -= quantity;
    if position.quantity <= 1e-9 {
        position.quantity = 0.0;
        position.avg_price = 0.0;
    }
    account.cash += total_sell;
    states[ticker_idx].record_sell(price);
    if let Some(sell_markers) = sell_markers {
        sell_markers.push(index as u32);
    }
    trade_summary.turnover += total_sell;
    trade_summary.sell_count += 1;
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

        fn allow_buy(&self, _genome: &Self::Genome, ctx: &DecisionContext) -> bool {
            ctx.index == 0 && ctx.position_quantity <= 0.0
        }

        fn allow_sell(&self, _genome: &Self::Genome, _ctx: &DecisionContext) -> bool {
            false
        }

        fn buy_budget(&self, _genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
            ctx.cash * 0.8
        }

        fn sell_budget(&self, _genome: &Self::Genome, _ctx: &DecisionContext) -> f64 {
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
    fn family_buy_budget_can_exceed_old_equal_weight_cap() {
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

    fn record_buy(&mut self, price: f64) {
        self.last_buy_price = Some(price);
        self.last_sell_price = None;
        self.lowest_rsi_since_flat = None;
        self.local_minimum = price;
        self.local_maximum = price;
    }

    fn record_sell(&mut self, price: f64) {
        self.last_sell_price = Some(price);
        self.last_buy_price = None;
        self.highest_rsi_since_long = None;
        self.local_minimum = price;
        self.local_maximum = price;
    }
}
