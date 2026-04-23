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
    metrics::{compute_metrics, BacktestMetrics, TradeSummary, STARTING_CASH},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketDataset {
    pub split_name: String,
    pub tickers: Vec<String>,
    pub bars: MappedHistorical,
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
    let indexes = market.bars.first().map(|bars| bars.len()).unwrap_or(0);
    let indicator_cfg = family.indicator_config(genome);

    let prices_by_ticker: Vec<Vec<f64>> = market.bars.iter().map(convert_historical).collect();
    let decider_rsi_by_ticker = compute_rsi_map(&prices_by_ticker, indicator_cfg.decider_rsi_alpha);
    let amount_rsi_by_ticker = compute_rsi_map(&prices_by_ticker, indicator_cfg.amount_rsi_alpha);
    let price_ema_by_ticker = compute_ema_map(&prices_by_ticker, indicator_cfg.price_ema_alpha);
    let fast_ema_by_ticker = indicator_cfg
        .fast_ema_alpha
        .map(|alpha| compute_ema_map(&prices_by_ticker, alpha));
    let slow_ema_by_ticker = indicator_cfg
        .slow_ema_alpha
        .map(|alpha| compute_ema_map(&prices_by_ticker, alpha));

    let benchmark_assets = benchmark_curve(&market.bars);
    let mut account = Account::new(STARTING_CASH, market.bars.len());
    let mut total_assets = Vec::with_capacity(indexes);
    let mut cash_curve = Vec::with_capacity(indexes);
    let mut positioned_by_ticker = vec![Vec::with_capacity(indexes); market.bars.len()];
    let mut buys_by_ticker = vec![Vec::new(); market.bars.len()];
    let mut sells_by_ticker = vec![Vec::new(); market.bars.len()];
    let mut states = market
        .bars
        .iter()
        .map(|bars| TickerRuntimeState::new(bars[0].close))
        .collect::<Vec<_>>();
    let mut trade_summary = TradeSummary::default();

    for index in 0..indexes {
        let mut positioned_total = 0.0;
        for (ticker_idx, bars) in market.bars.iter().enumerate() {
            let price = bars[index].close;
            let positioned = account.positions[ticker_idx].value_with_price(price);
            positioned_by_ticker[ticker_idx].push(positioned);
            positioned_total += positioned;
        }

        let assets = account.cash + positioned_total;
        cash_curve.push(account.cash);
        total_assets.push(assets);

        for (ticker_idx, bars) in market.bars.iter().enumerate() {
            let price = bars[index].close;
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
                let fraction = family.sell_fraction(genome, &ctx).clamp(0.0, 1.0);
                if try_sell(
                    ticker_idx,
                    index,
                    price,
                    fraction,
                    &mut account,
                    &mut states,
                    &mut sells_by_ticker,
                    &mut trade_summary,
                ) {
                    continue;
                }
            }

            if family.allow_buy(genome, &ctx) {
                let fraction = family.buy_fraction(genome, &ctx).clamp(0.0, 1.0);
                let max_position_value = assets / market.bars.len() as f64;
                let buy_budget = account.cash.min(
                    (max_position_value - account.positions[ticker_idx].value_with_price(price))
                        .max(0.0)
                        * fraction,
                );
                if try_buy(
                    ticker_idx,
                    index,
                    price,
                    buy_budget,
                    assets,
                    &mut account,
                    &mut states,
                    &mut buys_by_ticker,
                    &mut trade_summary,
                ) {
                    continue;
                }
            }
        }
    }

    let metrics = compute_metrics(&total_assets, &benchmark_assets, trade_summary);
    let trace = capture_trace.then_some(BacktestTrace {
        total_assets,
        cash: cash_curve,
        benchmark_assets,
        prices_by_ticker,
        positioned_by_ticker,
        buys_by_ticker,
        sells_by_ticker,
    });

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
            title: format!("{split_name} {ticker} Assets"),
            x_label: Some("Step".to_string()),
            y_label: Some("Assets".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Assets {
                total: to_f32(&trace.total_assets),
                cash: to_f32(&trace.cash),
                positioned: Some(to_f32(&trace.positioned_by_ticker[ticker_idx])),
                benchmark: Some(to_f32(&trace.benchmark_assets)),
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
            &trace.total_assets,
            &trace.cash,
            Some(&trace.positioned_by_ticker[ticker_idx]),
            Some(&trace.benchmark_assets),
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

fn benchmark_curve(bars: &MappedHistorical) -> Vec<f64> {
    if bars.is_empty() {
        return vec![STARTING_CASH];
    }
    let allocation = STARTING_CASH / bars.len() as f64;
    let quantities: Vec<f64> = bars
        .iter()
        .map(|ticker_bars| allocation / ticker_bars[0].close)
        .collect();

    (0..bars[0].len())
        .map(|index| {
            quantities
                .iter()
                .enumerate()
                .map(|(ticker_idx, quantity)| quantity * bars[ticker_idx][index].close)
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
    buys_by_ticker: &mut [Vec<u32>],
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
    buys_by_ticker[ticker_idx].push(index as u32);
    trade_summary.turnover += total_buy;
    trade_summary.buy_count += 1;
    true
}

fn try_sell(
    ticker_idx: usize,
    index: usize,
    price: f64,
    sell_fraction: f64,
    account: &mut Account,
    states: &mut [TickerRuntimeState],
    sells_by_ticker: &mut [Vec<u32>],
    trade_summary: &mut TradeSummary,
) -> bool {
    let position = &mut account.positions[ticker_idx];
    if position.quantity <= 0.0 || position.avg_price >= price {
        return false;
    }
    let position_value = position.value_with_price(price);
    let sell_budget = position_value * sell_fraction;
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
    sells_by_ticker[ticker_idx].push(index as u32);
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
