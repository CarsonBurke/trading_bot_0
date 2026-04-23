use serde::{Deserialize, Serialize};

pub const STARTING_CASH: f64 = 10_000.0;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub score: f64,
    pub final_assets: f64,
    pub return_pct: f64,
    pub benchmark_return_pct: f64,
    pub outperformance_pct: f64,
    pub max_drawdown_pct: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub turnover: f64,
    pub trade_count: usize,
    pub buy_count: usize,
    pub sell_count: usize,
    pub win_rate: f64,
    pub total_commissions: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TradeSummary {
    pub turnover: f64,
    pub buy_count: usize,
    pub sell_count: usize,
    pub profitable_sells: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PopulationStats {
    pub p05: f64,
    pub p50: f64,
    pub p95: f64,
    pub worst: f64,
    pub best: f64,
}

pub fn compute_metrics(
    total_assets: &[f64],
    benchmark_assets: &[f64],
    trades: TradeSummary,
) -> BacktestMetrics {
    let final_assets = total_assets.last().copied().unwrap_or(STARTING_CASH);
    let benchmark_final = benchmark_assets.last().copied().unwrap_or(STARTING_CASH);
    let return_pct = pct_change(STARTING_CASH, final_assets);
    let benchmark_return_pct = pct_change(STARTING_CASH, benchmark_final);
    let outperformance_pct = return_pct - benchmark_return_pct;
    let max_drawdown_pct = max_drawdown_pct(total_assets);
    let sharpe = sharpe_like(total_assets);
    let sortino = sortino_like(total_assets);
    let calmar = if max_drawdown_pct <= f64::EPSILON {
        return_pct.max(0.0)
    } else {
        return_pct / max_drawdown_pct
    };
    let trade_count = trades.buy_count + trades.sell_count;
    let win_rate = if trades.sell_count == 0 {
        0.0
    } else {
        trades.profitable_sells as f64 / trades.sell_count as f64
    };

    let score = outperformance_pct + 0.35 * return_pct - 1.15 * max_drawdown_pct
        + 10.0 * sharpe.clamp(-2.0, 2.0)
        + 6.0 * sortino.clamp(-2.0, 2.0)
        + 4.0 * calmar.clamp(-2.0, 2.0)
        - 0.01 * trade_count as f64;

    BacktestMetrics {
        score,
        final_assets,
        return_pct,
        benchmark_return_pct,
        outperformance_pct,
        max_drawdown_pct,
        sharpe,
        sortino,
        calmar,
        turnover: trades.turnover,
        trade_count,
        buy_count: trades.buy_count,
        sell_count: trades.sell_count,
        win_rate,
        total_commissions: 0.0,
    }
}

pub fn population_stats(mut values: Vec<f64>) -> PopulationStats {
    if values.is_empty() {
        return PopulationStats::default();
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    PopulationStats {
        p05: percentile(&values, 0.05),
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        worst: *values.first().unwrap_or(&0.0),
        best: *values.last().unwrap_or(&0.0),
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn pct_change(start: f64, end: f64) -> f64 {
    if start.abs() <= f64::EPSILON {
        0.0
    } else {
        ((end / start) - 1.0) * 100.0
    }
}

fn max_drawdown_pct(assets: &[f64]) -> f64 {
    let mut peak = STARTING_CASH;
    let mut max_drawdown = 0.0_f64;
    for &asset in assets {
        peak = peak.max(asset);
        if peak > 0.0 {
            let drawdown = ((peak - asset) / peak) * 100.0;
            max_drawdown = max_drawdown.max(drawdown);
        }
    }
    max_drawdown
}

fn sharpe_like(assets: &[f64]) -> f64 {
    let returns = step_returns(assets);
    ratio(&returns)
}

fn sortino_like(assets: &[f64]) -> f64 {
    let returns = step_returns(assets);
    let downside: Vec<f64> = returns.iter().copied().filter(|ret| *ret < 0.0).collect();
    if downside.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside_std = std_dev(&downside);
    if downside_std <= f64::EPSILON {
        0.0
    } else {
        mean / downside_std
    }
}

fn step_returns(assets: &[f64]) -> Vec<f64> {
    assets
        .windows(2)
        .filter_map(|window| {
            let previous = window[0];
            let next = window[1];
            if previous.abs() <= f64::EPSILON {
                None
            } else {
                Some((next / previous) - 1.0)
            }
        })
        .collect()
}

fn ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let std = std_dev(returns);
    if std <= f64::EPSILON {
        0.0
    } else {
        mean / std
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}
