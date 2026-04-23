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
    pub spread: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct BacktestMetricAccumulator {
    previous_asset: Option<f64>,
    final_assets: f64,
    peak_assets: f64,
    max_drawdown_pct: f64,
    return_sum: f64,
    return_sum_sq: f64,
    return_count: usize,
    downside_sum: f64,
    downside_sum_sq: f64,
    downside_count: usize,
}

impl Default for BacktestMetricAccumulator {
    fn default() -> Self {
        Self {
            previous_asset: None,
            final_assets: STARTING_CASH,
            peak_assets: STARTING_CASH,
            max_drawdown_pct: 0.0,
            return_sum: 0.0,
            return_sum_sq: 0.0,
            return_count: 0,
            downside_sum: 0.0,
            downside_sum_sq: 0.0,
            downside_count: 0,
        }
    }
}

impl BacktestMetricAccumulator {
    pub fn observe(&mut self, asset: f64) {
        if let Some(previous) = self.previous_asset {
            if previous.abs() > f64::EPSILON {
                let step_return = (asset / previous) - 1.0;
                self.return_sum += step_return;
                self.return_sum_sq += step_return * step_return;
                self.return_count += 1;
                if step_return < 0.0 {
                    self.downside_sum += step_return;
                    self.downside_sum_sq += step_return * step_return;
                    self.downside_count += 1;
                }
            }
        }

        self.peak_assets = self.peak_assets.max(asset);
        if self.peak_assets > 0.0 {
            let drawdown = ((self.peak_assets - asset) / self.peak_assets) * 100.0;
            self.max_drawdown_pct = self.max_drawdown_pct.max(drawdown);
        }
        self.final_assets = asset;
        self.previous_asset = Some(asset);
    }

    pub fn finish(self, benchmark_final: f64, trades: TradeSummary) -> BacktestMetrics {
        let return_pct = pct_change(STARTING_CASH, self.final_assets);
        let benchmark_return_pct = pct_change(STARTING_CASH, benchmark_final);
        let outperformance_pct = return_pct - benchmark_return_pct;
        let sharpe = ratio_from_moments(self.return_sum, self.return_sum_sq, self.return_count);
        let mean_return = if self.return_count == 0 {
            0.0
        } else {
            self.return_sum / self.return_count as f64
        };
        let downside_std =
            std_dev_from_moments(self.downside_sum, self.downside_sum_sq, self.downside_count);
        let sortino = if downside_std <= f64::EPSILON {
            0.0
        } else {
            mean_return / downside_std
        };
        let calmar = if self.max_drawdown_pct <= f64::EPSILON {
            return_pct.max(0.0)
        } else {
            return_pct / self.max_drawdown_pct
        };
        let trade_count = trades.buy_count + trades.sell_count;
        let win_rate = if trades.sell_count == 0 {
            0.0
        } else {
            trades.profitable_sells as f64 / trades.sell_count as f64
        };
        let score = outperformance_pct + 0.35 * return_pct - 1.15 * self.max_drawdown_pct
            + 10.0 * sharpe.clamp(-2.0, 2.0)
            + 6.0 * sortino.clamp(-2.0, 2.0)
            + 4.0 * calmar.clamp(-2.0, 2.0)
            - 0.01 * trade_count as f64;

        BacktestMetrics {
            score,
            final_assets: self.final_assets,
            return_pct,
            benchmark_return_pct,
            outperformance_pct,
            max_drawdown_pct: self.max_drawdown_pct,
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
}

pub fn population_stats(mut values: Vec<f64>) -> PopulationStats {
    if values.is_empty() {
        return PopulationStats::default();
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let p05 = percentile(&values, 0.05);
    let p50 = percentile(&values, 0.50);
    let p95 = percentile(&values, 0.95);
    PopulationStats {
        p05,
        p50,
        p95,
        worst: *values.first().unwrap_or(&0.0),
        best: *values.last().unwrap_or(&0.0),
        spread: p95 - p05,
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

fn ratio_from_moments(sum: f64, sum_sq: f64, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let mean = sum / count as f64;
    let std = std_dev_from_moments(sum, sum_sq, count);
    if std <= f64::EPSILON {
        0.0
    } else {
        mean / std
    }
}

fn std_dev_from_moments(sum: f64, sum_sq: f64, count: usize) -> f64 {
    if count < 2 {
        return 0.0;
    }
    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - (mean * mean);
    variance.max(0.0).sqrt()
}
