use super::env::Env;

const REWARD_SCALE: f64 = 20.0;
const SHARPE_LAMBDA: f64 = 100.0;
const SORTINO_LAMBDA: f64 = 100.0;
const RISK_ADJUSTED_REWARD_LAMBDA: f64 = 0.01;
const COMMISSIONS_PENALTY_LAMBDA: f64 = 0.01;

const HINDSIGHT_REWARD_SCALE: f64 = 1.0;
const HINDSIGHT_COMMISSION_PENALTY: f64 = 10.0;

impl Env {
    /// Hindsight allocation quality reward.
    ///
    /// Measures what fraction of achievable outperformance the agent captured:
    /// - reward ∈ [-1, +1] based on allocation quality
    /// - +1 = 100% in best asset, -1 = 100% in worst asset, 0 = equal weight
    /// - Independent of market direction (up/down days don't matter)
    /// - Only measures: "did you overweight winners and underweight losers?"
    ///
    /// Eliminates market noise by normalizing by the achievable range.
    pub(super) fn get_hindsight_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return 0.0;
        }

        let next_step = absolute_step + 1;
        let n_tickers = self.tickers.len();

        // Compute next-step returns for each ticker
        let mut returns: Vec<f64> = Vec::with_capacity(n_tickers);
        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_step];
            returns.push((next_price / current_price) - 1.0);
        }

        // Get current portfolio weights (ticker weights only, excluding cash)
        let total_assets = self.account.total_assets;
        let mut weights: Vec<f64> = Vec::with_capacity(n_tickers);
        let mut total_invested = 0.0;
        for ticker_idx in 0..n_tickers {
            let value = self.account.positions[ticker_idx]
                .value_with_price(self.prices[ticker_idx][absolute_step]);
            weights.push(value);
            total_invested += value;
        }

        // Normalize weights to sum to 1 (among invested portion only)
        if total_invested < 1e-8 {
            // Penalty for holding all cash - missing opportunity
            let best_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            return -best_return.max(0.0) * HINDSIGHT_REWARD_SCALE;
        }

        for w in &mut weights {
            *w /= total_invested;
        }

        // Compute mean return (equal-weight benchmark)
        let mean_return: f64 = returns.iter().sum::<f64>() / n_tickers as f64;

        // Compute agent's weighted return
        let agent_return: f64 = weights.iter().zip(returns.iter()).map(|(w, r)| w * r).sum();

        // Compute best and worst possible returns (100% in single asset)
        let best_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);

        // Active return = how much better/worse than equal weight
        let active_return = agent_return - mean_return;

        // Achievable range from mean
        let max_upside = best_return - mean_return;
        let max_downside = mean_return - worst_return;

        // Normalize active return to [-1, +1]
        let allocation_quality = if active_return >= 0.0 {
            if max_upside > 1e-10 {
                (active_return / max_upside).min(1.0)
            } else {
                0.0
            }
        } else if max_downside > 1e-10 {
            -((-active_return) / max_downside).min(1.0)
        } else {
            0.0
        };

        // Commission penalty (relative to assets)
        let commission_penalty = -(commissions / total_assets) * HINDSIGHT_COMMISSION_PENALTY;

        // Cash drag: penalize for not being fully invested when market is up
        let cash_fraction = self.account.cash / total_assets;
        let cash_drag = if mean_return > 0.0 {
            -cash_fraction * mean_return * 5.0
        } else {
            cash_fraction * (-mean_return) * 2.0
        };

        (allocation_quality + commission_penalty + cash_drag) * HINDSIGHT_REWARD_SCALE
    }

    #[allow(dead_code)]
    pub(super) fn get_unrealized_pnl_reward(&self, absolute_step: usize, _commissions: f64) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;

            let pnl_reward = (total_assets_after_trade / self.account.total_assets).ln();
            pnl_reward * REWARD_SCALE
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    pub(super) fn get_index_benchmark_pnl_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let strategy_log_return = (total_assets_after_trade / self.account.total_assets).ln();

            let mut index_log_return = 0.0;
            for ticker_idx in 0..self.tickers.len() {
                let current_price = self.prices[ticker_idx][absolute_step];
                let next_price = self.prices[ticker_idx][next_absolute_step];
                index_log_return += (next_price / current_price).ln();
            }
            index_log_return /= self.tickers.len() as f64;

            let excess_return = strategy_log_return - index_log_return;

            let commissions_relative = commissions / self.account.total_assets;
            let commissions_penalty = -commissions_relative * COMMISSIONS_PENALTY_LAMBDA;

            (excess_return + commissions_penalty) * REWARD_SCALE
        } else {
            0.0
        }
    }

    #[deprecated]
    #[allow(dead_code)]
    fn get_excess_returns_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let strategy_log_return = (total_assets_after_trade / self.account.total_assets).ln();

            let current_price = self.prices[0][absolute_step];
            let next_price = self.prices[0][next_absolute_step];
            let buy_hold_log_return = (next_price / current_price).ln();

            (strategy_log_return - buy_hold_log_return) * REWARD_SCALE
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn get_sharpe_ratio_adjusted_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let step_return = (total_assets_after_trade / self.account.total_assets).ln();

            let sharpe_ratio = step_return - SHARPE_LAMBDA * step_return * step_return;
            sharpe_ratio * REWARD_SCALE
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn get_sortino_ratio_adjusted_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let step_return = (total_assets_after_trade / self.account.total_assets).ln();

            let downside = if step_return < 0.0 {
                step_return * step_return
            } else {
                0.0
            };
            let sortino_ratio = step_return - SORTINO_LAMBDA * downside;
            sortino_ratio * REWARD_SCALE
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn get_risk_adjusted_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let step_return = (total_assets_after_trade / self.account.total_assets).ln();

            let downside = if step_return < 0.0 {
                -step_return
            } else {
                0.0
            };
            let rar = step_return - RISK_ADJUSTED_REWARD_LAMBDA * downside;

            let commissions_relative = commissions / self.account.total_assets;
            let commisions_penalty = -commissions_relative * RISK_ADJUSTED_REWARD_LAMBDA;
            let reward = commisions_penalty + rar;

            reward * REWARD_SCALE
        } else {
            0.0
        }
    }
}
