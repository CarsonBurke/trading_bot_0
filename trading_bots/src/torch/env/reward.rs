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
    /// Measures what fraction of achievable return the agent captured, treating
    /// cash as an asset with 0% return:
    /// - reward ∈ [-1, +1] based on allocation quality
    /// - +1 = optimal (100% in best asset or cash)
    /// - -1 = worst (100% in worst asset or fully invested when should hold cash)
    /// - 0 = midpoint between best and worst achievable
    ///
    /// Works correctly for any number of tickers including single-ticker case.
    pub(super) fn get_hindsight_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return 0.0;
        }

        let next_step = absolute_step + 1;
        let n_tickers = self.tickers.len();
        let total_assets = self.account.total_assets;

        // Compute agent's actual return and best/worst achievable
        // Cash is implicitly an asset with 0% return
        let mut agent_return = 0.0;
        let mut best_return: f64 = 0.0; // Can always hold cash
        let mut worst_return: f64 = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_step];
            let ticker_return = (next_price / current_price) - 1.0;

            // Agent's weighted contribution from this ticker
            let position_value = self.account.positions[ticker_idx]
                .value_with_price(current_price);
            let weight = position_value / total_assets;
            agent_return += weight * ticker_return;

            // Track best/worst (cash=0 is always an option)
            best_return = best_return.max(ticker_return);
            worst_return = worst_return.min(ticker_return);
        }
        // Cash weight contributes 0 to agent_return (weight * 0)

        // Achievable range
        let range = best_return - worst_return;
        if range < 1e-10 {
            // All returns ~equal, no meaningful allocation decision
            let commission_penalty = -(commissions / total_assets) * HINDSIGHT_COMMISSION_PENALTY;
            return commission_penalty * HINDSIGHT_REWARD_SCALE;
        }

        // Normalize: -1 at worst_return, +1 at best_return
        let allocation_quality = (2.0 * (agent_return - worst_return) / range - 1.0).clamp(-1.0, 1.0);

        // Commission penalty
        let commission_penalty = -(commissions / total_assets) * HINDSIGHT_COMMISSION_PENALTY;

        (allocation_quality + commission_penalty) * HINDSIGHT_REWARD_SCALE
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
