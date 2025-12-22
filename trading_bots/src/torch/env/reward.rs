use super::env::Env;

// === Action-outcome reward ===
// Rewards the model based on how well its allocation decisions aligned with
// subsequent price movements. Unlike hindsight reward (which compares to optimal),
// this rewards the model for its actual bets paying off.
const ACTION_OUTCOME_SCALE: f64 = 100.0;
const ACTION_OUTCOME_COMMISSION_PENALTY: f64 = 5.0;
// Conviction bonus: rewards concentrated bets that pay off.
// Conviction = how far from uniform allocation (measured by max weight deviation).
// Only applies when portfolio return is positive - wrong conviction is already
// penalized by negative base return.
const CONVICTION_BONUS: f64 = 1.0;

const REWARD_SCALE: f64 = 20.0;
const CASH_INFLATION_RATE: f64 = 0.0002;
const SHARPE_LAMBDA: f64 = 100.0;
const SORTINO_LAMBDA: f64 = 100.0;
const RISK_ADJUSTED_REWARD_LAMBDA: f64 = 0.01;
const COMMISSIONS_PENALTY_LAMBDA: f64 = 0.01;

const HINDSIGHT_REWARD_SCALE: f64 = 1.0;
const HINDSIGHT_COMMISSION_PENALTY: f64 = 10.0;
// Asymmetric scaling: in bull markets (best_return > 0), scale negative
// allocation_quality more harshly. This breaks the symmetry where bear market
// protection (+1) dominates over bull market opportunity cost (-1).
// 1.0 = symmetric (original), >1.0 = penalize missing upside more
const BULL_DOWNSIDE_SCALE: f64 = 1.5;

impl Env {
    pub(super) fn get_unrealized_pnl_reward_breakdown(
        &self,
        absolute_step: usize,
        commissions: f64,
    ) -> (f64, Vec<f64>) {
        let n_tickers = self.tickers.len();
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return (0.0, vec![0.0; n_tickers + 1]);
        }

        let next_absolute_step = absolute_step + 1;
        let total_assets = self.account.total_assets;
        let mut contributions = vec![0.0; n_tickers];
        let mut total_assets_next = self.account.cash;
        let mut index_log_return = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_absolute_step];
            let current_value = self.account.positions[ticker_idx].value_with_price(current_price);
            let next_value = self.account.positions[ticker_idx].value_with_price(next_price);
            total_assets_next += next_value;
            contributions[ticker_idx] = (next_value - current_value) / total_assets;
            index_log_return += (next_price / current_price).ln();
        }

        index_log_return /= n_tickers as f64;

        let portfolio_return = (total_assets_next - total_assets) / total_assets;
        let strategy_log_return = (total_assets_next / total_assets).ln();
        let pnl_reward = strategy_log_return * REWARD_SCALE;
        let reward_base = (strategy_log_return - index_log_return) * REWARD_SCALE;
        let cash_weight = (self.account.cash / total_assets).clamp(0.0, 1.0);
        let cash_penalty = -cash_weight * CASH_INFLATION_RATE * REWARD_SCALE;
        let reward = reward_base + cash_penalty;

        if portfolio_return.abs() < 1e-8 {
            let mut rewards = vec![0.0; n_tickers];
            rewards.push(reward);
            return (reward, rewards);
        }

        let per_ticker_rewards: Vec<f64> = contributions
            .iter()
            .map(|c| pnl_reward * (c / portfolio_return))
            .collect();
        let sum_per_ticker: f64 = per_ticker_rewards.iter().sum();
        let cash_reward = reward - sum_per_ticker;
        let mut rewards = per_ticker_rewards;
        rewards.push(cash_reward);
        (reward, rewards)
    }

    /// Hindsight allocation quality reward with asymmetric upside penalty.
    ///
    /// Measures what fraction of achievable return the agent captured, treating
    /// cash as an asset with 0% return. Applies extra penalty for missing upside
    /// in bull markets to prevent cash-hoarding local optima.
    ///
    /// - Base reward ∈ [-1, +1] based on allocation quality
    /// - Additional penalty when best_return > 0 and agent missed it
    pub(super) fn get_hindsight_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return 0.0;
        }

        let next_step = absolute_step + 1;
        let n_tickers = self.tickers.len();
        let total_assets = self.account.total_assets;

        let mut agent_return = 0.0;
        let mut best_return: f64 = 0.0;
        let mut worst_return: f64 = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_step];
            let ticker_return = (next_price / current_price) - 1.0;

            let position_value = self.account.positions[ticker_idx]
                .value_with_price(current_price);
            let weight = position_value / total_assets;
            agent_return += weight * ticker_return;

            best_return = best_return.max(ticker_return);
            worst_return = worst_return.min(ticker_return);
        }

        let range = best_return - worst_return;
        if range < 1e-10 {
            let commission_penalty = -(commissions / total_assets) * HINDSIGHT_COMMISSION_PENALTY;
            return commission_penalty * HINDSIGHT_REWARD_SCALE;
        }

        // Base allocation quality: [-1, +1]
        let allocation_quality = (2.0 * (agent_return - worst_return) / range - 1.0).clamp(-1.0, 1.0);

        // Asymmetric scaling: in bull markets, amplify negative allocation quality
        // This makes missing upside hurt more than capturing downside protection helps
        // Reward remains in [-1.5, +1] range instead of [-1, +1]
        let scaled_quality = if best_return > 0.0 && allocation_quality < 0.0 {
            allocation_quality * BULL_DOWNSIDE_SCALE
        } else {
            allocation_quality
        };

        let commission_penalty = -(commissions / total_assets) * HINDSIGHT_COMMISSION_PENALTY;

        (scaled_quality + commission_penalty) * HINDSIGHT_REWARD_SCALE
    }

    /// Action-outcome reward: rewards based on how the model's allocation performed.
    ///
    /// Key differences from hindsight reward:
    /// - Hindsight compares to optimal allocation (what you should have done)
    /// - Action-outcome rewards actual portfolio return (what your bets earned)
    ///
    /// This creates a direct link between allocation decisions and outcomes:
    /// - Positive weight on rising asset → positive reward
    /// - Positive weight on falling asset → negative reward
    /// - Cash position → zero contribution (neutral, not rewarded or penalized)
    ///
    /// The reward is the weighted sum of per-asset returns, scaled and with
    /// commission penalty. Dense signal at every step.
    pub(super) fn get_action_outcome_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return 0.0;
        }

        let next_step = absolute_step + 1;
        let total_assets = self.account.total_assets;
        let n_tickers = self.tickers.len();

        // Portfolio return based on current positions
        let mut portfolio_return = 0.0;
        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_step];
            let ticker_return = (next_price / current_price) - 1.0;

            let position_value = self.account.positions[ticker_idx]
                .value_with_price(current_price);
            let weight = position_value / total_assets;
            portfolio_return += weight * ticker_return;
        }
        // Cash contributes 0 return (weight * 0)

        // Log-return: better for compounding, symmetric around zero
        // Guard against extreme values that could cause NaN
        let clamped_return = portfolio_return.clamp(-0.5, 0.5);
        let log_return = (1.0 + clamped_return).ln();
        let base_reward = log_return * ACTION_OUTCOME_SCALE;

        // Conviction: deviation from uniform allocation
        // For N tickers + cash, uniform = 1/(N+1) each
        // Measured as sum of squared deviations from uniform (Herfindahl-style)
        let n_assets = (n_tickers + 1) as f64;
        let uniform_weight = 1.0 / n_assets;
        let cash_weight = (self.account.cash / total_assets).clamp(0.0, 1.0);

        let mut concentration = 0.0;
        for ticker_idx in 0..n_tickers {
            let w = (self.account.positions[ticker_idx].value_with_price(
                self.prices[ticker_idx][absolute_step]
            ) / total_assets).clamp(0.0, 1.0);
            concentration += (w - uniform_weight).powi(2);
        }
        concentration += (cash_weight - uniform_weight).powi(2);

        // Normalize: max possible is (n-1)/n when all in one asset
        let max_concentration = (n_assets - 1.0) / n_assets;
        let conviction = if max_concentration > 1e-6 {
            (concentration / max_concentration).sqrt().min(1.0)
        } else {
            0.0
        };

        // Conviction bonus only when profitable - amplifies good concentrated bets
        let conviction_bonus = if portfolio_return > 0.0 {
            conviction * log_return.abs() * ACTION_OUTCOME_SCALE * CONVICTION_BONUS
        } else {
            0.0
        };

        let commission_penalty = -(commissions / total_assets) * ACTION_OUTCOME_COMMISSION_PENALTY;

        base_reward + conviction_bonus + commission_penalty
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
