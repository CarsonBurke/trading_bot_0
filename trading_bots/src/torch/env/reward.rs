use super::env::Env;
use crate::types::Position;

const REWARD_SCALE: f64 = 20.0;
const OPPORTUNITY_COST_ONLY: bool = true;

impl Env {
    pub fn get_counterfactual_reward_breakdown(
        &self,
        absolute_step: usize,
        pre_total_assets: f64,
        _pre_cash: f64,
        _pre_positions: &[Position],
    ) -> (f64, Vec<f64>, f64) {
        let n_tickers = self.tickers.len();
        if self.step + 1 >= self.max_step || pre_total_assets <= 0.0 {
            return (0.0, vec![0.0; n_tickers], 0.0);
        }

        let next_absolute_step = absolute_step + 1;
        let mut actual_next = self.account.cash;

        for ticker_idx in 0..n_tickers {
            let next_price = self.prices[ticker_idx][next_absolute_step];
            actual_next += self.account.positions[ticker_idx].value_with_price(next_price);
        }

        if actual_next <= 0.0 {
            return (0.0, vec![0.0; n_tickers], 0.0);
        }
        let actual_return = (actual_next / pre_total_assets).ln();
        let mut per_ticker_rewards = vec![0.0; n_tickers];
        let mut miss_sum = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_absolute_step];
            let ticker_return = (next_price / current_price).ln();
            let counterfactual_return = ticker_return.max(0.0);
            let mut miss = counterfactual_return - actual_return;
            if OPPORTUNITY_COST_ONLY {
                miss = miss.max(0.0);
            }
            miss_sum += miss;
            per_ticker_rewards[ticker_idx] = -miss * REWARD_SCALE;
        }

        let reward = if n_tickers > 0 {
            -(miss_sum / n_tickers as f64) * REWARD_SCALE
        } else {
            0.0
        };

        (reward, per_ticker_rewards, 0.0)
    }

    #[allow(dead_code)]
    pub fn get_shadow_benchmark_reward_breakdown(
        &self,
        absolute_step: usize,
        _: f64,
    ) -> (f64, Vec<f64>, f64) {
        let n_tickers = self.tickers.len();
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return (0.0, vec![0.0; n_tickers], 0.0);
        }

        let next_absolute_step = absolute_step + 1;
        let total_assets = self.account.total_assets;
        let inv_total_assets = 1.0 / total_assets;
        let inv_n_tickers = 1.0 / n_tickers as f64;
        let mut contributions = vec![0.0; n_tickers];
        let mut total_assets_next = self.account.cash;
        let mut index_log_return = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_absolute_step];
            let position = &self.account.positions[ticker_idx];
            let current_value = position.value_with_price(current_price);
            let next_value = position.value_with_price(next_price);
            total_assets_next += next_value;
            contributions[ticker_idx] = (next_value - current_value) * inv_total_assets;
            index_log_return += (next_price / current_price).ln();
        }

        index_log_return *= inv_n_tickers;
        let strategy_log_return = (total_assets_next * inv_total_assets).ln() * REWARD_SCALE;
        let cash_weight = (self.account.cash * inv_total_assets).clamp(0.0, 1.0);
        let cash_reward = -cash_weight * index_log_return * REWARD_SCALE;

        let portfolio_return: f64 = contributions.iter().sum();
        let per_ticker_rewards: Vec<f64> = if portfolio_return.abs() < 1e-8 {
            vec![0.0; n_tickers]
        } else {
            let inv_portfolio_return = 1.0 / portfolio_return;
            contributions
                .iter()
                .map(|c| strategy_log_return * (c * inv_portfolio_return))
                .collect()
        };

        (strategy_log_return, per_ticker_rewards, cash_reward)
    }

    #[allow(dead_code)]
    pub fn get_hybrid_reward_breakdown(
        &self,
        absolute_step: usize,
        _: f64,
    ) -> (f64, Vec<f64>, f64) {
        let n_tickers = self.tickers.len();
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return (0.0, vec![0.0; n_tickers], 0.0);
        }

        let next_absolute_step = absolute_step + 1;
        let total_assets = self.account.total_assets;
        let inv_total_assets = 1.0 / total_assets;
        let inv_n_tickers = 1.0 / n_tickers as f64;
        let mut contributions = vec![0.0; n_tickers];
        let mut total_assets_next = self.account.cash;
        let mut index_log_return = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_absolute_step];
            let position = &self.account.positions[ticker_idx];
            let current_value = position.value_with_price(current_price);
            let next_value = position.value_with_price(next_price);
            total_assets_next += next_value;
            contributions[ticker_idx] = (next_value - current_value) * inv_total_assets;
            index_log_return += (next_price / current_price).ln();
        }

        let portfolio_return: f64 = contributions.iter().sum();
        let strategy_log_return = (total_assets_next * inv_total_assets).ln();
        index_log_return *= inv_n_tickers;

        let base_reward = REWARD_SCALE * strategy_log_return;

        let cash_weight = (self.account.cash * inv_total_assets).clamp(0.0, 1.0);
        let cash_penalty = -cash_weight * index_log_return.max(0.0) * REWARD_SCALE;

        let per_ticker_rewards: Vec<f64> = if portfolio_return.abs() < 1e-8 {
            vec![0.0; n_tickers]
        } else {
            let inv_portfolio_return = 1.0 / portfolio_return;
            contributions
                .iter()
                .map(|c| base_reward * (c * inv_portfolio_return))
                .collect()
        };

        (base_reward, per_ticker_rewards, cash_penalty)
    }

    #[allow(dead_code)]
    pub fn get_cash_upswing_penalty_reward_breakdown(
        &self,
        absolute_step: usize,
        _: f64,
    ) -> (f64, Vec<f64>, f64) {
        let n_tickers = self.tickers.len();
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return (0.0, vec![0.0; n_tickers], 0.0);
        }

        let next_absolute_step = absolute_step + 1;
        let total_assets = self.account.total_assets;
        let inv_total_assets = 1.0 / total_assets;
        let inv_n_tickers = 1.0 / n_tickers as f64;
        let mut contributions = vec![0.0; n_tickers];
        let mut total_assets_next = self.account.cash;
        let mut index_log_return = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_absolute_step];
            let position = &self.account.positions[ticker_idx];
            let current_value = position.value_with_price(current_price);
            let next_value = position.value_with_price(next_price);
            total_assets_next += next_value;
            contributions[ticker_idx] = (next_value - current_value) * inv_total_assets;
            index_log_return += (next_price / current_price).ln();
        }

        let portfolio_return: f64 = contributions.iter().sum();
        let strategy_log_return = (total_assets_next * inv_total_assets).ln() * REWARD_SCALE;
        index_log_return *= inv_n_tickers;

        let cash_weight = (self.account.cash * inv_total_assets).clamp(0.0, 1.0);
        let cash_penalty = -cash_weight * index_log_return.max(0.0) * REWARD_SCALE;

        let per_ticker_rewards: Vec<f64> = if portfolio_return.abs() < 1e-8 {
            vec![0.0; n_tickers]
        } else {
            let inv_portfolio_return = 1.0 / portfolio_return;
            contributions
                .iter()
                .map(|c| strategy_log_return * (c * inv_portfolio_return))
                .collect()
        };

        (strategy_log_return, per_ticker_rewards, cash_penalty)
    }

    #[allow(dead_code)]
    pub fn get_unrealized_pnl_reward_breakdown(
        &self,
        absolute_step: usize,
        _: f64,
    ) -> (f64, Vec<f64>) {
        let n_tickers = self.tickers.len();
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return (0.0, vec![0.0; n_tickers]);
        }

        let next_absolute_step = absolute_step + 1;
        let total_assets = self.account.total_assets;
        let inv_total_assets = 1.0 / total_assets;
        let mut contributions = vec![0.0; n_tickers];
        let mut total_assets_next = self.account.cash;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_absolute_step];
            let position = &self.account.positions[ticker_idx];
            let current_value = position.value_with_price(current_price);
            let next_value = position.value_with_price(next_price);
            total_assets_next += next_value;
            contributions[ticker_idx] = (next_value - current_value) * inv_total_assets;
        }

        let portfolio_return: f64 = contributions.iter().sum();
        let strategy_log_return = (total_assets_next * inv_total_assets).ln() * REWARD_SCALE;

        let per_ticker_rewards: Vec<f64> = if portfolio_return.abs() < 1e-8 {
            vec![0.0; n_tickers]
        } else {
            let inv_portfolio_return = 1.0 / portfolio_return;
            contributions
                .iter()
                .map(|c| strategy_log_return * (c * inv_portfolio_return))
                .collect()
        };
        (strategy_log_return, per_ticker_rewards)
    }

    #[allow(dead_code)]
    pub fn get_unrealized_pnl_reward(&self, absolute_step: usize, _commissions: f64) -> f64 {
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
    const HINDSIGHT_REWARD_SCALE: f64 = 1.0;
    #[allow(dead_code)]
    const HINDSIGHT_COMMISSION_PENALTY: f64 = 10.0;
    // Asymmetric scaling: in bull markets (best_return > 0), scale negative
    // allocation_quality more harshly. This breaks the symmetry where bear market
    // protection (+1) dominates over bull market opportunity cost (-1).
    // 1.0 = symmetric (original), >1.0 = penalize missing upside more
    #[allow(dead_code)]
    const BULL_DOWNSIDE_SCALE: f64 = 1.5;

    #[allow(dead_code)]
    pub fn get_hindsight_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return 0.0;
        }

        let next_step = absolute_step + 1;
        let n_tickers = self.tickers.len();
        let total_assets = self.account.total_assets;
        let inv_total_assets = 1.0 / total_assets;

        let mut agent_return = 0.0;
        let mut best_return: f64 = 0.0;
        let mut worst_return: f64 = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_step];
            let ticker_return = (next_price / current_price) - 1.0;

            let position_value = self.account.positions[ticker_idx].value_with_price(current_price);
            let weight = position_value * inv_total_assets;
            agent_return += weight * ticker_return;

            best_return = best_return.max(ticker_return);
            worst_return = worst_return.min(ticker_return);
        }

        let range = best_return - worst_return;
        if range < 1e-10 {
            let commission_penalty =
                -(commissions * inv_total_assets) * Self::HINDSIGHT_COMMISSION_PENALTY;
            return commission_penalty * Self::HINDSIGHT_REWARD_SCALE;
        }

        let inv_range = 1.0 / range;
        let allocation_quality =
            (2.0 * (agent_return - worst_return) * inv_range - 1.0).clamp(-1.0, 1.0);

        let scaled_quality = if best_return > 0.0 && allocation_quality < 0.0 {
            allocation_quality * Self::BULL_DOWNSIDE_SCALE
        } else {
            allocation_quality
        };

        let commission_penalty =
            -(commissions * inv_total_assets) * Self::HINDSIGHT_COMMISSION_PENALTY;

        (scaled_quality + commission_penalty) * Self::HINDSIGHT_REWARD_SCALE
    }

    #[allow(dead_code)]
    const ACTION_OUTCOME_SCALE: f64 = 100.0;
    #[allow(dead_code)]
    const ACTION_OUTCOME_COMMISSION_PENALTY: f64 = 5.0;
    // Conviction bonus: rewards concentrated bets that pay off.
    // Conviction = how far from uniform allocation (measured by max weight deviation).
    // Only applies when portfolio return is positive - wrong conviction is already
    // penalized by negative base return.
    #[allow(dead_code)]
    const CONVICTION_BONUS: f64 = 1.0;

    #[allow(dead_code)]
    pub fn get_action_outcome_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 >= self.max_step || self.account.total_assets <= 0.0 {
            return 0.0;
        }

        let next_step = absolute_step + 1;
        let total_assets = self.account.total_assets;
        let inv_total_assets = 1.0 / total_assets;
        let n_tickers = self.tickers.len();

        let n_assets = (n_tickers + 1) as f64;
        let uniform_weight = 1.0 / n_assets;

        let mut portfolio_return = 0.0;
        let mut concentration = 0.0;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_step];
            let ticker_return = (next_price / current_price) - 1.0;

            let position_value = self.account.positions[ticker_idx].value_with_price(current_price);
            let weight = (position_value * inv_total_assets).clamp(0.0, 1.0);
            portfolio_return += weight * ticker_return;
            let dev = weight - uniform_weight;
            concentration += dev * dev;
        }

        let cash_weight = (self.account.cash * inv_total_assets).clamp(0.0, 1.0);
        let cash_dev = cash_weight - uniform_weight;
        concentration += cash_dev * cash_dev;

        let clamped_return = portfolio_return.clamp(-0.5, 0.5);
        let log_return = (1.0 + clamped_return).ln();
        let base_reward = log_return * Self::ACTION_OUTCOME_SCALE;

        let max_concentration = (n_assets - 1.0) / n_assets;
        let conviction = if max_concentration > 1e-6 {
            (concentration / max_concentration).sqrt().min(1.0)
        } else {
            0.0
        };

        let conviction_bonus = if portfolio_return > 0.0 {
            conviction * log_return.abs() * Self::ACTION_OUTCOME_SCALE * Self::CONVICTION_BONUS
        } else {
            0.0
        };

        let commission_penalty =
            -(commissions * inv_total_assets) * Self::ACTION_OUTCOME_COMMISSION_PENALTY;

        base_reward + conviction_bonus + commission_penalty
    }

    #[allow(dead_code)]
    const COMMISSIONS_PENALTY_LAMBDA: f64 = 0.01;

    #[allow(dead_code)]
    pub fn get_index_benchmark_pnl_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
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
            let commissions_penalty = -commissions_relative * Self::COMMISSIONS_PENALTY_LAMBDA;

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
    const SHARPE_LAMBDA: f64 = 100.0;

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

            let sharpe_ratio = step_return - Self::SHARPE_LAMBDA * step_return * step_return;
            sharpe_ratio * REWARD_SCALE
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    const SORTINO_LAMBDA: f64 = 100.0;

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
            let sortino_ratio = step_return - Self::SORTINO_LAMBDA * downside;
            sortino_ratio * REWARD_SCALE
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    const RISK_ADJUSTED_REWARD_LAMBDA: f64 = 0.01;

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

            let downside = if step_return < 0.0 { -step_return } else { 0.0 };
            let rar = step_return - Self::RISK_ADJUSTED_REWARD_LAMBDA * downside;

            let commissions_relative = commissions / self.account.total_assets;
            let commisions_penalty = -commissions_relative * Self::RISK_ADJUSTED_REWARD_LAMBDA;
            let reward = commisions_penalty + rar;

            reward * REWARD_SCALE
        } else {
            0.0
        }
    }
}
