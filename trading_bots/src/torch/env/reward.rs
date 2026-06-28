use super::single::Env;
use shared::constants::TICKERS_COUNT;

pub(crate) const REWARD_SCALE: f64 = 20.0;

impl Env {
    pub fn get_unrealized_pnl_reward_breakdown(
        &self,
        absolute_step: usize,
        pre_total_assets: f64,
    ) -> (f64, [f32; TICKERS_COUNT]) {
        let n_tickers = self.tickers.len();
        if !self.has_next_transition() || pre_total_assets <= 0.0 {
            return (0.0, [0.0; TICKERS_COUNT]);
        }

        let next_absolute_step = absolute_step + 1;
        let inv_pre_total_assets = 1.0 / pre_total_assets;
        let mut contributions = [0.0; TICKERS_COUNT];
        let mut total_assets_next = self.account.cash;

        for ticker_idx in 0..n_tickers {
            let current_price = self.prices[ticker_idx][absolute_step];
            let next_price = self.prices[ticker_idx][next_absolute_step];
            let position = &self.account.positions[ticker_idx];
            let current_value = position.value_with_price(current_price);
            let next_value = position.value_with_price(next_price);
            total_assets_next += next_value;
            contributions[ticker_idx] = (next_value - current_value) * inv_pre_total_assets;
        }

        let portfolio_return: f64 = contributions.iter().sum();
        let strategy_log_return = (total_assets_next * inv_pre_total_assets).ln() * REWARD_SCALE;

        let mut per_ticker_rewards = [0.0; TICKERS_COUNT];
        if portfolio_return.abs() >= 1e-8 {
            let inv_portfolio_return = 1.0 / portfolio_return;
            for ticker_idx in 0..n_tickers {
                per_ticker_rewards[ticker_idx] = (strategy_log_return
                    * (contributions[ticker_idx] * inv_portfolio_return))
                    as f32;
            }
        }

        if n_tickers > 0 {
            let mean_reward = per_ticker_rewards
                .iter()
                .take(n_tickers)
                .map(|&v| v as f64)
                .sum::<f64>()
                / n_tickers as f64;
            let residual = (strategy_log_return - mean_reward) as f32;
            if residual != 0.0 {
                for reward in per_ticker_rewards.iter_mut().take(n_tickers) {
                    *reward += residual;
                }
            }
        }

        (strategy_log_return, per_ticker_rewards)
    }
}
