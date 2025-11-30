use crate::torch::constants::{
    ACTION_HISTORY_LEN, GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER,
    STATIC_OBSERVATIONS, TICKERS_COUNT,
};

use super::env::Env;

impl Env {
    pub(super) fn get_next_obs(&self) -> (Vec<f32>, Vec<f32>) {
        let mut price_deltas = Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        let absolute_step = self.episode_start_offset + self.step;

        for ticker_price_deltas in self.price_deltas.iter() {
            let start_idx = absolute_step.saturating_sub(PRICE_DELTAS_PER_TICKER - 1);
            let end_idx = (absolute_step + 1).min(ticker_price_deltas.len());
            let slice = &ticker_price_deltas[start_idx..end_idx];
            let to_take = slice.len().min(PRICE_DELTAS_PER_TICKER);

            let padding_needed = PRICE_DELTAS_PER_TICKER - to_take;
            if padding_needed > 0 {
                price_deltas.extend(std::iter::repeat(0.0f32).take(padding_needed));
            }
            price_deltas.extend(slice.iter().rev().take(to_take).map(|&x| x as f32));
        }

        let mut static_obs = Vec::with_capacity(STATIC_OBSERVATIONS);

        // === Global observations (GLOBAL_STATIC_OBS = 7) ===
        static_obs.push(1.0 - (self.step as f32 / (self.max_step - 1).max(1) as f32)); // step progress
        static_obs.push((self.account.cash / self.account.total_assets) as f32); // cash_percent
        static_obs.push(((self.account.total_assets / Self::STARTING_CASH) - 1.0) as f32); // pnl
        static_obs.push(if self.peak_assets > 0.0 {
            ((self.account.total_assets / self.peak_assets) - 1.0) as f32
        } else {
            0.0
        }); // drawdown
        static_obs.push((self.episode_history.total_commissions / Self::STARTING_CASH) as f32); // commissions
        static_obs.push(self.last_reward as f32); // last_reward
        static_obs.push(self.last_fill_ratio as f32); // last_fill_ratio
        debug_assert_eq!(static_obs.len(), GLOBAL_STATIC_OBS);

        // === Per-ticker observations (ticker-major format) ===
        let position_percents = self.account.position_percents(&self.prices, absolute_step);

        for ticker_index in 0..TICKERS_COUNT as usize {
            let current_price = self.prices[ticker_index][absolute_step];

            // Position percent
            static_obs.push(position_percents[ticker_index] as f32);

            // Unrealized P&L %
            static_obs.push(
                self.account.positions[ticker_index]
                    .appreciation(current_price) as f32,
            );

            // Momentum (20-step lookback)
            let past_step = absolute_step.saturating_sub(20);
            let past_price = self.prices[ticker_index][past_step];
            static_obs.push(((current_price / past_price) - 1.0) as f32);

            // Steps since last traded (normalized by max_step)
            let steps_since = absolute_step.saturating_sub(self.last_traded_step[ticker_index]);
            static_obs.push((steps_since as f32 / self.max_step as f32).min(1.0));

            // Action history for this ticker (most recent first)
            for i in 0..ACTION_HISTORY_LEN {
                if i < self.action_history.len() {
                    let action_idx = self.action_history.len() - 1 - i;
                    // buy_sell action for this ticker
                    static_obs.push(self.action_history[action_idx][ticker_index] as f32);
                    // hold action for this ticker
                    static_obs.push(
                        self.action_history[action_idx][TICKERS_COUNT as usize + ticker_index]
                            as f32,
                    );
                } else {
                    static_obs.push(0.0f32); // buy_sell padding
                    static_obs.push(0.0f32); // hold padding
                }
            }
            debug_assert_eq!(
                static_obs.len(),
                GLOBAL_STATIC_OBS + (ticker_index + 1) * PER_TICKER_STATIC_OBS
            );
        }

        (price_deltas, static_obs)
    }
}
