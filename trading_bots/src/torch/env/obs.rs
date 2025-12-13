use crate::torch::constants::{
    GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS,
    TICKERS_COUNT,
};

use super::env::Env;

impl Env {
    pub(super) fn get_next_obs(&self) -> (Vec<f32>, Vec<f32>) {
        let mut price_deltas = Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        let absolute_step = self.episode_start_offset + self.step;

        // Use permuted order for price deltas
        for &real_idx in &self.ticker_perm {
            let ticker_price_deltas = &self.price_deltas[real_idx];
            let start_idx = absolute_step.saturating_sub(PRICE_DELTAS_PER_TICKER - 1);
            let end_idx = (absolute_step + 1).min(ticker_price_deltas.len());
            let slice = &ticker_price_deltas[start_idx..end_idx];
            let to_take = slice.len().min(PRICE_DELTAS_PER_TICKER);

            let padding_needed = PRICE_DELTAS_PER_TICKER - to_take;
            if padding_needed > 0 {
                price_deltas.extend(std::iter::repeat(0.0f32).take(padding_needed));
            }
            price_deltas.extend(slice.iter().take(to_take).map(|&x| x as f32));
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

        // === Per-ticker observations (ticker-major format, permuted order) ===
        let position_percents = self.account.position_percents(&self.prices, absolute_step);

        for (perm_idx, &real_idx) in self.ticker_perm.iter().enumerate() {
            let current_price = self.prices[real_idx][absolute_step];

            // Position percent
            static_obs.push(if position_percents[real_idx].is_finite() {
                position_percents[real_idx] as f32
            } else {
                0.0
            });

            // Unrealized P&L %
            let unrealized = self.account.positions[real_idx].appreciation(current_price);
            static_obs.push(if unrealized.is_finite() {
                unrealized as f32
            } else {
                0.0
            });

            // Momentum (20-step lookback)
            let past_step = absolute_step.saturating_sub(20);
            let past_price = self.prices[real_idx][past_step];
            let momentum = if past_price > 0.0 && current_price.is_finite() && past_price.is_finite()
            {
                (current_price / past_price) - 1.0
            } else {
                0.0
            };
            static_obs.push(momentum as f32);

            // Trade activity EMA (0 = inactive, higher = more frequent trading)
            static_obs.push(if self.trade_activity_ema[real_idx].is_finite() {
                self.trade_activity_ema[real_idx] as f32
            } else {
                0.0
            });

            // Steps since last trade (normalized: 1.0 = just traded, decays toward 0)
            let steps_since = self.steps_since_trade[real_idx] as f64;
            static_obs.push((1.0 / (1.0 + steps_since / 50.0)) as f32);

            // Position age (normalized: 0 = no position, higher = longer held)
            let position_age = match self.position_open_step[real_idx] {
                Some(open_step) => (absolute_step.saturating_sub(open_step) as f64 / 500.0).min(1.0),
                None => 0.0,
            };
            static_obs.push(position_age as f32);

            // Target weight for this ticker
            static_obs.push(if self.target_weights[real_idx].is_finite() {
                self.target_weights[real_idx] as f32
            } else {
                0.0
            });

            debug_assert_eq!(
                static_obs.len(),
                GLOBAL_STATIC_OBS + (perm_idx + 1) * PER_TICKER_STATIC_OBS
            );
        }

        (price_deltas, static_obs)
    }
}
