use crate::torch::constants::{
    GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS,
    TICKERS_COUNT,
};

use super::env::Env;

impl Env {
    fn build_static_obs(&self, absolute_step: usize) -> Vec<f32> {
        let mut static_obs = Vec::with_capacity(STATIC_OBSERVATIONS);

        static_obs.push(1.0 - (self.step as f32 / (self.max_step - 1).max(1) as f32));
        static_obs.push((self.account.cash / self.account.total_assets) as f32);
        static_obs.push(((self.account.total_assets / Self::STARTING_CASH) - 1.0) as f32);
        static_obs.push(if self.peak_assets > 0.0 {
            ((self.account.total_assets / self.peak_assets) - 1.0) as f32
        } else {
            0.0
        });
        static_obs.push((self.episode_history.total_commissions / Self::STARTING_CASH) as f32);
        static_obs.push(self.last_reward as f32);
        static_obs.push(self.last_fill_ratio as f32);
        debug_assert_eq!(static_obs.len(), GLOBAL_STATIC_OBS);

        let position_percents = self.account.position_percents(&self.prices, absolute_step);

        for (perm_idx, &real_idx) in self.ticker_perm.iter().enumerate() {
            let m = &self.momentum[real_idx];
            let e = &self.earnings[real_idx];

            static_obs.push(position_percents[real_idx].clamp(-1.0, 1.0) as f32);
            static_obs.push(
                self.account.positions[real_idx]
                    .appreciation(self.prices[real_idx][absolute_step])
                    .clamp(-1.0, 1.0) as f32,
            );
            static_obs.push(self.trade_activity_ema[real_idx] as f32);
            static_obs.push((1.0 / (1.0 + self.steps_since_trade[real_idx] as f64 / 50.0)) as f32);
            static_obs.push(
                self.position_open_step[real_idx]
                    .map(|s| (absolute_step.saturating_sub(s) as f64 / 500.0).min(1.0))
                    .unwrap_or(0.0) as f32,
            );
            static_obs.push(self.target_weights[real_idx].clamp(0.0, 1.0) as f32);
            static_obs.push(m.mom_5[absolute_step] as f32);
            static_obs.push(
                ((self.prices[real_idx][absolute_step]
                    / self.prices[real_idx][absolute_step.saturating_sub(20)]
                    - 1.0)
                    .clamp(-0.5, 0.5)) as f32,
            );
            static_obs.push(m.mom_60[absolute_step] as f32);
            static_obs.push(m.mom_120[absolute_step] as f32);
            static_obs.push(m.mom_accel[absolute_step] as f32);
            static_obs.push(m.vol_adj_mom[absolute_step] as f32);
            static_obs.push(m.efficiency[absolute_step] as f32);
            static_obs.push(m.trend_strength[absolute_step] as f32);
            static_obs.push((m.rsi[absolute_step] * 2.0 - 1.0) as f32);
            static_obs.push(m.range_pos[absolute_step] as f32);
            static_obs.push((m.stoch_k[absolute_step] * 2.0 - 1.0) as f32);
            static_obs.push((m.zscore[absolute_step] / 3.0) as f32);
            static_obs.push(m.macd[absolute_step] as f32);
            static_obs.push(e.steps_to_next[absolute_step] as f32);
            static_obs.push(e.revenue_growth[absolute_step] as f32);
            static_obs.push(e.opex_growth[absolute_step] as f32);
            static_obs.push(e.net_profit_growth[absolute_step] as f32);
            static_obs.push(e.eps[absolute_step] as f32);
            static_obs.push(e.eps_surprise[absolute_step] as f32);

            debug_assert_eq!(
                static_obs.len(),
                GLOBAL_STATIC_OBS + (perm_idx + 1) * PER_TICKER_STATIC_OBS
            );
        }

        static_obs
    }

    pub(super) fn get_next_obs(&self) -> (Vec<f32>, Vec<f32>) {
        let mut price_deltas = Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        let absolute_step = self.episode_start_offset + self.step;

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

        let static_obs = self.build_static_obs(absolute_step);

        (price_deltas, static_obs)
    }

    pub(super) fn get_next_step_obs(&self) -> (Vec<f32>, Vec<f32>) {
        let absolute_step = self.episode_start_offset + self.step;
        let mut step_deltas = Vec::with_capacity(TICKERS_COUNT as usize);

        for &real_idx in &self.ticker_perm {
            let ticker_price_deltas = &self.price_deltas[real_idx];
            let v = ticker_price_deltas.get(absolute_step).copied().unwrap_or(0.0);
            step_deltas.push(v as f32);
        }

        (step_deltas, self.build_static_obs(absolute_step))
    }
}
