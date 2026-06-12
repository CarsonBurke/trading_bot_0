use shared::constants::{
    ACTION_COUNT as ACTION_COUNT_USIZE, ACTION_HISTORY_LEN as ACTION_HISTORY_LEN_USIZE,
    TICKERS_COUNT as TICKERS_COUNT_USIZE,
};

use tch::Tensor;

use super::single::{Env, EnvMarketSnapshot, SingleStep, SingleStepStep, TRADE_EMA_ALPHA};
use crate::torch::constants::{
    ACTION_HISTORY_LEN, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};

impl Env {
    pub(super) fn reset_single_to_episode(
        &mut self,
        market: &EnvMarketSnapshot,
        episode_start_offset: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.apply_market_snapshot(market);
        self.reset_existing_episode_state_at(episode_start_offset);
        self.ticker_perm.clone_from(&market.ticker_perm);
        self.get_next_obs()
    }

    pub(super) fn reset_single_resampled_training_episode(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.resample_training_tickers();
        self.reset_existing_episode_state_at(self.sample_episode_start_offset());
        self.get_next_obs()
    }

    pub(super) fn reset_step_single_to_episode(
        &mut self,
        market: &EnvMarketSnapshot,
        episode_start_offset: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.apply_market_snapshot(market);
        self.reset_existing_episode_state_at(episode_start_offset);
        self.ticker_perm.clone_from(&market.ticker_perm);

        let (step_deltas, static_obs) = self.get_next_step_obs();
        (step_deltas.to_vec(), static_obs.to_vec())
    }

    pub(crate) fn reset_step_single_resampled_training_episode(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.resample_training_tickers();
        self.reset_existing_episode_state_at(self.sample_episode_start_offset());

        let (step_deltas, static_obs) = self.get_next_step_obs();
        (step_deltas.to_vec(), static_obs.to_vec())
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        self.reset_existing_episode_state();

        let (price_deltas, static_obs) = self.get_next_obs();
        let price_deltas_tensor = Tensor::from_slice(&price_deltas)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        (price_deltas_tensor, static_obs_tensor)
    }

    /// Reset for VecEnv - returns raw vectors instead of tensors
    pub fn reset_single(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.reset_existing_episode_state();
        self.get_next_obs()
    }

    pub fn reset_step_single(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.reset_existing_episode_state();

        let (step_deltas, static_obs) = self.get_next_step_obs();
        (step_deltas.to_vec(), static_obs.to_vec())
    }

    fn execute_step_core(&mut self, actions: &[f64]) -> StepCoreResult {
        let absolute_step = self.episode_start_offset + self.step;
        let next_absolute_step = absolute_step + 1;
        self.account.update_total(&self.prices, absolute_step);

        for ema in &mut self.trade_activity_ema {
            *ema *= 1.0 - TRADE_EMA_ALPHA;
        }
        for steps in &mut self.steps_since_trade {
            *steps += 1;
        }

        let pre_total_assets = self.account.total_assets;

        let mut real_actions = [0.0; ACTION_COUNT_USIZE];
        for (perm_idx, &real_idx) in self.ticker_perm.iter().enumerate() {
            real_actions[real_idx] = actions[perm_idx];
        }

        if self.step == 0 {
            self.episode_history.action_step0 = Some(real_actions.to_vec());
        }

        if ACTION_HISTORY_LEN_USIZE > 0 {
            self.action_history.push_back(real_actions.to_vec());
            if self.action_history.len() > ACTION_HISTORY_LEN {
                self.action_history.pop_front();
            }
        }

        let _commissions = self.trade_by_target_weights(&real_actions, absolute_step);
        self.account.update_total(&self.prices, absolute_step);
        let (reward, reward_per_ticker) =
            self.get_unrealized_pnl_reward_breakdown(absolute_step, pre_total_assets);

        self.account.update_total(&self.prices, next_absolute_step);
        self.sync_realized_weights(next_absolute_step);
        if self.account.total_assets > self.peak_assets {
            self.peak_assets = self.account.total_assets;
        }

        let is_done = self.get_is_done();

        for (index, _) in self.tickers.iter().enumerate() {
            self.episode_history.positioned[index].push(
                self.account.positions[index]
                    .value_with_price(self.prices[index][next_absolute_step]),
            );
            self.episode_history.raw_actions[index].push(real_actions[index]);
            self.episode_history.target_weights[index].push(self.target_weights[index]);
        }
        self.episode_history.cash.push(self.account.cash);
        self.episode_history.rewards.push(reward);
        self.episode_history
            .cash_weight
            .push(self.target_weights[self.tickers.len()]);

        if is_done == 1.0 {
            self.episode_history.action_final = Some(real_actions.to_vec());
            self.handle_episode_end(next_absolute_step);
        }

        self.step += 1;
        StepCoreResult {
            reward,
            reward_per_ticker,
            is_done,
        }
    }

    /// Single-environment step for VecEnv
    pub fn step_single(&mut self, actions: &[f64]) -> SingleStep {
        let StepCoreResult {
            reward,
            reward_per_ticker,
            is_done,
        } = self.execute_step_core(actions);

        let (price_deltas, static_obs) = self.get_next_obs();
        SingleStep {
            reward,
            reward_per_ticker,
            price_deltas,
            static_obs: static_obs.try_into().unwrap(),
            is_done,
        }
    }

    pub fn step_step_single(&mut self, actions: &[f64]) -> SingleStepStep {
        let StepCoreResult {
            reward,
            reward_per_ticker,
            is_done,
        } = self.execute_step_core(actions);

        let (step_deltas, static_obs) = self.get_next_step_obs();
        SingleStepStep {
            reward,
            reward_per_ticker,
            step_deltas,
            static_obs,
            is_done,
        }
    }
}

struct StepCoreResult {
    reward: f64,
    reward_per_ticker: [f32; TICKERS_COUNT_USIZE],
    is_done: f32,
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::history::{
        episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory,
    };
    use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STEPS_PER_EPISODE};
    use crate::torch::env::earnings::EarningsIndicators;
    use crate::torch::env::macro_ind::MacroIndicators;
    use crate::torch::env::momentum::MomentumIndicators;
    use crate::torch::env::reward::REWARD_SCALE;
    use crate::torch::env::single::Env;
    use crate::types::Account;

    fn test_env_with_prices(prices: Vec<f64>, start_offset: usize) -> Env {
        let n = prices.len();
        let mut price_deltas = vec![0.0; n];
        for i in 1..n {
            price_deltas[i] = prices[i] / prices[i - 1] - 1.0;
        }

        Env {
            env_id: 0,
            step: 0,
            max_step: (n - start_offset).min(STEPS_PER_EPISODE) - 2,
            tickers: vec!["TEST".to_string()],
            prices: vec![prices.clone()],
            price_deltas: vec![price_deltas],
            account: Account::new(Env::STARTING_CASH, 1),
            episode_history: EpisodeHistory::new(1),
            meta_history: MetaHistory::default(),
            episode_start: Instant::now(),
            episode: 0,
            action_history: VecDeque::new(),
            episode_start_offset: start_offset,
            total_data_length: n,
            random_start: false,
            resample_tickers_on_reset: false,
            peak_assets: Env::STARTING_CASH,
            last_fill_ratio: 1.0,
            trade_activity_ema: vec![0.0],
            steps_since_trade: vec![0],
            position_open_step: vec![None],
            ticker_perm: vec![0],
            target_weights: vec![0.0, 1.0],
            realized_weights: vec![0.0, 1.0],
            momentum: vec![Arc::new(MomentumIndicators::compute(&prices))],
            earnings: vec![Arc::new(EarningsIndicators::empty(n))],
            macro_ind: Arc::new(MacroIndicators::empty(n)),
            record_history_io: false,
            gens_path: None,
        }
    }

    #[test]
    fn step_returns_reward_and_observation_on_consistent_next_mark() {
        let start = PRICE_DELTAS_PER_TICKER;
        let mut prices = vec![100.0; start + 3];
        prices[start] = 100.0;
        prices[start + 1] = 110.0;
        prices[start + 2] = 110.0;
        let mut env = test_env_with_prices(prices, start);

        let step = env.step_step_single(&[1.0]);

        let target_delta_value = Env::STARTING_CASH;
        let fill_ratio = Env::STARTING_CASH
            / (target_delta_value
                + (target_delta_value / 100.0) * crate::torch::constants::COMMISSION_RATE);
        let scaled_amount = target_delta_value * fill_ratio;
        let quantity = scaled_amount / 100.0;
        let commission = quantity * crate::torch::constants::COMMISSION_RATE;
        let expected_cash = Env::STARTING_CASH - scaled_amount - commission;
        let expected_position_next = quantity * 110.0;
        let expected_assets_next = expected_cash + expected_position_next;
        let expected_reward = (expected_assets_next / Env::STARTING_CASH).ln() * REWARD_SCALE;
        let expected_position_weight = expected_position_next / expected_assets_next;

        assert!((step.reward - expected_reward).abs() < 1e-10);
        assert!((env.account.total_assets - expected_assets_next).abs() < 1e-8);
        assert!((env.realized_weights[0] - expected_position_weight).abs() < 1e-12);
        assert_eq!(env.step, 1);
        assert!((step.step_deltas[0] as f64 - 0.1).abs() < 1e-6);

        let cash_percent = step.static_obs[0] as f64;
        let pnl = step.static_obs[1] as f64;
        let position_percent = step.static_obs[19] as f64;
        let realized_weight = step.static_obs[24] as f64;

        assert!((cash_percent - expected_cash / expected_assets_next).abs() < 1e-9);
        assert!((pnl - (expected_assets_next / Env::STARTING_CASH - 1.0)).abs() < 1e-6);
        assert!((position_percent - expected_position_weight).abs() < 1e-6);
        assert!((realized_weight - expected_position_weight).abs() < 1e-6);
        assert!((env.episode_history.positioned[0][0] - expected_position_next).abs() < 1e-8);
    }
}
