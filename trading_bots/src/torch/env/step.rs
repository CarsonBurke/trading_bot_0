use shared::constants::{
    ACTION_COUNT as ACTION_COUNT_USIZE, ACTION_HISTORY_LEN as ACTION_HISTORY_LEN_USIZE,
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

    /// Single-environment step for VecEnv
    pub fn step_single(&mut self, actions: &[f64]) -> SingleStep {
        let absolute_step = self.episode_start_offset + self.step;
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
        self.sync_realized_weights(absolute_step);
        let (reward, reward_per_ticker) =
            self.get_unrealized_pnl_reward_breakdown(absolute_step, pre_total_assets);

        self.last_reward = reward;
        if self.account.total_assets > self.peak_assets {
            self.peak_assets = self.account.total_assets;
        }

        let is_done = self.get_is_done();

        for (index, _) in self.tickers.iter().enumerate() {
            self.episode_history.positioned[index].push(
                self.account.positions[index].value_with_price(self.prices[index][absolute_step]),
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
            self.handle_episode_end(absolute_step);
        }

        self.step += 1;
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
        let absolute_step = self.episode_start_offset + self.step;
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
        self.sync_realized_weights(absolute_step);
        let (reward, reward_per_ticker) =
            self.get_unrealized_pnl_reward_breakdown(absolute_step, pre_total_assets);

        self.last_reward = reward;
        if self.account.total_assets > self.peak_assets {
            self.peak_assets = self.account.total_assets;
        }

        let is_done = self.get_is_done();

        for (index, _) in self.tickers.iter().enumerate() {
            self.episode_history.positioned[index].push(
                self.account.positions[index].value_with_price(self.prices[index][absolute_step]),
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
            self.handle_episode_end(absolute_step);
        }

        self.step += 1;
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
