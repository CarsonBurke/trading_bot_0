use super::batch::CpuStepBatch;
use super::core::{RingStepResult, VecEnv};
use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use rayon::prelude::*;
use tch::{Device, Tensor};

impl VecEnv {
    /// Step all envs and write results directly into pre-allocated GPU tensors.
    /// Uses a single batched CPU→GPU copy instead of per-env copies.
    pub fn step_into(
        &mut self,
        all_actions: &[Vec<f64>],
        out_price_deltas: &mut Tensor, // [nprocs, price_deltas_dim] on device
        out_static_obs: &mut Tensor,   // [nprocs, static_obs_dim] on device
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let price_deltas_dim = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_single(&all_actions[i]);
            self.reward_buf[i] = step.reward as f32;
            self.is_done_buf[i] = step.is_done;
            let pd_offset = i * price_deltas_dim;
            let so_offset = i * static_obs_dim;
            if step.is_done == 1.0 {
                let (price_deltas, static_obs) = env.reset_single();
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&static_obs);
            } else {
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&step.price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&step.static_obs);
            }
        }

        // Single batched copy to GPU
        let pd_cpu = self.tensor_from_f32(
            &self.price_deltas_buf,
            &[
                self.nprocs_i64(),
                TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
            ],
        );
        out_price_deltas.copy_(&pd_cpu);

        let so_cpu = self.tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        // Return small tensors - these go to GPU via arithmetic ops later
        let reward = self.tensor_from_f32(&self.reward_buf, &[self.nprocs_i64()]);
        let is_done = self.tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]);

        (reward, is_done)
    }

    pub fn step_into_full(
        &mut self,
        all_actions: &[Vec<f64>],
        out_price_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
    ) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let price_deltas_dim = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_single(&all_actions[i]);
            let reward_start = i * TICKERS_COUNT as usize;
            self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.reward_per_ticker);
            self.is_done_buf[i] = step.is_done;
            let pd_offset = i * price_deltas_dim;
            let so_offset = i * static_obs_dim;
            if step.is_done == 1.0 {
                let (price_deltas, static_obs) = env.reset_single();
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&static_obs);
            } else {
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&step.price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&step.static_obs);
            }
        }

        let pd_cpu = self.tensor_from_f32(
            &self.price_deltas_buf,
            &[
                self.nprocs_i64(),
                TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
            ],
        );
        out_price_deltas.copy_(&pd_cpu);

        let so_cpu = self.tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let rpt_cpu = self.tensor_from_f32(
            &self.reward_per_ticker_buf,
            &[self.nprocs_i64(), TICKERS_COUNT],
        );
        out_reward_per_ticker.copy_(&rpt_cpu);
        out_is_done.copy_(&self.tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]));
    }

    pub fn step_into_step(
        &mut self,
        all_actions: &[Vec<f64>],
        out_step_deltas: &mut Tensor, // [nprocs, TICKERS_COUNT] on device
        out_static_obs: &mut Tensor,  // [nprocs, static_obs_dim] on device
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_step_single(&all_actions[i]);
            self.reward_buf[i] = step.reward as f32;
            self.is_done_buf[i] = step.is_done;
            let step_offset = i * TICKERS_COUNT as usize;
            self.step_deltas_buf[step_offset..step_offset + TICKERS_COUNT as usize]
                .copy_from_slice(&step.step_deltas);
            let static_offset = i * static_obs_dim;
            self.static_obs_buf[static_offset..static_offset + static_obs_dim]
                .copy_from_slice(&step.static_obs);
        }

        let deltas_cpu =
            self.owned_tensor_from_f32(&self.step_deltas_buf, &[self.nprocs_i64(), TICKERS_COUNT]);
        out_step_deltas.copy_(&deltas_cpu);

        let so_cpu = self.owned_tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let reward = self.owned_tensor_from_f32(&self.reward_buf, &[self.nprocs_i64()]);
        let is_done = self.owned_tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]);

        (reward, is_done)
    }

    pub fn step_incremental(
        &mut self,
        actions_flat: &[f64],
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let action_dim = ACTION_COUNT as usize;
        let envs_len = self.envs.len();
        debug_assert_eq!(actions_flat.len(), envs_len * action_dim);

        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let action_start = i * action_dim;
            let action_slice = &actions_flat[action_start..action_start + action_dim];
            let step_start = i * TICKERS_COUNT as usize;
            let static_start = i * static_obs_dim;

            if self.done_mask[i] {
                self.reward_buf[i] = 0.0;
                self.is_done_buf[i] = 1.0;
                self.reward_per_ticker_buf[step_start..step_start + TICKERS_COUNT as usize]
                    .fill(0.0);
                self.step_deltas_buf[step_start..step_start + TICKERS_COUNT as usize]
                    .copy_from_slice(
                        &self.last_step_deltas[step_start..step_start + TICKERS_COUNT as usize],
                    );
                self.static_obs_buf[static_start..static_start + static_obs_dim].copy_from_slice(
                    &self.last_static_obs[static_start..static_start + static_obs_dim],
                );
                continue;
            }

            let step = env.step_step_single(action_slice);
            self.reward_buf[i] = step.reward as f32;
            let reward_start = i * TICKERS_COUNT as usize;
            self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.reward_per_ticker);
            self.is_done_buf[i] = step.is_done;
            self.step_deltas_buf[step_start..step_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.step_deltas);
            self.static_obs_buf[static_start..static_start + static_obs_dim]
                .copy_from_slice(&step.static_obs);

            self.last_step_deltas[step_start..step_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.step_deltas);
            self.last_static_obs[static_start..static_start + static_obs_dim]
                .copy_from_slice(&step.static_obs);
            if step.is_done == 1.0 {
                self.done_mask[i] = true;
            }
        }

        let so_cpu = self.owned_tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let reward = self.owned_tensor_from_f32(&self.reward_buf, &[self.nprocs_i64()]);
        let reward_per_ticker = self.owned_tensor_from_f32(
            &self.reward_per_ticker_buf,
            &[self.nprocs_i64(), TICKERS_COUNT],
        );
        let is_done = self.owned_tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]);
        let step_deltas =
            self.owned_tensor_from_f32(&self.step_deltas_buf, &[self.nprocs_i64(), TICKERS_COUNT]);

        (reward, reward_per_ticker, is_done, step_deltas)
    }

    pub fn step_incremental_tensor_into(
        &mut self,
        actions: &Tensor,
        out_static_obs: &mut Tensor,
        out_reward: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
        out_step_deltas: &mut Tensor,
    ) {
        let actions_cpu = actions.to_device(Device::Cpu);
        let actions_flat = Vec::<f64>::try_from(actions_cpu.flatten(0, -1)).unwrap();
        let (reward, reward_per_ticker, is_done, step_deltas) =
            self.step_incremental(&actions_flat, out_static_obs);
        out_reward.copy_(&reward);
        out_reward_per_ticker.copy_(&reward_per_ticker);
        out_is_done.copy_(&is_done);
        out_step_deltas.copy_(&step_deltas);
    }

    pub fn step_incremental_tensor(
        &mut self,
        actions: &Tensor,
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let actions_cpu = actions.to_device(Device::Cpu);
        let actions_flat = Vec::<f64>::try_from(actions_cpu.flatten(0, -1)).unwrap();
        self.step_incremental(&actions_flat, out_static_obs)
    }

    fn step_into_ring_cpu_into(
        &mut self,
        all_actions: &[f64],
        reset_indices: &mut Vec<usize>,
        reset_price_deltas: &mut Vec<f32>,
    ) {
        let action_dim = ACTION_COUNT as usize;
        debug_assert_eq!(all_actions.len(), self.envs.len() * action_dim);

        reset_indices.clear();
        reset_price_deltas.clear();
        let step_results = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let action_start = i * action_dim;
                let action_slice = &all_actions[action_start..action_start + action_dim];
                let step = env.step_step_single(action_slice);
                (
                    i,
                    RingStepResult {
                        reward_per_ticker: step.reward_per_ticker,
                        is_done: step.is_done,
                        step_deltas: step.step_deltas,
                        static_obs: step.static_obs,
                    },
                )
            })
            .collect::<Vec<_>>();

        let mut ordered_results: Vec<Option<RingStepResult>> =
            (0..self.nprocs).map(|_| None).collect();
        for (env_idx, result) in step_results {
            ordered_results[env_idx] = Some(result);
        }

        let mut used_specs = Vec::with_capacity(self.env_group_count());
        for group_idx in 0..self.env_group_count() {
            let (group_start, group_end) = self.group_bounds(group_idx);
            let group_done = (group_start..group_end).any(|env_idx| {
                ordered_results[env_idx]
                    .as_ref()
                    .expect("missing env step result")
                    .is_done
                    == 1.0
            });

            if group_done {
                let all_done = (group_start..group_end).all(|env_idx| {
                    ordered_results[env_idx]
                        .as_ref()
                        .expect("missing env step result")
                        .is_done
                        == 1.0
                });
                assert!(
                    all_done,
                    "env reset group {} desynced: partial terminal reset",
                    group_idx
                );

                let (mut leader_price_deltas, mut leader_static_obs) =
                    self.envs[group_start].reset_single_resampled_training_episode();
                let mut spec = self.current_group_episode(group_start);
                let mut attempts = 0;
                while Self::has_used_market_episode(&used_specs, &spec) && attempts < 128 {
                    let reset = self.envs[group_start].reset_single_resampled_training_episode();
                    leader_price_deltas = reset.0;
                    leader_static_obs = reset.1;
                    spec = self.current_group_episode(group_start);
                    attempts += 1;
                }
                assert!(
                    !Self::has_used_market_episode(&used_specs, &spec),
                    "failed to sample distinct env reset group after {} attempts",
                    attempts
                );
                used_specs.push(spec.clone());

                for env_idx in group_start..group_end {
                    let result = ordered_results[env_idx]
                        .as_ref()
                        .expect("missing env step result");
                    let reward_start = env_idx * TICKERS_COUNT as usize;
                    self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                        .copy_from_slice(&result.reward_per_ticker);
                    self.is_done_buf[env_idx] = result.is_done;
                }

                self.write_reset_ring_obs(
                    group_start,
                    &leader_price_deltas,
                    &leader_static_obs,
                    reset_indices,
                    reset_price_deltas,
                );
                for env_idx in group_start + 1..group_end {
                    let (price_deltas, static_obs) =
                        self.envs[env_idx].reset_single_to_episode(&spec.market, spec.start_offset);
                    self.write_reset_ring_obs(
                        env_idx,
                        &price_deltas,
                        &static_obs,
                        reset_indices,
                        reset_price_deltas,
                    );
                }
            } else {
                for env_idx in group_start..group_end {
                    let result = ordered_results[env_idx]
                        .as_ref()
                        .expect("missing env step result");
                    let reward_start = env_idx * TICKERS_COUNT as usize;
                    self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                        .copy_from_slice(&result.reward_per_ticker);
                    self.is_done_buf[env_idx] = result.is_done;
                    self.write_step_obs(env_idx, &result.step_deltas, &result.static_obs);
                }
                used_specs.push(self.current_group_episode(group_start));
            }
        }
    }

    pub fn step_into_ring_flat_into(
        &mut self,
        all_actions: &[f64],
        out_step_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
        reset_indices: &mut Vec<usize>,
        reset_price_deltas: &mut Vec<f32>,
    ) {
        self.step_into_ring_cpu_into(all_actions, reset_indices, reset_price_deltas);

        let deltas_cpu =
            self.tensor_from_f32(&self.step_deltas_buf, &[self.nprocs_i64(), TICKERS_COUNT]);
        out_step_deltas.copy_(&deltas_cpu);

        let so_cpu = self.tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let rpt_cpu = self.tensor_from_f32(
            &self.reward_per_ticker_buf,
            &[self.nprocs_i64(), TICKERS_COUNT],
        );
        out_reward_per_ticker.copy_(&rpt_cpu);
        out_is_done.copy_(&self.tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]));
    }

    pub fn step_from_actions_f32_into(
        &mut self,
        batch: &mut CpuStepBatch,
        out_step_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
    ) {
        debug_assert_eq!(
            batch.actions_f32.len(),
            self.envs.len() * ACTION_COUNT as usize
        );
        for (dst, src) in batch.actions_f64.iter_mut().zip(batch.actions_f32.iter()) {
            *dst = *src as f64;
        }
        self.step_into_ring_flat_into(
            &batch.actions_f64,
            out_step_deltas,
            out_static_obs,
            out_reward_per_ticker,
            out_is_done,
            &mut batch.reset_indices,
            &mut batch.reset_price_deltas,
        );
    }
}
