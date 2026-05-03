use std::time::Instant;
use tch::{autocast, Device, Kind, Tensor};

use crate::torch::constants::TICKERS_COUNT;

use super::config::DEBUG_NUMERICS;
use super::numeric_debug::debug_tensor_stats;
use super::sample::sample_rollout_actions_from_output;
use super::trainer::{RolloutData, Trainer};

impl Trainer {
    pub(super) fn collect_rollout(&mut self, episode: usize) -> RolloutData {
        println!(
            "rollout {}: collecting steps={} envs={} total_samples={}",
            episode,
            self.rollout_steps,
            self.rollout.nprocs,
            self.rollout_steps * self.rollout.nprocs
        );
        let rollout_collect_start = Instant::now();
        let mut terminal_resets = 0i64;
        let mut reset_layout_batches_cpu: Vec<Tensor> = Vec::new();
        let mut reset_layout_count = 0i64;
        let mut reset_slots_host =
            vec![0i64; (self.total_chunks * self.rollout.ppo_chunk_len) as usize];
        for step in 0..self.rollout_steps as usize {
            let chunk_row = (step as i64 / self.rollout.ppo_chunk_len) * self.rollout.nprocs;
            let chunk_offset = step as i64 % self.rollout.ppo_chunk_len;
            let (
                values,
                action_mean,
                action_log_std,
                action_std,
                action_latents,
                target_weights,
                action_log_prob,
            ) = sample_rollout_actions_from_output(
                self.streamed_output
                    .take()
                    .expect("streamed rollout output missing"),
                &self.hl_gauss,
            );

            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("action_mean", &action_mean, episode as i64, step);
                let _ = debug_tensor_stats("action_log_std", &action_log_std, episode as i64, step);
                let _ = debug_tensor_stats("action_std", &action_std, episode as i64, step);
                let _ = debug_tensor_stats("action_latents", &action_latents, episode as i64, step);
                let _ = debug_tensor_stats("target_weights", &target_weights, episode as i64, step);
                let _ =
                    debug_tensor_stats("action_log_prob", &action_log_prob, episode as i64, step);
            }
            if step as i64 % self.rollout.ppo_chunk_len == 0 {
                let boundary_layout = self
                    .stream_state
                    .uniform_layout
                    .view([self.rollout.nprocs, self.pd_dim])
                    .to_kind(self.replay_obs_kind);
                let _ = self
                    .s_chunk_start_layouts
                    .narrow(0, chunk_row, self.rollout.nprocs)
                    .copy_(&boundary_layout);
            }
            let _ = self
                .s_static_obs
                .narrow(0, chunk_row, self.rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&self.obs_static.unsqueeze(1));

            let _ = self
                .action_host_view
                .copy_(&target_weights.to_kind(Kind::Float));
            self.env.step_from_actions_f32_into(
                &mut self.cpu_step_batch,
                &mut self.step_deltas,
                &mut self.obs_static,
                &mut self.step_reward_per_ticker,
                &mut self.step_is_done,
            );
            let reset_indices = &self.cpu_step_batch.reset_indices;
            let reset_price_deltas = &self.cpu_step_batch.reset_price_deltas;
            let _ = self
                .s_step_deltas
                .narrow(0, chunk_row, self.rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&self.step_deltas.unsqueeze(1));
            let mut reset_replay = None;
            if !reset_indices.is_empty() {
                let reset_count = reset_indices.len() as i64;
                terminal_resets += reset_count;
                // `from_blob` over the VecEnv-owned CPU buffer avoids a Vec->Tensor
                // copy before the H->D transfer (single non-blocking copy to GPU).
                let reset_raw_view = unsafe {
                    Tensor::from_blob(
                        reset_price_deltas.as_ptr() as *const u8,
                        &[reset_count, self.raw_pd_dim],
                        &[],
                        Kind::Float,
                        Device::Cpu,
                    )
                };
                let reset_raw_batch = reset_raw_view.to_device(self.device);
                let reset_layouts_batch = self
                    .trading_model
                    .uniform_stream_layout_from_raw_input(&reset_raw_batch);
                // Persistent CPU staging for env indices; write then view via from_blob.
                for (slot, &env_idx) in self
                    .reset_env_indices_host
                    .iter_mut()
                    .zip(reset_indices.iter())
                    .take(reset_count as usize)
                {
                    *slot = env_idx as i64;
                }
                let reset_env_host_view = unsafe {
                    Tensor::from_blob(
                        self.reset_env_indices_host.as_ptr() as *const u8,
                        &[reset_count],
                        &[],
                        Kind::Int64,
                        Device::Cpu,
                    )
                };
                let reset_env_tensor = reset_env_host_view.to_device(self.device);
                let reset_row_idx = (&reset_env_tensor.unsqueeze(1) * TICKERS_COUNT
                    + &self.ticker_offsets)
                    .reshape([-1]);
                reset_replay = Some((
                    reset_env_tensor,
                    reset_row_idx,
                    reset_layouts_batch.shallow_clone(),
                ));
                for (reset_i, env_idx) in reset_indices.iter().enumerate() {
                    let slot_idx = ((chunk_row + *env_idx as i64) * self.rollout.ppo_chunk_len
                        + chunk_offset) as usize;
                    reset_slots_host[slot_idx] = reset_layout_count + reset_i as i64 + 1;
                }
                reset_layout_count += reset_count;
                reset_layout_batches_cpu.push(
                    reset_layouts_batch
                        .to_kind(self.replay_obs_kind)
                        .to_device(tch::Device::Cpu),
                );
            }

            let _ = self
                .s_action_latents
                .narrow(0, chunk_row, self.rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&action_latents.unsqueeze(1));
            let _ = self
                .s_old_log_probs
                .narrow(0, chunk_row, self.rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&action_log_prob.unsqueeze(1));

            let portfolio_reward =
                self.step_reward_per_ticker
                    .mean_dim([1].as_slice(), false, Kind::Float);

            if DEBUG_NUMERICS {
                let _ =
                    debug_tensor_stats("portfolio_reward", &portfolio_reward, episode as i64, step);
                let _ = debug_tensor_stats("values", &values, episode as i64, step);
                let _ =
                    debug_tensor_stats("step_is_done", &self.step_is_done, episode as i64, step);
            }
            let _ = self
                .s_rewards
                .narrow(0, chunk_row, self.rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&portfolio_reward.unsqueeze(1));
            let _ = self
                .s_dones
                .narrow(0, chunk_row, self.rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&self.step_is_done.unsqueeze(1));
            let _ = self
                .s_values
                .narrow(0, chunk_row, self.rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&values.unsqueeze(1));

            let trading_model = &self.trading_model;
            let step_deltas = &self.step_deltas;
            let obs_static = &self.obs_static;
            let stream_state = &mut self.stream_state;
            self.streamed_output = Some(tch::no_grad(|| {
                autocast(false, || {
                    if reset_indices.is_empty() {
                        trading_model.step_on_device_for_replay(
                            step_deltas,
                            obs_static,
                            stream_state,
                        )
                    } else {
                        // Advance state (layout shift + patch embed + layer-0 prefill)
                        // without running the full forward — the reset below will
                        // clobber reset-env state, and a single forward afterward
                        // covers all envs. Saves one full replay forward per reset step.
                        trading_model.advance_replay_stream_state(step_deltas, stream_state);
                        let (reset_env_tensor, reset_row_idx, reset_layouts_batch) =
                            reset_replay.as_ref().expect("reset replay payload missing");
                        trading_model.reset_uniform_stream_envs_from_layout_indexed(
                            stream_state,
                            reset_env_tensor,
                            reset_row_idx,
                            reset_layouts_batch,
                        );
                        trading_model
                            .forward_stream_state_on_device_for_replay(obs_static, stream_state)
                    }
                })
            }));
        }
        let rollout_collect_secs = rollout_collect_start.elapsed().as_secs_f32();
        println!(
            "rollout {}: collected samples={} terminal_resets={} time {:.2}s",
            episode,
            self.rollout_steps * self.rollout.nprocs,
            terminal_resets,
            rollout_collect_secs
        );

        RolloutData {
            reset_layout_batches_cpu,
            reset_layout_count,
            reset_slots_host,
        }
    }
}
