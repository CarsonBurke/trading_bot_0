use tch::{autocast, Kind, Tensor};

use super::gae::compute_gae_chunked;
use super::geometry::minibatch_samples_from_total;
use super::trainer::{AdvantageData, RolloutData, Trainer};

impl Trainer {
    pub(super) fn compute_advantages(
        &mut self,
        episode: usize,
        rollout_data: &RolloutData,
    ) -> AdvantageData {
        // Bootstrap value from final observation state (decode two-hot logits)
        let trading_model = &self.trading_model;
        let obs_static = &self.obs_static;
        let stream_state = &mut self.stream_state;
        let bootstrap_value = tch::no_grad(|| {
            let (value_logits, _, _, _) = autocast(false, || {
                trading_model.forward_stream_state_on_device_for_replay(obs_static, stream_state)
            });
            self.hl_gauss.decode(&value_logits)
        });

        let (advantages, returns) = compute_gae_chunked(
            &self.s_rewards,
            &self.s_values,
            &self.s_dones,
            &bootstrap_value,
            self.rollout_steps,
            self.rollout.nprocs,
            self.rollout.ppo_chunk_len,
            0.99,
            0.95,
            self.device,
        );
        let reset_layout_bank_cpu = if rollout_data.reset_layout_batches_cpu.is_empty() {
            Tensor::zeros(&[0, self.pd_dim], (self.replay_obs_kind, tch::Device::Cpu))
        } else {
            let reset_layout_refs: Vec<&Tensor> =
                rollout_data.reset_layout_batches_cpu.iter().collect();
            Tensor::cat(&reset_layout_refs, 0)
        };

        // Compute advantage stats once per rollout (before normalization)
        let adv_stats = tch::no_grad(|| {
            Tensor::stack(
                &[
                    advantages.mean(Kind::Float),
                    advantages.min(),
                    advantages.max(),
                ],
                0,
            )
        });

        let total_samples = self.rollout_steps * self.rollout.nprocs;
        let minibatch_size = minibatch_samples_from_total(total_samples, self.rollout.nprocs);
        let chunk_batch_size =
            ((minibatch_size + self.rollout.ppo_chunk_len - 1) / self.rollout.ppo_chunk_len).max(1);
        // Keep reset slots flat and gather only the current minibatch to avoid
        // carrying a second chunk-major copy of the rollout on device.
        let reset_slots_by_chunk = Tensor::from_slice(&rollout_data.reset_slots_host)
            .to_kind(Kind::Int64)
            .view([self.total_chunks, self.rollout.ppo_chunk_len])
            .to_device(self.device);
        println!(
            "rollout {}: {} update total_samples={} minibatch_size={} chunk_len={} chunk_batch={}",
            episode,
            "ppo",
            total_samples,
            minibatch_size,
            self.rollout.ppo_chunk_len,
            chunk_batch_size
        );
        AdvantageData {
            advantages,
            returns,
            adv_stats,
            reset_layout_bank_cpu,
            reset_slots_by_chunk,
            chunk_batch_size,
            reset_layout_count: rollout_data.reset_layout_count,
        }
    }
}
