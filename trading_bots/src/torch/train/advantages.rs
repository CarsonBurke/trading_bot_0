use tch::{autocast, Kind, Tensor};

use super::gae::compute_gae_chunked;
use super::geometry::minibatch_samples_from_total;
use super::trainer::{AdvantageData, RolloutData, Trainer};

const RANK_GAUSS_CLAMP: f64 = 0.999;

/// Map raw advantages to empirical Gaussian quantiles over the full population.
/// Ported from orbit-wars `_rank_gaussian_advantage`: rank via double argsort,
/// quantile `(rank + 0.5) / n`, then `sqrt(2) * erfinv(clamp(2q - 1, ±0.999))`.
pub(super) fn rank_gaussian_normalize(adv: &Tensor) -> Tensor {
    let flat = adv.to_kind(Kind::Float).flatten(0, -1);
    let n = flat.numel();
    if n == 0 {
        return flat;
    }
    let ranks = flat.argsort(0, false).argsort(0, false).to_kind(Kind::Float);
    let quantile = (ranks + 0.5) / n as f64;
    let centered = (quantile * 2.0 - 1.0).clamp(-RANK_GAUSS_CLAMP, RANK_GAUSS_CLAMP);
    (centered.erfinv() * 2.0_f64.sqrt()).reshape(adv.size())
}

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
            let (value_logits, _, _) = autocast(false, || {
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
            0.995,
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

        // Compute advantage stats once per rollout (raw GAE magnitudes)
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

        // Rank-Gaussian shaping over the full per-update advantage population.
        // This is the sole advantage normalization; minibatches consume it as-is.
        let advantages = tch::no_grad(|| rank_gaussian_normalize(&advantages));

        // Post-shaping advantage stats: this is what the policy actually trains on.
        let adv_stats_shaped = tch::no_grad(|| {
            Tensor::stack(
                &[
                    advantages.mean(Kind::Float),
                    advantages.std(false),
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
        let reset_chunks_have_slots = rollout_data
            .reset_slots_host
            .chunks(self.rollout.ppo_chunk_len as usize)
            .map(|slots| slots.iter().any(|slot| *slot > 0))
            .collect();
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
            adv_stats_shaped,
            reset_layout_bank_cpu,
            reset_slots_by_chunk,
            reset_chunks_have_slots,
            chunk_batch_size,
            reset_layout_count: rollout_data.reset_layout_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::rank_gaussian_normalize;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn rank_gaussian_is_monotonic_symmetric_and_finite() {
        let n = 1000;
        let ramp: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 3.0).collect();
        let x = Tensor::from_slice(&ramp).to_kind(Kind::Float);
        let z = rank_gaussian_normalize(&x);

        assert!(z.isfinite().all().int64_value(&[]) == 1, "output must be finite");

        let vals: Vec<f64> = z.iter::<f64>().unwrap().collect();
        for w in vals.windows(2) {
            assert!(w[1] >= w[0], "must be monotonic for sorted input");
        }

        let mean = z.mean(Kind::Float).double_value(&[]);
        assert!(mean.abs() < 1e-4, "zero-mean expected, got {mean}");

        // Symmetry: smallest maps near -largest.
        let lo = vals.first().copied().unwrap();
        let hi = vals.last().copied().unwrap();
        assert!((lo + hi).abs() < 1e-4, "symmetric extremes: {lo} vs {hi}");

        // Extremes near +/- sqrt(2)*erfinv(0.999) ~= 2.3268.
        let expected = 2.0_f64.sqrt() * inv_erf(0.999);
        assert!((hi - expected).abs() < 0.05, "hi={hi} expected~{expected}");
    }

    #[test]
    fn rank_gaussian_handles_ties_without_nan() {
        let x = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 2.0, 0.0]).to_kind(Kind::Float);
        let z = rank_gaussian_normalize(&x);
        assert!(z.isfinite().all().int64_value(&[]) == 1, "ties must not produce NaN");
    }

    #[test]
    fn rank_gaussian_empty_is_finite() {
        let x = Tensor::zeros([0], (Kind::Float, Device::Cpu));
        let z = rank_gaussian_normalize(&x);
        assert_eq!(z.numel(), 0);
    }

    fn inv_erf(y: f64) -> f64 {
        // Newton refinement of erfinv for the test's expected-extreme check.
        let a = 0.147;
        let ln = (1.0 - y * y).ln();
        let t = 2.0 / (std::f64::consts::PI * a) + ln / 2.0;
        let mut x = (t * t - ln / a).sqrt().sqrt() - t;
        x = x.copysign(y);
        for _ in 0..50 {
            let err = erf(x) - y;
            x -= err / (2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp());
        }
        x
    }

    fn erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp();
        y.copysign(x)
    }
}
