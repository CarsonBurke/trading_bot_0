#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use crate::torch::hl_gauss::HlGaussBins;
    use crate::torch::ppo::{compute_gae_chunked, hl_gauss_value_loss};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn hl_gauss_value_loss_matches_cross_entropy() {
        let hl_gauss = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let logits = Tensor::zeros([1, 21], (Kind::Float, tch::Device::Cpu));
        let _ = logits.get(0).get(10).fill_(0.3);
        let _ = logits.get(0).get(11).fill_(0.1);

        let returns = Tensor::from_slice(&[0.15f32]);

        let loss = hl_gauss_value_loss(&hl_gauss, &logits, &returns);

        let return_bins = hl_gauss.encode(&returns);
        let reference = -(&return_bins * logits.log_softmax(-1, Kind::Float)).sum_dim_intlist(
            [-1].as_slice(),
            false,
            Kind::Float,
        );

        let actual = loss.double_value(&[0]);
        let expected = reference.double_value(&[0]);

        assert!(
            approx_eq(actual, expected, 1e-6),
            "value loss mismatch: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn chunked_gae_matches_time_major_reference() {
        let rollout_steps = 6i64;
        let nprocs = 3i64;
        let chunk_len = 2i64;
        let total_chunks = (rollout_steps / chunk_len) * nprocs;
        let gamma = 0.9;
        let gae_lambda = 0.8;
        let mut rewards = vec![0f32; (total_chunks * chunk_len) as usize];
        let mut values = vec![0f32; (total_chunks * chunk_len) as usize];
        let mut dones = vec![0f32; (total_chunks * chunk_len) as usize];
        let bootstrap = vec![0.3f32, -0.2, 0.1];

        for t in 0..rollout_steps {
            let chunk_row = (t / chunk_len) * nprocs;
            let chunk_offset = t % chunk_len;
            for env in 0..nprocs {
                let idx = ((chunk_row + env) * chunk_len + chunk_offset) as usize;
                rewards[idx] = (t as f32) * 0.13 + (env as f32) * 0.07 - 0.2;
                values[idx] = (t as f32) * -0.03 + (env as f32) * 0.11;
                if t == 3 && env == 1 {
                    dones[idx] = 1.0;
                }
            }
        }

        let rewards_t = Tensor::from_slice(&rewards).view([total_chunks, chunk_len]);
        let values_t = Tensor::from_slice(&values).view([total_chunks, chunk_len]);
        let dones_t = Tensor::from_slice(&dones).view([total_chunks, chunk_len]);
        let bootstrap_t = Tensor::from_slice(&bootstrap);
        let (advantages, returns) = compute_gae_chunked(
            &rewards_t,
            &values_t,
            &dones_t,
            &bootstrap_t,
            rollout_steps,
            nprocs,
            chunk_len,
            gamma,
            gae_lambda,
            Device::Cpu,
        );

        let mut expected_adv = vec![0f32; rewards.len()];
        let mut expected_ret = vec![0f32; rewards.len()];
        let mut last_gae = vec![0f32; nprocs as usize];
        for t in (0..rollout_steps).rev() {
            let chunk_row = (t / chunk_len) * nprocs;
            let chunk_offset = t % chunk_len;
            for env in 0..nprocs {
                let idx = ((chunk_row + env) * chunk_len + chunk_offset) as usize;
                let next_value = if t == rollout_steps - 1 {
                    bootstrap[env as usize]
                } else {
                    let next_chunk_row = ((t + 1) / chunk_len) * nprocs;
                    let next_chunk_offset = (t + 1) % chunk_len;
                    values[((next_chunk_row + env) * chunk_len + next_chunk_offset) as usize]
                };
                let nonterminal = 1.0 - dones[idx];
                let delta = rewards[idx] + nonterminal * gamma as f32 * next_value - values[idx];
                last_gae[env as usize] =
                    delta + nonterminal * gamma as f32 * gae_lambda as f32 * last_gae[env as usize];
                expected_adv[idx] = last_gae[env as usize];
                expected_ret[idx] = expected_adv[idx] + values[idx];
            }
        }

        for chunk in 0..total_chunks {
            for offset in 0..chunk_len {
                let idx = (chunk * chunk_len + offset) as usize;
                let actual_adv = advantages.double_value(&[chunk, offset]);
                let actual_ret = returns.double_value(&[chunk, offset]);
                assert!(
                    approx_eq(actual_adv, expected_adv[idx] as f64, 1e-5),
                    "advantage mismatch at chunk={chunk} offset={offset}: expected {}, got {actual_adv}",
                    expected_adv[idx]
                );
                assert!(
                    approx_eq(actual_ret, expected_ret[idx] as f64, 1e-5),
                    "return mismatch at chunk={chunk} offset={offset}: expected {}, got {actual_ret}",
                    expected_ret[idx]
                );
            }
        }
    }
}
