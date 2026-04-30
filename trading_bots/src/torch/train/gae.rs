use tch::{Kind, Tensor};

use crate::torch::constants::TICKERS_COUNT;

pub(crate) fn build_no_reset_windowed_layouts(
    boundary_layout: &Tensor,
    step_deltas_chunk: &Tensor,
    chunk_count: i64,
    ppo_chunk_len: i64,
    flat_layout_len: i64,
) -> Tensor {
    let layout_rows = chunk_count * TICKERS_COUNT;
    let boundary_rows = boundary_layout.view([layout_rows, flat_layout_len]);
    let appended_deltas = if ppo_chunk_len > 1 {
        step_deltas_chunk
            .narrow(1, 0, ppo_chunk_len - 1)
            .permute([0, 2, 1])
            .contiguous()
            .view([layout_rows, ppo_chunk_len - 1])
            .to_kind(boundary_rows.kind())
    } else {
        Tensor::zeros(
            [layout_rows, 0],
            (boundary_rows.kind(), boundary_rows.device()),
        )
    };
    let extended = Tensor::cat(&[&boundary_rows, &appended_deltas], 1);
    extended
        .unfold(1, flat_layout_len, 1)
        .view([chunk_count, TICKERS_COUNT, ppo_chunk_len, flat_layout_len])
        .permute([0, 2, 1, 3])
        .contiguous()
        .view([chunk_count * ppo_chunk_len * TICKERS_COUNT, flat_layout_len])
}

/// Compute GAE advantages and returns from chunk-major rollout data.
pub(crate) fn compute_gae_chunked(
    rewards_by_chunk: &Tensor,
    values_by_chunk: &Tensor,
    dones_by_chunk: &Tensor,
    bootstrap_value: &Tensor,
    rollout_steps: i64,
    nprocs: i64,
    ppo_chunk_len: i64,
    gamma: f64,
    gae_lambda: f64,
    device: tch::Device,
) -> (Tensor, Tensor) {
    let chunks_per_rollout = rollout_steps / ppo_chunk_len;
    let total_chunks = chunks_per_rollout * nprocs;
    let advantages = Tensor::zeros(&[total_chunks, ppo_chunk_len], (Kind::Float, device));
    let returns = Tensor::zeros(&[total_chunks, ppo_chunk_len], (Kind::Float, device));

    tch::no_grad(|| {
        let mut last_gae = Tensor::zeros(&[nprocs], (Kind::Float, device));
        for t in (0..rollout_steps).rev() {
            let chunk_block = t / ppo_chunk_len;
            let chunk_offset = t % ppo_chunk_len;
            let chunk_row = chunk_block * nprocs;
            let next_values = if t == rollout_steps - 1 {
                bootstrap_value.shallow_clone()
            } else {
                let next_chunk_block = (t + 1) / ppo_chunk_len;
                let next_chunk_offset = (t + 1) % ppo_chunk_len;
                values_by_chunk
                    .narrow(0, next_chunk_block * nprocs, nprocs)
                    .select(1, next_chunk_offset)
            };
            let cur_values = values_by_chunk
                .narrow(0, chunk_row, nprocs)
                .select(1, chunk_offset);
            let rewards = rewards_by_chunk
                .narrow(0, chunk_row, nprocs)
                .select(1, chunk_offset);
            let dones = dones_by_chunk
                .narrow(0, chunk_row, nprocs)
                .select(1, chunk_offset);

            let delta = rewards + (1.0 - &dones) * gamma * &next_values - &cur_values;
            last_gae = delta + (1.0 - &dones) * gamma * gae_lambda * &last_gae;
            let _ = advantages
                .narrow(0, chunk_row, nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&last_gae.unsqueeze(1));
            let step_returns = &last_gae + &cur_values;
            let _ = returns
                .narrow(0, chunk_row, nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&step_returns.unsqueeze(1));
        }
    });

    (advantages.detach(), returns.detach())
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::{build_no_reset_windowed_layouts, compute_gae_chunked};
    use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, TICKERS_COUNT};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
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

    #[test]
    fn no_reset_windowed_layouts_match_iterative_shift_append() {
        let chunk_count = 3i64;
        let chunk_len = 5i64;
        let flat_layout_len = PRICE_DELTAS_PER_TICKER as i64;
        let pd_dim = TICKERS_COUNT * flat_layout_len;
        let boundary_layout = Tensor::arange(chunk_count * pd_dim, (Kind::Float, Device::Cpu))
            .view([chunk_count, pd_dim]);
        let step_deltas_chunk = (Tensor::arange(
            chunk_count * chunk_len * TICKERS_COUNT,
            (Kind::Float, Device::Cpu),
        ) + 10_000.0)
            .view([chunk_count, chunk_len, TICKERS_COUNT]);

        let actual = build_no_reset_windowed_layouts(
            &boundary_layout,
            &step_deltas_chunk,
            chunk_count,
            chunk_len,
            flat_layout_len,
        );

        let layout_rows = chunk_count * TICKERS_COUNT;
        let mut current = boundary_layout.view([layout_rows, flat_layout_len]);
        let mut expected_rows = Vec::with_capacity(chunk_len as usize);
        expected_rows.push(current.shallow_clone());
        for t in 1..chunk_len {
            let row_deltas = step_deltas_chunk.select(1, t - 1).reshape([layout_rows, 1]);
            current = Tensor::cat(
                &[&current.narrow(1, 1, flat_layout_len - 1), &row_deltas],
                1,
            );
            expected_rows.push(current.shallow_clone());
        }
        let expected = Tensor::stack(&expected_rows, 0)
            .view([chunk_len, chunk_count, TICKERS_COUNT, flat_layout_len])
            .permute([1, 0, 2, 3])
            .contiguous()
            .view([chunk_count * chunk_len * TICKERS_COUNT, flat_layout_len]);

        let max_diff = (&actual - &expected).abs().max().double_value(&[]);
        assert!(
            approx_eq(max_diff, 0.0, 1e-6),
            "windowed layout mismatch, max_diff={max_diff}"
        );
    }

    #[test]
    fn no_reset_windowed_layouts_handle_single_step_chunks() {
        let chunk_count = 2i64;
        let chunk_len = 1i64;
        let flat_layout_len = PRICE_DELTAS_PER_TICKER as i64;
        let pd_dim = TICKERS_COUNT * flat_layout_len;
        let boundary_layout = Tensor::arange(chunk_count * pd_dim, (Kind::Float, Device::Cpu))
            .view([chunk_count, pd_dim]);
        let step_deltas_chunk = Tensor::zeros(
            [chunk_count, chunk_len, TICKERS_COUNT],
            (Kind::Float, Device::Cpu),
        );

        let actual = build_no_reset_windowed_layouts(
            &boundary_layout,
            &step_deltas_chunk,
            chunk_count,
            chunk_len,
            flat_layout_len,
        );
        let expected = boundary_layout.view([chunk_count * TICKERS_COUNT, flat_layout_len]);

        let max_diff = (&actual - &expected).abs().max().double_value(&[]);
        assert!(
            approx_eq(max_diff, 0.0, 1e-6),
            "single-step window mismatch, max_diff={max_diff}"
        );
    }
}
