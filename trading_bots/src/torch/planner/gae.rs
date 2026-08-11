use tch::{Kind, Tensor};

pub const PLANNER_GAMMA: f64 = 0.995;
pub const PLANNER_GAE_LAMBDA: f64 = 0.95;

/// GAE over a dense `[time, environments]` planner rollout. Kept distinct from
/// `train::gae::compute_gae_chunked`, which operates on chunk-major PPO layouts
/// with a single `dones` flag and an explicit bootstrap value; this variant is
/// time-major and separates terminated (no bootstrap) from truncated (bootstrap
/// but stop the trace), matching the planner's non-terminating environment.
pub fn compute_planner_gae(
    rewards: &Tensor,
    values: &Tensor,
    next_values: &Tensor,
    terminated: &Tensor,
    truncated: &Tensor,
) -> (Tensor, Tensor) {
    let gamma = PLANNER_GAMMA;
    let gae_lambda = PLANNER_GAE_LAMBDA;
    assert_eq!(
        rewards.size(),
        values.size(),
        "rewards/values shape mismatch"
    );
    assert_eq!(
        rewards.size(),
        next_values.size(),
        "rewards/next_values shape mismatch"
    );
    assert_eq!(
        rewards.size(),
        terminated.size(),
        "rewards/terminated shape mismatch"
    );
    assert_eq!(
        rewards.size(),
        truncated.size(),
        "rewards/truncated shape mismatch"
    );
    assert_eq!(rewards.dim(), 2, "planner GAE expects [time, environments]");

    let shape = rewards.size();
    let steps = shape[0];
    let environments = shape[1];
    let device = rewards.device();

    tch::no_grad(|| {
        let rewards = rewards.to_kind(Kind::Float);
        let values = values.to_kind(Kind::Float);
        let next_values = next_values.to_kind(Kind::Float);
        let terminated = terminated.to_kind(Kind::Float).clamp(0.0, 1.0);
        let truncated = truncated.to_kind(Kind::Float).clamp(0.0, 1.0);
        let advantages = Tensor::zeros([steps, environments], (Kind::Float, device));
        let mut next_gae = Tensor::zeros([environments], (Kind::Float, device));

        for step in (0..steps).rev() {
            let terminal = terminated.get(step);
            let truncation = truncated.get(step);
            let bootstrap_mask = 1.0 - &terminal;
            let trace_mask =
                (Tensor::ones_like(&terminal) - &terminal - &truncation).clamp(0.0, 1.0);
            let delta = rewards.get(step) + gamma * bootstrap_mask * next_values.get(step)
                - values.get(step);
            next_gae = delta + gamma * gae_lambda * trace_mask * next_gae;
            advantages.get(step).copy_(&next_gae);
        }
        let returns = &advantages + &values;
        (advantages.detach(), returns.detach())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(tensor: &Tensor, t: i64, env: i64) -> f64 {
        tensor.double_value(&[t, env])
    }

    #[test]
    fn truncation_bootstraps_but_stops_trace() {
        let rewards = Tensor::from_slice(&[1.0f32, 100.0]).view([2, 1]);
        let values = Tensor::zeros([2, 1], (Kind::Float, tch::Device::Cpu));
        let next_values = Tensor::from_slice(&[2.0f32, 0.0]).view([2, 1]);
        let terminated = Tensor::zeros([2, 1], (Kind::Float, tch::Device::Cpu));
        let truncated = Tensor::from_slice(&[1.0f32, 0.0]).view([2, 1]);

        let (advantages, returns) =
            compute_planner_gae(&rewards, &values, &next_values, &terminated, &truncated);

        // Truncation bootstraps off next_values[0] but zeroes the trace, so the
        // large adv[1] never leaks into adv[0].
        let expected_first = 1.0 + PLANNER_GAMMA * 2.0;
        assert!((scalar(&advantages, 0, 0) - expected_first).abs() < 1e-5);
        assert!((scalar(&advantages, 1, 0) - 100.0).abs() < 1e-5);
        assert!((scalar(&returns, 0, 0) - expected_first).abs() < 1e-5);
    }

    #[test]
    fn terminal_does_not_bootstrap() {
        let rewards = Tensor::from_slice(&[1.0f32]).view([1, 1]);
        let values = Tensor::from_slice(&[0.25f32]).view([1, 1]);
        let next_values = Tensor::from_slice(&[999.0f32]).view([1, 1]);
        let terminated = Tensor::ones([1, 1], (Kind::Float, tch::Device::Cpu));
        let truncated = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let (advantages, returns) =
            compute_planner_gae(&rewards, &values, &next_values, &terminated, &truncated);
        assert!((scalar(&advantages, 0, 0) - 0.75).abs() < 1e-6);
        assert!((scalar(&returns, 0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trace_propagates_through_ordinary_steps() {
        let rewards = Tensor::from_slice(&[1.0f32, 2.0]).view([2, 1]);
        let values = Tensor::zeros([2, 1], (Kind::Float, tch::Device::Cpu));
        let next_values = Tensor::zeros([2, 1], (Kind::Float, tch::Device::Cpu));
        let boundaries = Tensor::zeros([2, 1], (Kind::Float, tch::Device::Cpu));
        let (advantages, _) =
            compute_planner_gae(&rewards, &values, &next_values, &boundaries, &boundaries);
        let expected_first = 1.0 + PLANNER_GAMMA * PLANNER_GAE_LAMBDA * 2.0;
        assert!((scalar(&advantages, 0, 0) - expected_first).abs() < 1e-5);
    }
}
