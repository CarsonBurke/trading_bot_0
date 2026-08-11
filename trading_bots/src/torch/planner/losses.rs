use anyhow::{bail, Result};
use tch::{Kind, Tensor};

use crate::torch::action_space::{beta_entropy, beta_log_prob, beta_reverse_kl};
use crate::torch::planner::portfolio::PLANNER_REWARD_SCALE;
use crate::torch::train::config::{PolicyObjective, POLICY_OBJECTIVE};
use crate::torch::train::numeric_debug::compute_explained_variance;
use crate::torch::train::update::asym_clip_policy_loss;
use crate::torch::value::hl_gauss::HlGaussBins;

pub(crate) const POSITIVE_WEIGHT: f64 = 0.5;
pub(crate) const REVERSE_KL_COEFFICIENT: f64 = 0.3;
pub(crate) const VALUE_LOSS_COEFFICIENT: f64 = 1.0;
pub(crate) const PLANNER_AUX_RETURN_COEF: f64 = 0.1;

pub(crate) fn normalize_ppo_advantages(advantages: &Tensor) -> Result<Tensor> {
    if advantages.numel() == 0 {
        bail!("planner advantage batch is empty");
    }
    if advantages.isfinite().all().int64_value(&[]) == 0 {
        bail!("planner advantage batch contains NaN or infinity");
    }

    let centered = advantages - advantages.mean(Kind::Float);
    let population_variance = centered.square().mean(Kind::Float);
    let population_std = population_variance.sqrt();
    let centered_normalized = &centered / population_std.clamp_min(1e-8);
    let scale_normalized =
        advantages / advantages.square().mean(Kind::Float).sqrt().clamp_min(1e-8);
    let normalized = scale_normalized.where_self(&population_std.le(1e-8), &centered_normalized);
    if normalized.isfinite().all().int64_value(&[]) == 0 {
        bail!("normalized planner advantages contain NaN or infinity");
    }
    Ok(normalized)
}

pub struct PlannerLosses {
    pub actor_loss: Tensor,
    pub critic_loss: Tensor,
    pub policy_loss: Tensor,
    pub reverse_kl: Tensor,
    pub entropy: Tensor,
    pub value_loss: Tensor,
    pub aux_return_loss: Tensor,
}

pub fn pmpo_policy_loss(
    advantages: &Tensor,
    action_log_probs: &Tensor,
    old_alpha: &Tensor,
    old_beta: &Tensor,
    new_alpha: &Tensor,
    new_beta: &Tensor,
) -> (Tensor, Tensor) {
    let advantages = advantages.flatten(0, -1).to_kind(Kind::Float);
    let action_log_probs = action_log_probs.flatten(0, -1).to_kind(Kind::Float);
    assert_eq!(advantages.size(), action_log_probs.size());

    let weights = advantages.tanh().abs().detach();
    let weighted_log_probs = weights * action_log_probs;
    let positive = advantages.ge(0.0).to_kind(Kind::Float);
    let negative = advantages.lt(0.0).to_kind(Kind::Float);
    let positive_likelihood = (&weighted_log_probs * &positive).sum(Kind::Float)
        / positive.sum(Kind::Float).clamp_min(1.0);
    let negative_likelihood = (&weighted_log_probs * &negative).sum(Kind::Float)
        / negative.sum(Kind::Float).clamp_min(1.0);
    let policy_loss =
        -POSITIVE_WEIGHT * positive_likelihood + (1.0 - POSITIVE_WEIGHT) * negative_likelihood;
    let reverse_kl = beta_reverse_kl(old_alpha, old_beta, new_alpha, new_beta).mean(Kind::Float);
    let loss = policy_loss + REVERSE_KL_COEFFICIENT * &reverse_kl;
    (loss, reverse_kl)
}

#[allow(clippy::too_many_arguments)]
pub fn planner_actor_critic_losses(
    hl_gauss: &HlGaussBins,
    value_logits: &Tensor,
    new_alpha: &Tensor,
    new_beta: &Tensor,
    actions: &Tensor,
    old_alpha: &Tensor,
    old_beta: &Tensor,
    advantages: &Tensor,
    returns: &Tensor,
    next_return: &Tensor,
    next_return_target: &Tensor,
) -> PlannerLosses {
    let batch_size = value_logits.size()[0];
    assert_eq!(new_alpha.size(), new_beta.size());
    assert_eq!(new_alpha.size(), actions.size());
    assert_eq!(new_alpha.size(), old_alpha.size());
    assert_eq!(new_alpha.size(), old_beta.size());
    assert_eq!(new_alpha.size()[0], batch_size);
    assert_eq!(advantages.numel() as i64, batch_size);
    assert_eq!(returns.numel() as i64, batch_size);

    let action_log_probs = beta_log_prob(actions, new_alpha, new_beta);
    let (policy_loss, reverse_kl) = match POLICY_OBJECTIVE {
        PolicyObjective::Pmpo => pmpo_policy_loss(
            advantages,
            &action_log_probs,
            old_alpha,
            old_beta,
            new_alpha,
            new_beta,
        ),
        // PPO: DAPO asymmetric clip on the (batch-normalized) advantages. The
        // reverse-KL is computed for the KL controller/early-stop only and is
        // NOT added to the loss — clipping is the trust region, matching the
        // baseline PPO path in `train::update`.
        PolicyObjective::Ppo => {
            let old_log_probs = beta_log_prob(actions, old_alpha, old_beta).detach();
            let ratio = (&action_log_probs - old_log_probs).exp();
            let (clip_loss, _clip_gap) = asym_clip_policy_loss(advantages, &ratio);
            let reverse_kl =
                beta_reverse_kl(old_alpha, old_beta, new_alpha, new_beta).mean(Kind::Float);
            (clip_loss, reverse_kl)
        }
    };
    let entropy = beta_entropy(new_alpha, new_beta).mean(Kind::Float);
    let value_targets = hl_gauss.encode(&returns.flatten(0, -1));
    let value_log_probs = value_logits.log_softmax(-1, Kind::Float);
    let value_loss = -(value_targets * value_log_probs)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
        .mean(Kind::Float);

    let aux_return_loss = (next_return.flatten(0, -1).to_kind(Kind::Float)
        - next_return_target.flatten(0, -1).to_kind(Kind::Float) * PLANNER_REWARD_SCALE)
        .square()
        .mean(Kind::Float);

    let actor_loss = policy_loss.shallow_clone();
    let critic_loss =
        VALUE_LOSS_COEFFICIENT * &value_loss + PLANNER_AUX_RETURN_COEF * &aux_return_loss;
    PlannerLosses {
        actor_loss,
        critic_loss,
        policy_loss,
        reverse_kl,
        entropy,
        value_loss,
        aux_return_loss,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CriticDiagnostics {
    pub beta_concentration_mean: f64,
    pub critic_explained_variance: f64,
}

/// Beta concentration and critic explained variance over the whole real rollout.
/// Explained variance uses the canonical `compute_explained_variance`
/// (NaN sentinel on zero target variance) so it agrees with pretrain metrics.
pub fn critic_diagnostics(
    alpha: &Tensor,
    beta: &Tensor,
    predicted_values: &Tensor,
    returns: &Tensor,
) -> CriticDiagnostics {
    tch::no_grad(|| {
        if predicted_values.numel() == 0 {
            return CriticDiagnostics::default();
        }
        let concentration = (alpha + beta).mean(Kind::Float).double_value(&[]);
        let explained_variance =
            compute_explained_variance(&predicted_values.flatten(0, -1), &returns.flatten(0, -1))
                .double_value(&[]);

        CriticDiagnostics {
            beta_concentration_mean: concentration,
            critic_explained_variance: explained_variance,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn pmpo_matches_sign_weighted_likelihood_with_zero_kl() {
        let advantages = Tensor::from_slice(&[1.0f32, -1.0]);
        let log_probs = Tensor::from_slice(&[-0.2f32, -0.7]);
        let alpha = Tensor::from_slice(&[2.0f32, 2.0]).view([2, 1]);
        let beta = Tensor::from_slice(&[3.0f32, 3.0]).view([2, 1]);
        let (loss, kl) = pmpo_policy_loss(&advantages, &log_probs, &alpha, &beta, &alpha, &beta);
        let weight = 1.0f64.tanh();
        let expected = -0.5 * weight * -0.2 + 0.5 * weight * -0.7;
        assert!((loss.double_value(&[]) - expected).abs() < 1e-6);
        assert!(kl.double_value(&[]).abs() < 1e-6);
    }

    #[test]
    fn actor_and_hl_gauss_critic_losses_are_finite_and_differentiable() {
        let bins = HlGaussBins::default_for(Device::Cpu);
        let value_logits =
            Tensor::zeros([3, bins.num_bins()], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let new_alpha =
            Tensor::full([3, 1], 2.0, (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let new_beta =
            Tensor::full([3, 1], 2.5, (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let actions = Tensor::from_slice(&[0.2f32, 0.5, 0.8]).view([3, 1]);
        let old_alpha = Tensor::full([3, 1], 2.1, (Kind::Float, Device::Cpu));
        let old_beta = Tensor::full([3, 1], 2.4, (Kind::Float, Device::Cpu));
        let advantages = Tensor::from_slice(&[1.0f32, -0.5, 0.25]);
        let returns = Tensor::from_slice(&[0.2f32, -0.1, 1.0]);
        let next_return = Tensor::from_slice(&[0.01f32, -0.02, 0.03])
            .view([3, 1])
            .set_requires_grad(true);
        let next_return_target = Tensor::from_slice(&[0.02f32, -0.01, 0.0]).view([3, 1]);
        let losses = planner_actor_critic_losses(
            &bins,
            &value_logits,
            &new_alpha,
            &new_beta,
            &actions,
            &old_alpha,
            &old_beta,
            &advantages,
            &returns,
            &next_return,
            &next_return_target,
        );
        let total = &losses.actor_loss + &losses.critic_loss;
        assert!(total.double_value(&[]).is_finite());
        assert!(losses.aux_return_loss.double_value(&[]).is_finite());
        total.backward();
        assert!(value_logits.grad().defined());
        assert!(new_alpha.grad().defined());
        assert!(new_beta.grad().defined());
        assert!(next_return.grad().defined());
    }

    #[test]
    fn singleton_and_constant_advantages_preserve_finite_policy_signal() {
        for raw_advantages in [
            Tensor::from_slice(&[3.0f32]),
            Tensor::from_slice(&[3.0f32, 3.0, 3.0, 3.0]),
        ] {
            let advantages = normalize_ppo_advantages(&raw_advantages).unwrap();
            assert_eq!(advantages.isfinite().all().int64_value(&[]), 1);
            assert!((advantages.mean(Kind::Float).double_value(&[]) - 1.0).abs() < 1e-6);

            let log_ratio = Tensor::zeros_like(&advantages).set_requires_grad(true);
            let (loss, _) = asym_clip_policy_loss(&advantages, &log_ratio.exp());
            assert!(loss.double_value(&[]).is_finite());
            loss.backward();
            assert!(log_ratio.grad().defined());
            assert_eq!(log_ratio.grad().isfinite().all().int64_value(&[]), 1);
            assert!(log_ratio.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
        }

        let zeros =
            normalize_ppo_advantages(&Tensor::zeros([3], (Kind::Float, Device::Cpu))).unwrap();
        assert_eq!(zeros.abs().sum(Kind::Float).double_value(&[]), 0.0);
    }

    #[test]
    fn non_finite_advantages_are_rejected() {
        assert!(normalize_ppo_advantages(&Tensor::from_slice(&[0.0f32, f32::NAN])).is_err());
        assert!(normalize_ppo_advantages(&Tensor::from_slice(&[0.0f32, f32::INFINITY])).is_err());
    }

    #[test]
    fn critic_diagnostics_cover_the_real_rollout() {
        let alpha = Tensor::full([4, 1], 2.0, (Kind::Float, Device::Cpu));
        let beta = Tensor::full([4, 1], 2.0, (Kind::Float, Device::Cpu));
        let values = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let returns = Tensor::from_slice(&[1.0f32, 2.0, 4.0, 3.0]);
        let metrics = critic_diagnostics(&alpha, &beta, &values, &returns);
        assert!((metrics.beta_concentration_mean - 4.0).abs() < 1e-9);
        assert!((metrics.critic_explained_variance - 0.6).abs() < 1e-8);
    }
}
