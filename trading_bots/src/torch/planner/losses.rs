use tch::{Kind, Tensor};

use crate::torch::action_space::{beta_entropy, beta_log_prob, beta_reverse_kl};
use crate::torch::train::numeric_debug::compute_explained_variance;
use crate::torch::value::hl_gauss::HlGaussBins;

use super::rollout::RolloutSource;

pub(crate) const POSITIVE_WEIGHT: f64 = 0.5;
pub(crate) const REVERSE_KL_COEFFICIENT: f64 = 0.3;
pub(crate) const VALUE_LOSS_COEFFICIENT: f64 = 1.0;

pub struct PlannerLosses {
    pub actor_loss: Tensor,
    pub critic_loss: Tensor,
    pub policy_loss: Tensor,
    pub reverse_kl: Tensor,
    pub entropy: Tensor,
    pub value_loss: Tensor,
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
    real_value_logits: &Tensor,
    fantasy_value_logits: &Tensor,
    sources: &Tensor,
    new_alpha: &Tensor,
    new_beta: &Tensor,
    actions: &Tensor,
    old_alpha: &Tensor,
    old_beta: &Tensor,
    advantages: &Tensor,
    returns: &Tensor,
) -> PlannerLosses {
    let batch_size = real_value_logits.size()[0];
    assert_eq!(real_value_logits.size(), fantasy_value_logits.size());
    assert_eq!(sources.numel() as i64, batch_size);
    assert_eq!(new_alpha.size(), new_beta.size());
    assert_eq!(new_alpha.size(), actions.size());
    assert_eq!(new_alpha.size(), old_alpha.size());
    assert_eq!(new_alpha.size(), old_beta.size());
    assert_eq!(new_alpha.size()[0], batch_size);
    assert_eq!(advantages.numel() as i64, batch_size);
    assert_eq!(returns.numel() as i64, batch_size);

    let action_log_probs = beta_log_prob(actions, new_alpha, new_beta);
    let (policy_loss, reverse_kl) = pmpo_policy_loss(
        advantages,
        &action_log_probs,
        old_alpha,
        old_beta,
        new_alpha,
        new_beta,
    );
    let entropy = beta_entropy(new_alpha, new_beta).mean(Kind::Float);
    let value_logits = route_value_logits(real_value_logits, fantasy_value_logits, sources);
    let value_targets = hl_gauss.encode(&returns.flatten(0, -1));
    let value_log_probs = value_logits.log_softmax(-1, Kind::Float);
    let value_loss = -(value_targets * value_log_probs)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
        .mean(Kind::Float);

    let actor_loss = policy_loss.shallow_clone();
    let critic_loss = VALUE_LOSS_COEFFICIENT * &value_loss;
    PlannerLosses {
        actor_loss,
        critic_loss,
        policy_loss,
        reverse_kl,
        entropy,
        value_loss,
    }
}

pub fn route_value_logits(
    real_value_logits: &Tensor,
    fantasy_value_logits: &Tensor,
    sources: &Tensor,
) -> Tensor {
    assert_eq!(real_value_logits.size(), fantasy_value_logits.size());
    let fantasy = sources
        .flatten(0, -1)
        .eq(RolloutSource::Fantasy as i64)
        .unsqueeze(-1);
    fantasy_value_logits.where_self(&fantasy, real_value_logits)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SourceCriticDiagnostics {
    pub beta_concentration_mean: f64,
    pub critic_explained_variance: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SplitCriticDiagnostics {
    pub real: SourceCriticDiagnostics,
    pub fantasy: SourceCriticDiagnostics,
}

/// Per-source Beta concentration and critic explained variance over the whole
/// rollout. Explained variance uses the canonical `compute_explained_variance`
/// (NaN sentinel on zero target variance) so it agrees with pretrain metrics.
pub fn split_critic_diagnostics(
    sources: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    predicted_values: &Tensor,
    returns: &Tensor,
) -> SplitCriticDiagnostics {
    SplitCriticDiagnostics {
        real: source_critic_diagnostics(
            RolloutSource::Real,
            sources,
            alpha,
            beta,
            predicted_values,
            returns,
        ),
        fantasy: source_critic_diagnostics(
            RolloutSource::Fantasy,
            sources,
            alpha,
            beta,
            predicted_values,
            returns,
        ),
    }
}

fn source_critic_diagnostics(
    source: RolloutSource,
    sources: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    predicted_values: &Tensor,
    returns: &Tensor,
) -> SourceCriticDiagnostics {
    tch::no_grad(|| {
        let indices = sources
            .flatten(0, -1)
            .eq(source as i64)
            .nonzero()
            .flatten(0, -1);
        if indices.numel() == 0 {
            return SourceCriticDiagnostics::default();
        }

        let alpha = alpha.index_select(0, &indices);
        let beta = beta.index_select(0, &indices);
        let predicted_values = predicted_values.flatten(0, -1).index_select(0, &indices);
        let returns = returns.flatten(0, -1).index_select(0, &indices);

        let concentration = (&alpha + &beta).mean(Kind::Float).double_value(&[]);
        let explained_variance =
            compute_explained_variance(&predicted_values, &returns).double_value(&[]);

        SourceCriticDiagnostics {
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
        let real_value_logits =
            Tensor::zeros([3, bins.num_bins()], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let fantasy_value_logits =
            Tensor::zeros([3, bins.num_bins()], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let sources = Tensor::from_slice(&[0i64, 1, 0]);
        let new_alpha =
            Tensor::full([3, 1], 2.0, (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let new_beta =
            Tensor::full([3, 1], 2.5, (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let actions = Tensor::from_slice(&[0.2f32, 0.5, 0.8]).view([3, 1]);
        let old_alpha = Tensor::full([3, 1], 2.1, (Kind::Float, Device::Cpu));
        let old_beta = Tensor::full([3, 1], 2.4, (Kind::Float, Device::Cpu));
        let advantages = Tensor::from_slice(&[1.0f32, -0.5, 0.25]);
        let returns = Tensor::from_slice(&[0.2f32, -0.1, 1.0]);
        let losses = planner_actor_critic_losses(
            &bins,
            &real_value_logits,
            &fantasy_value_logits,
            &sources,
            &new_alpha,
            &new_beta,
            &actions,
            &old_alpha,
            &old_beta,
            &advantages,
            &returns,
        );
        let total = &losses.actor_loss + &losses.critic_loss;
        assert!(total.double_value(&[]).is_finite());
        total.backward();
        assert!(real_value_logits.grad().defined());
        assert!(fantasy_value_logits.grad().defined());
        assert!(new_alpha.grad().defined());
        assert!(new_beta.grad().defined());
    }

    #[test]
    fn routes_each_source_to_its_critic_without_changing_actor_inputs() {
        let real = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).view([2, 2]);
        let fantasy = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0]).view([2, 2]);
        let sources =
            Tensor::from_slice(&[RolloutSource::Real as i64, RolloutSource::Fantasy as i64]);
        let routed = route_value_logits(&real, &fantasy, &sources);
        assert_eq!(routed.double_value(&[0, 0]), 1.0);
        assert_eq!(routed.double_value(&[0, 1]), 2.0);
        assert_eq!(routed.double_value(&[1, 0]), 30.0);
        assert_eq!(routed.double_value(&[1, 1]), 40.0);
    }

    #[test]
    fn critic_diagnostics_are_split_by_source() {
        let sources = Tensor::from_slice(&[0i64, 0, 1, 1]);
        let alpha = Tensor::full([4, 1], 2.0, (Kind::Float, Device::Cpu));
        let beta = Tensor::full([4, 1], 2.0, (Kind::Float, Device::Cpu));
        let values = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let returns = Tensor::from_slice(&[1.0f32, 2.0, 4.0, 3.0]);
        let metrics = split_critic_diagnostics(&sources, &alpha, &beta, &values, &returns);
        assert!((metrics.real.beta_concentration_mean - 4.0).abs() < 1e-9);
        assert!((metrics.fantasy.beta_concentration_mean - 4.0).abs() < 1e-9);
        assert_eq!(metrics.real.critic_explained_variance, 1.0);
        assert_eq!(metrics.fantasy.critic_explained_variance, -3.0);
    }
}
