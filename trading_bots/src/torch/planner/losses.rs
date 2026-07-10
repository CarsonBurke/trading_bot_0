use tch::{Kind, Tensor};

use crate::torch::action_space::{beta_entropy, beta_log_prob, beta_reverse_kl};
use crate::torch::value::hl_gauss::HlGaussBins;

use super::rollout::RolloutSource;

#[derive(Clone, Copy, Debug)]
pub struct PlannerLossConfig {
    pub positive_weight: f64,
    pub reverse_kl_coefficient: f64,
    pub entropy_coefficient: f64,
    pub value_loss_coefficient: f64,
}

impl Default for PlannerLossConfig {
    fn default() -> Self {
        Self {
            positive_weight: 0.5,
            reverse_kl_coefficient: 0.3,
            entropy_coefficient: 0.0,
            value_loss_coefficient: 1.0,
        }
    }
}

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
    config: PlannerLossConfig,
) -> (Tensor, Tensor) {
    assert!((0.0..=1.0).contains(&config.positive_weight));
    assert!(config.reverse_kl_coefficient >= 0.0);
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
    let policy_loss = -config.positive_weight * positive_likelihood
        + (1.0 - config.positive_weight) * negative_likelihood;
    let reverse_kl = beta_reverse_kl(old_alpha, old_beta, new_alpha, new_beta).mean(Kind::Float);
    let loss = policy_loss + config.reverse_kl_coefficient * &reverse_kl;
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
    config: PlannerLossConfig,
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
    let (policy_loss, reverse_kl) = pmpo_policy_loss(
        advantages,
        &action_log_probs,
        old_alpha,
        old_beta,
        new_alpha,
        new_beta,
        config,
    );
    let entropy = beta_entropy(new_alpha, new_beta).mean(Kind::Float);
    let value_targets = hl_gauss.encode(&returns.flatten(0, -1));
    let value_log_probs = value_logits.log_softmax(-1, Kind::Float);
    let value_loss = -(value_targets * value_log_probs)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
        .mean(Kind::Float);

    let actor_loss = &policy_loss - config.entropy_coefficient * &entropy;
    let critic_loss = config.value_loss_coefficient * &value_loss;
    PlannerLosses {
        actor_loss,
        critic_loss,
        policy_loss,
        reverse_kl,
        entropy,
        value_loss,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SourceOptimizationMetrics {
    pub samples: usize,
    pub action_mean: f64,
    pub action_boundary_fraction: f64,
    pub beta_concentration_mean: f64,
    pub entropy_mean: f64,
    pub reverse_kl_mean: f64,
    pub critic_explained_variance: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SplitOptimizationMetrics {
    pub real: SourceOptimizationMetrics,
    pub fantasy: SourceOptimizationMetrics,
}

#[allow(clippy::too_many_arguments)]
pub fn split_optimization_metrics(
    sources: &Tensor,
    actions: &Tensor,
    old_alpha: &Tensor,
    old_beta: &Tensor,
    new_alpha: &Tensor,
    new_beta: &Tensor,
    predicted_values: &Tensor,
    returns: &Tensor,
) -> SplitOptimizationMetrics {
    SplitOptimizationMetrics {
        real: source_optimization_metrics(
            RolloutSource::Real,
            sources,
            actions,
            old_alpha,
            old_beta,
            new_alpha,
            new_beta,
            predicted_values,
            returns,
        ),
        fantasy: source_optimization_metrics(
            RolloutSource::Fantasy,
            sources,
            actions,
            old_alpha,
            old_beta,
            new_alpha,
            new_beta,
            predicted_values,
            returns,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn source_optimization_metrics(
    source: RolloutSource,
    sources: &Tensor,
    actions: &Tensor,
    old_alpha: &Tensor,
    old_beta: &Tensor,
    new_alpha: &Tensor,
    new_beta: &Tensor,
    predicted_values: &Tensor,
    returns: &Tensor,
) -> SourceOptimizationMetrics {
    tch::no_grad(|| {
        let indices = sources
            .flatten(0, -1)
            .eq(source as i64)
            .nonzero()
            .flatten(0, -1);
        let samples = indices.numel();
        if samples == 0 {
            return SourceOptimizationMetrics::default();
        }

        let actions = actions.index_select(0, &indices);
        let old_alpha = old_alpha.index_select(0, &indices);
        let old_beta = old_beta.index_select(0, &indices);
        let new_alpha = new_alpha.index_select(0, &indices);
        let new_beta = new_beta.index_select(0, &indices);
        let predicted_values = predicted_values.flatten(0, -1).index_select(0, &indices);
        let returns = returns.flatten(0, -1).index_select(0, &indices);

        let action_mean = actions.mean(Kind::Float).double_value(&[]);
        let boundary_fraction = actions
            .lt(0.01)
            .logical_or(&actions.gt(0.99))
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        let concentration = (&new_alpha + &new_beta).mean(Kind::Float).double_value(&[]);
        let entropy = beta_entropy(&new_alpha, &new_beta)
            .mean(Kind::Float)
            .double_value(&[]);
        let reverse_kl = beta_reverse_kl(&old_alpha, &old_beta, &new_alpha, &new_beta)
            .mean(Kind::Float)
            .double_value(&[]);
        let target_variance = returns.var(false).double_value(&[]);
        let explained_variance = if target_variance > 1e-12 {
            1.0 - (&returns - predicted_values).var(false).double_value(&[]) / target_variance
        } else {
            0.0
        };

        SourceOptimizationMetrics {
            samples,
            action_mean,
            action_boundary_fraction: boundary_fraction,
            beta_concentration_mean: concentration,
            entropy_mean: entropy,
            reverse_kl_mean: reverse_kl,
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
        let config = PlannerLossConfig::default();
        let (loss, kl) = pmpo_policy_loss(
            &advantages,
            &log_probs,
            &alpha,
            &beta,
            &alpha,
            &beta,
            config,
        );
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
            PlannerLossConfig::default(),
        );
        let total = &losses.actor_loss + &losses.critic_loss;
        assert!(total.double_value(&[]).is_finite());
        total.backward();
        assert!(value_logits.grad().defined());
        assert!(new_alpha.grad().defined());
        assert!(new_beta.grad().defined());
    }

    #[test]
    fn optimization_metrics_are_split_by_source() {
        let sources = Tensor::from_slice(&[0i64, 0, 1, 1]);
        let actions = Tensor::from_slice(&[0.2f32, 0.4, 0.8, 1.0]).view([4, 1]);
        let old_alpha = Tensor::full([4, 1], 2.0, (Kind::Float, Device::Cpu));
        let old_beta = Tensor::full([4, 1], 2.0, (Kind::Float, Device::Cpu));
        let new_alpha = old_alpha.shallow_clone();
        let new_beta = old_beta.shallow_clone();
        let values = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let returns = Tensor::from_slice(&[1.0f32, 2.0, 4.0, 3.0]);
        let metrics = split_optimization_metrics(
            &sources, &actions, &old_alpha, &old_beta, &new_alpha, &new_beta, &values, &returns,
        );
        assert_eq!(metrics.real.samples, 2);
        assert_eq!(metrics.fantasy.samples, 2);
        assert!((metrics.real.action_mean - 0.3).abs() < 1e-6);
        assert!((metrics.fantasy.action_mean - 0.9).abs() < 1e-6);
        assert_eq!(metrics.real.critic_explained_variance, 1.0);
        assert_eq!(metrics.fantasy.critic_explained_variance, -3.0);
    }
}
