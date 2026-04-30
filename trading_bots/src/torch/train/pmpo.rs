use std::env;

use tch::{autocast, Device, Kind, Tensor};

use crate::torch::action_space::{
    beta_entropy, beta_kl_old_new, beta_log_prob, beta_log_prob_per_dim,
};
use crate::torch::cuda::graph::CudaGraph;
use crate::torch::model::TradingModel;
use crate::torch::value::hl_gauss::HlGaussBins;

use super::config::{
    PolicyObjective, DEBUG_NUMERICS, ENTROPY_COEF, PMPO_POS_TO_NEG_WEIGHT, PMPO_REVERSE_KL_COEF,
    PPO_CLIP_HIGH, PPO_CLIP_LOW, RPO_ALPHA_MAX, RPO_ALPHA_MIN, VALUE_LOSS_COEF,
};
use super::value_loss::hl_gauss_value_loss;

pub(crate) struct PmpoMinibatchMetrics {
    pub(crate) approx_kl: Tensor,
    pub(crate) action_loss: Tensor,
    pub(crate) value_loss: Tensor,
    pub(crate) reverse_kl_loss: Tensor,
    pub(crate) dist_entropy: Tensor,
    pub(crate) clipped: Tensor,
}

pub(crate) struct PmpoMinibatchOutputs {
    pub(crate) total_loss: Tensor,
    pub(crate) metrics: PmpoMinibatchMetrics,
}

pub(crate) fn compute_pmpo_minibatch_outputs(
    model: &TradingModel,
    hl_gauss: &HlGaussBins,
    windowed: &Tensor,
    static_flat: &Tensor,
    sample_count: i64,
    actions: &Tensor,
    old_log_probs: &Tensor,
    old_action_alpha: &Tensor,
    old_action_beta: &Tensor,
    advantages: &Tensor,
    returns: &Tensor,
) -> PmpoMinibatchOutputs {
    let (new_value_logits, action_alpha, action_beta, _action_std) = autocast(false, || {
        model.windowed_replay_forward(windowed, static_flat, sample_count)
    });
    let action_log_probs = beta_log_prob(actions, &action_alpha, &action_beta);
    let dist_entropy_per_sample = beta_entropy(&action_alpha, &action_beta);
    let action_log_probs_per_dim = beta_log_prob_per_dim(actions, &action_alpha, &action_beta);
    let log_ratio = &action_log_probs - old_log_probs;
    let reverse_kl_loss = beta_kl_old_new(
        old_action_alpha,
        old_action_beta,
        &action_alpha,
        &action_beta,
    )
    .mean(Kind::Float);

    let adv_weight = advantages.tanh().abs().unsqueeze(-1);
    let signed_log_probs = &action_log_probs_per_dim * adv_weight;
    let pos_mask = advantages
        .ge(0.0)
        .to_kind(Kind::Float)
        .unsqueeze(-1)
        .expand_as(&signed_log_probs);
    let neg_mask = advantages
        .lt(0.0)
        .to_kind(Kind::Float)
        .unsqueeze(-1)
        .expand_as(&signed_log_probs);
    let pos_count = pos_mask.sum(Kind::Float).clamp_min(1.0);
    let neg_count = neg_mask.sum(Kind::Float).clamp_min(1.0);
    let pos_loss = (&signed_log_probs * &pos_mask).sum(Kind::Float) / pos_count;
    let neg_loss = (&signed_log_probs * &neg_mask).sum(Kind::Float) / neg_count;
    let action_loss = -PMPO_POS_TO_NEG_WEIGHT * pos_loss
        + (1.0 - PMPO_POS_TO_NEG_WEIGHT) * neg_loss
        + &reverse_kl_loss * PMPO_REVERSE_KL_COEF;
    let value_loss = hl_gauss_value_loss(hl_gauss, &new_value_logits, returns).mean(Kind::Float);
    let dist_entropy = dist_entropy_per_sample.mean(Kind::Float);
    let total_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF + action_loss.shallow_clone()
        - &dist_entropy * ENTROPY_COEF;
    let ratio = log_ratio.exp();
    let approx_kl = (ratio.shallow_clone() - 1.0 - &log_ratio).mean(Kind::Float);
    let dev = ratio - 1.0;
    let clipped = (dev.le(-PPO_CLIP_LOW).to_kind(Kind::Float)
        + dev.ge(PPO_CLIP_HIGH).to_kind(Kind::Float))
    .sum(Kind::Float);

    PmpoMinibatchOutputs {
        total_loss,
        metrics: PmpoMinibatchMetrics {
            approx_kl: approx_kl.detach(),
            action_loss: action_loss.detach(),
            value_loss: value_loss.detach(),
            reverse_kl_loss: reverse_kl_loss.detach(),
            dist_entropy: dist_entropy.detach(),
            clipped: clipped.detach(),
        },
    }
}

pub(crate) fn zero_existing_grads(trainable_vars: &[Tensor]) {
    tch::no_grad(|| {
        for param in trainable_vars {
            let mut grad = param.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
        }
    });
}

pub(crate) fn cuda_graph_updates_enabled(
    policy_objective: PolicyObjective,
    device: Device,
) -> bool {
    let enabled = env::var("PPO_CUDA_GRAPHS")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);
    if !enabled {
        return false;
    }
    if !device.is_cuda() {
        println!("CUDA graph updates disabled because device is not CUDA");
        return false;
    }
    if !CudaGraph::is_available() {
        println!("CUDA graph updates disabled because this build lacks CUDA graph support");
        return false;
    }
    if DEBUG_NUMERICS {
        println!("CUDA graph updates disabled because DEBUG_NUMERICS is enabled");
        return false;
    }
    if policy_objective != PolicyObjective::Pmpo {
        println!("CUDA graph updates disabled for non-PMPO objective");
        return false;
    }
    if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
        println!("CUDA graph updates disabled while RPO noise is enabled");
        return false;
    }
    true
}
