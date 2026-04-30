use tch::Tensor;

use crate::torch::action_space::{beta_log_prob, sample_beta_action};
use crate::torch::model::ModelOutput;
use crate::torch::value::hl_gauss::HlGaussBins;

pub(crate) fn sample_rollout_actions_from_output(
    output: ModelOutput,
    hl_gauss: &HlGaussBins,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let (value_logits, action_alpha, action_beta, action_std) = output;

    // Decode critic logits to scalar values for GAE.
    let values = hl_gauss.decode(&value_logits);

    let target_weights = sample_beta_action(&action_alpha, &action_beta);
    let action_log_prob = beta_log_prob(&target_weights, &action_alpha, &action_beta);

    (
        values,
        action_alpha,
        action_beta,
        action_std,
        target_weights.shallow_clone(),
        target_weights,
        action_log_prob,
    )
}
