use tch::Tensor;

use crate::torch::action_space::{beta_log_prob, sample_beta_action};
use crate::torch::model::ModelOutput;
use crate::torch::value::hl_gauss::HlGaussBins;

pub(crate) fn sample_rollout_actions_from_output(
    output: ModelOutput,
    hl_gauss: &HlGaussBins,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let (value_logits, alpha, beta) = output;

    // Decode critic logits to scalar values for GAE.
    let values = hl_gauss.decode(&value_logits);

    let action = sample_beta_action(&alpha, &beta);
    let log_prob = beta_log_prob(&action, &alpha, &beta);

    (values, alpha, beta, action, log_prob)
}
