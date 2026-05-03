use tch::Tensor;

use crate::torch::action_space::{sample_squashed_gaussian_action, squashed_gaussian_log_prob};
use crate::torch::model::ModelOutput;
use crate::torch::value::hl_gauss::HlGaussBins;

pub(crate) fn sample_rollout_actions_from_output(
    output: ModelOutput,
    hl_gauss: &HlGaussBins,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    const LOG_2PI: f64 = 1.8378770664093453;
    let (value_logits, action_mean, action_log_std, action_std) = output;

    // Decode critic logits to scalar values for GAE.
    let values = hl_gauss.decode(&value_logits);

    let (action_latent, target_weights) =
        sample_squashed_gaussian_action(&action_mean, &action_std);
    let action_log_prob =
        squashed_gaussian_log_prob(&action_latent, &action_mean, &action_std, LOG_2PI);

    (
        values,
        action_mean,
        action_log_std,
        action_std,
        action_latent,
        target_weights,
        action_log_prob,
    )
}
