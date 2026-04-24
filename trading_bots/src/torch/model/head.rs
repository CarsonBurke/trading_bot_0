use tch::{Kind, Tensor};

use super::{linear_with_same_dtype, ModelOutput, TradingModel, LOG_STD_MAX, LOG_STD_MIN};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_from_final_hidden(&self, x: &Tensor, batch_size: i64) -> ModelOutput {
        let seq = x.size()[1];
        let live = x.select(1, seq - 1);

        let actor_in = live.view([batch_size, TICKERS_COUNT, self.model_dim]);
        let policy_mean_log_var = linear_with_same_dtype(&actor_in, &self.policy_mean_log_var)
            .sum_dim_intlist([1].as_slice(), false, Kind::Float);
        let action_mean = policy_mean_log_var.narrow(1, 0, policy_mean_log_var.size()[1] / 2);
        let action_log_var = policy_mean_log_var.narrow(
            1,
            policy_mean_log_var.size()[1] / 2,
            policy_mean_log_var.size()[1] / 2,
        ) + self
            .policy_log_var_offset
            .to_kind(Kind::Float)
            .view([1, -1]);
        let center = (LOG_STD_MIN + LOG_STD_MAX) * 0.5;
        let half_range = (LOG_STD_MAX - LOG_STD_MIN) * 0.5;
        let raw_log_std = &action_log_var * 0.5;
        let action_log_std = ((raw_log_std - center) / half_range).tanh() * half_range + center;
        let action_std = action_log_std.exp();

        let critic_in = live.view([batch_size, TICKERS_COUNT * self.model_dim]);
        let value_logits = linear_with_same_dtype(&critic_in, &self.value_proj);

        (
            value_logits.to_kind(Kind::Float),
            action_mean.to_kind(Kind::Float),
            action_std.to_kind(Kind::Float),
        )
    }
}
