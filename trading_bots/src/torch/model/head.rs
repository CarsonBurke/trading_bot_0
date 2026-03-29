use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, LOG_STD_MAX, LOG_STD_MIN};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_from_uniform_suffix(
        &self,
        x_suffix: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        let x_time = x_suffix.view([batch_size, TICKERS_COUNT, 4, self.model_dim]);
        let actor_cls = x_time.select(2, 1);
        let critic_cls = x_time.select(2, 2);

        let policy_mean_log_var = actor_cls
            .reshape([batch_size * TICKERS_COUNT, self.model_dim])
            .apply(&self.policy_mean_log_var)
            .reshape([batch_size, TICKERS_COUNT, -1])
            .sum_dim_intlist([1].as_slice(), false, Kind::Float);
        let action_mean = policy_mean_log_var.narrow(1, 0, policy_mean_log_var.size()[1] / 2);
        let action_log_var = policy_mean_log_var.narrow(
            1,
            policy_mean_log_var.size()[1] / 2,
            policy_mean_log_var.size()[1] / 2,
        );
        let action_std = action_log_var
            .clamp(2.0 * LOG_STD_MIN, 2.0 * LOG_STD_MAX)
            .g_mul_scalar(0.5)
            .exp();
        let value_logits = critic_cls
            .reshape([batch_size, TICKERS_COUNT * self.model_dim])
            .apply(&self.value_proj);
        (
            value_logits.to_kind(Kind::Float),
            action_mean.to_kind(Kind::Float),
            action_std.to_kind(Kind::Float),
        )
    }

    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let temporal_len = x_ssm.size()[1];
        let x_time = x_ssm.view([batch_size, TICKERS_COUNT, temporal_len, self.model_dim]);
        let actor_cls = x_time.select(2, temporal_len - 3);
        let critic_cls = x_time.select(2, temporal_len - 2);

        let policy_mean_log_var = actor_cls
            .reshape([batch_size * TICKERS_COUNT, self.model_dim])
            .apply(&self.policy_mean_log_var)
            .reshape([batch_size, TICKERS_COUNT, -1])
            .sum_dim_intlist([1].as_slice(), false, Kind::Float);
        let action_mean = policy_mean_log_var.narrow(1, 0, policy_mean_log_var.size()[1] / 2);
        let action_log_var = policy_mean_log_var.narrow(
            1,
            policy_mean_log_var.size()[1] / 2,
            policy_mean_log_var.size()[1] / 2,
        );
        let action_std = action_log_var
            .clamp(2.0 * LOG_STD_MIN, 2.0 * LOG_STD_MAX)
            .g_mul_scalar(0.5)
            .exp();

        // Value logits: [batch, NUM_BINS] for two-hot distributional critic
        let value_logits = critic_cls
            .reshape([batch_size, TICKERS_COUNT * self.model_dim])
            .apply(&self.value_proj);

        let value_logits = value_logits.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_std = action_std.to_kind(Kind::Float);

        ((value_logits, action_mean, action_std), None)
    }
}
