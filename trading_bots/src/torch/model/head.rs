use tch::{Kind, Tensor};

use super::{
    linear_with_same_dtype, DebugMetrics, ModelOutput, TradingModel, LOG_STD_MAX, LOG_STD_MIN,
    NUM_HEAD_CLS_TOKENS,
};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    fn head_from_readout_tokens(&self, readout_tokens: &Tensor, batch_size: i64) -> ModelOutput {
        let actor_cls =
            readout_tokens
                .select(1, 0)
                .view([batch_size, TICKERS_COUNT, self.model_dim]);
        let critic_cls = readout_tokens
            .select(1, 1)
            .view([batch_size, TICKERS_COUNT * self.model_dim]);

        let policy_mean_log_var = linear_with_same_dtype(&actor_cls, &self.policy_mean_log_var)
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
        let value_logits = linear_with_same_dtype(&critic_cls, &self.value_proj);
        (
            value_logits.to_kind(Kind::Float),
            action_mean.to_kind(Kind::Float),
            action_std.to_kind(Kind::Float),
        )
    }

    pub(super) fn head_from_uniform_suffix(
        &self,
        x_suffix: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        let queries = self.readout_queries(x_suffix);
        let readout_tokens = self.readout_block.forward(&queries, x_suffix);
        self.head_from_readout_tokens(&readout_tokens, batch_size)
    }

    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let queries = x_ssm.narrow(
            1,
            x_ssm.size()[1] - NUM_HEAD_CLS_TOKENS,
            NUM_HEAD_CLS_TOKENS,
        );
        let readout_tokens = self.readout_block.forward(&queries, x_ssm);
        (
            self.head_from_readout_tokens(&readout_tokens, batch_size),
            None,
        )
    }
}
