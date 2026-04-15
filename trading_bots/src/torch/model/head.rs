use tch::{Kind, Tensor};

use super::{
    linear_with_same_dtype, DebugMetrics, ModelOutput, TradingModel, LOG_STD_MAX, LOG_STD_MIN,
    NUM_HEAD_CLS_TOKENS,
};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    fn head_from_readout_tokens(&self, readout_tokens: &Tensor, batch_size: i64) -> ModelOutput {
        let readout_tokens = self.readout_head_ln.forward(readout_tokens);
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
        // Soft clamp via tanh: gradient never hits zero, so std always tracks mean
        let center = (LOG_STD_MIN + LOG_STD_MAX) * 0.5;
        let half_range = (LOG_STD_MAX - LOG_STD_MIN) * 0.5;
        let raw_log_std = &action_log_var * 0.5;
        let action_log_std = ((raw_log_std - center) / half_range).tanh() * half_range + center;
        let action_std = action_log_std.exp();
        let value_logits = linear_with_same_dtype(&critic_cls, &self.value_proj);
        (
            value_logits.to_kind(Kind::Float),
            action_mean.to_kind(Kind::Float),
            action_std.to_kind(Kind::Float),
        )
    }

    pub(super) fn head_from_uniform_suffix_with_prefix_cache(
        &self,
        x_suffix: &Tensor,
        batch_size: i64,
        prefix_k: &Tensor,
        prefix_v: &Tensor,
    ) -> ModelOutput {
        let queries = self.readout_queries(x_suffix);
        let (suffix_k, suffix_v) = self.readout_block.project_source(x_suffix);
        let prefix_k = if prefix_k.kind() == suffix_k.kind() {
            prefix_k.shallow_clone()
        } else {
            prefix_k.to_kind(suffix_k.kind())
        };
        let prefix_v = if prefix_v.kind() == suffix_v.kind() {
            prefix_v.shallow_clone()
        } else {
            prefix_v.to_kind(suffix_v.kind())
        };
        let source_k = Tensor::cat(&[prefix_k, suffix_k], 2);
        let source_v = Tensor::cat(&[prefix_v, suffix_v], 2);
        let readout_tokens = self
            .readout_block
            .forward_with_projected_source(&queries, &source_k, &source_v);
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
