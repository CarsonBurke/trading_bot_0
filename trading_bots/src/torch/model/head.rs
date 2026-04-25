use tch::{Kind, Tensor};

use super::{linear_with_same_dtype, ModelOutput, TradingModel};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_from_readout_source(&self, source: &Tensor, batch_size: i64) -> ModelOutput {
        let rows = source.size()[0];
        let queries = self
            .readout_queries
            .to_kind(source.kind())
            .unsqueeze(0)
            .expand([rows, 2, self.model_dim], false);
        let readout = self.readout_block.forward(&queries, source);
        let actor_readout = readout.select(1, 0);
        let critic_readout = readout.select(1, 1);

        let actor_in = actor_readout.view([batch_size, TICKERS_COUNT, self.model_dim]);
        let policy_mean_log_var =
            linear_with_same_dtype(&actor_in, &self.policy_mean_log_var).to_kind(Kind::Float);
        let action_mean = policy_mean_log_var.narrow(-1, 0, 1).squeeze_dim(-1);
        let action_log_var = policy_mean_log_var.narrow(-1, 1, 1).squeeze_dim(-1);
        let action_std = (&action_log_var * 0.5).exp();

        let critic_in = critic_readout.view([batch_size, TICKERS_COUNT * self.model_dim]);
        let value_logits = linear_with_same_dtype(&critic_in, &self.value_proj);

        (
            value_logits.to_kind(Kind::Float),
            action_mean.to_kind(Kind::Float),
            action_std.to_kind(Kind::Float),
            action_log_var.to_kind(Kind::Float),
        )
    }
}
