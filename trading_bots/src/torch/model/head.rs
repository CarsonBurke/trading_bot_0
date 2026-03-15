use tch::{Kind, Tensor};

use super::{ACTION_STD_MAX, ACTION_STD_MIN, DebugMetrics, ModelOutput, TradingModel};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let temporal_len = x_ssm.size()[1];
        let x_time = x_ssm.view([batch_size, TICKERS_COUNT, temporal_len, self.model_dim]);
        let actor_cls = x_time.select(2, 0);
        let critic_cls = x_time.select(2, 1);

        let policy_mean_std_logit = actor_cls
            .reshape([batch_size * TICKERS_COUNT, self.model_dim])
            .apply(&self.policy_mean_std_logit)
            .reshape([batch_size, TICKERS_COUNT, -1])
            .sum_dim_intlist([1].as_slice(), false, Kind::Float);
        let action_mean =
            policy_mean_std_logit.narrow(1, 0, policy_mean_std_logit.size()[1] / 2);
        let action_std_logit = policy_mean_std_logit.narrow(
            1,
            policy_mean_std_logit.size()[1] / 2,
            policy_mean_std_logit.size()[1] / 2,
        );
        let action_std = action_std_logit.sigmoid().g_mul_scalar(ACTION_STD_MAX - ACTION_STD_MIN)
            + ACTION_STD_MIN;

        let values = critic_cls
            .reshape([batch_size, TICKERS_COUNT * self.model_dim])
            .apply(&self.value_proj)
            .squeeze_dim(-1);

        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_std = action_std.to_kind(Kind::Float);

        ((values, action_mean, action_std), None)
    }
}
