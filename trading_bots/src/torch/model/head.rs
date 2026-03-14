use tch::{Kind, Tensor};

use super::{
    DebugMetrics, ModelOutput, TradingModel, LOG_STD_INIT, LOG_STD_MAX, LOG_STD_MIN, SDE_EPS,
};
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
        let sde_cls = x_time.select(2, 2);

        let action_mean = actor_cls
            .reshape([batch_size * TICKERS_COUNT, self.model_dim])
            .apply(&self.actor_proj)
            .reshape([batch_size, TICKERS_COUNT, -1])
            .sum_dim_intlist([1].as_slice(), false, Kind::Float);

        let sde_latent = sde_cls
            .reshape([batch_size, TICKERS_COUNT * self.model_dim])
            .tanh();
        let log_std = (&self.log_std_param + LOG_STD_INIT)
            .clamp(LOG_STD_MIN, LOG_STD_MAX)
            .to_kind(sde_latent.kind());
        let std_sq = log_std.exp().pow_tensor_scalar(2);
        let action_std = (sde_latent.pow_tensor_scalar(2).matmul(&std_sq) + SDE_EPS).sqrt();

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
