use tch::{Kind, Tensor};

use super::{
    DebugMetrics, LOG_STD_INIT, LOG_STD_MAX, LOG_STD_MIN, ModelOutput,
    SDE_EPS, SDE_PRESCALE, TradingModel,
};
use crate::torch::constants::TICKERS_COUNT;

use super::SDE_LATENT_DIM;

impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let temporal_len = x_ssm.size()[1];
        let x_time = x_ssm.view([batch_size, TICKERS_COUNT, temporal_len, self.model_dim]);
        let flat_dim_per = temporal_len * self.model_dim;
        let flat_dim_all = TICKERS_COUNT * flat_dim_per;

        let flat_per = x_time.reshape([batch_size * TICKERS_COUNT, flat_dim_per]);
        let flat_all = x_time.reshape([batch_size, flat_dim_all]);

        // Weightless RMSNorm on flattened vectors (computed in f32)
        let flat_per_normed = {
            let xf = flat_per.to_kind(Kind::Float);
            let rms = (xf.pow_tensor_scalar(2).mean_dim([-1].as_slice(), true, Kind::Float) + 1e-6).sqrt();
            (xf / rms).to_kind(flat_per.kind())
        };
        let flat_all_normed = {
            let xf = flat_all.to_kind(Kind::Float);
            let rms = (xf.pow_tensor_scalar(2).mean_dim([-1].as_slice(), true, Kind::Float) + 1e-6).sqrt();
            (xf / rms).to_kind(flat_all.kind())
        };

        // Actor: flat_per → ACTION_COUNT, summed across tickers
        let action_mean = flat_per_normed
            .apply(&self.actor_proj)
            .reshape([batch_size, TICKERS_COUNT, -1])
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            * self.mean_scale.to_kind(flat_per_normed.kind());

        // SDE: flat_per → 64 → 64 (tanh-bounded), then matmul with log_std_param
        let sde_h = flat_per_normed.apply(&self.sde_in_proj);
        let sde_latent = (self
            .sde_norm
            .forward(&sde_h.apply(&self.sde_fc))
            .apply(&self.sde_fc2)
            / SDE_PRESCALE)
            .tanh()
            .reshape([batch_size, TICKERS_COUNT * SDE_LATENT_DIM]);
        let log_std = (&self.log_std_param + LOG_STD_INIT)
            .clamp(LOG_STD_MIN, LOG_STD_MAX)
            .to_kind(sde_latent.kind());
        let std_sq = log_std.exp().pow_tensor_scalar(2);
        let action_std = (sde_latent.pow_tensor_scalar(2).matmul(&std_sq) + SDE_EPS).sqrt();

        // Critic: RMSNorm → projection
        let values = flat_all_normed
            .apply(&self.value_proj)
            .squeeze_dim(-1);

        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_std = action_std.to_kind(Kind::Float);

        (
            (values, action_mean, action_std),
            None,
        )
    }
}
