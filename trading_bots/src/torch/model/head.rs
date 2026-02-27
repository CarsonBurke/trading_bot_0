use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, SDE_EPS};
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

        // Actor: RMSNorm → projection (gain controls init scale directly)
        let action_mean = flat_per_normed
            .apply(&self.actor_proj)
            .reshape([batch_size, TICKERS_COUNT])
            * self.mean_scale.to_kind(flat_per_normed.kind());

        // gSDE: mean-pooled temporal features -> latent -> quadratic variance
        let sde_pool = x_time.mean_dim([2].as_slice(), false, x_time.kind());
        let sde_in = sde_pool.reshape([batch_size * TICKERS_COUNT, self.model_dim]);
        let sde_latent = self.sde_norm.forward(&sde_in.apply(&self.sde_fc))
            .apply(&self.sde_fc2)
            .tanh();
        let latent_sq = sde_latent.pow_tensor_scalar(2);
        let noise_w_sq_t = self
            .sde_out
            .ws
            .to_kind(latent_sq.kind())
            .pow_tensor_scalar(2)
            .transpose(0, 1);
        let variance = latent_sq.matmul(&noise_w_sq_t).reshape([batch_size, TICKERS_COUNT]);
        let action_noise_std = (variance + SDE_EPS).sqrt();

        // Critic: RMSNorm → projection
        let values = flat_all_normed
            .apply(&self.value_proj)
            .squeeze_dim(-1);

        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_noise_std = action_noise_std.to_kind(Kind::Float);

        (
            (values, action_mean, action_noise_std),
            None,
        )
    }
}
