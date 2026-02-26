use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, SDE_LATENT_DIM, SDE_EPS, SDE_NOISE_FLOOR};
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
        let scale_per = 1.0 / (flat_dim_per as f64).sqrt();
        let scale_all = 1.0 / (flat_dim_all as f64).sqrt();

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

        // Actor: RMSNorm → projection → 1/sqrt(d) (cash pinned to 0)
        let action_mean = flat_per_normed
            .apply(&self.actor_proj)
            .reshape([batch_size, TICKERS_COUNT])
            * scale_per;

        // gSDE: mean-pooled temporal features -> latent -> tanh -> fc3
        let sde_pool = x_time.mean_dim([2].as_slice(), false, x_time.kind());
        let sde_in = sde_pool.reshape([batch_size * TICKERS_COUNT, self.model_dim]).detach();
        let sde_latent = self.sde_norm.forward(&sde_in.apply(&self.sde_fc))
            .apply(&self.sde_fc2)
            .tanh()
            .apply(&self.sde_fc3);
        let sde_latent = sde_latent.reshape([batch_size, TICKERS_COUNT, SDE_LATENT_DIM]);
        let variance = sde_latent
            .pow_tensor_scalar(2)
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let action_noise_std = (variance + SDE_EPS).sqrt() + SDE_NOISE_FLOOR;

        // Critic: RMSNorm → projection → 1/sqrt(d) (detached from backbone)
        let values = flat_all_normed.detach()
            .apply(&self.value_proj)
            .squeeze_dim(-1)
            * scale_all;

        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_noise_std = action_noise_std.to_kind(Kind::Float);

        (
            (values, action_mean, action_noise_std),
            None,
        )
    }
}
