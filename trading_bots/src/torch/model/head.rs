use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, SDE_LATENT_DIM, LOG_STD_INIT, SDE_EPS};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let x = self.final_norm.forward(x_ssm);
        let temporal_len = x.size()[1];
        let x_time = x.view([batch_size, TICKERS_COUNT, temporal_len, self.model_dim]);

        let flat_dim_per = temporal_len * self.model_dim;
        let flat_dim_all = TICKERS_COUNT * flat_dim_per;
        let scale_per = 1.0 / (flat_dim_per as f64).sqrt();
        let scale_all = 1.0 / (flat_dim_all as f64).sqrt();

        let flat_per_ticker = x_time
            .reshape([batch_size, TICKERS_COUNT, flat_dim_per]);
        let flat_all = flat_per_ticker
            .reshape([batch_size, flat_dim_all]);

        // Weightless RMSNorm: x / rms(x), rms computed in f32
        let flat_per_normed = {
            let x = flat_per_ticker.reshape([batch_size * TICKERS_COUNT, flat_dim_per]);
            let xf = x.to_kind(Kind::Float);
            let rms = (xf.pow_tensor_scalar(2).mean_dim([-1].as_slice(), true, Kind::Float) + 1e-6).sqrt();
            (xf / rms).to_kind(x.kind()).reshape([batch_size, TICKERS_COUNT, flat_dim_per])
        };
        let flat_all_normed = {
            let xf = flat_all.to_kind(Kind::Float);
            let rms = (xf.pow_tensor_scalar(2).mean_dim([-1].as_slice(), true, Kind::Float) + 1e-6).sqrt();
            (xf / rms).to_kind(flat_all.kind())
        };

        // Actor: per-ticker flatten -> RMSNorm -> projection -> scale
        let ticker_logits = flat_per_normed
            .reshape([batch_size * TICKERS_COUNT, flat_dim_per])
            .apply(&self.actor_proj)
            .reshape([batch_size, TICKERS_COUNT])
            * scale_per;

        // Cash: all-ticker flatten -> RMSNorm -> projection -> scale
        let cash_logit = flat_all_normed.apply(&self.cash_proj).squeeze_dim(-1) * scale_all;

        let action_mean = Tensor::cat(&[ticker_logits, cash_logit.unsqueeze(-1)], -1);

        // gSDE: CLS token -> latent -> tanh -> fc3, variance over latent features
        let cls = x_time.select(2, 0); // [batch, tickers, model_dim]
        let sde_in = cls.reshape([batch_size * TICKERS_COUNT, self.model_dim]);
        let sde_latent = self.sde_norm.forward(&sde_in.apply(&self.sde_fc))
            .apply(&self.sde_fc2)
            .tanh()
            .apply(&self.sde_fc3);
        let sde_latent = sde_latent.reshape([batch_size, TICKERS_COUNT, SDE_LATENT_DIM]);
        let log_std = (&self.log_std_param + LOG_STD_INIT).clamp(-3.0, -0.5);
        let std_sq = log_std.exp().pow_tensor_scalar(2).transpose(0, 1);
        let variance = (sde_latent.pow_tensor_scalar(2) * std_sq.unsqueeze(0))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let action_noise_std = (variance + SDE_EPS).sqrt();

        // Critic: all-ticker flatten -> RMSNorm -> projection -> scale
        let values = flat_all_normed.apply(&self.value_proj).squeeze_dim(-1) * scale_all;

        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_noise_std = action_noise_std.to_kind(Kind::Float);

        (
            (values, action_mean, action_noise_std),
            None,
        )
    }
}
