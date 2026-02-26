use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, SDE_LATENT_DIM, SDE_EPS};
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

        let cls = x_time.select(2, 0); // [batch, tickers, model_dim]

        // Actor: per-ticker CLS → logit (cash pinned to 0 as reference category)
        let action_mean = cls
            .reshape([batch_size * TICKERS_COUNT, self.model_dim])
            .apply(&self.actor_proj)
            .reshape([batch_size, TICKERS_COUNT]);

        // gSDE: CLS token -> latent -> tanh -> fc3, variance over latent features
        let sde_in = cls.reshape([batch_size * TICKERS_COUNT, self.model_dim]);
        let sde_latent = self.sde_norm.forward(&sde_in.apply(&self.sde_fc))
            .apply(&self.sde_fc2)
            .tanh()
            .apply(&self.sde_fc3);
        let sde_latent = sde_latent.reshape([batch_size, TICKERS_COUNT, SDE_LATENT_DIM]);
        let log_std = self.log_std_param.clamp(-3.0, -0.5);
        let std_sq = log_std.exp().pow_tensor_scalar(2).transpose(0, 1);
        let variance = (sde_latent.pow_tensor_scalar(2) * std_sq.unsqueeze(0))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let action_noise_std = (variance + SDE_EPS).sqrt();

        // Critic: all CLS tokens → single scalar
        let values = cls
            .reshape([batch_size, TICKERS_COUNT * self.model_dim])
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
