use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS, LOG_STD_INIT, SDE_EPS};
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

        let mut ticker_repr = x_time.select(2, 0);

        let alpha_scale = RESIDUAL_ALPHA_MAX / (TIME_CROSS_LAYERS as f64).sqrt();
        let block = &self.inter_ticker_block;
        let alpha_ticker_attn = block.alpha_ticker_attn.sigmoid() * alpha_scale;
        let alpha_mlp = block.alpha_mlp.sigmoid() * alpha_scale;

        let x_ticker_norm = block
            .ticker_ln
            .forward(&ticker_repr.reshape([batch_size * TICKERS_COUNT, self.model_dim]))
            .reshape([batch_size, TICKERS_COUNT, self.model_dim]);
        let qkv = x_ticker_norm.apply(&block.ticker_qkv);
        let parts = qkv.split(self.model_dim, -1);
        let q = block.q_norm.forward(&parts[0]).unsqueeze(1);
        let k = block.k_norm.forward(&parts[1]).unsqueeze(1);
        let v = parts[2].unsqueeze(1);
        let ticker_ctx = Tensor::scaled_dot_product_attention(
            &q, &k, &v,
            None::<&Tensor>, 0.0, false, None, false,
        )
            .squeeze_dim(1)
            .apply(&block.ticker_out)
            .reshape([batch_size, TICKERS_COUNT, self.model_dim]);
        ticker_repr = ticker_repr + &ticker_ctx * &alpha_ticker_attn;

        let mlp_in = block
            .mlp_ln
            .forward(&ticker_repr.reshape([batch_size * TICKERS_COUNT, self.model_dim]));
        let mlp_proj = mlp_in.apply(&block.mlp_fc1);
        let mlp_parts = mlp_proj.split(self.ff_dim, -1);
        let mlp = (mlp_parts[0].silu() * &mlp_parts[1])
            .apply(&block.mlp_fc2)
            .reshape([batch_size, TICKERS_COUNT, self.model_dim]);
        ticker_repr = ticker_repr + &mlp * &alpha_mlp;

        // Actor head
        let mut actor_x = ticker_repr.reshape([batch_size * TICKERS_COUNT, self.model_dim]);
        for i in 0..self.actor_mlp_linears.len() {
            actor_x = actor_x.apply(&self.actor_mlp_linears[i]);
            actor_x = self.actor_mlp_norms[i].forward(&actor_x);
            actor_x = actor_x.silu();
        }

        let ticker_logits = actor_x
            .apply(&self.actor_out)
            .reshape([batch_size, TICKERS_COUNT]);

        let cash_logit = actor_x
            .reshape([batch_size, TICKERS_COUNT, self.model_dim])
            .mean_dim([1].as_slice(), false, actor_x.kind())
            .apply(&self.cash_proj)
            .squeeze_dim(-1);

        let action_mean = Tensor::cat(&[ticker_logits, cash_logit.unsqueeze(-1)], -1);

        // gSDE
        let sde_raw = actor_x.apply(&self.sde_fc);
        let sde_latent = self.sde_norm.forward(&sde_raw).apply(&self.sde_fc2).tanh();
        let sde_latent = sde_latent.reshape([batch_size, TICKERS_COUNT, -1]);
        let log_std = (&self.log_std_param + LOG_STD_INIT).clamp(-3.0, -0.5);
        let std_sq = log_std.exp().pow_tensor_scalar(2).transpose(0, 1);
        let variance = (sde_latent.pow_tensor_scalar(2) * std_sq.unsqueeze(0))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let action_noise_std = (variance + SDE_EPS).sqrt();

        // Scalar critic
        let mut critic_x = ticker_repr.reshape([batch_size, TICKERS_COUNT * self.model_dim]);
        for i in 0..self.value_mlp_linears.len() {
            critic_x = critic_x.apply(&self.value_mlp_linears[i]);
            critic_x = self.value_mlp_norms[i].forward(&critic_x);
            critic_x = critic_x.silu();
        }
        let values = critic_x.apply(&self.value_out).squeeze_dim(-1); // [batch]

        // Cast all outputs to fp32
        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_noise_std = action_noise_std.to_kind(Kind::Float);

        (
            (values, action_mean, action_noise_std),
            None,
        )
    }
}
