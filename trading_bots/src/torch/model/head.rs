use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        compute_values: bool,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let x = self.final_norm.forward(x_ssm);
        // [batch*tickers, seq_len, model_dim] -> [batch, tickers, seq_len, model_dim]
        let temporal_len = x.size()[1];
        let x_time = x.view([batch_size, TICKERS_COUNT, temporal_len, self.model_dim]);

        // Full-sequence PMA pooling with 2 learned queries per ticker.
        let mut ticker_repr = self
            .temporal_pma_block
            .forward(&x_time, batch_size, self.model_dim);

        // InterTickerBlock on ticker representations: [batch, tickers, dim]
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

        // Actor head: MLP per-ticker → flatten → project to latent → action_mean
        let mut actor_x = ticker_repr.reshape([batch_size * TICKERS_COUNT, self.model_dim]);
        for i in 0..self.actor_mlp_linears.len() {
            actor_x = actor_x.apply(&self.actor_mlp_linears[i]);
            actor_x = self.actor_mlp_norms[i].forward(&actor_x);
            actor_x = actor_x.silu();
        }
        let actor_latent = actor_x
            .reshape([batch_size, TICKERS_COUNT * self.model_dim])
            .apply(&self.actor_proj);
        let action_mean = actor_latent.apply(&self.actor_out);

        // Critic: PPO-style scalar value head.
        let critic_input = ticker_repr.reshape([batch_size, TICKERS_COUNT * self.model_dim]);
        let mut critic_x = critic_input.shallow_clone();
        for i in 0..self.value_mlp_linears.len() {
            critic_x = critic_x.apply(&self.value_mlp_linears[i]);
            critic_x = self.value_mlp_norms[i].forward(&critic_x);
            critic_x = critic_x.silu();
        }
        let critic_values = critic_x.apply(&self.value_out).squeeze_dim(-1);

        // Cast all outputs to fp32 for loss computation
        let values = if compute_values {
            critic_values.to_kind(Kind::Float)
        } else {
            Tensor::zeros(&[batch_size], (Kind::Float, ticker_repr.device()))
        };
        let critic_values = critic_values.to_kind(Kind::Float);
        let critic_input = critic_input.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let actor_latent = actor_latent.to_kind(Kind::Float);

        let debug_metrics = None;

        (
            (
                values,
                critic_values,
                critic_input,
                (action_mean, actor_latent),
            ),
            debug_metrics,
        )
    }
}
