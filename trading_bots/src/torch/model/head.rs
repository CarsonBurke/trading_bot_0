use tch::{Kind, Tensor};

use super::{
    DebugMetrics, ModelOutput, TradingModel, FF_DIM, MODEL_DIM,
    RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS,
};
use crate::torch::constants::TICKERS_COUNT;


impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        compute_values: bool,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let x = self.ssm_final_norm.forward(x_ssm);
        // [batch*tickers, seq_len, SSM_DIM] -> [batch, tickers, seq_len, SSM_DIM]
        let temporal_len = x.size()[1];
        let x_time = x.view([batch_size, TICKERS_COUNT, temporal_len, MODEL_DIM]);

        // Extract last token: [batch, tickers, MODEL_DIM]
        let last_tokens = x_time.narrow(2, temporal_len - 1, 1).squeeze_dim(2);

        let mut ticker_repr = last_tokens;

        // InterTickerBlock on ticker representations: [batch, tickers, dim]
        let alpha_scale = RESIDUAL_ALPHA_MAX / (TIME_CROSS_LAYERS as f64).sqrt();
        let block = &self.inter_ticker_block;
        let alpha_ticker_attn = block.alpha_ticker_attn.sigmoid() * alpha_scale;
        let alpha_mlp = block.alpha_mlp.sigmoid() * alpha_scale;

        let x_ticker_norm = block
            .ticker_ln
            .forward(&ticker_repr.reshape([batch_size * TICKERS_COUNT, MODEL_DIM]))
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        let q = x_ticker_norm.apply(&block.ticker_q);
        let k = x_ticker_norm.apply(&block.ticker_k);
        let v = x_ticker_norm.apply(&block.ticker_v);
        let kind = q.kind();
        let scores = q.matmul(&k.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt();
        let attn = scores.softmax(-1, Kind::Float).to_kind(kind);
        let ticker_ctx = attn
            .matmul(&v)
            .apply(&block.ticker_out)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        ticker_repr = ticker_repr + &ticker_ctx * &alpha_ticker_attn;

        let mlp_in = block
            .mlp_ln
            .forward(&ticker_repr.reshape([batch_size * TICKERS_COUNT, MODEL_DIM]));
        let mlp_proj = mlp_in.apply(&block.mlp_fc1);
        let mlp_parts = mlp_proj.split(FF_DIM, -1);
        let mlp = (mlp_parts[0].silu() * &mlp_parts[1])
            .apply(&block.mlp_fc2)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        ticker_repr = ticker_repr + &mlp * &alpha_mlp;

        // Actor: DreamerV3-style MLP → per-ticker scalar score + cash bias -> action_mean
        let mut actor_x = ticker_repr.reshape([batch_size * TICKERS_COUNT, MODEL_DIM]);
        for i in 0..self.actor_mlp_linears.len() {
            actor_x = actor_x.apply(&self.actor_mlp_linears[i]);
            actor_x = self.actor_mlp_norms[i].forward(&actor_x);
            actor_x = actor_x.silu();
        }
        let ticker_scores = actor_x
            .apply(&self.actor_score)
            .reshape([batch_size, TICKERS_COUNT]);
        let cash = self.cash_bias.expand(&[batch_size, 1], false);
        let cash = cash.to_kind(ticker_scores.kind());
        let action_mean = Tensor::cat(&[&ticker_scores, &cash], 1);

        // SDE latent: DreamerV3-style MLP per-ticker → flatten → project
        let mut sde_x = ticker_repr.reshape([batch_size * TICKERS_COUNT, MODEL_DIM]);
        for i in 0..self.sde_mlp_linears.len() {
            sde_x = sde_x.apply(&self.sde_mlp_linears[i]);
            sde_x = self.sde_mlp_norms[i].forward(&sde_x);
            sde_x = sde_x.silu();
        }
        let sde_latent = sde_x
            .reshape([batch_size, TICKERS_COUNT * MODEL_DIM])
            .apply(&self.sde_proj);

        // Critic: DreamerV3-style MLP (Linear → RMSNorm → SiLU) → Linear
        let critic_input = ticker_repr
            .reshape([batch_size, TICKERS_COUNT * MODEL_DIM]);
        let mut critic_x = critic_input.shallow_clone();
        for i in 0..self.value_mlp_linears.len() {
            critic_x = critic_x.apply(&self.value_mlp_linears[i]);
            critic_x = self.value_mlp_norms[i].forward(&critic_x);
            critic_x = critic_x.silu();
        }
        let critic_logits = critic_x.apply(&self.value_out);

        // Bins are scaled-symlog-spaced in raw return space
        // Weighted average gives value prediction directly in return space
        let values = if compute_values {
            let critic_probs = critic_logits.softmax(-1, Kind::Float);
            let bucket_centers = self.bucket_centers.to_kind(critic_probs.kind());
            let value = (&critic_probs * &bucket_centers)
                .sum_dim_intlist(-1, false, Kind::Float);
            value.to_kind(ticker_repr.kind())
        } else {
            Tensor::zeros(&[batch_size], (ticker_repr.kind(), ticker_repr.device()))
        };

        let debug_metrics = None;

        (
            (values, critic_logits, critic_input, (action_mean, sde_latent)),
            debug_metrics,
        )
    }
}
