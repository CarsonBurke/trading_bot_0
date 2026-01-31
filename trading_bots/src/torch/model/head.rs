use tch::{Kind, Tensor};

use super::{
    DebugMetrics, ModelOutput, TradingModel, FF_DIM, MODEL_DIM,
    RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS,
};
use crate::torch::constants::{PER_TICKER_STATIC_OBS, TICKERS_COUNT};


impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
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

        // Build exo context for last tokens only
        let global_ctx = global_static
            .apply(&self.time_global_ctx)
            .unsqueeze(1)
            .unsqueeze(1);
        let exo_ticker = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.time_ticker_ctx)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        let exo_global = global_ctx.squeeze_dim(1).squeeze_dim(1)
            .unsqueeze(1)
            .expand(&[batch_size, TICKERS_COUNT, MODEL_DIM], false);
        let exo_tokens = Tensor::cat(
            &[
                exo_global.unsqueeze(2),
                exo_ticker.unsqueeze(2),
            ],
            2,
        ); // [batch, tickers, 2, dim]

        // Cross-attention on last tokens only: [batch, tickers, dim] queries exo_tokens
        let last_tokens_flat = last_tokens
            .reshape([batch_size * TICKERS_COUNT, 1, MODEL_DIM]);
        let exo_tokens_flat = exo_tokens.reshape([batch_size * TICKERS_COUNT, 2, MODEL_DIM]);
        let q_exo = last_tokens_flat.apply(&self.static_cross_q);
        let k_exo = exo_tokens_flat.apply(&self.static_cross_k);
        let v_exo = exo_tokens_flat.apply(&self.static_cross_v);
        let exo_scores = q_exo.matmul(&k_exo.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt();
        let exo_attn = exo_scores.softmax(-1, Kind::Float).to_kind(q_exo.kind());
        let exo_ctx = exo_attn
            .matmul(&v_exo)
            .apply(&self.static_cross_out)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);

        // Add exo context to pooled tokens
        let mut pooled = &last_tokens + exo_ctx; // [batch, tickers, dim]

        // InterTickerBlock on pooled tokens: [batch, tickers, dim]
        let alpha_scale = RESIDUAL_ALPHA_MAX / (TIME_CROSS_LAYERS as f64).sqrt();
        let block = &self.inter_ticker_block;
        let alpha_ticker_attn = block.alpha_ticker_attn.sigmoid() * alpha_scale;
        let alpha_mlp = block.alpha_mlp.sigmoid() * alpha_scale;

        let x_ticker_norm = block
            .ticker_ln
            .forward(&pooled.reshape([batch_size * TICKERS_COUNT, MODEL_DIM]))
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        let latent_q = x_ticker_norm.apply(&block.ticker_latent_q);
        let kind = latent_q.kind();
        let latent_k_learned = if block.ticker_latent_k.kind() == kind {
            block.ticker_latent_k.shallow_clone()
        } else {
            block.ticker_latent_k.to_kind(kind)
        };
        let latent_k_frozen = if block.ticker_latent_k_frozen.kind() == kind {
            block.ticker_latent_k_frozen.shallow_clone()
        } else {
            block.ticker_latent_k_frozen.to_kind(kind)
        };
        let latent_k = Tensor::cat(&[&latent_k_learned, &latent_k_frozen], 0);
        let latent_scores =
            latent_q.matmul(&latent_k.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt();
        let latent_attn = latent_scores.softmax(-1, Kind::Float).to_kind(kind);
        let latent_v_base = if block.ticker_latent_v.kind() == kind {
            block.ticker_latent_v.shallow_clone()
        } else {
            block.ticker_latent_v.to_kind(kind)
        };
        let latent_v = Tensor::cat(&[&latent_v_base, &latent_v_base], 0);
        let latent_ctx = latent_attn
            .matmul(&latent_v)
            .apply(&block.ticker_out)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        pooled = pooled + &latent_ctx * &alpha_ticker_attn;

        let mlp_in = block
            .mlp_ln
            .forward(&pooled.reshape([batch_size * TICKERS_COUNT, MODEL_DIM]));
        let mlp_proj = mlp_in.apply(&block.mlp_fc1);
        let mlp_parts = mlp_proj.split(FF_DIM, -1);
        let mlp = (mlp_parts[0].silu() * &mlp_parts[1])
            .apply(&block.mlp_fc2)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        pooled = pooled + &mlp * &alpha_mlp;

        // Residual static enrichment: SSM signal preserved through skip connection
        let static_ctx = Tensor::cat(&[pooled.shallow_clone(), per_ticker_static.shallow_clone()], 2)
            .reshape([
                batch_size * TICKERS_COUNT,
                MODEL_DIM + PER_TICKER_STATIC_OBS as i64,
            ])
            .apply(&self.static_proj)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        let pooled_enriched = pooled + static_ctx;

        // Flatten per-ticker features, shared by actor and critic
        let flat_tickers = pooled_enriched.reshape([batch_size, TICKERS_COUNT * MODEL_DIM]);

        // Critic branch (reads flat_tickers directly)
        let value_hidden = self
            .value_ln
            .forward(&flat_tickers)
            .apply(&self.value_mlp_fc1)
            .silu()
            .apply(&self.value_mlp_fc2);
        let critic_logits = value_hidden.apply(&self.critic_out); // [batch, NUM_VALUE_BUCKETS]
        let values = if compute_values {
            let critic_probs = critic_logits.softmax(-1, Kind::Float);
            let bucket_centers = self.bucket_centers.to_kind(critic_probs.kind());
            let values_symlog = critic_probs
                .matmul(&bucket_centers.unsqueeze(-1))
                .squeeze_dim(-1); // [batch]
            super::symexp_tensor(&values_symlog).to_kind(pooled_enriched.kind())
        } else {
            Tensor::zeros(&[batch_size], (pooled_enriched.kind(), pooled_enriched.device()))
        };

        // Actor branch: dedicated MLP produces sde_latent (decoupled from critic)
        // No trailing activation — sde_latent h² must span both signs for Lattice covariance
        let sde_latent = flat_tickers
            .apply(&self.actor_mlp_fc1)
            .silu()
            .apply(&self.actor_mlp_fc2); // [batch, SDE_LATENT_DIM]
        let action_mean = sde_latent.apply(&self.actor_out);

        let debug_metrics = None;

        (
            (values, critic_logits, (action_mean, sde_latent)),
            debug_metrics,
        )
    }
}
