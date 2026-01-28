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
        let exo_ticker_base = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.time_ticker_ctx)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        let exo_ticker_embed = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.cross_ticker_embed)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        let exo_ticker = &exo_ticker_base + &exo_ticker_embed;
        let step_progress = global_static.narrow(1, 0, 1);
        let angle1 = &step_progress * (2.0 * std::f64::consts::PI);
        let angle2 = &step_progress * (4.0 * std::f64::consts::PI);
        let time_feats = Tensor::cat(&[angle1.sin(), angle1.cos(), angle2.sin(), angle2.cos()], 1);
        let time_pos = time_feats
            .apply(&self.time_pos_proj)
            .unsqueeze(1)
            .unsqueeze(1);
        let exo_global = global_ctx.squeeze_dim(1).squeeze_dim(1);
        let exo_time = time_pos.squeeze_dim(1).squeeze_dim(1);
        let exo_global = exo_global
            .unsqueeze(1)
            .expand(&[batch_size, TICKERS_COUNT, MODEL_DIM], false);
        let exo_time = exo_time
            .unsqueeze(1)
            .expand(&[batch_size, TICKERS_COUNT, MODEL_DIM], false);
        let exo_tokens = Tensor::cat(
            &[
                exo_global.unsqueeze(2),
                exo_ticker.unsqueeze(2),
                exo_time.unsqueeze(2),
            ],
            2,
        ); // [batch, tickers, 3, dim]

        // Cross-attention on last tokens only: [batch, tickers, dim] queries exo_tokens
        let last_tokens_flat = last_tokens
            .reshape([batch_size * TICKERS_COUNT, 1, MODEL_DIM]);
        let last_tokens_norm = self.last_token_ln.forward(&last_tokens_flat);
        let exo_tokens_flat = exo_tokens.reshape([batch_size * TICKERS_COUNT, 3, MODEL_DIM]);
        let q_exo = last_tokens_norm.apply(&self.static_cross_q);
        let k_exo = exo_tokens_flat.apply(&self.static_cross_k);
        let v_exo = exo_tokens_flat.apply(&self.static_cross_v);
        let exo_scores = q_exo.matmul(&k_exo.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt();
        let exo_attn = exo_scores.softmax(-1, Kind::Float).to_kind(q_exo.kind());
        let exo_ctx = exo_attn
            .matmul(&v_exo)
            .apply(&self.static_cross_out)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);

        // Add exo context and global injection to pooled tokens
        let mut pooled = &last_tokens + exo_ctx; // [batch, tickers, dim]
        let global_inject = global_static
            .apply(&self.global_inject_down)
            .apply(&self.global_inject_up);
        let global_inject = global_inject * self.global_inject_gate_raw.sigmoid();
        pooled = pooled + global_inject.unsqueeze(1);

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

        // Combine with per-ticker static and apply projection
        let combined = Tensor::cat(&[pooled, per_ticker_static.shallow_clone()], 2)
            .reshape([
                batch_size * TICKERS_COUNT,
                MODEL_DIM + PER_TICKER_STATIC_OBS as i64,
            ])
            .apply(&self.static_proj);
        let combined = self
            .ln_static_proj
            .forward(&combined)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]);
        let global_ctx = global_static
            .apply(&self.global_to_ticker)
            .unsqueeze(1);
        let pooled_enriched = combined + global_ctx;

        // Cash pathway for policy (still needed for actor)
        let cash_summary = pooled_enriched
            .mean_dim([1].as_slice(), false, pooled_enriched.kind())
            .apply(&self.cash_merge);
        let attn_entropy = Tensor::zeros([batch_size], (Kind::Float, pooled_enriched.device()));

        // Portfolio-level critic: pool all tickers into single representation
        let portfolio_repr = pooled_enriched
            .mean_dim([1].as_slice(), false, pooled_enriched.kind()); // [batch, dim]
        let value_hidden = self
            .value_ln
            .forward(&portfolio_repr)
            .apply(&self.value_mlp_fc1)
            .silu()
            .apply(&self.value_mlp_fc2);
        let critic_logits = value_hidden.apply(&self.critic_out); // [batch, NUM_VALUE_BUCKETS]
        let critic_probs = critic_logits.softmax(-1, Kind::Float);
        let bucket_centers = self.bucket_centers.to_kind(critic_probs.kind());
        let values = critic_probs
            .matmul(&bucket_centers.unsqueeze(-1))
            .squeeze_dim(-1); // [batch]
        let values = values.to_kind(pooled_enriched.kind());

        // Batch policy computation: combine ticker and cash inputs, apply policy_ln + head_proj once
        let pooled_flat = pooled_enriched.reshape([batch_size * TICKERS_COUNT, MODEL_DIM]);
        let cash_policy_flat = cash_summary.view([batch_size, MODEL_DIM]);
        let policy_input = Tensor::cat(&[pooled_flat, cash_policy_flat], 0);
        let policy_hidden = self
            .policy_ln
            .forward(&policy_input)
            .apply(&self.head_proj);
        let policy_hidden = policy_hidden.silu();
        let ticker_head_base = policy_hidden
            .narrow(0, 0, batch_size * TICKERS_COUNT)
            .reshape([batch_size, TICKERS_COUNT, super::HEAD_HIDDEN]);
        let cash_head_base = policy_hidden.narrow(0, batch_size * TICKERS_COUNT, batch_size);

        // Action mean (logits before softmax)
        let action_mean_ticker = ticker_head_base.apply(&self.actor_out).squeeze_dim(-1);
        let cash_logit = cash_head_base.apply(&self.actor_out).squeeze_dim(-1);
        let action_mean = Tensor::cat(&[action_mean_ticker, cash_logit.unsqueeze(1)], 1);

        // gSDE: state-dependent exploration via learned latent features
        // Tanh bounds sde_latent to [-1, 1] to prevent exploration collapse (matches SB3 convention)
        let sde_input = ticker_head_base.reshape([batch_size * TICKERS_COUNT, super::HEAD_HIDDEN]);
        let sde_latent = sde_input.apply(&self.sde_fc).tanh();
        let sde_latent = sde_latent.reshape([batch_size, TICKERS_COUNT, super::SDE_LATENT_DIM]);

        let debug_metrics = None;

        (
            (
                values,
                critic_logits,
                (action_mean, sde_latent),
                attn_entropy,
            ),
            debug_metrics,
        )
    }
}
