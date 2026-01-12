use tch::{Kind, Tensor};

use super::{
    symexp_tensor, DebugMetrics, ModelOutput, TradingModel, FF_DIM, MODEL_DIM, PMA_QUERIES,
    RESIDUAL_ALPHA_MAX, TEMPORAL_POOL_GROUPS, TIME_CROSS_LAYERS,
};
use crate::torch::constants::{PER_TICKER_STATIC_OBS, TICKERS_COUNT};

const POOL_TEMPERATURE_MIN: f64 = 0.1;
const POOL_TEMPERATURE_MAX: f64 = 8.0;

impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let x = self.post_ssm_ln.forward(x_ssm);
        let x = &x * x.apply(&self.ssm_gate).sigmoid();
        let x = x.permute([0, 2, 1]);
        let mut x = x.apply(&self.ssm_proj);
        let temporal_len = x.size()[2];

        let x_time = x
            .view([batch_size, TICKERS_COUNT, MODEL_DIM, temporal_len])
            .permute([0, 3, 1, 2]);
        let x_time = x_time;
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
        );
        let mut global_token = self
            .global_ticker_token
            .unsqueeze(0)
            .expand(&[batch_size, TICKERS_COUNT, MODEL_DIM], false);
        global_token = global_token + &exo_ticker + &exo_global + &exo_time;
        let global_token = global_token.unsqueeze(1);
        let mut x_time = Tensor::cat(&[x_time, global_token], 1);
        let temporal_len_all = temporal_len + 1;
        let x_time_flat = x_time
            .permute([0, 2, 1, 3])
            .reshape([batch_size * TICKERS_COUNT, temporal_len_all, MODEL_DIM]);
        let exo_tokens_flat =
            exo_tokens.reshape([batch_size * TICKERS_COUNT, 3, MODEL_DIM]);
        let q_exo = x_time_flat.apply(&self.static_cross_q);
        let k_exo = exo_tokens_flat.apply(&self.static_cross_k);
        let v_exo = exo_tokens_flat.apply(&self.static_cross_v);
        let exo_scores = q_exo.matmul(&k_exo.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt();
        let exo_attn = exo_scores.softmax(-1, Kind::Float).to_kind(q_exo.kind());
        let exo_ctx = exo_attn
            .matmul(&v_exo)
            .apply(&self.static_cross_out)
            .reshape([batch_size, TICKERS_COUNT, temporal_len_all, MODEL_DIM])
            .permute([0, 2, 1, 3]);
        x_time = x_time + exo_ctx;
        let global_inject = global_static
            .apply(&self.global_inject_down)
            .apply(&self.global_inject_up);
        let global_inject = global_inject * self.global_inject_gate_raw.sigmoid();
        x_time = x_time + global_inject.unsqueeze(1).unsqueeze(1);
        let alpha_scale = RESIDUAL_ALPHA_MAX / (TIME_CROSS_LAYERS as f64).sqrt();
        let block = &self.inter_ticker_block;
        let alpha_ticker_attn = block.alpha_ticker_attn.sigmoid() * alpha_scale;
        let alpha_mlp = block.alpha_mlp.sigmoid() * alpha_scale;
        let mut x_2d = x_time;
        let btk = batch_size * temporal_len_all * TICKERS_COUNT;
        let bt = batch_size * temporal_len_all;

        let x_ticker_norm = block
            .ticker_ln
            .forward(&x_2d.reshape([btk, MODEL_DIM]))
            .reshape([bt, TICKERS_COUNT, MODEL_DIM]);
        let latent_q = x_ticker_norm.apply(&block.ticker_latent_q);
        let latent_k = block.ticker_latent_k.to_kind(latent_q.kind());
        let latent_scores =
            latent_q.matmul(&latent_k.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt();
        let latent_attn = latent_scores.softmax(-1, Kind::Float).to_kind(latent_q.kind());
        let latent_v = block.ticker_latent_v.to_kind(latent_q.kind());
        let latent_ctx = latent_attn
            .matmul(&latent_v)
            .apply(&block.ticker_out)
            .reshape([batch_size, temporal_len_all, TICKERS_COUNT, MODEL_DIM]);
        x_2d = x_2d + &latent_ctx * &alpha_ticker_attn;

        let mlp_in = block
            .mlp_ln
            .forward(&x_2d.reshape([btk, MODEL_DIM]));
        let mlp_proj = mlp_in.apply(&block.mlp_fc1);
        let mlp_parts = mlp_proj.split(FF_DIM, -1);
        let mlp = (mlp_parts[0].silu() * &mlp_parts[1])
            .apply(&block.mlp_fc2)
            .reshape([batch_size, temporal_len_all, TICKERS_COUNT, MODEL_DIM]);
        x_2d = x_2d + &mlp * &alpha_mlp;
        let x_time = x_2d.permute([0, 2, 1, 3]).narrow(2, 0, temporal_len);
        let per_ticker_static_expanded = per_ticker_static
            .unsqueeze(2)
            .expand(
                &[
                    batch_size,
                    TICKERS_COUNT,
                    temporal_len,
                    PER_TICKER_STATIC_OBS as i64,
                ],
                false,
            );
        let combined = Tensor::cat(&[x_time, per_ticker_static_expanded], 3)
            .reshape([
                batch_size * TICKERS_COUNT * temporal_len,
                MODEL_DIM + PER_TICKER_STATIC_OBS as i64,
            ])
            .apply(&self.static_proj);
        let combined = self
            .ln_static_proj
            .forward(&combined)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, temporal_len, MODEL_DIM]);
        let global_ctx = global_static
            .apply(&self.global_to_ticker)
            .unsqueeze(1)
            .unsqueeze(1);
        let enriched = combined + global_ctx;

        let pooled_input = enriched
            .reshape([batch_size * TICKERS_COUNT, temporal_len, MODEL_DIM])
            .permute([0, 2, 1]);
        let static_proj = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.static_to_temporal);
        let static_proj = self
            .ln_static_temporal
            .forward(&static_proj)
            .unsqueeze(2)
            .expand(&[-1, -1, temporal_len], false);
        let combined_full = Tensor::cat(&[pooled_input.shallow_clone(), static_proj], 1);
        let combined_full_t = combined_full.permute([0, 2, 1]);
        let bt = batch_size * TICKERS_COUNT;
        let q = self
            .pma_queries
            .unsqueeze(0)
            .expand(&[bt, PMA_QUERIES, MODEL_DIM], false)
            .apply(&self.pma_q_proj);
        let k = combined_full_t.apply(&self.pma_k_proj);
        let v = combined_full_t.apply(&self.pma_v_proj);
        let q = q
            .view([bt, PMA_QUERIES, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .view([bt, temporal_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .view([bt, temporal_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let scores = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt();
        let attn = scores.softmax(-1, Kind::Float).to_kind(q.kind());
        let ctx = attn
            .matmul(&v)
            .permute([0, 2, 1, 3])
            .reshape([bt, PMA_QUERIES, MODEL_DIM])
            .apply(&self.pma_out);
        let pooled_enriched = ctx
            .mean_dim([1].as_slice(), false, Kind::Float)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM])
            .to_kind(enriched.kind());

        let head_base = self
            .policy_ln
            .forward(&pooled_enriched)
            .reshape([batch_size * TICKERS_COUNT, MODEL_DIM])
            .apply(&self.head_proj);
        let head_base = self
            .head_ln
            .forward(&head_base)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, super::HEAD_HIDDEN]);

        let token_seq = enriched.reshape([batch_size, TICKERS_COUNT * temporal_len, MODEL_DIM]);
        let cash_slope =
            self.cash_recent_slope_raw.sigmoid() * (2.0 / temporal_len as f64);
        let positions = self.decay_positions.narrow(0, 0, temporal_len);
        let distances = (temporal_len - 1) as f64 - &positions;
        let ones = self.decay_ones.narrow(0, 0, temporal_len);
        let cash_decay = (&ones - &distances * &cash_slope).clamp_min(0.0);
        let cash_decay = &cash_decay / cash_decay.sum(Kind::Float).clamp_min(1e-6);
        let cash_decay = cash_decay
            .reshape([1, 1, temporal_len, 1])
            .to_kind(enriched.kind());
        let cash_recent = (&enriched * &cash_decay)
            .sum_dim_intlist([2].as_slice(), false, enriched.kind())
            .mean_dim([1].as_slice(), false, enriched.kind());
        let cash_recent_proj = cash_recent.apply(&self.cash_recent_proj);
        let cash_recent_gate = cash_recent.apply(&self.cash_recent_gate).sigmoid();
        let cash_q = self
            .cash_queries
            .unsqueeze(0)
            .expand(&[batch_size, self.cash_pool_queries, MODEL_DIM], false)
            + (cash_recent_proj * cash_recent_gate).unsqueeze(1);
        let cash_q = cash_q
            .apply(&self.cash_q_proj);
        let cash_k = token_seq.apply(&self.cash_k_proj);
        let cash_v = token_seq.apply(&self.cash_v_proj);
        let cash_temp = self.cash_attn_temp_raw.exp().clamp(POOL_TEMPERATURE_MIN, POOL_TEMPERATURE_MAX);
        let scores =
            cash_q.matmul(&cash_k.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt() * cash_temp;
        let cash_attn_f = scores.softmax(-1, Kind::Float).to_kind(cash_q.kind());
        let cash_ctx = cash_attn_f.matmul(&cash_v);
        let global_tokens = self
            .global_tokens
            .unsqueeze(0)
            .expand(&[batch_size, super::GLOBAL_TOKEN_COUNT, MODEL_DIM], false)
            + global_static
                .apply(&self.global_token_proj)
                .unsqueeze(1);
        let global_summary = global_tokens
            .mean_dim([1].as_slice(), false, global_tokens.kind())
            .apply(&self.global_token_merge);
        let cash_summary = cash_ctx
            .mean_dim(1, false, cash_ctx.kind())
            .apply(&self.cash_merge);
        let cash_summary = cash_summary + global_summary;
        let entropy_denom = (temporal_len as f64).ln().max(1.0);
        let temporal_attn = attn
            .mean_dim([1, 2].as_slice(), false, Kind::Float)
            .reshape([batch_size, TICKERS_COUNT, temporal_len]);
        let temporal_attn_entropy = -(temporal_attn.clamp_min(1e-9)
            * temporal_attn.clamp_min(1e-9).log())
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .mean_dim([1].as_slice(), false, Kind::Float)
            / entropy_denom;
        let cash_attn_entropy = -(cash_attn_f.clamp_min(1e-9)
            * cash_attn_f.clamp_min(1e-9).log())
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .mean_dim([1].as_slice(), false, Kind::Float)
            / entropy_denom;
        let attn_entropy =
            (temporal_attn_entropy + cash_attn_entropy).to_kind(pooled_enriched.kind());

        let value_tokens = self
            .value_ln
            .forward(&pooled_enriched)
            .apply(&self.value_mlp_fc1)
            .silu()
            .apply(&self.value_mlp_fc2);
        let value_logits_ticker = value_tokens
            .reshape([batch_size * TICKERS_COUNT, MODEL_DIM])
            .apply(&self.critic_out)
            .reshape([batch_size, TICKERS_COUNT, super::NUM_VALUE_BUCKETS]);
        let value_cash = self
            .value_ln
            .forward(&cash_summary)
            .apply(&self.value_mlp_fc1)
            .silu()
            .apply(&self.value_mlp_fc2)
            .apply(&self.critic_out)
            .reshape([batch_size, 1, super::NUM_VALUE_BUCKETS]);
        let critic_logits = Tensor::cat(&[value_logits_ticker, value_cash], 1);
        let critic_probs = critic_logits.softmax(-1, Kind::Float);
        let bucket_centers = self.bucket_centers.to_kind(critic_probs.kind());
        let values_symlog = critic_probs
            .matmul(&bucket_centers.unsqueeze(-1))
            .squeeze_dim(-1);
        let values = symexp_tensor(&values_symlog).to_kind(pooled_enriched.kind());

        let action_mean_ticker = head_base.apply(&self.actor_out).squeeze_dim(-1);
        let cash_head_base = self
            .policy_ln
            .forward(&cash_summary)
            .apply(&self.head_proj);
        let cash_head_base = self.head_ln.forward(&cash_head_base).silu();
        let cash_logit = cash_head_base.apply(&self.actor_out).squeeze_dim(-1);
        let action_mean = Tensor::cat(&[action_mean_ticker, cash_logit.unsqueeze(1)], 1);
        let sde_input = head_base
            .reshape([batch_size * TICKERS_COUNT, super::HEAD_HIDDEN]);
        let sde_latent = sde_input.apply(&self.sde_fc);
        let sde_latent = (self.ln_sde.forward(&sde_latent) / 1.5).tanh();
        let sde_latent = sde_latent.reshape([batch_size, TICKERS_COUNT, -1]);
        let log_std = (&self.log_std_param + super::LOG_STD_INIT).clamp(-3.0, -0.5);
        let std_sq = log_std.exp().pow_tensor_scalar(2).transpose(0, 1);
        let variance = (sde_latent.pow_tensor_scalar(2) * std_sq.unsqueeze(0))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let action_log_std_ticker = (variance + super::SDE_EPS).sqrt().log();
        let cash_log_std = (&self.cash_log_std_param + super::LOG_STD_INIT).clamp(-3.0, -0.5);
        let action_log_std = Tensor::cat(
            &[
                action_log_std_ticker,
                cash_log_std.expand(&[batch_size, 1], false),
            ],
            1,
        );

        let debug_metrics = None;

        (
            (
                values,
                critic_logits,
                (action_mean, action_log_std),
                attn_entropy,
            ),
            debug_metrics,
        )
    }
}
