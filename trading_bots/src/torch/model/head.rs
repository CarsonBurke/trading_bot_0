use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, FF_DIM, RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS};
use crate::torch::constants::{ACTION_COUNT, PER_TICKER_STATIC_OBS, TICKERS_COUNT};

impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
        debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let x = x_ssm.apply(&self.ssm_proj) + &self.pos_embedding;
        let temporal_len = x.size()[2];

        let x_time = x
            .view([batch_size, TICKERS_COUNT, 256, temporal_len])
            .permute([0, 3, 1, 2]);
        let x_time = x_time;
        let global_ctx = global_static
            .apply(&self.time_global_ctx)
            .unsqueeze(1)
            .unsqueeze(1);
        let ticker_ctx = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.time_ticker_ctx)
            .reshape([batch_size, TICKERS_COUNT, 256])
            .unsqueeze(1);
        let step_progress = global_static.narrow(1, 0, 1);
        let angle1 = &step_progress * (2.0 * std::f64::consts::PI);
        let angle2 = &step_progress * (4.0 * std::f64::consts::PI);
        let time_feats = Tensor::cat(&[angle1.sin(), angle1.cos(), angle2.sin(), angle2.cos()], 1);
        let time_pos = time_feats
            .apply(&self.time_pos_proj)
            .unsqueeze(1)
            .unsqueeze(1);
        let x_time = x_time + global_ctx + ticker_ctx + time_pos;
        let pos = Tensor::arange(temporal_len, (Kind::Float, x.device()));
        let (rope_cos, rope_sin) = self.rope_cos_sin(&pos, x.kind(), x.device());
        let cross_ticker_embed_base = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.cross_ticker_embed)
            .reshape([batch_size, TICKERS_COUNT, 256]);
        let cross_ticker_embed = cross_ticker_embed_base
            .unsqueeze(1)
            .expand(&[batch_size, temporal_len, TICKERS_COUNT, 256], false);
        let mut time_alpha_attn_sum = 0.0;
        let mut time_alpha_mlp_sum = 0.0;
        let alpha_scale = RESIDUAL_ALPHA_MAX / (TIME_CROSS_LAYERS as f64).sqrt();
        let block = &self.time_cross_block;
        let alpha_attn = block.alpha_attn.sigmoid() * alpha_scale;
        let alpha_mlp = block.alpha_mlp.sigmoid() * alpha_scale;
        if debug {
            time_alpha_attn_sum += f64::try_from(&alpha_attn).unwrap_or(0.0);
            time_alpha_mlp_sum += f64::try_from(&alpha_mlp).unwrap_or(0.0);
        }
        let mut x_2d = x_time + &cross_ticker_embed;
        // Permute to [batch, tickers, temporal, 256] and merge batch*tickers before projections.
        // This avoids separate copies for Q, K, V by doing one reshape early.
        let x_norm = block
            .ln
            .forward(&x_2d.reshape([batch_size * temporal_len * TICKERS_COUNT, 256]))
            .reshape([batch_size, temporal_len, TICKERS_COUNT, 256])
            .permute([0, 2, 1, 3])
            .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
        let q = x_norm
            .apply(&block.attn_q)
            .reshape([batch_size * TICKERS_COUNT, temporal_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let kv = x_norm
            .apply(&block.attn_kv)
            .reshape([batch_size * TICKERS_COUNT, temporal_len, 2, self.kv_heads, self.head_dim])
            .permute([2, 0, 1, 3, 4]);
        let (k, v) = (kv.get(0), kv.get(1));
        let k = k.permute([0, 2, 1, 3]);
        let q = self.apply_rope_cached(&q, &rope_cos, &rope_sin);
        let k = self.apply_rope_cached(&k, &rope_cos, &rope_sin);
        let q = q
            .permute([0, 2, 1, 3])
            .reshape([batch_size, TICKERS_COUNT, temporal_len, self.num_heads, self.head_dim])
            .permute([0, 3, 1, 2, 4])
            .reshape([
                batch_size,
                self.num_heads,
                TICKERS_COUNT * temporal_len,
                self.head_dim,
            ]);
        let k = k
            .permute([0, 2, 1, 3])
            .reshape([batch_size, TICKERS_COUNT, temporal_len, self.kv_heads, self.head_dim])
            .permute([0, 3, 1, 2, 4])
            .reshape([
                batch_size,
                self.kv_heads,
                TICKERS_COUNT * temporal_len,
                self.head_dim,
            ]);
        let v = v
            .permute([0, 2, 1, 3])
            .reshape([batch_size, TICKERS_COUNT, temporal_len, self.kv_heads, self.head_dim])
            .permute([0, 3, 1, 2, 4])
            .reshape([
                batch_size,
                self.kv_heads,
                TICKERS_COUNT * temporal_len,
                self.head_dim,
            ]);
        let total_len = TICKERS_COUNT as i64 * temporal_len;
        let attn_out = if self.use_sdpa(total_len) {
            Tensor::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                Option::<Tensor>::None,
                0.0,
                false,
                None,
                true,
            )
        } else {
            let kv_repeat = self.num_heads / self.kv_heads;
            let k = k
                .unsqueeze(2)
                .expand(
                    &[
                        batch_size,
                        self.kv_heads,
                        kv_repeat,
                        TICKERS_COUNT * temporal_len,
                        self.head_dim,
                    ],
                    false,
                )
                .reshape([
                    batch_size,
                    self.num_heads,
                    TICKERS_COUNT * temporal_len,
                    self.head_dim,
                ]);
            let v = v
                .unsqueeze(2)
                .expand(
                    &[
                        batch_size,
                        self.kv_heads,
                        kv_repeat,
                        TICKERS_COUNT * temporal_len,
                        self.head_dim,
                    ],
                    false,
                )
                .reshape([
                    batch_size,
                    self.num_heads,
                    TICKERS_COUNT * temporal_len,
                    self.head_dim,
                ]);
            let attn = self.attn_softmax_fp32(&q, &k);
            attn.matmul(&v)
        };
        let attn_out = attn_out
            .permute([0, 2, 1, 3])
            .contiguous()
            .view([batch_size, total_len, 256])
            .apply(&block.attn_out)
            .reshape([batch_size, TICKERS_COUNT, temporal_len, 256])
            .permute([0, 2, 1, 3]);
        let mlp_in = x_norm.reshape([batch_size * TICKERS_COUNT * temporal_len, 256]);
        let mlp_proj = mlp_in.apply(&block.mlp_fc1);
        let mlp_parts = mlp_proj.split(FF_DIM, -1);
        let mlp = (mlp_parts[0].silu() * &mlp_parts[1])
            .apply(&block.mlp_fc2)
            .reshape([batch_size, TICKERS_COUNT, temporal_len, 256])
            .permute([0, 2, 1, 3]);
        x_2d = x_2d + &attn_out * &alpha_attn + &mlp * &alpha_mlp;
        let x_time = x_2d
            .permute([0, 2, 1, 3])
            .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
        // GQA-style pooling: kv_heads query groups, no expansion needed
        // pool_queries: [kv_heads, 256] - separate learnable query per group
        let pool_queries = self.pool_queries.unsqueeze(0);
        let global_ctx = global_static
            .apply(&self.cls_global_ctx)
            .unsqueeze(1)
            .expand(&[batch_size * TICKERS_COUNT, 1, 256], false);
        let ticker_ctx = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.cls_ticker_ctx)
            .unsqueeze(1)
            .expand(&[batch_size * TICKERS_COUNT, 1, 256], false);
        let q_base = pool_queries + global_ctx + ticker_ctx;
        let q_base_norm = self
            .ln_temporal_q
            .forward(&q_base.reshape([batch_size * TICKERS_COUNT * self.kv_heads, 256]))
            .reshape([batch_size * TICKERS_COUNT, self.kv_heads, 256]);
        let x_time_norm = self
            .ln_temporal_kv
            .forward(&x_time.reshape([batch_size * TICKERS_COUNT * temporal_len, 256]))
            .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
        // Gate each query group with last token
        let last_token = x_time_norm.narrow(1, temporal_len - 1, 1);
        let last_proj = last_token
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.temporal_last)
            .unsqueeze(1)
            .expand(&[batch_size * TICKERS_COUNT, self.kv_heads, 256], false);
        let gate = q_base_norm
            .reshape([batch_size * TICKERS_COUNT * self.kv_heads, 256])
            .apply(&self.temporal_gate)
            .sigmoid()
            .reshape([batch_size * TICKERS_COUNT, self.kv_heads, 256]);
        let query_base = &q_base_norm + gate * last_proj;
        // Q: [b*t, kv_heads, head_dim] - one head per query group
        let q_pool = query_base
            .reshape([batch_size * TICKERS_COUNT * self.kv_heads, 256])
            .apply(&self.temporal_q)
            .reshape([batch_size * TICKERS_COUNT, self.kv_heads, 1, self.head_dim]);
        // K, V: [b*t, kv_heads, temporal, head_dim] - distinct per group
        let k_pool = x_time_norm
            .apply(&self.temporal_k)
            .reshape([batch_size * TICKERS_COUNT, temporal_len, self.kv_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v_pool = x_time_norm
            .apply(&self.temporal_v)
            .reshape([batch_size * TICKERS_COUNT, temporal_len, self.kv_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let tau = self.temporal_tau_raw.exp().clamp(0.5, 4.0);
        let temporal_tau = if debug {
            f64::try_from(&tau).unwrap_or(0.0)
        } else {
            0.0
        };
        let q_pool = q_pool * tau;
        // Attention: [b*t, kv_heads, 1, head_dim] @ [b*t, kv_heads, head_dim, temporal]
        let mut temporal_attn_cache: Option<Tensor> = None;
        let temporal_attn_out = if self.use_sdpa(temporal_len) {
            Tensor::scaled_dot_product_attention(
                &q_pool,
                &k_pool,
                &v_pool,
                Option::<Tensor>::None,
                0.0,
                false,
                None,
                true,
            )
            .squeeze_dim(2)
            .reshape([batch_size * TICKERS_COUNT, self.kv_heads * self.head_dim])
            .apply(&self.temporal_attn_out)
        } else {
            let attn = self.attn_softmax_fp32(&q_pool, &k_pool);
            temporal_attn_cache = Some(attn.shallow_clone());
            attn.matmul(&v_pool)
                .squeeze_dim(2)
                .reshape([batch_size * TICKERS_COUNT, self.kv_heads * self.head_dim])
                .apply(&self.temporal_attn_out)
        };
        let pooled = temporal_attn_out;
        // Aggregate query groups for residual: mean over kv_heads dimension
        let query_base_agg = query_base.mean_dim(1, false, query_base.kind());
        let conv_features = self.ln_pool_out.forward(&(pooled + query_base_agg)).reshape([
            batch_size,
            TICKERS_COUNT,
            256,
        ]);
        let conv_features = conv_features;
        // Attention shape: [b*t, kv_heads, 1, temporal] -> average over heads for visualization
        let attn_avg = tch::no_grad(|| {
            let attn =
                temporal_attn_cache.unwrap_or_else(|| self.attn_softmax_fp32(&q_pool, &k_pool));
            attn.squeeze_dim(2)
                .mean_dim(1, false, x.kind())
                .reshape([batch_size, TICKERS_COUNT, temporal_len])
                .mean_dim(1, false, x.kind())
        });
        let debug_metrics = if debug {
            let (
                temporal_attn_entropy,
                temporal_attn_max,
                temporal_attn_eff_len,
                temporal_attn_center,
                temporal_attn_last_weight,
                cross_ticker_embed_norm,
            ) = tch::no_grad(|| {
                let attn = self.attn_softmax_fp32(&q_pool, &k_pool).squeeze_dim(2);
                // attn: [b*t, kv_heads, temporal]
                let entropy = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log())
                    .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
                    .mean(Kind::Float);
                let attn_mean = attn.mean_dim([0, 1].as_slice(), false, Kind::Float);
                let attn_sum = attn_mean.sum(Kind::Float).clamp_min(1e-9);
                let attn_norm = &attn_mean / &attn_sum;
                let attn_max = attn_norm.max();
                let eff_len =
                    attn_norm.pow_tensor_scalar(2).sum(Kind::Float).clamp_min(1e-9).reciprocal();
                let positions = Tensor::arange(temporal_len, (Kind::Float, attn.device()));
                let center = (&attn_norm * &positions).sum(Kind::Float);
                let last_weight = attn_norm.narrow(0, temporal_len - 1, 1).squeeze();
                let embed_norm = cross_ticker_embed_base
                    .pow_tensor_scalar(2)
                    .mean(Kind::Float)
                    .sqrt();
                (
                    f64::try_from(&entropy).unwrap_or(0.0),
                    f64::try_from(&attn_max).unwrap_or(0.0),
                    f64::try_from(&eff_len).unwrap_or(0.0),
                    f64::try_from(&center).unwrap_or(0.0),
                    f64::try_from(&last_weight).unwrap_or(0.0),
                    f64::try_from(&embed_norm).unwrap_or(0.0),
                )
            });
            Some(DebugMetrics {
                time_alpha_attn_mean: time_alpha_attn_sum / TIME_CROSS_LAYERS as f64,
                time_alpha_mlp_mean: time_alpha_mlp_sum / TIME_CROSS_LAYERS as f64,
                cross_alpha_attn_mean: 0.0,
                cross_alpha_mlp_mean: 0.0,
                temporal_tau,
                temporal_attn_entropy,
                temporal_attn_max,
                temporal_attn_eff_len,
                temporal_attn_center,
                temporal_attn_last_weight,
                cross_ticker_embed_norm,
            })
        } else {
            None
        };

        (
            self.head_common(&conv_features, global_static, per_ticker_static, batch_size, attn_avg),
            debug_metrics,
        )
    }

    fn head_common(
        &self,
        conv_features: &Tensor,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
        attn_out_vis: Tensor,
    ) -> ModelOutput {
        let combined = Tensor::cat(
            &[
                conv_features.shallow_clone(),
                per_ticker_static.shallow_clone(),
            ],
            2,
        )
        .reshape([
            batch_size * TICKERS_COUNT,
            256 + PER_TICKER_STATIC_OBS as i64,
        ])
        .apply(&self.static_proj);
        let combined =
            self.ln_static_proj
                .forward(&combined)
                .silu()
                .reshape([batch_size, TICKERS_COUNT, 256]);

        let global_ctx = global_static.apply(&self.global_to_ticker).unsqueeze(1);
        let enriched = &combined + global_ctx;

        let pool_logits = enriched
            .shallow_clone()
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.pool_scorer)
            .reshape([batch_size, TICKERS_COUNT, 1]);
        let pool_weights = pool_logits.softmax(1, Kind::Float);
        let pool_summary = (&enriched * &pool_weights).sum_dim_intlist(1, false, Kind::Float);
        let pool_summary = pool_summary.to_kind(enriched.kind());

        let actor_hidden = enriched
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.actor_fc1);
        let actor_hidden = self.ln_actor_fc1.forward(&actor_hidden).silu().reshape([
            batch_size,
            TICKERS_COUNT,
            256,
        ]);
        let actor_residual = actor_hidden
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.actor_fc2)
            .reshape([batch_size, TICKERS_COUNT, 256]);
        let actor_feat =
            (actor_residual + &actor_hidden).reshape([batch_size * TICKERS_COUNT, 256]);
        let actor_feat =
            self.ln_actor_fc2
                .forward(&actor_feat)
                .silu()
                .reshape([batch_size, TICKERS_COUNT, 256]);
        let values_ticker = enriched
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.value_ticker_out)
            .reshape([batch_size, TICKERS_COUNT]);
        let value_cash = pool_summary
            .apply(&self.cash_value_out)
            .reshape([batch_size, 1]);
        let values = Tensor::cat(&[values_ticker, value_cash], 1);

        let ticker_logits = actor_feat.apply(&self.actor_out).squeeze_dim(-1);
        let logit_scale = self.logit_scale_raw.exp();
        let cash_logit = pool_summary.apply(&self.cash_out).squeeze_dim(-1);
        let action_logits = Tensor::cat(&[ticker_logits, cash_logit.unsqueeze(1)], 1) * &logit_scale;
        let sde_input = actor_feat.reshape([batch_size * TICKERS_COUNT, 256]);
        let sde_input = if super::SDE_LEARN_FEATURES {
            sde_input
        } else {
            sde_input.detach()
        };
        let sde_latent = sde_input.apply(&self.sde_fc);
        let sde_scale_input = if super::SDE_LEARN_FEATURES {
            enriched.shallow_clone()
        } else {
            enriched.detach()
        };
        let sde_scale_ticker = sde_scale_input
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.sde_scale_ticker)
            .sigmoid()
            .reshape([batch_size, TICKERS_COUNT, 1]);
        let sde_latent = sde_latent
            .reshape([batch_size, TICKERS_COUNT, -1])
            .g_mul(&sde_scale_ticker);
        let cash_input = if super::SDE_LEARN_FEATURES {
            pool_summary.shallow_clone()
        } else {
            pool_summary.detach()
        };
        let cash_sde = cash_input.apply(&self.cash_sde).reshape([batch_size, 1, -1]);
        let sde_scale_cash_input = if super::SDE_LEARN_FEATURES {
            pool_summary.shallow_clone()
        } else {
            pool_summary.detach()
        };
        let sde_scale_cash = sde_scale_cash_input
            .apply(&self.sde_scale_cash)
            .sigmoid()
            .reshape([batch_size, 1, 1]);
        let cash_sde = cash_sde
            .g_mul(&sde_scale_cash)
            .reshape([batch_size, 1, -1]);
        let sde_latent = Tensor::cat(&[sde_latent, cash_sde], 1);
        let std = self.sde_std_matrix();
        let std_sq = std.pow_tensor_scalar(2).transpose(0, 1);
        let variance =
            (sde_latent.pow_tensor_scalar(2) * std_sq.unsqueeze(0))
                .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let action_log_std = (variance + super::SDE_EPS).sqrt().log();

        (
            values,
            (action_logits, action_log_std, sde_latent),
            attn_out_vis,
        )
    }
}
