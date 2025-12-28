use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, FF_DIM, RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS};
use crate::torch::constants::{PER_TICKER_STATIC_OBS, TICKERS_COUNT};

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
        let mut x_seq = x_time
            .reshape([batch_size, temporal_len, TICKERS_COUNT, 256])
            .permute([0, 2, 1, 3])
            .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
        let cross_ticker_embed_base = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.cross_ticker_embed)
            .reshape([batch_size, TICKERS_COUNT, 256]);
        let cross_ticker_embed = cross_ticker_embed_base
            .unsqueeze(1)
            .expand(&[batch_size, temporal_len, TICKERS_COUNT, 256], false);
        let mut time_alpha_attn_sum = 0.0;
        let mut time_alpha_mlp_sum = 0.0;
        let mut cross_alpha_attn_sum = 0.0;
        let mut cross_alpha_mlp_sum = 0.0;
        let alpha_scale = RESIDUAL_ALPHA_MAX / (TIME_CROSS_LAYERS as f64).sqrt();
        for block in &self.time_cross_blocks {
            let alpha_time_attn = block.alpha_time_attn.sigmoid() * alpha_scale;
            let alpha_time_mlp = block.alpha_time_mlp.sigmoid() * alpha_scale;
            if debug {
                time_alpha_attn_sum += f64::try_from(&alpha_time_attn).unwrap_or(0.0);
                time_alpha_mlp_sum += f64::try_from(&alpha_time_mlp).unwrap_or(0.0);
            }
            let x_time_norm = block
                .ln_time
                .forward(&x_seq.reshape([batch_size * TICKERS_COUNT * temporal_len, 256]))
                .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
            let q = x_time_norm
                .apply(&block.time_attn_q)
                .reshape([batch_size * TICKERS_COUNT, temporal_len, self.num_heads, self.head_dim])
                .permute([0, 2, 1, 3]);
            let kv = x_time_norm
                .apply(&block.time_attn_kv)
                .reshape([
                    batch_size * TICKERS_COUNT,
                    temporal_len,
                    2,
                    self.kv_heads,
                    self.head_dim,
                ])
                .permute([2, 0, 3, 1, 4]);
            let (k, v) = (kv.get(0), kv.get(1));
            let pos = Tensor::arange(temporal_len, (Kind::Float, x.device()));
            let q = self.apply_rope_single(&q, &pos);
            let k = self.apply_rope_single(&k, &pos);
            let kv_repeat = self.num_heads / self.kv_heads;
            let k = k.repeat(&[1, kv_repeat, 1, 1]);
            let v = v.repeat(&[1, kv_repeat, 1, 1]);
            let time_attn_out = if self.use_sdpa(temporal_len) {
                Tensor::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    Option::<Tensor>::None,
                    0.0,
                    false,
                    None,
                    false,
                )
                .permute([0, 2, 1, 3])
                .contiguous()
                .view([batch_size * TICKERS_COUNT, temporal_len, 256])
                .apply(&block.time_attn_out)
            } else {
                let attn = self.attn_softmax_fp32(&q, &k);
                attn.matmul(&v)
                    .permute([0, 2, 1, 3])
                    .contiguous()
                    .view([batch_size * TICKERS_COUNT, temporal_len, 256])
                    .apply(&block.time_attn_out)
            };
            let time_mlp_in =
                x_time_norm.reshape([batch_size * TICKERS_COUNT * temporal_len, 256]);
            let time_mlp_proj = time_mlp_in.apply(&block.time_mlp_fc1);
            let time_mlp_parts = time_mlp_proj.split(FF_DIM, -1);
            let time_mlp = (time_mlp_parts[0].silu() * &time_mlp_parts[1])
                .apply(&block.time_mlp_fc2)
                .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
            x_seq = x_seq + &time_attn_out * &alpha_time_attn + &time_mlp * &alpha_time_mlp;

            let alpha_cross_attn = block.alpha_cross_attn.sigmoid() * alpha_scale;
            let alpha_cross_mlp = block.alpha_cross_mlp.sigmoid() * alpha_scale;
            if debug {
                cross_alpha_attn_sum += f64::try_from(&alpha_cross_attn).unwrap_or(0.0);
                cross_alpha_mlp_sum += f64::try_from(&alpha_cross_mlp).unwrap_or(0.0);
            }
            let x_cross = x_seq
                .reshape([batch_size, TICKERS_COUNT, temporal_len, 256])
                .permute([0, 2, 1, 3]);
            let x_cross_norm = block
                .ln_cross
                .forward(&x_cross.reshape([batch_size * temporal_len * TICKERS_COUNT, 256]))
                .reshape([batch_size, temporal_len, TICKERS_COUNT, 256])
                + &cross_ticker_embed;
            let qkv = x_cross_norm
                .apply(&block.cross_attn_qkv)
                .reshape([batch_size, temporal_len, TICKERS_COUNT, 3, self.num_heads, self.head_dim])
                .permute([3, 0, 1, 4, 2, 5]);
            let (q, k, v) = (qkv.get(0), qkv.get(1), qkv.get(2));
            let q = q.reshape([batch_size * temporal_len, self.num_heads, TICKERS_COUNT, self.head_dim]);
            let k = k.reshape([batch_size * temporal_len, self.num_heads, TICKERS_COUNT, self.head_dim]);
            let v = v.reshape([batch_size * temporal_len, self.num_heads, TICKERS_COUNT, self.head_dim]);
            let cross_attn_out = if self.use_sdpa(TICKERS_COUNT as i64) {
                Tensor::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    Option::<Tensor>::None,
                    0.0,
                    false,
                    None,
                    false,
                )
                .reshape([batch_size, temporal_len, self.num_heads, TICKERS_COUNT, self.head_dim])
                .permute([0, 1, 3, 2, 4])
                .contiguous()
                .view([batch_size, temporal_len, TICKERS_COUNT, 256])
                .apply(&block.cross_attn_out)
            } else {
                let attn = self.attn_softmax_fp32(&q, &k);
                attn.matmul(&v)
                    .reshape([batch_size, temporal_len, self.num_heads, TICKERS_COUNT, self.head_dim])
                    .permute([0, 1, 3, 2, 4])
                    .contiguous()
                    .view([batch_size, temporal_len, TICKERS_COUNT, 256])
                    .apply(&block.cross_attn_out)
            };
            let cross_mlp_in =
                x_cross_norm.reshape([batch_size * temporal_len * TICKERS_COUNT, 256]);
            let cross_mlp_proj = cross_mlp_in.apply(&block.cross_mlp_fc1);
            let cross_mlp_parts = cross_mlp_proj.split(FF_DIM, -1);
            let cross_mlp = (cross_mlp_parts[0].silu() * &cross_mlp_parts[1])
                .apply(&block.cross_mlp_fc2)
                .reshape([batch_size, temporal_len, TICKERS_COUNT, 256]);
            let x_cross = x_cross + &cross_attn_out * &alpha_cross_attn + &cross_mlp * &alpha_cross_mlp;
            x_seq = x_cross
                .permute([0, 2, 1, 3])
                .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
        }
        let x_time = x_seq;
        let global_ctx = global_static
            .apply(&self.cls_global_ctx)
            .unsqueeze(1)
            .expand(&[batch_size, TICKERS_COUNT, 256], false);
        let ticker_ctx = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.cls_ticker_ctx)
            .reshape([batch_size, TICKERS_COUNT, 256]);
        let cls_base = self
            .cls_token
            .expand(&[batch_size, TICKERS_COUNT, 256], false);
        let q_base =
            (cls_base + global_ctx + ticker_ctx).reshape([batch_size * TICKERS_COUNT, 256]);
        let q_base_norm = self
            .ln_temporal_q
            .forward(&q_base)
            .reshape([batch_size * TICKERS_COUNT, 256]);
        let x_time_norm = self
            .ln_temporal_kv
            .forward(&x_time.reshape([batch_size * TICKERS_COUNT * temporal_len, 256]))
            .reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
        let last_token = x_time_norm.narrow(1, temporal_len - 1, 1).squeeze_dim(1);
        let gate = q_base_norm
            .apply(&self.temporal_gate)
            .sigmoid();
        let last_proj = last_token.apply(&self.temporal_last);
        let query_base = &q_base_norm + gate * last_proj;
        let q_pool = query_base
            .apply(&self.temporal_q)
            .reshape([batch_size * TICKERS_COUNT, 1, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k_pool = x_time_norm
            .apply(&self.temporal_k)
            .reshape([
                batch_size * TICKERS_COUNT,
                temporal_len,
                self.kv_heads,
                self.head_dim,
            ])
            .permute([0, 2, 1, 3]);
        let v_pool = x_time_norm
            .apply(&self.temporal_v)
            .reshape([
                batch_size * TICKERS_COUNT,
                temporal_len,
                self.kv_heads,
                self.head_dim,
            ])
            .permute([0, 2, 1, 3]);
        let pos = Tensor::arange(temporal_len, (Kind::Float, x.device()));
        let q_pos = Tensor::from_slice(&[(temporal_len - 1) as f64]).to_device(x.device());
        let q_pool = self.apply_rope_single(&q_pool, &q_pos);
        let k_pool = self.apply_rope_single(&k_pool, &pos);
        let kv_repeat = self.num_heads / self.kv_heads;
        let k_pool = k_pool.repeat(&[1, kv_repeat, 1, 1]);
        let v_pool = v_pool.repeat(&[1, kv_repeat, 1, 1]);
        let tau = self.temporal_tau_raw.exp().clamp(0.5, 4.0);
        let temporal_tau = if debug {
            f64::try_from(&tau).unwrap_or(0.0)
        } else {
            0.0
        };
        let q_pool = q_pool * tau;
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
                false,
            )
            .permute([0, 2, 1, 3])
            .contiguous()
            .view([batch_size * TICKERS_COUNT, 1, 256])
            .apply(&self.temporal_attn_out)
        } else {
            let attn = self.attn_softmax_fp32(&q_pool, &k_pool);
            temporal_attn_cache = Some(attn.shallow_clone());
            attn.matmul(&v_pool)
                .permute([0, 2, 1, 3])
                .contiguous()
                .view([batch_size * TICKERS_COUNT, 1, 256])
                .apply(&self.temporal_attn_out)
        };
        let pooled = temporal_attn_out.squeeze_dim(1);
        let conv_features = self.ln_pool_out.forward(&(pooled + query_base)).reshape([
            batch_size,
            TICKERS_COUNT,
            256,
        ]);
        let attn_avg = tch::no_grad(|| {
            let attn =
                temporal_attn_cache.unwrap_or_else(|| self.attn_softmax_fp32(&q_pool, &k_pool));
            attn.select(2, 0)
                .mean_dim(1, false, x.kind())
                .reshape([batch_size, TICKERS_COUNT, temporal_len])
                .mean_dim(1, false, x.kind())
        });
        let debug_metrics = if debug {
            let (temporal_attn_entropy, cross_ticker_embed_norm) = tch::no_grad(|| {
                let attn = self.attn_softmax_fp32(&q_pool, &k_pool);
                let attn = attn.squeeze_dim(2);
                let entropy = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log())
                    .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
                    .mean(Kind::Float);
                let embed_norm = cross_ticker_embed_base
                    .pow_tensor_scalar(2)
                    .mean(Kind::Float)
                    .sqrt();
                (
                    f64::try_from(&entropy).unwrap_or(0.0),
                    f64::try_from(&embed_norm).unwrap_or(0.0),
                )
            });
            Some(DebugMetrics {
                time_alpha_attn_mean: time_alpha_attn_sum / TIME_CROSS_LAYERS as f64,
                time_alpha_mlp_mean: time_alpha_mlp_sum / TIME_CROSS_LAYERS as f64,
                cross_alpha_attn_mean: cross_alpha_attn_sum / TIME_CROSS_LAYERS as f64,
                cross_alpha_mlp_mean: cross_alpha_mlp_sum / TIME_CROSS_LAYERS as f64,
                temporal_tau,
                temporal_attn_entropy,
                cross_ticker_embed_norm,
            })
        } else {
            None
        };

        (
            self.head_common(
                &conv_features,
                global_static,
                per_ticker_static,
                batch_size,
                attn_avg,
            ),
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
        let action_mean = ticker_logits * &logit_scale;
        let cash_logit = pool_summary.apply(&self.cash_out) * logit_scale;
        const LOG_STD_MIN: f64 = -5.0;
        const LOG_STD_MAX: f64 = 0.0;
        let sde_latent = actor_feat
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.sde_fc);
        let sde_latent =
            self.ln_sde
                .forward(&sde_latent)
                .tanh()
                .reshape([batch_size, TICKERS_COUNT, -1]);
        let latent_norm = sde_latent
            .pow_tensor_scalar(2)
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .clamp_min(1e-6);
        let log_std_raw = self.log_std_param.tanh();
        let log_std_base = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_raw + 1.0);
        let log_std: Tensor = log_std_base
            .unsqueeze(0)
            .expand(&[batch_size, TICKERS_COUNT], false)
            + 0.5 * latent_norm.log();
        let action_log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX);

        (
            values,
            (action_mean, action_log_std, sde_latent, cash_logit),
            attn_out_vis,
        )
    }
}
