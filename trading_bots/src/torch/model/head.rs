use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, FF_DIM, RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS};
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
        let alpha_scale = RESIDUAL_ALPHA_MAX / (TIME_CROSS_LAYERS as f64).sqrt();
        let block = &self.time_cross_block;
        let alpha_attn = block.alpha_attn.sigmoid() * alpha_scale;
        let alpha_mlp = block.alpha_mlp.sigmoid() * alpha_scale;
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
        let x_time = x_2d.permute([0, 2, 1, 3]);
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
        let combined = Tensor::cat(&[x_time.shallow_clone(), per_ticker_static_expanded], 3)
            .reshape([
                batch_size * TICKERS_COUNT * temporal_len,
                256 + PER_TICKER_STATIC_OBS as i64,
            ])
            .apply(&self.static_proj);
        let combined = self
            .ln_static_proj
            .forward(&combined)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, temporal_len, 256]);
        let global_ctx = global_static
            .apply(&self.global_to_ticker)
            .unsqueeze(1)
            .unsqueeze(1);
        let enriched = combined + global_ctx;

        let token_seq = enriched.reshape([batch_size * TICKERS_COUNT, temporal_len, 256]);
        let ticker_ctx = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.ticker_ctx_proj)
            .reshape([batch_size * TICKERS_COUNT, 1, 256]);
        let global_ctx = global_static
            .apply(&self.global_to_ticker)
            .unsqueeze(1)
            .expand(&[batch_size, TICKERS_COUNT, 256], false)
            .reshape([batch_size * TICKERS_COUNT, 1, 256]);
        let positions = self.decay_positions.narrow(0, 0, temporal_len);
        let distances = (temporal_len - 1) as f64 - &positions;
        let ones = self.decay_ones.narrow(0, 0, temporal_len);
        let ticker_slope =
            self.ticker_recent_slope_raw.sigmoid() * (2.0 / temporal_len as f64);
        let ticker_decay = (&ones - &distances * &ticker_slope).clamp_min(0.0);
        let ticker_decay = &ticker_decay / ticker_decay.sum(Kind::Float).clamp_min(1e-6);
        let ticker_decay = ticker_decay
            .reshape([1, temporal_len, 1])
            .to_kind(token_seq.kind());
        let recent_token = (&token_seq * &ticker_decay)
            .sum_dim_intlist([1].as_slice(), false, token_seq.kind());
        let recent_proj = recent_token.apply(&self.ticker_recent_proj);
        let recent_gate = recent_token.apply(&self.ticker_recent_gate).sigmoid();
        let ticker_queries = self
            .ticker_queries
            .unsqueeze(0)
            .expand(
                &[
                    batch_size * TICKERS_COUNT,
                    self.ticker_pool_queries,
                    256,
                ],
                false,
            );
        let ticker_q = (ticker_queries
            + ticker_ctx
            + global_ctx
            + (recent_proj * recent_gate).unsqueeze(1))
        .apply(&self.ticker_q_proj);
        let ticker_k = token_seq.apply(&self.ticker_k_proj);
        let ticker_v = token_seq.apply(&self.ticker_v_proj);
        let ticker_temp = self.ticker_attn_temp_raw.exp().clamp(POOL_TEMPERATURE_MIN, POOL_TEMPERATURE_MAX);
        let ticker_scores =
            ticker_q.matmul(&ticker_k.transpose(-2, -1)) / (256f64).sqrt() * ticker_temp;
        let ticker_attn_f = ticker_scores.softmax(-1, Kind::Float);
        let ticker_ctx_out = ticker_attn_f.matmul(&ticker_v);
        let pooled_enriched = ticker_ctx_out
            .mean_dim(1, false, Kind::Float)
            .apply(&self.ticker_merge)
            .reshape([batch_size, TICKERS_COUNT, 256])
            .to_kind(enriched.kind());

        let head_base = pooled_enriched
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.head_proj);
        let head_base = self
            .head_ln
            .forward(&head_base)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, super::HEAD_HIDDEN]);

        let token_weights_f = ticker_attn_f
            .reshape([
                batch_size,
                TICKERS_COUNT,
                self.ticker_pool_queries,
                temporal_len,
            ])
            .mean_dim([2].as_slice(), false, Kind::Float);
        let attn_avg = token_weights_f
            .mean_dim([0, 1].as_slice(), false, Kind::Float)
            .to_kind(x.kind())
            .reshape([temporal_len]);

        let token_seq =
            enriched.reshape([batch_size, TICKERS_COUNT * temporal_len, 256]);
        let cash_slope =
            self.cash_recent_slope_raw.sigmoid() * (2.0 / temporal_len as f64);
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
            .expand(&[batch_size, self.cash_pool_queries, 256], false)
            + (cash_recent_proj * cash_recent_gate).unsqueeze(1);
        let cash_q = cash_q
            .apply(&self.cash_q_proj);
        let cash_k = token_seq.apply(&self.cash_k_proj);
        let cash_v = token_seq.apply(&self.cash_v_proj);
        let cash_temp = self.cash_attn_temp_raw.exp().clamp(POOL_TEMPERATURE_MIN, POOL_TEMPERATURE_MAX);
        let scores = cash_q.matmul(&cash_k.transpose(-2, -1)) / (256f64).sqrt() * cash_temp;
        let cash_attn_f = scores.softmax(-1, Kind::Float);
        let cash_ctx = cash_attn_f.matmul(&cash_v);
        let cash_summary = cash_ctx
            .mean_dim(1, false, Kind::Float)
            .apply(&self.cash_merge)
            .to_kind(pooled_enriched.kind());
        let cash_head = cash_summary
            .apply(&self.head_proj)
            .to_kind(pooled_enriched.kind());
        let cash_head = self.head_ln.forward(&cash_head).silu();

        let entropy_denom = (temporal_len as f64).ln().max(1.0);
        let ticker_attn_entropy = -(ticker_attn_f.clamp_min(1e-9)
            * ticker_attn_f.clamp_min(1e-9).log())
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .reshape([
                batch_size,
                TICKERS_COUNT,
                self.ticker_pool_queries,
            ])
            .mean_dim([1, 2].as_slice(), false, Kind::Float)
            / entropy_denom;
        let cash_attn_entropy = -(cash_attn_f.clamp_min(1e-9)
            * cash_attn_f.clamp_min(1e-9).log())
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .mean_dim([1].as_slice(), false, Kind::Float)
            / entropy_denom;
        let attn_entropy =
            (ticker_attn_entropy + cash_attn_entropy).to_kind(pooled_enriched.kind());

        let values_ticker = head_base
            .reshape([batch_size * TICKERS_COUNT, super::HEAD_HIDDEN])
            .apply(&self.value_ticker_out)
            .reshape([batch_size, TICKERS_COUNT]);
        let value_cash = cash_head
            .apply(&self.cash_value_out)
            .reshape([batch_size, 1]);
        let values = Tensor::cat(&[values_ticker, value_cash], 1);

        let ticker_logits = head_base.apply(&self.actor_out).squeeze_dim(-1);
        let logit_scale = self.logit_scale_raw.exp();
        let cash_logit = cash_head.apply(&self.cash_out).squeeze_dim(-1);
        let action_logits = Tensor::cat(&[ticker_logits, cash_logit.unsqueeze(1)], 1) * &logit_scale;
        let sde_input = head_base.reshape([batch_size * TICKERS_COUNT, super::HEAD_HIDDEN]);
        let sde_input = if super::SDE_LEARN_FEATURES {
            sde_input
        } else {
            sde_input.detach()
        };
        let sde_latent = sde_input.apply(&self.sde_fc);
        let sde_scale_input = if super::SDE_LEARN_FEATURES {
            head_base.shallow_clone()
        } else {
            head_base.detach()
        };
        let sde_scale_ticker = sde_scale_input
            .reshape([batch_size * TICKERS_COUNT, super::HEAD_HIDDEN])
            .apply(&self.sde_scale_ticker)
            .sigmoid()
            .reshape([batch_size, TICKERS_COUNT, 1]);
        let sde_latent = sde_latent
            .reshape([batch_size, TICKERS_COUNT, -1])
            .g_mul(&sde_scale_ticker);
        let cash_input = if super::SDE_LEARN_FEATURES {
            cash_head.shallow_clone()
        } else {
            cash_head.detach()
        };
        let cash_sde = cash_input.apply(&self.cash_sde).reshape([batch_size, 1, -1]);
        let sde_scale_cash_input = if super::SDE_LEARN_FEATURES {
            cash_head.shallow_clone()
        } else {
            cash_head.detach()
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

        let debug_metrics = None;

        (
            (
                values,
                (action_logits, action_log_std, sde_latent),
                attn_avg,
                attn_entropy,
            ),
            debug_metrics,
        )
    }
}
