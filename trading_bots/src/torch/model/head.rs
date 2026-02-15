use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, TradingModel, NUM_VALUE_BUCKETS, RESIDUAL_ALPHA_MAX, TIME_CROSS_LAYERS, SDE_DIM};
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

        // Extract CLS token (position 0) as per-ticker representation.
        let mut ticker_repr = x_time.select(2, 0);

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

        // Actor head: per-ticker MLP → per-ticker logits + cash logit
        let mut actor_x = ticker_repr.reshape([batch_size * TICKERS_COUNT, self.model_dim]);
        for i in 0..self.actor_mlp_linears.len() {
            actor_x = actor_x.apply(&self.actor_mlp_linears[i]);
            actor_x = self.actor_mlp_norms[i].forward(&actor_x);
            actor_x = actor_x.silu();
        }

        // Per-ticker logits (shared actor_out: model_dim -> 1)
        let ticker_logits = actor_x
            .apply(&self.actor_out)
            .reshape([batch_size, TICKERS_COUNT]);

        // Cash logit from mean-pooled ticker representations
        let cash_logit = actor_x
            .reshape([batch_size, TICKERS_COUNT, self.model_dim])
            .mean_dim([1].as_slice(), false, actor_x.kind())
            .apply(&self.cash_proj)
            .squeeze_dim(-1);

        // action_mean: [B, ACTION_DIM]
        let action_mean = Tensor::cat(&[ticker_logits, cash_logit.unsqueeze(-1)], -1);

        // gSDE: FC → RMSNorm → SiLU → quadratic form for per-ticker variance
        let sde_latent = self.sde_norm.forward(
            &actor_x.apply(&self.sde_fc)
        ).silu().reshape([batch_size, TICKERS_COUNT, SDE_DIM]);

        // Per-ticker variance: Σ_j latent[b,i,j]² · exp(2·log_std[j,i])
        let ticker_std_sq = (&self.sde_log_std_param * 2.0).exp(); // [SDE_DIM, TICKERS_COUNT]
        let ticker_var = (sde_latent.pow_tensor_scalar(2) * ticker_std_sq.transpose(0, 1).unsqueeze(0))
            .sum_dim_intlist([-1].as_slice(), false, sde_latent.kind()); // [B, TICKERS_COUNT]
        let ticker_log_std = (ticker_var + 1e-8).log() * 0.5;

        // Cash is deterministic (redundant DOF on simplex — noise is pure gradient variance).
        // action_log_std is [B, TICKERS_COUNT] only; cash excluded from log_prob.
        let action_log_std = ticker_log_std;

        // Critic: distributional two-hot value head.
        let critic_input = ticker_repr.reshape([batch_size, TICKERS_COUNT * self.model_dim]);
        let mut critic_x = critic_input.shallow_clone();
        for i in 0..self.value_mlp_linears.len() {
            critic_x = critic_x.apply(&self.value_mlp_linears[i]);
            critic_x = self.value_mlp_norms[i].forward(&critic_x);
            critic_x = critic_x.silu();
        }
        let critic_logits = critic_x.apply(&self.value_out); // [batch, NUM_VALUE_BUCKETS]

        let values = if compute_values {
            let probs = critic_logits.softmax(-1, Kind::Float);
            let centers = self.bucket_centers.to_kind(probs.kind());
            let m: i64 = (NUM_VALUE_BUCKETS - 1) / 2; // 127
            let p1 = probs.narrow(-1, 0, m);
            let p2 = probs.narrow(-1, m, 1);
            let p3 = probs.narrow(-1, m + 1, m);
            let b1 = centers.narrow(0, 0, m);
            let b2 = centers.narrow(0, m, 1);
            let b3 = centers.narrow(0, m + 1, m);
            let left = (&p1 * &b1).flip([-1]);
            let right = &p3 * &b3;
            ((&p2 * &b2).sum_dim_intlist(-1, false, Kind::Float)
                + (&left + &right).sum_dim_intlist(-1, false, Kind::Float))
                .to_kind(ticker_repr.kind())
        } else {
            Tensor::zeros(&[batch_size], (ticker_repr.kind(), ticker_repr.device()))
        };

        // Cast all outputs to fp32 for loss computation
        let values = values.to_kind(Kind::Float);
        let critic_logits = critic_logits.to_kind(Kind::Float);
        let critic_input = critic_input.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_log_std = action_log_std.to_kind(Kind::Float);

        let debug_metrics = None;

        (
            (
                values,
                critic_logits,
                critic_input,
                action_mean,
                action_log_std,
            ),
            debug_metrics,
        )
    }
}
