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

        // Actor: per-ticker scalar score + cash bias -> action_mean
        let ticker_scores = ticker_repr
            .reshape([batch_size * TICKERS_COUNT, MODEL_DIM])
            .apply(&self.actor_score)
            .reshape([batch_size, TICKERS_COUNT]);
        let cash = self.cash_bias.expand(&[batch_size, 1], false);
        let cash = cash.to_kind(ticker_scores.kind());
        let action_mean = Tensor::cat(&[&ticker_scores, &cash], 1);

        // SDE latent: raw features for Lattice h² weighting (no normalization per paper)
        let sde_latent = ticker_repr
            .reshape([batch_size, TICKERS_COUNT * MODEL_DIM]);

        // Critic: direct projection (no MLP - backbone already processed)
        let critic_logits = ticker_repr
            .reshape([batch_size, TICKERS_COUNT * MODEL_DIM])
            .apply(&self.value_out);

        let values = if compute_values {
            let critic_probs = critic_logits.softmax(-1, Kind::Float);
            let bucket_centers = self.bucket_centers.to_kind(critic_probs.kind());
            let n = bucket_centers.size()[0];
            let m = (n - 1) / 2;
            let p_neg = critic_probs.narrow(-1, 0, m);
            let p_mid = critic_probs.narrow(-1, m, 1);
            let p_pos = critic_probs.narrow(-1, m + 1, m);
            let b_neg = bucket_centers.narrow(0, 0, m);
            let b_mid = bucket_centers.narrow(0, m, 1);
            let b_pos = bucket_centers.narrow(0, m + 1, m);
            let paired = (&p_neg * &b_neg).flip([-1]) + &p_pos * &b_pos;
            let wavg = paired.sum_dim_intlist(-1, false, Kind::Float)
                + (&p_mid * &b_mid).squeeze_dim(-1);
            wavg.to_kind(ticker_repr.kind())
        } else {
            Tensor::zeros(&[batch_size], (ticker_repr.kind(), ticker_repr.device()))
        };

        let debug_metrics = None;

        (
            (values, critic_logits, (action_mean, sde_latent)),
            debug_metrics,
        )
    }
}
