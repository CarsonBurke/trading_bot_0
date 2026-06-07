use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::model::blocks::cross_attn::CrossAttnFfnBlock;
use crate::torch::model::blocks::exogenous::ExogenousTickerBlock;

/// Set-Transformer Pooling-by-Multihead-Attention readout. Two learned seed
/// queries (index 0 = actor, 1 = critic) attend bidirectionally over all encoded
/// patch embeddings. One PMA block == one MAB in this codebase's pre-norm
/// residual idiom: cross-attention (attn-residual) followed by a squared-ReLU FF
/// (FF-residual), supplied by `CrossAttnFfnBlock`. No RoPE on seeds or keys: the
/// patch embeddings already carry position from the trunk's RoPE self-attention.
pub(in crate::torch::model) struct PmaReadout {
    seeds: Tensor,
    block: CrossAttnFfnBlock,
    model_dim: i64,
}

impl PmaReadout {
    pub(in crate::torch::model) fn new(
        p: &nn::Path,
        model_dim: i64,
        ff_dim: i64,
        init_scale: f64,
    ) -> Self {
        let seeds = p.var(
            "pma_seeds",
            &[2, model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        // Non-zero cross-attn output init so pooled patch context (and thus price
        // history) reaches the readout at initialization, rather than the seeds
        // passing through a zeroed residual unchanged. The FFN out-proj keeps its
        // residual-zero init (identity at start), matching the codebase idiom.
        let _ = init_scale;
        let pma_path = p / "pma";
        let cross_attn = ExogenousTickerBlock::new_with_output_init(
            &(&pma_path / "cross_attn"),
            model_dim,
            false,
        );
        let block = CrossAttnFfnBlock::new_with_cross_attn(&pma_path, model_dim, ff_dim, cross_attn);
        Self {
            seeds,
            block,
            model_dim,
        }
    }

    /// `encoded`: `[B, S, d]` post-trunk patch embeddings. Returns `[B, 2, d]`;
    /// `[:, 0]` is the actor readout, `[:, 1]` the critic readout.
    pub(in crate::torch::model) fn forward(&self, encoded: &Tensor) -> Tensor {
        let b = encoded.size()[0];
        let q = self
            .seeds
            .unsqueeze(0)
            .expand([b, 2, self.model_dim], false)
            .to_kind(encoded.kind());
        self.block.forward(&q, encoded)
    }
}
