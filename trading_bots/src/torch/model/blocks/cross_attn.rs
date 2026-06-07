use tch::{nn, Tensor};

use crate::torch::model::blocks::exogenous::ExogenousTickerBlock;
use crate::torch::model::blocks::ffn::ScaledFfn;

pub(in crate::torch::model) const CA_NUM_HEADS: i64 = 2;
pub(in crate::torch::model) const CA_HEAD_DIM: i64 = 128;

pub(in crate::torch::model) struct CrossAttnFfnBlock {
    cross_attn: ExogenousTickerBlock,
    ffn: ScaledFfn,
}

impl CrossAttnFfnBlock {
    pub(in crate::torch::model) fn new(p: &nn::Path, model_dim: i64, ff_dim: i64) -> Self {
        let cross_attn = ExogenousTickerBlock::new(&(p / "cross_attn"), model_dim);
        Self::new_with_cross_attn(p, model_dim, ff_dim, cross_attn)
    }

    pub(in crate::torch::model) fn new_with_cross_attn(
        p: &nn::Path,
        model_dim: i64,
        ff_dim: i64,
        cross_attn: ExogenousTickerBlock,
    ) -> Self {
        let ffn = ScaledFfn::new(p, model_dim, ff_dim);
        Self { cross_attn, ffn }
    }

    pub(in crate::torch::model) fn forward(&self, queries: &Tensor, source: &Tensor) -> Tensor {
        let (source_k, source_v) = self.project_source(source);
        self.forward_with_projected_source(queries, &source_k, &source_v)
    }

    pub(in crate::torch::model) fn project_source(&self, source: &Tensor) -> (Tensor, Tensor) {
        self.cross_attn.project_source(source)
    }

    pub(in crate::torch::model) fn forward_with_projected_source(
        &self,
        queries: &Tensor,
        source_k: &Tensor,
        source_v: &Tensor,
    ) -> Tensor {
        let x = self
            .cross_attn
            .forward_with_projected_source(queries, source_k, source_v);
        self.ffn.forward(&x)
    }
}
