use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::model::blocks::exogenous::ExogenousTickerBlock;
use crate::torch::model::init::{relu_sq_linear, linear_residual_out, linear_truncated};
use crate::torch::model::rmsnorm::RMSNorm;

pub(in crate::torch::model) const CA_NUM_HEADS: i64 = 2;
pub(in crate::torch::model) const CA_HEAD_DIM: i64 = 128;

pub(in crate::torch::model) struct CrossAttnFfnBlock {
    cross_attn: ExogenousTickerBlock,
    ffn_ln: RMSNorm,
    ffn_fc1: nn::Linear,
    ffn_fc2: nn::Linear,
    mlp_scale: Tensor,
}

impl CrossAttnFfnBlock {
    pub(in crate::torch::model) fn new(
        p: &nn::Path,
        model_dim: i64,
        ff_dim: i64,
        init_scale: f64,
    ) -> Self {
        let cross_attn = ExogenousTickerBlock::new(&(p / "cross_attn"), model_dim, init_scale);
        Self::new_with_cross_attn(p, model_dim, ff_dim, cross_attn)
    }

    pub(in crate::torch::model) fn new_with_cross_attn(
        p: &nn::Path,
        model_dim: i64,
        ff_dim: i64,
        cross_attn: ExogenousTickerBlock,
    ) -> Self {
        let ffn_ln = RMSNorm::new(&(p / "ffn_ln"), model_dim, 1e-6);
        let ffn_fc1 = linear_truncated(p, "ffn_fc1", model_dim, ff_dim);
        let ffn_fc2 = linear_residual_out(p, "ffn_fc2", ff_dim, model_dim);
        let mlp_scale = p.var("mlp_scale", &[model_dim], Init::Const(1.0));
        Self {
            cross_attn,
            ffn_ln,
            ffn_fc1,
            ffn_fc2,
            mlp_scale,
        }
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
        let ffn_out = relu_sq_linear(
            &self.ffn_ln.forward_linear(&x, &self.ffn_fc1),
            &self.ffn_fc2,
        );
        let ffn_out = &ffn_out * self.mlp_scale.to_kind(ffn_out.kind()).view([1, 1, -1]);
        x + ffn_out
    }
}
