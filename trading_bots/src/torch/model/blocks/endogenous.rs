use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::model::blocks::gqa::QK_GAIN_INIT;
use crate::torch::model::init::{
    relu_sq_linear, linear_residual_out, linear_truncated, linear_with_same_dtype,
};
use crate::torch::model::rmsnorm::RMSNorm;

pub(in crate::torch::model) struct EndogenousTickerBlock {
    ticker_ln: RMSNorm,
    ticker_qkv: nn::Linear,
    ticker_out: nn::Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    q_gain: Tensor,
    attn_scale: Tensor,
    mlp_scale: Tensor,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    mlp_ln: RMSNorm,
}

impl EndogenousTickerBlock {
    pub(in crate::torch::model) fn new(p: &nn::Path, model_dim: i64, ff_dim: i64) -> Self {
        let ticker_ln = RMSNorm::new(model_dim, 1e-6);
        let ticker_qkv = linear_truncated(p, "ticker_qkv", model_dim, 3 * model_dim);
        let ticker_out = linear_residual_out(p, "ticker_out", model_dim, model_dim);
        let q_norm = RMSNorm::new(model_dim, 1e-6);
        let k_norm = RMSNorm::new(model_dim, 1e-6);
        let q_gain = p.var("q_gain", &[1], Init::Const(QK_GAIN_INIT));
        let attn_scale = p.var("attn_scale", &[model_dim], Init::Const(1.0));
        let mlp_scale = p.var("mlp_scale", &[model_dim], Init::Const(1.0));
        let mlp_fc1 = linear_truncated(p, "mlp_fc1", model_dim, ff_dim);
        let mlp_fc2 = linear_residual_out(p, "mlp_fc2", ff_dim, model_dim);
        let mlp_ln = RMSNorm::new(model_dim, 1e-6);
        Self {
            ticker_ln,
            ticker_qkv,
            ticker_out,
            q_norm,
            k_norm,
            q_gain,
            attn_scale,
            mlp_scale,
            mlp_fc1,
            mlp_fc2,
            mlp_ln,
        }
    }

    pub(in crate::torch::model) fn forward(
        &self,
        x: &Tensor,
        model_dim: i64,
        _ff_dim: i64,
    ) -> Tensor {
        let (batch, num_items, _) = x.size3().unwrap();
        let qkv = self.ticker_ln.forward_linear(x, &self.ticker_qkv);
        let parts = qkv.split(model_dim, -1);
        // QKNorm (single-head, head_dim == model_dim)
        let q = self.q_norm.forward(&parts[0]).unsqueeze(1);
        let q = &q * self.q_gain.to_kind(q.kind()).view([1, 1, 1, 1]);
        let k = self.k_norm.forward(&parts[1]).unsqueeze(1);
        let v = parts[2].unsqueeze(1);
        let ctx = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )
        .squeeze_dim(1)
        .reshape([batch * num_items, model_dim]);
        let ctx =
            linear_with_same_dtype(&ctx, &self.ticker_out).reshape([batch, num_items, model_dim]);
        let ctx = &ctx * self.attn_scale.to_kind(ctx.kind()).view([1, 1, -1]);
        let x = x + ctx;
        let mlp = relu_sq_linear(
            &self.mlp_ln.forward_linear(&x, &self.mlp_fc1),
            &self.mlp_fc2,
        );
        let mlp = &mlp * self.mlp_scale.to_kind(mlp.kind()).view([1, 1, -1]);
        x + mlp
    }
}
