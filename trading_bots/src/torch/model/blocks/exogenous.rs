use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::model::blocks::cross_attn::{CA_HEAD_DIM, CA_NUM_HEADS};
use crate::torch::model::blocks::gqa::QK_GAIN_INIT;
use crate::torch::model::init::{
    linear_residual_out, linear_truncated, linear_with_same_dtype, relu_sq_linear,
};
use crate::torch::model::rmsnorm::RMSNorm;

pub(in crate::torch::model) struct ExogenousTickerBlock {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    ln_q: RMSNorm,
    ln_kv: RMSNorm,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    q_gain: Tensor,
    attn_scale: Tensor,
}

impl ExogenousTickerBlock {
    pub(in crate::torch::model) fn new(p: &nn::Path, model_dim: i64) -> Self {
        Self::new_with_output_init(p, model_dim, true)
    }

    pub(in crate::torch::model) fn new_with_output_init(
        p: &nn::Path,
        model_dim: i64,
        residual_zero_out: bool,
    ) -> Self {
        let ca_dim = CA_NUM_HEADS * CA_HEAD_DIM;
        let ln_q = RMSNorm::new(model_dim, 1e-6);
        let ln_kv = RMSNorm::new(model_dim, 1e-6);
        let q_norm = RMSNorm::new(CA_HEAD_DIM, 1e-6);
        let k_norm = RMSNorm::new(CA_HEAD_DIM, 1e-6);
        let q_gain = p.var("ca_q_gain", &[CA_NUM_HEADS], Init::Const(QK_GAIN_INIT));
        let attn_scale = p.var("ca_attn_scale", &[model_dim], Init::Const(1.0));
        let q_proj = linear_truncated(p, "ca_q", model_dim, ca_dim);
        let k_proj = linear_truncated(p, "ca_k", model_dim, ca_dim);
        let v_proj = linear_truncated(p, "ca_v", model_dim, ca_dim);
        let out_proj = if residual_zero_out {
            linear_residual_out(p, "ca_out", ca_dim, model_dim)
        } else {
            linear_truncated(p, "ca_out", ca_dim, model_dim)
        };
        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            ln_q,
            ln_kv,
            q_norm,
            k_norm,
            q_gain,
            attn_scale,
        }
    }

    pub(in crate::torch::model) fn project_source(&self, exo_kv: &Tensor) -> (Tensor, Tensor) {
        let b = exo_kv.size()[0];
        let exo_kv = self.ln_kv.forward(exo_kv);
        let exo_len = exo_kv.size()[1];
        let k = linear_with_same_dtype(&exo_kv, &self.k_proj)
            .reshape([b, exo_len, CA_NUM_HEADS, CA_HEAD_DIM])
            .permute([0, 2, 1, 3]);
        let k = self.k_norm.forward(&k);
        let v = linear_with_same_dtype(&exo_kv, &self.v_proj)
            .reshape([b, exo_len, CA_NUM_HEADS, CA_HEAD_DIM])
            .permute([0, 2, 1, 3]);
        (k, v)
    }

    pub(in crate::torch::model) fn forward_with_projected_source(
        &self,
        x: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Tensor {
        let (b, s, _d) = x.size3().unwrap();
        let q = self
            .ln_q
            .forward_linear(x, &self.q_proj)
            .reshape([b, s, CA_NUM_HEADS, CA_HEAD_DIM])
            .permute([0, 2, 1, 3]);
        let q = self.q_norm.forward(&q);
        let q = &q * self.q_gain.to_kind(q.kind()).view([1, CA_NUM_HEADS, 1, 1]);
        let k = if k.kind() == q.kind() {
            k.shallow_clone()
        } else {
            k.to_kind(q.kind())
        };
        let v = if v.kind() == q.kind() {
            v.shallow_clone()
        } else {
            v.to_kind(q.kind())
        };
        let out = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )
        .permute([0, 2, 1, 3])
        .contiguous()
        .reshape([b, s, CA_NUM_HEADS * CA_HEAD_DIM]);
        let out = linear_with_same_dtype(&out, &self.out_proj);
        let out = &out * self.attn_scale.to_kind(out.kind()).view([1, 1, -1]);
        x + out
    }
}

pub(in crate::torch::model) struct ExoMLP {
    in_ln: RMSNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ExoMLP {
    pub(in crate::torch::model) fn new(p: &nn::Path, model_dim: i64) -> Self {
        let in_ln = RMSNorm::new(model_dim, 1e-6);
        let fc1 = linear_truncated(p, "exo_mlp_fc1", model_dim, model_dim);
        let fc2 = linear_residual_out(p, "exo_mlp_fc2", model_dim, model_dim);
        Self { in_ln, fc1, fc2 }
    }

    pub(in crate::torch::model) fn forward(&self, x: &Tensor) -> Tensor {
        let h = relu_sq_linear(&self.in_ln.forward_linear(x, &self.fc1), &self.fc2);
        x + h
    }
}
