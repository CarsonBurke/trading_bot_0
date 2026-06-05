use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::model::blocks::cross_attn::{CA_HEAD_DIM, CA_NUM_HEADS};
use crate::torch::model::blocks::gqa::QK_GAIN_INIT;
use crate::torch::model::init::{linear_truncated, linear_with_same_dtype};
use crate::torch::model::rmsnorm::RMSNorm;

/// Shared bidirectional cross-attention readout. Actor/critic seed tokens form the
/// query set; keys/values are the causal patch hidden states with both seed tokens
/// appended, so each summary attends over every patch plus both tokens (bidirectional
/// actor<->critic communication). Nonzero (orthogonal) `out_proj` makes the attended
/// patch summary O(1) and state-dependent from step 1, so the residual
/// `query_tokens + out` carries price information into the actor/critic heads at init.
pub(in crate::torch::model) struct ActorCriticReadout {
    ln_kv: RMSNorm,
    ln_q: RMSNorm,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    q_gain: Tensor,
    attn_scale: Tensor,
}

impl ActorCriticReadout {
    pub(in crate::torch::model) fn new(p: &nn::Path, model_dim: i64) -> Self {
        let ca_dim = CA_NUM_HEADS * CA_HEAD_DIM;
        let ln_kv = RMSNorm::new(&(p / "ln_kv"), model_dim, 1e-6);
        let ln_q = RMSNorm::new(&(p / "ln_q"), model_dim, 1e-6);
        let q_proj = linear_truncated(p, "q_proj", model_dim, ca_dim);
        let k_proj = linear_truncated(p, "k_proj", model_dim, ca_dim);
        let v_proj = linear_truncated(p, "v_proj", model_dim, ca_dim);
        let out_proj = linear_truncated(p, "out_proj", ca_dim, model_dim);
        let q_norm = RMSNorm::new(&(p / "q_norm"), CA_HEAD_DIM, 1e-6);
        let k_norm = RMSNorm::new(&(p / "k_norm"), CA_HEAD_DIM, 1e-6);
        let q_gain = p.var("q_gain", &[CA_NUM_HEADS], Init::Const(QK_GAIN_INIT));
        let attn_scale = p.var("attn_scale", &[model_dim], Init::Const(1.0));
        Self {
            ln_kv,
            ln_q,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_norm,
            k_norm,
            q_gain,
            attn_scale,
        }
    }

    pub(in crate::torch::model) fn forward(
        &self,
        query_tokens: &Tensor,
        kv_input: &Tensor,
    ) -> Tensor {
        let (b, q_len, _d) = query_tokens.size3().unwrap();
        let kv = self.ln_kv.forward(kv_input);
        let kv_len = kv.size()[1];
        let q_in = self.ln_q.forward(query_tokens);

        let q = linear_with_same_dtype(&q_in, &self.q_proj)
            .reshape([b, q_len, CA_NUM_HEADS, CA_HEAD_DIM])
            .permute([0, 2, 1, 3]);
        let k = linear_with_same_dtype(&kv, &self.k_proj)
            .reshape([b, kv_len, CA_NUM_HEADS, CA_HEAD_DIM])
            .permute([0, 2, 1, 3]);
        let v = linear_with_same_dtype(&kv, &self.v_proj)
            .reshape([b, kv_len, CA_NUM_HEADS, CA_HEAD_DIM])
            .permute([0, 2, 1, 3]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let q = &q * self.q_gain.to_kind(q.kind()).view([1, CA_NUM_HEADS, 1, 1]);
        let k = if k.kind() == q.kind() {
            k
        } else {
            k.to_kind(q.kind())
        };
        let v = if v.kind() == q.kind() {
            v
        } else {
            v.to_kind(q.kind())
        };
        let attn = Tensor::scaled_dot_product_attention(
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
        .reshape([b, q_len, CA_NUM_HEADS * CA_HEAD_DIM]);
        let out = linear_with_same_dtype(&attn, &self.out_proj);
        let out = &out * self.attn_scale.to_kind(out.kind()).view([1, 1, -1]);
        query_tokens + out
    }
}
