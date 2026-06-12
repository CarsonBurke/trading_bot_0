use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::model::blocks::ffn::ScaledFfn;
use crate::torch::model::init::{linear_residual_out, linear_truncated, linear_with_same_dtype};
use crate::torch::model::rmsnorm::RMSNorm;
use crate::torch::model::rope::RotaryEmbedding;

pub(in crate::torch::model) const GQA_NUM_Q_HEADS: i64 = 4;
pub(in crate::torch::model) const GQA_NUM_KV_HEADS: i64 = 1;
pub(in crate::torch::model) const QK_GAIN_INIT: f64 = 1.0;
const MDHA_KERNEL_SIZE: usize = 3;

pub(in crate::torch::model) struct GqaBlock {
    attn_ln: RMSNorm,
    attn_qkv: nn::Linear,
    q_dconv: [Tensor; MDHA_KERNEL_SIZE],
    k_dconv: [Tensor; MDHA_KERNEL_SIZE],
    v_dconv: [Tensor; MDHA_KERNEL_SIZE],
    attn_out: nn::Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    q_gain: Tensor,
    attn_scale: Tensor,
    q_dim: i64,
    kv_dim: i64,
    resid_mix: Tensor,
    ffn: ScaledFfn,
}

impl GqaBlock {
    pub(in crate::torch::model) fn new(
        p: &nn::Path,
        model_dim: i64,
        ff_dim: i64,
        _layer_idx: usize,
    ) -> Self {
        let head_dim = model_dim / GQA_NUM_Q_HEADS;
        let kv_dim = GQA_NUM_KV_HEADS * head_dim;
        let qkv_dim = model_dim + 2 * kv_dim;
        let attn_ln = RMSNorm::new(model_dim, 1e-6);
        let attn_qkv = linear_truncated(p, "attn_qkv", model_dim, qkv_dim);
        let q_dconv = mdha_dconv_vars(p, "q_dconv", model_dim);
        let k_dconv = mdha_dconv_vars(p, "k_dconv", kv_dim);
        let v_dconv = mdha_dconv_vars(p, "v_dconv", kv_dim);
        let attn_out = linear_residual_out(p, "attn_out", model_dim, model_dim);
        let q_norm = RMSNorm::new(head_dim, 1e-6);
        let k_norm = RMSNorm::new(head_dim, 1e-6);
        let q_gain = p.var("q_gain", &[GQA_NUM_Q_HEADS], Init::Const(QK_GAIN_INIT));
        let attn_scale = p.var("attn_scale", &[model_dim], Init::Const(1.0));
        let resid_mix = p.var_copy(
            "resid_mix",
            &Tensor::stack(
                &[
                    &Tensor::ones([model_dim], (Kind::Float, p.device())),
                    &Tensor::zeros([model_dim], (Kind::Float, p.device())),
                ],
                0,
            ),
        );
        let ffn = ScaledFfn::new(p, model_dim, ff_dim);
        Self {
            attn_ln,
            attn_qkv,
            q_dconv,
            k_dconv,
            v_dconv,
            attn_out,
            q_norm,
            k_norm,
            q_gain,
            attn_scale,
            q_dim: model_dim,
            kv_dim,
            resid_mix,
            ffn,
        }
    }

    /// Bidirectional self-attention over the full patch sequence: no causal mask,
    /// full all-to-all attention. RoPE positions are supplied explicitly.
    pub(in crate::torch::model) fn forward_bidirectional(
        &self,
        x: &Tensor,
        x0: &Tensor,
        rope: &RotaryEmbedding,
        positions: &Tensor,
    ) -> Tensor {
        let (b, s, _d) = x.size3().unwrap();
        let mix = self.resid_mix.to_kind(x.kind());
        let x = x * mix.get(0).view([1, 1, -1]) + x0 * mix.get(1).view([1, 1, -1]);

        let (q, k, v) = self.project_qkv_with_positions(&x, rope, positions);

        let out = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            true,
        );
        let out = out.permute([0, 2, 1, 3]).contiguous().reshape([b, s, _d]);
        let out = linear_with_same_dtype(&out, &self.attn_out);
        let out = &out * self.attn_scale.to_kind(out.kind()).view([1, 1, -1]);
        self.apply_ffn(&(&x + out))
    }

    fn project_qkv_with_positions(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        positions: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let (b, s, d) = x.size3().unwrap();
        let head_dim = d / GQA_NUM_Q_HEADS;
        let normed = self.attn_ln.forward(x);
        let qkv = linear_with_same_dtype(&normed, &self.attn_qkv);
        let parts = qkv.split_with_sizes(&[self.q_dim, self.kv_dim, self.kv_dim], -1);
        let q = causal_depthwise_projection(&parts[0], &self.q_dconv)
            .reshape([b, s, GQA_NUM_Q_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let k = causal_depthwise_projection(&parts[1], &self.k_dconv)
            .reshape([b, s, GQA_NUM_KV_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let v = causal_depthwise_projection(&parts[2], &self.v_dconv)
            .reshape([b, s, GQA_NUM_KV_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let q = rope.apply_positions(&q, positions);
        let k = rope.apply_positions(&k, positions);
        let q = &q
            * self
                .q_gain
                .to_kind(q.kind())
                .view([1, GQA_NUM_Q_HEADS, 1, 1]);
        (q, k, v)
    }

    fn apply_ffn(&self, x: &Tensor) -> Tensor {
        self.ffn.forward(x)
    }
}

fn mdha_dconv_vars(p: &nn::Path, name: &str, channels: i64) -> [Tensor; MDHA_KERNEL_SIZE] {
    std::array::from_fn(|shift| {
        let init = if shift == 0 {
            0.5
        } else {
            0.5 / MDHA_KERNEL_SIZE as f64
        };
        p.var(&format!("{name}_{shift}"), &[channels], Init::Const(init))
    })
}

fn causal_shift(x: &Tensor, shift: i64) -> Tensor {
    if shift == 0 {
        return x.shallow_clone();
    }
    let (batch, seq, channels) = x.size3().unwrap();
    if shift >= seq {
        return Tensor::zeros([batch, seq, channels], (x.kind(), x.device()));
    }
    let pad = Tensor::zeros([batch, shift, channels], (x.kind(), x.device()));
    let kept = x.narrow(1, 0, seq - shift);
    Tensor::cat(&[pad, kept], 1)
}

fn causal_depthwise_projection(x: &Tensor, weights: &[Tensor; MDHA_KERNEL_SIZE]) -> Tensor {
    let mut out = x * weights[0].to_kind(x.kind()).view([1, 1, -1]);
    for (shift, weight) in weights.iter().enumerate().skip(1) {
        out += causal_shift(x, shift as i64) * weight.to_kind(x.kind()).view([1, 1, -1]);
    }
    out
}
