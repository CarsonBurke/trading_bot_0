mod forward;
mod head;
mod inference;
mod rmsnorm;

use clap::ValueEnum;
use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER,
    STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::hl_gauss::NUM_BINS;

use rmsnorm::RMSNorm;

struct EndogenousTickerBlock {
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
    fn new(p: &nn::Path, model_dim: i64, ff_dim: i64, _init_scale: f64) -> Self {
        let ticker_ln = RMSNorm::new(&(p / "ticker_ln"), model_dim, 1e-6);
        let ticker_qkv = linear_truncated(p, "ticker_qkv", model_dim, 3 * model_dim);
        let ticker_out = linear_residual_out(p, "ticker_out", model_dim, model_dim);
        let q_norm = RMSNorm::new(&(p / "q_norm"), model_dim, 1e-6);
        let k_norm = RMSNorm::new(&(p / "k_norm"), model_dim, 1e-6);
        let q_gain = p.var("q_gain", &[1], Init::Const(QK_GAIN_INIT));
        let attn_scale = p.var("attn_scale", &[model_dim], Init::Const(1.0));
        let mlp_scale = p.var("mlp_scale", &[model_dim], Init::Const(1.0));
        let mlp_fc1 = linear_truncated(p, "mlp_fc1", model_dim, ff_dim);
        let mlp_fc2 = linear_residual_out(p, "mlp_fc2", ff_dim, model_dim);
        let mlp_ln = RMSNorm::new(&(p / "mlp_ln"), model_dim, 1e-6);
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

    fn forward(&self, x: &Tensor, model_dim: i64, _ff_dim: i64) -> Tensor {
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
        let mlp = leaky_relu_sq_linear(
            &self.mlp_ln.forward_linear(&x, &self.mlp_fc1),
            &self.mlp_fc2,
        );
        let mlp = &mlp * self.mlp_scale.to_kind(mlp.kind()).view([1, 1, -1]);
        x + mlp
    }
}

const GQA_NUM_Q_HEADS: i64 = 4;
const GQA_NUM_KV_HEADS: i64 = 4;
const CA_NUM_HEADS: i64 = 2;
const CA_HEAD_DIM: i64 = 128;
const QK_GAIN_INIT: f64 = 1.5;
const ROPE_DIMS: i64 = 16;

fn rotate_half(x: &Tensor) -> Tensor {
    let last_dim = *x.size().last().unwrap();
    let half = last_dim / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[&(-&x2), &x1], -1)
}

struct RotaryEmbedding {
    cos_cached: Tensor, // [max_seq_len, rope_dims]
    sin_cached: Tensor, // [max_seq_len, rope_dims]
    rope_dims: i64,
}

impl RotaryEmbedding {
    fn new(max_seq_len: i64, head_dim: i64, rope_dims: i64, device: tch::Device) -> Self {
        let rd = rope_dims.min(head_dim);
        let half_rd = rd / 2;
        let exponents = Tensor::arange(half_rd, (Kind::Float, device)) * (2.0 / rd as f64);
        let inv_freq = (exponents * -(10000.0_f64.ln())).exp();
        let positions = Tensor::arange(max_seq_len, (Kind::Float, device));
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0);
        let cos_half = angles.cos();
        let sin_half = angles.sin();
        Self {
            cos_cached: Tensor::cat(&[&cos_half, &cos_half], -1).set_requires_grad(false),
            sin_cached: Tensor::cat(&[&sin_half, &sin_half], -1).set_requires_grad(false),
            rope_dims: rd,
        }
    }

    fn apply_from(&self, x: &Tensor, offset: i64) -> Tensor {
        // x: [batch, heads, seq_len, head_dim]
        let seq_len = x.size()[2];
        let head_dim = *x.size().last().unwrap();
        let cos = self.cos_cached.narrow(0, offset, seq_len).to_kind(x.kind());
        let sin = self.sin_cached.narrow(0, offset, seq_len).to_kind(x.kind());
        if self.rope_dims < head_dim {
            let x_rope = x.narrow(-1, 0, self.rope_dims);
            let x_pass = x.narrow(-1, self.rope_dims, head_dim - self.rope_dims);
            let rotated = &x_rope * &cos + rotate_half(&x_rope) * &sin;
            Tensor::cat(&[&rotated, &x_pass], -1)
        } else {
            x * &cos + rotate_half(x) * &sin
        }
    }
}

struct GqaBlock {
    attn_ln: RMSNorm,
    attn_qkv: nn::Linear,
    attn_out: nn::Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    q_gain: Tensor,
    attn_scale: Tensor,
    mlp_scale: Tensor,
    q_dim: i64,
    kv_dim: i64,
    resid_mix: Tensor,
    ln_scale_factor: f64,
    ffn_ln: RMSNorm,
    ffn_fc1: nn::Linear,
    ffn_fc2: nn::Linear,
}

impl GqaBlock {
    fn new(p: &nn::Path, model_dim: i64, ff_dim: i64, _init_scale: f64, layer_idx: usize) -> Self {
        let head_dim = model_dim / GQA_NUM_Q_HEADS;
        let kv_dim = GQA_NUM_KV_HEADS * head_dim;
        let qkv_dim = model_dim + 2 * kv_dim;
        let attn_ln = RMSNorm::new(&(p / "attn_ln"), model_dim, 1e-6);
        let attn_qkv = linear_truncated(p, "attn_qkv", model_dim, qkv_dim);
        let attn_out = linear_residual_out(p, "attn_out", model_dim, model_dim);
        let q_norm = RMSNorm::new(&(p / "q_norm"), head_dim, 1e-6);
        let k_norm = RMSNorm::new(&(p / "k_norm"), head_dim, 1e-6);
        let q_gain = p.var("q_gain", &[GQA_NUM_Q_HEADS], Init::Const(QK_GAIN_INIT));
        let attn_scale = p.var("attn_scale", &[model_dim], Init::Const(1.0));
        let mlp_scale = p.var("mlp_scale", &[model_dim], Init::Const(1.0));
        let ln_scale_factor = 1.0 / ((layer_idx + 1) as f64).sqrt();
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
        let ffn_ln = RMSNorm::new(&(p / "ffn_ln"), model_dim, 1e-6);
        let ffn_fc1 = linear_truncated(p, "ffn_fc1", model_dim, ff_dim);
        let ffn_fc2 = linear_residual_out(p, "ffn_fc2", ff_dim, model_dim);
        Self {
            attn_ln,
            attn_qkv,
            attn_out,
            q_norm,
            k_norm,
            q_gain,
            attn_scale,
            mlp_scale,
            q_dim: model_dim,
            kv_dim,
            resid_mix,
            ln_scale_factor,
            ffn_ln,
            ffn_fc1,
            ffn_fc2,
        }
    }

    fn forward(&self, x: &Tensor, x0: &Tensor, rope: &RotaryEmbedding, causal: bool) -> Tensor {
        let (b, s, _d) = x.size3().unwrap();
        let mix = self.resid_mix.to_kind(x.kind());
        let x = x * mix.get(0).view([1, 1, -1]) + x0 * mix.get(1).view([1, 1, -1]);

        let (q, k, v) = self.project_qkv(&x, rope, 0);

        let out = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            causal,
            None,
            true,
        );
        let out = out.permute([0, 2, 1, 3]).contiguous().reshape([b, s, _d]);
        let out = linear_with_same_dtype(&out, &self.attn_out);
        let out = &out * self.attn_scale.to_kind(out.kind()).view([1, 1, -1]);
        self.apply_ffn(&(&x + out))
    }

    fn project_qkv(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        rope_offset: i64,
    ) -> (Tensor, Tensor, Tensor) {
        let (b, s, d) = x.size3().unwrap();
        let head_dim = d / GQA_NUM_Q_HEADS;
        let normed = self.attn_ln.forward(x) * self.ln_scale_factor;
        let qkv = linear_with_same_dtype(&normed, &self.attn_qkv);
        let parts = qkv.split_with_sizes(&[self.q_dim, self.kv_dim, self.kv_dim], -1);
        let q = parts[0]
            .reshape([b, s, GQA_NUM_Q_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let k = parts[1]
            .reshape([b, s, GQA_NUM_KV_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let v = parts[2]
            .reshape([b, s, GQA_NUM_KV_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let q = rope.apply_from(&q, rope_offset);
        let k = rope.apply_from(&k, rope_offset);
        let q = &q
            * self
                .q_gain
                .to_kind(q.kind())
                .view([1, GQA_NUM_Q_HEADS, 1, 1]);
        (q, k, v)
    }

    fn apply_ffn(&self, x: &Tensor) -> Tensor {
        let normed = self.ffn_ln.forward(x) * self.ln_scale_factor;
        let ffn_out = leaky_relu_sq_linear(
            &linear_with_same_dtype(&normed, &self.ffn_fc1),
            &self.ffn_fc2,
        );
        let ffn_out = &ffn_out * self.mlp_scale.to_kind(ffn_out.kind()).view([1, 1, -1]);
        x + ffn_out
    }

    fn forward_prefix_and_cache(
        &self,
        x: &Tensor,
        x0: &Tensor,
        rope: &RotaryEmbedding,
    ) -> (Tensor, Tensor, Tensor) {
        let mix = self.resid_mix.to_kind(x.kind());
        let x = x * mix.get(0).view([1, 1, -1]) + x0 * mix.get(1).view([1, 1, -1]);
        let (q, k, v) = self.project_qkv(&x, rope, 0);
        let out = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            true,
            None,
            true,
        );
        let out = out.permute([0, 2, 1, 3]).contiguous().reshape(x.size());
        let attn_out = linear_with_same_dtype(&out, &self.attn_out);
        let attn_out = &attn_out * self.attn_scale.to_kind(attn_out.kind()).view([1, 1, -1]);
        let x = &x + attn_out;
        (self.apply_ffn(&x), k, v)
    }

    fn forward_suffix_with_cache(
        &self,
        x_suffix: &Tensor,
        x0_suffix: &Tensor,
        prefix_k: &Tensor,
        prefix_v: &Tensor,
        rope: &RotaryEmbedding,
        prefix_len: i64,
    ) -> Tensor {
        let mix = self.resid_mix.to_kind(x_suffix.kind());
        let x_suffix =
            x_suffix * mix.get(0).view([1, 1, -1]) + x0_suffix * mix.get(1).view([1, 1, -1]);
        let (q, suffix_k, suffix_v) = self.project_qkv(&x_suffix, rope, prefix_len);
        let prefix_k = if prefix_k.kind() == suffix_k.kind() {
            prefix_k.shallow_clone()
        } else {
            prefix_k.to_kind(suffix_k.kind())
        };
        let prefix_v = if prefix_v.kind() == suffix_v.kind() {
            prefix_v.shallow_clone()
        } else {
            prefix_v.to_kind(suffix_v.kind())
        };
        let all_k = Tensor::cat(&[&prefix_k, &suffix_k], -2);
        let all_v = Tensor::cat(&[&prefix_v, &suffix_v], -2);
        let out = Tensor::scaled_dot_product_attention(
            &q,
            &all_k,
            &all_v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            true,
        );
        let out = out
            .permute([0, 2, 1, 3])
            .contiguous()
            .reshape(x_suffix.size());
        let attn_out = linear_with_same_dtype(&out, &self.attn_out);
        let attn_out = &attn_out * self.attn_scale.to_kind(attn_out.kind()).view([1, 1, -1]);
        let x = &x_suffix + attn_out;
        self.apply_ffn(&x)
    }
}

struct ExogenousTickerBlock {
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
    fn new(p: &nn::Path, model_dim: i64, _init_scale: f64) -> Self {
        let ca_dim = CA_NUM_HEADS * CA_HEAD_DIM;
        let ln_q = RMSNorm::new(&(p / "ln_q"), model_dim, 1e-6);
        let ln_kv = RMSNorm::new(&(p / "ln_kv"), model_dim, 1e-6);
        let q_norm = RMSNorm::new(&(p / "ca_q_norm"), CA_HEAD_DIM, 1e-6);
        let k_norm = RMSNorm::new(&(p / "ca_k_norm"), CA_HEAD_DIM, 1e-6);
        let q_gain = p.var("ca_q_gain", &[CA_NUM_HEADS], Init::Const(QK_GAIN_INIT));
        let attn_scale = p.var("ca_attn_scale", &[model_dim], Init::Const(1.0));
        let q_proj = linear_truncated(p, "ca_q", model_dim, ca_dim);
        let k_proj = linear_truncated(p, "ca_k", model_dim, ca_dim);
        let v_proj = linear_truncated(p, "ca_v", model_dim, ca_dim);
        let out_proj = linear_residual_out(p, "ca_out", ca_dim, model_dim);
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

    fn project_source(&self, exo_kv: &Tensor) -> (Tensor, Tensor) {
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

    fn forward_with_projected_source(&self, x: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
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

struct ExoMLP {
    in_ln: RMSNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ExoMLP {
    fn new(p: &nn::Path, model_dim: i64, _init_scale: f64) -> Self {
        let in_ln = RMSNorm::new(&(p / "in_ln"), model_dim, 1e-6);
        let fc1 = linear_truncated(p, "exo_mlp_fc1", model_dim, model_dim);
        let fc2 = linear_residual_out(p, "exo_mlp_fc2", model_dim, model_dim);
        Self { in_ln, fc1, fc2 }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.in_ln.forward_linear(x, &self.fc1).silu();
        let h = linear_with_same_dtype(&h, &self.fc2);
        x + h
    }
}

struct CrossAttnFfnBlock {
    cross_attn: ExogenousTickerBlock,
    ffn_ln: RMSNorm,
    ffn_fc1: nn::Linear,
    ffn_fc2: nn::Linear,
    mlp_scale: Tensor,
}

impl CrossAttnFfnBlock {
    fn new(p: &nn::Path, model_dim: i64, ff_dim: i64, init_scale: f64) -> Self {
        let cross_attn = ExogenousTickerBlock::new(&(p / "cross_attn"), model_dim, init_scale);
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

    fn forward(&self, queries: &Tensor, source: &Tensor) -> Tensor {
        let (source_k, source_v) = self.project_source(source);
        self.forward_with_projected_source(queries, &source_k, &source_v)
    }

    fn project_source(&self, source: &Tensor) -> (Tensor, Tensor) {
        self.cross_attn.project_source(source)
    }

    fn forward_with_projected_source(
        &self,
        queries: &Tensor,
        source_k: &Tensor,
        source_v: &Tensor,
    ) -> Tensor {
        let x = self
            .cross_attn
            .forward_with_projected_source(queries, source_k, source_v);
        let ffn_out = leaky_relu_sq_linear(
            &self.ffn_ln.forward_linear(&x, &self.ffn_fc1),
            &self.ffn_fc2,
        );
        let ffn_out = &ffn_out * self.mlp_scale.to_kind(ffn_out.kind()).view([1, 1, -1]);
        x + ffn_out
    }
}

pub(super) fn linear_with_same_dtype(x: &Tensor, linear: &nn::Linear) -> Tensor {
    let weight = if linear.ws.kind() == x.kind() {
        linear.ws.shallow_clone()
    } else {
        linear.ws.to_kind(x.kind())
    };
    let bias = linear.bs.as_ref().map(|b| {
        if b.kind() == x.kind() {
            b.shallow_clone()
        } else {
            b.to_kind(x.kind())
        }
    });
    x.linear(&weight, bias.as_ref())
}

fn leaky_relu_sq_linear(x: &Tensor, out_proj: &nn::Linear) -> Tensor {
    let h = x.maximum(&(x * 0.5)).square();
    linear_with_same_dtype(&h, out_proj)
}

fn xavier_normal_std(in_features: i64, out_features: i64) -> f64 {
    (2.0 / (in_features + out_features) as f64).sqrt()
}

fn truncated_normal_std(in_features: i64, out_features: i64) -> f64 {
    xavier_normal_std(in_features, out_features) / 0.8796
}

fn truncated_normal_init(in_features: i64, out_features: i64) -> Init {
    Init::Randn {
        mean: 0.0,
        stdev: truncated_normal_std(in_features, out_features),
    }
}

fn linear_truncated(p: &nn::Path, name: &str, in_features: i64, out_features: i64) -> nn::Linear {
    nn::linear(
        p / name,
        in_features,
        out_features,
        nn::LinearConfig {
            ws_init: Init::Orthogonal { gain: 1.0 },
            bs_init: None,
            bias: false,
        },
    )
}

fn linear_residual_out(
    p: &nn::Path,
    name: &str,
    in_features: i64,
    out_features: i64,
) -> nn::Linear {
    nn::linear(
        p / name,
        in_features,
        out_features,
        nn::LinearConfig {
            ws_init: Init::Const(0.0),
            bs_init: None,
            bias: false,
        },
    )
}

const BASE_MODEL_DIM: i64 = 256;
const BASE_FF_DIM: i64 = 512;
const BASE_GQA_LAYERS: usize = 3;
const ABLATION_SMALL_MODEL_DIM: i64 = 96;
const ABLATION_SMALL_FF_DIM: i64 = 192;
const ABLATION_SMALL_GQA_LAYERS: usize = 1;
const UNIFORM_STREAM_PATCH_COUNT: i64 = 120;
const UNIFORM_STREAM_PATCH_SIZE: i64 = 50;
const UNIFORM_STREAM_LAYOUT_LEN: i64 = UNIFORM_STREAM_PATCH_COUNT * UNIFORM_STREAM_PATCH_SIZE;
const UNIFORM_STREAM_BOOTSTRAP_FULL_PATCHES: i64 = UNIFORM_STREAM_PATCH_COUNT - 1;
const UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL: i64 = PRICE_DELTAS_PER_TICKER as i64
    - UNIFORM_STREAM_BOOTSTRAP_FULL_PATCHES * UNIFORM_STREAM_PATCH_SIZE;
const INTER_TICKER_AFTER: usize = 1;
const NUM_EXO_TOKENS: i64 = STATIC_OBSERVATIONS as i64;
const PATCH_SCALAR_FEATS: i64 = 3;
fn residual_init_scale(num_residual_sublayers: usize) -> f64 {
    1.0 / (2.0 * num_residual_sublayers as f64).sqrt()
}

const BASE_PATCH_CONFIGS: &[(i64, i64)] = &[
    (3072, 128),
    (1536, 64),
    (768, 32),
    (384, 16),
    (128, 8),
    (64, 4),
    (46, 2),
    (2, 1),
];

const fn uniform_stream_patch_configs() -> [(i64, i64); UNIFORM_STREAM_PATCH_COUNT as usize] {
    let mut configs = [(0i64, 0i64); UNIFORM_STREAM_PATCH_COUNT as usize];
    let mut idx = 0usize;
    while idx < UNIFORM_STREAM_PATCH_COUNT as usize {
        configs[idx] = (UNIFORM_STREAM_PATCH_SIZE, UNIFORM_STREAM_PATCH_SIZE);
        idx += 1;
    }
    configs
}

const UNIFORM_STREAM_PATCH_CONFIGS: &[(i64, i64)] = &uniform_stream_patch_configs();

const ABLATION_SMALL_PATCH_CONFIGS: &[(i64, i64)] = &[(5632, 256), (360, 8), (8, 1)];

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ModelVariant {
    Base,
    #[value(
        name = "uniform-stream",
        alias = "uniform-256-stream",
        alias = "uniform256-stream"
    )]
    UniformStream,
    AblationSmall,
}

impl ModelVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::UniformStream => "uniform-stream",
            Self::AblationSmall => "ablation-small",
        }
    }
}

#[derive(Clone, Copy)]
struct ModelSpec {
    model_dim: i64,
    ff_dim: i64,
    gqa_layers: usize,
    patch_configs: &'static [(i64, i64)],
}

fn model_spec(variant: ModelVariant) -> ModelSpec {
    match variant {
        ModelVariant::Base => ModelSpec {
            model_dim: BASE_MODEL_DIM,
            ff_dim: BASE_FF_DIM,
            gqa_layers: BASE_GQA_LAYERS,
            patch_configs: BASE_PATCH_CONFIGS,
        },
        ModelVariant::UniformStream => ModelSpec {
            model_dim: BASE_MODEL_DIM,
            ff_dim: BASE_FF_DIM,
            gqa_layers: BASE_GQA_LAYERS,
            patch_configs: UNIFORM_STREAM_PATCH_CONFIGS,
        },
        ModelVariant::AblationSmall => ModelSpec {
            model_dim: ABLATION_SMALL_MODEL_DIM,
            ff_dim: ABLATION_SMALL_FF_DIM,
            gqa_layers: ABLATION_SMALL_GQA_LAYERS,
            patch_configs: ABLATION_SMALL_PATCH_CONFIGS,
        },
    }
}

fn compute_patch_totals(patch_configs: &[(i64, i64)]) -> (i64, i64) {
    let mut total_days = 0i64;
    let mut total_tokens = 0i64;
    for &(days, patch_size) in patch_configs {
        assert!(
            days % patch_size == 0,
            "days must be divisible by patch_size"
        );
        total_days += days;
        total_tokens += days / patch_size;
    }
    (total_days, total_tokens)
}

pub fn patch_seq_len_for_variant(variant: ModelVariant) -> i64 {
    compute_patch_totals(model_spec(variant).patch_configs).1
}

pub fn patch_ends_for_variant(variant: ModelVariant) -> Vec<i64> {
    let patch_configs = model_spec(variant).patch_configs;
    let seq_len = patch_seq_len_for_variant(variant);
    let mut ends = Vec::with_capacity(seq_len as usize);
    let mut total = 0i64;
    for &(days, patch_size) in patch_configs {
        let num = days / patch_size;
        for _ in 0..num {
            total += patch_size;
            ends.push(total);
        }
    }
    ends
}

/// (value_logits, action_mean, action_std, action_log_var)
pub type ModelOutput = (Tensor, Tensor, Tensor, Tensor);

pub struct DebugMetrics {
    pub temporal_tau: f64,
    pub temporal_attn_entropy: f64,
    pub temporal_attn_max: f64,
    pub temporal_attn_eff_len: f64,
    pub temporal_attn_center: f64,
    pub temporal_attn_last_weight: f64,
}

#[derive(Clone, Copy)]
pub struct TradingModelConfig {
    pub variant: ModelVariant,
}

impl Default for TradingModelConfig {
    fn default() -> Self {
        Self {
            variant: ModelVariant::Base,
        }
    }
}

/// Streaming state for inference
/// - Ring buffer holds full delta history
/// - Patch buffer accumulates deltas until full patch ready
/// - No model state needed (GQA is stateless, uses full forward pass)
pub struct StreamState {
    /// Ring buffer: [TICKERS_COUNT, PRICE_DELTAS_PER_TICKER]
    pub delta_ring: Tensor,
    /// Write position in ring buffer
    pub ring_pos: i64,
    /// Patch accumulator: [TICKERS_COUNT, FINEST_PATCH_SIZE]
    pub patch_buf: Tensor,
    /// Position within current patch
    pub patch_pos: i64,
    /// Whether initialized with full sequence
    pub initialized: bool,
    /// Uniform stream bucket layout: [batch*TICKERS_COUNT, patch_count, patch_size]
    pub uniform_layout: Tensor,
    /// Cached patch tokens for uniform streamed rollout: [batch*TICKERS_COUNT, patch_count, model_dim]
    pub uniform_patch_tokens: Tensor,
    /// Live fill per env for the tail bucket: [batch]
    pub uniform_live_fill: Tensor,
    /// Host mirror of live fill to avoid per-step device syncs during streamed rollout.
    pub uniform_live_fill_host: Vec<i64>,
    /// Prefix hidden state after layer-0 self-attention/FFN, before exogenous cross-attention.
    pub uniform_layer0_prefix_hidden: Tensor,
    /// Layer-0 prefix K cache for uniform streamed rollout.
    pub uniform_layer0_prefix_k: Tensor,
    /// Layer-0 prefix V cache for uniform streamed rollout.
    pub uniform_layer0_prefix_v: Tensor,
    /// Per-layer cached prefix K for uniform streamed rollout.
    pub uniform_prefix_k: Vec<Tensor>,
    /// Per-layer cached prefix V for uniform streamed rollout.
    pub uniform_prefix_v: Vec<Tensor>,
    /// Prefix x0 embedding (post-input_ln) for x0 residual mixing.
    pub uniform_prefix_x0: Tensor,
    /// Conditioned prefix hidden state after all GQA layers, before final_ln.
    pub uniform_conditioned_prefix_hidden: Tensor,
    /// Static features associated with the currently conditioned prefix cache.
    pub uniform_cached_static_features: Option<Tensor>,
    /// Exogenous tokens associated with the currently conditioned prefix cache.
    pub uniform_cached_exo_tokens: Option<Tensor>,
}

pub struct TradingModel {
    variant: ModelVariant,
    patch_configs: &'static [(i64, i64)],
    seq_len: i64,
    finest_patch_size: i64,
    model_dim: i64,
    ff_dim: i64,
    patch_embed_weight: Tensor,
    patch_config_ids: Tensor,
    patch_stream_proj: nn::Linear,
    input_ln: RMSNorm,
    final_ln: RMSNorm,
    gqa_layers: Vec<GqaBlock>,
    exogenous_ticker_block: CrossAttnFfnBlock,
    exo_mlp: ExoMLP,
    exo_embed_ln: RMSNorm,
    rope: RotaryEmbedding,
    exo_feat_w: Tensor,
    exo_feat_b: Tensor,
    endogenous_ticker_block: EndogenousTickerBlock,
    readout_queries: Tensor,
    readout_block: CrossAttnFfnBlock,
    policy_mean_log_var: nn::Linear,
    value_proj: nn::Linear,
    device: tch::Device,
}

impl TradingModel {
    pub fn price_input_dim(&self) -> i64 {
        match self.variant {
            ModelVariant::UniformStream => TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN,
            _ => TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        }
    }

    pub fn uniform_stream_bootstrap_live_fill(&self) -> i64 {
        UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL
    }

    pub fn input_kind(&self) -> Kind {
        self.patch_embed_weight.kind()
    }

    fn maybe_to_device(&self, input: &Tensor, device: tch::Device) -> Tensor {
        if input.device() == device {
            input.shallow_clone()
        } else {
            input.to_device(device)
        }
    }

    fn cast_inputs(&self, input: &Tensor) -> Tensor {
        let target_kind = self.activation_kind();
        if input.kind() == target_kind {
            input.shallow_clone()
        } else {
            input.to_kind(target_kind)
        }
    }

    fn activation_kind(&self) -> Kind {
        if self.device.is_cuda() {
            Kind::BFloat16
        } else {
            Kind::Float
        }
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn patch_seq_len(&self) -> i64 {
        self.seq_len
    }

    fn uniform_stream_layout_from_raw(&self, deltas: &Tensor) -> Tensor {
        let device = deltas.device();
        let batch = deltas.size()[0];
        let layout = Tensor::full(
            [batch, UNIFORM_STREAM_LAYOUT_LEN],
            f64::NAN,
            (Kind::Float, device),
        );
        let full_prefix = UNIFORM_STREAM_BOOTSTRAP_FULL_PATCHES * UNIFORM_STREAM_PATCH_SIZE;
        let _ = layout
            .narrow(1, 0, full_prefix)
            .copy_(&deltas.narrow(1, 0, full_prefix));
        let _ = layout
            .narrow(1, full_prefix, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL)
            .copy_(&deltas.narrow(1, full_prefix, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL));
        layout.to_kind(deltas.kind())
    }

    pub fn uniform_stream_layout_from_raw_input(&self, price_deltas: &Tensor) -> Tensor {
        assert_eq!(
            self.variant,
            ModelVariant::UniformStream,
            "uniform_stream_layout_from_raw_input is only valid for UniformStream",
        );
        let price = if price_deltas.dim() == 1 {
            price_deltas.unsqueeze(0)
        } else {
            price_deltas.shallow_clone()
        };
        let price = self.cast_inputs(&self.maybe_to_device(&price, self.device));
        let batch_size = price.size()[0];
        self.uniform_stream_layout_from_raw(
            &price
                .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
                .view([batch_size * TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]),
        )
        .view([batch_size, TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN])
    }

    pub fn new(p: &nn::Path) -> Self {
        Self::new_with_config(p, TradingModelConfig::default())
    }

    pub fn new_with_config(p: &nn::Path, config: TradingModelConfig) -> Self {
        let spec = model_spec(config.variant);
        assert_eq!(
            spec.model_dim % GQA_NUM_Q_HEADS,
            0,
            "model_dim must divide evenly across GQA query heads"
        );
        assert!(GQA_NUM_KV_HEADS > 0, "GQA must have at least one KV head");
        assert_eq!(ROPE_DIMS % 2, 0, "RoPE dimensions must be even");
        let gqa_layers_count = spec.gqa_layers;
        // SA + FFN per layer = 2 sublayers each, plus 1 CA sublayer after layer 0
        let num_residual_sublayers = gqa_layers_count * 2 + 1;
        let init_scale = residual_init_scale(num_residual_sublayers);
        let patch_configs = spec.patch_configs;
        let (total_days, seq_len) = compute_patch_totals(patch_configs);
        if config.variant == ModelVariant::UniformStream {
            assert_eq!(
                UNIFORM_STREAM_LAYOUT_LEN, PRICE_DELTAS_PER_TICKER as i64,
                "uniform stream layout must exactly match the raw observation history"
            );
            assert_eq!(
                UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL, UNIFORM_STREAM_PATCH_SIZE,
                "uniform stream bootstrap must fill the live patch exactly"
            );
            assert_eq!(
                total_days, UNIFORM_STREAM_LAYOUT_LEN,
                "uniform stream patch configs must sum to the layout length"
            );
        } else {
            assert!(
                total_days == PRICE_DELTAS_PER_TICKER as i64,
                "patch configs must sum to PRICE_DELTAS_PER_TICKER"
            );
        }
        let finest_patch_index = patch_configs.len() - 1;
        let finest_patch_size = patch_configs[finest_patch_index].1;
        let num_configs = patch_configs.len() as i64;
        let max_patch_size = patch_configs
            .iter()
            .map(|&(_, patch_size)| patch_size)
            .max()
            .unwrap_or(0);
        let max_input_dim = max_patch_size + PATCH_SCALAR_FEATS;
        let xavier_std = xavier_normal_std(max_input_dim, spec.model_dim);
        let patch_embed_weight = p.var(
            "patch_embed_weight",
            &[num_configs, max_input_dim, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: xavier_std,
            },
        );
        let patch_stream_proj = nn::linear(
            p / "patch_stream_proj",
            UNIFORM_STREAM_PATCH_SIZE + 1, // patch values + fill_fraction
            spec.model_dim,
            nn::LinearConfig {
                ws_init: truncated_normal_init(UNIFORM_STREAM_PATCH_SIZE + 1, spec.model_dim),
                bs_init: None,
                bias: false,
            },
        );
        let input_ln = RMSNorm::new(&(p / "input_ln"), spec.model_dim, 1e-6);
        let final_ln = RMSNorm::new(&(p / "final_ln"), spec.model_dim, 1e-6);
        let patch_config_ids = {
            let mut ids = Vec::with_capacity(seq_len as usize);
            for (cfg_idx, &(days, patch_size)) in patch_configs.iter().enumerate() {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    ids.push(cfg_idx as i64);
                }
            }
            Tensor::from_slice(&ids)
                .to_kind(Kind::Int64)
                .to_device(p.device())
        };

        let gqa_layers = (0..gqa_layers_count)
            .map(|i| {
                GqaBlock::new(
                    &(p / format!("gqa_{}", i)),
                    spec.model_dim,
                    spec.ff_dim,
                    init_scale,
                    i,
                )
            })
            .collect::<Vec<_>>();
        let exogenous_ticker_block = CrossAttnFfnBlock::new(
            &(p / "cross_attn_0"),
            spec.model_dim,
            spec.ff_dim,
            init_scale,
        );
        let exo_mlp = ExoMLP::new(&(p / "exo_mlp"), spec.model_dim, init_scale);
        let exo_embed_ln = RMSNorm::new(&(p / "exo_embed_ln"), spec.model_dim, 1e-6);
        let head_dim = spec.model_dim / GQA_NUM_Q_HEADS;
        let rope = RotaryEmbedding::new(seq_len, head_dim, ROPE_DIMS, p.device());
        let exo_feat_w = p.var(
            "exo_feat_w",
            &[NUM_EXO_TOKENS, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: xavier_normal_std(1, spec.model_dim),
            },
        );
        let exo_feat_b = p.var(
            "exo_feat_b",
            &[NUM_EXO_TOKENS, spec.model_dim],
            Init::Const(0.0),
        );
        let endogenous_ticker_block = EndogenousTickerBlock::new(
            &(p / "inter_ticker_0"),
            spec.model_dim,
            spec.ff_dim,
            init_scale,
        );
        let readout_queries = p.var(
            "actor_critic_readout_queries",
            &[2, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );
        let readout_block = CrossAttnFfnBlock::new(
            &(p / "actor_critic_readout"),
            spec.model_dim,
            spec.ff_dim,
            init_scale,
        );
        assert_eq!(
            ACTION_COUNT, TICKERS_COUNT,
            "per-ticker actor head requires one action per ticker"
        );
        let flat_all_tickers = TICKERS_COUNT * spec.model_dim;
        let policy_mean_log_var = nn::linear(
            p / "policy_mean_log_var",
            spec.model_dim,
            2,
            nn::LinearConfig {
                ws_init: Init::Randn {
                    mean: 0.0,
                    stdev: 0.01,
                },
                bs_init: None,
                bias: false,
            },
        );
        let value_proj = nn::linear(
            p / "value_proj",
            flat_all_tickers,
            NUM_BINS,
            nn::LinearConfig {
                ws_init: Init::Uniform {
                    lo: -1.0 / (flat_all_tickers as f64).sqrt(),
                    up: 1.0 / (flat_all_tickers as f64).sqrt(),
                },
                bs_init: None,
                bias: false,
            },
        );
        Self {
            variant: config.variant,
            patch_configs,
            seq_len,
            finest_patch_size,
            model_dim: spec.model_dim,
            ff_dim: spec.ff_dim,
            patch_embed_weight,
            patch_config_ids,
            patch_stream_proj,
            input_ln,
            final_ln,
            gqa_layers,
            exogenous_ticker_block,
            exo_mlp,
            exo_embed_ln,
            rope,
            exo_feat_w,
            exo_feat_b,
            endogenous_ticker_block,
            readout_queries,
            readout_block,
            policy_mean_log_var,
            value_proj,
            device: p.device(),
        }
    }

    fn parse_static(&self, static_features: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let global = static_features.narrow(1, 0, GLOBAL_STATIC_OBS as i64);
        let per_ticker = static_features
            .narrow(
                1,
                GLOBAL_STATIC_OBS as i64,
                TICKERS_COUNT * PER_TICKER_STATIC_OBS as i64,
            )
            .reshape([batch_size, TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64]);
        (global, per_ticker)
    }

    fn maybe_apply_endogenous_ticker(&self, x: &Tensor, layer_idx: usize) -> Tensor {
        if layer_idx != INTER_TICKER_AFTER || TICKERS_COUNT == 1 {
            return x.shallow_clone();
        }
        let bt = x.size()[0];
        let seq = x.size()[1];
        let batch_size = bt / TICKERS_COUNT;
        let x_4d = x.view([batch_size, TICKERS_COUNT, seq, self.model_dim]);
        let live = x_4d.narrow(2, seq - 1, 1);
        let live_for_mix =
            live.permute([0, 2, 1, 3])
                .reshape([batch_size, TICKERS_COUNT, self.model_dim]);
        let enriched_live = self
            .endogenous_ticker_block
            .forward(&live_for_mix, self.model_dim, self.ff_dim)
            .reshape([batch_size, 1, TICKERS_COUNT, self.model_dim])
            .permute([0, 2, 1, 3]);
        if seq == 1 {
            enriched_live.reshape([bt, seq, self.model_dim])
        } else {
            let past = x_4d.narrow(2, 0, seq - 1);
            Tensor::cat(&[&past, &enriched_live], 2).reshape([bt, seq, self.model_dim])
        }
    }

    /// Build exogenous KV bank: [batch*tickers, NUM_EXO_TOKENS, MODEL_DIM]
    /// Each of the 46 static features gets its own token via per-feature learned projection
    fn build_exo_kv(
        &self,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let global_exp = global_static.unsqueeze(1).expand(
            &[batch_size, TICKERS_COUNT, GLOBAL_STATIC_OBS as i64],
            false,
        );
        let all_feats = Tensor::cat(&[global_exp, per_ticker_static.shallow_clone()], -1);
        let all_feats = all_feats.reshape([batch_size * TICKERS_COUNT, NUM_EXO_TOKENS]);
        let feats_expanded = all_feats.unsqueeze(-1);
        let exo_feat_w = self.exo_feat_w.to_kind(feats_expanded.kind());
        let exo_feat_b = self.exo_feat_b.to_kind(feats_expanded.kind());
        feats_expanded * &exo_feat_w + &exo_feat_b
    }

    /// Build exo tokens with MLP refinement: [batch*tickers, NUM_EXO_TOKENS, MODEL_DIM]
    fn build_exo_tokens(
        &self,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let exo_kv = self.build_exo_kv(global_static, per_ticker_static, batch_size);
        self.exo_mlp.forward(&self.exo_embed_ln.forward(&exo_kv))
    }

    fn patch_latent_stem_on_device(&self, price_deltas: &Tensor, batch_size: i64) -> Tensor {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = if self.variant == ModelVariant::UniformStream {
            let expected_layout = TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN;
            assert_eq!(
                price_deltas.size()[1],
                expected_layout,
                "UniformStream full forward expects anchored layout input"
            );
            price_deltas
                .view([batch_size, TICKERS_COUNT, UNIFORM_STREAM_LAYOUT_LEN])
                .view([batch_tokens, UNIFORM_STREAM_LAYOUT_LEN])
        } else {
            price_deltas
                .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
                .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64])
        };

        let patch_tokens = self.patch_embed(&deltas);
        self.input_ln.forward(&patch_tokens)
    }

    /// Per-config enrichment avoids expanding each patch to the full history width,
    /// then projects all tokens in one fused einsum.
    fn patch_embed(&self, deltas: &Tensor) -> Tensor {
        if self.variant == ModelVariant::UniformStream {
            return self.patch_embed_stream(deltas);
        }
        let device = deltas.device();
        let kind = deltas.kind();
        let batch = deltas.size()[0];
        let max_patch_size = self.patch_embed_weight.size()[1] - PATCH_SCALAR_FEATS;

        // Phase 1: per-config enrichment, zero-padded to max_input_dim, then cat
        let max_input_dim = max_patch_size + PATCH_SCALAR_FEATS;
        let mut enriched_parts = Vec::with_capacity(self.patch_configs.len());
        let mut delta_offset = 0i64;
        for &(days, patch_size) in self.patch_configs {
            let n_patches = days / patch_size;
            let patches = deltas
                .narrow(1, delta_offset, days)
                .view([batch, n_patches, patch_size])
                .to_kind(Kind::Float);
            let mean = patches.mean_dim([2].as_slice(), true, Kind::Float);
            let var = (&patches - &mean).pow_tensor_scalar(2.0).mean_dim(
                [2].as_slice(),
                true,
                Kind::Float,
            );
            let std = (var + 1e-5).sqrt();
            let first = patches.narrow(2, 0, 1);
            let last = patches.narrow(2, patch_size - 1, 1);
            let slope = &last - &first;
            let enriched = Tensor::cat(&[&patches, &mean, &std, &slope], 2);
            // Zero-pad to max_input_dim so all configs share the einsum
            let pad_cols = max_input_dim - (patch_size + PATCH_SCALAR_FEATS);
            let padded = if pad_cols > 0 {
                let pad = Tensor::zeros(&[batch, n_patches, pad_cols], (Kind::Float, device));
                Tensor::cat(&[&enriched, &pad], 2)
            } else {
                enriched
            };
            enriched_parts.push(padded);
            delta_offset += days;
        }
        let enriched = Tensor::cat(&enriched_parts.iter().collect::<Vec<_>>(), 1).to_kind(kind);

        // Phase 2: fused projection over all tokens.
        let weight_per_patch = self
            .patch_embed_weight
            .index_select(0, &self.patch_config_ids)
            .to_kind(kind);
        let out = Tensor::einsum(
            "blm,lmd->bld",
            &[&enriched, &weight_per_patch],
            None::<&[i64]>,
        );
        out
    }

    fn patch_embed_stream_batch(&self, patch_vals: &Tensor, fill_counts: &Tensor) -> Tensor {
        let target_kind = patch_vals.kind();
        let patch_size = UNIFORM_STREAM_PATCH_SIZE;
        // Build position mask from fill counts — don't read NaN positions
        let positions = Tensor::arange(patch_size, (Kind::Int64, patch_vals.device()));
        let mask = positions
            .unsqueeze(0)
            .less_tensor(&fill_counts.unsqueeze(-1)); // [batch, patch_size]
        let clean = patch_vals
            .to_kind(Kind::Float)
            .where_self(&mask, &Tensor::from(0.0f32).to_device(patch_vals.device()));
        let fill_fraction = fill_counts.to_kind(Kind::Float).unsqueeze(-1) / patch_size as f64;
        let input = Tensor::cat(
            &[
                &clean.to_kind(target_kind),
                &fill_fraction.to_kind(target_kind),
            ],
            -1,
        ); // [batch, patch_size + 1]
        linear_with_same_dtype(&input, &self.patch_stream_proj)
    }

    fn patch_embed_stream(&self, deltas: &Tensor) -> Tensor {
        let batch = deltas.size()[0];
        let patches = deltas.view([
            batch * UNIFORM_STREAM_PATCH_COUNT,
            UNIFORM_STREAM_PATCH_SIZE,
        ]);
        // Compute fill counts per patch from valid (non-NaN) positions
        let fill_counts = patches
            .isnan()
            .logical_not()
            .to_kind(Kind::Int64)
            .sum_dim_intlist([1].as_slice(), false, Kind::Int64);
        self.patch_embed_stream_batch(&patches, &fill_counts).view([
            batch,
            UNIFORM_STREAM_PATCH_COUNT,
            self.model_dim,
        ])
    }
}
