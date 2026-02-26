mod forward;
mod head;
mod inference;
mod rmsnorm;

use clap::ValueEnum;
use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{
    GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS,
    TICKERS_COUNT,
};

use rmsnorm::RMSNorm;

struct InterTickerBlock {
    ticker_ln: RMSNorm,
    ticker_qkv: nn::Linear,
    ticker_out: nn::Linear,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    mlp_ln: RMSNorm,
}

impl InterTickerBlock {
    fn new(p: &nn::Path, model_dim: i64, ff_dim: i64) -> Self {
        let ticker_ln = RMSNorm::new(&(p / "ticker_ln"), model_dim, 1e-6);
        let ticker_qkv = linear_truncated(p, "ticker_qkv", model_dim, 3 * model_dim);
        let ticker_out = linear_residual_out(p, "ticker_out", model_dim, model_dim);
        let mlp_fc1 = linear_truncated(p, "mlp_fc1", model_dim, 2 * ff_dim);
        let mlp_fc2 = linear_residual_out(p, "mlp_fc2", ff_dim, model_dim);
        let mlp_ln = RMSNorm::new(&(p / "mlp_ln"), model_dim, 1e-6);
        Self {
            ticker_ln,
            ticker_qkv,
            ticker_out,
            mlp_fc1,
            mlp_fc2,
            mlp_ln,
        }
    }

    fn forward(&self, x: &Tensor, model_dim: i64, ff_dim: i64) -> Tensor {
        let (batch, num_items, _) = x.size3().unwrap();
        let x_norm = self.ticker_ln
            .forward(&x.reshape([batch * num_items, model_dim]))
            .reshape([batch, num_items, model_dim]);
        let qkv = x_norm.apply(&self.ticker_qkv);
        let parts = qkv.split(model_dim, -1);
        let q = parts[0].unsqueeze(1);
        let k = parts[1].unsqueeze(1);
        let v = parts[2].unsqueeze(1);
        let ctx = Tensor::scaled_dot_product_attention(
            &q, &k, &v,
            None::<&Tensor>, 0.0, false, None, false,
        )
            .squeeze_dim(1)
            .apply(&self.ticker_out)
            .reshape([batch, num_items, model_dim]);
        let x = x + ctx;
        let mlp_in = self.mlp_ln
            .forward(&x.reshape([batch * num_items, model_dim]));
        let mlp_proj = mlp_in.apply(&self.mlp_fc1);
        let mlp_parts = mlp_proj.split(ff_dim, -1);
        let mlp = (mlp_parts[0].silu() * &mlp_parts[1])
            .apply(&self.mlp_fc2)
            .reshape([batch, num_items, model_dim]);
        &x + mlp
    }
}

const EXO_CROSS_AFTER: &[usize] = &[0];

struct ExoCrossBlock {
    cross_ln: RMSNorm,
    cross_q: nn::Linear,
    cross_kv: nn::Linear,
    cross_out: nn::Linear,
    kv_dim: i64,
}

impl ExoCrossBlock {
    fn new(p: &nn::Path, model_dim: i64, cross_head_dim: i64) -> Self {
        let kv_dim = CROSS_NUM_KV_HEADS * cross_head_dim;
        let cross_ln = RMSNorm::new(&(p / "cross_ln"), model_dim, 1e-6);
        let cross_q = linear_truncated(p, "cross_q", model_dim, model_dim);
        let cross_kv = linear_truncated(p, "cross_kv", model_dim, 2 * kv_dim);
        let cross_out = linear_residual_out(p, "cross_out", model_dim, model_dim);
        Self {
            cross_ln,
            cross_q,
            cross_kv,
            cross_out,
            kv_dim,
        }
    }

    fn forward(&self, x: &Tensor, exo_kv: &Tensor, model_dim: i64, cross_head_dim: i64) -> Tensor {
        let (b, s, _d) = x.size3().unwrap_or_else(|_| {
            let (b, _d) = x.size2().unwrap();
            (b, 1, _d)
        });
        let x_3d = if x.dim() == 2 {
            x.unsqueeze(1)
        } else {
            x.shallow_clone()
        };
        let t = exo_kv.size()[1];

        let q = self.cross_ln.forward(&x_3d).apply(&self.cross_q);
        let kv = exo_kv.apply(&self.cross_kv);
        let kv_parts = kv.split(self.kv_dim, -1);

        let q = q
            .reshape([b, s, CROSS_NUM_Q_HEADS, cross_head_dim])
            .permute([0, 2, 1, 3]);
        let k = kv_parts[0]
            .reshape([b, t, CROSS_NUM_KV_HEADS, cross_head_dim])
            .permute([0, 2, 1, 3]);
        let v = kv_parts[1]
            .reshape([b, t, CROSS_NUM_KV_HEADS, cross_head_dim])
            .permute([0, 2, 1, 3]);

        let out = Tensor::scaled_dot_product_attention(
            &q, &k, &v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            true,
        );

        let out = out.permute([0, 2, 1, 3]).reshape([b, s, model_dim]);
        let out = out.apply(&self.cross_out);

        let result = &x_3d + out;
        if x.dim() == 2 {
            result.squeeze_dim(1)
        } else {
            result
        }
    }
}

const GQA_NUM_Q_HEADS: i64 = 4;
const GQA_NUM_KV_HEADS: i64 = 2;

fn rotate_half(x: &Tensor) -> Tensor {
    let last_dim = *x.size().last().unwrap();
    let half = last_dim / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[&(-&x2), &x1], -1)
}

struct RotaryEmbedding {
    cos_cached: Tensor, // [max_seq_len, head_dim]
    sin_cached: Tensor, // [max_seq_len, head_dim]
}

impl RotaryEmbedding {
    fn new(max_seq_len: i64, head_dim: i64, device: tch::Device) -> Self {
        let half_dim = head_dim / 2;
        let exponents =
            Tensor::arange(half_dim, (Kind::Float, device)) * (2.0 / head_dim as f64);
        let inv_freq = (exponents * -(10000.0_f64.ln())).exp(); // [half_dim]
        let positions = Tensor::arange(max_seq_len, (Kind::Float, device)); // [max_seq_len]
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0); // [max_seq_len, half_dim]
        let cos_half = angles.cos();
        let sin_half = angles.sin();
        Self {
            cos_cached: Tensor::cat(&[&cos_half, &cos_half], -1).set_requires_grad(false),
            sin_cached: Tensor::cat(&[&sin_half, &sin_half], -1).set_requires_grad(false),
        }
    }

    fn apply(&self, x: &Tensor) -> Tensor {
        // x: [batch, heads, seq_len, head_dim]
        let seq_len = x.size()[2];
        let cos = self.cos_cached.narrow(0, 0, seq_len).to_kind(x.kind());
        let sin = self.sin_cached.narrow(0, 0, seq_len).to_kind(x.kind());
        x * &cos + rotate_half(x) * &sin
    }
}

struct GqaBlock {
    attn_ln: RMSNorm,
    attn_qkv: nn::Linear,
    attn_out: nn::Linear,
    q_dim: i64,
    kv_dim: i64,
    ffn_ln: RMSNorm,
    ffn_fc1: nn::Linear,
    ffn_fc2: nn::Linear,
}

impl GqaBlock {
    fn new(p: &nn::Path, model_dim: i64, ff_dim: i64) -> Self {
        let head_dim = model_dim / GQA_NUM_Q_HEADS;
        let kv_dim = GQA_NUM_KV_HEADS * head_dim;
        let qkv_dim = model_dim + 2 * kv_dim;
        let attn_ln = RMSNorm::new(&(p / "attn_ln"), model_dim, 1e-6);
        let attn_qkv = linear_truncated(p, "attn_qkv", model_dim, qkv_dim);
        let attn_out = linear_residual_out(p, "attn_out", model_dim, model_dim);
        let ffn_ln = RMSNorm::new(&(p / "ffn_ln"), model_dim, 1e-6);
        let ffn_fc1 = linear_truncated(p, "ffn_fc1", model_dim, 2 * ff_dim);
        let ffn_fc2 = linear_residual_out(p, "ffn_fc2", ff_dim, model_dim);
        Self {
            attn_ln,
            attn_qkv,
            attn_out,
            q_dim: model_dim,
            kv_dim,
            ffn_ln,
            ffn_fc1,
            ffn_fc2,
        }
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding) -> Tensor {
        let (b, s, _d) = x.size3().unwrap();
        let head_dim = _d / GQA_NUM_Q_HEADS;

        // Self-attention with pre-norm
        let x_norm = self.attn_ln.forward(x);
        let qkv = x_norm.apply(&self.attn_qkv);
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

        let q = rope.apply(&q);
        let k = rope.apply(&k);

        let out = Tensor::scaled_dot_product_attention(
            &q, &k, &v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            true,
        )
            .permute([0, 2, 1, 3])
            .contiguous()
            .reshape([b, s, _d]);
        let out = out.apply(&self.attn_out);
        let x = x + out;

        // SwiGLU FFN with pre-norm
        let ffn_in = self.ffn_ln.forward(&x);
        let ffn_proj = ffn_in.apply(&self.ffn_fc1);
        let ffn_parts = ffn_proj.chunk(2, -1);
        let ffn_out = (ffn_parts[0].silu() * &ffn_parts[1]).apply(&self.ffn_fc2);
        &x + ffn_out
    }
}

fn truncated_normal_std(in_features: i64, out_features: i64) -> f64 {
    let denoms = (in_features + out_features) as f64 / 2.0;
    (1.0 / denoms).sqrt() / 0.8796
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
            ws_init: truncated_normal_init(in_features, out_features),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        },
    )
}

/// Output projection feeding into a residual add
fn linear_residual_out(
    p: &nn::Path,
    name: &str,
    in_features: i64,
    out_features: i64,
) -> nn::Linear {
    let base_std = truncated_normal_std(in_features, out_features);
    nn::linear(
        p / name,
        in_features,
        out_features,
        nn::LinearConfig {
            ws_init: Init::Randn {
                mean: 0.0,
                stdev: base_std,
            },
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        },
    )
}

const BASE_MODEL_DIM: i64 = 128;
const BASE_FF_DIM: i64 = 512;
const BASE_GQA_LAYERS: usize = 3;
const ABLATION_SMALL_MODEL_DIM: i64 = 64;
const ABLATION_SMALL_FF_DIM: i64 = 256;
const ABLATION_SMALL_GQA_LAYERS: usize = 1;
pub(super) const SDE_LATENT_DIM: i64 = 64;
pub(super) const SDE_EPS: f64 = 1e-6;
pub(super) const SDE_NOISE_FLOOR: f64 = 0.5;
const INTER_TICKER_AFTER: usize = 1;
const CROSS_NUM_Q_HEADS: i64 = 4;
const CROSS_NUM_KV_HEADS: i64 = 2;
const NUM_EXO_TOKENS: i64 = STATIC_OBSERVATIONS as i64;
const PATCH_SCALAR_FEATS: i64 = 3;

const BASE_PATCH_CONFIGS: &[(i64, i64)] = &[
    (4608, 128),
    (2048, 64),
    (1024, 32),
    (512, 16),
    (256, 8),
    (128, 4),
    (60, 1),
];

const ABLATION_SMALL_PATCH_CONFIGS: &[(i64, i64)] = &[(8192, 256), (384, 8), (60, 1)];

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ModelVariant {
    Base,
    AblationSmall,
}

impl ModelVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
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

/// (values, action_mean, action_noise_std)
pub type ModelOutput = (Tensor, Tensor, Tensor);

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
}

pub struct TradingModel {
    variant: ModelVariant,
    patch_configs: &'static [(i64, i64)],
    seq_len: i64,
    finest_patch_size: i64,
    model_dim: i64,
    ff_dim: i64,
    cross_head_dim: i64,
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    patch_config_ids: Tensor,
    gqa_layers: Vec<GqaBlock>,
    rope: RotaryEmbedding,
    exo_feat_w: Tensor,
    exo_feat_b: Tensor,
    exo_cross_blocks: Vec<ExoCrossBlock>,
    cls_token: Tensor,
    inter_ticker_block: InterTickerBlock,
    actor_proj: nn::Linear,
    value_proj: nn::Linear,
    sde_fc: nn::Linear,
    sde_norm: RMSNorm,
    sde_fc2: nn::Linear,
    sde_fc3: nn::Linear,
    device: tch::Device,
}

impl TradingModel {
    fn maybe_to_device(&self, input: &Tensor, device: tch::Device) -> Tensor {
        if input.device() == device {
            input.shallow_clone()
        } else {
            input.to_device(device)
        }
    }



    fn cast_inputs(&self, input: &Tensor) -> Tensor {
        let target_kind = self.patch_embed_weight.kind();
        if input.kind() == target_kind {
            input.shallow_clone()
        } else {
            input.to_kind(target_kind)
        }
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn patch_seq_len(&self) -> i64 {
        self.seq_len
    }

    pub fn new(p: &nn::Path) -> Self {
        Self::new_with_config(p, TradingModelConfig::default())
    }

    pub fn new_with_config(p: &nn::Path, config: TradingModelConfig) -> Self {
        let spec = model_spec(config.variant);
        let gqa_layers_count = spec.gqa_layers;
        let patch_configs = spec.patch_configs;
        let (total_days, seq_len) = compute_patch_totals(patch_configs);
        assert!(
            total_days == PRICE_DELTAS_PER_TICKER as i64,
            "patch configs must sum to PRICE_DELTAS_PER_TICKER"
        );
        let finest_patch_index = patch_configs.len() - 1;
        let finest_patch_size = patch_configs[finest_patch_index].1;
        let num_configs = patch_configs.len() as i64;
        let max_patch_size = patch_configs
            .iter()
            .map(|&(_, patch_size)| patch_size)
            .max()
            .unwrap_or(0);
        let max_input_dim = max_patch_size + PATCH_SCALAR_FEATS;
        let xavier_std = (2.0 / (max_input_dim + spec.model_dim) as f64).sqrt();
        let patch_embed_weight = p.var(
            "patch_embed_weight",
            &[num_configs, max_input_dim, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: xavier_std,
            },
        );
        let patch_embed_bias = p.var(
            "patch_embed_bias",
            &[num_configs, spec.model_dim],
            Init::Const(0.0),
        );
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
            .map(|i| GqaBlock::new(&(p / format!("gqa_{}", i)), spec.model_dim, spec.ff_dim))
            .collect::<Vec<_>>();
        let head_dim = spec.model_dim / GQA_NUM_Q_HEADS;
        let rope = RotaryEmbedding::new(seq_len + 1, head_dim, p.device());
        let exo_feat_w = p.var(
            "exo_feat_w",
            &[NUM_EXO_TOKENS, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: (1.0 / spec.model_dim as f64).sqrt(),
            },
        );
        let exo_feat_b = p.var(
            "exo_feat_b",
            &[NUM_EXO_TOKENS, spec.model_dim],
            Init::Const(0.0),
        );
        let cross_head_dim = spec.model_dim / CROSS_NUM_Q_HEADS;
        let exo_cross_blocks = EXO_CROSS_AFTER
            .iter()
            .enumerate()
            .map(|(i, _)| {
                ExoCrossBlock::new(
                    &(p / format!("exo_cross_{}", i)),
                    spec.model_dim,
                    cross_head_dim,
                )
            })
            .collect::<Vec<_>>();
        let cls_token = p.var("cls_token", &[1, 1, spec.model_dim], Init::Const(0.0));
        let inter_ticker_block =
            InterTickerBlock::new(&(p / "inter_ticker_0"), spec.model_dim, spec.ff_dim);
        let full_seq_len = seq_len + 1;
        let flat_per_ticker = full_seq_len * spec.model_dim;
        let flat_all_tickers = TICKERS_COUNT * flat_per_ticker;
        let actor_proj = nn::linear(
            p / "actor_proj",
            flat_per_ticker,
            1,
            nn::LinearConfig {
                ws_init: Init::Orthogonal { gain: 100.0 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        let value_proj = nn::linear(
            p / "value_proj",
            flat_all_tickers,
            1,
            nn::LinearConfig {
                ws_init: Init::Orthogonal { gain: 1.0 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        let sde_fc = nn::linear(
            p / "sde_fc",
            spec.model_dim,
            SDE_LATENT_DIM,
            nn::LinearConfig {
                ws_init: Init::Orthogonal { gain: 1.0 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        let sde_norm = RMSNorm::new(&(p / "sde_norm"), SDE_LATENT_DIM, 1e-5);
        let sde_fc2 = nn::linear(
            p / "sde_fc2",
            SDE_LATENT_DIM,
            SDE_LATENT_DIM,
            nn::LinearConfig {
                ws_init: Init::Orthogonal { gain: 1.0 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        let sde_fc3 = nn::linear(
            p / "sde_fc3",
            SDE_LATENT_DIM,
            SDE_LATENT_DIM,
            nn::LinearConfig {
                ws_init: Init::Orthogonal { gain: 0.01 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        Self {
            variant: config.variant,
            patch_configs,
            seq_len,
            finest_patch_size,
            model_dim: spec.model_dim,
            ff_dim: spec.ff_dim,
            cross_head_dim,
            patch_embed_weight,
            patch_embed_bias,
            patch_config_ids,
            gqa_layers,
            rope,
            exo_feat_w,
            exo_feat_b,
            exo_cross_blocks,
            cls_token,
            inter_ticker_block,
            actor_proj,
            value_proj,
            sde_fc,
            sde_norm,
            sde_fc2,
            sde_fc3,
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

    /// Apply exo cross-block if this layer index is in EXO_CROSS_AFTER
    fn maybe_apply_exo_cross(&self, x: &Tensor, exo_kv: &Tensor, layer_idx: usize) -> Tensor {
        if let Some(pos) = EXO_CROSS_AFTER.iter().position(|&i| i == layer_idx) {
            self.exo_cross_blocks[pos].forward(x, exo_kv, self.model_dim, self.cross_head_dim)
        } else {
            x.shallow_clone()
        }
    }

    fn maybe_apply_inter_ticker(&self, x: &Tensor, layer_idx: usize) -> Tensor {
        if layer_idx != INTER_TICKER_AFTER {
            return x.shallow_clone();
        }
        let bt = x.size()[0];
        let seq = x.size()[1];
        let batch_size = bt / TICKERS_COUNT;
        let x_4d = x.view([batch_size, TICKERS_COUNT, seq, self.model_dim]);
        let cls = x_4d.select(2, 0);
        let enriched_cls = self.inter_ticker_block.forward(&cls, self.model_dim, self.ff_dim);
        let non_cls = x_4d.narrow(2, 1, seq - 1);
        Tensor::cat(&[&enriched_cls.unsqueeze(2), &non_cls], 2)
            .reshape([bt, seq, self.model_dim])
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
        feats_expanded * &self.exo_feat_w + &self.exo_feat_b
    }

    fn patch_latent_stem(
        &self,
        price_deltas: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let price_deltas = self.maybe_to_device(price_deltas, self.device);
        self.patch_latent_stem_on_device(&price_deltas, batch_size)
    }

    fn patch_latent_stem_on_device(
        &self,
        price_deltas: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let kind = deltas.kind();
        let patch_tokens = self.patch_embed(&deltas);
        let cls = self.cls_token.to_kind(kind).expand([batch_tokens, 1, self.model_dim], false);
        Tensor::cat(&[&cls, &patch_tokens], 1)
    }

    /// Per-config enrichment (avoids [batch, 256, 8636] expand), then fused
    /// einsum projection across all 256 tokens in a single kernel.
    fn patch_embed(&self, deltas: &Tensor) -> Tensor {
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
        let enriched = Tensor::cat(&enriched_parts.iter().collect::<Vec<_>>(), 1)
            .to_kind(kind);

        // Phase 2: fused projection — single einsum over all 256 tokens
        let weight_per_patch = self.patch_embed_weight.index_select(0, &self.patch_config_ids);
        let bias_per_patch = self.patch_embed_bias.index_select(0, &self.patch_config_ids);
        let out = Tensor::einsum(
            "blm,lmd->bld",
            &[&enriched, &weight_per_patch],
            None::<&[i64]>,
        );
        out + bias_per_patch.unsqueeze(0)
    }

}
