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
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    mlp_ln: RMSNorm,
    alpha_ticker_attn: Tensor,
    alpha_mlp: Tensor,
}

impl InterTickerBlock {
    fn new(p: &nn::Path, model_dim: i64, ff_dim: i64) -> Self {
        let ticker_ln = RMSNorm::new(&(p / "ticker_ln"), model_dim, 1e-6);
        let ticker_qkv = linear_truncated(p, "ticker_qkv", model_dim, 3 * model_dim);
        let ticker_out = linear_truncated(p, "ticker_out", model_dim, model_dim);
        let q_norm = RMSNorm::new(&(p / "ticker_q_norm"), model_dim, 1e-6);
        let k_norm = RMSNorm::new(&(p / "ticker_k_norm"), model_dim, 1e-6);
        let mlp_fc1 = linear_truncated(p, "mlp_fc1", model_dim, 2 * ff_dim);
        let mlp_fc2 = linear_truncated(p, "mlp_fc2", ff_dim, model_dim);
        let mlp_ln = RMSNorm::new(&(p / "mlp_ln"), model_dim, 1e-6);
        let alpha_ticker_attn = p.var(
            "alpha_ticker_attn_raw",
            &[1],
            Init::Const(RESIDUAL_ALPHA_INIT),
        );
        let alpha_mlp = p.var("alpha_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            ticker_ln,
            ticker_qkv,
            ticker_out,
            q_norm,
            k_norm,
            mlp_fc1,
            mlp_fc2,
            mlp_ln,
            alpha_ticker_attn,
            alpha_mlp,
        }
    }
}

const EXO_CROSS_AFTER: &[usize] = &[0];

struct ExoCrossBlock {
    cross_ln: RMSNorm,
    cross_q: nn::Linear,
    cross_kv: nn::Linear,
    cross_out: nn::Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    kv_dim: i64,
    alpha_cross: Tensor,
}

impl ExoCrossBlock {
    fn new(p: &nn::Path, model_dim: i64, cross_head_dim: i64) -> Self {
        let kv_dim = CROSS_NUM_KV_HEADS * cross_head_dim;
        let cross_ln = RMSNorm::new(&(p / "cross_ln"), model_dim, 1e-6);
        let cross_q = linear_truncated(p, "cross_q", model_dim, model_dim);
        let cross_kv = linear_truncated(p, "cross_kv", model_dim, 2 * kv_dim);
        let cross_out = linear_truncated(p, "cross_out", model_dim, model_dim);
        let q_norm = RMSNorm::new(&(p / "cross_q_norm"), cross_head_dim, 1e-6);
        let k_norm = RMSNorm::new(&(p / "cross_k_norm"), cross_head_dim, 1e-6);
        let alpha_cross = p.var("alpha_cross", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            cross_ln,
            cross_q,
            cross_kv,
            cross_out,
            q_norm,
            k_norm,
            kv_dim,
            alpha_cross,
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

        // QKNorm: apply RMSNorm per-head on head_dim (last dimension)
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

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

        let result = &x_3d + self.alpha_cross.sigmoid() * RESIDUAL_ALPHA_MAX * out;
        if x.dim() == 2 {
            result.squeeze_dim(1)
        } else {
            result
        }
    }
}

const GQA_NUM_Q_HEADS: i64 = 4;
const GQA_NUM_KV_HEADS: i64 = 2;

struct GqaBlock {
    attn_ln: RMSNorm,
    attn_qkv: nn::Linear,
    attn_out: nn::Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    q_dim: i64,
    kv_dim: i64,
    alpha_attn: Tensor,
    ffn_ln: RMSNorm,
    ffn_fc1: nn::Linear,
    ffn_fc2: nn::Linear,
    alpha_ffn: Tensor,
}

impl GqaBlock {
    fn new(p: &nn::Path, model_dim: i64, ff_dim: i64) -> Self {
        let head_dim = model_dim / GQA_NUM_Q_HEADS;
        let kv_dim = GQA_NUM_KV_HEADS * head_dim;
        let qkv_dim = model_dim + 2 * kv_dim;
        let attn_ln = RMSNorm::new(&(p / "attn_ln"), model_dim, 1e-6);
        let attn_qkv = linear_truncated(p, "attn_qkv", model_dim, qkv_dim);
        let attn_out = linear_truncated(p, "attn_out", model_dim, model_dim);
        let q_norm = RMSNorm::new(&(p / "attn_q_norm"), head_dim, 1e-6);
        let k_norm = RMSNorm::new(&(p / "attn_k_norm"), head_dim, 1e-6);
        let alpha_attn = p.var("alpha_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let ffn_ln = RMSNorm::new(&(p / "ffn_ln"), model_dim, 1e-6);
        let ffn_fc1 = linear_truncated(p, "ffn_fc1", model_dim, 2 * ff_dim);
        let ffn_fc2 = linear_truncated(p, "ffn_fc2", ff_dim, model_dim);
        let alpha_ffn = p.var("alpha_ffn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            attn_ln,
            attn_qkv,
            attn_out,
            q_norm,
            k_norm,
            q_dim: model_dim,
            kv_dim,
            alpha_attn,
            ffn_ln,
            ffn_fc1,
            ffn_fc2,
            alpha_ffn,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
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

        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

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
        let x = x + self.alpha_attn.sigmoid() * RESIDUAL_ALPHA_MAX * out;

        // SwiGLU FFN with pre-norm
        let ffn_in = self.ffn_ln.forward(&x);
        let ffn_proj = ffn_in.apply(&self.ffn_fc1);
        let ffn_parts = ffn_proj.chunk(2, -1);
        let ffn_out = (ffn_parts[0].silu() * &ffn_parts[1]).apply(&self.ffn_fc2);
        &x + self.alpha_ffn.sigmoid() * RESIDUAL_ALPHA_MAX * ffn_out
    }
}

fn truncated_normal_init(in_features: i64, out_features: i64) -> Init {
    let denoms = (in_features + out_features) as f64 / 2.0;
    let std = (1.0 / denoms).sqrt() / 0.8796;
    Init::Randn {
        mean: 0.0,
        stdev: std,
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

const BASE_MODEL_DIM: i64 = 128;
const BASE_FF_DIM: i64 = 512;
const BASE_GQA_LAYERS: usize = 2;
const ABLATION_SMALL_MODEL_DIM: i64 = 64;
const ABLATION_SMALL_FF_DIM: i64 = 256;
const ABLATION_SMALL_GQA_LAYERS: usize = 1;
pub(crate) const SDE_LATENT_DIM: i64 = 128;
pub(crate) const GSDE_EPS: f64 = 1e-6;
pub(crate) const GSDE_LOG_STD_INIT: f64 = -2.0;
pub(crate) const ACTION_DIM: i64 = TICKERS_COUNT + 1;
const TIME_CROSS_LAYERS: usize = 1;
const RESIDUAL_ALPHA_MAX: f64 = 0.5;
const RESIDUAL_ALPHA_INIT: f64 = -2.0;
const ACTOR_MLP_LAYERS: usize = 1;
pub(crate) const GSDE_LEARN_FEATURES: bool = true;
const CRITIC_LAYERS: usize = 2;
const CROSS_NUM_Q_HEADS: i64 = 4;
const CROSS_NUM_KV_HEADS: i64 = 2;
const NUM_EXO_TOKENS: i64 = STATIC_OBSERVATIONS as i64;
const PATCH_SCALAR_FEATS: i64 = 3;
pub(crate) const NUM_VALUE_BUCKETS: i64 = 255;
// symexp(x,s) ≈ s*x near zero → linear spacing ~s*(range/N) ≈ 0.001 per bin
// full range: ±s*(e^range - 1) ≈ ±2.67
const SYMLOG_SCALE: f64 = 0.03;
const SYMLOG_LOG_RANGE: f64 = 4.5;

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

/// expln: numerically stable exp alternative that bounds growth for positive inputs
/// expln(x) = exp(x)        if x <= 0
///          = ln(x + 1) + 1 if x > 0
pub(crate) fn expln(t: &Tensor) -> Tensor {
    let below = t.le(0.0).to_kind(Kind::Float);
    let above = t.gt(0.0).to_kind(Kind::Float);
    let below_threshold = t.exp() * &below;
    let safe_t = t * &above + GSDE_EPS;
    let above_threshold = (safe_t.log1p() + 1.0) * &above;
    below_threshold + above_threshold
}

pub(crate) fn symlog_tensor(x: &Tensor, s: f64) -> Tensor {
    x.sign() * (x.abs() / s).log1p()
}

pub(crate) fn symexp_tensor(x: &Tensor, s: f64) -> Tensor {
    x.sign() * s * x.abs().expm1()
}

/// Two-hot CE loss with searchsorted bin finding.
/// targets: [batch] raw return values, log_probs: [batch, NUM_VALUE_BUCKETS],
/// centers: [NUM_VALUE_BUCKETS] monotonically increasing bin centers in raw space.
pub(crate) fn twohot_ce_loss(targets: &Tensor, log_probs: &Tensor, centers: &Tensor) -> Tensor {
    let n = centers.size()[0];
    let targets_f = targets.to_kind(log_probs.kind());
    let centers_f = centers.to_kind(log_probs.kind());

    let above = targets_f
        .f_searchsorted(&centers_f, false, false, "right", None::<&Tensor>)
        .unwrap()
        .clamp(1, n - 1);
    let below = (&above - 1).clamp(0, n - 1);

    let bin_below = centers_f.index_select(0, &below.flatten(0, -1)).reshape_as(&targets_f);
    let bin_above = centers_f.index_select(0, &above.flatten(0, -1)).reshape_as(&targets_f);

    let dist_below = (&bin_below - &targets_f).abs() + 1e-8;
    let dist_above = (&bin_above - &targets_f).abs() + 1e-8;
    let total = &dist_below + &dist_above;
    let w_below = &dist_above / &total;
    let w_above = &dist_below / &total;

    let lp_below = log_probs.gather(1, &below.unsqueeze(1), false).squeeze_dim(1);
    let lp_above = log_probs.gather(1, &above.unsqueeze(1), false).squeeze_dim(1);

    -(w_below * lp_below + w_above * lp_above)
}

/// (values, critic_logits, critic_input, (action_mean, actor_latent))
/// critic_logits: [batch, NUM_VALUE_BUCKETS] distributional value logits
/// critic_input: [batch, TICKERS_COUNT * MODEL_DIM] pre-MLP input for debugging/probing
/// actor_latent: [batch, SDE_LATENT_DIM] shared actor/gSDE latent
pub type ModelOutput = (Tensor, Tensor, Tensor, (Tensor, Tensor));

/// EMA copy of critic weights (not in the optimizer's VarStore).
/// Returns logits [batch, NUM_VALUE_BUCKETS] (distributional two-hot critic).
pub struct SlowCritic {
    mlp_weights: Vec<Tensor>,  // [CRITIC_LAYERS] linear weights
    mlp_biases: Vec<Tensor>,   // [CRITIC_LAYERS] linear biases
    norm_weights: Vec<Tensor>, // [CRITIC_LAYERS] RMSNorm weights
    out_weight: Tensor,
    out_bias: Tensor,
    rate: f64,
}

impl SlowCritic {
    /// Initialize from the live critic parameters (deep copy, detached)
    pub fn new(model: &TradingModel, rate: f64) -> Self {
        let mlp_weights: Vec<Tensor> = model
            .value_mlp_linears
            .iter()
            .map(|l| l.ws.detach().copy())
            .collect();
        let mlp_biases: Vec<Tensor> = model
            .value_mlp_linears
            .iter()
            .map(|l| l.bs.as_ref().unwrap().detach().copy())
            .collect();
        let norm_weights: Vec<Tensor> = model
            .value_mlp_norms
            .iter()
            .map(|n| n.weight().detach().copy())
            .collect();
        let out_weight = model.value_out.ws.detach().copy();
        let out_bias = model.value_out.bs.as_ref().unwrap().detach().copy();
        Self {
            mlp_weights,
            mlp_biases,
            norm_weights,
            out_weight,
            out_bias,
            rate,
        }
    }

    /// EMA update: slow = rate * live + (1-rate) * slow
    pub fn update(&mut self, model: &TradingModel) {
        let r = self.rate;
        let blend = |slow: &mut Tensor, live: &Tensor| {
            tch::no_grad(|| {
                let _ = slow.g_mul_scalar_(1.0 - r).g_add_(&(live.detach() * r));
            });
        };
        for (i, l) in model.value_mlp_linears.iter().enumerate() {
            blend(&mut self.mlp_weights[i], &l.ws);
            blend(&mut self.mlp_biases[i], l.bs.as_ref().unwrap());
        }
        for (i, n) in model.value_mlp_norms.iter().enumerate() {
            blend(&mut self.norm_weights[i], &n.weight());
        }
        blend(&mut self.out_weight, &model.value_out.ws);
        blend(&mut self.out_bias, model.value_out.bs.as_ref().unwrap());
    }

    /// Forward pass through slow critic (no grad). Returns scalar values [batch, 1].
    pub fn forward(&self, critic_input: &Tensor) -> Tensor {
        tch::no_grad(|| {
            let mut x = critic_input.shallow_clone();
            for i in 0..self.mlp_weights.len() {
                x = x.linear(&self.mlp_weights[i], Some(&self.mlp_biases[i]));
                // RMSNorm: x * weight / rms(x)
                let variance = x.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float);
                let rms = (variance + 1e-6).sqrt();
                x = (&x / &rms) * &self.norm_weights[i];
                x = x.silu();
            }
            x.linear(&self.out_weight, Some(&self.out_bias))
        })
    }
}

pub struct DebugMetrics {
    pub time_alpha_attn_mean: f64,
    pub time_alpha_mlp_mean: f64,
    pub cross_alpha_attn_mean: f64,
    pub cross_alpha_mlp_mean: f64,
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
    patch_pos_embed: Tensor,
    gqa_layers: Vec<GqaBlock>,
    final_norm: RMSNorm,
    exo_feat_w: Tensor,
    exo_feat_b: Tensor,
    exo_cross_blocks: Vec<ExoCrossBlock>,
    cls_token: Tensor,
    inter_ticker_block: InterTickerBlock,
    actor_mlp_linears: Vec<nn::Linear>,
    actor_mlp_norms: Vec<RMSNorm>,
    actor_proj: nn::Linear,
    actor_out: nn::Linear,
    value_mlp_linears: Vec<nn::Linear>,
    value_mlp_norms: Vec<RMSNorm>,
    value_out: nn::Linear,
    log_std_param: Tensor, // [SDE_LATENT_DIM, ACTION_DIM - 1]
    bucket_centers: Tensor, // [NUM_VALUE_BUCKETS] sorted ascending
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

    /// gSDE std matrix used to scale exploration noise matrices.
    pub fn sde_std(&self) -> Tensor {
        expln(&self.log_std_param)
    }

    pub fn bucket_centers(&self) -> &Tensor {
        &self.bucket_centers
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
        let patch_embed_weight = p.var(
            "patch_embed_weight",
            &[num_configs, max_input_dim, spec.model_dim],
            Init::Uniform {
                lo: -0.02,
                up: 0.02,
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
        let patch_pos_embed = p.var(
            "patch_pos_embed",
            &[seq_len, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );

        let gqa_layers = (0..gqa_layers_count)
            .map(|i| GqaBlock::new(&(p / format!("gqa_{}", i)), spec.model_dim, spec.ff_dim))
            .collect::<Vec<_>>();
        let final_norm = RMSNorm::new(&(p / "final_norm"), spec.model_dim, 1e-6);

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
        let cls_token = p.var("cls_token", &[1, 1, spec.model_dim], Init::Randn { mean: 0.0, stdev: 0.02 });
        let inter_ticker_block =
            InterTickerBlock::new(&(p / "inter_ticker_0"), spec.model_dim, spec.ff_dim);
        // Actor latent MLP per-ticker, then flatten → project to SDE_LATENT_DIM
        let actor_mlp_linears = (0..ACTOR_MLP_LAYERS)
            .map(|i| {
                nn::linear(
                    p / format!("actor_mlp_{}", i),
                    spec.model_dim,
                    spec.model_dim,
                    nn::LinearConfig {
                        ws_init: truncated_normal_init(spec.model_dim, spec.model_dim),
                        bs_init: Some(Init::Const(0.0)),
                        bias: true,
                    },
                )
            })
            .collect::<Vec<_>>();
        let actor_mlp_norms = (0..ACTOR_MLP_LAYERS)
            .map(|i| RMSNorm::new(&(p / format!("actor_mlp_norm_{}", i)), spec.model_dim, 1e-6))
            .collect::<Vec<_>>();
        let actor_proj = nn::linear(
            p / "actor_proj",
            TICKERS_COUNT * spec.model_dim,
            SDE_LATENT_DIM,
            nn::LinearConfig {
                ws_init: truncated_normal_init(TICKERS_COUNT * spec.model_dim, SDE_LATENT_DIM),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        let actor_out = nn::linear(
            p / "actor_out",
            SDE_LATENT_DIM,
            ACTION_DIM,
            nn::LinearConfig {
                ws_init: Init::Orthogonal { gain: 0.01 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        // Critic MLP: (Linear → RMSNorm → SiLU) → Linear(→scalar value), orthogonally initialized.
        let critic_in = TICKERS_COUNT * spec.model_dim;
        let critic_hidden = spec.ff_dim;
        let value_mlp_linears = (0..CRITIC_LAYERS)
            .map(|i| {
                let in_dim = if i == 0 { critic_in } else { critic_hidden };
                nn::linear(
                    p / format!("value_mlp_{}", i),
                    in_dim,
                    critic_hidden,
                    nn::LinearConfig {
                        ws_init: Init::Orthogonal { gain: 1.0 },
                        bs_init: Some(Init::Const(0.0)),
                        bias: true,
                    },
                )
            })
            .collect::<Vec<_>>();
        let value_mlp_norms = (0..CRITIC_LAYERS)
            .map(|i| RMSNorm::new(&(p / format!("value_mlp_norm_{}", i)), critic_hidden, 1e-6))
            .collect::<Vec<_>>();
        let value_out = nn::linear(
            p / "value_out",
            critic_hidden,
            NUM_VALUE_BUCKETS,
            nn::LinearConfig {
                ws_init: Init::Orthogonal { gain: 1.0 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        // gSDE log-std: [SDE_LATENT_DIM, ACTION_DIM - 1]
        // K-1 noise dims (last logit is gauge reference with zero noise)
        let log_std_param = p.var(
            "log_std",
            &[SDE_LATENT_DIM, ACTION_DIM - 1],
            Init::Const(GSDE_LOG_STD_INIT),
        );
        let half_n = (NUM_VALUE_BUCKETS - 1) / 2 + 1; // 128
        let half_log = Tensor::linspace(-SYMLOG_LOG_RANGE, 0.0, half_n, (Kind::Float, p.device()));
        let half = symexp_tensor(&half_log, SYMLOG_SCALE);
        let pos_half = half.narrow(0, 0, half_n - 1).flip([0]).neg();
        let bucket_centers = Tensor::cat(&[half, pos_half], 0); // [255], sorted ascending
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
            patch_pos_embed,
            gqa_layers,
            final_norm,
            exo_feat_w,
            exo_feat_b,
            exo_cross_blocks,
            cls_token,
            inter_ticker_block,
            actor_mlp_linears,
            actor_mlp_norms,
            actor_proj,
            actor_out,
            value_mlp_linears,
            value_mlp_norms,
            value_out,
            log_std_param,
            bucket_centers,
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
        let pos_embed = if self.patch_pos_embed.kind() == kind {
            self.patch_pos_embed.shallow_clone()
        } else {
            self.patch_pos_embed.to_kind(kind)
        };
        let patch_tokens = self.patch_embed(&deltas) + pos_embed;
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
