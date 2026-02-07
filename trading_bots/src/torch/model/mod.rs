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

pub(crate) const NUM_VALUE_BUCKETS: i64 = 255;
use crate::torch::ssm_ref::{
    stateful_mamba_block_cfg, Mamba2Config, Mamba2State, StatefulMambaRef,
};

use rmsnorm::RMSNorm;

struct InterTickerBlock {
    ticker_ln: RMSNorm,
    ticker_q: nn::Linear,
    ticker_k: nn::Linear,
    ticker_v: nn::Linear,
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
        let ticker_q = nn::linear(p / "ticker_q", model_dim, model_dim, Default::default());
        let ticker_k = nn::linear(p / "ticker_k", model_dim, model_dim, Default::default());
        let ticker_v = nn::linear(p / "ticker_v", model_dim, model_dim, Default::default());
        let ticker_out = nn::linear(p / "ticker_out", model_dim, model_dim, Default::default());
        let q_norm = RMSNorm::new(&(p / "ticker_q_norm"), model_dim, 1e-6);
        let k_norm = RMSNorm::new(&(p / "ticker_k_norm"), model_dim, 1e-6);
        let mlp_fc1 = nn::linear(p / "mlp_fc1", model_dim, 2 * ff_dim, Default::default());
        let mlp_fc2 = nn::linear(p / "mlp_fc2", ff_dim, model_dim, Default::default());
        let mlp_ln = RMSNorm::new(&(p / "mlp_ln"), model_dim, 1e-6);
        let alpha_ticker_attn = p.var(
            "alpha_ticker_attn_raw",
            &[1],
            Init::Const(RESIDUAL_ALPHA_INIT),
        );
        let alpha_mlp = p.var("alpha_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            ticker_ln,
            ticker_q,
            ticker_k,
            ticker_v,
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
const ATTN_LOGIT_CAP: f64 = 30.0;

struct ExoCrossBlock {
    cross_ln: RMSNorm,
    cross_q: nn::Linear,
    cross_k: nn::Linear,
    cross_v: nn::Linear,
    cross_out: nn::Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    alpha_cross: Tensor,
}

impl ExoCrossBlock {
    fn new(p: &nn::Path, model_dim: i64, cross_head_dim: i64) -> Self {
        let cross_ln = RMSNorm::new(&(p / "cross_ln"), model_dim, 1e-6);
        let cross_q = nn::linear(p / "cross_q", model_dim, model_dim, Default::default());
        let cross_k = nn::linear(
            p / "cross_k",
            model_dim,
            CROSS_NUM_KV_HEADS * cross_head_dim,
            Default::default(),
        );
        let cross_v = nn::linear(
            p / "cross_v",
            model_dim,
            CROSS_NUM_KV_HEADS * cross_head_dim,
            Default::default(),
        );
        let cross_out = nn::linear(p / "cross_out", model_dim, model_dim, Default::default());
        let q_norm = RMSNorm::new(&(p / "cross_q_norm"), cross_head_dim, 1e-6);
        let k_norm = RMSNorm::new(&(p / "cross_k_norm"), cross_head_dim, 1e-6);
        let alpha_cross = p.var("alpha_cross", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            cross_ln,
            cross_q,
            cross_k,
            cross_v,
            cross_out,
            q_norm,
            k_norm,
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
        let k = exo_kv.apply(&self.cross_k);
        let v = exo_kv.apply(&self.cross_v);

        let q = q
            .reshape([b, s, CROSS_NUM_Q_HEADS, cross_head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .reshape([b, t, CROSS_NUM_KV_HEADS, cross_head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([b, t, CROSS_NUM_KV_HEADS, cross_head_dim])
            .permute([0, 2, 1, 3]);

        // QKNorm: apply RMSNorm per-head on head_dim (last dimension)
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let heads_per_group = CROSS_NUM_Q_HEADS / CROSS_NUM_KV_HEADS;
        let k = k
            .unsqueeze(2)
            .expand(
                [b, CROSS_NUM_KV_HEADS, heads_per_group, t, cross_head_dim],
                false,
            )
            .reshape([b, CROSS_NUM_Q_HEADS, t, cross_head_dim]);
        let v = v
            .unsqueeze(2)
            .expand(
                [b, CROSS_NUM_KV_HEADS, heads_per_group, t, cross_head_dim],
                false,
            )
            .reshape([b, CROSS_NUM_Q_HEADS, t, cross_head_dim]);

        let scale = (cross_head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)) / scale;
        let scores = ATTN_LOGIT_CAP * (scores / ATTN_LOGIT_CAP).tanh();
        let attn = scores.softmax(-1, Kind::Float).to_kind(q.kind());
        let out = attn.matmul(&v);

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

fn truncated_normal_init(in_features: i64, out_features: i64) -> Init {
    let denoms = (in_features + out_features) as f64 / 2.0;
    let std = (1.0 / denoms).sqrt() / 0.8796;
    Init::Randn {
        mean: 0.0,
        stdev: std,
    }
}

const BASE_SSM_DIM: i64 = 128;
const BASE_MODEL_DIM: i64 = 128;
const BASE_FF_DIM: i64 = 512;
const BASE_SSM_LAYERS: usize = 2;
const ABLATION_SMALL_SSM_DIM: i64 = 64;
const ABLATION_SMALL_MODEL_DIM: i64 = 64;
const ABLATION_SMALL_FF_DIM: i64 = 256;
const ABLATION_SMALL_SSM_LAYERS: usize = 1;
pub(crate) const SDE_LATENT_DIM: i64 = 128;
pub(crate) const SDE_EPS: f64 = 1e-6;
pub(crate) const LATTICE_ALPHA: f64 = 1.0;
pub(crate) const LATTICE_STD_REG: f64 = 0.01;
pub(crate) const LATTICE_MIN_STD: f64 = 1e-3;
pub(crate) const LATTICE_MAX_STD: f64 = 1.0;
const LOG_STD_INIT: f64 = 0.0;
pub(crate) const ACTION_DIM: i64 = TICKERS_COUNT + 1;
const SYMLOG_SCALE: f64 = 0.1;
const SYMLOG_LOG_RANGE: f64 = 5.0;
const TIME_CROSS_LAYERS: usize = 1;
const RESIDUAL_ALPHA_MAX: f64 = 0.5;
const RESIDUAL_ALPHA_INIT: f64 = -4.0;
const ACTOR_MLP_LAYERS: usize = 1;
pub(crate) const LATTICE_LEARN_FEATURES: bool = false;
const CRITIC_LAYERS: usize = 2;
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
    ssm_dim: i64,
    model_dim: i64,
    ff_dim: i64,
    ssm_layers: usize,
    patch_configs: &'static [(i64, i64)],
}

fn model_spec(variant: ModelVariant) -> ModelSpec {
    match variant {
        ModelVariant::Base => ModelSpec {
            ssm_dim: BASE_SSM_DIM,
            model_dim: BASE_MODEL_DIM,
            ff_dim: BASE_FF_DIM,
            ssm_layers: BASE_SSM_LAYERS,
            patch_configs: BASE_PATCH_CONFIGS,
        },
        ModelVariant::AblationSmall => ModelSpec {
            ssm_dim: ABLATION_SMALL_SSM_DIM,
            model_dim: ABLATION_SMALL_MODEL_DIM,
            ff_dim: ABLATION_SMALL_FF_DIM,
            ssm_layers: ABLATION_SMALL_SSM_LAYERS,
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
    let safe_t = t * &above + 1e-8;
    let above_threshold = (safe_t.log1p() + 1.0) * &above;
    below_threshold + above_threshold
}

/// (values, critic_logits, critic_input, (action_mean, actor_latent))
/// critic_input: [batch, TICKERS_COUNT * MODEL_DIM] pre-MLP input for slow target
/// actor_latent: [batch, SDE_LATENT_DIM] shared actor/Lattice latent
pub type ModelOutput = (Tensor, Tensor, Tensor, (Tensor, Tensor));

/// DreamerV3-style slow target for value function stabilization.
/// Holds an EMA copy of critic weights (not in the optimizer's VarStore).
/// Forward pass produces target logits; regularization loss pulls live critic toward these.
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

    /// Forward pass through slow critic (no grad). Returns logits [batch, NUM_VALUE_BUCKETS].
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

pub(crate) fn symlog_tensor(x: &Tensor, s: f64) -> Tensor {
    x.sign() * (x.abs() / s + 1.0).log()
}

pub(crate) fn symexp_tensor(x: &Tensor, s: f64) -> Tensor {
    x.sign() * s * (x.abs().exp() - 1.0)
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
    pub ssm_layers_override: Option<usize>,
}

impl Default for TradingModelConfig {
    fn default() -> Self {
        Self {
            variant: ModelVariant::Base,
            ssm_layers_override: None,
        }
    }
}

/// Streaming state for O(1) inference per step
/// - Ring buffer holds full delta history for head computation
/// - Patch buffer accumulates deltas until full patch ready
/// - SSM state carries compressed history (only process new token each patch)
pub struct StreamState {
    /// Ring buffer: [TICKERS_COUNT, PRICE_DELTAS_PER_TICKER]
    pub delta_ring: Tensor,
    /// Write position in ring buffer
    pub ring_pos: i64,
    /// Patch accumulator: [TICKERS_COUNT, FINEST_PATCH_SIZE]
    pub patch_buf: Tensor,
    /// Position within current patch
    pub patch_pos: i64,
    /// SSM hidden state per layer (batched over tickers)
    pub ssm_states: Vec<Mamba2State>,
    /// Whether initialized with full sequence
    pub initialized: bool,
}

pub struct TradingModel {
    variant: ModelVariant,
    patch_configs: &'static [(i64, i64)],
    seq_len: i64,
    finest_patch_size: i64,
    finest_patch_index: usize,
    ssm_dim: i64,
    model_dim: i64,
    ff_dim: i64,
    cross_head_dim: i64,
    critic_hidden_dim: i64,
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    patch_dt_scale: Tensor,
    patch_sizes: Tensor,
    patch_config_ids: Tensor,
    patch_pos_embed: Tensor,
    ssm_layers: Vec<StatefulMambaRef>,
    ssm_norms: Vec<RMSNorm>,
    ssm_final_norm: RMSNorm,
    exo_feat_w: Tensor,
    exo_feat_b: Tensor,
    exo_cross_blocks: Vec<ExoCrossBlock>,
    inter_ticker_block: InterTickerBlock,
    actor_mlp_linears: Vec<nn::Linear>,
    actor_mlp_norms: Vec<RMSNorm>,
    actor_proj: nn::Linear,
    actor_out: nn::Linear,
    value_mlp_linears: Vec<nn::Linear>,
    value_mlp_norms: Vec<RMSNorm>,
    value_out: nn::Linear,
    // Lattice exploration: learned log-std for correlated + independent noise
    log_std_param: Tensor, // [SDE_LATENT_DIM, SDE_LATENT_DIM + ACTION_DIM]
    bucket_centers: Tensor,
    value_centers: Tensor,
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

    fn maybe_to_device_kind(&self, input: &Tensor, device: tch::Device, kind: Kind) -> Tensor {
        let input = self.maybe_to_device(input, device);
        if input.kind() == kind {
            input
        } else {
            input.to_kind(kind)
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

    pub fn value_centers(&self) -> &Tensor {
        &self.value_centers
    }

    /// Lattice stds: corr_std [SDE_LATENT_DIM, SDE_LATENT_DIM], ind_std [SDE_LATENT_DIM, ACTION_DIM]
    /// With dimension correction and clipping per the paper
    pub fn lattice_stds(&self) -> (Tensor, Tensor) {
        let log_std = self
            .log_std_param
            .clamp(LATTICE_MIN_STD.ln(), LATTICE_MAX_STD.ln());
        // Dimension correction: prevents variance from scaling with latent_dim
        let log_std = &log_std - 0.5 * (SDE_LATENT_DIM as f64).ln();
        let std = expln(&log_std);
        let corr_std = std.narrow(1, 0, SDE_LATENT_DIM);
        let ind_std = std.narrow(1, SDE_LATENT_DIM, ACTION_DIM);
        (corr_std, ind_std)
    }

    /// W_policy: actor_out weights [ACTION_DIM, SDE_LATENT_DIM] for Lattice covariance
    pub fn w_policy(&self) -> Tensor {
        self.actor_out.ws.shallow_clone()
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
        let ssm_layers_count = config.ssm_layers_override.unwrap_or(spec.ssm_layers);
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
            &[num_configs, max_input_dim, spec.ssm_dim],
            Init::Uniform {
                lo: -0.02,
                up: 0.02,
            },
        );
        let patch_embed_bias = p.var(
            "patch_embed_bias",
            &[num_configs, spec.ssm_dim],
            Init::Const(0.0),
        );
        let patch_dt_scale = {
            let mut scales = Vec::with_capacity(seq_len as usize);
            for &(days, patch_size) in patch_configs {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    scales.push(patch_size as f32);
                }
            }
            Tensor::from_slice(&scales)
                .view([1, seq_len, 1])
                .to_device(p.device())
        };
        let patch_sizes = {
            let mut sizes = Vec::with_capacity(seq_len as usize);
            for &(days, patch_size) in patch_configs {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    sizes.push(patch_size as f32);
                }
            }
            Tensor::from_slice(&sizes)
                .view([1, seq_len, 1])
                .to_device(p.device())
        };
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
            &[seq_len, spec.ssm_dim],
            Init::Const(0.0),
        );

        let ssm_cfg = Mamba2Config {
            d_model: spec.ssm_dim,
            d_ssm: Some(spec.ssm_dim),
            ..Mamba2Config::default()
        };
        let ssm_layers = (0..ssm_layers_count)
            .map(|i| stateful_mamba_block_cfg(&(p / format!("ssm_{}", i)), ssm_cfg.clone()))
            .collect::<Vec<_>>();
        let ssm_norms = (0..ssm_layers_count)
            .map(|i| RMSNorm::new(&(p / format!("ssm_norm_{}", i)), spec.ssm_dim, 1e-6))
            .collect::<Vec<_>>();
        let ssm_final_norm = RMSNorm::new(&(p / "ssm_final_norm"), spec.ssm_dim, 1e-6);

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
                ws_init: truncated_normal_init(SDE_LATENT_DIM, ACTION_DIM),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        // DreamerV3-style critic MLP: (Linear → RMSNorm → SiLU) → Linear(→bins, zero_init)
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
                        ws_init: truncated_normal_init(in_dim, critic_hidden),
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
                ws_init: Init::Const(0.0),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        // Lattice log-std: [SDE_LATENT_DIM, SDE_LATENT_DIM + ACTION_DIM]
        // First SDE_LATENT_DIM columns = corr_std, last ACTION_DIM columns = ind_std
        let log_std_param = p.var(
            "log_std",
            &[SDE_LATENT_DIM, SDE_LATENT_DIM + ACTION_DIM],
            Init::Const(LOG_STD_INIT),
        );
        // Scaled symlog bins in raw return space (s=0.1, log_range=5 → ±14.7)
        // Dense near 0 for typical ±0.3 returns, log-compressed tails to ±14.7
        let half_n = (NUM_VALUE_BUCKETS - 1) / 2 + 1; // 128
        let half_log = Tensor::linspace(-SYMLOG_LOG_RANGE, 0.0, half_n, (Kind::Float, p.device()));
        let half = symexp_tensor(&half_log, SYMLOG_SCALE);
        // Positive: negate and reverse first 127 elements (excluding zero)
        let pos_half = half.narrow(0, 0, half_n - 1).flip([0]).neg();
        let bucket_centers = Tensor::cat(&[half, pos_half], 0);
        let value_centers = bucket_centers.shallow_clone();
        Self {
            variant: config.variant,
            patch_configs,
            seq_len,
            finest_patch_size,
            finest_patch_index,
            ssm_dim: spec.ssm_dim,
            model_dim: spec.model_dim,
            ff_dim: spec.ff_dim,
            cross_head_dim,
            critic_hidden_dim: critic_hidden,
            patch_embed_weight,
            patch_embed_bias,
            patch_dt_scale,
            patch_sizes,
            patch_config_ids,
            patch_pos_embed,
            ssm_layers,
            ssm_norms,
            ssm_final_norm,
            exo_feat_w,
            exo_feat_b,
            exo_cross_blocks,
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
            value_centers,
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

    /// Apply exo cross-block if this SSM layer index is in EXO_CROSS_AFTER
    fn maybe_apply_exo_cross(&self, x: &Tensor, exo_kv: &Tensor, ssm_layer_idx: usize) -> Tensor {
        if let Some(pos) = EXO_CROSS_AFTER.iter().position(|&i| i == ssm_layer_idx) {
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

    fn patch_embed_single(&self, ticker_data: &Tensor) -> Tensor {
        let deltas = ticker_data.to_device(self.device);
        self.patch_embed(&deltas)
    }

    fn normalize_seq_idx(&self, seq_idx: &Tensor, batch_size: i64) -> Tensor {
        if seq_idx.numel() == 0 {
            return seq_idx.shallow_clone();
        }
        let seq_idx = self.maybe_to_device(seq_idx, self.device);
        self.normalize_seq_idx_on_device(&seq_idx, batch_size)
    }

    fn normalize_seq_idx_on_device(&self, seq_idx: &Tensor, batch_size: i64) -> Tensor {
        if seq_idx.numel() == 0 {
            return seq_idx.shallow_clone();
        }
        let sizes = seq_idx.size();
        if sizes.len() != 2 {
            panic!("seq_idx must be 2D");
        }
        if sizes[0] == batch_size && sizes[1] == TICKERS_COUNT * self.seq_len {
            seq_idx.view([batch_size * TICKERS_COUNT, self.seq_len])
        } else if sizes[0] == batch_size * TICKERS_COUNT && sizes[1] == self.seq_len {
            seq_idx.shallow_clone()
        } else {
            panic!("unexpected seq_idx shape {:?}", sizes);
        }
    }

    fn patch_latent_stem(
        &self,
        price_deltas: &Tensor,
        batch_size: i64,
        seq_idx: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor) {
        let price_deltas = self.maybe_to_device(price_deltas, self.device);
        let seq_idx = seq_idx.map(|seq_idx| self.maybe_to_device(seq_idx, self.device));
        self.patch_latent_stem_on_device(&price_deltas, batch_size, seq_idx.as_ref())
    }

    fn patch_latent_stem_on_device(
        &self,
        price_deltas: &Tensor,
        batch_size: i64,
        seq_idx: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor) {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let kind = deltas.kind();
        let pos_embed = self.maybe_to_device_kind(&self.patch_pos_embed, deltas.device(), kind);
        let x = self.patch_embed(&deltas) + pos_embed;
        let seq_idx = match seq_idx {
            Some(seq_idx) => self.normalize_seq_idx_on_device(seq_idx, batch_size),
            None => Self::build_seq_idx_from_padding(&deltas, &self.patch_sizes),
        };
        // Cast to int32 here so Python bridge doesn't need to cast per forward call.
        // causal_conv1d CUDA kernel requires int32 seq_idx.
        let seq_idx = if seq_idx.numel() > 0 {
            seq_idx.to_kind(Kind::Int)
        } else {
            seq_idx
        };
        let dt_scale = self
            .maybe_to_device_kind(&self.patch_dt_scale, deltas.device(), kind)
            .clamp_min(1e-4);
        (x, dt_scale, seq_idx)
    }

    fn build_seq_idx_from_padding(deltas: &Tensor, patch_sizes: &Tensor) -> Tensor {
        let device = deltas.device();
        let zeros = deltas.eq(0.0).to_kind(Kind::Float);
        let prefix = zeros.cumprod(1, Kind::Float);
        let leading = prefix
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
            .to_kind(Kind::Int64);
        let max_leading = leading.max().int64_value(&[]);
        if max_leading == 0 {
            return Tensor::zeros(&[0], (Kind::Int64, device));
        }
        let patch_sizes = if patch_sizes.device() == device {
            patch_sizes.shallow_clone()
        } else {
            patch_sizes.to_device(device)
        };
        let patch_ends = patch_sizes.to_kind(Kind::Int64).cumsum(1, Kind::Int64);
        let patch_ends = patch_ends.squeeze_dim(0).squeeze_dim(-1);
        let mask = leading.unsqueeze(1).ge_tensor(&patch_ends);
        mask.to_kind(Kind::Int64).neg()
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
        let enriched = Tensor::cat(&enriched_parts.iter().collect::<Vec<_>>(), 1);

        // Phase 2: fused projection — single einsum over all 256 tokens
        let config_ids = self.maybe_to_device(&self.patch_config_ids, device);
        let weight = self.maybe_to_device_kind(&self.patch_embed_weight, device, Kind::Float);
        let bias = self.maybe_to_device_kind(&self.patch_embed_bias, device, Kind::Float);
        let weight_per_patch = weight.index_select(0, &config_ids);
        let bias_per_patch = bias.index_select(0, &config_ids);
        let out = Tensor::einsum(
            "blm,lmd->bld",
            &[&enriched, &weight_per_patch],
            None::<&[i64]>,
        );
        let out = out + bias_per_patch.unsqueeze(0);
        out.to_kind(kind)
    }

    /// Single-config embedding for streaming inference (one patch at a time).
    fn embed_patch_config(&self, patches: &Tensor, config_idx: i64) -> Tensor {
        let kind = patches.kind();
        let patches_f = patches.to_kind(Kind::Float);
        let patch_len = patches_f.size()[2];
        let mean = patches_f.mean_dim([2].as_slice(), true, Kind::Float);
        let var =
            (&patches_f - &mean)
                .pow_tensor_scalar(2.0)
                .mean_dim([2].as_slice(), true, Kind::Float);
        let std = (var + 1e-5).sqrt();
        let first = patches_f.narrow(2, 0, 1);
        let last = patches_f.narrow(2, patch_len - 1, 1);
        let slope = &last - &first;
        let enriched = Tensor::cat(&[&patches_f, &mean, &std, &slope], 2);
        let input_dim = patch_len + PATCH_SCALAR_FEATS;
        let weight = self
            .patch_embed_weight
            .get(config_idx)
            .narrow(0, 0, input_dim);
        let bias = self.patch_embed_bias.get(config_idx);
        let out = enriched.matmul(&weight) + bias;
        out.to_kind(kind)
    }
}
