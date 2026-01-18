mod forward;
mod head;
mod inference;
mod rmsnorm;

use std::sync::OnceLock;

use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{
    GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT,
};
use crate::torch::ssm::{stateful_mamba_block_cfg, Mamba2Config, Mamba2State, StatefulMamba};

pub use shared::constants::GLOBAL_MACRO_OBS;

use rmsnorm::RMSNorm;

struct InterTickerBlock {
    ticker_ln: RMSNorm,
    ticker_rp_q: nn::Linear,
    ticker_rp_v: Tensor,
    ticker_rp_k_frozen: Tensor,
    ticker_latent_q: nn::Linear,
    ticker_latent_k: Tensor,
    ticker_latent_v: Tensor,
    ticker_out: nn::Linear,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    mlp_ln: RMSNorm,
    alpha_ticker_rp: Tensor,
    alpha_ticker_attn: Tensor,
    alpha_mlp: Tensor,
}

impl InterTickerBlock {
    fn new(p: &nn::Path, kv_heads: i64, head_dim: i64, ticker_latents: i64) -> Self {
        let _ = (kv_heads, head_dim);
        let ticker_ln = RMSNorm::new(&(p / "ticker_ln"), MODEL_DIM, 1e-6);
        let ticker_rp_q = nn::linear(p / "ticker_rp_q", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_rp_v = p.var(
            "ticker_rp_v",
            &[ticker_latents, MODEL_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (MODEL_DIM as f64).sqrt(),
            },
        );
        let ticker_rp_k_frozen = Tensor::randn(
            &[ticker_latents, MODEL_DIM],
            (Kind::Float, p.device()),
        ) * (1.0 / (MODEL_DIM as f64).sqrt());
        ticker_rp_k_frozen.set_requires_grad(false);
        let ticker_latent_q =
            nn::linear(p / "ticker_latent_q", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_latent_k = p.var(
            "ticker_latent_k",
            &[ticker_latents, MODEL_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (MODEL_DIM as f64).sqrt(),
            },
        );
        let ticker_latent_v = p.var(
            "ticker_latent_v",
            &[ticker_latents, MODEL_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (MODEL_DIM as f64).sqrt(),
            },
        );
        let ticker_out = nn::linear(p / "ticker_out", MODEL_DIM, MODEL_DIM, Default::default());
        let mlp_fc1 = nn::linear(p / "mlp_fc1", MODEL_DIM, 2 * FF_DIM, Default::default());
        let mlp_fc2 = nn::linear(p / "mlp_fc2", FF_DIM, MODEL_DIM, Default::default());
        let mlp_ln = RMSNorm::new(&(p / "mlp_ln"), MODEL_DIM, 1e-6);
        let alpha_ticker_rp = p.var("alpha_ticker_rp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_ticker_attn =
            p.var("alpha_ticker_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_mlp = p.var("alpha_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            ticker_ln,
            ticker_rp_q,
            ticker_rp_v,
            ticker_rp_k_frozen,
            ticker_latent_q,
            ticker_latent_k,
            ticker_latent_v,
            ticker_out,
            mlp_fc1,
            mlp_fc2,
            mlp_ln,
            alpha_ticker_rp,
            alpha_ticker_attn,
            alpha_mlp,
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

const SSM_DIM: i64 = 128;
const MODEL_DIM: i64 = 128;
const LOG_STD_INIT: f64 = -2.0;
const SDE_EPS: f64 = 1e-6;
const USE_SDPA: bool = true;
const SDPA_MIN_LEN: i64 = 64;
const SDPA_MIN_LEN_CROSS: i64 = 1;
const TIME_CROSS_LAYERS: usize = 1;
const FF_DIM: i64 = 512;
const HEAD_HIDDEN: i64 = 192;
const RESIDUAL_ALPHA_MAX: f64 = 0.5;
const RESIDUAL_ALPHA_INIT: f64 = -4.0;
const ROPE_BASE: f64 = 10000.0;
const TICKER_LATENT_FACTORS: i64 = 32;

pub(crate) fn symlog_tensor(t: &Tensor) -> Tensor {
    let abs = t.abs();
    t.sign() * (abs + 1.0).log()
}

pub(crate) fn symexp_tensor(t: &Tensor) -> Tensor {
    let abs = t.abs();
    t.sign() * (abs.exp() - 1.0)
}
const DEFAULT_CASH_POOL_QUERIES: i64 = 4;
const TEMPORAL_POOL_GROUPS: i64 = 4;
const PMA_QUERIES: i64 = 2;
const TEMPORAL_STATIC_DIM: i64 = 64;
const GLOBAL_TOKEN_COUNT: i64 = 4;

const PATCH_CONFIGS: [(i64, i64); 7] = [
    (1600, 64),
    (1024, 32),
    (512, 16),
    (128, 8),
    (64, 4),
    (32, 2),
    (40, 1),
];

const fn compute_patch_totals() -> (i64, i64) {
    let mut total_days = 0i64;
    let mut total_tokens = 0i64;
    let mut i = 0;
    while i < PATCH_CONFIGS.len() {
        let (days, patch_size) = PATCH_CONFIGS[i];
        assert!(days % patch_size == 0, "days must be divisible by patch_size");
        total_days += days;
        total_tokens += days / patch_size;
        i += 1;
    }
    (total_days, total_tokens)
}

const PATCH_TOTALS: (i64, i64) = compute_patch_totals();
const SEQ_LEN: i64 = PATCH_TOTALS.1;
pub(crate) const PATCH_SEQ_LEN: i64 = SEQ_LEN;
const PATCH_EXTRA_FEATS: i64 = 4;
const FINEST_PATCH_SIZE: i64 = 1;
const FINEST_PATCH_INDEX: usize = 6;

const _: () = assert!(
    PATCH_TOTALS.0 == PRICE_DELTAS_PER_TICKER as i64,
    "PATCH_CONFIGS days must equal PRICE_DELTAS_PER_TICKER"
);

static PATCH_ENDS_CPU: OnceLock<Vec<i64>> = OnceLock::new();

pub(crate) fn patch_ends_cpu() -> &'static [i64] {
    PATCH_ENDS_CPU
        .get_or_init(|| {
            let mut ends = Vec::with_capacity(SEQ_LEN as usize);
            let mut total = 0i64;
            for &(days, patch_size) in &PATCH_CONFIGS {
                let num = days / patch_size;
                for _ in 0..num {
                    total += patch_size;
                    ends.push(total);
                }
            }
            ends
        })
        .as_slice()
}

const NUM_VALUE_BUCKETS: i64 = 127;

// (values, critic_logits, (action_mean, action_log_std), attn_entropy)
pub type ModelOutput = (Tensor, Tensor, (Tensor, Tensor), Tensor);

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
    pub cross_ticker_embed_norm: f64,
}

#[derive(Clone, Copy)]
pub struct TradingModelConfig {
    pub ssm_layers: usize,
    pub cash_pool_queries: i64,
}

impl Default for TradingModelConfig {
    fn default() -> Self {
        Self {
            ssm_layers: 2,
            cash_pool_queries: DEFAULT_CASH_POOL_QUERIES,
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
    /// SSM hidden state per ticker per layer
    pub ssm_states: Vec<Mamba2State>,
    /// Whether initialized with full sequence
    pub initialized: bool,
}

pub struct TradingModel {
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    patch_ln_weight: Tensor,
    patch_dt_scale: Tensor,
    stem_pos_embed: Tensor,
    stem_scale_embed: Tensor,
    patch_gather_idx: Tensor,
    patch_mask: Tensor,
    patch_sizes: Tensor,
    patch_last_idx: Tensor,
    patch_config_ids: Tensor,
    ssm_layers: Vec<StatefulMamba>,
    ssm_norms: Vec<RMSNorm>,
    post_ssm_ln: RMSNorm,
    ssm_gate: nn::Linear,
    ssm_proj: nn::Conv1D,
    static_proj: nn::Linear,
    ln_static_proj: RMSNorm,
    static_to_temporal: nn::Linear,
    ln_static_temporal: RMSNorm,
    pma_queries: Tensor,
    pma_q_proj: nn::Linear,
    pma_k_proj: nn::Linear,
    pma_v_proj: nn::Linear,
    pma_out: nn::Linear,
    inter_ticker_block: InterTickerBlock,
    time_pos_proj: nn::Linear,
    time_global_ctx: nn::Linear,
    time_ticker_ctx: nn::Linear,
    static_cross_q: nn::Linear,
    static_cross_k: nn::Linear,
    static_cross_v: nn::Linear,
    static_cross_out: nn::Linear,
    cross_ticker_embed: nn::Linear,
    global_ticker_token: Tensor,
    global_tokens: Tensor,
    global_token_proj: nn::Linear,
    global_token_merge: nn::Linear,
    global_to_ticker: nn::Linear,
    global_inject_down: nn::Linear,
    global_inject_up: nn::Linear,
    global_inject_gate_raw: Tensor,
    head_proj: nn::Linear,
    head_ln: RMSNorm,
    policy_ln: RMSNorm,
    value_ln: RMSNorm,
    value_mlp_fc1: nn::Linear,
    value_mlp_fc2: nn::Linear,
    actor_out: nn::Linear,
    critic_out: nn::Linear,
    cash_log_std_param: Tensor,
    cash_queries: Tensor,
    cash_q_proj: nn::Linear,
    cash_k_proj: nn::Linear,
    cash_v_proj: nn::Linear,
    cash_merge: nn::Linear,
    cash_recent_proj: nn::Linear,
    cash_recent_gate: nn::Linear,
    cash_recent_slope_raw: Tensor,
    cash_attn_temp_raw: Tensor,
    sde_fc: nn::Linear,
    ln_sde: RMSNorm,
    log_std_param: Tensor,
    bucket_centers: Tensor,
    value_centers: Tensor,
    device: tch::Device,
    num_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    cash_pool_queries: i64,
    decay_positions: Tensor,
    decay_ones: Tensor,
}

impl TradingModel {
    fn use_sdpa(&self, seq_len: i64) -> bool {
        if !USE_SDPA {
            return false;
        }
        let _ = seq_len;
        true
    }

    fn use_sdpa_cross(&self, seq_len: i64) -> bool {
        if !USE_SDPA {
            return false;
        }
        let _ = seq_len;
        true
    }

    fn attn_softmax_fp32(&self, q: &Tensor, k: &Tensor) -> Tensor {
        let q_f = q.to_kind(Kind::Float);
        let k_f = k.to_kind(Kind::Float);
        let scores = (q_f.matmul(&k_f.transpose(-2, -1)) / (self.head_dim as f64).sqrt())
            .softmax(-1, Kind::Float);
        scores.to_kind(q.kind())
    }

    fn cast_inputs(&self, input: &Tensor) -> Tensor {
        let target_kind = self.log_std_param.kind();
        if input.kind() == target_kind {
            input.shallow_clone()
        } else {
            input.to_kind(target_kind)
        }
    }

    pub fn value_centers(&self) -> &Tensor {
        &self.value_centers
    }

    pub fn new(p: &nn::Path) -> Self {
        Self::new_with_config(p, TradingModelConfig::default())
    }

    pub fn new_with_config(p: &nn::Path, config: TradingModelConfig) -> Self {
        let num_configs = PATCH_CONFIGS.len() as i64;
        let max_patch_size = PATCH_CONFIGS
            .iter()
            .map(|&(_, patch_size)| patch_size)
            .max()
            .unwrap_or(0);
        let max_input_dim = max_patch_size + PATCH_EXTRA_FEATS;
        let patch_embed_weight = p.var(
            "patch_embed_weight",
            &[num_configs, max_input_dim, SSM_DIM],
            Init::Uniform { lo: -0.02, up: 0.02 },
        );
        let patch_embed_bias = p.var(
            "patch_embed_bias",
            &[num_configs, SSM_DIM],
            Init::Const(0.0),
        );
        let patch_ln_weight = p.var(
            "patch_ln_weight",
            &[num_configs, SSM_DIM],
            Init::Const(1.0),
        );
        let stem_scale_embed = p.var(
            "stem_scale_embed",
            &[num_configs, SSM_DIM],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );
        let stem_pos_embed = Self::build_sin_cos_pos_embed(SEQ_LEN, SSM_DIM, p.device())
            .unsqueeze(0);
        let patch_dt_scale = {
            let mut scales = Vec::with_capacity(SEQ_LEN as usize);
            for &(days, patch_size) in &PATCH_CONFIGS {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    scales.push(patch_size as f32);
                }
            }
            Tensor::from_slice(&scales)
                .view([1, SEQ_LEN, 1])
                .to_device(p.device())
        };
        let (patch_gather_idx, patch_mask, patch_sizes, patch_last_idx, patch_config_ids) = {
            let mut gather_idx = Vec::with_capacity((SEQ_LEN * max_patch_size) as usize);
            let mut mask = Vec::with_capacity((SEQ_LEN * max_patch_size) as usize);
            let mut sizes = Vec::with_capacity(SEQ_LEN as usize);
            let mut last_idx = Vec::with_capacity(SEQ_LEN as usize);
            let mut cfg_ids = Vec::with_capacity(SEQ_LEN as usize);
            let mut offset = 0i64;
            for (cfg_idx, &(days, patch_size)) in PATCH_CONFIGS.iter().enumerate() {
                let n_patches = days / patch_size;
                for p_idx in 0..n_patches {
                    let start = offset + p_idx * patch_size;
                    sizes.push(patch_size as f32);
                    last_idx.push((patch_size - 1) as i64);
                    cfg_ids.push(cfg_idx as i64);
                    for k in 0..max_patch_size {
                        if k < patch_size {
                            gather_idx.push(start + k);
                            mask.push(1.0);
                        } else {
                            gather_idx.push(-1);
                            mask.push(0.0);
                        }
                    }
                }
                offset += days;
            }
            let gather_idx = Tensor::from_slice(&gather_idx)
                .view([SEQ_LEN, max_patch_size])
                .to_kind(Kind::Int64)
                .to_device(p.device());
            let mask = Tensor::from_slice(&mask)
                .view([SEQ_LEN, max_patch_size])
                .to_device(p.device());
            let sizes = Tensor::from_slice(&sizes)
                .view([1, SEQ_LEN, 1])
                .to_device(p.device());
            let last_idx = Tensor::from_slice(&last_idx)
                .view([1, SEQ_LEN, 1])
                .to_kind(Kind::Int64)
                .to_device(p.device());
            let cfg_ids = Tensor::from_slice(&cfg_ids)
                .to_kind(Kind::Int64)
                .to_device(p.device());
            (gather_idx, mask, sizes, last_idx, cfg_ids)
        };

        let ssm_cfg = Mamba2Config {
            d_model: SSM_DIM,
            d_ssm: Some(SSM_DIM),
            ..Mamba2Config::default()
        };
        let ssm_layers = (0..config.ssm_layers)
            .map(|i| stateful_mamba_block_cfg(&(p / format!("ssm_{}", i)), ssm_cfg.clone()))
            .collect::<Vec<_>>();
        let ssm_norms = (0..config.ssm_layers)
            .map(|i| RMSNorm::new(&(p / format!("ssm_norm_{}", i)), SSM_DIM, 1e-6))
            .collect::<Vec<_>>();
        let post_ssm_ln = RMSNorm::new(&(p / "post_ssm_ln"), SSM_DIM, 1e-6);
        let ssm_gate = nn::linear(p / "ssm_gate", SSM_DIM, SSM_DIM, Default::default());
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, MODEL_DIM, 1, Default::default());

        let static_proj = nn::linear(
            p / "static_proj",
            MODEL_DIM + PER_TICKER_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let ln_static_proj = RMSNorm::new(&(p / "ln_static_proj"), MODEL_DIM, 1e-6);
        let static_to_temporal = nn::linear(
            p / "static_to_temporal",
            PER_TICKER_STATIC_OBS as i64,
            TEMPORAL_STATIC_DIM,
            Default::default(),
        );
        let ln_static_temporal =
            RMSNorm::new(&(p / "ln_static_temporal"), TEMPORAL_STATIC_DIM, 1e-6);
        let pma_queries = p.var(
            "pma_queries",
            &[PMA_QUERIES, MODEL_DIM],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );
        let pma_q_proj = nn::linear(p / "pma_q_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let pma_k_proj = nn::linear(
            p / "pma_k_proj",
            MODEL_DIM + TEMPORAL_STATIC_DIM,
            MODEL_DIM,
            Default::default(),
        );
        let pma_v_proj = nn::linear(
            p / "pma_v_proj",
            MODEL_DIM + TEMPORAL_STATIC_DIM,
            MODEL_DIM,
            Default::default(),
        );
        let pma_out = nn::linear(p / "pma_out", MODEL_DIM, MODEL_DIM, Default::default());

        let num_heads = 8i64;
        let kv_heads = 8i64;
        let head_dim = 16i64;
        assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
        assert_eq!(
            num_heads * head_dim,
            MODEL_DIM,
            "num_heads * head_dim must equal MODEL_DIM"
        );
        assert!(config.cash_pool_queries > 0, "cash_pool_queries must be > 0");
        let inter_ticker_block = InterTickerBlock::new(
            &(p / "inter_ticker_0"),
            kv_heads,
            head_dim,
            TICKER_LATENT_FACTORS,
        );
        let time_pos_proj = nn::linear(p / "time_pos_proj", 4, MODEL_DIM, Default::default());
        let time_global_ctx = nn::linear(
            p / "time_global_ctx",
            GLOBAL_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let time_ticker_ctx = nn::linear(
            p / "time_ticker_ctx",
            PER_TICKER_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let static_cross_q = nn::linear(p / "static_cross_q", MODEL_DIM, MODEL_DIM, Default::default());
        let static_cross_k = nn::linear(p / "static_cross_k", MODEL_DIM, MODEL_DIM, Default::default());
        let static_cross_v = nn::linear(p / "static_cross_v", MODEL_DIM, MODEL_DIM, Default::default());
        let static_cross_out = nn::linear(p / "static_cross_out", MODEL_DIM, MODEL_DIM, Default::default());
        let cross_ticker_embed = nn::linear(
            p / "cross_ticker_embed",
            PER_TICKER_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let global_ticker_token = p.var(
            "global_ticker_token",
            &[TICKERS_COUNT as i64, MODEL_DIM],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );
        let global_tokens = p.var(
            "global_tokens",
            &[GLOBAL_TOKEN_COUNT, MODEL_DIM],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );
        let global_token_proj = nn::linear(
            p / "global_token_proj",
            GLOBAL_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let global_token_merge =
            nn::linear(p / "global_token_merge", MODEL_DIM, MODEL_DIM, Default::default());
        let global_to_ticker = nn::linear(
            p / "global_to_ticker",
            GLOBAL_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let global_inject_down = nn::linear(
            p / "global_inject_down",
            GLOBAL_STATIC_OBS as i64,
            TEMPORAL_STATIC_DIM,
            Default::default(),
        );
        let global_inject_up = nn::linear(
            p / "global_inject_up",
            TEMPORAL_STATIC_DIM,
            MODEL_DIM,
            Default::default(),
        );
        let global_inject_gate_raw = p.var("global_inject_gate_raw", &[1], Init::Const(-2.0));
        let head_proj = nn::linear(
            p / "head_proj",
            MODEL_DIM,
            HEAD_HIDDEN,
            nn::LinearConfig {
                ws_init: truncated_normal_init(MODEL_DIM, HEAD_HIDDEN),
                ..Default::default()
            },
        );
        let head_ln = RMSNorm::new(&(p / "head_ln"), HEAD_HIDDEN, 1e-6);
        let policy_ln = RMSNorm::new(&(p / "policy_ln"), MODEL_DIM, 1e-6);
        let value_ln = RMSNorm::new(&(p / "value_ln"), MODEL_DIM, 1e-6);
        let value_mlp_fc1 = nn::linear(p / "value_mlp_fc1", MODEL_DIM, MODEL_DIM, Default::default());
        let value_mlp_fc2 = nn::linear(p / "value_mlp_fc2", MODEL_DIM, MODEL_DIM, Default::default());
        let actor_out = nn::linear(
            p / "actor_out",
            HEAD_HIDDEN,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(HEAD_HIDDEN, 1),
                ..Default::default()
            },
        );
        let cash_log_std_param = p.var("cash_log_std", &[1], Init::Const(0.0));
        let critic_out = nn::linear(
            p / "critic_out",
            MODEL_DIM,
            NUM_VALUE_BUCKETS,
            nn::LinearConfig {
                ws_init: truncated_normal_init(MODEL_DIM, NUM_VALUE_BUCKETS),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
        let cash_queries = p.var(
            "cash_queries",
            &[config.cash_pool_queries, MODEL_DIM],
            Init::Uniform {
                lo: -0.01,
                up: 0.01,
            },
        );
        let cash_q_proj = nn::linear(p / "cash_q_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let cash_k_proj = nn::linear(p / "cash_k_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let cash_v_proj = nn::linear(p / "cash_v_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let cash_merge = nn::linear(p / "cash_merge", MODEL_DIM, MODEL_DIM, Default::default());
        let cash_recent_proj = nn::linear(p / "cash_recent_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let cash_recent_gate = nn::linear(p / "cash_recent_gate", MODEL_DIM, MODEL_DIM, Default::default());
        let cash_recent_slope_raw = p.var("cash_recent_slope_raw", &[1], Init::Const(0.0));
        let cash_attn_temp_raw = p.var("cash_attn_temp_raw", &[1], Init::Const(0.0));
        const SDE_LATENT_DIM: i64 = 64;
        let sde_fc = nn::linear(p / "sde_fc", HEAD_HIDDEN, SDE_LATENT_DIM, Default::default());
        let ln_sde = RMSNorm::new(&(p / "ln_sde"), SDE_LATENT_DIM, 1e-6);
        let log_std_param = p.var(
            "log_std",
            &[SDE_LATENT_DIM, TICKERS_COUNT],
            Init::Const(0.0),
        );
        let value_clip = 10.0;
        let value_clip_symlog = symlog_tensor(&Tensor::from(value_clip)).double_value(&[]);
        let value_centers = Tensor::linspace(
            -value_clip_symlog,
            value_clip_symlog,
            NUM_VALUE_BUCKETS,
            (Kind::Float, p.device()),
        );
        let bucket_centers = value_centers.shallow_clone();
        let decay_positions = Tensor::arange(SEQ_LEN, (Kind::Float, p.device()));
        let decay_ones = Tensor::ones(&[SEQ_LEN], (Kind::Float, p.device()));
        Self {
            patch_embed_weight,
            patch_embed_bias,
            patch_ln_weight,
            patch_dt_scale,
            stem_pos_embed,
            stem_scale_embed,
            patch_gather_idx,
            patch_mask,
            patch_sizes,
            patch_last_idx,
            patch_config_ids,
            ssm_layers,
            ssm_norms,
            post_ssm_ln,
            ssm_gate,
            ssm_proj,
            static_proj,
            ln_static_proj,
            static_to_temporal,
            ln_static_temporal,
            pma_queries,
            pma_q_proj,
            pma_k_proj,
            pma_v_proj,
            pma_out,
            inter_ticker_block,
            time_pos_proj,
            time_global_ctx,
            time_ticker_ctx,
            static_cross_q,
            static_cross_k,
            static_cross_v,
            static_cross_out,
            cross_ticker_embed,
            global_ticker_token,
            global_tokens,
            global_token_proj,
            global_token_merge,
            global_to_ticker,
            global_inject_down,
            global_inject_up,
            global_inject_gate_raw,
            head_proj,
            head_ln,
            policy_ln,
            value_ln,
            value_mlp_fc1,
            value_mlp_fc2,
            actor_out,
            critic_out,
            cash_log_std_param,
            cash_queries,
            cash_q_proj,
            cash_k_proj,
            cash_v_proj,
            cash_merge,
            cash_recent_proj,
            cash_recent_gate,
            cash_recent_slope_raw,
            cash_attn_temp_raw,
            sde_fc,
            ln_sde,
            log_std_param,
            bucket_centers,
            value_centers,
            device: p.device(),
            num_heads,
            kv_heads,
            head_dim,
            cash_pool_queries: config.cash_pool_queries,
            decay_positions,
            decay_ones,
        }
    }

    fn rope_cos_sin(&self, positions: &Tensor, kind: Kind, device: tch::Device) -> (Tensor, Tensor) {
        let half = self.head_dim / 2;
        let idx = Tensor::arange(half, (Kind::Float, device));
        let inv_freq = (-(idx * 2.0 / self.head_dim as f64) * ROPE_BASE.ln()).exp();
        let freqs = positions.to_kind(Kind::Float).unsqueeze(1) * inv_freq.unsqueeze(0);
        let cos = freqs.cos().to_kind(kind);
        let sin = freqs.sin().to_kind(kind);
        (cos, sin)
    }

    fn apply_rope_cached(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let sizes = x.size();
        let (b, h, s, d) = (sizes[0], sizes[1], sizes[2], sizes[3]);
        let half = self.head_dim / 2;
        let cos = cos.unsqueeze(0).unsqueeze(0);
        let sin = sin.unsqueeze(0).unsqueeze(0);
        let x = x.view([b, h, s, half, 2]);
        let x_even = x.select(-1, 0);
        let x_odd = x.select(-1, 1);
        let rot = Tensor::stack(
            &[
                &x_even * &cos - &x_odd * &sin,
                &x_even * &sin + &x_odd * &cos,
            ],
            -1,
        );
        rot.view([b, h, s, d])
    }

    fn apply_rope_single(&self, x: &Tensor, positions: &Tensor) -> Tensor {
        let (cos, sin) = self.rope_cos_sin(positions, x.kind(), x.device());
        self.apply_rope_cached(x, &cos, &sin)
    }

    fn build_sin_cos_pos_embed(seq_len: i64, dim: i64, device: tch::Device) -> Tensor {
        let half = dim / 2;
        let positions = Tensor::arange(seq_len, (Kind::Float, device)).unsqueeze(1);
        let idx = Tensor::arange(half, (Kind::Float, device)).unsqueeze(0);
        let inv_freq = (-(idx * 2.0 / dim as f64) * ROPE_BASE.ln()).exp();
        let angles = positions * inv_freq;
        let sin = angles.sin();
        let cos = angles.cos();
        Tensor::cat(&[sin, cos], 1)
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

    fn patch_embed_single(&self, ticker_data: &Tensor) -> Tensor {
        let deltas = ticker_data.to_device(self.device);
        let kind = deltas.kind();
        let pos_embed = if self.stem_pos_embed.kind() == kind {
            self.stem_pos_embed.shallow_clone()
        } else {
            self.stem_pos_embed.to_kind(kind)
        };
        self.patch_embed_fused(&deltas) + pos_embed
    }

    fn normalize_seq_idx(&self, seq_idx: &Tensor, batch_size: i64) -> Tensor {
        if seq_idx.numel() == 0 {
            return seq_idx.shallow_clone();
        }
        let seq_idx = seq_idx.to_device(self.device);
        let sizes = seq_idx.size();
        if sizes.len() != 2 {
            panic!("seq_idx must be 2D");
        }
        if sizes[0] == batch_size && sizes[1] == TICKERS_COUNT * SEQ_LEN {
            seq_idx.view([batch_size * TICKERS_COUNT, SEQ_LEN])
        } else if sizes[0] == batch_size * TICKERS_COUNT && sizes[1] == SEQ_LEN {
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
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let kind = deltas.kind();
        let pos_embed = if self.stem_pos_embed.kind() == kind {
            self.stem_pos_embed.shallow_clone()
        } else {
            self.stem_pos_embed.to_kind(kind)
        };
        let x = self.patch_embed_fused(&deltas) + pos_embed;
        let seq_idx = match seq_idx {
            Some(seq_idx) => self.normalize_seq_idx(seq_idx, batch_size),
            None => Self::build_seq_idx_from_padding(
                &deltas,
                &self.patch_gather_idx,
                &self.patch_mask,
                &self.patch_sizes,
            ),
        };
        let dt_scale = self.patch_dt_scale.to_kind(kind).clamp_min(1e-4);
        (x, dt_scale, seq_idx)
    }

    fn build_seq_idx_from_padding(
        deltas: &Tensor,
        patch_gather_idx: &Tensor,
        patch_mask: &Tensor,
        patch_sizes: &Tensor,
    ) -> Tensor {
        let device = deltas.device();
        let zeros = deltas.eq(0.0).to_kind(Kind::Float);
        let prefix = zeros.cumprod(1, Kind::Float);
        let leading = prefix.sum_dim_intlist([1].as_slice(), false, Kind::Float).to_kind(Kind::Int64);
        let max_leading = leading.max().int64_value(&[]);
        if max_leading == 0 {
            return Tensor::zeros(&[0], (Kind::Int64, device));
        }
        let _ = patch_gather_idx;
        let _ = patch_mask;
        let patch_sizes = patch_sizes.to_device(device);
        let patch_ends = patch_sizes.to_kind(Kind::Int64).cumsum(1, Kind::Int64);
        let patch_ends = patch_ends.squeeze_dim(0).squeeze_dim(-1);
        let mask = leading.unsqueeze(1).ge_tensor(&patch_ends);
        mask.to_kind(Kind::Int64).neg()
    }

    fn enrich_patches(&self, patches: &Tensor) -> Tensor {
        let patch_len = patches.size()[2];
        let mean = patches
            .mean_dim([2].as_slice(), true, Kind::Float)
            .to_kind(patches.kind());
        let std = patches.std_dim(2, false, true).to_kind(patches.kind());
        let first = patches.narrow(2, 0, 1);
        let last = patches.narrow(2, patch_len - 1, 1);
        let slope = &last - &first;
        Tensor::cat(&[patches, &mean, &std, &last, &slope], 2)
    }

    fn patch_embed_fused(&self, deltas: &Tensor) -> Tensor {
        let device = deltas.device();
        let kind = deltas.kind();
        let batch_tokens = deltas.size()[0];
        let total_len = deltas.size()[1];
        let max_patch_size = self.patch_gather_idx.size()[1];
        let idx = self.patch_gather_idx.to_device(device);
        let mask = self.patch_mask.to_device(device).to_kind(kind);
        let idx_clamped = idx.clamp_min(0);
        let idx_exp = idx_clamped
            .unsqueeze(0)
            .expand(&[batch_tokens, SEQ_LEN, max_patch_size], false);
        let deltas_exp = deltas
            .unsqueeze(1)
            .expand(&[batch_tokens, SEQ_LEN, total_len], false);
        let mut patches = deltas_exp.gather(2, &idx_exp, false);
        let mask_exp = mask
            .unsqueeze(0)
            .expand(&[batch_tokens, SEQ_LEN, max_patch_size], false);
        patches = patches * &mask_exp;
        let sizes = self.patch_sizes.to_device(device).to_kind(Kind::Float);
        let sizes_f = sizes.to_kind(Kind::Float);
        let patches_f = patches.to_kind(Kind::Float);
        let mean = patches_f.sum_dim_intlist([2].as_slice(), true, Kind::Float) / &sizes_f;
        let var = (&patches_f - &mean).pow_tensor_scalar(2.0) * &mask_exp.to_kind(Kind::Float);
        let var = var.sum_dim_intlist([2].as_slice(), true, Kind::Float) / &sizes_f;
        let std = (var + 1e-5).sqrt();
        let first = patches_f.narrow(2, 0, 1);
        let last_idx = self.patch_last_idx.to_device(device);
        let last = patches_f.gather(2, &last_idx.expand(&[batch_tokens, SEQ_LEN, 1], false), false);
        let slope = &last - &first;
        let enriched = Tensor::cat(&[
            &patches_f,
            &mean,
            &std,
            &last,
            &slope,
        ], 2);
        let config_ids = self.patch_config_ids.to_device(device);
        let weight = self.patch_embed_weight.to_device(device).to_kind(Kind::Float);
        let bias = self.patch_embed_bias.to_device(device).to_kind(Kind::Float);
        let weight_per_patch = weight.index_select(0, &config_ids);
        let bias_per_patch = bias.index_select(0, &config_ids);
        let out = Tensor::einsum("blm,lmd->bld", &[&enriched, &weight_per_patch], None::<&[i64]>);
        let mut out = out + bias_per_patch.unsqueeze(0);
        let ln_weight = self.patch_ln_weight.to_device(device).to_kind(Kind::Float);
        let ln_weight = ln_weight.index_select(0, &config_ids).unsqueeze(0);
        let rms = (out.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float) + 1e-5).sqrt();
        out = out / rms * ln_weight;
        let scale = self.stem_scale_embed.to_device(device).to_kind(Kind::Float);
        let scale = scale.index_select(0, &config_ids).unsqueeze(0);
        (out + scale).to_kind(kind)
    }

    fn embed_patch_config(&self, patches: &Tensor, config_idx: i64) -> Tensor {
        let device = patches.device();
        let kind = patches.kind();
        let patches_f = patches.to_kind(Kind::Float);
        let patch_len = patches_f.size()[2];
        let mean = patches_f.mean_dim([2].as_slice(), true, Kind::Float);
        let std = patches_f.std_dim(2, false, true);
        let first = patches_f.narrow(2, 0, 1);
        let last = patches_f.narrow(2, patch_len - 1, 1);
        let slope = &last - &first;
        let enriched = Tensor::cat(&[&patches_f, &mean, &std, &last, &slope], 2);
        let weight = self.patch_embed_weight.get(config_idx);
        let bias = self.patch_embed_bias.get(config_idx);
        let out = enriched.matmul(&weight) + bias;
        let ln_weight = self.patch_ln_weight.get(config_idx);
        let rms = (out.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float) + 1e-5).sqrt();
        let out = out / rms * ln_weight;
        let scale = self.stem_scale_embed.get(config_idx);
        (out + scale).to_kind(kind)
    }
}
