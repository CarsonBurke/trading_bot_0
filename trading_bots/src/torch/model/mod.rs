mod forward;
mod head;
mod inference;
mod rmsnorm;

use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT,
};
use crate::torch::ssm::{stateful_mamba_block_cfg, Mamba2Config, Mamba2State, StatefulMamba};

pub use shared::constants::GLOBAL_MACRO_OBS;

use rmsnorm::RMSNorm;

struct TimeCrossBlock2d {
    time_rp_q: nn::Linear,
    time_rp_v: Tensor,
    time_rp_k_frozen: Tensor,
    time_latent_q: nn::Linear,
    time_latent_k: Tensor,
    time_latent_v: Tensor,
    time_out: nn::Linear,
    ln: RMSNorm,
    ticker_rp_q: nn::Linear,
    ticker_rp_v: Tensor,
    ticker_rp_k_frozen: Tensor,
    ticker_latent_q: nn::Linear,
    ticker_latent_k: Tensor,
    ticker_latent_v: Tensor,
    ticker_out: nn::Linear,
    ticker_ln: RMSNorm,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    mlp_ln: RMSNorm,
    alpha_attn: Tensor,
    alpha_time_rp: Tensor,
    alpha_ticker_rp: Tensor,
    alpha_ticker_attn: Tensor,
    alpha_mlp: Tensor,
}

impl TimeCrossBlock2d {
    fn new(p: &nn::Path, kv_heads: i64, head_dim: i64, ticker_latents: i64) -> Self {
        let _ = (kv_heads, head_dim);
        let time_rp_q = nn::linear(p / "time_rp_q", MODEL_DIM, MODEL_DIM, Default::default());
        let time_rp_v = p.var(
            "time_rp_v",
            &[TIME_LATENT_FACTORS, MODEL_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (MODEL_DIM as f64).sqrt(),
            },
        );
        let time_rp_k_frozen = Tensor::randn(
            &[TIME_LATENT_FACTORS, MODEL_DIM],
            (Kind::Float, p.device()),
        ) * (1.0 / (MODEL_DIM as f64).sqrt());
        time_rp_k_frozen.set_requires_grad(false);
        let time_latent_q =
            nn::linear(p / "time_latent_q", MODEL_DIM, MODEL_DIM, Default::default());
        let time_latent_k = p.var(
            "time_latent_k",
            &[TIME_LATENT_FACTORS, MODEL_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (MODEL_DIM as f64).sqrt(),
            },
        );
        let time_latent_v = p.var(
            "time_latent_v",
            &[TIME_LATENT_FACTORS, MODEL_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (MODEL_DIM as f64).sqrt(),
            },
        );
        let time_out = nn::linear(p / "time_out", MODEL_DIM, MODEL_DIM, Default::default());
        let ln = RMSNorm::new(&(p / "ln"), MODEL_DIM, 1e-5);
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
        let ticker_ln = RMSNorm::new(&(p / "ticker_ln"), MODEL_DIM, 1e-5);
        let mlp_fc1 = nn::linear(p / "mlp_fc1", MODEL_DIM, 2 * FF_DIM, Default::default());
        let mlp_fc2 = nn::linear(p / "mlp_fc2", FF_DIM, MODEL_DIM, Default::default());
        let mlp_ln = RMSNorm::new(&(p / "mlp_ln"), MODEL_DIM, 1e-5);
        let alpha_attn = p.var("alpha_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_time_rp = p.var("alpha_time_rp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_ticker_rp = p.var("alpha_ticker_rp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_ticker_attn =
            p.var("alpha_ticker_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_mlp = p.var("alpha_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            time_rp_q,
            time_rp_v,
            time_rp_k_frozen,
            time_latent_q,
            time_latent_k,
            time_latent_v,
            time_out,
            ln,
            ticker_rp_q,
            ticker_rp_v,
            ticker_rp_k_frozen,
            ticker_latent_q,
            ticker_latent_k,
            ticker_latent_v,
            ticker_out,
            ticker_ln,
            mlp_fc1,
            mlp_fc2,
            mlp_ln,
            alpha_attn,
            alpha_time_rp,
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
const LOGIT_SCALE_INIT: f64 = 0.3;
const LOG_STD_INIT: f64 = -2.0;
const SDE_EPS: f64 = 1e-6;
const SDE_LEARN_FEATURES: bool = false;
pub const LOGIT_SCALE_GROUP: usize = 1;
const USE_SDPA: bool = true;
const SDPA_MIN_LEN: i64 = 64;
const SDPA_MIN_LEN_CROSS: i64 = 1;
const TIME_CROSS_LAYERS: usize = 1;
const FF_DIM: i64 = 512;
const HEAD_HIDDEN: i64 = 192;
const RESIDUAL_ALPHA_MAX: f64 = 0.5;
const RESIDUAL_ALPHA_INIT: f64 = -2.0;
const ROPE_BASE: f64 = 10000.0;
const TIME_LATENT_FACTORS: i64 = 32;
const TICKER_LATENT_FACTORS: i64 = 32;
const DEFAULT_CASH_POOL_QUERIES: i64 = 4;
const DEFAULT_TICKER_POOL_QUERIES: i64 = 4;

// Uniform patch size for proper streaming support
// 3400 deltas / 17 = 200 tokens
const PATCH_SIZE: i64 = 17;
const PATCH_EXTRA_FEATS: i64 = 4;
const PATCH_INPUT_DIM: i64 = PATCH_SIZE + PATCH_EXTRA_FEATS;
const SEQ_LEN: i64 = PRICE_DELTAS_PER_TICKER as i64 / PATCH_SIZE;

const _: () = assert!(
    PRICE_DELTAS_PER_TICKER as i64 % PATCH_SIZE == 0,
    "PRICE_DELTAS must be divisible by PATCH_SIZE"
);

// (values, (action_logits, action_log_std, sde_latent), attn_weights, attn_entropy)
pub type ModelOutput = (Tensor, (Tensor, Tensor, Tensor), Tensor, Tensor);

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
    pub ticker_pool_queries: i64,
}

impl Default for TradingModelConfig {
    fn default() -> Self {
        Self {
            ssm_layers: 2,
            cash_pool_queries: DEFAULT_CASH_POOL_QUERIES,
            ticker_pool_queries: DEFAULT_TICKER_POOL_QUERIES,
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
    /// Patch accumulator: [TICKERS_COUNT, PATCH_SIZE]
    pub patch_buf: Tensor,
    /// Position within current patch
    pub patch_pos: i64,
    /// SSM hidden state per ticker per layer
    pub ssm_states: Vec<Mamba2State>,
    /// Cached SSM output sequence: [TICKERS_COUNT, SSM_DIM, SEQ_LEN]
    pub ssm_cache: Tensor,
    /// Whether initialized with full sequence
    pub initialized: bool,
}

pub struct TradingModel {
    patch_embed: nn::Linear,
    patch_ln: RMSNorm,
    ssm_layers: Vec<StatefulMamba>,
    ssm_norms: Vec<RMSNorm>,
    post_ssm_ln: RMSNorm,
    ssm_gate: nn::Linear,
    ssm_proj: nn::Conv1D,
    static_proj: nn::Linear,
    ln_static_proj: RMSNorm,
    time_cross_block: TimeCrossBlock2d,
    time_pos_proj: nn::Linear,
    time_global_ctx: nn::Linear,
    time_ticker_ctx: nn::Linear,
    static_cross_q: nn::Linear,
    static_cross_k: nn::Linear,
    static_cross_v: nn::Linear,
    static_cross_out: nn::Linear,
    cross_ticker_embed: nn::Linear,
    global_ticker_token: Tensor,
    ticker_queries: Tensor,
    ticker_q_proj: nn::Linear,
    ticker_k_proj: nn::Linear,
    ticker_v_proj: nn::Linear,
    ticker_merge: nn::Linear,
    ticker_ctx_proj: nn::Linear,
    ticker_recent_proj: nn::Linear,
    ticker_recent_gate: nn::Linear,
    ticker_recent_slope_raw: Tensor,
    ticker_attn_temp_raw: Tensor,
    global_to_ticker: nn::Linear,
    head_proj: nn::Linear,
    head_ln: RMSNorm,
    policy_ln: RMSNorm,
    value_ln: RMSNorm,
    value_mlp_fc1: nn::Linear,
    value_mlp_fc2: nn::Linear,
    actor_out: nn::Linear,
    cash_out: nn::Linear,
    value_ticker_out: nn::Linear,
    cash_value_out: nn::Linear,
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
    cash_sde: nn::Linear,
    sde_scale_ticker: nn::Linear,
    sde_scale_cash: nn::Linear,
    log_std_param: Tensor,
    logit_scale_raw: Tensor,
    device: tch::Device,
    num_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    cash_pool_queries: i64,
    ticker_pool_queries: i64,
    patch_pos_embed: Tensor,
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
        let target_kind = self.logit_scale_raw.kind();
        if input.kind() == target_kind {
            input.shallow_clone()
        } else {
            input.to_kind(target_kind)
        }
    }

    pub fn logit_scale(&self) -> Tensor {
        self.logit_scale_raw.exp()
    }

    pub fn sde_std_matrix(&self) -> Tensor {
        sde_std_from_log_std(&self.log_std_param)
    }

    pub fn new(p: &nn::Path) -> Self {
        Self::new_with_config(p, TradingModelConfig::default())
    }

    pub fn new_with_config(p: &nn::Path, config: TradingModelConfig) -> Self {
        let patch_embed = nn::linear(p / "patch_embed", PATCH_INPUT_DIM, SSM_DIM, Default::default());
        let patch_ln = RMSNorm::new(&(p / "patch_ln"), SSM_DIM, 1e-5);

        let ssm_cfg = Mamba2Config {
            d_model: SSM_DIM,
            d_ssm: Some(SSM_DIM),
            ..Mamba2Config::default()
        };
        let ssm_layers = (0..config.ssm_layers)
            .map(|i| stateful_mamba_block_cfg(&(p / format!("ssm_{}", i)), ssm_cfg.clone()))
            .collect::<Vec<_>>();
        let ssm_norms = (0..config.ssm_layers)
            .map(|i| RMSNorm::new(&(p / format!("ssm_norm_{}", i)), SSM_DIM, 1e-5))
            .collect::<Vec<_>>();
        let post_ssm_ln = RMSNorm::new(&(p / "post_ssm_ln"), SSM_DIM, 1e-5);
        let ssm_gate = nn::linear(p / "ssm_gate", SSM_DIM, SSM_DIM, Default::default());
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, MODEL_DIM, 1, Default::default());

        let static_proj = nn::linear(
            p / "static_proj",
            MODEL_DIM + PER_TICKER_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let ln_static_proj = RMSNorm::new(&(p / "ln_static_proj"), MODEL_DIM, 1e-5);

        let num_heads = 8i64;
        let kv_heads = 8i64;
        let head_dim = 16i64;
        assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
        assert!(config.cash_pool_queries > 0, "cash_pool_queries must be > 0");
        assert!(
            config.ticker_pool_queries > 0,
            "ticker_pool_queries must be > 0"
        );
        let time_cross_block = TimeCrossBlock2d::new(
            &(p / "time_cross_0"),
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
        let ticker_queries = p.var(
            "ticker_queries",
            &[config.ticker_pool_queries, MODEL_DIM],
            Init::Uniform {
                lo: -0.01,
                up: 0.01,
            },
        );
        let ticker_q_proj = nn::linear(p / "ticker_q_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_k_proj = nn::linear(p / "ticker_k_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_v_proj = nn::linear(p / "ticker_v_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_merge = nn::linear(p / "ticker_merge", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_ctx_proj = nn::linear(
            p / "ticker_ctx_proj",
            PER_TICKER_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let ticker_recent_proj =
            nn::linear(p / "ticker_recent_proj", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_recent_gate =
            nn::linear(p / "ticker_recent_gate", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_recent_slope_raw = p.var("ticker_recent_slope_raw", &[1], Init::Const(0.0));
        let ticker_attn_temp_raw = p.var("ticker_attn_temp_raw", &[1], Init::Const(0.0));
        let global_to_ticker = nn::linear(
            p / "global_to_ticker",
            GLOBAL_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let head_proj = nn::linear(
            p / "head_proj",
            MODEL_DIM,
            HEAD_HIDDEN,
            nn::LinearConfig {
                ws_init: truncated_normal_init(MODEL_DIM, HEAD_HIDDEN),
                ..Default::default()
            },
        );
        let head_ln = RMSNorm::new(&(p / "head_ln"), HEAD_HIDDEN, 1e-5);
        let policy_ln = RMSNorm::new(&(p / "policy_ln"), MODEL_DIM, 1e-5);
        let value_ln = RMSNorm::new(&(p / "value_ln"), MODEL_DIM, 1e-5);
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
        let cash_out = nn::linear(
            p / "cash_out",
            HEAD_HIDDEN,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(HEAD_HIDDEN, 1),
                ..Default::default()
            },
        );
        let value_ticker_out = nn::linear(
            p / "value_ticker_out",
            HEAD_HIDDEN,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(HEAD_HIDDEN, 1),
                ..Default::default()
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
        let cash_value_out = nn::linear(
            p / "cash_value_out",
            HEAD_HIDDEN,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(HEAD_HIDDEN, 1),
                ..Default::default()
            },
        );

        const SDE_LATENT_DIM: i64 = 64;
        let sde_fc = nn::linear(p / "sde_fc", HEAD_HIDDEN, SDE_LATENT_DIM, Default::default());
        let cash_sde = nn::linear(p / "cash_sde", HEAD_HIDDEN, SDE_LATENT_DIM, Default::default());
        let sde_scale_ticker = nn::linear(p / "sde_scale_ticker", HEAD_HIDDEN, 1, Default::default());
        let sde_scale_cash = nn::linear(p / "sde_scale_cash", HEAD_HIDDEN, 1, Default::default());
        let log_std_param = p.var(
            "log_std",
            &[SDE_LATENT_DIM, ACTION_COUNT],
            Init::Const(LOG_STD_INIT),
        );
        let logit_scale_raw = p.set_group(LOGIT_SCALE_GROUP).var(
            "logit_scale_raw",
            &[1],
            Init::Const(LOGIT_SCALE_INIT.ln()),
        );
        let patch_pos_embed = Self::build_sin_cos_pos_embed(SEQ_LEN, MODEL_DIM, p.device());
        let decay_positions = Tensor::arange(SEQ_LEN, (Kind::Float, p.device()));
        let decay_ones = Tensor::ones(&[SEQ_LEN], (Kind::Float, p.device()));
        Self {
            patch_embed,
            patch_ln,
            ssm_layers,
            ssm_norms,
            post_ssm_ln,
            ssm_gate,
            ssm_proj,
            static_proj,
            ln_static_proj,
            time_cross_block,
            time_pos_proj,
            time_global_ctx,
            time_ticker_ctx,
            static_cross_q,
            static_cross_k,
            static_cross_v,
            static_cross_out,
            cross_ticker_embed,
            global_ticker_token,
            ticker_queries,
            ticker_q_proj,
            ticker_k_proj,
            ticker_v_proj,
            ticker_merge,
            ticker_ctx_proj,
            ticker_recent_proj,
            ticker_recent_gate,
            ticker_recent_slope_raw,
            ticker_attn_temp_raw,
            global_to_ticker,
            head_proj,
            head_ln,
            policy_ln,
            value_ln,
            value_mlp_fc1,
            value_mlp_fc2,
            actor_out,
            cash_out,
            value_ticker_out,
            cash_value_out,
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
            cash_sde,
            sde_scale_ticker,
            sde_scale_cash,
            log_std_param,
            logit_scale_raw,
            device: p.device(),
            num_heads,
            kv_heads,
            head_dim,
            cash_pool_queries: config.cash_pool_queries,
            ticker_pool_queries: config.ticker_pool_queries,
            patch_pos_embed,
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
        let patches = ticker_data.view([1, SEQ_LEN, PATCH_SIZE]);
        let patches = self.enrich_patches(&patches);
        let x = patches.apply(&self.patch_embed);
        let x = self.patch_ln.forward(&x);
        x.permute([0, 2, 1])
    }

    fn patch_embed_all_with_static(&self, price_deltas: &Tensor, batch_size: i64) -> Tensor {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let patches = deltas.view([batch_tokens, SEQ_LEN, PATCH_SIZE]);
        let patches = self.enrich_patches(&patches);
        let x = patches.apply(&self.patch_embed);
        let x = self.patch_ln.forward(&x);
        x.permute([0, 2, 1])
    }

    fn enrich_patches(&self, patches: &Tensor) -> Tensor {
        let mean = patches
            .mean_dim([2].as_slice(), true, Kind::Float)
            .to_kind(patches.kind());
        let std = patches.std_dim(2, false, true).to_kind(patches.kind());
        let first = patches.narrow(2, 0, 1);
        let last = patches.narrow(2, PATCH_SIZE - 1, 1);
        let slope = &last - &first;
        Tensor::cat(&[patches, &mean, &std, &last, &slope], 2)
    }
}

pub(crate) fn sde_std_from_log_std(log_std: &Tensor) -> Tensor {
    let positive = log_std.gt(0.0).to_kind(Kind::Float);
    let non_positive = log_std.le(0.0).to_kind(Kind::Float);
    let below = log_std.exp() * &non_positive;
    let safe_log_std = log_std * &positive + SDE_EPS;
    let above = (safe_log_std.log1p() + 1.0) * &positive;
    below + above
}
