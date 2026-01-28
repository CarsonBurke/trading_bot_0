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
use crate::torch::ppo::VALUE_LOG_CLIP;
use crate::torch::ssm_ref::{stateful_mamba_block_cfg, Mamba2Config, Mamba2State, StatefulMambaRef};

pub use shared::constants::GLOBAL_MACRO_OBS;

use rmsnorm::RMSNorm;

struct InterTickerBlock {
    ticker_ln: RMSNorm,
    ticker_latent_q: nn::Linear,
    ticker_latent_k: Tensor,
    ticker_latent_k_frozen: Tensor,
    ticker_latent_v: Tensor,
    ticker_out: nn::Linear,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    mlp_ln: RMSNorm,
    alpha_ticker_attn: Tensor,
    alpha_mlp: Tensor,
}

impl InterTickerBlock {
    fn new(p: &nn::Path, ticker_latents: i64) -> Self {
        let ticker_ln = RMSNorm::new(&(p / "ticker_ln"), MODEL_DIM, 1e-6);
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
        let ticker_latent_k_frozen = p.var(
            "ticker_latent_k_frozen",
            &[ticker_latents, MODEL_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (MODEL_DIM as f64).sqrt(),
            },
        );
        let _ = ticker_latent_k_frozen.set_requires_grad(false);
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
        let alpha_ticker_attn =
            p.var("alpha_ticker_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_mlp = p.var("alpha_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            ticker_ln,
            ticker_latent_q,
            ticker_latent_k,
            ticker_latent_k_frozen,
            ticker_latent_v,
            ticker_out,
            mlp_fc1,
            mlp_fc2,
            mlp_ln,
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
const SSM_NHEADS: i64 = 2;
const SSM_HEADDIM: i64 = 64;
const SSM_DSTATE: i64 = 128;
pub(crate) const SDE_LATENT_DIM: i64 = 64;
pub(crate) const LOG_STD_INIT: f64 = -2.0;
pub(crate) const SDE_EPS: f64 = 1e-6;
const TIME_CROSS_LAYERS: usize = 1;
const FF_DIM: i64 = 512;
const HEAD_HIDDEN: i64 = 192;
const RESIDUAL_ALPHA_MAX: f64 = 0.5;
const RESIDUAL_ALPHA_INIT: f64 = -4.0;
const TICKER_LATENT_FACTORS: i64 = 8;

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
const TEMPORAL_STATIC_DIM: i64 = 64;

const PATCH_CONFIGS: [(i64, i64); 7] = [
    (4608, 128),  // 36 tokens - ~16 days
    (2048, 64),   // 32 tokens - ~7.1 days
    (1024, 32),   // 32 tokens - ~3.6 days
    (512, 16),    // 32 tokens - ~1.8 days
    (256, 8),     // 32 tokens - ~21 hrs
    (128, 4),     // 32 tokens - ~10.7 hrs
    (60, 1),      // 60 tokens - ~5 hrs (finest)
];
// Total: 8636 segments (~30 days), 256 tokens

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
// Enriched patch dim: 2 * max_patch_size (values + intra_dt) + 3 (mean, std, slope)
const PATCH_SCALAR_FEATS: i64 = 3;
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
// action_mean: [batch, TICKERS_COUNT + 1] logits before softmax
// action_log_std: [batch, TICKERS_COUNT + 1] per-action log std
/// (values, critic_logits, (action_mean, sde_latent), attn_entropy)
/// sde_latent: [batch, tickers, SDE_LATENT_DIM] for gSDE noise computation
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
}

impl Default for TradingModelConfig {
    fn default() -> Self {
        Self { ssm_layers: 2 }
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
    stem_scale_embed: Tensor,
    patch_gather_idx: Tensor,
    patch_mask: Tensor,
    patch_sizes: Tensor,
    patch_intra_dt: Tensor,
    patch_config_ids: Tensor,
    ssm_layers: Vec<StatefulMambaRef>,
    ssm_norms: Vec<RMSNorm>,
    ssm_gate: nn::Linear,
    ssm_proj: nn::Conv1D,
    static_proj: nn::Linear,
    ln_static_proj: RMSNorm,
    inter_ticker_block: InterTickerBlock,
    time_pos_proj: nn::Linear,
    time_global_ctx: nn::Linear,
    time_ticker_ctx: nn::Linear,
    last_token_ln: RMSNorm,
    static_cross_q: nn::Linear,
    static_cross_k: nn::Linear,
    static_cross_v: nn::Linear,
    static_cross_out: nn::Linear,
    cross_ticker_embed: nn::Linear,
    global_to_ticker: nn::Linear,
    global_inject_down: nn::Linear,
    global_inject_up: nn::Linear,
    global_inject_gate_raw: Tensor,
    head_proj: nn::Linear,
    policy_ln: RMSNorm,
    value_ln: RMSNorm,
    value_mlp_fc1: nn::Linear,
    value_mlp_fc2: nn::Linear,
    actor_out: nn::Linear,
    critic_out: nn::Linear,
    cash_merge: nn::Linear,
    // SDE for state-dependent exploration (gSDE)
    sde_fc: nn::Linear,
    log_std_param: Tensor,        // [SDE_LATENT_DIM, TICKERS_COUNT]
    cash_log_std_param: Tensor,   // [1] - state-independent cash exploration
    bucket_centers: Tensor,
    value_centers: Tensor,
    device: tch::Device,
}

impl TradingModel {
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

    /// Get exploration std for gSDE: [SDE_LATENT_DIM, TICKERS_COUNT]
    pub fn sde_std(&self) -> Tensor {
        (&self.log_std_param + LOG_STD_INIT).clamp(-3.0, -0.5).exp()
    }

    /// Get cash exploration log_std (state-independent): scalar
    pub fn cash_log_std(&self) -> Tensor {
        (&self.cash_log_std_param + LOG_STD_INIT).clamp(-3.0, -0.5)
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
        let max_input_dim = 2 * max_patch_size + PATCH_SCALAR_FEATS;
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
        let (patch_gather_idx, patch_mask, patch_sizes, patch_intra_dt, patch_config_ids) = {
            let mut gather_idx = Vec::with_capacity((SEQ_LEN * max_patch_size) as usize);
            let mut mask = Vec::with_capacity((SEQ_LEN * max_patch_size) as usize);
            let mut sizes = Vec::with_capacity(SEQ_LEN as usize);
            let mut intra_dt = Vec::with_capacity((SEQ_LEN * max_patch_size) as usize);
            let mut cfg_ids = Vec::with_capacity(SEQ_LEN as usize);
            let mut offset = 0i64;
            for (cfg_idx, &(days, patch_size)) in PATCH_CONFIGS.iter().enumerate() {
                let n_patches = days / patch_size;
                for p_idx in 0..n_patches {
                    let start = offset + p_idx * patch_size;
                    sizes.push(patch_size as f32);
                    cfg_ids.push(cfg_idx as i64);
                    for k in 0..max_patch_size {
                        if k < patch_size {
                            gather_idx.push(start + k);
                            mask.push(1.0);
                            // Normalized intra-patch temporal position [0, 1]
                            let denom = (patch_size - 1).max(1) as f32;
                            intra_dt.push(k as f32 / denom);
                        } else {
                            gather_idx.push(-1);
                            mask.push(0.0);
                            intra_dt.push(0.0);
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
            let intra_dt = Tensor::from_slice(&intra_dt)
                .view([SEQ_LEN, max_patch_size])
                .to_device(p.device());
            let cfg_ids = Tensor::from_slice(&cfg_ids)
                .to_kind(Kind::Int64)
                .to_device(p.device());
            (gather_idx, mask, sizes, intra_dt, cfg_ids)
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
        let ssm_gate = nn::linear(p / "ssm_gate", SSM_DIM, SSM_DIM, Default::default());
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, MODEL_DIM, 1, Default::default());

        let static_proj = nn::linear(
            p / "static_proj",
            MODEL_DIM + PER_TICKER_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let ln_static_proj = RMSNorm::new(&(p / "ln_static_proj"), MODEL_DIM, 1e-6);

        let inter_ticker_block = InterTickerBlock::new(&(p / "inter_ticker_0"), TICKER_LATENT_FACTORS);
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
        let last_token_ln = RMSNorm::new(&(p / "last_token_ln"), MODEL_DIM, 1e-6);
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
        let policy_ln = RMSNorm::new(&(p / "policy_ln"), MODEL_DIM, 1e-6);
        let value_ln = RMSNorm::new(&(p / "value_ln"), MODEL_DIM, 1e-6);
        let value_mlp_fc1 = nn::linear(p / "value_mlp_fc1", MODEL_DIM, MODEL_DIM, Default::default());
        let value_mlp_fc2 = nn::linear(p / "value_mlp_fc2", MODEL_DIM, MODEL_DIM, Default::default());
        
        let actor_out = nn::linear(
            p / "actor_out",
            HEAD_HIDDEN,
            1,
            nn::LinearConfig {
                ws_init: Init::Randn { mean: 0.0, stdev: 0.01 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );
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
        let cash_merge = nn::linear(p / "cash_merge", MODEL_DIM, MODEL_DIM, Default::default());

        // SDE for state-dependent exploration (SB3-style with learn_features=True)
        let sde_fc = nn::linear(p / "sde_fc", HEAD_HIDDEN, SDE_LATENT_DIM, Default::default());
        let log_std_param = p.var(
            "log_std",
            &[SDE_LATENT_DIM, TICKERS_COUNT],
            Init::Const(0.0),
        );
        let cash_log_std_param = p.var("cash_log_std", &[1], Init::Const(0.0));
        let value_centers = Tensor::linspace(
            -VALUE_LOG_CLIP,
            VALUE_LOG_CLIP,
            NUM_VALUE_BUCKETS,
            (Kind::Float, p.device()),
        );
        let bucket_centers = value_centers.shallow_clone();
        Self {
            patch_embed_weight,
            patch_embed_bias,
            patch_ln_weight,
            patch_dt_scale,
            stem_scale_embed,
            patch_gather_idx,
            patch_mask,
            patch_sizes,
            patch_intra_dt,
            patch_config_ids,
            ssm_layers,
            ssm_norms,
            ssm_gate,
            ssm_proj,
            static_proj,
            ln_static_proj,
            inter_ticker_block,
            time_pos_proj,
            time_global_ctx,
            time_ticker_ctx,
            last_token_ln,
            static_cross_q,
            static_cross_k,
            static_cross_v,
            static_cross_out,
            cross_ticker_embed,
            global_to_ticker,
            global_inject_down,
            global_inject_up,
            global_inject_gate_raw,
            head_proj,
            policy_ln,
            value_ln,
            value_mlp_fc1,
            value_mlp_fc2,
            actor_out,
            critic_out,
            cash_merge,
            sde_fc,
            log_std_param,
            cash_log_std_param,
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

    fn patch_embed_single(&self, ticker_data: &Tensor) -> Tensor {
        let deltas = ticker_data.to_device(self.device);
        self.patch_embed_fused(&deltas)
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
        let x = self.patch_embed_fused(&deltas);
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
        let sizes_f = self.patch_sizes.to_device(device).to_kind(Kind::Float);
        let patches_f = patches.to_kind(Kind::Float);
        let intra_dt = self.patch_intra_dt
            .to_device(device)
            .to_kind(Kind::Float)
            .unsqueeze(0)
            .expand(&[batch_tokens, SEQ_LEN, max_patch_size], false);
        let mean = patches_f.sum_dim_intlist([2].as_slice(), true, Kind::Float) / &sizes_f;
        let var = (&patches_f - &mean).pow_tensor_scalar(2.0) * &mask_exp.to_kind(Kind::Float);
        let var = var.sum_dim_intlist([2].as_slice(), true, Kind::Float) / &sizes_f;
        let std = (var + 1e-5).sqrt();
        let first = patches_f.narrow(2, 0, 1);
        let last_pos = (sizes_f - 1.0).clamp_min(0.0).to_kind(Kind::Int64);
        let last = patches_f.gather(2, &last_pos.expand(&[batch_tokens, SEQ_LEN, 1], false), false);
        let slope = &last - &first;
        let enriched = Tensor::cat(&[
            &patches_f,
            &intra_dt,
            &mean,
            &std,
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
        // Build intra_dt for this specific patch config (normalized [0, 1])
        let batch = patches_f.size()[0];
        let n_patches = patches_f.size()[1];
        let intra_dt = Tensor::arange(patch_len, (Kind::Float, device))
            / (patch_len - 1).max(1) as f64;
        let intra_dt = intra_dt
            .view([1, 1, patch_len])
            .expand([batch, n_patches, patch_len], false);
        let mean = patches_f.mean_dim([2].as_slice(), true, Kind::Float);
        let std = patches_f.std_dim(2, false, true);
        let first = patches_f.narrow(2, 0, 1);
        let last = patches_f.narrow(2, patch_len - 1, 1);
        let slope = &last - &first;
        let enriched = Tensor::cat(&[&patches_f, &intra_dt, &mean, &std, &slope], 2);
        let input_dim = 2 * patch_len + PATCH_SCALAR_FEATS;
        // Slice weight to match actual input dim (weights are sized for max_patch_size)
        let weight = self.patch_embed_weight.get(config_idx).narrow(0, 0, input_dim);
        let bias = self.patch_embed_bias.get(config_idx);
        let out = enriched.matmul(&weight) + bias;
        let ln_weight = self.patch_ln_weight.get(config_idx);
        let rms = (out.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float) + 1e-5).sqrt();
        let out = out / rms * ln_weight;
        let scale = self.stem_scale_embed.get(config_idx);
        (out + scale).to_kind(kind)
    }
}
