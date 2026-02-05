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


use rmsnorm::RMSNorm;

struct InterTickerBlock {
    ticker_ln: RMSNorm,
    ticker_q: nn::Linear,
    ticker_k: nn::Linear,
    ticker_v: nn::Linear,
    ticker_out: nn::Linear,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    mlp_ln: RMSNorm,
    alpha_ticker_attn: Tensor,
    alpha_mlp: Tensor,
}

impl InterTickerBlock {
    fn new(p: &nn::Path) -> Self {
        let ticker_ln = RMSNorm::new(&(p / "ticker_ln"), MODEL_DIM, 1e-6);
        let ticker_q = nn::linear(p / "ticker_q", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_k = nn::linear(p / "ticker_k", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_v = nn::linear(p / "ticker_v", MODEL_DIM, MODEL_DIM, Default::default());
        let ticker_out = nn::linear(p / "ticker_out", MODEL_DIM, MODEL_DIM, Default::default());
        let mlp_fc1 = nn::linear(p / "mlp_fc1", MODEL_DIM, 2 * FF_DIM, Default::default());
        let mlp_fc2 = nn::linear(p / "mlp_fc2", FF_DIM, MODEL_DIM, Default::default());
        let mlp_ln = RMSNorm::new(&(p / "mlp_ln"), MODEL_DIM, 1e-6);
        let alpha_ticker_attn =
            p.var("alpha_ticker_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_mlp = p.var("alpha_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            ticker_ln,
            ticker_q,
            ticker_k,
            ticker_v,
            ticker_out,
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
    cross_k: nn::Linear,
    cross_v: nn::Linear,
    cross_out: nn::Linear,
    alpha_cross: Tensor,
}

impl ExoCrossBlock {
    fn new(p: &nn::Path) -> Self {
        let cross_ln = RMSNorm::new(&(p / "cross_ln"), MODEL_DIM, 1e-6);
        let cross_q = nn::linear(p / "cross_q", MODEL_DIM, MODEL_DIM, Default::default());
        let cross_k = nn::linear(p / "cross_k", MODEL_DIM, MODEL_DIM, Default::default());
        let cross_v = nn::linear(p / "cross_v", MODEL_DIM, MODEL_DIM, Default::default());
        let cross_out = nn::linear(p / "cross_out", MODEL_DIM, MODEL_DIM, Default::default());
        let alpha_cross = p.var("alpha_cross", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            cross_ln, cross_q, cross_k, cross_v, cross_out, alpha_cross,
        }
    }

    fn forward(&self, x: &Tensor, exo_kv: &Tensor) -> Tensor {
        let q = self.cross_ln.forward(x).apply(&self.cross_q);
        let k = exo_kv.apply(&self.cross_k);
        let v = exo_kv.apply(&self.cross_v);
        let scores = q.matmul(&k.transpose(-2, -1)) / (MODEL_DIM as f64).sqrt();
        let attn = scores.softmax(-1, Kind::Float).to_kind(q.kind());
        let out = attn.matmul(&v).apply(&self.cross_out);
        x + self.alpha_cross.sigmoid() * RESIDUAL_ALPHA_MAX * out
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
pub(crate) const SDE_LATENT_DIM: i64 = TICKERS_COUNT * MODEL_DIM;
pub(crate) const SDE_EPS: f64 = 1e-6;
pub(crate) const LATTICE_ALPHA: f64 = 1.0;
pub(crate) const LATTICE_STD_REG: f64 = 0.01;
pub(crate) const LATTICE_MIN_STD: f64 = 1e-3;
pub(crate) const LATTICE_MAX_STD: f64 = 1.0;
const LOG_STD_INIT: f64 = 0.0;
pub(crate) const ACTION_DIM: i64 = TICKERS_COUNT + 1;
const TIME_CROSS_LAYERS: usize = 1;
const FF_DIM: i64 = 512;
const RESIDUAL_ALPHA_MAX: f64 = 0.5;
const RESIDUAL_ALPHA_INIT: f64 = -4.0;

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
// Enriched patch dim: max_patch_size (values) + 3 (mean, std, slope)
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

const NUM_VALUE_BUCKETS: i64 = 255; // Must be odd due to handling

/// (values, critic_logits, (action_mean, sde_latent))
/// sde_latent: [batch, SDE_LATENT_DIM] flat ticker features for Lattice noise
pub type ModelOutput = (Tensor, Tensor, (Tensor, Tensor));

pub(crate) fn symlog_tensor(x: &Tensor) -> Tensor {
    x.sign() * (x.abs() + 1.0).log()
}

pub(crate) fn symexp_tensor(x: &Tensor) -> Tensor {
    x.sign() * (x.abs().exp() - 1.0)
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
    /// SSM hidden state per layer (batched over tickers)
    pub ssm_states: Vec<Mamba2State>,
    /// Whether initialized with full sequence
    pub initialized: bool,
}

pub struct TradingModel {
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    patch_ln_weight: Tensor,
    patch_dt_scale: Tensor,
    patch_sizes: Tensor,
    patch_config_ids: Tensor,
    patch_pos_embed: Tensor,
    ssm_layers: Vec<StatefulMambaRef>,
    ssm_norms: Vec<RMSNorm>,
    ssm_final_norm: RMSNorm,
    exo_global_proj: nn::Linear,
    exo_ticker_proj: nn::Linear,
    exo_cross_blocks: Vec<ExoCrossBlock>,
    inter_ticker_block: InterTickerBlock,
    actor_score: nn::Linear,
    cash_bias: Tensor,
    actor_out: nn::Linear,
    value_out: nn::Linear,
    // Lattice exploration: learned log-std for correlated + independent noise
    log_std_param: Tensor,  // [SDE_LATENT_DIM, SDE_LATENT_DIM + ACTION_DIM]
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
        let log_std = self.log_std_param
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
        let max_input_dim = max_patch_size + PATCH_SCALAR_FEATS;
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
        let patch_sizes = {
            let mut sizes = Vec::with_capacity(SEQ_LEN as usize);
            for &(days, patch_size) in &PATCH_CONFIGS {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    sizes.push(patch_size as f32);
                }
            }
            Tensor::from_slice(&sizes)
                .view([1, SEQ_LEN, 1])
                .to_device(p.device())
        };
        let patch_config_ids = {
            let mut ids = Vec::with_capacity(SEQ_LEN as usize);
            for (cfg_idx, &(days, patch_size)) in PATCH_CONFIGS.iter().enumerate() {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    ids.push(cfg_idx as i64);
                }
            }
            Tensor::from_slice(&ids).to_kind(Kind::Int64).to_device(p.device())
        };
        let patch_pos_embed = p.var(
            "patch_pos_embed",
            &[SEQ_LEN, SSM_DIM],
            Init::Const(0.0),
        );

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
        let ssm_final_norm = RMSNorm::new(&(p / "ssm_final_norm"), SSM_DIM, 1e-6);

        let exo_global_proj = nn::linear(
            p / "exo_global_proj",
            GLOBAL_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let exo_ticker_proj = nn::linear(
            p / "exo_ticker_proj",
            PER_TICKER_STATIC_OBS as i64,
            MODEL_DIM,
            Default::default(),
        );
        let exo_cross_blocks = EXO_CROSS_AFTER
            .iter()
            .enumerate()
            .map(|(i, _)| ExoCrossBlock::new(&(p / format!("exo_cross_{}", i))))
            .collect::<Vec<_>>();
        let inter_ticker_block = InterTickerBlock::new(&(p / "inter_ticker_0"));
        let actor_score = nn::linear(p / "actor_score", MODEL_DIM, 1, nn::LinearConfig {
            ws_init: truncated_normal_init(MODEL_DIM, 1),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        });
        let cash_bias = p.var("cash_bias", &[1], Init::Const(0.0));
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
        let value_out = nn::linear(
            p / "value_out",
            TICKERS_COUNT * MODEL_DIM,
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
        // DreamerV3-style exponential bin spacing: symexp(linspace(-VALUE_LOG_CLIP, 0))
        // Dense near zero (where most returns land), sparse at extremes
        let half_n = (NUM_VALUE_BUCKETS - 1) / 2 + 1; // 128
        let neg_half = Tensor::linspace(-VALUE_LOG_CLIP, 0.0, half_n, (Kind::Float, p.device()));
        let neg_half = symexp_tensor(&neg_half);
        let pos_half = neg_half.narrow(0, 0, half_n - 1).flip([0]).neg();
        let value_centers = Tensor::cat(&[neg_half, pos_half], 0);
        let bucket_centers = value_centers.shallow_clone();
        Self {
            patch_embed_weight,
            patch_embed_bias,
            patch_ln_weight,
            patch_dt_scale,
            patch_sizes,
            patch_config_ids,
            patch_pos_embed,
            ssm_layers,
            ssm_norms,
            ssm_final_norm,
            exo_global_proj,
            exo_ticker_proj,
            exo_cross_blocks,
            inter_ticker_block,
            actor_score,
            cash_bias,
            actor_out,
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
            self.exo_cross_blocks[pos].forward(x, exo_kv)
        } else {
            x.shallow_clone()
        }
    }

    /// Build exogenous KV bank: [batch*tickers, 2, MODEL_DIM]
    /// Token 0 = projected global static, Token 1 = projected per-ticker static
    fn build_exo_kv(&self, global_static: &Tensor, per_ticker_static: &Tensor, batch_size: i64) -> Tensor {
        let global_emb = global_static.apply(&self.exo_global_proj); // [batch, MODEL_DIM]
        let ticker_emb = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.exo_ticker_proj)
            .reshape([batch_size, TICKERS_COUNT, MODEL_DIM]); // [batch, tickers, MODEL_DIM]
        let global_exp = global_emb
            .unsqueeze(1)
            .expand(&[batch_size, TICKERS_COUNT, MODEL_DIM], false); // [batch, tickers, MODEL_DIM]
        Tensor::stack(&[global_exp, ticker_emb], 2)
            .reshape([batch_size * TICKERS_COUNT, 2, MODEL_DIM]) // [batch*tickers, 2, MODEL_DIM]
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
        let leading = prefix.sum_dim_intlist([1].as_slice(), false, Kind::Float).to_kind(Kind::Int64);
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
        let mut enriched_parts = Vec::with_capacity(PATCH_CONFIGS.len());
        let mut delta_offset = 0i64;
        for &(days, patch_size) in &PATCH_CONFIGS {
            let n_patches = days / patch_size;
            let patches = deltas
                .narrow(1, delta_offset, days)
                .view([batch, n_patches, patch_size])
                .to_kind(Kind::Float);
            let mean = patches.mean_dim([2].as_slice(), true, Kind::Float);
            let var = (&patches - &mean).pow_tensor_scalar(2.0)
                .mean_dim([2].as_slice(), true, Kind::Float);
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
        let out = Tensor::einsum("blm,lmd->bld", &[&enriched, &weight_per_patch], None::<&[i64]>);
        let out = out + bias_per_patch.unsqueeze(0);

        // RMSNorm with per-config scale
        let ln_weight = self.maybe_to_device_kind(&self.patch_ln_weight, device, Kind::Float);
        let ln_weight = ln_weight.index_select(0, &config_ids).unsqueeze(0);
        let rms = (out.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float) + 1e-5).sqrt();
        (out / rms * ln_weight).to_kind(kind)
    }

    /// Single-config embedding for streaming inference (one patch at a time).
    fn embed_patch_config(&self, patches: &Tensor, config_idx: i64) -> Tensor {
        let kind = patches.kind();
        let patches_f = patches.to_kind(Kind::Float);
        let patch_len = patches_f.size()[2];
        let mean = patches_f.mean_dim([2].as_slice(), true, Kind::Float);
        let var = (&patches_f - &mean).pow_tensor_scalar(2.0)
            .mean_dim([2].as_slice(), true, Kind::Float);
        let std = (var + 1e-5).sqrt();
        let first = patches_f.narrow(2, 0, 1);
        let last = patches_f.narrow(2, patch_len - 1, 1);
        let slope = &last - &first;
        let enriched = Tensor::cat(&[&patches_f, &mean, &std, &slope], 2);
        let input_dim = patch_len + PATCH_SCALAR_FEATS;
        let weight = self.patch_embed_weight.get(config_idx).narrow(0, 0, input_dim);
        let bias = self.patch_embed_bias.get(config_idx);
        let out = enriched.matmul(&weight) + bias;
        let ln_weight = self.patch_ln_weight.get(config_idx);
        let rms = (out.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float) + 1e-5).sqrt();
        (out / rms * ln_weight).to_kind(kind)
    }
}
