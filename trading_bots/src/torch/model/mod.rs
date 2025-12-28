mod forward;
mod head;
mod inference;
mod rmsnorm;

use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT,
};
use crate::torch::ssm::{stateful_mamba_block, Mamba2State, StatefulMamba};

pub use shared::constants::GLOBAL_MACRO_OBS;

use rmsnorm::RMSNorm;

struct TimeCrossBlock {
    time_attn_q: nn::Linear,
    time_attn_kv: nn::Linear,
    time_attn_out: nn::Linear,
    ln_time: RMSNorm,
    time_mlp_fc1: nn::Linear,
    time_mlp_fc2: nn::Linear,
    alpha_time_attn: Tensor,
    alpha_time_mlp: Tensor,
    cross_attn_qkv: nn::Linear,
    cross_attn_out: nn::Linear,
    ln_cross: RMSNorm,
    cross_mlp_fc1: nn::Linear,
    cross_mlp_fc2: nn::Linear,
    alpha_cross_attn: Tensor,
    alpha_cross_mlp: Tensor,
}

impl TimeCrossBlock {
    fn new(p: &nn::Path, kv_heads: i64, head_dim: i64) -> Self {
        let time_attn_q = nn::linear(p / "time_attn_q", 256, 256, Default::default());
        let time_attn_kv = nn::linear(
            p / "time_attn_kv",
            256,
            2 * kv_heads * head_dim,
            Default::default(),
        );
        let time_attn_out = nn::linear(p / "time_attn_out", 256, 256, Default::default());
        let ln_time = RMSNorm::new(&(p / "ln_time"), 256, 1e-5);
        let time_mlp_fc1 = nn::linear(p / "time_mlp_fc1", 256, 2 * FF_DIM, Default::default());
        let time_mlp_fc2 = nn::linear(p / "time_mlp_fc2", FF_DIM, 256, Default::default());
        let alpha_time_attn = p.var("alpha_time_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_time_mlp = p.var("alpha_time_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let cross_attn_qkv = nn::linear(p / "cross_attn_qkv", 256, 256 * 3, Default::default());
        let cross_attn_out = nn::linear(p / "cross_attn_out", 256, 256, Default::default());
        let ln_cross = RMSNorm::new(&(p / "ln_cross"), 256, 1e-5);
        let cross_mlp_fc1 = nn::linear(p / "cross_mlp_fc1", 256, 2 * FF_DIM, Default::default());
        let cross_mlp_fc2 = nn::linear(p / "cross_mlp_fc2", FF_DIM, 256, Default::default());
        let alpha_cross_attn = p.var("alpha_cross_attn_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        let alpha_cross_mlp = p.var("alpha_cross_mlp_raw", &[1], Init::Const(RESIDUAL_ALPHA_INIT));
        Self {
            time_attn_q,
            time_attn_kv,
            time_attn_out,
            ln_time,
            time_mlp_fc1,
            time_mlp_fc2,
            alpha_time_attn,
            alpha_time_mlp,
            cross_attn_qkv,
            cross_attn_out,
            ln_cross,
            cross_mlp_fc1,
            cross_mlp_fc2,
            alpha_cross_attn,
            alpha_cross_mlp,
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

const SSM_DIM: i64 = 64;
const LOGIT_SCALE_INIT: f64 = 0.3;
pub const LOGIT_SCALE_GROUP: usize = 1;
const USE_SDPA: bool = true;
const SDPA_MIN_LEN: i64 = 64;
const TIME_CROSS_LAYERS: usize = 2;
const FF_DIM: i64 = 512;
const RESIDUAL_ALPHA_MAX: f64 = 0.5;
const RESIDUAL_ALPHA_INIT: f64 = -2.0;
const ROPE_BASE: f64 = 10000.0;

// Uniform patch size for proper streaming support
// 3400 deltas / 34 = 100 tokens
const PATCH_SIZE: i64 = 34;
const SEQ_LEN: i64 = PRICE_DELTAS_PER_TICKER as i64 / PATCH_SIZE;

const _: () = assert!(
    PRICE_DELTAS_PER_TICKER as i64 % PATCH_SIZE == 0,
    "PRICE_DELTAS must be divisible by PATCH_SIZE"
);

// (values, (ticker_mean, ticker_log_std, sde_latent, cash_logit), attn_weights)
pub type ModelOutput = (Tensor, (Tensor, Tensor, Tensor, Tensor), Tensor);

pub struct DebugMetrics {
    pub time_alpha_attn_mean: f64,
    pub time_alpha_mlp_mean: f64,
    pub cross_alpha_attn_mean: f64,
    pub cross_alpha_mlp_mean: f64,
    pub temporal_tau: f64,
    pub temporal_attn_entropy: f64,
    pub cross_ticker_embed_norm: f64,
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
    /// SSM hidden state per ticker
    pub ssm_states: Vec<Mamba2State>,
    /// Cached SSM output sequence: [TICKERS_COUNT, SSM_DIM, SEQ_LEN]
    pub ssm_cache: Tensor,
    /// Whether initialized with full sequence
    pub initialized: bool,
}

pub struct TradingModel {
    patch_embed: nn::Linear,
    patch_ln: RMSNorm,
    pos_emb: Tensor,
    ssm: StatefulMamba,
    ssm_proj: nn::Conv1D,
    pos_embedding: Tensor,
    static_to_ssm: nn::Linear,
    ln_static_ssm: RMSNorm,
    static_proj: nn::Linear,
    ln_static_proj: RMSNorm,
    time_cross_blocks: Vec<TimeCrossBlock>,
    time_pos_proj: nn::Linear,
    time_global_ctx: nn::Linear,
    time_ticker_ctx: nn::Linear,
    cross_ticker_embed: nn::Linear,
    cls_token: Tensor,
    ln_temporal_q: RMSNorm,
    ln_temporal_kv: RMSNorm,
    temporal_q: nn::Linear,
    temporal_k: nn::Linear,
    temporal_v: nn::Linear,
    temporal_last: nn::Linear,
    temporal_gate: nn::Linear,
    temporal_attn_out: nn::Linear,
    temporal_tau_raw: Tensor,
    ln_pool_out: RMSNorm,
    cls_global_ctx: nn::Linear,
    cls_ticker_ctx: nn::Linear,
    global_to_ticker: nn::Linear,
    actor_fc1: nn::Linear,
    ln_actor_fc1: RMSNorm,
    actor_fc2: nn::Linear,
    ln_actor_fc2: RMSNorm,
    actor_out: nn::Linear,
    cash_out: nn::Linear,
    pool_scorer: nn::Linear,
    value_ticker_out: nn::Linear,
    cash_value_out: nn::Linear,
    sde_fc: nn::Linear,
    ln_sde: RMSNorm,
    log_std_param: Tensor,
    logit_scale_raw: Tensor,
    device: tch::Device,
    num_heads: i64,
    kv_heads: i64,
    head_dim: i64,
}

impl TradingModel {
    fn use_sdpa(&self, seq_len: i64) -> bool {
        if !USE_SDPA {
            return false;
        }
        seq_len >= SDPA_MIN_LEN
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

    pub fn new(p: &nn::Path) -> Self {
        let patch_embed = nn::linear(p / "patch_embed", PATCH_SIZE, SSM_DIM, Default::default());
        let patch_ln = RMSNorm::new(&(p / "patch_ln"), SSM_DIM, 1e-5);
        let pos_emb = p.var(
            "pos_emb_stem",
            &[1, SEQ_LEN, SSM_DIM],
            Init::Uniform {
                lo: -0.01,
                up: 0.01,
            },
        );

        let ssm = stateful_mamba_block(&(p / "ssm"), SSM_DIM);
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, 256, 1, Default::default());
        let pos_embedding = p.var(
            "pos_emb",
            &[1, 256, SEQ_LEN],
            Init::Uniform {
                lo: -0.01,
                up: 0.01,
            },
        );

        let static_to_ssm = nn::linear(
            p / "static_to_ssm",
            PER_TICKER_STATIC_OBS as i64,
            SSM_DIM,
            Default::default(),
        );
        let ln_static_ssm = RMSNorm::new(&(p / "ln_static_ssm"), SSM_DIM, 1e-5);

        let static_proj = nn::linear(
            p / "static_proj",
            256 + PER_TICKER_STATIC_OBS as i64,
            256,
            Default::default(),
        );
        let ln_static_proj = RMSNorm::new(&(p / "ln_static_proj"), 256, 1e-5);

        let num_heads = 4i64;
        let kv_heads = 2i64;
        let head_dim = 64i64;
        assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
        let mut time_cross_blocks = Vec::with_capacity(TIME_CROSS_LAYERS);
        for i in 0..TIME_CROSS_LAYERS {
            time_cross_blocks.push(TimeCrossBlock::new(
                &(p / format!("time_cross_{}", i)),
                kv_heads,
                head_dim,
            ));
        }
        let time_pos_proj = nn::linear(p / "time_pos_proj", 4, 256, Default::default());
        let time_global_ctx = nn::linear(
            p / "time_global_ctx",
            GLOBAL_STATIC_OBS as i64,
            256,
            Default::default(),
        );
        let time_ticker_ctx = nn::linear(
            p / "time_ticker_ctx",
            PER_TICKER_STATIC_OBS as i64,
            256,
            Default::default(),
        );
        let cross_ticker_embed = nn::linear(
            p / "cross_ticker_embed",
            PER_TICKER_STATIC_OBS as i64,
            256,
            Default::default(),
        );
        let cls_token = p.var(
            "cls_token",
            &[1, 1, 256],
            Init::Uniform {
                lo: -0.01,
                up: 0.01,
            },
        );
        let ln_temporal_q = RMSNorm::new(&(p / "ln_temporal_q"), 256, 1e-5);
        let ln_temporal_kv = RMSNorm::new(&(p / "ln_temporal_kv"), 256, 1e-5);
        let temporal_q = nn::linear(p / "temporal_q", 256, 256, Default::default());
        let temporal_k = nn::linear(p / "temporal_k", 256, kv_heads * head_dim, Default::default());
        let temporal_v = nn::linear(p / "temporal_v", 256, kv_heads * head_dim, Default::default());
        let temporal_last = nn::linear(p / "temporal_last", 256, 256, Default::default());
        let temporal_gate = nn::linear(p / "temporal_gate", 256, 256, Default::default());
        let temporal_attn_out = nn::linear(p / "temporal_attn_out", 256, 256, Default::default());
        let temporal_tau_raw = p.var("temporal_tau_raw", &[1], Init::Const(0.0));
        let ln_pool_out = RMSNorm::new(&(p / "ln_pool_out"), 256, 1e-5);
        let cls_global_ctx = nn::linear(
            p / "cls_global_ctx",
            GLOBAL_STATIC_OBS as i64,
            256,
            Default::default(),
        );
        let cls_ticker_ctx = nn::linear(
            p / "cls_ticker_ctx",
            PER_TICKER_STATIC_OBS as i64,
            256,
            Default::default(),
        );

        let global_to_ticker = nn::linear(
            p / "global_to_ticker",
            GLOBAL_STATIC_OBS as i64,
            256,
            Default::default(),
        );
        let actor_fc1 = nn::linear(
            p / "actor_fc1",
            256,
            256,
            nn::LinearConfig {
                ws_init: truncated_normal_init(256, 256),
                ..Default::default()
            },
        );
        let ln_actor_fc1 = RMSNorm::new(&(p / "ln_actor_fc1"), 256, 1e-5);
        let actor_fc2 = nn::linear(
            p / "actor_fc2",
            256,
            256,
            nn::LinearConfig {
                ws_init: truncated_normal_init(256, 256),
                ..Default::default()
            },
        );
        let ln_actor_fc2 = RMSNorm::new(&(p / "ln_actor_fc2"), 256, 1e-5);
        let actor_out = nn::linear(
            p / "actor_out",
            256,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(256, 1),
                ..Default::default()
            },
        );
        let cash_out = nn::linear(
            p / "cash_out",
            256,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(256, 1),
                ..Default::default()
            },
        );
        let pool_scorer = nn::linear(
            p / "pool_scorer",
            256,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(256, 1),
                ..Default::default()
            },
        );
        let value_ticker_out = nn::linear(
            p / "value_ticker_out",
            256,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(256, 1),
                ..Default::default()
            },
        );
        let cash_value_out = nn::linear(
            p / "cash_value_out",
            256,
            1,
            nn::LinearConfig {
                ws_init: truncated_normal_init(256, 1),
                ..Default::default()
            },
        );

        const SDE_LATENT_DIM: i64 = 64;
        let sde_fc = nn::linear(p / "sde_fc", 256, SDE_LATENT_DIM, Default::default());
        let ln_sde = RMSNorm::new(&(p / "ln_sde"), SDE_LATENT_DIM, 1e-5);
        let log_std_param = p.var("log_std", &[ACTION_COUNT - 1], Init::Const(0.0));
        let logit_scale_raw = p.set_group(LOGIT_SCALE_GROUP).var(
            "logit_scale_raw",
            &[1],
            Init::Const(LOGIT_SCALE_INIT.ln()),
        );
        Self {
            patch_embed,
            patch_ln,
            pos_emb,
            ssm,
            ssm_proj,
            pos_embedding,
            static_to_ssm,
            ln_static_ssm,
            static_proj,
            ln_static_proj,
            time_cross_blocks,
            time_pos_proj,
            time_global_ctx,
            time_ticker_ctx,
            cross_ticker_embed,
            cls_token,
            ln_temporal_q,
            ln_temporal_kv,
            temporal_q,
            temporal_k,
            temporal_v,
            temporal_last,
            temporal_gate,
            temporal_attn_out,
            temporal_tau_raw,
            ln_pool_out,
            cls_global_ctx,
            cls_ticker_ctx,
            global_to_ticker,
            actor_fc1,
            ln_actor_fc1,
            actor_fc2,
            ln_actor_fc2,
            actor_out,
            cash_out,
            pool_scorer,
            value_ticker_out,
            cash_value_out,
            sde_fc,
            ln_sde,
            log_std_param,
            logit_scale_raw,
            device: p.device(),
            num_heads,
            kv_heads,
            head_dim,
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

    fn apply_rope_single(&self, x: &Tensor, positions: &Tensor) -> Tensor {
        let sizes = x.size();
        let (b, h, s, d) = (sizes[0], sizes[1], sizes[2], sizes[3]);
        let half = self.head_dim / 2;
        let (cos, sin) = self.rope_cos_sin(positions, x.kind(), x.device());
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

    fn per_ticker_static_ssm(&self, per_ticker_static: &Tensor, batch_size: i64) -> Tensor {
        let x = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.static_to_ssm);
        self.ln_static_ssm.forward(&x)
    }

    fn patch_embed_single(&self, ticker_data: &Tensor, static_ssm: &Tensor) -> Tensor {
        let x = ticker_data
            .view([1, SEQ_LEN, PATCH_SIZE])
            .apply(&self.patch_embed);
        let x = self.patch_ln.forward(&x);
        let static_ssm = static_ssm
            .view([1, 1, SSM_DIM])
            .expand(&[1, SEQ_LEN, SSM_DIM], false);
        (x + &self.pos_emb + static_ssm).permute([0, 2, 1])
    }

    fn patch_embed_all_with_static(
        &self,
        price_deltas: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let x = deltas
            .view([batch_tokens, SEQ_LEN, PATCH_SIZE])
            .apply(&self.patch_embed);
        let x = self.patch_ln.forward(&x);

        let pos_emb = self
            .pos_emb
            .expand(&[batch_tokens, SEQ_LEN, SSM_DIM], false);
        let static_ssm = self
            .per_ticker_static_ssm(per_ticker_static, batch_size)
            .unsqueeze(1)
            .expand(&[batch_tokens, SEQ_LEN, SSM_DIM], false);
        (x + pos_emb + static_ssm).permute([0, 2, 1])
    }
}
