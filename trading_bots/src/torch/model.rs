use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{
    GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT,
};
use crate::torch::ssm::{stateful_mamba_block, Mamba2State, StatefulMamba};

pub use shared::constants::GLOBAL_MACRO_OBS;

fn truncated_normal_init(in_features: i64, out_features: i64) -> Init {
    let denoms = (in_features + out_features) as f64 / 2.0;
    let std = (1.0 / denoms).sqrt() / 0.8796;
    Init::Randn { mean: 0.0, stdev: std }
}

const SSM_DIM: i64 = 64;

// Uniform patch size for proper streaming support
// 3400 deltas / 34 = 100 tokens
const PATCH_SIZE: i64 = 34;
const SEQ_LEN: i64 = PRICE_DELTAS_PER_TICKER as i64 / PATCH_SIZE;

const _: () = assert!(PRICE_DELTAS_PER_TICKER as i64 % PATCH_SIZE == 0, "PRICE_DELTAS must be divisible by PATCH_SIZE");

// (values, (action_mean, action_log_std, divisor), attn_weights)
pub type ModelOutput = (Tensor, (Tensor, Tensor, Tensor), Tensor);

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

impl StreamState {
    pub fn reset(&mut self) {
        let _ = self.delta_ring.zero_();
        self.ring_pos = 0;
        let _ = self.patch_buf.zero_();
        self.patch_pos = 0;
        for s in &mut self.ssm_states {
            s.reset();
        }
        let _ = self.ssm_cache.zero_();
        self.initialized = false;
    }
}

pub struct TradingModel {
    patch_embed: nn::Linear,
    patch_ln: nn::LayerNorm,
    pos_emb: Tensor,
    ssm: StatefulMamba,
    ssm_proj: nn::Conv1D,
    pos_embedding: Tensor,
    static_to_ssm: nn::Linear,
    ln_static_ssm: nn::LayerNorm,
    static_to_temporal: nn::Linear,
    ln_static_temporal: nn::LayerNorm,
    temporal_pools: [nn::Linear; 4],
    static_proj: nn::Linear,
    ln_static_proj: nn::LayerNorm,
    attn_qkv: nn::Linear,
    attn_out: nn::Linear,
    ln_attn: nn::LayerNorm,
    global_to_ticker: nn::Linear,
    ticker_ff1: nn::Linear,
    ticker_ff2: nn::Linear,
    ln_ticker_ff: nn::LayerNorm,
    actor_fc1: nn::Linear,
    ln_actor_fc1: nn::LayerNorm,
    actor_fc2: nn::Linear,
    ln_actor_fc2: nn::LayerNorm,
    actor_out: nn::Linear,
    pool_scorer: nn::Linear,
    value_ticker_out: nn::Linear,
    value_cash_out: nn::Linear,
    sde_fc: nn::Linear,
    ln_sde: nn::LayerNorm,
    log_std_param: Tensor,
    log_d_raw: Tensor,
    device: tch::Device,
    num_heads: i64,
    head_dim: i64,
}

impl TradingModel {
    pub fn new(p: &nn::Path, nact: i64) -> Self {
        let patch_embed = nn::linear(p / "patch_embed", PATCH_SIZE, SSM_DIM, Default::default());
        let patch_ln = nn::layer_norm(p / "patch_ln", vec![SSM_DIM], Default::default());
        let pos_emb = p.var("pos_emb_stem", &[1, SEQ_LEN, SSM_DIM], Init::Uniform { lo: -0.01, up: 0.01 });

        let ssm = stateful_mamba_block(&(p / "ssm"), SSM_DIM);
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, 256, 1, Default::default());
        let pos_embedding = p.var("pos_emb", &[1, 256, SEQ_LEN], Init::Uniform { lo: -0.01, up: 0.01 });

        let static_to_ssm = nn::linear(p / "static_to_ssm", PER_TICKER_STATIC_OBS as i64, SSM_DIM, Default::default());
        let ln_static_ssm = nn::layer_norm(p / "ln_static_ssm", vec![SSM_DIM], Default::default());
        let static_to_temporal = nn::linear(p / "static_to_temporal", PER_TICKER_STATIC_OBS as i64, 64, Default::default());
        let ln_static_temporal = nn::layer_norm(p / "ln_static_temporal", vec![64], Default::default());
        let temporal_pools = [
            nn::linear(p / "temporal_pool_0", 320, 1, Default::default()),
            nn::linear(p / "temporal_pool_1", 320, 1, Default::default()),
            nn::linear(p / "temporal_pool_2", 320, 1, Default::default()),
            nn::linear(p / "temporal_pool_3", 320, 1, Default::default()),
        ];

        let static_proj = nn::linear(p / "static_proj", 256 + PER_TICKER_STATIC_OBS as i64, 256, Default::default());
        let ln_static_proj = nn::layer_norm(p / "ln_static_proj", vec![256], Default::default());

        let num_heads = 4i64;
        let head_dim = 64i64;
        let attn_qkv = nn::linear(p / "attn_qkv", 256, 256 * 3, Default::default());
        let attn_out = nn::linear(p / "attn_out", 256, 256, Default::default());
        let ln_attn = nn::layer_norm(p / "ln_attn", vec![256], Default::default());

        let global_to_ticker = nn::linear(p / "global_to_ticker", GLOBAL_STATIC_OBS as i64, 256, Default::default());
        let ticker_ff1 = nn::linear(p / "ticker_ff1", 256, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 256), ..Default::default()
        });
        let ticker_ff2 = nn::linear(p / "ticker_ff2", 256, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 256), ..Default::default()
        });
        let ln_ticker_ff = nn::layer_norm(p / "ln_ticker_ff", vec![256], Default::default());
        let actor_fc1 = nn::linear(p / "actor_fc1", 256, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 256), ..Default::default()
        });
        let ln_actor_fc1 = nn::layer_norm(p / "ln_actor_fc1", vec![256], Default::default());
        let actor_fc2 = nn::linear(p / "actor_fc2", 256, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 256), ..Default::default()
        });
        let ln_actor_fc2 = nn::layer_norm(p / "ln_actor_fc2", vec![256], Default::default());
        let actor_out = nn::linear(p / "actor_out", 256, 1, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 1), ..Default::default()
        });
        let pool_scorer = nn::linear(p / "pool_scorer", 256, 1, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 1), ..Default::default()
        });
        let value_ticker_out = nn::linear(p / "value_ticker_out", 256, 1, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 1), ..Default::default()
        });
        let value_cash_out = nn::linear(p / "value_cash_out", 256, 1, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 1), ..Default::default()
        });

        // Logistic-normal: output K-1 unconstrained dims for softmax
        const SDE_LATENT_DIM: i64 = 64;
        let sde_fc = nn::linear(p / "sde_fc", 256, SDE_LATENT_DIM, Default::default());
        let ln_sde = nn::layer_norm(p / "ln_sde", vec![SDE_LATENT_DIM], Default::default());
        let log_std_param = p.var("log_std", &[SDE_LATENT_DIM, 1], Init::Const(-3.0));
        let log_d_raw = p.var("log_d_raw", &[nact], Init::Const(-0.3));

        Self {
            patch_embed, patch_ln, pos_emb,
            ssm, ssm_proj, pos_embedding,
            static_to_ssm, ln_static_ssm,
            static_to_temporal, ln_static_temporal, temporal_pools,
            static_proj, ln_static_proj,
            attn_qkv, attn_out, ln_attn,
            global_to_ticker, ticker_ff1, ticker_ff2, ln_ticker_ff,
            actor_fc1, ln_actor_fc1, actor_fc2, ln_actor_fc2, actor_out,
            pool_scorer, value_ticker_out, value_cash_out,
            sde_fc, ln_sde, log_std_param, log_d_raw,
            device: p.device(),
            num_heads, head_dim,
        }
    }

    /// Batch forward for training (parallel SSM scan)
    pub fn forward(&self, price_deltas: &Tensor, static_features: &Tensor, _train: bool) -> ModelOutput {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.patch_embed_all_with_static(&price_deltas, &per_ticker_static, batch_size);

        let x_for_ssm = x_stem.permute([0, 2, 1]);
        let x_ssm = self.ssm.forward(&x_for_ssm, false);
        let x_ssm = x_ssm.permute([0, 2, 1]);

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    /// Batched forward for multiple timesteps (each sample independent, parallel SSM)
    pub fn forward_sequence_with_state(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _state: &mut StreamState,
    ) -> ModelOutput {
        let seq_len = price_deltas.size()[0];
        let batch_size = price_deltas.size()[1];
        let total_samples = seq_len * batch_size;

        let price_deltas_flat = price_deltas.reshape([total_samples, -1]);
        let static_features_flat = static_features.reshape([total_samples, -1]);
        self.forward(&price_deltas_flat, &static_features_flat, true)
    }

    /// Initialize streaming state (single batch)
    pub fn init_stream_state(&self) -> StreamState {
        StreamState {
            delta_ring: Tensor::zeros(&[TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64], (Kind::Float, self.device)),
            ring_pos: 0,
            patch_buf: Tensor::zeros(&[TICKERS_COUNT, PATCH_SIZE], (Kind::Float, self.device)),
            patch_pos: 0,
            ssm_states: (0..TICKERS_COUNT).map(|_| self.ssm.init_state(1, self.device)).collect(),
            ssm_cache: Tensor::zeros(&[TICKERS_COUNT, SSM_DIM, SEQ_LEN], (Kind::Float, self.device)),
            initialized: false,
        }
    }

    /// Initialize batched streaming state for PPO rollout collection
    /// Note: For training with full observations at each step, state tracking is minimal
    pub fn init_stream_state_batched(&self, batch_size: i64) -> StreamState {
        StreamState {
            delta_ring: Tensor::zeros(&[batch_size * TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64], (Kind::Float, self.device)),
            ring_pos: 0,
            patch_buf: Tensor::zeros(&[batch_size * TICKERS_COUNT, PATCH_SIZE], (Kind::Float, self.device)),
            patch_pos: 0,
            ssm_states: (0..(batch_size * TICKERS_COUNT) as usize).map(|_| self.ssm.init_state(1, self.device)).collect(),
            ssm_cache: Tensor::zeros(&[batch_size * TICKERS_COUNT, SSM_DIM, SEQ_LEN], (Kind::Float, self.device)),
            initialized: false,
        }
    }

    /// Forward with state for rollout collection (uses full observation, state unused)
    pub fn forward_with_state(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _state: &mut StreamState,
    ) -> ModelOutput {
        self.forward(price_deltas, static_features, false)
    }

    /// Streaming inference step - O(1) per delta when patch not ready, O(SEQ_LEN) when patch ready
    /// First call with full observation initializes state, subsequent calls stream one delta at a time
    pub fn step(&self, new_deltas: &Tensor, static_features: &Tensor, state: &mut StreamState) -> ModelOutput {
        let new_deltas = new_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);

        // Full observation: initialize streaming state
        let full_obs = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        let is_full = (new_deltas.dim() == 1 && new_deltas.size()[0] == full_obs)
            || (new_deltas.dim() == 2 && new_deltas.size()[1] == full_obs);

        if is_full {
            return self.init_from_full(&new_deltas, &static_features, state);
        }

        // Streaming: add one delta per ticker
        let new_deltas = if new_deltas.dim() == 1 { new_deltas } else { new_deltas.squeeze_dim(0) };

        // Update ring buffer
        for t in 0..TICKERS_COUNT {
            let _ = state.delta_ring.get(t).narrow(0, state.ring_pos, 1)
                .copy_(&new_deltas.get(t).unsqueeze(0));
        }
        state.ring_pos = (state.ring_pos + 1) % PRICE_DELTAS_PER_TICKER as i64;

        // Accumulate in patch buffer
        let _ = state.patch_buf.narrow(1, state.patch_pos, 1)
            .copy_(&new_deltas.unsqueeze(1));
        state.patch_pos += 1;

        let static_features = if static_features.dim() == 1 { static_features.unsqueeze(0) } else { static_features };
        let (global_static, per_ticker_static) = self.parse_static(&static_features, 1);
        let static_ssm = self.per_ticker_static_ssm(&per_ticker_static, 1);

        // Patch complete: process new token through SSM
        if state.patch_pos >= PATCH_SIZE {
            state.patch_pos = 0;
            self.process_new_patch(state, &static_ssm);
            let _ = state.patch_buf.zero_();
        }

        // Head uses cached SSM output + current static features
        self.head_with_temporal_pool(&state.ssm_cache, &global_static, &per_ticker_static, 1)
    }

    /// Initialize streaming from full observation
    fn init_from_full(&self, price_deltas: &Tensor, static_features: &Tensor, state: &mut StreamState) -> ModelOutput {
        let price = if price_deltas.dim() == 1 { price_deltas.unsqueeze(0) } else { price_deltas.shallow_clone() };
        let static_features = if static_features.dim() == 1 { static_features.unsqueeze(0) } else { static_features.shallow_clone() };

        // Fill ring buffer
        let reshaped = price.view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);
        let _ = state.delta_ring.copy_(&reshaped);
        state.ring_pos = 0;
        state.patch_pos = 0;
        let _ = state.patch_buf.zero_();

        let (global_static, per_ticker_static) = self.parse_static(&static_features, 1);
        let static_ssm = self.per_ticker_static_ssm(&per_ticker_static, 1);

        // Process each ticker through SSM to capture final states
        for t in 0..TICKERS_COUNT as usize {
            let ticker_data = reshaped.get(t as i64).unsqueeze(0);
            let x_stem = self.patch_embed_single(&ticker_data, &static_ssm.get(t as i64));
            let x_ssm = self.ssm.forward_with_state(&x_stem.permute([0, 2, 1]), &mut state.ssm_states[t]);
            let _ = state.ssm_cache.get(t as i64).copy_(&x_ssm.squeeze_dim(0).permute([1, 0]));
        }

        state.initialized = true;
        self.head_with_temporal_pool(&state.ssm_cache, &global_static, &per_ticker_static, 1)
    }

    /// Process new patch through SSM, update cache by shifting and appending
    fn process_new_patch(&self, state: &mut StreamState, static_ssm: &Tensor) {
        // Embed the new patch: [TICKERS_COUNT, PATCH_SIZE] -> [TICKERS_COUNT, SSM_DIM]
        let patch_emb = state.patch_buf
            .view([TICKERS_COUNT, 1, PATCH_SIZE])
            .apply(&self.patch_embed)
            .apply(&self.patch_ln)
            .squeeze_dim(1)
            + static_ssm;

        // Process through SSM step for each ticker
        for t in 0..TICKERS_COUNT as usize {
            let x_in = patch_emb.get(t as i64).unsqueeze(0);
            let out = self.ssm.step(&x_in, &mut state.ssm_states[t]);

            // Shift cache left by 1, append new output
            let old_cache = state.ssm_cache.get(t as i64);
            let shifted = old_cache.narrow(1, 1, SEQ_LEN - 1);
            let _ = old_cache.narrow(1, 0, SEQ_LEN - 1).copy_(&shifted);
            let _ = old_cache.narrow(1, SEQ_LEN - 1, 1).copy_(&out.squeeze_dim(0).unsqueeze(1));
        }
    }

    fn parse_static(&self, static_features: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let global = static_features.narrow(1, 0, GLOBAL_STATIC_OBS as i64);
        let per_ticker = static_features
            .narrow(1, GLOBAL_STATIC_OBS as i64, TICKERS_COUNT * PER_TICKER_STATIC_OBS as i64)
            .reshape([batch_size, TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64]);
        (global, per_ticker)
    }

    fn per_ticker_static_ssm(&self, per_ticker_static: &Tensor, batch_size: i64) -> Tensor {
        per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.static_to_ssm)
            .apply(&self.ln_static_ssm)
    }

    /// Embed single ticker data: [1, PRICE_DELTAS] -> [1, SSM_DIM, SEQ_LEN]
    fn patch_embed_single(&self, ticker_data: &Tensor, static_ssm: &Tensor) -> Tensor {
        let x = ticker_data
            .view([1, SEQ_LEN, PATCH_SIZE])
            .apply(&self.patch_embed)
            .apply(&self.patch_ln);
        let static_ssm = static_ssm.view([1, 1, SSM_DIM]).expand(&[1, SEQ_LEN, SSM_DIM], false);
        (x + &self.pos_emb + static_ssm).permute([0, 2, 1])
    }

    /// Embed all tickers: [B, TICKERS*PRICE_DELTAS] -> [B*TICKERS, SSM_DIM, SEQ_LEN]
    fn patch_embed_all_with_static(&self, price_deltas: &Tensor, per_ticker_static: &Tensor, batch_size: i64) -> Tensor {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let x = deltas
            .view([batch_tokens, SEQ_LEN, PATCH_SIZE])
            .apply(&self.patch_embed)
            .apply(&self.patch_ln);

        let pos_emb = self.pos_emb.expand(&[batch_tokens, SEQ_LEN, SSM_DIM], false);
        let static_ssm = self.per_ticker_static_ssm(per_ticker_static, batch_size)
            .unsqueeze(1)
            .expand(&[batch_tokens, SEQ_LEN, SSM_DIM], false);
        (x + pos_emb + static_ssm).permute([0, 2, 1])
    }

    fn head_with_temporal_pool(&self, x_ssm: &Tensor, global_static: &Tensor, per_ticker_static: &Tensor, batch_size: i64) -> ModelOutput {
        let x = x_ssm.apply(&self.ssm_proj) + &self.pos_embedding;
        let temporal_len = x.size()[2];

        let static_proj = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&self.static_to_temporal)
            .apply(&self.ln_static_temporal)
            .unsqueeze(2)
            .expand(&[-1, -1, temporal_len], false);

        let x_groups = x.chunk(4, 1);
        let mut pooled_groups = Vec::with_capacity(4);
        let mut attn_sum = Tensor::zeros(&[batch_size * TICKERS_COUNT, temporal_len], (x.kind(), x.device()));

        // Combine full x (256) with static (64) for attention logits = 320
        let combined_full = Tensor::cat(&[x.shallow_clone(), static_proj.shallow_clone()], 1);

        for (group, pool) in x_groups.iter().zip(self.temporal_pools.iter()) {
            let logits = combined_full.permute([0, 2, 1]).apply(pool).squeeze_dim(-1);
            let attn = logits.softmax(-1, logits.kind());
            attn_sum = attn_sum + &attn;
            pooled_groups.push(group.bmm(&attn.unsqueeze(-1)).squeeze_dim(-1));
        }

        let pooled = Tensor::cat(&pooled_groups, 1);
        let attn_avg = (attn_sum / 4.0).reshape([batch_size, TICKERS_COUNT, -1]).mean_dim(1, false, x.kind());
        let conv_features = pooled.view([batch_size, TICKERS_COUNT, 256]);

        self.head_common(&conv_features, global_static, per_ticker_static, batch_size, attn_avg)
    }

    fn head_common(&self, conv_features: &Tensor, global_static: &Tensor, per_ticker_static: &Tensor, batch_size: i64, attn_out_vis: Tensor) -> ModelOutput {
        let combined = Tensor::cat(&[conv_features.shallow_clone(), per_ticker_static.shallow_clone()], 2)
            .reshape([batch_size * TICKERS_COUNT, 256 + PER_TICKER_STATIC_OBS as i64])
            .apply(&self.static_proj)
            .apply(&self.ln_static_proj)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, 256]);

        let qkv = combined.apply(&self.attn_qkv).reshape([batch_size, TICKERS_COUNT, 3, self.num_heads, self.head_dim]).permute([2, 0, 3, 1, 4]);
        let (q, k, v) = (qkv.get(0), qkv.get(1), qkv.get(2));
        let attn = (q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt()).softmax(-1, q.kind());
        let attn_out = attn.matmul(&v).permute([0, 2, 1, 3]).contiguous().view([batch_size, TICKERS_COUNT, 256]).apply(&self.attn_out);
        let ticker_features = (combined + attn_out).apply(&self.ln_attn);

        let global_ctx = global_static.apply(&self.global_to_ticker).unsqueeze(1);
        let enriched = &ticker_features + global_ctx;
        let enriched_ff = enriched
            .shallow_clone()
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.ticker_ff1)
            .silu()
            .apply(&self.ticker_ff2)
            .reshape([batch_size, TICKERS_COUNT, 256]);
        let enriched = (enriched_ff + &enriched)
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.ln_ticker_ff)
            .reshape([batch_size, TICKERS_COUNT, 256]);

        let pool_logits = enriched
            .shallow_clone()
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.pool_scorer)
            .reshape([batch_size, TICKERS_COUNT, 1]);
        let pool_weights = pool_logits.softmax(1, Kind::Float);
        let pool_summary = (&enriched * &pool_weights).sum_dim_intlist(1, false, Kind::Float);

        let actor_hidden = enriched
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.actor_fc1)
            .apply(&self.ln_actor_fc1)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, 256]);
        let actor_residual = actor_hidden
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.actor_fc2)
            .reshape([batch_size, TICKERS_COUNT, 256]);
        let actor_feat = (actor_residual + &actor_hidden)
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.ln_actor_fc2)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, 256]);

        let ticker_values = enriched
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.value_ticker_out)
            .reshape([batch_size, TICKERS_COUNT]);
        let cash_value = pool_summary.apply(&self.value_cash_out);
        let values = Tensor::cat(&[ticker_values.shallow_clone(), cash_value], 1);

        const MEAN_SCALE: f64 = 0.5;
        let action_mean = actor_feat.apply(&self.actor_out).squeeze_dim(-1).tanh() * MEAN_SCALE;
        // Soft bounds via tanh: log_std ∈ [LOG_STD_MIN, LOG_STD_MAX] with smooth gradients
        const LOG_STD_MIN: f64 = -5.0; // std = 0.007
        const LOG_STD_MAX: f64 = -0.693; // std = 0.5
        let latent = (&enriched + pool_summary.unsqueeze(1))
            .reshape([batch_size * TICKERS_COUNT, 256])
            .apply(&self.sde_fc)
            .apply(&self.ln_sde)
            .tanh()
            .reshape([batch_size, TICKERS_COUNT, -1]);
        let log_std_raw = self.log_std_param.tanh();
        let log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_raw + 1.0);
        let variance = latent.pow_tensor_scalar(2).matmul(&log_std.exp().pow_tensor_scalar(2));
        let action_log_std = (variance + 1e-6).sqrt().log().clamp(LOG_STD_MIN, LOG_STD_MAX).squeeze_dim(-1);

        const LOG_D_RAW_SCALE: f64 = 5.0;
        let divisor = self.log_d_raw.g_mul_scalar(LOG_D_RAW_SCALE).softplus() + 0.1;

        (values, (action_mean, action_log_std, divisor), attn_out_vis)
    }
}
