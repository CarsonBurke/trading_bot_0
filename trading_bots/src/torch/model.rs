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
// 3400 deltas / 20 = 170 tokens
const PATCH_SIZE: i64 = 20;
const SEQ_LEN: i64 = PRICE_DELTAS_PER_TICKER as i64 / PATCH_SIZE;

const _: () = assert!(PRICE_DELTAS_PER_TICKER as i64 % PATCH_SIZE == 0, "PRICE_DELTAS must be divisible by PATCH_SIZE");

// (critic_value, critic_logits, (action_mean, action_log_std, divisor), attn_weights)
pub type ModelOutput = (Tensor, Tensor, (Tensor, Tensor, Tensor), Tensor);

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
    static_to_temporal: nn::Linear,
    ln_static_temporal: nn::LayerNorm,
    temporal_pools: [nn::Linear; 4],
    static_proj: nn::Linear,
    ln_static_proj: nn::LayerNorm,
    attn_qkv: nn::Linear,
    attn_out: nn::Linear,
    ln_attn: nn::LayerNorm,
    pma_seeds: Tensor,
    pma_kv: nn::Linear,
    pma_q: nn::Linear,
    pma_out: nn::Linear,
    ln_pma: nn::LayerNorm,
    global_to_seed: nn::Linear,
    fc1: nn::Linear,
    ln_fc1: nn::LayerNorm,
    fc2_actor: nn::Linear,
    ln_fc2_actor: nn::LayerNorm,
    fc2_critic: nn::Linear,
    ln_fc2_critic: nn::LayerNorm,
    fc3_actor: nn::Linear,
    ln_fc3_actor: nn::LayerNorm,
    fc3_critic: nn::Linear,
    ln_fc3_critic: nn::LayerNorm,
    critic: nn::Linear,
    bucket_centers: Tensor,
    symlog_centers: Tensor, // For two-hot target computation
    actor_mean: nn::Linear,
    sde_fc: nn::Linear,
    ln_sde: nn::LayerNorm,
    log_std_param: Tensor,
    log_d_raw: Tensor,
    device: tch::Device,
    num_heads: i64,
    head_dim: i64,
    pma_num_seeds: i64,
}

impl TradingModel {
    pub fn new(p: &nn::Path, nact: i64) -> Self {
        let patch_embed = nn::linear(p / "patch_embed", PATCH_SIZE, SSM_DIM, Default::default());
        let patch_ln = nn::layer_norm(p / "patch_ln", vec![SSM_DIM], Default::default());
        let pos_emb = p.var("pos_emb_stem", &[1, SEQ_LEN, SSM_DIM], Init::Uniform { lo: -0.01, up: 0.01 });

        let ssm = stateful_mamba_block(&(p / "ssm"), SSM_DIM);
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, 256, 1, Default::default());
        let pos_embedding = p.var("pos_emb", &[1, 256, SEQ_LEN], Init::Uniform { lo: -0.01, up: 0.01 });

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

        let pma_num_seeds = 4i64;
        let pma_seeds = p.var("pma_seeds", &[pma_num_seeds, 256], Init::Uniform { lo: -0.1, up: 0.1 });
        let pma_kv = nn::linear(p / "pma_kv", 256, 256 * 2, Default::default());
        let pma_q = nn::linear(p / "pma_q", 256, 256, Default::default());
        let pma_out = nn::linear(p / "pma_out", 256, 256, Default::default());
        let ln_pma = nn::layer_norm(p / "ln_pma", vec![256], Default::default());
        let global_to_seed = nn::linear(p / "global_to_seed", GLOBAL_STATIC_OBS as i64, 256, Default::default());

        let fc1 = nn::linear(p / "l1", 1024, 512, nn::LinearConfig {
            ws_init: truncated_normal_init(1024, 512), ..Default::default()
        });
        let ln_fc1 = nn::layer_norm(p / "ln_fc1", vec![512], Default::default());
        let fc2_actor = nn::linear(p / "l2_actor", 512, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(512, 256), ..Default::default()
        });
        let ln_fc2_actor = nn::layer_norm(p / "ln_fc2_actor", vec![256], Default::default());
        let fc2_critic = nn::linear(p / "l2_critic", 512, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(512, 256), ..Default::default()
        });
        let ln_fc2_critic = nn::layer_norm(p / "ln_fc2_critic", vec![256], Default::default());
        let fc3_actor = nn::linear(p / "l3_actor", 256, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 256), ..Default::default()
        });
        let ln_fc3_actor = nn::layer_norm(p / "ln_fc3_actor", vec![256], Default::default());
        let fc3_critic = nn::linear(p / "l3_critic", 256, 256, nn::LinearConfig {
            ws_init: truncated_normal_init(256, 256), ..Default::default()
        });
        let ln_fc3_critic = nn::layer_norm(p / "ln_fc3_critic", vec![256], Default::default());

        const NUM_VALUE_BUCKETS: i64 = 255;
        let critic = nn::linear(p / "cl", 256, NUM_VALUE_BUCKETS, nn::LinearConfig {
            ws_init: truncated_normal_init(256, NUM_VALUE_BUCKETS),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        });
        // Distributional critic bins uniform in symlog space [-symlog(REWARD_RANGE), +symlog(REWARD_RANGE)]
        let symlog_clip = shared::symlog_target_clip();
        let symlog_centers =
            Tensor::linspace(-symlog_clip, symlog_clip, NUM_VALUE_BUCKETS, (Kind::Float, p.device()));
        // bucket_centers in raw space (symexp of symlog) for raw value expectation
        let bucket_centers = &symlog_centers.sign() * (&symlog_centers.abs().exp() - 1.0);

        // Logistic-normal: output K-1 unconstrained dims, append 0 before softmax
        let actor_mean = nn::linear(p / "al_mean", 256, nact - 1, nn::LinearConfig {
            ws_init: Init::Uniform { lo: -0.1, up: 0.1 },
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        });

        const SDE_LATENT_DIM: i64 = 64;
        let sde_fc = nn::linear(p / "sde_fc", 256, SDE_LATENT_DIM, Default::default());
        let ln_sde = nn::layer_norm(p / "ln_sde", vec![SDE_LATENT_DIM], Default::default());
        let log_std_param = p.var("log_std", &[SDE_LATENT_DIM, nact - 1], Init::Const(0.0));
        let log_d_raw = p.var("log_d_raw", &[nact], Init::Const(-0.3));

        Self {
            patch_embed, patch_ln, pos_emb,
            ssm, ssm_proj, pos_embedding,
            static_to_temporal, ln_static_temporal, temporal_pools,
            static_proj, ln_static_proj,
            attn_qkv, attn_out, ln_attn,
            pma_seeds, pma_kv, pma_q, pma_out, ln_pma, global_to_seed,
            fc1, ln_fc1, fc2_actor, ln_fc2_actor, fc2_critic, ln_fc2_critic,
            fc3_actor, ln_fc3_actor, fc3_critic, ln_fc3_critic,
            critic, bucket_centers, symlog_centers, actor_mean, sde_fc, ln_sde, log_std_param, log_d_raw,
            device: p.device(),
            num_heads, head_dim, pma_num_seeds,
        }
    }

    /// Get symlog bucket centers for two-hot target computation
    pub fn symlog_centers(&self) -> &Tensor {
        &self.symlog_centers
    }

    /// Batch forward for training (parallel SSM scan)
    pub fn forward(&self, price_deltas: &Tensor, static_features: &Tensor, _train: bool) -> ModelOutput {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.patch_embed_all(&price_deltas, batch_size);

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

        // Patch complete: process new token through SSM
        if state.patch_pos >= PATCH_SIZE {
            state.patch_pos = 0;
            self.process_new_patch(state);
            let _ = state.patch_buf.zero_();
        }

        // Head uses cached SSM output + current static features
        let static_features = if static_features.dim() == 1 { static_features.unsqueeze(0) } else { static_features };
        let (global_static, per_ticker_static) = self.parse_static(&static_features, 1);
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

        // Process each ticker through SSM to capture final states
        for t in 0..TICKERS_COUNT as usize {
            let ticker_data = reshaped.get(t as i64).unsqueeze(0);
            let x_stem = self.patch_embed_single(&ticker_data);
            let x_ssm = self.ssm.forward_with_state(&x_stem.permute([0, 2, 1]), &mut state.ssm_states[t]);
            let _ = state.ssm_cache.get(t as i64).copy_(&x_ssm.squeeze_dim(0).permute([1, 0]));
        }

        state.initialized = true;
        self.head_with_temporal_pool(&state.ssm_cache, &global_static, &per_ticker_static, 1)
    }

    /// Process new patch through SSM, update cache by shifting and appending
    fn process_new_patch(&self, state: &mut StreamState) {
        // Embed the new patch: [TICKERS_COUNT, PATCH_SIZE] -> [TICKERS_COUNT, SSM_DIM]
        let patch_emb = state.patch_buf
            .view([TICKERS_COUNT, 1, PATCH_SIZE])
            .apply(&self.patch_embed)
            .apply(&self.patch_ln)
            .squeeze_dim(1);

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

    /// Embed single ticker data: [1, PRICE_DELTAS] -> [1, SSM_DIM, SEQ_LEN]
    fn patch_embed_single(&self, ticker_data: &Tensor) -> Tensor {
        let x = ticker_data
            .view([1, SEQ_LEN, PATCH_SIZE])
            .apply(&self.patch_embed)
            .apply(&self.patch_ln);
        (x + &self.pos_emb).permute([0, 2, 1])
    }

    /// Embed all tickers: [B, TICKERS*PRICE_DELTAS] -> [B*TICKERS, SSM_DIM, SEQ_LEN]
    fn patch_embed_all(&self, price_deltas: &Tensor, batch_size: i64) -> Tensor {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let x = deltas
            .view([batch_tokens, SEQ_LEN, PATCH_SIZE])
            .apply(&self.patch_embed)
            .apply(&self.patch_ln);

        let pos_emb = self.pos_emb.expand(&[batch_tokens, SEQ_LEN, SSM_DIM], false);
        (x + pos_emb).permute([0, 2, 1])
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

        let seeds = self.pma_seeds.unsqueeze(0) + global_static.apply(&self.global_to_seed).unsqueeze(1);
        let pma_q = seeds.apply(&self.pma_q);
        let pma_kv = ticker_features.apply(&self.pma_kv).chunk(2, 2);
        let (pma_k, pma_v) = (&pma_kv[0], &pma_kv[1]);

        let q_h = pma_q.reshape([batch_size, self.pma_num_seeds, self.num_heads, self.head_dim]).permute([0, 2, 1, 3]);
        let k_h = pma_k.reshape([batch_size, TICKERS_COUNT, self.num_heads, self.head_dim]).permute([0, 2, 1, 3]);
        let v_h = pma_v.reshape([batch_size, TICKERS_COUNT, self.num_heads, self.head_dim]).permute([0, 2, 1, 3]);
        let pma_attn = (q_h.matmul(&k_h.transpose(-2, -1)) / (self.head_dim as f64).sqrt()).softmax(-1, q_h.kind());
        let pma_out = pma_attn.matmul(&v_h).permute([0, 2, 1, 3]).contiguous().view([batch_size, self.pma_num_seeds, 256]).apply(&self.pma_out);
        let pooled = (seeds + pma_out).apply(&self.ln_pma).reshape([batch_size, self.pma_num_seeds * 256]);

        let fc1 = pooled.apply(&self.fc1).apply(&self.ln_fc1).silu();
        let actor_fc2 = fc1.apply(&self.fc2_actor).apply(&self.ln_fc2_actor).silu();
        let critic_fc2 = fc1.apply(&self.fc2_critic).apply(&self.ln_fc2_critic).silu();
        let actor_feat = (actor_fc2.apply(&self.fc3_actor) + &actor_fc2).apply(&self.ln_fc3_actor).silu();
        let critic_feat = (critic_fc2.apply(&self.fc3_critic) + &critic_fc2).apply(&self.ln_fc3_critic).silu();

        let critic_logits = critic_feat.apply(&self.critic);
        let critic_probs = critic_logits.softmax(-1, Kind::Float);
        // Expectation in raw space (bins are uniform in symlog space for stability)
        let critic_value = critic_probs.mv(&self.bucket_centers);

        let action_mean = actor_feat.apply(&self.actor_mean);
        const LOG_STD_OFFSET: f64 = -2.0;
        let latent = actor_feat.apply(&self.sde_fc).apply(&self.ln_sde).tanh();
        let log_std = (&self.log_std_param + LOG_STD_OFFSET).clamp(-3.0, -0.5);
        let variance = latent.pow_tensor_scalar(2).matmul(&log_std.exp().pow_tensor_scalar(2));
        let action_log_std = (variance + 1e-6).sqrt().log().clamp(-10.0, 2.0);

        const LOG_D_RAW_SCALE: f64 = 5.0;
        let divisor = self.log_d_raw.g_mul_scalar(LOG_D_RAW_SCALE).softplus() + 0.1;

        (critic_value, critic_logits, (action_mean, action_log_std, divisor), attn_out_vis)
    }
}