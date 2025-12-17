use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{
    GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT,
};
use crate::torch::ssm::{stateful_mamba_block, Mamba2State, StatefulMamba};

fn truncated_normal_init(in_features: i64, out_features: i64) -> Init {
    let denoms = (in_features + out_features) as f64 / 2.0;
    let std = (1.0 / denoms).sqrt() / 0.8796;
    Init::Randn { mean: 0.0, stdev: std }
}

const SSM_DIM: i64 = 64;

// Variable patch sizes: fine resolution for recent, coarse for old (oldest first in array)
// Processed in order: oldest (coarse) → newest (fine)
// Each (days, patch_size) must have days % patch_size == 0
const PATCH_CONFIGS: [(i64, i64); 7] = [
    (1600, 64), // oldest 1600 days, patch_size=64 → 25 tokens
    (1024, 32), // 1024 days, patch_size=32 → 32 tokens
    (512, 16),  // 512 days, patch_size=16 → 32 tokens
    (128, 8),   // 128 days, patch_size=8 → 16 tokens
    (64, 4),    // 64 days, patch_size=4 → 16 tokens
    (32, 2),    // 32 days, patch_size=2 → 16 tokens
    (40, 1),    // newest 40 days, patch_size=1 → 40 tokens
]; // Total: 1600+1024+512+128+64+32+40 = 3400, Tokens: 25+32+32+16+16+16+40 = 177

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
const STEM_SEQ_LEN: i64 = PATCH_TOTALS.1; // 364 tokens
const FINEST_PATCH_SIZE: i64 = PATCH_CONFIGS[PATCH_CONFIGS.len() - 1].1;

const _: () = assert!(PATCH_TOTALS.0 == PRICE_DELTAS_PER_TICKER as i64, "PATCH_CONFIGS days must equal PRICE_DELTAS_PER_TICKER");

// (critic_value, critic_logits, (action_mean, action_log_std, divisor), attn_weights)
pub type ModelOutput = (Tensor, Tensor, (Tensor, Tensor, Tensor), Tensor);

/// Streaming state for incremental inference
/// - Maintains ring buffer of price deltas
/// - SSM state carries compressed history across steps
/// - Only processes new patch when PATCH_SIZE deltas accumulated
pub struct StreamState {
    /// Ring buffer of raw price deltas: [TICKERS_COUNT, PRICE_DELTAS_PER_TICKER]
    pub delta_buffer: Tensor,
    /// Current write position in delta buffer (oldest delta to overwrite)
    pub delta_pos: i64,
    /// Buffer for accumulating deltas until full patch: [TICKERS_COUNT, PATCH_SIZE]
    pub patch_buffer: Tensor,
    /// Position within current patch being built
    pub patch_pos: i64,
    /// SSM states for streaming (one per ticker for O(1) step)
    pub ssm_states: Vec<Mamba2State>,
    /// Batched SSM state for training rollouts
    pub ssm_state_batched: Mamba2State,
    /// Cached output from last full forward (used when patch not ready)
    pub last_output: Option<ModelOutput>,
    /// Whether we've processed at least one full sequence
    pub initialized: bool,
}

impl StreamState {
    pub fn reset(&mut self) {
        let _ = self.delta_buffer.zero_();
        self.delta_pos = 0;
        let _ = self.patch_buffer.zero_();
        self.patch_pos = 0;
        for s in &mut self.ssm_states {
            s.reset();
        }
        self.ssm_state_batched.reset();
        self.last_output = None;
        self.initialized = false;
    }
}

pub struct TradingModel {
    patch_embeds: Vec<nn::Linear>,
    patch_lns: Vec<nn::LayerNorm>,
    stem_pos_emb: Tensor,
    dt_scale: Tensor, // [1, STEM_SEQ_LEN, 1] - patch_size per position for dt scaling
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
        let mut patch_embeds = Vec::new();
        let mut patch_lns = Vec::new();
        for (i, &(_, patch_size)) in PATCH_CONFIGS.iter().enumerate() {
            patch_embeds.push(nn::linear(
                p / format!("patch_embed_{}", i),
                patch_size,
                SSM_DIM,
                Default::default(),
            ));
            patch_lns.push(nn::layer_norm(
                p / format!("patch_ln_{}", i),
                vec![SSM_DIM],
                Default::default(),
            ));
        }

        let stem_pos_emb = p.var(
            "stem_pos_emb",
            &[1, STEM_SEQ_LEN, SSM_DIM],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );

        // dt_scale: patch_size per position [1, L, 1] - used to scale dt in SSM
        let dt_scale = {
            let mut scales = Vec::with_capacity(STEM_SEQ_LEN as usize);
            for &(days, patch_size) in &PATCH_CONFIGS {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    scales.push(patch_size as f32);
                }
            }
            Tensor::from_slice(&scales)
                .view([1, STEM_SEQ_LEN, 1])
                .to_device(p.device())
        };

        let ssm = stateful_mamba_block(&(p / "ssm"), SSM_DIM);
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, 256, 1, Default::default());
        let pos_embedding = p.var(
            "pos_emb",
            &[1, 256, STEM_SEQ_LEN],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );

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
            patch_embeds,
            patch_lns,
            stem_pos_emb,
            dt_scale,
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
        let (x_stem, dt_scale) = self.patch_latent_stem(&price_deltas, batch_size);

        let x_for_ssm = x_stem.permute([0, 2, 1]);
        let x_ssm = self.ssm.forward_with_dt_scale(&x_for_ssm, Some(&dt_scale));
        let x_ssm = x_ssm.permute([0, 2, 1]);

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    /// Forward with recurrent SSM state - GPU efficient for training rollouts with memory
    pub fn forward_with_state(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let (x_stem, dt_scale) = self.patch_latent_stem(&price_deltas, batch_size);

        let x_for_ssm = x_stem.permute([0, 2, 1]);
        let x_ssm = self.ssm.forward_with_state_dt_scale(&x_for_ssm, &mut state.ssm_state_batched, Some(&dt_scale));
        let x_ssm = x_ssm.permute([0, 2, 1]);

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    /// Batched forward for multiple timesteps (no state carry - each sample independent)
    /// price_deltas: [seq_len, batch, features]
    /// static_features: [seq_len, batch, features]
    /// Returns: (critics, action_means, action_log_stds) each [seq_len * batch, ...]
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

    /// Batched forward with initial SSM states for efficient BPTT
    pub fn forward_with_ssm_states(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        ssm_initial_states: &Tensor,
    ) -> (ModelOutput, Tensor) {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let (x_stem, dt_scale) = self.patch_latent_stem(&price_deltas, batch_size);

        let x_for_ssm = x_stem.permute([0, 2, 1]);
        let (x_ssm_out, final_states) = self.ssm.forward_batched_init_dt_scale(&x_for_ssm, ssm_initial_states, Some(&dt_scale));
        let x_ssm = x_ssm_out.permute([0, 2, 1]);

        let output = self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size);
        (output, final_states)
    }

    /// Initialize streaming state for O(1) inference (batch_size=1)
    pub fn init_stream_state(&self) -> StreamState {
        self.init_stream_state_batched(1)
    }

    /// Initialize streaming state with specific batch size for training rollouts
    pub fn init_stream_state_batched(&self, batch_size: i64) -> StreamState {
        StreamState {
            delta_buffer: Tensor::zeros(&[TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64], (Kind::Float, self.device)),
            delta_pos: 0,
            patch_buffer: Tensor::zeros(&[TICKERS_COUNT, FINEST_PATCH_SIZE], (Kind::Float, self.device)),
            patch_pos: 0,
            ssm_states: (0..TICKERS_COUNT).map(|_| self.ssm.init_state(1, self.device)).collect(),
            ssm_state_batched: self.ssm.init_state(batch_size * TICKERS_COUNT, self.device),
            last_output: None,
            initialized: false,
        }
    }

    /// Streaming step for O(1) inference.
    /// Input: new_deltas [TICKERS_COUNT] - one new delta per ticker
    /// Returns (ready, output) where ready=true when a new patch was processed
    pub fn step(&self, new_deltas: &Tensor, static_features: &Tensor, state: &mut StreamState) -> (bool, ModelOutput) {
        let new_deltas = new_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);

        // Handle full observation for initialization
        let full_obs = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        if (new_deltas.dim() == 1 && new_deltas.size()[0] == full_obs)
            || (new_deltas.dim() == 2 && new_deltas.size()[0] == 1 && new_deltas.size()[1] == full_obs)
        {
            return self.init_stream_from_full(new_deltas, static_features, state);
        }

        // Streaming: add one delta per ticker
        // new_deltas shape: [TICKERS_COUNT]
        let new_deltas = if new_deltas.dim() == 1 { new_deltas } else { new_deltas.squeeze_dim(0) };

        // Update ring buffer at current position
        for t in 0..TICKERS_COUNT {
            let _ = state.delta_buffer.get(t).narrow(0, state.delta_pos, 1)
                .copy_(&new_deltas.get(t).unsqueeze(0));
        }
        state.delta_pos = (state.delta_pos + 1) % PRICE_DELTAS_PER_TICKER as i64;

        // Accumulate in patch buffer
        let _ = state.patch_buffer.narrow(1, state.patch_pos, 1)
            .copy_(&new_deltas.unsqueeze(1));
        state.patch_pos += 1;

        // Not enough deltas for a patch yet
        if state.patch_pos < FINEST_PATCH_SIZE {
            let output = state.last_output.as_ref().map(Self::clone_output).unwrap_or_else(|| self.zero_output());
            return (false, output);
        }

        // Patch ready - process it
        state.patch_pos = 0;
        let patch = state.patch_buffer.shallow_clone();
        let output = self.process_single_patch(&patch, &static_features, state);
        state.last_output = Some(Self::clone_output(&output));
        let _ = state.patch_buffer.zero_();
        (true, output)
    }

    fn init_stream_from_full(&self, new_deltas: Tensor, static_features: Tensor, state: &mut StreamState) -> (bool, ModelOutput) {
        let price = if new_deltas.dim() == 1 { new_deltas.unsqueeze(0) } else { new_deltas };
        let static_features = if static_features.dim() == 1 { static_features.unsqueeze(0) } else { static_features };

        // Fill delta buffer with the full sequence
        let reshaped = price.view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);
        let _ = state.delta_buffer.copy_(&reshaped);
        state.delta_pos = 0;
        state.patch_pos = 0;
        let _ = state.patch_buffer.zero_();

        // Initialize SSM states by processing all patches
        let output = self.forward_init_stream(&price, &static_features, state);
        state.last_output = Some(Self::clone_output(&output));
        state.initialized = true;
        (true, output)
    }

    /// Initialize streaming by processing full sequence per-ticker, capturing final SSM states
    fn forward_init_stream(&self, price_deltas: &Tensor, static_features: &Tensor, state: &mut StreamState) -> ModelOutput {
        let batch_size = price_deltas.size()[0];
        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);

        let per_ticker = price_deltas.view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);

        let mut ssm_outputs = Vec::with_capacity(TICKERS_COUNT as usize);
        for t in 0..TICKERS_COUNT as usize {
            let ticker_data = per_ticker.get(0).get(t as i64).unsqueeze(0);
            let (x_stem, dt_scale) = self.patch_single_ticker(&ticker_data);
            let x_ssm = self.ssm.forward_with_state_dt_scale(&x_stem.permute([0, 2, 1]), &mut state.ssm_states[t], Some(&dt_scale));
            ssm_outputs.push(x_ssm.permute([0, 2, 1]));
        }

        let x_ssm = Tensor::cat(&ssm_outputs, 0).unsqueeze(0);
        let x_ssm = x_ssm.view([TICKERS_COUNT, SSM_DIM, STEM_SEQ_LEN]);

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    /// Returns (x_stem [1, D, L], dt_scale [1, L, 1])
    fn patch_single_ticker(&self, ticker_data: &Tensor) -> (Tensor, Tensor) {
        // Data arrives oldest→newest, PATCH_CONFIGS is oldest→newest
        let mut patches = Vec::new();
        let mut offset = 0i64;
        for (i, &(days, patch_size)) in PATCH_CONFIGS.iter().enumerate() {
            let n_patches = days / patch_size;
            let chunk = ticker_data.narrow(1, offset, days);
            let p = chunk
                .view([1, n_patches, patch_size])
                .apply(&self.patch_embeds[i])
                .apply(&self.patch_lns[i]);
            patches.push(p);
            offset += days;
        }
        let x = Tensor::cat(&patches, 1);
        let pos_emb = self.stem_pos_emb.narrow(0, 0, 1);
        let x_stem = (x + pos_emb).permute([0, 2, 1]);
        (x_stem, self.dt_scale.shallow_clone())
    }

    /// Process a single delta through SSM with state carry (streaming inference)
    /// Note: With variable patching, streaming requires buffering deltas until enough for finest patch
    fn process_single_patch(&self, patch: &Tensor, static_features: &Tensor, state: &mut StreamState) -> ModelOutput {
        let static_features = if static_features.dim() == 1 { static_features.unsqueeze(0) } else { static_features.shallow_clone() };
        let (global_static, per_ticker_static) = self.parse_static(&static_features, 1);

        // For streaming: use finest patch size (1) embedding
        let patch_emb = patch.apply(&self.patch_embeds[0]).apply(&self.patch_lns[0]);

        let mut outputs = Vec::with_capacity(TICKERS_COUNT as usize);
        for t in 0..TICKERS_COUNT as usize {
            let x_in = patch_emb.get(t as i64).unsqueeze(0);
            let out = self.ssm.step(&x_in, &mut state.ssm_states[t]);
            outputs.push(out);
        }

        let x_ssm = Tensor::cat(&outputs, 0).unsqueeze(-1);

        // Use streaming head (no temporal pooling - single token per ticker)
        self.head_no_temporal_pool(&x_ssm, &global_static, &per_ticker_static, 1)
    }

    fn clone_output(o: &ModelOutput) -> ModelOutput {
        (o.0.shallow_clone(), o.1.shallow_clone(), (o.2.0.shallow_clone(), o.2.1.shallow_clone(), o.2.2.shallow_clone()), o.3.shallow_clone())
    }

    fn zero_output(&self) -> ModelOutput {
        let z_c = Tensor::zeros(&[1], (Kind::Float, self.device));
        let z_logits = Tensor::zeros(&[1, 255], (Kind::Float, self.device));
        // K-1 dims for logistic-normal
        let z_m = Tensor::zeros(&[1, TICKERS_COUNT], (Kind::Float, self.device));
        let z_s = Tensor::zeros(&[1, TICKERS_COUNT], (Kind::Float, self.device));
        let z_d = Tensor::ones(&[TICKERS_COUNT + 1], (Kind::Float, self.device));
        let z_a = Tensor::zeros(&[1, 1], (Kind::Float, self.device));
        (z_c, z_logits, (z_m, z_s, z_d), z_a)
    }

    fn parse_static(&self, static_features: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let global = static_features.narrow(1, 0, GLOBAL_STATIC_OBS as i64);
        let per_ticker = static_features
            .narrow(1, GLOBAL_STATIC_OBS as i64, TICKERS_COUNT * PER_TICKER_STATIC_OBS as i64)
            .reshape([batch_size, TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64]);
        (global, per_ticker)
    }

    /// Returns (x_stem [B*T, D, L], dt_scale [1, L, 1])
    fn patch_latent_stem(&self, price_deltas: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let batch_tokens = batch_size * TICKERS_COUNT;
        // Data arrives oldest→newest, PATCH_CONFIGS is oldest→newest
        let deltas = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64]);

        let mut patches = Vec::new();
        let mut offset = 0i64;
        for (i, &(days, patch_size)) in PATCH_CONFIGS.iter().enumerate() {
            let n_patches = days / patch_size;
            let chunk = deltas.narrow(1, offset, days);
            let p = chunk
                .view([batch_tokens, n_patches, patch_size])
                .apply(&self.patch_embeds[i])
                .apply(&self.patch_lns[i]);
            patches.push(p);
            offset += days;
        }

        let x = Tensor::cat(&patches, 1);
        let pos_emb = self.stem_pos_emb.expand(&[batch_tokens, STEM_SEQ_LEN, SSM_DIM], false);
        let x_stem = (x + pos_emb).permute([0, 2, 1]);
        (x_stem, self.dt_scale.shallow_clone())
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

    fn head_no_temporal_pool(&self, conv_features: &Tensor, global_static: &Tensor, per_ticker_static: &Tensor, batch_size: i64) -> ModelOutput {
        let dummy_attn = Tensor::zeros(&[batch_size, 1], (Kind::Float, self.device));
        self.head_common(conv_features, global_static, per_ticker_static, batch_size, dummy_attn)
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

pub type Model = Box<dyn Fn(&Tensor, &Tensor, bool) -> ModelOutput>;

pub fn model(p: &nn::Path, nact: i64) -> Model {
    let m = TradingModel::new(p, nact);
    Box::new(move |price_deltas, static_features, train| m.forward(price_deltas, static_features, train))
}
