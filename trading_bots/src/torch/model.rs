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
const PATCH_SIZE: i64 = 8;
const PATCHES_PER_TICKER: i64 = PRICE_DELTAS_PER_TICKER as i64 / PATCH_SIZE;
const COARSE_LATENTS: i64 = 32;
const MID_LATENTS: i64 = 64;
const MID_RAW_STEPS: i64 = 1024;
const MID_PATCHES: i64 = MID_RAW_STEPS / PATCH_SIZE;
const TAIL_PATCHES: i64 = 32;
const STEM_SEQ_LEN: i64 = COARSE_LATENTS + MID_LATENTS + TAIL_PATCHES;
const INPUT_BUFFER_SIZE: i64 = PATCH_SIZE;
const COARSE_DECAY: f64 = 0.9975;
const MID_DECAY: f64 = 0.9922;

// (critic_value, critic_logits, (action_mean, action_log_std, divisor), attn_weights)
pub type ModelOutput = (Tensor, Tensor, (Tensor, Tensor, Tensor), Tensor);

/// Streaming state for O(1) inference and training rollouts with memory
pub struct StreamState {
    pub input_buffer: Tensor,
    pub buffer_pos: i64,
    pub stream_mem_initialized: bool,
    pub tail_pos: i64,
    pub coarse_rp: Tensor,
    pub mid_rp: Tensor,
    pub tail_tokens: Tensor,
    /// Per-ticker states for O(1) step inference
    pub ssm_states: Vec<Mamba2State>,
    /// Batched state for GPU-efficient forward_with_state (training rollouts)
    pub ssm_state_batched: Mamba2State,
    pub last_output: Option<ModelOutput>,
}

impl StreamState {
    pub fn reset(&mut self) {
        let _ = self.input_buffer.zero_();
        self.buffer_pos = 0;
        self.stream_mem_initialized = false;
        self.tail_pos = 0;
        let _ = self.coarse_rp.zero_();
        let _ = self.mid_rp.zero_();
        let _ = self.tail_tokens.zero_();
        for s in &mut self.ssm_states {
            s.reset();
        }
        self.ssm_state_batched.reset();
        self.last_output = None;
    }
}

struct LatentPool {
    ln_tokens: nn::LayerNorm,
    ln_latents: nn::LayerNorm,
    latent_tokens: Tensor,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
}

impl LatentPool {
    fn new(p: &nn::Path, d_model: i64, num_latents: i64) -> Self {
        let ln_tokens = nn::layer_norm(p / "ln_tokens", vec![d_model], Default::default());
        let ln_latents = nn::layer_norm(p / "ln_latents", vec![d_model], Default::default());
        let latent_tokens =
            p.var("latent_tokens", &[num_latents, d_model], Init::Uniform { lo: -0.1, up: 0.1 });
        let q_proj = nn::linear(p / "q", d_model, d_model, Default::default());
        let k_proj = nn::linear(p / "k", d_model, d_model, Default::default());
        let v_proj = nn::linear(p / "v", d_model, d_model, Default::default());
        let out_proj = nn::linear(p / "out", d_model, d_model, Default::default());
        Self {
            ln_tokens,
            ln_latents,
            latent_tokens,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        }
    }

    fn forward_impl(&self, tokens: &Tensor, freeze_k: bool, freeze_v: bool) -> Tensor {
        debug_assert_eq!(tokens.dim(), 3);
        let batch = tokens.size()[0];
        let tokens = tokens.apply(&self.ln_tokens);
        let latents = self
            .latent_tokens
            .unsqueeze(0)
            .expand(&[batch, -1, -1], false)
            .apply(&self.ln_latents);

        let q = latents.apply(&self.q_proj);
        let k = if freeze_k {
            let ws = self.k_proj.ws.detach();
            let bs = self.k_proj.bs.as_ref().map(|b| b.detach());
            match &bs {
                Some(b) => tokens.linear(&ws, Some(b)),
                None => tokens.linear(&ws, None::<&Tensor>),
            }
        } else {
            tokens.apply(&self.k_proj)
        };
        let v = if freeze_v {
            let ws = self.v_proj.ws.detach();
            let bs = self.v_proj.bs.as_ref().map(|b| b.detach());
            match &bs {
                Some(b) => tokens.linear(&ws, Some(b)),
                None => tokens.linear(&ws, None::<&Tensor>),
            }
        } else {
            tokens.apply(&self.v_proj)
        };

        let scale = (q.size()[2] as f64).sqrt();
        let attn = (q.matmul(&k.transpose(-2, -1)) / scale).softmax(-1, Kind::Float);
        attn.matmul(&v).apply(&self.out_proj)
    }

    fn forward(&self, tokens: &Tensor) -> Tensor {
        self.forward_impl(tokens, false, false)
    }

    fn forward_random_proj(&self, tokens: &Tensor, freeze_v: bool) -> Tensor {
        self.forward_impl(tokens, true, freeze_v)
    }
}

struct TwoStageLatentPool {
    random_proj: LatentPool,
    pool: LatentPool,
    freeze_rp_v: bool,
}

impl TwoStageLatentPool {
    fn new(p: &nn::Path, d_model: i64, num_latents: i64, freeze_rp_v: bool) -> Self {
        Self {
            random_proj: LatentPool::new(&(p / "rp"), d_model, num_latents),
            pool: LatentPool::new(&(p / "pool"), d_model, num_latents),
            freeze_rp_v,
        }
    }

    fn forward(&self, tokens: &Tensor) -> Tensor {
        let rp_latents = self.random_proj.forward_random_proj(tokens, self.freeze_rp_v);
        self.pool.forward(&rp_latents)
    }

    fn init_rp_values(&self, batch: i64) -> Tensor {
        self.random_proj
            .latent_tokens
            .detach()
            .unsqueeze(0)
            .expand(&[batch, -1, -1], false)
            .contiguous()
    }

    fn update_rp_values(&self, rp_values: &Tensor, token: &Tensor, decay: f64) -> Tensor {
        debug_assert_eq!(rp_values.dim(), 3);
        debug_assert_eq!(token.dim(), 2);
        debug_assert_eq!(rp_values.size()[0], token.size()[0]);

        let q = token.apply(&self.random_proj.q_proj).to_kind(Kind::Float); // [B, D]
        let k_frozen = self.random_proj.latent_tokens.detach().to_kind(Kind::Float); // [M, D]
        let scores = q.matmul(&k_frozen.transpose(0, 1)) / (q.size()[1] as f64).sqrt(); // [B, M]
        let attn = scores.softmax(-1, Kind::Float);

        let token_v = if self.freeze_rp_v {
            let ws = self.random_proj.v_proj.ws.detach();
            let bs = self.random_proj.v_proj.bs.as_ref().map(|b| b.detach());
            match &bs {
                Some(b) => token.linear(&ws, Some(b)),
                None => token.linear(&ws, None::<&Tensor>),
            }
        } else {
            token.apply(&self.random_proj.v_proj)
        };
        let token_v = token_v.to_kind(Kind::Float);

        let updated = rp_values.to_kind(Kind::Float) * decay + attn.unsqueeze(-1) * token_v.unsqueeze(1);
        updated.apply(&self.random_proj.ln_latents).to_kind(rp_values.kind())
    }

    fn forward_from_rp_values(&self, rp_values: &Tensor) -> Tensor {
        self.pool.forward(rp_values)
    }
}

struct MultiBankLatentPool {
    coarse: TwoStageLatentPool,
    mid: TwoStageLatentPool,
}

impl MultiBankLatentPool {
    fn new(p: &nn::Path, d_model: i64, freeze_rp_v: bool) -> Self {
        assert_eq!(MID_RAW_STEPS % PATCH_SIZE, 0);
        Self {
            coarse: TwoStageLatentPool::new(&(p / "coarse"), d_model, COARSE_LATENTS, freeze_rp_v),
            mid: TwoStageLatentPool::new(&(p / "mid"), d_model, MID_LATENTS, freeze_rp_v),
        }
    }
}

pub struct TradingModel {
    patch_embed: nn::Linear,
    ln_patch: nn::LayerNorm,
    latent_pool: MultiBankLatentPool,
    stem_pos_emb: Tensor,
    stem_type_emb: Tensor,
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
        assert_eq!(PRICE_DELTAS_PER_TICKER as i64 % PATCH_SIZE, 0);
        assert!(TAIL_PATCHES <= PATCHES_PER_TICKER);
        assert!(MID_PATCHES <= PATCHES_PER_TICKER - TAIL_PATCHES);
        let patch_embed = nn::linear(p / "patch_embed", PATCH_SIZE, SSM_DIM, Default::default());
        let ln_patch = nn::layer_norm(p / "ln_patch", vec![SSM_DIM], Default::default());
        let latent_pool = MultiBankLatentPool::new(&(p / "latent_pool"), SSM_DIM, false);
        let stem_pos_emb = p.var(
            "stem_pos_emb",
            &[1, STEM_SEQ_LEN, SSM_DIM],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );
        let stem_type_emb = p.var(
            "stem_type_emb",
            &[3, SSM_DIM],
            Init::Uniform { lo: -0.01, up: 0.01 },
        );

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
            nn::linear(p / "temporal_pool_0", 128, 1, Default::default()),
            nn::linear(p / "temporal_pool_1", 128, 1, Default::default()),
            nn::linear(p / "temporal_pool_2", 128, 1, Default::default()),
            nn::linear(p / "temporal_pool_3", 128, 1, Default::default()),
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
            patch_embed,
            ln_patch,
            latent_pool,
            stem_pos_emb,
            stem_type_emb,
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
    pub fn forward(&self, price_deltas: &Tensor, static_features: &Tensor, train: bool) -> ModelOutput {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.patch_latent_stem(&price_deltas, batch_size);

        let x_for_ssm = x_stem.permute([0, 2, 1]);
        let x_ssm = self.ssm.forward(&x_for_ssm, train);
        let x_ssm = x_ssm.permute([0, 2, 1]);

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    /// Forward with recurrent SSM state - GPU efficient for training rollouts with memory
    pub fn forward_with_state(&self, price_deltas: &Tensor, static_features: &Tensor, _state: &mut StreamState) -> ModelOutput {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.patch_latent_stem(&price_deltas, batch_size);

        let x_for_ssm = x_stem.permute([0, 2, 1]); // [B*T, L, C]
        let x_ssm = self.ssm.forward(&x_for_ssm, false);
        let x_ssm = x_ssm.permute([0, 2, 1]); // [B*T, C, L]

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    /// Batched forward for multiple timesteps - parallelizes conv and head, sequential SSM
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

    /// Initialize streaming state for O(1) inference (batch_size=1)
    pub fn init_stream_state(&self) -> StreamState {
        self.init_stream_state_batched(1)
    }

    /// Initialize streaming state with specific batch size for training rollouts
    pub fn init_stream_state_batched(&self, batch_size: i64) -> StreamState {
        StreamState {
            input_buffer: Tensor::zeros(&[TICKERS_COUNT, INPUT_BUFFER_SIZE], (Kind::Float, self.device)),
            buffer_pos: 0,
            stream_mem_initialized: false,
            tail_pos: 0,
            coarse_rp: Tensor::zeros(&[TICKERS_COUNT, COARSE_LATENTS, SSM_DIM], (Kind::Float, self.device)),
            mid_rp: Tensor::zeros(&[TICKERS_COUNT, MID_LATENTS, SSM_DIM], (Kind::Float, self.device)),
            tail_tokens: Tensor::zeros(&[TICKERS_COUNT, TAIL_PATCHES, SSM_DIM], (Kind::Float, self.device)),
            ssm_states: (0..TICKERS_COUNT).map(|_| self.ssm.init_state(1, self.device)).collect(),
            ssm_state_batched: self.ssm.init_state(batch_size * TICKERS_COUNT, self.device),
            last_output: None,
        }
    }

    /// Streaming step for O(1) inference. Returns (ready, output).
    pub fn step(&self, new_deltas: &Tensor, static_features: &Tensor, state: &mut StreamState) -> (bool, ModelOutput) {
        let new_deltas = new_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);

        let full_obs = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        if (new_deltas.dim() == 1 && new_deltas.size()[0] == full_obs)
            || (new_deltas.dim() == 2 && new_deltas.size()[0] == 1 && new_deltas.size()[1] == full_obs)
        {
            let price = if new_deltas.dim() == 1 { new_deltas.unsqueeze(0) } else { new_deltas };
            let static_features = if static_features.dim() == 1 { static_features.unsqueeze(0) } else { static_features };
            let output = self.forward_with_state(&price, &static_features, state);
            state.last_output = Some(Self::clone_output(&output));
            return (true, output);
        }

        let _ = state.input_buffer.narrow(1, state.buffer_pos, 1).copy_(&new_deltas.unsqueeze(1));
        state.buffer_pos += 1;

        if state.buffer_pos < INPUT_BUFFER_SIZE {
            let output = state.last_output.as_ref().map(Self::clone_output).unwrap_or_else(|| self.zero_output());
            return (false, output);
        }

        state.buffer_pos = 0;
        let buffer = state.input_buffer.shallow_clone();
        let output = self.process_stream_buffer(&buffer, &static_features, state);
        state.last_output = Some(Self::clone_output(&output));
        (true, output)
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

    fn process_stream_buffer(&self, buffer: &Tensor, static_features: &Tensor, state: &mut StreamState) -> ModelOutput {
        self.ensure_stream_mem_initialized(state);

        let token = buffer.apply(&self.patch_embed).apply(&self.ln_patch).silu(); // [T, D]
        self.update_stream_mem(state, &token);
        self.stream_forward_from_mem(static_features, state)
    }

    fn ensure_stream_mem_initialized(&self, state: &mut StreamState) {
        if state.stream_mem_initialized {
            return;
        }
        state.coarse_rp = self
            .latent_pool
            .coarse
            .init_rp_values(TICKERS_COUNT)
            .to_device(self.device)
            .to_kind(Kind::Float);
        state.mid_rp = self
            .latent_pool
            .mid
            .init_rp_values(TICKERS_COUNT)
            .to_device(self.device)
            .to_kind(Kind::Float);
        let _ = state.tail_tokens.zero_();
        state.tail_pos = 0;
        state.stream_mem_initialized = true;
    }

    fn update_stream_mem(&self, state: &mut StreamState, token: &Tensor) {
        state.coarse_rp = self
            .latent_pool
            .coarse
            .update_rp_values(&state.coarse_rp, token, COARSE_DECAY);
        state.mid_rp = self
            .latent_pool
            .mid
            .update_rp_values(&state.mid_rp, token, MID_DECAY);

        let _ = state
            .tail_tokens
            .narrow(1, state.tail_pos, 1)
            .copy_(&token.unsqueeze(1));
        state.tail_pos = (state.tail_pos + 1) % TAIL_PATCHES;
    }

    fn ordered_tail_tokens(&self, state: &StreamState) -> Tensor {
        if state.tail_pos == 0 {
            state.tail_tokens.shallow_clone()
        } else {
            let a = state.tail_tokens.narrow(1, state.tail_pos, TAIL_PATCHES - state.tail_pos);
            let b = state.tail_tokens.narrow(1, 0, state.tail_pos);
            Tensor::cat(&[a, b], 1)
        }
    }

    fn stream_forward_from_mem(&self, static_features: &Tensor, state: &StreamState) -> ModelOutput {
        let batch_size = 1i64;
        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);

        let coarse = self.latent_pool.coarse.forward_from_rp_values(&state.coarse_rp);
        let mid = self.latent_pool.mid.forward_from_rp_values(&state.mid_rp);
        let tail = self.ordered_tail_tokens(state);

        let batch_tokens = TICKERS_COUNT;
        let type_coarse = self
            .stem_type_emb
            .get(0)
            .view([1, 1, SSM_DIM])
            .expand(&[batch_tokens, COARSE_LATENTS, SSM_DIM], false);
        let type_mid = self
            .stem_type_emb
            .get(1)
            .view([1, 1, SSM_DIM])
            .expand(&[batch_tokens, MID_LATENTS, SSM_DIM], false);
        let type_tail = self
            .stem_type_emb
            .get(2)
            .view([1, 1, SSM_DIM])
            .expand(&[batch_tokens, TAIL_PATCHES, SSM_DIM], false);
        let type_emb = Tensor::cat(&[type_coarse, type_mid, type_tail], 1);
        let pos_emb = self.stem_pos_emb.expand(&[batch_tokens, STEM_SEQ_LEN, SSM_DIM], false);

        let stem_tokens = Tensor::cat(&[coarse, mid, tail], 1) + type_emb + pos_emb; // [T, L, D]
        let x_ssm = self.ssm.forward(&stem_tokens, false).permute([0, 2, 1]); // [T, D, L]
        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    fn parse_static(&self, static_features: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let global = static_features.narrow(1, 0, GLOBAL_STATIC_OBS as i64);
        let per_ticker = static_features
            .narrow(1, GLOBAL_STATIC_OBS as i64, TICKERS_COUNT * PER_TICKER_STATIC_OBS as i64)
            .reshape([batch_size, TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64]);
        (global, per_ticker)
    }

    fn patch_latent_stem(&self, price_deltas: &Tensor, batch_size: i64) -> Tensor {
        let x = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_size * TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_size * TICKERS_COUNT, PATCHES_PER_TICKER, PATCH_SIZE])
            .apply(&self.patch_embed)
            .apply(&self.ln_patch)
            .silu();

        let tail_start = PATCHES_PER_TICKER - TAIL_PATCHES;
        let past_all = x.narrow(1, 0, tail_start);
        let mid_start = tail_start - MID_PATCHES;
        let past_mid = x.narrow(1, mid_start, MID_PATCHES);
        let tail = x.narrow(1, tail_start, TAIL_PATCHES);

        let coarse_latents = self.latent_pool.coarse.forward(&past_all);
        let mid_latents = self.latent_pool.mid.forward(&past_mid);
        let batch_tokens = batch_size * TICKERS_COUNT;
        let type_latent = self
            .stem_type_emb
            .get(0)
            .view([1, 1, SSM_DIM])
            .expand(&[batch_tokens, COARSE_LATENTS, SSM_DIM], false);
        let type_mid = self
            .stem_type_emb
            .get(1)
            .view([1, 1, SSM_DIM])
            .expand(&[batch_tokens, MID_LATENTS, SSM_DIM], false);
        let type_tail = self
            .stem_type_emb
            .get(2)
            .view([1, 1, SSM_DIM])
            .expand(&[batch_tokens, TAIL_PATCHES, SSM_DIM], false);
        let type_emb = Tensor::cat(&[type_latent, type_mid, type_tail], 1);
        let pos_emb = self.stem_pos_emb.expand(&[batch_tokens, STEM_SEQ_LEN, SSM_DIM], false);
        (Tensor::cat(&[coarse_latents, mid_latents, tail], 1) + type_emb + pos_emb).permute([0, 2, 1])
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

        for (group, pool) in x_groups.iter().zip(self.temporal_pools.iter()) {
            let combined = Tensor::cat(&[group.shallow_clone(), static_proj.shallow_clone()], 1);
            let logits = combined.permute([0, 2, 1]).apply(pool).squeeze_dim(-1);
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
        const LOG_STD_OFFSET: f64 = -3.8;
        let latent = actor_feat.apply(&self.sde_fc).apply(&self.ln_sde).tanh();
        let log_std = (&self.log_std_param + LOG_STD_OFFSET).clamp(-4.0, -1.0);
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
