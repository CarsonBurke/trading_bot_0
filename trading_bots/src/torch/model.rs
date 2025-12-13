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

const fn calc_conv_temporal_len(input_len: i64) -> i64 {
    let after_c1 = (input_len - 8) / 2 + 1;
    let after_c2 = (after_c1 - 5 + 4) / 2 + 1;
    (after_c2 - 3 + 2) / 2 + 1
}

const CONV_TEMPORAL_LEN: i64 = calc_conv_temporal_len(PRICE_DELTAS_PER_TICKER as i64);
const SSM_DIM: i64 = 64;
const INPUT_BUFFER_SIZE: i64 = 8;

pub type ModelOutput = (Tensor, (Tensor, Tensor, Tensor), Tensor);

/// Streaming state for O(1) inference and training rollouts with memory
pub struct StreamState {
    pub input_buffer: Tensor,
    pub buffer_pos: i64,
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
        for s in &mut self.ssm_states {
            s.reset();
        }
        self.ssm_state_batched.reset();
        self.last_output = None;
    }
}

pub struct TradingModel {
    c1: nn::Conv1D,
    gn2: nn::GroupNorm,
    c2_dw: nn::Conv1D,
    c2_pw: nn::Conv1D,
    gn3: nn::GroupNorm,
    c3_dw: nn::Conv1D,
    c3_pw: nn::Conv1D,
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
        let c1 = nn::conv1d(p / "c1", 1, 64, 8, nn::ConvConfig { stride: 2, ..Default::default() });
        let gn2 = nn::group_norm(p / "gn2", 8, 64, Default::default());
        let c2_dw = nn::conv1d(p / "c2_dw", 64, 64, 5, nn::ConvConfig { stride: 2, padding: 2, groups: 64, ..Default::default() });
        let c2_pw = nn::conv1d(p / "c2_pw", 64, 64, 1, Default::default());
        let gn3 = nn::group_norm(p / "gn3", 8, 64, Default::default());
        let c3_dw = nn::conv1d(p / "c3_dw", 64, 64, 3, nn::ConvConfig { stride: 2, padding: 1, groups: 64, ..Default::default() });
        let c3_pw = nn::conv1d(p / "c3_pw", 64, 64, 1, Default::default());

        let ssm = stateful_mamba_block(&(p / "ssm"), SSM_DIM);
        let ssm_proj = nn::conv1d(p / "ssm_proj", SSM_DIM, 256, 1, Default::default());
        let pos_embedding = p.var("pos_emb", &[1, 256, CONV_TEMPORAL_LEN], Init::Uniform { lo: -0.01, up: 0.01 });

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
        let bucket_centers = Tensor::linspace(-20.0, 20.0, NUM_VALUE_BUCKETS, (Kind::Float, p.device()));

        let actor_mean = nn::linear(p / "al_mean", 256, nact, nn::LinearConfig {
            ws_init: Init::Uniform { lo: -0.1, up: 0.1 },
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        });

        const SDE_LATENT_DIM: i64 = 64;
        let sde_fc = nn::linear(p / "sde_fc", 256, SDE_LATENT_DIM, Default::default());
        let ln_sde = nn::layer_norm(p / "ln_sde", vec![SDE_LATENT_DIM], Default::default());
        let log_std_param = p.var("log_std", &[SDE_LATENT_DIM, nact], Init::Const(0.0));
        let log_d_raw = p.var("log_d_raw", &[nact], Init::Const(-0.3));

        Self {
            c1, gn2, c2_dw, c2_pw, gn3, c3_dw, c3_pw,
            ssm, ssm_proj, pos_embedding,
            static_to_temporal, ln_static_temporal, temporal_pools,
            static_proj, ln_static_proj,
            attn_qkv, attn_out, ln_attn,
            pma_seeds, pma_kv, pma_q, pma_out, ln_pma, global_to_seed,
            fc1, ln_fc1, fc2_actor, ln_fc2_actor, fc2_critic, ln_fc2_critic,
            fc3_actor, ln_fc3_actor, fc3_critic, ln_fc3_critic,
            critic, bucket_centers, actor_mean, sde_fc, ln_sde, log_std_param, log_d_raw,
            device: p.device(),
            num_heads, head_dim, pma_num_seeds,
        }
    }

    /// Batch forward for training (parallel SSM scan)
    pub fn forward(&self, price_deltas: &Tensor, static_features: &Tensor, train: bool) -> ModelOutput {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.conv_stem(&price_deltas, batch_size);

        let x_for_ssm = x_stem.permute([0, 2, 1]);
        let x_ssm = self.ssm.forward(&x_for_ssm, train);
        let x_ssm = x_ssm.permute([0, 2, 1]);

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
    }

    /// Forward with recurrent SSM state - GPU efficient for training rollouts with memory
    pub fn forward_with_state(&self, price_deltas: &Tensor, static_features: &Tensor, state: &mut StreamState) -> ModelOutput {
        let price_deltas = price_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.conv_stem(&price_deltas, batch_size);

        // GPU-efficient SSM with batched state - chunked parallel scan
        let x_for_ssm = x_stem.permute([0, 2, 1]); // [B*T, L, C]
        let x_ssm = self.ssm.forward_with_state(&x_for_ssm, &mut state.ssm_state_batched);
        let x_ssm = x_ssm.permute([0, 2, 1]); // [B*T, C, L]

        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, batch_size)
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
            ssm_states: (0..TICKERS_COUNT).map(|_| self.ssm.init_state(1, self.device)).collect(),
            ssm_state_batched: self.ssm.init_state(batch_size * TICKERS_COUNT, self.device),
            last_output: None,
        }
    }

    /// Streaming step for O(1) inference. Returns (ready, output).
    pub fn step(&self, new_deltas: &Tensor, static_features: &Tensor, state: &mut StreamState) -> (bool, ModelOutput) {
        let new_deltas = new_deltas.to_device(self.device);
        let static_features = static_features.to_device(self.device);

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
        (o.0.shallow_clone(), (o.1.0.shallow_clone(), o.1.1.shallow_clone(), o.1.2.shallow_clone()), o.2.shallow_clone())
    }

    fn zero_output(&self) -> ModelOutput {
        let z_c = Tensor::zeros(&[1], (Kind::Float, self.device));
        let z_m = Tensor::zeros(&[1, TICKERS_COUNT + 1], (Kind::Float, self.device));
        let z_s = Tensor::zeros(&[1, TICKERS_COUNT + 1], (Kind::Float, self.device));
        let z_d = Tensor::ones(&[TICKERS_COUNT + 1], (Kind::Float, self.device));
        let z_a = Tensor::zeros(&[1, 1], (Kind::Float, self.device));
        (z_c, (z_m, z_s, z_d), z_a)
    }

    fn process_stream_buffer(&self, buffer: &Tensor, static_features: &Tensor, state: &mut StreamState) -> ModelOutput {
        let batch_size = 1i64;
        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);

        let x = buffer.unsqueeze(1);
        let x = x.apply(&self.c1).silu();
        let x = x.apply(&self.gn2).apply(&self.c2_dw).apply(&self.c2_pw).silu();
        let x = x.apply(&self.gn3).apply(&self.c3_dw).apply(&self.c3_pw).silu();

        let x_in = x.squeeze_dim(2);
        let mut ssm_outs = Vec::with_capacity(TICKERS_COUNT as usize);
        for t in 0..TICKERS_COUNT as usize {
            let out = self.ssm.step(&x_in.narrow(0, t as i64, 1), &mut state.ssm_states[t]);
            ssm_outs.push(out);
        }
        let x_ssm = Tensor::cat(&ssm_outs, 0);

        // ssm_proj is Conv1D(64, 256, kernel=1) which is equivalent to Linear
        // weights: [256, 64, 1] -> squeeze to [256, 64] -> transpose to [64, 256]
        let w = self.ssm_proj.ws.squeeze_dim(2).tr();
        let conv_features = x_ssm.matmul(&w);
        let conv_features = match &self.ssm_proj.bs {
            Some(b) => conv_features + b,
            None => conv_features,
        };
        let conv_features = conv_features.view([batch_size, TICKERS_COUNT, 256]);

        self.head_no_temporal_pool(&conv_features, &global_static, &per_ticker_static, batch_size)
    }

    fn parse_static(&self, static_features: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let global = static_features.narrow(1, 0, GLOBAL_STATIC_OBS as i64);
        let per_ticker = static_features
            .narrow(1, GLOBAL_STATIC_OBS as i64, TICKERS_COUNT * PER_TICKER_STATIC_OBS as i64)
            .reshape([batch_size, TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64]);
        (global, per_ticker)
    }

    fn conv_stem(&self, price_deltas: &Tensor, batch_size: i64) -> Tensor {
        let x = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_size * TICKERS_COUNT, 1, PRICE_DELTAS_PER_TICKER as i64]);
        let x = x.apply(&self.c1).silu();
        let x = x.apply(&self.gn2).apply(&self.c2_dw).apply(&self.c2_pw).silu();
        x.apply(&self.gn3).apply(&self.c3_dw).apply(&self.c3_pw).silu()
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

        let critic_probs = critic_feat.apply(&self.critic).softmax(-1, Kind::Float);
        let critic_symlog = critic_probs.matmul(&self.bucket_centers);
        let critic_value = critic_symlog.sign() * (critic_symlog.abs().exp() - 1.0);

        let action_mean = actor_feat.apply(&self.actor_mean);
        const LOG_STD_OFFSET: f64 = -4.0;
        let latent = actor_feat.apply(&self.sde_fc).apply(&self.ln_sde).tanh();
        let log_std = (&self.log_std_param + LOG_STD_OFFSET).clamp(-5.0, -2.0);
        let variance = latent.pow_tensor_scalar(2).matmul(&log_std.exp().pow_tensor_scalar(2));
        let action_log_std = (variance + 1e-6).sqrt().log();

        const LOG_D_RAW_SCALE: f64 = 5.0;
        let divisor = self.log_d_raw.g_mul_scalar(LOG_D_RAW_SCALE).softplus() + 0.1;

        (critic_value, (action_mean, action_log_std, divisor), attn_out_vis)
    }
}

pub type Model = Box<dyn Fn(&Tensor, &Tensor, bool) -> ModelOutput>;

pub fn model(p: &nn::Path, nact: i64) -> Model {
    let m = TradingModel::new(p, nact);
    Box::new(move |price_deltas, static_features, train| m.forward(price_deltas, static_features, train))
}
