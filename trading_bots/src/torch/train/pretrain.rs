use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use rand::{seq::SliceRandom, Rng};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use tch::{autocast, nn, nn::Module, nn::ModuleT, Device, Kind, Reduction, Tensor};

use crate::data::universe::cached_eligible_training_universe;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::env::{Env, OHLC_BAR_FEATURES};
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{ModelVariant, RotaryEmbedding, TradingModel, TradingModelConfig};
use crate::torch::optim::muon::{Muon, MuonConfig};
use shared::{
    paths::RUNS_PATH,
    report::{CandleBar, Report, ReportKind, ReportSeries, ScaleKind},
    run_dir::RunDir,
};

use super::config::{LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON};
use super::optimizer_glue::{muon_momentum_for_step, named_trainable_variables};

const HORIZON_FEATURE_DIM: i64 = 7;
const LEJEPA_SIGREG_PROJECTIONS: i64 = 1024;
const LEJEPA_SIGREG_POSITIONS: i64 = 256;
const LEJEPA_SIGREG_KNOTS: i64 = 17;
const LEJEPA_BAR_FEATURES: i64 = OHLC_BAR_FEATURES as i64;
const LEJEPA_ROLLOUT_BARS: i64 = 100;
const LEJEPA_ROLLOUT_EVAL_WINDOWS: usize = 64;
const LEJEPA_PROBE_LR: f64 = 1e-3;
const LEJEPA_WEIGHT_DECAY: f64 = 1e-3;
const LEJEPA_AR_LAYERS: usize = 6;
const LEJEPA_AR_FF_DIM: i64 = 1536;
const LEJEPA_PROJECTOR_HIDDEN_DIM: i64 = 2048;
const LEJEPA_HEAD_DIM: i64 = 64;
const LEJEPA_ROPE_DIMS: i64 = 32;
// Shortcut flow-matching head (dreamer4 shortcut forcing + JEDI latent bounding).
const LEJEPA_K_MAX: i64 = 64;
const LEJEPA_SIGNAL_EMBED_DIM: i64 = 32;
const LEJEPA_STEP_EMBED_DIM: i64 = 32;
const LEJEPA_NUM_STEP_SIZES_LOG2: i64 = 6; // log2 step size in {0..5}
const LEJEPA_FLOW_COND_DIM: i64 = 512;
const LEJEPA_FLOW_HIDDEN: i64 = 1024;
const LEJEPA_FLOW_BLOCKS: usize = 3;
const LEJEPA_ROLLOUT_STEPS: i64 = 4;
const LEJEPA_ROLLOUT_STEP_SIZE: i64 = LEJEPA_K_MAX / LEJEPA_ROLLOUT_STEPS;
// Mean/deterministic trajectory = Monte-Carlo average of K independent sampled rollouts.
const LEJEPA_MEAN_ROLLOUT_SAMPLES: i64 = 4;
const LEJEPA_ROLLOUT_EVAL_SAMPLES: usize = 4;
// Drift-mitigation context noise: ctx = (1-mix)*tokens + mix*randn, applied to the
// AR predictor input at both train and rollout time.
const LEJEPA_CTX_NOISE_MIX: f64 = 0.1;
// Batch-level Bernoulli: shortcut branch with prob 1 - 1/log2(K_MAX) = 1 - 1/6.
const LEJEPA_PROB_SHORTCUT: f64 = 1.0 - 1.0 / 6.0;
const LEJEPA_LATENT_BOUND: f64 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum PretrainObjective {
    MeanMse,
    Lejepa,
}

#[derive(Clone, Debug)]
pub struct PretrainArgs {
    pub weights: Option<String>,
    pub model_size: ModelVariant,
    pub run: Option<String>,
    pub epochs: usize,
    pub steps: Option<usize>,
    pub batch_size: usize,
    pub k_patches: usize,
    pub batches_per_epoch: usize,
    pub objective: PretrainObjective,
    pub lambda_lat: f64,
    pub lambda_sigreg: f64,
    pub probe_epochs: usize,
    pub target_scale: f64,
    pub validation_batches: usize,
    pub validate_every: usize,
    pub checkpoint_every: usize,
    pub log_step_losses: bool,
}

struct CausalLejepaLayer {
    qkv: nn::Linear,
    out_proj: nn::Linear,
    ff_gate: nn::Linear,
    ff_value: nn::Linear,
    ff_out: nn::Linear,
}

struct ProjectionMlp {
    fc1: nn::Linear,
    bn: nn::BatchNorm,
    fc2: nn::Linear,
}

struct LejepaBarPredictions {
    belief: Tensor,
}

// DiT AdaLN-zero block: a non-affine norm modulated by (shift, scale) and a
// zero-init gate on the residual, all produced from the conditioning vector.
struct LejepaFlowBlock {
    mod_fc: nn::Linear,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

// Shortcut flow-matching head. Discrete signal-level + step-size log2 embeddings
// condition an x-prediction DiT that operates directly in the (tanh-bounded)
// latent space -- no EMA standardization; SIGReg keeps tokens near-isotropic.
struct LejepaFlowHead {
    signal_embed: nn::Embedding,
    step_embed: nn::Embedding,
    cond_fc1: nn::Linear,
    cond_fc2: nn::Linear,
    in_proj: nn::Linear,
    blocks: Vec<LejepaFlowBlock>,
    final_mod: nn::Linear,
    out_proj: nn::Linear,
}

struct PretrainHeads {
    forecast_queries: Tensor,
    horizon_pos_proj: nn::Linear,
    forecast_q_proj: nn::Linear,
    forecast_k_proj: nn::Linear,
    forecast_v_proj: nn::Linear,
    forecast_out_proj: nn::Linear,
    return_mean: nn::Linear,
    bar_proj: nn::Linear,
    bar_enrich_fc1: nn::Linear,
    bar_enrich_fc2: nn::Linear,
    lejepa_projector: ProjectionMlp,
    lejepa_layers: Vec<CausalLejepaLayer>,
    lejepa_flow: LejepaFlowHead,
    rope: RotaryEmbedding,
    probe_input_ln: nn::LayerNorm,
    probe_head: nn::Linear,
    probe_logvar_head: nn::Linear,
    next_patch_embed: nn::Linear,
    latent_fc1: nn::Linear,
    latent_fc2: nn::Linear,
    horizon: i64,
    latent_dim: i64,
    forecast_heads: i64,
    lejepa_heads: i64,
    dropout: f64,
}

struct PretrainBatch {
    obs: Tensor,
    static_obs: Tensor,
    next_obs: Tensor,
    next_static_obs: Tensor,
    future_patches: Tensor,
    next_patch: Tensor,
    bar_history: Tensor,
    next_bars: Tensor,
}

impl PretrainBatch {
    fn len(&self) -> i64 {
        self.obs.size()[0]
    }
}

struct PretrainSampler {
    train_tickers: Vec<String>,
    train_envs: Vec<Env>,
    train_offsets_by_env: Vec<Vec<usize>>,
    train_batches_per_epoch: usize,
    train_batch_cursor: usize,
    k_patches: usize,
    patch_size: usize,
    target_scale: f64,
    device: Device,
}

#[derive(Clone, Copy)]
enum SplitKind {
    Train,
    Validation,
}

impl PretrainHeads {
    fn new(p: &nn::Path, latent_dim: i64, k_patches: i64, patch_size: i64) -> Self {
        let ff_dim = latent_dim * 2;
        let horizon = k_patches * patch_size;
        let forecast_heads = 4;
        let lejepa_heads = 4;
        assert_eq!(
            latent_dim % forecast_heads,
            0,
            "forecast attention heads must divide latent dim"
        );
        assert_eq!(
            latent_dim % lejepa_heads,
            0,
            "LEJEPA attention heads must divide latent dim"
        );
        assert_eq!(
            latent_dim / lejepa_heads,
            LEJEPA_HEAD_DIM,
            "LEJEPA head dim must match RoPE head dim"
        );
        let forecast_queries = p.var(
            "forecast_queries",
            &[horizon, latent_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let mut horizon_pos_proj = nn::linear(
            p / "horizon_pos_proj",
            HORIZON_FEATURE_DIM,
            latent_dim,
            Default::default(),
        );
        tch::no_grad(|| {
            let init = Tensor::randn(
                horizon_pos_proj.ws.size(),
                (horizon_pos_proj.ws.kind(), horizon_pos_proj.ws.device()),
            ) * 0.01;
            horizon_pos_proj.ws.copy_(&init);
            if let Some(bias) = horizon_pos_proj.bs.as_mut() {
                let _ = bias.zero_();
            }
        });
        let forecast_q_proj = nn::linear(
            p / "forecast_q_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let forecast_k_proj = nn::linear(
            p / "forecast_k_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let forecast_v_proj = nn::linear(
            p / "forecast_v_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let forecast_out_proj = nn::linear(
            p / "forecast_out_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let mut return_mean = nn::linear(p / "return_mean", latent_dim, 1, Default::default());
        tch::no_grad(|| {
            let init = Tensor::randn(
                return_mean.ws.size(),
                (return_mean.ws.kind(), return_mean.ws.device()),
            ) * 0.01;
            return_mean.ws.copy_(&init);
            if let Some(bias) = return_mean.bs.as_mut() {
                let _ = bias.zero_();
            }
        });
        let bar_proj = nn::linear(
            p / "bar_proj",
            LEJEPA_BAR_FEATURES,
            latent_dim,
            Default::default(),
        );
        let bar_enrich_fc1 =
            nn::linear(p / "bar_enrich_fc1", latent_dim, ff_dim, Default::default());
        let bar_enrich_fc2 =
            nn::linear(p / "bar_enrich_fc2", ff_dim, latent_dim, Default::default());
        let lejepa_projector = ProjectionMlp {
            fc1: nn::linear(
                p / "lejepa_projector_fc1",
                latent_dim,
                LEJEPA_PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            bn: nn::batch_norm1d(
                p / "lejepa_projector_bn",
                LEJEPA_PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            fc2: nn::linear(
                p / "lejepa_projector_fc2",
                LEJEPA_PROJECTOR_HIDDEN_DIM,
                latent_dim,
                Default::default(),
            ),
        };
        let mut lejepa_layers = Vec::with_capacity(LEJEPA_AR_LAYERS);
        for layer_idx in 0..LEJEPA_AR_LAYERS {
            let layer_name = format!("lejepa_layer_{layer_idx}");
            let layer_path = p / layer_name.as_str();
            lejepa_layers.push(CausalLejepaLayer {
                qkv: nn::linear(
                    &layer_path / "qkv",
                    latent_dim,
                    latent_dim * 3,
                    Default::default(),
                ),
                out_proj: nn::linear(
                    &layer_path / "out_proj",
                    latent_dim,
                    latent_dim,
                    Default::default(),
                ),
                ff_gate: nn::linear(
                    &layer_path / "ff_gate",
                    latent_dim,
                    LEJEPA_AR_FF_DIM,
                    Default::default(),
                ),
                ff_value: nn::linear(
                    &layer_path / "ff_value",
                    latent_dim,
                    LEJEPA_AR_FF_DIM,
                    Default::default(),
                ),
                ff_out: nn::linear(
                    &layer_path / "ff_out",
                    LEJEPA_AR_FF_DIM,
                    latent_dim,
                    Default::default(),
                ),
            });
        }
        let rope = RotaryEmbedding::new(
            PRICE_DELTAS_PER_TICKER as i64 + 1,
            LEJEPA_HEAD_DIM,
            LEJEPA_ROPE_DIMS,
            p.device(),
        );
        // AdaLN-zero: modulation, final-mod, and out_proj are zero-init so the head
        // starts as the identity-mean predictor (x_pred == 0 before training).
        let zero_init = |mut lin: nn::Linear| -> nn::Linear {
            tch::no_grad(|| {
                let _ = lin.ws.zero_();
                if let Some(b) = lin.bs.as_mut() {
                    let _ = b.zero_();
                }
            });
            lin
        };
        let mut lejepa_flow_blocks = Vec::with_capacity(LEJEPA_FLOW_BLOCKS);
        for block_idx in 0..LEJEPA_FLOW_BLOCKS {
            let block_name = format!("lejepa_flow_block_{block_idx}");
            let block_path = p / block_name.as_str();
            lejepa_flow_blocks.push(LejepaFlowBlock {
                mod_fc: zero_init(nn::linear(
                    &block_path / "mod",
                    LEJEPA_FLOW_COND_DIM,
                    latent_dim * 3,
                    Default::default(),
                )),
                fc1: nn::linear(
                    &block_path / "fc1",
                    latent_dim,
                    LEJEPA_FLOW_HIDDEN,
                    Default::default(),
                ),
                fc2: nn::linear(
                    &block_path / "fc2",
                    LEJEPA_FLOW_HIDDEN,
                    latent_dim,
                    Default::default(),
                ),
            });
        }
        let lejepa_flow = LejepaFlowHead {
            signal_embed: nn::embedding(
                p / "lejepa_flow_signal_embed",
                LEJEPA_K_MAX,
                LEJEPA_SIGNAL_EMBED_DIM,
                Default::default(),
            ),
            step_embed: nn::embedding(
                p / "lejepa_flow_step_embed",
                LEJEPA_NUM_STEP_SIZES_LOG2,
                LEJEPA_STEP_EMBED_DIM,
                Default::default(),
            ),
            cond_fc1: nn::linear(
                p / "lejepa_flow_cond_fc1",
                latent_dim + LEJEPA_SIGNAL_EMBED_DIM + LEJEPA_STEP_EMBED_DIM,
                LEJEPA_FLOW_COND_DIM,
                Default::default(),
            ),
            cond_fc2: nn::linear(
                p / "lejepa_flow_cond_fc2",
                LEJEPA_FLOW_COND_DIM,
                LEJEPA_FLOW_COND_DIM,
                Default::default(),
            ),
            in_proj: nn::linear(
                p / "lejepa_flow_in_proj",
                latent_dim,
                latent_dim,
                Default::default(),
            ),
            blocks: lejepa_flow_blocks,
            final_mod: zero_init(nn::linear(
                p / "lejepa_flow_final_mod",
                LEJEPA_FLOW_COND_DIM,
                latent_dim * 2,
                Default::default(),
            )),
            out_proj: zero_init(nn::linear(
                p / "lejepa_flow_out_proj",
                latent_dim,
                latent_dim,
                Default::default(),
            )),
        };
        let probe_input_ln =
            nn::layer_norm(p / "probe_input_ln", vec![latent_dim], Default::default());
        let probe_head = nn::linear(
            p / "probe_head",
            latent_dim,
            LEJEPA_BAR_FEATURES,
            Default::default(),
        );
        let probe_logvar_head = nn::linear(
            p / "probe_logvar_head",
            latent_dim,
            LEJEPA_BAR_FEATURES,
            Default::default(),
        );
        let next_patch_embed = nn::linear(
            p / "next_patch_embed",
            patch_size,
            latent_dim,
            Default::default(),
        );
        let latent_fc1 = nn::linear(p / "latent_fc1", latent_dim * 2, ff_dim, Default::default());
        let latent_fc2 = nn::linear(p / "latent_fc2", ff_dim, latent_dim, Default::default());
        Self {
            forecast_queries,
            horizon_pos_proj,
            forecast_q_proj,
            forecast_k_proj,
            forecast_v_proj,
            forecast_out_proj,
            return_mean,
            bar_proj,
            bar_enrich_fc1,
            bar_enrich_fc2,
            lejepa_projector,
            lejepa_layers,
            lejepa_flow,
            rope,
            probe_input_ln,
            probe_head,
            probe_logvar_head,
            next_patch_embed,
            latent_fc1,
            latent_fc2,
            horizon,
            latent_dim,
            forecast_heads,
            lejepa_heads,
            dropout: 0.1,
        }
    }

    fn horizon_features(&self, device: Device, kind: Kind) -> Tensor {
        let denom = (self.horizon - 1).max(1) as f64;
        let x = (Tensor::arange(self.horizon, (Kind::Float, device)) / denom).unsqueeze(-1);
        let centered = &x * 2.0 - 1.0;
        let squared = x.pow_tensor_scalar(2.0);
        let angle1 = &x * std::f64::consts::TAU;
        let sin1 = angle1.sin();
        let cos1 = angle1.cos();
        let angle2 = &x * (std::f64::consts::TAU * 2.0);
        let sin2 = angle2.sin();
        let cos2 = angle2.cos();
        Tensor::cat(&[&x, &centered, &squared, &sin1, &cos1, &sin2, &cos2], -1).to_kind(kind)
    }

    fn forecast_tokens(&self, patch_tokens: &Tensor, train: bool) -> (Tensor, i64, i64) {
        let size = patch_tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let source_len = size[2];
        let rows = batch * tickers;
        let source = patch_tokens.view([rows, source_len, self.latent_dim]);
        let horizon_features = self.horizon_features(source.device(), source.kind());
        let base_queries = self.forecast_queries.to_kind(source.kind())
            + self.horizon_pos_proj.forward(&horizon_features);
        let queries = base_queries
            .unsqueeze(0)
            .expand([rows, self.horizon, self.latent_dim], false);

        let head_dim = self.latent_dim / self.forecast_heads;
        let q = self
            .forecast_q_proj
            .forward(&queries)
            .view([rows, self.horizon, self.forecast_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let k = self
            .forecast_k_proj
            .forward(&source)
            .view([rows, source_len, self.forecast_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let v = self
            .forecast_v_proj
            .forward(&source)
            .view([rows, source_len, self.forecast_heads, head_dim])
            .permute([0, 2, 1, 3]);

        let attn_scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
        let attn = attn_scores
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train)
            .to_kind(v.kind());
        let attended = attn.matmul(&v).permute([0, 2, 1, 3]).contiguous().view([
            rows,
            self.horizon,
            self.latent_dim,
        ]);
        let forecast_tokens = queries
            + self
                .forecast_out_proj
                .forward(&attended)
                .dropout(self.dropout, train);
        (forecast_tokens, batch, tickers)
    }

    fn forecast_readout(&self, forecast_tokens: &Tensor, train: bool) -> Tensor {
        forecast_tokens.dropout(self.dropout, train)
    }

    fn return_mean_from_readout(&self, readout: &Tensor, batch: i64, tickers: i64) -> Tensor {
        self.return_mean
            .forward(&readout)
            .view([batch, tickers, self.horizon])
    }

    fn predict_return_mean(&self, patch_tokens: &Tensor, train: bool) -> Tensor {
        let (forecast_tokens, batch, tickers) = self.forecast_tokens(patch_tokens, train);
        let readout = self.forecast_readout(&forecast_tokens, train);
        self.return_mean_from_readout(&readout, batch, tickers)
    }

    fn encode_bar_tokens(&self, bars: &Tensor, train: bool) -> Tensor {
        let size = bars.size();
        let batch = size[0];
        let tickers = size[1];
        let length = size[2];
        let features = bars
            .view([batch * tickers * length, LEJEPA_BAR_FEATURES])
            .to_kind(Kind::Float)
            .nan_to_num(0.0, 0.0, 0.0);
        let h = self.bar_proj.forward(&features);
        let enriched = self.bar_enrich_fc2.forward(
            &normalize_last_dim(&self.bar_enrich_fc1.forward(&normalize_last_dim(&h))).gelu("none"),
        );
        let h = h + enriched;
        let tokens = self.projection_mlp(&h, &self.lejepa_projector, train);
        // JEDI differentiable latent bounding: keep encoder tokens in a fixed range
        // so the unit-N(0,I) flow prior and the x-predictions share the same scale.
        let tokens = latent_bound(&tokens);
        tokens.view([batch, tickers, length, self.latent_dim])
    }

    // AR transformer belief = final normalized representation, one per position.
    // The belief conditions the shortcut flow head to predict the next token.
    fn predict_lejepa_bar_predictions(
        &self,
        bar_tokens: &Tensor,
        train: bool,
    ) -> LejepaBarPredictions {
        let size = bar_tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let length = size[2];
        let rows = batch * tickers;
        let positions = Tensor::arange(length, (Kind::Int64, bar_tokens.device()));
        let mut x = bar_tokens.view([rows, length, self.latent_dim]);
        for layer in &self.lejepa_layers {
            x = self.causal_lejepa_layer(&x, layer, &positions, train);
        }
        let belief = normalize_last_dim(&x);
        LejepaBarPredictions {
            belief: belief.view([batch, tickers, length, self.latent_dim]),
        }
    }

    // Shortcut flow head, x-prediction. Given a noised latent `z` at discrete
    // `signal` level, `step_log2` step size, and belief `ctx`, predict the clean
    // latent. AdaLN-zero DiT operating directly in tanh-bounded latent space.
    // `z`, `ctx` are [rows, latent_dim]; `signal`, `step_log2` are int64 [rows].
    fn lejepa_flow_predict(
        &self,
        z: &Tensor,
        signal: &Tensor,
        step_log2: &Tensor,
        ctx: &Tensor,
    ) -> Tensor {
        let flow = &self.lejepa_flow;
        let latent_dim = self.latent_dim;
        let signal_emb = flow.signal_embed.forward(signal);
        let step_emb = flow.step_embed.forward(step_log2);
        let cond_in = Tensor::cat(&[ctx, &signal_emb, &step_emb], -1);
        let c = flow
            .cond_fc2
            .forward(&flow.cond_fc1.forward(&cond_in).silu())
            .silu();
        let mut h = flow.in_proj.forward(z);
        for block in &flow.blocks {
            let mods = block.mod_fc.forward(&c);
            let shift = mods.narrow(-1, 0, latent_dim);
            let scale = mods.narrow(-1, latent_dim, latent_dim);
            let gate = mods.narrow(-1, 2 * latent_dim, latent_dim);
            let modulated = &normalize_last_dim(&h) * (&scale + 1.0) + &shift;
            let enriched = block
                .fc2
                .forward(&block.fc1.forward(&modulated).gelu("none"));
            h = h + &gate * enriched;
        }
        let fmods = flow.final_mod.forward(&c);
        let shift_f = fmods.narrow(-1, 0, latent_dim);
        let scale_f = fmods.narrow(-1, latent_dim, latent_dim);
        let modulated = &normalize_last_dim(&h) * (&scale_f + 1.0) + &shift_f;
        latent_bound(&flow.out_proj.forward(&modulated))
    }

    // x-prediction -> rectified-flow velocity: v = (x_pred - noised) / (1 - t),
    // t = signal / K_MAX. Callers guarantee signal < K_MAX so (1 - t) >= 1/K_MAX.
    fn lejepa_flow_velocity(&self, x_pred: &Tensor, noised: &Tensor, signal: &Tensor) -> Tensor {
        let one_minus_t = (signal.to_kind(Kind::Float) / (-(LEJEPA_K_MAX as f64))) + 1.0;
        (x_pred - noised) / one_minus_t.unsqueeze(-1)
    }

    fn causal_lejepa_layer(
        &self,
        source: &Tensor,
        layer: &CausalLejepaLayer,
        positions: &Tensor,
        train: bool,
    ) -> Tensor {
        let size = source.size();
        let rows = size[0];
        let length = size[1];
        let head_dim = self.latent_dim / self.lejepa_heads;
        let normed = normalize_last_dim(source);
        let qkv = layer.qkv.forward(&normed);
        let parts = qkv.split(self.latent_dim, -1);
        let q = parts[0]
            .view([rows, length, self.lejepa_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let k = parts[1]
            .view([rows, length, self.lejepa_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let v = parts[2]
            .view([rows, length, self.lejepa_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let q = self.rope.apply_positions(&q, positions);
        let k = self.rope.apply_positions(&k, positions);
        let attn_kind = if source.device().is_cuda() {
            Kind::BFloat16
        } else {
            source.kind()
        };
        let attn = Tensor::scaled_dot_product_attention(
            &q.to_kind(attn_kind),
            &k.to_kind(attn_kind),
            &v.to_kind(attn_kind),
            None::<&Tensor>,
            0.0,
            true,
            None,
            false,
        )
        .to_kind(source.kind())
        .permute([0, 2, 1, 3])
        .contiguous()
        .view([rows, length, self.latent_dim]);
        let x = source + layer.out_proj.forward(&attn).dropout(self.dropout, train);
        let ff = self.causal_ff(layer, &x).dropout(self.dropout, train);
        x + ff
    }

    fn causal_ff(&self, layer: &CausalLejepaLayer, x: &Tensor) -> Tensor {
        let x = normalize_last_dim(x);
        let gate = layer.ff_gate.forward(&x).silu();
        let value = layer.ff_value.forward(&x);
        layer.ff_out.forward(&(gate * value))
    }

    fn projection_mlp(&self, x: &Tensor, mlp: &ProjectionMlp, train: bool) -> Tensor {
        let shape = x.size();
        let rows = x.numel() as i64 / self.latent_dim;
        let flat = x.view([rows, self.latent_dim]);
        let hidden = mlp.fc1.forward(&flat);
        let hidden = mlp.bn.forward_t(&hidden, train).gelu("none");
        mlp.fc2.forward(&hidden).view(shape.as_slice())
    }

    // Rank-preserving probe: maps a latent [.., D] -> (mean, logvar) each [.., 16].
    fn probe_ohlc_features(&self, latent: &Tensor) -> (Tensor, Tensor) {
        let normed = self.probe_input_ln.forward(latent);
        let mean = self.probe_head.forward(&normed);
        let logvar = self.probe_logvar_head.forward(&normed).clamp(-7.0, 7.0);
        (mean, logvar)
    }

    // Stochastic imagined rollout. `temperature == 0` yields the MEAN path: the
    // Monte-Carlo average (over decoded bar features) of K independent sampled
    // trajectories, each drawn at temperature 1. Any other temperature yields a
    // single sampled trajectory scaled by that temperature.
    fn lejepa_imagined_rollout(&self, context_bars: &Tensor, temperature: f64, train: bool) -> Tensor {
        if temperature == 0.0 {
            let k = LEJEPA_MEAN_ROLLOUT_SAMPLES;
            let mut acc: Option<Tensor> = None;
            for _ in 0..k {
                let (traj, _) = self.single_imagined_rollout(context_bars, 1.0, train, false);
                acc = Some(match acc {
                    Some(a) => a + traj,
                    None => traj,
                });
            }
            return acc.expect("K >= 1") / k as f64;
        }
        self.single_imagined_rollout(context_bars, temperature, train, false).0
    }

    // One stochastic autoregressive trajectory. Per generated frame: mix a fixed
    // per-position context noise into the token sequence (drift mitigation), run
    // the AR predictor to a belief, then sample the next latent with a 4-step
    // shortcut Euler integration from a fresh N(0,I) prior (scaled by temperature).
    // Each new latent is tanh-bounded, appended, and probe-decoded to an OHLC bar.
    fn single_imagined_rollout(
        &self,
        context_bars: &Tensor,
        temperature: f64,
        train: bool,
        collect_entropy: bool,
    ) -> (Tensor, Option<RolloutEntropy>) {
        let mut tokens = self.encode_bar_tokens(context_bars, train).detach();
        let size = tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let latent_dim = self.latent_dim;
        let rows = batch * tickers;
        let device = tokens.device();
        // Fixed per-position context noise, grown alongside the token sequence so
        // each frame's mixed context stays constant once it exists.
        let mut ctx_noise = Tensor::randn_like(&tokens);
        let step_log2_val = (LEJEPA_ROLLOUT_STEP_SIZE as f64).log2() as i64;
        let mut imagined = Vec::with_capacity(LEJEPA_ROLLOUT_BARS as usize);
        let mut ent_means: Vec<Tensor> = Vec::new();
        let mut tok_norm_sum = 0.0f64;
        let mut tok_norm_max = 0.0f64;
        for _ in 0..LEJEPA_ROLLOUT_BARS {
            let noisy_tokens = &tokens * (1.0 - LEJEPA_CTX_NOISE_MIX) + &ctx_noise * LEJEPA_CTX_NOISE_MIX;
            let belief = self.predict_lejepa_bar_predictions(&noisy_tokens, train).belief;
            let last = tokens.size()[2] - 1;
            let ctx = belief.narrow(2, last, 1).reshape([rows, latent_dim]);
            let mut z = Tensor::randn([rows, latent_dim], (Kind::Float, device)) * temperature;
            for k in 0..LEJEPA_ROLLOUT_STEPS {
                let sig_val = (k * LEJEPA_ROLLOUT_STEP_SIZE).min(LEJEPA_K_MAX - 1);
                let signal = Tensor::full([rows], sig_val, (Kind::Int64, device));
                let step_log2 = Tensor::full([rows], step_log2_val, (Kind::Int64, device));
                let x_pred = self.lejepa_flow_predict(&z, &signal, &step_log2, &ctx);
                let v = self.lejepa_flow_velocity(&x_pred, &z, &signal);
                z = z + v * (LEJEPA_ROLLOUT_STEP_SIZE as f64 / LEJEPA_K_MAX as f64);
            }
            // Final Euler step lands exactly on the (already latent-bounded) x_pred.
            let next_token = z.view([batch, tickers, 1, latent_dim]);
            let (mean, _logvar) = self.probe_ohlc_features(&next_token);
            let bar = mean.view([batch, LEJEPA_BAR_FEATURES]);
            imagined.push(bar.shallow_clone());
            if collect_entropy {
                let nt_n = z
                    .reshape([rows, latent_dim])
                    .square()
                    .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                    .sqrt();
                tok_norm_sum += nt_n.mean(Kind::Float).double_value(&[]);
                tok_norm_max = tok_norm_max.max(nt_n.max().double_value(&[]));
                ent_means.push(bar);
            }
            tokens = Tensor::cat(&[&tokens, &next_token], 2);
            ctx_noise = Tensor::cat(&[&ctx_noise, &Tensor::randn_like(&next_token)], 2);
            let len = tokens.size()[2];
            let max_len = PRICE_DELTAS_PER_TICKER as i64;
            if len > max_len {
                tokens = tokens.narrow(2, len - max_len, max_len);
                ctx_noise = ctx_noise.narrow(2, len - max_len, max_len);
            }
        }
        let entropy = if collect_entropy {
            let steps = LEJEPA_ROLLOUT_BARS as f64;
            let means_stack = Tensor::stack(&ent_means, 1);
            let mu = means_stack.mean_dim([1i64].as_slice(), true, Kind::Float);
            let mean_step_std = (&means_stack - &mu)
                .square()
                .mean_dim([1i64].as_slice(), false, Kind::Float)
                .sqrt()
                .mean(Kind::Float)
                .double_value(&[]);
            Some(RolloutEntropy {
                mean_step_std,
                tok_norm_mean: tok_norm_sum / steps,
                tok_norm_max,
            })
        } else {
            None
        };
        (Tensor::stack(&imagined, 1), entropy)
    }

    fn predict_next_latent(&self, latent: &Tensor, next_patch: &Tensor) -> Tensor {
        let next_patch_embed = self.next_patch_embed.forward(next_patch);
        let x = Tensor::cat(&[latent, &next_patch_embed], -1);
        let x = normalize_last_dim(&x);
        latent + self.latent_fc2.forward(&self.latent_fc1.forward(&x).relu())
    }
}

impl PretrainSampler {
    fn new(
        k_patches: usize,
        patch_size: usize,
        target_scale: f64,
        batches_per_epoch: usize,
        device: Device,
    ) -> Self {
        assert_eq!(
            TICKERS_COUNT, 1,
            "full-universe pretraining currently expects one ticker per observation"
        );
        assert!(
            batches_per_epoch > 0,
            "--batches-per-epoch must be positive"
        );
        let mut train_tickers = cached_eligible_training_universe().to_vec();
        train_tickers.shuffle(&mut rand::rng());
        let mut usable_train_tickers = Vec::with_capacity(train_tickers.len());
        let mut train_envs = Vec::with_capacity(train_tickers.len());
        let mut train_offsets_by_env = Vec::with_capacity(train_tickers.len());
        for ticker in train_tickers {
            let env = Env::new_with_tickers_and_recording(vec![ticker.clone()], true, false, None);
            let offsets = build_split_offsets(
                env.price_deltas[0].len(),
                k_patches,
                patch_size,
                SplitKind::Train,
            );
            if offsets.is_empty() {
                continue;
            }
            usable_train_tickers.push(ticker);
            train_envs.push(env);
            train_offsets_by_env.push(offsets);
        }
        assert!(
            !usable_train_tickers.is_empty(),
            "not enough market history for pretraining: train_tickers={}",
            usable_train_tickers.len()
        );
        Self {
            train_tickers: usable_train_tickers,
            train_envs,
            train_offsets_by_env,
            train_batches_per_epoch: batches_per_epoch,
            train_batch_cursor: 0,
            k_patches,
            patch_size,
            target_scale,
            device,
        }
    }

    fn start_epoch(&mut self) {
        self.train_batch_cursor = 0;
    }

    fn next_train_batch(&mut self, batch_size: usize) -> Option<PretrainBatch> {
        if self.train_batch_cursor >= self.train_batches_per_epoch {
            return None;
        }
        self.train_batch_cursor += 1;
        let mut rng = rand::rng();
        let samples = (0..batch_size)
            .map(|_| {
                let env_idx = rng.random_range(0..self.train_envs.len());
                let offsets = &self.train_offsets_by_env[env_idx];
                let offset = offsets[rng.random_range(0..offsets.len())];
                (env_idx, offset)
            })
            .collect::<Vec<_>>();
        Some(Self::batch_from_env_offsets(
            &mut self.train_envs,
            &samples,
            self.k_patches,
            self.patch_size,
            self.target_scale,
            self.device,
        ))
    }

    fn batch_from_offsets(
        env: &mut Env,
        offsets: &[usize],
        k_patches: usize,
        patch_size: usize,
        target_scale: f64,
        device: Device,
    ) -> PretrainBatch {
        let pd_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let so_dim = STATIC_OBSERVATIONS;
        let target_len = TICKERS_COUNT as usize * k_patches * patch_size;
        let next_patch_len = TICKERS_COUNT as usize * patch_size;
        let bar_history_len = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER * OHLC_BAR_FEATURES;
        let next_ohlc_len =
            TICKERS_COUNT as usize * LEJEPA_ROLLOUT_BARS as usize * OHLC_BAR_FEATURES;

        let mut obs = Vec::with_capacity(offsets.len() * pd_dim);
        let mut static_obs = Vec::with_capacity(offsets.len() * so_dim);
        let mut next_obs = Vec::with_capacity(offsets.len() * pd_dim);
        let mut next_static_obs = Vec::with_capacity(offsets.len() * so_dim);
        let mut future_patches = Vec::with_capacity(offsets.len() * target_len);
        let mut next_patch = Vec::with_capacity(offsets.len() * next_patch_len);
        let mut bar_history = Vec::with_capacity(offsets.len() * bar_history_len);
        let mut next_bars = Vec::with_capacity(offsets.len() * next_ohlc_len);

        for &offset in offsets {
            append_pretrain_sample(
                env,
                offset,
                k_patches,
                patch_size,
                target_scale,
                &mut obs,
                &mut static_obs,
                &mut next_obs,
                &mut next_static_obs,
                &mut future_patches,
                &mut next_patch,
                &mut bar_history,
                &mut next_bars,
            );
        }

        Self::batch_from_raw_parts(
            offsets.len(),
            obs,
            static_obs,
            next_obs,
            next_static_obs,
            future_patches,
            next_patch,
            bar_history,
            next_bars,
            k_patches,
            patch_size,
            device,
        )
    }

    fn batch_from_env_offsets(
        envs: &mut [Env],
        samples: &[(usize, usize)],
        k_patches: usize,
        patch_size: usize,
        target_scale: f64,
        device: Device,
    ) -> PretrainBatch {
        let pd_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let so_dim = STATIC_OBSERVATIONS;
        let target_len = TICKERS_COUNT as usize * k_patches * patch_size;
        let next_patch_len = TICKERS_COUNT as usize * patch_size;
        let bar_history_len = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER * OHLC_BAR_FEATURES;
        let next_ohlc_len =
            TICKERS_COUNT as usize * LEJEPA_ROLLOUT_BARS as usize * OHLC_BAR_FEATURES;

        let mut obs = Vec::with_capacity(samples.len() * pd_dim);
        let mut static_obs = Vec::with_capacity(samples.len() * so_dim);
        let mut next_obs = Vec::with_capacity(samples.len() * pd_dim);
        let mut next_static_obs = Vec::with_capacity(samples.len() * so_dim);
        let mut future_patches = Vec::with_capacity(samples.len() * target_len);
        let mut next_patch = Vec::with_capacity(samples.len() * next_patch_len);
        let mut bar_history = Vec::with_capacity(samples.len() * bar_history_len);
        let mut next_bars = Vec::with_capacity(samples.len() * next_ohlc_len);

        for &(env_idx, offset) in samples {
            append_pretrain_sample(
                &mut envs[env_idx],
                offset,
                k_patches,
                patch_size,
                target_scale,
                &mut obs,
                &mut static_obs,
                &mut next_obs,
                &mut next_static_obs,
                &mut future_patches,
                &mut next_patch,
                &mut bar_history,
                &mut next_bars,
            );
        }

        Self::batch_from_raw_parts(
            samples.len(),
            obs,
            static_obs,
            next_obs,
            next_static_obs,
            future_patches,
            next_patch,
            bar_history,
            next_bars,
            k_patches,
            patch_size,
            device,
        )
    }

    fn batch_from_raw_parts(
        sample_count: usize,
        obs: Vec<f32>,
        static_obs: Vec<f32>,
        next_obs: Vec<f32>,
        next_static_obs: Vec<f32>,
        future_patches: Vec<f32>,
        next_patch: Vec<f32>,
        bar_history: Vec<f32>,
        next_bars: Vec<f32>,
        k_patches: usize,
        patch_size: usize,
        device: Device,
    ) -> PretrainBatch {
        let batch = sample_count as i64;
        let pd_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let so_dim = STATIC_OBSERVATIONS;
        PretrainBatch {
            obs: Tensor::from_slice(&obs)
                .view([batch, pd_dim as i64])
                .to_device(device),
            static_obs: Tensor::from_slice(&static_obs)
                .view([batch, so_dim as i64])
                .to_device(device),
            next_obs: Tensor::from_slice(&next_obs)
                .view([batch, pd_dim as i64])
                .to_device(device),
            next_static_obs: Tensor::from_slice(&next_static_obs)
                .view([batch, so_dim as i64])
                .to_device(device),
            future_patches: Tensor::from_slice(&future_patches)
                .view([batch, TICKERS_COUNT, k_patches as i64, patch_size as i64])
                .to_device(device),
            next_patch: Tensor::from_slice(&next_patch)
                .view([batch, TICKERS_COUNT, patch_size as i64])
                .to_device(device),
            bar_history: Tensor::from_slice(&bar_history)
                .view([
                    batch,
                    TICKERS_COUNT,
                    PRICE_DELTAS_PER_TICKER as i64,
                    OHLC_BAR_FEATURES as i64,
                ])
                .to_device(device),
            next_bars: Tensor::from_slice(&next_bars)
                .view([
                    batch,
                    TICKERS_COUNT,
                    LEJEPA_ROLLOUT_BARS,
                    OHLC_BAR_FEATURES as i64,
                ])
                .to_device(device),
        }
    }
}

fn append_pretrain_sample(
    env: &mut Env,
    offset: usize,
    k_patches: usize,
    patch_size: usize,
    target_scale: f64,
    obs: &mut Vec<f32>,
    static_obs: &mut Vec<f32>,
    next_obs: &mut Vec<f32>,
    next_static_obs: &mut Vec<f32>,
    future_patches: &mut Vec<f32>,
    next_patch: &mut Vec<f32>,
    bar_history: &mut Vec<f32>,
    next_bars: &mut Vec<f32>,
) {
    let (obs_i, static_i) = env.reset_single_at_offset_for_pretrain(offset);
    let target_i =
        future_patches_for_current_perm(env, offset, k_patches, patch_size, target_scale);
    let next_patch_i = future_patches_for_current_perm(env, offset, 1, patch_size, 1.0);
    let bar_history_i = bar_history_for_current_perm(env, offset);
    let next_bars_i = next_bars_for_current_perm(env, offset);
    let (next_obs_i, next_static_i) =
        env.reset_single_at_offset_preserving_perm_for_pretrain(offset + patch_size);

    obs.extend(obs_i);
    static_obs.extend(static_i);
    future_patches.extend(target_i);
    next_patch.extend(next_patch_i);
    bar_history.extend(bar_history_i);
    next_bars.extend(next_bars_i);
    next_obs.extend(next_obs_i);
    next_static_obs.extend(next_static_i);
}

fn build_split_offsets(
    data_len: usize,
    k_patches: usize,
    patch_size: usize,
    split_kind: SplitKind,
) -> Vec<usize> {
    let min_offset = PRICE_DELTAS_PER_TICKER;
    let horizon = k_patches * patch_size;
    let next_latent_advance = patch_size;
    let max_target_advance = horizon
        .max(next_latent_advance)
        .max(LEJEPA_ROLLOUT_BARS as usize);
    let max_exclusive = data_len.saturating_sub(max_target_advance);
    if max_exclusive <= min_offset {
        return Vec::new();
    }
    let split_raw = min_offset + ((max_exclusive - min_offset) * 8 / 10).max(1);
    let split = align_up_to_step(split_raw, min_offset, patch_size);
    let train_max_exclusive = split.saturating_sub(max_target_advance);
    let (start, end) = match split_kind {
        SplitKind::Train => (min_offset, train_max_exclusive),
        SplitKind::Validation => (split, max_exclusive),
    };
    if start >= end {
        return Vec::new();
    }
    (start..end).step_by(patch_size).collect()
}

fn align_up_to_step(value: usize, origin: usize, step: usize) -> usize {
    let rem = (value - origin) % step;
    if rem == 0 {
        value
    } else {
        value + (step - rem)
    }
}

pub fn pretrain(args: PretrainArgs) -> Result<()> {
    assert_eq!(
        args.model_size,
        ModelVariant::UniformStream,
        "world-model pretraining currently supports --model-size uniform-stream only"
    );
    assert!(args.epochs > 0, "--epochs must be positive");
    if let Some(steps) = args.steps {
        assert!(steps > 0, "--steps must be positive when provided");
    }
    assert!(args.batch_size > 0, "--batch-size must be positive");
    assert!(args.k_patches > 0, "--k-patches must be positive");
    assert!(
        args.batches_per_epoch > 0,
        "--batches-per-epoch must be positive"
    );
    assert!(
        args.lambda_lat.is_finite() && args.lambda_lat >= 0.0,
        "--lambda-lat must be finite and non-negative"
    );
    assert!(
        args.lambda_sigreg.is_finite() && args.lambda_sigreg >= 0.0,
        "--lambda-sigreg must be finite and non-negative"
    );
    assert!(
        args.target_scale.is_finite() && args.target_scale > 0.0,
        "--target-scale must be finite and positive"
    );
    configure_threads();
    let device = tch::Device::cuda_if_available();
    println!("device is cuda: {}", device.is_cuda());
    configure_cuda();

    let run_dir =
        RunDir::create_fresh(RUNS_PATH, args.run.as_deref()).expect("failed to create run dir");
    println!("Run dir: {}", run_dir.root.display());

    let mut model_vs = nn::VarStore::new(device);
    let model = TradingModel::new_with_config(
        &model_vs.root(),
        TradingModelConfig {
            variant: args.model_size,
            ..TradingModelConfig::default()
        },
    );
    let start_weights = args.weights.as_deref().map(PathBuf::from);
    if let Some(path) = &start_weights {
        println!("Loading pretrain start weights from {}", path.display());
        let load_summary =
            load_var_store_partial(&mut model_vs, path).map_err(|err| anyhow!("{err}"))?;
        load_summary
            .require_complete()
            .map_err(|err| anyhow!("{err}"))?;
    }

    let patch_size = model.pretrain_patch_size();
    assert_eq!(
        args.k_patches as i64 * patch_size,
        args.k_patches as i64 * model.pretrain_patch_size()
    );
    let mut sampler = PretrainSampler::new(
        args.k_patches,
        patch_size as usize,
        args.target_scale,
        args.batches_per_epoch,
        device,
    );
    let mut head_vs = nn::VarStore::new(device);
    let heads = PretrainHeads::new(
        &head_vs.root(),
        model.pretrain_latent_dim(),
        args.k_patches as i64,
        patch_size,
    );
    if let Some(path) = start_weights.as_deref() {
        load_matching_pretrain_heads(&mut head_vs, path)?;
    }

    let mut named_vars = named_trainable_variables(&model_vs);
    named_vars.extend(
        named_trainable_variables(&head_vs)
            .into_iter()
            .map(|(name, tensor)| (format!("pretrain_heads.{name}"), tensor)),
    );
    let mut opt = Muon::new_named(
        &named_vars,
        MuonConfig {
            lr: MUON_LR,
            use_muon_for_2d: USE_MUON,
            momentum: MUON_MOMENTUM_WARMUP_START,
            adamw_lr: LEARNING_RATE,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            weight_decay: 0.0,
            adamw_wd: LEJEPA_WEIGHT_DECAY,
            force_adamw_name_substrings: vec![
                "policy_concentration".to_string(),
                "value_proj".to_string(),
                "forecast_".to_string(),
                "horizon_pos_proj".to_string(),
                "return_mean".to_string(),
                "bar_proj".to_string(),
                "bar_enrich_".to_string(),
                "lejepa_".to_string(),
                "probe_".to_string(),
            ],
            ..MuonConfig::default()
        },
    );
    let probe_named_vars = named_trainable_variables(&head_vs)
        .into_iter()
        .filter(|(name, _)| name.contains("probe_"))
        .collect::<Vec<_>>();
    let mut probe_opt = Muon::new_named(
        &probe_named_vars,
        MuonConfig {
            lr: MUON_LR,
            use_muon_for_2d: false,
            momentum: MUON_MOMENTUM_WARMUP_START,
            adamw_lr: LEJEPA_PROBE_LR,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            weight_decay: 0.0,
            adamw_wd: 0.0,
            quiet: true,
            ..MuonConfig::default()
        },
    );

    let mut optimizer_step = 0i64;
    let mut global_step = 0usize;
    let mut best_val = f64::INFINITY;
    let mut scalar_history = PretrainScalarHistory::default();
    let mut stop_requested = false;
    let final_path = run_dir.weights.join("pretrain_model.ot");
    let best_path = run_dir.weights.join("pretrain_model_best.ot");
    let final_heads_path = run_dir.weights.join("pretrain_heads.ot");
    let best_heads_path = run_dir.weights.join("pretrain_heads_best.ot");
    let mut train_epoch_log = BufWriter::new(File::create(
        run_dir.root.join("pretrain_train_epochs.csv"),
    )?);
    let mut validation_log =
        BufWriter::new(File::create(run_dir.root.join("pretrain_validation.csv"))?);
    writeln!(
        train_epoch_log,
        "epoch,global_step,total_loss,jepa_mse,sigreg,repr_std_mean,repr_std_min,pred_embed_std,target_embed_std,probe_mse,probe_mae,probe_bias,pred_abs,target_abs,pred_std,target_std,probe_terminal_mse,zero_mse,probe_explained_variance,next_lat,samples,batches"
    )?;
    writeln!(
        validation_log,
        "epoch,global_step,total_loss,jepa_mse,sigreg,repr_std_mean,repr_std_min,pred_embed_std,target_embed_std,probe_mse,probe_mae,probe_bias,pred_abs,target_abs,pred_std,target_std,probe_terminal_mse,zero_mse,probe_explained_variance,next_lat,rollout_mean_mse,rollout_sampled_mse,rollout_mse_delta,rollout_mse_delta_se,rollout_mse_t,rollout_mse_n,samples,tickers,batches"
    )?;
    let mut step_log = if args.log_step_losses {
        let mut log = BufWriter::new(File::create(run_dir.root.join("pretrain_train_steps.csv"))?);
        writeln!(
            log,
            "global_step,epoch,total_loss,jepa_mse,sigreg,repr_std_mean,repr_std_min,pred_embed_std,target_embed_std,probe_mse,probe_mae,probe_bias,pred_abs,target_abs,pred_std,target_std,probe_terminal_mse,zero_mse,probe_explained_variance,next_lat,samples"
        )?;
        Some(log)
    } else {
        None
    };

    'epoch_loop: for epoch in 1..=args.epochs {
        sampler.start_epoch();
        let mut train_epoch_loss = RunningLoss::new(device);
        let mut grad_norm_acc = GradNormAccum::default();
        println!(
            "pretrain epoch {epoch}/{} tickers={} batch_size={} batches_per_epoch={}",
            args.epochs,
            sampler.train_tickers.len(),
            args.batch_size,
            args.batches_per_epoch
        );

        while let Some(batch) = sampler.next_train_batch(args.batch_size) {
            global_step += 1;
            let losses = pretrain_loss(
                &model,
                &heads,
                &batch,
                args.objective,
                args.lambda_lat,
                args.lambda_sigreg,
                args.target_scale,
                true,
            );
            let batch_samples = batch.len() as usize;
            train_epoch_loss.add(&losses, batch_samples);
            assert_finite_loss(&losses.total, global_step);
            opt.zero_grad();
            losses.total.backward();
            grad_norm_acc.add(&pretrain_grad_norms(&named_vars, device));
            clip_all_grads(&named_vars, MAX_GRAD_NORM, device);
            opt.set_momentum(muon_momentum_for_step(optimizer_step));
            opt.step();
            optimizer_step += 1;

            let mut scalar_losses = None;
            if let Some(log) = step_log.as_mut() {
                let total_v = losses.total.double_value(&[]);
                let jepa_mse_v = losses.jepa_mse.double_value(&[]);
                let sigreg_v = losses.sigreg.double_value(&[]);
                let repr_std_mean_v = losses.repr_std_mean.double_value(&[]);
                let repr_std_min_v = losses.repr_std_min.double_value(&[]);
                let pred_embed_std_v = losses.pred_embed_std.double_value(&[]);
                let target_embed_std_v = losses.target_embed_std.double_value(&[]);
                let probe_nll_v = losses.probe_nll.double_value(&[]);
                let probe_mse_v = losses.probe_mse.double_value(&[]);
                let probe_mae_v = losses.probe_mae.double_value(&[]);
                let probe_bias_v = losses.probe_bias.double_value(&[]);
                let pred_abs_v = losses.pred_abs.double_value(&[]);
                let target_abs_v = losses.target_abs.double_value(&[]);
                let pred_std_v = losses.pred_std.double_value(&[]);
                let target_std_v = losses.target_std.double_value(&[]);
                let probe_terminal_mse_v = losses.probe_terminal_mse.double_value(&[]);
                let zero_mse_v = losses.zero_mse.double_value(&[]);
                let probe_explained_variance_v = losses.probe_explained_variance.double_value(&[]);
                let lat_v = losses.next_lat.double_value(&[]);
                writeln!(
                    log,
                    "{global_step},{epoch},{total_v:.9},{jepa_mse_v:.9},{sigreg_v:.9},{repr_std_mean_v:.9},{repr_std_min_v:.9},{pred_embed_std_v:.9},{target_embed_std_v:.9},{probe_mse_v:.9},{probe_mae_v:.9},{probe_bias_v:.9},{pred_abs_v:.9},{target_abs_v:.9},{pred_std_v:.9},{target_std_v:.9},{probe_terminal_mse_v:.9},{zero_mse_v:.9},{probe_explained_variance_v:.9},{lat_v:.9},{batch_samples}"
                )?;
                scalar_losses = Some((
                    total_v,
                    jepa_mse_v,
                    sigreg_v,
                    repr_std_mean_v,
                    repr_std_min_v,
                    pred_embed_std_v,
                    target_embed_std_v,
                    probe_nll_v,
                    probe_mse_v,
                    probe_mae_v,
                    probe_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    probe_terminal_mse_v,
                    zero_mse_v,
                    probe_explained_variance_v,
                    lat_v,
                ));
            }

            if global_step == 1 || global_step % 20 == 0 {
                let (
                    total_v,
                    jepa_mse_v,
                    sigreg_v,
                    repr_std_mean_v,
                    repr_std_min_v,
                    pred_embed_std_v,
                    target_embed_std_v,
                    _probe_nll_v,
                    probe_mse_v,
                    probe_mae_v,
                    probe_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    probe_terminal_mse_v,
                    zero_mse_v,
                    probe_explained_variance_v,
                    lat_v,
                ) = scalar_losses.unwrap_or_else(|| {
                    (
                        losses.total.double_value(&[]),
                        losses.jepa_mse.double_value(&[]),
                        losses.sigreg.double_value(&[]),
                        losses.repr_std_mean.double_value(&[]),
                        losses.repr_std_min.double_value(&[]),
                        losses.pred_embed_std.double_value(&[]),
                        losses.target_embed_std.double_value(&[]),
                        losses.probe_nll.double_value(&[]),
                        losses.probe_mse.double_value(&[]),
                        losses.probe_mae.double_value(&[]),
                        losses.probe_bias.double_value(&[]),
                        losses.pred_abs.double_value(&[]),
                        losses.target_abs.double_value(&[]),
                        losses.pred_std.double_value(&[]),
                        losses.target_std.double_value(&[]),
                        losses.probe_terminal_mse.double_value(&[]),
                        losses.zero_mse.double_value(&[]),
                        losses.probe_explained_variance.double_value(&[]),
                        losses.next_lat.double_value(&[]),
                    )
                });
                println!(
                    "pretrain epoch {epoch} step {global_step} train total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_mse={:.6} probe_mae={:.6} probe_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} probe_terminal_mse={:.6} zero_mse={:.6} probe_ev={:.2}% next_lat={:.6}",
                    total_v,
                    jepa_mse_v,
                    sigreg_v,
                    repr_std_mean_v,
                    repr_std_min_v,
                    pred_embed_std_v,
                    target_embed_std_v,
                    probe_mse_v,
                    probe_mae_v,
                    probe_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    probe_terminal_mse_v,
                    zero_mse_v,
                    probe_explained_variance_v * 100.0,
                    lat_v,
                );
            }

            if args.validate_every > 0 && global_step % args.validate_every == 0 {
                let val = validate_full(
                    &model,
                    &heads,
                    &mut sampler,
                    args.batch_size,
                    validation_batch_cap(args.validation_batches),
                    args.objective,
                    args.lambda_lat,
                    args.lambda_sigreg,
                    device,
                );
                println!(
                    "pretrain step {global_step} validation total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_mse={:.6} probe_mae={:.6} probe_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} probe_terminal_mse={:.6} zero_mse={:.6} probe_ev={:.2}% next_lat={:.6} rollout_mean_mse={:.6} rollout_sampled_mse={:.6} rollout_mse_delta={:.6} rollout_mse_delta_se={:.6} rollout_mse_t={:.6} rollout_mse_n={:.6} samples={} tickers={} batches={}",
                    val.total,
                    val.jepa_mse,
                    val.sigreg,
                    val.repr_std_mean,
                    val.repr_std_min,
                    val.pred_embed_std,
                    val.target_embed_std,
                    val.probe_mse,
                    val.probe_mae,
                    val.probe_bias,
                    val.pred_abs,
                    val.target_abs,
                    val.pred_std,
                    val.target_std,
                    val.probe_terminal_mse,
                    val.zero_mse,
                    val.probe_explained_variance * 100.0,
                    val.next_lat,
                    val.rollout_mean_mse,
                    val.rollout_sampled_mse,
                    val.rollout_mse_delta,
                    val.rollout_mse_delta_se,
                    val.rollout_mse_t,
                    val.rollout_mse_n,
                    val.samples,
                    val.tickers,
                    val.batches
                );
                writeln!(
                    validation_log,
                    "step:{global_step},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
                    val.total,
                    val.jepa_mse,
                    val.sigreg,
                    val.repr_std_mean,
                    val.repr_std_min,
                    val.pred_embed_std,
                    val.target_embed_std,
                    val.probe_mse,
                    val.probe_mae,
                    val.probe_bias,
                    val.pred_abs,
                    val.target_abs,
                    val.pred_std,
                    val.target_std,
                    val.probe_terminal_mse,
                    val.zero_mse,
                    val.probe_explained_variance,
                    val.next_lat,
                    val.rollout_mean_mse,
                    val.rollout_sampled_mse,
                    val.rollout_mse_delta,
                    val.rollout_mse_delta_se,
                    val.rollout_mse_t,
                    val.rollout_mse_n,
                    val.samples,
                    val.tickers,
                    val.batches
                )?;
                if val.total < best_val {
                    best_val = val.total;
                    model_vs.save(&best_path)?;
                    head_vs.save(&best_heads_path)?;
                    println!("Saved best pretrained model: {}", best_path.display());
                }
            }

            if args.checkpoint_every > 0 && global_step % args.checkpoint_every == 0 {
                let path = pretrain_step_model_path(&run_dir.weights, global_step);
                let heads_path = pretrain_step_heads_path(&run_dir.weights, global_step);
                model_vs.save(&path)?;
                head_vs.save(&heads_path)?;
                println!(
                    "Saved pretrained checkpoint: {} and {}",
                    path.display(),
                    heads_path.display()
                );
            }

            if args.steps.is_some_and(|max_steps| global_step >= max_steps) {
                stop_requested = true;
                break;
            }
        }

        let train = train_epoch_loss.finish();
        println!(
            "pretrain epoch {epoch} train_mean total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_mse={:.6} probe_mae={:.6} probe_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} probe_terminal_mse={:.6} zero_mse={:.6} probe_ev={:.2}% next_lat={:.6} samples={} batches={}",
            train.total,
            train.jepa_mse,
            train.sigreg,
            train.repr_std_mean,
            train.repr_std_min,
            train.pred_embed_std,
            train.target_embed_std,
            train.probe_mse,
            train.probe_mae,
            train.probe_bias,
            train.pred_abs,
            train.target_abs,
            train.pred_std,
            train.target_std,
            train.probe_terminal_mse,
            train.zero_mse,
            train.probe_explained_variance * 100.0,
            train.next_lat,
            train.samples,
            train.batches
        );
        let grad_norms = grad_norm_acc.mean();
        println!(
            "pretrain epoch {epoch} grad_norms grad_total={:.6} grad_ar={:.6} grad_encoder={:.6} grad_other={:.6} pnorm_ar={:.6} pnorm_encoder={:.6} steps={}",
            grad_norms.grad_total,
            grad_norms.grad_ar,
            grad_norms.grad_encoder,
            grad_norms.grad_other,
            grad_norms.pnorm_ar,
            grad_norms.pnorm_encoder,
            grad_norm_acc.steps
        );
        writeln!(
            train_epoch_log,
            "{epoch},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{}",
            train.total,
            train.jepa_mse,
            train.sigreg,
            train.repr_std_mean,
            train.repr_std_min,
            train.pred_embed_std,
            train.target_embed_std,
            train.probe_mse,
            train.probe_mae,
            train.probe_bias,
            train.pred_abs,
            train.target_abs,
            train.pred_std,
            train.target_std,
            train.probe_terminal_mse,
            train.zero_mse,
            train.probe_explained_variance,
            train.next_lat,
            train.samples,
            train.batches
        )?;
        train_epoch_log.flush()?;
        if let Some(log) = step_log.as_mut() {
            log.flush()?;
        }

        if args.objective == PretrainObjective::Lejepa && args.probe_epochs > 0 && !stop_requested {
            let probe = train_detached_probe(
                &model,
                &heads,
                &mut sampler,
                args.batch_size,
                args.probe_epochs,
                args.target_scale,
                &mut probe_opt,
                &probe_named_vars,
                device,
            );
            println!(
                "pretrain epoch {epoch} detached_probe_train probe_nll={:.6} ohlc_mse={:.6} ohlc_mae={:.6} pred_std={:.6} target_std={:.6} pred_sigma={:.6} pred_std_level={:.6} target_std_level={:.6} probe_ev_level={:.2}% pred_sigma_level={:.6} grad_probe={:.6} pnorm_probe={:.6} samples={} batches={} probe_epochs={}",
                probe.probe_nll,
                probe.probe_mse,
                probe.probe_mae,
                probe.pred_std,
                probe.target_std,
                probe.pred_sigma,
                probe.pred_std_level,
                probe.target_std_level,
                probe.probe_ev_level * 100.0,
                probe.pred_sigma_level,
                probe.grad_probe,
                probe.pnorm_probe,
                probe.samples,
                probe.batches,
                args.probe_epochs
            );
        }

        let val = validate_full(
            &model,
            &heads,
            &mut sampler,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            args.objective,
            args.lambda_lat,
            args.lambda_sigreg,
            device,
        );
        println!(
            "pretrain epoch {epoch} validation total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_mse={:.6} probe_mae={:.6} probe_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} probe_terminal_mse={:.6} zero_mse={:.6} probe_ev={:.2}% next_lat={:.6} rollout_mean_mse={:.6} rollout_sampled_mse={:.6} rollout_mse_delta={:.6} rollout_mse_delta_se={:.6} rollout_mse_t={:.6} rollout_mse_n={:.6} rollout_mean_dclose={:.9} rollout_mean_dclose_std={:.9} rollout_sampled_dclose={:.9} rollout_sampled_dclose_std={:.9} samples={} tickers={} batches={}",
            val.total,
            val.jepa_mse,
            val.sigreg,
            val.repr_std_mean,
            val.repr_std_min,
            val.pred_embed_std,
            val.target_embed_std,
            val.probe_mse,
            val.probe_mae,
            val.probe_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.probe_terminal_mse,
            val.zero_mse,
            val.probe_explained_variance * 100.0,
            val.next_lat,
            val.rollout_mean_mse,
            val.rollout_sampled_mse,
            val.rollout_mse_delta,
            val.rollout_mse_delta_se,
            val.rollout_mse_t,
            val.rollout_mse_n,
            val.rollout_mean_dclose,
            val.rollout_mean_dclose_std,
            val.rollout_sampled_dclose,
            val.rollout_sampled_dclose_std,
            val.samples,
            val.tickers,
            val.batches
        );
        println!(
            "pretrain epoch {epoch} skill ev_correct={:.6} ev_shuffled={:.6} ev_zero={:.6} belief_spread={:.6} belief_norm={:.6} batches={}",
            val.skill_ev_correct,
            val.skill_ev_shuffled,
            val.skill_ev_zero,
            val.skill_belief_spread,
            val.skill_belief_norm,
            val.skill_batches
        );
        writeln!(
            validation_log,
            "{epoch},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
            val.total,
            val.jepa_mse,
            val.sigreg,
            val.repr_std_mean,
            val.repr_std_min,
            val.pred_embed_std,
            val.target_embed_std,
            val.probe_mse,
            val.probe_mae,
            val.probe_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.probe_terminal_mse,
            val.zero_mse,
            val.probe_explained_variance,
            val.next_lat,
            val.rollout_mean_mse,
            val.rollout_sampled_mse,
            val.rollout_mse_delta,
            val.rollout_mse_delta_se,
            val.rollout_mse_t,
            val.rollout_mse_n,
            val.samples,
            val.tickers,
            val.batches
        )?;
        validation_log.flush()?;
        scalar_history.push(&train, &val);
        write_pretrain_scalar_meta_reports(&run_dir.gens, epoch, global_step, &scalar_history)?;
        write_pretrain_diagnostics(
            &model,
            &heads,
            &mut sampler,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            args.objective,
            epoch,
            global_step,
            &run_dir.gens,
            device,
        )?;
        if val.total < best_val {
            best_val = val.total;
            model_vs.save(&best_path)?;
            head_vs.save(&best_heads_path)?;
            println!("Saved best pretrained model: {}", best_path.display());
        }
        if stop_requested {
            break 'epoch_loop;
        }
    }

    if best_val == f64::INFINITY {
        let val = validate_full(
            &model,
            &heads,
            &mut sampler,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            args.objective,
            args.lambda_lat,
            args.lambda_sigreg,
            device,
        );
        best_val = val.total;
        writeln!(
            validation_log,
            "final,{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
            val.total,
            val.jepa_mse,
            val.sigreg,
            val.repr_std_mean,
            val.repr_std_min,
            val.pred_embed_std,
            val.target_embed_std,
            val.probe_mse,
            val.probe_mae,
            val.probe_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.probe_terminal_mse,
            val.zero_mse,
            val.probe_explained_variance,
            val.next_lat,
            val.rollout_mean_mse,
            val.rollout_sampled_mse,
            val.rollout_mse_delta,
            val.rollout_mse_delta_se,
            val.rollout_mse_t,
            val.rollout_mse_n,
            val.samples,
            val.tickers,
            val.batches
        )?;
        validation_log.flush()?;
        model_vs.save(&best_path)?;
        head_vs.save(&best_heads_path)?;
        println!("Saved best pretrained model: {}", best_path.display());
    }

    if best_path.exists() {
        model_vs.load(&best_path)?;
    }
    if best_heads_path.exists() {
        head_vs.load(&best_heads_path)?;
    }
    model_vs.save(&final_path)?;
    head_vs.save(&final_heads_path)?;
    println!(
        "Saved final pretrained model: {} (best validation total_loss {:.6})",
        final_path.display(),
        best_val
    );
    Ok(())
}

fn load_matching_pretrain_heads(head_vs: &mut nn::VarStore, model_path: &Path) -> Result<()> {
    let Some(heads_path) = matching_pretrain_heads_path(model_path) else {
        return Ok(());
    };
    if !heads_path.exists() {
        return Err(anyhow!(
            "matching pretrain heads {} not found for model checkpoint {}",
            heads_path.display(),
            model_path.display()
        ));
    }
    let load_summary =
        load_var_store_partial(head_vs, &heads_path).map_err(|err| anyhow!("{err}"))?;
    if let Err(err) = load_summary.require_complete() {
        println!(
            "Warm-starting pretrain heads with partial load from {} ({err}); newly added head params are freshly initialized",
            heads_path.display()
        );
    } else {
        println!("Loaded pretrain heads from {}", heads_path.display());
    }
    Ok(())
}

fn matching_pretrain_heads_path(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let name = model_path.file_name()?.to_str()?;
    match name {
        "pretrain_model.ot" => Some(parent.join("pretrain_heads.ot")),
        "pretrain_model_best.ot" => Some(parent.join("pretrain_heads_best.ot")),
        _ => name
            .strip_prefix("pretrain_step")
            .and_then(|suffix| suffix.strip_suffix(".ot"))
            .map(|step| parent.join(format!("pretrain_heads_step{step}.ot"))),
    }
}

fn pretrain_step_model_path(weights_dir: &Path, global_step: usize) -> PathBuf {
    weights_dir.join(format!("pretrain_step{global_step}.ot"))
}

fn pretrain_step_heads_path(weights_dir: &Path, global_step: usize) -> PathBuf {
    weights_dir.join(format!("pretrain_heads_step{global_step}.ot"))
}

fn pretrain_loss(
    model: &TradingModel,
    heads: &PretrainHeads,
    batch: &PretrainBatch,
    objective: PretrainObjective,
    lambda_lat: f64,
    lambda_sigreg: f64,
    target_scale: f64,
    train: bool,
) -> PretrainLoss {
    match objective {
        PretrainObjective::MeanMse => {
            mean_mse_pretrain_loss(model, heads, batch, lambda_lat, train)
        }
        PretrainObjective::Lejepa => {
            lejepa_pretrain_loss(model, heads, batch, lambda_sigreg, target_scale, train)
        }
    }
}

fn mean_mse_pretrain_loss(
    model: &TradingModel,
    heads: &PretrainHeads,
    batch: &PretrainBatch,
    lambda_lat: f64,
    train: bool,
) -> PretrainLoss {
    let batch_size = batch.obs.size()[0];
    let layout_len = model.pretrain_layout_len();
    let layouts = model
        .uniform_stream_layout_from_raw_input(&batch.obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);

    let (patch_tokens, latent) = if lambda_lat == 0.0 {
        let patch_tokens = autocast(false, || {
            model.pretrain_patch_tokens(&layouts, &batch.static_obs, batch_size)
        });
        (patch_tokens, None)
    } else {
        let (patch_tokens, latent) = autocast(false, || {
            model.pretrain_patch_tokens_and_actor_latents(&layouts, &batch.static_obs, batch_size)
        });
        (patch_tokens, Some(latent))
    };
    let (forecast_tokens, forecast_batch, forecast_tickers) =
        heads.forecast_tokens(&patch_tokens, train);
    let (repr_std_mean, repr_std_min) = representation_std_metrics(&patch_tokens);
    debug_assert_eq!(forecast_batch, batch_size);
    debug_assert_eq!(forecast_tickers, TICKERS_COUNT as i64);
    let forecast_readout = heads.forecast_readout(&forecast_tokens, train);
    let return_target = cumulative_future_returns(&batch.future_patches);
    let return_pred =
        heads.return_mean_from_readout(&forecast_readout, forecast_batch, forecast_tickers);
    let probe_mse = return_pred.mse_loss(&return_target, Reduction::Mean);
    let return_err = &return_pred - &return_target;
    let probe_mae = return_err.abs().mean(Kind::Float);
    let probe_bias = return_err.mean(Kind::Float);
    let pred_abs = return_pred.abs().mean(Kind::Float);
    let target_abs = return_target.abs().mean(Kind::Float);
    let pred_std = return_pred.std(false);
    let target_std = return_target.std(false);
    let terminal_idx = heads.horizon - 1;
    let terminal_pred = return_pred.select(-1, terminal_idx);
    let terminal_target = return_target.select(-1, terminal_idx);
    let probe_terminal_mse = terminal_pred.mse_loss(&terminal_target, Reduction::Mean);
    let zero_mse = return_target.pow_tensor_scalar(2.0).mean(Kind::Float);
    let probe_explained_variance = explained_variance_tensor(&probe_mse, &zero_mse);
    let base_loss = probe_mse.shallow_clone();

    if lambda_lat == 0.0 {
        let next_lat = Tensor::zeros([], (Kind::Float, pred_abs.device()));
        return PretrainLoss {
            total: base_loss,
            jepa_mse: zero_like_scalar(&probe_mse),
            sigreg: zero_like_scalar(&probe_mse),
            repr_std_mean,
            repr_std_min,
            pred_embed_std: zero_like_scalar(&probe_mse),
            target_embed_std: zero_like_scalar(&probe_mse),
            probe_nll: zero_like_scalar(&probe_mse),
            probe_mae,
            probe_mse,
            pred_std,
            target_std,
            probe_bias,
            pred_abs,
            target_abs,
            next_lat,
            probe_terminal_mse,
            zero_mse,
            probe_explained_variance,
        };
    }

    let latent = latent.expect("latent pretrain state should be computed when lambda_lat > 0");
    let next_layouts = model
        .uniform_stream_layout_from_raw_input(&batch.next_obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);
    let next_latent = tch::no_grad(|| {
        autocast(false, || {
            model.pretrain_actor_latents(&next_layouts, &batch.next_static_obs, batch_size)
        })
    });
    let pred_next_latent = heads.predict_next_latent(&latent, &batch.next_patch);
    let latent_loss = pred_next_latent.smooth_l1_loss(&next_latent, Reduction::Mean, 1.0);
    let total = &base_loss + &latent_loss * lambda_lat;
    PretrainLoss {
        total,
        jepa_mse: zero_like_scalar(&probe_mse),
        sigreg: zero_like_scalar(&probe_mse),
        repr_std_mean,
        repr_std_min,
        pred_embed_std: zero_like_scalar(&probe_mse),
        target_embed_std: zero_like_scalar(&probe_mse),
        probe_nll: zero_like_scalar(&probe_mse),
        probe_mae,
        probe_mse,
        pred_std,
        target_std,
        probe_bias,
        pred_abs,
        target_abs,
        next_lat: latent_loss,
        probe_terminal_mse,
        zero_mse,
        probe_explained_variance,
    }
}

fn lejepa_pretrain_loss(
    model: &TradingModel,
    heads: &PretrainHeads,
    batch: &PretrainBatch,
    lambda_sigreg: f64,
    target_scale: f64,
    train: bool,
) -> PretrainLoss {
    let _ = model;

    let full = Tensor::cat(&[&batch.bar_history, &batch.next_bars.narrow(2, 0, 1)], 2);
    let all_tokens = autocast(false, || heads.encode_bar_tokens(&full, train));
    let length = batch.bar_history.size()[2];
    let bar_tokens = all_tokens.narrow(2, 0, length);
    // Un-detached shift-by-1 target (le-wm recipe): the clean next-token target
    // carries gradient into the shared online encoder through both the flow's
    // x-target and its noised input. SIGReg, not stop-grad, prevents collapse.
    let target_bar_tokens = all_tokens.narrow(2, 1, length);
    let latest_token = all_tokens.select(2, length);
    // Drift-mitigation noisy context: mix fixed unit noise into the predictor
    // INPUT only (same 10% as rollout). Clean targets / SIGReg keep the noiseless
    // encoder tokens.
    let noisy_bar_tokens =
        &bar_tokens * (1.0 - LEJEPA_CTX_NOISE_MIX) + Tensor::randn_like(&bar_tokens) * LEJEPA_CTX_NOISE_MIX;
    let belief = heads
        .predict_lejepa_bar_predictions(&noisy_bar_tokens, train)
        .belief;
    let latent_dim = heads.latent_dim;
    let rows = target_bar_tokens.numel() as i64 / latent_dim;
    // Attached belief conditioning and clean x-target both flow gradient to the
    // encoder (le-wm recipe): the flow is the sole dynamics loss.
    let ctx = belief.reshape([rows, latent_dim]);
    let clean = target_bar_tokens.reshape([rows, latent_dim]);
    let pred_loss = lejepa_flow_loss(heads, &ctx, &clean);

    let total_positions = all_tokens.size()[2];
    let k = LEJEPA_SIGREG_POSITIONS.min(total_positions);
    let perm = Tensor::randperm(total_positions, (Kind::Int64, all_tokens.device()));
    let sample_idx = Tensor::cat(
        &[
            &perm.narrow(0, 0, k - 1),
            &Tensor::from_slice(&[total_positions - 1]).to_device(all_tokens.device()),
        ],
        0,
    );
    let sigreg_tokens = all_tokens.index_select(2, &sample_idx);
    let batch_tickers = sigreg_tokens.size()[0] * sigreg_tokens.size()[1];
    let sigreg = sigreg_loss(&sigreg_tokens.permute([2, 0, 1, 3]).contiguous().reshape([
        k,
        batch_tickers,
        heads.latent_dim,
    ]));
    let (repr_std_mean, repr_std_min) = representation_std_metrics(&latest_token);
    let target_embed_std = target_bar_tokens.std(false);

    let total = &pred_loss + &sigreg * lambda_sigreg;

    // jepa_mse column = detached one-shot flow estimate MSE (x_pred from the zero
    // prior at signal 0), a stable dashboard proxy for the deterministic next-token
    // prediction. Gradients never touch these.
    let (jepa_mse, pred_embed_std) = tch::no_grad(|| {
        let device = clean.device();
        let zero = Tensor::zeros_like(&clean);
        let signal0 = Tensor::zeros([rows], (Kind::Int64, device));
        let step0 = Tensor::zeros([rows], (Kind::Int64, device));
        let x_pred = heads.lejepa_flow_predict(&zero, &signal0, &step0, &ctx.detach());
        (
            x_pred.mse_loss(&clean, Reduction::Mean),
            x_pred.std(false),
        )
    });

    let probe_target = scaled_next_ohlc_features(&batch.next_bars, target_scale);
    let probe = ohlc_probe_metrics(
        heads,
        &latest_token.detach().unsqueeze(2),
        &probe_target,
    );
    let zero_mse = probe_target.pow_tensor_scalar(2.0).mean(Kind::Float);
    let probe_explained_variance = explained_variance_tensor(&probe.probe_mse, &zero_mse);
    let next_lat = zero_like_scalar(&jepa_mse);
    PretrainLoss {
        total,
        jepa_mse,
        sigreg,
        repr_std_mean,
        repr_std_min,
        pred_embed_std,
        target_embed_std,
        probe_nll: probe.probe_nll,
        probe_mae: probe.probe_mae,
        probe_mse: probe.probe_mse,
        pred_std: probe.pred_std,
        target_std: probe.target_std,
        probe_bias: probe.probe_bias,
        pred_abs: probe.pred_abs,
        target_abs: probe.target_abs,
        next_lat,
        probe_terminal_mse: probe.probe_terminal_mse,
        zero_mse,
        probe_explained_variance,
    }
}

// dreamer4 shortcut-forcing objective. Per batch, a Bernoulli(p=1-1/6) picks the
// SHORTCUT branch (self-consistency distillation of a size-d step into two size-d/2
// steps, flow-space MSE weighted by (1-t)^2, plus the ramp-weighted x-space MSE to
// clean data that dreamer4 keeps on every batch), else the PLAIN branch (per-position
// diffusion-forcing x-prediction MSE, ramp-weighted 0.9t+0.1). `ctx` (belief) and
// `clean` (next-token target) are attached [rows, latent_dim] so gradient reaches
// the encoder through the flow.
fn lejepa_flow_loss(heads: &PretrainHeads, ctx: &Tensor, clean: &Tensor) -> Tensor {
    let rows = clean.size()[0];
    let device = clean.device();
    let shortcut = Tensor::rand([], (Kind::Float, device)).double_value(&[]) < LEJEPA_PROB_SHORTCUT;
    if !shortcut {
        // Plain branch: signal ~ Uniform{0..K_MAX-1} independently per position.
        let signal = Tensor::randint(LEJEPA_K_MAX, [rows], (Kind::Int64, device));
        let step_log2 = Tensor::zeros([rows], (Kind::Int64, device));
        let t = signal.to_kind(Kind::Float) / LEJEPA_K_MAX as f64;
        let t_col = t.unsqueeze(-1);
        let noise = Tensor::randn_like(clean);
        let noised = &noise * (&t_col * -1.0 + 1.0) + clean * &t_col;
        let x_pred = heads.lejepa_flow_predict(&noised, &signal, &step_log2, ctx);
        let per_sample = (x_pred - clean)
            .square()
            .mean_dim([-1i64].as_slice(), false, Kind::Float);
        let ramp = &t * 0.9 + 0.1;
        return (per_sample * ramp).mean(Kind::Float);
    }
    // Shortcut branch: step size d = 2^log2 with log2 ~ Uniform{1..6}; signal ~
    // Uniform{0..K_MAX-1} discretized down to a multiple of d.
    let step_log2 = Tensor::randint_low(1, LEJEPA_NUM_STEP_SIZES_LOG2, [rows], (Kind::Int64, device));
    let d_f = step_log2.to_kind(Kind::Float).exp2();
    let raw = Tensor::randint(LEJEPA_K_MAX, [rows], (Kind::Int64, device)).to_kind(Kind::Float);
    let sig_f = (&raw / &d_f).floor() * &d_f;
    let sig = sig_f.to_kind(Kind::Int64);
    let half_f = &d_f / 2.0;
    let t1 = &sig_f / LEJEPA_K_MAX as f64;
    let t1_col = t1.unsqueeze(-1);
    let noise = Tensor::randn_like(clean);
    let noised = &noise * (&t1_col * -1.0 + 1.0) + clean * &t1_col;
    let step_log2_minus1 = &step_log2 - 1;
    // Self-consistency target: two size-(d/2) sub-steps, averaged flow, stop-grad.
    let target_flow = tch::no_grad(|| {
        let x1 = heads.lejepa_flow_predict(&noised, &sig, &step_log2_minus1, ctx);
        let flow1 = heads.lejepa_flow_velocity(&x1, &noised, &sig);
        let denoised = &noised + &flow1 * (&half_f / LEJEPA_K_MAX as f64).unsqueeze(-1);
        let sig2_f = &sig_f + &half_f;
        let sig2 = sig2_f.to_kind(Kind::Int64);
        let x2 = heads.lejepa_flow_predict(&denoised, &sig2, &step_log2_minus1, ctx);
        let flow2 = heads.lejepa_flow_velocity(&x2, &denoised, &sig2);
        ((flow1 + flow2) / 2.0).detach()
    });
    // Main full-step-d prediction: regressed toward the target in flow-space, plus
    // the always-on x-space grounding to clean data (dreamer4 keeps both terms).
    let x_pred = heads.lejepa_flow_predict(&noised, &sig, &step_log2, ctx);
    let pred_flow = heads.lejepa_flow_velocity(&x_pred, &noised, &sig);
    let weight = (&t1 * -1.0 + 1.0).square();
    let consistency = (pred_flow - target_flow)
        .square()
        .mean_dim([-1i64].as_slice(), false, Kind::Float)
        * weight;
    let ramp = &t1 * 0.9 + 0.1;
    let grounding = (x_pred - clean)
        .square()
        .mean_dim([-1i64].as_slice(), false, Kind::Float)
        * ramp;
    (consistency + grounding).mean(Kind::Float)
}

fn cumulative_future_returns(future_patches: &Tensor) -> Tensor {
    let size = future_patches.size();
    future_patches
        .view([size[0], size[1], size[2] * size[3]])
        .cumsum(-1, Kind::Float)
}

fn scaled_next_ohlc_features(next_ohlc_patch: &Tensor, target_scale: f64) -> Tensor {
    next_ohlc_patch.narrow(2, 0, 1) * target_scale
}

fn zero_like_scalar(reference: &Tensor) -> Tensor {
    Tensor::zeros([], (Kind::Float, reference.device()))
}

fn explained_variance_tensor(mse: &Tensor, zero_mse: &Tensor) -> Tensor {
    Tensor::ones([], (Kind::Float, mse.device())) - mse / zero_mse.clamp_min(1e-12)
}

fn explained_variance_value(mse: f64, zero_mse: f64) -> f64 {
    if zero_mse <= 1e-12 || !zero_mse.is_finite() || !mse.is_finite() {
        0.0
    } else {
        1.0 - mse / zero_mse
    }
}

fn sigreg_loss(tokens: &Tensor) -> Tensor {
    let size = tokens.size();
    let samples = size[1];
    let dim = size[2];
    let proj_in = tokens.to_kind(Kind::Float);
    let mut directions = Tensor::randn(
        [dim, LEJEPA_SIGREG_PROJECTIONS],
        (Kind::Float, tokens.device()),
    );
    directions = &directions
        / directions
            .norm_scalaropt_dim(2, [0i64].as_slice(), true)
            .clamp_min(1e-7);
    let t = Tensor::linspace(
        0.0,
        3.0,
        LEJEPA_SIGREG_KNOTS,
        (Kind::Float, tokens.device()),
    );
    let dt = 3.0 / (LEJEPA_SIGREG_KNOTS - 1) as f64;
    let weights = Tensor::full(
        [LEJEPA_SIGREG_KNOTS],
        2.0 * dt,
        (Kind::Float, tokens.device()),
    );
    let _ = weights.narrow(0, 0, 1).fill_(dt);
    let _ = weights.narrow(0, LEJEPA_SIGREG_KNOTS - 1, 1).fill_(dt);
    let phi = (-t.square() * 0.5).exp();
    let weights = weights * &phi;
    let proj = proj_in.matmul(&directions);
    let x_t = proj.unsqueeze(-1) * t.view([1, 1, 1, -1]);
    let cos_err = x_t.cos().mean_dim([1i64].as_slice(), false, Kind::Float) - phi.view([1, 1, -1]);
    let sin_err = x_t.sin().mean_dim([1i64].as_slice(), false, Kind::Float);
    let err = cos_err.square() + sin_err.square();
    let weighted =
        (err * weights.view([1, 1, -1])).sum_dim_intlist([-1i64].as_slice(), false, Kind::Float);
    weighted.mean(Kind::Float) * samples as f64
}

fn representation_std_metrics(tokens: &Tensor) -> (Tensor, Tensor) {
    let dim = *tokens.size().last().unwrap();
    let flat = tokens.view([-1, dim]).to_kind(Kind::Float);
    let mean = flat.mean_dim([0i64].as_slice(), true, Kind::Float);
    let feature_std = (&flat - &mean)
        .pow_tensor_scalar(2.0)
        .mean_dim([0i64].as_slice(), false, Kind::Float)
        .clamp_min(1e-12)
        .sqrt();
    (feature_std.mean(Kind::Float), feature_std.min())
}

struct ProbeLoss {
    probe_nll: Tensor,
    probe_mae: Tensor,
    probe_mse: Tensor,
    pred_std: Tensor,
    target_std: Tensor,
    probe_bias: Tensor,
    pred_abs: Tensor,
    target_abs: Tensor,
    probe_terminal_mse: Tensor,
    pred_sigma: Tensor,
    pred_std_level: Tensor,
    target_std_level: Tensor,
    probe_ev_level: Tensor,
    pred_sigma_level: Tensor,
}

// Gaussian NLL of the probe decode over positions kept by `mask` [.., 1]. Used to
// train the probe on flow x-predictions at signal>=32 only.
fn masked_probe_nll(heads: &PretrainHeads, latent: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    let (mean, logvar) = heads.probe_ohlc_features(latent);
    let err = &mean - target;
    let nll_elem = &logvar + err.pow_tensor_scalar(2.0) * logvar.neg().exp();
    let features = *nll_elem.size().last().unwrap() as f64;
    let denom = (mask.sum(Kind::Float) * features).clamp_min(1.0);
    (nll_elem * mask).sum(Kind::Float) / denom * 0.5
}

fn ohlc_probe_metrics(heads: &PretrainHeads, belief: &Tensor, target: &Tensor) -> ProbeLoss {
    let (mean, logvar) = heads.probe_ohlc_features(belief);
    let err = &mean - target;
    let probe_mse = mean.mse_loss(target, Reduction::Mean);
    let probe_mae = err.abs().mean(Kind::Float);
    let probe_bias = err.mean(Kind::Float);
    let pred_abs = mean.abs().mean(Kind::Float);
    let target_abs = target.abs().mean(Kind::Float);
    let pred_std = mean.std(false);
    let target_std = target.std(false);
    let probe_terminal_mse = mean
        .select(2, 0)
        .mse_loss(&target.select(2, 0), Reduction::Mean);

    let nll_elem = &logvar + err.pow_tensor_scalar(2.0) * logvar.neg().exp();
    let probe_nll = nll_elem.mean(Kind::Float) * 0.5;
    let pred_sigma = (&logvar * 0.5).exp().mean(Kind::Float);

    let mean_level = mean.select(3, 0);
    let target_level = target.select(3, 0);
    let logvar_level = logvar.select(3, 0);
    let pred_std_level = mean_level.std(false);
    let target_std_level = target_level.std(false);
    let level_mse = mean_level.mse_loss(&target_level, Reduction::Mean);
    let level_zero_mse = target_level.pow_tensor_scalar(2.0).mean(Kind::Float);
    let probe_ev_level = explained_variance_tensor(&level_mse, &level_zero_mse);
    let pred_sigma_level = (&logvar_level * 0.5).exp().mean(Kind::Float);

    ProbeLoss {
        probe_nll,
        probe_mae,
        probe_mse,
        pred_std,
        target_std,
        probe_bias,
        pred_abs,
        target_abs,
        probe_terminal_mse,
        pred_sigma,
        pred_std_level,
        target_std_level,
        probe_ev_level,
        pred_sigma_level,
    }
}

fn predict_future_returns(
    model: &TradingModel,
    heads: &PretrainHeads,
    batch: &PretrainBatch,
) -> Tensor {
    let batch_size = batch.obs.size()[0];
    let layout_len = model.pretrain_layout_len();
    let layouts = model
        .uniform_stream_layout_from_raw_input(&batch.obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);
    let patch_tokens = autocast(false, || {
        model.pretrain_patch_tokens(&layouts, &batch.static_obs, batch_size)
    });
    heads.predict_return_mean(&patch_tokens, false)
}

struct ValidationLoss {
    total: f64,
    jepa_mse: f64,
    sigreg: f64,
    repr_std_mean: f64,
    repr_std_min: f64,
    pred_embed_std: f64,
    target_embed_std: f64,
    probe_nll: f64,
    probe_mae: f64,
    probe_mse: f64,
    pred_std: f64,
    target_std: f64,
    probe_bias: f64,
    pred_abs: f64,
    target_abs: f64,
    next_lat: f64,
    probe_terminal_mse: f64,
    zero_mse: f64,
    probe_explained_variance: f64,
    rollout_mean_mse: f64,
    rollout_sampled_mse: f64,
    rollout_mse_delta: f64,
    rollout_mse_delta_se: f64,
    rollout_mse_t: f64,
    rollout_mse_n: f64,
    rollout_mean_dclose: f64,
    rollout_mean_dclose_std: f64,
    rollout_sampled_dclose: f64,
    rollout_sampled_dclose_std: f64,
    skill_ev_correct: f64,
    skill_ev_shuffled: f64,
    skill_ev_zero: f64,
    skill_belief_spread: f64,
    skill_belief_norm: f64,
    skill_batches: usize,
    samples: usize,
    tickers: usize,
    batches: usize,
}

struct PretrainLoss {
    total: Tensor,
    jepa_mse: Tensor,
    sigreg: Tensor,
    repr_std_mean: Tensor,
    repr_std_min: Tensor,
    pred_embed_std: Tensor,
    target_embed_std: Tensor,
    probe_nll: Tensor,
    probe_mae: Tensor,
    probe_mse: Tensor,
    pred_std: Tensor,
    target_std: Tensor,
    probe_bias: Tensor,
    pred_abs: Tensor,
    target_abs: Tensor,
    next_lat: Tensor,
    probe_terminal_mse: Tensor,
    zero_mse: Tensor,
    probe_explained_variance: Tensor,
}

struct RunningLoss {
    total_sum: Tensor,
    jepa_mse_sum: Tensor,
    sigreg_sum: Tensor,
    repr_std_mean_sum: Tensor,
    repr_std_min_sum: Tensor,
    pred_embed_std_sum: Tensor,
    target_embed_std_sum: Tensor,
    probe_nll_sum: Tensor,
    probe_mae_sum: Tensor,
    probe_mse_sum: Tensor,
    pred_std_sum: Tensor,
    target_std_sum: Tensor,
    probe_bias_sum: Tensor,
    pred_abs_sum: Tensor,
    target_abs_sum: Tensor,
    next_lat_sum: Tensor,
    probe_terminal_mse_sum: Tensor,
    zero_mse_sum: Tensor,
    samples: usize,
    batches: usize,
}

impl RunningLoss {
    fn new(device: Device) -> Self {
        Self {
            total_sum: Tensor::zeros([], (Kind::Float, device)),
            jepa_mse_sum: Tensor::zeros([], (Kind::Float, device)),
            sigreg_sum: Tensor::zeros([], (Kind::Float, device)),
            repr_std_mean_sum: Tensor::zeros([], (Kind::Float, device)),
            repr_std_min_sum: Tensor::zeros([], (Kind::Float, device)),
            pred_embed_std_sum: Tensor::zeros([], (Kind::Float, device)),
            target_embed_std_sum: Tensor::zeros([], (Kind::Float, device)),
            probe_nll_sum: Tensor::zeros([], (Kind::Float, device)),
            probe_mae_sum: Tensor::zeros([], (Kind::Float, device)),
            probe_mse_sum: Tensor::zeros([], (Kind::Float, device)),
            pred_std_sum: Tensor::zeros([], (Kind::Float, device)),
            target_std_sum: Tensor::zeros([], (Kind::Float, device)),
            probe_bias_sum: Tensor::zeros([], (Kind::Float, device)),
            pred_abs_sum: Tensor::zeros([], (Kind::Float, device)),
            target_abs_sum: Tensor::zeros([], (Kind::Float, device)),
            next_lat_sum: Tensor::zeros([], (Kind::Float, device)),
            probe_terminal_mse_sum: Tensor::zeros([], (Kind::Float, device)),
            zero_mse_sum: Tensor::zeros([], (Kind::Float, device)),
            samples: 0,
            batches: 0,
        }
    }

    fn add(&mut self, losses: &PretrainLoss, samples: usize) {
        tch::no_grad(|| {
            let weight = samples as f64;
            self.total_sum += losses.total.detach() * weight;
            self.jepa_mse_sum += losses.jepa_mse.detach() * weight;
            self.sigreg_sum += losses.sigreg.detach() * weight;
            self.repr_std_mean_sum += losses.repr_std_mean.detach() * weight;
            self.repr_std_min_sum += losses.repr_std_min.detach() * weight;
            self.pred_embed_std_sum += losses.pred_embed_std.detach() * weight;
            self.target_embed_std_sum += losses.target_embed_std.detach() * weight;
            self.probe_nll_sum += losses.probe_nll.detach() * weight;
            self.probe_mae_sum += losses.probe_mae.detach() * weight;
            self.probe_mse_sum += losses.probe_mse.detach() * weight;
            self.pred_std_sum += losses.pred_std.detach() * weight;
            self.target_std_sum += losses.target_std.detach() * weight;
            self.probe_bias_sum += losses.probe_bias.detach() * weight;
            self.pred_abs_sum += losses.pred_abs.detach() * weight;
            self.target_abs_sum += losses.target_abs.detach() * weight;
            self.next_lat_sum += losses.next_lat.detach() * weight;
            self.probe_terminal_mse_sum += losses.probe_terminal_mse.detach() * weight;
            self.zero_mse_sum += losses.zero_mse.detach() * weight;
            self.samples += samples;
            self.batches += 1;
        });
    }

    fn finish(self) -> TrainEpochLoss {
        assert!(self.samples > 0, "train epoch is empty");
        let denom = self.samples as f64;
        let probe_mse = self.probe_mse_sum.double_value(&[]) / denom;
        let zero_mse = self.zero_mse_sum.double_value(&[]) / denom;
        TrainEpochLoss {
            total: self.total_sum.double_value(&[]) / denom,
            jepa_mse: self.jepa_mse_sum.double_value(&[]) / denom,
            sigreg: self.sigreg_sum.double_value(&[]) / denom,
            repr_std_mean: self.repr_std_mean_sum.double_value(&[]) / denom,
            repr_std_min: self.repr_std_min_sum.double_value(&[]) / denom,
            pred_embed_std: self.pred_embed_std_sum.double_value(&[]) / denom,
            target_embed_std: self.target_embed_std_sum.double_value(&[]) / denom,
            probe_nll: self.probe_nll_sum.double_value(&[]) / denom,
            probe_mae: self.probe_mae_sum.double_value(&[]) / denom,
            probe_mse,
            pred_std: self.pred_std_sum.double_value(&[]) / denom,
            target_std: self.target_std_sum.double_value(&[]) / denom,
            probe_bias: self.probe_bias_sum.double_value(&[]) / denom,
            pred_abs: self.pred_abs_sum.double_value(&[]) / denom,
            target_abs: self.target_abs_sum.double_value(&[]) / denom,
            next_lat: self.next_lat_sum.double_value(&[]) / denom,
            probe_terminal_mse: self.probe_terminal_mse_sum.double_value(&[]) / denom,
            zero_mse,
            probe_explained_variance: explained_variance_value(probe_mse, zero_mse),
            samples: self.samples,
            batches: self.batches,
        }
    }
}

struct TrainEpochLoss {
    total: f64,
    jepa_mse: f64,
    sigreg: f64,
    repr_std_mean: f64,
    repr_std_min: f64,
    pred_embed_std: f64,
    target_embed_std: f64,
    probe_nll: f64,
    probe_mae: f64,
    probe_mse: f64,
    pred_std: f64,
    target_std: f64,
    probe_bias: f64,
    pred_abs: f64,
    target_abs: f64,
    next_lat: f64,
    probe_terminal_mse: f64,
    zero_mse: f64,
    probe_explained_variance: f64,
    samples: usize,
    batches: usize,
}

struct ProbeTrainSummary {
    probe_nll: f64,
    probe_mae: f64,
    probe_mse: f64,
    pred_std: f64,
    target_std: f64,
    pred_sigma: f64,
    pred_std_level: f64,
    target_std_level: f64,
    probe_ev_level: f64,
    pred_sigma_level: f64,
    grad_probe: f64,
    pnorm_probe: f64,
    samples: usize,
    batches: usize,
}

#[derive(Default)]
struct PretrainScalarHistory {
    train_mse: Vec<f32>,
    eval_mse: Vec<f32>,
    train_sigreg: Vec<f32>,
    eval_sigreg: Vec<f32>,
    train_jepa_mse: Vec<f32>,
    eval_jepa_mse: Vec<f32>,
    train_repr_std_mean: Vec<f32>,
    eval_repr_std_mean: Vec<f32>,
    train_repr_std_min: Vec<f32>,
    eval_repr_std_min: Vec<f32>,
    train_pred_embed_std: Vec<f32>,
    eval_pred_embed_std: Vec<f32>,
    train_target_embed_std: Vec<f32>,
    eval_target_embed_std: Vec<f32>,
    train_probe_nll: Vec<f32>,
    eval_probe_nll: Vec<f32>,
    train_probe_mae: Vec<f32>,
    eval_probe_mae: Vec<f32>,
    train_probe_explained_variance: Vec<f32>,
    eval_probe_explained_variance: Vec<f32>,
    train_pred_std: Vec<f32>,
    eval_pred_std: Vec<f32>,
    train_target_std: Vec<f32>,
    eval_target_std: Vec<f32>,
    train_probe_terminal_mse: Vec<f32>,
    eval_probe_terminal_mse: Vec<f32>,
}

impl PretrainScalarHistory {
    fn push(&mut self, train: &TrainEpochLoss, val: &ValidationLoss) {
        self.train_mse.push(train.probe_mse as f32);
        self.eval_mse.push(val.probe_mse as f32);
        self.train_sigreg.push(train.sigreg as f32);
        self.eval_sigreg.push(val.sigreg as f32);
        self.train_jepa_mse.push(train.jepa_mse as f32);
        self.eval_jepa_mse.push(val.jepa_mse as f32);
        self.train_repr_std_mean.push(train.repr_std_mean as f32);
        self.eval_repr_std_mean.push(val.repr_std_mean as f32);
        self.train_repr_std_min.push(train.repr_std_min as f32);
        self.eval_repr_std_min.push(val.repr_std_min as f32);
        self.train_pred_embed_std.push(train.pred_embed_std as f32);
        self.eval_pred_embed_std.push(val.pred_embed_std as f32);
        self.train_target_embed_std
            .push(train.target_embed_std as f32);
        self.eval_target_embed_std.push(val.target_embed_std as f32);
        self.train_probe_nll.push(train.probe_nll as f32);
        self.eval_probe_nll.push(val.probe_nll as f32);
        self.train_probe_mae.push(train.probe_mae as f32);
        self.eval_probe_mae.push(val.probe_mae as f32);
        self.train_probe_explained_variance
            .push(train.probe_explained_variance as f32);
        self.eval_probe_explained_variance
            .push(val.probe_explained_variance as f32);
        self.train_pred_std.push(train.pred_std as f32);
        self.eval_pred_std.push(val.pred_std as f32);
        self.train_target_std.push(train.target_std as f32);
        self.eval_target_std.push(val.target_std as f32);
        self.train_probe_terminal_mse
            .push(train.probe_terminal_mse as f32);
        self.eval_probe_terminal_mse
            .push(val.probe_terminal_mse as f32);
    }
}

fn write_pretrain_scalar_meta_reports(
    gens_dir: &Path,
    epoch: usize,
    global_step: usize,
    history: &PretrainScalarHistory,
) -> Result<()> {
    let epoch_dir = gens_dir.join(epoch.to_string());
    fs::create_dir_all(&epoch_dir)?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_probe_mse.report.bin"),
        format!("Pretrain Probe MSE - epoch {epoch} step {global_step}"),
        "target-scaled prediction MSE",
        &history.train_mse,
        &history.eval_mse,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_sigreg.report.bin"),
        format!("Pretrain SIGReg Loss - epoch {epoch} step {global_step}"),
        "SIGReg loss",
        &history.train_sigreg,
        &history.eval_sigreg,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_jepa_mse.report.bin"),
        format!("Pretrain JEPA MSE - epoch {epoch} step {global_step}"),
        "embedding MSE",
        &history.train_jepa_mse,
        &history.eval_jepa_mse,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_repr_std_mean.report.bin"),
        format!("Pretrain Repr Std Mean - epoch {epoch} step {global_step}"),
        "mean feature std",
        &history.train_repr_std_mean,
        &history.eval_repr_std_mean,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_repr_std_min.report.bin"),
        format!("Pretrain Repr Std Min - epoch {epoch} step {global_step}"),
        "minimum feature std",
        &history.train_repr_std_min,
        &history.eval_repr_std_min,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_pred_embed_std.report.bin"),
        format!("Pretrain Pred Embed Std - epoch {epoch} step {global_step}"),
        "predicted embedding std",
        &history.train_pred_embed_std,
        &history.eval_pred_embed_std,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_target_embed_std.report.bin"),
        format!("Pretrain Target Embed Std - epoch {epoch} step {global_step}"),
        "target embedding std",
        &history.train_target_embed_std,
        &history.eval_target_embed_std,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_probe_mae.report.bin"),
        format!("Pretrain Probe MAE - epoch {epoch} step {global_step}"),
        "target-scaled prediction MAE",
        &history.train_probe_mae,
        &history.eval_probe_mae,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_probe_explained_variance.report.bin"),
        format!("Pretrain Probe Explained Variance - epoch {epoch} step {global_step}"),
        "1 - probe MSE / zero baseline MSE",
        &history.train_probe_explained_variance,
        &history.eval_probe_explained_variance,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_pred_std.report.bin"),
        format!("Pretrain Probe Pred Std - epoch {epoch} step {global_step}"),
        "probe prediction std",
        &history.train_pred_std,
        &history.eval_pred_std,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_target_std.report.bin"),
        format!("Pretrain Probe Target Std - epoch {epoch} step {global_step}"),
        "probe target std",
        &history.train_target_std,
        &history.eval_target_std,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_probe_terminal_mse.report.bin"),
        format!("Pretrain Probe Terminal MSE - epoch {epoch} step {global_step}"),
        "last predicted bar MSE",
        &history.train_probe_terminal_mse,
        &history.eval_probe_terminal_mse,
    )
}

fn write_pretrain_scalar_report(
    path: &Path,
    title: String,
    y_label: &str,
    train: &[f32],
    eval: &[f32],
) -> Result<()> {
    write_report_file(
        path,
        &Report {
            title,
            x_label: Some("epoch".to_string()),
            y_label: Some(y_label.to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "train".to_string(),
                        values: train.to_vec(),
                    },
                    ReportSeries {
                        label: "eval".to_string(),
                        values: eval.to_vec(),
                    },
                ],
            },
        },
    )
}

fn train_detached_probe(
    model: &TradingModel,
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    batch_size: usize,
    probe_epochs: usize,
    target_scale: f64,
    probe_opt: &mut Muon,
    probe_named_vars: &[(String, Tensor)],
    device: Device,
) -> ProbeTrainSummary {
    let _ = model;
    let mut nll_sum = 0.0;
    let mut mse_sum = 0.0;
    let mut mae_sum = 0.0;
    let mut pred_std_sum = 0.0;
    let mut target_std_sum = 0.0;
    let mut pred_sigma_sum = 0.0;
    let mut pred_std_level_sum = 0.0;
    let mut target_std_level_sum = 0.0;
    let mut probe_ev_level_sum = 0.0;
    let mut pred_sigma_level_sum = 0.0;
    let mut grad_probe_sum = 0.0;
    let mut pnorm_probe_sum = 0.0;
    let mut samples = 0usize;
    let mut batches = 0usize;

    for probe_epoch in 0..probe_epochs {
        sampler.start_epoch();
        while let Some(batch) = sampler.next_train_batch(batch_size) {
            let batch_samples = batch.len() as usize;
            // The probe is trained on BOTH real encoded tokens AND detached flow
            // x-predictions (sampled per position, masked to signal>=32 so it only
            // decodes reasonably-clean latents), matching the rollout distribution.
            // All encoder/AR/flow inputs are detached so the probe never shapes them.
            let (token, pred_emb, pred_target, pred_mask) = tch::no_grad(|| {
                autocast(false, || {
                    let real_token = heads.encode_bar_tokens(&batch.next_bars, false).narrow(2, 0, 1);
                    let full =
                        Tensor::cat(&[&batch.bar_history, &batch.next_bars.narrow(2, 0, 1)], 2);
                    let all_tokens = heads.encode_bar_tokens(&full, false);
                    let length = batch.bar_history.size()[2];
                    let bar_tokens = all_tokens.narrow(2, 0, length);
                    let latent_dim = heads.latent_dim;
                    let noisy = &bar_tokens * (1.0 - LEJEPA_CTX_NOISE_MIX)
                        + Tensor::randn_like(&bar_tokens) * LEJEPA_CTX_NOISE_MIX;
                    let belief = heads.predict_lejepa_bar_predictions(&noisy, false).belief;
                    let clean_next = all_tokens.narrow(2, 1, length);
                    let bt = belief.size();
                    let (b, tk) = (bt[0], bt[1]);
                    let rows = b * tk * length;
                    let signal = Tensor::randint(LEJEPA_K_MAX, [b, tk, length], (Kind::Int64, device));
                    let t = (signal.to_kind(Kind::Float) / LEJEPA_K_MAX as f64).unsqueeze(-1);
                    let noise = Tensor::randn_like(&clean_next);
                    let noised = &noise * (&t * -1.0 + 1.0) + &clean_next * &t;
                    let x_pred = heads
                        .lejepa_flow_predict(
                            &noised.reshape([rows, latent_dim]),
                            &signal.reshape([rows]),
                            &Tensor::zeros([rows], (Kind::Int64, device)),
                            &belief.reshape([rows, latent_dim]),
                        )
                        .reshape([b, tk, length, latent_dim]);
                    let mask = signal.ge(LEJEPA_K_MAX / 2).to_kind(Kind::Float).unsqueeze(-1);
                    let pred_target = full.narrow(2, 1, length) * target_scale;
                    (real_token, x_pred, pred_target, mask)
                })
            });
            let ohlc_target = scaled_next_ohlc_features(&batch.next_bars, target_scale);
            let probe = ohlc_probe_metrics(heads, &token.detach(), &ohlc_target);
            let probe_pred_nll = masked_probe_nll(heads, &pred_emb.detach(), &pred_target, &pred_mask);
            let probe_loss = &probe.probe_nll + &probe_pred_nll;
            assert_finite_loss(&probe_loss, probe_epoch + 1);
            probe_opt.zero_grad();
            probe_loss.backward();
            let (grad_probe, pnorm_probe) = named_grad_param_l2(probe_named_vars, device);
            grad_probe_sum += grad_probe;
            pnorm_probe_sum += pnorm_probe;
            clip_all_grads(probe_named_vars, MAX_GRAD_NORM, device);
            probe_opt.step();

            nll_sum += probe.probe_nll.double_value(&[]) * batch_samples as f64;
            mse_sum += probe.probe_mse.double_value(&[]) * batch_samples as f64;
            mae_sum += probe.probe_mae.double_value(&[]) * batch_samples as f64;
            pred_std_sum += probe.pred_std.double_value(&[]) * batch_samples as f64;
            target_std_sum += probe.target_std.double_value(&[]) * batch_samples as f64;
            pred_sigma_sum += probe.pred_sigma.double_value(&[]) * batch_samples as f64;
            pred_std_level_sum += probe.pred_std_level.double_value(&[]) * batch_samples as f64;
            target_std_level_sum += probe.target_std_level.double_value(&[]) * batch_samples as f64;
            probe_ev_level_sum += probe.probe_ev_level.double_value(&[]) * batch_samples as f64;
            pred_sigma_level_sum += probe.pred_sigma_level.double_value(&[]) * batch_samples as f64;
            samples += batch_samples;
            batches += 1;
        }
    }

    assert!(samples > 0, "detached probe training set is empty");
    let denom = samples as f64;
    ProbeTrainSummary {
        probe_nll: nll_sum / denom,
        probe_mae: mae_sum / denom,
        probe_mse: mse_sum / denom,
        pred_std: pred_std_sum / denom,
        target_std: target_std_sum / denom,
        pred_sigma: pred_sigma_sum / denom,
        pred_std_level: pred_std_level_sum / denom,
        target_std_level: target_std_level_sum / denom,
        probe_ev_level: probe_ev_level_sum / denom,
        pred_sigma_level: pred_sigma_level_sum / denom,
        grad_probe: grad_probe_sum / batches.max(1) as f64,
        pnorm_probe: pnorm_probe_sum / batches.max(1) as f64,
        samples,
        batches,
    }
}

fn validation_batch_cap(validation_batches: usize) -> Option<usize> {
    (validation_batches > 0).then_some(validation_batches)
}

fn validate_full(
    model: &TradingModel,
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    batch_size: usize,
    max_batches: Option<usize>,
    objective: PretrainObjective,
    lambda_lat: f64,
    lambda_sigreg: f64,
    device: Device,
) -> ValidationLoss {
    tch::no_grad(|| {
        let mut total_sum = 0.0;
        let mut jepa_mse_sum = 0.0;
        let mut sigreg_sum = 0.0;
        let mut repr_std_mean_sum = 0.0;
        let mut repr_std_min_sum = 0.0;
        let mut pred_embed_std_sum = 0.0;
        let mut target_embed_std_sum = 0.0;
        let mut probe_nll_sum = 0.0;
        let mut probe_mae_sum = 0.0;
        let mut probe_mse_sum = 0.0;
        let mut pred_std_sum = 0.0;
        let mut target_std_sum = 0.0;
        let mut probe_bias_sum = 0.0;
        let mut pred_abs_sum = 0.0;
        let mut target_abs_sum = 0.0;
        let mut next_lat_sum = 0.0;
        let mut probe_terminal_mse_sum = 0.0;
        let mut zero_mse_sum = 0.0;
        // Belief-ablation skill test (read-only): does the flow head actually USE the
        // AR belief? Accumulated per Lejepa validation batch, meaned below.
        let mut skill_ev_correct_sum = 0.0;
        let mut skill_ev_shuffled_sum = 0.0;
        let mut skill_ev_zero_sum = 0.0;
        let mut skill_belief_spread_sum = 0.0;
        let mut skill_belief_norm_sum = 0.0;
        let mut skill_batches = 0usize;
        let mut samples = 0usize;
        let mut tickers = 0usize;
        let mut batches = 0usize;
        let mut rollout_ctx: Vec<Tensor> = Vec::new();
        let mut rollout_actual: Vec<Tensor> = Vec::new();
        let mut rollout_windows = 0usize;

        let k_patches = sampler.k_patches;
        let patch_size = sampler.patch_size;
        let target_scale = sampler.target_scale;
        for env in sampler.train_envs.iter_mut() {
            if max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }

            let offsets = build_split_offsets(
                env.price_deltas[0].len(),
                k_patches,
                patch_size,
                SplitKind::Validation,
            );
            if offsets.is_empty() {
                continue;
            }
            tickers += 1;

            for chunk in offsets.chunks(batch_size) {
                if max_batches.is_some_and(|limit| batches >= limit) {
                    break;
                }

                let batch = PretrainSampler::batch_from_offsets(
                    env,
                    chunk,
                    k_patches,
                    patch_size,
                    target_scale,
                    device,
                );
                let batch_samples = batch.len() as usize;
                let losses = pretrain_loss(
                    model,
                    heads,
                    &batch,
                    objective,
                    lambda_lat,
                    lambda_sigreg,
                    target_scale,
                    false,
                );
                let return_target = match objective {
                    PretrainObjective::MeanMse => cumulative_future_returns(&batch.future_patches),
                    PretrainObjective::Lejepa => {
                        scaled_next_ohlc_features(&batch.next_bars, target_scale)
                    }
                };
                let zero_mse_loss = return_target.pow_tensor_scalar(2.0).mean(Kind::Float);
                total_sum += losses.total.double_value(&[]) * batch_samples as f64;
                jepa_mse_sum += losses.jepa_mse.double_value(&[]) * batch_samples as f64;
                sigreg_sum += losses.sigreg.double_value(&[]) * batch_samples as f64;
                repr_std_mean_sum += losses.repr_std_mean.double_value(&[]) * batch_samples as f64;
                repr_std_min_sum += losses.repr_std_min.double_value(&[]) * batch_samples as f64;
                pred_embed_std_sum +=
                    losses.pred_embed_std.double_value(&[]) * batch_samples as f64;
                target_embed_std_sum +=
                    losses.target_embed_std.double_value(&[]) * batch_samples as f64;
                probe_nll_sum += losses.probe_nll.double_value(&[]) * batch_samples as f64;
                probe_mae_sum += losses.probe_mae.double_value(&[]) * batch_samples as f64;
                probe_mse_sum += losses.probe_mse.double_value(&[]) * batch_samples as f64;
                pred_std_sum += losses.pred_std.double_value(&[]) * batch_samples as f64;
                target_std_sum += losses.target_std.double_value(&[]) * batch_samples as f64;
                probe_bias_sum += losses.probe_bias.double_value(&[]) * batch_samples as f64;
                pred_abs_sum += losses.pred_abs.double_value(&[]) * batch_samples as f64;
                target_abs_sum += losses.target_abs.double_value(&[]) * batch_samples as f64;
                next_lat_sum += losses.next_lat.double_value(&[]) * batch_samples as f64;
                probe_terminal_mse_sum +=
                    losses.probe_terminal_mse.double_value(&[]) * batch_samples as f64;
                zero_mse_sum += zero_mse_loss.double_value(&[]) * batch_samples as f64;
                samples += batch_samples;
                batches += 1;

                if matches!(objective, PretrainObjective::Lejepa)
                    && rollout_windows < LEJEPA_ROLLOUT_EVAL_WINDOWS
                {
                    let take = batch_samples.min(LEJEPA_ROLLOUT_EVAL_WINDOWS - rollout_windows);
                    rollout_ctx.push(batch.bar_history.narrow(0, 0, take as i64));
                    rollout_actual.push(batch.next_bars.narrow(0, 0, take as i64));
                    rollout_windows += take;
                }

                // Belief-ablation skill test: how much predictive info does the AR
                // belief carry through the flow head? ev = 1 - mse/var (centered
                // marginal variance) of the one-shot flow estimate. Comparing
                // belief vs row-shuffled vs zero context isolates the conditioning.
                if matches!(objective, PretrainObjective::Lejepa) {
                    let latent_dim = heads.latent_dim;
                    let full =
                        Tensor::cat(&[&batch.bar_history, &batch.next_bars.narrow(2, 0, 1)], 2);
                    let all_tokens = autocast(false, || heads.encode_bar_tokens(&full, false));
                    let length = batch.bar_history.size()[2];
                    let target = all_tokens.narrow(2, 1, length);
                    let belief = heads
                        .predict_lejepa_bar_predictions(&all_tokens.narrow(2, 0, length), false)
                        .belief;
                    let rows = target.numel() as i64 / latent_dim;
                    let z1 = target.reshape([rows, latent_dim]);
                    let b = belief.reshape([rows, latent_dim]);
                    let z1_mean = z1.mean_dim([0i64].as_slice(), true, Kind::Float);
                    let var = (&z1 - &z1_mean)
                        .square()
                        .mean(Kind::Float)
                        .double_value(&[]);
                    // One-shot flow estimate (x_pred from the zero prior at signal 0)
                    // as the deterministic next-token prediction for the ablation.
                    let signal0 = Tensor::zeros([rows], (Kind::Int64, device));
                    let step0 = Tensor::zeros([rows], (Kind::Int64, device));
                    let ev = |ctx: &Tensor| -> f64 {
                        let est = heads.lejepa_flow_predict(
                            &Tensor::zeros_like(&z1),
                            &signal0,
                            &step0,
                            ctx,
                        );
                        let mse = est.mse_loss(&z1, Reduction::Mean).double_value(&[]);
                        1.0 - mse / var
                    };
                    let perm = Tensor::randperm(rows, (Kind::Int64, device));
                    skill_ev_correct_sum += ev(&b);
                    skill_ev_shuffled_sum += ev(&b.index_select(0, &perm));
                    skill_ev_zero_sum += ev(&Tensor::zeros_like(&b));
                    let b_mean = b.mean_dim([0i64].as_slice(), true, Kind::Float);
                    skill_belief_spread_sum += (&b - &b_mean)
                        .square()
                        .mean_dim([0i64].as_slice(), false, Kind::Float)
                        .sqrt()
                        .mean(Kind::Float)
                        .double_value(&[]);
                    skill_belief_norm_sum += b
                        .norm_scalaropt_dim(2, [-1i64].as_slice(), false)
                        .mean(Kind::Float)
                        .double_value(&[]);
                    skill_batches += 1;
                }
            }
        }

        // Mean skill metrics; NaN when no rollout-capable (Lejepa) batches ran.
        let (
            skill_ev_correct,
            skill_ev_shuffled,
            skill_ev_zero,
            skill_belief_spread,
            skill_belief_norm,
        ) = if skill_batches > 0 {
            let n = skill_batches as f64;
            (
                skill_ev_correct_sum / n,
                skill_ev_shuffled_sum / n,
                skill_ev_zero_sum / n,
                skill_belief_spread_sum / n,
                skill_belief_norm_sum / n,
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        };

        // Stochastic imagined-rollout MSE against the actual future in raw OHLC
        // space. The MEAN path is the Monte-Carlo average of K sampled rollouts;
        // the SAMPLED column is the expected per-window MSE over independent draws.
        // A paired t-test over windows reports whether averaging beats sampling.
        // Per-window MSE reduces over horizon x 16 features; chunked at `batch_size`.
        let (
            rollout_mean_mse,
            rollout_sampled_mse,
            rollout_mse_delta,
            rollout_mse_delta_se,
            rollout_mse_t,
            rollout_mse_n,
            rollout_mean_dclose,
            rollout_mean_dclose_std,
            rollout_sampled_dclose,
            rollout_sampled_dclose_std,
        ) = match objective {
            PretrainObjective::Lejepa if !rollout_ctx.is_empty() => {
                let ctx = Tensor::cat(&rollout_ctx, 0);
                let actual = Tensor::cat(&rollout_actual, 0).view([
                    -1,
                    LEJEPA_ROLLOUT_BARS,
                    LEJEPA_BAR_FEATURES,
                ]);
                let n_total = ctx.size()[0];
                let chunk = batch_size as i64;
                let mut mean_mse: Vec<f64> = Vec::with_capacity(rollout_windows);
                let mut sampled_mse: Vec<f64> = Vec::with_capacity(rollout_windows);
                // Decoded close-delta (feature row[3]) accumulators over all rollout
                // bars x windows, separately for the mean and sampled paths.
                let mut mean_dclose_sum = 0.0f64;
                let mut mean_dclose_sqsum = 0.0f64;
                let mut mean_dclose_n = 0i64;
                let mut sampled_dclose_sum = 0.0f64;
                let mut sampled_dclose_sqsum = 0.0f64;
                let mut sampled_dclose_n = 0i64;
                let mut start = 0;
                while start < n_total {
                    let len = chunk.min(n_total - start);
                    let ctx_c = ctx.narrow(0, start, len);
                    let actual_c = actual.narrow(0, start, len);
                    let mean_roll =
                        heads.lejepa_imagined_rollout(&ctx_c, 0.0, false) / target_scale;
                    let mean_dclose = mean_roll.narrow(2, 3, 1);
                    mean_dclose_sum += mean_dclose.sum(Kind::Float).double_value(&[]);
                    mean_dclose_sqsum += mean_dclose.square().sum(Kind::Float).double_value(&[]);
                    mean_dclose_n += mean_dclose.numel() as i64;
                    let mean_pw = (&mean_roll - &actual_c).pow_tensor_scalar(2.0).mean_dim(
                        [1i64, 2].as_slice(),
                        false,
                        Kind::Float,
                    );
                    let mut sampled_pw = Tensor::zeros([len], (Kind::Float, device));
                    for _ in 0..LEJEPA_ROLLOUT_EVAL_SAMPLES {
                        let sampled_roll =
                            heads.lejepa_imagined_rollout(&ctx_c, 1.0, false) / target_scale;
                        let sampled_dclose = sampled_roll.narrow(2, 3, 1);
                        sampled_dclose_sum += sampled_dclose.sum(Kind::Float).double_value(&[]);
                        sampled_dclose_sqsum +=
                            sampled_dclose.square().sum(Kind::Float).double_value(&[]);
                        sampled_dclose_n += sampled_dclose.numel() as i64;
                        sampled_pw += (&sampled_roll - &actual_c).pow_tensor_scalar(2.0).mean_dim(
                            [1i64, 2].as_slice(),
                            false,
                            Kind::Float,
                        );
                    }
                    let sampled_pw = sampled_pw / LEJEPA_ROLLOUT_EVAL_SAMPLES as f64;
                    mean_mse.extend(
                        tensor_to_vec_f32(&mean_pw)
                            .expect("rollout mean mse")
                            .into_iter()
                            .map(|x| x as f64),
                    );
                    sampled_mse.extend(
                        tensor_to_vec_f32(&sampled_pw)
                            .expect("rollout sampled mse")
                            .into_iter()
                            .map(|x| x as f64),
                    );
                    start += len;
                }
                let n = mean_mse.len();
                let mean_avg = mean_mse.iter().sum::<f64>() / n as f64;
                let sampled_avg = sampled_mse.iter().sum::<f64>() / n as f64;
                // Paired significance of (sampled - mean) MSE across windows.
                let diffs: Vec<f64> =
                    sampled_mse.iter().zip(&mean_mse).map(|(s, m)| s - m).collect();
                let delta = diffs.iter().sum::<f64>() / n as f64;
                let (se, t) = if n >= 2 {
                    let var = diffs.iter().map(|d| (d - delta).powi(2)).sum::<f64>()
                        / (n as f64 - 1.0);
                    let se = (var / n as f64).sqrt();
                    let t = if se > 0.0 { delta / se } else { 0.0 };
                    (se, t)
                } else {
                    (0.0, 0.0)
                };
                let mean_dclose_avg = mean_dclose_sum / mean_dclose_n as f64;
                let mean_dclose_std = (mean_dclose_sqsum / mean_dclose_n as f64
                    - mean_dclose_avg.powi(2))
                .max(0.0)
                .sqrt();
                let sampled_dclose_avg = sampled_dclose_sum / sampled_dclose_n as f64;
                let sampled_dclose_std = (sampled_dclose_sqsum / sampled_dclose_n as f64
                    - sampled_dclose_avg.powi(2))
                .max(0.0)
                .sqrt();
                (
                    mean_avg,
                    sampled_avg,
                    delta,
                    se,
                    t,
                    n as f64,
                    mean_dclose_avg,
                    mean_dclose_std,
                    sampled_dclose_avg,
                    sampled_dclose_std,
                )
            }
            // NaN/0 = not-applicable: MeanMse has no imagined rollout.
            _ => (
                f64::NAN,
                f64::NAN,
                0.0,
                0.0,
                0.0,
                0.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ),
        };

        assert!(samples > 0, "validation set is empty");
        let probe_mse = probe_mse_sum / samples as f64;
        let zero_mse = zero_mse_sum / samples as f64;
        ValidationLoss {
            total: total_sum / samples as f64,
            jepa_mse: jepa_mse_sum / samples as f64,
            sigreg: sigreg_sum / samples as f64,
            repr_std_mean: repr_std_mean_sum / samples as f64,
            repr_std_min: repr_std_min_sum / samples as f64,
            pred_embed_std: pred_embed_std_sum / samples as f64,
            target_embed_std: target_embed_std_sum / samples as f64,
            probe_nll: probe_nll_sum / samples as f64,
            probe_mae: probe_mae_sum / samples as f64,
            probe_mse,
            pred_std: pred_std_sum / samples as f64,
            target_std: target_std_sum / samples as f64,
            probe_bias: probe_bias_sum / samples as f64,
            pred_abs: pred_abs_sum / samples as f64,
            target_abs: target_abs_sum / samples as f64,
            next_lat: next_lat_sum / samples as f64,
            probe_terminal_mse: probe_terminal_mse_sum / samples as f64,
            zero_mse,
            probe_explained_variance: explained_variance_value(probe_mse, zero_mse),
            rollout_mean_mse,
            rollout_sampled_mse,
            rollout_mse_delta,
            rollout_mse_delta_se,
            rollout_mse_t,
            rollout_mse_n,
            rollout_mean_dclose,
            rollout_mean_dclose_std,
            rollout_sampled_dclose,
            rollout_sampled_dclose_std,
            skill_ev_correct,
            skill_ev_shuffled,
            skill_ev_zero,
            skill_belief_spread,
            skill_belief_norm,
            skill_batches,
            samples,
            tickers,
            batches,
        }
    })
}

struct DiagnosticTrace {
    label: String,
    loss: f64,
    actual: Vec<f32>,
    predicted: Vec<f32>,
}

struct RolloutEntropy {
    mean_step_std: f64,
    tok_norm_mean: f64,
    tok_norm_max: f64,
}

struct VariantCandles {
    label: String,
    actual: Vec<CandleBar>,
    mean: Vec<CandleBar>,
    sampled: Vec<CandleBar>,
}

fn write_pretrain_diagnostics(
    model: &TradingModel,
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    batch_size: usize,
    max_batches: Option<usize>,
    objective: PretrainObjective,
    epoch: usize,
    global_step: usize,
    gens_dir: &Path,
    device: Device,
) -> Result<()> {
    const TRACE_COUNT: usize = 8;
    const WORST_COUNT: usize = 8;

    let epoch_dir = gens_dir.join(epoch.to_string());
    let samples_dir = epoch_dir.join("samples");
    fs::create_dir_all(&samples_dir)?;

    let horizon = match objective {
        PretrainObjective::MeanMse => sampler.k_patches * sampler.patch_size,
        PretrainObjective::Lejepa => LEJEPA_ROLLOUT_BARS as usize,
    };
    let mut abs_sum = vec![0.0f64; horizon];
    let mut sq_sum = vec![0.0f64; horizon];
    let mut bias_sum = vec![0.0f64; horizon];
    let mut count = 0usize;
    let mut first_traces = Vec::new();
    let mut worst_traces: Vec<DiagnosticTrace> = Vec::new();
    let mut variant_traces: Vec<VariantCandles> = Vec::new();
    let mut ent_mstep = 0.0f64;
    let mut ent_tnmean = 0.0f64;
    let mut ent_tnmax = 0.0f64;
    let mut ent_n = 0usize;

    let k_patches = sampler.k_patches;
    let patch_size = sampler.patch_size;
    let target_scale = sampler.target_scale;
    tch::no_grad(|| -> Result<()> {
        let mut batches = 0usize;
        for (ticker, env) in sampler
            .train_tickers
            .iter()
            .zip(sampler.train_envs.iter_mut())
        {
            if max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }
            let offsets = build_split_offsets(
                env.price_deltas[0].len(),
                k_patches,
                patch_size,
                SplitKind::Validation,
            );
            if offsets.is_empty() {
                continue;
            }

            for chunk in offsets.chunks(batch_size) {
                if max_batches.is_some_and(|limit| batches >= limit) {
                    break;
                }
                let batch = PretrainSampler::batch_from_offsets(
                    env,
                    chunk,
                    k_patches,
                    patch_size,
                    target_scale,
                    device,
                );
                let mut mean_ohlc: Option<Vec<f32>> = None;
                let (predicted, actual, predicted_ohlc, actual_ohlc) = match objective {
                    PretrainObjective::MeanMse => {
                        let pred = predict_future_returns(model, heads, &batch);
                        let actual = cumulative_future_returns(&batch.future_patches);
                        (
                            tensor_to_vec_f32(&pred)?,
                            tensor_to_vec_f32(&actual)?,
                            None,
                            None,
                        )
                    }
                    PretrainObjective::Lejepa => {
                        // One sampled trajectory for the candle diagnostics + rollout
                        // step-variance instrumentation.
                        let (imagined, entropy) = heads.single_imagined_rollout(
                            &batch.bar_history,
                            1.0,
                            false,
                            true,
                        );
                        if let Some(e) = entropy {
                            ent_mstep += e.mean_step_std;
                            ent_tnmean += e.tok_norm_mean;
                            ent_tnmax = ent_tnmax.max(e.tok_norm_max);
                            ent_n += 1;
                        }
                        let predicted_ohlc = imagined / target_scale;
                        mean_ohlc = Some(tensor_to_vec_f32(&predicted_ohlc)?);
                        (
                            Vec::new(),
                            Vec::new(),
                            Some(tensor_to_vec_f32(&predicted_ohlc)?),
                            Some(tensor_to_vec_f32(&batch.next_bars)?),
                        )
                    }
                };

                for (sample_idx, &offset) in chunk.iter().enumerate() {
                    let mut sample_abs = 0.0;
                    let (actual_sample, pred_sample) = match objective {
                        PretrainObjective::MeanMse => {
                            let start = sample_idx * horizon;
                            let end = start + horizon;
                            let actual_sample = actual[start..end].to_vec();
                            let pred_sample = predicted[start..end].to_vec();
                            for h in 0..horizon {
                                let err = pred_sample[h] as f64 - actual_sample[h] as f64;
                                abs_sum[h] += err.abs();
                                sq_sum[h] += err * err;
                                bias_sum[h] += err;
                                sample_abs += err.abs();
                            }
                            (actual_sample, pred_sample)
                        }
                        PretrainObjective::Lejepa => {
                            let feature_start = sample_idx * horizon * OHLC_BAR_FEATURES;
                            let feature_end = feature_start + horizon * OHLC_BAR_FEATURES;
                            let actual_features =
                                &actual_ohlc.as_ref().expect("LEJEPA actual OHLC missing")
                                    [feature_start..feature_end];
                            let predicted_features = &predicted_ohlc
                                .as_ref()
                                .expect("LEJEPA predicted OHLC missing")
                                [feature_start..feature_end];
                            for h in 0..horizon {
                                let start = h * OHLC_BAR_FEATURES;
                                let end = start + OHLC_BAR_FEATURES;
                                let mut bar_abs = 0.0;
                                let mut bar_sq = 0.0;
                                let mut bar_bias = 0.0;
                                for (&pred, &actual) in predicted_features[start..end]
                                    .iter()
                                    .zip(actual_features[start..end].iter())
                                {
                                    let err = pred as f64 - actual as f64;
                                    bar_abs += err.abs();
                                    bar_sq += err * err;
                                    bar_bias += err;
                                }
                                let denom = OHLC_BAR_FEATURES as f64;
                                abs_sum[h] += bar_abs / denom;
                                sq_sum[h] += bar_sq / denom;
                                bias_sum[h] += bar_bias / denom;
                                sample_abs += bar_abs / denom;
                            }
                            // Seed both chains from the TRUE last context bar
                            // (index `offset`), whose sanitized OHLC are the
                            // denominators the windowed rows (bars `offset+1..`)
                            // were built against. Reconstruct its proportions
                            // from its own intra-bar channels so the
                            // telescoping is exact; actual and predicted share
                            // this real seed so their candles stay comparable.
                            let seed_idx = env.ticker_perm[0];
                            let seed =
                                seed_candle_from_feature_row(&env.ohlc_features[seed_idx][offset]);
                            if variant_traces.len() < TRACE_COUNT {
                                let mean_features =
                                    &mean_ohlc.as_ref().expect("LEJEPA mean OHLC missing")
                                        [feature_start..feature_end];
                                variant_traces.push(VariantCandles {
                                    label: format!("sample_{:02}", variant_traces.len() + 1),
                                    actual: chained_candles_from_ohlc_features(
                                        actual_features,
                                        &seed,
                                    ),
                                    mean: chained_candles_from_ohlc_features(mean_features, &seed),
                                    sampled: chained_candles_from_ohlc_features(
                                        predicted_features,
                                        &seed,
                                    ),
                                });
                            }
                            (Vec::new(), Vec::new())
                        }
                    };
                    count += 1;
                    let loss = sample_abs / horizon as f64;
                    let trace = DiagnosticTrace {
                        label: format!("{}_offset_{}", ticker, offset),
                        loss,
                        actual: actual_sample,
                        predicted: pred_sample,
                    };

                    if first_traces.len() < TRACE_COUNT {
                        first_traces.push(DiagnosticTrace {
                            label: format!("sample_{:02}_{}", first_traces.len() + 1, trace.label),
                            loss,
                            actual: trace.actual.clone(),
                            predicted: trace.predicted.clone(),
                        });
                    }

                    worst_traces.push(trace);
                    worst_traces.sort_by(|a, b| {
                        b.loss
                            .partial_cmp(&a.loss)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    worst_traces.truncate(WORST_COUNT);
                }
                batches += 1;
            }
        }
        Ok(())
    })?;

    assert!(count > 0, "pretrain diagnostics validation set is empty");
    let denom = count as f64;
    let mae = abs_sum
        .iter()
        .map(|v| (*v / denom) as f32)
        .collect::<Vec<_>>();
    let rmse = sq_sum
        .iter()
        .map(|v| (*v / denom).sqrt() as f32)
        .collect::<Vec<_>>();
    let bias = bias_sum
        .iter()
        .map(|v| (*v / denom) as f32)
        .collect::<Vec<_>>();

    write_report_file(
        &epoch_dir.join("pretrain_horizon_error.report.bin"),
        &Report {
            title: format!("Pretrain Horizon Error - epoch {epoch} step {global_step}"),
            x_label: Some("forecast step".to_string()),
            y_label: Some(match objective {
                PretrainObjective::MeanMse => "target-scaled cumulative log return".to_string(),
                PretrainObjective::Lejepa => "OHLC feature error".to_string(),
            }),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "MAE".to_string(),
                        values: mae.clone(),
                    },
                    ReportSeries {
                        label: "RMSE".to_string(),
                        values: rmse.clone(),
                    },
                    ReportSeries {
                        label: "Bias".to_string(),
                        values: bias,
                    },
                ],
            },
        },
    )?;
    for (i, trace) in first_traces.iter().enumerate() {
        write_trace_reports(
            &samples_dir,
            &format!("sample_{:02}", i + 1),
            "Sample",
            epoch,
            global_step,
            trace,
        )?;
    }
    for (i, trace) in worst_traces.iter().enumerate() {
        write_trace_reports(
            &samples_dir,
            &format!("worst_{:02}", i + 1),
            "Worst",
            epoch,
            global_step,
            trace,
        )?;
    }
    for vt in &variant_traces {
        write_variant_candle_report(
            &samples_dir,
            &vt.label,
            "mean",
            epoch,
            global_step,
            &vt.actual,
            &vt.mean,
        )?;
        write_variant_candle_report(
            &samples_dir,
            &vt.label,
            "sampled",
            epoch,
            global_step,
            &vt.actual,
            &vt.sampled,
        )?;
    }
    if ent_n > 0 {
        let n = ent_n as f64;
        println!(
            "lejepa rollout: mean_step_std={:.5} tok_norm_mean={:.4} tok_norm_max={:.4}",
            ent_mstep / n,
            ent_tnmean / n,
            ent_tnmax,
        );
    }

    Ok(())
}

fn write_variant_candle_report(
    dir: &Path,
    prefix: &str,
    variant: &str,
    epoch: usize,
    global_step: usize,
    actual: &[CandleBar],
    predicted: &[CandleBar],
) -> Result<()> {
    write_report_file(
        &dir.join(format!("{prefix}_{variant}_candles.report.bin")),
        &Report {
            title: format!(
                "Pretrain Rollout {variant} Candles - epoch {epoch} step {global_step} - {prefix}"
            ),
            x_label: Some("forecast bar".to_string()),
            y_label: Some("relative price".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::CandleCompare {
                actual: actual.to_vec(),
                predicted: predicted.to_vec(),
            },
        },
    )
}

fn write_trace_reports(
    dir: &Path,
    prefix: &str,
    group: &str,
    epoch: usize,
    global_step: usize,
    trace: &DiagnosticTrace,
) -> Result<()> {
    if !trace.actual.is_empty() && !trace.predicted.is_empty() {
        let error = trace
            .predicted
            .iter()
            .zip(trace.actual.iter())
            .map(|(pred, actual)| pred - actual)
            .collect::<Vec<_>>();
        write_report_file(
            &dir.join(format!("{prefix}_deltas.report.bin")),
            &Report {
                title: format!(
                    "Pretrain {group} Returns - epoch {epoch} step {global_step} - {} - MAE {:.5}",
                    trace.label, trace.loss
                ),
                x_label: Some("forecast step".to_string()),
                y_label: Some("target-scaled cumulative log return".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "actual".to_string(),
                            values: trace.actual.clone(),
                        },
                        ReportSeries {
                            label: "predicted".to_string(),
                            values: trace.predicted.clone(),
                        },
                        ReportSeries {
                            label: "error".to_string(),
                            values: error,
                        },
                    ],
                },
            },
        )?;
    }
    Ok(())
}

fn chained_candles_from_ohlc_features(features: &[f32], seed: &CandleBar) -> Vec<CandleBar> {
    let mut prev = seed.clone();
    features
        .chunks_exact(OHLC_BAR_FEATURES)
        .map(|row| {
            let candle = candle_from_ohlc_feature_row(row, &prev);
            prev = candle.clone();
            candle
        })
        .collect()
}

/// Reconstruct a bar's sanitized OHLC proportions from its own intra-bar feature
/// channels for use as a chain seed. With the close-anchored decode the chain
/// only consumes `seed.close`, so anchor `close` at 1.0 and derive
/// open/high/low from it via the O/C, H/C, L/C channels (mirroring the decode).
/// Seeding a chain with this bar telescopes into the following bars' sanitized
/// OHLC up to the `1/close` scale.
fn seed_candle_from_feature_row(row: &[f32]) -> CandleBar {
    let close = 1.0f64;
    let open = (close * (1.0 + row[6] as f64)).max(1e-6);
    let high0 = (close * (1.0 + row[9] as f64)).max(1e-6);
    let low0 = (close * (1.0 + row[12] as f64)).max(1e-6);
    let high = open.max(high0).max(low0).max(close);
    let low = open.min(high0).min(low0).min(close).max(1e-6);
    CandleBar {
        open: open as f32,
        high: high as f32,
        low: low as f32,
        close: close as f32,
    }
}

fn candle_from_ohlc_feature_row(row: &[f32], prev: &CandleBar) -> CandleBar {
    // Close-anchored decode: only the close level chains across bars, and the
    // intra-bar open/high/low are derived from this bar's own close via its
    // O/C, H/C, L/C channels. This bounds the per-bar high-low range instead of
    // letting the four channels diffuse apart independently across a rollout.
    let close = (prev.close as f64 * (1.0 + row[3] as f64)).max(1e-6);
    let open = (close * (1.0 + row[6] as f64)).max(1e-6);
    let high0 = (close * (1.0 + row[9] as f64)).max(1e-6);
    let low0 = (close * (1.0 + row[12] as f64)).max(1e-6);
    let high = open.max(high0).max(low0).max(close);
    let low = open.min(high0).min(low0).min(close).max(1e-6);
    CandleBar {
        open: open as f32,
        high: high as f32,
        low: low as f32,
        close: close as f32,
    }
}

fn tensor_to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    let tensor = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .view([-1]);
    let numel = tensor.numel();
    let mut values = vec![0.0f32; numel];
    tensor.copy_data(&mut values, numel);
    Ok(values)
}

fn write_report_file(path: &Path, report: &Report) -> Result<()> {
    let bytes = postcard::to_stdvec(report).context("failed to encode report")?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn future_patches_for_current_perm(
    env: &Env,
    offset: usize,
    k_patches: usize,
    patch_size: usize,
    target_scale: f64,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(TICKERS_COUNT as usize * k_patches * patch_size);
    let first_future = offset + 1;
    for &real_idx in &env.ticker_perm {
        let deltas = &env.price_deltas[real_idx];
        for patch_i in 0..k_patches {
            let start = first_future + patch_i * patch_size;
            let end = start + patch_size;
            out.extend(
                deltas[start..end]
                    .iter()
                    .map(|&v| (v * target_scale) as f32),
            );
        }
    }
    out
}

fn bar_history_for_current_perm(env: &Env, offset: usize) -> Vec<f32> {
    let start = offset + 1 - PRICE_DELTAS_PER_TICKER;
    let end = offset + 1;
    let mut out =
        Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER * OHLC_BAR_FEATURES);
    for &real_idx in &env.ticker_perm {
        append_ohlc_feature_window(&env.ohlc_features[real_idx], start, end, &mut out);
    }
    out
}

fn next_bars_for_current_perm(env: &Env, offset: usize) -> Vec<f32> {
    let start = offset + 1;
    let end = start + LEJEPA_ROLLOUT_BARS as usize;
    let mut out = Vec::with_capacity(
        TICKERS_COUNT as usize * LEJEPA_ROLLOUT_BARS as usize * OHLC_BAR_FEATURES,
    );
    for &real_idx in &env.ticker_perm {
        append_ohlc_feature_window(&env.ohlc_features[real_idx], start, end, &mut out);
    }
    out
}

fn append_ohlc_feature_window(
    features: &[[f32; OHLC_BAR_FEATURES]],
    start: usize,
    end: usize,
    out: &mut Vec<f32>,
) {
    for row in &features[start..end] {
        out.extend_from_slice(row);
    }
}

// JEDI differentiable latent bounding: z = B * tanh(z / B). Keeps encoder tokens
// and flow x-predictions within [-B, B] so they share the unit-N(0,I) prior scale.
fn latent_bound(z: &Tensor) -> Tensor {
    (z / LEJEPA_LATENT_BOUND).tanh() * LEJEPA_LATENT_BOUND
}

fn normalize_last_dim(x: &Tensor) -> Tensor {
    let mean = x.mean_dim([-1].as_slice(), true, Kind::Float);
    let centered = x - &mean;
    let var = centered
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    centered / (var + 1e-5).sqrt()
}

fn clip_all_grads(named_vars: &[(String, Tensor)], max_grad_norm: f64, device: Device) {
    tch::no_grad(|| {
        let mut total_norm_sq = Tensor::zeros([], (Kind::Float, device));
        let mut grads = Vec::new();
        for (_, param) in named_vars {
            let grad = param.grad();
            if grad.defined() {
                total_norm_sq += grad.square().sum(Kind::Float);
                grads.push(grad);
            }
        }
        let total_norm = total_norm_sq.sqrt();
        let coef = (Tensor::from(max_grad_norm as f32).to_device(device) / (&total_norm + 1e-6))
            .clamp_max(1.0);
        for mut grad in grads {
            let coef = coef.to_kind(grad.kind());
            let _ = grad.g_mul_(&coef);
        }
    });
}

#[derive(Clone, Copy, PartialEq)]
enum LejepaGradGroup {
    Encoder,
    Ar,
    Other,
}

// Routes a trainable parameter to its learning-dynamics group by name. Order
// matters: the per-bar encoder/projector (which also matches `lejepa_projector`)
// is checked first, leaving the AR transformer + shortcut flow head as the
// remaining `lejepa_` params. Everything else (base model params with no
// gradient during LeJEPA pretrain) is a catch-all.
fn lejepa_grad_group(name: &str) -> LejepaGradGroup {
    if name.contains("bar_proj")
        || name.contains("bar_enrich_")
        || name.contains("lejepa_projector")
    {
        LejepaGradGroup::Encoder
    } else if name.contains("lejepa_") {
        LejepaGradGroup::Ar
    } else {
        LejepaGradGroup::Other
    }
}

#[derive(Clone, Copy, Default)]
struct PretrainGradNorms {
    grad_total: f64,
    grad_encoder: f64,
    grad_ar: f64,
    grad_other: f64,
    pnorm_encoder: f64,
    pnorm_ar: f64,
}

// Pure instrumentation: reads `.grad()` and weights without mutating either, so
// it never perturbs training. Computes the global L2 grad norm per group plus
// the weight L2 norm per group (for update-to-weight ratios). Undefined grads
// are skipped so partially-active subgraphs are handled safely.
fn pretrain_grad_norms(named_vars: &[(String, Tensor)], device: Device) -> PretrainGradNorms {
    tch::no_grad(|| {
        let mut grad_encoder_sq = Tensor::zeros([], (Kind::Float, device));
        let mut grad_ar_sq = Tensor::zeros([], (Kind::Float, device));
        let mut grad_other_sq = Tensor::zeros([], (Kind::Float, device));
        let mut pnorm_encoder_sq = Tensor::zeros([], (Kind::Float, device));
        let mut pnorm_ar_sq = Tensor::zeros([], (Kind::Float, device));
        for (name, param) in named_vars {
            let group = lejepa_grad_group(name);
            let grad = param.grad();
            if grad.defined() {
                let sq = grad.square().sum(Kind::Float);
                match group {
                    LejepaGradGroup::Encoder => grad_encoder_sq += &sq,
                    LejepaGradGroup::Ar => grad_ar_sq += &sq,
                    LejepaGradGroup::Other => grad_other_sq += &sq,
                }
            }
            let psq = param.square().sum(Kind::Float);
            match group {
                LejepaGradGroup::Encoder => pnorm_encoder_sq += &psq,
                LejepaGradGroup::Ar => pnorm_ar_sq += &psq,
                LejepaGradGroup::Other => {}
            }
        }
        let grad_total_sq = &grad_encoder_sq + &grad_ar_sq + &grad_other_sq;
        PretrainGradNorms {
            grad_total: grad_total_sq.sqrt().double_value(&[]),
            grad_encoder: grad_encoder_sq.sqrt().double_value(&[]),
            grad_ar: grad_ar_sq.sqrt().double_value(&[]),
            grad_other: grad_other_sq.sqrt().double_value(&[]),
            pnorm_encoder: pnorm_encoder_sq.sqrt().double_value(&[]),
            pnorm_ar: pnorm_ar_sq.sqrt().double_value(&[]),
        }
    })
}

// Global L2 grad norm and weight L2 norm over an arbitrary param slice (used for
// the detached probe, whose params are all `probe_`).
fn named_grad_param_l2(named_vars: &[(String, Tensor)], device: Device) -> (f64, f64) {
    tch::no_grad(|| {
        let mut grad_sq = Tensor::zeros([], (Kind::Float, device));
        let mut param_sq = Tensor::zeros([], (Kind::Float, device));
        for (_, param) in named_vars {
            let grad = param.grad();
            if grad.defined() {
                grad_sq += grad.square().sum(Kind::Float);
            }
            param_sq += param.square().sum(Kind::Float);
        }
        (
            grad_sq.sqrt().double_value(&[]),
            param_sq.sqrt().double_value(&[]),
        )
    })
}

#[derive(Default)]
struct GradNormAccum {
    grad_total: f64,
    grad_encoder: f64,
    grad_ar: f64,
    grad_other: f64,
    pnorm_encoder: f64,
    pnorm_ar: f64,
    steps: usize,
}

impl GradNormAccum {
    fn add(&mut self, norms: &PretrainGradNorms) {
        self.grad_total += norms.grad_total;
        self.grad_encoder += norms.grad_encoder;
        self.grad_ar += norms.grad_ar;
        self.grad_other += norms.grad_other;
        self.pnorm_encoder += norms.pnorm_encoder;
        self.pnorm_ar += norms.pnorm_ar;
        self.steps += 1;
    }

    fn mean(&self) -> PretrainGradNorms {
        let denom = self.steps.max(1) as f64;
        PretrainGradNorms {
            grad_total: self.grad_total / denom,
            grad_encoder: self.grad_encoder / denom,
            grad_ar: self.grad_ar / denom,
            grad_other: self.grad_other / denom,
            pnorm_encoder: self.pnorm_encoder / denom,
            pnorm_ar: self.pnorm_ar / denom,
        }
    }
}

fn assert_finite_loss(loss: &Tensor, step: usize) {
    let loss_v = loss.double_value(&[]);
    assert!(
        loss_v.is_finite(),
        "non-finite pretrain loss at step {step}: {loss_v}"
    );
}

fn configure_threads() {
    if let Some(threads) = std::env::var("TORCH_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
    {
        tch::set_num_threads(threads);
    } else {
        tch::set_num_threads(1);
    }
    if let Some(threads) = std::env::var("TORCH_NUM_INTEROP_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
    {
        tch::set_num_interop_threads(threads);
    } else {
        tch::set_num_interop_threads(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bar_history_for_current_perm, build_split_offsets, candle_from_ohlc_feature_row,
        chained_candles_from_ohlc_features, cumulative_future_returns, lejepa_flow_loss,
        future_patches_for_current_perm, next_bars_for_current_perm,
        seed_candle_from_feature_row, sigreg_loss, CandleBar, PretrainHeads, SplitKind,
        LEJEPA_BAR_FEATURES, LEJEPA_ROLLOUT_BARS,
    };
    use crate::torch::{
        constants::PRICE_DELTAS_PER_TICKER,
        env::{build_ohlc_features, Env, OHLC_BAR_FEATURES},
        model::{ModelVariant, TradingModel, TradingModelConfig},
    };
    use tch::nn;
    use tch::Tensor;

    #[test]
    fn cumulative_future_returns_flattens_patches_and_accumulates_horizon() {
        let future_patches =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).view([1, 1, 2, 3]);
        let cumulative = cumulative_future_returns(&future_patches);
        let expected = Tensor::from_slice(&[1.0f32, 3.0, 6.0, 10.0, 15.0, 21.0]).view([1, 1, 6]);
        let max_diff = (cumulative - expected).abs().max().double_value(&[]);
        assert!(max_diff < 1e-6, "cumulative target mismatch: {max_diff}");
    }

    #[test]
    fn future_patches_follow_current_ticker_permutation() {
        let mut env = Env::new(false);
        let offset = crate::torch::constants::PRICE_DELTAS_PER_TICKER;
        let _ = env.reset_single_at_offset_for_pretrain(offset);
        let patches = future_patches_for_current_perm(&env, offset, 2, 3, 1.0);
        assert_eq!(
            patches.len(),
            crate::torch::constants::TICKERS_COUNT as usize * 2 * 3
        );
        let real_idx = env.ticker_perm[0];
        assert_eq!(patches[0], env.price_deltas[real_idx][offset + 1] as f32);
        assert_eq!(patches[3], env.price_deltas[real_idx][offset + 4] as f32);
    }

    #[test]
    fn bar_history_matches_observation_window_and_close_deltas() {
        let mut env = Env::new(false);
        let offset = crate::torch::constants::PRICE_DELTAS_PER_TICKER;
        let _ = env.reset_single_at_offset_for_pretrain(offset);
        let bars = bar_history_for_current_perm(&env, offset);
        assert_eq!(
            bars.len(),
            crate::torch::constants::TICKERS_COUNT as usize
                * PRICE_DELTAS_PER_TICKER
                * OHLC_BAR_FEATURES
        );

        let real_idx = env.ticker_perm[0];
        let first_bar_idx = offset + 1 - PRICE_DELTAS_PER_TICKER;
        let last_bar_idx = offset;
        assert_eq!(bars[3], env.ohlc_features[real_idx][first_bar_idx][3]);
        let last_close_delta_offset = (PRICE_DELTAS_PER_TICKER - 1) * OHLC_BAR_FEATURES + 3;
        assert_eq!(
            bars[last_close_delta_offset],
            env.ohlc_features[real_idx][last_bar_idx][3]
        );

        // Cross-check against an INDEPENDENT recomputation from the raw bars, so
        // the assertions test the feature derivation rather than the copied memory.
        let ticker = env.tickers[real_idx].clone();
        let raw_bars = crate::data::historical::get_historical_data(Some(&[ticker.as_str()]));
        let recomputed = build_ohlc_features(&raw_bars[0]);
        assert_eq!(bars[3], recomputed[first_bar_idx][3]);
        assert_eq!(bars[last_close_delta_offset], recomputed[last_bar_idx][3]);
    }

    #[test]
    fn ohlc_feature_round_trip_recovers_candle() {
        use ibapi::market_data::historical::Bar;
        use time::{Duration, OffsetDateTime};
        let mk = |open: f64, high: f64, low: f64, close: f64| Bar {
            date: OffsetDateTime::UNIX_EPOCH + Duration::minutes(5),
            open,
            high,
            low,
            close,
            volume: 1_000.0,
            wap: close,
            count: 1,
        };
        let prev = mk(100.0, 105.0, 98.0, 102.0);
        let cur = mk(102.0, 108.0, 101.0, 106.0);
        let feats = build_ohlc_features(&[prev, cur]);
        // Close-anchored decode consumes only prev.close; the other prev fields
        // are bogus here to prove they no longer leak into reconstruction.
        let prev_candle = CandleBar {
            open: 1.0,
            high: 999.0,
            low: 0.001,
            close: 102.0,
        };
        let candle = candle_from_ohlc_feature_row(&feats[1], &prev_candle);
        // close = prev.close*(1+C/prevC), then open/high/low derive from close
        // via O/C, H/C, L/C, recovering cur's sanitized OHLC.
        assert!(
            (candle.close - 106.0).abs() < 1e-3,
            "close {}",
            candle.close
        );
        assert!((candle.open - 102.0).abs() < 1e-3, "open {}", candle.open);
        assert!((candle.high - 108.0).abs() < 1e-3, "high {}", candle.high);
        assert!((candle.low - 101.0).abs() < 1e-3, "low {}", candle.low);
    }

    #[test]
    fn chained_candles_recover_sanitized_ohlc_shapes() {
        use ibapi::market_data::historical::Bar;
        use time::{Duration, OffsetDateTime};
        let mk = |open: f64, high: f64, low: f64, close: f64| Bar {
            date: OffsetDateTime::UNIX_EPOCH + Duration::minutes(5),
            open,
            high,
            low,
            close,
            volume: 1_000.0,
            wap: close,
            count: 1,
        };
        // >=3 real bars with distinct, non-trivial OHLC proportions.
        let bars = vec![
            mk(100.0, 105.0, 98.0, 102.0),
            mk(102.0, 108.0, 101.0, 106.0),
            mk(106.0, 107.0, 99.0, 100.0),
            mk(100.0, 104.0, 97.0, 103.0),
        ];
        let feats = build_ohlc_features(&bars);

        // Chain-decode the windowed bars (rows 1..n) seeded from the TRUE first
        // bar's sanitized OHLC, exactly as the production diagnostics path does.
        let seed = seed_candle_from_feature_row(&feats[0]);
        let mut windowed = Vec::new();
        for row in &feats[1..] {
            windowed.extend_from_slice(row);
        }
        let candles = chained_candles_from_ohlc_features(&windowed, &seed);
        assert_eq!(candles.len(), bars.len() - 1);

        // The seed anchors bar0's close at 1.0, so the close-anchored chain
        // recovers each later bar's SANITIZED OHLC scaled by 1/bar0.close.
        // Shape fidelity, not just per-row math: fails with a flat {1,1,1,1} seed.
        let scale = 1.0 / bars[0].close;
        for (i, candle) in candles.iter().enumerate() {
            let bar = &bars[i + 1];
            let high_san = bar.high.max(bar.open).max(bar.close);
            let low_san = bar.low.min(bar.open).min(bar.close);
            assert!(
                (candle.open as f64 - bar.open * scale).abs() < 1e-3,
                "bar {} open {} vs {}",
                i + 1,
                candle.open,
                bar.open * scale
            );
            assert!(
                (candle.high as f64 - high_san * scale).abs() < 1e-3,
                "bar {} high {} vs {}",
                i + 1,
                candle.high,
                high_san * scale
            );
            assert!(
                (candle.low as f64 - low_san * scale).abs() < 1e-3,
                "bar {} low {} vs {}",
                i + 1,
                candle.low,
                low_san * scale
            );
            assert!(
                (candle.close as f64 - bar.close * scale).abs() < 1e-3,
                "bar {} close {} vs {}",
                i + 1,
                candle.close,
                bar.close * scale
            );
        }
    }

    #[test]
    fn next_bars_start_after_current_offset() {
        let mut env = Env::new(false);
        let offset = crate::torch::constants::PRICE_DELTAS_PER_TICKER;
        let _ = env.reset_single_at_offset_for_pretrain(offset);
        let bars = next_bars_for_current_perm(&env, offset);
        assert_eq!(
            bars.len(),
            crate::torch::constants::TICKERS_COUNT as usize
                * LEJEPA_ROLLOUT_BARS as usize
                * OHLC_BAR_FEATURES
        );

        let real_idx = env.ticker_perm[0];
        assert_eq!(bars[3], env.ohlc_features[real_idx][offset + 1][3]);

        // Independent recomputation from the raw bars (derivation cross-check).
        let ticker = env.tickers[real_idx].clone();
        let raw_bars = crate::data::historical::get_historical_data(Some(&[ticker.as_str()]));
        let recomputed = build_ohlc_features(&raw_bars[0]);
        assert_eq!(bars[3], recomputed[offset + 1][3]);
    }

    #[test]
    fn uniform_stream_pretrain_patch_size_is_25() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
                ..TradingModelConfig::default()
            },
        );
        assert_eq!(model.pretrain_patch_size(), 25);
        assert_eq!(model.pretrain_patch_token_count(), 240);
        assert_eq!(model.pretrain_layout_len(), PRICE_DELTAS_PER_TICKER as i64);
    }

    #[test]
    fn split_offsets_allow_last_future_safe_patch_aligned_anchor() {
        let data_len = PRICE_DELTAS_PER_TICKER + 801;
        let offsets = build_split_offsets(data_len, 16, 25, SplitKind::Validation);
        let last = *offsets
            .last()
            .expect("validation offsets should be non-empty");
        assert_eq!(last + 1 + 16 * 25, data_len);
    }

    #[test]
    fn train_split_keeps_forecast_targets_before_validation_contexts() {
        let data_len = PRICE_DELTAS_PER_TICKER + 10_000;
        let train = build_split_offsets(data_len, 16, 25, SplitKind::Train);
        let validation = build_split_offsets(data_len, 16, 25, SplitKind::Validation);
        let last_train = *train.last().expect("train offsets should be non-empty");
        let first_validation = *validation
            .first()
            .expect("validation offsets should be non-empty");
        assert!(last_train + 16 * 25 <= first_validation);
    }

    #[test]
    fn sigreg_penalizes_per_position_scale_above_unit() {
        let _guard = tch::no_grad_guard();
        let positions = 8i64;
        let samples = 512i64;
        let dim = 32i64;
        let opts = (tch::Kind::Float, tch::Device::Cpu);
        let unit = Tensor::randn([positions, samples, dim], opts);
        let inflated = &unit * 3.0_f64.sqrt();
        let unit_loss = sigreg_loss(&unit).double_value(&[]);
        let inflated_loss = sigreg_loss(&inflated).double_value(&[]);
        assert!(
            inflated_loss > unit_loss * 4.0,
            "variance-3 sigreg {inflated_loss} should dwarf unit-variance {unit_loss}"
        );
    }

    #[test]
    fn probe_predicts_single_next_bar() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let latent_dim = 256;
        let heads = PretrainHeads::new(&vs.root(), latent_dim, 16, 25);
        let batch = 4;
        let belief = Tensor::randn(
            [batch, 1, 1, latent_dim],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        let (pred, logvar) = heads.probe_ohlc_features(&belief);
        assert_eq!(pred.size(), vec![batch, 1, 1, LEJEPA_BAR_FEATURES]);
        assert_eq!(logvar.size(), vec![batch, 1, 1, LEJEPA_BAR_FEATURES]);

        let sigma = (&logvar * 0.5).exp();
        assert!(
            sigma.isfinite().all().int64_value(&[]) != 0,
            "predicted sigma must be finite"
        );
        let min_sigma = sigma.min().double_value(&[]);
        assert!(
            min_sigma > 0.0,
            "predicted sigma must be positive, got {min_sigma}"
        );
    }

    #[test]
    fn imagined_rollout_grows_tokens_and_yields_rollout_bars() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let latent_dim = 256;
        let heads = PretrainHeads::new(&vs.root(), latent_dim, 16, 25);
        let batch = 2;
        let context_len = 8;
        let context = Tensor::randn(
            [batch, 1, context_len, LEJEPA_BAR_FEATURES],
            (tch::Kind::Float, tch::Device::Cpu),
        );

        let imagined = heads.lejepa_imagined_rollout(&context, 1.0, false);
        assert_eq!(
            imagined.size(),
            vec![batch, LEJEPA_ROLLOUT_BARS, LEJEPA_BAR_FEATURES]
        );
        assert!(
            imagined.isfinite().all().int64_value(&[]) != 0,
            "imagined rollout must be finite"
        );
    }

    #[test]
    fn flow_loss_handles_gapped_narrowed_targets() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let latent_dim = 256;
        let heads = PretrainHeads::new(&vs.root(), latent_dim, 16, 25);
        let batch = 2;
        let length = 6;
        let bars = Tensor::randn(
            [batch, 1, length + 1, LEJEPA_BAR_FEATURES],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        // Mirrors lejepa_pretrain_loss: belief and clean target are reshaped from a
        // non-contiguous narrowed slice that folds a gapped position dim into rows.
        let all_tokens = heads.encode_bar_tokens(&bars, true);
        let bar_tokens = all_tokens.narrow(2, 0, length);
        let target_bar_tokens = all_tokens.narrow(2, 1, length);
        let belief = heads.predict_lejepa_bar_predictions(&bar_tokens, true).belief;
        let rows = target_bar_tokens.numel() as i64 / latent_dim;
        let ctx = belief.reshape([rows, latent_dim]);
        let clean = target_bar_tokens.reshape([rows, latent_dim]);
        let loss = lejepa_flow_loss(&heads, &ctx, &clean);
        let value = loss.double_value(&[]);
        assert!(value.is_finite(), "flow loss must be finite, got {value}");
        assert!(value >= 0.0, "flow loss must be non-negative, got {value}");
    }
}
