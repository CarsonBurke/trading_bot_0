use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{rngs::StdRng, SeedableRng};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};
use tch::{autocast, nn, nn::Module, nn::ModuleT, Device, Kind, Reduction, Tensor};

use crate::data::universe::cached_eligible_training_universe;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::env::{Env, OHLC_BAR_FEATURES};
use crate::torch::fa4::pope_flash_attention_prefill;
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{ModelVariant, TradingModel, TradingModelConfig};
use crate::torch::optim::muon::{Muon, MuonConfig};
#[cfg(test)]
use crate::torch::pope::pope_attention_reference;
use crate::torch::pope::{
    init_pope_theta_bias, pope_expand_qk_fp32, PolarQk, PopeThetaInit, POPE_FREQUENCY_BASE,
};
use crate::torch::world_model::{
    world_model_metadata_path, LejepaWorldModel, WorldModelMetadata, LEJEPA_AR_FF_DIM,
    LEJEPA_AR_LAYERS, LEJEPA_CACHE_CONTRACT, LEJEPA_HEADS, LEJEPA_HEAD_DIM,
    LEJEPA_NORMALIZATION_EPS, LEJEPA_PREDICTOR_BLOCKS, LEJEPA_PREDICTOR_HIDDEN_MULT,
    LEJEPA_PROBE_LOGVAR_LIMIT, LEJEPA_PROJECTOR_HIDDEN_DIM, OHLC_FEATURE_SCALE,
};
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
// Epps-Pulley statistic scale. le-wm uses `* proj.size(-2)` = its batch B = 128.
// Our micro-batch has only `batch_tickers` samples on the SIGReg sample axis, so
// scaling by that count makes the regularizer ~16x weaker than le-wm at the same
// lambda, which lets the encoder collapse. Match le-wm's absolute calibration.
const LEJEPA_SIGREG_REFERENCE_SAMPLES: i64 = 128;
const LEJEPA_BAR_FEATURES: i64 = OHLC_BAR_FEATURES as i64;
const LEJEPA_ROLLOUT_BARS: i64 = 100;
const LEJEPA_ROLLOUT_EVAL_WINDOWS: usize = 64;
const LEJEPA_PROBE_LR: f64 = 1e-3;
const LEJEPA_WEIGHT_DECAY: f64 = 1e-3;
const LEJEPA_PROBE_VAR_EMA_DECAY: f64 = 0.99;
const INTER_BAR_FEATURES: i64 = 4;
const TRAIN_ANCHOR_STRIDE_MULTIPLIER: usize = 4;
/// Number of fixed validation windows tracked by the candle-snapshot diagnostic.
const CANDLE_SNAPSHOT_WINDOWS: usize = 4;
/// Fixed seed so the candle-snapshot windows are the same for a whole run.
const CANDLE_SNAPSHOT_SEED: u64 = 0xC0FFEE;

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
    pub eval_skill_only: bool,
    pub batch_size: usize,
    pub k_patches: usize,
    pub objective: PretrainObjective,
    pub lambda_lat: f64,
    pub lambda_sigreg: f64,
    pub target_scale: f64,
    pub validation_batches: usize,
    pub validate_every: usize,
    pub checkpoint_every: usize,
    pub step_val_every: usize,
    pub candle_snapshot_every: usize,
}

struct CausalLejepaLayer {
    qkv: nn::Linear,
    pope_theta_bias: Tensor,
    out_proj: nn::Linear,
    ff_gate: nn::Linear,
    ff_value: nn::Linear,
    ff_out: nn::Linear,
}

struct ProjectionMlp {
    fc1: nn::Linear,
    gamma: Tensor,
    fc2: nn::Linear,
}

struct LejepaBarPredictions {
    belief: Tensor,
}

struct LejepaPredictorBlock {
    gate: nn::Linear,
    value: nn::Linear,
    out: nn::Linear,
}

struct LejepaPredictorHead {
    in_proj: nn::Linear,
    blocks: Vec<LejepaPredictorBlock>,
    out_proj: nn::Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValidationMode {
    Fast,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PretrainExecutionMode {
    Train,
    EvaluateOnly,
}

fn pretrain_execution_mode(args: &PretrainArgs) -> Result<PretrainExecutionMode> {
    if args.eval_skill_only
        && (args.steps != Some(0)
            || args.weights.is_none()
            || args.objective != PretrainObjective::Lejepa)
    {
        return Err(anyhow!(
            "--eval-skill-only requires --steps 0, --weights <checkpoint>, and --objective lejepa"
        ));
    }
    match args.steps {
        Some(0) if args.weights.is_none() => Err(anyhow!(
            "--steps 0 is evaluation-only and requires --weights <checkpoint>"
        )),
        Some(0) => Ok(PretrainExecutionMode::EvaluateOnly),
        Some(_) | None => Ok(PretrainExecutionMode::Train),
    }
}

fn enforce_single_sampler_pass(
    execution_mode: PretrainExecutionMode,
    epochs: usize,
    step_limit: Option<usize>,
    batches_per_epoch: usize,
) -> Result<()> {
    if execution_mode == PretrainExecutionMode::EvaluateOnly {
        return Ok(());
    }
    let available_batches = epochs.saturating_mul(batches_per_epoch);
    let planned_batches = step_limit
        .unwrap_or(available_batches)
        .min(available_batches);
    if planned_batches > batches_per_epoch {
        return Err(anyhow!(
            "training plan would consume {planned_batches} batches across repeated sampler passes, but the raw-transition reuse cap allows at most one {batches_per_epoch}-batch pass; reduce --steps or use --epochs 1"
        ));
    }
    Ok(())
}

fn strict_pope_prefill_attention(qk: &PolarQk, value_bshd: &Tensor) -> Tensor {
    if value_bshd.device().is_cuda() {
        return autocast(true, || pope_flash_attention_prefill(qk, value_bshd))
            .unwrap_or_else(|error| panic!("strict FA4 PoPE prefill failed: {error:#}"));
    }
    #[cfg(test)]
    {
        return pope_attention_reference(qk, value_bshd, true);
    }
    #[cfg(not(test))]
    panic!("PoPE pretraining requires CUDA with the strict FA4 bridge");
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
    lejepa_predictor: LejepaPredictorHead,
    probe_input_ln: nn::LayerNorm,
    probe_head: nn::Linear,
    probe_logvar_head: nn::Linear,
    // Non-trainable EMA running variance used only by the detached OHLC probe.
    feat_var_ema: Tensor,
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
    train_pairs: Vec<(usize, usize)>,
    train_cursor: usize,
    val_pairs: Vec<(usize, usize)>,
    val_eval_cursor: usize,
    test_pairs: Vec<(usize, usize)>,
    k_patches: usize,
    patch_size: usize,
    target_scale: f64,
    device: Device,
}

#[derive(Clone, Copy)]
enum SplitKind {
    Train,
    Validation,
    Test,
}

impl PretrainHeads {
    fn new(p: &nn::Path, latent_dim: i64, k_patches: i64, patch_size: i64) -> Self {
        let ff_dim = latent_dim * 2;
        let horizon = k_patches * patch_size;
        let forecast_heads = 4;
        let lejepa_heads = LEJEPA_HEADS;
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
            gamma: p.var(
                "lejepa_projector_norm_gamma",
                &[LEJEPA_PROJECTOR_HIDDEN_DIM],
                nn::Init::Const(1.0),
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
                pope_theta_bias: init_pope_theta_bias(
                    &layer_path,
                    "pope_theta_bias",
                    LEJEPA_HEADS,
                    LEJEPA_HEAD_DIM,
                    PRICE_DELTAS_PER_TICKER as i64,
                    PopeThetaInit::TwoPi,
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
        let predictor_hidden = latent_dim * LEJEPA_PREDICTOR_HIDDEN_MULT;
        let mut predictor_blocks = Vec::with_capacity(LEJEPA_PREDICTOR_BLOCKS);
        for block_idx in 0..LEJEPA_PREDICTOR_BLOCKS {
            let block_path = p / format!("lejepa_predictor_block_{block_idx}");
            predictor_blocks.push(LejepaPredictorBlock {
                gate: nn::linear(
                    &block_path / "gate",
                    latent_dim,
                    predictor_hidden,
                    Default::default(),
                ),
                value: nn::linear(
                    &block_path / "value",
                    latent_dim,
                    predictor_hidden,
                    Default::default(),
                ),
                out: nn::linear(
                    &block_path / "out",
                    predictor_hidden,
                    latent_dim,
                    Default::default(),
                ),
            });
        }
        let lejepa_predictor = LejepaPredictorHead {
            in_proj: nn::linear(
                p / "lejepa_predictor_in_proj",
                latent_dim,
                latent_dim,
                Default::default(),
            ),
            blocks: predictor_blocks,
            out_proj: nn::linear(
                p / "lejepa_predictor_out_proj",
                latent_dim,
                latent_dim,
                Default::default(),
            ),
        };
        let feat_var_ema = p.ones_no_train("lejepa_feat_var_ema", &[LEJEPA_BAR_FEATURES]);
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
            lejepa_predictor,
            probe_input_ln,
            probe_head,
            probe_logvar_head,
            feat_var_ema,
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
        let scale = Tensor::from_slice(&OHLC_FEATURE_SCALE).to_device(features.device());
        let features = features / scale;
        let h = self.bar_proj.forward(&features);
        let enriched = self.bar_enrich_fc2.forward(
            &normalize_last_dim(&self.bar_enrich_fc1.forward(&normalize_last_dim(&h))).gelu("none"),
        );
        let h = h + enriched;
        let tokens = h.view([batch, tickers, length, self.latent_dim]);
        self.project_lejepa_tokens(&tokens, train)
    }

    fn project_lejepa_tokens(&self, tokens: &Tensor, train: bool) -> Tensor {
        self.projection_mlp(tokens, &self.lejepa_projector, train)
    }

    // AR transformer belief = final normalized representation, one per position.
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

    // Deterministic conditional-mean next-latent prediction from the AR belief.
    fn lejepa_predict_next(&self, belief: &Tensor) -> Tensor {
        let predictor = &self.lejepa_predictor;
        let mut h = predictor.in_proj.forward(&normalize_last_dim(belief));
        for block in &predictor.blocks {
            let normed = normalize_last_dim(&h);
            let gated = block.gate.forward(&normed).silu() * block.value.forward(&normed);
            h += block.out.forward(&gated);
        }
        predictor.out_proj.forward(&normalize_last_dim(&h))
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
        let q = parts[0].view([rows, length, self.lejepa_heads, head_dim]);
        let k = parts[1].view([rows, length, self.lejepa_heads, head_dim]);
        let v = parts[2].view([rows, length, self.lejepa_heads, head_dim]);
        let polar = pope_expand_qk_fp32(
            &q,
            &k,
            positions,
            positions,
            &layer.pope_theta_bias,
            POPE_FREQUENCY_BASE,
        );
        let attn_kind = if source.device().is_cuda() {
            Kind::BFloat16
        } else {
            source.kind()
        };
        let polar = PolarQk {
            query: polar.query.to_kind(attn_kind).contiguous(),
            key: polar.key.to_kind(attn_kind).contiguous(),
        };
        let value = v.to_kind(attn_kind).contiguous();
        let attn = strict_pope_prefill_attention(&polar, &value)
            .to_kind(source.kind())
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

    fn projection_mlp(&self, x: &Tensor, mlp: &ProjectionMlp, _train: bool) -> Tensor {
        let shape = x.size();
        let rows = x.numel() as i64 / self.latent_dim;
        let flat = x.view([rows, self.latent_dim]);
        let hidden = mlp.fc1.forward(&flat);
        let hidden = hidden
            .internal_fused_rms_norm([LEJEPA_PROJECTOR_HIDDEN_DIM], Some(&mlp.gamma), Some(1e-6))
            .0
            .gelu("none");
        mlp.fc2.forward(&hidden).view(shape.as_slice())
    }

    // Rank-preserving probe: maps a latent [.., D] -> (mean, logvar) each [.., 16].
    fn probe_ohlc_features(&self, latent: &Tensor) -> (Tensor, Tensor) {
        let normed = self.probe_input_ln.forward(latent);
        let mean = self.probe_head.forward(&normed);
        let logvar = self
            .probe_logvar_head
            .forward(&normed)
            .clamp(-LEJEPA_PROBE_LOGVAR_LIMIT, LEJEPA_PROBE_LOGVAR_LIMIT);
        (mean, logvar)
    }

    fn lejepa_imagined_rollout(&self, context_bars: &Tensor) -> Tensor {
        self.lejepa_imagined_rollout_inner(context_bars, false).0
    }

    // Deterministic autoregressive rollout: the AR belief is fed through the
    // conditional-mean predictor, decoded to OHLC via the probe, and appended.
    fn lejepa_imagined_rollout_inner(
        &self,
        context_bars: &Tensor,
        collect_entropy: bool,
    ) -> (Tensor, Option<RolloutEntropy>) {
        let mut tokens = self.encode_bar_tokens(context_bars, false).detach();
        let size = tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let latent_dim = self.latent_dim;
        let rows = batch * tickers;
        let mut imagined = Vec::with_capacity(LEJEPA_ROLLOUT_BARS as usize);
        let mut ent_means: Vec<Tensor> = Vec::new();
        let mut tok_norm_sum = 0.0f64;
        let mut tok_norm_max = 0.0f64;
        for _ in 0..LEJEPA_ROLLOUT_BARS {
            let belief = self.predict_lejepa_bar_predictions(&tokens, false).belief;
            let last = tokens.size()[2] - 1;
            let ctx = belief.narrow(2, last, 1).reshape([rows, latent_dim]);
            let next_z = self.lejepa_predict_next(&ctx);
            let next_token = next_z.view([batch, tickers, 1, latent_dim]);
            let (mean, _logvar) = self.probe_ohlc_features(&next_token);
            let bar = mean.view([batch, LEJEPA_BAR_FEATURES]);
            imagined.push(bar.shallow_clone());
            if collect_entropy {
                let nt_n = next_token
                    .reshape([batch * tickers, latent_dim])
                    .square()
                    .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                    .sqrt();
                tok_norm_sum += nt_n.mean(Kind::Float).double_value(&[]);
                tok_norm_max = tok_norm_max.max(nt_n.max().double_value(&[]));
                ent_means.push(bar);
            }
            tokens = Tensor::cat(&[&tokens, &next_token], 2);
            let len = tokens.size()[2];
            let max_len = PRICE_DELTAS_PER_TICKER as i64;
            if len > max_len {
                tokens = tokens.narrow(2, len - max_len, max_len);
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
    fn new(k_patches: usize, patch_size: usize, target_scale: f64, device: Device) -> Self {
        assert_eq!(
            TICKERS_COUNT, 1,
            "full-universe pretraining currently expects one ticker per observation"
        );
        let mut train_tickers = cached_eligible_training_universe().to_vec();
        train_tickers.shuffle(&mut rand::rng());
        let mut usable_train_tickers = Vec::with_capacity(train_tickers.len());
        let mut train_envs = Vec::with_capacity(train_tickers.len());
        let mut dense_train_pairs = Vec::new();
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
            let env_idx = train_envs.len();
            dense_train_pairs.extend(offsets.into_iter().map(|offset| (env_idx, offset)));
            usable_train_tickers.push(ticker);
            train_envs.push(env);
        }
        let train_pairs = retain_spaced_train_pairs(dense_train_pairs);
        assert!(
            !usable_train_tickers.is_empty(),
            "not enough market history for pretraining: train_tickers={}",
            usable_train_tickers.len()
        );
        assert!(
            !train_pairs.is_empty(),
            "no training pairs available for pretraining"
        );
        let mut val_pairs = Vec::new();
        let mut test_pairs = Vec::new();
        for (env_idx, env) in train_envs.iter().enumerate() {
            let data_len = env.price_deltas[0].len();
            for offset in
                build_split_offsets(data_len, k_patches, patch_size, SplitKind::Validation)
            {
                val_pairs.push((env_idx, offset));
            }
            for offset in build_split_offsets(data_len, k_patches, patch_size, SplitKind::Test) {
                test_pairs.push((env_idx, offset));
            }
        }
        val_pairs.shuffle(&mut rand::rng());
        test_pairs.shuffle(&mut rand::rng());
        Self {
            train_tickers: usable_train_tickers,
            train_envs,
            train_pairs,
            train_cursor: 0,
            val_pairs,
            val_eval_cursor: 0,
            test_pairs,
            k_patches,
            patch_size,
            target_scale,
            device,
        }
    }

    /// Draw one round-robin validation mini-batch from the pre-shuffled validation
    /// pair list, cycling with a persistent cursor. `None` when no validation pairs.
    fn next_val_eval_batch(&mut self, batch_size: usize) -> Option<PretrainBatch> {
        if self.val_pairs.is_empty() {
            return None;
        }
        let mut picks = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            picks.push(self.val_pairs[self.val_eval_cursor]);
            self.val_eval_cursor = (self.val_eval_cursor + 1) % self.val_pairs.len();
        }
        Some(self.batch_for_pairs(&picks))
    }

    /// Batches per full-data epoch, emergent from batch size (final partial chunk dropped).
    fn batches_per_epoch(&self, batch_size: usize) -> usize {
        self.train_pairs.len() / batch_size
    }

    /// Reshuffle all train pairs so the next epoch is a fresh full pass over the data.
    fn start_epoch(&mut self) {
        self.train_pairs.shuffle(&mut rand::rng());
        self.train_cursor = 0;
    }

    /// Advance the epoch cursor by one consecutive chunk of `batch_size` pairs,
    /// returning `None` once fewer than `batch_size` pairs remain (partial chunk dropped).
    fn take_train_chunk(&mut self, batch_size: usize) -> Option<&[(usize, usize)]> {
        let end = self.train_cursor + batch_size;
        if end > self.train_pairs.len() {
            return None;
        }
        let start = self.train_cursor;
        self.train_cursor = end;
        Some(&self.train_pairs[start..end])
    }

    fn next_train_batch(&mut self, batch_size: usize) -> Option<PretrainBatch> {
        let samples = self.take_train_chunk(batch_size)?.to_vec();
        Some(Self::batch_from_env_offsets(
            &mut self.train_envs,
            &samples,
            self.k_patches,
            self.patch_size,
            self.target_scale,
            self.device,
        ))
    }

    fn batch_for_pairs(&mut self, pairs: &[(usize, usize)]) -> PretrainBatch {
        Self::batch_from_env_offsets(
            &mut self.train_envs,
            pairs,
            self.k_patches,
            self.patch_size,
            self.target_scale,
            self.device,
        )
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
    // Chronological 80/10/10 split. Two patch-aligned split points carve the usable
    // anchor range into train (first 80%), validation (next 10%), test (final 10%,
    // most-future). A per-split margin of `max_target_advance` keeps each split's
    // forecast/rollout targets from leaking into the next split's contexts.
    let usable = max_exclusive - min_offset;
    let split_train = align_up_to_step(
        min_offset + (usable * 8 / 10).max(1),
        min_offset,
        patch_size,
    );
    let split_val = align_up_to_step(
        min_offset + (usable * 9 / 10).max(1),
        min_offset,
        patch_size,
    );
    let (start, end) = match split_kind {
        SplitKind::Train => (min_offset, split_train.saturating_sub(max_target_advance)),
        SplitKind::Validation => (split_train, split_val.saturating_sub(max_target_advance)),
        SplitKind::Test => (split_val, max_exclusive),
    };
    if start >= end {
        return Vec::new();
    }
    (start..end).step_by(patch_size).collect()
}

fn retain_spaced_train_pairs(pairs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut seen_per_ticker = Vec::<usize>::new();
    pairs
        .into_iter()
        .filter(|&(ticker, _)| {
            if ticker >= seen_per_ticker.len() {
                seen_per_ticker.resize(ticker + 1, 0);
            }
            let seen = &mut seen_per_ticker[ticker];
            let retain = *seen % TRAIN_ANCHOR_STRIDE_MULTIPLIER == 0;
            *seen += 1;
            retain
        })
        .collect()
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
    let execution_mode = pretrain_execution_mode(&args)?;
    assert_eq!(
        args.model_size,
        ModelVariant::UniformStream,
        "world-model pretraining currently supports --model-size uniform-stream only"
    );
    assert!(args.epochs > 0, "--epochs must be positive");
    assert!(args.batch_size > 0, "--batch-size must be positive");
    assert!(args.k_patches > 0, "--k-patches must be positive");
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
        device,
    );
    assert!(
        sampler.batches_per_epoch(args.batch_size) > 0,
        "--batch-size {} is larger than the available pretrain training pairs {}; reduce batch size or widen the training universe",
        args.batch_size,
        sampler.train_pairs.len()
    );
    enforce_single_sampler_pass(
        execution_mode,
        args.epochs,
        args.steps,
        sampler.batches_per_epoch(args.batch_size),
    )?;
    let mut head_vs = nn::VarStore::new(device);
    let heads = PretrainHeads::new(
        &head_vs.root(),
        model.pretrain_latent_dim(),
        args.k_patches as i64,
        patch_size,
    );
    if let Some(path) = start_weights.as_deref() {
        load_matching_pretrain_heads(&mut head_vs, path, args.objective)?;
    }

    let mut named_vars = named_trainable_variables(&model_vs);
    // Probe params train online via their own optimizer every step, so they are
    // excluded from the model optimizer to avoid double-updates.
    named_vars.extend(
        named_trainable_variables(&head_vs)
            .into_iter()
            .filter(|(name, _)| !name.contains("probe_"))
            .map(|(name, tensor)| (format!("pretrain_heads.{name}"), tensor)),
    );
    let mut lejepa_muon_allowlist = vec![
        "bar_proj".to_string(),
        "bar_enrich".to_string(),
        "lejepa_projector_fc".to_string(),
        "lejepa_layer_".to_string(),
        "lejepa_predictor_in_proj".to_string(),
        "lejepa_predictor_out_proj".to_string(),
    ];
    for block_idx in 0..LEJEPA_PREDICTOR_BLOCKS {
        lejepa_muon_allowlist.push(format!("lejepa_predictor_block_{block_idx}.gate"));
        lejepa_muon_allowlist.push(format!("lejepa_predictor_block_{block_idx}.value"));
        lejepa_muon_allowlist.push(format!("lejepa_predictor_block_{block_idx}.out"));
    }
    let (force_adamw_name_substrings, muon_name_allowlist) =
        if args.objective == PretrainObjective::Lejepa {
            (vec!["pope_theta_bias".to_string()], lejepa_muon_allowlist)
        } else {
            (
                vec![
                    "policy_concentration".to_string(),
                    "value_proj".to_string(),
                    "forecast_".to_string(),
                    "horizon_pos_proj".to_string(),
                    "return_mean".to_string(),
                    "bar_proj".to_string(),
                    "bar_enrich_".to_string(),
                    "lejepa_".to_string(),
                ],
                Vec::new(),
            )
        };
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
            adamw_no_weight_decay_name_substrings: vec!["pope_theta_bias".to_string()],
            force_adamw_name_substrings,
            muon_name_allowlist,
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
    let mut best_rollout_mean_mse = f64::INFINITY;
    let mut scalar_history = PretrainScalarHistory::default();
    let mut stop_requested = false;
    let final_path = run_dir.weights.join("pretrain_model.ot");
    let best_path = run_dir.weights.join("pretrain_model_best.ot");
    let final_heads_path = run_dir.weights.join("pretrain_heads.ot");
    let best_heads_path = run_dir.weights.join("pretrain_heads_best.ot");
    // Dense per-step train-vs-val total-loss curve (val is NaN on non-eval steps),
    // emitted as the `pretrain_step_loss` report at each epoch end.
    let mut step_loss_train: Vec<f32> = Vec::new();
    let mut step_loss_val: Vec<f32> = Vec::new();
    // Candle deterministic-rollout scalar diagnostics, accumulated across snapshots
    // and emitted as reports from within `write_candle_snapshots`.
    let mut candle_rollout_mse_hist: Vec<f32> = Vec::new();
    let mut candle_rollout_dclose_hist: Vec<f32> = Vec::new();

    // Fixed validation windows for the candle-snapshot diagnostic, chosen once so
    // the same rollouts are tracked across the whole run.
    let candle_windows: Vec<(usize, usize)> = {
        let mut snap_rng = StdRng::seed_from_u64(CANDLE_SNAPSHOT_SEED);
        sampler
            .val_pairs
            .choose_multiple(
                &mut snap_rng,
                CANDLE_SNAPSHOT_WINDOWS.min(sampler.val_pairs.len()),
            )
            .copied()
            .collect()
    };

    if execution_mode == PretrainExecutionMode::EvaluateOnly {
        let input_model_path = start_weights
            .as_deref()
            .expect("evaluation-only weights validated before initialization");
        if args.objective == PretrainObjective::Lejepa {
            let input_heads_path =
                matching_pretrain_heads_path(input_model_path).ok_or_else(|| {
                    anyhow!("cannot derive pretrain heads path from input checkpoint")
                })?;
            let input_metadata_path = world_model_metadata_path(&input_heads_path);
            let verified = LejepaWorldModel::load(&input_heads_path, &input_metadata_path, device)
                .with_context(|| {
                    format!(
                        "evaluation-only input is not a complete compatible world model: {}",
                        input_heads_path.display()
                    )
                })?;
            drop(verified);
        }
        if args.eval_skill_only {
            let validation = evaluate_skill_panel(
                &heads,
                &mut sampler,
                SplitKind::Validation,
                args.batch_size,
                device,
            );
            println!(
                "pretrain skill-only validation ev_correct={:.9} ev_shuffled={:.9} ev_zero={:.9} sse_correct={:.9} sse_shuffled={:.9} sse_zero={:.9} sst={:.9} windows={} tickers={} rows={}",
                validation.ev_correct,
                validation.ev_shuffled,
                validation.ev_zero,
                validation.sse_correct,
                validation.sse_shuffled,
                validation.sse_zero,
                validation.sst,
                validation.windows,
                validation.tickers,
                validation.rows,
            );
            let test = if sampler.test_pairs.is_empty() {
                None
            } else {
                let metrics = evaluate_skill_panel(
                    &heads,
                    &mut sampler,
                    SplitKind::Test,
                    args.batch_size,
                    device,
                );
                println!(
                    "pretrain skill-only test ev_correct={:.9} ev_shuffled={:.9} ev_zero={:.9} sse_correct={:.9} sse_shuffled={:.9} sse_zero={:.9} sst={:.9} windows={} tickers={} rows={}",
                    metrics.ev_correct,
                    metrics.ev_shuffled,
                    metrics.ev_zero,
                    metrics.sse_correct,
                    metrics.sse_shuffled,
                    metrics.sse_zero,
                    metrics.sst,
                    metrics.windows,
                    metrics.tickers,
                    metrics.rows,
                );
                Some(metrics)
            };
            write_skill_panel_results(&run_dir, validation, test)?;
            println!(
                "Skill-only pretrain report written to {}",
                run_dir.root.display()
            );
            return Ok(());
        }
        let validation = validate_full(
            &model,
            &heads,
            &mut sampler,
            SplitKind::Validation,
            args.batch_size,
            None,
            args.objective,
            args.lambda_lat,
            args.lambda_sigreg,
            device,
            ValidationMode::Full,
        );
        print_step_eval_summary("validation-eval-only", 0, &validation);
        if args.objective == PretrainObjective::Lejepa {
            let deployed = deployed_cached_rollout_mse(
                &head_vs,
                &heads,
                &mut sampler,
                args.batch_size,
                &run_dir.weights,
                args.target_scale,
                device,
                SplitKind::Validation,
            )?;
            println!(
                "pretrain eval-only validation deployed_cached_rollout_mean_mse={deployed:.9} cache_contract={LEJEPA_CACHE_CONTRACT}"
            );
        }
        write_pretrain_diagnostics(
            &model,
            &heads,
            &mut sampler,
            args.batch_size,
            None,
            args.objective,
            0,
            0,
            &run_dir.gens,
            device,
            true,
        )?;
        if !sampler.test_pairs.is_empty() {
            let test = validate_full(
                &model,
                &heads,
                &mut sampler,
                SplitKind::Test,
                args.batch_size,
                None,
                args.objective,
                args.lambda_lat,
                args.lambda_sigreg,
                device,
                ValidationMode::Full,
            );
            print_step_eval_summary("test-eval-only", 0, &test);
            if args.objective == PretrainObjective::Lejepa {
                let deployed = deployed_cached_rollout_mse(
                    &head_vs,
                    &heads,
                    &mut sampler,
                    args.batch_size,
                    &run_dir.weights,
                    args.target_scale,
                    device,
                    SplitKind::Test,
                )?;
                println!(
                    "pretrain eval-only test deployed_cached_rollout_mean_mse={deployed:.9} cache_contract={LEJEPA_CACHE_CONTRACT}"
                );
            }
        }
        println!(
            "Evaluation-only pretrain reports written to {}",
            run_dir.root.display()
        );
        return Ok(());
    }

    'epoch_loop: for epoch in 1..=args.epochs {
        sampler.start_epoch();
        let mut train_epoch_loss = RunningLoss::new(device);
        let mut grad_norm_acc = GradNormAccum::default();
        let batches_per_epoch = sampler.batches_per_epoch(args.batch_size);
        println!(
            "pretrain epoch {epoch}/{} tickers={} batch_size={} batches_per_epoch={}",
            args.epochs,
            sampler.train_tickers.len(),
            args.batch_size,
            batches_per_epoch
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

            // Online probe: one optimizer step over detached real and predicted
            // token groups, using an optimizer disjoint from the world model.
            if args.objective == PretrainObjective::Lejepa && !losses.probe_groups.is_empty() {
                probe_step(
                    &heads,
                    &mut probe_opt,
                    &probe_named_vars,
                    &losses.probe_groups,
                    device,
                );
            }

            let total_v = losses.total.double_value(&[]);
            let jepa_mse_v = losses.jepa_mse.double_value(&[]);
            let sigreg_v = losses.sigreg.double_value(&[]);
            let repr_std_mean_v = losses.repr_std_mean.double_value(&[]);
            let repr_std_min_v = losses.repr_std_min.double_value(&[]);
            let pred_embed_std_v = losses.pred_embed_std.double_value(&[]);
            let target_embed_std_v = losses.target_embed_std.double_value(&[]);
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

            // Per-N-step validation mini-batch: same loss battery on one round-robin
            // validation batch, appended as val_* columns (empty on non-eval steps).
            let step_val = (args.step_val_every > 0 && global_step % args.step_val_every == 0)
                .then(|| {
                    tch::no_grad(|| {
                        sampler
                            .next_val_eval_batch(args.batch_size)
                            .map(|val_batch| {
                                let vl = pretrain_loss(
                                    &model,
                                    &heads,
                                    &val_batch,
                                    args.objective,
                                    args.lambda_lat,
                                    args.lambda_sigreg,
                                    args.target_scale,
                                    false,
                                );
                                (
                                    vl.total.double_value(&[]),
                                    vl.jepa_mse.double_value(&[]),
                                    vl.sigreg.double_value(&[]),
                                    vl.probe_mse.double_value(&[]),
                                    vl.probe_mae.double_value(&[]),
                                )
                            })
                    })
                })
                .flatten();
            step_loss_train.push(total_v as f32);
            step_loss_val.push(step_val.map(|(vt, ..)| vt as f32).unwrap_or(f32::NAN));

            if global_step == 1 || global_step % 20 == 0 {
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
                    SplitKind::Validation,
                    args.batch_size,
                    validation_batch_cap(args.validation_batches),
                    args.objective,
                    args.lambda_lat,
                    args.lambda_sigreg,
                    device,
                    ValidationMode::Fast,
                );
                let deployed_rollout_mean_mse = if args.objective == PretrainObjective::Lejepa {
                    deployed_cached_rollout_mse(
                        &head_vs,
                        &heads,
                        &mut sampler,
                        args.batch_size,
                        &run_dir.weights,
                        args.target_scale,
                        device,
                        SplitKind::Validation,
                    )?
                } else {
                    f64::NAN
                };
                print_step_eval_summary("validation", global_step, &val);
                // Periodic full-validation loss takes priority over the inline
                // step-val mini-batch in the step-loss overlay for the current step.
                if let Some(last) = step_loss_val.last_mut() {
                    *last = val.total as f32;
                }
                if args.objective == PretrainObjective::Lejepa {
                    println!(
                        "pretrain step {global_step} deployed_cached_rollout_mean_mse={deployed_rollout_mean_mse:.9} cache_contract={LEJEPA_CACHE_CONTRACT}"
                    );
                }
                if is_better_pretrain_checkpoint(
                    args.objective,
                    &val,
                    deployed_rollout_mean_mse,
                    best_val,
                    best_rollout_mean_mse,
                ) {
                    best_val = val.total;
                    best_rollout_mean_mse = deployed_rollout_mean_mse;
                    model_vs.save(&best_path)?;
                    save_pretrain_heads_checkpoint(
                        &head_vs,
                        &best_heads_path,
                        model.pretrain_latent_dim(),
                        args.target_scale,
                        args.objective,
                    )?;
                    println!("Saved best pretrained model: {}", best_path.display());
                }
            }

            if args.candle_snapshot_every > 0
                && global_step % args.candle_snapshot_every == 0
                && !candle_windows.is_empty()
            {
                write_candle_snapshots(
                    &heads,
                    &mut sampler,
                    &candle_windows,
                    epoch,
                    global_step,
                    &run_dir.gens,
                    &mut candle_rollout_mse_hist,
                    &mut candle_rollout_dclose_hist,
                )?;
            }

            if args.checkpoint_every > 0 && global_step % args.checkpoint_every == 0 {
                let path = pretrain_step_model_path(&run_dir.weights, global_step);
                let heads_path = pretrain_step_heads_path(&run_dir.weights, global_step);
                model_vs.save(&path)?;
                save_pretrain_heads_checkpoint(
                    &head_vs,
                    &heads_path,
                    model.pretrain_latent_dim(),
                    args.target_scale,
                    args.objective,
                )?;
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
        let final_epoch = stop_requested || epoch == args.epochs;
        let val = validate_full(
            &model,
            &heads,
            &mut sampler,
            SplitKind::Validation,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            args.objective,
            args.lambda_lat,
            args.lambda_sigreg,
            device,
            ValidationMode::Fast,
        );
        let deployed_rollout_mean_mse = if args.objective == PretrainObjective::Lejepa {
            deployed_cached_rollout_mse(
                &head_vs,
                &heads,
                &mut sampler,
                args.batch_size,
                &run_dir.weights,
                args.target_scale,
                device,
                SplitKind::Validation,
            )?
        } else {
            f64::NAN
        };
        print_validation_summary(&format!("pretrain epoch {epoch} validation"), &val);
        if args.objective == PretrainObjective::Lejepa {
            println!(
                "pretrain epoch {epoch} deployed_cached_rollout_mean_mse={deployed_rollout_mean_mse:.9} cache_contract={LEJEPA_CACHE_CONTRACT}"
            );
        }
        scalar_history.push(&train, &val);
        write_pretrain_scalar_meta_reports(&run_dir.gens, epoch, global_step, &scalar_history)?;
        write_pretrain_step_loss_report(
            &run_dir.gens,
            epoch,
            global_step,
            &step_loss_train,
            &step_loss_val,
        )?;
        if final_epoch {
            let final_validation = validate_full(
                &model,
                &heads,
                &mut sampler,
                SplitKind::Validation,
                args.batch_size,
                validation_batch_cap(args.validation_batches),
                args.objective,
                args.lambda_lat,
                args.lambda_sigreg,
                device,
                ValidationMode::Full,
            );
            print_step_eval_summary("final-validation", global_step, &final_validation);
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
                args.validation_batches == 0,
            )?;
        }

        // Held-out TEST battery at each epoch end (and on --steps early stop, since
        // the final epoch's end IS run end). Deep validation above stays untouched.
        if final_epoch && !sampler.test_pairs.is_empty() {
            let test = validate_full(
                &model,
                &heads,
                &mut sampler,
                SplitKind::Test,
                args.batch_size,
                validation_batch_cap(args.validation_batches),
                args.objective,
                args.lambda_lat,
                args.lambda_sigreg,
                device,
                ValidationMode::Full,
            );
            print_step_eval_summary("test", global_step, &test);
            write_pretrain_test_report(&run_dir.gens, epoch, global_step, &test)?;
        }

        if is_better_pretrain_checkpoint(
            args.objective,
            &val,
            deployed_rollout_mean_mse,
            best_val,
            best_rollout_mean_mse,
        ) {
            best_val = val.total;
            best_rollout_mean_mse = deployed_rollout_mean_mse;
            model_vs.save(&best_path)?;
            save_pretrain_heads_checkpoint(
                &head_vs,
                &best_heads_path,
                model.pretrain_latent_dim(),
                args.target_scale,
                args.objective,
            )?;
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
            SplitKind::Validation,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            args.objective,
            args.lambda_lat,
            args.lambda_sigreg,
            device,
            ValidationMode::Full,
        );
        best_val = val.total;
        model_vs.save(&best_path)?;
        save_pretrain_heads_checkpoint(
            &head_vs,
            &best_heads_path,
            model.pretrain_latent_dim(),
            args.target_scale,
            args.objective,
        )?;
        println!("Saved best pretrained model: {}", best_path.display());
    }

    if best_path.exists() {
        model_vs.load(&best_path)?;
    }
    if best_heads_path.exists() {
        head_vs.load(&best_heads_path)?;
    }
    model_vs.save(&final_path)?;
    save_pretrain_heads_checkpoint(
        &head_vs,
        &final_heads_path,
        model.pretrain_latent_dim(),
        args.target_scale,
        args.objective,
    )?;
    println!(
        "Saved final pretrained model: {} (best validation total_loss {:.6})",
        final_path.display(),
        best_val
    );
    Ok(())
}

fn load_matching_pretrain_heads(
    head_vs: &mut nn::VarStore,
    model_path: &Path,
    objective: PretrainObjective,
) -> Result<()> {
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
    if objective == PretrainObjective::Lejepa {
        let metadata_path = world_model_metadata_path(&heads_path);
        let metadata = WorldModelMetadata::load(&metadata_path).with_context(|| {
            format!(
                "LEJEPA warm-start requires compatible PoPE metadata {}",
                metadata_path.display()
            )
        })?;
        metadata.validate_checkpoint(&heads_path)?;
    }
    let load_summary =
        load_var_store_partial(head_vs, &heads_path).map_err(|err| anyhow!("{err}"))?;
    if objective == PretrainObjective::Lejepa {
        load_summary.require_complete().map_err(|error| {
            anyhow!(
                "LEJEPA warm-start {} is not a complete PoPE checkpoint: {error}",
                heads_path.display()
            )
        })?;
        println!("Loaded PoPE pretrain heads from {}", heads_path.display());
    } else if let Err(err) = load_summary.require_complete() {
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

fn save_pretrain_heads_checkpoint(
    head_vs: &nn::VarStore,
    checkpoint: &Path,
    latent_dim: i64,
    target_scale: f64,
    objective: PretrainObjective,
) -> Result<()> {
    head_vs.save(checkpoint)?;
    if matches!(objective, PretrainObjective::Lejepa) {
        WorldModelMetadata::save_for_checkpoint(checkpoint, latent_dim, target_scale)?;
    } else {
        let metadata_path = world_model_metadata_path(checkpoint);
        if metadata_path.exists() {
            fs::remove_file(metadata_path)?;
        }
    }
    Ok(())
}

fn deployed_cached_rollout_mse(
    head_vs: &nn::VarStore,
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    batch_size: usize,
    weights_dir: &Path,
    target_scale: f64,
    device: Device,
    split: SplitKind,
) -> Result<f64> {
    let split_pairs = match split {
        SplitKind::Train => &sampler.train_pairs,
        SplitKind::Validation => &sampler.val_pairs,
        SplitKind::Test => &sampler.test_pairs,
    };
    let samples = ticker_stratified_panel(split_pairs);
    if samples.is_empty() {
        return Err(anyhow!(
            "deployed rollout evaluation has no validation windows"
        ));
    }
    let checkpoint = weights_dir.join("pretrain_heads_promotion_candidate.ot");
    save_pretrain_heads_checkpoint(
        head_vs,
        &checkpoint,
        heads.latent_dim,
        target_scale,
        PretrainObjective::Lejepa,
    )?;
    let metadata_path = world_model_metadata_path(&checkpoint);
    let world_model = LejepaWorldModel::load(&checkpoint, &metadata_path, device)?;
    let mut squared_error_sum = 0.0;
    let mut elements = 0usize;
    for chunk in samples.chunks(batch_size) {
        let batch = PretrainSampler::batch_from_env_offsets(
            &mut sampler.train_envs,
            chunk,
            sampler.k_patches,
            sampler.patch_size,
            target_scale,
            device,
        );
        let prediction = world_model.predict(&batch.bar_history, LEJEPA_ROLLOUT_BARS)?;
        let actual = batch
            .next_bars
            .view([-1, LEJEPA_ROLLOUT_BARS, LEJEPA_BAR_FEATURES]);
        let squared_error = (&prediction.ohlc_mean - actual).square();
        squared_error_sum += squared_error.sum(Kind::Float).double_value(&[]);
        elements += squared_error.numel();
    }
    drop(world_model);
    fs::remove_file(&checkpoint)?;
    fs::remove_file(metadata_path)?;
    Ok(squared_error_sum / elements as f64)
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
            probe_groups: Vec::new(),
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
        probe_groups: Vec::new(),
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
    let target_bar_tokens = all_tokens.narrow(2, 1, length);
    let latest_token = all_tokens.select(2, length);
    let size = bar_tokens.size();
    let (batch_size, tickers, latent_dim) = (size[0], size[1], heads.latent_dim);
    let rows = batch_size * tickers * length;

    let belief = heads
        .predict_lejepa_bar_predictions(&bar_tokens, train)
        .belief;
    let belief_rows = belief.reshape([rows, latent_dim]);

    // Canonical LeWM objective: the next-token target remains attached so the
    // encoder is optimized end-to-end from both prediction branches. SIGReg is the
    // anti-collapse mechanism.
    let pred = heads.lejepa_predict_next(&belief_rows);
    let latent_target = target_bar_tokens.reshape([rows, latent_dim]);
    let latent_pred_loss = (&pred - &latent_target).square().mean(Kind::Float);
    let next_features = full.narrow(2, 1, length);

    let sigreg = sampled_sigreg_loss(&all_tokens, train);
    let (repr_std_mean, repr_std_min) = representation_std_metrics(&latest_token);
    let target_embed_std = target_bar_tokens.std(false);

    let total = &latent_pred_loss + &sigreg * lambda_sigreg;
    let jepa_mse = latent_pred_loss.detach();
    let pred_embed_std = pred.std(false).detach();

    let probe_target = scaled_next_ohlc_features(&batch.next_bars, target_scale);
    if train {
        update_probe_var_ema(
            &heads.feat_var_ema,
            &probe_target.reshape([-1, LEJEPA_BAR_FEATURES]),
        );
    }
    let (clean_latest_token, deterministic_endpoint) = tch::no_grad(|| {
        let inference_tokens = autocast(false, || heads.encode_bar_tokens(&full, false));
        let inference_bar_tokens = inference_tokens.narrow(2, 0, length);
        let inference_belief = heads
            .predict_lejepa_bar_predictions(&inference_bar_tokens, false)
            .belief
            .reshape([rows, latent_dim]);
        let endpoint = heads.lejepa_predict_next(&inference_belief);
        (inference_tokens.select(2, length).unsqueeze(2), endpoint)
    });

    let real_probe_input = clean_latest_token;
    let probe = ohlc_probe_metrics(heads, &real_probe_input, &probe_target);
    let pred_probe_target = next_features * target_scale;
    let probe_groups = vec![
        (real_probe_input, probe_target.shallow_clone()),
        (
            deterministic_endpoint.reshape([batch_size, tickers, length, latent_dim]),
            pred_probe_target,
        ),
    ];
    let zero_mse = probe_target.pow_tensor_scalar(2.0).mean(Kind::Float);
    let probe_explained_variance = explained_variance_tensor(&probe.probe_mse, &zero_mse);
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
        next_lat: latent_pred_loss.detach(),
        probe_terminal_mse: probe.probe_terminal_mse,
        zero_mse,
        probe_explained_variance,
        probe_groups,
    }
}

// In-place EMA update of a per-feature running variance buffer (no grad). `target`
// is [rows, features]; the buffer is [features].
fn update_probe_var_ema(ema: &Tensor, target: &Tensor) {
    tch::no_grad(|| {
        let mean = target.mean_dim([0i64].as_slice(), true, Kind::Float);
        let var = (target - &mean)
            .square()
            .mean_dim([0i64].as_slice(), false, Kind::Float);
        let updated = ema * LEJEPA_PROBE_VAR_EMA_DECAY + var * (1.0 - LEJEPA_PROBE_VAR_EMA_DECAY);
        ema.shallow_clone().copy_(&updated);
    });
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

fn sampled_sigreg_loss(tokens: &Tensor, train: bool) -> Tensor {
    let total_positions = tokens.size()[2];
    let k = LEJEPA_SIGREG_POSITIONS.min(total_positions);
    let sample_idx = if train {
        let last = Tensor::from_slice(&[total_positions - 1]).to_device(tokens.device());
        if k == 1 {
            last
        } else {
            let perm = Tensor::randperm(total_positions - 1, (Kind::Int64, tokens.device()));
            Tensor::cat(&[&perm.narrow(0, 0, k - 1), &last], 0)
        }
    } else {
        let idx = if k == 1 {
            vec![total_positions - 1]
        } else {
            (0..k)
                .map(|i| i * (total_positions - 1) / (k - 1))
                .collect::<Vec<_>>()
        };
        Tensor::from_slice(&idx).to_device(tokens.device())
    };
    let sigreg_tokens = tokens.index_select(2, &sample_idx);
    let batch_tickers = sigreg_tokens.size()[0] * sigreg_tokens.size()[1];
    sigreg_loss_impl(&sigreg_tokens.permute([2, 0, 1, 3]).contiguous().reshape([
        k,
        batch_tickers,
        tokens.size()[3],
    ]))
}

fn sigreg_loss_impl(tokens: &Tensor) -> Tensor {
    let size = tokens.size();
    let dim = size[2];
    let mut directions = Tensor::randn(
        [dim, LEJEPA_SIGREG_PROJECTIONS],
        (Kind::Float, tokens.device()),
    );
    directions = &directions / directions.norm_scalaropt_dim(2, [0i64].as_slice(), true);
    sigreg_loss_with_directions(tokens, &directions)
}

fn sigreg_loss_with_directions(tokens: &Tensor, directions: &Tensor) -> Tensor {
    let proj_in = tokens.to_kind(Kind::Float);
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
    weighted.mean(Kind::Float) * LEJEPA_SIGREG_REFERENCE_SAMPLES as f64
}

fn record_evaluated_tickers(evaluated_tickers: &mut HashSet<usize>, chunk: &[(usize, usize)]) {
    evaluated_tickers.extend(chunk.iter().map(|(env_idx, _)| *env_idx));
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

    let feat_var = heads.feat_var_ema.detach().clamp_min(1e-8);
    let mean_loss = (err.square() / &feat_var).mean(Kind::Float);
    let logvar_nll = (&logvar + err.square().detach() * logvar.neg().exp()).mean(Kind::Float) * 0.5;
    let probe_nll = mean_loss + logvar_nll;

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

#[derive(Default)]
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
    measurement: MeasurementMetrics,
    samples: usize,
    tickers: usize,
    batches: usize,
}

fn is_better_pretrain_checkpoint(
    objective: PretrainObjective,
    validation: &ValidationLoss,
    deployed_rollout_mean_mse: f64,
    best_total: f64,
    best_rollout_mean_mse: f64,
) -> bool {
    if !validation.total.is_finite() || validation.total >= best_total {
        return false;
    }
    match objective {
        PretrainObjective::MeanMse => true,
        PretrainObjective::Lejepa => {
            deployed_rollout_mean_mse.is_finite()
                && deployed_rollout_mean_mse < best_rollout_mean_mse
        }
    }
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
    // Detached (probe_input, target) groups for the online probe's single step.
    // Empty for objectives without an online probe.
    probe_groups: Vec<(Tensor, Tensor)>,
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
    train_total: Vec<f32>,
    eval_total: Vec<f32>,
    train_probe_bias: Vec<f32>,
    eval_probe_bias: Vec<f32>,
    train_pred_abs: Vec<f32>,
    eval_pred_abs: Vec<f32>,
    train_target_abs: Vec<f32>,
    eval_target_abs: Vec<f32>,
    train_zero_mse: Vec<f32>,
    eval_zero_mse: Vec<f32>,
    train_next_lat: Vec<f32>,
    eval_next_lat: Vec<f32>,
    eval_rollout_mean_mse: Vec<f32>,
    // Honest skill measurement (validation split, Lejepa only).
    eval_pred_probe_ev_mean: Vec<f32>,
    eval_latent_ev_marginal: Vec<f32>,
    eval_skill_belief_norm: Vec<f32>,
    eval_pred_probe_ev_persist: Vec<f32>,
    eval_pred_probe_diracc: Vec<f32>,
    eval_return_ev_persist: Vec<f32>,
    eval_return_diracc: Vec<f32>,
    eval_return_rank_ic: Vec<f32>,
    eval_latent_ev_persist: Vec<f32>,
    eval_skill_ev_correct: Vec<f32>,
    eval_skill_ev_shuffled: Vec<f32>,
    eval_skill_ev_zero: Vec<f32>,
    eval_skill_belief_spread: Vec<f32>,
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
        self.train_total.push(train.total as f32);
        self.eval_total.push(val.total as f32);
        self.train_probe_bias.push(train.probe_bias as f32);
        self.eval_probe_bias.push(val.probe_bias as f32);
        self.train_pred_abs.push(train.pred_abs as f32);
        self.eval_pred_abs.push(val.pred_abs as f32);
        self.train_target_abs.push(train.target_abs as f32);
        self.eval_target_abs.push(val.target_abs as f32);
        self.train_zero_mse.push(train.zero_mse as f32);
        self.eval_zero_mse.push(val.zero_mse as f32);
        self.train_next_lat.push(train.next_lat as f32);
        self.eval_next_lat.push(val.next_lat as f32);
        self.eval_rollout_mean_mse.push(val.rollout_mean_mse as f32);
        let m = &val.measurement;
        self.eval_pred_probe_ev_mean
            .push(m.pred_probe_ev_mean as f32);
        self.eval_latent_ev_marginal
            .push(m.latent_ev_marginal as f32);
        self.eval_skill_belief_norm.push(m.skill_belief_norm as f32);
        self.eval_pred_probe_ev_persist
            .push(m.pred_probe_ev_persist as f32);
        self.eval_pred_probe_diracc.push(m.pred_probe_diracc as f32);
        self.eval_return_ev_persist.push(m.return_ev_persist as f32);
        self.eval_return_diracc.push(m.return_diracc as f32);
        self.eval_return_rank_ic.push(m.return_rank_ic as f32);
        self.eval_latent_ev_persist.push(m.latent_ev_persist as f32);
        self.eval_skill_ev_correct.push(m.skill_ev_correct as f32);
        self.eval_skill_ev_shuffled.push(m.skill_ev_shuffled as f32);
        self.eval_skill_ev_zero.push(m.skill_ev_zero as f32);
        self.eval_skill_belief_spread
            .push(m.skill_belief_spread as f32);
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
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_pred_probe_ev_persist.report.bin"),
        format!("Pretrain Pred-Probe EV vs Persistence - epoch {epoch} step {global_step}"),
        "1 - probe SSE / persistence SSE",
        &history.eval_pred_probe_ev_persist,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_dir_acc_close.report.bin"),
        format!("Pretrain Pred-Probe Directional Accuracy - epoch {epoch} step {global_step}"),
        "sign-agreement fraction",
        &history.eval_pred_probe_diracc,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_return_ev_persist.report.bin"),
        format!("Pretrain Probe Return EV vs Persistence - epoch {epoch} step {global_step}"),
        "1 - probe return SSE / persistence SSE",
        &history.eval_return_ev_persist,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_return_dir_acc.report.bin"),
        format!("Pretrain Probe Return Directional Accuracy - epoch {epoch} step {global_step}"),
        "sign-agreement fraction",
        &history.eval_return_diracc,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_return_rank_ic.report.bin"),
        format!("Pretrain Probe Return Rank IC - epoch {epoch} step {global_step}"),
        "Spearman rank correlation",
        &history.eval_return_rank_ic,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_latent_ev_persist.report.bin"),
        format!("Pretrain Latent EV vs Persistence - epoch {epoch} step {global_step}"),
        "1 - latent SSE / latent-persistence SSE",
        &history.eval_latent_ev_persist,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_belief_spread.report.bin"),
        format!("Pretrain Belief Spread - epoch {epoch} step {global_step}"),
        "AR belief cross-batch spread",
        &history.eval_skill_belief_spread,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_total_loss.report.bin"),
        format!("Pretrain Total Loss - epoch {epoch} step {global_step}"),
        "total loss",
        &history.train_total,
        &history.eval_total,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_probe_bias.report.bin"),
        format!("Pretrain Probe Bias - epoch {epoch} step {global_step}"),
        "probe bias",
        &history.train_probe_bias,
        &history.eval_probe_bias,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_pred_abs.report.bin"),
        format!("Pretrain Probe Pred Abs - epoch {epoch} step {global_step}"),
        "probe pred abs mean",
        &history.train_pred_abs,
        &history.eval_pred_abs,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_target_abs.report.bin"),
        format!("Pretrain Probe Target Abs - epoch {epoch} step {global_step}"),
        "probe target abs mean",
        &history.train_target_abs,
        &history.eval_target_abs,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_zero_mse.report.bin"),
        format!("Pretrain Zero-Baseline MSE - epoch {epoch} step {global_step}"),
        "zero-baseline MSE",
        &history.train_zero_mse,
        &history.eval_zero_mse,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_next_lat.report.bin"),
        format!("Pretrain Next-Latent Loss - epoch {epoch} step {global_step}"),
        "next-latent loss",
        &history.train_next_lat,
        &history.eval_next_lat,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_rollout_mean_mse.report.bin"),
        format!("Pretrain Imagined Rollout MSE - epoch {epoch} step {global_step}"),
        "imagined rollout MSE",
        &history.eval_rollout_mean_mse,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_pred_probe_ev_mean.report.bin"),
        format!("Pretrain Pred-Probe EV vs Marginal - epoch {epoch} step {global_step}"),
        "1 - probe SSE / marginal SSE",
        &history.eval_pred_probe_ev_mean,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_latent_ev_marginal.report.bin"),
        format!("Pretrain Latent EV vs Marginal - epoch {epoch} step {global_step}"),
        "1 - latent SSE / marginal SSE",
        &history.eval_latent_ev_marginal,
    )?;
    write_pretrain_val_scalar_report(
        &epoch_dir.join("pretrain_belief_norm.report.bin"),
        format!("Pretrain Belief Norm - epoch {epoch} step {global_step}"),
        "AR belief L2 norm",
        &history.eval_skill_belief_norm,
    )?;
    write_report_file(
        &epoch_dir.join("pretrain_skill_ablation.report.bin"),
        &Report {
            title: format!("Pretrain Skill Ablation EV - epoch {epoch} step {global_step}"),
            x_label: Some("epoch".to_string()),
            y_label: Some("belief-conditioned EV".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "correct".to_string(),
                        values: history.eval_skill_ev_correct.clone(),
                    },
                    ReportSeries {
                        label: "shuffled".to_string(),
                        values: history.eval_skill_ev_shuffled.clone(),
                    },
                    ReportSeries {
                        label: "zero".to_string(),
                        values: history.eval_skill_ev_zero.clone(),
                    },
                ],
            },
        },
    )
}

fn write_pretrain_val_scalar_report(
    path: &Path,
    title: String,
    y_label: &str,
    values: &[f32],
) -> Result<()> {
    write_report_file(
        path,
        &Report {
            title,
            x_label: Some("epoch".to_string()),
            y_label: Some(y_label.to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![ReportSeries {
                    label: "val".to_string(),
                    values: values.to_vec(),
                }],
            },
        },
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

// Step-indexed train-vs-val total-loss curve. The val series is sparse (NaN on
// non-eval steps); the report renderer filters non-finite values.
fn write_pretrain_step_loss_report(
    gens_dir: &Path,
    epoch: usize,
    global_step: usize,
    train: &[f32],
    val: &[f32],
) -> Result<()> {
    let epoch_dir = gens_dir.join(epoch.to_string());
    fs::create_dir_all(&epoch_dir)?;
    write_report_file(
        &epoch_dir.join("pretrain_step_loss.report.bin"),
        &Report {
            title: format!("Pretrain Step Loss - epoch {epoch} step {global_step}"),
            x_label: Some("step".to_string()),
            y_label: Some("total loss".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "train".to_string(),
                        values: train.to_vec(),
                    },
                    ReportSeries {
                        label: "val".to_string(),
                        values: val.to_vec(),
                    },
                ],
            },
        },
    )
}

// Consolidated single-point held-out test battery (one series per non-bookkeeping
// metric), mirroring the single-point/multi-series `pretrain_skill_eval` idiom.
fn write_pretrain_test_report(
    gens_dir: &Path,
    epoch: usize,
    global_step: usize,
    test: &ValidationLoss,
) -> Result<()> {
    let epoch_dir = gens_dir.join(epoch.to_string());
    fs::create_dir_all(&epoch_dir)?;
    let m = &test.measurement;
    let series = [
        ("total", test.total),
        ("jepa_mse", test.jepa_mse),
        ("sigreg", test.sigreg),
        ("repr_std_mean", test.repr_std_mean),
        ("repr_std_min", test.repr_std_min),
        ("pred_embed_std", test.pred_embed_std),
        ("target_embed_std", test.target_embed_std),
        ("probe_mse", test.probe_mse),
        ("probe_mae", test.probe_mae),
        ("probe_bias", test.probe_bias),
        ("pred_abs", test.pred_abs),
        ("target_abs", test.target_abs),
        ("pred_std", test.pred_std),
        ("target_std", test.target_std),
        ("probe_terminal_mse", test.probe_terminal_mse),
        ("zero_mse", test.zero_mse),
        ("probe_explained_variance", test.probe_explained_variance),
        ("next_lat", test.next_lat),
        ("rollout_mean_mse", test.rollout_mean_mse),
        ("pred_probe_ev_mean", m.pred_probe_ev_mean),
        ("pred_probe_ev_persist", m.pred_probe_ev_persist),
        ("pred_probe_diracc", m.pred_probe_diracc),
        ("return_ev_persist", m.return_ev_persist),
        ("return_diracc", m.return_diracc),
        ("return_rank_ic", m.return_rank_ic),
        ("latent_ev_marginal", m.latent_ev_marginal),
        ("latent_ev_persist", m.latent_ev_persist),
        ("skill_ev_correct", m.skill_ev_correct),
        ("skill_ev_shuffled", m.skill_ev_shuffled),
        ("skill_ev_zero", m.skill_ev_zero),
        ("skill_belief_spread", m.skill_belief_spread),
        ("skill_belief_norm", m.skill_belief_norm),
    ]
    .into_iter()
    .map(|(label, value)| ReportSeries {
        label: label.to_string(),
        values: vec![value as f32],
    })
    .collect();
    write_report_file(
        &epoch_dir.join("pretrain_test.report.bin"),
        &Report {
            title: format!("Pretrain Held-out Test Battery - epoch {epoch} step {global_step}"),
            x_label: Some("metric".to_string()),
            y_label: Some("value".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine { series },
        },
    )
}

// Gaussian NLL of the probe's OHLC decode against `target`, the online probe's
// training signal. Mirrors the nll term in `ohlc_probe_metrics` without the extra
// diagnostic reductions, so it stays cheap over long pred_emb position dims.
// Probe training loss. The mean is fit with per-feature variance-standardized MSE
// so the tiny return features (idx 0-3) are weighted comparably to the large
// intra-bar shape features. The logvar is calibrated with a Gaussian NLL on the
// detached residual, keeping raw-scale variance for the planner.
fn probe_nll(heads: &PretrainHeads, belief: &Tensor, target: &Tensor) -> Tensor {
    let (mean, logvar) = heads.probe_ohlc_features(belief);
    let err = &mean - target;
    let feat_var = heads.feat_var_ema.detach().clamp_min(1e-8);
    let mean_loss = (err.square() / &feat_var).mean(Kind::Float);
    let logvar_nll = (&logvar + err.square().detach() * logvar.neg().exp()).mean(Kind::Float) * 0.5;
    mean_loss + logvar_nll
}

// One online probe optimizer step. Detachment at this boundary prevents any
// supervised probe gradient from reaching the world model.
fn probe_step(
    heads: &PretrainHeads,
    probe_opt: &mut Muon,
    probe_named_vars: &[(String, Tensor)],
    groups: &[(Tensor, Tensor)],
    device: Device,
) {
    let loss = groups
        .iter()
        .map(|(input, target)| probe_nll(heads, &input.detach(), &target.detach()))
        .reduce(|a, b| a + b)
        .expect("probe_step requires at least one group");
    probe_opt.zero_grad();
    loss.backward();
    clip_all_grads(probe_named_vars, MAX_GRAD_NORM, device);
    probe_opt.step();
}

fn print_step_eval_summary(kind: &str, global_step: usize, v: &ValidationLoss) {
    print_validation_summary(&format!("pretrain step {global_step} {kind}"), v);
}

fn print_validation_summary(label: &str, v: &ValidationLoss) {
    println!(
        "{label} total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_mse={:.6} probe_mae={:.6} probe_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} probe_terminal_mse={:.6} zero_mse={:.6} probe_ev={:.2}% next_lat={:.6} rollout_mean_mse={:.6} samples={} tickers={} batches={}",
        v.total,
        v.jepa_mse,
        v.sigreg,
        v.repr_std_mean,
        v.repr_std_min,
        v.pred_embed_std,
        v.target_embed_std,
        v.probe_mse,
        v.probe_mae,
        v.probe_bias,
        v.pred_abs,
        v.target_abs,
        v.pred_std,
        v.target_std,
        v.probe_terminal_mse,
        v.zero_mse,
        v.probe_explained_variance * 100.0,
        v.next_lat,
        v.rollout_mean_mse,
        v.samples,
        v.tickers,
        v.batches
    );
    let m = &v.measurement;
    println!(
        "{label} skill pred_probe_ev_mean={:.4} pred_probe_ev_persist={:.4} pred_probe_diracc={:.4} (n={:.0}) return_ev_persist={:.4} return_diracc={:.4} (n={:.0}) return_rank_ic={:.4} latent_ev_marginal={:.4} latent_ev_persist={:.4} belief_ev_correct={:.4} belief_ev_shuffled={:.4} belief_ev_zero={:.4} belief_spread={:.4} belief_norm={:.4} batches={}",
        m.pred_probe_ev_mean,
        m.pred_probe_ev_persist,
        m.pred_probe_diracc,
        m.pred_probe_diracc_n,
        m.return_ev_persist,
        m.return_diracc,
        m.return_diracc_n,
        m.return_rank_ic,
        m.latent_ev_marginal,
        m.latent_ev_persist,
        m.skill_ev_correct,
        m.skill_ev_shuffled,
        m.skill_ev_zero,
        m.skill_belief_spread,
        m.skill_belief_norm,
        m.skill_batches
    );
}

// Deterministic (temperature 0) imagined rollouts on a fixed set of validation
// windows. Writes a predicted-vs-actual CandleCompare report per window and
// accumulates the snapshot-indexed rollout MSE / decoded close-delta into running
// histories emitted as scalar reports.
fn write_candle_snapshots(
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    windows: &[(usize, usize)],
    epoch: usize,
    global_step: usize,
    gens_dir: &Path,
    mse_hist: &mut Vec<f32>,
    dclose_hist: &mut Vec<f32>,
) -> Result<()> {
    let target_scale = sampler.target_scale;
    let batch = sampler.batch_for_pairs(windows);
    tch::no_grad(|| -> Result<()> {
        let roll = heads.lejepa_imagined_rollout(&batch.bar_history) / target_scale;
        let actual = batch
            .next_bars
            .view([-1, LEJEPA_ROLLOUT_BARS, LEJEPA_BAR_FEATURES]);
        let rollout_mean_mse = (&roll - &actual)
            .pow_tensor_scalar(2.0)
            .mean_dim([1i64, 2].as_slice(), false, Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        let rollout_mean_dclose = roll.narrow(2, 3, 1).mean(Kind::Float).double_value(&[]);

        let pred_features = tensor_to_vec_f32(&roll)?;
        let actual_features = tensor_to_vec_f32(&actual)?;
        let stride = LEJEPA_ROLLOUT_BARS as usize * OHLC_BAR_FEATURES;
        let snapshot_dir = gens_dir.join(epoch.to_string()).join("candle_snapshots");
        fs::create_dir_all(&snapshot_dir)?;
        for (i, &(env_idx, offset)) in windows.iter().enumerate() {
            let env = &sampler.train_envs[env_idx];
            let seed = seed_candle_from_feature_row(&env.ohlc_features[env.ticker_perm[0]][offset]);
            let start = i * stride;
            let end = start + stride;
            let actual_candles =
                chained_candles_from_ohlc_features(&actual_features[start..end], &seed);
            let pred_candles =
                chained_candles_from_ohlc_features(&pred_features[start..end], &seed);
            write_report_file(
                &snapshot_dir.join(format!(
                    "step{global_step}_window{:02}_candles.report.bin",
                    i + 1
                )),
                &Report {
                    title: format!(
                        "Pretrain Candle Snapshot - step {global_step} - window {:02}",
                        i + 1
                    ),
                    x_label: Some("forecast bar".to_string()),
                    y_label: Some("relative price".to_string()),
                    scale: ScaleKind::Linear,
                    kind: ReportKind::CandleCompare {
                        actual: actual_candles,
                        predicted: pred_candles,
                    },
                },
            )?;
        }
        mse_hist.push(rollout_mean_mse as f32);
        dclose_hist.push(rollout_mean_dclose as f32);
        let epoch_dir = gens_dir.join(epoch.to_string());
        fs::create_dir_all(&epoch_dir)?;
        write_report_file(
            &epoch_dir.join("pretrain_candle_rollout_mse.report.bin"),
            &Report {
                title: format!("Pretrain Candle Rollout MSE - epoch {epoch} step {global_step}"),
                x_label: Some("snapshot".to_string()),
                y_label: Some("deterministic rollout MSE".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine {
                    series: vec![ReportSeries {
                        label: "mse".to_string(),
                        values: mse_hist.clone(),
                    }],
                },
            },
        )?;
        write_report_file(
            &epoch_dir.join("pretrain_candle_rollout_dclose.report.bin"),
            &Report {
                title: format!(
                    "Pretrain Candle Rollout Close Delta - epoch {epoch} step {global_step}"
                ),
                x_label: Some("snapshot".to_string()),
                y_label: Some("mean decoded close delta".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine {
                    series: vec![ReportSeries {
                        label: "dclose".to_string(),
                        values: dclose_hist.clone(),
                    }],
                },
            },
        )?;
        Ok(())
    })
}

fn validation_batch_cap(validation_batches: usize) -> Option<usize> {
    (validation_batches > 0).then_some(validation_batches)
}

fn ticker_stratified_panel(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut by_ticker: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for &(ticker, offset) in pairs {
        by_ticker.entry(ticker).or_default().push(offset);
    }
    by_ticker
        .into_iter()
        .map(|(ticker, mut offsets)| {
            offsets.sort_unstable();
            let median = offsets[offsets.len() / 2];
            (ticker, median)
        })
        .collect()
}

fn validation_pairs(split_pairs: &[(usize, usize)], mode: ValidationMode) -> Vec<(usize, usize)> {
    match mode {
        ValidationMode::Fast => ticker_stratified_panel(split_pairs),
        ValidationMode::Full => split_pairs.to_vec(),
    }
}

#[derive(Clone, Copy, Debug)]
struct SkillPanelMetrics {
    ev_correct: f64,
    ev_shuffled: f64,
    ev_zero: f64,
    sse_correct: f64,
    sse_shuffled: f64,
    sse_zero: f64,
    sst: f64,
    windows: usize,
    tickers: usize,
    rows: i64,
}

fn evaluate_skill_panel(
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    split: SplitKind,
    batch_size: usize,
    device: Device,
) -> SkillPanelMetrics {
    let split_pairs = match split {
        SplitKind::Train => &sampler.train_pairs,
        SplitKind::Validation => &sampler.val_pairs,
        SplitKind::Test => &sampler.test_pairs,
    };
    let panel = ticker_stratified_panel(split_pairs);
    assert!(!panel.is_empty(), "skill panel is empty");
    tch::no_grad(|| {
        let latent_dim = heads.latent_dim;
        let mut target_sum = Tensor::zeros([latent_dim], (Kind::Float, device));
        let mut target_square_sum = 0.0;
        let mut sse_correct = 0.0;
        let mut sse_shuffled = 0.0;
        let mut sse_zero = 0.0;
        let mut total_rows = 0i64;
        for chunk in panel.chunks(batch_size) {
            let batch = PretrainSampler::batch_from_env_offsets(
                &mut sampler.train_envs,
                chunk,
                sampler.k_patches,
                sampler.patch_size,
                sampler.target_scale,
                device,
            );
            let full = Tensor::cat(&[&batch.bar_history, &batch.next_bars.narrow(2, 0, 1)], 2);
            let tokens = autocast(false, || heads.encode_bar_tokens(&full, false));
            let length = batch.bar_history.size()[2];
            let source = tokens.narrow(2, 0, length);
            let target = tokens.narrow(2, 1, length).reshape([-1, latent_dim]);
            let belief_sequence = heads.predict_lejepa_bar_predictions(&source, false).belief;
            let belief = belief_sequence.reshape([-1, latent_dim]);
            let shuffled_belief = belief_sequence.flip([2]).reshape([-1, latent_dim]);
            let correct = heads.lejepa_predict_next(&belief);
            let shuffled = heads.lejepa_predict_next(&shuffled_belief);
            let zero = heads.lejepa_predict_next(&Tensor::zeros_like(&belief));
            sse_correct += (&correct - &target)
                .square()
                .sum(Kind::Float)
                .double_value(&[]);
            sse_shuffled += (&shuffled - &target)
                .square()
                .sum(Kind::Float)
                .double_value(&[]);
            sse_zero += (&zero - &target)
                .square()
                .sum(Kind::Float)
                .double_value(&[]);
            target_sum += target.sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
            target_square_sum += target.square().sum(Kind::Float).double_value(&[]);
            total_rows += target.size()[0];
        }
        let sst = target_square_sum
            - target_sum.square().sum(Kind::Float).double_value(&[]) / total_rows as f64;
        let ev = |sse: f64| if sst > 1e-12 { 1.0 - sse / sst } else { 0.0 };
        SkillPanelMetrics {
            ev_correct: ev(sse_correct),
            ev_shuffled: ev(sse_shuffled),
            ev_zero: ev(sse_zero),
            sse_correct,
            sse_shuffled,
            sse_zero,
            sst,
            windows: panel.len(),
            tickers: panel.len(),
            rows: total_rows,
        }
    })
}

fn write_skill_panel_results(
    run_dir: &RunDir,
    validation: SkillPanelMetrics,
    test: Option<SkillPanelMetrics>,
) -> Result<()> {
    let mut series = vec![
        ReportSeries {
            label: "validation correct".to_owned(),
            values: vec![validation.ev_correct as f32],
        },
        ReportSeries {
            label: "validation shuffled".to_owned(),
            values: vec![validation.ev_shuffled as f32],
        },
        ReportSeries {
            label: "validation zero".to_owned(),
            values: vec![validation.ev_zero as f32],
        },
    ];
    if let Some(test) = test {
        series.extend([
            ReportSeries {
                label: "test correct".to_owned(),
                values: vec![test.ev_correct as f32],
            },
            ReportSeries {
                label: "test shuffled".to_owned(),
                values: vec![test.ev_shuffled as f32],
            },
            ReportSeries {
                label: "test zero".to_owned(),
                values: vec![test.ev_zero as f32],
            },
        ]);
    }
    let report_dir = run_dir.gens.join("0");
    fs::create_dir_all(&report_dir)?;
    write_report_file(
        &report_dir.join("pretrain_skill_eval.report.bin"),
        &Report {
            title: "Pretrain Latent Skill Evaluation".to_owned(),
            x_label: Some("panel".to_owned()),
            y_label: Some("global explained variance".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine { series },
        },
    )
}

// Honest skill measurement for the JEPA world model, accumulated over validation
// batches. All baselines are corrected: feature EV is standardized per feature and
// measured against persistence (previous bar) and the unconditional mean; latent EV
// is measured against latent-persistence, not just the marginal variance.
#[derive(Default)]
struct MeasurementMetrics {
    // Prediction-probe (probe of the predicted next latent vs actual next features).
    pred_probe_ev_mean: f64,
    pred_probe_ev_persist: f64,
    pred_probe_diracc: f64,
    pred_probe_diracc_n: f64,
    // Inter-bar return diagnostics decoded from the detached OHLC probe (idx 0-3).
    return_ev_persist: f64,
    return_diracc: f64,
    return_diracc_n: f64,
    return_rank_ic: f64,
    // Latent space.
    latent_ev_marginal: f64,
    latent_ev_persist: f64,
    // Belief ablation (does the predictor use the AR belief?).
    skill_ev_correct: f64,
    skill_ev_shuffled: f64,
    skill_ev_zero: f64,
    skill_belief_spread: f64,
    skill_belief_norm: f64,
    skill_batches: usize,
}

struct MeasurementAccum {
    pred_probe_sse: Tensor,
    persist_sse: Tensor,
    actual_sum: Tensor,
    actual_sqsum: Tensor,
    rows: i64,
    probe_dir_correct: f64,
    probe_dir_n: f64,
    rank_ic_sum: f64,
    rank_ic_batches: f64,
    latent_target_sum: Tensor,
    latent_target_sqsum: f64,
    latent_pred_sse: f64,
    latent_persist_sse: f64,
    skill_ev_correct_sum: f64,
    skill_ev_shuffled_sum: f64,
    skill_ev_zero_sum: f64,
    skill_belief_spread_sum: f64,
    skill_belief_norm_sum: f64,
    skill_batches: usize,
    latent_dim: i64,
}

impl MeasurementAccum {
    fn new(latent_dim: i64, device: Device) -> Self {
        let feats = LEJEPA_BAR_FEATURES;
        Self {
            pred_probe_sse: Tensor::zeros([feats], (Kind::Float, device)),
            persist_sse: Tensor::zeros([feats], (Kind::Float, device)),
            actual_sum: Tensor::zeros([feats], (Kind::Float, device)),
            actual_sqsum: Tensor::zeros([feats], (Kind::Float, device)),
            rows: 0,
            probe_dir_correct: 0.0,
            probe_dir_n: 0.0,
            rank_ic_sum: 0.0,
            rank_ic_batches: 0.0,
            latent_target_sum: Tensor::zeros([latent_dim], (Kind::Float, device)),
            latent_target_sqsum: 0.0,
            latent_pred_sse: 0.0,
            latent_persist_sse: 0.0,
            skill_ev_correct_sum: 0.0,
            skill_ev_shuffled_sum: 0.0,
            skill_ev_zero_sum: 0.0,
            skill_belief_spread_sum: 0.0,
            skill_belief_norm_sum: 0.0,
            skill_batches: 0,
            latent_dim,
        }
    }

    fn add(&mut self, heads: &PretrainHeads, batch: &PretrainBatch, target_scale: f64) {
        let latent_dim = self.latent_dim;
        let full = Tensor::cat(&[&batch.bar_history, &batch.next_bars.narrow(2, 0, 1)], 2);
        let tokens = autocast(false, || heads.encode_bar_tokens(&full, false));
        let length = batch.bar_history.size()[2];
        let batch_size = tokens.size()[0];
        let tickers = tokens.size()[1];
        let bar_tokens = tokens.narrow(2, 0, length);
        let target_tokens = tokens.narrow(2, 1, length).reshape([-1, latent_dim]);
        let belief = heads
            .predict_lejepa_bar_predictions(&bar_tokens, false)
            .belief
            .reshape([-1, latent_dim]);
        let rows = belief.size()[0];
        let pred = heads.lejepa_predict_next(&belief);

        // Latent space: prediction vs marginal and latent-persistence baselines.
        self.latent_pred_sse += (&pred - &target_tokens)
            .square()
            .sum(Kind::Float)
            .double_value(&[]);
        let cur_tokens = bar_tokens.reshape([-1, latent_dim]);
        self.latent_persist_sse += (&cur_tokens - &target_tokens)
            .square()
            .sum(Kind::Float)
            .double_value(&[]);
        self.latent_target_sum +=
            target_tokens.sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        self.latent_target_sqsum += target_tokens.square().sum(Kind::Float).double_value(&[]);

        // Feature space (raw units).
        let actual_feats = full.narrow(2, 1, length).reshape([-1, LEJEPA_BAR_FEATURES]);
        let prev_feats = full.narrow(2, 0, length).reshape([-1, LEJEPA_BAR_FEATURES]);
        let pred_token = pred.reshape([batch_size, tickers, length, latent_dim]);
        let (probe_mean, _) = heads.probe_ohlc_features(&pred_token);
        let probe_feats = (probe_mean / target_scale).reshape([-1, LEJEPA_BAR_FEATURES]);

        self.pred_probe_sse += (&probe_feats - &actual_feats).square().sum_dim_intlist(
            [0i64].as_slice(),
            false,
            Kind::Float,
        );
        self.persist_sse += (&prev_feats - &actual_feats).square().sum_dim_intlist(
            [0i64].as_slice(),
            false,
            Kind::Float,
        );
        self.actual_sum += actual_feats.sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        self.actual_sqsum +=
            actual_feats
                .square()
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        self.rows += rows;

        // Directional accuracy on the close return (idx 3), ties (pred==0) excluded.
        let actual3 = actual_feats.select(1, 3);
        let (pc, pn) = directional_accuracy(&probe_feats.select(1, 3), &actual3);
        self.probe_dir_correct += pc;
        self.probe_dir_n += pn;

        // Cross-sectional rank-IC on the close return (idx 3): at each (window, lag)
        // rank the tickers against one another and correlate with the realized
        // ranking, averaging over all (window, lag) points. This requires a real
        // cross-section (TICKERS_COUNT > 1); with the single-ticker universe the
        // batch axis is a time-anchor mix of unrelated windows, not a cross-section,
        // so the metric stays dormant and reports NaN rather than a fabricated
        // pooled number.
        if tickers >= 2 {
            let return3_wt = probe_feats
                .select(1, 3)
                .reshape([batch_size, tickers, length]);
            let actual3_wt = actual3.reshape([batch_size, tickers, length]);
            for b in 0..batch_size {
                let pred_window = return3_wt.select(0, b);
                let actual_window = actual3_wt.select(0, b);
                for t in 0..length {
                    self.rank_ic_sum +=
                        rank_ic(&pred_window.select(1, t), &actual_window.select(1, t));
                    self.rank_ic_batches += 1.0;
                }
            }
        }

        // Belief ablation: correct vs row-shuffled vs zero context, EV vs marginal.
        let target_mean = target_tokens.mean_dim([0i64].as_slice(), true, Kind::Float);
        let marginal = (&target_tokens - &target_mean)
            .square()
            .mean(Kind::Float)
            .double_value(&[])
            .max(1e-12);
        let ev = |ctx: &Tensor| -> f64 {
            let est = heads.lejepa_predict_next(ctx);
            let mse = est
                .mse_loss(&target_tokens, Reduction::Mean)
                .double_value(&[]);
            1.0 - mse / marginal
        };
        let perm = Tensor::randperm(rows, (Kind::Int64, belief.device()));
        self.skill_ev_correct_sum += ev(&belief);
        self.skill_ev_shuffled_sum += ev(&belief.index_select(0, &perm));
        self.skill_ev_zero_sum += ev(&Tensor::zeros_like(&belief));
        let belief_mean = belief.mean_dim([0i64].as_slice(), true, Kind::Float);
        self.skill_belief_spread_sum += (&belief - &belief_mean)
            .square()
            .mean_dim([0i64].as_slice(), false, Kind::Float)
            .sqrt()
            .mean(Kind::Float)
            .double_value(&[]);
        self.skill_belief_norm_sum += belief
            .norm_scalaropt_dim(2, [-1i64].as_slice(), false)
            .mean(Kind::Float)
            .double_value(&[]);
        self.skill_batches += 1;
    }

    fn finish(self) -> MeasurementMetrics {
        if self.skill_batches == 0 {
            return MeasurementMetrics::default();
        }
        let n = self.rows as f64;
        let feat_var = &self.actual_sqsum / n - (&self.actual_sum / n).square();
        let feat_var = feat_var.clamp_min(1e-12);
        let ret_idx = INTER_BAR_FEATURES;
        // Standardized aggregate over the return-feature group (idx 0-3): each
        // feature's SSE is divided by its variance, so equal weight regardless of
        // scale. Mean baseline SSE per feature equals its variance, hence group size.
        let std_group = |sse: &Tensor| -> f64 {
            (sse.narrow(0, 0, ret_idx) / n / feat_var.narrow(0, 0, ret_idx))
                .sum(Kind::Float)
                .double_value(&[])
        };
        let probe_num = std_group(&self.pred_probe_sse);
        let persist_den = std_group(&self.persist_sse).max(1e-12);
        let group = ret_idx as f64;
        let marginal_sse = self.latent_target_sqsum
            - self
                .latent_target_sum
                .square()
                .sum(Kind::Float)
                .double_value(&[])
                / n;
        let batches = self.skill_batches as f64;
        MeasurementMetrics {
            pred_probe_ev_mean: 1.0 - probe_num / group,
            pred_probe_ev_persist: 1.0 - probe_num / persist_den,
            pred_probe_diracc: safe_ratio(self.probe_dir_correct, self.probe_dir_n),
            pred_probe_diracc_n: self.probe_dir_n,
            return_ev_persist: 1.0 - probe_num / persist_den,
            return_diracc: safe_ratio(self.probe_dir_correct, self.probe_dir_n),
            return_diracc_n: self.probe_dir_n,
            return_rank_ic: safe_ratio(self.rank_ic_sum, self.rank_ic_batches),
            latent_ev_marginal: 1.0 - self.latent_pred_sse / marginal_sse.max(1e-12),
            latent_ev_persist: 1.0 - self.latent_pred_sse / self.latent_persist_sse.max(1e-12),
            skill_ev_correct: self.skill_ev_correct_sum / batches,
            skill_ev_shuffled: self.skill_ev_shuffled_sum / batches,
            skill_ev_zero: self.skill_ev_zero_sum / batches,
            skill_belief_spread: self.skill_belief_spread_sum / batches,
            skill_belief_norm: self.skill_belief_norm_sum / batches,
            skill_batches: self.skill_batches,
        }
    }
}

fn safe_ratio(correct: f64, total: f64) -> f64 {
    if total > 0.0 {
        correct / total
    } else {
        f64::NAN
    }
}

// Sign-agreement accuracy on paired predictions/targets. Excludes ties (either side
// exactly zero). Returns (correct, counted).
fn directional_accuracy(pred: &Tensor, actual: &Tensor) -> (f64, f64) {
    let counted = pred.sign().abs() * actual.sign().abs();
    let agree = (pred.sign() * actual.sign()).clamp_min(0.0) * &counted;
    (
        agree.sum(Kind::Float).double_value(&[]),
        counted.sum(Kind::Float).double_value(&[]),
    )
}

// Spearman rank correlation (rank-IC) between two flat vectors.
fn rank_ic(pred: &Tensor, actual: &Tensor) -> f64 {
    let n = pred.size()[0];
    if n < 2 {
        return 0.0;
    }
    let rank = |t: &Tensor| t.argsort(0, false).argsort(0, false).to_kind(Kind::Float);
    let pr = rank(pred);
    let ar = rank(actual);
    let pm = pr.mean(Kind::Float);
    let am = ar.mean(Kind::Float);
    let pc = &pr - &pm;
    let ac = &ar - &am;
    let cov = (&pc * &ac).sum(Kind::Float).double_value(&[]);
    let denom = (pc.square().sum(Kind::Float).double_value(&[])
        * ac.square().sum(Kind::Float).double_value(&[]))
    .sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

fn validate_full(
    model: &TradingModel,
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    split: SplitKind,
    batch_size: usize,
    max_batches: Option<usize>,
    objective: PretrainObjective,
    lambda_lat: f64,
    lambda_sigreg: f64,
    device: Device,
    mode: ValidationMode,
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
        // Honest skill measurement (corrected baselines), accumulated over every
        // Lejepa validation batch in both Fast and Full modes.
        let mut measurement = MeasurementAccum::new(heads.latent_dim, device);
        let mut samples = 0usize;
        let mut batches = 0usize;
        let mut rollout_ctx: Vec<Tensor> = Vec::new();
        let mut rollout_actual: Vec<Tensor> = Vec::new();
        let mut rollout_windows = 0usize;

        let target_scale = sampler.target_scale;
        let split_pairs = match split {
            SplitKind::Train => sampler.train_pairs.clone(),
            SplitKind::Validation => sampler.val_pairs.clone(),
            SplitKind::Test => sampler.test_pairs.clone(),
        };
        let full_window_count = split_pairs.len();
        let pairs = validation_pairs(&split_pairs, mode);
        if mode == ValidationMode::Fast {
            let full_batches = full_window_count.div_ceil(batch_size);
            let fast_batches = pairs.len().div_ceil(batch_size);
            println!(
                "fast validation panel: tickers={} windows={} batches={} full_windows={} full_batches={} sampled_euler_rollouts=0",
                pairs.len(),
                pairs.len(),
                fast_batches,
                full_window_count,
                full_batches,
            );
        }
        let rollout_window_limit = if mode == ValidationMode::Full && max_batches.is_none() {
            pairs.len()
        } else {
            LEJEPA_ROLLOUT_EVAL_WINDOWS.min(pairs.len())
        };
        let mut evaluated_tickers = HashSet::new();

        for chunk in pairs.chunks(batch_size) {
            if mode == ValidationMode::Full && max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }
            record_evaluated_tickers(&mut evaluated_tickers, chunk);

            let batch = PretrainSampler::batch_from_env_offsets(
                &mut sampler.train_envs,
                chunk,
                sampler.k_patches,
                sampler.patch_size,
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
            pred_embed_std_sum += losses.pred_embed_std.double_value(&[]) * batch_samples as f64;
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

            if mode == ValidationMode::Full
                && matches!(objective, PretrainObjective::Lejepa)
                && rollout_windows < rollout_window_limit
            {
                let take = batch_samples.min(rollout_window_limit - rollout_windows);
                rollout_ctx.push(batch.bar_history.narrow(0, 0, take as i64));
                rollout_actual.push(batch.next_bars.narrow(0, 0, take as i64));
                rollout_windows += take;
            }

            // Honest skill measurement over corrected baselines (both modes).
            if matches!(objective, PretrainObjective::Lejepa) {
                measurement.add(heads, &batch, target_scale);
            }
        }

        let measurement = measurement.finish();

        // Deterministic autoregressive rollout MSE (Full validation only).
        let rollout_mean_mse = match objective {
            PretrainObjective::Lejepa if !rollout_ctx.is_empty() => {
                let ctx = Tensor::cat(&rollout_ctx, 0);
                let actual = Tensor::cat(&rollout_actual, 0).view([
                    -1,
                    LEJEPA_ROLLOUT_BARS,
                    LEJEPA_BAR_FEATURES,
                ]);
                let n_total = ctx.size()[0];
                let chunk = batch_size as i64;
                let mut squared_error_sum = 0.0f64;
                let mut elements = 0i64;
                let mut start = 0;
                while start < n_total {
                    let len = chunk.min(n_total - start);
                    let ctx_c = ctx.narrow(0, start, len);
                    let actual_c = actual.narrow(0, start, len);
                    let roll = heads.lejepa_imagined_rollout(&ctx_c) / target_scale;
                    let squared = (&roll - &actual_c).square();
                    squared_error_sum += squared.sum(Kind::Float).double_value(&[]);
                    elements += squared.numel() as i64;
                    start += len;
                }
                squared_error_sum / elements as f64
            }
            // NaN = not-applicable (MeanMse has no imagined rollout; Fast skips it).
            _ => f64::NAN,
        };
        let _ = rollout_windows;

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
            measurement,
            samples,
            tickers: evaluated_tickers.len(),
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
    predicted: Vec<CandleBar>,
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
    panel_only: bool,
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
            let offsets = if panel_only {
                vec![offsets[offsets.len() / 2]]
            } else {
                offsets
            };

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
                        let (imagined, entropy) =
                            heads.lejepa_imagined_rollout_inner(&batch.bar_history, true);
                        if let Some(e) = entropy {
                            ent_mstep += e.mean_step_std;
                            ent_tnmean += e.tok_norm_mean;
                            ent_tnmax = ent_tnmax.max(e.tok_norm_max);
                            ent_n += 1;
                        }
                        let prediction = imagined / target_scale;
                        (
                            Vec::new(),
                            Vec::new(),
                            Some(tensor_to_vec_f32(&prediction)?),
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
                                variant_traces.push(VariantCandles {
                                    label: format!("sample_{:02}", variant_traces.len() + 1),
                                    actual: chained_candles_from_ohlc_features(
                                        actual_features,
                                        &seed,
                                    ),
                                    predicted: chained_candles_from_ohlc_features(
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
            "predicted",
            epoch,
            global_step,
            &vt.actual,
            &vt.predicted,
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

fn normalize_last_dim(x: &Tensor) -> Tensor {
    let mean = x.mean_dim([-1].as_slice(), true, Kind::Float);
    let centered = x - &mean;
    let var = centered
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    centered / (var + LEJEPA_NORMALIZATION_EPS).sqrt()
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
// matters: the per-bar encoder/projector is checked first, leaving the AR
// transformer and predictor head as the remaining `lejepa_` params.
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
        chained_candles_from_ohlc_features, cumulative_future_returns, enforce_single_sampler_pass,
        future_patches_for_current_perm, next_bars_for_current_perm, record_evaluated_tickers,
        retain_spaced_train_pairs, save_pretrain_heads_checkpoint, seed_candle_from_feature_row,
        sigreg_loss_with_directions, ticker_stratified_panel, validation_pairs, CandleBar,
        PretrainArgs, PretrainExecutionMode, PretrainHeads, PretrainObjective, PretrainSampler,
        SplitKind, ValidationMode, LEJEPA_AR_LAYERS, LEJEPA_BAR_FEATURES, LEJEPA_HEADS,
        LEJEPA_HEAD_DIM, LEJEPA_ROLLOUT_BARS, LEJEPA_SIGREG_PROJECTIONS,
        TRAIN_ANCHOR_STRIDE_MULTIPLIER,
    };
    use crate::torch::{
        constants::PRICE_DELTAS_PER_TICKER,
        env::{build_ohlc_features, Env, OHLC_BAR_FEATURES},
        model::{ModelVariant, TradingModel, TradingModelConfig},
    };
    use tch::nn;
    use tch::{Kind, Tensor};

    #[test]
    fn only_lejepa_checkpoints_emit_world_model_metadata() {
        let temp_dir = std::env::temp_dir().join(format!(
            "trading-bot-pretrain-metadata-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let mean_checkpoint = temp_dir.join("mean.ot");
        let lejepa_checkpoint = temp_dir.join("lejepa.ot");
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let _weight = vs.root().var("weight", &[1], nn::Init::Const(1.0));

        std::fs::write(
            crate::torch::world_model::world_model_metadata_path(&mean_checkpoint),
            b"stale metadata",
        )
        .unwrap();

        save_pretrain_heads_checkpoint(
            &vs,
            &mean_checkpoint,
            256,
            100.0,
            PretrainObjective::MeanMse,
        )
        .unwrap();
        assert!(!crate::torch::world_model::world_model_metadata_path(&mean_checkpoint).exists());

        save_pretrain_heads_checkpoint(
            &vs,
            &lejepa_checkpoint,
            256,
            100.0,
            PretrainObjective::Lejepa,
        )
        .unwrap();
        assert!(crate::torch::world_model::world_model_metadata_path(&lejepa_checkpoint).exists());
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn zero_steps_reaches_eval_only_mode_only_with_weights() {
        let mut args = PretrainArgs {
            weights: Some("checkpoint.ot".to_owned()),
            model_size: ModelVariant::UniformStream,
            run: Some("eval-only-test".to_owned()),
            epochs: 1,
            steps: Some(0),
            eval_skill_only: false,
            batch_size: 8,
            k_patches: 16,
            objective: PretrainObjective::Lejepa,
            lambda_lat: 0.0,
            lambda_sigreg: 0.1,
            target_scale: 100.0,
            validation_batches: 0,
            validate_every: 0,
            checkpoint_every: 0,
            step_val_every: 0,
            candle_snapshot_every: 0,
        };
        assert_eq!(
            super::pretrain_execution_mode(&args).unwrap(),
            PretrainExecutionMode::EvaluateOnly
        );
        args.eval_skill_only = true;
        assert_eq!(
            super::pretrain_execution_mode(&args).unwrap(),
            PretrainExecutionMode::EvaluateOnly
        );
        args.objective = PretrainObjective::MeanMse;
        assert!(super::pretrain_execution_mode(&args).is_err());
        args.objective = PretrainObjective::Lejepa;
        args.weights = None;
        assert!(super::pretrain_execution_mode(&args).is_err());
        args.weights = Some("checkpoint.ot".to_owned());
        args.steps = Some(1);
        assert!(super::pretrain_execution_mode(&args).is_err());
        args.eval_skill_only = false;
        assert_eq!(
            super::pretrain_execution_mode(&args).unwrap(),
            PretrainExecutionMode::Train
        );
    }

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
        let mut env = Env::new_without_macro_for_test(false);
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
        let mut env = Env::new_without_macro_for_test(false);
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
        let mut env = Env::new_without_macro_for_test(false);
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
        // The most-future TEST split now reaches the last forecast-safe anchor.
        let data_len = PRICE_DELTAS_PER_TICKER + 801;
        let offsets = build_split_offsets(data_len, 16, 25, SplitKind::Test);
        let last = *offsets.last().expect("test offsets should be non-empty");
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
    fn three_way_split_is_ordered_disjoint_and_aligned() {
        use std::collections::HashSet;
        let data_len = PRICE_DELTAS_PER_TICKER + 10_000;
        let (k, ps) = (16usize, 25usize);
        let train = build_split_offsets(data_len, k, ps, SplitKind::Train);
        let val = build_split_offsets(data_len, k, ps, SplitKind::Validation);
        let test = build_split_offsets(data_len, k, ps, SplitKind::Test);
        assert!(!train.is_empty() && !val.is_empty() && !test.is_empty());

        // Each split is a contiguous patch-aligned stride from the shared origin.
        for offsets in [&train, &val, &test] {
            for pair in offsets.windows(2) {
                assert_eq!(pair[1] - pair[0], ps, "offsets must step by patch_size");
            }
            for &o in offsets.iter() {
                assert_eq!(
                    (o - PRICE_DELTAS_PER_TICKER) % ps,
                    0,
                    "offset must be aligned to the patch stride"
                );
            }
        }

        // Chronological order train < val < test, with per-split target margins
        // keeping each split's forecast targets out of the next split's contexts.
        assert!(*train.last().unwrap() < *val.first().unwrap());
        assert!(*val.last().unwrap() < *test.first().unwrap());
        assert!(train.last().unwrap() + k * ps <= *val.first().unwrap());
        assert!(val.last().unwrap() + k * ps <= *test.first().unwrap());

        // Fully disjoint anchor sets.
        let tset: HashSet<_> = train.iter().collect();
        let vset: HashSet<_> = val.iter().collect();
        let eset: HashSet<_> = test.iter().collect();
        assert!(tset.is_disjoint(&vset));
        assert!(vset.is_disjoint(&eset));
        assert!(tset.is_disjoint(&eset));
    }

    #[test]
    fn train_pair_thinning_restarts_per_ticker_and_preserves_input_order() {
        let dense = vec![
            (0, 100),
            (1, 300),
            (0, 125),
            (1, 325),
            (0, 150),
            (1, 350),
            (0, 175),
            (1, 375),
            (0, 200),
            (1, 400),
        ];
        assert_eq!(
            retain_spaced_train_pairs(dense),
            vec![(0, 100), (1, 300), (0, 200), (1, 400)]
        );
    }

    #[test]
    fn retained_train_anchors_are_one_hundred_bars_apart_per_ticker() {
        let patch_size = 25usize;
        let offsets = build_split_offsets(
            PRICE_DELTAS_PER_TICKER + 20_000,
            16,
            patch_size,
            SplitKind::Train,
        );
        let pairs = [0usize, 1]
            .into_iter()
            .flat_map(|ticker| offsets.iter().copied().map(move |offset| (ticker, offset)))
            .collect();
        let retained = retain_spaced_train_pairs(pairs);

        for ticker in [0usize, 1] {
            let ticker_offsets: Vec<_> = retained
                .iter()
                .filter_map(|&(pair_ticker, offset)| (pair_ticker == ticker).then_some(offset))
                .collect();
            assert_eq!(ticker_offsets.first(), offsets.first());
            for adjacent in ticker_offsets.windows(2) {
                assert_eq!(
                    adjacent[1] - adjacent[0],
                    patch_size * TRAIN_ANCHOR_STRIDE_MULTIPLIER
                );
            }
            assert_eq!(
                ticker_offsets.len(),
                offsets.len().div_ceil(TRAIN_ANCHOR_STRIDE_MULTIPLIER)
            );
        }
    }

    #[test]
    fn one_retained_sampler_pass_caps_raw_row_reuse_at_sixty_one() {
        let first_anchor = PRICE_DELTAS_PER_TICKER;
        let dense = (first_anchor..=first_anchor + 12_000)
            .step_by(25)
            .map(|offset| (0usize, offset))
            .collect();
        let retained = retain_spaced_train_pairs(dense);
        let last_row = retained.last().unwrap().1 + 1;
        let mut row_reuse = vec![0usize; last_row + 1];
        let mut transition_reuse = vec![0usize; last_row + 1];
        for &(_, anchor) in &retained {
            for count in &mut row_reuse[anchor + 1 - PRICE_DELTAS_PER_TICKER..=anchor + 1] {
                *count += 1;
            }
            for count in &mut transition_reuse[anchor + 2 - PRICE_DELTAS_PER_TICKER..=anchor + 1] {
                *count += 1;
            }
        }
        let max_row_reuse = *row_reuse.iter().max().unwrap();
        let max_transition_reuse = *transition_reuse.iter().max().unwrap();
        let retained_stride = 25 * TRAIN_ANCHOR_STRIDE_MULTIPLIER;
        assert_eq!(
            max_row_reuse,
            (PRICE_DELTAS_PER_TICKER + 1).div_ceil(retained_stride)
        );
        assert_eq!(max_row_reuse, 61);
        assert_eq!(
            max_transition_reuse,
            PRICE_DELTAS_PER_TICKER.div_ceil(retained_stride)
        );
        assert_eq!(max_transition_reuse, 60);
    }

    #[test]
    fn training_plan_cannot_repeat_the_reuse_safe_sampler_pool() {
        assert!(enforce_single_sampler_pass(PretrainExecutionMode::Train, 1, None, 10).is_ok());
        assert!(enforce_single_sampler_pass(PretrainExecutionMode::Train, 3, Some(10), 10).is_ok());
        assert!(enforce_single_sampler_pass(PretrainExecutionMode::Train, 2, None, 10).is_err());
        assert!(
            enforce_single_sampler_pass(PretrainExecutionMode::Train, 3, Some(11), 10).is_err()
        );
        assert!(
            enforce_single_sampler_pass(PretrainExecutionMode::EvaluateOnly, 3, None, 10).is_ok()
        );
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
        let directions = Tensor::randn([dim, LEJEPA_SIGREG_PROJECTIONS], opts);
        let directions = &directions / directions.norm_scalaropt_dim(2, [0i64].as_slice(), true);
        let unit_loss = sigreg_loss_with_directions(&unit, &directions).double_value(&[]);
        let inflated_loss = sigreg_loss_with_directions(&inflated, &directions).double_value(&[]);
        // The statistic is scaled by the fixed reference count (128), not the actual
        // sample axis, so the Gaussian null at 512 samples sits near
        // 128 * (1.052 / 512) ~= 0.263 rather than the n-scaled 1.052.
        assert!(
            (unit_loss - 0.263).abs() < 0.1,
            "reference-scaled Gaussian null should be near 0.263, got {unit_loss}"
        );
        assert!(
            inflated_loss > unit_loss * 4.0,
            "variance-3 sigreg {inflated_loss} should dwarf unit-variance {unit_loss}"
        );
    }

    #[test]
    fn sigreg_scale_is_independent_of_the_sample_count() {
        let _guard = tch::no_grad_guard();
        let base = Tensor::randn([3, 4, 16], (tch::Kind::Float, tch::Device::Cpu));
        let duplicated = Tensor::cat(&[&base, &base], 1);
        let directions = Tensor::randn(
            [16, LEJEPA_SIGREG_PROJECTIONS],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        let directions = &directions / directions.norm_scalaropt_dim(2, [0i64].as_slice(), true);
        let base_loss = sigreg_loss_with_directions(&base, &directions).double_value(&[]);
        let duplicated_loss =
            sigreg_loss_with_directions(&duplicated, &directions).double_value(&[]);
        // Scaling by the fixed reference count (not the actual sample axis) makes the
        // statistic invariant to duplicating identical samples: le-wm's `* B` gain is a
        // strength calibration, and pinning it decouples SIGReg from our micro-batch size.
        assert!(
            (duplicated_loss / base_loss - 1.0).abs() < 1e-5,
            "reference-scaled SIGReg must be invariant to sample count: {base_loss} vs {duplicated_loss}"
        );
    }

    #[test]
    fn capped_validation_ticker_count_only_includes_processed_chunks() {
        let pairs = vec![(0usize, 10usize), (1, 11), (1, 12), (2, 13), (3, 14)];
        let batch_size = 2;
        let max_batches = 2;
        let mut evaluated_tickers = std::collections::HashSet::new();
        let mut batches = 0;
        for chunk in pairs.chunks(batch_size) {
            if batches >= max_batches {
                break;
            }
            record_evaluated_tickers(&mut evaluated_tickers, chunk);
            batches += 1;
        }

        assert_eq!(evaluated_tickers.len(), 3);
        assert!(evaluated_tickers.contains(&0));
        assert!(evaluated_tickers.contains(&1));
        assert!(evaluated_tickers.contains(&2));
        assert!(!evaluated_tickers.contains(&3));
    }

    #[test]
    fn fast_validation_panel_has_one_deterministic_window_per_ticker() {
        let pairs = vec![(2, 30), (0, 40), (1, 50), (0, 20), (2, 10), (1, 60)];
        assert_eq!(
            ticker_stratified_panel(&pairs),
            vec![(0, 40), (1, 60), (2, 30)]
        );
    }

    #[test]
    fn lejepa_uses_full_two_pi_initialized_pope64_in_every_layer() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        tch::manual_seed(41);
        let heads = PretrainHeads::new(&vs.root(), 256, 16, 25);
        assert_eq!(heads.lejepa_layers.len(), LEJEPA_AR_LAYERS);
        let variables = vs.variables();
        for (index, layer) in heads.lejepa_layers.iter().enumerate() {
            let bias = &layer.pope_theta_bias;
            assert_eq!(bias.size(), vec![LEJEPA_HEADS, LEJEPA_HEAD_DIM]);
            assert_eq!(
                variables
                    .get(&format!("lejepa_layer_{index}.pope_theta_bias"))
                    .unwrap()
                    .size(),
                vec![LEJEPA_HEADS, LEJEPA_HEAD_DIM]
            );
            assert!(bias.max().double_value(&[]) <= 0.0);
            assert!(bias.min().double_value(&[]) >= -2.0 * std::f64::consts::PI);
            assert!(
                bias.abs().max().double_value(&[]) > 0.0,
                "two-pi initialization must not collapse to zero phase"
            );
        }
    }

    #[test]
    fn lejepa_pope_phase_biases_receive_finite_gradients() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        tch::manual_seed(43);
        let heads = PretrainHeads::new(&vs.root(), 256, 16, 25);
        let tokens = Tensor::randn([2, 1, 8, 256], (Kind::Float, tch::Device::Cpu));
        let belief = heads.predict_lejepa_bar_predictions(&tokens, true).belief;
        let weights = Tensor::arange(256, (Kind::Float, tch::Device::Cpu)).view([1, 1, 1, 256]);
        (&belief * weights).sum(Kind::Float).backward();
        for layer in &heads.lejepa_layers {
            let grad = layer.pope_theta_bias.grad();
            assert!(grad.defined());
            assert!(grad.isfinite().all().int64_value(&[]) != 0);
            assert!(
                grad.abs().max().double_value(&[]) > 0.0,
                "PoPE phase bias must receive a learning signal"
            );
        }
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

        let mut tokens = heads.encode_bar_tokens(&context, false);
        let start_len = tokens.size()[2];
        for step in 0..3 {
            let preds = heads.predict_lejepa_bar_predictions(&tokens, false);
            let last = tokens.size()[2] - 1;
            let ctx = preds.belief.narrow(2, last, 1).reshape([batch, latent_dim]);
            let next_latent = heads
                .lejepa_predict_next(&ctx)
                .view([batch, 1, 1, latent_dim]);
            tokens = Tensor::cat(&[&tokens, &next_latent], 2);
            assert_eq!(tokens.size()[2], start_len + step + 1);
        }

        let imagined = heads.lejepa_imagined_rollout(&context);
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
    fn predictor_output_is_bounded_and_rollout_is_repeatable() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let latent_dim = 256;
        let heads = PretrainHeads::new(&vs.root(), latent_dim, 16, 25);
        // The predictor is deterministic, so an AR rollout must be exactly repeatable.
        let context = Tensor::randn(
            [1, 1, 8, LEJEPA_BAR_FEATURES],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        let first = heads.lejepa_imagined_rollout(&context);
        let second = heads.lejepa_imagined_rollout(&context);
        assert_eq!((first - second).abs().max().double_value(&[]), 0.0);
    }

    #[test]
    fn jepa_latent_loss_backprops_through_prediction_and_target_branches() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let latent_dim = 256;
        let heads = PretrainHeads::new(&vs.root(), latent_dim, 16, 25);
        // `belief` and `target` stand in for the two attached encoder branches.
        let belief = Tensor::randn([16, latent_dim], (tch::Kind::Float, tch::Device::Cpu))
            .set_requires_grad(true);
        let target = Tensor::randn([16, latent_dim], (tch::Kind::Float, tch::Device::Cpu))
            .set_requires_grad(true);
        let pred = heads.lejepa_predict_next(&belief);
        let loss = (&pred - &target).square().mean(tch::Kind::Float);
        loss.backward();

        let belief_grad = belief.grad();
        assert!(
            belief_grad.defined()
                && belief_grad.abs().sum(tch::Kind::Float).double_value(&[]) > 0.0,
            "gradient must reach the attached belief (encoder)"
        );
        let target_grad = target.grad();
        assert!(
            target_grad.defined()
                && target_grad.abs().sum(tch::Kind::Float).double_value(&[]) > 0.0,
            "gradient must reach the attached next-token target"
        );
        let predictor_grad: f64 = vs
            .variables()
            .iter()
            .filter(|(name, _)| name.contains("lejepa_predictor"))
            .map(|(_, v)| {
                let g = v.grad();
                if g.defined() {
                    g.abs().sum(tch::Kind::Float).double_value(&[])
                } else {
                    0.0
                }
            })
            .sum();
        assert!(
            predictor_grad > 0.0,
            "gradient must reach the predictor head weights"
        );
    }

    #[test]
    fn raw_jepa_mse_matches_hand_computation() {
        let pred = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).reshape([2, 2]);
        let target = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 0.0]).reshape([2, 2]);
        // sq err = [[1,4],[4,16]]; raw mean = 6.25.
        let loss = (&pred - &target).square().mean(tch::Kind::Float);
        assert!((loss.double_value(&[]) - 6.25).abs() < 1e-5);
    }

    #[test]
    fn update_probe_var_ema_tracks_running_per_feature_variance() {
        let ema = Tensor::ones([2], (tch::Kind::Float, tch::Device::Cpu));
        // Column 0 has variance 1 (values 0,2); column 1 has variance 0.
        let target = Tensor::from_slice(&[0.0f32, 0.0, 2.0, 0.0]).reshape([2, 2]);
        super::update_probe_var_ema(&ema, &target);
        // ema*0.99 + var*0.01 -> [1.0, 0.99].
        assert!((ema.double_value(&[0]) - 1.0).abs() < 1e-6);
        assert!((ema.double_value(&[1]) - 0.99).abs() < 1e-6);
    }

    #[test]
    fn lejepa_parameters_have_no_attached_return_auxiliary() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let _heads = PretrainHeads::new(&vs.root(), 256, 16, 25);
        assert!(
            vs.variables()
                .keys()
                .all(|name| !name.contains("lejepa_return") && !name.contains("return_var_ema")),
            "canonical LeJEPA must not contain a training-only return head"
        );
    }

    #[test]
    fn directional_accuracy_and_rank_ic_match_synthetic_expectation() {
        let device = tch::Device::Cpu;
        // Signs: pred [+,-,+,0], actual [+,-,-,+]. Ties (pred==0) excluded, so
        // counted=3, agree on first two -> 2/3.
        let pred = Tensor::from_slice(&[1.0f32, -1.0, 1.0, 0.0]).to_device(device);
        let actual = Tensor::from_slice(&[1.0f32, -1.0, -1.0, 1.0]).to_device(device);
        let (correct, counted) = super::directional_accuracy(&pred, &actual);
        assert_eq!(counted, 3.0);
        assert_eq!(correct, 2.0);

        // Perfectly monotone ranking -> rank-IC = 1; reversed -> -1.
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_device(device);
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0]).to_device(device);
        assert!((super::rank_ic(&a, &b) - 1.0).abs() < 1e-6);
        let c = Tensor::from_slice(&[40.0f32, 30.0, 20.0, 10.0]).to_device(device);
        assert!((super::rank_ic(&a, &c) + 1.0).abs() < 1e-6);
    }

    fn synthetic_measurement_metrics(tickers: i64) -> super::MeasurementMetrics {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let latent_dim = 256;
        let heads = PretrainHeads::new(&vs.root(), latent_dim, 16, 25);
        let mut accum = super::MeasurementAccum::new(latent_dim, tch::Device::Cpu);
        let dummy = Tensor::zeros([1], (tch::Kind::Float, tch::Device::Cpu));
        for _ in 0..3 {
            let bar_history = Tensor::randn(
                [2, tickers, 6, LEJEPA_BAR_FEATURES],
                (tch::Kind::Float, tch::Device::Cpu),
            );
            let next_bars = Tensor::randn(
                [2, tickers, 1, LEJEPA_BAR_FEATURES],
                (tch::Kind::Float, tch::Device::Cpu),
            );
            let batch = super::PretrainBatch {
                obs: dummy.shallow_clone(),
                static_obs: dummy.shallow_clone(),
                next_obs: dummy.shallow_clone(),
                next_static_obs: dummy.shallow_clone(),
                future_patches: dummy.shallow_clone(),
                next_patch: dummy.shallow_clone(),
                bar_history,
                next_bars,
            };
            accum.add(&heads, &batch, 100.0);
        }
        accum.finish()
    }

    #[test]
    fn measurement_accum_produces_finite_metrics_on_synthetic_batches() {
        let metrics = synthetic_measurement_metrics(1);
        assert_eq!(metrics.skill_batches, 3);
        // The pooled/aggregate metrics are always defined.
        for value in [
            metrics.pred_probe_ev_mean,
            metrics.pred_probe_ev_persist,
            metrics.return_ev_persist,
            metrics.latent_ev_marginal,
            metrics.latent_ev_persist,
            metrics.skill_ev_correct,
            metrics.skill_ev_shuffled,
            metrics.skill_ev_zero,
        ] {
            assert!(
                value.is_finite(),
                "measurement metric must be finite: {value}"
            );
        }
        // Cross-sectional rank-IC is undefined with a single-ticker universe: it must
        // stay dormant (NaN), not report a fabricated pooled number.
        assert!(
            metrics.return_rank_ic.is_nan(),
            "cross-sectional rank-IC must be dormant when tickers < 2"
        );
    }

    #[test]
    fn cross_sectional_rank_ic_is_defined_with_multiple_tickers() {
        // With a genuine ticker cross-section the metric is computed and finite.
        let metrics = synthetic_measurement_metrics(4);
        assert!(
            metrics.return_rank_ic.is_finite(),
            "cross-sectional rank-IC must be finite when tickers >= 2"
        );
        assert!((-1.0..=1.0).contains(&metrics.return_rank_ic));
    }

    #[test]
    fn lejepa_checkpoint_promotion_requires_both_metrics_to_improve() {
        let validation = super::ValidationLoss {
            total: 0.9,
            ..Default::default()
        };
        assert!(super::is_better_pretrain_checkpoint(
            PretrainObjective::Lejepa,
            &validation,
            0.8,
            1.0,
            0.9,
        ));
        assert!(!super::is_better_pretrain_checkpoint(
            PretrainObjective::Lejepa,
            &validation,
            1.0,
            1.0,
            0.9,
        ));
        assert!(!super::is_better_pretrain_checkpoint(
            PretrainObjective::Lejepa,
            &validation,
            0.8,
            0.8,
            0.9,
        ));
    }

    #[test]
    fn promotion_candidate_loads_through_deployed_world_model() {
        let temp_dir = std::env::temp_dir().join(format!(
            "trading-bot-promotion-candidate-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let checkpoint = temp_dir.join("candidate.ot");
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let _heads = PretrainHeads::new(&vs.root(), 256, 16, 25);
        vs.save(&checkpoint).unwrap();
        let metadata_path = crate::torch::world_model::WorldModelMetadata::save_for_checkpoint(
            &checkpoint,
            256,
            100.0,
        )
        .unwrap();
        let world_model = crate::torch::world_model::LejepaWorldModel::load(
            &checkpoint,
            &metadata_path,
            tch::Device::Cpu,
        )
        .unwrap();
        let context = Tensor::randn(
            [1, 1, 8, LEJEPA_BAR_FEATURES],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        let prediction = world_model.predict(&context, 2).unwrap();
        assert_eq!(prediction.latent.size(), vec![1, 2, 256]);
        assert_eq!(prediction.ohlc_mean.size(), vec![1, 2, LEJEPA_BAR_FEATURES]);
        std::fs::remove_dir_all(temp_dir).unwrap();
    }

    fn synthetic_sampler(n: usize) -> PretrainSampler {
        PretrainSampler {
            train_tickers: Vec::new(),
            train_envs: Vec::new(),
            train_pairs: (0..n).map(|i| (0usize, i)).collect(),
            train_cursor: 0,
            val_pairs: Vec::new(),
            val_eval_cursor: 0,
            test_pairs: Vec::new(),
            k_patches: 1,
            patch_size: 1,
            target_scale: 1.0,
            device: tch::Device::Cpu,
        }
    }

    #[test]
    fn sampler_epoch_yields_floor_disjoint_batches_without_repeats() {
        use std::collections::HashSet;

        let n = 101;
        let batch_size = 7;
        let mut sampler = synthetic_sampler(n);
        let expected_batches = n / batch_size;
        assert_eq!(sampler.batches_per_epoch(batch_size), expected_batches);

        sampler.start_epoch();
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut batches = 0usize;
        while let Some(chunk) = sampler.take_train_chunk(batch_size) {
            assert_eq!(chunk.len(), batch_size, "every yielded chunk is full");
            for &pair in chunk {
                assert!(seen.insert(pair), "pair repeated within an epoch: {pair:?}");
            }
            batches += 1;
        }
        assert_eq!(batches, expected_batches, "exactly floor(N/batch) batches");
        assert_eq!(seen.len(), expected_batches * batch_size);
        assert!(seen.len() <= n, "epoch covers at most N pairs");
        assert!(
            n - seen.len() < batch_size,
            "only a partial final chunk is dropped"
        );
    }

    #[test]
    fn full_validation_retains_every_split_pair() {
        let pairs = vec![(0, 10), (0, 20), (0, 30), (1, 40), (1, 50)];

        assert_eq!(validation_pairs(&pairs, ValidationMode::Full), pairs);
        assert_eq!(
            validation_pairs(&pairs, ValidationMode::Fast),
            vec![(0, 20), (1, 50)]
        );
    }
}
