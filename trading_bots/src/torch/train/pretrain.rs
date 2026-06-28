use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use rand::{seq::SliceRandom, Rng};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use tch::{autocast, nn, nn::Module, Device, Kind, Reduction, Tensor};

use crate::data::universe::cached_eligible_training_universe;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::env::Env;
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{ModelVariant, TradingModel, TradingModelConfig};
use crate::torch::optim::muon::{Muon, MuonConfig};
use shared::{
    paths::RUNS_PATH,
    report::{Report, ReportKind, ReportSeries, ScaleKind},
    run_dir::RunDir,
};

use super::config::{LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON};
use super::optimizer_glue::{muon_momentum_for_step, named_trainable_variables};

const HORIZON_FEATURE_DIM: i64 = 7;
const LEJEPA_SIGREG_PROJECTIONS: i64 = 1024;
const LEJEPA_SIGREG_KNOTS: i64 = 17;
const LEJEPA_BAR_FEATURES: i64 = 7;
const LEJEPA_PATCH_VIT_LAYERS: usize = 3;
const LEJEPA_AR_LAYERS: usize = 5;
const LEJEPA_PROJECTOR_HIDDEN_DIM: i64 = 2048;

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

struct PatchVitLayer {
    qkv: nn::Linear,
    out_proj: nn::Linear,
    ff_gate: nn::Linear,
    ff_value: nn::Linear,
    ff_out: nn::Linear,
}

struct ProjectionMlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

struct LejepaPatchPredictions {
    belief: Tensor,
    projected: Tensor,
}

struct PretrainHeads {
    forecast_queries: Tensor,
    horizon_pos_proj: nn::Linear,
    forecast_q_proj: nn::Linear,
    forecast_k_proj: nn::Linear,
    forecast_v_proj: nn::Linear,
    forecast_out_proj: nn::Linear,
    return_mean: nn::Linear,
    patch_bar_proj: nn::Linear,
    patch_cls: Tensor,
    patch_bar_pos: Tensor,
    patch_vit_layers: Vec<PatchVitLayer>,
    patch_token_proj: nn::Linear,
    lejepa_projector: ProjectionMlp,
    lejepa_pos: Tensor,
    lejepa_layers: Vec<CausalLejepaLayer>,
    lejepa_pred_proj: nn::Linear,
    lejepa_pred_projector: ProjectionMlp,
    probe_fc1: nn::Linear,
    probe_out: nn::Linear,
    next_patch_embed: nn::Linear,
    latent_fc1: nn::Linear,
    latent_fc2: nn::Linear,
    horizon: i64,
    patch_size: i64,
    latent_dim: i64,
    patch_vit_dim: i64,
    forecast_heads: i64,
    patch_vit_heads: i64,
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
}

impl PretrainBatch {
    fn len(&self) -> i64 {
        self.obs.size()[0]
    }
}

struct PretrainSampler {
    train_tickers: Vec<String>,
    val_tickers: Vec<String>,
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
    fn new(
        p: &nn::Path,
        latent_dim: i64,
        patch_token_count: i64,
        k_patches: i64,
        patch_size: i64,
    ) -> Self {
        let ff_dim = latent_dim * 2;
        let patch_vit_dim = latent_dim / 2;
        let patch_vit_ff_dim = patch_vit_dim * 2;
        let horizon = k_patches * patch_size;
        let forecast_heads = 4;
        let patch_vit_heads = 4;
        let lejepa_heads = 4;
        assert_eq!(
            latent_dim % forecast_heads,
            0,
            "forecast attention heads must divide latent dim"
        );
        assert_eq!(
            patch_vit_dim % patch_vit_heads,
            0,
            "patch ViT attention heads must divide patch ViT dim"
        );
        assert_eq!(
            latent_dim % lejepa_heads,
            0,
            "LEJEPA attention heads must divide latent dim"
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
        let patch_bar_proj = nn::linear(
            p / "patch_bar_proj",
            LEJEPA_BAR_FEATURES,
            patch_vit_dim,
            Default::default(),
        );
        let patch_cls = p.var(
            "patch_cls",
            &[patch_vit_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let patch_bar_pos = p.var(
            "patch_bar_pos",
            &[patch_size, patch_vit_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let mut patch_vit_layers = Vec::with_capacity(LEJEPA_PATCH_VIT_LAYERS);
        for layer_idx in 0..LEJEPA_PATCH_VIT_LAYERS {
            let layer_name = format!("patch_vit_layer_{layer_idx}");
            let layer_path = p / layer_name.as_str();
            patch_vit_layers.push(PatchVitLayer {
                qkv: nn::linear(
                    &layer_path / "qkv",
                    patch_vit_dim,
                    patch_vit_dim * 3,
                    Default::default(),
                ),
                out_proj: nn::linear(
                    &layer_path / "out_proj",
                    patch_vit_dim,
                    patch_vit_dim,
                    Default::default(),
                ),
                ff_gate: nn::linear(
                    &layer_path / "ff_gate",
                    patch_vit_dim,
                    patch_vit_ff_dim,
                    Default::default(),
                ),
                ff_value: nn::linear(
                    &layer_path / "ff_value",
                    patch_vit_dim,
                    patch_vit_ff_dim,
                    Default::default(),
                ),
                ff_out: nn::linear(
                    &layer_path / "ff_out",
                    patch_vit_ff_dim,
                    patch_vit_dim,
                    Default::default(),
                ),
            });
        }
        let patch_token_proj = nn::linear(
            p / "patch_token_proj",
            patch_vit_dim,
            latent_dim,
            Default::default(),
        );
        let lejepa_projector = ProjectionMlp {
            fc1: nn::linear(
                p / "lejepa_projector_fc1",
                latent_dim,
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
                    ff_dim,
                    Default::default(),
                ),
                ff_value: nn::linear(
                    &layer_path / "ff_value",
                    latent_dim,
                    ff_dim,
                    Default::default(),
                ),
                ff_out: nn::linear(
                    &layer_path / "ff_out",
                    ff_dim,
                    latent_dim,
                    Default::default(),
                ),
            });
        }
        let lejepa_pos = p.var(
            "lejepa_pos",
            &[patch_token_count, latent_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let lejepa_pred_proj = nn::linear(
            p / "lejepa_pred_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let lejepa_pred_projector = ProjectionMlp {
            fc1: nn::linear(
                p / "lejepa_pred_projector_fc1",
                latent_dim,
                LEJEPA_PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            fc2: nn::linear(
                p / "lejepa_pred_projector_fc2",
                LEJEPA_PROJECTOR_HIDDEN_DIM,
                latent_dim,
                Default::default(),
            ),
        };
        let probe_fc1 = nn::linear(p / "probe_fc1", latent_dim, ff_dim, Default::default());
        let probe_out = nn::linear(p / "probe_out", ff_dim, patch_size, Default::default());
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
            patch_bar_proj,
            patch_cls,
            patch_bar_pos,
            patch_vit_layers,
            patch_token_proj,
            lejepa_projector,
            lejepa_pos,
            lejepa_layers,
            lejepa_pred_proj,
            lejepa_pred_projector,
            probe_fc1,
            probe_out,
            next_patch_embed,
            latent_fc1,
            latent_fc2,
            horizon,
            patch_size,
            latent_dim,
            patch_vit_dim,
            forecast_heads,
            patch_vit_heads,
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

    fn encode_lejepa_patches(&self, layouts: &Tensor, batch_size: i64) -> Tensor {
        let rows = layouts.size()[0];
        let patches = layouts.size()[1] / self.patch_size;
        let bars = layouts
            .view([rows * patches, self.patch_size])
            .to_kind(Kind::Float)
            .nan_to_num(0.0, 0.0, 0.0);
        self.encode_lejepa_patch_rows(&bars).view([
            batch_size,
            TICKERS_COUNT,
            patches,
            self.latent_dim,
        ])
    }

    fn encode_lejepa_next_patch(&self, next_patch: &Tensor) -> Tensor {
        let size = next_patch.size();
        let batch = size[0];
        let tickers = size[1];
        let bars = next_patch
            .view([batch * tickers, self.patch_size])
            .to_kind(Kind::Float)
            .nan_to_num(0.0, 0.0, 0.0);
        self.encode_lejepa_patch_rows(&bars)
            .view([batch, tickers, self.latent_dim])
    }

    fn encode_lejepa_patch_rows(&self, bars: &Tensor) -> Tensor {
        let rows = bars.size()[0];
        let delta = bars.unsqueeze(-1);
        let abs_delta = bars.abs().unsqueeze(-1);
        let squared_delta = bars.square().unsqueeze(-1);
        let cumulative = bars.cumsum(-1, Kind::Float).unsqueeze(-1);
        let patch_mean = bars
            .mean_dim([1i64].as_slice(), true, Kind::Float)
            .unsqueeze(-1)
            .expand([rows, self.patch_size, 1], false);
        let centered = bars - &patch_mean.squeeze_dim(-1);
        let patch_std = centered
            .pow_tensor_scalar(2.0)
            .mean_dim([1i64].as_slice(), true, Kind::Float)
            .clamp_min(1e-12)
            .sqrt()
            .unsqueeze(-1)
            .expand([rows, self.patch_size, 1], false);
        let denom = (self.patch_size - 1).max(1) as f64;
        let position =
            ((Tensor::arange(self.patch_size, (Kind::Float, bars.device())) / denom) * 2.0 - 1.0)
                .view([1, self.patch_size, 1])
                .expand([rows, self.patch_size, 1], false);
        let features = Tensor::cat(
            &[
                &delta,
                &abs_delta,
                &squared_delta,
                &cumulative,
                &patch_mean,
                &patch_std,
                &position,
            ],
            -1,
        );
        let bar_pos = self.patch_bar_pos.to_kind(features.kind()).view([
            1,
            self.patch_size,
            self.patch_vit_dim,
        ]);
        let bar_tokens = self.patch_bar_proj.forward(&features) + bar_pos;
        let cls = self
            .patch_cls
            .to_kind(bar_tokens.kind())
            .view([1, 1, self.patch_vit_dim])
            .expand([rows, 1, self.patch_vit_dim], false);
        let mut x = Tensor::cat(&[&cls, &bar_tokens], 1);
        for layer in &self.patch_vit_layers {
            x = self.patch_vit_layer(&x, layer);
        }
        let patch_token = self
            .patch_token_proj
            .forward(&normalize_last_dim(&x).select(1, 0));
        self.projection_mlp(&patch_token, &self.lejepa_projector)
    }

    fn patch_vit_layer(&self, source: &Tensor, layer: &PatchVitLayer) -> Tensor {
        let size = source.size();
        let rows = size[0];
        let tokens = size[1];
        let normed = normalize_last_dim(source);
        let qkv = layer.qkv.forward(&normed);
        let parts = qkv.split(self.patch_vit_dim, -1);
        let head_dim = self.patch_vit_dim / self.patch_vit_heads;
        let q = parts[0]
            .view([rows, tokens, self.patch_vit_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let k = parts[1]
            .view([rows, tokens, self.patch_vit_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let v = parts[2]
            .view([rows, tokens, self.patch_vit_heads, head_dim])
            .permute([0, 2, 1, 3]);
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
            false,
            None,
            true,
        )
        .to_kind(source.kind())
        .permute([0, 2, 1, 3])
        .contiguous()
        .view([rows, tokens, self.patch_vit_dim]);
        let x = source + layer.out_proj.forward(&attn);
        let normed = normalize_last_dim(&x);
        let gate = layer.ff_gate.forward(&normed).silu();
        let value = layer.ff_value.forward(&normed);
        x + layer.ff_out.forward(&(gate * value))
    }

    fn predict_lejepa_patch_predictions(
        &self,
        patch_tokens: &Tensor,
        train: bool,
    ) -> LejepaPatchPredictions {
        let size = patch_tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let patches = size[2];
        let rows = batch * tickers;
        let pos = self
            .lejepa_pos
            .narrow(0, 0, patches)
            .view([1, patches, self.latent_dim]);
        let mut x = patch_tokens.view([rows, patches, self.latent_dim]) + pos;
        let causal_mask =
            Tensor::ones([patches, patches], (Kind::Float, x.device())).triu(1) * -1e9;
        for layer in &self.lejepa_layers {
            x = self.causal_lejepa_layer(&x, layer, &causal_mask, train);
        }
        let belief = self.lejepa_pred_proj.forward(&normalize_last_dim(&x));
        let projected = self.projection_mlp(&belief, &self.lejepa_pred_projector);
        LejepaPatchPredictions {
            belief: belief.view([batch, tickers, patches, self.latent_dim]),
            projected: projected.view([batch, tickers, patches, self.latent_dim]),
        }
    }

    fn predict_lejepa_next_patch_belief(&self, patch_tokens: &Tensor, train: bool) -> Tensor {
        let predicted = self.predict_lejepa_patch_predictions(patch_tokens, train);
        predicted.belief.select(2, predicted.belief.size()[2] - 1)
    }

    fn causal_lejepa_layer(
        &self,
        source: &Tensor,
        layer: &CausalLejepaLayer,
        causal_mask: &Tensor,
        train: bool,
    ) -> Tensor {
        let size = source.size();
        let rows = size[0];
        let patches = size[1];
        let normed = normalize_last_dim(source);
        let qkv = layer.qkv.forward(&normed);
        let parts = qkv.split(self.latent_dim, -1);
        let head_dim = self.latent_dim / self.lejepa_heads;
        let q = parts[0]
            .view([rows, patches, self.lejepa_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let k = parts[1]
            .view([rows, patches, self.lejepa_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let v = parts[2]
            .view([rows, patches, self.lejepa_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let attn_scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
        let attn = (attn_scores + causal_mask.view([1, 1, patches, patches]))
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train)
            .to_kind(v.kind());
        let attn = attn.matmul(&v).permute([0, 2, 1, 3]).contiguous().view([
            rows,
            patches,
            self.latent_dim,
        ]);
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

    fn projection_mlp(&self, x: &Tensor, mlp: &ProjectionMlp) -> Tensor {
        let shape = x.size();
        let rows = x.numel() as i64 / self.latent_dim;
        let flat = x.view([rows, self.latent_dim]);
        let hidden = mlp.fc1.forward(&flat);
        let hidden = normalize_feature_batch(&hidden).gelu("none");
        mlp.fc2.forward(&hidden).view(shape.as_slice())
    }

    fn probe_return_mean(&self, predicted_patch_embed: &Tensor) -> Tensor {
        let size = predicted_patch_embed.size();
        let batch = size[0];
        let tickers = size[1];
        let h = self
            .probe_fc1
            .forward(predicted_patch_embed)
            .relu()
            .view([batch * tickers, -1]);
        let raw = self
            .probe_out
            .forward(&h)
            .view([batch, tickers, self.patch_size]);
        raw
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
        let val_tickers = cached_eligible_training_universe().to_vec();
        let mut train_tickers = val_tickers.clone();
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
            val_tickers,
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

        let mut obs = Vec::with_capacity(offsets.len() * pd_dim);
        let mut static_obs = Vec::with_capacity(offsets.len() * so_dim);
        let mut next_obs = Vec::with_capacity(offsets.len() * pd_dim);
        let mut next_static_obs = Vec::with_capacity(offsets.len() * so_dim);
        let mut future_patches = Vec::with_capacity(offsets.len() * target_len);
        let mut next_patch = Vec::with_capacity(offsets.len() * next_patch_len);

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

        let mut obs = Vec::with_capacity(samples.len() * pd_dim);
        let mut static_obs = Vec::with_capacity(samples.len() * so_dim);
        let mut next_obs = Vec::with_capacity(samples.len() * pd_dim);
        let mut next_static_obs = Vec::with_capacity(samples.len() * so_dim);
        let mut future_patches = Vec::with_capacity(samples.len() * target_len);
        let mut next_patch = Vec::with_capacity(samples.len() * next_patch_len);

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
) {
    let (obs_i, static_i) = env.reset_single_at_offset_for_pretrain(offset);
    let target_i =
        future_patches_for_current_perm(env, offset, k_patches, patch_size, target_scale);
    let next_patch_i = future_patches_for_current_perm(env, offset, 1, patch_size, 1.0);
    let (next_obs_i, next_static_i) =
        env.reset_single_at_offset_preserving_perm_for_pretrain(offset + patch_size);

    obs.extend(obs_i);
    static_obs.extend(static_i);
    future_patches.extend(target_i);
    next_patch.extend(next_patch_i);
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
    let max_target_advance = horizon.max(next_latent_advance);
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
        model.pretrain_patch_token_count(),
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
            adamw_wd: 0.0,
            force_adamw_name_substrings: vec![
                "policy_concentration".to_string(),
                "value_proj".to_string(),
                "forecast_".to_string(),
                "horizon_pos_proj".to_string(),
                "return_mean".to_string(),
                "patch_bar_".to_string(),
                "patch_cls".to_string(),
                "patch_token_proj".to_string(),
                "patch_vit_".to_string(),
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
            adamw_lr: LEARNING_RATE,
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
        "epoch,global_step,total_loss,jepa_mse,sigreg,repr_std_mean,repr_std_min,pred_embed_std,target_embed_std,probe_nll,return_mse,return_mae,return_bias,pred_abs,target_abs,pred_std,target_std,terminal_mse,next_lat,samples,batches"
    )?;
    writeln!(
        validation_log,
        "epoch,global_step,total_loss,jepa_mse,sigreg,repr_std_mean,repr_std_min,pred_embed_std,target_embed_std,probe_nll,return_mse,return_mae,return_bias,pred_abs,target_abs,pred_std,target_std,terminal_mse,next_lat,zero_mse,samples,tickers,batches"
    )?;
    let mut step_log = if args.log_step_losses {
        let mut log = BufWriter::new(File::create(run_dir.root.join("pretrain_train_steps.csv"))?);
        writeln!(
            log,
            "global_step,epoch,total_loss,jepa_mse,sigreg,repr_std_mean,repr_std_min,pred_embed_std,target_embed_std,probe_nll,return_mse,return_mae,return_bias,pred_abs,target_abs,pred_std,target_std,terminal_mse,next_lat,samples"
        )?;
        Some(log)
    } else {
        None
    };

    'epoch_loop: for epoch in 1..=args.epochs {
        sampler.start_epoch();
        let mut train_epoch_loss = RunningLoss::new(device);
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
                let return_mse_v = losses.return_mse.double_value(&[]);
                let return_mae_v = losses.return_mae.double_value(&[]);
                let return_bias_v = losses.return_bias.double_value(&[]);
                let pred_abs_v = losses.pred_abs.double_value(&[]);
                let target_abs_v = losses.target_abs.double_value(&[]);
                let pred_std_v = losses.pred_std.double_value(&[]);
                let target_std_v = losses.target_std.double_value(&[]);
                let terminal_mse_v = losses.terminal_mse.double_value(&[]);
                let lat_v = losses.next_lat.double_value(&[]);
                writeln!(
                    log,
                    "{global_step},{epoch},{total_v:.9},{jepa_mse_v:.9},{sigreg_v:.9},{repr_std_mean_v:.9},{repr_std_min_v:.9},{pred_embed_std_v:.9},{target_embed_std_v:.9},{probe_nll_v:.9},{return_mse_v:.9},{return_mae_v:.9},{return_bias_v:.9},{pred_abs_v:.9},{target_abs_v:.9},{pred_std_v:.9},{target_std_v:.9},{terminal_mse_v:.9},{lat_v:.9},{batch_samples}"
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
                    return_mse_v,
                    return_mae_v,
                    return_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    terminal_mse_v,
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
                    probe_nll_v,
                    return_mse_v,
                    return_mae_v,
                    return_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    terminal_mse_v,
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
                        losses.return_mse.double_value(&[]),
                        losses.return_mae.double_value(&[]),
                        losses.return_bias.double_value(&[]),
                        losses.pred_abs.double_value(&[]),
                        losses.target_abs.double_value(&[]),
                        losses.pred_std.double_value(&[]),
                        losses.target_std.double_value(&[]),
                        losses.terminal_mse.double_value(&[]),
                        losses.next_lat.double_value(&[]),
                    )
                });
                println!(
                    "pretrain epoch {epoch} step {global_step} train total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_nll={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6}",
                    total_v,
                    jepa_mse_v,
                    sigreg_v,
                    repr_std_mean_v,
                    repr_std_min_v,
                    pred_embed_std_v,
                    target_embed_std_v,
                    probe_nll_v,
                    return_mse_v,
                    return_mae_v,
                    return_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    terminal_mse_v,
                    lat_v
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
                    "pretrain step {global_step} validation total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_nll={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6} zero_mse={:.6} samples={} tickers={} batches={}",
                    val.total,
                    val.jepa_mse,
                    val.sigreg,
                    val.repr_std_mean,
                    val.repr_std_min,
                    val.pred_embed_std,
                    val.target_embed_std,
                    val.probe_nll,
                    val.return_mse,
                    val.return_mae,
                    val.return_bias,
                    val.pred_abs,
                    val.target_abs,
                    val.pred_std,
                    val.target_std,
                    val.terminal_mse,
                    val.next_lat,
                    val.zero_mse,
                    val.samples,
                    val.tickers,
                    val.batches
                );
                writeln!(
                    validation_log,
                    "step:{global_step},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
                    val.total,
                    val.jepa_mse,
                    val.sigreg,
                    val.repr_std_mean,
                    val.repr_std_min,
                    val.pred_embed_std,
                    val.target_embed_std,
                    val.probe_nll,
                    val.return_mse,
                    val.return_mae,
                    val.return_bias,
                    val.pred_abs,
                    val.target_abs,
                    val.pred_std,
                    val.target_std,
                    val.terminal_mse,
                    val.next_lat,
                    val.zero_mse,
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
            "pretrain epoch {epoch} train_mean total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_nll={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6} samples={} batches={}",
            train.total,
            train.jepa_mse,
            train.sigreg,
            train.repr_std_mean,
            train.repr_std_min,
            train.pred_embed_std,
            train.target_embed_std,
            train.probe_nll,
            train.return_mse,
            train.return_mae,
            train.return_bias,
            train.pred_abs,
            train.target_abs,
            train.pred_std,
            train.target_std,
            train.terminal_mse,
            train.next_lat,
            train.samples,
            train.batches
        );
        writeln!(
            train_epoch_log,
            "{epoch},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{}",
            train.total,
            train.jepa_mse,
            train.sigreg,
            train.repr_std_mean,
            train.repr_std_min,
            train.pred_embed_std,
            train.target_embed_std,
            train.probe_nll,
            train.return_mse,
            train.return_mae,
            train.return_bias,
            train.pred_abs,
            train.target_abs,
            train.pred_std,
            train.target_std,
            train.terminal_mse,
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
                "pretrain epoch {epoch} detached_probe_train probe_nll={:.6} return_mse={:.6} return_mae={:.6} pred_std={:.6} target_std={:.6} samples={} batches={} probe_epochs={}",
                probe.probe_nll,
                probe.return_mse,
                probe.return_mae,
                probe.pred_std,
                probe.target_std,
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
            "pretrain epoch {epoch} validation total_loss={:.6} jepa_mse={:.6} sigreg={:.6} repr_std_mean={:.6} repr_std_min={:.6} pred_embed_std={:.6} target_embed_std={:.6} probe_nll={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6} zero_mse={:.6} samples={} tickers={} batches={}",
            val.total,
            val.jepa_mse,
            val.sigreg,
            val.repr_std_mean,
            val.repr_std_min,
            val.pred_embed_std,
            val.target_embed_std,
            val.probe_nll,
            val.return_mse,
            val.return_mae,
            val.return_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.terminal_mse,
            val.next_lat,
            val.zero_mse,
            val.samples,
            val.tickers,
            val.batches
        );
        writeln!(
            validation_log,
            "{epoch},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
            val.total,
            val.jepa_mse,
            val.sigreg,
            val.repr_std_mean,
            val.repr_std_min,
            val.pred_embed_std,
            val.target_embed_std,
            val.probe_nll,
            val.return_mse,
            val.return_mae,
            val.return_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.terminal_mse,
            val.next_lat,
            val.zero_mse,
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
            "final,{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
            val.total,
            val.jepa_mse,
            val.sigreg,
            val.repr_std_mean,
            val.repr_std_min,
            val.pred_embed_std,
            val.target_embed_std,
            val.probe_nll,
            val.return_mse,
            val.return_mae,
            val.return_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.terminal_mse,
            val.next_lat,
            val.zero_mse,
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
    load_summary
        .require_complete()
        .map_err(|err| anyhow!("failed to load complete pretrain heads: {err}"))?;
    println!("Loaded pretrain heads from {}", heads_path.display());
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
    let return_mse = return_pred.mse_loss(&return_target, Reduction::Mean);
    let return_err = &return_pred - &return_target;
    let return_mae = return_err.abs().mean(Kind::Float);
    let return_bias = return_err.mean(Kind::Float);
    let pred_abs = return_pred.abs().mean(Kind::Float);
    let target_abs = return_target.abs().mean(Kind::Float);
    let pred_std = return_pred.std(false);
    let target_std = return_target.std(false);
    let terminal_idx = heads.horizon - 1;
    let terminal_pred = return_pred.select(-1, terminal_idx);
    let terminal_target = return_target.select(-1, terminal_idx);
    let terminal_mse = terminal_pred.mse_loss(&terminal_target, Reduction::Mean);
    let base_loss = return_mse.shallow_clone();

    if lambda_lat == 0.0 {
        let next_lat = Tensor::zeros([], (Kind::Float, pred_abs.device()));
        return PretrainLoss {
            total: base_loss,
            jepa_mse: zero_like_scalar(&return_mse),
            sigreg: zero_like_scalar(&return_mse),
            repr_std_mean,
            repr_std_min,
            pred_embed_std: zero_like_scalar(&return_mse),
            target_embed_std: zero_like_scalar(&return_mse),
            probe_nll: zero_like_scalar(&return_mse),
            return_mae,
            return_mse,
            pred_std,
            target_std,
            return_bias,
            pred_abs,
            target_abs,
            next_lat,
            terminal_mse,
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
        jepa_mse: zero_like_scalar(&return_mse),
        sigreg: zero_like_scalar(&return_mse),
        repr_std_mean,
        repr_std_min,
        pred_embed_std: zero_like_scalar(&return_mse),
        target_embed_std: zero_like_scalar(&return_mse),
        probe_nll: zero_like_scalar(&return_mse),
        return_mae,
        return_mse,
        pred_std,
        target_std,
        return_bias,
        pred_abs,
        target_abs,
        next_lat: latent_loss,
        terminal_mse,
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
    let batch_size = batch.obs.size()[0];
    let layout_len = model.pretrain_layout_len();
    let layouts = model
        .uniform_stream_layout_from_raw_input(&batch.obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);

    let patch_tokens = autocast(false, || heads.encode_lejepa_patches(&layouts, batch_size));
    let final_target_token = autocast(false, || heads.encode_lejepa_next_patch(&batch.next_patch));
    let source_target_tokens = patch_tokens.narrow(2, 1, patch_tokens.size()[2] - 1);
    let final_target_token = final_target_token.unsqueeze(2);
    let target_patch_tokens = Tensor::cat(&[&source_target_tokens, &final_target_token], 2);
    let predictions = heads.predict_lejepa_patch_predictions(&patch_tokens, train);
    let pred_patch_embeds = predictions.projected;
    let jepa_mse = pred_patch_embeds.mse_loss(&target_patch_tokens, Reduction::Mean);
    let sigreg_tokens = Tensor::cat(&[&patch_tokens, &final_target_token], 2);
    let sigreg = sigreg_loss(&sigreg_tokens);
    let (repr_std_mean, repr_std_min) = representation_std_metrics(&sigreg_tokens);
    let pred_embed_std = pred_patch_embeds.std(false);
    let target_embed_std = target_patch_tokens.std(false);
    let total = &jepa_mse + &sigreg * lambda_sigreg;

    let pred_next_embed = predictions
        .belief
        .select(2, predictions.belief.size()[2] - 1);
    let probe_target = scaled_next_patch_cumulative_returns(&batch.next_patch, target_scale);
    let probe = probe_metrics(heads, &pred_next_embed.detach(), &probe_target);
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
        return_mae: probe.return_mae,
        return_mse: probe.return_mse,
        pred_std: probe.pred_std,
        target_std: probe.target_std,
        return_bias: probe.return_bias,
        pred_abs: probe.pred_abs,
        target_abs: probe.target_abs,
        next_lat,
        terminal_mse: probe.terminal_mse,
    }
}

fn cumulative_future_returns(future_patches: &Tensor) -> Tensor {
    let size = future_patches.size();
    future_patches
        .view([size[0], size[1], size[2] * size[3]])
        .cumsum(-1, Kind::Float)
}

fn scaled_next_patch_cumulative_returns(next_patch: &Tensor, target_scale: f64) -> Tensor {
    (next_patch * target_scale).cumsum(-1, Kind::Float)
}

fn zero_like_scalar(reference: &Tensor) -> Tensor {
    Tensor::zeros([], (Kind::Float, reference.device()))
}

fn sigreg_loss(patch_tokens: &Tensor) -> Tensor {
    let size = patch_tokens.size();
    let rows = size[0] * size[1];
    let patches = size[2];
    let dim = size[3];
    let proj = patch_tokens
        .view([rows, patches, dim])
        .transpose(0, 1)
        .to_kind(Kind::Float);
    let mut directions = Tensor::randn(
        [dim, LEJEPA_SIGREG_PROJECTIONS],
        (Kind::Float, patch_tokens.device()),
    );
    directions = &directions
        / directions
            .norm_scalaropt_dim(2, [0i64].as_slice(), true)
            .clamp_min(1e-7);
    let t = Tensor::linspace(
        0.0,
        3.0,
        LEJEPA_SIGREG_KNOTS,
        (Kind::Float, patch_tokens.device()),
    );
    let dt = 3.0 / (LEJEPA_SIGREG_KNOTS - 1) as f64;
    let weights = Tensor::full(
        [LEJEPA_SIGREG_KNOTS],
        2.0 * dt,
        (Kind::Float, patch_tokens.device()),
    );
    let _ = weights.narrow(0, 0, 1).fill_(dt);
    let _ = weights.narrow(0, LEJEPA_SIGREG_KNOTS - 1, 1).fill_(dt);
    let phi = (-t.square() * 0.5).exp();
    let weights = weights * &phi;
    let x_t = proj.matmul(&directions).unsqueeze(-1) * t.view([1, 1, 1, -1]);
    let cos_err = x_t.cos().mean_dim([1i64].as_slice(), false, Kind::Float) - &phi;
    let sin_err = x_t.sin().mean_dim([1i64].as_slice(), false, Kind::Float);
    let err = cos_err.square() + sin_err.square();
    (err * weights.view([1, 1, -1]))
        .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
        .mean(Kind::Float)
        * rows as f64
}

fn representation_std_metrics(patch_tokens: &Tensor) -> (Tensor, Tensor) {
    let dim = patch_tokens.size()[3];
    let flat = patch_tokens.view([-1, dim]).to_kind(Kind::Float);
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
    return_mae: Tensor,
    return_mse: Tensor,
    pred_std: Tensor,
    target_std: Tensor,
    return_bias: Tensor,
    pred_abs: Tensor,
    target_abs: Tensor,
    terminal_mse: Tensor,
}

fn probe_metrics(
    heads: &PretrainHeads,
    predicted_patch_embed: &Tensor,
    target: &Tensor,
) -> ProbeLoss {
    let mean = heads.probe_return_mean(predicted_patch_embed);
    let err = &mean - target;
    let return_mse = mean.mse_loss(target, Reduction::Mean);
    let return_mae = err.abs().mean(Kind::Float);
    let return_bias = err.mean(Kind::Float);
    let pred_abs = mean.abs().mean(Kind::Float);
    let target_abs = target.abs().mean(Kind::Float);
    let pred_std = mean.std(false);
    let target_std = target.std(false);
    let terminal_idx = target.size()[2] - 1;
    let terminal_mse = mean
        .select(-1, terminal_idx)
        .mse_loss(&target.select(-1, terminal_idx), Reduction::Mean);
    let probe_nll = zero_like_scalar(&return_mse);
    ProbeLoss {
        probe_nll,
        return_mae,
        return_mse,
        pred_std,
        target_std,
        return_bias,
        pred_abs,
        target_abs,
        terminal_mse,
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

fn predict_lejepa_next_patch_belief(
    model: &TradingModel,
    heads: &PretrainHeads,
    batch: &PretrainBatch,
) -> Tensor {
    let batch_size = batch.obs.size()[0];
    let layout_len = model.pretrain_layout_len();
    let layouts = model
        .uniform_stream_layout_from_raw_input(&batch.obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);
    let patch_tokens = autocast(false, || heads.encode_lejepa_patches(&layouts, batch_size));
    heads.predict_lejepa_next_patch_belief(&patch_tokens, false)
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
    return_mae: f64,
    return_mse: f64,
    pred_std: f64,
    target_std: f64,
    return_bias: f64,
    pred_abs: f64,
    target_abs: f64,
    next_lat: f64,
    terminal_mse: f64,
    zero_mse: f64,
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
    return_mae: Tensor,
    return_mse: Tensor,
    pred_std: Tensor,
    target_std: Tensor,
    return_bias: Tensor,
    pred_abs: Tensor,
    target_abs: Tensor,
    next_lat: Tensor,
    terminal_mse: Tensor,
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
    return_mae_sum: Tensor,
    return_mse_sum: Tensor,
    pred_std_sum: Tensor,
    target_std_sum: Tensor,
    return_bias_sum: Tensor,
    pred_abs_sum: Tensor,
    target_abs_sum: Tensor,
    next_lat_sum: Tensor,
    terminal_mse_sum: Tensor,
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
            return_mae_sum: Tensor::zeros([], (Kind::Float, device)),
            return_mse_sum: Tensor::zeros([], (Kind::Float, device)),
            pred_std_sum: Tensor::zeros([], (Kind::Float, device)),
            target_std_sum: Tensor::zeros([], (Kind::Float, device)),
            return_bias_sum: Tensor::zeros([], (Kind::Float, device)),
            pred_abs_sum: Tensor::zeros([], (Kind::Float, device)),
            target_abs_sum: Tensor::zeros([], (Kind::Float, device)),
            next_lat_sum: Tensor::zeros([], (Kind::Float, device)),
            terminal_mse_sum: Tensor::zeros([], (Kind::Float, device)),
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
            self.return_mae_sum += losses.return_mae.detach() * weight;
            self.return_mse_sum += losses.return_mse.detach() * weight;
            self.pred_std_sum += losses.pred_std.detach() * weight;
            self.target_std_sum += losses.target_std.detach() * weight;
            self.return_bias_sum += losses.return_bias.detach() * weight;
            self.pred_abs_sum += losses.pred_abs.detach() * weight;
            self.target_abs_sum += losses.target_abs.detach() * weight;
            self.next_lat_sum += losses.next_lat.detach() * weight;
            self.terminal_mse_sum += losses.terminal_mse.detach() * weight;
            self.samples += samples;
            self.batches += 1;
        });
    }

    fn finish(self) -> TrainEpochLoss {
        assert!(self.samples > 0, "train epoch is empty");
        let denom = self.samples as f64;
        TrainEpochLoss {
            total: self.total_sum.double_value(&[]) / denom,
            jepa_mse: self.jepa_mse_sum.double_value(&[]) / denom,
            sigreg: self.sigreg_sum.double_value(&[]) / denom,
            repr_std_mean: self.repr_std_mean_sum.double_value(&[]) / denom,
            repr_std_min: self.repr_std_min_sum.double_value(&[]) / denom,
            pred_embed_std: self.pred_embed_std_sum.double_value(&[]) / denom,
            target_embed_std: self.target_embed_std_sum.double_value(&[]) / denom,
            probe_nll: self.probe_nll_sum.double_value(&[]) / denom,
            return_mae: self.return_mae_sum.double_value(&[]) / denom,
            return_mse: self.return_mse_sum.double_value(&[]) / denom,
            pred_std: self.pred_std_sum.double_value(&[]) / denom,
            target_std: self.target_std_sum.double_value(&[]) / denom,
            return_bias: self.return_bias_sum.double_value(&[]) / denom,
            pred_abs: self.pred_abs_sum.double_value(&[]) / denom,
            target_abs: self.target_abs_sum.double_value(&[]) / denom,
            next_lat: self.next_lat_sum.double_value(&[]) / denom,
            terminal_mse: self.terminal_mse_sum.double_value(&[]) / denom,
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
    return_mae: f64,
    return_mse: f64,
    pred_std: f64,
    target_std: f64,
    return_bias: f64,
    pred_abs: f64,
    target_abs: f64,
    next_lat: f64,
    terminal_mse: f64,
    samples: usize,
    batches: usize,
}

struct ProbeTrainSummary {
    probe_nll: f64,
    return_mae: f64,
    return_mse: f64,
    pred_std: f64,
    target_std: f64,
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
    train_return_mae: Vec<f32>,
    eval_return_mae: Vec<f32>,
    train_pred_std: Vec<f32>,
    eval_pred_std: Vec<f32>,
    train_target_std: Vec<f32>,
    eval_target_std: Vec<f32>,
    train_terminal_mse: Vec<f32>,
    eval_terminal_mse: Vec<f32>,
}

impl PretrainScalarHistory {
    fn push(&mut self, train: &TrainEpochLoss, val: &ValidationLoss) {
        self.train_mse.push(train.return_mse as f32);
        self.eval_mse.push(val.return_mse as f32);
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
        self.train_return_mae.push(train.return_mae as f32);
        self.eval_return_mae.push(val.return_mae as f32);
        self.train_pred_std.push(train.pred_std as f32);
        self.eval_pred_std.push(val.pred_std as f32);
        self.train_target_std.push(train.target_std as f32);
        self.eval_target_std.push(val.target_std as f32);
        self.train_terminal_mse.push(train.terminal_mse as f32);
        self.eval_terminal_mse.push(val.terminal_mse as f32);
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
        &epoch_dir.join("pretrain_return_mse.report.bin"),
        format!("Pretrain Return MSE - epoch {epoch} step {global_step}"),
        "target-scaled cumulative log return MSE",
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
        &epoch_dir.join("pretrain_probe_nll.report.bin"),
        format!("Pretrain Probe NLL - epoch {epoch} step {global_step}"),
        "detached probe NLL",
        &history.train_probe_nll,
        &history.eval_probe_nll,
    )?;
    write_pretrain_scalar_report(
        &epoch_dir.join("pretrain_return_mae.report.bin"),
        format!("Pretrain Return MAE - epoch {epoch} step {global_step}"),
        "target-scaled cumulative log return MAE",
        &history.train_return_mae,
        &history.eval_return_mae,
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
        &epoch_dir.join("pretrain_terminal_mse.report.bin"),
        format!("Pretrain Terminal MSE - epoch {epoch} step {global_step}"),
        "next patch terminal cumulative return MSE",
        &history.train_terminal_mse,
        &history.eval_terminal_mse,
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
    let mut nll_sum = 0.0;
    let mut mse_sum = 0.0;
    let mut mae_sum = 0.0;
    let mut pred_std_sum = 0.0;
    let mut target_std_sum = 0.0;
    let mut samples = 0usize;
    let mut batches = 0usize;

    for probe_epoch in 0..probe_epochs {
        sampler.start_epoch();
        while let Some(batch) = sampler.next_train_batch(batch_size) {
            let batch_samples = batch.len() as usize;
            let predicted_patch_embed =
                tch::no_grad(|| predict_lejepa_next_patch_belief(model, heads, &batch));
            let target = scaled_next_patch_cumulative_returns(&batch.next_patch, target_scale);
            let probe = probe_metrics(heads, &predicted_patch_embed.detach(), &target);
            assert_finite_loss(&probe.return_mse, probe_epoch + 1);
            probe_opt.zero_grad();
            probe.return_mse.backward();
            clip_all_grads(probe_named_vars, MAX_GRAD_NORM, device);
            probe_opt.step();

            nll_sum += probe.probe_nll.double_value(&[]) * batch_samples as f64;
            mse_sum += probe.return_mse.double_value(&[]) * batch_samples as f64;
            mae_sum += probe.return_mae.double_value(&[]) * batch_samples as f64;
            pred_std_sum += probe.pred_std.double_value(&[]) * batch_samples as f64;
            target_std_sum += probe.target_std.double_value(&[]) * batch_samples as f64;
            samples += batch_samples;
            batches += 1;
        }
    }

    assert!(samples > 0, "detached probe training set is empty");
    let denom = samples as f64;
    ProbeTrainSummary {
        probe_nll: nll_sum / denom,
        return_mae: mae_sum / denom,
        return_mse: mse_sum / denom,
        pred_std: pred_std_sum / denom,
        target_std: target_std_sum / denom,
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
        let mut return_mae_sum = 0.0;
        let mut return_mse_sum = 0.0;
        let mut pred_std_sum = 0.0;
        let mut target_std_sum = 0.0;
        let mut return_bias_sum = 0.0;
        let mut pred_abs_sum = 0.0;
        let mut target_abs_sum = 0.0;
        let mut next_lat_sum = 0.0;
        let mut terminal_mse_sum = 0.0;
        let mut zero_mse_sum = 0.0;
        let mut samples = 0usize;
        let mut tickers = 0usize;
        let mut batches = 0usize;

        for ticker in sampler.val_tickers.clone() {
            if max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }

            let mut env = Env::new_with_tickers_and_recording(vec![ticker], false, false, None);
            let offsets = build_split_offsets(
                env.price_deltas[0].len(),
                sampler.k_patches,
                sampler.patch_size,
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
                    &mut env,
                    chunk,
                    sampler.k_patches,
                    sampler.patch_size,
                    sampler.target_scale,
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
                    sampler.target_scale,
                    false,
                );
                let return_target = match objective {
                    PretrainObjective::MeanMse => cumulative_future_returns(&batch.future_patches),
                    PretrainObjective::Lejepa => scaled_next_patch_cumulative_returns(
                        &batch.next_patch,
                        sampler.target_scale,
                    ),
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
                return_mae_sum += losses.return_mae.double_value(&[]) * batch_samples as f64;
                return_mse_sum += losses.return_mse.double_value(&[]) * batch_samples as f64;
                pred_std_sum += losses.pred_std.double_value(&[]) * batch_samples as f64;
                target_std_sum += losses.target_std.double_value(&[]) * batch_samples as f64;
                return_bias_sum += losses.return_bias.double_value(&[]) * batch_samples as f64;
                pred_abs_sum += losses.pred_abs.double_value(&[]) * batch_samples as f64;
                target_abs_sum += losses.target_abs.double_value(&[]) * batch_samples as f64;
                next_lat_sum += losses.next_lat.double_value(&[]) * batch_samples as f64;
                terminal_mse_sum += losses.terminal_mse.double_value(&[]) * batch_samples as f64;
                zero_mse_sum += zero_mse_loss.double_value(&[]) * batch_samples as f64;
                samples += batch_samples;
                batches += 1;
            }
        }

        assert!(samples > 0, "validation set is empty");
        ValidationLoss {
            total: total_sum / samples as f64,
            jepa_mse: jepa_mse_sum / samples as f64,
            sigreg: sigreg_sum / samples as f64,
            repr_std_mean: repr_std_mean_sum / samples as f64,
            repr_std_min: repr_std_min_sum / samples as f64,
            pred_embed_std: pred_embed_std_sum / samples as f64,
            target_embed_std: target_embed_std_sum / samples as f64,
            probe_nll: probe_nll_sum / samples as f64,
            return_mae: return_mae_sum / samples as f64,
            return_mse: return_mse_sum / samples as f64,
            pred_std: pred_std_sum / samples as f64,
            target_std: target_std_sum / samples as f64,
            return_bias: return_bias_sum / samples as f64,
            pred_abs: pred_abs_sum / samples as f64,
            target_abs: target_abs_sum / samples as f64,
            next_lat: next_lat_sum / samples as f64,
            terminal_mse: terminal_mse_sum / samples as f64,
            zero_mse: zero_mse_sum / samples as f64,
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
        PretrainObjective::Lejepa => sampler.patch_size,
    };
    let mut abs_sum = vec![0.0f64; horizon];
    let mut sq_sum = vec![0.0f64; horizon];
    let mut bias_sum = vec![0.0f64; horizon];
    let mut count = 0usize;
    let mut first_traces = Vec::new();
    let mut worst_traces: Vec<DiagnosticTrace> = Vec::new();

    tch::no_grad(|| -> Result<()> {
        let mut batches = 0usize;
        for ticker in sampler.val_tickers.clone() {
            if max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }
            let mut env =
                Env::new_with_tickers_and_recording(vec![ticker.clone()], false, false, None);
            let offsets = build_split_offsets(
                env.price_deltas[0].len(),
                sampler.k_patches,
                sampler.patch_size,
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
                    &mut env,
                    chunk,
                    sampler.k_patches,
                    sampler.patch_size,
                    sampler.target_scale,
                    device,
                );
                let (pred, actual_returns) = match objective {
                    PretrainObjective::MeanMse => (
                        predict_future_returns(model, heads, &batch),
                        cumulative_future_returns(&batch.future_patches),
                    ),
                    PretrainObjective::Lejepa => {
                        let predicted_patch_embed =
                            predict_lejepa_next_patch_belief(model, heads, &batch);
                        (
                            heads.probe_return_mean(&predicted_patch_embed),
                            scaled_next_patch_cumulative_returns(
                                &batch.next_patch,
                                sampler.target_scale,
                            ),
                        )
                    }
                };
                let actual = tensor_to_vec_f32(&actual_returns)?;
                let predicted = tensor_to_vec_f32(&pred)?;

                for (sample_idx, &offset) in chunk.iter().enumerate() {
                    let start = sample_idx * horizon;
                    let end = start + horizon;
                    let actual_sample = &actual[start..end];
                    let pred_sample = &predicted[start..end];
                    let mut sample_abs = 0.0;

                    for h in 0..horizon {
                        let err = pred_sample[h] as f64 - actual_sample[h] as f64;
                        abs_sum[h] += err.abs();
                        sq_sum[h] += err * err;
                        bias_sum[h] += err;
                        sample_abs += err.abs();
                    }
                    count += 1;
                    let loss = sample_abs / horizon as f64;
                    let trace = DiagnosticTrace {
                        label: format!("{}_offset_{}", ticker, offset),
                        loss,
                        actual: actual_sample.to_vec(),
                        predicted: pred_sample.to_vec(),
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
            y_label: Some("target-scaled cumulative log return".to_string()),
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

    Ok(())
}

fn write_trace_reports(
    dir: &Path,
    prefix: &str,
    group: &str,
    epoch: usize,
    global_step: usize,
    trace: &DiagnosticTrace,
) -> Result<()> {
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
    Ok(())
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

fn normalize_last_dim(x: &Tensor) -> Tensor {
    let mean = x.mean_dim([-1].as_slice(), true, Kind::Float);
    let centered = x - &mean;
    let var = centered
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    centered / (var + 1e-5).sqrt()
}

fn normalize_feature_batch(x: &Tensor) -> Tensor {
    let mean = x.mean_dim([0i64].as_slice(), true, Kind::Float);
    let centered = x - &mean;
    let var = centered
        .pow_tensor_scalar(2.0)
        .mean_dim([0i64].as_slice(), true, Kind::Float);
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
        build_split_offsets, cumulative_future_returns, future_patches_for_current_perm, SplitKind,
    };
    use crate::torch::{
        constants::PRICE_DELTAS_PER_TICKER,
        env::Env,
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
}
