//! Causal-reader population SIGReg, temporal attached-target LeJEPA and fixed conditional prediction.
//! A view is a position; its population is the independent batch axis, never flattened time.
use anyhow::{ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use tch::{nn, Device, Kind, Tensor};

use super::{
    corpus::Batch,
    model::{linear, projection, ModelConfig, Statistics, CHANNELS},
};
use crate::torch::cuda;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ReaderNorm {
    #[default]
    Rms,
    None,
}
impl ReaderNorm {
    pub fn is_rms(&self) -> bool {
        *self == Self::Rms
    }
}
impl fmt::Display for ReaderNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Rms => "rms",
            Self::None => "none",
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SigregPlacement {
    #[default]
    Off,
    Local,
    State,
    Both,
}
impl SigregPlacement {
    pub fn enabled(self) -> bool {
        self != Self::Off
    }
    pub fn is_off(&self) -> bool {
        !self.enabled()
    }
    pub fn local(self) -> bool {
        matches!(self, Self::Local | Self::Both)
    }
    pub fn state(self) -> bool {
        matches!(self, Self::State | Self::Both)
    }
}
impl fmt::Display for SigregPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Local => "local",
            Self::State => "state",
            Self::Both => "both",
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum JepaMode {
    #[default]
    Off,
    LatentOne,
    LatentMulti,
    Anchored,
    AnchoredNoSigreg,
    AnchoredReconstruct,
    AnchoredProjected,
    AnchoredProjectedNoSigreg,
    /// Predict per-offset changes in the projected target, and SIGReg those changes.
    AnchoredTemporalProjected,
    /// Control for temporal projected prediction without SIGReg.
    AnchoredTemporalProjectedNoSigreg,
    AnchoredConditional,
    AnchoredProjectedSmall,
    /// Attached temporal prediction only; reader placement controls every regularizer.
    Unanchored,
}
impl JepaMode {
    pub fn enabled(self) -> bool {
        self != Self::Off
    }
    pub fn is_off(&self) -> bool {
        !self.enabled()
    }
    pub fn unanchored(self) -> bool {
        self == Self::Unanchored
    }
    pub fn detached_forecast(self) -> bool {
        matches!(self, Self::LatentOne | Self::LatentMulti)
    }
    pub fn regularized(self) -> bool {
        self.enabled()
            && !matches!(
                self,
                Self::AnchoredNoSigreg
                    | Self::AnchoredProjectedNoSigreg
                    | Self::AnchoredTemporalProjectedNoSigreg
                    | Self::AnchoredConditional
                    | Self::Unanchored
            )
    }
    pub fn projected(self) -> bool {
        matches!(
            self,
            Self::AnchoredProjected
                | Self::AnchoredProjectedNoSigreg
                | Self::AnchoredTemporalProjected
                | Self::AnchoredTemporalProjectedNoSigreg
                | Self::AnchoredProjectedSmall
        )
    }
    pub fn temporal(self) -> bool {
        matches!(
            self,
            Self::AnchoredTemporalProjected | Self::AnchoredTemporalProjectedNoSigreg
        )
    }
    pub fn conditional(self) -> bool {
        self == Self::AnchoredConditional
    }
    pub fn target_width(self, d_model: i64) -> i64 {
        if self == Self::AnchoredProjectedSmall {
            16
        } else {
            d_model
        }
    }
    /// Legacy nonregularized modes retain their sampled population diagnostics and RNG stream.
    pub fn needs_random(self) -> bool {
        self.enabled() && !self.conditional()
    }
    pub fn offsets(self) -> &'static [i64] {
        if self == Self::LatentOne {
            &[1]
        } else {
            &[1, 2, 4, 8, 12]
        }
    }
}
impl fmt::Display for JepaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::LatentOne => "latent-one",
            Self::LatentMulti => "latent-multi",
            Self::Anchored => "anchored",
            Self::AnchoredNoSigreg => "anchored-no-sigreg",
            Self::AnchoredReconstruct => "anchored-reconstruct",
            Self::AnchoredProjected => "anchored-projected",
            Self::AnchoredProjectedNoSigreg => "anchored-projected-no-sigreg",
            Self::AnchoredTemporalProjected => "anchored-temporal-projected",
            Self::AnchoredTemporalProjectedNoSigreg => "anchored-temporal-projected-no-sigreg",
            Self::AnchoredConditional => "anchored-conditional",
            Self::AnchoredProjectedSmall => "anchored-projected-small",
            Self::Unanchored => "unanchored",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
pub struct JepaConfig {
    #[arg(
        id = "jepa_predictor_width",
        long = "jepa-predictor-width",
        default_value_t = 256
    )]
    pub predictor_width: i64,
    #[arg(
        id = "jepa_prediction_weight",
        long = "jepa-prediction-weight",
        default_value_t = 1.0
    )]
    pub prediction_weight: f64,
    #[arg(
        id = "jepa_sigreg_weight",
        long = "jepa-sigreg-weight",
        default_value_t = 0.09
    )]
    pub sigreg_weight: f64,
    #[arg(
        id = "jepa_reconstruction_weight",
        long = "jepa-reconstruction-weight",
        default_value_t = 1.0
    )]
    pub reconstruction_weight: f64,
    #[arg(
        id = "jepa_directions",
        long = "jepa-directions",
        default_value_t = 256
    )]
    pub directions: i64,
    #[arg(id = "jepa_views", long = "jepa-views", default_value_t = 8)]
    pub views: i64,
    /// Independent host RNG stream, not libtorch's or the training loader's generator.
    #[arg(id = "jepa_seed", long = "jepa-seed", default_value_t = 0)]
    pub seed: u64,
}
impl Default for JepaConfig {
    fn default() -> Self {
        Self {
            predictor_width: 256,
            prediction_weight: 1.,
            sigreg_weight: 0.09,
            reconstruction_weight: 1.,
            directions: 256,
            views: 8,
            seed: 0,
        }
    }
}
impl JepaConfig {
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
    pub fn validate(&self, config: &ModelConfig) -> Result<()> {
        ensure!(
            (1..=2048).contains(&self.predictor_width),
            "JEPA predictor width must be in 1..=2048"
        );
        ensure!(
            (1..=1024).contains(&self.directions),
            "JEPA directions must be in 1..=1024"
        );
        ensure!(
            (1..=8).contains(&self.views),
            "JEPA sampled views must be in 1..=8"
        );
        for (name, value) in [
            ("prediction", self.prediction_weight),
            ("SIGReg", self.sigreg_weight),
            ("reconstruction", self.reconstruction_weight),
        ] {
            ensure!(
                value.is_finite() && value >= 0.,
                "JEPA {name} weight must be finite and nonnegative"
            );
        }
        if config.jepa_mode.enabled() {
            let last = *config.jepa_mode.offsets().last().unwrap();
            let first = (config.min_history + config.patch_len - 1) / config.patch_len - 1;
            ensure!(config.origins() > first + last, "JEPA context must contain a minimum-history source and its furthest nonoverlapping target patch");
            ensure!(
                self.prediction_weight > 0.,
                "an enabled JEPA arm needs a positive prediction weight"
            );
            if config.jepa_mode.regularized() {
                ensure!(
                    self.sigreg_weight > 0.,
                    "use anchored-no-sigreg or anchored-projected-no-sigreg to disable SIGReg explicitly"
                );
            }
            if config.jepa_mode == JepaMode::AnchoredReconstruct {
                ensure!(
                    self.reconstruction_weight > 0.,
                    "anchored-reconstruct needs positive reconstruction weight"
                );
            }
            if config.jepa_mode.unanchored() {
                ensure!(
                    self.prediction_weight == 1.
                        && self.sigreg_weight == 0.09
                        && self.reconstruction_weight == 0.,
                    "unanchored requires prediction weight 1, total SIGReg weight 0.09 (placement off disables it), and reconstruction weight 0"
                );
            }
        }
        if config.sigreg_placement.enabled() {
            ensure!(
                self.sigreg_weight > 0.,
                "enabled reader SIGReg needs positive weight; use --sigreg-placement off for the control"
            );
        }
        Ok(())
    }
}

pub const CONDITIONAL_FREQUENCIES: [f32; 5] = [0.25, 0.5, 1., 2., 4.];
pub const CONDITIONAL_FEATURES: i64 = 10;

/// Fixed source-anchored characteristic targets, not an output of any learned encoder.
pub struct ConditionalTargets {
    /// fp32 `[B,O-12,K,10]`, interleaved `[cos(w0*y),sin(w0*y),...]`.
    pub values: Tensor,
    /// fp32 `[B,O-12,K]`: complete source patch/history AND complete future interval.
    pub mask: Tensor,
}

/// Local observations are the unconstrained patch embeddings entering the trunk. Their input uses
/// causal prefix normalization, so they are NOT pure raw-local observations. `state` is causal.
/// Predictions share sources `0..O-Kmax`, with horizon axis ordered as `horizons` (bar units).
pub struct RepresentationViews {
    /// `[B,O,d_model]` before gainless RMSNorm, same width/encoder in every arm.
    pub observation: Tensor,
    /// `[B,O,target_width]` attached latent targets: q(observation) in projected modes, otherwise
    /// the observation itself. The deterministic pointwise q is never an input to the trunk.
    pub target: Tensor,
    /// `[B,O,d_model]` actual causal forecast-reader states, with `reader_norm` applied.
    pub state: Tensor,
    /// `[B,O-Kmax,K,target_width]`, or last dimension 10 for conditional characteristic prediction.
    pub prediction: Option<Tensor>,
    /// `[B,O,patch_len*4]` fixed normalized price features, never exogenous coordinates.
    /// Undefined only inside unanchored training; public views retain these for frozen readers.
    pub reconstruction_target: Tensor,
    pub conditional: Option<ConditionalTargets>,
    pub horizons: Vec<i64>,
}

/// Legacy diagnostic slots in `Losses::jepa`, detached fp32 [8]. No term is forecast NLL.
/// 0 latent MSE; 1 population SIGReg; 2 price-token reconstruction MSE;
/// 3 latent persistence MSE; 4 valid source-target pairs; 5 mean valid population N per view;
/// 6 target population std at sampled positions; 7 predictor population std at the last
/// available source, averaged over its direct horizons.
/// Conditional mode uses slots 0/3 for CF MSE/psi(0) persistence, leaves slots 1/2 unused,
/// and reports batch population statistics for the fixed ten-dimensional characteristic targets.
pub const DIAGNOSTIC_COUNT: i64 = 8;
/// Unanchored preserves the latent eight slots and appends the independent reader packet.
/// Slots 1/2 are zero unavailable target-SIGReg/reconstruction terms, never overwritten.
/// Slots 8..16 are local SIGReg, weighted placement sum, state SIGReg, eligible views,
/// row-view pairs, mean N, local std and state std, including population diagnostics for off.
pub const UNANCHORED_DIAGNOSTIC_COUNT: i64 = 16;
/// Reader placement uses slots 0/2 for raw local/state SIGReg (zero when inactive), 1 for
/// the weighted contribution, 3/4/5 for eligible views/valid row-view pairs/mean N, and
/// 6/7 for local/state population standard deviation. No latent prediction loss is present.
pub fn diagnostic_labels(config: &ModelConfig) -> Vec<&'static str> {
    if config.jepa_mode.unanchored() {
        return vec![
            "latent MSE",
            "unused target SIGReg",
            "unused reconstruction",
            "latent persistence MSE",
            "valid source-target pairs",
            "latent sampled population N",
            "observation population std",
            "prediction population std (last source)",
            "reader local SIGReg",
            "reader SIGReg weighted contribution",
            "reader state SIGReg",
            "reader SIGReg eligible views (N >= 2)",
            "reader SIGReg valid row-view pairs",
            "reader SIGReg mean valid batch population per view",
            "reader local population std",
            "reader state population std",
        ];
    }
    if config.sigreg_placement.enabled() {
        return vec![
            if config.sigreg_placement.local() {
                "reader local SIGReg"
            } else {
                "unused local SIGReg"
            },
            "reader SIGReg weighted contribution",
            if config.sigreg_placement.state() {
                "reader state SIGReg"
            } else {
                "unused state SIGReg"
            },
            "reader SIGReg eligible views (N >= 2)",
            "reader SIGReg valid row-view pairs",
            "reader SIGReg mean valid batch population per view",
            "reader local population std",
            "reader state population std",
        ];
    }
    let mode = config.jepa_mode;
    if mode.conditional() {
        return vec![
            "conditional CF MSE",
            "unused SIGReg",
            "unused reconstruction",
            "conditional CF persistence MSE (psi(0))",
            "conditional CF valid source-horizon pairs",
            "conditional CF mean valid batch population per source/horizon",
            "conditional CF target population std (last source)",
            "conditional CF prediction population std (last source)",
        ];
    }
    let mut labels = vec![
        "latent MSE",
        "population SIGReg",
        "price-token reconstruction MSE",
        "latent persistence MSE",
        "valid source-target pairs",
        "SIGReg population N",
        if mode.projected() {
            "projected-target population std"
        } else {
            "observation population std"
        },
        "prediction population std (last source)",
    ];
    if mode.temporal() {
        labels[0] = "temporal projected-target delta MSE";
        labels[1] = "temporal projected-target delta SIGReg";
        labels[3] = "temporal projected-target delta persistence MSE";
        labels[6] = "temporal projected-target delta population std";
    }
    if config.decision_sign_enabled() {
        labels.push("decision sign logistic loss");
    }
    labels
}

struct ConditionalGeometry {
    frequencies: Tensor,
    horizon_scale: Tensor,
    persistence: Tensor,
}

pub(super) struct JepaHeads {
    predictor_hidden: nn::Linear,
    predictor: nn::Linear,
    reconstruction: Option<nn::Linear>,
    target_projector: Option<(nn::Linear, nn::Linear)>,
    output_dim: i64,
    offsets: &'static [i64],
    conditional: Option<ConditionalGeometry>,
}
impl JepaHeads {
    pub fn new(path: nn::Path, config: &ModelConfig) -> Self {
        let z = config.d_model;
        let hidden = config.jepa.predictor_width;
        let output_dim = if config.jepa_mode.conditional() {
            CONDITIONAL_FEATURES
        } else {
            config.jepa_mode.target_width(z)
        };
        Self {
            predictor_hidden: projection(&path / "predictor_hidden", z, hidden, true),
            predictor: projection(
                &path / "predictor",
                hidden,
                output_dim * config.jepa_mode.offsets().len() as i64,
                true,
            ),
            reconstruction: (config.jepa_mode == JepaMode::AnchoredReconstruct).then(|| {
                projection(
                    &path / "reconstruction",
                    z,
                    config.patch_len * CHANNELS,
                    true,
                )
            }),
            // Allocate only after the existing heads, preserving their initialization stream.
            target_projector: config.jepa_mode.projected().then(|| {
                (
                    projection(&path / "target_hidden", z, z, true),
                    projection(&path / "target_output", z, output_dim, true),
                )
            }),
            output_dim,
            offsets: config.jepa_mode.offsets(),
            conditional: config.jepa_mode.conditional().then(|| {
                let scale: Vec<f32> = config
                    .jepa_horizons()
                    .iter()
                    .map(|&h| (h as f32).sqrt())
                    .collect();
                ConditionalGeometry {
                    frequencies: Tensor::from_slice(&CONDITIONAL_FREQUENCIES)
                        .to_device(path.device()),
                    horizon_scale: Tensor::from_slice(&scale).to_device(path.device()),
                    persistence: Tensor::from_slice(&[1f32, 0., 1., 0., 1., 0., 1., 0., 1., 0.])
                        .to_device(path.device()),
                }
            }),
        }
    }
    pub fn target(&self, observation: &Tensor) -> Tensor {
        match &self.target_projector {
            Some((hidden, output)) => linear(&linear(observation, hidden).gelu("tanh"), output),
            None => observation.shallow_clone(),
        }
    }
    pub fn prediction(&self, state: &Tensor) -> Tensor {
        let sources = state.size()[1] - self.offsets.last().unwrap();
        linear(
            &linear(&state.narrow(1, 0, sources), &self.predictor_hidden).gelu("tanh"),
            &self.predictor,
        )
        .reshape([
            state.size()[0],
            sources,
            self.offsets.len() as i64,
            self.output_dim,
        ])
    }
    pub fn conditional_targets(
        &self,
        config: &ModelConfig,
        batch: &Batch,
        stats: &Statistics,
    ) -> Option<ConditionalTargets> {
        let geometry = self.conditional.as_ref()?;
        Some(tch::no_grad(|| {
            let sources = config.origins() - self.offsets.last().unwrap();
            // Origin closes/market levels are strided data views, not normalized future tokens.
            let selected = |values: &Tensor| {
                Tensor::stack(
                    &self
                        .offsets
                        .iter()
                        .map(|&k| values.narrow(1, k, sources))
                        .collect::<Vec<_>>(),
                    2,
                )
            };
            let source = |values: &Tensor| values.narrow(1, 0, sources).unsqueeze(-1);
            let neutral = selected(&stats.log_close)
                - source(&stats.log_close)
                - source(&stats.beta) * (selected(&stats.market) - source(&stats.market));
            let y = neutral / (source(&stats.sigma) * &geometry.horizon_scale);
            let phases = y.unsqueeze(-1) * &geometry.frequencies;
            let values = Tensor::stack(&[phases.cos(), phases.sin()], -1).flatten(-2, -1);

            // Prefix INVALID counts make every t+1..=t+h bar load-bearing without constructing
            // dense [B,O,4,192] targets or even [B,O,192] validity windows.
            let valid = batch.valid.narrow(1, 0, config.seq_len);
            let invalid = valid
                .eq(0.)
                .to_kind(Kind::Float)
                .cumsum(1, Kind::Float)
                .reshape([-1, config.origins(), config.patch_len])
                .select(2, config.patch_len - 1);
            let interval = (selected(&invalid) - source(&invalid))
                .eq(0.)
                .to_kind(Kind::Float);
            let source_valid = valid
                .reshape([-1, config.origins(), config.patch_len])
                .narrow(1, 0, sources)
                .amin([-1i64].as_slice(), false)
                * stats.mask.narrow(1, 0, sources);
            ConditionalTargets {
                values,
                mask: interval * source_valid.unsqueeze(-1),
            }
        }))
    }
    pub fn conditional_objective(
        &self,
        config: &ModelConfig,
        views: &RepresentationViews,
        targets: &ConditionalTargets,
    ) -> (Tensor, Tensor) {
        let geometry = self.conditional.as_ref().expect("conditional geometry");
        let prediction = views
            .prediction
            .as_ref()
            .expect("conditional predictor")
            .to_kind(Kind::Float);
        let mse = masked_mse(&prediction, &targets.values, &targets.mask);
        let objective = &mse * config.jepa.prediction_weight;
        let diagnostics = tch::no_grad(|| {
            let last = prediction.size()[1] - 1;
            let valid = targets.mask.select(1, last).transpose(0, 1);
            let zero = Tensor::zeros([], (Kind::Float, prediction.device()));
            Tensor::stack(
                &[
                    mse.detach(),
                    zero.shallow_clone(),
                    zero,
                    masked_mse(&geometry.persistence, &targets.values, &targets.mask),
                    targets.mask.sum(Kind::Float),
                    targets
                        .mask
                        .sum_dim_intlist([0i64].as_slice(), false, Kind::Float)
                        .mean(Kind::Float),
                    population_std(&targets.values.select(1, last).transpose(0, 1), &valid),
                    population_std(&prediction.select(1, last).transpose(0, 1), &valid),
                ],
                0,
            )
        });
        (objective, diagnostics)
    }
    pub fn objective(
        &self,
        config: &ModelConfig,
        views: &RepresentationViews,
        valid: &Tensor,
        source_valid: &Tensor,
        random: Option<&JepaRandom>,
    ) -> (Tensor, Tensor) {
        if let Some(targets) = &views.conditional {
            return self.conditional_objective(config, views, targets);
        }
        assert!(
            !config.jepa_mode.regularized() || random.is_some(),
            "SIGReg needs the uploaded population draw"
        );
        let prediction = views
            .prediction
            .as_ref()
            .expect("enabled JEPA predictor")
            .to_kind(Kind::Float);
        let sources = prediction.size()[1];
        // Attached targets: the observation encoder (and optional q) receives target gradients.
        let mut targets = Tensor::stack(
            &self
                .offsets
                .iter()
                .map(|&k| views.target.narrow(1, k, sources))
                .collect::<Vec<_>>(),
            2,
        )
        .to_kind(Kind::Float);
        if config.jepa_mode.temporal() {
            let source = views.target.narrow(1, 0, sources).unsqueeze(2);
            targets = &targets - &source;
        }
        let mask = pair_mask(valid, source_valid, self.offsets);
        let count = mask.sum(Kind::Float);
        let latent = masked_mse(&prediction, &targets, &mask);
        let zero = Tensor::zeros([], (Kind::Float, prediction.device()));
        if config.jepa_mode.unanchored() {
            let (regularizer, reader) = reader_sigreg_objective(
                config,
                &views.observation,
                &views.state,
                source_valid,
                random.expect("unanchored shares the population draw in all four placements"),
            );
            let objective = &latent * config.jepa.prediction_weight + regularizer;
            let diagnostics = tch::no_grad(|| {
                let persistence = masked_mse(
                    &views
                        .target
                        .narrow(1, 0, sources)
                        .unsqueeze(2)
                        .to_kind(Kind::Float),
                    &targets,
                    &mask,
                );
                let pred_std = population_std(
                    &prediction.select(1, sources - 1).transpose(0, 1),
                    &mask.select(1, sources - 1).transpose(0, 1),
                );
                let latent = Tensor::stack(
                    &[
                        latent.detach(),
                        zero.shallow_clone(),
                        zero,
                        persistence,
                        count,
                        reader.select(0, 5),
                        reader.select(0, 6),
                        pred_std,
                    ],
                    0,
                );
                Tensor::cat(&[latent, reader], 0)
            });
            return (objective, diagnostics);
        }
        let (regularizer, population, target_std) = match random {
            Some(random) if config.jepa_mode.temporal() => {
                let source = views
                    .target
                    .index_select(1, &random.positions)
                    .to_kind(Kind::Float);
                let source_valid = source_valid
                    .index_select(1, &random.positions)
                    .transpose(0, 1);
                let mut regularizer = zero.shallow_clone();
                let mut population = zero.shallow_clone();
                let mut target_std = zero.shallow_clone();
                for &offset in self.offsets {
                    let positions = &random.positions + offset;
                    let valid = (&source_valid * valid.index_select(1, &positions).transpose(0, 1))
                        .to_kind(Kind::Float);
                    let target_views = views
                        .target
                        .index_select(1, &positions)
                        .to_kind(Kind::Float)
                        - &source;
                    let target_views = target_views.transpose(0, 1);
                    let (reg, pop) = if config.jepa_mode.regularized() {
                        population_sigreg(&target_views, &valid, random)
                    } else {
                        (
                            zero.shallow_clone(),
                            valid
                                .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                                .mean(Kind::Float),
                        )
                    };
                    regularizer = &regularizer + reg;
                    population = &population + pop;
                    target_std =
                        &target_std + tch::no_grad(|| population_std(&target_views, &valid));
                }
                let count = self.offsets.len() as f64;
                (regularizer / count, population / count, target_std / count)
            }
            Some(random) => {
                let target_views = views
                    .target
                    .index_select(1, &random.positions)
                    .transpose(0, 1)
                    .to_kind(Kind::Float);
                let valid = source_valid
                    .index_select(1, &random.positions)
                    .transpose(0, 1);
                let (reg, population) = if config.jepa_mode.regularized() {
                    population_sigreg(&target_views, &valid, random)
                } else {
                    (
                        zero.shallow_clone(),
                        valid
                            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                            .mean(Kind::Float),
                    )
                };
                (
                    reg,
                    population,
                    tch::no_grad(|| population_std(&target_views, &valid)),
                )
            }
            None => (
                zero.shallow_clone(),
                zero.shallow_clone(),
                zero.shallow_clone(),
            ),
        };
        let reconstruction = match &self.reconstruction {
            Some(decoder) => masked_mse(
                &linear(&views.observation, decoder).to_kind(Kind::Float),
                &views.reconstruction_target,
                source_valid,
            ),
            None => zero,
        };
        let objective = &latent * config.jepa.prediction_weight
            + &regularizer
                * if config.jepa_mode.regularized() {
                    config.jepa.sigreg_weight
                } else {
                    0.
                }
            + &reconstruction * config.jepa.reconstruction_weight;
        let diagnostics = tch::no_grad(|| {
            let persistence = if config.jepa_mode.temporal() {
                masked_mse(&targets.zeros_like(), &targets, &mask)
            } else {
                masked_mse(
                    &views
                        .target
                        .narrow(1, 0, sources)
                        .unsqueeze(2)
                        .to_kind(Kind::Float),
                    &targets,
                    &mask,
                )
            };
            let pred_std = population_std(
                &prediction.select(1, sources - 1).transpose(0, 1),
                &mask.select(1, sources - 1).transpose(0, 1),
            );
            Tensor::stack(
                &[
                    latent.detach(),
                    regularizer.detach(),
                    reconstruction.detach(),
                    persistence,
                    count,
                    population,
                    target_std,
                    pred_std,
                ],
                0,
            )
        });
        (objective, diagnostics)
    }
}

pub(super) fn pair_mask(valid: &Tensor, source_valid: &Tensor, offsets: &[i64]) -> Tensor {
    let sources = valid.size()[1] - offsets.last().unwrap();
    Tensor::stack(
        &offsets
            .iter()
            .map(|&k| valid.narrow(1, k, sources))
            .collect::<Vec<_>>(),
        2,
    ) * source_valid.narrow(1, 0, sources).unsqueeze(-1)
}
pub(super) fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    ((prediction - target)
        .square()
        .mean_dim([-1i64].as_slice(), false, Kind::Float)
        * mask)
        .sum(Kind::Float)
        / mask.sum(Kind::Float).clamp_min(1.)
}
fn population_std(values: &Tensor, valid: &Tensor) -> Tensor {
    let count = valid
        .sum_dim_intlist([1i64].as_slice(), true, Kind::Float)
        .clamp_min(1.);
    let weight = valid.unsqueeze(-1);
    let mean = (values * &weight).sum_dim_intlist([1i64].as_slice(), true, Kind::Float)
        / count.unsqueeze(-1);
    let variance =
        ((values - mean).square() * weight).sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            / &count;
    variance.mean(Kind::Float).sqrt()
}

/// Independent host draws uploaded into fixed-address operands BEFORE graph replay.
/// Drawing normal sphere directions on the host never advances CUDA or loader RNG state.
pub(super) struct JepaRandom {
    pub directions: Tensor,
    pub positions: Tensor,
    knots: Tensor,
    normal_ecf: Tensor,
    coefficients: Tensor,
    rng: ChaCha8Rng,
    values: Vec<f32>,
    candidates: Vec<i64>,
    device: Device,
}
impl JepaRandom {
    pub fn new(config: &ModelConfig, device: Device) -> Self {
        let first = (config.min_history + config.patch_len - 1) / config.patch_len - 1;
        // Temporal projected targets need every sampled source plus its furthest future
        // offset. Other modes retain the legacy full source population, including the final
        // context patch.
        let last_exclusive = if config.jepa_mode.temporal() {
            config.origins() - config.jepa_mode.offsets().last().copied().unwrap()
        } else {
            config.origins()
        };
        let views = config.jepa.views.min(last_exclusive - first);
        let width = config.jepa_mode.target_width(config.d_model);
        let knots = Tensor::linspace(0., 3., 17, (Kind::Float, device));
        let normal_ecf = (-knots.square() * 0.5).exp();
        let coefficients = Tensor::full([17], 0.375, (Kind::Float, device));
        let _ = coefficients.narrow(0, 0, 1).fill_(0.1875);
        let _ = coefficients.narrow(0, 16, 1).fill_(0.1875);
        Self {
            directions: Tensor::zeros([width, config.jepa.directions], (Kind::Float, device)),
            positions: Tensor::zeros([views], (Kind::Int64, device)),
            coefficients: coefficients * &normal_ecf,
            knots,
            normal_ecf,
            rng: ChaCha8Rng::seed_from_u64(config.jepa.seed ^ 0x4a45_5041_5349_4752),
            values: vec![0.; (width * config.jepa.directions) as usize],
            candidates: (first..last_exclusive).collect(),
            device,
        }
    }
    pub fn refresh(&mut self) -> Result<()> {
        let shape = self.directions.size();
        let (width, projections) = (shape[0] as usize, shape[1] as usize);
        let values = &mut self.values;
        for p in 0..projections {
            let mut norm = 0.;
            for d in (0..width).step_by(2) {
                let radius = (-2. * (1. - self.rng.random::<f64>()).ln()).sqrt();
                let (sin, cos) = (std::f64::consts::TAU * self.rng.random::<f64>()).sin_cos();
                let first = radius * cos;
                values[d * projections + p] = first as f32;
                norm += first * first;
                if d + 1 < width {
                    let second = radius * sin;
                    values[(d + 1) * projections + p] = second as f32;
                    norm += second * second;
                }
            }
            let norm = norm.sqrt() as f32;
            for d in 0..width {
                values[d * projections + p] /= norm;
            }
        }
        let candidates = &mut self.candidates;
        let views = self.positions.size()[0] as usize;
        for i in 0..views {
            let selected = self.rng.random_range(i..candidates.len());
            candidates.swap(i, selected);
        }
        let staged = |tensor: Tensor| {
            if self.device.is_cuda() {
                tensor.pin_memory(self.device)
            } else {
                tensor
            }
        };
        cuda::copy_nonblocking(
            &mut self.directions,
            &staged(Tensor::from_slice(values).reshape(shape)),
        )
        .map_err(|err| anyhow::anyhow!("uploading JEPA directions: {err}"))?;
        cuda::copy_nonblocking(
            &mut self.positions,
            &staged(Tensor::from_slice(&candidates[..views])),
        )
        .map_err(|err| anyhow::anyhow!("uploading JEPA positions: {err}"))
    }
}

/// Epps-Pulley statistic at each view independently. Invalid rows are absent from that view's
/// population; N is its actual valid batch count. Views with N<2 contribute no statistic.
pub(super) fn population_sigreg(
    embeddings: &Tensor,
    valid: &Tensor,
    random: &JepaRandom,
) -> (Tensor, Tensor) {
    let count = valid.sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
    let denominator = count.clamp_min(1.).reshape([-1, 1, 1]);
    let phase =
        embeddings.matmul(&random.directions).unsqueeze(-1) * random.knots.reshape([1, 1, 1, 17]);
    let weight = valid.unsqueeze(-1).unsqueeze(-1);
    let real = (phase.cos() * &weight).sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
        / &denominator
        - &random.normal_ecf;
    let imag =
        (phase.sin() * weight).sum_dim_intlist([1i64].as_slice(), false, Kind::Float) / denominator;
    let integrated = ((real.square() + imag.square()) * &random.coefficients)
        .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
        .mean_dim([-1i64].as_slice(), false, Kind::Float);
    let eligible = count.ge(2.).to_kind(Kind::Float);
    let loss = (integrated * &count * &eligible).sum(Kind::Float)
        / eligible.sum(Kind::Float).clamp_min(1.);
    (loss, count.mean(Kind::Float))
}

/// Regularize the actual shared interfaces, without an auxiliary head or a temporal population.
/// Both placements use the exact same directions, source positions and complete-history mask.
pub(super) fn reader_sigreg_objective(
    config: &ModelConfig,
    observation: &Tensor,
    state: &Tensor,
    source_valid: &Tensor,
    random: &JepaRandom,
) -> (Tensor, Tensor) {
    assert!(config.sigreg_placement.enabled() || config.jepa_mode.unanchored());
    assert_eq!(config.reader_norm, ReaderNorm::None);
    let sample = |values: &Tensor| {
        values
            .index_select(1, &random.positions)
            .transpose(0, 1)
            .to_kind(Kind::Float)
    };
    let valid = source_valid
        .index_select(1, &random.positions)
        .transpose(0, 1);
    let local_views = sample(observation);
    let state_views = sample(state);
    let zero = Tensor::zeros([], (Kind::Float, state.device()));
    let local = if config.sigreg_placement.local() {
        population_sigreg(&local_views, &valid, random).0
    } else {
        zero.shallow_clone()
    };
    let contextual = if config.sigreg_placement.state() {
        population_sigreg(&state_views, &valid, random).0
    } else {
        zero
    };
    let weight = config.jepa.sigreg_weight
        * if config.sigreg_placement == SigregPlacement::Both {
            0.5
        } else {
            1.
        };
    let objective = (&local + &contextual) * weight;
    let diagnostics = tch::no_grad(|| {
        let count = valid.sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
        Tensor::stack(
            &[
                local.detach(),
                objective.detach(),
                contextual.detach(),
                count.ge(2.).to_kind(Kind::Float).sum(Kind::Float),
                count.sum(Kind::Float),
                count.mean(Kind::Float),
                population_std(&local_views, &valid),
                population_std(&state_views, &valid),
            ],
            0,
        )
    });
    (objective, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<Device> {
        if std::env::var("TIMEXER_SEGMENT_GPU_TEST").as_deref() != Ok("1") {
            return None;
        }
        assert!(tch::Cuda::is_available());
        crate::torch::cuda::cfg::configure_cuda();
        Some(Device::Cuda(0))
    }

    fn config() -> ModelConfig {
        ModelConfig {
            seq_len: 256,
            pred_len: 16,
            patch_len: 16,
            min_history: 32,
            d_model: 8,
            heads: 2,
            jepa_mode: JepaMode::AnchoredNoSigreg,
            jepa: JepaConfig {
                predictor_width: 16,
                directions: 16,
                views: 2,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn temporal_projected_objective_uses_per_offset_target_deltas() {
        let config = ModelConfig {
            jepa_mode: JepaMode::AnchoredTemporalProjectedNoSigreg,
            ..config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let heads = JepaHeads::new(store.root(), &config);
        let observation = Tensor::randn([2, 16, 8], (Kind::Float, Device::Cpu));
        let target = heads.target(&observation);
        let sources = 4;
        let expected = Tensor::stack(
            &config
                .jepa_mode
                .offsets()
                .iter()
                .map(|&offset| target.narrow(1, offset, sources) - target.narrow(1, 0, sources))
                .collect::<Vec<_>>(),
            2,
        );
        let prediction = Tensor::zeros_like(&expected).set_requires_grad(true);
        let valid = Tensor::ones([2, 16], (Kind::Float, Device::Cpu));
        let views = RepresentationViews {
            prediction: Some(prediction),
            observation: observation.shallow_clone(),
            target,
            state: Tensor::zeros([2, 16, 8], (Kind::BFloat16, Device::Cpu)),
            reconstruction_target: Tensor::zeros([2, 16, 64], (Kind::Float, Device::Cpu)),
            conditional: None,
            horizons: config.jepa_horizons(),
        };
        let (objective, diagnostics) = heads.objective(&config, &views, &valid, &valid, None);
        let mask = Tensor::ones([2, sources, 5], (Kind::Float, Device::Cpu));
        let expected_loss = masked_mse(&Tensor::zeros_like(&expected), &expected, &mask);
        assert!((objective.double_value(&[]) - expected_loss.double_value(&[])).abs() < 1e-6);
        assert_eq!(
            diagnostics.double_value(&[3]),
            objective.double_value(&[]),
            "temporal persistence is the zero-delta baseline"
        );
    }

    #[test]
    fn cuda_jepa_targets_are_attached_and_invalid_or_short_history_pairs_have_zero_gradient() {
        let _rng = crate::torch::test_rng::exclusive();
        let Some(device) = gpu() else {
            return;
        };
        let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
        for mode in [
            JepaMode::AnchoredNoSigreg,
            JepaMode::AnchoredProjectedNoSigreg,
            JepaMode::AnchoredTemporalProjected,
            JepaMode::AnchoredTemporalProjectedNoSigreg,
            JepaMode::Unanchored,
        ] {
            let mut config = ModelConfig {
                jepa_mode: mode,
                ..config()
            };
            if mode.unanchored() {
                config.reader_norm = ReaderNorm::None;
                config.future_calendar = false;
                config.jepa.reconstruction_weight = 0.;
            }
            let mut random =
                (mode.regularized() || mode.unanchored()).then(|| JepaRandom::new(&config, device));
            if let Some(random) = &mut random {
                random.refresh().unwrap();
            }
            let store = nn::VarStore::new(device);
            let heads = JepaHeads::new(store.root(), &config);
            let observation =
                Tensor::randn([4, 16, 8], (Kind::Float, device)).set_requires_grad(true);
            let state = Tensor::randn([4, 16, 8], (Kind::BFloat16, device)).set_requires_grad(true);
            let valid = Tensor::ones([4, 16], (Kind::Float, device));
            let _ = valid.narrow(1, 2, 1).fill_(0.);
            let source_valid = valid.copy();
            let _ = source_valid.narrow(1, 0, 1).fill_(0.);
            let views = RepresentationViews {
                prediction: Some(heads.prediction(&state)),
                observation: observation.shallow_clone(),
                target: heads.target(&observation),
                state,
                reconstruction_target: Tensor::zeros([4, 16, 64], (Kind::Float, device)),
                conditional: None,
                horizons: config.jepa_horizons(),
            };
            let (objective, diagnostics) =
                heads.objective(&config, &views, &valid, &source_valid, random.as_ref());
            objective.backward();
            assert!(
                views.state.grad().abs().sum(Kind::Float).double_value(&[]) > 0.,
                "temporal predictor detached the source state"
            );
            let gradient = observation.grad();
            assert!(gradient.defined(), "future target encoder was detached");
            let invalid_prefix = if mode.temporal() {
                gradient.narrow(1, 0, 1)
            } else {
                gradient.narrow(1, 0, 3)
            };
            assert_eq!(
                invalid_prefix.abs().sum(Kind::Float).double_value(&[]),
                0.,
                "masked source/target positions must have zero target gradient"
            );
            assert!(
                gradient
                    .narrow(1, 3, 1)
                    .abs()
                    .sum(Kind::Float)
                    .double_value(&[])
                    > 0.
            );
            if let Some((hidden, output)) = &heads.target_projector {
                for weight in [&hidden.ws, &output.ws] {
                    let gradient = weight.grad();
                    assert!(gradient.defined(), "target projector was detached");
                    assert!(gradient.abs().sum(Kind::Float).double_value(&[]) > 0.);
                }
            }
            let expected_pairs = (0..4)
                .flat_map(|source| {
                    config
                        .jepa_mode
                        .offsets()
                        .iter()
                        .map(move |&k| (source, source + k))
                })
                .filter(|&(source, target)| source >= 1 && source != 2 && target != 2)
                .count()
                * 4;
            assert_eq!(diagnostics.double_value(&[4]), expected_pairs as f64);
        }
    }

    #[test]
    fn cuda_jepa_sigreg_uses_batch_population_and_never_pools_time_views() {
        let _rng = crate::torch::test_rng::exclusive();
        let Some(device) = gpu() else {
            return;
        };
        let mut random = JepaRandom::new(&config(), device);
        random.refresh().unwrap();
        let rows = Tensor::arange(64 * 8, (Kind::Float, device))
            .reshape([64, 8])
            .sin();
        let views = Tensor::stack(&[&rows + 3., &rows - 3.], 0);
        let valid = Tensor::ones([2, 64], (Kind::Float, device));
        let combined = population_sigreg(&views, &valid, &random)
            .0
            .double_value(&[]);
        let separate = (0..2)
            .map(|v| {
                population_sigreg(&views.narrow(0, v, 1), &valid.narrow(0, v, 1), &random)
                    .0
                    .double_value(&[])
            })
            .sum::<f64>()
            / 2.;
        assert!((combined - separate).abs() < 1e-4);
        let pooled = population_sigreg(
            &views.reshape([1, 128, 8]),
            &valid.reshape([1, 128]),
            &random,
        )
        .0
        .double_value(&[]);
        assert!(
            (combined - pooled).abs() > 1e-3,
            "temporal pooling was invisible to the statistic"
        );
        let (duplicated, n) =
            population_sigreg(&views.repeat([1, 2, 1]), &valid.repeat([1, 2]), &random);
        assert!((duplicated.double_value(&[]) - 2. * combined).abs() < 1e-3);
        assert_eq!(n.double_value(&[]), 128.);
        let invalid = Tensor::zeros_like(&valid);
        let _ = invalid.narrow(1, 0, 1).fill_(1.);
        assert_eq!(
            population_sigreg(&views, &invalid, &random)
                .0
                .double_value(&[]),
            0.
        );
    }

    #[test]
    fn cuda_jepa_resident_randomness_refreshes_reproducibly_without_advancing_torch_rng() {
        let _rng = crate::torch::test_rng::exclusive();
        let Some(device) = gpu() else {
            return;
        };
        let mut a = JepaRandom::new(&config(), device);
        let mut b = JepaRandom::new(&config(), device);
        let addresses = (a.directions.data_ptr(), a.positions.data_ptr());
        tch::manual_seed(971);
        let expected = Tensor::randn([16], (Kind::Float, device));
        tch::manual_seed(971);
        a.refresh().unwrap();
        b.refresh().unwrap();
        assert!(expected.equal(&Tensor::randn([16], (Kind::Float, device))));
        let first_directions = a.directions.copy();
        let first_positions = a.positions.copy();
        let mut positions_changed = false;
        for _ in 0..8 {
            a.refresh().unwrap();
            b.refresh().unwrap();
            assert!(a.directions.equal(&b.directions));
            assert!(a.positions.equal(&b.positions));
            positions_changed |= !a.positions.equal(&first_positions);
        }
        assert!(!a.directions.equal(&first_directions));
        assert!(positions_changed);
        assert_eq!(addresses, (a.directions.data_ptr(), a.positions.data_ptr()));
        assert!(a.positions.ge(1).all().int64_value(&[]) != 0);
        assert!(a.positions.lt(16).all().int64_value(&[]) != 0);
        let lengths = a
            .directions
            .square()
            .sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        assert!((lengths - 1.).abs().max().double_value(&[]) < 1e-6);
    }

    #[test]
    fn cuda_reader_both_is_the_same_draw_convex_combination_and_masks_have_zero_gradient() {
        let _rng = crate::torch::test_rng::exclusive();
        let Some(device) = gpu() else {
            return;
        };
        let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
        let config = ModelConfig {
            seq_len: 64,
            jepa_mode: JepaMode::Off,
            reader_norm: ReaderNorm::None,
            sigreg_placement: SigregPlacement::Both,
            future_calendar: false,
            jepa: JepaConfig {
                views: 3,
                directions: 16,
                ..Default::default()
            },
            ..config()
        };
        config.validate().unwrap();
        let mut random = JepaRandom::new(&config, device);
        random.refresh().unwrap();
        // Every legal source is sampled, including the last patch despite no future JEPA target.
        let positions = Vec::<i64>::try_from(random.positions.to_device(Device::Cpu)).unwrap();
        assert_eq!(
            positions
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            [1, 2, 3].into_iter().collect()
        );
        let observation = Tensor::randn([4, 4, 8], (Kind::Float, device)).set_requires_grad(true);
        let state = (Tensor::randn([4, 4, 8], (Kind::Float, device)) * 2.).set_requires_grad(true);
        let valid = Tensor::from_slice(&[
            0f32, 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0.,
        ])
        .reshape([4, 4])
        .to_device(device);
        let (both, diagnostics) =
            reader_sigreg_objective(&config, &observation, &state, &valid, &random);
        let local_config = ModelConfig {
            sigreg_placement: SigregPlacement::Local,
            ..config.clone()
        };
        let state_config = ModelConfig {
            sigreg_placement: SigregPlacement::State,
            ..config.clone()
        };
        let local = reader_sigreg_objective(&local_config, &observation, &state, &valid, &random).0;
        let contextual =
            reader_sigreg_objective(&state_config, &observation, &state, &valid, &random).0;
        assert!(
            (both.double_value(&[])
                - 0.5 * (local.double_value(&[]) + contextual.double_value(&[])))
            .abs()
                < 1e-6
        );
        assert_eq!(diagnostics.kind(), Kind::Float);
        assert_eq!(diagnostics.double_value(&[3]), 2.);
        assert_eq!(diagnostics.double_value(&[4]), 6.);
        assert_eq!(diagnostics.double_value(&[5]), 2.);
        let local_gradient = Tensor::run_backward(&[&local], &[&observation], true, false);
        let state_gradient = Tensor::run_backward(&[&contextual], &[&state], true, false);
        let gradients = Tensor::run_backward(&[&both], &[&observation, &state], false, false);
        let eligible = &valid * Tensor::from_slice(&[0f32, 1., 0., 1.]).to_device(device);
        for (gradient, separate) in gradients
            .iter()
            .zip([&local_gradient[0], &state_gradient[0]])
        {
            assert!((gradient - separate * 0.5).abs().max().double_value(&[]) < 1e-6);
            assert!(
                (gradient * eligible.eq(0.).unsqueeze(-1))
                    .abs()
                    .max()
                    .double_value(&[])
                    == 0.
            );
            assert!(gradient.abs().sum(Kind::Float).double_value(&[]) > 0.);
        }
    }

    #[test]
    fn cuda_sigreg_allows_persistent_population_diversity_without_pooling_time_or_singletons() {
        let _rng = crate::torch::test_rng::exclusive();
        let Some(device) = gpu() else {
            return;
        };
        let mut random = JepaRandom::new(&config(), device);
        random.refresh().unwrap();
        let rows = Tensor::randn([1, 64, 8], (Kind::Float, device));
        let valid = Tensor::ones([1, 64], (Kind::Float, device));
        let one = population_sigreg(&rows, &valid, &random).0;
        // Each example is perfectly persistent through time; that does not collapse its batch population.
        let persistent =
            population_sigreg(&rows.repeat([4, 1, 1]), &valid.repeat([4, 1]), &random).0;
        assert!((one.double_value(&[]) - persistent.double_value(&[])).abs() < 1e-5);
        let collapsed = population_sigreg(&Tensor::zeros_like(&rows), &valid, &random).0;
        assert!(one.double_value(&[]) < collapsed.double_value(&[]));
        // Adding a view with only one valid row must not dilute the mean over eligible views.
        let singleton = Tensor::zeros_like(&valid);
        let _ = singleton.narrow(1, 0, 1).fill_(1.);
        let extended = population_sigreg(
            &Tensor::cat(&[&rows, &Tensor::full_like(&rows, 20.)], 0),
            &Tensor::cat(&[&valid, &singleton], 0),
            &random,
        )
        .0;
        assert!((one.double_value(&[]) - extended.double_value(&[])).abs() < 1e-5);
    }
}
