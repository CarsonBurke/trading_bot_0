//! Origin-only direct forecasters: every attention key is an already completed bar.
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::{nn, Device, Kind, Tensor};

use super::data::{CLOCK_DIM, CONTEXT, FEATURES, HORIZONS};
use crate::torch::world_model::{
    BarTrunk, BAR_ARCHITECTURE, BAR_CACHE_CONTRACT, BAR_FF_DIM, BAR_HEADS, BAR_LAYERS,
    BAR_MODEL_DIM,
};

pub const WIDTH: i64 = 256;
pub const HEADS: i64 = 8;
pub const LAYERS: usize = 4;
pub const FF_WIDTH: i64 = 1024;
pub const BINS: i64 = 128;
pub const PATCHES: i64 = 236;
pub const PATCH_LAYOUT: [(i64, i64, i64); 3] = [(0, 1408, 32), (1408, 512, 8), (1920, 128, 1)];
const DROPOUT: f64 = 0.1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    #[default]
    #[value(name = "single-ticker-timexer")]
    SingleTickerTimeXer,
    PatchTst,
    DLinear,
    NLinear,
    BarTrunk,
    #[value(name = "raw-timexer")]
    RawTimeXer,
}

impl ModelKind {
    pub fn is_probabilistic(self) -> bool {
        self != Self::RawTimeXer
    }
}

pub fn architecture_contract(kind: ModelKind) -> String {
    let variant = match kind {
        ModelKind::SingleTickerTimeXer => {
            "prenorm-asymmetric-global-only-cross-variate-validity-channel-id"
        }
        ModelKind::PatchTst => "channel-independent-shared-patch-encoder-linear-channel-fusion",
        ModelKind::DLinear => "moving-average25-trend-seasonal-shared-temporal-categorical",
        ModelKind::NLinear => "last-value-centered-shared-temporal-categorical",
        ModelKind::BarTrunk => "actual-bardist-causal-pope-fa4-trunk-private-same-ticker-input-map",
        ModelKind::RawTimeXer => "raw-postnorm-patch32-revin-no-channel-id-flatten-mse",
    };
    let mut contract = format!("single-ticker-direct-v1|{variant}|context={CONTEXT}|patches={PATCH_LAYOUT:?}|width={WIDTH}|heads={HEADS}|layers={LAYERS}|ff={FF_WIDTH}|dropout={DROPOUT}|horizons={HORIZONS:?}|bins={BINS}|clock={CLOCK_DIM}|features={FEATURES:?}");
    if kind == ModelKind::BarTrunk {
        contract.push_str(&format!(
            "|trunk={BAR_ARCHITECTURE}|trunk-cache={BAR_CACHE_CONTRACT}|trunk-width={BAR_MODEL_DIM}|trunk-layers={BAR_LAYERS}|trunk-heads={BAR_HEADS}|trunk-ff={BAR_FF_DIM}"
        ));
    }
    contract
}

fn linear(input: &Tensor, layer: &nn::Linear) -> Tensor {
    input.linear(
        &layer.ws.to_kind(input.kind()),
        layer.bs.as_ref().map(|bias| bias.to_kind(input.kind())),
    )
}

#[derive(Debug)]
struct LayerNorm {
    inner: nn::LayerNorm,
    config: nn::LayerNormConfig,
}

impl nn::Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Native CUDA LayerNorm requires affine tensors to match the activation dtype.
        input.layer_norm(
            &self.inner.normalized_shape,
            self.inner
                .ws
                .as_ref()
                .map(|weight| weight.to_kind(input.kind())),
            self.inner
                .bs
                .as_ref()
                .map(|bias| bias.to_kind(input.kind())),
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}

fn norm(path: &nn::Path, width: i64) -> LayerNorm {
    let config = nn::LayerNormConfig::default();
    LayerNorm {
        inner: nn::layer_norm(path, vec![width], config),
        config,
    }
}

fn embedding(path: &nn::Path, name: &str, dims: &[i64]) -> Tensor {
    path.var(
        name,
        dims,
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    )
}

struct Attention {
    query: nn::Linear,
    key_value: nn::Linear,
    output: nn::Linear,
}

impl Attention {
    fn new(path: &nn::Path) -> Self {
        Self {
            query: nn::linear(path / "query", WIDTH, WIDTH, Default::default()),
            key_value: nn::linear(path / "key_value", WIDTH, 2 * WIDTH, Default::default()),
            output: nn::linear(path / "output", WIDTH, WIDTH, Default::default()),
        }
    }

    fn forward(&self, query: &Tensor, source: &Tensor, train: bool) -> Tensor {
        let batch = query.size()[0];
        let query_len = query.size()[1];
        let split = |value: &Tensor| {
            value
                .reshape([batch, -1, HEADS, WIDTH / HEADS])
                .transpose(1, 2)
        };
        let q = split(&linear(query, &self.query));
        let kv = linear(source, &self.key_value).split(WIDTH, -1);
        let attended = Tensor::scaled_dot_product_attention(
            &q,
            &split(&kv[0]),
            &split(&kv[1]),
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )
        .transpose(1, 2)
        .reshape([batch, query_len, WIDTH]);
        linear(&attended, &self.output).dropout(DROPOUT, train)
    }
}

struct FeedForward {
    first: nn::Linear,
    second: nn::Linear,
}

impl FeedForward {
    fn new(path: &nn::Path) -> Self {
        Self {
            first: nn::linear(path / "first", WIDTH, FF_WIDTH, Default::default()),
            second: nn::linear(path / "second", FF_WIDTH, WIDTH, Default::default()),
        }
    }

    fn forward(&self, input: &Tensor, train: bool) -> Tensor {
        linear(
            &linear(input, &self.first)
                .gelu("none")
                .dropout(DROPOUT, train),
            &self.second,
        )
        .dropout(DROPOUT, train)
    }
}

struct EncoderBlock {
    self_norm: LayerNorm,
    self_attention: Attention,
    cross: Option<(LayerNorm, LayerNorm, Attention)>,
    ff_norm: LayerNorm,
    ff: FeedForward,
}

impl EncoderBlock {
    fn new(path: &nn::Path, bridge: bool) -> Self {
        Self {
            self_norm: norm(&(path / "self_norm"), WIDTH),
            self_attention: Attention::new(&(path / "self_attention")),
            cross: bridge.then(|| {
                (
                    norm(&(path / "global_norm"), WIDTH),
                    norm(&(path / "variate_norm"), WIDTH),
                    Attention::new(&(path / "global_cross_attention")),
                )
            }),
            ff_norm: norm(&(path / "ff_norm"), WIDTH),
            ff: FeedForward::new(&(path / "ff")),
        }
    }

    fn forward(&self, input: &Tensor, exogenous: Option<&Tensor>, train: bool) -> Tensor {
        let normalized = input.apply(&self.self_norm);
        let mut state = input + self.self_attention.forward(&normalized, &normalized, train);
        if let Some((query_norm, source_norm, attention)) = &self.cross {
            let patch_count = state.size()[1] - 1;
            let global = state.narrow(1, patch_count, 1);
            let update = attention.forward(
                &global.apply(query_norm),
                &exogenous
                    .expect("bridge requires variate tokens")
                    .apply(source_norm),
                train,
            );
            state = Tensor::cat(&[state.narrow(1, 0, patch_count), global + update], 1);
        }
        &state + self.ff.forward(&state.apply(&self.ff_norm), train)
    }
}

struct MultiscalePatches {
    projections: Vec<nn::Linear>,
    elapsed_projection: nn::Linear,
    elapsed_coordinates: Tensor,
}

impl MultiscalePatches {
    fn new(path: &nn::Path) -> Self {
        let projections = PATCH_LAYOUT
            .iter()
            .map(|(_, _, length)| {
                nn::linear(
                    path / format!("patch_{length}"),
                    *length,
                    WIDTH,
                    Default::default(),
                )
            })
            .collect();
        let mut coordinates = Vec::with_capacity(PATCHES as usize * 3);
        for (start, length, patch) in PATCH_LAYOUT {
            for token in 0..length / patch {
                let age = (CONTEXT as i64 - (start + (token + 1) * patch)) as f32;
                coordinates.extend_from_slice(&[
                    age / CONTEXT as f32,
                    age.ln_1p() / (CONTEXT as f32).ln(),
                    (patch as f32).ln() / 32.0_f32.ln(),
                ]);
            }
        }
        Self {
            projections,
            elapsed_projection: nn::linear(path / "elapsed_position", 3, WIDTH, Default::default()),
            elapsed_coordinates: Tensor::from_slice(&coordinates)
                .reshape([1, PATCHES, 3])
                .to_device(path.device()),
        }
    }

    fn values(&self, input: &Tensor) -> Tensor {
        let batch = input.size()[0];
        let tokens: Vec<_> = PATCH_LAYOUT
            .iter()
            .zip(&self.projections)
            .map(|((start, length, patch), projection)| {
                linear(
                    &input
                        .narrow(1, *start, *length)
                        .reshape([batch, length / patch, *patch]),
                    projection,
                )
            })
            .collect();
        Tensor::cat(&tokens, 1)
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.values(input)
            + linear(
                &self.elapsed_coordinates.to_kind(input.kind()),
                &self.elapsed_projection,
            )
    }
}

struct HorizonHead {
    horizon: Tensor,
    clock: nn::Linear,
    query_norm: LayerNorm,
    source_norm: LayerNorm,
    attention: Attention,
    output_norm: LayerNorm,
    output: nn::Linear,
}

impl HorizonHead {
    fn new(path: &nn::Path) -> Self {
        Self {
            horizon: embedding(path, "horizon", &[1, HORIZONS.len() as i64, WIDTH]),
            clock: nn::linear(
                path / "future_clock",
                CLOCK_DIM as i64,
                WIDTH,
                Default::default(),
            ),
            query_norm: norm(&(path / "query_norm"), WIDTH),
            source_norm: norm(&(path / "source_norm"), WIDTH),
            attention: Attention::new(&(path / "attention")),
            output_norm: norm(&(path / "output_norm"), WIDTH),
            output: nn::linear(path / "logits", WIDTH, BINS, Default::default()),
        }
    }

    fn forward(&self, patches: &Tensor, global: &Tensor, clock: &Tensor, train: bool) -> Tensor {
        let query = self.horizon.to_kind(patches.kind()) + linear(clock, &self.clock) + global;
        let update = self.attention.forward(
            &query.apply(&self.query_norm),
            &patches.apply(&self.source_norm),
            train,
        );
        linear(&(query + update).apply(&self.output_norm), &self.output)
    }
}

struct AsymmetricModel {
    patches: MultiscalePatches,
    global: Tensor,
    variate: nn::Linear,
    channel: Tensor,
    layers: Vec<EncoderBlock>,
    final_norm: LayerNorm,
    head: HorizonHead,
}

impl AsymmetricModel {
    fn new(path: &nn::Path) -> Self {
        Self {
            patches: MultiscalePatches::new(&(path / "patches")),
            global: embedding(path, "global_token", &[1, 1, WIDTH]),
            variate: nn::linear(
                path / "variate",
                2 * CONTEXT as i64,
                WIDTH,
                Default::default(),
            ),
            channel: embedding(path, "channel", &[1, FEATURES.len() as i64, WIDTH]),
            layers: (0..LAYERS)
                .map(|i| EncoderBlock::new(&(path / format!("block_{i}")), true))
                .collect(),
            final_norm: norm(&(path / "final_norm"), WIDTH),
            head: HorizonHead::new(&(path / "head")),
        }
    }

    fn forward(
        &self,
        endogenous: &Tensor,
        exogenous: &Tensor,
        validity: &Tensor,
        clock: &Tensor,
        train: bool,
    ) -> Tensor {
        let variates = linear(&Tensor::cat(&[exogenous, validity], -1), &self.variate)
            + self.channel.to_kind(endogenous.kind());
        let mut state = Tensor::cat(
            &[
                self.patches.forward(endogenous),
                self.global
                    .to_kind(endogenous.kind())
                    .expand([endogenous.size()[0], 1, WIDTH], true),
            ],
            1,
        );
        for layer in &self.layers {
            state = layer.forward(&state, Some(&variates), train);
        }
        state = state.apply(&self.final_norm);
        self.head.forward(
            &state.narrow(1, 0, PATCHES),
            &state.narrow(1, PATCHES, 1),
            clock,
            train,
        )
    }
}

struct PatchTstModel {
    patches: MultiscalePatches,
    validity: MultiscalePatches,
    channel: Tensor,
    layers: Vec<EncoderBlock>,
    fusion: nn::Linear,
    final_norm: LayerNorm,
    head: HorizonHead,
}

impl PatchTstModel {
    fn new(path: &nn::Path) -> Self {
        Self {
            patches: MultiscalePatches::new(&(path / "patches")),
            validity: MultiscalePatches::new(&(path / "validity")),
            channel: embedding(path, "channel", &[1, FEATURES.len() as i64 + 1, 1, WIDTH]),
            layers: (0..LAYERS)
                .map(|i| EncoderBlock::new(&(path / format!("block_{i}")), false))
                .collect(),
            fusion: nn::linear(
                path / "channel_fusion",
                (FEATURES.len() as i64 + 1) * WIDTH,
                WIDTH,
                Default::default(),
            ),
            final_norm: norm(&(path / "final_norm"), WIDTH),
            head: HorizonHead::new(&(path / "head")),
        }
    }

    fn forward(
        &self,
        endogenous: &Tensor,
        exogenous: &Tensor,
        validity: &Tensor,
        clock: &Tensor,
        train: bool,
    ) -> Tensor {
        let batch = endogenous.size()[0];
        let channels = FEATURES.len() as i64 + 1;
        let values = Tensor::cat(&[endogenous.unsqueeze(1), exogenous.shallow_clone()], 1)
            .reshape([batch * channels, CONTEXT as i64]);
        let valid = Tensor::cat(
            &[
                Tensor::ones_like(endogenous).unsqueeze(1),
                validity.shallow_clone(),
            ],
            1,
        )
        .reshape([batch * channels, CONTEXT as i64]);
        let mut state = ((self.patches.forward(&values) + self.validity.values(&valid))
            .reshape([batch, channels, PATCHES, WIDTH])
            + self.channel.to_kind(values.kind()))
        .reshape([batch * channels, PATCHES, WIDTH]);
        for layer in &self.layers {
            state = layer.forward(&state, None, train);
        }
        let channel_states = state
            .reshape([batch, channels, PATCHES, WIDTH])
            .permute([0, 2, 1, 3])
            .reshape([batch, PATCHES, channels * WIDTH]);
        let patches = linear(&channel_states, &self.fusion).apply(&self.final_norm);
        let global = patches.mean_dim([1i64].as_slice(), true, patches.kind());
        self.head.forward(&patches, &global, clock, train)
    }
}

struct LinearModel {
    kind: ModelKind,
    seasonal: nn::Linear,
    trend: Option<nn::Linear>,
    categorical: nn::Linear,
    clock: nn::Linear,
}

impl LinearModel {
    fn new(path: &nn::Path, kind: ModelKind) -> Self {
        let mut config = nn::LinearConfig::default();
        config.ws_init = nn::Init::Const(1.0 / CONTEXT as f64);
        Self {
            kind,
            seasonal: nn::linear(
                path / "seasonal",
                CONTEXT as i64,
                HORIZONS.len() as i64,
                config,
            ),
            trend: (kind == ModelKind::DLinear).then(|| {
                nn::linear(
                    path / "trend",
                    CONTEXT as i64,
                    HORIZONS.len() as i64,
                    config,
                )
            }),
            categorical: nn::linear(
                path / "categorical",
                1 + 2 * FEATURES.len() as i64,
                BINS,
                Default::default(),
            ),
            clock: nn::linear(
                path / "future_clock",
                CLOCK_DIM as i64,
                BINS,
                Default::default(),
            ),
        }
    }

    fn forward(
        &self,
        endogenous: &Tensor,
        exogenous: &Tensor,
        validity: &Tensor,
        clock: &Tensor,
    ) -> Tensor {
        let history = Tensor::cat(
            &[
                endogenous.unsqueeze(1),
                exogenous.shallow_clone(),
                validity.shallow_clone(),
            ],
            1,
        );
        let forecasts = if self.kind == ModelKind::DLinear {
            let trend = history
                .replication_pad1d([12, 12])
                .avg_pool1d([25], [1], [0], false, true);
            linear(&(&history - &trend), &self.seasonal)
                + linear(&trend, self.trend.as_ref().unwrap())
        } else {
            let origin = history.narrow(2, CONTEXT as i64 - 1, 1).detach();
            linear(&(history - &origin), &self.seasonal) + origin
        };
        linear(&forecasts.transpose(1, 2), &self.categorical) + linear(clock, &self.clock)
    }
}

struct BarTrunkModel {
    input: nn::Linear,
    trunk: BarTrunk,
    representation: nn::Linear,
    head: HorizonHead,
}

impl BarTrunkModel {
    fn new(path: &nn::Path) -> Self {
        Self {
            input: nn::linear(
                path / "same_ticker_input",
                1 + 2 * FEATURES.len() as i64,
                BAR_MODEL_DIM,
                Default::default(),
            ),
            trunk: BarTrunk::new(&(path / "actual_bar_trunk")),
            representation: nn::linear(
                path / "representation",
                BAR_MODEL_DIM,
                WIDTH,
                Default::default(),
            ),
            head: HorizonHead::new(&(path / "head")),
        }
    }

    fn forward(
        &self,
        endogenous: &Tensor,
        exogenous: &Tensor,
        validity: &Tensor,
        clock: &Tensor,
        train: bool,
    ) -> Tensor {
        let history = Tensor::cat(
            &[
                endogenous.unsqueeze(1),
                exogenous.shallow_clone(),
                validity.shallow_clone(),
            ],
            1,
        )
        .transpose(1, 2);
        let states = tch::autocast(matches!(history.device(), Device::Cuda(_)), || {
            self.trunk
                .forward_embedded(&linear(&history, &self.input), 0, train)
        });
        let states = linear(&states.to_kind(history.kind()), &self.representation);
        self.head.forward(
            &states,
            &states.narrow(1, CONTEXT as i64 - 1, 1),
            clock,
            train,
        )
    }
}

struct RawBlock {
    attention: Attention,
    cross: Attention,
    norms: [LayerNorm; 3],
    ff: FeedForward,
}

impl RawBlock {
    fn new(path: &nn::Path) -> Self {
        Self {
            attention: Attention::new(&(path / "self_attention")),
            cross: Attention::new(&(path / "cross_attention")),
            norms: std::array::from_fn(|i| norm(&(path / format!("norm_{i}")), WIDTH)),
            ff: FeedForward::new(&(path / "ff")),
        }
    }

    fn forward(&self, input: &Tensor, exogenous: &Tensor, train: bool) -> Tensor {
        let state = (input + self.attention.forward(input, input, train)).apply(&self.norms[0]);
        let last = state.size()[1] - 1;
        let global = state.narrow(1, last, 1);
        let global =
            (&global + self.cross.forward(&global, exogenous, train)).apply(&self.norms[1]);
        let state = Tensor::cat(&[state.narrow(1, 0, last), global], 1);
        (&state + self.ff.forward(&state, train)).apply(&self.norms[2])
    }
}

struct RawTimeXerModel {
    patch: nn::Linear,
    position: Tensor,
    global: Tensor,
    variate: nn::Linear,
    layers: Vec<RawBlock>,
    final_norm: LayerNorm,
    head: nn::Linear,
}

impl RawTimeXerModel {
    fn new(path: &nn::Path) -> Self {
        let tokens = CONTEXT as i64 / 32;
        let positions: Vec<f32> = (0..tokens)
            .flat_map(|p| {
                (0..WIDTH).map(move |d| {
                    let phase = p as f64 / 10000_f64.powf((2 * (d / 2)) as f64 / WIDTH as f64);
                    if d % 2 == 0 {
                        phase.sin() as f32
                    } else {
                        phase.cos() as f32
                    }
                })
            })
            .collect();
        Self {
            patch: nn::linear(
                path / "patch",
                32,
                WIDTH,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            position: Tensor::from_slice(&positions)
                .reshape([1, tokens, WIDTH])
                .to_device(path.device()),
            global: embedding(path, "global_token", &[1, 1, WIDTH]),
            variate: nn::linear(path / "variate", CONTEXT as i64, WIDTH, Default::default()),
            layers: (0..LAYERS)
                .map(|i| RawBlock::new(&(path / format!("block_{i}"))))
                .collect(),
            final_norm: norm(&(path / "final_norm"), WIDTH),
            head: nn::linear(
                path / "flatten_head",
                (tokens + 1) * WIDTH,
                HORIZONS.len() as i64,
                Default::default(),
            ),
        }
    }

    fn forward(&self, endogenous: &Tensor, exogenous: &Tensor, train: bool) -> Tensor {
        let batch = endogenous.size()[0];
        let mean = endogenous
            .mean_dim([1i64].as_slice(), true, endogenous.kind())
            .detach();
        let scale = (endogenous.var_dim([1i64].as_slice(), false, true) + 1e-5).sqrt();
        let normalized = (endogenous - &mean) / &scale;
        let exo_mean = exogenous
            .mean_dim([2i64].as_slice(), true, exogenous.kind())
            .detach();
        let exo_scale = (exogenous.var_dim([2i64].as_slice(), false, true) + 1e-5).sqrt();
        let variates =
            linear(&((exogenous - exo_mean) / exo_scale), &self.variate).dropout(DROPOUT, train);
        let patches = linear(
            &normalized.reshape([batch, CONTEXT as i64 / 32, 32]),
            &self.patch,
        ) + self.position.to_kind(endogenous.kind());
        let mut state = Tensor::cat(
            &[
                patches,
                self.global
                    .to_kind(endogenous.kind())
                    .expand([batch, 1, WIDTH], true),
            ],
            1,
        )
        .dropout(DROPOUT, train);
        for layer in &self.layers {
            state = layer.forward(&state, &variates, train);
        }
        linear(
            &state.apply(&self.final_norm).transpose(1, 2).flatten(1, -1),
            &self.head,
        )
        .dropout(DROPOUT, train)
            * scale
            + mean
    }
}

enum Model {
    Asymmetric(AsymmetricModel),
    PatchTst(PatchTstModel),
    Linear(LinearModel),
    BarTrunk(BarTrunkModel),
    Raw(RawTimeXerModel),
}

pub struct ForecastModel {
    model: Model,
    kind: ModelKind,
}

impl ForecastModel {
    pub fn new(path: &nn::Path, kind: ModelKind) -> Self {
        let model = match kind {
            ModelKind::SingleTickerTimeXer => Model::Asymmetric(AsymmetricModel::new(path)),
            ModelKind::PatchTst => Model::PatchTst(PatchTstModel::new(path)),
            ModelKind::DLinear | ModelKind::NLinear => Model::Linear(LinearModel::new(path, kind)),
            ModelKind::BarTrunk => Model::BarTrunk(BarTrunkModel::new(path)),
            ModelKind::RawTimeXer => Model::Raw(RawTimeXerModel::new(path)),
        };
        Self { model, kind }
    }

    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// Inputs contain one forecast origin per row; this is never a dense sequence-prediction API.
    pub fn forward(
        &self,
        endogenous: &Tensor,
        exogenous: &Tensor,
        validity: &Tensor,
        future_clock: &Tensor,
        train: bool,
    ) -> Tensor {
        let batch = endogenous.size()[0];
        assert_eq!(endogenous.size(), [batch, CONTEXT as i64]);
        assert_eq!(
            exogenous.size(),
            [batch, FEATURES.len() as i64, CONTEXT as i64]
        );
        assert_eq!(validity.size(), exogenous.size());
        assert_eq!(
            future_clock.size(),
            [batch, HORIZONS.len() as i64, CLOCK_DIM as i64]
        );
        #[cfg(not(test))]
        assert!(
            matches!(endogenous.device(), Device::Cuda(_)),
            "forecasters require CUDA; CPU is only supported in structural tests"
        );
        let kind = match endogenous.device() {
            Device::Cuda(_) => Kind::BFloat16,
            _ => Kind::Float,
        };
        let endogenous = endogenous.to_kind(kind);
        let validity = validity.to_kind(kind);
        let exogenous = exogenous
            .to_kind(kind)
            .where_self(&validity.gt(0.0), &Tensor::zeros_like(&validity));
        let clock = future_clock.to_kind(kind);
        match &self.model {
            Model::Asymmetric(model) => {
                model.forward(&endogenous, &exogenous, &validity, &clock, train)
            }
            Model::PatchTst(model) => {
                model.forward(&endogenous, &exogenous, &validity, &clock, train)
            }
            Model::Linear(model) => model.forward(&endogenous, &exogenous, &validity, &clock),
            Model::BarTrunk(model) => {
                model.forward(&endogenous, &exogenous, &validity, &clock, train)
            }
            Model::Raw(model) => model.forward(&endogenous, &exogenous, train),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_layer_norm_preserves_activation_dtype_and_master_gradients() {
        tch::set_num_threads(1);
        let store = nn::VarStore::new(Device::Cpu);
        let normalization = norm(&store.root(), WIDTH);
        for kind in [Kind::Float, Kind::BFloat16] {
            let input = Tensor::randn([2, 3, WIDTH], (kind, Device::Cpu)).set_requires_grad(true);
            let output = input.apply(&normalization);
            assert_eq!(output.kind(), kind);
            let expected = input.layer_norm(
                [WIDTH],
                normalization.inner.ws.as_ref().map(|x| x.to_kind(kind)),
                normalization.inner.bs.as_ref().map(|x| x.to_kind(kind)),
                normalization.config.eps,
                normalization.config.cudnn_enabled,
            );
            assert!(output.equal(&expected));
            (&output * Tensor::randn_like(&output))
                .sum(Kind::Float)
                .backward();
            assert_eq!(input.grad().kind(), kind);
        }
        for parameter in store.variables().values() {
            assert_eq!(parameter.kind(), Kind::Float);
            assert_eq!(parameter.grad().kind(), Kind::Float);
            assert_eq!(parameter.grad().isfinite().all().int64_value(&[]), 1);
            assert!(parameter.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
        }
    }

    #[test]
    fn bfloat16_asymmetric_model_backpropagates_to_float_master_parameters() {
        tch::set_num_threads(1);
        let store = nn::VarStore::new(Device::Cpu);
        let model = AsymmetricModel::new(&store.root());
        let (history, exo, validity, clock) = input(1);
        let output = model.forward(
            &history.to_kind(Kind::BFloat16),
            &exo.to_kind(Kind::BFloat16),
            &validity.to_kind(Kind::BFloat16),
            &clock.to_kind(Kind::BFloat16),
            true,
        );
        assert_eq!(output.kind(), Kind::BFloat16);
        assert_eq!(output.size(), [1, 6, 128]);
        output
            .log_softmax(-1, Kind::Float)
            .select(-1, 0)
            .neg()
            .mean(Kind::Float)
            .backward();
        for (name, parameter) in store.variables() {
            assert_eq!(parameter.kind(), Kind::Float, "{name}");
            assert!(parameter.grad().defined(), "missing gradient: {name}");
            assert_eq!(parameter.grad().kind(), Kind::Float, "{name}");
            assert_eq!(
                parameter.grad().isfinite().all().int64_value(&[]),
                1,
                "{name}"
            );
        }
    }

    fn input(batch: i64) -> (Tensor, Tensor, Tensor, Tensor) {
        (
            Tensor::randn([batch, CONTEXT as i64], (Kind::Float, Device::Cpu)),
            Tensor::randn(
                [batch, FEATURES.len() as i64, CONTEXT as i64],
                (Kind::Float, Device::Cpu),
            ),
            Tensor::ones(
                [batch, FEATURES.len() as i64, CONTEXT as i64],
                (Kind::Float, Device::Cpu),
            ),
            Tensor::zeros(
                [batch, HORIZONS.len() as i64, CLOCK_DIM as i64],
                (Kind::Float, Device::Cpu),
            ),
        )
    }

    #[test]
    fn multiscale_layout_covers_every_bar_exactly_once() {
        let mut seen = vec![0; CONTEXT];
        let mut count = 0;
        for (start, length, patch) in PATCH_LAYOUT {
            assert_eq!(length % patch, 0);
            for i in start..start + length {
                seen[i as usize] += 1;
            }
            count += length / patch;
        }
        assert!(seen.iter().all(|count| *count == 1));
        assert_eq!(count, PATCHES);
        let store = nn::VarStore::new(Device::Cpu);
        let patches = MultiscalePatches::new(&store.root());
        let history =
            Tensor::ones([1, CONTEXT as i64], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        patches
            .forward(&history)
            .square()
            .sum(Kind::Float)
            .backward();
        assert_eq!(history.grad().ne(0.0).all().int64_value(&[]), 1);
    }

    #[test]
    fn exogenous_bridge_only_updates_global_within_one_block() {
        let store = nn::VarStore::new(Device::Cpu);
        let block = EncoderBlock::new(&store.root(), true);
        let state = Tensor::randn([1, PATCHES + 1, WIDTH], (Kind::Float, Device::Cpu));
        let exo = Tensor::randn(
            [1, FEATURES.len() as i64, WIDTH],
            (Kind::Float, Device::Cpu),
        );
        let first = block.forward(&state, Some(&exo), false);
        let changed = Tensor::randn_like(&Tensor::zeros(
            [1, FEATURES.len() as i64, WIDTH],
            (Kind::Float, Device::Cpu),
        ));
        let third = block.forward(&state, Some(&changed), false);
        assert!(first
            .narrow(1, 0, PATCHES)
            .equal(&third.narrow(1, 0, PATCHES)));
        assert!(!first.narrow(1, PATCHES, 1).allclose(
            &third.narrow(1, PATCHES, 1),
            1e-6,
            1e-6,
            false
        ));
    }

    #[test]
    fn probabilistic_model_is_row_independent_and_masks_missing_values() {
        tch::set_num_threads(1);
        let store = nn::VarStore::new(Device::Cpu);
        let model = ForecastModel::new(&store.root(), ModelKind::SingleTickerTimeXer);
        let parameter_count: usize = store.variables().values().map(Tensor::numel).sum();
        println!("SingleTickerTimeXer parameters: {parameter_count}");
        let (history, exo, validity, clock) = input(2);
        let _ = validity.narrow(1, 0, 1).fill_(0.0);
        let output = model.forward(&history, &exo, &validity, &clock, false);
        assert_eq!(output.size(), [2, 6, 128]);
        let one = model.forward(
            &history.narrow(0, 0, 1),
            &exo.narrow(0, 0, 1),
            &validity.narrow(0, 0, 1),
            &clock.narrow(0, 0, 1),
            false,
        );
        assert!(output.narrow(0, 0, 1).allclose(&one, 1e-5, 1e-5, false));
        let _ = exo.narrow(1, 0, 1).fill_(f64::NAN);
        let masked = model.forward(&history, &exo, &validity, &clock, false);
        assert!(output.equal(&masked));
        let changed_clock = clock + 0.25;
        assert!(!output.allclose(
            &model.forward(&history, &exo, &validity, &changed_clock, false),
            1e-6,
            1e-6,
            false
        ));
        output
            .log_softmax(-1, Kind::Float)
            .select(-1, 0)
            .neg()
            .mean(Kind::Float)
            .backward();
        for name in [
            "patches.patch_1.weight",
            "patches.patch_8.weight",
            "patches.patch_32.weight",
            "variate.weight",
            "channel",
            "head.horizon",
        ] {
            let parameter = store
                .variables()
                .remove(name)
                .unwrap_or_else(|| panic!("missing parameter {name}"));
            assert!(parameter.grad().defined(), "no gradient for {name}");
            assert!(
                parameter.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0,
                "zero gradient for {name}"
            );
        }
    }

    #[test]
    fn bar_trunk_embedded_adapter_preserves_actual_causal_layers() {
        tch::set_num_threads(1);
        let store = nn::VarStore::new(Device::Cpu);
        let trunk = BarTrunk::new(&store.root());
        let dof = Tensor::randn(
            [1, 8, crate::torch::bar_dist::BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        );
        let bins = Tensor::zeros(dof.size(), (Kind::Int64, Device::Cpu));
        let clock = Tensor::zeros(
            [1, 8, crate::torch::dataset::BAR_TIME_FEATURES as i64],
            (Kind::Int64, Device::Cpu),
        );
        let original = trunk.forward(&dof, &bins, &clock, 0, false);
        let embedded =
            trunk.forward_embedded(&trunk.token_embedding(&dof, &bins, &clock), 0, false);
        assert!(original.equal(&embedded));
    }

    #[test]
    fn baseline_output_contracts() {
        tch::set_num_threads(1);
        let (history, exo, validity, clock) = input(1);
        for kind in [
            ModelKind::DLinear,
            ModelKind::NLinear,
            ModelKind::PatchTst,
            ModelKind::RawTimeXer,
        ] {
            let store = nn::VarStore::new(Device::Cpu);
            let model = ForecastModel::new(&store.root(), kind);
            let output = model.forward(&history, &exo, &validity, &clock, false);
            let expected = if kind.is_probabilistic() {
                vec![1, 6, 128]
            } else {
                vec![1, 6]
            };
            assert_eq!(output.size(), expected);
            assert_eq!(output.isfinite().all().int64_value(&[]), 1);
        }
    }
}
