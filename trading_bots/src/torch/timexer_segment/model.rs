use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use tch::{nn, Device, Kind, Tensor};

pub const CHANNELS: i64 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum DecoderKind {
    #[default]
    Joint,
    Legacy,
}

fn legacy_decoder() -> DecoderKind {
    DecoderKind::Legacy
}
fn is_legacy_decoder(decoder: &DecoderKind) -> bool {
    *decoder == DecoderKind::Legacy
}

pub struct Forecast {
    pub standardized: Tensor,
    pub prices: Tensor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[arg(long, default_value_t = 6000)]
    pub seq_len: i64,
    #[arg(long, default_value_t = 192)]
    pub pred_len: i64,
    #[arg(long, default_value_t = 16)]
    pub patch_len: i64,
    #[arg(long, default_value_t = 512)]
    pub d_model: i64,
    #[arg(long, default_value_t = 8)]
    pub n_heads: i64,
    #[arg(long, default_value_t = 2)]
    pub e_layers: usize,
    #[arg(long, default_value_t = 2048)]
    pub d_ff: i64,
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,
    #[arg(long)]
    #[serde(default)]
    pub volume_features: bool,
    #[arg(long, value_enum, default_value_t = DecoderKind::Joint)]
    #[serde(default = "legacy_decoder", skip_serializing_if = "is_legacy_decoder")]
    pub decoder: DecoderKind,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            seq_len: 6000,
            pred_len: 192,
            patch_len: 16,
            d_model: 512,
            n_heads: 8,
            e_layers: 2,
            d_ff: 2048,
            dropout: 0.1,
            volume_features: false,
            decoder: DecoderKind::Joint,
        }
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.seq_len > 0 && self.pred_len > 0,
            "sequence lengths must be positive"
        );
        ensure!(
            self.patch_len > 0 && self.seq_len % self.patch_len == 0,
            "patches must cover the entire context exactly"
        );
        ensure!(
            self.d_model > 0 && self.d_model % 2 == 0,
            "model width must be positive and even"
        );
        ensure!(
            self.n_heads > 0 && self.d_model % self.n_heads == 0,
            "model width must be divisible by attention heads"
        );
        ensure!(
            self.e_layers > 0 && self.d_ff > 0,
            "encoder and feedforward widths must be positive"
        );
        ensure!(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "dropout must be in [0, 1)"
        );
        Ok(())
    }
}

fn projection(path: nn::Path, input: i64, output: i64, bias: bool) -> nn::Linear {
    // Match torch.nn.Linear, including its fan-in initialization.
    let bound = 1.0 / (input as f64).sqrt();
    let init = nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: init,
            bs_init: Some(init),
            bias,
        },
    )
}

fn linear(input: &Tensor, layer: &nn::Linear) -> Tensor {
    input.linear(
        &layer.ws.to_kind(input.kind()),
        layer.bs.as_ref().map(|bias| bias.to_kind(input.kind())),
    )
}

struct LayerNorm(nn::LayerNorm);

impl LayerNorm {
    fn new(path: nn::Path, width: i64) -> Self {
        Self(nn::layer_norm(path, vec![width], Default::default()))
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(
            &self.0.normalized_shape,
            self.0
                .ws
                .as_ref()
                .map(|weight| weight.to_kind(input.kind())),
            self.0.bs.as_ref().map(|bias| bias.to_kind(input.kind())),
            1e-5,
            true,
        )
    }
}

struct Attention {
    query: nn::Linear,
    key_value: nn::Linear,
    output: nn::Linear,
    heads: i64,
    width: i64,
    dropout: f64,
}

impl Attention {
    fn new(path: nn::Path, config: &ModelConfig) -> Self {
        let width = config.d_model;
        Self {
            query: projection(&path / "query", width, width, true),
            key_value: projection(&path / "key_value", width, 2 * width, true),
            output: projection(&path / "output", width, width, true),
            heads: config.n_heads,
            width,
            dropout: config.dropout,
        }
    }

    fn forward(&self, query: &Tensor, source: &Tensor, train: bool) -> Tensor {
        let batch = query.size()[0];
        let length = query.size()[1];
        let split = |tensor: &Tensor| {
            tensor
                .reshape([batch, -1, self.heads, self.width / self.heads])
                .transpose(1, 2)
        };
        let q = split(&linear(query, &self.query));
        let kv = linear(source, &self.key_value).split(self.width, -1);
        let attended = Tensor::scaled_dot_product_attention(
            &q,
            &split(&kv[0]),
            &split(&kv[1]),
            None::<&Tensor>,
            if train { self.dropout } else { 0.0 },
            false,
            None,
            false,
        )
        .transpose(1, 2)
        .reshape([batch, length, self.width]);
        linear(&attended, &self.output).dropout(self.dropout, train)
    }
}

struct EncoderBlock {
    self_attention: Attention,
    cross_attention: Attention,
    norms: [LayerNorm; 3],
    first: nn::Linear,
    second: nn::Linear,
    dropout: f64,
}

impl EncoderBlock {
    fn new(path: nn::Path, config: &ModelConfig) -> Self {
        Self {
            self_attention: Attention::new(&path / "self_attention", config),
            cross_attention: Attention::new(&path / "cross_attention", config),
            norms: std::array::from_fn(|index| {
                LayerNorm::new(&path / format!("norm_{index}"), config.d_model)
            }),
            first: projection(&path / "first", config.d_model, config.d_ff, true),
            second: projection(&path / "second", config.d_ff, config.d_model, true),
            dropout: config.dropout,
        }
    }

    fn forward(&self, input: &Tensor, variates: &Tensor, train: bool) -> Tensor {
        let state =
            self.norms[0].forward(&(input + self.self_attention.forward(input, input, train)));
        let patch_count = state.size()[1] - 1;
        let global = state.narrow(1, patch_count, 1);
        let queries = global.reshape([variates.size()[0], CHANNELS, state.size()[2]]);
        let cross = self
            .cross_attention
            .forward(&queries, variates, train)
            .reshape_as(&global);
        let global = self.norms[1].forward(&(global + cross));
        let state = Tensor::cat(&[state.narrow(1, 0, patch_count), global], 1);
        let ff = linear(
            &linear(&state, &self.first)
                .gelu("none")
                .dropout(self.dropout, train),
            &self.second,
        )
        .dropout(self.dropout, train);
        self.norms[2].forward(&(state + ff))
    }
}

fn normalize(input: &Tensor) -> (Tensor, Tensor, Tensor) {
    let input = input.to_kind(Kind::Float);
    let mean = input
        .mean_dim([1i64].as_slice(), true, Kind::Float)
        .detach();
    let centered = input - &mean;
    let scale = (centered.var_dim([1i64].as_slice(), false, true) + 1e-5).sqrt();
    (&centered / &scale, mean, scale)
}

fn normalize_auxiliary(auxiliary: &Tensor) -> Tensor {
    let values = auxiliary.narrow(2, 0, 1).to_kind(Kind::Float);
    let valid = auxiliary.narrow(2, 1, 1).to_kind(Kind::Float);
    let count = valid
        .sum_dim_intlist([1i64].as_slice(), true, Kind::Float)
        .clamp_min(1.0);
    let mean = (&values * &valid).sum_dim_intlist([1i64].as_slice(), true, Kind::Float) / &count;
    let centered = (values - mean) * &valid;
    let variance = centered
        .square()
        .sum_dim_intlist([1i64].as_slice(), true, Kind::Float)
        / count;
    Tensor::cat(&[centered / (variance + 1e-5).sqrt(), valid], 2)
}

fn positional_encoding(patches: i64, width: i64, device: Device) -> Tensor {
    let values: Vec<f32> = (0..patches)
        .flat_map(|position| {
            (0..width).map(move |index| {
                let angle =
                    position as f64 / 10000_f64.powf((2 * (index / 2)) as f64 / width as f64);
                if index % 2 == 0 {
                    angle.sin() as f32
                } else {
                    angle.cos() as f32
                }
            })
        })
        .collect();
    Tensor::from_slice(&values)
        .reshape([1, patches, width])
        .to_device(device)
}

/// Upstream TimeXer forecast_multi on the four OHLC channels of one ticker.
pub struct SegmentModel {
    config: ModelConfig,
    patch: nn::Linear,
    global: Tensor,
    position: Tensor,
    variate: nn::Linear,
    layers: Vec<EncoderBlock>,
    final_norm: LayerNorm,
    head: nn::Linear,
    geometry_mix: Option<nn::Linear>,
}

impl SegmentModel {
    pub fn new(path: &nn::Path, config: &ModelConfig) -> Self {
        config
            .validate()
            .expect("invalid TimeXer segment model configuration");
        let patches = config.seq_len / config.patch_len;
        Self {
            config: config.clone(),
            patch: projection(path / "patch", config.patch_len, config.d_model, false),
            global: path.var(
                "global_token",
                &[1, CHANNELS, 1, config.d_model],
                nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            ),
            position: positional_encoding(patches, config.d_model, path.device()),
            variate: projection(path / "variate", config.seq_len, config.d_model, true),
            layers: (0..config.e_layers)
                .map(|index| EncoderBlock::new(path / format!("block_{index}"), config))
                .collect(),
            final_norm: LayerNorm::new(path / "final_norm", config.d_model),
            head: projection(
                path / "head",
                config.d_model * (patches + 1),
                config.pred_len,
                true,
            ),
            geometry_mix: (config.decoder == DecoderKind::Joint)
                .then(|| projection(path / "geometry_mix", CHANNELS, CHANNELS, true)),
        }
    }

    pub fn forward(
        &self,
        input: &Tensor,
        price_scaling: &Tensor,
        geometry_context: &Tensor,
        train: bool,
    ) -> Forecast {
        self.forward_with_aux(input, None, price_scaling, geometry_context, train)
    }

    pub fn forward_with_aux(
        &self,
        input: &Tensor,
        auxiliary: Option<&Tensor>,
        price_scaling: &Tensor,
        geometry_context: &Tensor,
        train: bool,
    ) -> Forecast {
        assert_eq!(
            auxiliary.is_some(),
            self.config.volume_features,
            "volume feature configuration mismatch"
        );
        assert!(
            matches!(input.device(), Device::Cuda(_)),
            "TimeXer segment inference and training require CUDA"
        );
        let batch = input.size()[0];
        assert_eq!(input.size(), [batch, self.config.seq_len, CHANNELS]);
        assert_eq!(price_scaling.size(), [batch, 2, CHANNELS]);
        assert_eq!(price_scaling.device(), input.device());
        let (normalized, mean, scale) = normalize(input);
        let normalized = normalized.to_kind(Kind::BFloat16).transpose(1, 2);
        let patches = self.config.seq_len / self.config.patch_len;
        let width = self.config.d_model;
        let tokens = linear(
            &normalized.reshape([batch * CHANNELS, patches, self.config.patch_len]),
            &self.patch,
        ) + self.position.to_kind(Kind::BFloat16);
        let mut state = Tensor::cat(
            &[
                tokens.reshape([batch, CHANNELS, patches, width]),
                self.global
                    .to_kind(Kind::BFloat16)
                    .expand([batch, CHANNELS, 1, width], true),
            ],
            2,
        )
        .reshape([batch * CHANNELS, patches + 1, width])
        .dropout(self.config.dropout, train);
        let variate_history = if let Some(auxiliary) = auxiliary {
            assert_eq!(auxiliary.size(), [batch, self.config.seq_len, 2]);
            assert_eq!(auxiliary.device(), input.device());
            Tensor::cat(
                &[
                    normalized,
                    normalize_auxiliary(auxiliary)
                        .to_kind(Kind::BFloat16)
                        .transpose(1, 2),
                ],
                1,
            )
        } else {
            normalized
        };
        let variates = linear(&variate_history, &self.variate).dropout(self.config.dropout, train);
        for layer in &self.layers {
            state = layer.forward(&state, &variates, train);
        }
        let flattened = self
            .final_norm
            .forward(&state)
            .reshape([batch, CHANNELS, patches + 1, width])
            .transpose(2, 3)
            .flatten(2, 3);
        let output = linear(&flattened, &self.head)
            .dropout(self.config.dropout, train)
            .transpose(1, 2)
            .to_kind(Kind::Float);
        let global_mean = price_scaling.narrow(1, 0, 1).to_kind(Kind::Float);
        let global_scale = price_scaling.narrow(1, 1, 1).to_kind(Kind::Float);
        match &self.geometry_mix {
            Some(mix) => {
                let coordinates = linear(&output, mix);
                assert_eq!(geometry_context.size(), [batch, 3]);
                assert_eq!(geometry_context.device(), input.device());
                let prices = decode_joint(&coordinates, geometry_context);
                let standardized = (&prices - global_mean) / global_scale;
                Forecast {
                    standardized,
                    prices,
                }
            }
            None => {
                let standardized = output * scale + mean;
                let prices = &standardized * global_scale + global_mean;
                Forecast {
                    standardized,
                    prices,
                }
            }
        }
    }
}

fn decode_joint(coordinates: &Tensor, geometry_context: &Tensor) -> Tensor {
    let smallest = f32::MIN_POSITIVE as f64;
    let largest = f32::MAX as f64;
    let coordinates = coordinates.to_kind(Kind::Double);
    let stats = geometry_context.to_kind(Kind::Double).unsqueeze(1);
    let close_mean = stats.narrow(2, 0, 1).clamp_min(smallest);
    let close_scale = stats.narrow(2, 1, 1).clamp_min(smallest);
    let ratio = (&close_mean / &close_scale).clamp_min(smallest);
    let inverse_softplus = &ratio + (-&ratio).expm1().neg().log();
    let close = ((inverse_softplus + coordinates.narrow(2, 0, 1)).softplus() * close_scale)
        .clamp(smallest, largest);
    let range_scale = stats.narrow(2, 2, 1).clamp_min(f32::EPSILON as f64);
    let relative_range = (range_scale * coordinates.narrow(2, 1, 1).softplus()
        / std::f64::consts::LN_2)
        .clamp_max(largest);
    let close_position = coordinates.narrow(2, 2, 1).sigmoid();
    let open_position = coordinates.narrow(2, 3, 1).sigmoid();
    let low = (&close / (&close_position * &relative_range + 1.0)).clamp_min(smallest);
    let high: Tensor = &close + (1.0 - close_position) * relative_range * &low;
    let high = high.clamp_max(largest);
    let open = low
        .lerp_tensor(&high, &open_position)
        .maximum(&low)
        .minimum(&high);
    Tensor::cat(&[open, high, low, close], 2).to_kind(Kind::Float)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_complete_uniform_patches() {
        let mut config = ModelConfig::default();
        config.validate().unwrap();
        config.seq_len = 97;
        assert!(config.validate().is_err());
        config.seq_len = 96;
        config.dropout = f64::NAN;
        assert!(config.validate().is_err());
    }

    #[test]
    fn historical_model_configuration_retains_its_context_and_no_auxiliary_inputs() {
        let mut serialized = serde_json::to_value(ModelConfig::default()).unwrap();
        let fields = serialized.as_object_mut().unwrap();
        fields.insert("seq_len".into(), serde_json::json!(96));
        fields.remove("volume_features");
        fields.remove("decoder");
        let config: ModelConfig = serde_json::from_value(serialized).unwrap();
        assert_eq!(config.seq_len, 96);
        assert!(!config.volume_features);
        assert_eq!(config.decoder, DecoderKind::Legacy);
        assert!(serde_json::to_value(&config)
            .unwrap()
            .get("decoder")
            .is_none());
        config.validate().unwrap();
    }

    fn assert_valid_candles(prices: &Tensor) {
        let open = prices.narrow(2, 0, 1);
        let high = prices.narrow(2, 1, 1);
        let low = prices.narrow(2, 2, 1);
        let close = prices.narrow(2, 3, 1);
        assert_eq!(prices.isfinite().all().int64_value(&[]), 1);
        assert_eq!(prices.gt(0.0).all().int64_value(&[]), 1);
        assert_eq!(
            high.ge_tensor(&open.maximum(&close)).all().int64_value(&[]),
            1
        );
        assert_eq!(
            low.le_tensor(&open.minimum(&close)).all().int64_value(&[]),
            1
        );
    }

    #[test]
    fn joint_decoder_uses_historical_mean_and_preserves_price_gradients() {
        let stats = Tensor::from_slice(&[100.0f32, 2.0, 0.02, 0.000001, 5.0, 0.04]).reshape([2, 3]);
        let coordinates =
            Tensor::zeros([2, 3, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let prices = decode_joint(&coordinates, &stats);
        assert_valid_candles(&prices);
        assert!((prices.double_value(&[0, 0, 3]) - 100.0).abs() < 1e-5);
        assert!((prices.double_value(&[1, 0, 3]) - 0.000001).abs() < 1e-11);
        let means = Tensor::from_slice(&[90.0f32, 93.0, 89.0, 91.0, 1000.0, 1001.0, 999.0, 1000.5])
            .reshape([2, 1, 4]);
        let scales =
            Tensor::from_slice(&[2.0f32, 3.0, 4.0, 5.0, 90.0, 91.0, 92.0, 93.0]).reshape([2, 1, 4]);
        let standardized = (&prices - means) / scales;
        standardized.square().mean(Kind::Float).backward();
        assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        for coordinate in 0..4 {
            assert!(
                coordinates
                    .grad()
                    .narrow(2, coordinate, 1)
                    .abs()
                    .sum(Kind::Float)
                    .double_value(&[])
                    > 0.0
            );
        }
    }

    #[test]
    fn joint_decoder_remains_valid_at_flat_and_saturated_position_limits() {
        let mut coordinates = Vec::new();
        for close in [-1e6f32, 0.0, 1e6] {
            for range in [-1000.0, 0.0, 1e6] {
                for close_position in [-1000.0, 0.0, 1000.0] {
                    for open_position in [-1000.0, 0.0, 1000.0] {
                        coordinates.extend([close, range, close_position, open_position]);
                    }
                }
            }
        }
        let coordinates = Tensor::from_slice(&coordinates)
            .reshape([1, -1, 4])
            .set_requires_grad(true);
        let stats = Tensor::from_slice(&[100.0f32, 3.0, f32::EPSILON]).reshape([1, 3]);
        let prices = decode_joint(&coordinates, &stats);
        assert_valid_candles(&prices);
        prices.mean(Kind::Float).backward();
        assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        let flat = decode_joint(
            &Tensor::from_slice(&[0.0f32, -1000.0, 0.0, 0.0]).reshape([1, 1, 4]),
            &stats,
        );
        assert_eq!(flat.max().double_value(&[]), flat.min().double_value(&[]));
        let boundary = decode_joint(
            &Tensor::from_slice(&[f32::MAX, f32::MAX, -1000.0, 1000.0]).reshape([1, 1, 4]),
            &stats,
        );
        assert_valid_candles(&boundary);
    }

    #[test]
    fn extreme_finite_price_scale_ratio_keeps_anchor_and_gradients_finite() {
        let mean = 3e38f32;
        let stats = Tensor::from_slice(&[mean, 0.00316, 1.0]).reshape([1, 3]);
        let coordinates = Tensor::from_slice(&[0.0f32, 100.0, -100.0, 0.0])
            .reshape([1, 1, 4])
            .set_requires_grad(true);
        let prices = decode_joint(&coordinates, &stats);
        assert_valid_candles(&prices);
        assert_eq!(prices.double_value(&[0, 0, 3]), f64::from(mean));
        prices.to_kind(Kind::Double).mean(Kind::Double).backward();
        assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
    }

    #[test]
    fn decoder_adds_only_random_geometry_mix_and_preserves_legacy_parameters() {
        let legacy = ModelConfig {
            seq_len: 96,
            decoder: DecoderKind::Legacy,
            ..ModelConfig::default()
        };
        let old_store = nn::VarStore::new(Device::Cpu);
        let old = SegmentModel::new(&old_store.root(), &legacy);
        assert!(old.geometry_mix.is_none());
        assert!(!old_store
            .variables()
            .keys()
            .any(|name| name.starts_with("geometry_mix.")));
        let new_store = nn::VarStore::new(Device::Cpu);
        let new = SegmentModel::new(
            &new_store.root(),
            &ModelConfig {
                decoder: DecoderKind::Joint,
                ..legacy
            },
        );
        let mix = new.geometry_mix.as_ref().unwrap();
        assert_eq!(mix.ws.size(), [4, 4]);
        assert!(mix.ws.abs().sum(Kind::Float).double_value(&[]) > 0.0);
        assert!(new.head.ws.abs().sum(Kind::Float).double_value(&[]) > 0.0);
        assert_eq!(
            new_store
                .trainable_variables()
                .iter()
                .map(Tensor::numel)
                .sum::<usize>()
                - old_store
                    .trainable_variables()
                    .iter()
                    .map(Tensor::numel)
                    .sum::<usize>(),
            20
        );
    }

    #[test]
    fn normalization_preserves_small_price_changes_and_reconstructs_levels() {
        let input = Tensor::from_slice(&[
            100.0f32, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07,
        ])
        .reshape([1, 2, CHANNELS]);
        let (normalized, mean, scale) = normalize(&input);
        assert_eq!(normalized.kind(), Kind::Float);
        assert!(normalized.abs().min().double_value(&[]) > 0.9);
        assert!((&normalized * scale + mean).allclose(&input, 1e-6, 1e-6, false));
        let constant = Tensor::full([1, 96, CHANNELS], 123.0, (Kind::Float, Device::Cpu));
        let (normalized, _, scale) = normalize(&constant);
        assert_eq!(normalized.abs().max().double_value(&[]), 0.0);
        assert!((scale.min().double_value(&[]) - 1e-5f64.sqrt()).abs() < 1e-7);
    }

    #[test]
    fn volume_normalization_ignores_missing_positions_and_preserves_validity() {
        let auxiliary = Tensor::from_slice(&[2.0f32, 1.0, 0.0, 0.0, 4.0, 1.0]).reshape([1, 3, 2]);
        let normalized = normalize_auxiliary(&auxiliary);
        assert!(normalized.narrow(2, 1, 1).equal(&auxiliary.narrow(2, 1, 1)));
        assert_eq!(normalized.double_value(&[0, 1, 0]), 0.0);
        assert!((normalized.double_value(&[0, 0, 0]) + 1.0).abs() < 1e-5);
        assert!((normalized.double_value(&[0, 2, 0]) - 1.0).abs() < 1e-5);
        let missing = normalize_auxiliary(&Tensor::zeros([2, 96, 2], (Kind::Float, Device::Cpu)));
        assert_eq!(missing.abs().sum(Kind::Float).double_value(&[]), 0.0);
    }

    #[test]
    fn shared_dense_head_and_float_master_parameters_match_contract() {
        let store = nn::VarStore::new(Device::Cpu);
        let model = SegmentModel::new(&store.root(), &ModelConfig::default());
        assert_eq!(model.head.ws.size(), [192, 376 * 512]);
        assert_eq!(model.global.size(), [1, CHANNELS, 1, 512]);
        assert!(model.patch.bs.is_none());
        for (_, tensor) in store.variables() {
            assert_eq!(tensor.kind(), Kind::Float);
        }
        let bound = 1.0 / 16.0f64.sqrt();
        assert!(model.patch.ws.abs().max().double_value(&[]) <= bound);
    }

    #[test]
    fn multivariate_global_queries_keep_forecast_origins_separate() {
        let globals = Tensor::arange(2 * CHANNELS * 3, (Kind::Int64, Device::Cpu)).reshape([
            2 * CHANNELS,
            1,
            3,
        ]);
        let queries = globals.reshape([2, CHANNELS, 3]);
        assert_eq!(queries.get(0).max().int64_value(&[]), 11);
        assert_eq!(queries.get(1).min().int64_value(&[]), 12);
        assert!(queries.reshape_as(&globals).equal(&globals));
    }

    #[test]
    fn cuda_forward_backward_and_origin_independence() {
        if std::env::var("TIMEXER_SEGMENT_GPU_TEST").as_deref() != Ok("1") {
            return;
        }
        assert!(tch::Cuda::is_available());
        crate::torch::cuda::cfg::configure_cuda();
        crate::torch::cuda::cfg::enable_tf32_matmul().unwrap();
        let device = Device::Cuda(0);
        let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
        tch::manual_seed(20260905);
        tch::Cuda::manual_seed_all(20260905);
        let config = ModelConfig {
            decoder: DecoderKind::Legacy,
            ..ModelConfig::default()
        };
        let store = nn::VarStore::new(device);
        let model = SegmentModel::new(&store.root(), &config);
        let input = Tensor::randn([2, config.seq_len, CHANNELS], (Kind::Float, device)) * 0.02
            + Tensor::from_slice(&[0.5f32, 0.51, 0.49, 0.505])
                .to_device(device)
                .reshape([1, 1, CHANNELS]);
        let price_scaling = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            .to_device(device)
            .reshape([1, 2, 4])
            .repeat([2, 1, 1]);
        {
            let _guard = tch::no_grad_guard();
            let baseline = model
                .forward(&input, &price_scaling, &Tensor::new(), false)
                .standardized;
            assert_eq!(baseline.size(), [2, config.pred_len, CHANNELS]);
            assert_eq!(baseline.kind(), Kind::Float);
            assert_eq!(baseline.isfinite().all().int64_value(&[]), 1);
            let changed = Tensor::cat(&[input.narrow(0, 0, 1) + 100.0, input.narrow(0, 1, 1)], 0);
            let changed = model
                .forward(&changed, &price_scaling, &Tensor::new(), false)
                .standardized;
            assert!(
                baseline.get(1).equal(&changed.get(1)),
                "another origin changed this row's forecast"
            );
            assert!(
                baseline
                    .get(0)
                    .allclose(&(changed.get(0) - 100.0), 1e-3, 1e-3, false),
                "window normalization lost level-shift equivariance"
            );
        }

        let prediction = model
            .forward(&input, &price_scaling, &Tensor::new(), true)
            .standardized;
        let target = input
            .narrow(1, config.seq_len - 1, 1)
            .expand_as(&prediction);
        let loss = (prediction - target).square().mean(Kind::Float);
        assert!(loss.double_value(&[]).is_finite());
        loss.backward();
        for (name, parameter) in store.variables() {
            let gradient = parameter.grad();
            assert!(gradient.defined(), "missing gradient: {name}");
            assert_eq!(parameter.kind(), Kind::Float, "master dtype: {name}");
            assert_eq!(gradient.kind(), Kind::Float, "gradient dtype: {name}");
            assert_eq!(
                gradient.isfinite().all().int64_value(&[]),
                1,
                "nonfinite gradient: {name}"
            );
        }
        for parameter in [
            &model.patch.ws,
            &model.global,
            &model.variate.ws,
            &model.head.ws,
        ] {
            assert!(
                parameter.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0,
                "an architectural branch received no gradient"
            );
        }
    }
}
