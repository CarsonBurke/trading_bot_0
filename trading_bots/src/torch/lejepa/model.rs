use crate::torch::cuda::cfg::pin_bfloat16_autocast;
use tch::{autocast, nn, nn::Module, Kind, Tensor};

use crate::torch::fa4::pope_flash_attention_prefill;
#[cfg(test)]
use crate::torch::pope::pope_attention_reference;
use crate::torch::pope::{
    init_pope_theta_bias, pope_expand_qk, PolarQk, PopePhases, PopeThetaInit, POPE_FREQUENCY_BASE,
};

pub const ARCHITECTURE: &str = "lejepa-msejepa-causal-ar-pope64-fa4-v6";
pub const FEATURE_LAYOUT: &str = "torch-env-ohlc-features-fixed-scale-v2";
pub const LATENT_DIM: i64 = 256;
pub const BAR_FEATURES: i64 = 16;
pub const MAX_CONTEXT_BARS: i64 = 6_000;
pub const AR_LAYERS: usize = 6;
pub const AR_FF_DIM: i64 = 1_536;
pub const HEADS: i64 = 4;
pub const HEAD_DIM: i64 = 64;
pub const PROJECTOR_HIDDEN_DIM: i64 = 2_048;
pub const PREDICTOR_HIDDEN_MULT: i64 = 4;
pub const PREDICTOR_BLOCKS: usize = 2;
pub const NORMALIZATION_EPS: f64 = 1e-5;
pub const PROJECTOR_RMS_EPS: f64 = 1e-6;
pub const DROPOUT: f64 = 0.1;
pub const OHLC_FEATURE_SCALE: [f32; BAR_FEATURES as usize] = [
    2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 4e-3, 2e-3, 2e-3, 4e-3, 2e-3, 2e-3, 2e-3, 2e-3,
];

struct CausalLayer {
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

struct PredictorBlock {
    gate: nn::Linear,
    value: nn::Linear,
    out: nn::Linear,
}

struct PredictorHead {
    in_proj: nn::Linear,
    blocks: Vec<PredictorBlock>,
    out_proj: nn::Linear,
}

pub struct MseJepaForward {
    pub all_tokens: Tensor,
    pub predictions: Tensor,
    pub targets: Tensor,
}

pub struct MseJepaModel {
    bar_proj: nn::Linear,
    bar_enrich_fc1: nn::Linear,
    bar_enrich_fc2: nn::Linear,
    projector: ProjectionMlp,
    layers: Vec<CausalLayer>,
    predictor: PredictorHead,
    feature_scale: Tensor,
    pope_phases: PopePhases,
    dropout: f64,
}

impl MseJepaModel {
    pub fn new(p: &nn::Path) -> Self {
        let bar_proj = nn::linear(p / "bar_proj", BAR_FEATURES, LATENT_DIM, Default::default());
        let bar_enrich_fc1 = nn::linear(
            p / "bar_enrich_fc1",
            LATENT_DIM,
            LATENT_DIM * 2,
            Default::default(),
        );
        let bar_enrich_fc2 = nn::linear(
            p / "bar_enrich_fc2",
            LATENT_DIM * 2,
            LATENT_DIM,
            Default::default(),
        );
        let projector = ProjectionMlp {
            fc1: nn::linear(
                p / "lejepa_projector_fc1",
                LATENT_DIM,
                PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            gamma: p.var(
                "lejepa_projector_norm_gamma",
                &[PROJECTOR_HIDDEN_DIM],
                nn::Init::Const(1.0),
            ),
            fc2: nn::linear(
                p / "lejepa_projector_fc2",
                PROJECTOR_HIDDEN_DIM,
                LATENT_DIM,
                Default::default(),
            ),
        };
        let mut layers = Vec::with_capacity(AR_LAYERS);
        for index in 0..AR_LAYERS {
            let path = p / format!("lejepa_layer_{index}");
            layers.push(CausalLayer {
                qkv: nn::linear(
                    &path / "qkv",
                    LATENT_DIM,
                    LATENT_DIM * 3,
                    Default::default(),
                ),
                pope_theta_bias: init_pope_theta_bias(
                    &path,
                    "pope_theta_bias",
                    HEADS,
                    HEAD_DIM,
                    MAX_CONTEXT_BARS,
                    PopeThetaInit::TwoPi,
                ),
                out_proj: nn::linear(
                    &path / "out_proj",
                    LATENT_DIM,
                    LATENT_DIM,
                    Default::default(),
                ),
                ff_gate: nn::linear(&path / "ff_gate", LATENT_DIM, AR_FF_DIM, Default::default()),
                ff_value: nn::linear(
                    &path / "ff_value",
                    LATENT_DIM,
                    AR_FF_DIM,
                    Default::default(),
                ),
                ff_out: nn::linear(&path / "ff_out", AR_FF_DIM, LATENT_DIM, Default::default()),
            });
        }
        let hidden = LATENT_DIM * PREDICTOR_HIDDEN_MULT;
        let mut blocks = Vec::with_capacity(PREDICTOR_BLOCKS);
        for index in 0..PREDICTOR_BLOCKS {
            let path = p / format!("lejepa_predictor_block_{index}");
            blocks.push(PredictorBlock {
                gate: nn::linear(&path / "gate", LATENT_DIM, hidden, Default::default()),
                value: nn::linear(&path / "value", LATENT_DIM, hidden, Default::default()),
                out: nn::linear(&path / "out", hidden, LATENT_DIM, Default::default()),
            });
        }
        let predictor = PredictorHead {
            in_proj: nn::linear(
                p / "lejepa_predictor_in_proj",
                LATENT_DIM,
                LATENT_DIM,
                Default::default(),
            ),
            blocks,
            out_proj: nn::linear(
                p / "lejepa_predictor_out_proj",
                LATENT_DIM,
                LATENT_DIM,
                Default::default(),
            ),
        };
        let feature_scale = Tensor::from_slice(&OHLC_FEATURE_SCALE).to_device(p.device());
        let pope_positions = Tensor::arange(MAX_CONTEXT_BARS, (Kind::Int64, p.device()));
        let pope_phases = PopePhases::new(&pope_positions, p.device(), POPE_FREQUENCY_BASE);
        Self {
            bar_proj,
            bar_enrich_fc1,
            bar_enrich_fc2,
            projector,
            layers,
            predictor,
            feature_scale,
            pope_phases,
            dropout: DROPOUT,
        }
    }

    pub fn forward(&self, bars: &Tensor, train: bool) -> MseJepaForward {
        let size = bars.size();
        assert_eq!(
            size.len(),
            4,
            "MSE-JEPA bars must be [batch,tickers,time,16]"
        );
        assert_eq!(size[1], 1, "MSE-JEPA is a single-symbol pretrainer");
        assert_eq!(size[2], MAX_CONTEXT_BARS + 1);
        assert_eq!(size[3], BAR_FEATURES);
        // The attached encoder/target branch stays fp32. CUDA autocast is limited
        // to the trunk/head scope so its dense GEMMs use bf16; residual boundaries
        // are explicitly cast back below.
        let all_tokens = autocast(false, || self.encode(bars, train));
        let source = all_tokens.narrow(2, 0, MAX_CONTEXT_BARS);
        let targets = all_tokens.narrow(2, 1, MAX_CONTEXT_BARS);
        let predictions = if source.device().is_cuda() {
            pin_bfloat16_autocast();
            autocast(true, || {
                let beliefs = self.causal_beliefs(&source, train);
                self.predict_next(&beliefs)
            })
        } else {
            let beliefs = self.causal_beliefs(&source, train);
            self.predict_next(&beliefs)
        };
        MseJepaForward {
            all_tokens,
            predictions,
            targets,
        }
    }
    /// Predict the latent immediately following an observed OHLC context.
    pub fn predict_next_latent(&self, bars: &Tensor) -> Tensor {
        let size = bars.size();
        assert_eq!(
            size.len(),
            4,
            "MSE-JEPA context must be [batch,tickers,time,16]"
        );
        assert_eq!(size[1], 1, "MSE-JEPA is a single-symbol model");
        assert!(
            size[2] > 0 && size[2] <= MAX_CONTEXT_BARS,
            "MSE-JEPA context length must be in 1..={MAX_CONTEXT_BARS}"
        );
        assert_eq!(size[3], BAR_FEATURES);
        // Match training precision: encode in fp32, then autocast only the CUDA
        // causal trunk and predictor.
        let tokens = autocast(false, || self.encode(bars, false));
        if tokens.device().is_cuda() {
            pin_bfloat16_autocast();
            autocast(true, || {
                let beliefs = self.causal_beliefs(&tokens, false);
                let last = beliefs.narrow(2, size[2] - 1, 1);
                self.predict_next(&last)
            })
        } else {
            let beliefs = self.causal_beliefs(&tokens, false);
            let last = beliefs.narrow(2, size[2] - 1, 1);
            self.predict_next(&last)
        }
    }

    pub fn encode(&self, bars: &Tensor, train: bool) -> Tensor {
        let size = bars.size();
        let rows = size[0] * size[1] * size[2];
        let features = bars
            .view([rows, BAR_FEATURES])
            .to_kind(Kind::Float)
            .nan_to_num(0.0, 0.0, 0.0);
        let features = features / &self.feature_scale;
        let h = self.bar_proj.forward(&features);
        let enriched = self.bar_enrich_fc2.forward(
            &normalize_last_dim(&self.bar_enrich_fc1.forward(&normalize_last_dim(&h))).gelu("none"),
        );
        let shape = [size[0], size[1], size[2], LATENT_DIM];
        self.project(&(h + enriched).view(shape), train)
    }

    fn project(&self, tokens: &Tensor, _train: bool) -> Tensor {
        let shape = tokens.size();
        let flat = tokens.view([-1, LATENT_DIM]);
        let hidden = self.projector.fc1.forward(&flat);
        let hidden = hidden
            .internal_fused_rms_norm(
                [PROJECTOR_HIDDEN_DIM],
                Some(&self.projector.gamma),
                Some(PROJECTOR_RMS_EPS),
            )
            .0
            .gelu("none");
        self.projector.fc2.forward(&hidden).view(shape.as_slice())
    }

    fn causal_beliefs(&self, tokens: &Tensor, train: bool) -> Tensor {
        let size = tokens.size();
        let rows = size[0] * size[1];
        let length = size[2];
        let narrowed_phases =
            (length != MAX_CONTEXT_BARS).then(|| self.pope_phases.narrow(0, length));
        let phases = narrowed_phases.as_ref().unwrap_or(&self.pope_phases);
        let mut x = tokens.view([rows, length, LATENT_DIM]);
        for layer in &self.layers {
            x = self.causal_layer(&x, layer, phases, train);
        }
        normalize_last_dim(&x).view([size[0], size[1], length, LATENT_DIM])
    }

    fn predict_next(&self, beliefs: &Tensor) -> Tensor {
        let mut h = self
            .predictor
            .in_proj
            .forward(&normalize_last_dim(beliefs))
            .to_kind(Kind::Float);
        for block in &self.predictor.blocks {
            let normed = normalize_last_dim(&h);
            let gated = block.gate.forward(&normed).silu() * block.value.forward(&normed);
            h += block.out.forward(&gated).to_kind(Kind::Float);
        }
        self.predictor
            .out_proj
            .forward(&normalize_last_dim(&h))
            .to_kind(Kind::Float)
    }

    fn causal_layer(
        &self,
        source: &Tensor,
        layer: &CausalLayer,
        phases: &PopePhases,
        train: bool,
    ) -> Tensor {
        let size = source.size();
        let rows = size[0];
        let length = size[1];
        let qkv = layer.qkv.forward(&normalize_last_dim(source));
        let parts = qkv.split(LATENT_DIM, -1);
        let reshape = |tensor: &Tensor| tensor.view([rows, length, HEADS, HEAD_DIM]);
        let q = reshape(&parts[0]);
        let k = reshape(&parts[1]);
        let v = reshape(&parts[2]);
        let kind = if source.device().is_cuda() {
            Kind::BFloat16
        } else {
            source.kind()
        };
        let polar = pope_expand_qk(&q, &k, phases, &layer.pope_theta_bias, kind);
        let value = v.to_kind(kind).contiguous();
        let attended = strict_attention(&polar, &value)
            .to_kind(Kind::Float)
            .contiguous()
            .view([rows, length, LATENT_DIM]);
        let attention_residual = layer
            .out_proj
            .forward(&attended)
            .to_kind(Kind::Float)
            .dropout(self.dropout, train);
        let x = source + attention_residual;
        let normed = normalize_last_dim(&x);
        let ff = layer
            .ff_out
            .forward(&(layer.ff_gate.forward(&normed).silu() * layer.ff_value.forward(&normed)))
            .to_kind(Kind::Float);
        x + ff.dropout(self.dropout, train)
    }
}

fn strict_attention(qk: &PolarQk, value: &Tensor) -> Tensor {
    if value.device().is_cuda() {
        return autocast(true, || pope_flash_attention_prefill(qk, value))
            .unwrap_or_else(|error| panic!("strict FA4 PoPE prefill failed: {error:#}"));
    }
    #[cfg(test)]
    {
        return pope_attention_reference(qk, value, true);
    }
    #[cfg(not(test))]
    panic!("MSE-JEPA pretraining requires CUDA with the strict FA4 bridge");
}

fn normalize_last_dim(x: &Tensor) -> Tensor {
    let mean = x.mean_dim([-1i64].as_slice(), true, Kind::Float);
    let centered = x - &mean;
    let variance = centered
        .square()
        .mean_dim([-1i64].as_slice(), true, Kind::Float);
    centered / (variance + NORMALIZATION_EPS).sqrt()
}

const _: () = {
    assert!(LATENT_DIM == HEADS * HEAD_DIM);
    assert!(BAR_FEATURES == 16);
    assert!(MAX_CONTEXT_BARS == 6_000);
};

#[cfg(test)]
mod tests {
    use super::{MseJepaModel, AR_LAYERS, PREDICTOR_BLOCKS};
    use crate::torch::test_rng;
    use std::collections::BTreeSet;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn tensor_names_are_exactly_the_c277_mse_jepa_core() {
        let _torch_rng_guard = test_rng::shared();
        let var_store = nn::VarStore::new(Device::Cpu);
        let _model = MseJepaModel::new(&var_store.root());
        let actual = var_store.variables().into_keys().collect::<BTreeSet<_>>();
        let mut expected = [
            "bar_proj.weight",
            "bar_proj.bias",
            "bar_enrich_fc1.weight",
            "bar_enrich_fc1.bias",
            "bar_enrich_fc2.weight",
            "bar_enrich_fc2.bias",
            "lejepa_projector_fc1.weight",
            "lejepa_projector_fc1.bias",
            "lejepa_projector_norm_gamma",
            "lejepa_projector_fc2.weight",
            "lejepa_projector_fc2.bias",
            "lejepa_predictor_in_proj.weight",
            "lejepa_predictor_in_proj.bias",
            "lejepa_predictor_out_proj.weight",
            "lejepa_predictor_out_proj.bias",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        for layer in 0..AR_LAYERS {
            expected.insert(format!("lejepa_layer_{layer}.pope_theta_bias"));
            for projection in ["qkv", "out_proj", "ff_gate", "ff_value", "ff_out"] {
                expected.insert(format!("lejepa_layer_{layer}.{projection}.weight"));
                expected.insert(format!("lejepa_layer_{layer}.{projection}.bias"));
            }
        }
        for block in 0..PREDICTOR_BLOCKS {
            for projection in ["gate", "value", "out"] {
                expected.insert(format!(
                    "lejepa_predictor_block_{block}.{projection}.weight"
                ));
                expected.insert(format!("lejepa_predictor_block_{block}.{projection}.bias"));
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn restored_model_exposes_next_latent_inference() {
        let _torch_rng_guard = test_rng::shared();
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        let bars = Tensor::randn([2, 1, 4, 16], (Kind::Float, Device::Cpu));
        let prediction = model.predict_next_latent(&bars);
        assert_eq!(prediction.size(), vec![2, 1, 1, 256]);
        assert!(prediction.isfinite().all().int64_value(&[]) != 0);
    }
}
