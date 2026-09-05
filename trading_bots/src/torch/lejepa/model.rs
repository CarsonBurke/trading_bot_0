use crate::torch::cuda::cfg::pin_bfloat16_autocast;
use tch::{autocast, nn, nn::Module, Kind, Tensor};

use super::checkpoint::EmissionGradientMode;
use crate::torch::bar_dist::{BarEmissionHead, BarSupports};
use crate::torch::fa4::{pope_flash_attention_decode_q1, pope_flash_attention_prefill};
#[cfg(test)]
use crate::torch::pope::pope_attention_reference;
use crate::torch::pope::{
    init_pope_theta_bias, pope_expand_qk, PolarQk, PopePhases, PopeThetaInit, POPE_FREQUENCY_BASE,
};

pub const ARCHITECTURE: &str =
    "lejepa-dbwm-transition7-causal-ar-pope64-fa4-flow-two-stage-readout-v13";
pub const FEATURE_LAYOUT: &str = "transition-bar-ohlcv-logshape-gk-volume-delta-fixed-scale-v4";
pub const LATENT_DIM: i64 = 512;
pub const BAR_FEATURES: i64 = 7;
pub const MAX_CONTEXT_BARS: i64 = 6_000;
pub const AR_LAYERS: usize = 6;
pub const AR_FF_DIM: i64 = 1_024;
pub const HEADS: i64 = 8;
pub const HEAD_DIM: i64 = 64;
pub const PROJECTOR_HIDDEN_DIM: i64 = 2_048;
pub const FLOW_BLOCKS: usize = 3;
pub const FLOW_HIDDEN_DIM: i64 = 2_048;
/// DiT-style sinusoidal `tau` features: `FLOW_TIME_FEATURES / 2` geometric frequencies
/// down to `1 / FLOW_TIME_MAX_PERIOD`, applied to `tau * FLOW_TIME_SCALE`, as `cat(cos, sin)`.
pub const FLOW_TIME_FEATURES: i64 = 256;
pub const FLOW_TIME_SCALE: f64 = 1_000.0;
pub const FLOW_TIME_MAX_PERIOD: f64 = 10_000.0;
pub const NORMALIZATION_EPS: f64 = 1e-5;
pub const PROJECTOR_RMS_EPS: f64 = 1e-6;
pub const DROPOUT: f64 = 0.1;
pub const TRANSITION_BAR_FEATURE_SCALE: [f32; BAR_FEATURES as usize] =
    [4e-3, 2e-3, 2e-3, 2e-3, 2e-3, 0.5, 1.0];

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

/// adaLN-zero gated-MLP block: `x + gate * mlp(norm(x) * (1 + scale) + shift)` with
/// `(shift, scale, gate)` read from the conditioning vector through a zero-initialised
/// projection, so every block is the identity at init.
struct FlowBlock {
    modulation: nn::Linear,
    gate: nn::Linear,
    value: nn::Linear,
    out: nn::Linear,
}

/// Conditional flow-matching velocity field `v(x_tau, tau, h)` over next-bar tokens.
struct FlowPredictor {
    time_frequencies: Tensor,
    time_fc1: nn::Linear,
    time_fc2: nn::Linear,
    belief_proj: nn::Linear,
    in_proj: nn::Linear,
    blocks: Vec<FlowBlock>,
    final_modulation: nn::Linear,
    out_proj: nn::Linear,
}

/// Belief-side flow conditioning shared by every velocity evaluation on the same rows.
struct FlowConditioning {
    normed: Tensor,
    projected: Tensor,
}

pub struct MseJepaForward {
    pub all_tokens: Tensor,
    /// `[batch, 1, MAX_CONTEXT_BARS, LATENT_DIM]` normalized causal beliefs.
    pub beliefs: Tensor,
    /// `[batch, 1, MAX_CONTEXT_BARS, LATENT_DIM]` attached same-pass next-bar tokens.
    pub targets: Tensor,
}

pub struct MseJepaModel {
    bar_proj: nn::Linear,
    bar_enrich_fc1: nn::Linear,
    bar_enrich_fc2: nn::Linear,
    projector: ProjectionMlp,
    layers: Vec<CausalLayer>,
    flow: FlowPredictor,
    emission: BarEmissionHead,
    feature_scale: Tensor,
    pope_phases: PopePhases,
    dropout: f64,
}

/// Separately constructible same-time factorized token readout. It is deliberately absent
/// from [`MseJepaModel`] so core checkpoints and optimizers cannot carry an online probe.
pub struct MseJepaTokenProbe {
    head: BarEmissionHead,
}

impl MseJepaTokenProbe {
    pub fn new(p: &nn::Path) -> Self {
        Self {
            head: BarEmissionHead::new_unconditioned(p, LATENT_DIM),
        }
    }

    pub fn logits(&self, tokens: &Tensor, target_bins: &Tensor) -> Tensor {
        self.head.logits_unconditioned(tokens, target_bins)
    }

    pub fn reset(&mut self, seed: u64) {
        self.head.reset_unconditioned(seed);
    }
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
        let mut blocks = Vec::with_capacity(FLOW_BLOCKS);
        for index in 0..FLOW_BLOCKS {
            let path = p / format!("lejepa_flow_block_{index}");
            blocks.push(FlowBlock {
                modulation: zero_linear(&path / "modulation", LATENT_DIM, LATENT_DIM * 3),
                gate: nn::linear(
                    &path / "gate",
                    LATENT_DIM,
                    FLOW_HIDDEN_DIM,
                    Default::default(),
                ),
                value: nn::linear(
                    &path / "value",
                    LATENT_DIM,
                    FLOW_HIDDEN_DIM,
                    Default::default(),
                ),
                out: nn::linear(
                    &path / "out",
                    FLOW_HIDDEN_DIM,
                    LATENT_DIM,
                    Default::default(),
                ),
            });
        }
        let half = FLOW_TIME_FEATURES / 2;
        let time_frequencies = (Tensor::arange(half, (Kind::Float, p.device()))
            * (-FLOW_TIME_MAX_PERIOD.ln() / half as f64))
            .exp();
        let flow = FlowPredictor {
            time_frequencies,
            time_fc1: nn::linear(
                p / "lejepa_flow_time_fc1",
                FLOW_TIME_FEATURES,
                LATENT_DIM,
                Default::default(),
            ),
            time_fc2: nn::linear(
                p / "lejepa_flow_time_fc2",
                LATENT_DIM,
                LATENT_DIM,
                Default::default(),
            ),
            belief_proj: nn::linear(
                p / "lejepa_flow_belief_proj",
                LATENT_DIM,
                LATENT_DIM,
                Default::default(),
            ),
            in_proj: nn::linear(
                p / "lejepa_flow_in_proj",
                LATENT_DIM * 2,
                LATENT_DIM,
                Default::default(),
            ),
            blocks,
            final_modulation: zero_linear(
                p / "lejepa_flow_final_modulation",
                LATENT_DIM,
                LATENT_DIM * 2,
            ),
            out_proj: zero_linear(p / "lejepa_flow_out_proj", LATENT_DIM, LATENT_DIM),
        };
        let emission = BarEmissionHead::new_unconditioned(&(p / "lejepa_emission"), LATENT_DIM);
        let feature_scale = Tensor::from_slice(&TRANSITION_BAR_FEATURE_SCALE).to_device(p.device());
        let pope_positions = Tensor::arange(MAX_CONTEXT_BARS, (Kind::Int64, p.device()));
        let pope_phases = PopePhases::new(&pope_positions, p.device(), POPE_FREQUENCY_BASE);
        Self {
            bar_proj,
            bar_enrich_fc1,
            bar_enrich_fc2,
            projector,
            layers,
            flow,
            emission,
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
            "MSE-JEPA bars must be [batch,tickers,time,7]"
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
        let beliefs = if source.device().is_cuda() {
            pin_bfloat16_autocast();
            autocast(true, || self.causal_beliefs(&source, train))
        } else {
            self.causal_beliefs(&source, train)
        };
        MseJepaForward {
            all_tokens,
            beliefs,
            targets,
        }
    }

    /// Flow velocity `v(x_tau, tau, h)` for interpolants `x_tau` `[..., LATENT_DIM]`,
    /// times `tau` `[..., 1]` (broadcastable over the lead dims), and beliefs sharing the
    /// interpolants' lead dims. Callers own the autocast scope, as for [`Self::emission_logits`].
    pub fn velocity(&self, x_tau: &Tensor, tau: &Tensor, beliefs: &Tensor) -> Tensor {
        let conditioning = self.flow_conditioning(&normalize_last_dim(beliefs));
        self.velocity_field(x_tau, &self.time_embedding(tau), &conditioning)
    }

    /// Teacher-forced belief-conditioned emission logits `[..., BAR_DOF, NUM_BAR_BINS]`.
    /// `target_bins` must share the beliefs' leading dimensions and come from
    /// [`BarSupports::bin_ids`].
    pub fn emission_logits(&self, beliefs: &Tensor, target_bins: &Tensor) -> Tensor {
        self.emission.logits_unconditioned(beliefs, target_bins)
    }

    /// Core-training emission logits. The experiment arm changes only the CE gradient edge
    /// into causal beliefs; values and deployed head parameters are identical.
    pub(crate) fn training_emission_logits(
        &self,
        beliefs: &Tensor,
        target_bins: &Tensor,
        mode: EmissionGradientMode,
    ) -> Tensor {
        match mode {
            EmissionGradientMode::Attached => self.emission_logits(beliefs, target_bins),
            EmissionGradientMode::Detached => self.emission_logits(&beliefs.detach(), target_bins),
        }
    }

    /// Reset the entire deployed readout independently of backbone construction RNG.
    pub(crate) fn reset_emission(&mut self, seed: u64) {
        self.emission.reset_unconditioned(seed);
    }

    /// Ancestral 5-DOF bar sample from beliefs, `[..., BAR_DOF]` raw values.
    pub fn sample_bars(
        &self,
        beliefs: &Tensor,
        supports: &BarSupports,
        temperature: f64,
    ) -> Tensor {
        self.emission
            .sample_unconditioned(beliefs, supports, temperature)
    }

    /// Next-latent draws `[samples, ..beliefs dims..]` from `steps` Heun steps of the flow
    /// ODE started at fresh `N(0, I)` noise.
    pub(crate) fn sample_next_latents(&self, beliefs: &Tensor, samples: i64, steps: i64) -> Tensor {
        assert!(samples > 0, "flow sampling needs at least one draw");
        let mut shape = vec![samples];
        shape.extend_from_slice(&beliefs.size());
        let initial = tch::no_grad(|| Tensor::randn(shape, (Kind::Float, beliefs.device())));
        self.sample_next_latents_from(beliefs, &initial, steps)
    }

    /// [`Self::sample_next_latents`] started from caller-owned base noise
    /// `[samples, ..beliefs dims..]`, so a fixed validation panel scores the same draws.
    pub(crate) fn sample_next_latents_from(
        &self,
        beliefs: &Tensor,
        initial: &Tensor,
        steps: i64,
    ) -> Tensor {
        validate_sequence(beliefs, LATENT_DIM, "MSE-JEPA causal beliefs");
        assert!(steps > 0, "Heun sampling needs at least one ODE step");
        let size = initial.size();
        assert!(
            size.len() == 5 && size[0] > 0 && size[1..] == beliefs.size()[..],
            "flow base noise must be [samples, ..beliefs dims..]"
        );
        tch::no_grad(|| {
            if beliefs.device().is_cuda() {
                pin_bfloat16_autocast();
                autocast(true, || self.heun_sample(beliefs, initial, steps))
            } else {
                self.heun_sample(beliefs, initial, steps)
            }
        })
    }

    /// Frozen-representation entry point used by rollout evaluation.
    ///
    /// The returned token is the projector output that the causal trunk was trained on, not the
    /// non-contextual `h + enriched` encoder residual. Keeping this crate-visible prevents
    /// rollout code from reaching into mutable model internals while making the re-encode
    /// contract explicit.
    pub(crate) fn post_projector_tokens(&self, bars: &Tensor) -> Tensor {
        validate_sequence(bars, BAR_FEATURES, "MSE-JEPA bars");
        autocast(false, || self.encode(bars, false))
    }

    /// Prefill the trunk KV cache from projected tokens, returning the normalized causal
    /// beliefs for every prefilled position.
    pub(crate) fn prefill_cached(&self, tokens: &Tensor, cache: &mut MseJepaKvCache) -> Tensor {
        validate_sequence(tokens, LATENT_DIM, "MSE-JEPA projected tokens");
        if tokens.device().is_cuda() {
            pin_bfloat16_autocast();
            autocast(true, || self.prefill_inner(tokens, cache))
        } else {
            self.prefill_inner(tokens, cache)
        }
    }

    /// Append one projected token per row to the cached context and return its normalized
    /// belief `[batch, 1, 1, LATENT_DIM]`.
    pub(crate) fn decode_cached(&self, token: &Tensor, cache: &mut MseJepaKvCache) -> Tensor {
        let size = token.size();
        assert_eq!(
            size,
            [size[0], 1, 1, LATENT_DIM],
            "cached decode consumes exactly one token per row"
        );
        if token.device().is_cuda() {
            pin_bfloat16_autocast();
            autocast(true, || self.decode_inner(token, cache))
        } else {
            self.decode_inner(token, cache)
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
            x = self.causal_layer_kv(&x, layer, phases, train).0;
        }
        normalize_last_dim(&x).view([size[0], size[1], length, LATENT_DIM])
    }

    fn time_embedding(&self, tau: &Tensor) -> Tensor {
        let angles = tau.to_kind(Kind::Float) * FLOW_TIME_SCALE * &self.flow.time_frequencies;
        let features = Tensor::cat(&[angles.cos(), angles.sin()], -1);
        self.flow
            .time_fc2
            .forward(&self.flow.time_fc1.forward(&features).silu())
    }

    fn flow_conditioning(&self, normed: &Tensor) -> FlowConditioning {
        FlowConditioning {
            normed: normed.shallow_clone(),
            projected: self.flow.belief_proj.forward(normed),
        }
    }

    fn velocity_field(
        &self,
        x_tau: &Tensor,
        time_embedding: &Tensor,
        conditioning: &FlowConditioning,
    ) -> Tensor {
        let flow = &self.flow;
        let c = (&conditioning.projected + time_embedding).silu();
        let mut x = flow
            .in_proj
            .forward(&Tensor::cat(&[x_tau, &conditioning.normed], -1))
            .to_kind(Kind::Float);
        for block in &flow.blocks {
            let parts = block
                .modulation
                .forward(&c)
                .to_kind(Kind::Float)
                .chunk(3, -1);
            let modulated = normalize_last_dim(&x) * (&parts[1] + 1.0) + &parts[0];
            let gated = block.gate.forward(&modulated).silu() * block.value.forward(&modulated);
            x += &parts[2] * block.out.forward(&gated).to_kind(Kind::Float);
        }
        let parts = flow
            .final_modulation
            .forward(&c)
            .to_kind(Kind::Float)
            .chunk(2, -1);
        flow.out_proj
            .forward(&(normalize_last_dim(&x) * (&parts[1] + 1.0) + &parts[0]))
            .to_kind(Kind::Float)
    }

    /// Heun on the uniform grid `tau_k = k / steps`, every fan draw batched per ODE step;
    /// the velocity parameterisation has no `tau = 1` singularity, so the last step is full.
    fn heun_sample(&self, beliefs: &Tensor, initial: &Tensor, steps: i64) -> Tensor {
        let shape = initial.size();
        // Project once per belief row, then fan out: the projection is tau-independent.
        let conditioning = self.flow_conditioning(&normalize_last_dim(beliefs));
        let fan = |tensor: &Tensor| {
            tensor
                .unsqueeze(0)
                .expand(shape.as_slice(), true)
                .reshape([-1, LATENT_DIM])
        };
        let conditioning = FlowConditioning {
            normed: fan(&conditioning.normed),
            projected: fan(&conditioning.projected),
        };
        let device = initial.device();
        // Fresh storage: a flattened fp32 view would alias the caller's base noise and the
        // in-place integration below would overwrite it.
        let mut x = initial
            .to_kind(Kind::Float)
            .reshape([-1, LATENT_DIM])
            .copy();
        let dt = 1.0 / steps as f64;
        let grid = |k: i64| Tensor::full([1, 1], k as f64 * dt, (Kind::Float, device));
        let mut embedding = self.time_embedding(&grid(0));
        for k in 0..steps {
            let v1 = self.velocity_field(&x, &embedding, &conditioning);
            embedding = self.time_embedding(&grid(k + 1));
            let v2 = self.velocity_field(&(&x + &v1 * dt), &embedding, &conditioning);
            x += (v1 + v2) * (dt * 0.5);
        }
        x.reshape(shape)
    }

    fn prefill_inner(&self, tokens: &Tensor, cache: &mut MseJepaKvCache) -> Tensor {
        let size = tokens.size();
        let rows = size[0] * size[1];
        let length = size[2];
        let narrowed_phases =
            (length != MAX_CONTEXT_BARS).then(|| self.pope_phases.narrow(0, length));
        let phases = narrowed_phases.as_ref().unwrap_or(&self.pope_phases);
        let capacity = ((length as u64).next_power_of_two() as i64).min(MAX_CONTEXT_BARS);
        let mut x = tokens.view([rows, length, LATENT_DIM]);
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next, key, value) = self.causal_layer_kv(&x, layer, phases, false);
            x = next;
            layers.push(MseJepaLayerKv::prefilled(&key, &value, capacity));
        }
        cache.layers = layers;
        cache.length = length;
        cache.next_position = length;
        cache.write_index = length % capacity;
        normalize_last_dim(&x).view([size[0], size[1], length, LATENT_DIM])
    }

    fn decode_inner(&self, token: &Tensor, cache: &mut MseJepaKvCache) -> Tensor {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "MSE-JEPA KV cache has an incompatible layer count"
        );
        cache.ensure_append_capacity();
        let size = token.size();
        let rows = size[0] * size[1];
        let position = Tensor::from_slice(&[cache.next_position]).to_device(token.device());
        let phases = PopePhases::new(&position, token.device(), POPE_FREQUENCY_BASE);
        let write_index = cache.write_index;
        let previous_length = cache.length;
        let mut x = token.view([rows, 1, LATENT_DIM]);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let qkv = layer.qkv.forward(&normalize_last_dim(&x));
            let parts = qkv.split(LATENT_DIM, -1);
            let reshape = |tensor: &Tensor| tensor.view([rows, 1, HEADS, HEAD_DIM]);
            let q = reshape(&parts[0]);
            let k = reshape(&parts[1]);
            let v = reshape(&parts[2]);
            let kind = if x.device().is_cuda() {
                Kind::BFloat16
            } else {
                x.kind()
            };
            let polar = pope_expand_qk(&q, &k, &phases, &layer.pope_theta_bias, kind);
            let value = v.to_kind(kind).contiguous();
            layer_cache.key.narrow(1, write_index, 1).copy_(&polar.key);
            layer_cache.value.narrow(1, write_index, 1).copy_(&value);
            let (active_key, active_value) = layer_cache.active_after_write(previous_length);
            let attended = strict_attention_decode(&polar.query, &active_key, &active_value)
                .to_kind(Kind::Float)
                .contiguous()
                .view([rows, 1, LATENT_DIM]);
            let with_attention = &x + layer.out_proj.forward(&attended).to_kind(Kind::Float);
            let normed = normalize_last_dim(&with_attention);
            let ff = layer
                .ff_out
                .forward(&(layer.ff_gate.forward(&normed).silu() * layer.ff_value.forward(&normed)))
                .to_kind(Kind::Float);
            x = with_attention + ff;
        }
        cache.finish_append();
        normalize_last_dim(&x).view([size[0], size[1], 1, LATENT_DIM])
    }

    fn causal_layer_kv(
        &self,
        source: &Tensor,
        layer: &CausalLayer,
        phases: &PopePhases,
        train: bool,
    ) -> (Tensor, Tensor, Tensor) {
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
        (x + ff.dropout(self.dropout, train), polar.key, value)
    }
}

/// Circular per-layer KV cache holding PoPE-expanded keys and raw values, mirroring the
/// bar world model's cache contract: positions are absolute and baked into the key
/// phases, so eviction is a plain modular overwrite and the attention span stays within
/// the trained [`MAX_CONTEXT_BARS`] window of relative distances.
pub struct MseJepaKvCache {
    layers: Vec<MseJepaLayerKv>,
    next_position: i64,
    length: i64,
    write_index: i64,
}

struct MseJepaLayerKv {
    /// `[rows, capacity, HEADS, POPE_QK_DIM]`
    key: Tensor,
    /// `[rows, capacity, HEADS, POPE_DIM]`
    value: Tensor,
}

impl MseJepaLayerKv {
    fn prefilled(key: &Tensor, value: &Tensor, capacity: i64) -> Self {
        let len = key.size()[1];
        let storage = |source: &Tensor| {
            let size = source.size();
            let buffer = Tensor::zeros(
                [size[0], capacity, size[2], size[3]],
                (source.kind(), source.device()),
            );
            buffer.narrow(1, 0, len).copy_(source);
            buffer
        };
        Self {
            key: storage(key),
            value: storage(value),
        }
    }

    fn repeat_batch(&self, factor: i64) -> Self {
        Self {
            key: self.key.repeat_interleave_self_int(factor, 0, None),
            value: self.value.repeat_interleave_self_int(factor, 0, None),
        }
    }

    fn grow(&mut self, capacity: i64, length: i64) {
        let grown = |source: &Tensor| {
            let size = source.size();
            let buffer = Tensor::zeros(
                [size[0], capacity, size[2], size[3]],
                (source.kind(), source.device()),
            );
            buffer
                .narrow(1, 0, length)
                .copy_(&source.narrow(1, 0, length));
            buffer
        };
        self.key = grown(&self.key);
        self.value = grown(&self.value);
    }

    fn active_after_write(&self, previous_length: i64) -> (Tensor, Tensor) {
        let capacity = self.key.size()[1];
        if previous_length < capacity {
            let length = previous_length + 1;
            return (
                self.key.narrow(1, 0, length),
                self.value.narrow(1, 0, length),
            );
        }
        (self.key.shallow_clone(), self.value.shallow_clone())
    }
}

impl MseJepaKvCache {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            next_position: 0,
            length: 0,
            write_index: 0,
        }
    }

    pub fn cached_bars(&self) -> i64 {
        self.length
    }

    pub fn next_position(&self) -> i64 {
        self.next_position
    }

    /// Interleave every row `factor` times, so one prefill over `B` histories serves
    /// `B * factor` independent continuations. Row order matches
    /// `Tensor::repeat_interleave` on dim 0.
    pub fn repeat_batch(&self, factor: i64) -> Self {
        assert!(factor > 0, "cache batch factor must be positive");
        Self {
            layers: self
                .layers
                .iter()
                .map(|layer| layer.repeat_batch(factor))
                .collect(),
            next_position: self.next_position,
            length: self.length,
            write_index: self.write_index,
        }
    }

    fn capacity(&self) -> i64 {
        self.layers
            .first()
            .map(|layer| layer.key.size()[1])
            .unwrap_or(0)
    }

    fn ensure_append_capacity(&mut self) {
        let capacity = self.capacity();
        if self.length < capacity || capacity == MAX_CONTEXT_BARS {
            return;
        }
        let grown = (capacity.max(1) * 2).min(MAX_CONTEXT_BARS);
        for layer in &mut self.layers {
            layer.grow(grown, self.length);
        }
        self.write_index = self.length;
    }

    fn finish_append(&mut self) {
        let capacity = self.capacity();
        if self.length < MAX_CONTEXT_BARS {
            self.length += 1;
        }
        self.write_index = (self.write_index + 1) % capacity;
        self.next_position += 1;
    }
}

impl Default for MseJepaKvCache {
    fn default() -> Self {
        Self::new()
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

fn strict_attention_decode(query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
    if value.device().is_cuda() {
        return autocast(true, || pope_flash_attention_decode_q1(query, key, value))
            .unwrap_or_else(|error| panic!("strict FA4 PoPE decode failed: {error:#}"));
    }
    #[cfg(test)]
    {
        return pope_attention_reference(
            &PolarQk {
                query: query.shallow_clone(),
                key: key.shallow_clone(),
            },
            value,
            false,
        );
    }
    #[cfg(not(test))]
    panic!("MSE-JEPA cached decode requires CUDA with the strict FA4 bridge");
}

fn validate_sequence(tensor: &Tensor, width: i64, label: &str) {
    let size = tensor.size();
    assert_eq!(
        size.len(),
        4,
        "{label} must be [batch,tickers,time,{width}]"
    );
    assert_eq!(size[1], 1, "MSE-JEPA is a single-symbol model");
    assert!(
        size[2] > 0 && size[2] <= MAX_CONTEXT_BARS,
        "MSE-JEPA sequence length must be in 1..={MAX_CONTEXT_BARS}"
    );
    assert_eq!(size[3], width, "{label} width mismatch");
}

fn normalize_last_dim(x: &Tensor) -> Tensor {
    let mean = x.mean_dim([-1i64].as_slice(), true, Kind::Float);
    let centered = x - &mean;
    let variance = centered
        .square()
        .mean_dim([-1i64].as_slice(), true, Kind::Float);
    centered / (variance + NORMALIZATION_EPS).sqrt()
}

fn zero_linear(path: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    nn::linear(
        path,
        in_dim,
        out_dim,
        nn::LinearConfig {
            ws_init: nn::Init::Const(0.0),
            bs_init: Some(nn::Init::Const(0.0)),
            bias: true,
        },
    )
}

const _: () = {
    assert!(LATENT_DIM == HEADS * HEAD_DIM);
    assert!(BAR_FEATURES == 7);
    assert!(MAX_CONTEXT_BARS == 6_000);
    assert!(FLOW_TIME_FEATURES % 2 == 0);
    assert!(FLOW_BLOCKS == 3 && FLOW_HIDDEN_DIM == 4 * LATENT_DIM);
};

#[cfg(test)]
mod tests {
    use super::{
        MseJepaKvCache, MseJepaModel, MseJepaTokenProbe, ARCHITECTURE, AR_LAYERS, BAR_FEATURES,
        FEATURE_LAYOUT, FLOW_BLOCKS, FLOW_TIME_FEATURES, LATENT_DIM, TRANSITION_BAR_FEATURE_SCALE,
    };
    use crate::torch::bar_dist::{BAR_DOF, BAR_DOF_NAMES, NUM_BAR_BINS};
    use crate::torch::lejepa::checkpoint::EmissionGradientMode;
    use crate::torch::test_rng;
    use std::collections::BTreeSet;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn tensor_names_preserve_the_dbwm_core_capacity() {
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
            "lejepa_flow_time_fc1.weight",
            "lejepa_flow_time_fc1.bias",
            "lejepa_flow_time_fc2.weight",
            "lejepa_flow_time_fc2.bias",
            "lejepa_flow_belief_proj.weight",
            "lejepa_flow_belief_proj.bias",
            "lejepa_flow_in_proj.weight",
            "lejepa_flow_in_proj.bias",
            "lejepa_flow_final_modulation.weight",
            "lejepa_flow_final_modulation.bias",
            "lejepa_flow_out_proj.weight",
            "lejepa_flow_out_proj.bias",
            "lejepa_emission.bar_prefix_embed",
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
        for block in 0..FLOW_BLOCKS {
            for projection in ["modulation", "gate", "value", "out"] {
                expected.insert(format!("lejepa_flow_block_{block}.{projection}.weight"));
                expected.insert(format!("lejepa_flow_block_{block}.{projection}.bias"));
            }
        }
        for name in BAR_DOF_NAMES {
            expected.insert(format!("lejepa_emission.bar_dof_head_{name}.weight"));
            expected.insert(format!("lejepa_emission.bar_dof_head_{name}.bias"));
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn transition_bar_feature_contract_is_versioned_and_fixed_scale() {
        assert_eq!(BAR_FEATURES, 7);
        assert_eq!(
            TRANSITION_BAR_FEATURE_SCALE,
            [4e-3, 2e-3, 2e-3, 2e-3, 2e-3, 0.5, 1.0]
        );
        assert_eq!(
            ARCHITECTURE,
            "lejepa-dbwm-transition7-causal-ar-pope64-fa4-flow-two-stage-readout-v13"
        );
        assert_eq!(
            FEATURE_LAYOUT,
            "transition-bar-ohlcv-logshape-gk-volume-delta-fixed-scale-v4"
        );
    }

    /// Pushes the adaLN-zero head off its identity init so a sampler test exercises a
    /// non-trivial velocity field.
    fn randomize_flow_zero_init(var_store: &nn::VarStore) {
        tch::no_grad(|| {
            for (name, mut variable) in var_store.variables() {
                if name.contains("modulation") || name.starts_with("lejepa_flow_out_proj") {
                    variable.copy_(&(Tensor::randn_like(&variable) * 0.05));
                }
            }
        });
    }

    #[test]
    fn flow_head_is_the_identity_with_zero_velocity_at_init() {
        let _torch_rng_guard = test_rng::shared();
        tch::manual_seed(17);
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        let flow_parameters: i64 = var_store
            .variables()
            .iter()
            .filter(|(name, _)| name.starts_with("lejepa_flow_"))
            .map(|(_, variable)| variable.numel() as i64)
            .sum();
        assert_eq!(flow_parameters, 13_784_576);
        let beliefs = Tensor::randn([2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu));
        let x_tau = Tensor::randn([2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu));
        let tau = Tensor::rand([2, 1, 4, 1], (Kind::Float, Device::Cpu));
        let velocity = model.velocity(&x_tau, &tau, &beliefs);
        assert_eq!(velocity.size(), vec![2, 1, 4, LATENT_DIM]);
        assert_eq!(velocity.abs().max().double_value(&[]), 0.0);
        let initial = Tensor::randn([3, 2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu));
        let sampled = model.sample_next_latents_from(&beliefs, &initial, 4);
        assert!(
            sampled.equal(&initial),
            "zero velocity must leave base noise untouched"
        );
    }

    #[test]
    fn sinusoidal_time_features_separate_the_flow_endpoints() {
        let _torch_rng_guard = test_rng::shared();
        tch::manual_seed(17);
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        let frequencies = &model.flow.time_frequencies;
        assert_eq!(frequencies.size(), vec![FLOW_TIME_FEATURES / 2]);
        assert!((frequencies.double_value(&[0]) - 1.0).abs() < 1e-6);
        assert!(frequencies.double_value(&[FLOW_TIME_FEATURES / 2 - 1]) > 1e-4);
        let start = model.time_embedding(&Tensor::zeros([1, 1], (Kind::Float, Device::Cpu)));
        let end = model.time_embedding(&Tensor::ones([1, 1], (Kind::Float, Device::Cpu)));
        assert_eq!(start.size(), vec![1, LATENT_DIM]);
        assert!(start.isfinite().all().int64_value(&[]) != 0);
        assert!((start - end).abs().max().double_value(&[]) > 1e-3);
    }

    #[test]
    fn heun_samples_are_shaped_finite_integrated_and_noise_dependent() {
        let _torch_rng_guard = test_rng::shared();
        tch::manual_seed(17);
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        randomize_flow_zero_init(&var_store);
        let beliefs = Tensor::randn([2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu));
        let drawn = model.sample_next_latents(&beliefs, 3, 8);
        assert_eq!(drawn.size(), vec![3, 2, 1, 4, LATENT_DIM]);
        assert!(drawn.isfinite().all().int64_value(&[]) != 0);
        let spread = (drawn.narrow(0, 0, 1) - drawn.narrow(0, 1, 1))
            .abs()
            .max()
            .double_value(&[]);
        assert!(spread > 0.0, "independent base draws must differ");

        let initial = Tensor::randn([2, 2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu));
        let untouched = initial.copy();
        let coarse = model.sample_next_latents_from(&beliefs, &initial, 1);
        let fine = model.sample_next_latents_from(&beliefs, &initial, 8);
        let replay = model.sample_next_latents_from(&beliefs, &initial, 8);
        assert!(
            initial.equal(&untouched),
            "sampling must not mutate the base noise"
        );
        assert!(fine.equal(&replay), "fixed base noise must replay exactly");
        assert!((&fine - &initial).abs().max().double_value(&[]) > 1e-4);
        assert!((&fine - &coarse).abs().max().double_value(&[]) > 1e-6);
    }

    #[test]
    fn emission_logits_are_belief_conditioned_per_chain_factor() {
        let _torch_rng_guard = test_rng::shared();
        tch::manual_seed(17);
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        let beliefs = Tensor::randn([2, 1, 3, LATENT_DIM], (Kind::Float, Device::Cpu));
        let bins = Tensor::zeros([2, 1, 3, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let logits = model.emission_logits(&beliefs, &bins);
        assert_eq!(logits.size(), vec![2, 1, 3, BAR_DOF as i64, NUM_BAR_BINS]);
        assert!(logits.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn emission_gradient_arms_have_identical_logits_and_differ_only_at_beliefs() {
        let _torch_rng_guard = test_rng::exclusive();
        tch::manual_seed(17);
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        let values = Tensor::randn([2, 1, 3, LATENT_DIM], (Kind::Float, Device::Cpu));
        let attached_beliefs = values.copy().set_requires_grad(true);
        let detached_beliefs = values.copy().set_requires_grad(true);
        let bins = Tensor::zeros([2, 1, 3, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let attached = model.training_emission_logits(
            &attached_beliefs,
            &bins,
            EmissionGradientMode::Attached,
        );
        let detached = model.training_emission_logits(
            &detached_beliefs,
            &bins,
            EmissionGradientMode::Detached,
        );
        assert!(
            attached.equal(&detached),
            "gradient mode changed emission values"
        );
        (-attached
            .log_softmax(-1, Kind::Float)
            .select(-1, 0)
            .mean(Kind::Float))
        .backward();
        assert!(
            attached_beliefs.grad().defined()
                && attached_beliefs.grad().abs().max().double_value(&[]) > 0.0,
            "attached emission CE did not reach beliefs"
        );
        (-detached
            .log_softmax(-1, Kind::Float)
            .select(-1, 0)
            .mean(Kind::Float))
        .backward();
        assert!(
            !detached_beliefs.grad().defined(),
            "detached emission CE reached beliefs"
        );
        let variables = var_store.variables();
        let emission_bias = &variables["lejepa_emission.bar_dof_head_r.bias"];
        assert!(
            emission_bias.grad().defined()
                && emission_bias.grad().abs().max().double_value(&[]) > 0.0,
            "detached arm also detached deployed emission parameters"
        );
    }

    #[test]
    fn emission_and_token_head_reset_are_identical_from_the_dedicated_seed() {
        let _torch_rng_guard = test_rng::exclusive();
        let var_store = nn::VarStore::new(Device::Cpu);
        let mut model = MseJepaModel::new(&var_store.root());
        let mut probe = MseJepaTokenProbe::new(&(var_store.root() / "posthoc_token_probe"));
        model.reset_emission(91);
        probe.reset(91);
        let inputs = Tensor::randn([7, LATENT_DIM], (Kind::Float, Device::Cpu));
        let bins = Tensor::zeros([7, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        assert!(
            model
                .emission_logits(&inputs, &bins)
                .equal(&probe.logits(&inputs, &bins)),
            "identically seeded fresh readouts differ"
        );
    }

    #[test]
    fn cached_decode_matches_the_uncached_belief_of_the_appended_position() {
        let _rng = test_rng::exclusive();
        tch::manual_seed(17);
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        let tokens = Tensor::randn([2, 1, 9, LATENT_DIM], (Kind::Float, Device::Cpu));
        let context = tokens.narrow(2, 0, 8);
        let appended = tokens.narrow(2, 8, 1);
        let mut cache = MseJepaKvCache::new();
        let prefilled = model.prefill_cached(&context, &mut cache);
        assert_eq!(prefilled.size(), vec![2, 1, 8, LATENT_DIM]);
        assert_eq!(cache.cached_bars(), 8);
        let decoded = model.decode_cached(&appended, &mut cache);
        assert_eq!(cache.next_position(), 9);
        let reference = model.causal_beliefs(&tokens, false).narrow(2, 8, 1);
        let difference = (decoded - reference).abs().max().double_value(&[]);
        assert!(
            difference < 1e-4,
            "cached decode drifted from the uncached trunk: {difference}"
        );
    }
}
