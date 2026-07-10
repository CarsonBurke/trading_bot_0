use std::{
    fs::File,
    io::{BufReader, BufWriter, Read},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use ring::digest::{Context as DigestContext, SHA256};
use serde::{Deserialize, Serialize};
use tch::{nn, nn::Module, nn::ModuleT, Device, Kind, Tensor};

use crate::torch::{
    constants::PRICE_DELTAS_PER_TICKER, env::OHLC_BAR_FEATURES, load::load_var_store_partial,
};

const METADATA_VERSION: u32 = 1;
const ARCHITECTURE: &str = "lejepa-causal-ar-v1";
const FEATURE_LAYOUT: &str = "torch-env-ohlc-features-v1";
const AR_LAYERS: usize = 6;
const AR_FF_DIM: i64 = 1536;
const PROJECTOR_HIDDEN_DIM: i64 = 2048;
const HEAD_DIM: i64 = 64;
const ROPE_DIMS: i64 = 32;
const HEADS: i64 = 4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldModelMetadata {
    pub format_version: u32,
    pub architecture: String,
    pub feature_layout: String,
    pub latent_dim: i64,
    pub bar_feature_dim: i64,
    pub max_context_bars: i64,
    pub target_scale: f64,
    pub checkpoint_sha256: String,
}

impl WorldModelMetadata {
    pub fn for_checkpoint(
        checkpoint: impl AsRef<Path>,
        latent_dim: i64,
        target_scale: f64,
    ) -> Result<Self> {
        validate_target_scale(target_scale)?;
        Ok(Self {
            format_version: METADATA_VERSION,
            architecture: ARCHITECTURE.to_owned(),
            feature_layout: FEATURE_LAYOUT.to_owned(),
            latent_dim,
            bar_feature_dim: OHLC_BAR_FEATURES as i64,
            max_context_bars: PRICE_DELTAS_PER_TICKER as i64,
            target_scale,
            checkpoint_sha256: checkpoint_sha256(checkpoint)?,
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open world-model metadata {}", path.display()))?;
        serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("failed to parse world-model metadata {}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("failed to create world-model metadata {}", path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)
            .with_context(|| format!("failed to write world-model metadata {}", path.display()))
    }

    pub fn save_for_checkpoint(
        checkpoint: impl AsRef<Path>,
        latent_dim: i64,
        target_scale: f64,
    ) -> Result<PathBuf> {
        let checkpoint = checkpoint.as_ref();
        let metadata_path = world_model_metadata_path(checkpoint);
        Self::for_checkpoint(checkpoint, latent_dim, target_scale)?.save(&metadata_path)?;
        Ok(metadata_path)
    }

    pub fn validate_checkpoint(&self, checkpoint: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let actual = checkpoint_sha256(checkpoint.as_ref())?;
        if actual != self.checkpoint_sha256 {
            bail!(
                "world-model checkpoint hash mismatch: metadata={}, actual={actual}",
                self.checkpoint_sha256
            );
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<()> {
        if self.format_version != METADATA_VERSION {
            bail!(
                "unsupported world-model metadata version {}, expected {METADATA_VERSION}",
                self.format_version
            );
        }
        if self.architecture != ARCHITECTURE {
            bail!(
                "incompatible world-model architecture {}, expected {ARCHITECTURE}",
                self.architecture
            );
        }
        if self.feature_layout != FEATURE_LAYOUT {
            bail!(
                "incompatible world-model feature layout {}, expected {FEATURE_LAYOUT}",
                self.feature_layout
            );
        }
        if self.latent_dim <= 0 || self.latent_dim % HEADS != 0 {
            bail!("latent_dim must be positive and divisible by {HEADS}");
        }
        if self.latent_dim / HEADS != HEAD_DIM {
            bail!(
                "latent_dim must be {} for the trained LEJEPA architecture",
                HEAD_DIM * HEADS
            );
        }
        if self.bar_feature_dim != OHLC_BAR_FEATURES as i64 {
            bail!(
                "incompatible bar feature dimension {}, expected {}",
                self.bar_feature_dim,
                OHLC_BAR_FEATURES
            );
        }
        if self.max_context_bars != PRICE_DELTAS_PER_TICKER as i64 {
            bail!(
                "incompatible context length {}, expected {}",
                self.max_context_bars,
                PRICE_DELTAS_PER_TICKER
            );
        }
        validate_target_scale(self.target_scale)
    }
}

pub fn world_model_metadata_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("metadata.json")
}

#[derive(Debug)]
pub struct WorldModelPrediction {
    pub latent: Tensor,
    pub ohlc_mean: Tensor,
    pub ohlc_logvar: Tensor,
}

#[derive(Debug)]
struct LayerKvCache {
    key: Tensor,
    value: Tensor,
}

impl LayerKvCache {
    fn fork(&self) -> Self {
        Self {
            key: self.key.shallow_clone(),
            value: self.value.shallow_clone(),
        }
    }
}

#[derive(Debug)]
struct CausalKvCache {
    layers: Vec<LayerKvCache>,
    next_position: i64,
    max_tokens: i64,
}

impl CausalKvCache {
    fn fork(&self) -> Self {
        Self {
            layers: self.layers.iter().map(LayerKvCache::fork).collect(),
            next_position: self.next_position,
            max_tokens: self.max_tokens,
        }
    }

    fn cached_tokens(&self) -> i64 {
        self.layers
            .first()
            .map(|layer| layer.key.size()[2])
            .unwrap_or(0)
    }
}

/// Stateful frozen-world-model inference context.
///
/// Forecasting forks this state, so generated tokens never enter the real-history
/// cache. Appending an actual bar advances the base cache by exactly one token.
pub struct LejepaWorldModelSession {
    cache: CausalKvCache,
    last_belief: Tensor,
    batch_size: i64,
    checkpoint_sha256: String,
}

impl LejepaWorldModelSession {
    pub fn cached_tokens(&self) -> i64 {
        self.cache.cached_tokens()
    }

    pub fn batch_size(&self) -> i64 {
        self.batch_size
    }

    pub fn checkpoint_sha256(&self) -> &str {
        &self.checkpoint_sha256
    }

    pub fn fork(&self) -> Self {
        Self {
            cache: self.cache.fork(),
            last_belief: self.last_belief.shallow_clone(),
            batch_size: self.batch_size,
            checkpoint_sha256: self.checkpoint_sha256.clone(),
        }
    }

    pub fn forecast(&self, model: &LejepaWorldModel, horizon: i64) -> Result<WorldModelPrediction> {
        model.predict_from_session(self, horizon)
    }

    pub fn append_actual_bar(
        &mut self,
        model: &LejepaWorldModel,
        actual_bar: &Tensor,
    ) -> Result<()> {
        model.append_actual_bar(self, actual_bar)
    }
}

pub struct LejepaWorldModel {
    var_store: nn::VarStore,
    core: LejepaInferenceCore,
    metadata: WorldModelMetadata,
}

impl LejepaWorldModel {
    pub fn load(
        checkpoint: impl AsRef<Path>,
        metadata_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let checkpoint = checkpoint.as_ref();
        let metadata = WorldModelMetadata::load(metadata_path)?;
        metadata.validate_checkpoint(checkpoint)?;

        let mut var_store = nn::VarStore::new(device);
        let core = LejepaInferenceCore::new(&var_store.root(), metadata.latent_dim);
        load_var_store_partial(&mut var_store, checkpoint)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .require_complete()
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .with_context(|| {
                format!(
                    "world-model checkpoint {} is missing required LEJEPA inference tensors",
                    checkpoint.display()
                )
            })?;
        var_store.freeze();

        Ok(Self {
            var_store,
            core,
            metadata,
        })
    }

    pub fn metadata(&self) -> &WorldModelMetadata {
        &self.metadata
    }

    pub fn device(&self) -> Device {
        self.var_store.device()
    }

    pub fn all_parameters_frozen(&self) -> bool {
        self.var_store
            .variables()
            .values()
            .all(|tensor| !tensor.requires_grad())
    }

    pub fn predict(&self, context_bars: &Tensor, horizon: i64) -> Result<WorldModelPrediction> {
        validate_context(context_bars, &self.metadata, horizon)?;
        let session = self.start_session(context_bars)?;
        self.predict_from_session(&session, horizon)
    }

    pub fn start_session(&self, context_bars: &Tensor) -> Result<LejepaWorldModelSession> {
        validate_context(context_bars, &self.metadata, 1)?;
        let context_bars = context_bars.to_device(self.device()).to_kind(Kind::Float);
        Ok(tch::no_grad(|| {
            let tokens = self.core.encode_bars(&context_bars).detach();
            let (belief, cache) = self.core.prefill_cache(&tokens);
            let last = belief.size()[2] - 1;
            LejepaWorldModelSession {
                cache,
                last_belief: belief.narrow(2, last, 1).detach(),
                batch_size: context_bars.size()[0],
                checkpoint_sha256: self.metadata.checkpoint_sha256.clone(),
            }
        }))
    }

    pub fn predict_from_session(
        &self,
        session: &LejepaWorldModelSession,
        horizon: i64,
    ) -> Result<WorldModelPrediction> {
        self.validate_session(session)?;
        if horizon <= 0 {
            bail!("world-model horizon must be positive");
        }
        Ok(tch::no_grad(|| {
            self.core
                .predict_cached(session, horizon, self.metadata.target_scale)
        }))
    }

    pub fn append_actual_bar(
        &self,
        session: &mut LejepaWorldModelSession,
        actual_bar: &Tensor,
    ) -> Result<()> {
        self.validate_session(session)?;
        validate_actual_bar(actual_bar, session.batch_size, &self.metadata)?;
        let actual_bar = actual_bar.to_device(self.device()).to_kind(Kind::Float);
        tch::no_grad(|| {
            let token = self.core.encode_bars(&actual_bar).detach();
            session.last_belief = self.core.append_cached(&mut session.cache, &token).detach();
        });
        Ok(())
    }

    fn validate_session(&self, session: &LejepaWorldModelSession) -> Result<()> {
        if session.checkpoint_sha256 != self.metadata.checkpoint_sha256 {
            bail!("world-model session was created by a different checkpoint");
        }
        if session.cache.layers.len() != self.core.layers.len() {
            bail!("world-model session has an incompatible layer cache");
        }
        if session.last_belief.device() != self.device() {
            bail!("world-model session is on a different device");
        }
        Ok(())
    }
}

struct CausalLayer {
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

struct LejepaInferenceCore {
    bar_proj: nn::Linear,
    bar_enrich_fc1: nn::Linear,
    bar_enrich_fc2: nn::Linear,
    projector: ProjectionMlp,
    layers: Vec<CausalLayer>,
    pred_projector: ProjectionMlp,
    probe_input_ln: nn::LayerNorm,
    probe_head: nn::Linear,
    probe_logvar_head: nn::Linear,
    latent_dim: i64,
}

impl LejepaInferenceCore {
    fn new(p: &nn::Path, latent_dim: i64) -> Self {
        let ff_dim = latent_dim * 2;
        let bar_proj = nn::linear(
            p / "bar_proj",
            OHLC_BAR_FEATURES as i64,
            latent_dim,
            Default::default(),
        );
        let bar_enrich_fc1 =
            nn::linear(p / "bar_enrich_fc1", latent_dim, ff_dim, Default::default());
        let bar_enrich_fc2 =
            nn::linear(p / "bar_enrich_fc2", ff_dim, latent_dim, Default::default());
        let projector = ProjectionMlp {
            fc1: nn::linear(
                p / "lejepa_projector_fc1",
                latent_dim,
                PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            bn: nn::batch_norm1d(
                p / "lejepa_projector_bn",
                PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            fc2: nn::linear(
                p / "lejepa_projector_fc2",
                PROJECTOR_HIDDEN_DIM,
                latent_dim,
                Default::default(),
            ),
        };
        let layers = (0..AR_LAYERS)
            .map(|index| {
                let layer_path = p / format!("lejepa_layer_{index}");
                CausalLayer {
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
                        AR_FF_DIM,
                        Default::default(),
                    ),
                    ff_value: nn::linear(
                        &layer_path / "ff_value",
                        latent_dim,
                        AR_FF_DIM,
                        Default::default(),
                    ),
                    ff_out: nn::linear(
                        &layer_path / "ff_out",
                        AR_FF_DIM,
                        latent_dim,
                        Default::default(),
                    ),
                }
            })
            .collect();
        let pred_projector = ProjectionMlp {
            fc1: nn::linear(
                p / "lejepa_pred_proj_fc1",
                latent_dim,
                PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            bn: nn::batch_norm1d(
                p / "lejepa_pred_proj_bn",
                PROJECTOR_HIDDEN_DIM,
                Default::default(),
            ),
            fc2: nn::linear(
                p / "lejepa_pred_proj_fc2",
                PROJECTOR_HIDDEN_DIM,
                latent_dim,
                Default::default(),
            ),
        };
        let probe_input_ln =
            nn::layer_norm(p / "probe_input_ln", vec![latent_dim], Default::default());
        let probe_head = nn::linear(
            p / "probe_head",
            latent_dim,
            OHLC_BAR_FEATURES as i64,
            Default::default(),
        );
        let probe_logvar_head = nn::linear(
            p / "probe_logvar_head",
            latent_dim,
            OHLC_BAR_FEATURES as i64,
            Default::default(),
        );
        Self {
            bar_proj,
            bar_enrich_fc1,
            bar_enrich_fc2,
            projector,
            layers,
            pred_projector,
            probe_input_ln,
            probe_head,
            probe_logvar_head,
            latent_dim,
        }
    }

    fn predict_cached(
        &self,
        session: &LejepaWorldModelSession,
        horizon: i64,
        target_scale: f64,
    ) -> WorldModelPrediction {
        let mut cache = session.cache.fork();
        let mut belief = session.last_belief.shallow_clone();
        let mut latents = Vec::with_capacity(horizon as usize);
        let mut means = Vec::with_capacity(horizon as usize);
        let mut logvars = Vec::with_capacity(horizon as usize);
        for _ in 0..horizon {
            let next_token = self.project(&belief, &self.pred_projector);
            let (scaled_mean, scaled_logvar) = self.probe(&next_token);
            latents.push(next_token.squeeze_dim(2).squeeze_dim(1));
            means.push((scaled_mean / target_scale).squeeze_dim(2).squeeze_dim(1));
            logvars.push(
                raw_logvar(&scaled_logvar, target_scale)
                    .squeeze_dim(2)
                    .squeeze_dim(1),
            );
            belief = self.append_cached(&mut cache, &next_token);
        }
        WorldModelPrediction {
            latent: Tensor::stack(&latents, 1).detach(),
            ohlc_mean: Tensor::stack(&means, 1).detach(),
            ohlc_logvar: Tensor::stack(&logvars, 1).detach(),
        }
    }

    #[cfg(test)]
    fn predict_full(
        &self,
        context: &Tensor,
        horizon: i64,
        target_scale: f64,
    ) -> WorldModelPrediction {
        let mut tokens = self.encode_bars(context).detach();
        let mut latents = Vec::with_capacity(horizon as usize);
        let mut means = Vec::with_capacity(horizon as usize);
        let mut logvars = Vec::with_capacity(horizon as usize);
        for _ in 0..horizon {
            let belief = self.causal_belief(&tokens);
            let last = tokens.size()[2] - 1;
            let next_token = self.project(&belief.narrow(2, last, 1), &self.pred_projector);
            let (scaled_mean, scaled_logvar) = self.probe(&next_token);
            latents.push(next_token.squeeze_dim(2).squeeze_dim(1));
            means.push((scaled_mean / target_scale).squeeze_dim(2).squeeze_dim(1));
            logvars.push(
                raw_logvar(&scaled_logvar, target_scale)
                    .squeeze_dim(2)
                    .squeeze_dim(1),
            );
            tokens = Tensor::cat(&[&tokens, &next_token], 2);
        }
        WorldModelPrediction {
            latent: Tensor::stack(&latents, 1).detach(),
            ohlc_mean: Tensor::stack(&means, 1).detach(),
            ohlc_logvar: Tensor::stack(&logvars, 1).detach(),
        }
    }

    fn encode_bars(&self, bars: &Tensor) -> Tensor {
        let size = bars.size();
        let batch = size[0];
        let tickers = size[1];
        let length = size[2];
        let features = bars
            .reshape([batch * tickers * length, OHLC_BAR_FEATURES as i64])
            .nan_to_num(0.0, 0.0, 0.0);
        let h = self.bar_proj.forward(&features);
        let enriched = self.bar_enrich_fc2.forward(
            &normalize_last_dim(&self.bar_enrich_fc1.forward(&normalize_last_dim(&h))).gelu("none"),
        );
        let tokens = (h + enriched).view([batch, tickers, length, self.latent_dim]);
        self.project(&tokens, &self.projector)
    }

    #[cfg(test)]
    fn causal_belief(&self, tokens: &Tensor) -> Tensor {
        let size = tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let length = size[2];
        let positions = Tensor::arange(length, (Kind::Int64, tokens.device()));
        let mut x = tokens.reshape([batch * tickers, length, self.latent_dim]);
        for layer in &self.layers {
            x = self.causal_layer_full(&x, layer, &positions).0;
        }
        normalize_last_dim(&x).view([batch, tickers, length, self.latent_dim])
    }

    fn prefill_cache(&self, tokens: &Tensor) -> (Tensor, CausalKvCache) {
        self.prefill_cache_with_max(tokens, PRICE_DELTAS_PER_TICKER as i64)
    }

    fn prefill_cache_with_max(&self, tokens: &Tensor, max_tokens: i64) -> (Tensor, CausalKvCache) {
        let size = tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let length = size[2];
        assert!(max_tokens > 0, "KV-cache window must be positive");
        assert!(
            length <= max_tokens,
            "prefill length exceeds KV-cache window"
        );
        let positions = Tensor::arange(length, (Kind::Int64, tokens.device()));
        let mut x = tokens.reshape([batch * tickers, length, self.latent_dim]);
        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next, key, value) = self.causal_layer_full(&x, layer, &positions);
            x = next;
            caches.push(LayerKvCache { key, value });
        }
        let belief = normalize_last_dim(&x).view([batch, tickers, length, self.latent_dim]);
        (
            belief,
            CausalKvCache {
                layers: caches,
                next_position: length,
                max_tokens,
            },
        )
    }

    fn causal_layer_full(
        &self,
        source: &Tensor,
        layer: &CausalLayer,
        positions: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let rows = source.size()[0];
        let length = source.size()[1];
        let qkv = layer.qkv.forward(&normalize_last_dim(source));
        let parts = qkv.split(self.latent_dim, -1);
        let reshape = |tensor: &Tensor| {
            tensor
                .view([rows, length, HEADS, HEAD_DIM])
                .permute([0, 2, 1, 3])
        };
        let q = apply_rotary_positions(&reshape(&parts[0]), positions);
        let k = apply_rotary_positions(&reshape(&parts[1]), positions);
        let v = reshape(&parts[2]);
        let attention_kind = attention_kind(source);
        let cached_key = k.to_kind(attention_kind);
        let cached_value = v.to_kind(attention_kind);
        let attention = Tensor::scaled_dot_product_attention(
            &q.to_kind(attention_kind),
            &cached_key,
            &cached_value,
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
        let x = source + layer.out_proj.forward(&attention);
        let normed = normalize_last_dim(&x);
        let ff = layer
            .ff_out
            .forward(&(layer.ff_gate.forward(&normed).silu() * layer.ff_value.forward(&normed)));
        (x + ff, cached_key.detach(), cached_value.detach())
    }

    fn append_cached(&self, cache: &mut CausalKvCache, token: &Tensor) -> Tensor {
        let size = token.size();
        let batch = size[0];
        let tickers = size[1];
        assert_eq!(size[2], 1, "KV-cache append requires exactly one token");
        assert_eq!(size[3], self.latent_dim);
        assert_eq!(cache.layers.len(), self.layers.len());

        let position = Tensor::from_slice(&[cache.next_position])
            .to_kind(Kind::Int64)
            .to_device(token.device());
        let max_tokens = cache.max_tokens;
        let mut x = token.reshape([batch * tickers, 1, self.latent_dim]);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let qkv = layer.qkv.forward(&normalize_last_dim(&x));
            let parts = qkv.split(self.latent_dim, -1);
            let reshape = |tensor: &Tensor| {
                tensor
                    .view([batch * tickers, 1, HEADS, HEAD_DIM])
                    .permute([0, 2, 1, 3])
            };
            let attention_kind = attention_kind(&x);
            let query =
                apply_rotary_positions(&reshape(&parts[0]), &position).to_kind(attention_kind);
            let key =
                apply_rotary_positions(&reshape(&parts[1]), &position).to_kind(attention_kind);
            let value = reshape(&parts[2]).to_kind(attention_kind);
            layer_cache.key = Tensor::cat(&[&layer_cache.key, &key], 2).detach();
            layer_cache.value = Tensor::cat(&[&layer_cache.value, &value], 2).detach();
            let cached_length = layer_cache.key.size()[2];
            if cached_length > max_tokens {
                let start = cached_length - max_tokens;
                layer_cache.key = layer_cache.key.narrow(2, start, max_tokens).contiguous();
                layer_cache.value = layer_cache.value.narrow(2, start, max_tokens).contiguous();
            }

            let attention = Tensor::scaled_dot_product_attention(
                &query,
                &layer_cache.key,
                &layer_cache.value,
                None::<&Tensor>,
                0.0,
                false,
                None,
                false,
            )
            .to_kind(x.kind())
            .permute([0, 2, 1, 3])
            .contiguous()
            .view([batch * tickers, 1, self.latent_dim]);
            let residual = &x + layer.out_proj.forward(&attention);
            let normed = normalize_last_dim(&residual);
            let ff = layer.ff_out.forward(
                &(layer.ff_gate.forward(&normed).silu() * layer.ff_value.forward(&normed)),
            );
            x = residual + ff;
        }
        cache.next_position += 1;
        normalize_last_dim(&x).view([batch, tickers, 1, self.latent_dim])
    }

    fn project(&self, value: &Tensor, projector: &ProjectionMlp) -> Tensor {
        let shape = value.size();
        let flat = value.reshape([-1, self.latent_dim]);
        let hidden = projector.fc1.forward(&flat);
        projector
            .fc2
            .forward(&projector.bn.forward_t(&hidden, false).gelu("none"))
            .reshape(shape.as_slice())
    }

    fn probe(&self, latent: &Tensor) -> (Tensor, Tensor) {
        let normalized = self.probe_input_ln.forward(latent);
        (
            self.probe_head.forward(&normalized),
            self.probe_logvar_head.forward(&normalized).clamp(-7.0, 7.0),
        )
    }
}

fn validate_context(context: &Tensor, metadata: &WorldModelMetadata, horizon: i64) -> Result<()> {
    metadata.validate_schema()?;
    if horizon <= 0 {
        bail!("world-model horizon must be positive");
    }
    let shape = context.size();
    if shape.len() != 4 {
        bail!("context bars must have shape [batch, 1, length, features]");
    }
    if shape[0] <= 0 || shape[1] != 1 || shape[2] <= 0 {
        bail!("context bars require a positive batch/length and exactly one ticker");
    }
    if shape[2] > metadata.max_context_bars {
        bail!(
            "context has {} bars, exceeding maximum {}",
            shape[2],
            metadata.max_context_bars
        );
    }
    if shape[3] != metadata.bar_feature_dim {
        bail!(
            "context has {} features, expected {}",
            shape[3],
            metadata.bar_feature_dim
        );
    }
    Ok(())
}

fn validate_actual_bar(
    actual_bar: &Tensor,
    expected_batch: i64,
    metadata: &WorldModelMetadata,
) -> Result<()> {
    let shape = actual_bar.size();
    if shape != [expected_batch, 1, 1, metadata.bar_feature_dim] {
        bail!(
            "actual bar must have shape [{expected_batch}, 1, 1, {}], got {shape:?}",
            metadata.bar_feature_dim
        );
    }
    Ok(())
}

fn attention_kind(value: &Tensor) -> Kind {
    if value.device().is_cuda() {
        Kind::BFloat16
    } else {
        value.kind()
    }
}

fn apply_rotary_positions(value: &Tensor, positions: &Tensor) -> Tensor {
    let head_dim = *value.size().last().unwrap();
    let rope_dims = ROPE_DIMS.min(head_dim);
    let half = rope_dims / 2;
    let exponents = Tensor::arange(half, (Kind::Float, value.device())) * (2.0 / rope_dims as f64);
    let inv_frequency = (exponents * -(10000.0_f64.ln())).exp();
    let angles = positions
        .to_device(value.device())
        .to_kind(Kind::Float)
        .unsqueeze(1)
        * inv_frequency.unsqueeze(0);
    let cosine_half = angles.cos();
    let sine_half = angles.sin();
    let cosine = Tensor::cat(&[&cosine_half, &cosine_half], -1).to_kind(value.kind());
    let sine = Tensor::cat(&[&sine_half, &sine_half], -1).to_kind(value.kind());
    let rotary = value.narrow(-1, 0, rope_dims);
    let first = rotary.narrow(-1, 0, half);
    let second = rotary.narrow(-1, half, half);
    let rotated_half = Tensor::cat(&[&(-&second), &first], -1);
    let rotated = rotary * cosine + rotated_half * sine;
    if rope_dims < head_dim {
        Tensor::cat(
            &[&rotated, &value.narrow(-1, rope_dims, head_dim - rope_dims)],
            -1,
        )
    } else {
        rotated
    }
}

fn raw_logvar(scaled_logvar: &Tensor, target_scale: f64) -> Tensor {
    scaled_logvar - 2.0 * target_scale.ln()
}

fn validate_target_scale(target_scale: f64) -> Result<()> {
    if !target_scale.is_finite() || target_scale <= 0.0 {
        bail!("world-model target_scale must be finite and positive");
    }
    Ok(())
}

fn checkpoint_sha256(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut file = File::open(path)
        .with_context(|| format!("failed to open world-model checkpoint {}", path.display()))?;
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash checkpoint {}", path.display()))?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
    }
    Ok(context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn normalize_last_dim(value: &Tensor) -> Tensor {
    let variance = value
        .pow_tensor_scalar(2.0)
        .mean_dim([-1i64].as_slice(), true, Kind::Float);
    value * (variance + 1e-6).rsqrt().to_kind(value.kind())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "trading-bot-world-model-{name}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    fn test_metadata(target_scale: f64) -> WorldModelMetadata {
        WorldModelMetadata {
            format_version: METADATA_VERSION,
            architecture: ARCHITECTURE.to_owned(),
            feature_layout: FEATURE_LAYOUT.to_owned(),
            latent_dim: 256,
            bar_feature_dim: OHLC_BAR_FEATURES as i64,
            max_context_bars: PRICE_DELTAS_PER_TICKER as i64,
            target_scale,
            checkpoint_sha256: String::new(),
        }
    }

    fn test_model(seed: i64) -> LejepaWorldModel {
        tch::manual_seed(seed);
        let mut var_store = nn::VarStore::new(Device::Cpu);
        let core = LejepaInferenceCore::new(&var_store.root(), 256);
        var_store.freeze();
        LejepaWorldModel {
            var_store,
            core,
            metadata: test_metadata(100.0),
        }
    }

    fn deterministic_bars(batch: i64, length: i64) -> Tensor {
        Tensor::arange(
            batch * length * OHLC_BAR_FEATURES as i64,
            (Kind::Float, Device::Cpu),
        )
        .view([batch, 1, length, OHLC_BAR_FEATURES as i64])
            / 1000.0
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f64) {
        let max_difference = (actual - expected).abs().max().double_value(&[]);
        assert!(
            max_difference <= tolerance,
            "maximum difference {max_difference} exceeded {tolerance}"
        );
    }

    #[test]
    fn raw_unit_conversion_scales_mean_and_variance() {
        let scaled_logvar = Tensor::from_slice(&[(9.0f64).ln()]);
        let raw = raw_logvar(&scaled_logvar, 3.0);
        assert!((raw.double_value(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn validates_context_contract() {
        let metadata = test_metadata(100.0);
        let valid = Tensor::zeros([2, 1, 8, 16], (Kind::Float, Device::Cpu));
        validate_context(&valid, &metadata, 100).unwrap();
        let wrong_features = Tensor::zeros([2, 1, 8, 15], (Kind::Float, Device::Cpu));
        assert!(validate_context(&wrong_features, &metadata, 100).is_err());
        assert!(validate_context(&valid, &metadata, 0).is_err());
    }

    #[test]
    fn metadata_round_trip_and_hash_validation() {
        let checkpoint = temp_path("checkpoint.ot");
        let metadata_path = temp_path("metadata.json");
        fs::write(&checkpoint, b"checkpoint bytes").unwrap();
        let metadata = WorldModelMetadata::for_checkpoint(&checkpoint, 256, 100.0).unwrap();
        metadata.save(&metadata_path).unwrap();
        let loaded = WorldModelMetadata::load(&metadata_path).unwrap();
        assert_eq!(loaded, metadata);
        loaded.validate_checkpoint(&checkpoint).unwrap();
        fs::write(&checkpoint, b"changed checkpoint bytes").unwrap();
        assert!(loaded.validate_checkpoint(&checkpoint).is_err());
        let _ = fs::remove_file(checkpoint);
        let _ = fs::remove_file(metadata_path);
    }

    #[test]
    fn metadata_path_tracks_checkpoint_stem() {
        assert_eq!(
            world_model_metadata_path("weights/pretrain_heads_best.ot"),
            PathBuf::from("weights/pretrain_heads_best.metadata.json")
        );
        assert_eq!(
            world_model_metadata_path("weights/pretrain_heads_step100"),
            PathBuf::from("weights/pretrain_heads_step100.metadata.json")
        );
    }

    #[test]
    fn predictions_have_expected_shapes_and_are_detached() {
        let model = test_model(11);
        let context = Tensor::randn([2, 1, 4, 16], (Kind::Float, Device::Cpu));
        let prediction = model.predict(&context, 3).unwrap();
        assert_eq!(prediction.latent.size(), vec![2, 3, 256]);
        assert_eq!(prediction.ohlc_mean.size(), vec![2, 3, 16]);
        assert_eq!(prediction.ohlc_logvar.size(), vec![2, 3, 16]);
        assert!(!prediction.latent.requires_grad());
        assert!(!prediction.ohlc_mean.requires_grad());
        assert!(!prediction.ohlc_logvar.requires_grad());
        assert!(model.all_parameters_frozen());
    }

    #[test]
    fn cached_rollout_matches_full_prefix_autoregression() {
        let model = test_model(19);
        let context = deterministic_bars(2, 5);
        let cached = model.predict(&context, 4).unwrap();
        let full = tch::no_grad(|| model.core.predict_full(&context, 4, 100.0));

        // Incremental and full SDPA use different matrix multiplication shapes,
        // so their fp32 reduction order is not bit-identical.
        assert_close(&cached.latent, &full.latent, 5e-5);
        assert_close(&cached.ohlc_mean, &full.ohlc_mean, 5e-5);
        assert_close(&cached.ohlc_logvar, &full.ohlc_logvar, 5e-5);
    }

    #[test]
    fn forecasting_does_not_mutate_base_session() {
        let model = test_model(23);
        let context = deterministic_bars(1, 4);
        let session = model.start_session(&context).unwrap();
        let cached_before = session.cached_tokens();
        let first = session.forecast(&model, 3).unwrap();
        let second = session.forecast(&model, 3).unwrap();

        assert_eq!(session.cached_tokens(), cached_before);
        assert_close(&first.latent, &second.latent, 0.0);
        assert_close(&first.ohlc_mean, &second.ohlc_mean, 0.0);
    }

    #[test]
    fn appending_actual_bar_advances_cache_and_matches_full_context() {
        let model = test_model(29);
        let context = deterministic_bars(1, 4);
        let actual = deterministic_bars(1, 1) + 0.75;
        let mut session = model.start_session(&context).unwrap();
        session.append_actual_bar(&model, &actual).unwrap();
        assert_eq!(session.cached_tokens(), 5);

        let cached = session.forecast(&model, 2).unwrap();
        let extended = Tensor::cat(&[&context, &actual], 2);
        let full = tch::no_grad(|| model.core.predict_full(&extended, 2, 100.0));
        assert_close(&cached.latent, &full.latent, 2e-5);
        assert_close(&cached.ohlc_mean, &full.ohlc_mean, 2e-6);
    }

    #[test]
    fn kv_cache_evicts_oldest_tokens_at_window_limit() {
        let model = test_model(30);
        let context = deterministic_bars(1, 3);
        let tokens = tch::no_grad(|| model.core.encode_bars(&context).detach());
        let (_, mut cache) = model.core.prefill_cache_with_max(&tokens, 3);
        let appended = tokens.narrow(2, 2, 1);

        for expected_position in 4..=8 {
            tch::no_grad(|| {
                let _ = model.core.append_cached(&mut cache, &appended);
            });
            assert_eq!(cache.cached_tokens(), 3);
            assert_eq!(cache.next_position, expected_position);
            assert!(cache
                .layers
                .iter()
                .all(|layer| layer.key.size()[2] == 3 && layer.value.size()[2] == 3));
        }
    }

    #[test]
    fn saved_inference_tensor_names_load_completely() {
        let checkpoint = temp_path("inference-names.ot");
        let metadata_path = world_model_metadata_path(&checkpoint);
        let source = test_model(31);
        source.var_store.save(&checkpoint).unwrap();
        WorldModelMetadata::save_for_checkpoint(&checkpoint, 256, 100.0).unwrap();

        let loaded = LejepaWorldModel::load(&checkpoint, &metadata_path, Device::Cpu).unwrap();
        assert!(loaded.all_parameters_frozen());
        let mut source_names = source
            .var_store
            .variables()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let mut loaded_names = loaded
            .var_store
            .variables()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        source_names.sort();
        loaded_names.sort();
        assert_eq!(source_names, loaded_names);

        let _ = fs::remove_file(checkpoint);
        let _ = fs::remove_file(metadata_path);
    }
}
