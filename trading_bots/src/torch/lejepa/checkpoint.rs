use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, ensure, Context, Result};
use clap::ValueEnum;
use ring::digest::{digest, Context as DigestContext, SHA256};
use serde::{Deserialize, Serialize};
use tch::nn;

use crate::torch::bar_dist::{BAR_LABEL_SIGMA_RATIO, BAR_PREFIX_EMBED_DIM, NUM_BAR_BINS};
use crate::torch::hashing::file_sha256;
use crate::torch::load::load_var_store_partial;
use crate::torch::pope::{POPE_ATTENTION_SCALE, POPE_DIM, POPE_FREQUENCY_BASE, POPE_QK_DIM};

use super::model::{
    ARCHITECTURE, AR_FF_DIM, AR_LAYERS, BAR_FEATURES, DROPOUT, FEATURE_LAYOUT, FLOW_BLOCKS,
    FLOW_HIDDEN_DIM, FLOW_TIME_FEATURES, FLOW_TIME_MAX_PERIOD, FLOW_TIME_SCALE, HEADS, HEAD_DIM,
    LATENT_DIM, MAX_CONTEXT_BARS, NORMALIZATION_EPS, PROJECTOR_HIDDEN_DIM,
    TRANSITION_BAR_FEATURE_SCALE,
};

const BUNDLE_VERSION: u32 = 14;
const FAMILY: &str = "mse_jepa";
const ARTIFACT_ROLE: &str = "self-contained-mse-jepa-model";
pub const CANONICAL_READOUT_FIT_STEPS: u64 = 4_096;
pub const CANONICAL_READOUT_FIT_BATCH_SIZE: u64 = 8;
pub const CANONICAL_READOUT_TOKEN_ROWS: u64 = 4_096;
pub const CANONICAL_READOUT_VALIDATION_WINDOWS: u64 = 8;
pub const CANONICAL_READOUT_FIT_SEED: u64 = 0x5245_4144_4f55_5431;
pub const CANONICAL_READOUT_OPTIMIZER_RECIPE_ID: &str =
    "adamw-lr3e-4-beta1.9-beta2.999-wd.01-eps1e-8-one-update-per-batch-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointKind {
    Core,
    FittedReadout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "kebab-case")]
pub enum EmissionGradientMode {
    Attached,
    Detached,
}

impl std::fmt::Display for EmissionGradientMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Attached => "attached",
            Self::Detached => "detached",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthenticatedCheckpoint {
    pub path: PathBuf,
    pub kind: CheckpointKind,
    pub checkpoint_sha256: String,
    pub lineage_sha256: String,
    pub sigreg_lambda: f64,
    pub flow_lambda: f64,
    pub provenance: CheckpointProvenance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreInitialization {
    Fresh,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreTrainingOrigin {
    pub initialization: CoreInitialization,
    pub train_seed: u64,
    pub batch_size: u64,
    pub resolution_secs: u32,
    pub min_bars: u64,
    pub split_bounds: (i64, i64),
    pub split_bounds_pinned: bool,
    pub corpus_fingerprint: String,
    pub supports_scoring_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointProvenance {
    pub completed_steps: u64,
    pub planned_steps: u64,
    pub steps_per_pass: u64,
    pub optimizer_recipe_id: String,
    pub emission_gradient_mode: EmissionGradientMode,
    pub weight_readout: WeightReadout,
    pub core_origin: CoreTrainingOrigin,
    pub head_fit: Option<HeadFitProvenance>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadFitProvenance {
    pub source_checkpoint_sha256: String,
    pub source_lineage_sha256: String,
    pub source_weight_readout: WeightReadout,
    pub source_core_origin: CoreTrainingOrigin,
    pub seed: u64,
    pub steps: u64,
    pub batch_size: u64,
    pub token_rows_per_step: u64,
    pub validation_windows: u64,
    pub optimizer_recipe_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WeightReadout {
    Raw,
    TailEma {
        start_step: u64,
        horizon_steps: u64,
        blend: f64,
        update_count: u64,
        eligible_name_sha256: String,
    },
}

impl WeightReadout {
    fn lineage_fields(&self) -> String {
        match self {
            Self::Raw => "raw".to_owned(),
            Self::TailEma {
                start_step,
                horizon_steps,
                blend,
                update_count,
                eligible_name_sha256,
            } => format!(
                "tail_ema:start_step={start_step}:horizon_steps={horizon_steps}:blend_bits={:016x}:update_count={update_count}:eligible_name_sha256={eligible_name_sha256}",
                blend.to_bits()
            ),
        }
    }
}

impl CoreTrainingOrigin {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.initialization == CoreInitialization::Fresh,
            "MSE-JEPA core must originate from a fresh initialization"
        );
        ensure!(
            self.batch_size > 0 && self.resolution_secs > 0 && self.min_bars > 0,
            "MSE-JEPA core data/training dimensions must be positive"
        );
        ensure!(
            self.split_bounds.0 < self.split_bounds.1,
            "MSE-JEPA core split bounds are unordered"
        );
        ensure!(
            is_sha256_hex(&self.corpus_fingerprint),
            "MSE-JEPA corpus fingerprint is empty or malformed"
        );
        ensure!(
            is_sha256_hex(&self.supports_scoring_sha256),
            "MSE-JEPA BarSupports scoring digest is empty or malformed"
        );
        Ok(())
    }

    fn lineage_fields(&self) -> String {
        format!(
            "initialization=fresh:train_seed={}:batch_size={}:resolution_secs={}:min_bars={}:split_start_ms={}:split_end_ms={}:split_bounds_pinned={}:corpus_fingerprint={}:supports_scoring_sha256={}",
            self.train_seed,
            self.batch_size,
            self.resolution_secs,
            self.min_bars,
            self.split_bounds.0,
            self.split_bounds.1,
            self.split_bounds_pinned,
            self.corpus_fingerprint,
            self.supports_scoring_sha256,
        )
    }
}

impl CheckpointProvenance {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.planned_steps > 0,
            "MSE-JEPA planned steps must be positive"
        );
        ensure!(
            self.steps_per_pass > 0,
            "MSE-JEPA steps per pass must be positive"
        );
        ensure!(
            self.completed_steps <= self.planned_steps,
            "MSE-JEPA completed steps exceed planned steps"
        );
        ensure!(
            !self.optimizer_recipe_id.is_empty(),
            "MSE-JEPA optimizer recipe ID must not be empty"
        );
        self.core_origin.validate()?;
        if let WeightReadout::TailEma {
            start_step,
            horizon_steps,
            blend,
            update_count,
            eligible_name_sha256,
        } = &self.weight_readout
        {
            ensure!(
                self.completed_steps > 0,
                "MSE-JEPA tail-EMA readout requires completed training steps"
            );
            ensure!(
                *start_step <= self.completed_steps && *start_step <= self.planned_steps,
                "MSE-JEPA tail-EMA start step is outside the training range"
            );
            ensure!(
                *horizon_steps > 0,
                "MSE-JEPA tail-EMA horizon must be positive"
            );
            ensure!(
                blend.is_finite() && *blend > 0.0 && *blend <= 1.0,
                "MSE-JEPA tail-EMA blend must be finite and in (0, 1]"
            );
            let first_update_step = (*start_step).max(1);
            let expected_updates = self.completed_steps - first_update_step + 1;
            ensure!(
                *update_count == expected_updates,
                "MSE-JEPA tail-EMA update count does not match the completed step range"
            );
            ensure!(
                is_sha256_hex(eligible_name_sha256),
                "MSE-JEPA tail-EMA eligible-name SHA-256 is empty or malformed"
            );
        }
        if let Some(fit) = &self.head_fit {
            ensure!(
                self.completed_steps == self.planned_steps,
                "posthoc readout fitting requires a fully completed core checkpoint"
            );
            ensure!(
                is_sha256_hex(&fit.source_checkpoint_sha256)
                    && is_sha256_hex(&fit.source_lineage_sha256),
                "posthoc readout source hashes are empty or malformed"
            );
            ensure!(
                fit.source_weight_readout == self.weight_readout,
                "posthoc readout source weight identity disagrees with the fitted backbone"
            );
            ensure!(
                fit.source_core_origin == self.core_origin,
                "posthoc readout data/support origin disagrees with the fitted backbone"
            );
            ensure!(
                fit.steps > 0 && fit.batch_size > 0 && fit.token_rows_per_step > 0,
                "posthoc readout fit schedule must be positive"
            );
            ensure!(
                fit.validation_windows > 0 && !fit.optimizer_recipe_id.is_empty(),
                "posthoc readout fit provenance is incomplete"
            );
            ensure!(
                fit.steps == CANONICAL_READOUT_FIT_STEPS
                    && fit.batch_size == CANONICAL_READOUT_FIT_BATCH_SIZE
                    && fit.token_rows_per_step == CANONICAL_READOUT_TOKEN_ROWS
                    && fit.validation_windows == CANONICAL_READOUT_VALIDATION_WINDOWS
                    && fit.seed == CANONICAL_READOUT_FIT_SEED
                    && fit.optimizer_recipe_id == CANONICAL_READOUT_OPTIMIZER_RECIPE_ID,
                "posthoc readout fit provenance is not the canonical official recipe"
            );
        }
        Ok(())
    }

    fn validate_checkpoint_name(&self, checkpoint: &Path) -> Result<()> {
        let name = checkpoint
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        let name_is_tail_ema = name.contains("tail_ema");
        let readout_is_tail_ema = matches!(self.weight_readout, WeightReadout::TailEma { .. });
        ensure!(
            name_is_tail_ema == readout_is_tail_ema,
            "MSE-JEPA checkpoint filename and weight readout disagree"
        );
        let name_is_fitted = name.contains("fitted");
        ensure!(
            name_is_fitted == self.head_fit.is_some(),
            "MSE-JEPA checkpoint filename and posthoc-fit stage disagree"
        );
        Ok(())
    }

    fn lineage_fields(&self) -> String {
        let readout = self.weight_readout.lineage_fields();
        let fit = self.head_fit.as_ref().map_or_else(
            || "core".to_owned(),
            |fit| {
                format!(
                    "fitted:source_checkpoint_sha256={}:source_lineage_sha256={}:source_weight_readout={}:source_core_origin={}:seed={}:steps={}:batch_size={}:token_rows_per_step={}:validation_windows={}:optimizer_recipe_id_len={}:optimizer_recipe_id={}",
                    fit.source_checkpoint_sha256,
                    fit.source_lineage_sha256,
                    fit.source_weight_readout.lineage_fields(),
                    fit.source_core_origin.lineage_fields(),
                    fit.seed,
                    fit.steps,
                    fit.batch_size,
                    fit.token_rows_per_step,
                    fit.validation_windows,
                    fit.optimizer_recipe_id.len(),
                    fit.optimizer_recipe_id,
                )
            },
        );
        format!(
            "completed_steps={};planned_steps={};steps_per_pass={};optimizer_recipe_id_len={};optimizer_recipe_id={};emission_gradient_mode={};core_origin={};weight_readout={readout};head_fit={fit}",
            self.completed_steps,
            self.planned_steps,
            self.steps_per_pass,
            self.optimizer_recipe_id.len(),
            self.optimizer_recipe_id,
            self.emission_gradient_mode,
            self.core_origin.lineage_fields(),
        )
    }
}

/// Hashes a set of parameter names in sorted order as repeated
/// `u64::to_be_bytes(name.len()) || name.as_bytes()` records.
pub fn eligible_name_sha256<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<String> {
    let mut names = names.into_iter().collect::<Vec<_>>();
    ensure!(
        !names.is_empty(),
        "tail-EMA eligible parameter names must not be empty"
    );
    names.sort_unstable();
    ensure!(
        names.iter().all(|name| !name.is_empty()),
        "tail-EMA eligible parameter names must not contain an empty name"
    );
    ensure!(
        names.windows(2).all(|pair| pair[0] != pair[1]),
        "tail-EMA eligible parameter names must be unique"
    );

    let mut digest = DigestContext::new(&SHA256);
    for name in names {
        digest.update(&(name.len() as u64).to_be_bytes());
        digest.update(name.as_bytes());
    }
    Ok(hex_digest(digest.finish().as_ref()))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MseJepaMetadata {
    pub format_version: u32,
    pub family: String,
    pub artifact_role: String,
    pub architecture: String,
    pub feature_layout: String,
    pub latent_dim: i64,
    pub bar_feature_dim: i64,
    pub max_context_bars: i64,
    pub sigreg_lambda: f64,
    pub flow_lambda: f64,
    pub provenance: CheckpointProvenance,
    pub checkpoint_sha256: String,
    pub lineage_sha256: String,
}

impl MseJepaMetadata {
    pub fn for_checkpoint(
        checkpoint: impl AsRef<Path>,
        sigreg_lambda: f64,
        flow_lambda: f64,
        provenance: CheckpointProvenance,
    ) -> Result<Self> {
        ensure!(
            sigreg_lambda.is_finite() && sigreg_lambda >= 0.0,
            "MSE-JEPA SIGReg lambda must be finite and non-negative"
        );
        ensure!(
            flow_lambda.is_finite() && flow_lambda >= 0.0,
            "MSE-JEPA flow lambda must be finite and non-negative"
        );
        provenance.validate()?;
        let checkpoint = checkpoint.as_ref();
        provenance.validate_checkpoint_name(checkpoint)?;
        let mut metadata = Self {
            format_version: BUNDLE_VERSION,
            family: FAMILY.to_owned(),
            artifact_role: ARTIFACT_ROLE.to_owned(),
            architecture: ARCHITECTURE.to_owned(),
            feature_layout: FEATURE_LAYOUT.to_owned(),
            latent_dim: LATENT_DIM,
            bar_feature_dim: BAR_FEATURES,
            max_context_bars: MAX_CONTEXT_BARS,
            sigreg_lambda,
            flow_lambda,
            checkpoint_sha256: file_sha256(checkpoint)?,
            provenance,
            lineage_sha256: String::new(),
        };
        metadata.lineage_sha256 = metadata.compute_lineage_sha256();
        Ok(metadata)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_reader(BufReader::new(File::open(path)?))
            .with_context(|| format!("failed parsing MSE-JEPA metadata {}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let path = path.as_ref();
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, self)
            .with_context(|| format!("failed writing MSE-JEPA metadata {}", path.display()))?;
        writer
            .flush()
            .with_context(|| format!("failed flushing MSE-JEPA metadata {}", path.display()))
    }

    pub fn validate_checkpoint(&self, checkpoint: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let checkpoint = checkpoint.as_ref();
        self.provenance.validate_checkpoint_name(checkpoint)?;
        let actual = file_sha256(checkpoint)?;
        ensure!(
            actual == self.checkpoint_sha256,
            "MSE-JEPA checkpoint hash mismatch: metadata={}, actual={actual}",
            self.checkpoint_sha256
        );
        Ok(())
    }

    fn validate_schema(&self) -> Result<()> {
        ensure!(
            self.format_version == BUNDLE_VERSION,
            "unsupported MSE-JEPA bundle version {}",
            self.format_version
        );
        ensure!(
            self.family == FAMILY,
            "cross-family checkpoint rejected: {}",
            self.family
        );
        ensure!(
            self.artifact_role == ARTIFACT_ROLE,
            "MSE-JEPA artifact role mismatch: {}",
            self.artifact_role
        );
        ensure!(
            self.architecture == ARCHITECTURE,
            "MSE-JEPA architecture mismatch: {}",
            self.architecture
        );
        ensure!(
            self.feature_layout == FEATURE_LAYOUT,
            "MSE-JEPA feature layout mismatch: {}",
            self.feature_layout
        );
        ensure!(
            self.latent_dim == LATENT_DIM,
            "MSE-JEPA latent width mismatch"
        );
        ensure!(
            self.bar_feature_dim == BAR_FEATURES,
            "MSE-JEPA feature width mismatch"
        );
        ensure!(
            self.max_context_bars == MAX_CONTEXT_BARS,
            "MSE-JEPA context mismatch"
        );
        ensure!(
            self.sigreg_lambda.is_finite() && self.sigreg_lambda >= 0.0,
            "MSE-JEPA SIGReg lambda is invalid"
        );
        ensure!(
            self.flow_lambda.is_finite() && self.flow_lambda >= 0.0,
            "MSE-JEPA flow lambda is invalid"
        );
        ensure!(
            is_sha256_hex(&self.checkpoint_sha256),
            "MSE-JEPA checkpoint SHA-256 is empty or malformed"
        );
        self.provenance.validate()?;
        let expected = self.compute_lineage_sha256();
        ensure!(
            self.lineage_sha256 == expected,
            "MSE-JEPA lineage mismatch: metadata={}, expected={expected}",
            self.lineage_sha256
        );
        Ok(())
    }

    fn compute_lineage_sha256(&self) -> String {
        let model_lineage = format!(
            "format_version={};family={};artifact_role={};architecture={};feature_layout={};latent_dim={};bar_feature_dim={};max_context_bars={};sigreg_lambda_bits={:016x};flow_lambda_bits={:016x};weights_sha256={};ar_layers={};ar_ff_dim={};projector_hidden_dim={};head_dim={};heads={};pope_dim={};pope_qk_dim={};pope_frequency_base_bits={:016x};pope_attention_scale_bits={:016x};pope_theta_init=two-pi-block-aware;pope_layout=real-then-imag;fa4_contract=strict-bshd-qk128-v64;transition=conditional-flow-matching-linear-path-logit-normal-tau-velocity-v1;flow_head=adaln-zero-gated-mlp-direct-belief-concat;flow_blocks={};flow_hidden_dim={};flow_time_features={};flow_time_scale_bits={:016x};flow_time_max_period_bits={:016x};flow_sampler=heun-uniform-grid;core_emission=online-belief-conditioned-bar-head-arm-controlled-gradient;posthoc_readouts=fresh-joint-emission-and-same-time-token-heads-v1;emission_bins={};emission_prefix_embed_dim={};emission_label_sigma_ratio_bits={:016x};normalization_eps_bits={:016x};projector_norm=rms;projector_rms_eps_bits={:016x};dropout_bits={:016x};sigreg=lewm-c8a4417-n1024-k17-t3-actual-n-attached-future-v1;feature_scale_bits={}",
            self.format_version,
            self.family,
            self.artifact_role,
            self.architecture,
            self.feature_layout,
            self.latent_dim,
            self.bar_feature_dim,
            self.max_context_bars,
            self.sigreg_lambda.to_bits(),
            self.flow_lambda.to_bits(),
            self.checkpoint_sha256,
            AR_LAYERS,
            AR_FF_DIM,
            PROJECTOR_HIDDEN_DIM,
            HEAD_DIM,
            HEADS,
            POPE_DIM,
            POPE_QK_DIM,
            POPE_FREQUENCY_BASE.to_bits(),
            POPE_ATTENTION_SCALE.to_bits(),
            FLOW_BLOCKS,
            FLOW_HIDDEN_DIM,
            FLOW_TIME_FEATURES,
            FLOW_TIME_SCALE.to_bits(),
            FLOW_TIME_MAX_PERIOD.to_bits(),
            NUM_BAR_BINS,
            BAR_PREFIX_EMBED_DIM,
            BAR_LABEL_SIGMA_RATIO.to_bits(),
            NORMALIZATION_EPS.to_bits(),
            super::model::PROJECTOR_RMS_EPS.to_bits(),
            DROPOUT.to_bits(),
            feature_scale_bits(),
        );
        lineage_digest(&format!(
            "{model_lineage};{}",
            self.provenance.lineage_fields()
        ))
    }
}

pub fn bundle_metadata_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("metadata.json")
}

pub fn save_bundle(
    var_store: &nn::VarStore,
    checkpoint: impl AsRef<Path>,
    sigreg_lambda: f64,
    flow_lambda: f64,
    provenance: CheckpointProvenance,
) -> Result<PathBuf> {
    let checkpoint = checkpoint.as_ref();
    let name = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    ensure!(
        name.contains("mse_jepa"),
        "MSE-JEPA bundles require an unambiguous mse_jepa filename"
    );
    provenance.validate()?;
    provenance.validate_checkpoint_name(checkpoint)?;
    var_store.save(checkpoint)?;
    let metadata_path = bundle_metadata_path(checkpoint);
    MseJepaMetadata::for_checkpoint(checkpoint, sigreg_lambda, flow_lambda, provenance)?
        .save(&metadata_path)?;
    Ok(metadata_path)
}

pub fn authenticate(input: impl AsRef<Path>) -> Result<AuthenticatedCheckpoint> {
    let input = input.as_ref();
    ensure!(
        input.exists(),
        "MSE-JEPA checkpoint does not exist: {}",
        input.display()
    );
    let name = input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    ensure!(
        name.contains("mse_jepa"),
        "cross-family, legacy, or ambiguously named checkpoint rejected: {name}"
    );
    let metadata = MseJepaMetadata::load(bundle_metadata_path(input))?;
    metadata.validate_checkpoint(input)?;
    let kind = if metadata.provenance.head_fit.is_some() {
        CheckpointKind::FittedReadout
    } else {
        CheckpointKind::Core
    };
    Ok(AuthenticatedCheckpoint {
        path: input.to_path_buf(),
        kind,
        checkpoint_sha256: metadata.checkpoint_sha256,
        lineage_sha256: metadata.lineage_sha256,
        sigreg_lambda: metadata.sigreg_lambda,
        flow_lambda: metadata.flow_lambda,
        provenance: metadata.provenance,
    })
}

pub fn load_authenticated(
    var_store: &mut nn::VarStore,
    input: impl AsRef<Path>,
) -> Result<AuthenticatedCheckpoint> {
    let authenticated = authenticate(input)?;
    let summary = load_var_store_partial(var_store, &authenticated.path)
        .map_err(|error| anyhow!("{error}"))?;
    summary.require_complete().map_err(|error| {
        anyhow!(
            "authenticated MSE-JEPA checkpoint {} is incomplete: {error}",
            authenticated.path.display()
        )
    })?;
    Ok(authenticated)
}

fn feature_scale_bits() -> String {
    TRANSITION_BAR_FEATURE_SCALE
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn lineage_digest(canonical: &str) -> String {
    hex_digest(digest(&SHA256, canonical.as_bytes()).as_ref())
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}
#[cfg(test)]
mod tests {
    use super::{
        authenticate, bundle_metadata_path, eligible_name_sha256, CheckpointKind,
        CheckpointProvenance, CoreInitialization, CoreTrainingOrigin, EmissionGradientMode,
        HeadFitProvenance, MseJepaMetadata, WeightReadout, CANONICAL_READOUT_FIT_BATCH_SIZE,
        CANONICAL_READOUT_FIT_SEED, CANONICAL_READOUT_FIT_STEPS,
        CANONICAL_READOUT_OPTIMIZER_RECIPE_ID, CANONICAL_READOUT_TOKEN_ROWS,
        CANONICAL_READOUT_VALIDATION_WINDOWS,
    };
    use crate::torch::lejepa::model::{ARCHITECTURE, BAR_FEATURES, FEATURE_LAYOUT, LATENT_DIM};
    use crate::torch::lejepa::sigreg::DEFAULT_SIGREG_LAMBDA;
    use std::fs;
    use std::path::Path;

    fn fixture_dir() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("mse-jepa-checkpoint-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        path
    }
    fn core_origin() -> CoreTrainingOrigin {
        CoreTrainingOrigin {
            initialization: CoreInitialization::Fresh,
            train_seed: 0x5eed,
            batch_size: 8,
            resolution_secs: 300,
            min_bars: 60_002,
            split_bounds: (1_700_000_000_000, 1_710_000_000_000),
            split_bounds_pinned: true,
            corpus_fingerprint: "c".repeat(64),
            supports_scoring_sha256: "d".repeat(64),
        }
    }

    fn raw_provenance() -> CheckpointProvenance {
        CheckpointProvenance {
            completed_steps: 42,
            planned_steps: 50,
            steps_per_pass: 50,
            optimizer_recipe_id: "test-dbwm-recipe".to_owned(),
            emission_gradient_mode: EmissionGradientMode::Attached,
            weight_readout: WeightReadout::Raw,
            core_origin: core_origin(),
            head_fit: None,
        }
    }

    fn tail_ema_provenance() -> CheckpointProvenance {
        CheckpointProvenance {
            weight_readout: WeightReadout::TailEma {
                start_step: 40,
                horizon_steps: 3,
                blend: 0.6,
                update_count: 3,
                eligible_name_sha256: eligible_name_sha256([
                    "ar.blocks.0.weight",
                    "flow.out_proj.weight",
                ])
                .unwrap(),
            },
            ..raw_provenance()
        }
    }

    fn fitted_provenance() -> CheckpointProvenance {
        let mut provenance = raw_provenance();
        provenance.completed_steps = provenance.planned_steps;
        provenance.head_fit = Some(HeadFitProvenance {
            source_checkpoint_sha256: "a".repeat(64),
            source_lineage_sha256: "b".repeat(64),
            source_weight_readout: WeightReadout::Raw,
            source_core_origin: provenance.core_origin.clone(),
            seed: CANONICAL_READOUT_FIT_SEED,
            steps: CANONICAL_READOUT_FIT_STEPS,
            batch_size: CANONICAL_READOUT_FIT_BATCH_SIZE,
            token_rows_per_step: CANONICAL_READOUT_TOKEN_ROWS,
            validation_windows: CANONICAL_READOUT_VALIDATION_WINDOWS,
            optimizer_recipe_id: CANONICAL_READOUT_OPTIMIZER_RECIPE_ID.to_owned(),
        });
        provenance
    }

    fn metadata(checkpoint: &Path, provenance: CheckpointProvenance) -> MseJepaMetadata {
        MseJepaMetadata::for_checkpoint(checkpoint, DEFAULT_SIGREG_LAMBDA, 1.0, provenance).unwrap()
    }

    fn write_unchecked(checkpoint: &Path, metadata: &MseJepaMetadata) {
        fs::write(
            bundle_metadata_path(checkpoint),
            serde_json::to_vec(metadata).unwrap(),
        )
        .unwrap();
    }

    fn assert_rehashed_metadata_is_rejected(checkpoint: &Path, mut metadata: MseJepaMetadata) {
        metadata.lineage_sha256 = metadata.compute_lineage_sha256();
        write_unchecked(checkpoint, &metadata);
        assert!(authenticate(checkpoint).is_err());
    }

    #[test]
    fn new_bundle_metadata_is_family_named_hashed_and_lineage_bound() {
        let root = fixture_dir();
        let checkpoint = root.join("mse_jepa_best.ot");
        fs::write(&checkpoint, b"self-contained model fixture").unwrap();
        let metadata = metadata(&checkpoint, raw_provenance());
        metadata.save(bundle_metadata_path(&checkpoint)).unwrap();
        assert_eq!(metadata.format_version, 14);
        assert_eq!(metadata.architecture, ARCHITECTURE);
        assert_eq!(metadata.feature_layout, FEATURE_LAYOUT);
        assert_eq!(metadata.latent_dim, LATENT_DIM);
        assert_eq!(metadata.bar_feature_dim, BAR_FEATURES);
        assert_eq!(metadata.flow_lambda, 1.0);
        let authenticated = authenticate(&checkpoint).unwrap();
        assert_eq!(authenticated.path, checkpoint);
        assert_eq!(authenticated.kind, CheckpointKind::Core);
        assert_eq!(authenticated.provenance, raw_provenance());

        fs::write(&checkpoint, b"tampered model fixture").unwrap();
        assert!(authenticate(&checkpoint).is_err());
        assert!(MseJepaMetadata::for_checkpoint(
            &checkpoint,
            DEFAULT_SIGREG_LAMBDA,
            -1.0,
            raw_provenance(),
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_and_tail_ema_readouts_cannot_masquerade_as_each_other() {
        let root = fixture_dir();
        let raw_checkpoint = root.join("mse_jepa.ot");
        let ema_checkpoint = root.join("mse_jepa_tail_ema.ot");
        fs::write(&raw_checkpoint, b"same weights").unwrap();
        fs::write(&ema_checkpoint, b"same weights").unwrap();

        let raw = metadata(&raw_checkpoint, raw_provenance());
        let ema = metadata(&ema_checkpoint, tail_ema_provenance());
        assert_ne!(raw.lineage_sha256, ema.lineage_sha256);
        raw.save(bundle_metadata_path(&raw_checkpoint)).unwrap();
        ema.save(bundle_metadata_path(&ema_checkpoint)).unwrap();
        assert!(matches!(
            authenticate(&raw_checkpoint)
                .unwrap()
                .provenance
                .weight_readout,
            WeightReadout::Raw
        ));
        assert!(matches!(
            authenticate(&ema_checkpoint)
                .unwrap()
                .provenance
                .weight_readout,
            WeightReadout::TailEma { .. }
        ));

        let mut disguised_raw = raw.clone();
        disguised_raw.provenance.weight_readout = tail_ema_provenance().weight_readout;
        assert_rehashed_metadata_is_rejected(&raw_checkpoint, disguised_raw);

        let mut disguised_ema = ema;
        disguised_ema.provenance.weight_readout = WeightReadout::Raw;
        assert_rehashed_metadata_is_rejected(&ema_checkpoint, disguised_ema);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn fitted_stage_and_source_hashes_are_authenticated() {
        let root = fixture_dir();
        let checkpoint = root.join("mse_jepa_fitted.ot");
        fs::write(&checkpoint, b"fitted endpoint").unwrap();
        let fitted = metadata(&checkpoint, fitted_provenance());
        fitted.save(bundle_metadata_path(&checkpoint)).unwrap();
        let authenticated = authenticate(&checkpoint).unwrap();
        assert_eq!(authenticated.kind, CheckpointKind::FittedReadout);

        let mut tampered = fitted.clone();
        tampered
            .provenance
            .head_fit
            .as_mut()
            .unwrap()
            .source_checkpoint_sha256 = "c".repeat(64);
        write_unchecked(&checkpoint, &tampered);
        assert!(authenticate(&checkpoint).is_err());
        fitted.save(bundle_metadata_path(&checkpoint)).unwrap();
        let mut mismatched_origin = fitted.clone();
        mismatched_origin
            .provenance
            .head_fit
            .as_mut()
            .unwrap()
            .source_core_origin
            .train_seed += 1;
        assert_rehashed_metadata_is_rejected(&checkpoint, mismatched_origin);
        fitted.save(bundle_metadata_path(&checkpoint)).unwrap();

        let mut noncanonical_fit = fitted.clone();
        noncanonical_fit.provenance.head_fit.as_mut().unwrap().steps -= 1;
        assert_rehashed_metadata_is_rejected(&checkpoint, noncanonical_fit);
        fitted.save(bundle_metadata_path(&checkpoint)).unwrap();

        let core_name = root.join("mse_jepa.ot");
        fs::rename(&checkpoint, &core_name).unwrap();
        fs::rename(
            bundle_metadata_path(&checkpoint),
            bundle_metadata_path(&core_name),
        )
        .unwrap();
        assert!(authenticate(&core_name).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn every_provenance_number_and_readout_detail_is_lineage_bound() {
        let root = fixture_dir();
        let checkpoint = root.join("mse_jepa_tail_ema.ot");
        fs::write(&checkpoint, b"tail EMA lineage fixture").unwrap();
        let original = metadata(&checkpoint, tail_ema_provenance());
        let original_lineage = original.compute_lineage_sha256();

        let mut variants = Vec::new();
        let mut changed = original.clone();
        changed.provenance.completed_steps += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.planned_steps += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.steps_per_pass += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.optimizer_recipe_id.push_str("-changed");
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.train_seed += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.batch_size += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.resolution_secs += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.min_bars += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.split_bounds.0 += 1;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.split_bounds_pinned = false;
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.corpus_fingerprint = "e".repeat(64);
        variants.push(changed);
        let mut changed = original.clone();
        changed.provenance.core_origin.supports_scoring_sha256 = "f".repeat(64);
        variants.push(changed);

        let readout_changes: [fn(&mut WeightReadout); 5] = [
            |readout: &mut WeightReadout| {
                if let WeightReadout::TailEma { start_step, .. } = readout {
                    *start_step += 1;
                }
            },
            |readout: &mut WeightReadout| {
                if let WeightReadout::TailEma { horizon_steps, .. } = readout {
                    *horizon_steps += 1;
                }
            },
            |readout: &mut WeightReadout| {
                if let WeightReadout::TailEma { blend, .. } = readout {
                    *blend = 0.61;
                }
            },
            |readout: &mut WeightReadout| {
                if let WeightReadout::TailEma { update_count, .. } = readout {
                    *update_count += 1;
                }
            },
            |readout: &mut WeightReadout| {
                if let WeightReadout::TailEma {
                    eligible_name_sha256,
                    ..
                } = readout
                {
                    *eligible_name_sha256 = "a".repeat(64);
                }
            },
        ];
        for change_readout in readout_changes {
            let mut changed = original.clone();
            change_readout(&mut changed.provenance.weight_readout);
            variants.push(changed);
        }

        for changed in variants {
            assert_ne!(changed.compute_lineage_sha256(), original_lineage);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_provenance_fails_authentication_even_with_rehashed_lineage() {
        let root = fixture_dir();
        let raw_checkpoint = root.join("mse_jepa.ot");
        fs::write(&raw_checkpoint, b"raw malformed provenance fixture").unwrap();
        let raw = metadata(&raw_checkpoint, raw_provenance());

        let mut malformed = raw.clone();
        malformed.provenance.completed_steps = malformed.provenance.planned_steps + 1;
        assert_rehashed_metadata_is_rejected(&raw_checkpoint, malformed);
        let mut malformed = raw.clone();
        malformed.provenance.planned_steps = 0;
        assert_rehashed_metadata_is_rejected(&raw_checkpoint, malformed);
        let mut malformed = raw.clone();
        malformed.provenance.steps_per_pass = 0;
        assert_rehashed_metadata_is_rejected(&raw_checkpoint, malformed);
        let mut malformed = raw;
        malformed.provenance.optimizer_recipe_id.clear();
        assert_rehashed_metadata_is_rejected(&raw_checkpoint, malformed);

        let ema_checkpoint = root.join("mse_jepa_tail_ema.ot");
        fs::write(&ema_checkpoint, b"EMA malformed provenance fixture").unwrap();
        let ema = metadata(&ema_checkpoint, tail_ema_provenance());
        for malformed_readout in [
            WeightReadout::TailEma {
                start_step: 43,
                horizon_steps: 3,
                blend: 0.6,
                update_count: 1,
                eligible_name_sha256: "a".repeat(64),
            },
            WeightReadout::TailEma {
                start_step: 40,
                horizon_steps: 0,
                blend: 0.6,
                update_count: 3,
                eligible_name_sha256: "a".repeat(64),
            },
            WeightReadout::TailEma {
                start_step: 40,
                horizon_steps: 3,
                blend: 0.0,
                update_count: 3,
                eligible_name_sha256: "a".repeat(64),
            },
            WeightReadout::TailEma {
                start_step: 40,
                horizon_steps: 3,
                blend: 1.1,
                update_count: 3,
                eligible_name_sha256: "a".repeat(64),
            },
            WeightReadout::TailEma {
                start_step: 40,
                horizon_steps: 3,
                blend: 0.6,
                update_count: 2,
                eligible_name_sha256: "a".repeat(64),
            },
            WeightReadout::TailEma {
                start_step: 40,
                horizon_steps: 3,
                blend: 0.6,
                update_count: 3,
                eligible_name_sha256: String::new(),
            },
            WeightReadout::TailEma {
                start_step: 40,
                horizon_steps: 3,
                blend: 0.6,
                update_count: 3,
                eligible_name_sha256: "A".repeat(64),
            },
        ] {
            let mut malformed = ema.clone();
            malformed.provenance.weight_readout = malformed_readout;
            assert_rehashed_metadata_is_rejected(&ema_checkpoint, malformed);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn eligible_name_hash_has_a_canonical_ordered_encoding() {
        let first = eligible_name_sha256(["z.weight", "a.weight"]).unwrap();
        let second = eligible_name_sha256(["a.weight", "z.weight"]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        assert!(eligible_name_sha256(std::iter::empty()).is_err());
        assert!(eligible_name_sha256([""]).is_err());
        assert!(eligible_name_sha256(["same", "same"]).is_err());
    }

    #[test]
    fn historical_v11_v12_bundles_are_incompatible() {
        let root = fixture_dir();
        let checkpoint = root.join("mse_jepa_old.ot");
        fs::write(&checkpoint, b"historical feature model").unwrap();

        let current = metadata(&checkpoint, raw_provenance());
        for (format_version, architecture, feature_layout, bar_feature_dim) in [
            (
                8,
                "lejepa-msejepa-intrabar6-causal-ar-pope64-fa4-v7",
                "single-bar-ohlcv-logshape-gk-volume-fixed-scale-v3",
                6,
            ),
            (
                9,
                "lejepa-msejepa-transition7-causal-ar-pope64-fa4-v8",
                "transition-bar-ohlcv-logshape-gk-volume-delta-fixed-scale-v4",
                7,
            ),
            (
                10,
                "lejepa-dbwm-transition7-causal-ar-pope64-fa4-v9",
                "transition-bar-ohlcv-logshape-gk-volume-delta-fixed-scale-v4",
                7,
            ),
            (
                11,
                "lejepa-dbwm-transition7-causal-ar-pope64-fa4-v10",
                "transition-bar-ohlcv-logshape-gk-volume-delta-fixed-scale-v4",
                7,
            ),
            (
                12,
                "lejepa-dbwm-transition7-causal-ar-pope64-fa4-flow-emission-stopbelief-v12",
                FEATURE_LAYOUT,
                BAR_FEATURES,
            ),
            (
                13,
                "lejepa-dbwm-transition7-causal-ar-pope64-fa4-flow-emission-stopbelief-v12",
                FEATURE_LAYOUT,
                BAR_FEATURES,
            ),
            (
                13,
                "lejepa-dbwm-transition7-causal-ar-pope64-fa4-flow-v11",
                FEATURE_LAYOUT,
                BAR_FEATURES,
            ),
        ] {
            let mut old = current.clone();
            old.format_version = format_version;
            old.architecture = architecture.to_owned();
            old.feature_layout = feature_layout.to_owned();
            old.bar_feature_dim = bar_feature_dim;
            assert_rehashed_metadata_is_rejected(&checkpoint, old);
        }

        let mut pre_provenance = serde_json::to_value(&current).unwrap();
        pre_provenance.as_object_mut().unwrap().remove("provenance");
        fs::write(
            bundle_metadata_path(&checkpoint),
            serde_json::to_vec(&pre_provenance).unwrap(),
        )
        .unwrap();
        assert!(authenticate(&checkpoint).is_err());

        let legacy_heads = root.join("pretrain_heads.ot");
        fs::write(&legacy_heads, b"legacy heads").unwrap();
        assert!(authenticate(&legacy_heads).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn categorical_checkpoint_names_cannot_cross_into_mse_jepa() {
        let root = fixture_dir();
        let checkpoint = root.join("pretrain_best.ot");
        fs::write(&checkpoint, b"categorical").unwrap();
        assert!(authenticate(&checkpoint).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
