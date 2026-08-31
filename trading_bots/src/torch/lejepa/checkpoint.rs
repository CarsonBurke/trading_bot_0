use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, ensure, Context, Result};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use tch::nn;

use crate::torch::hashing::file_sha256;
use crate::torch::load::load_var_store_partial;
use crate::torch::pope::{POPE_ATTENTION_SCALE, POPE_DIM, POPE_FREQUENCY_BASE, POPE_QK_DIM};

use super::model::{
    ARCHITECTURE, AR_FF_DIM, AR_LAYERS, BAR_FEATURES, DROPOUT, FEATURE_LAYOUT, HEADS, HEAD_DIM,
    LATENT_DIM, MAX_CONTEXT_BARS, NORMALIZATION_EPS, OHLC_FEATURE_SCALE, PREDICTOR_BLOCKS,
    PREDICTOR_HIDDEN_MULT, PROJECTOR_HIDDEN_DIM,
};

const LEGACY_VERSION: u32 = 5;
const BUNDLE_VERSION: u32 = 7;
const FAMILY: &str = "mse_jepa";
const ARTIFACT_ROLE: &str = "self-contained-mse-jepa-model";
const LEGACY_CACHE_CONTRACT: &str = "stateful-circular-pope-absolute-bshd-k128-v64-fa4-v2";
const LEGACY_PROBE_LOGVAR_LIMIT: f64 = 7.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointKind {
    SelfContained,
    LegacyHeads,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedCheckpoint {
    pub path: PathBuf,
    pub kind: CheckpointKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    pub checkpoint_sha256: String,
    pub lineage_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct LegacyV5Metadata {
    format_version: u32,
    architecture: String,
    feature_layout: String,
    latent_dim: i64,
    bar_feature_dim: i64,
    max_context_bars: i64,
    target_scale: f64,
    checkpoint_sha256: String,
    lineage_sha256: String,
}

impl MseJepaMetadata {
    pub fn for_checkpoint(checkpoint: impl AsRef<Path>, sigreg_lambda: f64) -> Result<Self> {
        ensure!(
            sigreg_lambda.is_finite() && sigreg_lambda >= 0.0,
            "MSE-JEPA SIGReg lambda must be finite and non-negative"
        );
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
            checkpoint_sha256: file_sha256(checkpoint)?,
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
        let expected = self.compute_lineage_sha256();
        ensure!(
            self.lineage_sha256 == expected,
            "MSE-JEPA lineage mismatch: metadata={}, expected={expected}",
            self.lineage_sha256
        );
        Ok(())
    }

    fn compute_lineage_sha256(&self) -> String {
        lineage_digest(&format!(
            "format_version={};family={};artifact_role={};architecture={};feature_layout={};latent_dim={};bar_feature_dim={};max_context_bars={};sigreg_lambda_bits={:016x};weights_sha256={};ar_layers={};ar_ff_dim={};projector_hidden_dim={};head_dim={};heads={};pope_dim={};pope_qk_dim={};pope_frequency_base_bits={:016x};pope_attention_scale_bits={:016x};pope_theta_init=two-pi-block-aware;pope_layout=real-then-imag;fa4_contract=strict-bshd-qk128-v64;predictor_hidden_mult={};predictor_blocks={};normalization_eps_bits={:016x};projector_norm=rms;projector_rms_eps_bits={:016x};dropout_bits={:016x};sigreg=lewm-c8a4417-n1024-k17-t3-actual-n-attached-future-v1;feature_scale_bits={}",
            self.format_version,
            self.family,
            self.artifact_role,
            self.architecture,
            self.feature_layout,
            self.latent_dim,
            self.bar_feature_dim,
            self.max_context_bars,
            self.sigreg_lambda.to_bits(),
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
            PREDICTOR_HIDDEN_MULT,
            PREDICTOR_BLOCKS,
            NORMALIZATION_EPS.to_bits(),
            super::model::PROJECTOR_RMS_EPS.to_bits(),
            DROPOUT.to_bits(),
            feature_scale_bits(),
        ))
    }
}

impl LegacyV5Metadata {
    fn load(path: &Path) -> Result<Self> {
        serde_json::from_reader(BufReader::new(File::open(path)?))
            .with_context(|| format!("failed parsing legacy MSE-JEPA metadata {}", path.display()))
    }

    fn validate_checkpoint(&self, checkpoint: &Path) -> Result<()> {
        self.validate_schema()?;
        let actual = file_sha256(checkpoint)?;
        ensure!(
            actual == self.checkpoint_sha256,
            "legacy MSE-JEPA checkpoint hash mismatch: metadata={}, actual={actual}",
            self.checkpoint_sha256
        );
        Ok(())
    }

    fn validate_schema(&self) -> Result<()> {
        ensure!(
            self.format_version == LEGACY_VERSION,
            "legacy v1 and non-v5 metadata are unsupported"
        );
        ensure!(
            self.architecture == ARCHITECTURE,
            "legacy architecture rejected: {}",
            self.architecture
        );
        ensure!(
            self.feature_layout == FEATURE_LAYOUT,
            "legacy feature layout rejected: {}",
            self.feature_layout
        );
        ensure!(
            self.latent_dim == LATENT_DIM,
            "legacy latent width mismatch"
        );
        ensure!(
            self.bar_feature_dim == BAR_FEATURES,
            "legacy feature width mismatch"
        );
        ensure!(
            self.max_context_bars == MAX_CONTEXT_BARS,
            "legacy context mismatch"
        );
        ensure!(
            self.target_scale.is_finite() && self.target_scale > 0.0,
            "legacy target scale is invalid"
        );
        let expected = self.compute_lineage_sha256();
        ensure!(
            self.lineage_sha256 == expected,
            "legacy MSE-JEPA lineage mismatch: metadata={}, expected={expected}",
            self.lineage_sha256
        );
        Ok(())
    }

    fn compute_lineage_sha256(&self) -> String {
        lineage_digest(&format!(
            "format_version={};architecture={};feature_layout={};latent_dim={};bar_feature_dim={};max_context_bars={};target_scale_bits={:016x};weights_sha256={};ar_layers={};ar_ff_dim={};projector_hidden_dim={};head_dim={};heads={};pope_dim={};pope_qk_dim={};pope_frequency_base_bits={:016x};pope_attention_scale_bits={:016x};pope_theta_init=two-pi-block-aware;pope_layout=real-then-imag;fa4_contract=strict-bshd-qk128-v64;predictor_hidden_mult={};predictor_blocks={};normalization_eps_bits={:016x};probe_logvar_limit_bits={:016x};cache_contract={};ohlc_feature_scale_bits={}",
            self.format_version,
            self.architecture,
            self.feature_layout,
            self.latent_dim,
            self.bar_feature_dim,
            self.max_context_bars,
            self.target_scale.to_bits(),
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
            PREDICTOR_HIDDEN_MULT,
            PREDICTOR_BLOCKS,
            NORMALIZATION_EPS.to_bits(),
            LEGACY_PROBE_LOGVAR_LIMIT.to_bits(),
            LEGACY_CACHE_CONTRACT,
            feature_scale_bits(),
        ))
    }
}

pub fn bundle_metadata_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("metadata.json")
}

fn legacy_metadata_path(checkpoint: &Path) -> PathBuf {
    checkpoint.with_extension("metadata.json")
}

pub fn save_bundle(
    var_store: &nn::VarStore,
    checkpoint: impl AsRef<Path>,
    sigreg_lambda: f64,
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
    var_store.save(checkpoint)?;
    let metadata_path = bundle_metadata_path(checkpoint);
    MseJepaMetadata::for_checkpoint(checkpoint, sigreg_lambda)?.save(&metadata_path)?;
    Ok(metadata_path)
}

pub fn authenticate(input: impl AsRef<Path>) -> Result<AuthenticatedCheckpoint> {
    let input = input.as_ref();
    let resolved = resolve_legacy_heads(input).unwrap_or_else(|| input.to_path_buf());
    ensure!(
        resolved.exists(),
        "MSE-JEPA checkpoint does not exist: {}",
        resolved.display()
    );
    let name = resolved
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if name.starts_with("pretrain_heads") {
        let metadata = LegacyV5Metadata::load(&legacy_metadata_path(&resolved))?;
        metadata.validate_checkpoint(&resolved)?;
        return Ok(AuthenticatedCheckpoint {
            path: resolved,
            kind: CheckpointKind::LegacyHeads,
        });
    }
    ensure!(
        name.contains("mse_jepa"),
        "cross-family or ambiguously named checkpoint rejected: {name}"
    );
    let metadata = MseJepaMetadata::load(bundle_metadata_path(&resolved))?;
    metadata.validate_checkpoint(&resolved)?;
    Ok(AuthenticatedCheckpoint {
        path: resolved,
        kind: CheckpointKind::SelfContained,
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

fn resolve_legacy_heads(input: &Path) -> Option<PathBuf> {
    let parent = input.parent()?;
    let name = input.file_name()?.to_str()?;
    match name {
        "pretrain_model.ot" => Some(parent.join("pretrain_heads.ot")),
        "pretrain_model_best.ot" => Some(parent.join("pretrain_heads_best.ot")),
        _ => name
            .strip_prefix("pretrain_step")
            .and_then(|suffix| suffix.strip_suffix(".ot"))
            .map(|step| parent.join(format!("pretrain_heads_step{step}.ot"))),
    }
}

fn feature_scale_bits() -> String {
    OHLC_FEATURE_SCALE
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn lineage_digest(canonical: &str) -> String {
    digest(&SHA256, canonical.as_bytes())
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        authenticate, bundle_metadata_path, legacy_metadata_path, CheckpointKind, LegacyV5Metadata,
        MseJepaMetadata,
    };
    use crate::torch::hashing::file_sha256;
    use crate::torch::lejepa::model::{
        ARCHITECTURE, BAR_FEATURES, FEATURE_LAYOUT, LATENT_DIM, MAX_CONTEXT_BARS,
    };
    use crate::torch::lejepa::sigreg::DEFAULT_SIGREG_LAMBDA;
    use std::fs;

    fn fixture_dir() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("mse-jepa-checkpoint-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn legacy_metadata(checkpoint: &std::path::Path) -> LegacyV5Metadata {
        let mut metadata = LegacyV5Metadata {
            format_version: 5,
            architecture: ARCHITECTURE.to_owned(),
            feature_layout: FEATURE_LAYOUT.to_owned(),
            latent_dim: LATENT_DIM,
            bar_feature_dim: BAR_FEATURES,
            max_context_bars: MAX_CONTEXT_BARS,
            target_scale: 100.0,
            checkpoint_sha256: file_sha256(checkpoint).unwrap(),
            lineage_sha256: String::new(),
        };
        metadata.lineage_sha256 = metadata.compute_lineage_sha256();
        metadata
    }

    #[test]
    fn legacy_v5_lineage_matches_a_real_c277_artifact() {
        let metadata = LegacyV5Metadata {
            format_version: 5,
            architecture: ARCHITECTURE.to_owned(),
            feature_layout: FEATURE_LAYOUT.to_owned(),
            latent_dim: LATENT_DIM,
            bar_feature_dim: BAR_FEATURES,
            max_context_bars: MAX_CONTEXT_BARS,
            target_scale: 100.0,
            checkpoint_sha256: "2279fdd99decd74b7c3c949047c0cb7bbf748a20e124456109504ec8af4afde0"
                .to_owned(),
            lineage_sha256: String::new(),
        };
        assert_eq!(
            metadata.compute_lineage_sha256(),
            "12a0e03a4db50ab8f3abedd98caf5fc5ab064e5e150d1e4024ef5ed0a2701c26"
        );
    }

    #[test]
    fn new_bundle_metadata_is_family_named_hashed_and_lineage_bound() {
        let root = fixture_dir();
        let checkpoint = root.join("mse_jepa_best.ot");
        fs::write(&checkpoint, b"self-contained model fixture").unwrap();
        let metadata = MseJepaMetadata::for_checkpoint(&checkpoint, DEFAULT_SIGREG_LAMBDA).unwrap();
        metadata.save(bundle_metadata_path(&checkpoint)).unwrap();
        assert_eq!(metadata.format_version, 7);
        let authenticated = authenticate(&checkpoint).unwrap();
        assert_eq!(authenticated.path, checkpoint);
        assert_eq!(authenticated.kind, CheckpointKind::SelfContained);

        let mut obsolete_bundle = metadata.clone();
        obsolete_bundle.format_version = 6;
        obsolete_bundle.lineage_sha256 = obsolete_bundle.compute_lineage_sha256();
        obsolete_bundle
            .save(bundle_metadata_path(&checkpoint))
            .unwrap_err();

        let mut wrong_family = metadata;
        wrong_family.family = "categorical".to_owned();
        wrong_family.lineage_sha256 = wrong_family.compute_lineage_sha256();
        fs::write(
            bundle_metadata_path(&checkpoint),
            serde_json::to_vec(&wrong_family).unwrap(),
        )
        .unwrap();
        assert!(authenticate(&checkpoint).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_v5_requires_both_artifact_hash_and_exact_lineage() {
        let root = fixture_dir();
        let checkpoint = root.join("pretrain_heads.ot");
        fs::write(&checkpoint, b"historical heads fixture").unwrap();
        let metadata = legacy_metadata(&checkpoint);
        fs::write(
            legacy_metadata_path(&checkpoint),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        assert_eq!(
            authenticate(&checkpoint).unwrap().kind,
            CheckpointKind::LegacyHeads
        );

        fs::write(&checkpoint, b"tampered").unwrap();
        assert!(authenticate(&checkpoint).is_err());
        fs::write(&checkpoint, b"historical heads fixture").unwrap();
        let mut wrong_lineage = metadata;
        wrong_lineage.lineage_sha256 = "00".repeat(32);
        fs::write(
            legacy_metadata_path(&checkpoint),
            serde_json::to_vec(&wrong_lineage).unwrap(),
        )
        .unwrap();
        assert!(authenticate(&checkpoint).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn historical_model_path_resolves_only_to_authenticated_heads() {
        let root = fixture_dir();
        let model = root.join("pretrain_model.ot");
        let heads = root.join("pretrain_heads.ot");
        fs::write(&model, b"unbound companion model").unwrap();
        fs::write(&heads, b"bound heads").unwrap();
        let metadata = legacy_metadata(&heads);
        fs::write(
            legacy_metadata_path(&heads),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        let authenticated = authenticate(&model).unwrap();
        assert_eq!(authenticated.path, heads);
        assert_eq!(authenticated.kind, CheckpointKind::LegacyHeads);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn v1_plainflow_and_other_v5_tuples_are_rejected() {
        let root = fixture_dir();
        let checkpoint = root.join("pretrain_heads.ot");
        fs::write(&checkpoint, b"legacy").unwrap();
        for (version, architecture) in [
            (1, "lejepa-causal-ar-v1"),
            (5, "lejepa-plainflow-causal-ar-v5"),
            (5, "lejepa-msejepa-causal-ar-pope64-fa4-v5"),
        ] {
            let mut metadata = legacy_metadata(&checkpoint);
            metadata.format_version = version;
            metadata.architecture = architecture.to_owned();
            metadata.lineage_sha256 = metadata.compute_lineage_sha256();
            fs::write(
                legacy_metadata_path(&checkpoint),
                serde_json::to_vec(&metadata).unwrap(),
            )
            .unwrap();
            assert!(authenticate(&checkpoint).is_err());
        }
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
