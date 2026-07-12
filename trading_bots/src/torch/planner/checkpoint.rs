use std::{
    fs::{self, File},
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tch::nn;

use crate::torch::hashing::file_sha256;
use crate::torch::optim::muon::Muon;
use crate::torch::{
    constants::{ACTION_THRESHOLD, COMMISSION_RATE},
    train::config::{
        LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM, MUON_MOMENTUM_WARMUP_START,
        MUON_MOMENTUM_WARMUP_STEPS, USE_MUON,
    },
};

use super::{
    gae::{PLANNER_GAE_LAMBDA, PLANNER_GAMMA},
    losses::{POSITIVE_WEIGHT, REVERSE_KL_COEFFICIENT, VALUE_LOSS_COEFFICIENT},
    portfolio::PLANNER_REWARD_SCALE,
    PLANNER_HEADS, PLANNER_LATENT_DIM, PLANNER_LAYERS, PLANNER_MODEL_DIM, PLANNER_OHLC_DIM,
    PLANNER_PORTFOLIO_DIM,
};

const FORMAT_VERSION: u32 = 2;
const ARCHITECTURE: &str = "world-model-planner-bidirectional-dual-critic-v2";
const INPUT_NORMALIZATION: &str = "wm-feature-scale-relative-logvar-v1";
const REAL_SOURCE_FRACTION: f64 = 0.5;
pub(crate) const OPTIMIZATION_EPOCHS: usize = 3;
pub(crate) const TARGET_KL: f64 = 0.035;
pub(crate) const KL_CONTROLLER_HALF_LIFE: f64 = 50.0;
pub(crate) const KL_MIN_LR_SCALE: f64 = 0.01;
pub(crate) const KL_MAX_LR_SCALE: f64 = 10.0;
pub(crate) const FANTASY_CLOSE_DELTA_MIN: f64 = -0.25;
pub(crate) const FANTASY_CLOSE_DELTA_MAX: f64 = 0.25;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannerTrainingContract {
    pub reward_scale: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub real_source_fraction: f64,
    pub pmpo_positive_weight: f64,
    pub pmpo_reverse_kl_coefficient: f64,
    pub value_loss_coefficient: f64,
    pub optimization_epochs: usize,
    pub target_kl: f64,
    pub kl_controller_half_life: f64,
    pub kl_min_lr_scale: f64,
    pub kl_max_lr_scale: f64,
    pub input_normalization: String,
    pub muon_lr: f64,
    pub adamw_lr: f64,
    pub adamw_beta1: f64,
    pub adamw_beta2: f64,
    pub adamw_epsilon: f64,
    pub use_muon: bool,
    pub muon_momentum: f64,
    pub muon_momentum_warmup_start: f64,
    pub muon_momentum_warmup_steps: i64,
    pub muon_nesterov: bool,
    pub muon_beta2: f64,
    pub muon_weight_decay: f64,
    pub muon_newton_schulz_steps: usize,
    pub max_grad_norm: f64,
    pub adamw_weight_decay: f64,
    pub optimizer_routing: String,
    pub action_threshold: f64,
    pub commission_rate: f64,
    pub fantasy_close_delta_min: f64,
    pub fantasy_close_delta_max: f64,
    pub fantasy_sanitization: String,
    pub validation_policy: String,
    pub base_seed: u64,
}

impl PlannerTrainingContract {
    fn new(base_seed: u64) -> Self {
        Self {
            reward_scale: PLANNER_REWARD_SCALE,
            gamma: PLANNER_GAMMA,
            gae_lambda: PLANNER_GAE_LAMBDA,
            real_source_fraction: REAL_SOURCE_FRACTION,
            pmpo_positive_weight: POSITIVE_WEIGHT,
            pmpo_reverse_kl_coefficient: REVERSE_KL_COEFFICIENT,
            value_loss_coefficient: VALUE_LOSS_COEFFICIENT,
            optimization_epochs: OPTIMIZATION_EPOCHS,
            target_kl: TARGET_KL,
            kl_controller_half_life: KL_CONTROLLER_HALF_LIFE,
            kl_min_lr_scale: KL_MIN_LR_SCALE,
            kl_max_lr_scale: KL_MAX_LR_SCALE,
            input_normalization: INPUT_NORMALIZATION.to_owned(),
            muon_lr: MUON_LR,
            adamw_lr: LEARNING_RATE,
            adamw_beta1: 0.9,
            adamw_beta2: 0.95,
            adamw_epsilon: 1e-8,
            use_muon: USE_MUON,
            muon_momentum: MUON_MOMENTUM,
            muon_momentum_warmup_start: MUON_MOMENTUM_WARMUP_START,
            muon_momentum_warmup_steps: MUON_MOMENTUM_WARMUP_STEPS,
            muon_nesterov: true,
            muon_beta2: 0.95,
            muon_weight_decay: 0.0,
            muon_newton_schulz_steps: 5,
            max_grad_norm: MAX_GRAD_NORM,
            adamw_weight_decay: 0.0,
            optimizer_routing: "muon-matrices-adamw-policy-value-v1".to_owned(),
            action_threshold: ACTION_THRESHOLD,
            commission_rate: COMMISSION_RATE,
            fantasy_close_delta_min: FANTASY_CLOSE_DELTA_MIN,
            fantasy_close_delta_max: FANTASY_CLOSE_DELTA_MAX,
            fantasy_sanitization: "finite-check-clamp-before-observe-and-execute-v2".to_owned(),
            validation_policy:
                "fixed-ticker-stratified-16-every-50-median-wealth-dd0.30-turnover0.50-v1"
                    .to_owned(),
            base_seed,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannerCheckpointMetadata {
    pub format_version: u32,
    pub architecture: String,
    pub world_model_lineage_sha256: String,
    pub world_model_weights_sha256: String,
    pub horizon: usize,
    pub context_bars: usize,
    pub model_dim: i64,
    pub layers: usize,
    pub heads: i64,
    pub latent_dim: i64,
    pub ohlc_dim: i64,
    pub portfolio_dim: i64,
    pub optimizer_steps: u64,
    pub training_contract: PlannerTrainingContract,
    pub kl_ema: f64,
    pub kl_lr_scale: f64,
    /// Cumulative training updates across all resumes; seeds the CSV `update`
    /// axis so it stays monotonic when appending after a restart.
    #[serde(default)]
    pub cumulative_updates: u64,
    /// SHA-256 of the planner weights file this metadata commits. Written last in
    /// the atomic save; a mismatch on load means a torn or corrupted checkpoint.
    #[serde(default)]
    pub weights_sha256: String,
    /// SHA-256 of the optimizer-state sidecar this metadata commits. Detects
    /// post-commit corruption or a mismatched sidecar before restoring moments.
    #[serde(default)]
    pub optimizer_sha256: String,
}

impl PlannerCheckpointMetadata {
    pub fn new(
        world_model_lineage_sha256: impl Into<String>,
        world_model_weights_sha256: impl Into<String>,
        horizon: usize,
        context_bars: usize,
        optimizer_steps: u64,
        cumulative_updates: u64,
        base_seed: u64,
        kl_ema: f64,
        kl_lr_scale: f64,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            architecture: ARCHITECTURE.to_owned(),
            world_model_lineage_sha256: world_model_lineage_sha256.into(),
            world_model_weights_sha256: world_model_weights_sha256.into(),
            horizon,
            context_bars,
            model_dim: PLANNER_MODEL_DIM,
            layers: PLANNER_LAYERS,
            heads: PLANNER_HEADS,
            latent_dim: PLANNER_LATENT_DIM,
            ohlc_dim: PLANNER_OHLC_DIM,
            portfolio_dim: PLANNER_PORTFOLIO_DIM,
            optimizer_steps,
            training_contract: PlannerTrainingContract::new(base_seed),
            kl_ema,
            kl_lr_scale,
            cumulative_updates,
            weights_sha256: String::new(),
            optimizer_sha256: String::new(),
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_reader(BufReader::new(File::open(path).with_context(|| {
            format!("failed opening planner metadata {}", path.display())
        })?))
        .with_context(|| format!("failed parsing planner metadata {}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let path = path.as_ref();
        ensure_parent_dir(path)?;
        serde_json::to_writer_pretty(
            BufWriter::new(
                File::create(path).with_context(|| {
                    format!("failed creating planner metadata {}", path.display())
                })?,
            ),
            self,
        )
        .with_context(|| format!("failed writing planner metadata {}", path.display()))
    }

    pub fn validate(
        &self,
        world_model_lineage_sha256: &str,
        expected_horizon: Option<usize>,
        expected_base_seed: Option<u64>,
    ) -> Result<()> {
        self.validate_schema()?;
        if self.world_model_lineage_sha256 != world_model_lineage_sha256 {
            bail!(
                "planner/world-model mismatch: planner requires {}, loaded {}",
                self.world_model_lineage_sha256,
                world_model_lineage_sha256
            );
        }
        if let Some(horizon) = expected_horizon {
            if self.horizon != horizon {
                bail!(
                    "planner horizon mismatch: checkpoint={}, requested={horizon}",
                    self.horizon
                );
            }
        }
        if let Some(seed) = expected_base_seed {
            if self.training_contract.base_seed != seed {
                bail!(
                    "planner base seed mismatch: checkpoint={}, requested={seed}",
                    self.training_contract.base_seed
                );
            }
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION || self.architecture != ARCHITECTURE {
            bail!("unsupported planner checkpoint metadata");
        }
        if self.model_dim != PLANNER_MODEL_DIM
            || self.layers != PLANNER_LAYERS
            || self.heads != PLANNER_HEADS
            || self.latent_dim != PLANNER_LATENT_DIM
            || self.ohlc_dim != PLANNER_OHLC_DIM
            || self.portfolio_dim != PLANNER_PORTFOLIO_DIM
        {
            bail!("planner checkpoint architecture dimensions are incompatible");
        }
        if self.horizon == 0
            || self.context_bars == 0
            || self.world_model_lineage_sha256.is_empty()
            || self.world_model_weights_sha256.is_empty()
            || self.weights_sha256.is_empty()
            || self.optimizer_sha256.is_empty()
            || self.training_contract
                != PlannerTrainingContract::new(self.training_contract.base_seed)
            || !self.kl_ema.is_finite()
            || self.kl_ema <= 0.0
            || !self.kl_lr_scale.is_finite()
            || self.kl_lr_scale <= 0.0
        {
            bail!("planner checkpoint metadata contains empty required fields");
        }
        Ok(())
    }
}

pub fn planner_metadata_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("metadata.json")
}

pub fn planner_optimizer_state_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("optimizer.ot")
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed creating planner checkpoint directory {}",
                parent.display()
            )
        })?;
    }
    Ok(())
}

/// Verify a committed checkpoint file against the SHA-256 recorded in metadata. A
/// non-empty `expected` that does not match is a hard error (corruption, a torn
/// write, or a mismatched sidecar); an empty `expected` (pre-sha checkpoint) skips.
fn verify_file_sha256(path: &Path, expected: &str, label: &str) -> Result<()> {
    if expected.is_empty() {
        return Ok(());
    }
    let actual = file_sha256(path)?;
    if actual != expected {
        bail!(
            "planner {label} {} does not match metadata sha256 (corrupted or torn checkpoint): metadata={expected}, actual={actual}",
            path.display()
        );
    }
    Ok(())
}

/// Verify an optimizer-state sidecar against the SHA-256 recorded in checkpoint
/// metadata before its moments are restored.
pub fn verify_optimizer_state(path: impl AsRef<Path>, expected: &str) -> Result<()> {
    verify_file_sha256(path.as_ref(), expected, "optimizer state")
}

fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Atomically persist planner weights, optimizer state, and metadata. Every file
/// is staged to a temp sibling then renamed into place, metadata last: metadata
/// carries the SHA-256 of the final weights, so a crash between renames leaves a
/// stale metadata whose hash no longer matches the weights and is rejected on
/// load rather than silently paired.
pub fn save_planner_checkpoint(
    var_store: &nn::VarStore,
    checkpoint: impl AsRef<Path>,
    metadata: &PlannerCheckpointMetadata,
    optimizer: &Muon,
) -> Result<()> {
    let checkpoint = checkpoint.as_ref();
    ensure_parent_dir(checkpoint)?;

    let metadata_path = planner_metadata_path(checkpoint);
    let optimizer_path = planner_optimizer_state_path(checkpoint);
    let weights_tmp = temp_sibling(checkpoint);
    let optimizer_tmp = temp_sibling(&optimizer_path);
    let metadata_tmp = temp_sibling(&metadata_path);

    var_store
        .save(&weights_tmp)
        .with_context(|| format!("failed saving planner weights {}", weights_tmp.display()))?;
    optimizer.save_state(&optimizer_tmp)?;

    let mut metadata = metadata.clone();
    metadata.weights_sha256 = file_sha256(&weights_tmp)?;
    metadata.optimizer_sha256 = file_sha256(&optimizer_tmp)?;
    metadata.save(&metadata_tmp)?;

    fs::rename(&weights_tmp, checkpoint)
        .with_context(|| format!("failed committing planner weights {}", checkpoint.display()))?;
    fs::rename(&optimizer_tmp, &optimizer_path).with_context(|| {
        format!(
            "failed committing planner optimizer state {}",
            optimizer_path.display()
        )
    })?;
    fs::rename(&metadata_tmp, &metadata_path).with_context(|| {
        format!(
            "failed committing planner metadata {}",
            metadata_path.display()
        )
    })?;
    Ok(())
}

pub fn load_planner_checkpoint(
    var_store: &mut nn::VarStore,
    checkpoint: impl AsRef<Path>,
    world_model_lineage_sha256: &str,
    expected_horizon: Option<usize>,
    expected_base_seed: Option<u64>,
) -> Result<PlannerCheckpointMetadata> {
    let checkpoint = checkpoint.as_ref();
    let metadata = PlannerCheckpointMetadata::load(planner_metadata_path(checkpoint))?;
    metadata.validate(
        world_model_lineage_sha256,
        expected_horizon,
        expected_base_seed,
    )?;
    verify_file_sha256(checkpoint, &metadata.weights_sha256, "checkpoint weights")?;
    var_store
        .load(checkpoint)
        .with_context(|| format!("failed loading planner weights {}", checkpoint.display()))?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_rejects_wrong_world_model() {
        let mut metadata = PlannerCheckpointMetadata::new(
            "lineage-a",
            "weights-a",
            100,
            6_000,
            10,
            10,
            7,
            0.035,
            1.0,
        );
        metadata.weights_sha256 = "planner-weights".to_owned();
        metadata.optimizer_sha256 = "planner-optimizer".to_owned();
        assert!(metadata.validate("lineage-b", Some(100), Some(7)).is_err());
        assert!(metadata.validate("lineage-a", Some(50), Some(7)).is_err());
        assert!(metadata.validate("lineage-a", Some(100), Some(8)).is_err());
        metadata.validate("lineage-a", Some(100), Some(7)).unwrap();
    }

    #[test]
    fn metadata_path_tracks_checkpoint() {
        assert_eq!(
            planner_metadata_path("weights/planner.ot"),
            PathBuf::from("weights/planner.metadata.json")
        );
        assert_eq!(
            planner_optimizer_state_path("weights/planner.ot"),
            PathBuf::from("weights/planner.optimizer.ot")
        );
    }

    #[test]
    fn metadata_rejects_training_contract_drift() {
        let mut metadata =
            PlannerCheckpointMetadata::new("lineage-a", "weights-a", 100, 128, 1, 1, 7, 0.035, 1.0);
        metadata.weights_sha256 = "planner-weights".to_owned();
        metadata.optimizer_sha256 = "planner-optimizer".to_owned();
        metadata.training_contract.gamma = 0.9;
        assert!(metadata.validate("lineage-a", Some(100), Some(7)).is_err());
    }

    #[test]
    fn checkpoint_roundtrip_restores_weights_and_optimizer_state() {
        use crate::torch::optim::muon::{Muon, MuonConfig};
        use crate::torch::train::optimizer_glue::named_trainable_variables;
        use tch::{Device, Kind, Tensor};

        let dir = std::env::temp_dir().join(format!(
            "planner-ckpt-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("planner.ot");

        let vs = tch::nn::VarStore::new(Device::Cpu);
        let proj = vs.root().sub("proj");
        let _w = proj.randn("weight", &[8, 4], 0.0, 1.0);
        let _b = proj.zeros("bias", &[8]);

        let named = named_trainable_variables(&vs);
        let mut optimizer = Muon::new_named(
            &named,
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        // Drive a step so momentum/second-moment/AdamW buffers become non-zero.
        let trainable: Vec<Tensor> = named.iter().map(|(_, t)| t.shallow_clone()).collect();
        let loss = Tensor::stack(
            &trainable
                .iter()
                .map(|t| t.square().sum(Kind::Float))
                .collect::<Vec<_>>(),
            0,
        )
        .sum(Kind::Float);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        let metadata =
            PlannerCheckpointMetadata::new("lineage-a", "weights-a", 100, 128, 1, 1, 7, 0.035, 1.0);
        save_planner_checkpoint(&vs, &checkpoint, &metadata, &optimizer).unwrap();

        // Optimizer sidecar round-trips exactly.
        let mut restored = Muon::new_named(
            &named_trainable_variables(&vs),
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        restored
            .load_state(planner_optimizer_state_path(&checkpoint))
            .unwrap();
        assert_eq!(restored.state_bytes(), optimizer.state_bytes());

        // Weights + metadata load and the sha validates.
        let mut loaded_vs = tch::nn::VarStore::new(Device::Cpu);
        {
            let lproj = loaded_vs.root().sub("proj");
            let _ = lproj.randn("weight", &[8, 4], 0.0, 1.0);
            let _ = lproj.zeros("bias", &[8]);
        }
        let loaded =
            load_planner_checkpoint(&mut loaded_vs, &checkpoint, "lineage-a", Some(100), Some(7))
                .unwrap();
        assert_eq!(loaded.cumulative_updates, 1);
        assert!(!loaded.weights_sha256.is_empty());
        assert!(!loaded.optimizer_sha256.is_empty());

        // Optimizer sidecar sha validates intact and rejects corruption.
        let optimizer_state = planner_optimizer_state_path(&checkpoint);
        verify_optimizer_state(&optimizer_state, &loaded.optimizer_sha256).unwrap();
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            let mut f = OpenOptions::new()
                .append(true)
                .open(&optimizer_state)
                .unwrap();
            f.write_all(b"corruption").unwrap();
        }
        assert!(verify_optimizer_state(&optimizer_state, &loaded.optimizer_sha256).is_err());

        // A corrupted weights file is rejected against the metadata sha.
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&checkpoint).unwrap();
            f.write_all(b"corruption").unwrap();
        }
        assert!(load_planner_checkpoint(
            &mut loaded_vs,
            &checkpoint,
            "lineage-a",
            Some(100),
            Some(7),
        )
        .is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
