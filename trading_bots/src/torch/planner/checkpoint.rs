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
    losses::{
        PLANNER_AUX_RETURN_COEF, POSITIVE_WEIGHT, REVERSE_KL_COEFFICIENT, VALUE_LOSS_COEFFICIENT,
    },
    portfolio::PLANNER_REWARD_SCALE,
    PLANNER_BELIEF_DIM, PLANNER_HEADS, PLANNER_LATENT_DIM, PLANNER_LAYERS, PLANNER_MODEL_DIM,
    PLANNER_PORTFOLIO_DIM,
};

const FORMAT_VERSION: u32 = 6;
const ARCHITECTURE: &str = "world-model-planner-disjoint-policy-critic-v6";
const INPUT_NORMALIZATION: &str = "belief-token-latent-rms-normalized-with-log-scale-v1";
const ROLLOUT_DATA: &str = "frozen-wm-observation-real-next-price-after-costs-v1";
pub(crate) const OPTIMIZATION_EPOCHS: usize = 3;
pub(crate) const TARGET_KL: f64 = 0.035;
pub(crate) const KL_CONTROLLER_HALF_LIFE: f64 = 50.0;
pub(crate) const KL_MIN_LR_SCALE: f64 = 0.01;
pub(crate) const KL_MAX_LR_SCALE: f64 = 10.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannerTrainingContract {
    pub reward_scale: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub rollout_data: String,
    pub pmpo_positive_weight: f64,
    pub pmpo_reverse_kl_coefficient: f64,
    pub value_loss_coefficient: f64,
    pub aux_return_coefficient: f64,
    pub optimization_epochs: usize,
    pub target_kl: f64,
    pub kl_controller_half_life: f64,
    pub kl_min_lr_scale: f64,
    pub kl_max_lr_scale: f64,
    pub kl_lr_routing: String,
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
    pub validation_policy: String,
    pub base_seed: u64,
}

impl PlannerTrainingContract {
    fn new(base_seed: u64) -> Self {
        Self {
            reward_scale: PLANNER_REWARD_SCALE,
            gamma: PLANNER_GAMMA,
            gae_lambda: PLANNER_GAE_LAMBDA,
            rollout_data: ROLLOUT_DATA.to_owned(),
            pmpo_positive_weight: POSITIVE_WEIGHT,
            pmpo_reverse_kl_coefficient: REVERSE_KL_COEFFICIENT,
            value_loss_coefficient: VALUE_LOSS_COEFFICIENT,
            aux_return_coefficient: PLANNER_AUX_RETURN_COEF,
            optimization_epochs: OPTIMIZATION_EPOCHS,
            target_kl: TARGET_KL,
            kl_controller_half_life: KL_CONTROLLER_HALF_LIFE,
            kl_min_lr_scale: KL_MIN_LR_SCALE,
            kl_max_lr_scale: KL_MAX_LR_SCALE,
            kl_lr_routing: "adaptive-entire-policy-branch-disjoint-critic-v2".to_owned(),
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
            optimizer_routing: "separate-policy-and-critic-normuon-optimizers-v5".to_owned(),
            action_threshold: ACTION_THRESHOLD,
            commission_rate: COMMISSION_RATE,
            validation_policy:
                "fixed-ticker-time-stratified-64-every-50-paired-outperformance-profitable-dd0.30-turnover0.50-v3"
                    .to_owned(),
            base_seed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannerSelectedValidation {
    pub update: u64,
    pub median_wealth_ratio: f64,
    pub median_buy_and_hold_wealth_ratio: f64,
    pub mean_outperformance_ratio: f64,
    pub median_outperformance_ratio: f64,
    pub outperformance_fraction: f64,
    pub mean_max_drawdown: f64,
    pub mean_turnover: f64,
}

impl PlannerSelectedValidation {
    fn is_valid(self, cumulative_updates: u64) -> bool {
        self.update == cumulative_updates
            && self.median_wealth_ratio.is_finite()
            && self.median_wealth_ratio > 0.0
            && self.median_buy_and_hold_wealth_ratio.is_finite()
            && self.median_buy_and_hold_wealth_ratio > 0.0
            && self.mean_outperformance_ratio.is_finite()
            && self.median_outperformance_ratio.is_finite()
            && self.outperformance_fraction.is_finite()
            && (0.0..=1.0).contains(&self.outperformance_fraction)
            && self.mean_max_drawdown.is_finite()
            && (0.0..=1.0).contains(&self.mean_max_drawdown)
            && self.mean_turnover.is_finite()
            && self.mean_turnover >= 0.0
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
    pub belief_dim: i64,
    pub portfolio_dim: i64,
    pub actor_optimizer_steps: u64,
    pub critic_optimizer_steps: u64,
    pub run_lineage_id: String,
    pub training_contract: PlannerTrainingContract,
    pub kl_ema: f64,
    pub kl_lr_scale: f64,
    /// Cumulative training updates across all resumes; keeps checkpoint and
    /// native report generations monotonic after a restart.
    #[serde(default)]
    pub cumulative_updates: u64,
    /// SHA-256 of the planner weights file this metadata commits. Written last in
    /// the atomic save; a mismatch on load means a torn or corrupted checkpoint.
    #[serde(default)]
    pub weights_sha256: String,
    /// SHA-256 hashes of the optimizer-state sidecars this metadata commits.
    /// Detect post-commit corruption or mismatched sidecars before restoring moments.
    pub actor_optimizer_sha256: String,
    pub critic_optimizer_sha256: String,
    pub actor_initialized_adamw: Vec<String>,
    pub critic_initialized_adamw: Vec<String>,
    /// Present only on immutable model-selection checkpoints. Keeping the score
    /// in the same metadata commit as the weight hashes makes selection atomic.
    #[serde(default)]
    pub selected_validation: Option<PlannerSelectedValidation>,
}

impl PlannerCheckpointMetadata {
    pub fn new(
        world_model_lineage_sha256: impl Into<String>,
        world_model_weights_sha256: impl Into<String>,
        horizon: usize,
        context_bars: usize,
        actor_optimizer_steps: u64,
        critic_optimizer_steps: u64,
        cumulative_updates: u64,
        run_lineage_id: impl Into<String>,
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
            belief_dim: PLANNER_BELIEF_DIM,
            portfolio_dim: PLANNER_PORTFOLIO_DIM,
            actor_optimizer_steps,
            critic_optimizer_steps,
            run_lineage_id: run_lineage_id.into(),
            training_contract: PlannerTrainingContract::new(base_seed),
            kl_ema,
            kl_lr_scale,
            cumulative_updates,
            weights_sha256: String::new(),
            actor_optimizer_sha256: String::new(),
            critic_optimizer_sha256: String::new(),
            actor_initialized_adamw: Vec::new(),
            critic_initialized_adamw: Vec::new(),
            selected_validation: None,
        }
    }

    pub fn with_selected_validation(mut self, selected: PlannerSelectedValidation) -> Self {
        self.selected_validation = Some(selected);
        self
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
            || self.belief_dim != PLANNER_BELIEF_DIM
            || self.portfolio_dim != PLANNER_PORTFOLIO_DIM
        {
            bail!("planner checkpoint architecture dimensions are incompatible");
        }
        if self.horizon == 0
            || self.context_bars == 0
            || self.world_model_lineage_sha256.is_empty()
            || self.world_model_weights_sha256.is_empty()
            || self.weights_sha256.is_empty()
            || self.actor_optimizer_sha256.is_empty()
            || self.critic_optimizer_sha256.is_empty()
            || self.run_lineage_id.is_empty()
            || self
                .selected_validation
                .is_some_and(|selected| !selected.is_valid(self.cumulative_updates))
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

pub fn planner_actor_optimizer_state_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("actor_optimizer.ot")
}

pub fn planner_critic_optimizer_state_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("critic_optimizer.ot")
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

pub fn load_committed_planner_metadata(
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
    verify_optimizer_state(
        planner_actor_optimizer_state_path(checkpoint),
        &metadata.actor_optimizer_sha256,
    )?;
    verify_optimizer_state(
        planner_critic_optimizer_state_path(checkpoint),
        &metadata.critic_optimizer_sha256,
    )?;
    Ok(metadata)
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
    actor_optimizer: &Muon,
    critic_optimizer: &Muon,
) -> Result<PlannerCheckpointMetadata> {
    let checkpoint = checkpoint.as_ref();
    ensure_parent_dir(checkpoint)?;

    let metadata_path = planner_metadata_path(checkpoint);
    let actor_optimizer_path = planner_actor_optimizer_state_path(checkpoint);
    let critic_optimizer_path = planner_critic_optimizer_state_path(checkpoint);
    let weights_tmp = temp_sibling(checkpoint);
    let actor_optimizer_tmp = temp_sibling(&actor_optimizer_path);
    let critic_optimizer_tmp = temp_sibling(&critic_optimizer_path);
    let metadata_tmp = temp_sibling(&metadata_path);

    var_store
        .save(&weights_tmp)
        .with_context(|| format!("failed saving planner weights {}", weights_tmp.display()))?;
    actor_optimizer.save_state(&actor_optimizer_tmp)?;
    critic_optimizer.save_state(&critic_optimizer_tmp)?;

    let mut metadata = metadata.clone();
    metadata.actor_initialized_adamw = actor_optimizer.initialized_adamw_names();
    metadata.critic_initialized_adamw = critic_optimizer.initialized_adamw_names();
    let actor_steps = i64::try_from(metadata.actor_optimizer_steps)
        .context("planner actor optimizer step count exceeds i64")?;
    let critic_steps = i64::try_from(metadata.critic_optimizer_steps)
        .context("planner critic optimizer step count exceeds i64")?;
    actor_optimizer.validate_state_strict(
        &actor_optimizer_tmp,
        &metadata.actor_initialized_adamw,
        actor_steps,
    )?;
    critic_optimizer.validate_state_strict(
        &critic_optimizer_tmp,
        &metadata.critic_initialized_adamw,
        critic_steps,
    )?;
    metadata.weights_sha256 = file_sha256(&weights_tmp)?;
    metadata.actor_optimizer_sha256 = file_sha256(&actor_optimizer_tmp)?;
    metadata.critic_optimizer_sha256 = file_sha256(&critic_optimizer_tmp)?;
    metadata.save(&metadata_tmp)?;

    File::open(&weights_tmp)?.sync_all()?;
    File::open(&actor_optimizer_tmp)?.sync_all()?;
    File::open(&critic_optimizer_tmp)?.sync_all()?;
    File::open(&metadata_tmp)?.sync_all()?;

    fs::rename(&weights_tmp, checkpoint)
        .with_context(|| format!("failed committing planner weights {}", checkpoint.display()))?;
    fs::rename(&actor_optimizer_tmp, &actor_optimizer_path).with_context(|| {
        format!(
            "failed committing planner actor optimizer state {}",
            actor_optimizer_path.display()
        )
    })?;
    fs::rename(&critic_optimizer_tmp, &critic_optimizer_path).with_context(|| {
        format!(
            "failed committing planner critic optimizer state {}",
            critic_optimizer_path.display()
        )
    })?;
    fs::rename(&metadata_tmp, &metadata_path).with_context(|| {
        format!(
            "failed committing planner metadata {}",
            metadata_path.display()
        )
    })?;
    if let Some(parent) = checkpoint
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        File::open(parent)?.sync_all()?;
    }
    Ok(metadata)
}

pub fn load_planner_checkpoint(
    var_store: &mut nn::VarStore,
    checkpoint: impl AsRef<Path>,
    world_model_lineage_sha256: &str,
    expected_horizon: Option<usize>,
    expected_base_seed: Option<u64>,
) -> Result<PlannerCheckpointMetadata> {
    let checkpoint = checkpoint.as_ref();
    let metadata = load_committed_planner_metadata(
        checkpoint,
        world_model_lineage_sha256,
        expected_horizon,
        expected_base_seed,
    )?;
    var_store
        .load(checkpoint)
        .with_context(|| format!("failed loading planner weights {}", checkpoint.display()))?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::test_rng;

    #[test]
    fn metadata_rejects_wrong_world_model() {
        let mut metadata = PlannerCheckpointMetadata::new(
            "lineage-a",
            "weights-a",
            100,
            6_000,
            10,
            10,
            10,
            "run-a",
            7,
            0.035,
            1.0,
        );
        metadata.weights_sha256 = "planner-weights".to_owned();
        metadata.actor_optimizer_sha256 = "planner-actor-optimizer".to_owned();
        metadata.critic_optimizer_sha256 = "planner-critic-optimizer".to_owned();
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
            planner_actor_optimizer_state_path("weights/planner.ot"),
            PathBuf::from("weights/planner.actor_optimizer.ot")
        );
        assert_eq!(
            planner_critic_optimizer_state_path("weights/planner.ot"),
            PathBuf::from("weights/planner.critic_optimizer.ot")
        );
    }

    #[test]
    fn metadata_rejects_training_contract_drift() {
        let mut metadata = PlannerCheckpointMetadata::new(
            "lineage-a",
            "weights-a",
            100,
            128,
            1,
            1,
            1,
            "run-a",
            7,
            0.035,
            1.0,
        );
        metadata.weights_sha256 = "planner-weights".to_owned();
        metadata.actor_optimizer_sha256 = "planner-actor-optimizer".to_owned();
        metadata.critic_optimizer_sha256 = "planner-critic-optimizer".to_owned();
        metadata.training_contract.gamma = 0.9;
        assert!(metadata.validate("lineage-a", Some(100), Some(7)).is_err());
    }

    #[test]
    fn metadata_rejects_pre_actor_only_kl_lr_routing_format() {
        let mut metadata = PlannerCheckpointMetadata::new(
            "lineage-a",
            "weights-a",
            100,
            128,
            1,
            1,
            1,
            "run-a",
            7,
            0.035,
            1.0,
        );
        metadata.weights_sha256 = "planner-weights".to_owned();
        metadata.actor_optimizer_sha256 = "planner-actor-optimizer".to_owned();
        metadata.critic_optimizer_sha256 = "planner-critic-optimizer".to_owned();
        metadata.format_version = 4;

        assert!(metadata.validate("lineage-a", Some(100), Some(7)).is_err());
    }

    #[test]
    fn training_contract_records_disjoint_optimizer_routing() {
        let contract = PlannerTrainingContract::new(7);
        assert_eq!(
            contract.kl_lr_routing,
            "adaptive-entire-policy-branch-disjoint-critic-v2"
        );
        assert_eq!(
            contract.optimizer_routing,
            "separate-policy-and-critic-normuon-optimizers-v5"
        );
    }

    #[test]
    fn selected_validation_must_match_the_committed_update() {
        let mut metadata = PlannerCheckpointMetadata::new(
            "lineage-a",
            "weights-a",
            100,
            128,
            1,
            50,
            50,
            "run-a",
            7,
            0.035,
            1.0,
        )
        .with_selected_validation(PlannerSelectedValidation {
            update: 49,
            median_wealth_ratio: 1.1,
            median_buy_and_hold_wealth_ratio: 1.0,
            mean_outperformance_ratio: 0.1,
            median_outperformance_ratio: 0.1,
            outperformance_fraction: 0.75,
            mean_max_drawdown: 0.1,
            mean_turnover: 0.2,
        });
        metadata.weights_sha256 = "planner-weights".to_owned();
        metadata.actor_optimizer_sha256 = "planner-actor-optimizer".to_owned();
        metadata.critic_optimizer_sha256 = "planner-critic-optimizer".to_owned();
        assert!(metadata.validate("lineage-a", Some(100), Some(7)).is_err());
        metadata.selected_validation.as_mut().unwrap().update = 50;
        metadata.validate("lineage-a", Some(100), Some(7)).unwrap();
    }

    #[test]
    fn checkpoint_roundtrip_restores_weights_and_optimizer_state() {
        use crate::torch::optim::muon::{Muon, MuonConfig, StepKind};
        use crate::torch::train::optimizer_glue::named_trainable_variables;
        use tch::{Device, Kind};

        let _torch_rng_guard = test_rng::shared();
        let dir = std::env::temp_dir().join(format!(
            "planner-ckpt-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("planner.ot");

        let vs = tch::nn::VarStore::new(Device::Cpu);
        let _actor = vs.root().sub("policy").randn("weight", &[8, 4], 0.0, 1.0);
        let _critic = vs.root().sub("critic").randn("weight", &[8, 4], 0.0, 1.0);
        let named = named_trainable_variables(&vs);
        let actor_named = named
            .iter()
            .filter(|(name, _)| name.starts_with("policy."))
            .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
            .collect::<Vec<_>>();
        let critic_named = named
            .iter()
            .filter(|(name, _)| name.starts_with("critic."))
            .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
            .collect::<Vec<_>>();
        let mut actor_optimizer = Muon::new_named(
            &actor_named,
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        let mut critic_optimizer = Muon::new_named(
            &critic_named,
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        actor_named[0].1.square().sum(Kind::Float).backward();
        actor_optimizer.step(StepKind::Primary);
        actor_optimizer.zero_grad();
        for _ in 0..2 {
            critic_named[0].1.square().sum(Kind::Float).backward();
            critic_optimizer.step(StepKind::Primary);
            critic_optimizer.zero_grad();
        }

        let metadata = PlannerCheckpointMetadata::new(
            "lineage-a",
            "weights-a",
            100,
            128,
            1,
            2,
            1,
            "run-a",
            7,
            0.035,
            1.0,
        );
        save_planner_checkpoint(
            &vs,
            &checkpoint,
            &metadata,
            &actor_optimizer,
            &critic_optimizer,
        )
        .unwrap();

        let mut restored_actor = Muon::new_named(
            &actor_named,
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        let mut restored_critic = Muon::new_named(
            &critic_named,
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        restored_actor
            .load_state(planner_actor_optimizer_state_path(&checkpoint))
            .unwrap();
        restored_critic
            .load_state(planner_critic_optimizer_state_path(&checkpoint))
            .unwrap();
        assert_eq!(restored_actor.state_bytes(), actor_optimizer.state_bytes());
        assert_eq!(
            restored_critic.state_bytes(),
            critic_optimizer.state_bytes()
        );

        // Weights + metadata load and the sha validates.
        let mut loaded_vs = tch::nn::VarStore::new(Device::Cpu);
        {
            let _ = loaded_vs
                .root()
                .sub("policy")
                .randn("weight", &[8, 4], 0.0, 1.0);
            let _ = loaded_vs
                .root()
                .sub("critic")
                .randn("weight", &[8, 4], 0.0, 1.0);
        }
        let loaded =
            load_planner_checkpoint(&mut loaded_vs, &checkpoint, "lineage-a", Some(100), Some(7))
                .unwrap();
        assert_eq!(loaded.cumulative_updates, 1);
        assert_eq!(loaded.actor_optimizer_steps, 1);
        assert_eq!(loaded.critic_optimizer_steps, 2);
        assert!(!loaded.weights_sha256.is_empty());
        assert!(!loaded.actor_optimizer_sha256.is_empty());
        assert!(!loaded.critic_optimizer_sha256.is_empty());
        assert_ne!(
            loaded.actor_optimizer_sha256,
            loaded.critic_optimizer_sha256
        );
        assert_eq!(
            loaded.actor_initialized_adamw,
            actor_optimizer.initialized_adamw_names()
        );
        assert_eq!(
            loaded.critic_initialized_adamw,
            critic_optimizer.initialized_adamw_names()
        );

        let mismatched_checkpoint = dir.join("planner-mismatched.ot");
        let mut mismatched_metadata = metadata.clone();
        mismatched_metadata.actor_optimizer_steps += 1;
        assert!(save_planner_checkpoint(
            &vs,
            &mismatched_checkpoint,
            &mismatched_metadata,
            &actor_optimizer,
            &critic_optimizer,
        )
        .is_err());
        assert!(!mismatched_checkpoint.exists());

        // Optimizer sidecar sha validates intact and rejects corruption.
        let optimizer_state = planner_actor_optimizer_state_path(&checkpoint);
        verify_optimizer_state(&optimizer_state, &loaded.actor_optimizer_sha256).unwrap();
        verify_optimizer_state(
            planner_critic_optimizer_state_path(&checkpoint),
            &loaded.critic_optimizer_sha256,
        )
        .unwrap();
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            let mut f = OpenOptions::new()
                .append(true)
                .open(&optimizer_state)
                .unwrap();
            f.write_all(b"corruption").unwrap();
        }
        assert!(verify_optimizer_state(&optimizer_state, &loaded.actor_optimizer_sha256).is_err());

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
