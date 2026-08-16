use anyhow::{bail, ensure, Context, Result};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tch::{autocast, nn, Device, Kind, Tensor};

use crate::torch::action_space::BETA_SAMPLE_EPS;
use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::env::{CpuStepBatch, ValidatedVecEnvSnapshot, VecEnv, VecEnvSnapshot};
use crate::torch::hashing::file_sha256;
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{
    ModelOutput, ModelVariant, StreamState, TradingModel, TradingModelConfig,
};
use crate::torch::optim::muon::{newton_schulz_polynomial_bits, Muon, MuonConfig};
use crate::torch::value::hl_gauss::{
    HlGaussBins, DIRECT_SIGMA_RATIO, NUM_BINS, SYMLOG_SUPPORT_MAX, SYMLOG_SUPPORT_MIN,
};
use shared::{paths::RUNS_PATH, run_dir::RunDir};

use super::config::{
    CLIP_EPS_HIGH, CLIP_EPS_LOW, CRITIC_PRETRAIN_EPISODES, ENTROPY_COEF, GAE_GAMMA, GAE_LAMBDA,
    KL_LR_EMA_HALF_LIFE, KL_LR_MAX_SCALE, KL_LR_MIN_SCALE, KL_LR_TARGET, KL_STOP_MULTIPLIER,
    LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM, MUON_MOMENTUM_WARMUP_START,
    MUON_MOMENTUM_WARMUP_STEPS, OPTIM_EPOCHS, PMPO_KL_COEF, PMPO_POS_TO_NEG_WEIGHT,
    POLICY_OBJECTIVE, RET_PERC_FLOOR, RET_PERC_HI, RET_PERC_LO, TARGET_KL, USE_MUON,
    VALUE_LOSS_COEF,
};
use super::geometry::{minibatch_samples_from_total, rollout_geometry, RolloutGeometry};
use super::optimizer_glue::{
    apply_lr_scale, grad_clip_groups, named_trainable_variables, GradClipGroups, KlLrController,
    KlLrControllerState, ACTOR_GRAD_CLIP_PATTERNS, CRITIC_GRAD_CLIP_PATTERNS, KL_LR_SCALE_EXPONENT,
};
use super::update::PpoUpdateCudaGraph;

pub(super) struct RolloutData {
    pub(super) reset_layout_batches_cpu: Vec<Tensor>,
    pub(super) reset_layout_count: i64,
    pub(super) reset_slots_host: Vec<i64>,
}

pub(super) struct AdvantageData {
    pub(super) advantages: Tensor,
    pub(super) returns: Tensor,
    pub(super) adv_stats: Tensor,
    pub(super) adv_stats_shaped: Tensor,
    pub(super) reset_layout_bank_cpu: Tensor,
    pub(super) reset_slots_by_chunk: Tensor,
    pub(super) reset_chunks_have_slots: Vec<bool>,
    pub(super) chunk_batch_size: i64,
    pub(super) reset_layout_count: i64,
}

pub(super) struct UpdateMetrics {
    pub(super) total_policy_loss_weighted: Tensor,
    pub(super) total_value_loss_weighted: Tensor,
    pub(super) total_clip_gap_weighted: Tensor,
    pub(super) actor_grad_norm_sum: Tensor,
    pub(super) critic_grad_norm_sum: Tensor,
    pub(super) total_sample_count: i64,
    pub(super) grad_norm_count: i64,
    pub(super) total_clip_violations: Tensor,
    pub(super) total_ratio_samples: i64,
    pub(super) total_entropy_weighted: Tensor,
    pub(super) entropy_min: Tensor,
    pub(super) entropy_max: Tensor,
    pub(super) mean_epoch_approx_kl: f64,
    pub(super) lr_scale: f64,
    pub(super) kl_lr_scale_next: f64,
    pub(super) kl_lr_ema: f64,
    pub(super) kl_lr_signal: f64,
}

pub(super) struct Trainer {
    pub(super) vs: nn::VarStore,
    pub(super) trading_model: TradingModel,
    pub(super) trainable_vars: Vec<Tensor>,
    pub(super) named_trainable_vars: Vec<(String, Tensor)>,
    pub(super) grad_clip_groups: GradClipGroups,
    pub(super) opt: Muon,
    pub(super) optimizer_step: i64,
    pub(super) muon_momentum_step: i64,
    pub(super) kl_lr_controller: KlLrController,
    pub(super) env: VecEnv,
    pub(super) device: Device,
    pub(super) rollout: RolloutGeometry,
    pub(super) hl_gauss: HlGaussBins,
    pub(super) run_dir: RunDir,
    pub(super) start_update: usize,
    // Geometry-derived constants
    pub(super) rollout_steps: i64,
    pub(super) total_chunks: i64,
    pub(super) raw_pd_dim: i64,
    pub(super) pd_dim: i64,
    pub(super) so_dim: i64,
    pub(super) replay_obs_kind: Kind,
    // Rollout storage buffers
    pub(super) s_chunk_start_layouts: Tensor,
    pub(super) s_static_obs: Tensor,
    pub(super) s_step_deltas: Tensor,
    pub(super) s_actions: Tensor,
    pub(super) s_old_log_probs: Tensor,
    pub(super) s_old_alphas: Tensor,
    pub(super) s_old_betas: Tensor,
    pub(super) s_rewards: Tensor,
    pub(super) s_dones: Tensor,
    pub(super) s_values: Tensor,
    // Per-step working tensors
    pub(super) obs_static: Tensor,
    pub(super) step_deltas: Tensor,
    pub(super) stream_state: StreamState,
    pub(super) streamed_output: Option<ModelOutput>,
    pub(super) step_reward_per_ticker: Tensor,
    pub(super) step_is_done: Tensor,
    pub(super) cpu_step_batch: CpuStepBatch,
    pub(super) action_host_view: Tensor,
    pub(super) reset_env_indices_host: Vec<i64>,
    pub(super) ticker_offsets: Tensor,
    pub(super) ppo_update_graph: Option<PpoUpdateCudaGraph>,
    pub(super) seed: u64,
    contract: PpoTrainingContract,
}

const PPO_CHECKPOINT_FORMAT_VERSION: u32 = 5;
const PPO_CHECKPOINT_PHASE: &str = "ready-for-rollout";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct PpoTrainingContract {
    model_variant: String,
    device: String,
    rollout: RolloutGeometry,
    minibatch_samples: i64,
    optim_epochs: i64,
    ticker_count: i64,
    action_count: i64,
    price_context: usize,
    static_observations: usize,
    objective: String,
    rng_algorithm: String,
    libtorch_runtime: String,
    cuda_graphs_requested: bool,
    torch_num_threads: i32,
    torch_num_interop_threads: i32,
    environment_semantics: String,
    optimizer: PpoOptimizerContract,
    objective_semantics: PpoObjectiveContract,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct PpoObjectiveContract {
    gae_gamma: u64,
    gae_lambda: u64,
    clip_eps_low: u64,
    clip_eps_high: u64,
    target_kl: u64,
    kl_stop_multiplier: u64,
    value_loss_coef: u64,
    entropy_coef: u64,
    max_grad_norm: u64,
    kl_lr_target: u64,
    kl_lr_ema_half_life: u64,
    kl_lr_min_scale: u64,
    kl_lr_max_scale: u64,
    kl_lr_scale_exponent: u64,
    critic_pretrain_episodes: usize,
    policy_objective: String,
    pmpo_pos_to_neg_weight: u64,
    pmpo_kl_coef: u64,
    return_percentile_low: u64,
    return_percentile_high: u64,
    return_percentile_floor: u64,
    beta_sample_epsilon: u64,
    value_bins: i64,
    value_support_min: u64,
    value_support_max: u64,
    value_sigma_ratio: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct PpoOptimizerContract {
    lr: u64,
    use_muon_for_2d: bool,
    momentum_start: u64,
    momentum_end: u64,
    momentum_warmup_steps: i64,
    nesterov: bool,
    beta2: u64,
    weight_decay: u64,
    adamw_lr: u64,
    adamw_beta1: u64,
    adamw_beta2: u64,
    adamw_eps: u64,
    adamw_weight_decay: u64,
    adamw_no_weight_decay_names: Vec<String>,
    ns_steps: usize,
    ns_polynomial: [u64; 3],
    force_adamw_names: Vec<String>,
    muon_allowlist: Vec<String>,
    per_attention_head_ortho: bool,
    per_attention_output_head_ortho: bool,
    attention_head_dim: i64,
    cross_attention_head_dim: i64,
    actor_grad_clip_patterns: Vec<String>,
    critic_grad_clip_patterns: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PpoCheckpointMetadata {
    format_version: u32,
    phase: String,
    next_update: usize,
    seed: u64,
    contract: PpoTrainingContract,
    optimizer_step: i64,
    muon_momentum_step: i64,
    weights_sha256: String,
    optimizer_sha256: String,
    trajectory_sha256: String,
    initialized_adamw: Vec<String>,
    kl_lr_controller: KlLrControllerState,
}

struct ValidatedPpoCheckpoint {
    metadata: PpoCheckpointMetadata,
    trajectory: ValidatedVecEnvSnapshot,
}

struct ParsedPpoCheckpoint {
    metadata: PpoCheckpointMetadata,
    trajectory: VecEnvSnapshot,
}

fn ppo_metadata_path(weights_path: &Path) -> PathBuf {
    weights_path.with_extension("resume.json")
}

fn ppo_optimizer_path(weights_path: &Path) -> PathBuf {
    weights_path.with_extension("optimizer.ot")
}

fn ppo_trajectory_path(weights_path: &Path) -> PathBuf {
    weights_path.with_extension("trajectory.postcard")
}

fn temp_sibling(path: &Path, transaction_id: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{transaction_id}"));
    path.with_file_name(name)
}

fn is_ppo_checkpoint_path(path: &Path) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_prefix("ppo_ep"))
        .is_some_and(|episode| episode.parse::<usize>().is_ok())
}

fn should_resume_from_path(path: &Path) -> bool {
    is_ppo_checkpoint_path(path) || ppo_metadata_path(path).exists()
}

fn completed_update_for_resume(next_update: usize) -> Result<usize> {
    next_update
        .checked_sub(1)
        .context("PPO checkpoint does not follow a completed update")
}

fn preflight_optimizer_sidecar(path: &Path, metadata: &PpoCheckpointMetadata) -> Result<()> {
    let loaded = Tensor::load_multi_with_device(path, Device::Cpu)
        .with_context(|| format!("failed reading optimizer state {}", path.display()))?
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    let global_step = loaded
        .get("__muon_step_count__")
        .context("optimizer state is missing its global step")?;
    ensure!(
        global_step.numel() == 1 && global_step.int64_value(&[]) == metadata.optimizer_step,
        "optimizer step disagrees with checkpoint metadata"
    );
    for name in &metadata.initialized_adamw {
        let m = loaded
            .get(&format!("{name}.__adamw_m"))
            .with_context(|| format!("optimizer state is missing AdamW m for {name}"))?;
        let v = loaded
            .get(&format!("{name}.__adamw_v"))
            .with_context(|| format!("optimizer state is missing AdamW v for {name}"))?;
        let step = loaded
            .get(&format!("{name}.__adamw_step_count"))
            .with_context(|| format!("optimizer state is missing AdamW step for {name}"))?;
        ensure!(
            m.size() == v.size(),
            "AdamW moment shape mismatch for {name}"
        );
        ensure!(step.numel() == 1, "AdamW step is not scalar for {name}");
    }
    for (name, tensor) in &loaded {
        let (counterpart, same_shape) = if let Some(base) = name.strip_suffix(".__momentum") {
            (Some(format!("{base}.__second_momentum")), false)
        } else if let Some(base) = name.strip_suffix(".__second_momentum") {
            (Some(format!("{base}.__momentum")), false)
        } else if let Some(base) = name.strip_suffix(".__adamw_m") {
            (Some(format!("{base}.__adamw_v")), true)
        } else if let Some(base) = name.strip_suffix(".__adamw_v") {
            (Some(format!("{base}.__adamw_m")), true)
        } else {
            (None, false)
        };
        if let Some(counterpart) = counterpart {
            let other = loaded
                .get(&counterpart)
                .with_context(|| format!("optimizer tensor {name} has no {counterpart}"))?;
            ensure!(
                !same_shape || tensor.size() == other.size(),
                "optimizer pair shape mismatch"
            );
        }
        if tensor.is_floating_point() {
            ensure!(
                tensor.isfinite().all().int64_value(&[]) != 0,
                "optimizer tensor {name} contains non-finite state"
            );
        }
    }
    Ok(())
}

fn training_contract(
    model_variant: ModelVariant,
    device: Device,
    rollout: RolloutGeometry,
    torch_num_threads: i32,
    torch_num_interop_threads: i32,
    optimizer: &MuonConfig,
) -> PpoTrainingContract {
    let libtorch_runtime = include_str!("../../../../.pytorch-version").trim();
    PpoTrainingContract {
        model_variant: model_variant.as_str().to_owned(),
        device: format!("{device:?}"),
        rollout,
        minibatch_samples: minibatch_samples_from_total(rollout.total_samples, rollout.nprocs),
        optim_epochs: OPTIM_EPOCHS,
        ticker_count: TICKERS_COUNT,
        action_count: ACTION_COUNT,
        price_context: PRICE_DELTAS_PER_TICKER,
        static_observations: STATIC_OBSERVATIONS,
        objective: "ppo".to_owned(),
        rng_algorithm: format!(
            "libtorch::standard_gamma/{libtorch_runtime};rand_chacha::ChaCha12Rng/0.9"
        ),
        libtorch_runtime: libtorch_runtime.to_owned(),
        cuda_graphs_requested: device.is_cuda()
            && env::var("PPO_CUDA_GRAPHS").ok().as_deref() != Some("0"),
        torch_num_threads,
        torch_num_interop_threads,
        environment_semantics: "causal-env-v2".to_owned(),
        optimizer: PpoOptimizerContract {
            lr: optimizer.lr.to_bits(),
            use_muon_for_2d: optimizer.use_muon_for_2d,
            momentum_start: optimizer.momentum.to_bits(),
            momentum_end: MUON_MOMENTUM.to_bits(),
            momentum_warmup_steps: MUON_MOMENTUM_WARMUP_STEPS,
            nesterov: optimizer.nesterov,
            beta2: optimizer.beta2.to_bits(),
            weight_decay: optimizer.weight_decay.to_bits(),
            adamw_lr: optimizer.adamw_lr.to_bits(),
            adamw_beta1: optimizer.adamw_betas.0.to_bits(),
            adamw_beta2: optimizer.adamw_betas.1.to_bits(),
            adamw_eps: optimizer.adamw_eps.to_bits(),
            adamw_weight_decay: optimizer.adamw_wd.to_bits(),
            adamw_no_weight_decay_names: optimizer.adamw_no_weight_decay_name_substrings.clone(),
            ns_steps: optimizer.ns_steps,
            ns_polynomial: newton_schulz_polynomial_bits(),
            force_adamw_names: optimizer.force_adamw_name_substrings.clone(),
            muon_allowlist: optimizer.muon_name_allowlist.clone(),
            per_attention_head_ortho: optimizer.per_attention_head_ortho,
            per_attention_output_head_ortho: optimizer.per_attention_output_head_ortho,
            attention_head_dim: optimizer.attention_head_dim,
            cross_attention_head_dim: optimizer.cross_attention_head_dim,
            actor_grad_clip_patterns: ACTOR_GRAD_CLIP_PATTERNS
                .iter()
                .map(|pattern| (*pattern).to_owned())
                .collect(),
            critic_grad_clip_patterns: CRITIC_GRAD_CLIP_PATTERNS
                .iter()
                .map(|pattern| (*pattern).to_owned())
                .collect(),
        },
        objective_semantics: PpoObjectiveContract {
            gae_gamma: GAE_GAMMA.to_bits(),
            gae_lambda: GAE_LAMBDA.to_bits(),
            clip_eps_low: CLIP_EPS_LOW.to_bits(),
            clip_eps_high: CLIP_EPS_HIGH.to_bits(),
            target_kl: TARGET_KL.to_bits(),
            kl_stop_multiplier: KL_STOP_MULTIPLIER.to_bits(),
            value_loss_coef: VALUE_LOSS_COEF.to_bits(),
            entropy_coef: ENTROPY_COEF.to_bits(),
            max_grad_norm: MAX_GRAD_NORM.to_bits(),
            kl_lr_target: KL_LR_TARGET.to_bits(),
            kl_lr_ema_half_life: KL_LR_EMA_HALF_LIFE.to_bits(),
            kl_lr_min_scale: KL_LR_MIN_SCALE.to_bits(),
            kl_lr_max_scale: KL_LR_MAX_SCALE.to_bits(),
            kl_lr_scale_exponent: KL_LR_SCALE_EXPONENT.to_bits(),
            critic_pretrain_episodes: CRITIC_PRETRAIN_EPISODES,
            policy_objective: match POLICY_OBJECTIVE {
                super::config::PolicyObjective::Ppo => "ppo",
                super::config::PolicyObjective::Pmpo => "pmpo",
            }
            .to_owned(),
            pmpo_pos_to_neg_weight: PMPO_POS_TO_NEG_WEIGHT.to_bits(),
            pmpo_kl_coef: PMPO_KL_COEF.to_bits(),
            return_percentile_low: RET_PERC_LO.to_bits(),
            return_percentile_high: RET_PERC_HI.to_bits(),
            return_percentile_floor: RET_PERC_FLOOR.to_bits(),
            beta_sample_epsilon: BETA_SAMPLE_EPS.to_bits(),
            value_bins: NUM_BINS,
            value_support_min: SYMLOG_SUPPORT_MIN.to_bits(),
            value_support_max: SYMLOG_SUPPORT_MAX.to_bits(),
            value_sigma_ratio: DIRECT_SIGMA_RATIO.to_bits(),
        },
    }
}

fn ppo_muon_config() -> MuonConfig {
    MuonConfig {
        lr: MUON_LR,
        use_muon_for_2d: USE_MUON,
        momentum: MUON_MOMENTUM_WARMUP_START,
        adamw_lr: LEARNING_RATE,
        adamw_betas: (0.9, 0.95),
        adamw_eps: 1e-8,
        weight_decay: 0.0,
        adamw_wd: 0.0,
        force_adamw_name_substrings: vec![
            "policy_concentration".to_string(),
            "resid_mix".to_string(),
            "value_proj".to_string(),
        ],
        ..MuonConfig::default()
    }
}

fn load_ppo_checkpoint_files(
    weights_path: &Path,
    expected_contract: &PpoTrainingContract,
    expected_seed: u64,
) -> Result<ParsedPpoCheckpoint> {
    let metadata_path = ppo_metadata_path(weights_path);
    let metadata: PpoCheckpointMetadata =
        serde_json::from_slice(&fs::read(&metadata_path).with_context(|| {
            format!(
                "failed reading PPO resume metadata {}",
                metadata_path.display()
            )
        })?)
        .with_context(|| {
            format!(
                "failed parsing PPO resume metadata {}",
                metadata_path.display()
            )
        })?;
    if metadata.format_version != PPO_CHECKPOINT_FORMAT_VERSION {
        bail!(
            "unsupported PPO checkpoint format {} in {}",
            metadata.format_version,
            metadata_path.display()
        );
    }
    if metadata.phase != PPO_CHECKPOINT_PHASE {
        bail!("unsupported PPO checkpoint phase {:?}", metadata.phase);
    }
    if &metadata.contract != expected_contract {
        bail!(
            "PPO checkpoint training contract mismatch: saved={:?}, requested={:?}",
            metadata.contract,
            expected_contract
        );
    }
    if metadata.seed != expected_seed {
        bail!(
            "PPO checkpoint seed mismatch: saved={}, requested={expected_seed}",
            metadata.seed
        );
    }
    if metadata.optimizer_step < 0 {
        bail!("PPO checkpoint has a negative optimizer step");
    }
    if !(0..=metadata.optimizer_step).contains(&metadata.muon_momentum_step) {
        bail!(
            "PPO checkpoint Muon momentum step {} is outside 0..={}",
            metadata.muon_momentum_step,
            metadata.optimizer_step
        );
    }
    let mut controller_preflight = KlLrController::new(
        KL_LR_TARGET,
        KL_LR_EMA_HALF_LIFE,
        KL_LR_MIN_SCALE,
        KL_LR_MAX_SCALE,
    );
    if !controller_preflight.restore_state(metadata.kl_lr_controller) {
        bail!("PPO checkpoint contains invalid KL-LR controller state");
    }
    if metadata
        .initialized_adamw
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        bail!("PPO checkpoint AdamW state names are not sorted and unique");
    }
    completed_update_for_resume(metadata.next_update)?;
    let optimizer_path = ppo_optimizer_path(weights_path);
    let trajectory_path = ppo_trajectory_path(weights_path);
    let weights_sha256 = file_sha256(weights_path)?;
    if weights_sha256 != metadata.weights_sha256 {
        bail!(
            "PPO checkpoint weights hash mismatch for {}",
            weights_path.display()
        );
    }
    let optimizer_sha256 = file_sha256(&optimizer_path)?;
    if optimizer_sha256 != metadata.optimizer_sha256 {
        bail!(
            "PPO checkpoint optimizer hash mismatch for {}",
            optimizer_path.display()
        );
    }
    preflight_optimizer_sidecar(&optimizer_path, &metadata)?;
    let trajectory_sha256 = file_sha256(&trajectory_path)?;
    if trajectory_sha256 != metadata.trajectory_sha256 {
        bail!(
            "PPO checkpoint trajectory hash mismatch for {}",
            trajectory_path.display()
        );
    }
    let trajectory = VecEnvSnapshot::from_bytes(&fs::read(&trajectory_path)?)?;
    Ok(ParsedPpoCheckpoint {
        metadata,
        trajectory,
    })
}

fn load_ppo_checkpoint(
    weights_path: &Path,
    expected_contract: &PpoTrainingContract,
    expected_seed: u64,
) -> Result<ValidatedPpoCheckpoint> {
    let parsed = load_ppo_checkpoint_files(weights_path, expected_contract, expected_seed)?;
    let trajectory = parsed
        .trajectory
        .preflight(expected_contract.rollout.nprocs as usize, expected_seed)?;
    Ok(ValidatedPpoCheckpoint {
        metadata: parsed.metadata,
        trajectory,
    })
}

fn save_ppo_checkpoint_bundle(
    vs: &nn::VarStore,
    opt: &Muon,
    weights_path: &Path,
    next_update: usize,
    seed: u64,
    contract: &PpoTrainingContract,
    env: &VecEnv,
    optimizer_step: i64,
    muon_momentum_step: i64,
    kl_lr_controller: &KlLrController,
) -> Result<()> {
    let parent = weights_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)?;
    }
    let optimizer_path = ppo_optimizer_path(weights_path);
    let metadata_path = ppo_metadata_path(weights_path);
    let trajectory_path = ppo_trajectory_path(weights_path);
    let transaction_id = uuid::Uuid::new_v4().to_string();
    let weights_tmp = temp_sibling(weights_path, &transaction_id);
    let optimizer_tmp = temp_sibling(&optimizer_path, &transaction_id);
    let metadata_tmp = temp_sibling(&metadata_path, &transaction_id);
    let trajectory_tmp = temp_sibling(&trajectory_path, &transaction_id);

    vs.save(&weights_tmp)
        .with_context(|| format!("failed saving PPO weights {}", weights_tmp.display()))?;
    opt.save_state(&optimizer_tmp)?;
    fs::write(&trajectory_tmp, env.snapshot()?.to_bytes()?)?;
    let metadata = PpoCheckpointMetadata {
        format_version: PPO_CHECKPOINT_FORMAT_VERSION,
        phase: PPO_CHECKPOINT_PHASE.to_owned(),
        next_update,
        seed,
        contract: contract.clone(),
        optimizer_step,
        muon_momentum_step,
        weights_sha256: file_sha256(&weights_tmp)?,
        optimizer_sha256: file_sha256(&optimizer_tmp)?,
        trajectory_sha256: file_sha256(&trajectory_tmp)?,
        initialized_adamw: opt.initialized_adamw_names(),
        kl_lr_controller: kl_lr_controller.state(),
    };
    fs::write(&metadata_tmp, serde_json::to_vec_pretty(&metadata)?)?;

    File::open(&weights_tmp)?.sync_all()?;
    File::open(&optimizer_tmp)?.sync_all()?;
    File::open(&metadata_tmp)?.sync_all()?;
    File::open(&trajectory_tmp)?.sync_all()?;
    fs::rename(&weights_tmp, weights_path)?;
    fs::rename(&optimizer_tmp, &optimizer_path)?;
    fs::rename(&trajectory_tmp, &trajectory_path)?;
    fs::rename(&metadata_tmp, &metadata_path)?;
    if let Some(parent) = parent {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

impl Trainer {
    pub(super) fn new(
        weights_path: Option<&str>,
        model_variant: ModelVariant,
        run_name: Option<String>,
        seed: u64,
    ) -> Result<Self> {
        if model_variant != ModelVariant::UniformStream {
            bail!(
                "PPO rollout collection supports --model-size uniform-stream only, got {}",
                model_variant.as_str()
            );
        }
        let rollout = rollout_geometry();
        let torch_num_threads = env::var("TORCH_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|threads| *threads > 0)
            .unwrap_or(1);
        let torch_num_interop_threads = env::var("TORCH_NUM_INTEROP_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|threads| *threads > 0)
            .unwrap_or(1);
        tch::set_num_threads(torch_num_threads);
        tch::set_num_interop_threads(torch_num_interop_threads);

        let device = tch::Device::cuda_if_available();
        let optimizer_config = ppo_muon_config();
        let contract = training_contract(
            model_variant,
            device,
            rollout,
            torch_num_threads,
            torch_num_interop_threads,
            &optimizer_config,
        );
        println!("device is cuda: {}", device.is_cuda());
        configure_cuda();
        println!(
            "ppo rollout geometry: nprocs={} seq_len={} total_samples={} chunk_len={} objective=ppo",
            rollout.nprocs, rollout.seq_len, rollout.total_samples, rollout.ppo_chunk_len,
        );
        let weights_path = weights_path.map(Path::new);
        let resume_checkpoint = weights_path.and_then(|path| {
            if should_resume_from_path(path) {
                Some(
                    load_ppo_checkpoint(path, &contract, seed).unwrap_or_else(|error| {
                        panic!(
                            "PPO checkpoint {} is not a complete valid resume bundle: {error:#}",
                            path.display()
                        )
                    }),
                )
            } else {
                None
            }
        });

        tch::manual_seed(deterministic_stream_seed(seed, 0x4d4f_4445_4c49_4e49, 0) as i64);
        let mut vs = nn::VarStore::new(device);
        let trading_model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: model_variant,
                ..TradingModelConfig::default()
            },
        );
        let named_trainable_vars = named_trainable_variables(&vs);
        let grad_clip_groups = grad_clip_groups(&named_trainable_vars);
        let trainable_vars: Vec<Tensor> = named_trainable_vars
            .iter()
            .map(|(_, tensor)| tensor.shallow_clone())
            .collect();
        let mut opt = Muon::new_named(&named_trainable_vars, optimizer_config);
        if let (Some(path), Some(checkpoint)) = (weights_path, resume_checkpoint.as_ref()) {
            opt.validate_state_strict(
                ppo_optimizer_path(path),
                &checkpoint.metadata.initialized_adamw,
                checkpoint.metadata.optimizer_step,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "PPO optimizer {} failed preflight before state restoration: {error:#}",
                    ppo_optimizer_path(path).display()
                )
            });
        }

        let (start_update, run_dir) = if let Some(path) = weights_path {
            let is_resume = resume_checkpoint.is_some();
            if is_resume {
                println!("Loading complete PPO resume bundle from {}", path.display());
            } else {
                println!("Warm-starting PPO weights from {}", path.display());
            }
            let load_summary = load_var_store_partial(&mut vs, path).unwrap();
            load_summary.require_complete().unwrap();
            let rd = if is_resume {
                let run = RunDir::from_weights_path_in(path, RUNS_PATH)
                    .expect("complete PPO resume weights must belong to a managed run");
                if let Some(requested) = run_name.as_deref() {
                    let source_name = run
                        .root
                        .file_name()
                        .and_then(|name| name.to_str())
                        .expect("resume run name is not UTF-8");
                    assert_eq!(
                        requested, source_name,
                        "--run cannot redirect a complete PPO resume bundle; omit it or select its source run"
                    );
                }
                run.activate(RUNS_PATH)
                    .expect("failed to activate resumed run");
                run
            } else {
                RunDir::create_fresh(RUNS_PATH, run_name.as_deref())
                    .expect("failed to create run dir")
            };
            let next_update = resume_checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.metadata.next_update)
                .unwrap_or(0);
            if is_resume {
                println!("Resuming at PPO update {next_update}");
            }
            (next_update, rd)
        } else {
            println!("Starting training from scratch");
            let rd = RunDir::create_fresh(RUNS_PATH, run_name.as_deref())
                .expect("failed to create run dir");
            (0, rd)
        };
        let gens_path = run_dir.gens.to_string_lossy().to_string();
        println!("Run dir: {}", run_dir.root.display());

        let mut optimizer_step = 0i64;
        let mut muon_momentum_step = 0i64;
        let mut kl_lr_controller = KlLrController::new(
            KL_LR_TARGET,
            KL_LR_EMA_HALF_LIFE,
            KL_LR_MIN_SCALE,
            KL_LR_MAX_SCALE,
        );

        let mut env = VecEnv::new_seeded(
            true,
            model_variant,
            gens_path.clone(),
            rollout.nprocs as usize,
            seed,
        );
        let resumed = resume_checkpoint.is_some();
        if let Some(checkpoint) = resume_checkpoint {
            let path = weights_path.expect("resume weights path missing");
            env.restore_snapshot(checkpoint.trajectory);
            opt.load_state_strict(
                ppo_optimizer_path(path),
                &checkpoint.metadata.initialized_adamw,
            )
            .unwrap_or_else(|error| panic!("failed restoring PPO optimizer state: {error:#}"));
            optimizer_step = checkpoint.metadata.optimizer_step;
            muon_momentum_step = checkpoint.metadata.muon_momentum_step;
            assert!(
                kl_lr_controller.restore_state(checkpoint.metadata.kl_lr_controller),
                "PPO checkpoint contains invalid KL-LR controller state"
            );
            println!(
                "Restored PPO optimizer at step {} (Muon momentum step {}) and KL-LR controller scale {:.3}, ema {:.4}",
                optimizer_step,
                muon_momentum_step,
                kl_lr_controller.scale(),
                kl_lr_controller.ema()
            );
        }

        let hl_gauss = HlGaussBins::default_for(device);

        let rollout_steps = rollout.seq_len;
        assert_eq!(
            rollout_steps % rollout.ppo_chunk_len,
            0,
            "PPO_CHUNK_LEN must divide rollout length"
        );
        let chunks_per_rollout = rollout_steps / rollout.ppo_chunk_len;
        let total_chunks = chunks_per_rollout * rollout.nprocs;

        let raw_pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        let pd_dim = trading_model.price_input_dim();
        let so_dim = STATIC_OBSERVATIONS as i64;
        let replay_obs_kind = trading_model.input_kind();
        let s_chunk_start_layouts = Tensor::zeros(
            &[chunks_per_rollout * rollout.nprocs, pd_dim],
            (replay_obs_kind, device),
        );
        let s_static_obs = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len, so_dim],
            (replay_obs_kind, device),
        );
        let s_step_deltas = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len, TICKERS_COUNT],
            (replay_obs_kind, device),
        );
        let s_actions = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len, ACTION_COUNT],
            (Kind::Float, device),
        );
        let s_old_log_probs = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len],
            (Kind::Float, device),
        );
        let s_old_alphas = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len, ACTION_COUNT],
            (Kind::Float, device),
        );
        let s_old_betas = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len, ACTION_COUNT],
            (Kind::Float, device),
        );
        let s_rewards = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len],
            (Kind::Float, device),
        );
        let s_dones = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len],
            (Kind::Float, device),
        );
        let s_values = Tensor::zeros(
            &[total_chunks, rollout.ppo_chunk_len],
            (Kind::Float, device),
        );

        let (obs_price_cpu, obs_static_cpu) = if resumed {
            env.current_full_observation()
        } else {
            env.reset()
        };
        let mut obs_static = Tensor::zeros(
            &[rollout.nprocs, STATIC_OBSERVATIONS as i64],
            (replay_obs_kind, device),
        );
        let step_deltas =
            Tensor::zeros(&[rollout.nprocs, TICKERS_COUNT], (replay_obs_kind, device));
        obs_static.copy_(&obs_static_cpu);
        let obs_price = obs_price_cpu.to_device(device);
        let mut stream_state = trading_model.init_replay_stream_state_batched(rollout.nprocs);
        let stream_layout = trading_model.uniform_stream_layout_from_raw_input(&obs_price);
        let streamed_output = Some(tch::no_grad(|| {
            autocast(false, || {
                trading_model.step_on_device_for_replay(
                    &stream_layout,
                    &obs_static,
                    &mut stream_state,
                )
            })
        }));
        let step_reward_per_ticker =
            Tensor::zeros(&[rollout.nprocs, TICKERS_COUNT], (Kind::Float, device));
        let step_is_done = Tensor::zeros(&[rollout.nprocs], (Kind::Float, device));
        let cpu_step_batch = CpuStepBatch::new(
            rollout.nprocs as usize,
            ACTION_COUNT as usize,
            raw_pd_dim as usize,
        );
        let action_host_view = unsafe {
            Tensor::from_blob(
                cpu_step_batch.actions_f32.as_ptr() as *const u8,
                &[rollout.nprocs, ACTION_COUNT],
                &[],
                Kind::Float,
                Device::Cpu,
            )
        };
        // Persistent CPU staging for reset env indices (one i64 per env, reused each step).
        let reset_env_indices_host: Vec<i64> = vec![0i64; rollout.nprocs as usize];
        let ticker_offsets = Tensor::arange(TICKERS_COUNT, (Kind::Int64, device));

        Ok(Self {
            vs,
            trading_model,
            trainable_vars,
            named_trainable_vars,
            grad_clip_groups,
            opt,
            optimizer_step,
            muon_momentum_step,
            kl_lr_controller,
            env,
            device,
            rollout,
            hl_gauss,
            run_dir,
            start_update,
            rollout_steps,
            total_chunks,
            raw_pd_dim,
            pd_dim,
            so_dim,
            replay_obs_kind,
            s_chunk_start_layouts,
            s_static_obs,
            s_step_deltas,
            s_actions,
            s_old_log_probs,
            s_old_alphas,
            s_old_betas,
            s_rewards,
            s_dones,
            s_values,
            obs_static,
            step_deltas,
            stream_state,
            streamed_output,
            step_reward_per_ticker,
            step_is_done,
            cpu_step_batch,
            action_host_view,
            reset_env_indices_host,
            ticker_offsets,
            ppo_update_graph: None,
            seed,
            contract,
        })
    }

    pub(super) fn deterministic_seed(&self, domain: u64, update: usize) -> u64 {
        deterministic_stream_seed(self.seed, domain, update)
    }

    fn refresh_rollout_frontier(&mut self) {
        let (obs_price_cpu, obs_static_cpu) = self.env.current_full_observation();
        self.obs_static.copy_(&obs_static_cpu);
        let obs_price = obs_price_cpu.to_device(self.device);
        self.stream_state = self
            .trading_model
            .init_replay_stream_state_batched(self.rollout.nprocs);
        let stream_layout = self
            .trading_model
            .uniform_stream_layout_from_raw_input(&obs_price);
        self.streamed_output = Some(tch::no_grad(|| {
            autocast(false, || {
                self.trading_model.step_on_device_for_replay(
                    &stream_layout,
                    &self.obs_static,
                    &mut self.stream_state,
                )
            })
        }));
    }

    pub(super) async fn run(&mut self) -> Result<()> {
        for update in self.start_update..1000000 {
            let rollout_data = self.collect_rollout(update);
            let advantage_data = self.compute_advantages(update, &rollout_data);
            let lr_scale = self.kl_lr_controller.scale();
            apply_lr_scale(&mut self.opt, lr_scale);
            let mut update_metrics = self.update_policy(update, &advantage_data)?;
            let kl_lr_signal = update_metrics.kl_lr_signal;
            // Critic-only pretraining leaves the policy path fixed; keep the
            // controller at its initial state until actor optimization starts.
            if update >= CRITIC_PRETRAIN_EPISODES {
                self.kl_lr_controller.observe(kl_lr_signal);
            }
            update_metrics.lr_scale = lr_scale;
            update_metrics.kl_lr_signal = kl_lr_signal;
            update_metrics.kl_lr_ema = self.kl_lr_controller.ema();
            update_metrics.kl_lr_scale_next = self.kl_lr_controller.scale();
            self.log_episode(update, &advantage_data, &update_metrics)?;
            self.refresh_rollout_frontier();
            self.maybe_checkpoint(update);
        }
        Ok(())
    }

    pub(super) fn maybe_checkpoint(&self, update: usize) {
        if update > 0 && update % 50 == 0 {
            let path = self.run_dir.weights.join(format!("ppo_ep{update}.ot"));
            match save_ppo_checkpoint_bundle(
                &self.vs,
                &self.opt,
                &path,
                update
                    .checked_add(1)
                    .expect("PPO update counter overflowed"),
                self.seed,
                &self.contract,
                &self.env,
                self.optimizer_step,
                self.muon_momentum_step,
                &self.kl_lr_controller,
            ) {
                Ok(()) => println!("Saved complete PPO resume bundle: {}", path.display()),
                Err(error) => println!("Error while saving PPO resume bundle: {error:#}"),
            }
        }
    }
}

fn deterministic_stream_seed(seed: u64, domain: u64, update: usize) -> u64 {
    let mut z = seed ^ domain ^ (update as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::constants::{ACTION_COUNT, STEPS_PER_EPISODE};
    use crate::torch::env::synthetic_env;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;
    use std::sync::Mutex;

    #[test]
    fn trainer_rejects_non_streaming_models_without_panicking() {
        for variant in [ModelVariant::Base, ModelVariant::AblationSmall] {
            let Err(error) = Trainer::new(None, variant, None, 20260811) else {
                panic!("unsupported PPO model variant must return an error");
            };
            assert!(error.to_string().contains("uniform-stream"));
        }
    }

    // libtorch's generator is process-global. Keep tests which reseed it from
    // perturbing one another when Rust's test harness runs them concurrently.
    static TORCH_RNG_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn resume_contract_covers_runtime_and_effective_objective_constants() {
        let contract = training_contract(
            ModelVariant::UniformStream,
            Device::Cpu,
            RolloutGeometry {
                nprocs: 1,
                seq_len: 2,
                ppo_chunk_len: 1,
                total_samples: 2,
            },
            1,
            1,
            &ppo_muon_config(),
        );

        assert_eq!(contract.libtorch_runtime, "2.12.1+cu130");
        assert!(contract.rng_algorithm.contains("libtorch::standard_gamma"));
        assert_eq!(
            contract.objective_semantics.return_percentile_low,
            RET_PERC_LO.to_bits()
        );
        assert_eq!(
            contract.objective_semantics.beta_sample_epsilon,
            BETA_SAMPLE_EPS.to_bits()
        );
        assert_eq!(contract.objective_semantics.value_bins, NUM_BINS);
        assert_eq!(
            contract.objective_semantics.kl_lr_scale_exponent,
            KL_LR_SCALE_EXPONENT.to_bits()
        );
        assert_eq!(
            contract.optimizer.actor_grad_clip_patterns,
            vec!["policy_concentration"]
        );
        assert_eq!(
            contract.optimizer.critic_grad_clip_patterns,
            vec!["value_proj", "next_return_head"]
        );
        assert_eq!(
            contract.objective_semantics.value_sigma_ratio,
            DIRECT_SIGMA_RATIO.to_bits()
        );
    }

    #[derive(Debug, Default, PartialEq, Eq)]
    struct InterruptionTrace {
        action_bits: Vec<Vec<u32>>,
        reward_bits: Vec<u64>,
        dones: Vec<u32>,
        minibatch_orders: Vec<Vec<usize>>,
        env_frontiers: Vec<(usize, usize, Vec<usize>, u64)>,
    }

    struct MiniPpo {
        vs: nn::VarStore,
        parameter: Tensor,
        opt: Muon,
        env: VecEnv,
        controller: KlLrController,
        optimizer_step: i64,
        muon_momentum_step: i64,
        seed: u64,
        contract: PpoTrainingContract,
    }

    impl MiniPpo {
        fn new(seed: u64) -> Self {
            tch::manual_seed(deterministic_stream_seed(seed, 0x4d49_4e49, 0) as i64);
            let vs = nn::VarStore::new(Device::Cpu);
            let parameter = vs
                .root()
                .var("policy", &[ACTION_COUNT], nn::Init::Const(0.125));
            let named = named_trainable_variables(&vs);
            let optimizer_config = MuonConfig {
                lr: 1e-3,
                use_muon_for_2d: true,
                adamw_lr: 1e-3,
                adamw_betas: (0.9, 0.95),
                adamw_eps: 1e-8,
                weight_decay: 0.0,
                adamw_wd: 0.0,
                ..MuonConfig::default()
            };
            let contract = training_contract(
                ModelVariant::UniformStream,
                Device::Cpu,
                RolloutGeometry {
                    nprocs: 1,
                    seq_len: 2,
                    ppo_chunk_len: 1,
                    total_samples: 2,
                },
                1,
                1,
                &optimizer_config,
            );
            let opt = Muon::new_named(&named, optimizer_config);
            let mut env = synthetic_env();
            let hold = vec![0.0; ACTION_COUNT as usize];
            for _ in 0..(STEPS_PER_EPISODE - 6) {
                assert_eq!(env.step_step_single(&hold).is_done, 0.0);
            }
            Self {
                vs,
                parameter,
                opt,
                env: VecEnv::from_test_envs(vec![env]),
                controller: KlLrController::new(0.035, 50.0, 0.01, 10.0),
                optimizer_step: 0,
                muon_momentum_step: 0,
                seed,
                contract,
            }
        }

        fn update(&mut self, update: usize, trace: &mut InterruptionTrace) {
            tch::manual_seed(
                deterministic_stream_seed(self.seed, 0x524f_4c4c_4f55_5453, update) as i64,
            );
            let mut losses = Vec::new();
            for _ in 0..2 {
                let noise = Tensor::rand([ACTION_COUNT], (Kind::Float, Device::Cpu));
                let actions = (&self.parameter + noise).sigmoid();
                let action_values = Vec::<f32>::try_from(actions.shallow_clone()).unwrap();
                let transition = self.env.envs[0].step_step_single(
                    &action_values
                        .iter()
                        .map(|value| *value as f64)
                        .collect::<Vec<_>>(),
                );
                trace
                    .action_bits
                    .push(action_values.iter().map(|value| value.to_bits()).collect());
                trace.reward_bits.push(transition.reward.to_bits());
                trace.dones.push(transition.is_done.to_bits());
                losses.push(actions.mean(Kind::Float) * (1.0 + transition.reward as f64));
                if transition.is_done == 1.0 {
                    self.env.envs[0].reset_existing_episode_state();
                }
            }

            let mut order = (0..8usize).collect::<Vec<_>>();
            let mut rng = ChaCha12Rng::seed_from_u64(deterministic_stream_seed(
                self.seed,
                0x4d49_4e49_4241_5443,
                update,
            ));
            order.shuffle(&mut rng);
            trace.minibatch_orders.push(order);

            self.opt.zero_grad();
            Tensor::stack(&losses, 0).mean(Kind::Float).backward();
            crate::torch::train::optimizer_glue::step_optimizer(
                &mut self.opt,
                &mut self.optimizer_step,
                &mut self.muon_momentum_step,
                true,
            );
            self.controller.observe(0.01 + update as f64 * 0.001);
            let env = &self.env.envs[0];
            trace.env_frontiers.push((
                env.episode,
                env.step,
                env.ticker_perm.clone(),
                env.test_rng_counter(),
            ));
        }

        fn save(&self, directory: &Path, next_update: usize) -> Result<()> {
            fs::create_dir_all(directory)?;
            save_ppo_checkpoint_bundle(
                &self.vs,
                &self.opt,
                &directory.join("ppo_ep0.ot"),
                next_update,
                self.seed,
                &self.contract,
                &self.env,
                self.optimizer_step,
                self.muon_momentum_step,
                &self.controller,
            )
        }

        fn resume(seed: u64, directory: &Path) -> Result<(Self, usize)> {
            let mut resumed = Self::new(seed);
            let weights_path = directory.join("ppo_ep0.ot");
            let checkpoint = load_ppo_checkpoint_files(&weights_path, &resumed.contract, seed)?;
            resumed.opt.validate_state_strict(
                ppo_optimizer_path(&weights_path),
                &checkpoint.metadata.initialized_adamw,
                checkpoint.metadata.optimizer_step,
            )?;
            resumed.vs.load(&weights_path)?;
            resumed.opt.load_state_strict(
                ppo_optimizer_path(&weights_path),
                &checkpoint.metadata.initialized_adamw,
            )?;
            resumed.optimizer_step = checkpoint.metadata.optimizer_step;
            resumed.muon_momentum_step = checkpoint.metadata.muon_momentum_step;
            resumed
                .env
                .restore_snapshot_from_current_markets(checkpoint.trajectory)?;
            assert!(resumed
                .controller
                .restore_state(checkpoint.metadata.kl_lr_controller));
            let next_update = checkpoint.metadata.next_update;
            Ok((resumed, next_update))
        }
    }

    #[derive(Debug, Default, PartialEq, Eq)]
    struct ProductionPpoTrace {
        actions: Vec<Vec<u32>>,
        log_probs: Vec<Vec<u32>>,
        rewards: Vec<Vec<u32>>,
        dones: Vec<Vec<u32>>,
        frontiers: Vec<(usize, usize, Vec<usize>, u64)>,
    }

    fn tensor_bits(tensor: &Tensor) -> Vec<u32> {
        Vec::<f32>::try_from(tensor.flatten(0, -1).to_device(Device::Cpu))
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn production_test_trainer(seed: u64) -> (Trainer, PathBuf) {
        let device = Device::Cpu;
        let rollout = RolloutGeometry {
            nprocs: 1,
            seq_len: 2,
            ppo_chunk_len: 1,
            total_samples: 2,
        };
        let optimizer_config = ppo_muon_config();
        let contract = training_contract(
            ModelVariant::UniformStream,
            device,
            rollout,
            1,
            1,
            &optimizer_config,
        );
        tch::manual_seed(deterministic_stream_seed(seed, 0x4d4f_4445_4c49_4e49, 0) as i64);
        let vs = nn::VarStore::new(device);
        let trading_model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
                ..TradingModelConfig::default()
            },
        );
        let named_trainable_vars = named_trainable_variables(&vs);
        let grad_clip_groups = grad_clip_groups(&named_trainable_vars);
        let trainable_vars = named_trainable_vars
            .iter()
            .map(|(_, tensor)| tensor.shallow_clone())
            .collect::<Vec<_>>();
        let opt = Muon::new_named(&named_trainable_vars, optimizer_config);

        let mut single = synthetic_env();
        let hold = vec![0.0; ACTION_COUNT as usize];
        for _ in 0..(STEPS_PER_EPISODE - 6) {
            assert_eq!(single.step_step_single(&hold).is_done, 0.0);
        }
        let mut env = VecEnv::from_test_envs(vec![single]);
        let run_root = std::env::temp_dir().join(format!(
            "trading-bot-production-ppo-test-{}",
            uuid::Uuid::new_v4()
        ));
        let gens = run_root.join("gens");
        let weights = run_root.join("weights");
        fs::create_dir_all(&gens).unwrap();
        fs::create_dir_all(&weights).unwrap();
        let run_dir = RunDir {
            root: run_root.clone(),
            gens,
            weights,
            log_file: run_root.join("training.log"),
        };

        let rollout_steps = rollout.seq_len;
        let total_chunks = rollout_steps / rollout.ppo_chunk_len * rollout.nprocs;
        let raw_pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        let pd_dim = trading_model.price_input_dim();
        let so_dim = STATIC_OBSERVATIONS as i64;
        let replay_obs_kind = trading_model.input_kind();
        let s_chunk_start_layouts =
            Tensor::zeros([total_chunks, pd_dim], (replay_obs_kind, device));
        let s_static_obs = Tensor::zeros(
            [total_chunks, rollout.ppo_chunk_len, so_dim],
            (replay_obs_kind, device),
        );
        let s_step_deltas = Tensor::zeros(
            [total_chunks, rollout.ppo_chunk_len, TICKERS_COUNT],
            (replay_obs_kind, device),
        );
        let s_actions = Tensor::zeros(
            [total_chunks, rollout.ppo_chunk_len, ACTION_COUNT],
            (Kind::Float, device),
        );
        let s_old_log_probs =
            Tensor::zeros([total_chunks, rollout.ppo_chunk_len], (Kind::Float, device));
        let s_old_alphas = Tensor::zeros(
            [total_chunks, rollout.ppo_chunk_len, ACTION_COUNT],
            (Kind::Float, device),
        );
        let s_old_betas = Tensor::zeros(
            [total_chunks, rollout.ppo_chunk_len, ACTION_COUNT],
            (Kind::Float, device),
        );
        let s_rewards = Tensor::zeros([total_chunks, rollout.ppo_chunk_len], (Kind::Float, device));
        let s_dones = Tensor::zeros([total_chunks, rollout.ppo_chunk_len], (Kind::Float, device));
        let s_values = Tensor::zeros([total_chunks, rollout.ppo_chunk_len], (Kind::Float, device));

        let (obs_price_cpu, obs_static_cpu) = env.current_full_observation();
        let mut obs_static = Tensor::zeros(
            [rollout.nprocs, STATIC_OBSERVATIONS as i64],
            (replay_obs_kind, device),
        );
        obs_static.copy_(&obs_static_cpu);
        let step_deltas = Tensor::zeros([rollout.nprocs, TICKERS_COUNT], (replay_obs_kind, device));
        let mut stream_state = trading_model.init_replay_stream_state_batched(rollout.nprocs);
        let stream_layout =
            trading_model.uniform_stream_layout_from_raw_input(&obs_price_cpu.to_device(device));
        let streamed_output = Some(tch::no_grad(|| {
            trading_model.step_on_device_for_replay(&stream_layout, &obs_static, &mut stream_state)
        }));
        let step_reward_per_ticker =
            Tensor::zeros([rollout.nprocs, TICKERS_COUNT], (Kind::Float, device));
        let step_is_done = Tensor::zeros([rollout.nprocs], (Kind::Float, device));
        let cpu_step_batch = CpuStepBatch::new(1, ACTION_COUNT as usize, raw_pd_dim as usize);
        let action_host_view = unsafe {
            Tensor::from_blob(
                cpu_step_batch.actions_f32.as_ptr() as *const u8,
                &[rollout.nprocs, ACTION_COUNT],
                &[],
                Kind::Float,
                Device::Cpu,
            )
        };

        (
            Trainer {
                vs,
                trading_model,
                trainable_vars,
                named_trainable_vars,
                grad_clip_groups,
                opt,
                optimizer_step: 0,
                muon_momentum_step: 0,
                kl_lr_controller: KlLrController::new(
                    KL_LR_TARGET,
                    KL_LR_EMA_HALF_LIFE,
                    KL_LR_MIN_SCALE,
                    KL_LR_MAX_SCALE,
                ),
                env,
                device,
                rollout,
                hl_gauss: HlGaussBins::default_for(device),
                run_dir,
                start_update: 100,
                rollout_steps,
                total_chunks,
                raw_pd_dim,
                pd_dim,
                so_dim,
                replay_obs_kind,
                s_chunk_start_layouts,
                s_static_obs,
                s_step_deltas,
                s_actions,
                s_old_log_probs,
                s_old_alphas,
                s_old_betas,
                s_rewards,
                s_dones,
                s_values,
                obs_static,
                step_deltas,
                stream_state,
                streamed_output,
                step_reward_per_ticker,
                step_is_done,
                cpu_step_batch,
                action_host_view,
                reset_env_indices_host: vec![0],
                ticker_offsets: Tensor::arange(TICKERS_COUNT, (Kind::Int64, device)),
                ppo_update_graph: None,
                seed,
                contract,
            },
            run_root,
        )
    }

    fn production_update(trainer: &mut Trainer, update: usize, trace: &mut ProductionPpoTrace) {
        let rollout_data = trainer.collect_rollout(update);
        trace.actions.push(tensor_bits(&trainer.s_actions));
        trace.log_probs.push(tensor_bits(&trainer.s_old_log_probs));
        trace.rewards.push(tensor_bits(&trainer.s_rewards));
        trace.dones.push(tensor_bits(&trainer.s_dones));
        let advantage_data = trainer.compute_advantages(update, &rollout_data);
        let lr_scale = trainer.kl_lr_controller.scale();
        apply_lr_scale(&mut trainer.opt, lr_scale);
        let mut metrics = trainer.update_policy(update, &advantage_data).unwrap();
        let signal = metrics.kl_lr_signal;
        if update >= CRITIC_PRETRAIN_EPISODES {
            trainer.kl_lr_controller.observe(signal);
        }
        metrics.lr_scale = lr_scale;
        metrics.kl_lr_signal = signal;
        metrics.kl_lr_ema = trainer.kl_lr_controller.ema();
        metrics.kl_lr_scale_next = trainer.kl_lr_controller.scale();
        trainer
            .log_episode(update, &advantage_data, &metrics)
            .unwrap();
        trainer.refresh_rollout_frontier();
        let env = &trainer.env.envs[0];
        trace.frontiers.push((
            env.episode,
            env.step,
            env.ticker_perm.clone(),
            env.test_rng_counter(),
        ));
    }

    #[test]
    fn ppo_checkpoint_paths_distinguish_resume_from_warm_start() {
        let checkpoint = Path::new("weights/ppo_ep50.ot");
        assert!(is_ppo_checkpoint_path(checkpoint));
        assert_eq!(
            ppo_optimizer_path(checkpoint),
            PathBuf::from("weights/ppo_ep50.optimizer.ot")
        );
        assert_eq!(
            ppo_metadata_path(checkpoint),
            PathBuf::from("weights/ppo_ep50.resume.json")
        );
        assert_eq!(
            ppo_trajectory_path(checkpoint),
            PathBuf::from("weights/ppo_ep50.trajectory.postcard")
        );
        assert!(!is_ppo_checkpoint_path(Path::new(
            "weights/pretrain_model.ot"
        )));
        assert_eq!(completed_update_for_resume(51).unwrap(), 50);
        assert!(completed_update_for_resume(0).is_err());
    }

    #[test]
    fn deterministic_streams_resume_at_the_same_update_boundary() {
        let uninterrupted = (0..4)
            .map(|update| deterministic_stream_seed(19, 7, update))
            .collect::<Vec<_>>();
        let resumed = (2..4)
            .map(|update| deterministic_stream_seed(19, 7, update))
            .collect::<Vec<_>>();
        assert_eq!(&uninterrupted[2..], resumed);
        assert_ne!(
            deterministic_stream_seed(19, 7, 2),
            deterministic_stream_seed(19, 8, 2)
        );
    }

    #[test]
    fn interruption_resume_matches_three_uninterrupted_ppo_updates() {
        let _torch_rng_guard = TORCH_RNG_TEST_LOCK.lock().unwrap();
        let seed = 0x5eed_u64;
        let mut uninterrupted = MiniPpo::new(seed);
        let mut uninterrupted_trace = InterruptionTrace::default();
        for update in 0..3 {
            uninterrupted.update(update, &mut uninterrupted_trace);
        }

        let mut interrupted = MiniPpo::new(seed);
        let mut resumed_trace = InterruptionTrace::default();
        interrupted.update(0, &mut resumed_trace);
        let directory = std::env::temp_dir().join(format!(
            "trading-bot-ppo-resume-test-{}",
            uuid::Uuid::new_v4()
        ));
        interrupted.save(&directory, 1).unwrap();
        let (mut resumed, next_update) = MiniPpo::resume(seed, &directory).unwrap();
        assert_eq!(next_update, 1);
        for update in next_update..3 {
            resumed.update(update, &mut resumed_trace);
        }

        assert_eq!(uninterrupted_trace, resumed_trace);
        assert_eq!(uninterrupted.optimizer_step, resumed.optimizer_step);
        assert_eq!(uninterrupted.muon_momentum_step, resumed.muon_momentum_step);
        assert_eq!(
            uninterrupted.controller.ema().to_bits(),
            resumed.controller.ema().to_bits()
        );
        assert_eq!(
            uninterrupted.controller.scale().to_bits(),
            resumed.controller.scale().to_bits()
        );
        assert_eq!(
            Vec::<f32>::try_from(uninterrupted.parameter.shallow_clone()).unwrap(),
            Vec::<f32>::try_from(resumed.parameter.shallow_clone()).unwrap()
        );
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn production_ppo_frontier_is_exact_across_interruption() {
        let _torch_rng_guard = TORCH_RNG_TEST_LOCK.lock().unwrap();
        let seed = 0xface_5eed_u64;
        let (mut uninterrupted, uninterrupted_root) = production_test_trainer(seed);
        let mut uninterrupted_trace = ProductionPpoTrace::default();
        for update in 100..103 {
            production_update(&mut uninterrupted, update, &mut uninterrupted_trace);
        }

        let (mut interrupted, interrupted_root) = production_test_trainer(seed);
        let mut resumed_trace = ProductionPpoTrace::default();
        production_update(&mut interrupted, 100, &mut resumed_trace);
        let checkpoint_path = interrupted.run_dir.weights.join("ppo_ep100.ot");
        save_ppo_checkpoint_bundle(
            &interrupted.vs,
            &interrupted.opt,
            &checkpoint_path,
            101,
            seed,
            &interrupted.contract,
            &interrupted.env,
            interrupted.optimizer_step,
            interrupted.muon_momentum_step,
            &interrupted.kl_lr_controller,
        )
        .unwrap();

        let (mut resumed, resumed_root) = production_test_trainer(seed);
        let checkpoint =
            load_ppo_checkpoint_files(&checkpoint_path, &resumed.contract, seed).unwrap();
        resumed
            .opt
            .validate_state_strict(
                ppo_optimizer_path(&checkpoint_path),
                &checkpoint.metadata.initialized_adamw,
                checkpoint.metadata.optimizer_step,
            )
            .unwrap();
        resumed.vs.load(&checkpoint_path).unwrap();
        resumed
            .opt
            .load_state_strict(
                ppo_optimizer_path(&checkpoint_path),
                &checkpoint.metadata.initialized_adamw,
            )
            .unwrap();
        resumed.optimizer_step = checkpoint.metadata.optimizer_step;
        resumed.muon_momentum_step = checkpoint.metadata.muon_momentum_step;
        resumed
            .env
            .restore_snapshot_from_current_markets(checkpoint.trajectory)
            .unwrap();
        assert!(resumed
            .kl_lr_controller
            .restore_state(checkpoint.metadata.kl_lr_controller));
        resumed.refresh_rollout_frontier();
        for update in checkpoint.metadata.next_update..103 {
            production_update(&mut resumed, update, &mut resumed_trace);
        }

        assert_eq!(uninterrupted_trace, resumed_trace);
        assert_eq!(uninterrupted.optimizer_step, resumed.optimizer_step);
        assert_eq!(uninterrupted.muon_momentum_step, resumed.muon_momentum_step);
        assert_eq!(
            uninterrupted.kl_lr_controller.ema().to_bits(),
            resumed.kl_lr_controller.ema().to_bits()
        );
        for ((left_name, left), (right_name, right)) in uninterrupted
            .named_trainable_vars
            .iter()
            .zip(&resumed.named_trainable_vars)
        {
            assert_eq!(left_name, right_name);
            assert_eq!(
                tensor_bits(left),
                tensor_bits(right),
                "parameter {left_name}"
            );
        }

        fs::remove_dir_all(uninterrupted_root).unwrap();
        fs::remove_dir_all(interrupted_root).unwrap();
        fs::remove_dir_all(resumed_root).unwrap();
    }
}
