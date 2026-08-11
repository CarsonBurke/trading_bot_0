use anyhow::{bail, Context, Result};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tch::{autocast, nn, Device, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::env::{CpuStepBatch, VecEnv};
use crate::torch::hashing::file_sha256;
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{
    ModelOutput, ModelVariant, StreamState, TradingModel, TradingModelConfig,
};
use crate::torch::optim::muon::{Muon, MuonConfig};
use crate::torch::value::hl_gauss::HlGaussBins;
use shared::{paths::RUNS_PATH, run_dir::RunDir};

use super::config::{
    CRITIC_PRETRAIN_EPISODES, KL_LR_EMA_HALF_LIFE, KL_LR_MAX_SCALE, KL_LR_MIN_SCALE, KL_LR_TARGET,
    LEARNING_RATE, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON,
};
use super::geometry::{rollout_geometry, RolloutGeometry};
use super::optimizer_glue::{
    apply_lr_scale, grad_clip_groups, named_trainable_variables, GradClipGroups, KlLrController,
    KlLrControllerState,
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
    pub(super) last_minibatch_approx_kl: f64,
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
    pub(super) kl_lr_controller: KlLrController,
    pub(super) env: VecEnv,
    pub(super) device: Device,
    pub(super) rollout: RolloutGeometry,
    pub(super) hl_gauss: HlGaussBins,
    pub(super) run_dir: RunDir,
    pub(super) start_episode: usize,
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
}

const PPO_CHECKPOINT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PpoCheckpointMetadata {
    format_version: u32,
    next_episode: usize,
    optimizer_step: i64,
    weights_sha256: String,
    optimizer_sha256: String,
    kl_lr_controller: KlLrControllerState,
}

fn ppo_metadata_path(weights_path: &Path) -> PathBuf {
    weights_path.with_extension("resume.json")
}

fn ppo_optimizer_path(weights_path: &Path) -> PathBuf {
    weights_path.with_extension("optimizer.ot")
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

fn completed_episode_for_resume(next_episode: usize) -> Result<usize> {
    next_episode
        .checked_sub(1)
        .context("PPO checkpoint does not follow a completed episode")
}

fn load_ppo_checkpoint_metadata(weights_path: &Path) -> Result<PpoCheckpointMetadata> {
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
    if metadata.optimizer_step < 0 {
        bail!("PPO checkpoint has a negative optimizer step");
    }
    completed_episode_for_resume(metadata.next_episode)?;
    let optimizer_path = ppo_optimizer_path(weights_path);
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
    Ok(metadata)
}

fn save_ppo_checkpoint_bundle(
    vs: &nn::VarStore,
    opt: &Muon,
    weights_path: &Path,
    completed_episode: usize,
    optimizer_step: i64,
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
    let transaction_id = uuid::Uuid::new_v4().to_string();
    let weights_tmp = temp_sibling(weights_path, &transaction_id);
    let optimizer_tmp = temp_sibling(&optimizer_path, &transaction_id);
    let metadata_tmp = temp_sibling(&metadata_path, &transaction_id);

    vs.save(&weights_tmp)
        .with_context(|| format!("failed saving PPO weights {}", weights_tmp.display()))?;
    opt.save_state(&optimizer_tmp)?;
    let metadata = PpoCheckpointMetadata {
        format_version: PPO_CHECKPOINT_FORMAT_VERSION,
        next_episode: completed_episode
            .checked_add(1)
            .context("PPO checkpoint episode overflow")?,
        optimizer_step,
        weights_sha256: file_sha256(&weights_tmp)?,
        optimizer_sha256: file_sha256(&optimizer_tmp)?,
        kl_lr_controller: kl_lr_controller.state(),
    };
    fs::write(&metadata_tmp, serde_json::to_vec_pretty(&metadata)?)?;

    File::open(&weights_tmp)?.sync_all()?;
    File::open(&optimizer_tmp)?.sync_all()?;
    File::open(&metadata_tmp)?.sync_all()?;
    fs::rename(&weights_tmp, weights_path)?;
    fs::rename(&optimizer_tmp, &optimizer_path)?;
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
    ) -> Result<Self> {
        if model_variant != ModelVariant::UniformStream {
            bail!(
                "PPO rollout collection supports --model-size uniform-stream only, got {}",
                model_variant.as_str()
            );
        }
        let rollout = rollout_geometry();
        if let Some(threads) = env::var("TORCH_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            tch::set_num_threads(threads);
        } else {
            tch::set_num_threads(1);
        }
        if let Some(threads) = env::var("TORCH_NUM_INTEROP_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            tch::set_num_interop_threads(threads);
        } else {
            tch::set_num_interop_threads(1);
        }

        let device = tch::Device::cuda_if_available();
        println!("device is cuda: {}", device.is_cuda());
        configure_cuda();
        println!(
            "ppo rollout geometry: nprocs={} seq_len={} total_samples={} chunk_len={} objective=ppo",
            rollout.nprocs, rollout.seq_len, rollout.total_samples, rollout.ppo_chunk_len,
        );
        let weights_path = weights_path.map(Path::new);
        let resume_metadata = weights_path.and_then(|path| {
            if should_resume_from_path(path) {
                Some(load_ppo_checkpoint_metadata(path).unwrap_or_else(|error| {
                    panic!(
                        "PPO checkpoint {} is not a complete valid resume bundle: {error:#}",
                        path.display()
                    )
                }))
            } else {
                None
            }
        });

        let mut vs = nn::VarStore::new(device);
        let trading_model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: model_variant,
                ..TradingModelConfig::default()
            },
        );

        let (start_episode, run_dir) = if let Some(path) = weights_path {
            let is_resume = resume_metadata.is_some();
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
            let next_episode = resume_metadata
                .as_ref()
                .map(|metadata| metadata.next_episode)
                .unwrap_or(0);
            if is_resume {
                println!("Resuming at episode {next_episode}");
            }
            (next_episode, rd)
        } else {
            println!("Starting training from scratch");
            let rd = RunDir::create_fresh(RUNS_PATH, run_name.as_deref())
                .expect("failed to create run dir");
            (0, rd)
        };
        let gens_path = run_dir.gens.to_string_lossy().to_string();
        println!("Run dir: {}", run_dir.root.display());

        let named_trainable_vars = named_trainable_variables(&vs);
        let grad_clip_groups = grad_clip_groups(&named_trainable_vars);
        let trainable_vars: Vec<Tensor> = named_trainable_vars
            .iter()
            .map(|(_, tensor)| tensor.shallow_clone())
            .collect();
        let mut opt = Muon::new_named(
            &named_trainable_vars,
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
            },
        );
        let mut optimizer_step = 0i64;
        let mut kl_lr_controller = KlLrController::new(
            KL_LR_TARGET,
            KL_LR_EMA_HALF_LIFE,
            KL_LR_MIN_SCALE,
            KL_LR_MAX_SCALE,
        );

        let mut env = VecEnv::new(
            true,
            model_variant,
            gens_path.clone(),
            rollout.nprocs as usize,
        );
        if let Some(metadata) = resume_metadata {
            let path = weights_path.expect("resume weights path missing");
            opt.load_state(ppo_optimizer_path(path))
                .unwrap_or_else(|error| panic!("failed restoring PPO optimizer state: {error:#}"));
            optimizer_step = metadata.optimizer_step;
            assert!(
                kl_lr_controller.restore_state(metadata.kl_lr_controller),
                "PPO checkpoint contains invalid KL-LR controller state"
            );
            println!(
                "Restored PPO optimizer at step {} and KL-LR controller scale {:.3}, ema {:.4}",
                optimizer_step,
                kl_lr_controller.scale(),
                kl_lr_controller.ema()
            );

            env.set_episode(start_episode);
            let meta_history = &mut env.primary_mut().meta_history;
            let completed_episode = completed_episode_for_resume(start_episode)
                .expect("validated PPO resume metadata became invalid");
            meta_history
                .load_from_episode(completed_episode, &gens_path)
                .unwrap_or_else(|error| panic!("failed restoring PPO report history: {error:#}"));
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

        let (obs_price_cpu, obs_static_cpu) = env.reset();
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
            kl_lr_controller,
            env,
            device,
            rollout,
            hl_gauss,
            run_dir,
            start_episode,
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
        })
    }

    pub(super) async fn run(&mut self) -> Result<()> {
        for episode in self.start_episode..1000000 {
            let rollout_data = self.collect_rollout(episode);
            let advantage_data = self.compute_advantages(episode, &rollout_data);
            let lr_scale = self.kl_lr_controller.scale();
            apply_lr_scale(&mut self.opt, lr_scale);
            let mut update_metrics = self.update_policy(episode, &advantage_data);
            let kl_lr_signal = update_metrics.last_minibatch_approx_kl;
            // During critic-only pretraining the policy KL reflects trunk drift,
            // not actor learning, so it must not steer the KL-adaptive LR.
            if episode >= CRITIC_PRETRAIN_EPISODES {
                self.kl_lr_controller.observe(kl_lr_signal);
            }
            update_metrics.lr_scale = lr_scale;
            update_metrics.kl_lr_signal = kl_lr_signal;
            update_metrics.kl_lr_ema = self.kl_lr_controller.ema();
            update_metrics.kl_lr_scale_next = self.kl_lr_controller.scale();
            self.log_episode(episode, &advantage_data, &update_metrics)?;
            self.maybe_checkpoint(episode);
        }
        Ok(())
    }

    pub(super) fn maybe_checkpoint(&self, episode: usize) {
        if episode > 0 && episode % 50 == 0 {
            let path = self.run_dir.weights.join(format!("ppo_ep{episode}.ot"));
            match save_ppo_checkpoint_bundle(
                &self.vs,
                &self.opt,
                &path,
                episode,
                self.optimizer_step,
                &self.kl_lr_controller,
            ) {
                Ok(()) => println!("Saved complete PPO resume bundle: {}", path.display()),
                Err(error) => println!("Error while saving PPO resume bundle: {error:#}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trainer_rejects_non_streaming_models_without_panicking() {
        for variant in [ModelVariant::Base, ModelVariant::AblationSmall] {
            let Err(error) = Trainer::new(None, variant, None) else {
                panic!("unsupported PPO model variant must return an error");
            };
            assert!(error.to_string().contains("uniform-stream"));
        }
    }

    fn test_optimizer(named: &[(String, Tensor)]) -> Muon {
        Muon::new_named(
            named,
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        )
    }

    #[test]
    fn ppo_checkpoint_bundle_roundtrips_resume_state_and_next_episode() {
        let dir = std::env::temp_dir().join(format!(
            "trading-bot-ppo-checkpoint-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("ppo_ep41.ot");

        let vs = nn::VarStore::new(Device::Cpu);
        let weight = vs.root().var("weight", &[2, 2], nn::Init::Const(1.0));
        let named = vec![("weight".to_owned(), weight.shallow_clone())];
        let mut opt = test_optimizer(&named);
        weight.sum(Kind::Float).backward();
        opt.step();
        let expected_weight_sum = weight.sum(Kind::Float).double_value(&[]);
        let mut controller = KlLrController::new(0.02, 20.0, 0.1, 10.0);
        controller.restore(0.04, 2.5);

        save_ppo_checkpoint_bundle(&vs, &opt, &checkpoint, 41, 17, &controller).unwrap();
        let metadata = load_ppo_checkpoint_metadata(&checkpoint).unwrap();
        assert_eq!(metadata.next_episode, 42);
        assert_eq!(metadata.optimizer_step, 17);
        let mut restored_controller = KlLrController::new(0.02, 20.0, 0.1, 10.0);
        assert!(restored_controller.restore_state(metadata.kl_lr_controller));
        assert!((restored_controller.ema() - 0.04).abs() < 1e-12);
        assert!((restored_controller.scale() - 2.5).abs() < 1e-12);

        let mut restored_vs = nn::VarStore::new(Device::Cpu);
        let restored_weight = restored_vs
            .root()
            .var("weight", &[2, 2], nn::Init::Const(0.0));
        restored_vs.load(&checkpoint).unwrap();
        assert_eq!(
            restored_weight.sum(Kind::Float).double_value(&[]),
            expected_weight_sum
        );
        let restored_named = vec![("weight".to_owned(), restored_weight)];
        let mut restored_opt = test_optimizer(&restored_named);
        restored_opt
            .load_state(ppo_optimizer_path(&checkpoint))
            .unwrap();

        assert!(checkpoint.exists());
        assert!(ppo_optimizer_path(&checkpoint).exists());
        assert!(ppo_metadata_path(&checkpoint).exists());
        fs::write(ppo_optimizer_path(&checkpoint), b"corrupt optimizer").unwrap();
        assert!(load_ppo_checkpoint_metadata(&checkpoint).is_err());
        fs::remove_dir_all(dir).unwrap();
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
        assert!(!is_ppo_checkpoint_path(Path::new(
            "weights/pretrain_model.ot"
        )));
        assert_eq!(completed_episode_for_resume(51).unwrap(), 50);
        assert!(completed_episode_for_resume(0).is_err());
    }
}
