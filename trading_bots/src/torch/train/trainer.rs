use std::env;
use std::path::Path;
use tch::{autocast, nn, Device, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::env::{CpuStepBatch, VecEnv};
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{
    ModelOutput, ModelVariant, StreamState, TradingModel, TradingModelConfig,
};
use crate::torch::optim::muon::{Muon, MuonConfig};
use crate::torch::value::hl_gauss::HlGaussBins;
use shared::{paths::RUNS_PATH, run_dir::RunDir};

use super::config::{LEARNING_RATE, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON};
use super::geometry::{rollout_geometry, RolloutGeometry};
use super::optimizer_glue::{grad_clip_groups, named_trainable_variables, GradClipGroups};

pub(super) struct RolloutData {
    pub(super) reset_layout_batches_cpu: Vec<Tensor>,
    pub(super) reset_layout_count: i64,
    pub(super) reset_slots_host: Vec<i64>,
}

pub(super) struct AdvantageData {
    pub(super) advantages: Tensor,
    pub(super) returns: Tensor,
    pub(super) adv_stats: Tensor,
    pub(super) reset_layout_bank_cpu: Tensor,
    pub(super) reset_slots_by_chunk: Tensor,
    pub(super) reset_chunks_have_slots: Vec<bool>,
    pub(super) chunk_batch_size: i64,
    pub(super) reset_layout_count: i64,
}

pub(super) struct UpdateMetrics {
    pub(super) total_policy_loss_weighted: Tensor,
    pub(super) total_value_loss_weighted: Tensor,
    pub(super) total_spo_penalty_weighted: Tensor,
    pub(super) grad_norm_sum: Tensor,
    pub(super) total_sample_count: i64,
    pub(super) grad_norm_count: i64,
    pub(super) total_spo_bound_violations: Tensor,
    pub(super) total_ratio_samples: i64,
    pub(super) total_entropy_weighted: Tensor,
    pub(super) entropy_min: Tensor,
    pub(super) entropy_max: Tensor,
    pub(super) last_minibatch_approx_kl: f64,
}

pub(super) struct Trainer {
    pub(super) vs: nn::VarStore,
    pub(super) trading_model: TradingModel,
    pub(super) trainable_vars: Vec<Tensor>,
    pub(super) named_trainable_vars: Vec<(String, Tensor)>,
    pub(super) grad_clip_groups: GradClipGroups,
    pub(super) opt: Muon,
    pub(super) optimizer_step: i64,
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
}

impl Trainer {
    pub(super) fn new(
        weights_path: Option<&str>,
        model_variant: ModelVariant,
        run_name: Option<String>,
    ) -> Self {
        let rollout = rollout_geometry();
        assert_eq!(
            model_variant,
            ModelVariant::UniformStream,
            "PPO rollout collection is streaming-only; train with --model-size uniform-stream"
        );
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
        let mut vs = nn::VarStore::new(device);
        let trading_model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: model_variant,
                ..TradingModelConfig::default()
            },
        );

        let (start_episode, run_dir) = if let Some(path) = weights_path {
            println!("Loading weights from {}", path);
            let load_summary = load_var_store_partial(&mut vs, path).unwrap();
            load_summary.require_complete().unwrap();
            let ep = Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("ppo_ep"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            if ep > 0 {
                println!("Resuming from episode {}", ep);
            }
            // If weights path is inside runs/*/weights/, resume that run dir
            let p = Path::new(path);
            let is_run_weights = p
                .parent()
                .and_then(|d| d.file_name())
                .map(|n| n == "weights")
                .unwrap_or(false)
                && p.ancestors()
                    .any(|a| a.file_name().map(|n| n == "runs").unwrap_or(false));
            let rd = if is_run_weights {
                RunDir::from_weights_path(p).expect("failed to open run dir from weights path")
            } else {
                RunDir::create_fresh(RUNS_PATH, run_name.as_deref())
                    .expect("failed to create run dir")
            };
            (ep, rd)
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
        let opt = Muon::new_named(
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
                    "actor_live_proj".to_string(),
                    "critic_live_proj".to_string(),
                    "policy_concentration".to_string(),
                    "resid_mix".to_string(),
                    "value_proj".to_string(),
                ],
                ..MuonConfig::default()
            },
        );
        let optimizer_step = 0i64;

        let mut env = VecEnv::new(
            true,
            model_variant,
            gens_path.clone(),
            rollout.nprocs as usize,
        );
        if start_episode > 0 {
            env.set_episode(start_episode);
            env.primary_mut()
                .meta_history
                .load_from_episode(start_episode, &gens_path);
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

        Self {
            vs,
            trading_model,
            trainable_vars,
            named_trainable_vars,
            grad_clip_groups,
            opt,
            optimizer_step,
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
        }
    }

    pub(super) async fn run(&mut self) {
        for episode in self.start_episode..1000000 {
            let rollout_data = self.collect_rollout(episode);
            let advantage_data = self.compute_advantages(episode, &rollout_data);
            let update_metrics = self.update_policy(episode, &advantage_data);
            self.log_episode(episode, &advantage_data, &update_metrics);
            self.maybe_checkpoint(episode);
        }
    }

    pub(super) fn maybe_checkpoint(&self, episode: usize) {
        if episode > 0 && episode % 50 == 0 {
            let path = format!("{}/ppo_ep{}.ot", self.run_dir.weights.display(), episode);
            if let Err(err) = self.vs.save(&path) {
                println!("Error while saving weights: {}", err);
            } else {
                println!("Saved model weights: {}", path);
            }
        }
    }
}
