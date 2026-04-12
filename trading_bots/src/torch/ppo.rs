use rand::seq::SliceRandom;
use std::env;
use std::path::Path;
use std::time::Instant;
use tch::{autocast, nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::action_space::{
    sigmoid_target_weight, transformed_action_log_prob, transformed_action_log_prob_entropy_and_var,
};
use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::cuda_cfg::configure_cuda;
use crate::torch::env::{CpuStepBatch, VecEnv};
use crate::torch::hl_gauss::HlGaussBins;
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{ModelOutput, ModelVariant, TradingModel, TradingModelConfig};
use shared::{paths::RUNS_PATH, run_dir::RunDir};

const LEARNING_RATE: f64 = 3e-4;
pub const DEFAULT_NPROCS: i64 = 16;
const DEFAULT_SEQ_LEN: i64 = 4000;
const DEFAULT_TOTAL_SAMPLES: i64 = DEFAULT_NPROCS * DEFAULT_SEQ_LEN;
const DEFAULT_PPO_CHUNK_LEN: i64 = 50;
const DEFAULT_SUB_CHUNK_LEN: i64 = 25;
const DEFAULT_PPO_MINIBATCH_RATIO: f64 = 0.1;
const OPTIM_EPOCHS: i64 = 3;
const PPO_CLIP_LOW: f64 = 0.2;
const PPO_CLIP_HIGH: f64 = 0.2;
const TARGET_KL: f64 = 0.03;
const KL_STOP_MULTIPLIER: f64 = 1.5;
const VALUE_LOSS_COEF: f64 = 0.5;
const ENTROPY_COEF: f64 = 0.0;
const MAX_GRAD_NORM: f64 = 0.5;
pub(crate) const DEBUG_NUMERICS: bool = false;
const LOG_2PI: f64 = 1.8378770664093453;

// RPO: Random Policy Optimization - adds bounded noise to action mean during training and intentionally not during rollout
// Alpha is learned via induced KL targeting. Set all to 0.0 to disable.
const RPO_ALPHA_MIN: f64 = 0.01;
const RPO_ALPHA_MAX: f64 = 0.0;
const RPO_ALPHA_INIT: f64 = 0.0; // CleanRL impl found 0.1 reliably improved results in all test envs over PPO
const RPO_TARGET_KL: f64 = 0.018;
const ALPHA_LOSS_COEF: f64 = 0.1;
const MAX_DELTA_ALPHA: f64 = 0.2;

#[derive(Clone, Copy, Debug)]
struct RolloutGeometry {
    nprocs: i64,
    seq_len: i64,
    ppo_chunk_len: i64,
    sub_chunk_len: i64,
    total_samples: i64,
}

fn parse_positive_i64_env(name: &str) -> Option<i64> {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|&v| v > 0)
}

fn align_up(value: i64, multiple: i64) -> i64 {
    debug_assert!(multiple > 0);
    ((value + multiple - 1) / multiple) * multiple
}

fn rollout_geometry() -> RolloutGeometry {
    let nprocs = parse_positive_i64_env("PPO_NPROCS").unwrap_or(DEFAULT_NPROCS);
    let ppo_chunk_len = parse_positive_i64_env("PPO_CHUNK_LEN").unwrap_or(DEFAULT_PPO_CHUNK_LEN);
    let sub_chunk_len =
        parse_positive_i64_env("PPO_SUB_CHUNK_LEN").unwrap_or(DEFAULT_SUB_CHUNK_LEN);
    assert_eq!(
        ppo_chunk_len % sub_chunk_len,
        0,
        "PPO_CHUNK_LEN must be divisible by PPO_SUB_CHUNK_LEN"
    );

    let seq_len = if let Some(seq_len) = parse_positive_i64_env("PPO_SEQ_LEN") {
        align_up(seq_len.max(ppo_chunk_len), ppo_chunk_len)
    } else {
        let target_total_samples =
            parse_positive_i64_env("PPO_TOTAL_SAMPLES").unwrap_or(DEFAULT_TOTAL_SAMPLES);
        let target_seq_len = (target_total_samples + nprocs - 1) / nprocs;
        align_up(target_seq_len.max(ppo_chunk_len), ppo_chunk_len)
    };

    RolloutGeometry {
        nprocs,
        seq_len,
        ppo_chunk_len,
        sub_chunk_len,
        total_samples: nprocs * seq_len,
    }
}

fn minibatch_samples_from_total(total_samples: i64, nprocs: i64) -> i64 {
    let ratio = env::var("PPO_MINIBATCH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&v| v > 0.0 && v <= 1.0)
        .unwrap_or(DEFAULT_PPO_MINIBATCH_RATIO);
    let target = ((total_samples as f64) * ratio).round() as i64;
    let aligned = ((target + nprocs - 1) / nprocs).max(1) * nprocs;
    aligned.min(total_samples)
}

fn clip_grad_norm_on_device(
    trainable_vars: &[Tensor],
    max_grad_norm: f64,
    device: Device,
) -> Tensor {
    tch::no_grad(|| {
        let mut total_norm_sq = Tensor::zeros([], (Kind::Float, device));
        let mut has_grads = false;
        for v in trainable_vars {
            let g = v.grad();
            if g.defined() {
                total_norm_sq += g.square().sum(Kind::Float);
                has_grads = true;
            }
        }
        if !has_grads {
            return Tensor::zeros([], (Kind::Float, device));
        }

        let total_norm = total_norm_sq.sqrt();
        let clip_coef = Tensor::from(max_grad_norm as f32).to_device(device) / (&total_norm + 1e-6);
        let clip_coef = clip_coef.clamp_max(1.0);

        for v in trainable_vars {
            let mut g = v.grad();
            if g.defined() {
                let coef = clip_coef.to_kind(g.kind());
                let _ = g.g_mul_(&coef);
            }
        }

        total_norm
    })
}

fn debug_tensor_stats(name: &str, t: &Tensor, episode: i64, step: usize) -> bool {
    let has_nan = t.isnan().any().int64_value(&[]) != 0;
    let has_inf = t.isinf().any().int64_value(&[]) != 0;
    if has_nan || has_inf {
        let mean = t.mean(Kind::Float).double_value(&[]);
        let min = t.min().double_value(&[]);
        let max = t.max().double_value(&[]);
        println!(
            "Non-finite in {} at ep {} step {} nan={} inf={} mean={:.6} min={:.6} max={:.6}",
            name, episode, step, has_nan, has_inf, mean, min, max
        );
        return false;
    }
    true
}

fn sample_rollout_actions_from_output(
    output: ModelOutput,
    hl_gauss: &HlGaussBins,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let (value_logits, action_mean, action_std) = output;

    // Decode critic logits to scalar values for GAE.
    let values = hl_gauss.decode(&value_logits);

    let batch = action_mean.size()[0];
    let z = Tensor::randn(&[batch, ACTION_COUNT], (Kind::Float, action_mean.device()));
    let noise = &action_std * &z;
    let latent_actions = &action_mean + &noise;
    let target_weights = sigmoid_target_weight(&latent_actions);
    let action_log_prob =
        transformed_action_log_prob(&latent_actions, &action_mean, &action_std, LOG_2PI);

    (
        values,
        action_mean,
        action_std,
        latent_actions,
        target_weights,
        action_log_prob,
    )
}

/// Compute GAE advantages and returns from rollout data.
fn compute_gae(
    s_rewards: &Tensor,
    s_values: &Tensor,
    s_dones: &Tensor,
    bootstrap_value: &Tensor,
    rollout_steps: i64,
    nprocs: i64,
    gamma: f64,
    gae_lambda: f64,
    device: tch::Device,
) -> (Tensor, Tensor) {
    let memory_size = rollout_steps * nprocs;
    let advantages = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let returns = Tensor::zeros(&[memory_size], (Kind::Float, device));

    tch::no_grad(|| {
        let mut last_gae = Tensor::zeros(&[nprocs], (Kind::Float, device));
        for t in (0..rollout_steps).rev() {
            let mem_idx = t * nprocs;
            let next_values = if t == rollout_steps - 1 {
                bootstrap_value.shallow_clone()
            } else {
                s_values.narrow(0, (t + 1) * nprocs, nprocs)
            };
            let cur_values = s_values.narrow(0, mem_idx, nprocs);
            let rewards = s_rewards.narrow(0, mem_idx, nprocs);
            let dones = s_dones.narrow(0, mem_idx, nprocs);

            let delta = rewards + (1.0 - &dones) * gamma * &next_values - &cur_values;
            last_gae = delta + (1.0 - &dones) * gamma * gae_lambda * &last_gae;
            let _ = advantages.narrow(0, mem_idx, nprocs).copy_(&last_gae);
            let step_returns = &last_gae + &cur_values;
            let _ = returns.narrow(0, mem_idx, nprocs).copy_(&step_returns);
        }
    });

    (advantages.detach(), returns.detach())
}

/// Compute explained variance on a subset of the rollout.
/// EV using pre-training rollout values, matching CleanRL.
fn compute_explained_variance(rollout_values: &Tensor, returns: &Tensor) -> Tensor {
    tch::no_grad(|| {
        let residuals = rollout_values - returns;
        let mean_ret = returns.mean(Kind::Float);
        let var_ret = returns.square().mean(Kind::Float) - mean_ret.square();
        let var_resid = residuals.square().mean(Kind::Float);
        Tensor::from(1.0) - &var_resid / var_ret.clamp_min(1e-8)
    })
}

pub(crate) fn hl_gauss_value_loss(
    hl_gauss: &HlGaussBins,
    value_logits: &Tensor,
    returns: &Tensor,
) -> Tensor {
    let return_bins = hl_gauss.encode(returns);
    let log_probs = value_logits.log_softmax(-1, Kind::Float);
    -(&return_bins * &log_probs).sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

/// Compute action std and RPO alpha stats from a sample batch.
fn compute_action_std_stats(
    model: &TradingModel,
    price_deltas: &Tensor,
    static_obs: &Tensor,
    rpo_rho: &Tensor,
    device: tch::Device,
) -> Tensor {
    tch::no_grad(|| {
        let (_, _, action_std) = autocast(true, || {
            model.forward_on_device(price_deltas, static_obs, false)
        });
        let rpo_alpha_val = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
            (RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid()).squeeze()
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };
        Tensor::stack(
            &[
                action_std.mean(Kind::Float),
                action_std.min(),
                action_std.max(),
                rpo_alpha_val,
            ],
            0,
        )
    })
}

pub async fn train(
    weights_path: Option<&str>,
    model_variant: ModelVariant,
    run_name: Option<String>,
) {
    let rollout = rollout_geometry();
    assert_eq!(
        model_variant,
        ModelVariant::Uniform256Stream,
        "PPO rollout collection is streaming-only; train with --model-size uniform-256-stream"
    );
    assert_eq!(
        rollout.ppo_chunk_len % rollout.sub_chunk_len,
        0,
        "PPO_CHUNK_LEN must be divisible by PPO_SUB_CHUNK_LEN"
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
        "ppo rollout geometry: nprocs={} seq_len={} total_samples={} chunk_len={} sub_chunk_len={}",
        rollout.nprocs,
        rollout.seq_len,
        rollout.total_samples,
        rollout.ppo_chunk_len,
        rollout.sub_chunk_len
    );
    let mut vs = nn::VarStore::new(device);
    let trading_model = TradingModel::new_with_config(
        &vs.root(),
        TradingModelConfig {
            variant: model_variant,
            ..TradingModelConfig::default()
        },
    );

    // RPO alpha via sigmoid: alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(rho)
    // Keep rho outside VarStore so AdamW only updates model parameters.
    let mut rpo_rho = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
        let p_init = (RPO_ALPHA_INIT - RPO_ALPHA_MIN) / (RPO_ALPHA_MAX - RPO_ALPHA_MIN);
        let p_init = p_init.clamp(1e-6, 1.0 - 1e-6);
        let rho_init = (p_init / (1.0 - p_init)).ln();
        Tensor::full([1], rho_init, (Kind::Float, device)).set_requires_grad(true)
    } else {
        Tensor::zeros(&[1], (Kind::Float, device))
    };

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
            RunDir::create_fresh(RUNS_PATH, run_name.as_deref()).expect("failed to create run dir")
        };
        (ep, rd)
    } else {
        println!("Starting training from scratch");
        let rd =
            RunDir::create_fresh(RUNS_PATH, run_name.as_deref()).expect("failed to create run dir");
        (0, rd)
    };
    let gens_path = run_dir.gens.to_string_lossy().to_string();
    println!("Run dir: {}", run_dir.root.display());

    let mut opt = nn::AdamW::default()
        .wd(0.0)
        .eps(1e-5)
        .build(&vs, LEARNING_RATE)
        .expect("failed to build AdamW optimizer");
    let trainable_vars = vs.trainable_variables();

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
    let memory_size = rollout_steps * rollout.nprocs;
    assert_eq!(
        rollout_steps % rollout.ppo_chunk_len,
        0,
        "PPO_CHUNK_LEN must divide rollout length"
    );
    let chunks_per_rollout = rollout_steps / rollout.ppo_chunk_len;
    let total_chunks = chunks_per_rollout * rollout.nprocs;
    let mut chunk_mem_indices_host =
        Vec::with_capacity((total_chunks * rollout.ppo_chunk_len) as usize);
    for chunk_id in 0..total_chunks {
        let env_idx = chunk_id % rollout.nprocs;
        let chunk_start = (chunk_id / rollout.nprocs) * rollout.ppo_chunk_len;
        for offset in 0..rollout.ppo_chunk_len {
            chunk_mem_indices_host.push((chunk_start + offset) * rollout.nprocs + env_idx);
        }
    }
    let chunk_mem_indices = Tensor::from_slice(&chunk_mem_indices_host)
        .to_kind(Kind::Int64)
        .to_device(device)
        .view([total_chunks, rollout.ppo_chunk_len]);

    let raw_pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
    let pd_dim = trading_model.price_input_dim();
    let so_dim = STATIC_OBSERVATIONS as i64;
    let replay_obs_kind = trading_model.input_kind();
    let uniform_reset_live_fill = trading_model.uniform_stream_bootstrap_live_fill();

    let s_chunk_start_layouts = Tensor::zeros(
        &[chunks_per_rollout * rollout.nprocs, pd_dim],
        (replay_obs_kind, device),
    );
    let s_step_deltas = Tensor::zeros(&[memory_size, TICKERS_COUNT], (replay_obs_kind, device));
    let s_static_obs = Tensor::zeros(&[memory_size, so_dim], (replay_obs_kind, device));
    let s_actions = Tensor::zeros(&[memory_size, ACTION_COUNT], (Kind::Float, device));
    let s_old_log_probs = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let s_rewards = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let s_dones = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let s_values = Tensor::zeros(&[memory_size], (Kind::Float, device));

    if start_episode > 0 {
        let meta_path = format!(
            "{}/ppo_ep{}.rpo.json",
            run_dir.weights.display(),
            start_episode
        );
        let meta_path = if Path::new(&meta_path).exists() {
            meta_path
        } else {
            format!("../weights/ppo_ep{}.rpo.json", start_episode)
        };
        if let Ok(json) = std::fs::read_to_string(&meta_path) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) {
                if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    if let Some(rho) = parsed["rpo_rho"].as_f64() {
                        tch::no_grad(|| {
                            let _ =
                                rpo_rho.copy_(&Tensor::from_slice(&[rho as f32]).to_device(device));
                        });
                        println!("Loaded RPO rho: {:.6}", rho);
                    }
                }
            }
        }
    }

    for episode in start_episode..1000000 {
        let (obs_price_cpu, obs_static_cpu) = env.reset();
        let mut obs_price = Tensor::zeros(&[rollout.nprocs, raw_pd_dim], (Kind::Float, device));
        let mut obs_static = Tensor::zeros(
            &[rollout.nprocs, STATIC_OBSERVATIONS as i64],
            (Kind::Float, device),
        );
        let ring_len = PRICE_DELTAS_PER_TICKER as i64;
        let base_idx = Tensor::arange(ring_len, (Kind::Int64, device));
        let mut ring_pos = ring_len - 1;
        let mut ring_buf = Tensor::zeros(
            &[rollout.nprocs, TICKERS_COUNT, ring_len],
            (Kind::Float, device),
        );
        let mut step_deltas =
            Tensor::zeros(&[rollout.nprocs, TICKERS_COUNT], (Kind::Float, device));
        obs_price.copy_(&obs_price_cpu);
        obs_static.copy_(&obs_static_cpu);
        ring_buf.copy_(&obs_price.view([rollout.nprocs, TICKERS_COUNT, ring_len]));
        let mut stream_state = trading_model.init_stream_state_batched(rollout.nprocs);
        let stream_layout = trading_model.uniform_stream_layout_from_raw_input(&obs_price);
        let mut streamed_output = Some(tch::no_grad(|| {
            autocast(true, || {
                trading_model.step_on_device(&stream_layout, &obs_static, &mut stream_state)
            })
        }));
        let mut step_reward_per_ticker =
            Tensor::zeros(&[rollout.nprocs, TICKERS_COUNT], (Kind::Float, device));
        let mut step_is_done = Tensor::zeros(&[rollout.nprocs], (Kind::Float, device));
        let mut reset_layout_batches: Vec<Tensor> = Vec::new();
        let mut reset_layout_count = 0i64;
        let mut reset_slots_host = vec![0i64; memory_size as usize];
        let mut cpu_step_batch = CpuStepBatch::new(
            rollout.nprocs as usize,
            ACTION_COUNT as usize,
            raw_pd_dim as usize,
        );
        for step in 0..rollout_steps as usize {
            let mem_idx = step as i64 * rollout.nprocs;
            let (values, action_mean, action_std, latent_actions, target_weights, action_log_prob) =
                sample_rollout_actions_from_output(
                    streamed_output
                        .take()
                        .expect("streamed rollout output missing"),
                    &hl_gauss,
                );

            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("action_mean", &action_mean, episode as i64, step);
                let _ = debug_tensor_stats("action_std", &action_std, episode as i64, step);
                let _ = debug_tensor_stats("latent_actions", &latent_actions, episode as i64, step);
                let _ = debug_tensor_stats("target_weights", &target_weights, episode as i64, step);
                let _ =
                    debug_tensor_stats("action_log_prob", &action_log_prob, episode as i64, step);
            }
            if step as i64 % rollout.ppo_chunk_len == 0 {
                let chunk_row = (step as i64 / rollout.ppo_chunk_len) * rollout.nprocs;
                let boundary_layout = stream_state
                    .uniform_layout
                    .view([rollout.nprocs, pd_dim])
                    .to_kind(replay_obs_kind);
                let _ = s_chunk_start_layouts
                    .narrow(0, chunk_row, rollout.nprocs)
                    .copy_(&boundary_layout);
            }
            let _ = s_static_obs
                .narrow(0, mem_idx, rollout.nprocs)
                .copy_(&obs_static);

            let mut action_host_view = unsafe {
                Tensor::from_blob(
                    cpu_step_batch.actions_f32.as_ptr() as *const u8,
                    &[rollout.nprocs, ACTION_COUNT],
                    &[],
                    Kind::Float,
                    Device::Cpu,
                )
            };
            let _ = action_host_view.copy_(&target_weights.to_kind(Kind::Float));
            env.step_from_actions_f32_into(
                &mut cpu_step_batch,
                &mut step_deltas,
                &mut obs_static,
                &mut step_reward_per_ticker,
                &mut step_is_done,
            );
            let reset_indices = &cpu_step_batch.reset_indices;
            let reset_price_deltas = &cpu_step_batch.reset_price_deltas;
            let _ = s_step_deltas
                .narrow(0, mem_idx, rollout.nprocs)
                .copy_(&step_deltas);

            ring_pos = (ring_pos + 1) % ring_len;
            let _ = ring_buf
                .narrow(2, ring_pos, 1)
                .copy_(&step_deltas.unsqueeze(-1));

            let idx = (&base_idx + (ring_pos + 1)).remainder(ring_len);
            if !reset_indices.is_empty() {
                let reset_count = reset_indices.len() as i64;
                let reset_raw_batch = Tensor::from_slice(reset_price_deltas)
                    .view([reset_count, raw_pd_dim])
                    .to_device(device);
                let reset_layouts_batch =
                    trading_model.uniform_stream_layout_from_raw_input(&reset_raw_batch);
                let reset_ring_ordered =
                    reset_raw_batch.view([reset_count, TICKERS_COUNT, ring_len]);
                let mut reset_ring_batch = Tensor::zeros(
                    &[reset_count, TICKERS_COUNT, ring_len],
                    (Kind::Float, device),
                );
                let _ = reset_ring_batch.index_copy_(2, &idx, &reset_ring_ordered);
                let reset_env_indices: Vec<i64> = reset_indices
                    .iter()
                    .map(|&env_idx| env_idx as i64)
                    .collect();
                let reset_env_tensor = Tensor::from_slice(&reset_env_indices)
                    .to_kind(Kind::Int64)
                    .to_device(device);
                let _ = ring_buf.index_copy_(0, &reset_env_tensor, &reset_ring_batch);
                for (reset_i, env_idx) in reset_indices.iter().enumerate() {
                    reset_slots_host[(mem_idx + *env_idx as i64) as usize] =
                        reset_layout_count + reset_i as i64 + 1;
                }
                reset_layout_count += reset_count;
                reset_layout_batches.push(reset_layouts_batch.to_kind(replay_obs_kind));
            }

            if !reset_indices.is_empty() {
                let ordered = ring_buf.index_select(2, &idx);
                obs_price.copy_(&ordered.view([rollout.nprocs, raw_pd_dim]));
            }

            let _ = s_actions
                .narrow(0, mem_idx, rollout.nprocs)
                .copy_(&latent_actions);
            let _ = s_old_log_probs
                .narrow(0, mem_idx, rollout.nprocs)
                .copy_(&action_log_prob);

            let portfolio_reward =
                step_reward_per_ticker.mean_dim([1].as_slice(), false, Kind::Float);

            if DEBUG_NUMERICS {
                let _ =
                    debug_tensor_stats("portfolio_reward", &portfolio_reward, episode as i64, step);
                let _ = debug_tensor_stats("values", &values, episode as i64, step);
                let _ = debug_tensor_stats("step_is_done", &step_is_done, episode as i64, step);
            }
            let _ = s_rewards
                .narrow(0, mem_idx, rollout.nprocs)
                .copy_(&portfolio_reward);
            let _ = s_dones
                .narrow(0, mem_idx, rollout.nprocs)
                .copy_(&step_is_done);
            let _ = s_values.narrow(0, mem_idx, rollout.nprocs).copy_(&values);

            streamed_output = Some(tch::no_grad(|| {
                autocast(true, || {
                    if reset_indices.is_empty() {
                        trading_model.step_on_device(&step_deltas, &obs_static, &mut stream_state)
                    } else {
                        let stream_layout =
                            trading_model.uniform_stream_layout_from_raw_input(&obs_price);
                        trading_model.step_on_device(&stream_layout, &obs_static, &mut stream_state)
                    }
                })
            }));
        }

        // Bootstrap value from final observation state (decode two-hot logits)
        let bootstrap_value = tch::no_grad(|| {
            let bootstrap_price = trading_model.uniform_stream_snapshot(&stream_state);
            let (value_logits, _, _) = autocast(true, || {
                trading_model.forward_on_device(&bootstrap_price, &obs_static, false)
            });
            hl_gauss.decode(&value_logits)
        });

        let (advantages, returns) = compute_gae(
            &s_rewards,
            &s_values,
            &s_dones,
            &bootstrap_value,
            rollout_steps,
            rollout.nprocs,
            0.99,
            0.95,
            device,
        );
        let reset_layout_bank = if reset_layout_batches.is_empty() {
            Tensor::zeros(&[0, pd_dim], (replay_obs_kind, device))
        } else {
            let reset_layout_refs: Vec<&Tensor> = reset_layout_batches.iter().collect();
            Tensor::cat(&reset_layout_refs, 0)
        };

        // Compute advantage stats once per rollout (before normalization)
        let adv_stats = tch::no_grad(|| {
            Tensor::stack(
                &[
                    advantages.mean(Kind::Float),
                    advantages.min(),
                    advantages.max(),
                ],
                0,
            )
        });

        let total_samples = rollout_steps * rollout.nprocs;
        let minibatch_size = minibatch_samples_from_total(total_samples, rollout.nprocs);
        let chunk_batch_size =
            ((minibatch_size + rollout.ppo_chunk_len - 1) / rollout.ppo_chunk_len).max(1);
        // Per-chunk reset slot ids laid out as [total_chunks, chunk_len], matching
        // `chunk_mem_indices_host` ordering. Entry 0 == no reset; >0 == 1-indexed slot.
        // Precomputed host-side so the inner sub-chunk loop never needs a device->host
        // sync to decide whether a reset occurred at the current step.
        let mut resets_by_chunk_flat: Vec<i64> =
            Vec::with_capacity((total_chunks * rollout.ppo_chunk_len) as usize);
        for &mem_idx in &chunk_mem_indices_host {
            resets_by_chunk_flat.push(reset_slots_host[mem_idx as usize]);
        }
        println!(
            "ppo update: total_samples={} minibatch_size={} chunk_len={} chunk_batch={}",
            total_samples, minibatch_size, rollout.ppo_chunk_len, chunk_batch_size
        );
        // Full-rollout advantage normalization (not per-minibatch)
        let adv_norm = (&advantages - advantages.mean(Kind::Float)) / (advantages.std(true) + 1e-8);
        let chunk_mem_indices_flat = chunk_mem_indices.reshape([-1]);
        let s_static_obs_by_chunk = s_static_obs.index_select(0, &chunk_mem_indices_flat).view([
            total_chunks,
            rollout.ppo_chunk_len,
            so_dim,
        ]);
        let s_step_deltas_by_chunk = s_step_deltas
            .index_select(0, &chunk_mem_indices_flat)
            .view([total_chunks, rollout.ppo_chunk_len, TICKERS_COUNT]);
        let s_actions_by_chunk = s_actions.index_select(0, &chunk_mem_indices_flat).view([
            total_chunks,
            rollout.ppo_chunk_len,
            ACTION_COUNT,
        ]);
        let s_old_log_probs_by_chunk = s_old_log_probs
            .index_select(0, &chunk_mem_indices_flat)
            .view([total_chunks, rollout.ppo_chunk_len]);
        let returns_by_chunk = returns
            .index_select(0, &chunk_mem_indices_flat)
            .view([total_chunks, rollout.ppo_chunk_len]);
        let adv_norm_by_chunk = adv_norm
            .index_select(0, &chunk_mem_indices_flat)
            .view([total_chunks, rollout.ppo_chunk_len]);

        let mut total_kl_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_policy_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_value_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        // Explained variance: EV = 1 - Var(residuals) / Var(targets)
        let mut grad_norm_sum = Tensor::zeros([], (Kind::Float, device));
        let mut total_sample_count = 0i64;
        let mut grad_norm_count = 0i64;
        let mut total_clipped = Tensor::zeros([], (Kind::Float, device));
        let mut total_ratio_samples = 0i64;
        let mut total_entropy_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut entropy_min = Tensor::from(f64::INFINITY)
            .to_kind(Kind::Float)
            .to_device(device);
        let mut entropy_max = Tensor::from(f64::NEG_INFINITY)
            .to_kind(Kind::Float)
            .to_device(device);

        let mut fwd_time_us = 0u64;
        let mut bwd_time_us = 0u64;

        let mut last_minibatch_approx_kl = 0.0f64;
        let mut perm_host: Vec<i64> = (0..total_chunks).collect();
        let mut perm_gpu = Tensor::zeros([total_chunks], (Kind::Int64, device));
        let mut mb_reset_rows: Vec<Vec<usize>> = (0..rollout.ppo_chunk_len as usize)
            .map(|_| Vec::new())
            .collect();
        let mut mb_reset_slots: Vec<Vec<i64>> = (0..rollout.ppo_chunk_len as usize)
            .map(|_| Vec::new())
            .collect();
        let mut rng = rand::rng();

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            perm_host.shuffle(&mut rng);
            let perm_cpu = Tensor::from_slice(&perm_host)
                .to_kind(Kind::Int64)
                .to_device(device);
            perm_gpu.copy_(&perm_cpu);

            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;

            for (chunk_i, mb_start) in (0..total_chunks)
                .step_by(chunk_batch_size as usize)
                .enumerate()
            {
                let mb_end = (mb_start + chunk_batch_size).min(total_chunks);
                let chunk_count = mb_end - mb_start;
                let chunk_ids_host = &perm_host[mb_start as usize..mb_end as usize];
                let chunk_ids = perm_gpu.narrow(0, mb_start, chunk_count);
                let boundary_layout = s_chunk_start_layouts.index_select(0, &chunk_ids);
                let so_chunk = s_static_obs_by_chunk.index_select(0, &chunk_ids);
                let step_deltas_chunk = s_step_deltas_by_chunk.index_select(0, &chunk_ids);
                let adv_mb_by_chunk = adv_norm_by_chunk.index_select(0, &chunk_ids);
                let ret_mb_by_chunk = returns_by_chunk.index_select(0, &chunk_ids);
                let old_log_probs_by_chunk = s_old_log_probs_by_chunk.index_select(0, &chunk_ids);
                let act_mb_by_chunk = s_actions_by_chunk.index_select(0, &chunk_ids);

                for rows in &mut mb_reset_rows {
                    rows.clear();
                }
                for slots in &mut mb_reset_slots {
                    slots.clear();
                }
                for (row_in_mb, &chunk_id) in chunk_ids_host.iter().enumerate() {
                    let base = (chunk_id * rollout.ppo_chunk_len) as usize;
                    for pos in 0..rollout.ppo_chunk_len as usize {
                        let slot = resets_by_chunk_flat[base + pos];
                        if slot > 0 {
                            mb_reset_rows[pos].push(row_in_mb);
                            mb_reset_slots[pos].push(slot - 1);
                        }
                    }
                }
                let mut mb_reset_env_idx_tensors =
                    Vec::with_capacity(rollout.ppo_chunk_len as usize);
                let mut mb_reset_row_idx_tensors =
                    Vec::with_capacity(rollout.ppo_chunk_len as usize);
                let mut mb_reset_live_fill_tensors =
                    Vec::with_capacity(rollout.ppo_chunk_len as usize);
                let mut mb_reset_layouts = Vec::with_capacity(rollout.ppo_chunk_len as usize);
                for pos in 0..rollout.ppo_chunk_len as usize {
                    let rows = &mb_reset_rows[pos];
                    if rows.is_empty() {
                        mb_reset_env_idx_tensors.push(None);
                        mb_reset_row_idx_tensors.push(None);
                        mb_reset_live_fill_tensors.push(None);
                        mb_reset_layouts.push(None);
                        continue;
                    }
                    let env_idx_host: Vec<i64> = rows.iter().map(|&row| row as i64).collect();
                    let row_idx_host: Vec<i64> = rows
                        .iter()
                        .flat_map(|&row| {
                            (0..TICKERS_COUNT)
                                .map(move |ticker_idx| row as i64 * TICKERS_COUNT + ticker_idx)
                        })
                        .collect();
                    let reset_slot_ids = Tensor::from_slice(&mb_reset_slots[pos])
                        .to_kind(Kind::Int64)
                        .to_device(device);
                    mb_reset_env_idx_tensors.push(Some(
                        Tensor::from_slice(&env_idx_host)
                            .to_kind(Kind::Int64)
                            .to_device(device),
                    ));
                    mb_reset_row_idx_tensors.push(Some(
                        Tensor::from_slice(&row_idx_host)
                            .to_kind(Kind::Int64)
                            .to_device(device),
                    ));
                    mb_reset_live_fill_tensors.push(Some(Tensor::full(
                        [rows.len() as i64],
                        uniform_reset_live_fill,
                        (Kind::Int64, device),
                    )));
                    mb_reset_layouts.push(Some(reset_layout_bank.index_select(0, &reset_slot_ids)));
                }

                let mut fwd_start = Instant::now();
                let mut chunk_state = trading_model.init_stream_state_batched(chunk_count);
                let num_sub_chunks = rollout.ppo_chunk_len / rollout.sub_chunk_len;
                let minibatch_sample_count = chunk_count * rollout.ppo_chunk_len;
                let sub_chunk_sample_count = chunk_count * rollout.sub_chunk_len;
                let mut global_pos: i64 = 0;
                let mut minibatch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
                let mut minibatch_kl_count = 0i64;

                for sub_idx in 0..num_sub_chunks {
                    let mut sub_value_logits: Vec<Tensor> =
                        Vec::with_capacity(rollout.sub_chunk_len as usize);
                    let mut sub_action_mean: Vec<Tensor> =
                        Vec::with_capacity(rollout.sub_chunk_len as usize);
                    let mut sub_action_std: Vec<Tensor> =
                        Vec::with_capacity(rollout.sub_chunk_len as usize);
                    let sub_start = sub_idx * rollout.sub_chunk_len;

                    for _ in 0..rollout.sub_chunk_len {
                        let output = if global_pos == 0 {
                            let first_static = so_chunk.select(1, 0);
                            autocast(true, || {
                                trading_model.step_on_device(
                                    &boundary_layout,
                                    &first_static,
                                    &mut chunk_state,
                                )
                            })
                        } else {
                            let prev_step_deltas = step_deltas_chunk.select(1, global_pos - 1);
                            let current_static = so_chunk.select(1, global_pos);
                            let step_reset_rows = &mb_reset_rows[(global_pos - 1) as usize];
                            if step_reset_rows.is_empty() {
                                autocast(true, || {
                                    trading_model.step_on_device(
                                        &prev_step_deltas,
                                        &current_static,
                                        &mut chunk_state,
                                    )
                                })
                            } else {
                                let _ = autocast(true, || {
                                    trading_model.step_on_device(
                                        &prev_step_deltas,
                                        &current_static,
                                        &mut chunk_state,
                                    )
                                });
                                let reset_pos = (global_pos - 1) as usize;
                                trading_model.reset_uniform_stream_envs_from_layout_indexed(
                                    &mut chunk_state,
                                    step_reset_rows,
                                    mb_reset_env_idx_tensors[reset_pos].as_ref().unwrap(),
                                    mb_reset_row_idx_tensors[reset_pos].as_ref().unwrap(),
                                    mb_reset_layouts[reset_pos].as_ref().unwrap(),
                                    mb_reset_live_fill_tensors[reset_pos].as_ref().unwrap(),
                                    uniform_reset_live_fill,
                                );
                                autocast(true, || {
                                    trading_model.forward_stream_state_on_device(
                                        &current_static,
                                        &mut chunk_state,
                                    )
                                })
                            }
                        };
                        sub_value_logits.push(output.0);
                        sub_action_mean.push(output.1);
                        sub_action_std.push(output.2);
                        global_pos += 1;
                    }

                    let new_value_logits =
                        Tensor::stack(&sub_value_logits, 1).reshape([sub_chunk_sample_count, -1]);
                    let action_mean = Tensor::stack(&sub_action_mean, 1)
                        .reshape([sub_chunk_sample_count, ACTION_COUNT]);
                    let action_std = Tensor::stack(&sub_action_std, 1)
                        .reshape([sub_chunk_sample_count, ACTION_COUNT]);

                    // Slice sub-chunk-matching rows from the full-minibatch flat tensors.
                    // Layout is chunk-major: [chunk_count, chunk_len, ...] flattened.
                    let sub_adv = adv_mb_by_chunk
                        .narrow(1, sub_start, rollout.sub_chunk_len)
                        .reshape([-1]);
                    let sub_ret = ret_mb_by_chunk
                        .narrow(1, sub_start, rollout.sub_chunk_len)
                        .reshape([-1]);
                    let sub_old_log_probs = old_log_probs_by_chunk
                        .narrow(1, sub_start, rollout.sub_chunk_len)
                        .reshape([-1]);
                    let sub_act = act_mb_by_chunk
                        .narrow(1, sub_start, rollout.sub_chunk_len)
                        .reshape([-1, ACTION_COUNT]);

                    let latent_actions_mb = sub_act;

                    // RPO (CleanRL-style): iid uniform perturbation on each action-mean dimension.
                    let (rpo_alpha, action_mean_perturbed) = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                        let alpha =
                            RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                        let alpha_detached = alpha.detach();
                        let rpo_noise = Tensor::empty(
                            [sub_chunk_sample_count, ACTION_COUNT],
                            (Kind::Float, device),
                        )
                        .uniform_(-1.0, 1.0)
                            * &alpha_detached;
                        (alpha, &action_mean + rpo_noise)
                    } else {
                        (
                            Tensor::zeros(&[1], (Kind::Float, device)),
                            action_mean.shallow_clone(),
                        )
                    };

                    let (action_log_probs, dist_entropy_per_sample, action_var) =
                        transformed_action_log_prob_entropy_and_var(
                            &latent_actions_mb,
                            &action_mean_perturbed,
                            &action_std,
                            LOG_2PI,
                        );

                    if DEBUG_NUMERICS {
                        let _ = debug_tensor_stats(
                            "latent_actions_mb",
                            &latent_actions_mb,
                            _epoch,
                            chunk_i,
                        );
                        let _ = debug_tensor_stats(
                            "old_log_probs_mb",
                            &sub_old_log_probs,
                            _epoch,
                            chunk_i,
                        );
                        let _ = debug_tensor_stats("action_mean", &action_mean, _epoch, chunk_i);
                        let _ = debug_tensor_stats("action_std", &action_std, _epoch, chunk_i);
                    }

                    let log_ratio = &action_log_probs - &sub_old_log_probs;

                    if DEBUG_NUMERICS {
                        let _ = debug_tensor_stats(
                            "action_log_probs",
                            &action_log_probs,
                            _epoch,
                            chunk_i,
                        );
                        let _ = debug_tensor_stats("log_ratio", &log_ratio, _epoch, chunk_i);
                    }
                    let ratio = log_ratio.exp();
                    let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_LOW, 1.0 + PPO_CLIP_HIGH);

                    // Single portfolio advantage - no more weighted combination
                    let action_loss =
                        -Tensor::min_other(&(&ratio * &sub_adv), &(&ratio_clipped * &sub_adv))
                            .mean(Kind::Float);

                    // Two-hot distributional value loss
                    if DEBUG_NUMERICS {
                        let _ = debug_tensor_stats("ret_mb", &sub_ret, _epoch, chunk_i);
                        let _ = debug_tensor_stats(
                            "new_value_logits",
                            &new_value_logits,
                            _epoch,
                            chunk_i,
                        );
                        let _ = debug_tensor_stats("adv_mb", &sub_adv, _epoch, chunk_i);
                    }

                    let value_loss = hl_gauss_value_loss(&hl_gauss, &new_value_logits, &sub_ret)
                        .mean(Kind::Float);

                    let dist_entropy = dist_entropy_per_sample.mean(Kind::Float);
                    let dist_entropy_detached = dist_entropy.detach();

                    let sub_ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                        + action_loss.shallow_clone()
                        - &dist_entropy * ENTROPY_COEF;

                    // RPO alpha loss
                    let alpha_loss = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                        let inv_var_mean = action_var
                            .detach()
                            .clamp_min(1e-4)
                            .reciprocal()
                            .mean(Kind::Float);
                        let d = ACTION_COUNT as f64;
                        let induced_kl = rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                        (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0) * ALPHA_LOSS_COEF
                    } else {
                        Tensor::zeros([], (Kind::Float, device))
                    };

                    // Scale loss by this sub-chunk's fraction of the full minibatch so that
                    // gradients accumulated across all sub-chunks of all chunks equal the
                    // gradient that a single mean-over-minibatch backward would produce.
                    let scale = (sub_chunk_sample_count as f64) / (minibatch_sample_count as f64);
                    let scaled_sub_loss = (sub_ppo_loss + alpha_loss) * scale;

                    fwd_time_us += fwd_start.elapsed().as_micros() as u64;
                    let bwd_start = Instant::now();
                    scaled_sub_loss.backward();
                    bwd_time_us += bwd_start.elapsed().as_micros() as u64;

                    trading_model.detach_stream_state(&mut chunk_state);
                    fwd_start = Instant::now();

                    let approx_kl_val =
                        tch::no_grad(|| (log_ratio.exp() - 1.0 - &log_ratio).mean(Kind::Float));
                    if DEBUG_NUMERICS {
                        let _ =
                            debug_tensor_stats("approx_kl_val", &approx_kl_val, _epoch, chunk_i);
                    }
                    let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * sub_chunk_sample_count as f64));
                    let _ =
                        minibatch_kl_gpu.g_add_(&(&approx_kl_val * sub_chunk_sample_count as f64));
                    let _ = total_policy_loss_weighted
                        .g_add_(&(&action_loss.detach() * sub_chunk_sample_count as f64));
                    let _ = total_value_loss_weighted
                        .g_add_(&(&value_loss.detach() * sub_chunk_sample_count as f64));
                    let _ =
                        total_kl_weighted.g_add_(&(&approx_kl_val * sub_chunk_sample_count as f64));
                    let _ = total_entropy_weighted
                        .g_add_(&(&dist_entropy_detached * sub_chunk_sample_count as f64));
                    entropy_min = entropy_min.min_other(&dist_entropy_detached);
                    entropy_max = entropy_max.max_other(&dist_entropy_detached);
                    epoch_kl_count += sub_chunk_sample_count;
                    minibatch_kl_count += sub_chunk_sample_count;
                    total_sample_count += sub_chunk_sample_count;

                    let _ = total_clipped.g_add_(&tch::no_grad(|| {
                        let dev = &ratio - 1.0;
                        let clipped_lo = dev.le(-PPO_CLIP_LOW).to_kind(Kind::Float);
                        let clipped_hi = dev.ge(PPO_CLIP_HIGH).to_kind(Kind::Float);
                        (clipped_lo + clipped_hi).sum(Kind::Float)
                    }));
                    total_ratio_samples += sub_chunk_sample_count;
                }

                if DEBUG_NUMERICS {
                    let has_nan_grad = tch::no_grad(|| {
                        let mut found = false;
                        for v in &trainable_vars {
                            let g = v.grad();
                            if g.defined()
                                && (g.isnan().any().int64_value(&[]) != 0
                                    || g.isinf().any().int64_value(&[]) != 0)
                            {
                                found = true;
                                break;
                            }
                        }
                        found
                    });
                    if has_nan_grad {
                        println!("ERROR: Non-finite gradients detected!");
                    }
                }

                let batch_grad_norm =
                    clip_grad_norm_on_device(&trainable_vars, MAX_GRAD_NORM, device);
                grad_norm_sum += &batch_grad_norm;
                grad_norm_count += 1;

                opt.step();
                if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let max_delta_rho = MAX_DELTA_ALPHA / (0.25 * (RPO_ALPHA_MAX - RPO_ALPHA_MIN));
                    tch::no_grad(|| {
                        let mut rho_grad = rpo_rho.grad();
                        if rho_grad.defined() {
                            let rho_step =
                                (-LEARNING_RATE * &rho_grad).clamp(-max_delta_rho, max_delta_rho);
                            let _ = rpo_rho.g_add_(&rho_step);
                            let _ = rho_grad.zero_();
                        }
                    });
                }
                opt.zero_grad();

                last_minibatch_approx_kl =
                    minibatch_kl_gpu.double_value(&[]) / minibatch_kl_count as f64;
            }

            let mean_epoch_kl = epoch_kl_gpu.double_value(&[]) / epoch_kl_count as f64;
            println!(
                "Epoch {}/{}: KL {:.4}",
                _epoch + 1,
                OPTIM_EPOCHS,
                mean_epoch_kl
            );
            if mean_epoch_kl > TARGET_KL * KL_STOP_MULTIPLIER {
                break 'epoch_loop;
            }
        }

        println!(
            "fwd: {:.1}ms  bwd: {:.1}ms",
            fwd_time_us as f64 / 1000.0,
            bwd_time_us as f64 / 1000.0
        );

        let max_param_norm = tch::no_grad(|| {
            let norms: Vec<Tensor> = trainable_vars.iter().map(|v| v.norm()).collect();
            if norms.is_empty() {
                0.0f64
            } else {
                Tensor::stack(&norms, 0).max().double_value(&[])
            }
        });
        if max_param_norm > 1000.0 {
            println!(
                "WARNING: Large parameter norm detected: {:.2}",
                max_param_norm
            );
        }

        // Compute all metrics on GPU, single transfer to CPU
        let (mean_policy_loss_t, mean_value_loss_t, mean_grad_norm_t, clip_frac_t) =
            if total_sample_count > 0 {
                let n = total_sample_count as f64;
                let mean_policy = &total_policy_loss_weighted / n;
                let mean_value = &total_value_loss_weighted / n;
                let grad_norm = if grad_norm_count > 0 {
                    &grad_norm_sum / (grad_norm_count as f64)
                } else {
                    Tensor::zeros([], (Kind::Float, device))
                };
                let clip = if total_ratio_samples > 0 {
                    &total_clipped / (total_ratio_samples as f64)
                } else {
                    Tensor::zeros([], (Kind::Float, device))
                };
                (mean_policy, mean_value, grad_norm, clip)
            } else {
                (
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                )
            };

        let explained_var_t = if total_sample_count > 0 {
            compute_explained_variance(&s_values, &returns)
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };

        let entropy_mean_t = if total_sample_count > 0 {
            &total_entropy_weighted / total_sample_count as f64
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };

        let log_std_stats = compute_action_std_stats(
            &trading_model,
            &s_chunk_start_layouts.narrow(0, 0, rollout.nprocs),
            &s_static_obs.narrow(0, 0, rollout.nprocs),
            &rpo_rho,
            device,
        );
        let return_range_stats = hl_gauss.range_stats(&returns);
        // Single GPU->CPU transfer for all scalars
        let all_scalars = Tensor::cat(
            &[
                mean_policy_loss_t.view([1]),
                mean_value_loss_t.view([1]),
                explained_var_t.view([1]),
                mean_grad_norm_t.view([1]),
                clip_frac_t.view([1]),
                adv_stats.view([3]),
                log_std_stats.view([4]),
                entropy_mean_t.view([1]),
                entropy_min.view([1]),
                entropy_max.view([1]),
                return_range_stats.view([6]),
            ],
            0,
        );
        let all_scalars_vec: Vec<f64> = Vec::try_from(all_scalars.to_device(tch::Device::Cpu))
            .unwrap_or_else(|_| vec![0.0; 21]);
        let mean_policy_loss = all_scalars_vec[0];
        let mean_value_loss = all_scalars_vec[1];
        let explained_var = all_scalars_vec[2];
        let mean_grad_norm = all_scalars_vec[3];
        let clip_frac = all_scalars_vec[4];
        let (adv_mean, adv_min, adv_max) =
            (all_scalars_vec[5], all_scalars_vec[6], all_scalars_vec[7]);
        let log_std_stats_vec = &all_scalars_vec[8..12];
        let (entropy_mean, entropy_min_val, entropy_max_val) = (
            all_scalars_vec[12],
            all_scalars_vec[13],
            all_scalars_vec[14],
        );
        let (
            return_min,
            return_max,
            support_min,
            support_max,
            below_support_frac,
            above_support_frac,
        ) = (
            all_scalars_vec[15],
            all_scalars_vec[16],
            all_scalars_vec[17],
            all_scalars_vec[18],
            all_scalars_vec[19],
            all_scalars_vec[20],
        );

        let primary = env.primary_mut();
        primary
            .meta_history
            .record_advantage_stats(adv_mean, adv_min, adv_max);
        // Record action std stats in place of logit noise stats
        primary.meta_history.record_logit_noise_stats(
            log_std_stats_vec[0],
            log_std_stats_vec[1],
            log_std_stats_vec[2],
            log_std_stats_vec[3],
        );
        primary.meta_history.record_clip_fraction(clip_frac);
        primary.meta_history.record_policy_loss(mean_policy_loss);
        primary.meta_history.record_value_loss(mean_value_loss);
        primary.meta_history.record_explained_var(explained_var);
        primary.meta_history.record_grad_norm(mean_grad_norm);
        primary
            .meta_history
            .record_policy_entropy(entropy_mean, entropy_min_val, entropy_max_val);
        primary
            .meta_history
            .record_approx_kl(last_minibatch_approx_kl);
        primary.meta_history.record_hl_gauss_range_stats(
            return_min,
            return_max,
            support_min,
            support_max,
            below_support_frac,
            above_support_frac,
        );

        println!(
            "  Policy: {:.4}, Value: {:.4} (EV: {:.3}), GradNorm: {:.4}",
            mean_policy_loss, mean_value_loss, explained_var, mean_grad_norm
        );

        if episode > 0 && episode % 50 == 0 {
            let path = format!("{}/ppo_ep{}.ot", run_dir.weights.display(), episode);
            if let Err(err) = vs.save(&path) {
                println!("Error while saving weights: {}", err);
            } else {
                println!("Saved model weights: {}", path);
                let meta_path = format!("{}/ppo_ep{}.rpo.json", run_dir.weights.display(), episode);
                let json = serde_json::json!({
                    "rpo_rho": if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                        Some(rpo_rho.double_value(&[]))
                    } else {
                        None
                    },
                });
                if let Err(err) = std::fs::write(&meta_path, json.to_string()) {
                    println!("Error saving RPO state: {}", err);
                }
            }
        }
    }
}
