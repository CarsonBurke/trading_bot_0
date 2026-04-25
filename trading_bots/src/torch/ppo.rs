use rand::seq::SliceRandom;
use std::env;
use std::path::Path;
use std::time::Instant;
use tch::{autocast, nn, Device, Kind, Tensor};

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
use crate::torch::muon::{Muon, MuonConfig};
use shared::{paths::RUNS_PATH, run_dir::RunDir};

/// NorMuon LR for 2D weight matrices (orthogonalized updates, RMS-match scaling).
const MUON_LR: f64 = 3e-4;
/// AdamW LR for 1D params (biases, norms) and the standalone rho scalar.
const LEARNING_RATE: f64 = 3e-4;
const USE_MUON: bool = true;
pub const DEFAULT_NPROCS: i64 = 16;
const DEFAULT_SEQ_LEN: i64 = 2000;
const DEFAULT_TOTAL_SAMPLES: i64 = DEFAULT_NPROCS * DEFAULT_SEQ_LEN;
const DEFAULT_PPO_CHUNK_LEN: i64 = 60;
const DEFAULT_PPO_MINIBATCH_RATIO: f64 = 1.0 / 16.0;
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

fn named_trainable_variables(vs: &nn::VarStore) -> Vec<(String, Tensor)> {
    let mut vars: Vec<(String, Tensor)> = vs
        .variables()
        .into_iter()
        .filter(|(_, tensor)| tensor.requires_grad())
        .collect();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    vars
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

fn tensor_is_finite(t: &Tensor) -> bool {
    t.isfinite().all().int64_value(&[]) != 0
}

fn tensor_summary(name: &str, t: &Tensor) -> String {
    let mean = t.mean(Kind::Float).double_value(&[]);
    let min = t.min().double_value(&[]);
    let max = t.max().double_value(&[]);
    format!(
        "{} shape={:?} mean={:.6} min={:.6} max={:.6}",
        name,
        t.size(),
        mean,
        min,
        max
    )
}

fn log_first_non_finite_tensor(
    logged: &mut bool,
    stage: &str,
    episode: usize,
    epoch: i64,
    chunk_i: usize,
    tensors: &[(&str, &Tensor)],
) {
    if *logged {
        return;
    }
    for (name, tensor) in tensors {
        if !tensor_is_finite(tensor) {
            println!(
                "NUMERIC ROOT CAUSE: stage={} episode={} epoch={} chunk={} {}",
                stage,
                episode,
                epoch + 1,
                chunk_i,
                tensor_summary(name, tensor)
            );
            for (other_name, other_tensor) in tensors {
                println!("  {}", tensor_summary(other_name, other_tensor));
            }
            *logged = true;
            return;
        }
    }
}

fn log_first_non_finite_var(
    logged: &mut bool,
    stage: &str,
    episode: usize,
    epoch: i64,
    chunk_i: usize,
    vars: &[Tensor],
    use_grad: bool,
) {
    if *logged {
        return;
    }
    for (idx, var) in vars.iter().enumerate() {
        let candidate = if use_grad {
            var.grad()
        } else {
            var.shallow_clone()
        };
        if !candidate.defined() || tensor_is_finite(&candidate) {
            continue;
        }
        println!(
            "NUMERIC ROOT CAUSE: stage={} episode={} epoch={} chunk={} param_idx={} param_shape={:?}",
            stage,
            episode,
            epoch + 1,
            chunk_i,
            idx,
            var.size()
        );
        println!(
            "  {}",
            tensor_summary(if use_grad { "grad" } else { "param" }, &candidate)
        );
        println!("  {}", tensor_summary("param_snapshot", var));
        *logged = true;
        return;
    }
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

pub(crate) fn build_no_reset_windowed_layouts(
    boundary_layout: &Tensor,
    step_deltas_chunk: &Tensor,
    chunk_count: i64,
    ppo_chunk_len: i64,
    flat_layout_len: i64,
) -> Tensor {
    let layout_rows = chunk_count * TICKERS_COUNT;
    let boundary_rows = boundary_layout.view([layout_rows, flat_layout_len]);
    let appended_deltas = if ppo_chunk_len > 1 {
        step_deltas_chunk
            .narrow(1, 0, ppo_chunk_len - 1)
            .permute([0, 2, 1])
            .contiguous()
            .view([layout_rows, ppo_chunk_len - 1])
            .to_kind(boundary_rows.kind())
    } else {
        Tensor::zeros(
            [layout_rows, 0],
            (boundary_rows.kind(), boundary_rows.device()),
        )
    };
    let extended = Tensor::cat(&[&boundary_rows, &appended_deltas], 1);
    extended
        .unfold(1, flat_layout_len, 1)
        .view([chunk_count, TICKERS_COUNT, ppo_chunk_len, flat_layout_len])
        .permute([0, 2, 1, 3])
        .contiguous()
        .view([chunk_count * ppo_chunk_len * TICKERS_COUNT, flat_layout_len])
}

/// Compute GAE advantages and returns from chunk-major rollout data.
pub(crate) fn compute_gae_chunked(
    rewards_by_chunk: &Tensor,
    values_by_chunk: &Tensor,
    dones_by_chunk: &Tensor,
    bootstrap_value: &Tensor,
    rollout_steps: i64,
    nprocs: i64,
    ppo_chunk_len: i64,
    gamma: f64,
    gae_lambda: f64,
    device: tch::Device,
) -> (Tensor, Tensor) {
    let chunks_per_rollout = rollout_steps / ppo_chunk_len;
    let total_chunks = chunks_per_rollout * nprocs;
    let advantages = Tensor::zeros(&[total_chunks, ppo_chunk_len], (Kind::Float, device));
    let returns = Tensor::zeros(&[total_chunks, ppo_chunk_len], (Kind::Float, device));

    tch::no_grad(|| {
        let mut last_gae = Tensor::zeros(&[nprocs], (Kind::Float, device));
        for t in (0..rollout_steps).rev() {
            let chunk_block = t / ppo_chunk_len;
            let chunk_offset = t % ppo_chunk_len;
            let chunk_row = chunk_block * nprocs;
            let next_values = if t == rollout_steps - 1 {
                bootstrap_value.shallow_clone()
            } else {
                let next_chunk_block = (t + 1) / ppo_chunk_len;
                let next_chunk_offset = (t + 1) % ppo_chunk_len;
                values_by_chunk
                    .narrow(0, next_chunk_block * nprocs, nprocs)
                    .select(1, next_chunk_offset)
            };
            let cur_values = values_by_chunk
                .narrow(0, chunk_row, nprocs)
                .select(1, chunk_offset);
            let rewards = rewards_by_chunk
                .narrow(0, chunk_row, nprocs)
                .select(1, chunk_offset);
            let dones = dones_by_chunk
                .narrow(0, chunk_row, nprocs)
                .select(1, chunk_offset);

            let delta = rewards + (1.0 - &dones) * gamma * &next_values - &cur_values;
            last_gae = delta + (1.0 - &dones) * gamma * gae_lambda * &last_gae;
            let _ = advantages
                .narrow(0, chunk_row, nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&last_gae.unsqueeze(1));
            let step_returns = &last_gae + &cur_values;
            let _ = returns
                .narrow(0, chunk_row, nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&step_returns.unsqueeze(1));
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
        "ppo rollout geometry: nprocs={} seq_len={} total_samples={} chunk_len={}",
        rollout.nprocs, rollout.seq_len, rollout.total_samples, rollout.ppo_chunk_len
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

    let named_trainable_vars = named_trainable_variables(&vs);
    let trainable_vars: Vec<Tensor> = named_trainable_vars
        .iter()
        .map(|(_, tensor)| tensor.shallow_clone())
        .collect();
    let mut opt = Muon::new(
        &trainable_vars,
        MuonConfig {
            lr: MUON_LR,
            use_muon_for_2d: USE_MUON,
            adamw_lr: LEARNING_RATE,
            adamw_eps: 1e-6,
            ..MuonConfig::default()
        },
    );

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

    let (obs_price_cpu, obs_static_cpu) = env.reset();
    let mut obs_static = Tensor::zeros(
        &[rollout.nprocs, STATIC_OBSERVATIONS as i64],
        (replay_obs_kind, device),
    );
    let mut step_deltas =
        Tensor::zeros(&[rollout.nprocs, TICKERS_COUNT], (replay_obs_kind, device));
    obs_static.copy_(&obs_static_cpu);
    let obs_price = obs_price_cpu.to_device(device);
    let mut stream_state = trading_model.init_replay_stream_state_batched(rollout.nprocs);
    let stream_layout = trading_model.uniform_stream_layout_from_raw_input(&obs_price);
    let mut streamed_output = Some(tch::no_grad(|| {
        autocast(true, || {
            trading_model.step_on_device_for_replay(&stream_layout, &obs_static, &mut stream_state)
        })
    }));
    let mut step_reward_per_ticker =
        Tensor::zeros(&[rollout.nprocs, TICKERS_COUNT], (Kind::Float, device));
    let mut step_is_done = Tensor::zeros(&[rollout.nprocs], (Kind::Float, device));
    let mut cpu_step_batch = CpuStepBatch::new(
        rollout.nprocs as usize,
        ACTION_COUNT as usize,
        raw_pd_dim as usize,
    );
    let mut action_host_view = unsafe {
        Tensor::from_blob(
            cpu_step_batch.actions_f32.as_ptr() as *const u8,
            &[rollout.nprocs, ACTION_COUNT],
            &[],
            Kind::Float,
            Device::Cpu,
        )
    };
    // Persistent CPU staging for reset env indices (one i64 per env, reused each step).
    let mut reset_env_indices_host: Vec<i64> = vec![0i64; rollout.nprocs as usize];
    let ticker_offsets = Tensor::arange(TICKERS_COUNT, (Kind::Int64, device));

    for episode in start_episode..1000000 {
        let mut reset_layout_batches_cpu: Vec<Tensor> = Vec::new();
        let mut reset_layout_count = 0i64;
        let mut reset_slots_host = vec![0i64; (total_chunks * rollout.ppo_chunk_len) as usize];
        for step in 0..rollout_steps as usize {
            let chunk_row = (step as i64 / rollout.ppo_chunk_len) * rollout.nprocs;
            let chunk_offset = step as i64 % rollout.ppo_chunk_len;
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
                let boundary_layout = stream_state
                    .uniform_layout
                    .view([rollout.nprocs, pd_dim])
                    .to_kind(replay_obs_kind);
                let _ = s_chunk_start_layouts
                    .narrow(0, chunk_row, rollout.nprocs)
                    .copy_(&boundary_layout);
            }
            let _ = s_static_obs
                .narrow(0, chunk_row, rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&obs_static.unsqueeze(1));

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
                .narrow(0, chunk_row, rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&step_deltas.unsqueeze(1));
            let mut reset_replay = None;
            if !reset_indices.is_empty() {
                let reset_count = reset_indices.len() as i64;
                // `from_blob` over the VecEnv-owned CPU buffer avoids a Vec->Tensor
                // copy before the H->D transfer (single non-blocking copy to GPU).
                let reset_raw_view = unsafe {
                    Tensor::from_blob(
                        reset_price_deltas.as_ptr() as *const u8,
                        &[reset_count, raw_pd_dim],
                        &[],
                        Kind::Float,
                        Device::Cpu,
                    )
                };
                let reset_raw_batch = reset_raw_view.to_device(device);
                let reset_layouts_batch =
                    trading_model.uniform_stream_layout_from_raw_input(&reset_raw_batch);
                // Persistent CPU staging for env indices; write then view via from_blob.
                for (slot, &env_idx) in reset_env_indices_host
                    .iter_mut()
                    .zip(reset_indices.iter())
                    .take(reset_count as usize)
                {
                    *slot = env_idx as i64;
                }
                let reset_env_host_view = unsafe {
                    Tensor::from_blob(
                        reset_env_indices_host.as_ptr() as *const u8,
                        &[reset_count],
                        &[],
                        Kind::Int64,
                        Device::Cpu,
                    )
                };
                let reset_env_tensor = reset_env_host_view.to_device(device);
                let reset_row_idx = (&reset_env_tensor.unsqueeze(1) * TICKERS_COUNT
                    + &ticker_offsets)
                    .reshape([-1]);
                reset_replay = Some((
                    reset_env_tensor,
                    reset_row_idx,
                    reset_layouts_batch.shallow_clone(),
                ));
                for (reset_i, env_idx) in reset_indices.iter().enumerate() {
                    let slot_idx = ((chunk_row + *env_idx as i64) * rollout.ppo_chunk_len
                        + chunk_offset) as usize;
                    reset_slots_host[slot_idx] = reset_layout_count + reset_i as i64 + 1;
                }
                reset_layout_count += reset_count;
                reset_layout_batches_cpu.push(
                    reset_layouts_batch
                        .to_kind(replay_obs_kind)
                        .to_device(tch::Device::Cpu),
                );
            }

            let _ = s_actions
                .narrow(0, chunk_row, rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&latent_actions.unsqueeze(1));
            let _ = s_old_log_probs
                .narrow(0, chunk_row, rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&action_log_prob.unsqueeze(1));

            let portfolio_reward =
                step_reward_per_ticker.mean_dim([1].as_slice(), false, Kind::Float);

            if DEBUG_NUMERICS {
                let _ =
                    debug_tensor_stats("portfolio_reward", &portfolio_reward, episode as i64, step);
                let _ = debug_tensor_stats("values", &values, episode as i64, step);
                let _ = debug_tensor_stats("step_is_done", &step_is_done, episode as i64, step);
            }
            let _ = s_rewards
                .narrow(0, chunk_row, rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&portfolio_reward.unsqueeze(1));
            let _ = s_dones
                .narrow(0, chunk_row, rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&step_is_done.unsqueeze(1));
            let _ = s_values
                .narrow(0, chunk_row, rollout.nprocs)
                .narrow(1, chunk_offset, 1)
                .copy_(&values.unsqueeze(1));

            streamed_output = Some(tch::no_grad(|| {
                autocast(true, || {
                    if reset_indices.is_empty() {
                        trading_model.step_on_device_for_replay(
                            &step_deltas,
                            &obs_static,
                            &mut stream_state,
                        )
                    } else {
                        // Advance state (layout shift + patch embed + layer-0 prefill)
                        // without running the full forward — the reset below will
                        // clobber reset-env state, and a single forward afterward
                        // covers all envs. Saves one full replay forward per reset step.
                        trading_model.advance_replay_stream_state(&step_deltas, &mut stream_state);
                        let (reset_env_tensor, reset_row_idx, reset_layouts_batch) =
                            reset_replay.as_ref().expect("reset replay payload missing");
                        trading_model.reset_uniform_stream_envs_from_layout_indexed(
                            &mut stream_state,
                            reset_env_tensor,
                            reset_row_idx,
                            reset_layouts_batch,
                        );
                        trading_model.forward_stream_state_on_device_for_replay(
                            &obs_static,
                            &mut stream_state,
                        )
                    }
                })
            }));
        }

        // Bootstrap value from final observation state (decode two-hot logits)
        let bootstrap_value = tch::no_grad(|| {
            let (value_logits, _, _) = autocast(true, || {
                trading_model
                    .forward_stream_state_on_device_for_replay(&obs_static, &mut stream_state)
            });
            hl_gauss.decode(&value_logits)
        });

        let (advantages, returns) = compute_gae_chunked(
            &s_rewards,
            &s_values,
            &s_dones,
            &bootstrap_value,
            rollout_steps,
            rollout.nprocs,
            rollout.ppo_chunk_len,
            0.99,
            0.95,
            device,
        );
        let reset_layout_bank_cpu = if reset_layout_batches_cpu.is_empty() {
            Tensor::zeros(&[0, pd_dim], (replay_obs_kind, tch::Device::Cpu))
        } else {
            let reset_layout_refs: Vec<&Tensor> = reset_layout_batches_cpu.iter().collect();
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
        // Keep reset slots flat and gather only the current minibatch to avoid
        // carrying a second chunk-major copy of the rollout on device.
        let reset_slots_by_chunk = Tensor::from_slice(&reset_slots_host)
            .to_kind(Kind::Int64)
            .view([total_chunks, rollout.ppo_chunk_len])
            .to_device(device);
        println!(
            "ppo update: total_samples={} minibatch_size={} chunk_len={} chunk_batch={}",
            total_samples, minibatch_size, rollout.ppo_chunk_len, chunk_batch_size
        );
        // Full-rollout advantage normalization (not per-minibatch)
        let adv_norm = (&advantages - advantages.mean(Kind::Float)) / (advantages.std(true) + 1e-8);

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
        let mut logged_numeric_root_cause = false;

        let mut last_minibatch_approx_kl = 0.0f64;
        let mut perm_host: Vec<i64> = (0..total_chunks).collect();
        let mut perm_gpu = Tensor::zeros([total_chunks], (Kind::Int64, device));
        let mut rng = rand::rng();

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            perm_host.shuffle(&mut rng);
            let perm_cpu = Tensor::from_slice(&perm_host)
                .to_kind(Kind::Int64)
                .to_device(device);
            perm_gpu.copy_(&perm_cpu);

            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;
            // Track last minibatch's KL mean on-device; fetch once at end of epoch
            // to avoid a host/device sync on every minibatch.
            let mut last_minibatch_kl_mean_gpu: Option<Tensor> = None;

            for (chunk_i, mb_start) in (0..total_chunks)
                .step_by(chunk_batch_size as usize)
                .enumerate()
            {
                let mb_end = (mb_start + chunk_batch_size).min(total_chunks);
                let chunk_count = mb_end - mb_start;
                let chunk_ids = perm_gpu.narrow(0, mb_start, chunk_count);
                let boundary_layout = s_chunk_start_layouts.index_select(0, &chunk_ids);
                let so_chunk = s_static_obs.index_select(0, &chunk_ids);
                let step_deltas_chunk = s_step_deltas.index_select(0, &chunk_ids);
                let adv_mb_by_chunk = adv_norm.index_select(0, &chunk_ids);
                let ret_mb_by_chunk = returns.index_select(0, &chunk_ids);
                let old_log_probs_by_chunk = s_old_log_probs.index_select(0, &chunk_ids);
                let act_mb_by_chunk = s_actions.index_select(0, &chunk_ids);
                let reset_slots_chunk = reset_slots_by_chunk.index_select(0, &chunk_ids);

                let fwd_start = Instant::now();
                let minibatch_sample_count = chunk_count * rollout.ppo_chunk_len;

                // Full-chunk batched windowed forward: build all ppo_chunk_len
                // windowed layouts at once and fire a single batched forward with
                // effective batch = chunk_count * ppo_chunk_len. No sub-chunk
                // gradient accumulation needed — one forward, one backward per
                // minibatch. Each window is its own 255-token causal prefix +
                // live-token suffix, so streaming semantics are preserved per window.
                let flat_layout_len = boundary_layout.size()[1] / TICKERS_COUNT;
                let has_reset_slots =
                    reset_layout_count > 0 && reset_slots_chunk.max().int64_value(&[]) > 0;
                let windowed = if has_reset_slots {
                    let layout_rows = chunk_count * TICKERS_COUNT;
                    let mut current_layout = boundary_layout.view([layout_rows, flat_layout_len]);
                    let mut windowed_rows: Vec<Tensor> =
                        Vec::with_capacity(rollout.ppo_chunk_len as usize);
                    for t in 0..rollout.ppo_chunk_len {
                        if t == 0 {
                            // Window 0: boundary layout unchanged (mirrors the `is_full`
                            // init path in step_on_device_for_replay).
                            windowed_rows.push(current_layout.shallow_clone());
                        } else {
                            let prev_step_deltas = step_deltas_chunk.select(1, t - 1); // [chunk_count, TICKERS]
                            let row_deltas = prev_step_deltas.reshape([layout_rows, 1]);
                            current_layout = trading_model
                                .shift_layout_append_delta(&current_layout, &row_deltas);
                            // Reset after shift-append to preserve bank layouts verbatim.
                            let step_reset_slots = reset_slots_chunk.select(1, t - 1); // [chunk_count]
                            let reset_chunk_idx = step_reset_slots.gt(0).nonzero().squeeze_dim(1);
                            if reset_chunk_idx.size()[0] > 0 {
                                let reset_slot_ids =
                                    step_reset_slots.index_select(0, &reset_chunk_idx) - 1;
                                let reset_slot_ids_cpu = reset_slot_ids.to_device(tch::Device::Cpu);
                                let reset_layouts = reset_layout_bank_cpu
                                    .index_select(0, &reset_slot_ids_cpu)
                                    .to_device(device);
                                let reset_row_idx = (&reset_chunk_idx.unsqueeze(1) * TICKERS_COUNT
                                    + &ticker_offsets)
                                    .reshape([-1]);
                                current_layout = current_layout.index_copy(
                                    0,
                                    &reset_row_idx,
                                    &reset_layouts.view([-1, flat_layout_len]),
                                );
                            }
                            windowed_rows.push(current_layout.shallow_clone());
                        }
                    }
                    Tensor::stack(&windowed_rows, 0)
                        .view([
                            rollout.ppo_chunk_len,
                            chunk_count,
                            TICKERS_COUNT,
                            flat_layout_len,
                        ])
                        .permute([1, 0, 2, 3])
                        .contiguous()
                        .view([
                            chunk_count * rollout.ppo_chunk_len * TICKERS_COUNT,
                            flat_layout_len,
                        ])
                } else {
                    build_no_reset_windowed_layouts(
                        &boundary_layout,
                        &step_deltas_chunk,
                        chunk_count,
                        rollout.ppo_chunk_len,
                        flat_layout_len,
                    )
                };
                let static_flat = so_chunk.reshape([minibatch_sample_count, so_dim]);

                let (new_value_logits, action_mean, action_std) = autocast(true, || {
                    trading_model.windowed_replay_forward(
                        &windowed,
                        &static_flat,
                        minibatch_sample_count,
                    )
                });

                // Flatten rollout-captured targets to minibatch-flat form (chunk-major).
                let adv_flat = adv_mb_by_chunk.reshape([-1]);
                let ret_flat = ret_mb_by_chunk.reshape([-1]);
                let old_log_probs_flat = old_log_probs_by_chunk.reshape([-1]);
                let act_flat = act_mb_by_chunk.reshape([-1, ACTION_COUNT]);

                let (rpo_alpha, action_mean_perturbed) = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let alpha = RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                    let alpha_detached = alpha.detach();
                    let rpo_noise = Tensor::empty(
                        [minibatch_sample_count, ACTION_COUNT],
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
                        &act_flat,
                        &action_mean_perturbed,
                        &action_std,
                        LOG_2PI,
                    );

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("latent_actions_mb", &act_flat, _epoch, chunk_i);
                    let _ = debug_tensor_stats(
                        "old_log_probs_mb",
                        &old_log_probs_flat,
                        _epoch,
                        chunk_i,
                    );
                    let _ = debug_tensor_stats("action_mean", &action_mean, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_std", &action_std, _epoch, chunk_i);
                }

                let log_ratio = &action_log_probs - &old_log_probs_flat;

                if DEBUG_NUMERICS {
                    let _ =
                        debug_tensor_stats("action_log_probs", &action_log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_ratio", &log_ratio, _epoch, chunk_i);
                }
                let ratio = log_ratio.exp();
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_LOW, 1.0 + PPO_CLIP_HIGH);

                log_first_non_finite_tensor(
                    &mut logged_numeric_root_cause,
                    "forward",
                    episode,
                    _epoch,
                    chunk_i,
                    &[
                        ("action_mean", &action_mean),
                        ("action_std", &action_std),
                        ("action_log_probs", &action_log_probs),
                        ("old_log_probs", &old_log_probs_flat),
                        ("log_ratio", &log_ratio),
                        ("ratio", &ratio),
                        ("new_value_logits", &new_value_logits),
                        ("adv_flat", &adv_flat),
                        ("ret_flat", &ret_flat),
                    ],
                );

                let action_loss =
                    -Tensor::min_other(&(&ratio * &adv_flat), &(&ratio_clipped * &adv_flat))
                        .mean(Kind::Float);

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("ret_mb", &ret_flat, _epoch, chunk_i);
                    let _ =
                        debug_tensor_stats("new_value_logits", &new_value_logits, _epoch, chunk_i);
                    let _ = debug_tensor_stats("adv_mb", &adv_flat, _epoch, chunk_i);
                }

                let value_loss =
                    hl_gauss_value_loss(&hl_gauss, &new_value_logits, &ret_flat).mean(Kind::Float);

                let dist_entropy = dist_entropy_per_sample.mean(Kind::Float);
                let dist_entropy_detached = dist_entropy.detach();

                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - &dist_entropy * ENTROPY_COEF;

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

                let total_loss = ppo_loss.shallow_clone() + alpha_loss.shallow_clone();

                log_first_non_finite_tensor(
                    &mut logged_numeric_root_cause,
                    "loss",
                    episode,
                    _epoch,
                    chunk_i,
                    &[
                        ("action_loss", &action_loss),
                        ("value_loss", &value_loss),
                        ("dist_entropy", &dist_entropy),
                        ("ppo_loss", &ppo_loss),
                        ("alpha_loss", &alpha_loss),
                        ("total_loss", &total_loss),
                    ],
                );

                fwd_time_us += fwd_start.elapsed().as_micros() as u64;
                let bwd_start = Instant::now();
                total_loss.backward();
                bwd_time_us += bwd_start.elapsed().as_micros() as u64;

                log_first_non_finite_var(
                    &mut logged_numeric_root_cause,
                    "grads_after_backward",
                    episode,
                    _epoch,
                    chunk_i,
                    &trainable_vars,
                    true,
                );

                let approx_kl_val =
                    tch::no_grad(|| (log_ratio.exp() - 1.0 - &log_ratio).mean(Kind::Float));
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("approx_kl_val", &approx_kl_val, _epoch, chunk_i);
                }
                let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * minibatch_sample_count as f64));
                let _ = total_policy_loss_weighted
                    .g_add_(&(&action_loss.detach() * minibatch_sample_count as f64));
                let _ = total_value_loss_weighted
                    .g_add_(&(&value_loss.detach() * minibatch_sample_count as f64));
                let _ = total_kl_weighted.g_add_(&(&approx_kl_val * minibatch_sample_count as f64));
                let _ = total_entropy_weighted
                    .g_add_(&(&dist_entropy_detached * minibatch_sample_count as f64));
                entropy_min = entropy_min.min_other(&dist_entropy_detached);
                entropy_max = entropy_max.max_other(&dist_entropy_detached);
                epoch_kl_count += minibatch_sample_count;
                total_sample_count += minibatch_sample_count;

                let _ = total_clipped.g_add_(&tch::no_grad(|| {
                    let dev = &ratio - 1.0;
                    let clipped_lo = dev.le(-PPO_CLIP_LOW).to_kind(Kind::Float);
                    let clipped_hi = dev.ge(PPO_CLIP_HIGH).to_kind(Kind::Float);
                    (clipped_lo + clipped_hi).sum(Kind::Float)
                }));
                total_ratio_samples += minibatch_sample_count;

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
                log_first_non_finite_var(
                    &mut logged_numeric_root_cause,
                    "params_after_step",
                    episode,
                    _epoch,
                    chunk_i,
                    &trainable_vars,
                    false,
                );
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

                // One forward/backward per minibatch now: the minibatch's KL is
                // exactly approx_kl_val. Track the last one for end-of-epoch early stop.
                last_minibatch_kl_mean_gpu = Some(approx_kl_val.shallow_clone());
            }

            // Single end-of-epoch host sync covering both the epoch-mean KL and
            // the last-minibatch KL used for early stopping. Avoids per-minibatch
            // D2H stalls that previously blocked the training pipeline.
            let mean_epoch_kl = if let Some(last_mb) = last_minibatch_kl_mean_gpu {
                let stacked = Tensor::stack(
                    &[&(&epoch_kl_gpu / epoch_kl_count.max(1) as f64), &last_mb],
                    0,
                )
                .to_kind(Kind::Double)
                .to_device(tch::Device::Cpu);
                let vec = Vec::<f64>::try_from(stacked).unwrap_or_else(|_| vec![0.0, 0.0]);
                // Preserve prior-epoch value if this epoch somehow had zero minibatches.
                last_minibatch_approx_kl = vec[1];
                vec[0]
            } else {
                // Epoch had no minibatches; keep prior `last_minibatch_approx_kl`.
                0.0
            };
            println!(
                "Epoch {}/{}: KL {:.4} (last mb {:.4})",
                _epoch + 1,
                OPTIM_EPOCHS,
                mean_epoch_kl,
                last_minibatch_approx_kl
            );
            if last_minibatch_approx_kl > TARGET_KL * KL_STOP_MULTIPLIER {
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
            &s_static_obs.narrow(0, 0, rollout.nprocs).select(1, 0),
            &rpo_rho,
            device,
        );
        let return_range_stats = hl_gauss.range_stats(&returns);
        // Compute all metrics on GPU, single transfer to CPU.
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
