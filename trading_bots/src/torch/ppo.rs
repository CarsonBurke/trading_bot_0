use std::env;
use std::path::Path;
use std::time::Instant;
use tch::{autocast, nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::action_space::{
    gaussian_entropy, sigmoid_target_weight, transformed_action_log_prob,
};
use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::env::{CpuStepBatch, VecEnv};
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{ModelOutput, ModelVariant, TradingModel, TradingModelConfig};
use crate::torch::sdp::force_flash_sdp;
use crate::torch::two_hot::TwoHotBins;
use shared::{paths::RUNS_PATH, run_dir::RunDir};

const LEARNING_RATE: f64 = 1e-4;
pub const NPROCS: i64 = 16;
const SEQ_LEN: i64 = 4000;
const PPO_CHUNK_LEN: i64 = 250;
const DEFAULT_PPO_MINIBATCH_RATIO: f64 = 0.0625;
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

fn minibatch_samples_from_total(total_samples: i64) -> i64 {
    let ratio = env::var("PPO_MINIBATCH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&v| v > 0.0 && v <= 1.0)
        .unwrap_or(DEFAULT_PPO_MINIBATCH_RATIO);
    let target = ((total_samples as f64) * ratio).round() as i64;
    let aligned = ((target + NPROCS - 1) / NPROCS).max(1) * NPROCS;
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

fn cpu_tensor_from_f32(data: &[f32], size: &[i64]) -> Tensor {
    unsafe {
        Tensor::from_blob(
            data.as_ptr() as *const u8,
            size,
            &[],
            Kind::Float,
            Device::Cpu,
        )
    }
}

fn sample_rollout_actions_from_output(
    output: ModelOutput,
    two_hot: &TwoHotBins,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let (value_logits, action_mean, action_std) = output;

    // Decode two-hot logits to scalar values for GAE
    let values = two_hot.decode(&value_logits);

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
    gamma: f64,
    gae_lambda: f64,
    device: tch::Device,
) -> (Tensor, Tensor) {
    let memory_size = rollout_steps * NPROCS;
    let advantages = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let returns = Tensor::zeros(&[memory_size], (Kind::Float, device));

    tch::no_grad(|| {
        let mut last_gae = Tensor::zeros(&[NPROCS], (Kind::Float, device));
        for t in (0..rollout_steps).rev() {
            let mem_idx = t * NPROCS;
            let next_values = if t == rollout_steps - 1 {
                bootstrap_value.shallow_clone()
            } else {
                s_values.narrow(0, (t + 1) * NPROCS, NPROCS)
            };
            let cur_values = s_values.narrow(0, mem_idx, NPROCS);
            let rewards = s_rewards.narrow(0, mem_idx, NPROCS);
            let dones = s_dones.narrow(0, mem_idx, NPROCS);

            let delta = rewards + (1.0 - &dones) * gamma * &next_values - &cur_values;
            last_gae = delta + (1.0 - &dones) * gamma * gae_lambda * &last_gae;
            let _ = advantages.narrow(0, mem_idx, NPROCS).copy_(&last_gae);
            let step_returns = &last_gae + &cur_values;
            let _ = returns.narrow(0, mem_idx, NPROCS).copy_(&step_returns);
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

pub(crate) fn two_hot_value_loss(
    two_hot: &TwoHotBins,
    value_logits: &Tensor,
    returns: &Tensor,
) -> Tensor {
    let return_bins = two_hot.encode(returns);
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
    assert_eq!(
        model_variant,
        ModelVariant::Uniform256Stream,
        "PPO rollout collection is streaming-only; train with --model-size uniform-256-stream"
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
    force_flash_sdp();
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
        load_var_store_partial(&mut vs, path).unwrap();
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

    let mut env = VecEnv::new(true, model_variant, gens_path.clone());
    if start_episode > 0 {
        env.set_episode(start_episode);
        env.primary_mut()
            .meta_history
            .load_from_episode(start_episode, &gens_path);
    }

    let two_hot = TwoHotBins::default_for(device);

    let rollout_steps = SEQ_LEN;
    let memory_size = rollout_steps * NPROCS;
    assert_eq!(
        rollout_steps % PPO_CHUNK_LEN,
        0,
        "PPO_CHUNK_LEN must divide rollout length"
    );
    let chunks_per_rollout = rollout_steps / PPO_CHUNK_LEN;
    let total_chunks = chunks_per_rollout * NPROCS;
    let mut chunk_mem_indices_host = Vec::with_capacity((total_chunks * PPO_CHUNK_LEN) as usize);
    for chunk_id in 0..total_chunks {
        let env_idx = chunk_id % NPROCS;
        let chunk_start = (chunk_id / NPROCS) * PPO_CHUNK_LEN;
        for offset in 0..PPO_CHUNK_LEN {
            chunk_mem_indices_host.push((chunk_start + offset) * NPROCS + env_idx);
        }
    }
    let chunk_mem_indices = Tensor::from_slice(&chunk_mem_indices_host)
        .to_kind(Kind::Int64)
        .to_device(device)
        .view([total_chunks, PPO_CHUNK_LEN]);

    let raw_pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
    let pd_dim = trading_model.price_input_dim();
    let so_dim = STATIC_OBSERVATIONS as i64;
    let replay_obs_kind = trading_model.input_kind();

    let s_chunk_start_layouts = Tensor::zeros(
        &[chunks_per_rollout * NPROCS, pd_dim],
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
        let mut obs_price = Tensor::zeros(&[NPROCS, raw_pd_dim], (Kind::Float, device));
        let mut obs_static =
            Tensor::zeros(&[NPROCS, STATIC_OBSERVATIONS as i64], (Kind::Float, device));
        let ring_len = PRICE_DELTAS_PER_TICKER as i64;
        let base_idx = Tensor::arange(ring_len, (Kind::Int64, device));
        let mut ring_pos = ring_len - 1;
        let mut ring_buf = Tensor::zeros(&[NPROCS, TICKERS_COUNT, ring_len], (Kind::Float, device));
        let mut step_deltas = Tensor::zeros(&[NPROCS, TICKERS_COUNT], (Kind::Float, device));
        obs_price.copy_(&obs_price_cpu);
        obs_static.copy_(&obs_static_cpu);
        ring_buf.copy_(&obs_price.view([NPROCS, TICKERS_COUNT, ring_len]));
        let mut stream_state = trading_model.init_stream_state_batched(NPROCS);
        let stream_layout = trading_model.uniform_stream_layout_from_raw_input(&obs_price);
        let mut streamed_output = Some(tch::no_grad(|| {
            autocast(true, || {
                trading_model.step_on_device(&stream_layout, &obs_static, &mut stream_state)
            })
        }));
        let mut step_reward_per_ticker =
            Tensor::zeros(&[NPROCS, TICKERS_COUNT], (Kind::Float, device));
        let mut step_is_done = Tensor::zeros(&[NPROCS], (Kind::Float, device));
        let mut reset_layout_store: Vec<Tensor> = Vec::new();
        let mut reset_slots_host = vec![0i64; memory_size as usize];
        let mut cpu_step_batch = CpuStepBatch::new(
            NPROCS as usize,
            ACTION_COUNT as usize,
            TICKERS_COUNT as usize,
            STATIC_OBSERVATIONS,
            raw_pd_dim as usize,
        );
        for step in 0..rollout_steps as usize {
            let mem_idx = step as i64 * NPROCS;
            let (values, action_mean, action_std, latent_actions, target_weights, action_log_prob) =
                sample_rollout_actions_from_output(
                    streamed_output
                        .take()
                        .expect("streamed rollout output missing"),
                    &two_hot,
                );

            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("action_mean", &action_mean, episode as i64, step);
                let _ = debug_tensor_stats("action_std", &action_std, episode as i64, step);
                let _ = debug_tensor_stats("latent_actions", &latent_actions, episode as i64, step);
                let _ = debug_tensor_stats("target_weights", &target_weights, episode as i64, step);
                let _ =
                    debug_tensor_stats("action_log_prob", &action_log_prob, episode as i64, step);
            }
            if step as i64 % PPO_CHUNK_LEN == 0 {
                let chunk_row = (step as i64 / PPO_CHUNK_LEN) * NPROCS;
                let boundary_layout = stream_state
                    .uniform_layout
                    .view([NPROCS, pd_dim])
                    .to_kind(replay_obs_kind);
                let _ = s_chunk_start_layouts
                    .narrow(0, chunk_row, NPROCS)
                    .copy_(&boundary_layout);
            }
            let _ = s_static_obs.narrow(0, mem_idx, NPROCS).copy_(&obs_static);

            let target_weights_cpu = target_weights.to_device(Device::Cpu).to_kind(Kind::Float);
            let actions_len = cpu_step_batch.actions_f32.len();
            target_weights_cpu.copy_data(&mut cpu_step_batch.actions_f32, actions_len);
            env.step_from_actions_f32(&mut cpu_step_batch);
            step_deltas.copy_(&cpu_tensor_from_f32(
                &cpu_step_batch.step_deltas,
                &[NPROCS, TICKERS_COUNT],
            ));
            obs_static.copy_(&cpu_tensor_from_f32(
                &cpu_step_batch.static_obs,
                &[NPROCS, STATIC_OBSERVATIONS as i64],
            ));
            step_reward_per_ticker.copy_(&cpu_tensor_from_f32(
                &cpu_step_batch.reward_per_ticker,
                &[NPROCS, TICKERS_COUNT],
            ));
            step_is_done.copy_(&cpu_tensor_from_f32(&cpu_step_batch.is_done, &[NPROCS]));
            let reset_indices = &cpu_step_batch.reset_indices;
            let reset_price_deltas = &cpu_step_batch.reset_price_deltas;
            let _ = s_step_deltas.narrow(0, mem_idx, NPROCS).copy_(&step_deltas);

            ring_pos = (ring_pos + 1) % ring_len;
            let _ = ring_buf
                .narrow(2, ring_pos, 1)
                .copy_(&step_deltas.unsqueeze(-1));

            if !reset_indices.is_empty() {
                let idx = (&base_idx + (ring_pos + 1)).remainder(ring_len);
                let pd_dim_usize = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
                let reset_layout_tensor = Tensor::from_slice(reset_price_deltas)
                    .view([reset_indices.len() as i64, raw_pd_dim])
                    .to_device(device);
                let reset_layouts_batch =
                    trading_model.uniform_stream_layout_from_raw_input(&reset_layout_tensor);
                for (reset_i, env_idx) in reset_indices.iter().enumerate() {
                    let start = reset_i * pd_dim_usize;
                    let end = start + pd_dim_usize;
                    let ordered = Tensor::from_slice(&reset_price_deltas[start..end])
                        .view([TICKERS_COUNT, ring_len])
                        .to_device(device);
                    let mut ring_env = ring_buf.narrow(0, *env_idx as i64, 1).squeeze_dim(0);
                    let _ = ring_env.index_copy_(1, &idx, &ordered);
                    reset_layout_store.push(
                        reset_layouts_batch
                            .get(reset_i as i64)
                            .to_kind(replay_obs_kind),
                    );
                    reset_slots_host[(mem_idx + *env_idx as i64) as usize] =
                        reset_layout_store.len() as i64;
                }
            }

            let idx = (&base_idx + (ring_pos + 1)).remainder(ring_len);
            let ordered = ring_buf.index_select(2, &idx);
            obs_price.copy_(&ordered.view([NPROCS, raw_pd_dim]));

            let _ = s_actions.narrow(0, mem_idx, NPROCS).copy_(&latent_actions);
            let _ = s_old_log_probs
                .narrow(0, mem_idx, NPROCS)
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
                .narrow(0, mem_idx, NPROCS)
                .copy_(&portfolio_reward);
            let _ = s_dones.narrow(0, mem_idx, NPROCS).copy_(&step_is_done);
            let _ = s_values.narrow(0, mem_idx, NPROCS).copy_(&values);

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
            two_hot.decode(&value_logits)
        });

        let (advantages, returns) = compute_gae(
            &s_rewards,
            &s_values,
            &s_dones,
            &bootstrap_value,
            rollout_steps,
            0.99,
            0.95,
            device,
        );

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

        let total_samples = rollout_steps * NPROCS;
        let minibatch_size = minibatch_samples_from_total(total_samples);
        let chunk_batch_size = ((minibatch_size + PPO_CHUNK_LEN - 1) / PPO_CHUNK_LEN).max(1);
        let s_reset_slots = Tensor::from_slice(&reset_slots_host)
            .to_kind(Kind::Int64)
            .to_device(device);
        println!(
            "ppo update: total_samples={} minibatch_size={} chunk_len={} chunk_batch={}",
            total_samples, minibatch_size, PPO_CHUNK_LEN, chunk_batch_size
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

        let mut first_epoch_kl = 0.0f64;

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            let perm = Tensor::randperm(total_chunks, (Kind::Int64, device));

            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;

            for (chunk_i, mb_start) in (0..total_chunks)
                .step_by(chunk_batch_size as usize)
                .enumerate()
            {
                let mb_end = (mb_start + chunk_batch_size).min(total_chunks);
                let chunk_count = mb_end - mb_start;
                let chunk_ids = perm.narrow(0, mb_start, chunk_count);
                let mb_inds = chunk_mem_indices.index_select(0, &chunk_ids).reshape([-1]);
                let chunk_sample_count = chunk_count * PPO_CHUNK_LEN;
                let boundary_layout = s_chunk_start_layouts.index_select(0, &chunk_ids);
                let so_chunk_flat = s_static_obs.index_select(0, &mb_inds);
                let step_deltas_chunk_flat = s_step_deltas.index_select(0, &mb_inds);
                let act_mb = s_actions.index_select(0, &mb_inds);
                let ret_mb = returns.index_select(0, &mb_inds);
                let adv_mb = adv_norm.index_select(0, &mb_inds);
                let old_log_probs_mb = s_old_log_probs.index_select(0, &mb_inds);
                let reset_slots_chunk = s_reset_slots
                    .index_select(0, &mb_inds)
                    .view([chunk_count, PPO_CHUNK_LEN]);
                let so_chunk = so_chunk_flat.view([chunk_count, PPO_CHUNK_LEN, so_dim]);
                let step_deltas_chunk =
                    step_deltas_chunk_flat.view([chunk_count, PPO_CHUNK_LEN, TICKERS_COUNT]);

                let fwd_start = Instant::now();
                let mut chunk_state = trading_model.init_stream_state_batched(chunk_count);
                let first_static = so_chunk.select(1, 0);
                let first_output = autocast(true, || {
                    trading_model.step_on_device(&boundary_layout, &first_static, &mut chunk_state)
                });
                let mut value_logits_steps = vec![first_output.0];
                let mut action_mean_steps = vec![first_output.1];
                let mut action_std_steps = vec![first_output.2];
                trading_model.refresh_stream_state_storage_for_autograd(&mut chunk_state);
                for pos in 1..PPO_CHUNK_LEN {
                    let prev_step_deltas = step_deltas_chunk.select(1, pos - 1);
                    let current_static = so_chunk.select(1, pos);
                    let reset_slot_pos = reset_slots_chunk.select(1, pos - 1);
                    let reset_mask = reset_slot_pos.gt(0);
                    let has_reset = reset_mask.any().int64_value(&[]) != 0;
                    let output = if !has_reset {
                        autocast(true, || {
                            trading_model.step_on_device(
                                &prev_step_deltas,
                                &current_static,
                                &mut chunk_state,
                            )
                        })
                    } else {
                        let reset_rows_tensor = reset_mask.nonzero().squeeze_dim(1);
                        let reset_rows_i64: Vec<i64> =
                            Vec::try_from(reset_rows_tensor.to_device(Device::Cpu)).unwrap();
                        let reset_rows: Vec<usize> =
                            reset_rows_i64.iter().map(|&idx| idx as usize).collect();
                        let reset_slot_ids: Vec<i64> = Vec::try_from(
                            reset_slot_pos
                                .index_select(0, &reset_rows_tensor)
                                .to_device(Device::Cpu),
                        )
                        .unwrap();
                        let reset_layout_parts = reset_slot_ids
                            .iter()
                            .map(|&slot| reset_layout_store[(slot - 1) as usize].shallow_clone())
                            .collect::<Vec<_>>();
                        let _ = autocast(true, || {
                            trading_model.step_on_device(
                                &prev_step_deltas,
                                &current_static,
                                &mut chunk_state,
                            )
                        });
                        let reset_layout_batch = Tensor::stack(&reset_layout_parts, 0);
                        trading_model.reset_uniform_stream_envs_from_layout(
                            &mut chunk_state,
                            &reset_rows,
                            &reset_layout_batch,
                        );
                        autocast(true, || {
                            trading_model
                                .forward_stream_state_on_device(&current_static, &mut chunk_state)
                        })
                    };
                    value_logits_steps.push(output.0);
                    action_mean_steps.push(output.1);
                    action_std_steps.push(output.2);
                    trading_model.refresh_stream_state_storage_for_autograd(&mut chunk_state);
                }
                let new_value_logits =
                    Tensor::stack(&value_logits_steps, 1).reshape([chunk_sample_count, -1]);
                let action_mean = Tensor::stack(&action_mean_steps, 1)
                    .reshape([chunk_sample_count, ACTION_COUNT]);
                let action_std =
                    Tensor::stack(&action_std_steps, 1).reshape([chunk_sample_count, ACTION_COUNT]);

                let latent_actions_mb = act_mb;

                // RPO (CleanRL-style): iid uniform perturbation on each action-mean dimension.
                let (rpo_alpha, action_mean_perturbed) = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let alpha = RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                    let alpha_detached = alpha.detach();
                    let rpo_noise =
                        Tensor::empty([chunk_sample_count, ACTION_COUNT], (Kind::Float, device))
                            .uniform_(-1.0, 1.0)
                            * &alpha_detached;
                    (alpha, &action_mean + rpo_noise)
                } else {
                    (
                        Tensor::zeros(&[1], (Kind::Float, device)),
                        action_mean.shallow_clone(),
                    )
                };

                let action_log_probs = transformed_action_log_prob(
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
                    let _ =
                        debug_tensor_stats("old_log_probs_mb", &old_log_probs_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_mean", &action_mean, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_std", &action_std, _epoch, chunk_i);
                }

                let log_ratio = &action_log_probs - &old_log_probs_mb;

                if DEBUG_NUMERICS {
                    let _ =
                        debug_tensor_stats("action_log_probs", &action_log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_ratio", &log_ratio, _epoch, chunk_i);
                }
                let ratio = log_ratio.exp();
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_LOW, 1.0 + PPO_CLIP_HIGH);

                // Single portfolio advantage - no more weighted combination
                let action_loss =
                    -Tensor::min_other(&(&ratio * &adv_mb), &(&ratio_clipped * &adv_mb))
                        .mean(Kind::Float);

                // Two-hot distributional value loss with PPO-style value clipping
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("ret_mb", &ret_mb, _epoch, chunk_i);
                    let _ =
                        debug_tensor_stats("new_value_logits", &new_value_logits, _epoch, chunk_i);
                    let _ = debug_tensor_stats("adv_mb", &adv_mb, _epoch, chunk_i);
                }

                let value_loss =
                    two_hot_value_loss(&two_hot, &new_value_logits, &ret_mb).mean(Kind::Float);

                let dist_entropy = gaussian_entropy(&action_std, LOG_2PI).mean(Kind::Float);
                let dist_entropy_detached = dist_entropy.detach();

                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - &dist_entropy * ENTROPY_COEF;

                // RPO alpha loss: target induced KL for uniform logit perturbation.
                let alpha_loss = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let action_var = action_std.pow_tensor_scalar(2).detach();
                    let inv_var_mean = action_var.clamp_min(1e-4).reciprocal().mean(Kind::Float);
                    let d = ACTION_COUNT as f64;
                    let induced_kl = rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                    (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0) * ALPHA_LOSS_COEF
                } else {
                    Tensor::zeros([], (Kind::Float, device))
                };

                let scaled_ppo_loss = &ppo_loss
                    * (chunk_sample_count as f64 / (chunk_batch_size * PPO_CHUNK_LEN) as f64);
                let scaled_alpha_loss = &alpha_loss
                    * (chunk_sample_count as f64 / (chunk_batch_size * PPO_CHUNK_LEN) as f64);
                let total_chunk_loss = scaled_ppo_loss + scaled_alpha_loss;

                fwd_time_us += fwd_start.elapsed().as_micros() as u64;
                let bwd_start = Instant::now();
                total_chunk_loss.backward();
                bwd_time_us += bwd_start.elapsed().as_micros() as u64;

                let approx_kl_val =
                    tch::no_grad(|| (log_ratio.exp() - 1.0 - &log_ratio).mean(Kind::Float));
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("approx_kl_val", &approx_kl_val, _epoch, chunk_i);
                }
                let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                let _ = total_policy_loss_weighted
                    .g_add_(&(&action_loss.detach() * chunk_sample_count as f64));
                let _ = total_value_loss_weighted
                    .g_add_(&(&value_loss.detach() * chunk_sample_count as f64));
                let _ = total_kl_weighted.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                let _ = total_entropy_weighted
                    .g_add_(&(&dist_entropy_detached * chunk_sample_count as f64));
                entropy_min = entropy_min.min_other(&dist_entropy_detached);
                entropy_max = entropy_max.max_other(&dist_entropy_detached);
                epoch_kl_count += chunk_sample_count;
                total_sample_count += chunk_sample_count;

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

                let _ = total_clipped.g_add_(&tch::no_grad(|| {
                    let dev = &ratio - 1.0;
                    let clipped_lo = dev.le(-PPO_CLIP_LOW).to_kind(Kind::Float);
                    let clipped_hi = dev.ge(PPO_CLIP_HIGH).to_kind(Kind::Float);
                    (clipped_lo + clipped_hi).sum(Kind::Float)
                }));
                total_ratio_samples += chunk_sample_count;
            }

            let mean_epoch_kl = epoch_kl_gpu.double_value(&[]) / epoch_kl_count as f64;
            if _epoch == 0 {
                first_epoch_kl = mean_epoch_kl;
            }
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
            &s_chunk_start_layouts.narrow(0, 0, NPROCS),
            &s_static_obs.narrow(0, 0, NPROCS),
            &rpo_rho,
            device,
        );
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
            ],
            0,
        );
        let all_scalars_vec: Vec<f64> = Vec::try_from(all_scalars.to_device(tch::Device::Cpu))
            .unwrap_or_else(|_| vec![0.0; 15]);
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
        primary.meta_history.record_approx_kl(first_epoch_kl);

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
