use std::time::Instant;
use shared::paths::WEIGHTS_PATH;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, OBSERVATION_SPACE, PRICE_DELTAS_PER_TICKER, RETROACTIVE_BUY_REWARD,
    STATIC_OBSERVATIONS, STEPS_PER_EPISODE, TICKERS_COUNT,
};
use crate::torch::model::model;
use crate::torch::env::Env;

pub const NPROCS: i64 = 1; // Parallel environments for better GPU utilization
const UPDATES: i64 = 1000000;
const OPTIM_MINIBATCH: i64 = 256; // Mini-batch size for GPU processing (avoids OOM)
const OPTIM_EPOCHS: i64 = 4;

const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)
const GAMMA: f64 = 0.995;
const GAE_LAMBDA: f64 = 0.98;

// PPO hyperparameters
const PPO_CLIP_RATIO: f64 = 0.2; // Clip range for policy ratio (the trust region)
const VALUE_CLIP_RANGE: f64 = 0.2; // Clip range for value function
const ENTROPY_COEF: f64 = 0.001; // Entropy bonus (reduced - exploration from std is enough)
const VALUE_LOSS_COEF: f64 = 0.5; // Value loss coefficient
const MAX_GRAD_NORM: f64 = 0.5; // Gradient clipping norm
// KL is monitored for diagnostics only - PPO clip handles trust region
const LEARNING_RATE: f64 = 2e-5;

pub fn train(weights_path: Option<&str>) {
    let mut env = Env::new(true);

    // Generate data analysis charts for training data
    println!("Generating data analysis charts...");
    let data_dir = "../training/data";
    for (ticker_idx, ticker) in env.tickers.iter().enumerate() {
        if let Err(e) = crate::charts::data_analysis::create_data_analysis_charts(
            ticker,
            &env.prices[ticker_idx],
            data_dir,
        ) {
            eprintln!("Warning: Failed to create data analysis charts for {}: {}", ticker, e);
        }
    }

    if let Err(e) = crate::charts::data_analysis::create_index_chart(
        &env.tickers,
        &env.prices,
        data_dir,
    ) {
        eprintln!("Warning: Failed to create index chart: {}", e);
    }

    println!("Data analysis charts generated in {}/", data_dir);

    // n_steps is the episode length - use constant since all episodes are same length
    // Episodes are capped at 1500 steps, minus 2 for buffer = 1498
    let n_steps = STEPS_PER_EPISODE as i64;
    let memory_size = n_steps * NPROCS;
    println!("action space: {} ({} tickers + cash)", ACTION_COUNT, TICKERS_COUNT);
    println!("observation space: {:?}", OBSERVATION_SPACE);

    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CUDA device count: {}", tch::Cuda::device_count());
    println!("cuDNN available: {}", tch::Cuda::cudnn_is_available());

    let device = tch::Device::cuda_if_available();
    println!("Using device {:?}", device);

    let mut vs = nn::VarStore::new(device);
    let model = model(&vs.root(), ACTION_COUNT);

    if let Some(path) = weights_path {
        println!("Loading weights from: {}", path);
        vs.load(path).expect("Failed to load weights");
        println!("Weights loaded successfully, resuming training");
    } else {
        println!("Starting training from scratch");
    }

    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

    // Disabled FP16 - with GAP reducing params from 39M to 284K, FP32 easily fits in VRAM
    // FP16 causes NaN issues in tch-rs, especially with complex architectures
    // vs.half();

    // Create device-specific kind
    let float_kind = (Kind::Float, device);
    let int64_kind = (Kind::Int64, device);

    let mut sum_rewards = Tensor::zeros([NPROCS], float_kind);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;

    let (current_price_deltas_init, current_static_obs_init) = env.reset();
    let mut current_price_deltas = current_price_deltas_init;
    let mut current_static_obs = current_static_obs_init.to_device(device);

    // Separate storage for price deltas and static observations
    // Store price deltas on CPU to save VRAM (2400 deltas × 12000 steps is large)
    // Only move to GPU during training batches
    let s_price_deltas = Tensor::zeros(
        [
            n_steps + 1,
            NPROCS,
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ],
        (Kind::Float, Device::Cpu), // Store on CPU in FP32
    );
    let s_static_obs = Tensor::zeros(
        [n_steps + 1, NPROCS, STATIC_OBSERVATIONS as i64],
        (Kind::Float, device), // Keep static obs on GPU in FP32
    );

    for episode in 0..UPDATES {
        let mut episode_reward = 0.0;

        let s_values = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));
        let s_rewards = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));
        let s_actions = Tensor::zeros([n_steps, NPROCS, ACTION_COUNT], (Kind::Float, device));
        let s_masks = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));
        let s_log_probs = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));

        // Custom loop
        let (price_deltas_reset, static_obs_reset) = env.reset();
        current_price_deltas = price_deltas_reset; // Keep in FP32
        current_static_obs = static_obs_reset.to_device(device); // Keep in FP32
        env.episode = episode as usize;

        s_price_deltas.get(0).copy_(&current_price_deltas); // CPU to CPU (FP32)
        s_static_obs.get(0).copy_(&current_static_obs);

        // Use a separate index (s) for tensor storage, starting from 0
        // Loop through the episode using relative steps (0 to max_step)
        let mut s: i64 = 0;
        for step in 0..env.max_step {
            env.step = step;

            let (critic, (action_mean, action_log_std, divisor), attn_weights) = tch::no_grad(|| {
                // Move price deltas from CPU to GPU (FP32)
                let price_deltas_gpu = s_price_deltas.get(s).to_device(device);
                let static_obs = s_static_obs.get(s);
                model(
                    &price_deltas_gpu,
                    &static_obs,
                    false, // eval mode during rollout for stable observations
                )
            });

            // Sample from Gaussian distribution
            let action_std = action_log_std.exp();
            let noise = Tensor::randn_like(&action_mean);
            let z = &action_mean + &action_std * noise;

            // Compute log probability for the sampled z
            let z_normalized = (&z - &action_mean) / &action_std;
            let z_squared = z_normalized.pow_tensor_scalar(2);
            let two_log_std = &action_log_std * 2.0;
            let log_prob_z = (&z_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);

            // Softsign squashing: action = (z/d) / (1 + |z/d|) = z / (d + |z|)
            let z_abs = z.abs();
            let actions = &z / (&divisor + &z_abs);

            // Log Jacobian for softsign: log|da/dz| = log(d) - 2*log(d + |z|)
            let log_jacobian = divisor.log() - (&divisor + &z_abs).log() * 2.0;

            // Final log probability
            let action_log_prob =
                (log_prob_z - log_jacobian).sum_dim_intlist(-1, false, Kind::Float);

            let actions_flat = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(ACTION_COUNT as usize)
                .map(|chunk| chunk.to_vec())
                .collect();

            // println!("Actions: {:?}", actions_vec);

            let step_state = env.step(actions_vec);

            // Record static observations and attention weights for TUI visualization (last 100 steps only)
            if step >= env.max_step.saturating_sub(100) {
                let static_obs_vec = Vec::<f32>::try_from(s_static_obs.get(s).flatten(0, -1)).unwrap_or_default();
                let attn_weights_vec = Vec::<f32>::try_from(attn_weights.flatten(0, -1)).unwrap_or_default();
                env.episode_history.static_observations.push(static_obs_vec);
                env.episode_history.attention_weights.push(attn_weights_vec);
            }

            // println!("Step reward: {:?}", step_state.reward);

            let reward = step_state.reward.to_device(device);
            let is_done = step_state.is_done.to_device(device);
            // Keep price deltas on CPU for storage, only move to GPU when needed
            let price_deltas = step_state.price_deltas;
            let static_obs = step_state.static_obs.to_device(device);

            sum_rewards += &reward;

            let reward_float = f64::try_from((&sum_rewards * &is_done).sum(Kind::Float)).unwrap();
            episode_reward += reward_float;
            total_rewards += reward_float;
            total_episodes += f64::try_from(is_done.sum(Kind::Float)).unwrap();

            let masks = Tensor::from(1f32).to_device(device) - &is_done;
            sum_rewards *= &masks;

            s_actions.get(s).copy_(&actions);
            s_values.get(s).copy_(&critic.squeeze_dim(-1));
            s_log_probs.get(s).copy_(&action_log_prob);
            s_price_deltas.get(s + 1).copy_(&price_deltas);
            s_static_obs.get(s + 1).copy_(&static_obs);
            s_rewards.get(s).copy_(&reward);
            s_masks.get(s).copy_(&masks);

            current_price_deltas = price_deltas;
            current_static_obs = static_obs;

            s += 1; // Increment storage index
        }

        if RETROACTIVE_BUY_REWARD {
            env.apply_retroactive_rewards(&s_rewards);
        }

        let price_deltas_batch = s_price_deltas
            .narrow(0, 0, n_steps)
            .reshape([memory_size, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_batch = s_static_obs
            .narrow(0, 0, n_steps)
            .reshape([memory_size, STATIC_OBSERVATIONS as i64]);

        let (advantages, returns) = {
            let gae = Tensor::zeros([n_steps + 1, NPROCS], (Kind::Float, device));
            let returns = Tensor::zeros([n_steps + 1, NPROCS], (Kind::Float, device));

            // Bootstrap value for next state
            let next_value = tch::no_grad(|| {
                let price_deltas_gpu = s_price_deltas.get(-1).to_device(device);
                let static_obs = s_static_obs.get(-1);
                let (critic, _, _) = model(&price_deltas_gpu, &static_obs, false);
                critic.view([NPROCS])
            });

            // Compute GAE backwards
            for s in (0..n_steps).rev() {
                let value_t = s_values.get(s);
                let value_next = if s == n_steps - 1 {
                    next_value.shallow_clone()
                } else {
                    s_values.get(s + 1)
                };

                // TD error: δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
                let delta = s_rewards.get(s) + GAMMA * &value_next * s_masks.get(s) - &value_t;

                // GAE: A_t = δ_t + γ * λ * mask * A_{t+1}
                let gae_next = if s == n_steps - 1 {
                    Tensor::zeros_like(&delta)
                } else {
                    gae.get(s + 1)
                };
                let gae_t = delta + GAMMA * GAE_LAMBDA * s_masks.get(s) * gae_next;
                gae.get(s).copy_(&gae_t);

                // Returns: R_t = A_t + V(s_t)
                let return_t = &gae_t + &value_t;
                returns.get(s).copy_(&return_t);
            }

            (
                gae.narrow(0, 0, n_steps).view([memory_size, 1]),
                returns.narrow(0, 0, n_steps).view([memory_size, 1]),
            )
        };
        let actions = s_actions.view([memory_size, ACTION_COUNT]);
        let old_log_probs = s_log_probs.view([memory_size]);

        // Record advantage stats before normalization
        let adv_mean_val = f64::try_from(advantages.mean(Kind::Float)).unwrap_or(0.0);
        let adv_min_val = f64::try_from(advantages.min()).unwrap_or(0.0);
        let adv_max_val = f64::try_from(advantages.max()).unwrap_or(0.0);
        env.meta_history.record_advantage_stats(adv_mean_val, adv_min_val, adv_max_val);

        // Normalize advantages globally (not per-minibatch) for consistent policy gradients
        let adv_mean = advantages.mean(Kind::Float);
        let adv_std = advantages.std(false);
        let advantages = ((&advantages - adv_mean) / (adv_std + 1e-8)).squeeze_dim(-1);

        // Normalize returns and old values to prevent huge value loss early in training
        let ret_mean = returns.mean(Kind::Float);
        let ret_std = returns.std(false) + 1e-8;
        let returns = (&returns - &ret_mean) / &ret_std;
        let s_values = (&s_values - &ret_mean) / &ret_std; // normalize old values with same stats

        let opt_start = Instant::now();
        let mut total_kl = 0.0;
        let mut num_kl_samples = 0;
        let mut total_loss = 0.0;
        let mut num_loss_samples = 0;

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            let mut epoch_kl_sum = 0.0;
            let mut epoch_kl_count = 0;

            // Shuffle indices for this epoch
            let shuffled_indices = Tensor::randperm(memory_size, int64_kind);

            // Process in mini-batches to avoid VRAM overflow
            let num_minibatches = (memory_size + OPTIM_MINIBATCH - 1) / OPTIM_MINIBATCH;

            for minibatch_idx in 0..num_minibatches {
                let start_idx = minibatch_idx * OPTIM_MINIBATCH;
                let end_idx = (start_idx + OPTIM_MINIBATCH).min(memory_size);
                let minibatch_size = end_idx - start_idx;

                // Get indices for this mini-batch
                let batch_indexes = shuffled_indices.narrow(0, start_idx, minibatch_size);
                let batch_indexes_cpu = batch_indexes.to_device(tch::Device::Cpu);

                // Move only mini-batch to GPU (in FP32)
                let price_deltas_sample = price_deltas_batch
                    .index_select(0, &batch_indexes_cpu)
                    .to_device(device);
                let static_obs_sample = static_obs_batch.index_select(0, &batch_indexes);
                let actions_sample = actions.index_select(0, &batch_indexes);
                let returns_sample = returns.index_select(0, &batch_indexes);
                let advantages_sample = advantages.index_select(0, &batch_indexes);
                let old_log_probs_sample = old_log_probs.index_select(0, &batch_indexes);
                let (critic, (action_mean, action_log_std, divisor), _attn_weights) =
                    model(&price_deltas_sample, &static_obs_sample, true);

                // Recover z from softsign actions: a = z / (d + |z|)
                // Solving for z: z = a * d / (1 - |a|)
                // Clamp actions to prevent division by zero at |a| = 1
                let eps = 1e-6;
                let actions_clamped = actions_sample.clamp(-1.0 + eps, 1.0 - eps);
                let z = &actions_clamped * &divisor / (1.0 - actions_clamped.abs());

                // Compute log probability of z under Gaussian N(action_mean, action_std)
                let action_std = action_log_std.exp();
                let z_normalized = (&z - &action_mean) / &action_std;
                let z_squared = z_normalized.pow_tensor_scalar(2);
                let two_log_std = &action_log_std * 2.0;
                let log_prob_z = (&z_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);

                // Log Jacobian for softsign: log|da/dz| = log(d) - 2*log(d + |z|)
                let z_abs = z.abs();
                let log_jacobian = divisor.log() - (&divisor + &z_abs).log() * 2.0;

                // Final log probability: log p(action) = log p(z) - log|da/dz|
                let action_log_probs =
                    (log_prob_z - log_jacobian).sum_dim_intlist(-1, false, Kind::Float);

                // Entropy approximation: Use base Gaussian entropy
                let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * action_log_std;
                let dist_entropy = entropy_components
                    .g_mul_scalar(0.5)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);

                // Old predictions from buffer
                let values_old_sample = s_values
                    .reshape([memory_size, 1])
                    .index_select(0, &batch_indexes);

                // Normalize critic to match normalized returns
                let critic_normalized = (&critic - &ret_mean) / &ret_std;

                // Clipped value prediction
                let value_pred_clipped = &values_old_sample
                    + (&critic_normalized - &values_old_sample).clamp(-VALUE_CLIP_RANGE, VALUE_CLIP_RANGE);

                // Unclipped and clipped MSE vs returns
                let value_loss_unclipped = (&critic_normalized - &returns_sample).pow_tensor_scalar(2);
                let value_loss_clipped =
                    (&value_pred_clipped - &returns_sample).pow_tensor_scalar(2);

                // Final value loss
                let value_loss = value_loss_unclipped
                    .max_other(&value_loss_clipped)
                    .mean(Kind::Float);

                // PPO clipped objective with pre-computed GAE advantages (globally normalized)
                let ratio = (&action_log_probs - &old_log_probs_sample).exp();
                let unclipped_obj = &ratio * &advantages_sample;

                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_RATIO, 1.0 + PPO_CLIP_RATIO);
                let clipped_obj = ratio_clipped * &advantages_sample;

                let action_loss =
                    -Tensor::min_other(&unclipped_obj, &clipped_obj).mean(Kind::Float);

                // Compute approximate KL divergence for early stopping
                // KL ≈ (ratio - 1) - log(ratio)
                let approx_kl = (&old_log_probs_sample - &action_log_probs).mean(Kind::Float);
                let approx_kl_value = f64::try_from(approx_kl).unwrap();

                epoch_kl_sum += approx_kl_value;
                epoch_kl_count += 1;

                let loss = value_loss * VALUE_LOSS_COEF + action_loss - dist_entropy * ENTROPY_COEF;
                let loss_value = f64::try_from(loss.shallow_clone()).unwrap();
                total_loss += loss_value;
                num_loss_samples += 1;

                opt.backward_step_clip_norm(&loss, MAX_GRAD_NORM);
            }

            // Monitor KL divergence (diagnostic only - PPO clip handles trust region)
            let mean_epoch_kl = epoch_kl_sum / epoch_kl_count as f64;
            total_kl += mean_epoch_kl;
            num_kl_samples += 1;

            // Safety check: stop only if something is catastrophically wrong
            if mean_epoch_kl > 50.0 {
                println!("Epoch {}/{}: KL {:.1} (catastrophic), stopping epochs", _epoch + 1, OPTIM_EPOCHS, mean_epoch_kl);
                break 'epoch_loop;
            }
            
            println!("Epoch {}/{}: KL {:.1}", _epoch + 1, OPTIM_EPOCHS, mean_epoch_kl);
        }

        // Record std and divisor stats every episode
        let (mean_std, min_std, max_std, mean_div) = tch::no_grad(|| {
            let price_deltas_gpu = s_price_deltas.get(0).to_device(device);
            let static_obs = s_static_obs.get(0);
            let (_, (_, action_log_std, divisor), _) = model(&price_deltas_gpu, &static_obs, false);
            let std = action_log_std.exp();
            (
                f64::try_from(std.mean(Kind::Float)).unwrap_or(0.0),
                f64::try_from(std.min()).unwrap_or(0.0),
                f64::try_from(std.max()).unwrap_or(0.0),
                f64::try_from(divisor.mean(Kind::Float)).unwrap_or(0.0),
            )
        });
        env.meta_history.record_std_stats(mean_std, min_std, max_std);
        env.meta_history.record_divisor(mean_div);

        // Record mean loss every episode
        let mean_loss = total_loss / num_loss_samples as f64;
        env.meta_history.record_loss(mean_loss);

        let opt_end = Instant::now();

        if episode > 0 && episode % 25 == 0 {
            // Debug: Check if exploration has collapsed or network diverged
            let (_, (debug_mean, debug_log_std, debug_divisor), _attn_weights) = tch::no_grad(|| {
                let price_deltas_gpu = s_price_deltas.get(0).to_device(device);
                let static_obs = s_static_obs.get(0);
                model(&price_deltas_gpu, &static_obs, false)
            });

            // Show actual squashed actions using softsign
            let debug_actions = &debug_mean / (&debug_divisor + debug_mean.abs());
            let mean_squashed_action = f64::try_from(debug_actions.mean(Kind::Float)).unwrap();
            let mean_raw_action = f64::try_from(debug_mean.mean(Kind::Float)).unwrap();
            let mean_std = f64::try_from(debug_log_std.exp().mean(Kind::Float)).unwrap();
            let max_raw_action = f64::try_from(debug_mean.abs().max()).unwrap();
            let mean_divisor = f64::try_from(debug_divisor.mean(Kind::Float)).unwrap();

            let avg_kl = if num_kl_samples > 0 {
                total_kl / num_kl_samples as f64
            } else {
                0.0
            };

            println!(
                "[Ep {:6}] Episodes: {:.0}, Avg reward: {:.4}, Opt time: {:.2}s, Avg KL: {:.4}, Action (squashed): {:.3}, Action (raw): {:.1}, Max |raw|: {:.1}, Std: {:.4}, Div: {:.2}",
                episode,
                total_episodes,
                total_rewards / total_episodes,
                opt_end.duration_since(opt_start).as_secs_f32(),
                avg_kl,
                mean_squashed_action,
                mean_raw_action,
                max_raw_action,
                mean_std,
                mean_divisor
            );

            // Warn if network is diverging
            if max_raw_action > 100.0 {
                println!(
                    "WARNING: Network may be diverging! Raw action magnitude: {:.1}",
                    max_raw_action
                );
            }

            total_rewards = 0.;
            total_episodes = 0.;
        }
        if episode > 0 && episode % 50 == 0 {
            std::fs::create_dir_all("weights").ok();
            if let Err(err) = vs.save(format!("{WEIGHTS_PATH}/ppo_ep{}.ot", episode)) {
                println!("Error while saving weights: {}", err)
            } else {
                println!("Saved model weights: {WEIGHTS_PATH}/ppo_ep{}.ot", episode);
            }
        }
    }
}
