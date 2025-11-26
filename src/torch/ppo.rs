use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::constants::{OBSERVATION_SPACE, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, STEPS_PER_EPISODE, TICKERS_COUNT};
use crate::torch::step::Env;
use crate::torch::model::model;

pub const NPROCS: i64 = 1; // Parallel environments for better GPU utilization
const UPDATES: i64 = 1000000;
const OPTIM_MINIBATCH: i64 = 512;  // Mini-batch size for GPU processing (avoids OOM)
const OPTIM_EPOCHS: i64 = 4; 

const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)
const GAMMA: f64 = 0.995;
const GAE_LAMBDA: f64 = 0.98;

pub fn train() {
    let mut env = Env::new();
    // n_steps is the episode length - use constant since all episodes are same length
    // Episodes are capped at 1500 steps, minus 2 for buffer = 1498
    let n_steps = STEPS_PER_EPISODE as i64;
    let memory_size = n_steps * NPROCS;
    println!("action space: {} (2 actions per ticker: buy/sell + hold)", TICKERS_COUNT * 2);
    println!("observation space: {:?}", OBSERVATION_SPACE);

    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CUDA device count: {}", tch::Cuda::device_count());
    println!("cuDNN available: {}", tch::Cuda::cudnn_is_available());

    let device = tch::Device::cuda_if_available();
    println!("Using device {:?}", device);

    let mut vs = nn::VarStore::new(device);
    let model = model(&vs.root(), TICKERS_COUNT * 2);  // 2 actions per ticker: buy/sell + hold
    // Reduced learning rate to prevent premature convergence
    let mut opt = nn::Adam::default().build(&vs, 2e-4).unwrap();

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
    let mut current_price_deltas = current_price_deltas_init;  // Keep in FP32
    let mut current_static_obs = current_static_obs_init.to_device(device);  // Keep in FP32

    // Separate storage for price deltas and static observations
    // Store price deltas on CPU to save VRAM (2400 deltas × 12000 steps is large)
    // Only move to GPU during training batches
    let s_price_deltas = Tensor::zeros(
        [
            n_steps + 1,
            NPROCS,
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ],
        (Kind::Float, Device::Cpu),  // Store on CPU in FP32
    );
    let s_static_obs = Tensor::zeros(
        [
            n_steps + 1,
            NPROCS,
            STATIC_OBSERVATIONS as i64,
        ],
        (Kind::Float, device),  // Keep static obs on GPU in FP32
    );

    for episode in 0..UPDATES {
        let mut episode_reward = 0.0;

        let s_values = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));
        let s_rewards = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));
        let s_actions = Tensor::zeros([n_steps, NPROCS, TICKERS_COUNT * 2], (Kind::Float, device));
        let s_masks = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));
        let s_log_probs = Tensor::zeros([n_steps, NPROCS], (Kind::Float, device));

        // Custom loop
        let (price_deltas_reset, static_obs_reset) = env.reset();
        current_price_deltas = price_deltas_reset;  // Keep in FP32
        current_static_obs = static_obs_reset.to_device(device);  // Keep in FP32
        env.episode = episode as usize;

        s_price_deltas.get(0).copy_(&current_price_deltas);  // CPU to CPU (FP32)
        s_static_obs.get(0).copy_(&current_static_obs);

        // Use a separate index (s) for tensor storage, starting from 0
        // Loop through the episode using relative steps (0 to max_step)
        let mut s: i64 = 0;
        for step in 0..env.max_step {
            env.step = step;

            let (critic, (action_mean, action_log_std)) = tch::no_grad(|| {
                // Move price deltas from CPU to GPU (already in FP16)
                let price_deltas_gpu = s_price_deltas.get(s).to_device(device);
                let static_obs_half = s_static_obs.get(s);
                model(
                    &price_deltas_gpu,
                    &static_obs_half,
                    false,  // eval mode during rollout for stable observations
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

            // Softer sigmoid squashing: action = 2 * sigmoid(z/2) - 1
            // Dividing z by 2 makes the squashing much gentler:
            // - sigmoid(z/2) changes more slowly than sigmoid(z)
            // - Reduces saturation at extremes
            // - With std range [0.368, 2.718], provides good exploration without extreme saturation
            let actions = (&z / 2.0).sigmoid() * 2.0 - 1.0;

            // Compute log Jacobian for the squashing transformation
            let sigmoid_z_half = (&z / 2.0).sigmoid();
            let one_minus_sigmoid = Tensor::from(1.0) - &sigmoid_z_half;
            let log_jacobian = sigmoid_z_half.log() + one_minus_sigmoid.log();

            // Final log probability
            let action_log_prob = (log_prob_z - log_jacobian).sum_dim_intlist(-1, false, Kind::Float);

            // Flatten the actions tensor [NPROCS, TICKERS_COUNT] to 1D before converting
            let actions_flat = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(TICKERS_COUNT as usize * 2)
                .map(|chunk| chunk.to_vec())
                .collect();

            // println!("Actions: {:?}", actions_vec);

            let step_state = env.step(actions_vec);

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

            s += 1;  // Increment storage index
        }

        println!(
            "episode rewards: {}",
            episode_reward
        );

        let price_deltas_batch = s_price_deltas.narrow(0, 0, n_steps).view([
            memory_size,
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ]);
        let static_obs_batch = s_static_obs.narrow(0, 0, n_steps).view([
            memory_size,
            STATIC_OBSERVATIONS as i64,
        ]);

        let (advantages, returns) = {
            let gae = Tensor::zeros([n_steps + 1, NPROCS], (Kind::Float, device));
            let returns = Tensor::zeros([n_steps + 1, NPROCS], (Kind::Float, device));

            // Bootstrap value for next state
            let next_value = tch::no_grad(|| {
                let price_deltas_gpu = s_price_deltas.get(-1).to_device(device);
                let static_obs = s_static_obs.get(-1);
                model(&price_deltas_gpu, &static_obs, false).0.view([NPROCS])
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

            (gae.narrow(0, 0, n_steps).view([memory_size, 1]),
             returns.narrow(0, 0, n_steps).view([memory_size, 1]))
        };
        let actions = s_actions.view([memory_size, TICKERS_COUNT * 2]);
        let old_log_probs = s_log_probs.view([memory_size]);

        let opt_start = Instant::now();

        for _epoch in 0..OPTIM_EPOCHS {
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
                let price_deltas_sample = price_deltas_batch.index_select(0, &batch_indexes_cpu).to_device(device);
                let static_obs_sample = static_obs_batch.index_select(0, &batch_indexes);
                let actions_sample = actions.index_select(0, &batch_indexes);
                let returns_sample = returns.index_select(0, &batch_indexes);
                let advantages_sample = advantages.index_select(0, &batch_indexes);
                let old_log_probs_sample = old_log_probs.index_select(0, &batch_indexes);

                let (critic, (action_mean, action_log_std)) = model(&price_deltas_sample, &static_obs_sample, true);  // train mode during optimization
                // All outputs in FP32

                // Recover the unsquashed Gaussian samples z from actions
                // action = 2 * sigmoid(z/2) - 1  =>  z = 2 * logit((action + 1) / 2)
                // Add small epsilon for numerical stability
                let eps = 1e-6;
                let action_01 = (&actions_sample + 1.0) / 2.0;  // Transform from [-1, 1] to [0, 1]
                let action_01_clamped = action_01.clamp(eps, 1.0 - eps);  // Prevent log(0)
                let one_minus_action = Tensor::from(1.0) - &action_01_clamped;
                let z = (&action_01_clamped.log() - one_minus_action.log()) * 2.0;  // 2 * logit function

                // Compute log probability of z under Gaussian N(action_mean, action_std)
                let action_std = action_log_std.exp();
                let z_normalized = (&z - &action_mean) / &action_std;
                let z_squared = z_normalized.pow_tensor_scalar(2);
                let two_log_std = &action_log_std * 2.0;
                let log_prob_z = (&z_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);

                // Compute log Jacobian correction for softer sigmoid squashing
                // action = 2 * sigmoid(z/2) - 1
                // |d_action/dz| = sigmoid(z/2) * (1 - sigmoid(z/2))
                // log|d_action/dz| = log(sigmoid(z/2)) + log(1 - sigmoid(z/2))
                let sigmoid_z_half = (&z / 2.0).sigmoid();
                let one_minus_sigmoid = Tensor::from(1.0) - &sigmoid_z_half;
                let log_jacobian = sigmoid_z_half.log() + one_minus_sigmoid.log();

                // Final log probability: log p(action) = log p(z) - log|d_action/dz|
                let action_log_probs = (log_prob_z - log_jacobian)
                    .sum_dim_intlist(-1, false, Kind::Float);

                // Entropy approximation: Use base Gaussian entropy
                // Note: Exact entropy of sigmoid-squashed distribution is complex to compute
                // Using Gaussian entropy as approximation is standard practice for regularization
                let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * action_log_std;
                let dist_entropy = entropy_components
                    .g_mul_scalar(0.5)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);

                // Old predictions from buffer
                let values_old_sample = s_values
                    .view([memory_size, 1])
                    .index_select(0, &batch_indexes);

                // Clipped value prediction
                let value_pred_clipped =
                    &values_old_sample + (&critic - &values_old_sample).clamp(-0.2, 0.2);

                // Unclipped and clipped MSE vs returns
                let value_loss_unclipped = (&critic - &returns_sample).pow_tensor_scalar(2);
                let value_loss_clipped   = (&value_pred_clipped - &returns_sample).pow_tensor_scalar(2);

                // Final value loss
                let value_loss = value_loss_unclipped
                    .max_other(&value_loss_clipped)
                    .mean(Kind::Float);

                // PPO clipped objective with pre-computed GAE advantages
                let adv_mean = advantages_sample.mean(Kind::Float);
                let adv_std = advantages_sample.std(false);
                let advantages_normalized = ((&advantages_sample - adv_mean) / (adv_std + 1e-8)).squeeze_dim(-1);

                let ratio = (action_log_probs - &old_log_probs_sample).exp();
                let unclipped_obj = &ratio * &advantages_normalized;

                let ratio_clipped = ratio.clamp(0.8, 1.2);
                let clipped_obj = ratio_clipped * &advantages_normalized;

                let action_loss = -Tensor::min_other(&unclipped_obj, &clipped_obj).mean(Kind::Float);
                // Relatively high 0.05 entropy bonus to encourage more exploration
                let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.05;

                // Moderately aggressive clip
                opt.backward_step_clip_norm(&loss, 2.0);
            }  // End mini-batch loop
        }  // End epoch loop
        
        let opt_end = Instant::now();
        
        if episode > 0 && episode % 25 == 0 {
            // Debug: Check if exploration has collapsed or network diverged
            let (_, (debug_mean, debug_log_std)) = tch::no_grad(|| {
                let price_deltas_gpu = s_price_deltas.get(0).to_device(device);
                let static_obs_half = s_static_obs.get(0);
                model(&price_deltas_gpu, &static_obs_half, false)  // Stays in FP16
            });

            // Show actual squashed actions, not raw mean
            let debug_actions = (&debug_mean / 2.0).sigmoid() * 2.0 - 1.0;
            let mean_squashed_action = f64::try_from(debug_actions.mean(Kind::Float)).unwrap();
            let mean_raw_action = f64::try_from(debug_mean.mean(Kind::Float)).unwrap();
            let mean_std = f64::try_from(debug_log_std.exp().mean(Kind::Float)).unwrap();
            let max_raw_action = f64::try_from(debug_mean.abs().max()).unwrap();

            println!(
                "[Ep {:6}] Episodes: {:.0}, Avg reward: {:.4}, Opt time: {:.2}s, Action (squashed): {:.3}, Action (raw): {:.1}, Max |raw|: {:.1}, Std: {:.4}",
                episode,
                total_episodes,
                total_rewards / total_episodes,
                opt_end.duration_since(opt_start).as_secs_f32(),
                mean_squashed_action,
                mean_raw_action,
                max_raw_action,
                mean_std
            );

            // Warn if network is diverging
            if max_raw_action > 100.0 {
                println!("WARNING: Network may be diverging! Raw action magnitude: {:.1}", max_raw_action);
            }

            total_rewards = 0.;
            total_episodes = 0.;
        }
        if episode > 0 && episode % 100 == 0 {
            std::fs::create_dir_all("weights").ok();
            if let Err(err) = vs.save(format!("weights/ppo_ep{}.ot", episode)) {
                println!("Error while saving weights: {}", err)
            } else {
                println!("Saved model weights: weights/ppo_ep{}.ot", episode);
            }
        }
    }
}

// Pretty sure I can ignore this, used for inference after training?
// Seems like it yes. Loads in weights and then infers a bunch
//
// pub fn sample<T: AsRef<std::path::Path>>(weight_file: T) -> cpython::PyResult<()> {
//     let env = Environment::new(false);
//     println!("action space: {}", env.action_space());
//     println!("observation space: {:?}", env.observation_space());

//     let mut vs = nn::VarStore::new(tch::Device::Cpu);
//     let model = model(&vs.root(), env.action_space());
//     vs.load(weight_file).unwrap();

//     let mut frame_stack = FrameStack::new(1, NSTACK);
//     let mut obs = frame_stack.update(&env.reset()?, None);

//     for _index in 0..5000 {
//         let (_critic, actor) = tch::no_grad(|| model(obs));
//         let probs = actor.softmax(-1, Kind::Float);
//         let actions = probs.multinomial(1, true).squeeze_dim(-1);
//         let step = env.step(Vec::<i64>::try_from(&actions).unwrap())?;

//         let masks = Tensor::from(1f32) - step.is_done;
//         obs = frame_stack.update(&step.obs, Some(&masks));
//     }
//     Ok(())
// }
