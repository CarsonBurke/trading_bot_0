use shared::paths::WEIGHTS_PATH;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, OBSERVATION_SPACE, PRICE_DELTAS_PER_TICKER, RETROACTIVE_BUY_REWARD,
    STATIC_OBSERVATIONS, STEPS_PER_EPISODE, TICKERS_COUNT,
};
use crate::torch::env::VecEnv;
use crate::torch::model::TradingModel;
use crate::torch::ssm::Mamba2State;

pub const NPROCS: i64 = 6; // Parallel environments for better GPU utilization
const UPDATES: i64 = 1000000;
const OPTIM_EPOCHS: i64 = 4;
const CHUNK_SIZE: i64 = 64; // Steps per chunk for sequence-aware PPO with SSM state restoration

const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)
const GAMMA: f64 = 0.995;
const GAE_LAMBDA: f64 = 0.98;

// Symlog transform (from Dreamer-v3): handles diverse reward scales gracefully
// symlog compresses large values while preserving sign: symlog(x) = sign(x) * ln(|x| + 1)
fn symlog(x: &Tensor) -> Tensor {
    x.sign() * (x.abs() + 1.0).log()
}

/// Two-hot encode values into bucket distribution
/// Both values and bin_centers are in symlog space (Scheme B: everything symlog)
/// bin_centers are uniformly spaced in symlog space [-20, 20]
fn twohot_encode(values: &Tensor, bin_centers: &Tensor) -> Tensor {
    let num_bins = bin_centers.size()[0];
    let values_flat = values.flatten(0, -1);
    let batch_size = values_flat.size()[0];

    // Get bin range using tensor ops (avoid CPU sync)
    let bin_min = bin_centers.min();
    let bin_max = bin_centers.max();
    let bin_range = (&bin_max - &bin_min).clamp_min(1e-8);

    // Clamp values to bin range
    let values_clamped = values_flat.maximum(&bin_min).minimum(&bin_max);

    // Compute fractional bin index: (value - min) / range * (num_bins - 1)
    let frac_idx = (&values_clamped - &bin_min) / &bin_range * (num_bins - 1) as f64;

    // Get lower and upper bin indices
    let idx_low = frac_idx.floor().clamp(0.0, (num_bins - 2) as f64).to_kind(Kind::Int64);
    let idx_high = (&idx_low + 1).clamp(0, num_bins - 1);

    // Weight for upper bin is the fractional part
    let weight_high = (&frac_idx - frac_idx.floor()).clamp(0.0, 1.0);
    let weight_low: Tensor = 1.0 - &weight_high;

    // Build two-hot distribution using scatter_add (accumulates correctly)
    let mut twohot = Tensor::zeros([batch_size, num_bins], (Kind::Float, values.device()));
    let _ = twohot.scatter_add_(-1, &idx_low.unsqueeze(-1), &weight_low.unsqueeze(-1));
    let _ = twohot.scatter_add_(-1, &idx_high.unsqueeze(-1), &weight_high.unsqueeze(-1));

    twohot
}

fn clone_state_deep(s: &Mamba2State) -> Mamba2State {
    let mut conv = Tensor::zeros_like(&s.conv_state);
    let mut ssm = Tensor::zeros_like(&s.ssm_state);
    let _ = conv.copy_(&s.conv_state.detach());
    let _ = ssm.copy_(&s.ssm_state.detach());
    Mamba2State {
        conv_state: conv,
        ssm_state: ssm,
    }
}

// gSDE: resample exploration noise every N env steps (temporally correlated exploration).
// This matches the common SB3-style behavior while keeping per-step log-prob computation unchanged.
const SDE_SAMPLE_FREQ: usize = 4;

// PPO hyperparameters
const PPO_CLIP_RATIO: f64 = 0.2; // Clip range for policy ratio (the trust region)
const VALUE_CLIP_RANGE: f64 = 0.2; // Clip range for value function
const ENTROPY_COEF: f64 = 0.0; // Entropy bonus (reduced - exploration from std is enough)
const VALUE_LOSS_COEF: f64 = 0.5; // Value loss coefficient
const MAX_GRAD_NORM: f64 = 0.5; // Gradient clipping norm
                                // Conservative KL early stopping (SB3-style)
const TARGET_KL: f64 = 0.03;
const KL_STOP_MULTIPLIER: f64 = 1.5;
const LEARNING_RATE: f64 = 2e-5;
const GRAD_ACCUM_STEPS: usize = 2; // Accumulate gradients over k chunks before stepping (was 4, reduced for more updates)

// Symlog clipping - prevents extreme values even after compression
const REWARD_CLIP: f64 = 10.0;

pub fn train(weights_path: Option<&str>) {
    let mut env = VecEnv::new(true);

    // Generate data analysis charts for training data
    println!("Generating data analysis charts...");
    let data_dir = "../training/data";
    for (ticker_idx, ticker) in env.tickers().iter().enumerate() {
        if let Err(e) = crate::charts::data_analysis::create_data_analysis_charts(
            ticker,
            &env.prices()[ticker_idx],
            data_dir,
        ) {
            eprintln!(
                "Warning: Failed to create data analysis charts for {}: {}",
                ticker, e
            );
        }
    }

    if let Err(e) =
        crate::charts::data_analysis::create_index_chart(env.tickers(), env.prices(), data_dir)
    {
        eprintln!("Warning: Failed to create index chart: {}", e);
    }

    println!("Data analysis charts generated in {}/", data_dir);

    let max_steps = STEPS_PER_EPISODE as i64;
    println!(
        "action space: {} ({} tickers + cash)",
        ACTION_COUNT, TICKERS_COUNT
    );
    println!("observation space: {:?}", OBSERVATION_SPACE);

    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CUDA device count: {}", tch::Cuda::device_count());
    println!("cuDNN available: {}", tch::Cuda::cudnn_is_available());

    let device = tch::Device::cuda_if_available();
    println!("Using device {:?}", device);

    let mut vs = nn::VarStore::new(device);
    let trading_model = TradingModel::new(&vs.root(), ACTION_COUNT);

    if let Some(path) = weights_path {
        println!("Loading weights from: {}", path);
        // Debug: print VarStore variable names
        println!("VarStore variables:");
        for (name, _) in vs.variables() {
            println!("  {}", name);
        }
        match vs.load(path) {
            Ok(()) => println!("Weights loaded successfully, resuming training"),
            Err(e) => panic!("Failed to load weights: {:?}", e),
        }
    } else {
        println!("Starting training from scratch");
    }

    // Recurrent state for rollout collection (maintains SSM memory across steps)
    let mut stream_state = trading_model.init_stream_state_batched(NPROCS);

    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

    // Disabled FP16 - with GAP reducing params from 39M to 284K, FP32 easily fits in VRAM
    // FP16 causes NaN issues in tch-rs, especially with complex architectures
    // vs.half();

    // Create device-specific kind
    let float_kind = (Kind::Float, device);

    let mut sum_rewards = Tensor::zeros([NPROCS], float_kind);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;

    let (current_price_deltas_init, current_static_obs_init) = env.reset();
    let mut current_price_deltas = current_price_deltas_init;
    let mut current_static_obs = current_static_obs_init.to_device(device);

    // Store price deltas on GPU - ~122MB is negligible and avoids CPU-GPU transfers
    let s_price_deltas = Tensor::zeros(
        [
            max_steps + 1,
            NPROCS,
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ],
        (Kind::Float, device),
    );
    let s_static_obs = Tensor::zeros(
        [max_steps + 1, NPROCS, STATIC_OBSERVATIONS as i64],
        (Kind::Float, device), // Keep static obs on GPU in FP32
    );

    for episode in 0..UPDATES {
        // Custom loop
        let (price_deltas_reset, static_obs_reset) = env.reset();
        current_price_deltas = price_deltas_reset; // Keep in FP32
        current_static_obs = static_obs_reset.to_device(device); // Keep in FP32
        env.set_episode(episode as usize);

        let rollout_steps = env.max_step() as i64;
        let memory_size = rollout_steps * NPROCS;

        let s_values = Tensor::zeros([rollout_steps, NPROCS], (Kind::Float, device));
        let mut s_rewards = Tensor::zeros([rollout_steps, NPROCS], (Kind::Float, device));
        let s_actions =
            Tensor::zeros([rollout_steps, NPROCS, ACTION_COUNT - 1], (Kind::Float, device));
        let s_masks = Tensor::zeros([rollout_steps, NPROCS], (Kind::Float, device));
        let s_log_probs = Tensor::zeros([rollout_steps, NPROCS], (Kind::Float, device));

        // Reset recurrent state at episode boundary
        stream_state.reset();

        s_price_deltas.get(0).copy_(&current_price_deltas); // CPU to CPU (FP32)
        s_static_obs.get(0).copy_(&current_static_obs);

        // Store SSM states at chunk boundaries for sequence-aware training
        let num_chunks = (rollout_steps + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunk_ssm_states: Vec<Mamba2State> = Vec::with_capacity(num_chunks as usize);

        // Use a separate index (s) for tensor storage, starting from 0
        // Loop through the episode using relative steps (0 to max_step)
        let mut s: i64 = 0;
        let mut sde_noise: Option<Tensor> = None;
        for step in 0..env.max_step() {
            env.set_step(step);

            // Save SSM state at chunk boundaries BEFORE processing (deep copy via allocate + copy_)
            if s % CHUNK_SIZE == 0 {
                chunk_ssm_states.push(clone_state_deep(&stream_state.ssm_state_batched));
            }

            let (critic, _critic_logits, (action_mean, action_log_std, _divisor), attn_weights) =
                tch::no_grad(|| {
                    let price_deltas_step = s_price_deltas.get(s);
                    let static_obs = s_static_obs.get(s);
                    trading_model.forward_with_state(
                        &price_deltas_step,
                        &static_obs,
                        &mut stream_state,
                    )
                });

            // Logistic-normal sampling: model outputs K-1 dims, we append 0 for reference
            // u ~ N(μ, σ), u_ext = [u, 0], y = softmax(u_ext)
            let action_std = action_log_std.exp();
            if sde_noise.is_none() || (step % SDE_SAMPLE_FREQ == 0) {
                sde_noise = Some(Tensor::randn_like(&action_mean));
            }
            let noise = sde_noise.as_ref().unwrap();
            let u = &action_mean + &action_std * noise; // [batch, K-1]

            // Append 0 as reference category: u_ext = [u, 0]
            let zeros = Tensor::zeros([NPROCS, 1], (Kind::Float, device));
            let u_ext = Tensor::cat(&[u.shallow_clone(), zeros], 1); // [batch, K]

            // Softmax to get portfolio weights on simplex
            let actions = u_ext.softmax(-1, Kind::Float); // [batch, K]

            // Log-prob with Jacobian: log π(y) = log N(u; μ, σ) - Σ log(y_i)
            let u_normalized = (&u - &action_mean) / &action_std;
            let u_squared = u_normalized.pow_tensor_scalar(2);
            let two_log_std = &action_log_std * 2.0;
            let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
            let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float); // sum over K-1 dims

            // Jacobian correction: -Σ log(y_i) for all K dims (use log_softmax for stability)
            let log_jacobian = u_ext
                .log_softmax(-1, Kind::Float)
                .sum_dim_intlist(-1, false, Kind::Float);
            let action_log_prob = log_prob_gaussian - log_jacobian;

            let actions_flat = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(ACTION_COUNT as usize)
                .map(|chunk| chunk.to_vec())
                .collect();

            // println!("Actions: {:?}", actions_vec);

            let step_state = env.step(actions_vec);

            // Record static observations and attention weights for TUI visualization (last 100 steps only)
            if step >= env.max_step().saturating_sub(100) {
                let static_obs_vec =
                    Vec::<f32>::try_from(s_static_obs.get(s).flatten(0, -1)).unwrap_or_default();
                let attn_weights_vec =
                    Vec::<f32>::try_from(attn_weights.flatten(0, -1)).unwrap_or_default();
                env.primary_mut()
                    .episode_history
                    .static_observations
                    .push(static_obs_vec);
                env.primary_mut()
                    .episode_history
                    .attention_weights
                    .push(attn_weights_vec);
            }

            // println!("Step reward: {:?}", step_state.reward);

            let reward = step_state.reward.to_device(device);
            let is_done = step_state.is_done.to_device(device);
            // Keep price deltas on CPU for storage, only move to GPU when needed
            let price_deltas = step_state.price_deltas;
            let static_obs = step_state.static_obs.to_device(device);

            sum_rewards += &reward;

            let reward_float = f64::try_from((&sum_rewards * &is_done).sum(Kind::Float)).unwrap();
            total_rewards += reward_float;
            total_episodes += f64::try_from(is_done.sum(Kind::Float)).unwrap();

            let masks = Tensor::from(1f32).to_device(device) - &is_done;
            sum_rewards *= &masks;

            s_actions.get(s).copy_(&u); // Store u (K-1 dims) for training
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

        // Symlog rewards directly - no pre-scaling needed
        // Symlog compresses outliers while preserving sign (Dreamer-v3 style)
        s_rewards.copy_(&symlog(&s_rewards).clamp(-REWARD_CLIP, REWARD_CLIP));

        let price_deltas_batch = s_price_deltas
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_batch = s_static_obs
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, STATIC_OBSERVATIONS as i64]);

        // Snapshot end-of-rollout state for bootstrap (before we mutate stream_state)
        let end_state = clone_state_deep(&stream_state.ssm_state_batched);

        // Critic now outputs values directly in symlog space (expectation over symlog bins)
        // No need to symlog - values are already in symlog space

        let (advantages, returns) = {
            let gae = Tensor::zeros([rollout_steps + 1, NPROCS], (Kind::Float, device));
            let returns = Tensor::zeros([rollout_steps + 1, NPROCS], (Kind::Float, device));

            // Bootstrap value for next state (restore end state to avoid mutation)
            let next_value = tch::no_grad(|| {
                stream_state
                    .ssm_state_batched
                    .conv_state
                    .copy_(&end_state.conv_state);
                stream_state
                    .ssm_state_batched
                    .ssm_state
                    .copy_(&end_state.ssm_state);
                let price_deltas_step = s_price_deltas.get(rollout_steps);
                let static_obs = s_static_obs.get(rollout_steps);
                let (critic, _, _, _) = trading_model.forward_with_state(
                    &price_deltas_step,
                    &static_obs,
                    &mut stream_state,
                );
                critic.view([NPROCS])
            });
            // Critic output is already in symlog space

            // Compute GAE backwards - all in symlog space
            // Rewards symlog'd, values in symlog space (via distributional critic)
            for s in (0..rollout_steps).rev() {
                let value_t = s_values.get(s);
                let value_next = if s == rollout_steps - 1 {
                    next_value.shallow_clone()
                } else {
                    s_values.get(s + 1)
                };

                // TD error in symlog space: δ_t = r_symlog + γ * V_symlog(s_{t+1}) * mask - V_symlog(s_t)
                let delta = s_rewards.get(s) + GAMMA * &value_next * s_masks.get(s) - &value_t;

                // GAE: A_t = δ_t + γ * λ * mask * A_{t+1}
                let gae_next = if s == rollout_steps - 1 {
                    Tensor::zeros_like(&delta)
                } else {
                    gae.get(s + 1)
                };
                let gae_t = delta + GAMMA * GAE_LAMBDA * s_masks.get(s) * gae_next;
                gae.get(s).copy_(&gae_t);

                // Returns in symlog space: R_symlog = A + V_symlog
                let return_t = &gae_t + &value_t;
                returns.get(s).copy_(&return_t);
            }

            (
                gae.narrow(0, 0, rollout_steps).view([memory_size, 1]),
                returns.narrow(0, 0, rollout_steps).view([memory_size, 1]),
            )
        };
        let actions = s_actions.view([memory_size, ACTION_COUNT - 1]);
        let old_log_probs = s_log_probs.view([memory_size]);
        let s_values_flat = s_values.view([memory_size]).detach();

        // Record advantage stats before normalization
        let adv_mean_val = f64::try_from(advantages.mean(Kind::Float)).unwrap_or(0.0);
        let adv_min_val = f64::try_from(advantages.min()).unwrap_or(0.0);
        let adv_max_val = f64::try_from(advantages.max()).unwrap_or(0.0);
        env.primary_mut().meta_history.record_advantage_stats(
            adv_mean_val,
            adv_min_val,
            adv_max_val,
        );

        // Normalize advantages globally (not per-minibatch) for consistent policy gradients
        // Detach to prevent backprop through GAE computation
        let adv_mean = advantages.mean(Kind::Float);
        let adv_std = advantages.std(false);
        let advantages = ((&advantages - adv_mean) / (adv_std + 1e-8))
            .squeeze_dim(-1)
            .detach();

        // Returns already in symlog space, just detach
        let returns = returns.detach();

        let opt_start = Instant::now();
        // Accumulate on GPU - only sync once at end of all epochs (weighted by samples)
        let mut total_kl_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_sample_count = 0i64;
        let mut grad_norm_sum = Tensor::zeros([], (Kind::Float, device));
        let mut grad_norm_count = 0i64;
        // Clip fraction diagnostic
        let mut total_clipped = 0i64;
        let mut total_ratio_samples = 0i64;

        // Pre-allocate zeros for logistic-normal (max size: MINI_BATCH_STEPS * NPROCS)
        const MINI_BATCH_STEPS: i64 = 16;
        let zeros_max = Tensor::zeros([MINI_BATCH_STEPS * NPROCS, 1], (Kind::Float, device));

        // Shuffle chunk order for gradient diversity (but maintain temporal order within chunks)
        let mut chunk_order: Vec<usize> = (0..chunk_ssm_states.len()).collect();

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;

            // Shuffle chunk order each epoch
            use rand::seq::SliceRandom;
            chunk_order.shuffle(&mut rand::rng());

            // Process chunks with gradient accumulation (backward per chunk to free graph, step every k)
            opt.zero_grad();
            let total_epoch_samples = rollout_steps * NPROCS;
            let samples_per_accum =
                (CHUNK_SIZE * NPROCS * GRAD_ACCUM_STEPS as i64).min(total_epoch_samples);

            for (chunk_i, &chunk_idx) in chunk_order.iter().enumerate() {
                let chunk_start_step = (chunk_idx as i64) * CHUNK_SIZE;
                let chunk_end_step = ((chunk_idx as i64 + 1) * CHUNK_SIZE).min(rollout_steps);
                let chunk_len = chunk_end_step - chunk_start_step;
                let chunk_sample_count = chunk_len * NPROCS;
                let chunk_sample_start = chunk_start_step * NPROCS;

                let price_deltas_chunk =
                    price_deltas_batch.narrow(0, chunk_sample_start, chunk_sample_count);

                // Restore SSM state from saved chunk boundary
                stream_state
                    .ssm_state_batched
                    .conv_state
                    .copy_(&chunk_ssm_states[chunk_idx].conv_state);
                stream_state
                    .ssm_state_batched
                    .ssm_state
                    .copy_(&chunk_ssm_states[chunk_idx].ssm_state);

                let mut mini_batch_start = 0i64;

                while mini_batch_start < chunk_len {
                    let mini_batch_len = MINI_BATCH_STEPS.min(chunk_len - mini_batch_start);
                    let mini_batch_samples = mini_batch_len * NPROCS;

                    // Batched forward: parallelize conv/head, sequential SSM
                    let mb_global_start = chunk_start_step + mini_batch_start;

                    // Gather price_deltas for all steps: [seq, batch, features]
                    let price_deltas_seq = price_deltas_chunk
                        .narrow(0, mini_batch_start * NPROCS, mini_batch_samples)
                        .view([mini_batch_len, NPROCS, -1]);
                    let static_obs_seq = static_obs_batch
                        .narrow(0, mb_global_start * NPROCS, mini_batch_samples)
                        .view([mini_batch_len, NPROCS, -1]);

                    let (critics, critic_logits, (action_means, action_log_stds, _), _) =
                        trading_model.forward_sequence_with_state(
                            &price_deltas_seq,
                            &static_obs_seq,
                            &mut stream_state,
                        );
                    let critics = critics.view([mini_batch_samples, 1]);
                    let critic_logits = critic_logits.view([mini_batch_samples, 255]);

                    // Get stored data for this mini-batch
                    let mb_sample_start = (chunk_start_step + mini_batch_start) * NPROCS;

                    let actions_mb = actions.narrow(0, mb_sample_start, mini_batch_samples);
                    let returns_mb = returns.narrow(0, mb_sample_start, mini_batch_samples);
                    let advantages_mb = advantages.narrow(0, mb_sample_start, mini_batch_samples);
                    let old_log_probs_mb =
                        old_log_probs.narrow(0, mb_sample_start, mini_batch_samples);

                    // Logistic-normal log-prob with Jacobian
                    let u = actions_mb;
                    let zeros = zeros_max.narrow(0, 0, mini_batch_samples);
                    let u_ext = Tensor::cat(&[u.shallow_clone(), zeros], 1);

                    // log N(u; μ, σ)
                    let action_std = action_log_stds.exp();
                    let u_normalized = (&u - &action_means) / &action_std;
                    let u_squared = u_normalized.pow_tensor_scalar(2);
                    let two_log_std = &action_log_stds * 2.0;
                    let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
                    let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);

                    // Jacobian: -Σ log(y_i) = -Σ log_softmax(u_ext)_i (numerically stable)
                    let log_jacobian = u_ext
                        .log_softmax(-1, Kind::Float)
                        .sum_dim_intlist(-1, false, Kind::Float);
                    let action_log_probs = log_prob_gaussian - log_jacobian;

                    // Entropy of Gaussian (K-1 dims)
                    let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * &action_log_stds;
                    let dist_entropy = entropy_components
                        .g_mul_scalar(0.5)
                        .sum_dim_intlist(-1, false, Kind::Float)
                        .mean(Kind::Float);

                    // Value loss: CE for distributional head + clipped MSE on expectation
                    // Returns are in symlog space, bin centers are in symlog space
                    let target_twohot =
                        twohot_encode(&returns_mb, trading_model.symlog_centers());
                    let log_probs = critic_logits.log_softmax(-1, Kind::Float);
                    let ce_loss = -(target_twohot * log_probs).sum_dim_intlist(-1, false, Kind::Float).mean(Kind::Float);

                    // PPO-style clipped MSE on expected value (symlog space)
                    let old_values_mb = s_values_flat.narrow(0, mb_sample_start, mini_batch_samples);
                    let values_pred = critics.squeeze_dim(-1);
                    let returns_t = returns_mb.squeeze_dim(-1);
                    let values_clipped = &old_values_mb + (&values_pred - &old_values_mb).clamp(-VALUE_CLIP_RANGE, VALUE_CLIP_RANGE);

                    let v_loss_unclipped = (&values_pred - &returns_t).pow_tensor_scalar(2);
                    let v_loss_clipped = (&values_clipped - &returns_t).pow_tensor_scalar(2);
                    let v_loss_clip = Tensor::max_other(&v_loss_unclipped, &v_loss_clipped).mean(Kind::Float);

                    let value_loss: Tensor = ce_loss + 0.01 * v_loss_clip;

                    // PPO clipped objective
                    let ratio = (&action_log_probs - &old_log_probs_mb).exp();

                    // Clip fraction diagnostic
                    let clipped_count = tch::no_grad(|| {
                        let deviation = (&ratio - 1.0).abs();
                        i64::try_from(deviation.gt(PPO_CLIP_RATIO).sum(Kind::Int64)).unwrap_or(0)
                    });
                    total_clipped += clipped_count;
                    total_ratio_samples += mini_batch_samples;

                    let unclipped_obj = &ratio * &advantages_mb;
                    let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_RATIO, 1.0 + PPO_CLIP_RATIO);
                    let clipped_obj = ratio_clipped * &advantages_mb;
                    let action_loss =
                        -Tensor::min_other(&unclipped_obj, &clipped_obj).mean(Kind::Float);

                    // KL and loss for metrics (extract before backward frees graph)
                    let approx_kl_val =
                        tch::no_grad(|| (&old_log_probs_mb - &action_log_probs).mean(Kind::Float));
                    let loss =
                        value_loss * VALUE_LOSS_COEF + action_loss - dist_entropy * ENTROPY_COEF;
                    let loss_val = tch::no_grad(|| loss.shallow_clone());

                    // Backward immediately (frees graph, accumulates grads)
                    let scaled_loss =
                        &loss * (mini_batch_samples as f64 / samples_per_accum as f64);
                    scaled_loss.backward();

                    // Detach SSM state after backward to break graph for next mini-batch (TBPTT)
                    stream_state.ssm_state_batched.conv_state =
                        stream_state.ssm_state_batched.conv_state.detach();
                    stream_state.ssm_state_batched.ssm_state =
                        stream_state.ssm_state_batched.ssm_state.detach();

                    // Accumulate metrics on GPU
                    let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * mini_batch_samples as f64));
                    let _ = total_loss_weighted.g_add_(&(&loss_val * mini_batch_samples as f64));
                    let _ = total_kl_weighted.g_add_(&(&approx_kl_val * mini_batch_samples as f64));
                    epoch_kl_count += mini_batch_samples;
                    total_sample_count += mini_batch_samples;

                    mini_batch_start += mini_batch_len;
                }

                // Step optimizer after accumulating k chunks
                if (chunk_i + 1) % GRAD_ACCUM_STEPS == 0 || chunk_i == chunk_order.len() - 1 {
                    let batch_grad_norm = tch::no_grad(|| {
                        let mut norm_sq = Tensor::zeros([], (Kind::Float, device));
                        for v in opt.trainable_variables() {
                            let g = v.grad();
                            if g.defined() {
                                norm_sq += g.pow_tensor_scalar(2).sum(Kind::Float);
                            }
                        }
                        norm_sq.sqrt()
                    });
                    grad_norm_sum += &batch_grad_norm;
                    grad_norm_count += 1;

                    opt.clip_grad_norm(MAX_GRAD_NORM);
                    opt.step();
                    opt.zero_grad();
                }
            }

            // Single GPU->CPU sync per epoch for KL early stopping check
            let mean_epoch_kl = f64::try_from(&epoch_kl_gpu / epoch_kl_count as f64).unwrap_or(0.0);

            if mean_epoch_kl.is_finite() && mean_epoch_kl > TARGET_KL * KL_STOP_MULTIPLIER {
                println!(
                    "Epoch {}/{}: KL {:.4} > {:.4}, stopping",
                    _epoch + 1,
                    OPTIM_EPOCHS,
                    mean_epoch_kl,
                    TARGET_KL * KL_STOP_MULTIPLIER
                );
                break 'epoch_loop;
            }

            println!(
                "Epoch {}/{}: KL {:.4}",
                _epoch + 1,
                OPTIM_EPOCHS,
                mean_epoch_kl
            );
        }

        // Record std and divisor stats every episode
        let (mean_std, min_std, max_std, mean_div) = tch::no_grad(|| {
            let price_deltas_step = s_price_deltas.get(0);
            let static_obs = s_static_obs.get(0);
            let (_, _, (_, action_log_std, divisor), _) =
                trading_model.forward(&price_deltas_step, &static_obs, false);
            let std = action_log_std.exp();
            (
                f64::try_from(std.mean(Kind::Float)).unwrap_or(0.0),
                f64::try_from(std.min()).unwrap_or(0.0),
                f64::try_from(std.max()).unwrap_or(0.0),
                f64::try_from(divisor.mean(Kind::Float)).unwrap_or(0.0),
            )
        });
        env.primary_mut()
            .meta_history
            .record_std_stats(mean_std, min_std, max_std);
        env.primary_mut().meta_history.record_divisor(mean_div);

        // Single GPU->CPU sync for loss and grad norm at end of all epochs
        let mean_loss = if total_sample_count > 0 {
            f64::try_from(&total_loss_weighted / total_sample_count as f64).unwrap_or(0.0)
        } else {
            0.0
        };
        env.primary_mut().meta_history.record_loss(mean_loss);
        let mean_grad_norm = if grad_norm_count > 0 {
            f64::try_from(&grad_norm_sum / grad_norm_count as f64).unwrap_or(0.0)
        } else {
            0.0
        };
        env.primary_mut()
            .meta_history
            .record_grad_norm(mean_grad_norm);

        if episode > 0 && episode % 25 == 0 {
            // Debug: Check if exploration has collapsed or network diverged
            let (_, _, (debug_mean, debug_log_std, _debug_divisor), _attn_weights) =
                tch::no_grad(|| {
                    let price_deltas_step = s_price_deltas.get(0);
                    let static_obs = s_static_obs.get(0);
                    trading_model.forward(&price_deltas_step, &static_obs, false)
                });

            let mean_std = f64::try_from(debug_log_std.exp().mean(Kind::Float)).unwrap();
            let max_raw_action = f64::try_from(debug_mean.abs().max()).unwrap();

            let avg_kl = if total_sample_count > 0 {
                f64::try_from(&total_kl_weighted / total_sample_count as f64).unwrap_or(0.0)
            } else {
                0.0
            };
            // Per-dimension KL for comparison (total KL scales with ACTION_COUNT)
            let kl_per_dim = avg_kl / ACTION_COUNT as f64;

            let opt_end = Instant::now();

            // Compute clip fraction
            let clip_frac = if total_ratio_samples > 0 {
                total_clipped as f64 / total_ratio_samples as f64
            } else {
                0.0
            };

            println!(
                "[Ep {:6}] Episodes: {:.0}, Avg reward: {:.4}, Opt time: {:.2}s, KL: {:.4} ({:.4}/dim), Clip: {:.1}%, Std: {:.4}",
                episode,
                total_episodes,
                total_rewards / total_episodes,
                opt_end.duration_since(opt_start).as_secs_f32(),
                avg_kl,
                kl_per_dim,
                clip_frac * 100.0,
                mean_std
            );

            // Warn if network is diverging
            if max_raw_action > 100.0 {
                println!(
                    "WARNING: Network may be diverging! Raw action magnitude: {:.1}",
                    max_raw_action
                );
            }
            // Warn if clip fraction is too high (policy changing too fast)
            if clip_frac > 0.3 {
                println!("WARNING: High clip fraction ({:.1}%)", clip_frac * 100.0);
            }

            total_rewards = 0.;
            total_episodes = 0.;
        }
        if episode > 0 && episode % 50 == 0 {
            std::fs::create_dir_all("weights").ok();
            if let Err(err) = vs.save(format!("{WEIGHTS_PATH}/ppo_ep{}.safetensors", episode)) {
                println!("Error while saving weights: {}", err)
            } else {
                println!("Saved model weights: {WEIGHTS_PATH}/ppo_ep{}.safetensors", episode);
            }
        }
    }
}
