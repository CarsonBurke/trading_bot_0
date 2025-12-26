use shared::paths::WEIGHTS_PATH;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, OBSERVATION_SPACE, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS,
    STEPS_PER_EPISODE, TICKERS_COUNT,
};
use crate::torch::env::VecEnv;
use crate::torch::model::{TradingModel, LOGIT_SCALE_GROUP};
use crate::torch::load::load_var_store_partial;

struct GpuRollingBuffer {
    buffer: Tensor,
    pos: i64,
}

impl GpuRollingBuffer {
    fn new(device: Device) -> Self {
        Self {
            buffer: Tensor::zeros(
                [NPROCS, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64],
                (Kind::Float, device),
            ),
            pos: 0,
        }
    }

    fn init_from_vecs(&mut self, all_deltas: &[Vec<f32>]) {
        for (i, deltas) in all_deltas.iter().enumerate() {
            let t =
                Tensor::from_slice(deltas).view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);
            self.buffer.get(i as i64).copy_(&t);
        }
        self.pos = 0;
    }

    fn push(&mut self, new_deltas: &Tensor, is_done: &Tensor, reset_deltas: Option<&[Vec<f32>]>) {
        self.buffer
            .narrow(2, self.pos, 1)
            .copy_(&new_deltas.unsqueeze(-1));
        self.pos = (self.pos + 1) % PRICE_DELTAS_PER_TICKER as i64;

        if let Some(resets) = reset_deltas {
            let done_mask = Vec::<f32>::try_from(is_done.flatten(0, -1)).unwrap_or_default();
            for (i, (&done, deltas)) in done_mask.iter().zip(resets.iter()).enumerate() {
                if done > 0.5 {
                    let t = Tensor::from_slice(deltas)
                        .view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);
                    self.buffer.get(i as i64).copy_(&t);
                }
            }
        }
    }

    fn get_flat(&self) -> Tensor {
        let ordered = if self.pos == 0 {
            self.buffer.shallow_clone()
        } else {
            let len = PRICE_DELTAS_PER_TICKER as i64;
            let older = self.buffer.narrow(2, self.pos, len - self.pos);
            let newer = self.buffer.narrow(2, 0, self.pos);
            Tensor::cat(&[older, newer], 2)
        };
        ordered.view([NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64])
    }

    fn reset(&mut self) {
        let _ = self.buffer.zero_();
        self.pos = 0;
    }
}

pub const NPROCS: i64 = 16; // Parallel environments for better GPU utilization
const UPDATES: i64 = 1000000;
const OPTIM_EPOCHS: i64 = 4;
const CHUNK_SIZE: i64 = 64; // Steps per chunk for PPO updates

const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)
const GAMMA: f64 = 0.99;
const GAE_LAMBDA: f64 = 0.95;

fn symlog(x: &Tensor) -> Tensor {
    x.sign() * (x.abs() + 1.0).log()
}

// gSDE: resample exploration noise every N env steps (temporally correlated exploration).
// This matches the common SB3-style behavior while keeping per-step log-prob computation unchanged.
const SDE_SAMPLE_FREQ: usize = 1;

// PPO hyperparameters
const PPO_CLIP_RATIO: f64 = 0.2; // Clip range for policy ratio (the trust region)
const VALUE_CLIP_RANGE: f64 = 0.0; // 0.0 disables value clipping
const ENTROPY_COEF: f64 = 0.0;
const VALUE_LOSS_COEF: f64 = 0.5; // Value loss coefficient
const MAX_GRAD_NORM: f64 = 0.5; // Gradient clipping norm
                                // Conservative KL early stopping (SB3-style)
const TARGET_KL: f64 = 0.03;
const KL_STOP_MULTIPLIER: f64 = 1.5;
const LEARNING_RATE: f64 = 4e-5;
const LOGIT_SCALE_LR_MULT: f64 = 10.0;

// RPO: adaptive alpha targeting induced KL (total KL, not per-dim)
const RPO_ALPHA_MIN: f64 = 0.005;
const RPO_ALPHA_MAX: f64 = 0.5;
const RPO_TARGET_KL: f64 = 0.018;
const RPO_ALPHA_INIT: f64 = 0.1;
const ALPHA_LOSS_COEF: f64 = 0.1;
const GRAD_ACCUM_STEPS: usize = 2; // Accumulate gradients over k chunks before stepping (was 4, reduced for more updates)

pub fn train(weights_path: Option<&str>) {
    let mut env = VecEnv::new(true);

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
    let trading_model = TradingModel::new(&vs.root());

    if let Some(path) = weights_path {
        println!("Loading weights from: {}", path);
        match load_var_store_partial(&mut vs, path) {
            Ok(_) => println!("Weights loaded successfully, resuming training"),
            Err(e) => panic!("Failed to load weights: {:?}", e),
        }
    } else {
        println!("Starting training from scratch");
    }

    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();
    opt.set_lr_group(LOGIT_SCALE_GROUP, LEARNING_RATE * LOGIT_SCALE_LR_MULT);

    // Disabled FP16 - with GAP reducing params from 39M to 284K, FP32 easily fits in VRAM
    // FP16 causes NaN issues in tch-rs, especially with complex architectures
    // vs.half();

    // Create device-specific kind
    let float_kind = (Kind::Float, device);

    let mut sum_rewards = Tensor::zeros([NPROCS], float_kind);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;
    let mut total_rewards_gpu = Tensor::zeros([], float_kind);
    let mut total_episodes_gpu = Tensor::zeros([], float_kind);

    let mut s_values_buf: Option<Tensor> = None;
    let mut s_rewards_buf: Option<Tensor> = None;
    let mut s_actions_buf: Option<Tensor> = None;
    let mut s_masks_buf: Option<Tensor> = None;
    let mut s_log_probs_buf: Option<Tensor> = None;

    // RPO alpha: sigmoid parameterization for smooth gradients
    // rho -> alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(rho)
    // Compute init rho from desired alpha: rho = logit((alpha - min) / (max - min))
    let p_init = (RPO_ALPHA_INIT - RPO_ALPHA_MIN) / (RPO_ALPHA_MAX - RPO_ALPHA_MIN);
    let rho_init = (p_init / (1.0 - p_init)).ln(); // logit
    let mut rpo_rho = vs
        .root()
        .var("rpo_alpha_rho", &[1], nn::Init::Const(rho_init));
    // Note: rpo_rho is in main VarStore, so it will be saved/loaded with model weights

    let _ = env.reset();

    let mut rolling_buffer = GpuRollingBuffer::new(device);
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
        (Kind::Float, device),
    );

    for episode in 0..UPDATES {
        let mut get_buf =
            |buf: &mut Option<Tensor>, size: &[i64], kind: (Kind, Device)| -> Tensor {
                let size_vec = size.to_vec();
                match buf {
                    Some(t) => {
                        if t.size() != size_vec {
                            *t = Tensor::zeros(size, kind);
                        } else {
                            let _ = t.zero_();
                        }
                        t.shallow_clone()
                    }
                    None => {
                        let t = Tensor::zeros(size, kind);
                        *buf = Some(t);
                        buf.as_ref().unwrap().shallow_clone()
                    }
                }
            };

        let (init_deltas, static_obs_reset) = env.reset_incremental();
        env.set_episode(episode as usize);

        rolling_buffer.init_from_vecs(&init_deltas);
        s_price_deltas.get(0).copy_(&rolling_buffer.get_flat());
        s_static_obs.get(0).copy_(&static_obs_reset);

        let rollout_steps = env.max_step() as i64;
        let memory_size = rollout_steps * NPROCS;

        let s_values = get_buf(
            &mut s_values_buf,
            &[rollout_steps, NPROCS, ACTION_COUNT],
            float_kind,
        );
        let mut s_rewards = get_buf(
            &mut s_rewards_buf,
            &[rollout_steps, NPROCS, ACTION_COUNT],
            float_kind,
        );
        let s_actions = get_buf(
            &mut s_actions_buf,
            &[rollout_steps, NPROCS, ACTION_COUNT - 1],
            float_kind,
        );
        let s_masks = get_buf(&mut s_masks_buf, &[rollout_steps, NPROCS], float_kind);
        let s_log_probs = get_buf(&mut s_log_probs_buf, &[rollout_steps, NPROCS], float_kind);

        let _ = total_rewards_gpu.zero_();
        let _ = total_episodes_gpu.zero_();
        let zeros_rollout = Tensor::zeros([NPROCS, 1], float_kind);

        // Use a separate index (s) for tensor storage, starting from 0
        // Loop through the episode using relative steps (0 to max_step)
        let mut s: i64 = 0;
        let mut sde_noise: Option<Tensor> = None;
        let mut last_attn_weights: Option<Tensor> = None;
        for step in 0..env.max_step() {
            env.set_step(step);

            let (values, (action_mean, action_log_std, sde_latent), attn_weights) =
                tch::no_grad(|| {
                    let price_deltas_step = s_price_deltas.get(s);
                    let static_obs = s_static_obs.get(s);
                    trading_model.forward(&price_deltas_step, &static_obs, false)
                });
            last_attn_weights = Some(attn_weights.shallow_clone());

            // Logistic-normal with softmax simplex projection
            let action_log_std_clamped = action_log_std.clamp(-20.0, 5.0);
            let action_std = action_log_std_clamped.exp();
            let latent_norm = sde_latent
                .pow_tensor_scalar(2)
                .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
                .sqrt()
                .clamp_min(1e-6);

            if sde_noise.is_none() || (step % SDE_SAMPLE_FREQ == 0) {
                sde_noise = Some(Tensor::randn_like(&sde_latent));
            }
            let noise = sde_noise.as_ref().unwrap();
            let noise_raw = (&sde_latent * noise).sum_dim_intlist([-1].as_slice(), false, Kind::Float);
            let noise_scaled = &noise_raw * &action_std / &latent_norm;
            let action_mean_noisy = &action_mean + &noise_scaled;
            let u = action_mean_noisy.shallow_clone(); // [batch, K-1]
            let u_ext = Tensor::cat(&[u.shallow_clone(), zeros_rollout.shallow_clone()], 1);
            let actions = u_ext.softmax(-1, Kind::Float);

            // Log-prob with softmax Jacobian
            let u_normalized = (&u - &action_mean) / &action_std;
            let u_squared = u_normalized.pow_tensor_scalar(2);
            let two_log_std = &action_log_std_clamped * 2.0;
            let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
            let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
            let log_jacobian = u_ext
                .log_softmax(-1, Kind::Float)
                .sum_dim_intlist(-1, false, Kind::Float);
            let action_log_prob = log_prob_gaussian - log_jacobian;

            let mut out_so = s_static_obs.get(s + 1);
            let (reward, reward_per_ticker, is_done, step_deltas) =
                env.step_incremental_tensor(&actions, &mut out_so);

            let step_deltas_gpu = step_deltas.to_device(device);
            rolling_buffer.push(&step_deltas_gpu, &is_done, None);
            s_price_deltas.get(s + 1).copy_(&rolling_buffer.get_flat());

            let reward = reward.to_device(device);
            let reward_per_ticker = symlog(&reward_per_ticker);
            let is_done = is_done.to_device(device);

            sum_rewards += &reward;
            let completed_rewards = (&sum_rewards * &is_done).sum(Kind::Float);
            let completed_episodes = is_done.sum(Kind::Float);
            let _ = total_rewards_gpu.g_add_(&completed_rewards);
            let _ = total_episodes_gpu.g_add_(&completed_episodes);

            let masks = Tensor::from(1f32).to_device(device) - &is_done;
            sum_rewards *= &masks;

            s_actions.get(s).copy_(&u); // Store pre-softmax u for training
            s_values.get(s).copy_(&values);
            s_log_probs.get(s).copy_(&action_log_prob);
            s_rewards.get(s).copy_(&reward_per_ticker);
            s_masks.get(s).copy_(&masks);

            s += 1; // Increment storage index
        }

        if let Some(attn) = last_attn_weights.take() {
            let static_obs_vec =
                Vec::<f32>::try_from(s_static_obs.get(rollout_steps - 1).flatten(0, -1))
                    .unwrap_or_default();
            let attn_weights_vec = Vec::<f32>::try_from(attn.flatten(0, -1)).unwrap_or_default();
            env.primary_mut()
                .episode_history
                .static_observations
                .push(static_obs_vec);
            env.primary_mut()
                .episode_history
                .attention_weights
                .push(attn_weights_vec);
        }

        total_rewards += f64::try_from(&total_rewards_gpu).unwrap_or(0.0);
        total_episodes += f64::try_from(&total_episodes_gpu).unwrap_or(0.0);

        let price_deltas_batch = s_price_deltas
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_batch = s_static_obs
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, STATIC_OBSERVATIONS as i64]);

        let (advantages, returns) = {
            let gae = Tensor::zeros(
                [rollout_steps + 1, NPROCS, ACTION_COUNT],
                (Kind::Float, device),
            );
            let returns = Tensor::zeros(
                [rollout_steps + 1, NPROCS, ACTION_COUNT],
                (Kind::Float, device),
            );

            // Bootstrap value for next state (SSM state is already at end-of-rollout)
            let next_value = tch::no_grad(|| {
                let price_deltas_step = s_price_deltas.get(rollout_steps);
                let static_obs = s_static_obs.get(rollout_steps);
                let (values, _, _) = trading_model.forward(&price_deltas_step, &static_obs, false);
                values
            });
            // Compute GAE backwards per ticker.
            for s in (0..rollout_steps).rev() {
                let value_t = s_values.get(s);
                let value_next = if s == rollout_steps - 1 {
                    next_value.shallow_clone()
                } else {
                    s_values.get(s + 1)
                };

                let mask = s_masks.get(s).unsqueeze(-1);
                let delta = s_rewards.get(s) + GAMMA * &value_next * &mask - &value_t;

                // GAE: A_t = δ_t + γ * λ * mask * A_{t+1}
                let gae_next = if s == rollout_steps - 1 {
                    Tensor::zeros_like(&delta)
                } else {
                    gae.get(s + 1)
                };
                let gae_t = delta + GAMMA * GAE_LAMBDA * &mask * gae_next;
                gae.get(s).copy_(&gae_t);

                let return_t = &gae_t + &value_t;
                returns.get(s).copy_(&return_t);
            }

            (
                gae.narrow(0, 0, rollout_steps)
                    .view([memory_size, ACTION_COUNT]),
                returns
                    .narrow(0, 0, rollout_steps)
                    .view([memory_size, ACTION_COUNT]),
            )
        };
        let actions = s_actions.view([memory_size, ACTION_COUNT - 1]);
        let old_log_probs = s_log_probs.view([memory_size]);
        let s_values_flat = s_values.view([memory_size, ACTION_COUNT]).detach();

        // Record advantage stats before normalization
        let adv_stats = Tensor::stack(
            &[
                advantages.mean(Kind::Float),
                advantages.min(),
                advantages.max(),
            ],
            0,
        );
        let adv_stats_vec = Vec::<f64>::try_from(adv_stats).unwrap_or_default();
        let adv_mean_val = *adv_stats_vec.get(0).unwrap_or(&0.0);
        let adv_min_val = *adv_stats_vec.get(1).unwrap_or(&0.0);
        let adv_max_val = *adv_stats_vec.get(2).unwrap_or(&0.0);
        env.primary_mut().meta_history.record_advantage_stats(
            adv_mean_val,
            adv_min_val,
            adv_max_val,
        );

        // Normalize advantages per ticker across batch/time for consistent gradients as K grows
        // Detach to prevent backprop through GAE computation
        let adv_mean = advantages.mean_dim(0, true, Kind::Float);
        let adv_var =
            (&advantages - &adv_mean)
                .pow_tensor_scalar(2.0)
                .mean_dim(0, true, Kind::Float);
        let adv_std = adv_var.sqrt().clamp_min(1e-4);
        let advantages = ((&advantages - adv_mean) / adv_std)
            .clamp(-3.0, 3.0)
            .detach();

        // Returns are raw, just detach
        let returns = returns.detach();

        let opt_start = Instant::now();
        // Accumulate on GPU - only sync once at end of all epochs (weighted by samples)
        let mut total_kl_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_policy_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_value_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_sample_count = 0i64;
        let mut grad_norm_sum = Tensor::zeros([], (Kind::Float, device));
        let mut grad_norm_count = 0i64;
        // Clip fraction diagnostic
        let mut total_clipped = Tensor::zeros([], (Kind::Float, device));
        let mut total_ratio_samples = 0i64;

        let num_chunks = (rollout_steps + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunk_order: Vec<usize> = (0..num_chunks as usize).collect();
        let zeros_mb = Tensor::zeros([CHUNK_SIZE * NPROCS, 1], float_kind);

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;

            // Shuffle chunk order each epoch for gradient diversity
            use rand::seq::SliceRandom;
            chunk_order.shuffle(&mut rand::rng());

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
                let static_obs_chunk =
                    static_obs_batch.narrow(0, chunk_sample_start, chunk_sample_count);

                let (values, (action_means, action_log_stds, _), _) = trading_model
                    .forward(&price_deltas_chunk, &static_obs_chunk, true);
                let values = values.view([chunk_sample_count, ACTION_COUNT]);

                let actions_mb = actions.narrow(0, chunk_sample_start, chunk_sample_count);
                let returns_mb = returns.narrow(0, chunk_sample_start, chunk_sample_count);
                let advantages_mb = advantages.narrow(0, chunk_sample_start, chunk_sample_count);
                let old_log_probs_mb =
                    old_log_probs.narrow(0, chunk_sample_start, chunk_sample_count);

                let u = actions_mb;
                let zeros = zeros_mb.narrow(0, 0, chunk_sample_count);
                let u_ext = Tensor::cat(&[u.shallow_clone(), zeros], 1);

                // log N(u; μ_perturbed, σ)
                let action_log_stds_clamped = action_log_stds.clamp(-20.0, 5.0);
                let action_std = action_log_stds_clamped.exp();
                let two_log_std = &action_log_stds_clamped * 2.0;
                // RPO alpha: sigmoid parameterization for smooth gradients everywhere
                let rpo_alpha =
                    RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                let rpo_alpha_detached = rpo_alpha.detach();
                let rpo_noise =
                    Tensor::empty_like(&action_means).uniform_(-1.0, 1.0) * &rpo_alpha_detached;
                let action_means_noisy = &action_means + &rpo_noise;

                let u_normalized = (&u - &action_means_noisy) / &action_std;
                let u_squared = u_normalized.pow_tensor_scalar(2);
                let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
                let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
                let log_jacobian = u_ext
                    .log_softmax(-1, Kind::Float)
                    .sum_dim_intlist(-1, false, Kind::Float);
                let action_log_probs = log_prob_gaussian - log_jacobian;

                // Entropy of Gaussian (per ticker)
                let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * &action_log_stds;
                let dist_entropy = entropy_components.g_mul_scalar(0.5).mean(Kind::Float);

                // PPO-style clipped MSE on per-ticker values (raw space)
                let old_values_mb =
                    s_values_flat.narrow(0, chunk_sample_start, chunk_sample_count);
                let values_pred = values;
                let returns_t = returns_mb;
                let values_clipped = &old_values_mb
                    + (&values_pred - &old_values_mb)
                        .clamp(-VALUE_CLIP_RANGE, VALUE_CLIP_RANGE);

                let v_loss_unclipped = (&values_pred - &returns_t).pow_tensor_scalar(2.0);
                let value_loss = if VALUE_CLIP_RANGE > 0.0 {
                    let v_loss_clipped = (&values_clipped - &returns_t).pow_tensor_scalar(2.0);
                    Tensor::max_other(&v_loss_unclipped, &v_loss_clipped).mean(Kind::Float)
                } else {
                    v_loss_unclipped.mean(Kind::Float)
                };

                // PPO clipped objective
                let ratio = (&action_log_probs - &old_log_probs_mb).exp();

                // Clip fraction diagnostic
                let clipped_count = tch::no_grad(|| {
                    let deviation = (&ratio - 1.0).abs();
                    deviation
                        .gt(PPO_CLIP_RATIO)
                        .to_kind(Kind::Float)
                        .sum(Kind::Float)
                });
                let _ = total_clipped.g_add_(&clipped_count);
                total_ratio_samples += chunk_sample_count;

                let ticker_adv = advantages_mb.narrow(1, 0, ACTION_COUNT - 1);
                let ticker_sum =
                    ticker_adv.sum_dim_intlist([-1].as_slice(), false, Kind::Float);
                let cash_adv = advantages_mb
                    .narrow(1, ACTION_COUNT - 1, 1)
                    .squeeze_dim(-1);
                let cash_weight = 0.25;
                let denom = (ACTION_COUNT - 1) as f64 + cash_weight;
                let advantages_reduced = (ticker_sum + cash_adv * cash_weight) / denom;
                let unclipped_obj = &ratio * &advantages_reduced;
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_RATIO, 1.0 + PPO_CLIP_RATIO);
                let clipped_obj = ratio_clipped * &advantages_reduced;
                let action_loss =
                    -Tensor::min_other(&unclipped_obj, &clipped_obj).mean(Kind::Float);

                // KL and loss for metrics (extract before backward frees graph)
                let approx_kl_val =
                    tch::no_grad(|| (&old_log_probs_mb - &action_log_probs).mean(Kind::Float));
                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - dist_entropy * ENTROPY_COEF;
                let loss_val = tch::no_grad(|| ppo_loss.shallow_clone());
                let policy_loss_val = tch::no_grad(|| action_loss.shallow_clone());
                let value_loss_val = tch::no_grad(|| value_loss.shallow_clone());

                // Backward PPO loss (alpha detached, so no alpha gradients here)
                let scaled_ppo_loss =
                    &ppo_loss * (chunk_sample_count as f64 / samples_per_accum as f64);
                scaled_ppo_loss.backward();

                // Alpha loss: target induced KL using detached network outputs
                // For diagonal Gaussian with z_i ~ U(-alpha, alpha):
                // E[KL] = sum_i E[z_i^2] / (2*sigma_i^2) = sum_i (alpha^2/3) / (2*sigma_i^2)
                //       = d * (alpha^2/6) * mean(1/sigma^2)  where d = ACTION_COUNT
                let action_std_detached = action_log_stds.detach().clamp(-20.0, 5.0).exp();
                let var = action_std_detached.pow_tensor_scalar(2);
                let inv_var_mean = var.clamp_min(1e-4).reciprocal().mean(Kind::Float);
                let d = ACTION_COUNT as f64;
                let induced_kl: Tensor =
                    rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                let alpha_loss = (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0);
                (alpha_loss * ALPHA_LOSS_COEF).backward();

                // Accumulate metrics on GPU
                let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                let _ = total_loss_weighted.g_add_(&(&loss_val * chunk_sample_count as f64));
                let _ = total_policy_loss_weighted.g_add_(&(&policy_loss_val * chunk_sample_count as f64));
                let _ = total_value_loss_weighted.g_add_(&(&value_loss_val * chunk_sample_count as f64));
                let _ = total_kl_weighted.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                epoch_kl_count += chunk_sample_count;
                total_sample_count += chunk_sample_count;

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

                    // Clamp delta rho on GPU (conservative bound for delta alpha)
                    // Max sigmoid slope is 0.25, so |Δα| <= range * 0.25 * |Δρ|
                    // Thus |Δρ| <= MAX_DELTA_ALPHA / (0.25 * range) ensures |Δα| <= MAX_DELTA_ALPHA
                    const MAX_DELTA_ALPHA: f64 = 0.02;
                    let range = RPO_ALPHA_MAX - RPO_ALPHA_MIN;
                    let max_delta_rho = MAX_DELTA_ALPHA / (0.25 * range);

                    tch::no_grad(|| {
                        let rho_before = rpo_rho.detach();
                        opt.step();
                        let delta_rho = &rpo_rho - &rho_before;
                        let delta_rho_clamped = delta_rho.clamp(-max_delta_rho, max_delta_rho);
                        let rho_new = rho_before + delta_rho_clamped;
                        let _ = rpo_rho.copy_(&rho_new);
                    });
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

        // Get current alpha value for logging/charting
        // Record std stats every episode
        let stats_tensor = tch::no_grad(|| {
            let price_deltas_step = s_price_deltas.get(0);
            let static_obs = s_static_obs.get(0);
            let (_, (_, action_log_std, _), _) =
                trading_model.forward(&price_deltas_step, &static_obs, false);
            let std = action_log_std.exp();
            let rpo_alpha =
                (RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid()).squeeze();
            let logit_scale = trading_model.logit_scale().squeeze();
            Tensor::stack(
                &[
                    std.mean(Kind::Float),
                    std.min(),
                    std.max(),
                    rpo_alpha,
                    logit_scale,
                ],
                0,
            )
        });
        let stats_vec = Vec::<f64>::try_from(stats_tensor).unwrap_or_default();
        let mean_std = *stats_vec.get(0).unwrap_or(&0.0);
        let min_std = *stats_vec.get(1).unwrap_or(&0.0);
        let max_std = *stats_vec.get(2).unwrap_or(&0.0);
        let rpo_alpha = *stats_vec.get(3).unwrap_or(&0.15);
        let logit_scale = *stats_vec.get(4).unwrap_or(&1.0);
        env.primary_mut()
            .meta_history
            .record_std_stats(mean_std, min_std, max_std, rpo_alpha);
        env.primary_mut().meta_history.record_logit_scale(logit_scale);

        // Single GPU->CPU sync for loss and grad norm at end of all epochs
        let mean_losses = if total_sample_count > 0 {
            Tensor::stack(
                &[
                    &total_loss_weighted,
                    &total_policy_loss_weighted,
                    &total_value_loss_weighted,
                ],
                0,
            ) / (total_sample_count as f64)
        } else {
            Tensor::zeros([3], (Kind::Float, device))
        };
        let mean_grad_norm = if grad_norm_count > 0 {
            &grad_norm_sum / (grad_norm_count as f64)
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };
        let mean_all = Tensor::cat(&[mean_losses, mean_grad_norm.unsqueeze(0)], 0);
        let mean_all_vec = Vec::<f64>::try_from(mean_all).unwrap_or_default();
        let mean_loss = *mean_all_vec.get(0).unwrap_or(&0.0);
        let mean_policy_loss = *mean_all_vec.get(1).unwrap_or(&0.0);
        let mean_value_loss = *mean_all_vec.get(2).unwrap_or(&0.0);
        let mean_grad_norm = *mean_all_vec.get(3).unwrap_or(&0.0);
        env.primary_mut().meta_history.record_loss(mean_loss);
        env.primary_mut()
            .meta_history
            .record_policy_loss(mean_policy_loss);
        env.primary_mut()
            .meta_history
            .record_value_loss(mean_value_loss);
        env.primary_mut()
            .meta_history
            .record_grad_norm(mean_grad_norm);

        if episode > 0 && episode % 25 == 0 {
            // Debug: Check if exploration has collapsed or network diverged
            let (_, (debug_mean, debug_log_std, _), _attn_weights) =
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
                let total_clipped_val = f64::try_from(&total_clipped).unwrap_or(0.0);
                total_clipped_val / total_ratio_samples as f64
            } else {
                0.0
            };

            println!(
                "[Ep {:6}] Episodes: {:.0}, Avg reward: {:.4}, Opt time: {:.2}s, KL: {:.4} ({:.4}/dim), Clip: {:.1}%, Std: {:.4}, RPO: {:.3}",
                episode,
                total_episodes,
                total_rewards / total_episodes,
                opt_end.duration_since(opt_start).as_secs_f32(),
                avg_kl,
                kl_per_dim,
                clip_frac * 100.0,
                mean_std,
                rpo_alpha
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
                println!(
                    "Saved model weights: {WEIGHTS_PATH}/ppo_ep{}.safetensors",
                    episode
                );
            }
        }
    }
}
