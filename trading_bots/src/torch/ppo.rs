use shared::paths::WEIGHTS_PATH;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, OBSERVATION_SPACE, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS,
    STEPS_PER_EPISODE, TICKERS_COUNT,
};
use crate::torch::env::VecEnv;
use crate::torch::model::{TradingModel, TradingModelConfig, LOGIT_SCALE_GROUP};
use crate::torch::load::load_var_store_partial;
use std::fs;
use std::process;
use std::sync::OnceLock;
use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::UsedGpuMemory;

struct GpuRollingBuffer {
    buffer: Tensor,
    pos: i64,
    kind: Kind,
}

impl GpuRollingBuffer {
    fn new(device: Device, kind: Kind) -> Self {
        Self {
            buffer: Tensor::zeros(
                [NPROCS, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64],
                (kind, device),
            ),
            pos: 0,
            kind,
        }
    }

    fn init_from_vecs(&mut self, all_deltas: &[Vec<f32>]) {
        for (i, deltas) in all_deltas.iter().enumerate() {
            let t =
                Tensor::from_slice(deltas)
                    .to_kind(self.kind)
                    .view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);
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
                        .to_kind(self.kind)
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
const LEARNING_RATE: f64 = 1e-4;
const LOGIT_SCALE_LR_MULT: f64 = 10.0;

// RPO: adaptive alpha targeting induced KL (total KL, not per-dim)
const RPO_ALPHA_MIN: f64 = 0.005;
const RPO_ALPHA_MAX: f64 = 0.5;
const RPO_TARGET_KL: f64 = 0.018;
const RPO_ALPHA_INIT: f64 = 0.1;
const ALPHA_LOSS_COEF: f64 = 0.1;
const ADV_MIXED_WEIGHT: f64 = 0.5;
const GRAD_ACCUM_STEPS: usize = 2; // Accumulate gradients over k chunks before stepping (was 4, reduced for more updates)
const DEBUG_TEMPORAL_REPORTS: bool = true;
const DEBUG_MEMORY_REPORTS: bool = true;

fn read_rss_kb() -> Option<u64> {
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let mut parts = statm.split_whitespace();
    let _ = parts.next()?;
    let rss_pages: u64 = parts.next()?.parse().ok()?;
    let page_kb = (unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64) / 1024;
    Some(rss_pages * page_kb)
}

fn nvml_handle() -> Option<&'static Nvml> {
    static NVML: OnceLock<Nvml> = OnceLock::new();
    if NVML.get().is_none() {
        if let Ok(nvml) = Nvml::init() {
            let _ = NVML.set(nvml);
        } else {
            return None;
        }
    }
    NVML.get()
}

fn read_gpu_mem_mb() -> Option<u64> {
    let nvml = nvml_handle()?;
    let device = nvml.device_by_index(0).ok()?;
    let pid = process::id();
    let processes = device.running_compute_processes().ok()?;
    for proc in processes {
        if proc.pid == pid {
            return match proc.used_gpu_memory {
                UsedGpuMemory::Used(bytes) => Some(bytes / (1024 * 1024)),
                UsedGpuMemory::Unavailable => None,
            };
        }
    }
    None
}

pub fn train(weights_path: Option<&str>) {
    if std::env::var("PYTORCH_ALLOC_CONF").is_err() {
        std::env::set_var("PYTORCH_ALLOC_CONF", "expandable_segments:True");
    }
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
    
    let train_kind = Kind::BFloat16;
    vs.bfloat16();
    println!("Using dtype bf16 for model forward");
    
    let trading_model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());

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

    // Create device-specific kind
    let model_kind = (train_kind, device);
    let stats_kind = (Kind::Float, device);

    let mut sum_rewards = Tensor::zeros([NPROCS], stats_kind);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;
    let mut total_rewards_gpu = Tensor::zeros([], stats_kind);
    let mut total_episodes_gpu = Tensor::zeros([], stats_kind);

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

    let mut rolling_buffer = GpuRollingBuffer::new(device, train_kind);
    let s_price_deltas = Tensor::zeros(
        [
            max_steps + 1,
            NPROCS,
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ],
        model_kind,
    );
    let s_static_obs = Tensor::zeros(
        [max_steps + 1, NPROCS, STATIC_OBSERVATIONS as i64],
        stats_kind,
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
            &[rollout_steps, NPROCS, TICKERS_COUNT + 1],
            stats_kind,
        );
        let mut s_rewards = get_buf(
            &mut s_rewards_buf,
            &[rollout_steps, NPROCS, TICKERS_COUNT + 1],
            stats_kind,
        );
        let s_actions = get_buf(
            &mut s_actions_buf,
            &[rollout_steps, NPROCS, ACTION_COUNT],
            stats_kind,
        );
        let s_masks = get_buf(&mut s_masks_buf, &[rollout_steps, NPROCS], stats_kind);
        let s_log_probs = get_buf(&mut s_log_probs_buf, &[rollout_steps, NPROCS], stats_kind);

        let _ = total_rewards_gpu.zero_();
        let _ = total_episodes_gpu.zero_();

        // Use a separate index (s) for tensor storage, starting from 0
        // Loop through the episode using relative steps (0 to max_step)
        let mut s: i64 = 0;
        let mut sde_noise: Option<Tensor> = None;
        for step in 0..env.max_step() {
            env.set_step(step);

            let (values, (action_logits, action_log_std, sde_latent), _attn_weights) =
                tch::no_grad(|| {
                    let price_deltas_step = s_price_deltas.get(s);
                    let static_obs = s_static_obs.get(s);
                    trading_model.forward(&price_deltas_step, &static_obs, false)
                });
            let values = values.to_kind(Kind::Float);
            let action_logits = action_logits.to_kind(Kind::Float);
            let action_log_std = action_log_std.to_kind(Kind::Float);
            let sde_latent = sde_latent.to_kind(Kind::Float);

            // Logistic-normal with softmax simplex projection
            let action_log_std_clamped = action_log_std.clamp(-20.0, 5.0);
            let action_std = action_log_std_clamped.exp();
            if sde_noise.is_none() || (step % SDE_SAMPLE_FREQ == 0) {
                let std_matrix = trading_model.sde_std_matrix().to_kind(Kind::Float);
                let eps = Tensor::randn(
                    &[
                        sde_latent.size()[0],
                        std_matrix.size()[0],
                        std_matrix.size()[1],
                    ],
                    stats_kind,
                );
                let noise_mat = eps * std_matrix.unsqueeze(0);
                sde_noise = Some(noise_mat);
            }
            let noise_mat = sde_noise.as_ref().unwrap().permute([0, 2, 1]);
            let noise_raw =
                (&sde_latent * noise_mat).sum_dim_intlist([-1].as_slice(), false, Kind::Float);
            let action_logits_noisy = &action_logits + &noise_raw;
            let u = action_logits_noisy.shallow_clone();
            let actions = u.softmax(-1, Kind::Float);

            // Log-prob with softmax Jacobian
            let u_normalized = (&u - &action_logits) / &action_std;
            let u_squared = u_normalized.pow_tensor_scalar(2);
            let two_log_std = &action_log_std_clamped * 2.0;
            let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
            let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
            let log_weights = u.log_softmax(-1, Kind::Float);
            let log_det = log_weights.sum_dim_intlist(-1, false, Kind::Float);
            let action_log_prob = log_prob_gaussian - log_det;

            let mut out_so = s_static_obs.get(s + 1);
            let (reward, reward_per_ticker, is_done, step_deltas) =
                env.step_incremental_tensor(&actions, &mut out_so);

            let step_deltas_gpu = step_deltas.to_device(device);
            rolling_buffer.push(&step_deltas_gpu, &is_done, None);
            s_price_deltas.get(s + 1).copy_(&rolling_buffer.get_flat());

            let reward = reward.to_device(device).to_kind(Kind::Float);
            let reward_per_ticker = reward_per_ticker.to_device(device).to_kind(Kind::Float);
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
            let rewards_full = Tensor::zeros([NPROCS, TICKERS_COUNT + 1], stats_kind);
            rewards_full
                .narrow(1, 0, TICKERS_COUNT)
                .copy_(&reward_per_ticker);
            s_rewards.get(s).copy_(&rewards_full);
            s_masks.get(s).copy_(&masks);

            s += 1; // Increment storage index
        }

        let static_obs_vec =
            Vec::<f32>::try_from(s_static_obs.get(rollout_steps - 1).flatten(0, -1))
                .unwrap_or_default();
        env.primary_mut()
            .episode_history
            .static_observations
            .push(static_obs_vec);

        total_rewards += f64::try_from(&total_rewards_gpu).unwrap_or(0.0);
        total_episodes += f64::try_from(&total_episodes_gpu).unwrap_or(0.0);

        let price_deltas_batch = s_price_deltas
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64])
            .detach();
        let static_obs_batch = s_static_obs
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, STATIC_OBSERVATIONS as i64])
            .detach();

        let (advantages, returns) = {
            let gae = Tensor::zeros(
                [rollout_steps + 1, NPROCS, TICKERS_COUNT + 1],
                (Kind::Float, device),
            );
            let returns = Tensor::zeros(
                [rollout_steps + 1, NPROCS, TICKERS_COUNT + 1],
                (Kind::Float, device),
            );

            // Bootstrap value for next state (SSM state is already at end-of-rollout)
            let next_value = tch::no_grad(|| {
                let price_deltas_step = s_price_deltas.get(rollout_steps);
                let static_obs = s_static_obs.get(rollout_steps);
                let (values, _, _) = trading_model.forward(&price_deltas_step, &static_obs, false);
                values.to_kind(Kind::Float)
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
                    .view([memory_size, TICKERS_COUNT + 1]),
                returns
                    .narrow(0, 0, rollout_steps)
                    .view([memory_size, TICKERS_COUNT + 1]),
            )
        };
        let actions = s_actions.view([memory_size, ACTION_COUNT]).detach();
        let old_log_probs = s_log_probs.view([memory_size]).detach();
        let s_values_flat = s_values.view([memory_size, TICKERS_COUNT + 1]).detach();

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
        let mut total_kl_weighted = Tensor::zeros([], stats_kind);
        let mut total_loss_weighted = Tensor::zeros([], stats_kind);
        let mut total_policy_loss_weighted = Tensor::zeros([], stats_kind);
        let mut total_value_loss_weighted = Tensor::zeros([], stats_kind);
        let mut total_sample_count = 0i64;
        let mut grad_norm_sum = Tensor::zeros([], stats_kind);
        let mut grad_norm_count = 0i64;
        // Clip fraction diagnostic
        let mut total_clipped = Tensor::zeros([], stats_kind);
        let mut total_ratio_samples = 0i64;

        let num_chunks = (rollout_steps + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunk_order: Vec<usize> = (0..num_chunks as usize).collect();

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            let mut epoch_kl_gpu = Tensor::zeros([], stats_kind);
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

                let (values, (action_logits, action_log_stds, _), _) = trading_model
                    .forward(&price_deltas_chunk, &static_obs_chunk, true);
                let values = values
                    .to_kind(Kind::Float)
                    .view([chunk_sample_count, TICKERS_COUNT + 1]);
                let action_logits = action_logits.to_kind(Kind::Float);
                let action_log_stds = action_log_stds.to_kind(Kind::Float);

                let actions_mb = actions.narrow(0, chunk_sample_start, chunk_sample_count);
                let returns_mb = returns.narrow(0, chunk_sample_start, chunk_sample_count);
                let advantages_mb = advantages.narrow(0, chunk_sample_start, chunk_sample_count);
                let old_log_probs_mb =
                    old_log_probs.narrow(0, chunk_sample_start, chunk_sample_count);

                let u = actions_mb;

                // log N(u; μ_perturbed, σ)
                let action_log_stds_clamped = action_log_stds.clamp(-20.0, 5.0);
                let action_std = action_log_stds_clamped.exp();
                let two_log_std = &action_log_stds_clamped * 2.0;
                // RPO alpha: sigmoid parameterization for smooth gradients everywhere
                let rpo_alpha =
                    RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                let rpo_alpha_detached = rpo_alpha.detach();
                let rpo_noise =
                    Tensor::empty_like(&action_logits).uniform_(-1.0, 1.0) * &rpo_alpha_detached;
                let action_logits_noisy = &action_logits + &rpo_noise;

                let u_normalized = (&u - &action_logits_noisy) / &action_std;
                let u_squared = u_normalized.pow_tensor_scalar(2);
                let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
                let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
                let log_weights = u.log_softmax(-1, Kind::Float);
                let log_det = log_weights.sum_dim_intlist(-1, false, Kind::Float);
                let action_log_probs = log_prob_gaussian - log_det;

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

                // PPO clipped objective (joint, weighted advantage)
                let ratio = (&action_log_probs - &old_log_probs_mb).exp();
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_RATIO, 1.0 + PPO_CLIP_RATIO);
                let action_weights = u.softmax(-1, Kind::Float).detach();
                let adv_mean = advantages_mb.mean_dim([-1].as_slice(), false, Kind::Float);
                let adv_weighted =
                    (&advantages_mb * &action_weights).sum_dim_intlist(-1, false, Kind::Float);
                let advantages_reduced =
                    adv_mean + adv_weighted * ADV_MIXED_WEIGHT;
                let unclipped_obj = &ratio * &advantages_reduced;
                let clipped_obj = ratio_clipped * &advantages_reduced;
                let action_loss =
                    -Tensor::min_other(&unclipped_obj, &clipped_obj).mean(Kind::Float);

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

                // KL and loss for metrics (extract before backward frees graph)
                let approx_kl_val = tch::no_grad(|| {
                    let delta = &action_log_probs - &old_log_probs_mb;
                    (delta.exp() - 1.0 - delta).mean(Kind::Float)
                });
                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - dist_entropy * ENTROPY_COEF;
                let loss_val = tch::no_grad(|| ppo_loss.shallow_clone());
                let policy_loss_val = tch::no_grad(|| action_loss.shallow_clone());
                let value_loss_val = tch::no_grad(|| value_loss.shallow_clone());

                // Alpha loss: target induced KL using detached network outputs
                // For diagonal Gaussian with z_i ~ U(-alpha, alpha):
                // E[KL] = sum_i E[z_i^2] / (2*sigma_i^2) = sum_i (alpha^2/3) / (2*sigma_i^2)
                //       = d * (alpha^2/6) * mean(1/sigma^2)  where d = TICKERS_COUNT
                let action_std_detached = action_log_stds.detach().clamp(-20.0, 5.0).exp();
                let var = action_std_detached.pow_tensor_scalar(2);
                let inv_var_mean = var.clamp_min(1e-4).reciprocal().mean(Kind::Float);
                let d = TICKERS_COUNT as f64;
                let induced_kl: Tensor =
                    rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                let alpha_loss = (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0);

                let scaled_ppo_loss =
                    &ppo_loss * (chunk_sample_count as f64 / samples_per_accum as f64);
                (&scaled_ppo_loss + alpha_loss * ALPHA_LOSS_COEF).backward();

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
            let sde_scale = trading_model.sde_scale().squeeze();
            Tensor::stack(
                &[
                    std.mean(Kind::Float),
                    std.min(),
                    std.max(),
                    rpo_alpha,
                    logit_scale,
                    sde_scale,
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
        let sde_scale = *stats_vec.get(5).unwrap_or(&1.0);
        env.primary_mut()
            .meta_history
            .record_std_stats(mean_std, min_std, max_std, rpo_alpha);
        env.primary_mut().meta_history.record_logit_scale(logit_scale);
        env.primary_mut().meta_history.record_sde_scale(sde_scale);

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
        if DEBUG_TEMPORAL_REPORTS {
            let ( _out, debug) = tch::no_grad(|| {
                let price_deltas_step = s_price_deltas.get(0);
                let static_obs = s_static_obs.get(0);
                trading_model.forward_with_debug(&price_deltas_step, &static_obs, false)
            });
            env.primary_mut().meta_history.record_temporal_debug(
                debug.time_alpha_attn_mean,
                debug.time_alpha_mlp_mean,
                debug.cross_alpha_attn_mean,
                debug.cross_alpha_mlp_mean,
                debug.temporal_tau,
                debug.temporal_attn_entropy,
                debug.temporal_attn_max,
                debug.temporal_attn_eff_len,
                debug.temporal_attn_center,
                debug.temporal_attn_last_weight,
                debug.cross_ticker_embed_norm,
            );
        }
        if DEBUG_MEMORY_REPORTS {
            let rss_kb = read_rss_kb().unwrap_or(0);
            let cuda_mb = read_gpu_mem_mb().unwrap_or(0);
            println!(
                "[Ep {:6}] mem rss={}KB cuda_proc={}MB",
                episode, rss_kb, cuda_mb
            );
        }

        if episode > 0 && episode % 25 == 0 {
            // Debug: Check if exploration has collapsed or network diverged
            let (_, (debug_logits, debug_log_std, _), _attn_weights) =
                tch::no_grad(|| {
                    let price_deltas_step = s_price_deltas.get(0);
                    let static_obs = s_static_obs.get(0);
                    trading_model.forward(&price_deltas_step, &static_obs, false)
                });

            let mean_std = f64::try_from(debug_log_std.exp().mean(Kind::Float)).unwrap();
            let max_raw_action = f64::try_from(debug_logits.abs().max()).unwrap();

            let avg_kl = if total_sample_count > 0 {
                f64::try_from(&total_kl_weighted / total_sample_count as f64).unwrap_or(0.0)
            } else {
                0.0
            };
            // Per-dimension KL for comparison (total KL scales with ticker dims)
            let kl_per_dim = avg_kl / (TICKERS_COUNT as f64);

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
