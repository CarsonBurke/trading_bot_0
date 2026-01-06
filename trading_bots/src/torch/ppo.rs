use shared::paths::WEIGHTS_PATH;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::constants::{
    ACTION_COUNT, OBSERVATION_SPACE, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS,
    STEPS_PER_EPISODE, TICKERS_COUNT,
};
use crate::torch::env::VecEnv;
use crate::torch::model::{symlog_tensor, TradingModel, TradingModelConfig};
use crate::torch::load::load_var_store_partial;
use std::fs;
use std::process;
use std::sync::OnceLock;
use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::UsedGpuMemory;

#[cfg(feature = "perf_timing")]
#[derive(Default)]
struct PerfTimers {
    rollout_us: u64,
    rollout_fwd_us: u64,
    env_step_us: u64,
    buffer_us: u64,
    prep_us: u64,
    gae_us: u64,
    update_fwd_us: u64,
    update_bwd_us: u64,
}

#[cfg(feature = "perf_timing")]
impl PerfTimers {
    fn log(&self) {
        println!(
            "timing: rollout={:.1}ms fwd={:.1}ms env={:.1}ms buf={:.1}ms prep={:.1}ms gae={:.1}ms update_fwd={:.1}ms update_bwd={:.1}ms",
            self.rollout_us as f64 / 1000.0,
            self.rollout_fwd_us as f64 / 1000.0,
            self.env_step_us as f64 / 1000.0,
            self.buffer_us as f64 / 1000.0,
            self.prep_us as f64 / 1000.0,
            self.gae_us as f64 / 1000.0,
            self.update_fwd_us as f64 / 1000.0,
            self.update_bwd_us as f64 / 1000.0
        );
    }
}

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

    fn push(
        &mut self,
        new_deltas: &Tensor,
        reset_deltas: Option<(&[Vec<f32>], &Tensor)>,
    ) {
        self.buffer
            .narrow(2, self.pos, 1)
            .copy_(&new_deltas.unsqueeze(-1));
        self.pos = (self.pos + 1) % PRICE_DELTAS_PER_TICKER as i64;

        if let Some((resets, is_done)) = reset_deltas {
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
const GAMMA: f64 = 0.997;
const GAE_LAMBDA: f64 = 0.95;

fn huber_elementwise(err: &Tensor, delta: f64) -> Tensor {
    let abs_err = err.abs();
    let quad = err.pow_tensor_scalar(2).g_mul_scalar(0.5);
    let lin = (&abs_err - 0.5 * delta).g_mul_scalar(delta);
    let mask = abs_err.le(delta).to_kind(Kind::Bool);
    quad.where_self(&mask, &lin)
}

fn twohot_encode(values: &Tensor, bin_centers: &Tensor) -> Tensor {
    let num_bins = bin_centers.size()[0];
    let values_flat = values.flatten(0, -1);
    let batch_size = values_flat.size()[0];

    let bin_min = bin_centers.min();
    let bin_max = bin_centers.max();
    let bin_range = (&bin_max - &bin_min).clamp_min(1e-8);

    let values_clamped = values_flat.maximum(&bin_min).minimum(&bin_max);
    let frac_idx = (&values_clamped - &bin_min) / &bin_range * (num_bins - 1) as f64;
    let idx_low = frac_idx
        .floor()
        .clamp(0.0, (num_bins - 2) as f64)
        .to_kind(Kind::Int64);
    let idx_high = (&idx_low + 1).clamp(0, num_bins - 1);
    let weight_high = (&frac_idx - frac_idx.floor()).clamp(0.0, 1.0);
    let weight_low: Tensor = 1.0 - &weight_high;

    let mut twohot = Tensor::zeros([batch_size, num_bins], (Kind::Float, values.device()));
    let _ = twohot.scatter_add_(-1, &idx_low.unsqueeze(-1), &weight_low.unsqueeze(-1));
    let _ = twohot.scatter_add_(-1, &idx_high.unsqueeze(-1), &weight_high.unsqueeze(-1));

    twohot
}

// PPO hyperparameters
const PPO_CLIP_RATIO: f64 = 0.2; // Clip for trust region
const ENTROPY_COEF: f64 = 0.0;
const ATTENTION_ENTROPY_COEF: f64 = 0.002;
const VALUE_LOSS_COEF: f64 = 0.5; // Value loss coefficient
const MAX_GRAD_NORM: f64 = 0.5; // Gradient clipping norm
const VALUE_LOG_CLIP: f64 = 10.0;
const CRITIC_MAE_NORM: f64 = VALUE_LOG_CLIP;
const CRITIC_ENTROPY_COEF: f64 = 1e-3;
                                // Conservative KL early stopping (SB3-style)
const TARGET_KL: f64 = 0.03;
const KL_STOP_MULTIPLIER: f64 = 1.5;
const LEARNING_RATE: f64 = 2e-4;

// RPO: adaptive alpha targeting induced KL (total KL, not per-dim)
const RPO_ALPHA_MIN: f64 = 0.005;
const RPO_ALPHA_MAX: f64 = 0.5;
const RPO_TARGET_KL: f64 = 0.018;
const RPO_ALPHA_INIT: f64 = 0.1;
const ALPHA_LOSS_COEF: f64 = 0.1;
const ADV_MIXED_WEIGHT: f64 = 0.5;
const GRAD_ACCUM_STEPS: usize = 2; // Accumulate gradients over k chunks before stepping (was 4, reduced for more updates)
const DEBUG_MEMORY_REPORTS: bool = false;
const DEBUG_NUMERICS: bool = true;

fn debug_tensor_stats(name: &str, t: &Tensor, episode: i64, step: usize) -> bool {
    let has_nan = t.isnan().any().int64_value(&[]) != 0;
    let has_inf = t.isinf().any().int64_value(&[]) != 0;
    if has_nan || has_inf {
        let mean = f64::try_from(t.mean(Kind::Float)).unwrap_or(f64::NAN);
        let min = f64::try_from(t.min()).unwrap_or(f64::NAN);
        let max = f64::try_from(t.max()).unwrap_or(f64::NAN);
        println!(
            "Non-finite in {} at ep {} step {} nan={} inf={} mean={:.6} min={:.6} max={:.6}",
            name, episode, step, has_nan, has_inf, mean, min, max
        );
        return false;
    }
    true
}

fn log_tensor_summary(name: &str, t: &Tensor) {
    let mean = f64::try_from(t.mean(Kind::Float)).unwrap_or(f64::NAN);
    let min = f64::try_from(t.min()).unwrap_or(f64::NAN);
    let max = f64::try_from(t.max()).unwrap_or(f64::NAN);
    let abs_max = f64::try_from(t.abs().max()).unwrap_or(f64::NAN);
    println!(
        "  {}: mean={:.6} min={:.6} max={:.6} abs_max={:.6}",
        name, mean, min, max, abs_max
    );
}

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

    // Create device-specific kind
    let model_kind = (train_kind, device);
    let stats_kind = (Kind::Float, device);

    let mut sum_rewards = Tensor::zeros([NPROCS], stats_kind);
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
    let mut step_reward = Tensor::zeros([NPROCS], stats_kind);
    let mut step_reward_per_ticker =
        Tensor::zeros([NPROCS, TICKERS_COUNT], stats_kind);
    let mut step_cash_reward = Tensor::zeros([NPROCS], stats_kind);
    let mut step_is_done = Tensor::zeros([NPROCS], stats_kind);
    let mut step_deltas = Tensor::zeros([NPROCS, TICKERS_COUNT], (train_kind, device));

    for episode in 0..UPDATES {
        #[cfg(feature = "perf_timing")]
        let mut perf = PerfTimers::default();
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
        #[cfg(feature = "perf_timing")]
        let rollout_start = Instant::now();
        for step in 0..env.max_step() {
            env.set_step(step);

            #[cfg(feature = "perf_timing")]
            let fwd_start = Instant::now();
            let (values, _critic_logits, (action_mean, action_log_std), _attn_entropy) =
                tch::no_grad(|| {
                    let price_deltas_step = s_price_deltas.get(s);
                    let static_obs = s_static_obs.get(s);
                    trading_model.forward(&price_deltas_step, &static_obs, false)
                });
            #[cfg(feature = "perf_timing")]
            {
                perf.rollout_fwd_us += fwd_start.elapsed().as_micros() as u64;
            }
            let values = values.to_kind(Kind::Float);
            let action_mean = action_mean.to_kind(Kind::Float);
            let action_log_std = action_log_std.to_kind(Kind::Float);

            let action_std = action_log_std.exp();
            let noise_ticker = Tensor::randn([NPROCS, TICKERS_COUNT], stats_kind);
            let noise_cash = Tensor::zeros([NPROCS, 1], stats_kind);
            let noise = Tensor::cat(&[noise_ticker, noise_cash], 1);
            let u = &action_mean + &action_std * noise;
            let actions = u.softmax(-1, Kind::Float);

            let u_normalized = (&u - &action_mean) / &action_std;
            let u_squared = u_normalized.pow_tensor_scalar(2);
            let two_log_std = &action_log_std * 2.0;
            let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
            let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
            let log_det = u
                .log_softmax(-1, Kind::Float)
                .sum_dim_intlist(-1, false, Kind::Float);
            let action_log_prob = &log_prob_gaussian - &log_det;

            if DEBUG_NUMERICS {
                let ok = debug_tensor_stats("action_mean", &action_mean, episode, step)
                    && debug_tensor_stats("action_log_std", &action_log_std, episode, step)
                    && debug_tensor_stats("u", &u, episode, step)
                    && debug_tensor_stats("log_det", &log_det, episode, step)
                    && debug_tensor_stats("action_log_prob", &action_log_prob, episode, step);
                if !ok {
                    return;
                }
            }

            let mut out_so = s_static_obs.get(s + 1);
            #[cfg(feature = "perf_timing")]
            let env_start = Instant::now();
            env.step_incremental_tensor_into(
                &actions,
                &mut out_so,
                &mut step_reward,
                &mut step_reward_per_ticker,
                &mut step_cash_reward,
                &mut step_is_done,
                &mut step_deltas,
            );
            #[cfg(feature = "perf_timing")]
            {
                perf.env_step_us += env_start.elapsed().as_micros() as u64;
            }

            #[cfg(feature = "perf_timing")]
            let buf_start = Instant::now();
            rolling_buffer.push(&step_deltas, None);
            s_price_deltas.get(s + 1).copy_(&rolling_buffer.get_flat());
            #[cfg(feature = "perf_timing")]
            {
                perf.buffer_us += buf_start.elapsed().as_micros() as u64;
            }

            sum_rewards += &step_reward;
            let completed_rewards = (&sum_rewards * &step_is_done).sum(Kind::Float);
            let completed_episodes = step_is_done.sum(Kind::Float);
            let _ = total_rewards_gpu.g_add_(&completed_rewards);
            let _ = total_episodes_gpu.g_add_(&completed_episodes);

            let masks = Tensor::from(1f32).to_device(device) - &step_is_done;
            sum_rewards *= &masks;

            s_actions.get(s).copy_(&u); // Store pre-softmax u for training
            s_values.get(s).copy_(&values);
            s_log_probs.get(s).copy_(&action_log_prob);
            let rewards_full = Tensor::zeros([NPROCS, TICKERS_COUNT + 1], stats_kind);
            rewards_full
                .narrow(1, 0, TICKERS_COUNT)
                .copy_(&step_reward_per_ticker);
            rewards_full
                .narrow(1, TICKERS_COUNT, 1)
                .copy_(&step_cash_reward.unsqueeze(1));
            s_rewards.get(s).copy_(&rewards_full);
            s_masks.get(s).copy_(&masks);

            s += 1; // Increment storage index
        }
        #[cfg(feature = "perf_timing")]
        {
            perf.rollout_us += rollout_start.elapsed().as_micros() as u64;
        }

        let static_obs_vec =
            Vec::<f32>::try_from(s_static_obs.get(rollout_steps - 1).flatten(0, -1))
                .unwrap_or_default();
        env.primary_mut()
            .episode_history
            .static_observations
            .push(static_obs_vec);

        #[cfg(feature = "perf_timing")]
        let prep_start = Instant::now();
        let price_deltas_batch = s_price_deltas
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64])
            .detach();
        let static_obs_batch = s_static_obs
            .narrow(0, 0, rollout_steps)
            .reshape([memory_size, STATIC_OBSERVATIONS as i64])
            .detach();
        #[cfg(feature = "perf_timing")]
        {
            perf.prep_us += prep_start.elapsed().as_micros() as u64;
        }

        #[cfg(feature = "perf_timing")]
        let gae_start = Instant::now();
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
                let (values, _, _, _) =
                    trading_model.forward(&price_deltas_step, &static_obs, false);
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
        #[cfg(feature = "perf_timing")]
        {
            perf.gae_us += gae_start.elapsed().as_micros() as u64;
        }
        let actions = s_actions.view([memory_size, ACTION_COUNT]).detach();
        let old_log_probs = s_log_probs.view([memory_size]).detach();
        let s_values_flat = s_values.view([memory_size, TICKERS_COUNT + 1]).detach();
        let action_weights = actions.softmax(-1, Kind::Float).detach();

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
        let mut total_value_mae_weighted = Tensor::zeros([], stats_kind);
        let mut total_sample_count = 0i64;
        let mut grad_norm_sum = Tensor::zeros([], stats_kind);
        let mut grad_norm_count = 0i64;
        // Clip fraction diagnostic
        let mut total_clipped = Tensor::zeros([], stats_kind);
        let mut total_ratio_samples = 0i64;

        let num_chunks = (rollout_steps + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunk_order: Vec<usize> = (0..num_chunks as usize).collect();
        let mut fwd_time_us = 0u64;
        let mut bwd_time_us = 0u64;

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
                let fwd_start = Instant::now();
                let chunk_start_step = (chunk_idx as i64) * CHUNK_SIZE;
                let chunk_end_step = ((chunk_idx as i64 + 1) * CHUNK_SIZE).min(rollout_steps);
                let chunk_len = chunk_end_step - chunk_start_step;
                let chunk_sample_count = chunk_len * NPROCS;
                let chunk_sample_start = chunk_start_step * NPROCS;
                let price_deltas_chunk =
                    price_deltas_batch.narrow(0, chunk_sample_start, chunk_sample_count);
                let static_obs_chunk =
                    static_obs_batch.narrow(0, chunk_sample_start, chunk_sample_count);

                let (values, critic_logits, (action_mean, action_log_stds), attn_entropy) =
                    trading_model.forward(&price_deltas_chunk, &static_obs_chunk, true);
                let values = values
                    .to_kind(Kind::Float)
                    .view([chunk_sample_count, TICKERS_COUNT + 1]);
                let action_mean = action_mean.to_kind(Kind::Float);
                let action_log_stds = action_log_stds.to_kind(Kind::Float);
                let attn_entropy = attn_entropy.to_kind(Kind::Float);

                let actions_mb = actions.narrow(0, chunk_sample_start, chunk_sample_count);
                let returns_mb = returns.narrow(0, chunk_sample_start, chunk_sample_count);
                let advantages_mb = advantages.narrow(0, chunk_sample_start, chunk_sample_count);
                let old_log_probs_mb =
                    old_log_probs.narrow(0, chunk_sample_start, chunk_sample_count);

                let u = actions_mb;

                // log N(u; μ_perturbed, σ)
                let action_std = action_log_stds.exp();
                let two_log_std = &action_log_stds * 2.0;
                // RPO alpha: sigmoid parameterization for smooth gradients everywhere
                let rpo_alpha =
                    RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                let rpo_alpha_detached = rpo_alpha.detach();
                let rpo_noise_ticker =
                    Tensor::empty([chunk_sample_count, TICKERS_COUNT], stats_kind)
                        .uniform_(-1.0, 1.0)
                        * &rpo_alpha_detached;
                let rpo_noise_cash = Tensor::zeros([chunk_sample_count, 1], stats_kind);
                let rpo_noise = Tensor::cat(&[rpo_noise_ticker, rpo_noise_cash], 1);
                let action_mean_noisy = &action_mean + &rpo_noise;

                let u_normalized = (&u - &action_mean_noisy) / &action_std;
                let u_squared = u_normalized.pow_tensor_scalar(2);
                let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
                let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
                let log_det = u
                    .log_softmax(-1, Kind::Float)
                    .sum_dim_intlist(-1, false, Kind::Float);
                let action_log_probs = (log_prob_gaussian - log_det).nan_to_num(0.0, 0.0, 0.0);

                // Entropy of Gaussian (per ticker)
                let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * &action_log_stds;
                let dist_entropy = entropy_components.g_mul_scalar(0.5).mean(Kind::Float);
                let attn_entropy_mean = attn_entropy.mean(Kind::Float);

                let returns_clipped = returns_mb.clamp(-VALUE_LOG_CLIP, VALUE_LOG_CLIP);
                let returns_symlog = symlog_tensor(&returns_clipped);
                let target_twohot = twohot_encode(&returns_symlog, trading_model.value_centers())
                    .view([chunk_sample_count, TICKERS_COUNT + 1, -1]);
                let log_probs = critic_logits
                    .to_kind(Kind::Float)
                    .log_softmax(-1, Kind::Float);
                let log_probs_ce = log_probs.shallow_clone();
                let ce_loss = -(target_twohot * log_probs_ce)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);
                let probs = log_probs.exp();
                let critic_entropy = -(probs * &log_probs)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);

                let values_pred = values;
                let returns_t = returns_mb;
                let value_loss: Tensor = ce_loss - CRITIC_ENTROPY_COEF * critic_entropy;
                let value_mae = (&values_pred - &returns_t)
                    .abs()
                    .mean(Kind::Float)
                    / CRITIC_MAE_NORM;

                // PPO clipped objective (joint, weighted advantage)
                let old_log_probs_mb = old_log_probs_mb.nan_to_num(0.0, 0.0, 0.0);
                let log_ratio_raw = &action_log_probs - &old_log_probs_mb;
                let log_ratio = log_ratio_raw.tanh() * 0.3;
                if DEBUG_NUMERICS {
                    let lr_nan = log_ratio.isnan().any().int64_value(&[]) != 0;
                    let lr_inf = log_ratio.isinf().any().int64_value(&[]) != 0;
                    if lr_nan || lr_inf {
                        println!(
                            "Non-finite log_ratio at ep {} chunk {} nan={} inf={}",
                            episode, chunk_idx, lr_nan, lr_inf
                        );
                        log_tensor_summary("log_ratio_raw", &log_ratio_raw);
                        log_tensor_summary("log_ratio", &log_ratio);
                        log_tensor_summary("action_log_probs", &action_log_probs);
                        log_tensor_summary("old_log_probs", &old_log_probs_mb);
                        log_tensor_summary("action_mean", &action_mean);
                        log_tensor_summary("action_mean_noisy", &action_mean_noisy);
                        log_tensor_summary("action_log_stds", &action_log_stds);
                        let rpo_alpha_val = f64::try_from(&rpo_alpha).unwrap_or(f64::NAN);
                        println!("  rpo_alpha: {:.6}", rpo_alpha_val);
                        return;
                    }
                }
                let ratio = log_ratio.exp();
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_RATIO, 1.0 + PPO_CLIP_RATIO);
                let action_weights = action_weights.narrow(0, chunk_sample_start, chunk_sample_count);
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
                let approx_kl_val =
                    tch::no_grad(|| (log_ratio.exp() - 1.0 - &log_ratio).mean(Kind::Float));
                if DEBUG_NUMERICS {
                    let kl_nan = approx_kl_val.isnan().int64_value(&[]) != 0;
                    let kl_inf = approx_kl_val.isinf().int64_value(&[]) != 0;
                    if kl_nan || kl_inf {
                        let kl_val = f64::try_from(&approx_kl_val).unwrap_or(f64::NAN);
                        println!(
                            "Non-finite approx_kl at ep {} chunk {} val={:.6} nan={} inf={}",
                            episode, chunk_idx, kl_val, kl_nan, kl_inf
                        );
                        log_tensor_summary("log_ratio_raw", &log_ratio_raw);
                        log_tensor_summary("log_ratio", &log_ratio);
                        log_tensor_summary("action_log_probs", &action_log_probs);
                        log_tensor_summary("old_log_probs", &old_log_probs_mb);
                        log_tensor_summary("action_mean", &action_mean);
                        log_tensor_summary("action_mean_noisy", &action_mean_noisy);
                        log_tensor_summary("action_log_stds", &action_log_stds);
                        let rpo_alpha_val = f64::try_from(&rpo_alpha).unwrap_or(f64::NAN);
                        println!("  rpo_alpha: {:.6}", rpo_alpha_val);
                        return;
                    }
                }
                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - dist_entropy * ENTROPY_COEF
                    - attn_entropy_mean * ATTENTION_ENTROPY_COEF;
                let loss_val = tch::no_grad(|| ppo_loss.shallow_clone());
                let policy_loss_val = tch::no_grad(|| action_loss.shallow_clone());
                let value_loss_val = tch::no_grad(|| value_loss.shallow_clone());

                // Alpha loss: target induced KL using detached network outputs
                // For diagonal Gaussian with z_i ~ U(-alpha, alpha):
                // E[KL] = sum_i E[z_i^2] / (2*sigma_i^2) = sum_i (alpha^2/3) / (2*sigma_i^2)
                //       = d * (alpha^2/6) * mean(1/sigma^2)  where d = TICKERS_COUNT
                let action_std_detached = action_log_stds.detach().exp();
                let var = action_std_detached.pow_tensor_scalar(2);
                let inv_var_mean = var.clamp_min(1e-4).reciprocal().mean(Kind::Float);
                let d = ACTION_COUNT as f64;
                let induced_kl: Tensor =
                    rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                let alpha_loss = (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0);

                let scaled_ppo_loss =
                    &ppo_loss * (chunk_sample_count as f64 / samples_per_accum as f64);
                fwd_time_us += fwd_start.elapsed().as_micros() as u64;
                let bwd_start = Instant::now();
                (&scaled_ppo_loss + alpha_loss * ALPHA_LOSS_COEF).backward();
                bwd_time_us += bwd_start.elapsed().as_micros() as u64;

                // Accumulate metrics on GPU
                let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                let _ = total_loss_weighted.g_add_(&(&loss_val * chunk_sample_count as f64));
                let _ = total_policy_loss_weighted.g_add_(&(&policy_loss_val * chunk_sample_count as f64));
                let _ = total_value_loss_weighted.g_add_(&(&value_loss_val * chunk_sample_count as f64));
                let _ = total_value_mae_weighted.g_add_(&(&value_mae * chunk_sample_count as f64));
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

        println!(
            "fwd: {:.1}ms  bwd: {:.1}ms",
            fwd_time_us as f64 / 1000.0,
            bwd_time_us as f64 / 1000.0
        );
        #[cfg(feature = "perf_timing")]
        {
            perf.update_fwd_us = fwd_time_us;
            perf.update_bwd_us = bwd_time_us;
            perf.log();
        }

        // Record std stats every episode
        let stats_tensor = tch::no_grad(|| {
            let price_deltas_step = s_price_deltas.get(0);
            let static_obs = s_static_obs.get(0);
            let (_, _, (_, action_log_std), _) =
                trading_model.forward(&price_deltas_step, &static_obs, false);
            let action_std = action_log_std.exp();
            let std_mean = action_std.mean(Kind::Float);
            let std_min = action_std.min();
            let std_max = action_std.max();
            let rpo_alpha =
                (RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid()).squeeze();
            Tensor::stack(
                &[std_mean, std_min, std_max, rpo_alpha],
                0,
            )
        });
        let stats_vec = Vec::<f64>::try_from(stats_tensor).unwrap_or_default();
        let logit_noise_mean = *stats_vec.get(0).unwrap_or(&0.0);
        let logit_noise_min = *stats_vec.get(1).unwrap_or(&0.0);
        let logit_noise_max = *stats_vec.get(2).unwrap_or(&0.0);
        let rpo_alpha = *stats_vec.get(3).unwrap_or(&0.0);
        env.primary_mut()
            .meta_history
            .record_logit_noise_stats(logit_noise_mean, logit_noise_min, logit_noise_max, rpo_alpha);
        let clip_frac = if total_ratio_samples > 0 {
            let total_clipped_val = f64::try_from(&total_clipped).unwrap_or(0.0);
            total_clipped_val / total_ratio_samples as f64
        } else {
            0.0
        };
        env.primary_mut().meta_history.record_clip_fraction(clip_frac);

        // Single GPU->CPU sync for loss and grad norm at end of all epochs
        let mean_losses = if total_sample_count > 0 {
            Tensor::stack(
                &[
                    &total_loss_weighted,
                    &total_policy_loss_weighted,
                    &total_value_loss_weighted,
                    &total_value_mae_weighted,
                ],
                0,
            ) / (total_sample_count as f64)
        } else {
            Tensor::zeros([4], (Kind::Float, device))
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
        let mean_value_mae = *mean_all_vec.get(3).unwrap_or(&0.0);
        let mean_grad_norm = *mean_all_vec.get(4).unwrap_or(&0.0);
        env.primary_mut().meta_history.record_loss(mean_loss);
        env.primary_mut()
            .meta_history
            .record_policy_loss(mean_policy_loss);
        env.primary_mut()
            .meta_history
            .record_value_loss(mean_value_loss);
        env.primary_mut()
            .meta_history
            .record_value_mae(mean_value_mae);
        env.primary_mut()
            .meta_history
            .record_grad_norm(mean_grad_norm);
        if DEBUG_MEMORY_REPORTS {
            let rss_kb = read_rss_kb().unwrap_or(0);
            let cuda_mb = read_gpu_mem_mb().unwrap_or(0);
            println!(
                "[Ep {:6}] mem rss={}KB cuda_proc={}MB",
                episode, rss_kb, cuda_mb
            );
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
