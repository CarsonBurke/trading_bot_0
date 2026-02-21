use std::env;
use std::time::Instant;
use tch::{nn, Kind, Tensor};

use super::Fp32Adam;

use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::env::VecEnv;
use crate::torch::model::{
    twohot_ce_loss, ModelVariant, TradingModel, TradingModelConfig, ACTION_DIM,
};

const LEARNING_RATE: f64 = 3e-4;
pub const NPROCS: i64 = 16;
const SEQ_LEN: i64 = 4000;
const CHUNK_SIZE: i64 = 128;
const OPTIM_EPOCHS: i64 = 4;
const PPO_CLIP_LOW: f64 = 0.2;
const PPO_CLIP_HIGH: f64 = 0.28; // DAPO-style asymmetric: wider upper bound prevents entropy collapse
const TARGET_KL: f64 = 0.03;
const KL_STOP_MULTIPLIER: f64 = 1.5;
const VALUE_LOSS_COEF: f64 = 0.5;
const ENTROPY_COEF: f64 = 0.0;
const MAX_GRAD_NORM: f64 = 0.5;
const GRAD_ACCUM_STEPS: usize = 1;
pub(crate) const DEBUG_NUMERICS: bool = false;
const LOG_2PI: f64 = 1.8378770664093453;

// RPO: Random Policy Optimization - adds bounded noise to action mean during training and intentionally not during rollout
// Alpha is learned via induced KL targeting. Set all to 0.0 to disable.
const RPO_ALPHA_MIN: f64 = 0.05;
const RPO_ALPHA_MAX: f64 = 0.5;
const RPO_ALPHA_INIT: f64 = 0.1; // CleanRL impl found 0.1 reliably improved results in all test envs over PPO
const RPO_TARGET_KL: f64 = 0.018;
const ALPHA_LOSS_COEF: f64 = 0.1;
const MAX_DELTA_ALPHA: f64 = 0.2;

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

fn log_tensor_summary(name: &str, t: &Tensor) {
    let mean = t.mean(Kind::Float).double_value(&[]);
    let min = t.min().double_value(&[]);
    let max = t.max().double_value(&[]);
    let abs_max = t.abs().max().double_value(&[]);
    println!(
        "  {}: mean={:.6} min={:.6} max={:.6} abs_max={:.6}",
        name, mean, min, max, abs_max
    );
}

struct GpuRollingBuffer {
    capacity: i64,
    device: tch::Device,
    data: Tensor,
    ptr: i64,
}

impl GpuRollingBuffer {
    fn new(capacity: i64, dim: i64, kind: Kind, device: tch::Device) -> Self {
        let data = Tensor::zeros(&[capacity, dim], (kind, device));
        Self {
            capacity,
            device,
            data,
            ptr: 0,
        }
    }

    fn push(&mut self, t: &Tensor) {
        let n = t.size()[0];
        if self.ptr + n > self.capacity {
            self.ptr = 0;
        }
        let _ = self.data.narrow(0, self.ptr, n).copy_(t);
        self.ptr += n;
    }

    fn get(&self, step: i64) -> Tensor {
        self.data.narrow(0, step * NPROCS, NPROCS)
    }
}

/// 1D Gaussian log-prob: log N(diff; 0, diag(std^2))
fn gaussian_log_prob(diff: &Tensor, std: &Tensor) -> Tensor {
    let var = std.pow_tensor_scalar(2);
    let log_std = std.log();
    let mahal = diff.pow_tensor_scalar(2) / &var;
    let per_dim: Tensor = -(mahal + LOG_2PI) * 0.5 - &log_std;
    per_dim.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

/// 1D diagonal Gaussian entropy: H = sum(log std + 0.5(1 + log 2pi))
fn gaussian_entropy(std: &Tensor) -> Tensor {
    (std.log() + 0.5 * (1.0 + LOG_2PI))
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

/// Categorical entropy of softmax policy: H = -sum(p * log p)
fn softmax_entropy(logits: &Tensor) -> Tensor {
    let log_probs = logits.log_softmax(-1, Kind::Float);
    let probs = logits.softmax(-1, Kind::Float);
    -(probs * log_probs).sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

fn sample_rollout_actions(
    model: &TradingModel,
    obs_price: &Tensor,
    obs_static: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    tch::no_grad(|| {
        let (values, _, _, action_mean, action_noise_std) =
            model.forward_on_device(obs_price, obs_static, false);
        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_noise_std = action_noise_std.to_kind(Kind::Float);

        let batch = action_mean.size()[0];
        let z = Tensor::randn(
            &[batch, TICKERS_COUNT],
            (Kind::Float, action_mean.device()),
        );
        let ticker_noise = &action_noise_std * &z; // [B, TICKERS_COUNT]
        let action_log_prob = gaussian_log_prob(&ticker_noise, &action_noise_std);
        let zero_cash = Tensor::zeros(&[batch, 1], (Kind::Float, action_mean.device()));
        let noise = Tensor::cat(&[ticker_noise, zero_cash], -1); // [B, ACTION_DIM]
        let u = &action_mean + &noise;
        let actions = u.softmax(-1, Kind::Float);

        (values, action_mean, u, actions, action_log_prob)
    })
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
fn compute_explained_variance(
    model: &TradingModel,
    price_deltas: &Tensor,
    static_obs: &Tensor,
    returns: &Tensor,
    rollout_steps: i64,
) -> Tensor {
    let ev_steps = CHUNK_SIZE.min(rollout_steps);
    let ev_samples = ev_steps * NPROCS;
    tch::no_grad(|| {
        let pd_ev = price_deltas.narrow(0, 0, ev_samples);
        let so_ev = static_obs.narrow(0, 0, ev_samples);
        let ret_ev = returns.narrow(0, 0, ev_samples);
        let (values_raw, _, _, _, _) = model.forward_on_device(&pd_ev, &so_ev, true);
        let values = values_raw.to_kind(Kind::Float).view([ev_samples]);
        let targets = ret_ev;
        let residuals = &values - &targets;
        let mean_target = targets.mean(Kind::Float);
        let var_targets = targets.square().mean(Kind::Float) - mean_target.square();
        let var_residuals = residuals.square().mean(Kind::Float);
        Tensor::from(1.0) - &var_residuals / var_targets.clamp_min(1e-8)
    })
}

/// Compute action std and RPO alpha stats from a sample batch.
fn compute_action_std_stats(
    model: &TradingModel,
    s_price_deltas: &GpuRollingBuffer,
    s_static_obs: &GpuRollingBuffer,
    rpo_rho: &Tensor,
    device: tch::Device,
) -> Tensor {
    tch::no_grad(|| {
        let (_, _, _, _, action_noise_std) =
            model.forward_on_device(&s_price_deltas.get(0), &s_static_obs.get(0), false);
        let action_std = action_noise_std.to_kind(Kind::Float);
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

pub async fn train(weights_path: Option<&str>, model_variant: ModelVariant) {
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
    let mut vs = nn::VarStore::new(device);
    let trading_model = TradingModel::new_with_config(
        &vs.root(),
        TradingModelConfig {
            variant: model_variant,
            ..TradingModelConfig::default()
        },
    );

    // RPO alpha via sigmoid: alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(rho)
    // Kept outside VarStore so Fp32Adam doesn't pick it up (custom clamped update rule)
    let mut rpo_rho = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
        let p_init = (RPO_ALPHA_INIT - RPO_ALPHA_MIN) / (RPO_ALPHA_MAX - RPO_ALPHA_MIN);
        let p_init = p_init.clamp(1e-6, 1.0 - 1e-6);
        let rho_init = (p_init / (1.0 - p_init)).ln();
        vs.root()
            .var("rpo_alpha_rho", &[1], nn::Init::Const(rho_init))
    } else {
        Tensor::zeros(&[1], (Kind::Float, device))
    };

    let start_episode = if let Some(path) = weights_path {
        println!("Loading weights from {}", path);
        vs.load(path).unwrap();
        // Extract episode number from filename like "ppo_ep500.ot"
        let ep = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("ppo_ep"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        if ep > 0 {
            println!("Resuming from episode {}", ep);
        }
        ep
    } else {
        println!("Starting training from scratch");
        0
    };

    // Cast all model parameters to bf16 for faster compute.
    // Done after weight loading so loaded fp32 weights get cast.
    // Adam optimizer handles mixed bf16 parameters correctly.
    vs.bfloat16();

    let mut opt = Fp32Adam::new(&vs, LEARNING_RATE);

    let mut env = VecEnv::new(true, model_variant);
    if start_episode > 0 {
        env.set_episode(start_episode);
        env.primary_mut()
            .meta_history
            .load_from_episode(start_episode);
    }

    let rollout_steps = SEQ_LEN;
    let memory_size = rollout_steps * NPROCS;

    let pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
    let so_dim = STATIC_OBSERVATIONS as i64;

    let mut s_price_deltas = GpuRollingBuffer::new(memory_size, pd_dim, Kind::Float, device);
    let mut s_static_obs = GpuRollingBuffer::new(memory_size, so_dim, Kind::Float, device);
    let s_actions = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
    let s_old_log_probs = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let s_rewards = Tensor::zeros(&[memory_size], (Kind::Float, device)); // portfolio-level reward
    let s_dones = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let s_values = Tensor::zeros(&[memory_size], (Kind::Float, device)); // portfolio-level value
    let s_action_weights = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));

    for episode in start_episode..1000000 {
        let (obs_price_cpu, obs_static_cpu) = env.reset();
        let mut obs_price = Tensor::zeros(&[NPROCS, pd_dim], (Kind::Float, device));
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
        let mut step_reward_per_ticker =
            Tensor::zeros(&[NPROCS, TICKERS_COUNT], (Kind::Float, device));
        let mut step_is_done = Tensor::zeros(&[NPROCS], (Kind::Float, device));
        for step in 0..rollout_steps as usize {
            let (values, action_mean, u, actions, action_log_prob) =
                sample_rollout_actions(&trading_model, &obs_price, &obs_static);

            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("action_mean", &action_mean, episode as i64, step);
                let _ = debug_tensor_stats("u", &u, episode as i64, step);
                let _ =
                    debug_tensor_stats("action_log_prob", &action_log_prob, episode as i64, step);
            }
            s_price_deltas.push(&obs_price);
            s_static_obs.push(&obs_static);

            let (reset_indices, reset_price_deltas) = env.step_into_ring_tensor(
                &actions,
                &mut step_deltas,
                &mut obs_static,
                &mut step_reward_per_ticker,
                &mut step_is_done,
            );

            ring_pos = (ring_pos + 1) % ring_len;
            let _ = ring_buf
                .narrow(2, ring_pos, 1)
                .copy_(&step_deltas.unsqueeze(-1));

            if !reset_indices.is_empty() {
                let idx = (&base_idx + (ring_pos + 1)).remainder(ring_len);
                let pd_dim_usize = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
                for (reset_i, env_idx) in reset_indices.iter().enumerate() {
                    let start = reset_i * pd_dim_usize;
                    let end = start + pd_dim_usize;
                    let ordered = Tensor::from_slice(&reset_price_deltas[start..end])
                        .view([TICKERS_COUNT, ring_len])
                        .to_device(device);
                    let mut ring_env = ring_buf.narrow(0, *env_idx as i64, 1).squeeze_dim(0);
                    let _ = ring_env.index_copy_(1, &idx, &ordered);
                }
            }

            let idx = (&base_idx + (ring_pos + 1)).remainder(ring_len);
            let ordered = ring_buf.index_select(2, &idx);
            obs_price.copy_(&ordered.view([NPROCS, pd_dim]));

            let mem_idx = step as i64 * NPROCS;
            let _ = s_actions.narrow(0, mem_idx, NPROCS).copy_(&u); // Store pre-softmax u for training
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
            let _ = s_action_weights.narrow(0, mem_idx, NPROCS).copy_(&actions);
        }

        // Bootstrap value from final observation state (raw return space)
        let bootstrap_value = tch::no_grad(|| {
            let (values_raw, _, _, _, _) =
                trading_model.forward_on_device(&obs_price, &obs_static, false);
            values_raw.to_kind(Kind::Float)
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

        let price_deltas_batch = s_price_deltas.data.shallow_clone();
        let static_obs_batch = s_static_obs.data.shallow_clone();
        let action_weights_batch = s_action_weights.shallow_clone();

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
        let mut entropy_min = Tensor::from(f64::INFINITY).to_device(device);
        let mut entropy_max = Tensor::from(f64::NEG_INFINITY).to_device(device);

        let mut fwd_time_us = 0u64;
        let mut bwd_time_us = 0u64;

        let num_chunks = (rollout_steps + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunk_order: Vec<usize> = (0..num_chunks as usize).collect();

        let mut first_epoch_kl = 0.0f64;

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            use rand::seq::SliceRandom;
            chunk_order.shuffle(&mut rand::rng());
            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;
            let total_epoch_samples = rollout_steps * NPROCS;
            let samples_per_accum =
                (CHUNK_SIZE * NPROCS * GRAD_ACCUM_STEPS as i64).min(total_epoch_samples);

            for (chunk_i, &chunk_idx) in chunk_order.iter().enumerate() {
                let chunk_start_step = chunk_idx as i64 * CHUNK_SIZE;
                let chunk_end_step = ((chunk_idx as i64 + 1) * CHUNK_SIZE).min(rollout_steps);
                let chunk_len = chunk_end_step - chunk_start_step;
                let chunk_sample_count = chunk_len * NPROCS;
                let chunk_sample_start = chunk_start_step * NPROCS;

                let pd_chunk = price_deltas_batch.narrow(0, chunk_sample_start, chunk_sample_count);
                let so_chunk = static_obs_batch.narrow(0, chunk_sample_start, chunk_sample_count);
                let act_mb = s_actions.narrow(0, chunk_sample_start, chunk_sample_count);
                let ret_mb = returns.narrow(0, chunk_sample_start, chunk_sample_count);
                let adv_mb_raw = advantages.narrow(0, chunk_sample_start, chunk_sample_count);
                let adv_mb =
                    (&adv_mb_raw - adv_mb_raw.mean(Kind::Float)) / (adv_mb_raw.std(true) + 1e-8);
                let old_log_probs_mb =
                    s_old_log_probs.narrow(0, chunk_sample_start, chunk_sample_count);
                let _weight_mb =
                    action_weights_batch.narrow(0, chunk_sample_start, chunk_sample_count);

                let fwd_start = Instant::now();
                let (_, critic_logits, _critic_input, action_mean, action_noise_std) =
                    trading_model.forward_no_values_on_device(&pd_chunk, &so_chunk, true);

                let action_mean = action_mean.to_kind(Kind::Float);
                let action_noise_std = action_noise_std.to_kind(Kind::Float);

                // act_mb contains the stored u (pre-softmax logits)
                let u = act_mb;

                // RPO (CleanRL-style): iid uniform perturbation on each action-mean dimension.
                let (rpo_alpha, action_mean_perturbed) = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let alpha = RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                    let alpha_detached = alpha.detach();
                    let rpo_noise =
                        Tensor::empty([chunk_sample_count, ACTION_DIM], (Kind::Float, device))
                            .uniform_(-1.0, 1.0)
                            * &alpha_detached;
                    (alpha, &action_mean + rpo_noise)
                } else {
                    (
                        Tensor::zeros(&[1], (Kind::Float, device)),
                        action_mean.shallow_clone(),
                    )
                };

                // Only ticker dims for log_prob (cash is deterministic, excluded)
                let diff = (&u - &action_mean_perturbed).narrow(-1, 0, TICKERS_COUNT);
                let action_log_probs = gaussian_log_prob(&diff, &action_noise_std);

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("u", &u, _epoch, chunk_i);
                    let _ =
                        debug_tensor_stats("old_log_probs_mb", &old_log_probs_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_mean", &action_mean, _epoch, chunk_i);
                    let _ =
                        debug_tensor_stats("action_noise_std", &action_noise_std, _epoch, chunk_i);
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

                // Distributional two-hot CE value loss.
                let critic_logits = critic_logits.to_kind(Kind::Float);
                let log_probs = critic_logits.log_softmax(-1, Kind::Float);
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("ret_mb", &ret_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("critic_logits", &critic_logits, _epoch, chunk_i);
                    let _ = debug_tensor_stats("adv_mb", &adv_mb, _epoch, chunk_i);
                }
                let value_loss =
                    twohot_ce_loss(&ret_mb, &log_probs, trading_model.bucket_centers())
                        .mean(Kind::Float);

                let gauss_ent = gaussian_entropy(&action_noise_std).mean(Kind::Float);
                let cat_ent = softmax_entropy(&action_mean).mean(Kind::Float);
                let dist_entropy = &gauss_ent + &cat_ent;
                let dist_entropy_detached = dist_entropy.detach();

                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - &dist_entropy * ENTROPY_COEF;

                // RPO alpha loss: target induced KL for uniform logit perturbation.
                let alpha_loss = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let action_var = action_noise_std.pow_tensor_scalar(2).detach();
                    let inv_var_mean = action_var.clamp_min(1e-4).reciprocal().mean(Kind::Float);
                    let d = ACTION_DIM as f64;
                    let induced_kl = rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                    (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0) * ALPHA_LOSS_COEF
                } else {
                    Tensor::zeros([], (Kind::Float, device))
                };

                let scaled_ppo_loss =
                    &ppo_loss * (chunk_sample_count as f64 / samples_per_accum as f64);
                let scaled_alpha_loss =
                    &alpha_loss * (chunk_sample_count as f64 / samples_per_accum as f64);
                let total_chunk_loss =
                    (scaled_ppo_loss + scaled_alpha_loss) / GRAD_ACCUM_STEPS as f64;

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

                if (chunk_i + 1) % GRAD_ACCUM_STEPS == 0 || chunk_i == chunk_order.len() - 1 {
                    if DEBUG_NUMERICS {
                        let has_nan_grad = tch::no_grad(|| {
                            let mut found = false;
                            for v in opt.trainable_variables() {
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

                    if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                        let max_delta_rho =
                            MAX_DELTA_ALPHA / (0.25 * (RPO_ALPHA_MAX - RPO_ALPHA_MIN));
                        tch::no_grad(|| {
                            let rho_before = rpo_rho.detach();
                            opt.step();
                            let rho_new = &rho_before
                                + (&rpo_rho - &rho_before).clamp(-max_delta_rho, max_delta_rho);
                            let _ = rpo_rho.copy_(&rho_new);
                        });
                    } else {
                        opt.step();
                    }
                    opt.zero_grad();
                }

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
            let norms: Vec<Tensor> = opt.trainable_variables().iter().map(|v| v.norm()).collect();
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
            compute_explained_variance(
                &trading_model,
                &price_deltas_batch,
                &static_obs_batch,
                &returns,
                rollout_steps,
            )
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
            &s_price_deltas,
            &s_static_obs,
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
            let _ = std::fs::create_dir_all("../weights");
            let path = format!("../weights/ppo_ep{}.ot", episode);
            if let Err(err) = vs.save(&path) {
                println!("Error while saving weights: {}", err);
            } else {
                println!("Saved model weights: {}", path);
            }
        }
    }
}
