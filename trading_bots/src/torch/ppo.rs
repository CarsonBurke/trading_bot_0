use std::env;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::constants::TICKERS;
use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::env::VecEnv;
use crate::torch::model::{symexp_tensor, symlog_tensor, TradingModel, PATCH_SEQ_LEN};

const LEARNING_RATE: f64 = 1e-4;
pub const NPROCS: i64 = 16;
const SEQ_LEN: i64 = 4000;
const CHUNK_SIZE: i64 = 128;
const OPTIM_EPOCHS: i64 = 4;
const PPO_CLIP_RATIO: f64 = 0.2;
const TARGET_KL: f64 = 0.03;
const KL_STOP_MULTIPLIER: f64 = 1.5;
const VALUE_LOSS_COEF: f64 = 0.5;
const ENTROPY_COEF: f64 = 0.0;
const MAX_GRAD_NORM: f64 = 0.5;
pub const VALUE_LOG_CLIP: f64 = 8.0;
const CRITIC_ENTROPY_COEF: f64 = 0.01;
const GRAD_ACCUM_STEPS: usize = 2;
pub(crate) const DEBUG_NUMERICS: bool = false;
const LOG_2PI: f64 = 1.8378770664093453;

// RPO: Random Policy Optimization - adds bounded noise to action mean during training
// Alpha is learned via induced KL targeting. Set all to 0.0 to disable.
const RPO_ALPHA_MIN: f64 = 0.01;
const RPO_ALPHA_MAX: f64 = 0.3;
const RPO_ALPHA_INIT: f64 = 0.1;
const RPO_TARGET_KL: f64 = 0.018;
const ALPHA_LOSS_COEF: f64 = 1.0;
const MAX_DELTA_ALPHA: f64 = 0.02;

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

fn twohot_encode(t: &Tensor, centers: &Tensor) -> Tensor {
    let n_buckets = centers.size()[0];
    let device = t.device();
    let flat_t = t.flatten(0, -1);
    let n_elements = flat_t.size()[0];

    let centers_expanded = centers.unsqueeze(0);
    let flat_t_expanded = flat_t.unsqueeze(1);

    let diff = (&flat_t_expanded - &centers_expanded).abs();
    let idx = diff.argmin(1, false);

    let low_idx = idx.shallow_clone();
    let high_idx = (idx + 1).clamp(0, n_buckets - 1);

    let low_val = centers.index_select(0, &low_idx);
    let high_val = centers.index_select(0, &high_idx);

    let dist = (&high_val - &low_val).clamp_min(1e-6);
    let weight_high = (&flat_t - &low_val) / &dist;
    let weight_high = weight_high.clamp(0.0, 1.0);
    let weight_low = weight_high.g_mul_scalar(-1.0) + 1.0;

    let mut out = Tensor::zeros(&[n_elements, n_buckets], (Kind::Float, device));
    let _ = out.scatter_(1, &low_idx.unsqueeze(1), &weight_low.unsqueeze(1));
    let _ = out.scatter_(1, &high_idx.unsqueeze(1), &weight_high.unsqueeze(1));

    let mut shape = t.size();
    shape.push(n_buckets);
    out.view(shape.as_slice())
}

pub async fn train(weights_path: Option<&str>) {
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
    let trading_model = TradingModel::new(&vs.root());

    // RPO alpha via sigmoid: alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(rho)
    let mut rpo_rho = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
        let p_init = (RPO_ALPHA_INIT - RPO_ALPHA_MIN) / (RPO_ALPHA_MAX - RPO_ALPHA_MIN);
        let p_init = p_init.clamp(1e-6, 1.0 - 1e-6);
        let rho_init = (p_init / (1.0 - p_init)).ln();
        vs.root().var("rpo_alpha_rho", &[1], nn::Init::Const(rho_init))
    } else {
        // RPO disabled - create dummy tensor
        Tensor::zeros(&[1], (Kind::Float, device))
    };

    if let Some(path) = weights_path {
        println!("Loading weights from {}", path);
        vs.load(path).unwrap();
    } else {
        println!("Starting training from scratch");
    }

    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut env = VecEnv::new(true);

    let rollout_steps = SEQ_LEN;
    let memory_size = rollout_steps * NPROCS;

    let pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
    let so_dim = STATIC_OBSERVATIONS as i64;
    let seq_idx_dim = TICKERS_COUNT * PATCH_SEQ_LEN;

    let mut s_price_deltas = GpuRollingBuffer::new(memory_size, pd_dim, Kind::Float, device);
    let mut s_static_obs = GpuRollingBuffer::new(memory_size, so_dim, Kind::Float, device);
    let mut s_seq_idx = GpuRollingBuffer::new(memory_size, seq_idx_dim, Kind::Int64, device);
    let mut s_actions = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
    let mut s_old_log_probs = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let mut s_rewards = Tensor::zeros(&[memory_size], (Kind::Float, device)); // portfolio-level reward
    let mut s_dones = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let mut s_values = Tensor::zeros(&[memory_size], (Kind::Float, device)); // portfolio-level value
    let mut s_action_weights =
        Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));

    for episode in 0..1000000 {
        let (obs_price_cpu, obs_static_cpu, obs_seq_idx_cpu) = env.reset();
        let mut obs_price = Tensor::zeros(&[NPROCS, pd_dim], (Kind::Float, device));
        let mut obs_static =
            Tensor::zeros(&[NPROCS, STATIC_OBSERVATIONS as i64], (Kind::Float, device));
        let mut obs_seq_idx = Tensor::zeros(&[NPROCS, seq_idx_dim], (Kind::Int64, device));
        obs_price.copy_(&obs_price_cpu);
        obs_static.copy_(&obs_static_cpu);
        obs_seq_idx.copy_(&obs_seq_idx_cpu);
        let mut step_reward_per_ticker =
            Tensor::zeros(&[NPROCS, TICKERS_COUNT], (Kind::Float, device));
        let mut step_cash_reward = Tensor::zeros(&[NPROCS], (Kind::Float, device));
        let mut step_is_done = Tensor::zeros(&[NPROCS], (Kind::Float, device));

        let action_dim = (TICKERS_COUNT + 1) as usize;
        let mut actions_flat = vec![0.0f64; NPROCS as usize * action_dim];
        let mut actions_vec = vec![vec![0.0f64; action_dim]; NPROCS as usize];

        let stats_kind = (Kind::Float, device);
        for step in 0..rollout_steps as usize {
            let (values, _, (action_mean, action_log_std), _) = tch::no_grad(|| {
                trading_model.forward_with_seq_idx(
                    &obs_price,
                    &obs_static,
                    Some(&obs_seq_idx),
                    false,
                )
            });
            let values = values.to_kind(Kind::Float);
            let action_mean = action_mean.to_kind(Kind::Float);
            let action_log_std = action_log_std.to_kind(Kind::Float);

            // Sample: u = mean + std * noise (tickers only), then softmax for simplex actions
            // action_log_std is [batch, tickers] - no cash
            let action_std = action_log_std.exp();
            let noise_ticker = Tensor::randn([NPROCS, TICKERS_COUNT], stats_kind);
            let action_mean_ticker = action_mean.narrow(1, 0, TICKERS_COUNT);
            let action_mean_cash = action_mean.narrow(1, TICKERS_COUNT, 1);
            let u_ticker = &action_mean_ticker + &action_std * &noise_ticker;
            let u = Tensor::cat(&[u_ticker, action_mean_cash], 1);
            let actions = u.softmax(-1, Kind::Float);

            // Log prob: Gaussian on tickers only (cash is deterministic)
            let u_ticker = u.narrow(1, 0, TICKERS_COUNT);
            let u_normalized = (&u_ticker - &action_mean_ticker) / &action_std;
            let u_squared = u_normalized.pow_tensor_scalar(2);
            let two_log_std = &action_log_std * 2.0;
            let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
            let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
            let log_det = u
                .log_softmax(-1, Kind::Float)
                .sum_dim_intlist(-1, false, Kind::Float);
            let action_log_prob = &log_prob_gaussian - &log_det;

            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("action_mean", &action_mean, episode as i64, step);
                let _ = debug_tensor_stats("action_log_std", &action_log_std, episode as i64, step);
                let _ = debug_tensor_stats("u", &u, episode as i64, step);
                let _ = debug_tensor_stats("action_log_prob", &action_log_prob, episode as i64, step);
            }
            let actions_cpu = actions
                .flatten(0, -1)
                .to_device(tch::Device::Cpu)
                .to_kind(Kind::Double);
            tch::Cuda::synchronize(0);
            let actions_flat_len = actions_flat.len();
            actions_cpu.copy_data(&mut actions_flat, actions_flat_len);
            for i in 0..NPROCS as usize {
                let start = i * action_dim;
                actions_vec[i].copy_from_slice(&actions_flat[start..start + action_dim]);
            }
            env.step_into_full(
                &actions_vec,
                &mut obs_price,
                &mut obs_static,
                &mut step_reward_per_ticker,
                &mut step_cash_reward,
                &mut step_is_done,
                &mut obs_seq_idx,
            );

            let mem_idx = step as i64 * NPROCS;

            s_price_deltas.push(&obs_price);
            s_static_obs.push(&obs_static);
            s_seq_idx.push(&obs_seq_idx);
            let _ = s_actions.narrow(0, mem_idx, NPROCS).copy_(&u); // Store pre-softmax u for training
            let _ = s_old_log_probs
                .narrow(0, mem_idx, NPROCS)
                .copy_(&action_log_prob);

            // Portfolio-level reward: include cash penalty to avoid trivial cash-hold policy.
            let portfolio_reward =
                step_reward_per_ticker.sum_dim_intlist([1].as_slice(), false, Kind::Float) + &step_cash_reward;
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
            let _ = s_action_weights
                .narrow(0, mem_idx, NPROCS)
                .copy_(&actions);
        }

        // Compute GAE on portfolio-level values
        let advantages = Tensor::zeros(&[memory_size], (Kind::Float, device));
        let returns = Tensor::zeros(&[memory_size], (Kind::Float, device));
        let gamma = 0.99f64;
        let gae_lambda = 0.95f64;

        tch::no_grad(|| {
            let mut last_gae = Tensor::zeros(&[NPROCS], (Kind::Float, device));
            for t in (0..rollout_steps).rev() {
                let mem_idx = t * NPROCS;
                let next_values = if t == rollout_steps - 1 {
                    Tensor::zeros(&[NPROCS], (Kind::Float, device))
                } else {
                    s_values.narrow(0, (t + 1) * NPROCS, NPROCS)
                };
                let cur_values = s_values.narrow(0, mem_idx, NPROCS);
                let rewards = s_rewards.narrow(0, mem_idx, NPROCS);
                let dones = s_dones.narrow(0, mem_idx, NPROCS);

                let delta = rewards + (1.0 - &dones) * gamma * &next_values - &cur_values;
                last_gae = delta + (1.0 - &dones) * gamma * gae_lambda * &last_gae;
                let _ = advantages.narrow(0, mem_idx, NPROCS).copy_(&last_gae);
                let _ = returns
                    .narrow(0, mem_idx, NPROCS)
                    .copy_(&(&last_gae + &cur_values));
            }
        });

        let advantages = advantages.detach();
        let returns = returns.detach();

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
        let seq_idx_batch = s_seq_idx.data.shallow_clone();
        let action_weights_batch = s_action_weights.shallow_clone();

        let mut total_kl_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_policy_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_value_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        // Explained variance in symlog space: EV = 1 - Var(residuals) / Var(targets)
        let mut total_residual_sq = Tensor::zeros([], (Kind::Float, device));
        let mut total_target_sum = Tensor::zeros([], (Kind::Float, device));
        let mut total_target_sq_sum = Tensor::zeros([], (Kind::Float, device));
        let mut grad_norm_sum = Tensor::zeros([], (Kind::Float, device));
        let mut total_sample_count = 0i64;
        let mut grad_norm_count = 0i64;
        let mut total_clipped = Tensor::zeros([], (Kind::Float, device));
        let mut total_ratio_samples = 0i64;

        let mut fwd_time_us = 0u64;
        let mut bwd_time_us = 0u64;

        let num_chunks = (rollout_steps + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunk_order: Vec<usize> = (0..num_chunks as usize).collect();


        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            use rand::seq::SliceRandom;
            chunk_order.shuffle(&mut rand::rng());
            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;

            for (chunk_i, &chunk_idx) in chunk_order.iter().enumerate() {
                let chunk_start_step = chunk_idx as i64 * CHUNK_SIZE;
                let chunk_end_step = ((chunk_idx as i64 + 1) * CHUNK_SIZE).min(rollout_steps);
                let chunk_len = chunk_end_step - chunk_start_step;
                let chunk_sample_count = chunk_len * NPROCS;
                let chunk_sample_start = chunk_start_step * NPROCS;

                let pd_chunk = price_deltas_batch.narrow(0, chunk_sample_start, chunk_sample_count);
                let so_chunk = static_obs_batch.narrow(0, chunk_sample_start, chunk_sample_count);
                let seq_idx_chunk = seq_idx_batch.narrow(0, chunk_sample_start, chunk_sample_count);
                let act_mb = s_actions.narrow(0, chunk_sample_start, chunk_sample_count);
                let ret_mb = returns.narrow(0, chunk_sample_start, chunk_sample_count);
                let adv_mb_raw = advantages.narrow(0, chunk_sample_start, chunk_sample_count);
                // Per-minibatch advantage normalization (CleanRL style)
                let adv_mb = {
                    let mb_mean = adv_mb_raw.mean(Kind::Float);
                    let mb_std = adv_mb_raw.std(false) + 1e-8;
                    (&adv_mb_raw - mb_mean) / mb_std
                };
                let old_log_probs_mb =
                    s_old_log_probs.narrow(0, chunk_sample_start, chunk_sample_count);
                let weight_mb =
                    action_weights_batch.narrow(0, chunk_sample_start, chunk_sample_count);

                let fwd_start = Instant::now();
                let (values, critic_logits, (action_mean, action_log_stds), _attn_entropy) =
                    trading_model.forward_with_seq_idx(
                        &pd_chunk,
                        &so_chunk,
                        Some(&seq_idx_chunk),
                        true,
                    );

                let values = values.to_kind(Kind::Float).view([chunk_sample_count]); // portfolio-level
                let action_mean = action_mean.to_kind(Kind::Float);
                let action_log_stds = action_log_stds.to_kind(Kind::Float);

                // act_mb contains the stored u (pre-softmax logits)
                let u = act_mb;

                // RPO: sigmoid parameterization for smooth gradients
                let (rpo_alpha, action_mean_perturbed) = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let alpha = RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                    let alpha_detached = alpha.detach();
                    let rpo_noise_ticker = Tensor::empty(
                        [chunk_sample_count, TICKERS_COUNT],
                        (Kind::Float, device),
                    )
                    .uniform_(-1.0, 1.0)
                        * &alpha_detached;
                    let rpo_noise_cash = Tensor::zeros([chunk_sample_count, 1], (Kind::Float, device));
                    let rpo_noise = Tensor::cat(&[rpo_noise_ticker, rpo_noise_cash], 1);
                    (alpha, &action_mean + rpo_noise)
                } else {
                    (Tensor::zeros(&[1], (Kind::Float, device)), action_mean.shallow_clone())
                };

                // Log prob: Gaussian on tickers only (cash is deterministic)
                let action_std = action_log_stds.exp();
                let two_log_std = &action_log_stds * 2.0;
                let u_ticker = u.narrow(1, 0, TICKERS_COUNT);
                let action_mean_ticker = action_mean_perturbed.narrow(1, 0, TICKERS_COUNT);
                let u_normalized = (&u_ticker - &action_mean_ticker) / &action_std;
                let u_squared = u_normalized.pow_tensor_scalar(2);
                let log_prob_u = (&u_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);
                let log_prob_gaussian = log_prob_u.sum_dim_intlist(-1, false, Kind::Float);
                let log_det = u
                    .log_softmax(-1, Kind::Float)
                    .sum_dim_intlist(-1, false, Kind::Float);
                let action_log_probs = (log_prob_gaussian - log_det).nan_to_num(0.0, 0.0, 0.0);

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("u", &u, _epoch, chunk_i);
                    let _ = debug_tensor_stats("old_log_probs_mb", &old_log_probs_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_mean", &action_mean, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_log_stds", &action_log_stds, _epoch, chunk_i);
                }

                let log_ratio = &action_log_probs - &old_log_probs_mb;
                // Soft clamp log_ratio to prevent extreme probability ratios
                // let log_ratio = log_ratio_raw.tanh() * 0.3;
                if DEBUG_NUMERICS {
                    let _ =
                        debug_tensor_stats("action_log_probs", &action_log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_ratio", &log_ratio, _epoch, chunk_i);
                }
                let ratio = log_ratio.exp();
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_RATIO, 1.0 + PPO_CLIP_RATIO);

                // Single portfolio advantage - no more weighted combination
                let action_loss =
                    -Tensor::min_other(&(&ratio * &adv_mb), &(&ratio_clipped * &adv_mb))
                        .mean(Kind::Float);

                // Portfolio-level value loss: ret_mb is [chunk_sample_count], critic_logits is [chunk_sample_count, NUM_VALUE_BUCKETS]
                let returns_symlog = symlog_tensor(&ret_mb.clamp(-VALUE_LOG_CLIP, VALUE_LOG_CLIP));
                let target_twohot = twohot_encode(&returns_symlog, trading_model.value_centers())
                    .view([chunk_sample_count, -1]);
                let log_probs = critic_logits
                    .to_kind(Kind::Float)
                    .log_softmax(-1, Kind::Float);
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("returns_symlog", &returns_symlog, _epoch, chunk_i);
                    let _ = debug_tensor_stats("target_twohot", &target_twohot, _epoch, chunk_i);
                    let _ = debug_tensor_stats("critic_logits", &critic_logits, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_probs", &log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("adv_mb", &adv_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("ret_mb", &ret_mb, _epoch, chunk_i);
                }
                let ce_loss = -(target_twohot * &log_probs)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);
                let critic_entropy = -(log_probs.exp() * &log_probs)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);
                let value_loss = ce_loss - CRITIC_ENTROPY_COEF * critic_entropy;

                // Entropy of Gaussian (per action dimension)
                let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * &action_log_stds;
                let dist_entropy = entropy_components.g_mul_scalar(0.5).mean(Kind::Float);

                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - dist_entropy * ENTROPY_COEF;

                // RPO alpha loss: target induced KL
                let alpha_loss = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let action_std_detached = action_log_stds.detach().exp();
                    let var = action_std_detached.pow_tensor_scalar(2);
                    let inv_var_mean = var.clamp_min(1e-4).reciprocal().mean(Kind::Float);
                    let d = TICKERS_COUNT as f64;
                    let induced_kl = rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                    (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0) * ALPHA_LOSS_COEF
                } else {
                    Tensor::zeros([], (Kind::Float, device))
                };

                let total_chunk_loss = (ppo_loss.shallow_clone() + alpha_loss) / GRAD_ACCUM_STEPS as f64;

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
                let _ =
                    total_policy_loss_weighted.g_add_(&(&action_loss * chunk_sample_count as f64));
                let _ =
                    total_value_loss_weighted.g_add_(&(&value_loss * chunk_sample_count as f64));
                // Explained variance in symlog space (reuse returns_symlog from loss computation)
                let values_symlog = symlog_tensor(&values.clamp(-VALUE_LOG_CLIP, VALUE_LOG_CLIP));
                let residuals_symlog = &values_symlog - &returns_symlog;
                let _ = total_residual_sq.g_add_(&residuals_symlog.square().sum(Kind::Float));
                let _ = total_target_sum.g_add_(&returns_symlog.sum(Kind::Float));
                let _ = total_target_sq_sum.g_add_(&returns_symlog.square().sum(Kind::Float));
                let _ = total_kl_weighted.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
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
                        let max_delta_rho = MAX_DELTA_ALPHA / (0.25 * (RPO_ALPHA_MAX - RPO_ALPHA_MIN));
                        tch::no_grad(|| {
                            let rho_before = rpo_rho.detach();
                            opt.step();
                            let rho_new = &rho_before + (&rpo_rho - &rho_before).clamp(-max_delta_rho, max_delta_rho);
                            let _ = rpo_rho.copy_(&rho_new);
                        });
                    } else {
                        opt.step();
                    }
                    opt.zero_grad();
                }

                let _ = total_clipped.g_add_(&tch::no_grad(|| {
                    (&ratio - 1.0)
                        .abs()
                        .gt(PPO_CLIP_RATIO)
                        .to_kind(Kind::Float)
                        .sum(Kind::Float)
                }));
                total_ratio_samples += chunk_sample_count;
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
        let (mean_policy_loss_t, mean_value_loss_t, explained_var_t, mean_grad_norm_t, clip_frac_t) =
            if total_sample_count > 0 {
                let n = total_sample_count as f64;
                let mean_policy = &total_policy_loss_weighted / n;
                let mean_value = &total_value_loss_weighted / n;
                // EV = 1 - Var(residuals) / Var(targets)
                let mean_target = &total_target_sum / n;
                let var_targets = &total_target_sq_sum / n - mean_target.square();
                let var_residuals = &total_residual_sq / n;
                let explained_var =
                    Tensor::from(1.0) - &var_residuals / var_targets.clamp_min(1e-8);
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
                (mean_policy, mean_value, explained_var, grad_norm, clip)
            } else {
                (
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                )
            };

        // Compute action std stats + rpo_alpha
        let log_std_stats = tch::no_grad(|| {
            let (_, _, (_, action_log_std), _) = trading_model.forward_with_seq_idx(
                &s_price_deltas.get(0),
                &s_static_obs.get(0),
                Some(&s_seq_idx.get(0)),
                false,
            );
            let action_std = action_log_std.exp();
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
        });

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
            ],
            0,
        );
        let all_scalars_vec: Vec<f64> = Vec::try_from(all_scalars.to_device(tch::Device::Cpu))
            .unwrap_or_else(|_| vec![0.0; 12]);
        let mean_policy_loss = all_scalars_vec[0];
        let mean_value_loss = all_scalars_vec[1];
        let explained_var = all_scalars_vec[2];
        let mean_grad_norm = all_scalars_vec[3];
        let clip_frac = all_scalars_vec[4];
        let (adv_mean, adv_min, adv_max) =
            (all_scalars_vec[5], all_scalars_vec[6], all_scalars_vec[7]);
        let log_std_stats_vec = &all_scalars_vec[8..12];

        let primary = env.primary_mut();
        primary
            .meta_history
            .record_advantage_stats(adv_mean, adv_min, adv_max);
        // Record log-std stats in place of logit noise stats
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
