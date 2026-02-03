use std::env;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::torch::constants::{
    PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::env::VecEnv;
use crate::torch::model::{
    TradingModel, ACTION_DIM, LATTICE_ALPHA, LATTICE_STD_REG, PATCH_SEQ_LEN,
    SDE_EPS, SDE_LATENT_DIM,
};

const LEARNING_RATE: f64 = 3e-4;
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
pub const VALUE_LOG_CLIP: f64 = 10.0;
const VALUE_RAW_CLIP: f64 = 22025.465794806718; // symexp(VALUE_LOG_CLIP) = exp(10) - 1
const CRITIC_ENTROPY_COEF: f64 = 0.0;
const GRAD_ACCUM_STEPS: usize = 1;
pub(crate) const DEBUG_NUMERICS: bool = false;
const LOG_2PI: f64 = 1.8378770664093453;

// RPO: Random Policy Optimization - adds bounded noise to action mean during training and intentionally not during rollout
// Alpha is learned via induced KL targeting. Set all to 0.0 to disable.
const RPO_ALPHA_MIN: f64 = 0.01;
const RPO_ALPHA_MAX: f64 = 0.2;
const RPO_ALPHA_INIT: f64 = 0.1; // CleanRL impl found 0.1 reliably improved results in all test envs over PPO
const RPO_TARGET_KL: f64 = 0.018;
const ALPHA_LOSS_COEF: f64 = 0.1;
const MAX_DELTA_ALPHA: f64 = 0.2;

struct ReturnNormalizer {
    percentile_low: f64,
    percentile_high: f64,
    initialized: bool,
}

impl ReturnNormalizer {
    fn new() -> Self {
        Self { percentile_low: 0.0, percentile_high: 0.0, initialized: false }
    }

    fn update(&mut self, returns: &Tensor) -> f64 {
        const DECAY: f64 = 0.99;
        const LOW_PCT: f64 = 0.05;
        const HIGH_PCT: f64 = 0.95;

        let sorted = returns.to_kind(Kind::Float).sort(0, false).0;
        let n = sorted.size()[0];
        let low_idx = ((n as f64 * LOW_PCT).floor() as i64).clamp(0, n - 1);
        let high_idx = ((n as f64 * HIGH_PCT).floor() as i64).clamp(0, n - 1);
        let p5: f64 = sorted.double_value(&[low_idx]);
        let p95: f64 = sorted.double_value(&[high_idx]);

        if !self.initialized {
            self.percentile_low = p5;
            self.percentile_high = p95;
            self.initialized = true;
        } else {
            self.percentile_low = DECAY * self.percentile_low + (1.0 - DECAY) * p5;
            self.percentile_high = DECAY * self.percentile_high + (1.0 - DECAY) * p95;
        }

        let range = self.percentile_high - self.percentile_low;
        range.max(1.0)
    }
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

/// DreamerV3-style twohot CE loss for non-uniform bin spacing.
/// Finds the two nearest bins via searchsorted, weights by distance.
fn twohot_log_prob_loss(targets: &Tensor, log_probs: &Tensor, centers: &Tensor) -> Tensor {
    let centers = centers.to_kind(log_probs.kind());
    let n_buckets = centers.size()[0];
    let targets = targets.to_kind(log_probs.kind());

    // below = number of bins <= target, minus 1 (index of bin at or just below)
    let below = centers
        .unsqueeze(0)
        .le_tensor(&targets.unsqueeze(1))
        .to_kind(Kind::Int64)
        .sum_dim_intlist(-1, false, Kind::Int64)
        - 1;
    let below = below.clamp(0, n_buckets - 1);
    // above = n_buckets - number of bins > target (index of bin at or just above)
    let above = n_buckets
        - centers
            .unsqueeze(0)
            .gt_tensor(&targets.unsqueeze(1))
            .to_kind(Kind::Int64)
            .sum_dim_intlist(-1, false, Kind::Int64);
    let above = above.clamp(0, n_buckets - 1);

    let bin_below = centers.index_select(0, &below.flatten(0, -1)).reshape_as(&targets);
    let bin_above = centers.index_select(0, &above.flatten(0, -1)).reshape_as(&targets);

    // When below == above (exact bin hit), both distances are 0.
    // clamp_min ensures 0/eps → 0 for both, but we gather the same log_prob
    // for both below and above, so weight_below + weight_above doesn't matter
    // as long as they sum to 1. Adding eps to both distances achieves this.
    let dist_to_below = (&bin_below - &targets).abs() + 1e-8;
    let dist_to_above = (&bin_above - &targets).abs() + 1e-8;
    let total = &dist_to_below + &dist_to_above;
    let weight_below = &dist_to_above / &total;
    let weight_above = &dist_to_below / &total;

    let log_p_below = log_probs
        .gather(1, &below.unsqueeze(1), false)
        .squeeze_dim(1);
    let log_p_above = log_probs
        .gather(1, &above.unsqueeze(1), false)
        .squeeze_dim(1);

    -(weight_below * log_p_below + weight_above * log_p_above).mean(Kind::Float)
}

/// MVN log prob + log|Σ|: returns (log_prob, log_det_sigma) to avoid redundant Cholesky
fn mvn_log_prob(sigma_mat: &Tensor, diff: &Tensor, k: i64) -> (Tensor, Tensor) {
    let jitter = Tensor::eye(k, (Kind::Float, sigma_mat.device())).unsqueeze(0) * SDE_EPS;
    let chol = (sigma_mat + &jitter).linalg_cholesky(false);
    let diff_col = diff.unsqueeze(-1);
    let y = chol.linalg_solve_triangular(&diff_col, false, true, false).squeeze_dim(-1);
    let mahal = y.pow_tensor_scalar(2).sum_dim_intlist([-1].as_slice(), false, Kind::Float);
    let log_det = chol.diagonal(0, -2, -1).log()
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float) * 2.0;
    let log_prob = (&mahal + &log_det + k as f64 * LOG_2PI).g_mul_scalar(-0.5);
    (log_prob, log_det)
}

/// Build Lattice covariance matrix [batch, ACTION_DIM, ACTION_DIM]
/// Σ = α²·W·diag(h²@σ_corr²)·Wᵀ + diag(h²@σ_ind²)
fn build_lattice_covariance(
    sde_latent: &Tensor,   // [batch, SDE_LATENT_DIM]
    corr_std: &Tensor,     // [SDE_LATENT_DIM, SDE_LATENT_DIM]
    ind_std: &Tensor,      // [SDE_LATENT_DIM, ACTION_DIM]
    w_policy: &Tensor,     // [ACTION_DIM, SDE_LATENT_DIM]
) -> Tensor {
    // latent_corr_var[b,d] = sum_j(h[b,j]^2 * sigma_corr[j,d]^2) -- [batch, SDE_LATENT_DIM]
    let latent_corr_var = sde_latent.pow_tensor_scalar(2)
        .matmul(&corr_std.pow_tensor_scalar(2));

    // sigma_corr = alpha^2 * W * diag(latent_corr_var) * W^T -- [batch, ACTION_DIM, ACTION_DIM]
    // w_scaled[b,a,d] = W[a,d] * latent_corr_var[b,d]
    let w_scaled = w_policy.unsqueeze(0) * latent_corr_var.unsqueeze(1);
    let sigma_corr = (LATTICE_ALPHA * LATTICE_ALPHA)
        * w_scaled.matmul(&w_policy.transpose(0, 1));

    // latent_ind_var[b,a] = sum_j(h[b,j]^2 * sigma_ind[j,a]^2) + reg^2 -- [batch, ACTION_DIM]
    let latent_ind_var = sde_latent.pow_tensor_scalar(2)
        .matmul(&ind_std.pow_tensor_scalar(2))
        + LATTICE_STD_REG * LATTICE_STD_REG;

    &sigma_corr + &Tensor::diag_embed(&latent_ind_var, 0, -2, -1)
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

    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut env = VecEnv::new(true);
    if start_episode > 0 {
        env.set_episode(start_episode);
        env.primary_mut().meta_history.load_from_episode(start_episode);
    }

    let rollout_steps = SEQ_LEN;
    let memory_size = rollout_steps * NPROCS;

    let pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
    let so_dim = STATIC_OBSERVATIONS as i64;
    let seq_idx_dim = TICKERS_COUNT * PATCH_SEQ_LEN;

    let mut s_price_deltas = GpuRollingBuffer::new(memory_size, pd_dim, Kind::Float, device);
    let mut s_static_obs = GpuRollingBuffer::new(memory_size, so_dim, Kind::Float, device);
    let mut s_seq_idx = GpuRollingBuffer::new(memory_size, seq_idx_dim, Kind::Int64, device);
    let s_actions = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
    let s_old_log_probs = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let s_rewards = Tensor::zeros(&[memory_size], (Kind::Float, device)); // portfolio-level reward
    let s_dones = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let s_values = Tensor::zeros(&[memory_size], (Kind::Float, device)); // portfolio-level value
    let s_action_weights =
        Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
    // Store lattice stds used during rollout for consistent log_prob computation
    let mut rollout_corr_std = Tensor::zeros(&[SDE_LATENT_DIM, SDE_LATENT_DIM], (Kind::Float, device));
    let mut rollout_ind_std = Tensor::zeros(&[SDE_LATENT_DIM, ACTION_DIM], (Kind::Float, device));

    let mut return_normalizer = ReturnNormalizer::new();

    for episode in start_episode..1000000 {
        let (obs_price_cpu, obs_static_cpu, obs_seq_idx_cpu) = env.reset();
        let mut obs_price = Tensor::zeros(&[NPROCS, pd_dim], (Kind::Float, device));
        let mut obs_static =
            Tensor::zeros(&[NPROCS, STATIC_OBSERVATIONS as i64], (Kind::Float, device));
        let mut obs_seq_idx = Tensor::zeros(&[NPROCS, seq_idx_dim], (Kind::Int64, device));
        let ring_len = PRICE_DELTAS_PER_TICKER as i64;
        let base_idx = Tensor::arange(ring_len, (Kind::Int64, device));
        let mut ring_pos = ring_len - 1;
        let mut ring_buf = Tensor::zeros(
            &[NPROCS, TICKERS_COUNT, ring_len],
            (Kind::Float, device),
        );
        let mut step_deltas = Tensor::zeros(&[NPROCS, TICKERS_COUNT], (Kind::Float, device));
        obs_price.copy_(&obs_price_cpu);
        obs_static.copy_(&obs_static_cpu);
        obs_seq_idx.copy_(&obs_seq_idx_cpu);
        ring_buf.copy_(&obs_price.view([NPROCS, TICKERS_COUNT, ring_len]));
        let mut step_reward_per_ticker =
            Tensor::zeros(&[NPROCS, TICKERS_COUNT], (Kind::Float, device));
        let mut step_cash_reward = Tensor::zeros(&[NPROCS], (Kind::Float, device));
        let mut step_is_done = Tensor::zeros(&[NPROCS], (Kind::Float, device));

        let stats_kind = (Kind::Float, device);

        // Capture lattice stds at start of rollout for consistent log_prob during training
        tch::no_grad(|| {
            let (corr, ind) = trading_model.lattice_stds();
            rollout_corr_std.copy_(&corr);
            rollout_ind_std.copy_(&ind);
        });

        for step in 0..rollout_steps as usize {
            let (values, action_mean, u, actions, action_log_prob) = tch::no_grad(|| {
                let (values, _, (action_mean, sde_latent)) = trading_model.forward_with_seq_idx_on_device(
                    &obs_price,
                    &obs_static,
                    Some(&obs_seq_idx),
                    false,
                );
                let values = values.to_kind(Kind::Float);
                let action_mean = action_mean.to_kind(Kind::Float);
                let sde_latent = sde_latent.to_kind(Kind::Float);

                let corr_exploration_mat = Tensor::randn([SDE_LATENT_DIM, SDE_LATENT_DIM], stats_kind)
                    * &rollout_corr_std;
                let ind_exploration_mat = Tensor::randn([SDE_LATENT_DIM, ACTION_DIM], stats_kind)
                    * &rollout_ind_std;

                // Correlated noise: perturb shared latent, project through W
                let latent_noise = sde_latent.matmul(&corr_exploration_mat); // [batch, SDE_LATENT_DIM]
                let w = trading_model.w_policy(); // [ACTION_DIM, SDE_LATENT_DIM]
                let correlated_action_noise = LATTICE_ALPHA
                    * latent_noise.matmul(&w.transpose(0, 1)); // [batch, ACTION_DIM]

                // Independent noise: project shared latent through ind noise
                let independent_action_noise = sde_latent.matmul(&ind_exploration_mat); // [batch, ACTION_DIM]

                let noise = &correlated_action_noise + &independent_action_noise;
                let u = &action_mean + &noise;
                let actions = u.softmax(-1, Kind::Float);

                // Log prob via MVN
                let sigma_mat = build_lattice_covariance(
                    &sde_latent, &rollout_corr_std, &rollout_ind_std, &trading_model.w_policy(),
                );
                let diff = &u - &action_mean;
                let (log_prob_gaussian, _) = mvn_log_prob(&sigma_mat, &diff, ACTION_DIM);
                let log_det_jac = u.log_softmax(-1, Kind::Float)
                    .sum_dim_intlist(-1, false, Kind::Float);
                let action_log_prob = &log_prob_gaussian - &log_det_jac;

                (values, action_mean, u, actions, action_log_prob)
            });

            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("action_mean", &action_mean, episode as i64, step);
                let _ = debug_tensor_stats("u", &u, episode as i64, step);
                let _ = debug_tensor_stats("action_log_prob", &action_log_prob, episode as i64, step);
            }
            s_price_deltas.push(&obs_price);
            s_static_obs.push(&obs_static);
            s_seq_idx.push(&obs_seq_idx);

            let (reset_indices, reset_price_deltas) = env.step_into_ring_tensor(
                &actions,
                &mut step_deltas,
                &mut obs_static,
                &mut step_reward_per_ticker,
                &mut step_cash_reward,
                &mut step_is_done,
            );

            ring_pos = (ring_pos + 1) % ring_len;
            let _ = ring_buf
                .narrow(2, ring_pos, 1)
                .copy_(&step_deltas.unsqueeze(-1));

            if !reset_indices.is_empty() {
                let idx = (&base_idx + (ring_pos + 1)).remainder(ring_len);
                let pd_dim_usize =
                    (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
                for (reset_i, env_idx) in reset_indices.iter().enumerate() {
                    let start = reset_i * pd_dim_usize;
                    let end = start + pd_dim_usize;
                    let ordered = Tensor::from_slice(&reset_price_deltas[start..end])
                        .view([TICKERS_COUNT, ring_len])
                        .to_device(device);
                    let mut ring_env = ring_buf
                        .narrow(0, *env_idx as i64, 1)
                        .squeeze_dim(0);
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

            // Portfolio-level reward: include cash penalty to avoid trivial cash-hold policy.
            let portfolio_reward =
                step_reward_per_ticker.mean_dim([1].as_slice(), false, Kind::Float) + &step_cash_reward;
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
            let _ = s_values.narrow(0, mem_idx, NPROCS);
            let _ = s_action_weights
                .narrow(0, mem_idx, NPROCS)
                .copy_(&actions);
        }

        // Compute GAE on portfolio-level values
        let advantages = Tensor::zeros(&[memory_size], (Kind::Float, device));
        let returns = Tensor::zeros(&[memory_size], (Kind::Float, device));
        let gamma = 0.99f64;
        let gae_lambda = 0.95f64;

        // Bootstrap value from final observation state
        let bootstrap_value = tch::no_grad(|| {
            let (values, _, _) = trading_model.forward_with_seq_idx_on_device(
                &obs_price,
                &obs_static,
                Some(&obs_seq_idx),
                false,
            );
            values.to_kind(Kind::Float)
        });

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

        let advantages = advantages.detach();
        let returns = returns.detach();

        let raw_clip = VALUE_RAW_CLIP;
        let max_ret_abs = returns.abs().max().double_value(&[]);
        if max_ret_abs > raw_clip {
            eprintln!(
                "Warning: returns exceed bin range: max_abs={:.6} clip={:.6}",
                max_ret_abs, raw_clip
            );
        }

        let ret_norm_scale = return_normalizer.update(&returns);

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
                let adv_mb = &adv_mb_raw / ret_norm_scale;
                let old_log_probs_mb =
                    s_old_log_probs.narrow(0, chunk_sample_start, chunk_sample_count);
                let _weight_mb =
                    action_weights_batch.narrow(0, chunk_sample_start, chunk_sample_count);

                let fwd_start = Instant::now();
                let (_, critic_logits, (action_mean, sde_latent)) =
                    trading_model.forward_with_seq_idx_no_values_on_device(
                        &pd_chunk,
                        &so_chunk,
                        Some(&seq_idx_chunk),
                        true,
                    );

                let action_mean = action_mean.to_kind(Kind::Float);
                let sde_latent = sde_latent.to_kind(Kind::Float); // [chunk, SDE_LATENT_DIM]

                // act_mb contains the stored u (pre-softmax logits)
                let u = act_mb;

                // learn_features=false: detach sde_latent for covariance (h²) path
                // Action mean gradients still flow through the attached sde_latent
                let sde_latent_detached = sde_latent.detach();
                let (corr_std, ind_std) = trading_model.lattice_stds();
                let sigma_mat = build_lattice_covariance(
                    &sde_latent_detached, &corr_std, &ind_std, &trading_model.w_policy(),
                );

                // RPO: sigmoid parameterization for smooth gradients
                let (rpo_alpha, action_mean_perturbed) = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let alpha = RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                    let alpha_detached = alpha.detach();
                    let rpo_noise = Tensor::empty(
                        [chunk_sample_count, ACTION_DIM],
                        (Kind::Float, device),
                    )
                    .uniform_(-1.0, 1.0)
                        * &alpha_detached;
                    (alpha, &action_mean + rpo_noise)
                } else {
                    (Tensor::zeros(&[1], (Kind::Float, device)), action_mean.shallow_clone())
                };

                let diff = &u - &action_mean_perturbed;
                let (log_prob_gaussian, log_det_sigma) = mvn_log_prob(&sigma_mat, &diff, ACTION_DIM);
                let log_det_jac = u
                    .log_softmax(-1, Kind::Float)
                    .sum_dim_intlist(-1, false, Kind::Float);
                let action_log_probs = log_prob_gaussian - log_det_jac;

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("u", &u, _epoch, chunk_i);
                    let _ = debug_tensor_stats("old_log_probs_mb", &old_log_probs_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_mean", &action_mean, _epoch, chunk_i);
                    let _ = debug_tensor_stats("sigma_mat", &sigma_mat, _epoch, chunk_i);
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

                // Portfolio-level value loss
                let log_probs = critic_logits
                    .to_kind(Kind::Float)
                    .log_softmax(-1, Kind::Float);
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("ret_mb", &ret_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("critic_logits", &critic_logits, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_probs", &log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("adv_mb", &adv_mb, _epoch, chunk_i);
                }
                // Bins are in raw return space (symexp-spaced), targets stay raw
                let ce_loss = twohot_log_prob_loss(
                    &ret_mb,
                    &log_probs,
                    trading_model.value_centers(),
                );
                let critic_entropy = -(log_probs.exp() * &log_probs)
                    .sum_dim_intlist(-1, false, Kind::Float)
                    .mean(Kind::Float);
                let value_loss = ce_loss - CRITIC_ENTROPY_COEF * critic_entropy;

                // MVN entropy: 0.5 * (k*(1+ln(2pi)) + ln|Sigma|), reuses Cholesky from log_prob
                let dist_entropy = (ACTION_DIM as f64 * (1.0 + LOG_2PI) + &log_det_sigma)
                    .g_mul_scalar(0.5).mean(Kind::Float);
                let dist_entropy_detached = dist_entropy.detach();

                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - &dist_entropy * ENTROPY_COEF;

                // RPO alpha loss: target induced KL (uniform noise in logit space)
                let alpha_loss = if RPO_ALPHA_MAX > RPO_ALPHA_MIN {
                    let action_var = sigma_mat.diagonal(0, -2, -1).detach();
                    let inv_var_mean = action_var.clamp_min(1e-4).reciprocal().mean(Kind::Float);
                    let d = ACTION_DIM as f64;
                    let induced_kl =
                        rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
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
                let _ = total_kl_weighted.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                let _ = total_entropy_weighted.g_add_(&(&dist_entropy_detached * chunk_sample_count as f64));
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
            let ev_steps = CHUNK_SIZE.min(rollout_steps);
            let ev_samples = ev_steps * NPROCS;
            tch::no_grad(|| {
                let pd_ev = price_deltas_batch.narrow(0, 0, ev_samples);
                let so_ev = static_obs_batch.narrow(0, 0, ev_samples);
                let seq_ev = seq_idx_batch.narrow(0, 0, ev_samples);
                let ret_ev = returns.narrow(0, 0, ev_samples);
                let (values, _, _) = trading_model.forward_with_seq_idx_on_device(
                    &pd_ev,
                    &so_ev,
                    Some(&seq_ev),
                    true,
                );
                let raw_clip = VALUE_RAW_CLIP;
                let values = values
                    .to_kind(Kind::Float)
                    .view([ev_samples])
                    .clamp(-raw_clip, raw_clip);
                let residuals = &values - &ret_ev;
                let mean_target = ret_ev.mean(Kind::Float);
                let var_targets =
                    ret_ev.square().mean(Kind::Float) - mean_target.square();
                let var_residuals = residuals.square().mean(Kind::Float);
                Tensor::from(1.0) - &var_residuals / var_targets.clamp_min(1e-8)
            })
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };

        let entropy_mean_t = if total_sample_count > 0 {
            &total_entropy_weighted / total_sample_count as f64
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };

        // Compute action std stats + rpo_alpha
        let log_std_stats = tch::no_grad(|| {
            let (_, _, (_, sde_latent)) = trading_model.forward_with_seq_idx_on_device(
                &s_price_deltas.get(0),
                &s_static_obs.get(0),
                Some(&s_seq_idx.get(0)),
                false,
            );
            let (corr_std, ind_std) = trading_model.lattice_stds();
            let sigma_mat = build_lattice_covariance(
                &sde_latent, &corr_std, &ind_std, &trading_model.w_policy(),
            );
            let action_std = sigma_mat.diagonal(0, -2, -1).clamp_min(1e-6).sqrt();
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
        let (entropy_mean, entropy_min_val, entropy_max_val) =
            (all_scalars_vec[12], all_scalars_vec[13], all_scalars_vec[14]);

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
        primary.meta_history.record_policy_entropy(entropy_mean, entropy_min_val, entropy_max_val);

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
