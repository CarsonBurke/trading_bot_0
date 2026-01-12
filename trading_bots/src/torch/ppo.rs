use std::env;
use std::time::Instant;
use tch::{nn, Kind, Tensor, nn::OptimizerConfig};

use crate::constants::TICKERS;
use crate::torch::constants::{ACTION_COUNT, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS};
use crate::torch::model::{TradingModel, symlog_tensor, symexp_tensor};
use crate::torch::env::VecEnv;

pub const NPROCS: i64 = 16;
const SEQ_LEN: i64 = 4000;
const CHUNK_SIZE: i64 = 128;
const OPTIM_EPOCHS: i64 = 4;
const PPO_CLIP_RATIO: f64 = 0.2;
const TARGET_KL: f64 = 0.03;
const KL_STOP_MULTIPLIER: f64 = 1.5;
const VALUE_LOSS_COEF: f64 = 0.5;
const ENTROPY_COEF: f64 = 0.01;
const ATTENTION_ENTROPY_COEF: f64 = 0.001;
const ALPHA_LOSS_COEF: f64 = 0.1;
const MAX_GRAD_NORM: f64 = 0.5;
const RPO_ALPHA_MIN: f64 = 0.1;
const RPO_ALPHA_MAX: f64 = 0.5;
const RPO_TARGET_KL: f64 = 0.018;
const VALUE_LOG_CLIP: f64 = 10.0;
const CRITIC_ENTROPY_COEF: f64 = 0.01;
const CRITIC_MAE_NORM: f64 = 100.0;
const LOG_2PI: f64 = 1.8378770664093453;
const ADV_MIXED_WEIGHT: f64 = 0.5;
const GRAD_ACCUM_STEPS: usize = 2;
pub(crate) const DEBUG_NUMERICS: bool = false;

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
    if let Some(threads) = env::var("TORCH_NUM_THREADS").ok().and_then(|v| v.parse::<i32>().ok()) {
        tch::set_num_threads(threads);
    } else {
        tch::set_num_threads(1);
    }
    if let Some(threads) = env::var("TORCH_NUM_INTEROP_THREADS").ok().and_then(|v| v.parse::<i32>().ok()) {
        tch::set_num_interop_threads(threads);
    } else {
        tch::set_num_interop_threads(1);
    }

    let device = tch::Device::cuda_if_available();
    println!("device is cuda: {}", device.is_cuda());

    let mut vs = nn::VarStore::new(device);
    let trading_model = TradingModel::new(&vs.root());
    
    if let Some(path) = weights_path {
        println!("Loading weights from {}", path);
        vs.load(path).unwrap();
    } else {
        println!("Starting training from scratch");
    }
    
    let mut opt = nn::Adam::default().build(&vs, 2e-4).unwrap();

    let mut env = VecEnv::new(true);

    let rollout_steps = SEQ_LEN;
    let memory_size = rollout_steps * NPROCS;

    let pd_dim = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
    let so_dim = STATIC_OBSERVATIONS as i64;

    let mut s_price_deltas = GpuRollingBuffer::new(memory_size, pd_dim, Kind::Float, device);
    let mut s_static_obs = GpuRollingBuffer::new(memory_size, so_dim, Kind::Float, device);
    let mut s_actions = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
    let mut s_old_log_probs = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let mut s_rewards = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
    let mut s_dones = Tensor::zeros(&[memory_size], (Kind::Float, device));
    let mut s_values = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
    let mut s_action_weights = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));

    let mut rpo_rho = vs.root().var("rpo_rho", &[1], nn::Init::Const(0.0));

    for episode in 0..1000000 {
        let (obs_price_cpu, obs_static_cpu) = env.reset();
        let mut obs_price = Tensor::zeros(&[NPROCS, pd_dim], (Kind::Float, device));
        let mut obs_static = Tensor::zeros(&[NPROCS, STATIC_OBSERVATIONS as i64], (Kind::Float, device));
        obs_price.copy_(&obs_price_cpu);
        obs_static.copy_(&obs_static_cpu);
        let mut step_reward_per_ticker =
            Tensor::zeros(&[NPROCS, TICKERS_COUNT], (Kind::Float, device));
        let mut step_cash_reward = Tensor::zeros(&[NPROCS], (Kind::Float, device));
        let mut step_is_done = Tensor::zeros(&[NPROCS], (Kind::Float, device));
        
        let action_dim = (TICKERS_COUNT + 1) as usize;
        let mut actions_flat = vec![0.0f64; NPROCS as usize * action_dim];
        let mut actions_vec = vec![vec![0.0f64; action_dim]; NPROCS as usize];

        for step in 0..rollout_steps as usize {
            let (values, _, (action_mean, action_log_std), _) = tch::no_grad(|| {
                trading_model.forward(&obs_price, &obs_static, false)
            });

            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("action_mean", &action_mean, episode as i64, step);
                let _ = debug_tensor_stats("action_log_std", &action_log_std, episode as i64, step);
            }

            let action_std = action_log_std.exp();
            let u = &action_mean + Tensor::randn_like(&action_mean) * &action_std;
            
            let u_normalized = (&u - &action_mean) / &action_std;
            let log_prob_gaussian = (u_normalized.pow_tensor_scalar(2) + action_log_std * 2.0 + LOG_2PI).sum_dim_intlist(-1, false, Kind::Float).g_mul_scalar(-0.5);
            let log_det = u.log_softmax(-1, Kind::Float).sum_dim_intlist(-1, false, Kind::Float);
            let action_log_prob = log_prob_gaussian - log_det;

            let actions_softmax = u.softmax(-1, Kind::Float);
            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("u", &u, episode as i64, step);
                let _ = debug_tensor_stats("action_log_prob", &action_log_prob, episode as i64, step);
                let _ = debug_tensor_stats("actions_softmax", &actions_softmax, episode as i64, step);
            }
            let actions_cpu = actions_softmax
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
            );

            let mem_idx = step as i64 * NPROCS;
            
            s_price_deltas.push(&obs_price);
            s_static_obs.push(&obs_static);
            let _ = s_actions.narrow(0, mem_idx, NPROCS).copy_(&u);
            let _ = s_old_log_probs.narrow(0, mem_idx, NPROCS).copy_(&action_log_prob);
            
            let rewards_combined =
                Tensor::cat(&[&step_reward_per_ticker, &step_cash_reward.unsqueeze(1)], 1);
            if DEBUG_NUMERICS {
                let _ = debug_tensor_stats("rewards_combined", &rewards_combined, episode as i64, step);
                let _ = debug_tensor_stats("values", &values, episode as i64, step);
                let _ = debug_tensor_stats("step_is_done", &step_is_done, episode as i64, step);
            }
            let _ = s_rewards.narrow(0, mem_idx, NPROCS).copy_(&rewards_combined);
            let _ = s_dones.narrow(0, mem_idx, NPROCS).copy_(&step_is_done);
            let _ = s_values.narrow(0, mem_idx, NPROCS).copy_(&values);
            let _ = s_action_weights.narrow(0, mem_idx, NPROCS).copy_(&actions_softmax);
        }

        // Compute GAE
        let advantages = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
        let returns = Tensor::zeros(&[memory_size, TICKERS_COUNT + 1], (Kind::Float, device));
        let gamma = 0.99f64;
        let gae_lambda = 0.95f64;

        tch::no_grad(|| {
            let mut last_gae = Tensor::zeros(&[NPROCS, TICKERS_COUNT + 1], (Kind::Float, device));
            for t in (0..rollout_steps).rev() {
                let mem_idx = t * NPROCS;
                let next_values = if t == rollout_steps - 1 {
                    Tensor::zeros(&[NPROCS, TICKERS_COUNT + 1], (Kind::Float, device))
                } else {
                    s_values.narrow(0, (t + 1) * NPROCS, NPROCS)
                };
                let cur_values = s_values.narrow(0, mem_idx, NPROCS);
                let rewards = s_rewards.narrow(0, mem_idx, NPROCS);
                let dones = s_dones.narrow(0, mem_idx, NPROCS).unsqueeze(1);
                
                let delta = rewards + (1.0 - &dones) * gamma * next_values - &cur_values;
                last_gae = delta + (1.0 - &dones) * gamma * gae_lambda * last_gae;
                let _ = advantages.narrow(0, mem_idx, NPROCS).copy_(&last_gae);
                let _ = returns.narrow(0, mem_idx, NPROCS).copy_(&(&last_gae + &cur_values));
            }
        });

        let adv_mean = advantages.mean_dim(0, true, Kind::Float);
        let adv_std = advantages.std_dim(0, true, false).clamp_min(1e-4);
        let advantages = ((advantages - adv_mean) / adv_std).clamp(-3.0, 3.0).detach();
        let returns = returns.detach();

        let price_deltas_batch = s_price_deltas.data.shallow_clone();
        let static_obs_batch = s_static_obs.data.shallow_clone();
        let action_weights_batch = s_action_weights.shallow_clone();

        let mut total_kl_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_policy_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_value_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_value_mae_weighted = Tensor::zeros([], (Kind::Float, device));
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
                let act_mb = s_actions.narrow(0, chunk_sample_start, chunk_sample_count);
                let ret_mb = returns.narrow(0, chunk_sample_start, chunk_sample_count);
                let adv_mb = advantages.narrow(0, chunk_sample_start, chunk_sample_count);
                let old_log_probs_mb = s_old_log_probs.narrow(0, chunk_sample_start, chunk_sample_count);
                let weight_mb = action_weights_batch.narrow(0, chunk_sample_start, chunk_sample_count);

                let fwd_start = Instant::now();
                let (values, critic_logits, (action_mean, action_log_stds), attn_entropy) =
                    trading_model.forward(&pd_chunk, &so_chunk, true);
                
                let values = values.to_kind(Kind::Float).view([chunk_sample_count, TICKERS_COUNT + 1]);
                let action_mean = action_mean.to_kind(Kind::Float);
                let action_log_stds = action_log_stds.to_kind(Kind::Float);
                
                let rpo_alpha = RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid();
                let rpo_noise = Tensor::cat(&[
                    Tensor::empty([chunk_sample_count, TICKERS_COUNT], (Kind::Float, device)).uniform_(-1.0, 1.0) * rpo_alpha.detach(),
                    Tensor::zeros([chunk_sample_count, 1], (Kind::Float, device))
                ], 1);
                let action_mean_noisy = &action_mean + &rpo_noise;

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("act_mb", &act_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("old_log_probs_mb", &old_log_probs_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_mean", &action_mean, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_log_stds", &action_log_stds, _epoch, chunk_i);
                }

                let u_normalized = (&act_mb - &action_mean_noisy) / action_log_stds.exp();
                let log_prob_gaussian = (u_normalized.pow_tensor_scalar(2) + &action_log_stds * 2.0 + LOG_2PI).sum_dim_intlist(-1, false, Kind::Float).g_mul_scalar(-0.5);
                let log_det = act_mb.log_softmax(-1, Kind::Float).sum_dim_intlist(-1, false, Kind::Float);
                let action_log_probs = log_prob_gaussian - log_det;

                let log_ratio = (&action_log_probs - &old_log_probs_mb).tanh() * 0.3;
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("action_log_probs", &action_log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_ratio", &log_ratio, _epoch, chunk_i);
                }
                let ratio = log_ratio.exp();
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_RATIO, 1.0 + PPO_CLIP_RATIO);
                
                let adv_reduced = adv_mb.mean_dim([-1].as_slice(), false, Kind::Float) + (&adv_mb * &weight_mb).sum_dim_intlist(-1, false, Kind::Float) * ADV_MIXED_WEIGHT;
                let action_loss = -Tensor::min_other(&(&ratio * &adv_reduced), &(&ratio_clipped * &adv_reduced)).mean(Kind::Float);

                let returns_symlog = symlog_tensor(&ret_mb.clamp(-VALUE_LOG_CLIP, VALUE_LOG_CLIP));
                let target_twohot = twohot_encode(&returns_symlog, trading_model.value_centers()).view([chunk_sample_count, TICKERS_COUNT + 1, -1]);
                let log_probs = critic_logits.to_kind(Kind::Float).log_softmax(-1, Kind::Float);
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("returns_symlog", &returns_symlog, _epoch, chunk_i);
                    let _ = debug_tensor_stats("target_twohot", &target_twohot, _epoch, chunk_i);
                    let _ = debug_tensor_stats("critic_logits", &critic_logits, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_probs", &log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("adv_mb", &adv_mb, _epoch, chunk_i);
                    let _ = debug_tensor_stats("ret_mb", &ret_mb, _epoch, chunk_i);
                }
                let ce_loss = -(target_twohot * &log_probs).sum_dim_intlist(-1, false, Kind::Float).mean(Kind::Float);
                let critic_entropy = -(log_probs.exp() * &log_probs).sum_dim_intlist(-1, false, Kind::Float).mean(Kind::Float);
                let value_loss = ce_loss - CRITIC_ENTROPY_COEF * critic_entropy;

                let dist_entropy = (&action_log_stds * 2.0 + (1.0 + LOG_2PI)).g_mul_scalar(0.5).mean(Kind::Float);
                
                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF + action_loss.shallow_clone() - dist_entropy * ENTROPY_COEF - attn_entropy.mean(Kind::Float) * ATTENTION_ENTROPY_COEF;
                
                let inv_var_mean = action_log_stds.detach().exp().pow_tensor_scalar(2).clamp_min(1e-4).reciprocal().mean(Kind::Float);
                let induced_kl = rpo_alpha.pow_tensor_scalar(2) * (ACTION_COUNT as f64 / 6.0) * inv_var_mean;
                let alpha_loss = (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0);

                let total_chunk_loss = (ppo_loss.shallow_clone() + alpha_loss * ALPHA_LOSS_COEF) / GRAD_ACCUM_STEPS as f64;
                
                fwd_time_us += fwd_start.elapsed().as_micros() as u64;
                let bwd_start = Instant::now();
                total_chunk_loss.backward();
                bwd_time_us += bwd_start.elapsed().as_micros() as u64;

                let approx_kl_val = tch::no_grad(|| (log_ratio.exp() - 1.0 - &log_ratio).mean(Kind::Float));
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("approx_kl_val", &approx_kl_val, _epoch, chunk_i);
                }
                let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                let _ = total_loss_weighted.g_add_(&(&ppo_loss * chunk_sample_count as f64));
                let _ = total_policy_loss_weighted.g_add_(&(&action_loss * chunk_sample_count as f64));
                let _ = total_value_loss_weighted.g_add_(&(&value_loss * chunk_sample_count as f64));
                let _ = total_value_mae_weighted.g_add_(&((&values - &ret_mb).abs().mean(Kind::Float) * chunk_sample_count as f64));
                let _ = total_kl_weighted.g_add_(&(&approx_kl_val * chunk_sample_count as f64));
                epoch_kl_count += chunk_sample_count;
                total_sample_count += chunk_sample_count;

                if (chunk_i + 1) % GRAD_ACCUM_STEPS == 0 || chunk_i == chunk_order.len() - 1 {
                    if DEBUG_NUMERICS {
                        let has_nan_grad = tch::no_grad(|| {
                            let mut found = false;
                            for v in opt.trainable_variables() {
                                let g = v.grad();
                                if g.defined() && (g.isnan().any().int64_value(&[]) != 0 || g.isinf().any().int64_value(&[]) != 0) {
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
                            if g.defined() { norm_sq += g.pow_tensor_scalar(2).sum(Kind::Float); }
                        }
                        norm_sq.sqrt()
                    });
                    grad_norm_sum += &batch_grad_norm;
                    grad_norm_count += 1;

                    opt.clip_grad_norm(MAX_GRAD_NORM);
                    opt.step();
                    opt.zero_grad();
                }
                
                let _ = total_clipped.g_add_(&tch::no_grad(|| (&ratio - 1.0).abs().gt(PPO_CLIP_RATIO).to_kind(Kind::Float).sum(Kind::Float)));
                total_ratio_samples += chunk_sample_count;
            }
            
            let mean_epoch_kl = epoch_kl_gpu.double_value(&[]) / epoch_kl_count as f64;
            println!("Epoch {}/{}: KL {:.4}", _epoch + 1, OPTIM_EPOCHS, mean_epoch_kl);
            if mean_epoch_kl > TARGET_KL * KL_STOP_MULTIPLIER { break 'epoch_loop; }
        }

        println!("fwd: {:.1}ms  bwd: {:.1}ms", fwd_time_us as f64 / 1000.0, bwd_time_us as f64 / 1000.0);
        
        let max_param_norm = tch::no_grad(|| {
            let mut max_norm = 0.0f64;
            for v in opt.trainable_variables() {
                let norm = v.norm().double_value(&[]);
                if norm > max_norm { max_norm = norm; }
            }
            max_norm
        });
        if max_param_norm > 1000.0 { println!("WARNING: Large parameter norm detected: {:.2}", max_param_norm); }

        let mean_losses = if total_sample_count > 0 {
            Tensor::stack(&[&total_loss_weighted, &total_policy_loss_weighted, &total_value_loss_weighted, &total_value_mae_weighted], 0) / (total_sample_count as f64)
        } else { Tensor::zeros([4], (Kind::Float, device)) };
        let mean_grad_norm = if grad_norm_count > 0 { (&grad_norm_sum / (grad_norm_count as f64)).double_value(&[]) } else { 0.0 };
        let mean_losses_vec: Vec<f64> = Vec::try_from(mean_losses.to_device(tch::Device::Cpu)).unwrap();
        let (mean_loss, mean_policy_loss, mean_value_loss, mean_value_mae) = (
            mean_losses_vec[0], mean_losses_vec[1], mean_losses_vec[2], mean_losses_vec[3]
        );

        let logit_stats = tch::no_grad(|| {
            let (_, _, (_, action_log_std), _) = trading_model.forward(&s_price_deltas.get(0), &s_static_obs.get(0), false);
            let action_std = action_log_std.exp();
            let rpo_alpha = (RPO_ALPHA_MIN + (RPO_ALPHA_MAX - RPO_ALPHA_MIN) * rpo_rho.sigmoid()).squeeze();
            Tensor::stack(&[action_std.mean(Kind::Float), action_std.min(), action_std.max(), rpo_alpha], 0)
        });
        let stats_vec: Vec<f64> = Vec::try_from(logit_stats.to_device(tch::Device::Cpu)).unwrap_or_else(|_| vec![0.0; 4]);
        let clip_frac = if total_ratio_samples > 0 { total_clipped.double_value(&[]) / total_ratio_samples as f64 } else { 0.0 };

        let primary = env.primary_mut();
        primary.meta_history.record_logit_noise_stats(stats_vec[0], stats_vec[1], stats_vec[2], stats_vec[3]);
        primary.meta_history.record_clip_fraction(clip_frac);
        primary.meta_history.record_loss(mean_loss);
        primary.meta_history.record_policy_loss(mean_policy_loss);
        primary.meta_history.record_value_loss(mean_value_loss);
        primary.meta_history.record_value_mae(mean_value_mae / CRITIC_MAE_NORM);
        primary.meta_history.record_grad_norm(mean_grad_norm);

        println!("  Loss: {:.4} (Policy: {:.4}, Value: {:.4}, MAE: {:.4}) GradNorm: {:.4}", 
                 mean_loss, mean_policy_loss, mean_value_loss, mean_value_mae / CRITIC_MAE_NORM, mean_grad_norm);
    }
}
