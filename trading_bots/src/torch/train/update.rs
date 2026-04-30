use rand::seq::SliceRandom;
use std::time::Instant;
use tch::{autocast, Kind, Tensor};

use crate::torch::action_space::{
    beta_entropy, beta_kl_old_new, beta_log_prob, beta_log_prob_per_dim, beta_variance,
};
use crate::torch::constants::{ACTION_COUNT, TICKERS_COUNT};
use crate::torch::model::TradingModel;

use super::config::{
    PolicyObjective, ALPHA_LOSS_COEF, DEBUG_NUMERICS, ENTROPY_COEF, KL_STOP_MULTIPLIER,
    MAX_GRAD_NORM, OPTIM_EPOCHS, PMPO_POS_TO_NEG_WEIGHT, PMPO_REVERSE_KL_COEF, PPO_CLIP_HIGH,
    PPO_CLIP_LOW, RPO_ALPHA_MAX, RPO_ALPHA_MIN, RPO_TARGET_KL, TARGET_KL, VALUE_LOSS_COEF,
};
use super::cuda_graph_minibatch::PmpoMinibatchCudaGraph;
use super::gae::build_no_reset_windowed_layouts;
use super::numeric_debug::{
    debug_tensor_stats, log_first_non_finite_tensor, log_first_non_finite_var,
    log_named_var_extremes,
};
use super::optimizer_glue::{clip_grad_norm_on_device, step_optimizer};
use super::trainer::{AdvantageData, Trainer, UpdateMetrics};
use super::value_loss::hl_gauss_value_loss;

impl Trainer {
    pub(super) fn update_policy(
        &mut self,
        episode: usize,
        adv_data: &AdvantageData,
    ) -> UpdateMetrics {
        let device = self.device;
        let mut total_policy_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_value_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_reverse_kl_weighted = Tensor::zeros([], (Kind::Float, device));
        // Explained variance: EV = 1 - Var(residuals) / Var(targets)
        let mut grad_norm_sum = Tensor::zeros([], (Kind::Float, device));
        let mut total_sample_count = 0i64;
        let mut grad_norm_count = 0i64;
        let mut total_clipped = Tensor::zeros([], (Kind::Float, device));
        let mut total_ratio_samples = 0i64;
        let mut total_entropy_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut entropy_min = Tensor::from(f64::INFINITY)
            .to_kind(Kind::Float)
            .to_device(device);
        let mut entropy_max = Tensor::from(f64::NEG_INFINITY)
            .to_kind(Kind::Float)
            .to_device(device);

        let mut fwd_time_us = 0u64;
        let mut bwd_time_us = 0u64;
        let mut graph_update_time_us = 0u64;
        let mut logged_replay_input_non_finite = false;
        let mut logged_forward_non_finite = false;
        let mut logged_loss_non_finite = false;
        let mut logged_grad_non_finite = false;
        let mut logged_param_non_finite = false;
        let mut logged_forward_probe = false;

        let mut last_minibatch_approx_kl = 0.0f64;
        let mut perm_host: Vec<i64> = (0..self.total_chunks).collect();
        let mut perm_gpu = Tensor::zeros([self.total_chunks], (Kind::Int64, device));
        let mut rng = rand::rng();

        'epoch_loop: for _epoch in 0..OPTIM_EPOCHS {
            perm_host.shuffle(&mut rng);
            let perm_cpu = Tensor::from_slice(&perm_host)
                .to_kind(Kind::Int64)
                .to_device(device);
            perm_gpu.copy_(&perm_cpu);

            let mut epoch_kl_gpu = Tensor::zeros([], (Kind::Float, device));
            let mut epoch_kl_count = 0i64;
            // Track last minibatch's KL mean on-device; fetch once at end of epoch
            // to avoid a host/device sync on every minibatch.
            let mut last_minibatch_kl_mean_gpu: Option<Tensor> = None;

            for (chunk_i, mb_start) in (0..self.total_chunks)
                .step_by(adv_data.chunk_batch_size as usize)
                .enumerate()
            {
                let mb_end = (mb_start + adv_data.chunk_batch_size).min(self.total_chunks);
                let chunk_count = mb_end - mb_start;
                let chunk_ids = perm_gpu.narrow(0, mb_start, chunk_count);
                let boundary_layout = self.s_chunk_start_layouts.index_select(0, &chunk_ids);
                let so_chunk = self.s_static_obs.index_select(0, &chunk_ids);
                let step_deltas_chunk = self.s_step_deltas.index_select(0, &chunk_ids);
                let adv_mb_by_chunk = match self.policy_objective {
                    PolicyObjective::Ppo => adv_data.adv_norm.index_select(0, &chunk_ids),
                    PolicyObjective::Pmpo => adv_data.advantages.index_select(0, &chunk_ids),
                };
                let ret_mb_by_chunk = adv_data.returns.index_select(0, &chunk_ids);
                let old_log_probs_by_chunk = self.s_old_log_probs.index_select(0, &chunk_ids);
                let act_mb_by_chunk = self.s_actions.index_select(0, &chunk_ids);
                let old_action_alpha_by_chunk =
                    self.s_old_action_alpha.index_select(0, &chunk_ids);
                let old_action_beta_by_chunk = self.s_old_action_beta.index_select(0, &chunk_ids);
                let reset_slots_chunk = adv_data.reset_slots_by_chunk.index_select(0, &chunk_ids);

                let fwd_start = Instant::now();
                let minibatch_sample_count = chunk_count * self.rollout.ppo_chunk_len;

                // Full-chunk batched windowed forward: build all ppo_chunk_len
                // windowed layouts at once and fire a single batched forward with
                // effective batch = chunk_count * ppo_chunk_len. No sub-chunk
                // gradient accumulation needed — one forward, one backward per
                // minibatch. Each window is its own 255-token causal prefix +
                // live-token suffix, so streaming semantics are preserved per window.
                let flat_layout_len = boundary_layout.size()[1] / TICKERS_COUNT;
                let has_reset_slots = adv_data.reset_layout_count > 0
                    && reset_slots_chunk.max().int64_value(&[]) > 0;
                let windowed = if has_reset_slots {
                    let layout_rows = chunk_count * TICKERS_COUNT;
                    let mut current_layout =
                        boundary_layout.view([layout_rows, flat_layout_len]);
                    let mut windowed_rows: Vec<Tensor> =
                        Vec::with_capacity(self.rollout.ppo_chunk_len as usize);
                    for t in 0..self.rollout.ppo_chunk_len {
                        if t == 0 {
                            // Window 0: boundary layout unchanged (mirrors the `is_full`
                            // init path in step_on_device_for_replay).
                            windowed_rows.push(current_layout.shallow_clone());
                        } else {
                            let prev_step_deltas = step_deltas_chunk.select(1, t - 1); // [chunk_count, TICKERS]
                            let row_deltas = prev_step_deltas.reshape([layout_rows, 1]);
                            current_layout = self
                                .trading_model
                                .shift_layout_append_delta(&current_layout, &row_deltas);
                            // Reset after shift-append to preserve bank layouts verbatim.
                            let step_reset_slots = reset_slots_chunk.select(1, t - 1); // [chunk_count]
                            let reset_chunk_idx =
                                step_reset_slots.gt(0).nonzero().squeeze_dim(1);
                            if reset_chunk_idx.size()[0] > 0 {
                                let reset_slot_ids =
                                    step_reset_slots.index_select(0, &reset_chunk_idx) - 1;
                                let reset_slot_ids_cpu =
                                    reset_slot_ids.to_device(tch::Device::Cpu);
                                let reset_layouts = adv_data
                                    .reset_layout_bank_cpu
                                    .index_select(0, &reset_slot_ids_cpu)
                                    .to_device(device);
                                let reset_row_idx = (&reset_chunk_idx.unsqueeze(1) * TICKERS_COUNT
                                    + &self.ticker_offsets)
                                    .reshape([-1]);
                                current_layout = current_layout.index_copy(
                                    0,
                                    &reset_row_idx,
                                    &reset_layouts.view([-1, flat_layout_len]),
                                );
                            }
                            windowed_rows.push(current_layout.shallow_clone());
                        }
                    }
                    Tensor::stack(&windowed_rows, 0)
                        .view([
                            self.rollout.ppo_chunk_len,
                            chunk_count,
                            TICKERS_COUNT,
                            flat_layout_len,
                        ])
                        .permute([1, 0, 2, 3])
                        .contiguous()
                        .view([
                            chunk_count * self.rollout.ppo_chunk_len * TICKERS_COUNT,
                            flat_layout_len,
                        ])
                } else {
                    build_no_reset_windowed_layouts(
                        &boundary_layout,
                        &step_deltas_chunk,
                        chunk_count,
                        self.rollout.ppo_chunk_len,
                        flat_layout_len,
                    )
                };
                let static_flat = so_chunk.reshape([minibatch_sample_count, self.so_dim]);

                // Flatten rollout-captured targets to minibatch-flat form (chunk-major).
                let adv_flat = adv_mb_by_chunk.reshape([-1]);
                let ret_flat = ret_mb_by_chunk.reshape([-1]);
                let old_log_probs_flat = old_log_probs_by_chunk.reshape([-1]);
                let act_flat = act_mb_by_chunk.reshape([-1, ACTION_COUNT]);
                let old_action_alpha_flat =
                    old_action_alpha_by_chunk.reshape([-1, ACTION_COUNT]);
                let old_action_beta_flat = old_action_beta_by_chunk.reshape([-1, ACTION_COUNT]);

                if log_first_non_finite_tensor(
                    &mut logged_replay_input_non_finite,
                    "replay_inputs",
                    episode,
                    _epoch,
                    chunk_i,
                    &[
                        ("windowed", &windowed),
                        ("static_flat", &static_flat),
                        ("actions", &act_flat),
                        ("old_log_probs", &old_log_probs_flat),
                        ("old_action_alpha", &old_action_alpha_flat),
                        ("old_action_beta", &old_action_beta_flat),
                        ("advantages", &adv_flat),
                        ("returns", &ret_flat),
                    ],
                ) {
                    log_named_var_extremes(
                        "params_at_replay_input_failure",
                        episode,
                        _epoch,
                        chunk_i,
                        &self.named_trainable_vars,
                        false,
                        12,
                    );
                }

                if self.use_cuda_graph_updates && self.policy_objective == PolicyObjective::Pmpo {
                    if self.pmpo_cuda_graph.is_none() {
                        self.pmpo_cuda_graph = Some(PmpoMinibatchCudaGraph::new(
                            device,
                            &windowed,
                            &static_flat,
                            &act_flat,
                            &old_log_probs_flat,
                            &old_action_alpha_flat,
                            &old_action_beta_flat,
                            &adv_flat,
                            &ret_flat,
                            minibatch_sample_count,
                        ));
                    }
                    let graph_matches = self.pmpo_cuda_graph.as_ref().is_some_and(|graph| {
                        graph.matches(
                            &windowed,
                            &static_flat,
                            &act_flat,
                            &old_log_probs_flat,
                            &old_action_alpha_flat,
                            &old_action_beta_flat,
                            &adv_flat,
                            &ret_flat,
                            minibatch_sample_count,
                        )
                    });
                    if graph_matches {
                        let graph_start = Instant::now();
                        let metrics = self.pmpo_cuda_graph.as_mut().unwrap().capture_or_replay(
                            &self.trading_model,
                            &self.hl_gauss,
                            &self.trainable_vars,
                            device,
                            &windowed,
                            &static_flat,
                            &act_flat,
                            &old_log_probs_flat,
                            &old_action_alpha_flat,
                            &old_action_beta_flat,
                            &adv_flat,
                            &ret_flat,
                        );
                        graph_update_time_us += graph_start.elapsed().as_micros() as u64;

                        if log_first_non_finite_var(
                            &mut logged_grad_non_finite,
                            "grads_after_backward",
                            episode,
                            _epoch,
                            chunk_i,
                            &self.named_trainable_vars,
                            true,
                        ) {
                            log_named_var_extremes(
                                "grads_after_backward_top_abs",
                                episode,
                                _epoch,
                                chunk_i,
                                &self.named_trainable_vars,
                                true,
                                12,
                            );
                        }

                        let approx_kl_val = metrics.approx_kl.shallow_clone();
                        let dist_entropy_detached = metrics.dist_entropy.detach();
                        let _ = epoch_kl_gpu
                            .g_add_(&(&approx_kl_val * minibatch_sample_count as f64));
                        let _ = total_policy_loss_weighted.g_add_(
                            &(&metrics.action_loss.detach() * minibatch_sample_count as f64),
                        );
                        let _ = total_value_loss_weighted.g_add_(
                            &(&metrics.value_loss.detach() * minibatch_sample_count as f64),
                        );
                        let _ = total_reverse_kl_weighted.g_add_(
                            &(&metrics.reverse_kl_loss.detach()
                                * minibatch_sample_count as f64),
                        );
                        let _ = total_entropy_weighted
                            .g_add_(&(&dist_entropy_detached * minibatch_sample_count as f64));
                        entropy_min = entropy_min.min_other(&dist_entropy_detached);
                        entropy_max = entropy_max.max_other(&dist_entropy_detached);
                        epoch_kl_count += minibatch_sample_count;
                        total_sample_count += minibatch_sample_count;
                        let _ = total_clipped.g_add_(&metrics.clipped);
                        total_ratio_samples += minibatch_sample_count;

                        let batch_grad_norm =
                            clip_grad_norm_on_device(&self.trainable_vars, MAX_GRAD_NORM, device);
                        grad_norm_sum += &batch_grad_norm;
                        grad_norm_count += 1;

                        step_optimizer(&mut self.opt, &mut self.optimizer_step);
                        if log_first_non_finite_var(
                            &mut logged_param_non_finite,
                            "params_after_step",
                            episode,
                            _epoch,
                            chunk_i,
                            &self.named_trainable_vars,
                            false,
                        ) {
                            log_named_var_extremes(
                                "params_after_step_top_abs",
                                episode,
                                _epoch,
                                chunk_i,
                                &self.named_trainable_vars,
                                false,
                                12,
                            );
                        }
                        self.rpo_rho_step();
                        self.opt.zero_grad();
                        last_minibatch_kl_mean_gpu = Some(approx_kl_val.shallow_clone());
                        continue;
                    }
                }

                let (new_value_logits, action_alpha, action_beta, action_std) =
                    autocast(false, || {
                        self.trading_model.windowed_replay_forward(
                            &windowed,
                            &static_flat,
                            minibatch_sample_count,
                        )
                    });

                let action_log_probs = beta_log_prob(&act_flat, &action_alpha, &action_beta);
                let dist_entropy_per_sample = beta_entropy(&action_alpha, &action_beta);
                let action_var = beta_variance(&action_alpha, &action_beta);
                let action_log_probs_per_dim =
                    if self.policy_objective == PolicyObjective::Pmpo {
                        beta_log_prob_per_dim(&act_flat, &action_alpha, &action_beta)
                    } else {
                        Tensor::zeros([0], (Kind::Float, device))
                    };

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("actions_mb", &act_flat, _epoch, chunk_i);
                    let _ = debug_tensor_stats(
                        "old_log_probs_mb",
                        &old_log_probs_flat,
                        _epoch,
                        chunk_i,
                    );
                    let _ = debug_tensor_stats("action_alpha", &action_alpha, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_std", &action_std, _epoch, chunk_i);
                }

                let log_ratio = &action_log_probs - &old_log_probs_flat;

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats(
                        "action_log_probs",
                        &action_log_probs,
                        _epoch,
                        chunk_i,
                    );
                    let _ = debug_tensor_stats("log_ratio", &log_ratio, _epoch, chunk_i);
                }
                let ratio = log_ratio.exp();
                let ratio_clipped = ratio.clamp(1.0 - PPO_CLIP_LOW, 1.0 + PPO_CLIP_HIGH);

                if log_first_non_finite_tensor(
                    &mut logged_forward_non_finite,
                    "forward",
                    episode,
                    _epoch,
                    chunk_i,
                    &[
                        ("action_alpha", &action_alpha),
                        ("action_std", &action_std),
                        ("action_beta", &action_beta),
                        ("action_log_probs", &action_log_probs),
                        ("old_log_probs", &old_log_probs_flat),
                        ("old_action_alpha", &old_action_alpha_flat),
                        ("old_action_beta", &old_action_beta_flat),
                        ("log_ratio", &log_ratio),
                        ("ratio", &ratio),
                        ("new_value_logits", &new_value_logits),
                        ("adv_flat", &adv_flat),
                        ("ret_flat", &ret_flat),
                    ],
                ) {
                    log_named_var_extremes(
                        "params_at_forward_failure",
                        episode,
                        _epoch,
                        chunk_i,
                        &self.named_trainable_vars,
                        false,
                        12,
                    );
                    if !logged_forward_probe {
                        logged_forward_probe = true;
                        TradingModel::set_replay_numeric_probe(true);
                        tch::no_grad(|| {
                            let _ = autocast(false, || {
                                self.trading_model.windowed_replay_forward(
                                    &windowed,
                                    &static_flat,
                                    minibatch_sample_count,
                                )
                            });
                        });
                        TradingModel::set_replay_numeric_probe(false);
                    }
                }

                let reverse_kl_loss = match self.policy_objective {
                    PolicyObjective::Ppo => Tensor::zeros([], (Kind::Float, device)),
                    PolicyObjective::Pmpo => beta_kl_old_new(
                        &old_action_alpha_flat,
                        &old_action_beta_flat,
                        &action_alpha,
                        &action_beta,
                    )
                    .mean(Kind::Float),
                };

                let action_loss = match self.policy_objective {
                    PolicyObjective::Ppo => {
                        -Tensor::min_other(&(&ratio * &adv_flat), &(&ratio_clipped * &adv_flat))
                            .mean(Kind::Float)
                    }
                    PolicyObjective::Pmpo => {
                        let adv_weight = adv_flat.tanh().abs().unsqueeze(-1);
                        let signed_log_probs = &action_log_probs_per_dim * adv_weight;
                        let pos_mask = adv_flat
                            .ge(0.0)
                            .to_kind(Kind::Float)
                            .unsqueeze(-1)
                            .expand_as(&signed_log_probs);
                        let neg_mask = adv_flat
                            .lt(0.0)
                            .to_kind(Kind::Float)
                            .unsqueeze(-1)
                            .expand_as(&signed_log_probs);
                        let pos_count = pos_mask.sum(Kind::Float).clamp_min(1.0);
                        let neg_count = neg_mask.sum(Kind::Float).clamp_min(1.0);
                        let pos_loss =
                            (&signed_log_probs * &pos_mask).sum(Kind::Float) / pos_count;
                        let neg_loss =
                            (&signed_log_probs * &neg_mask).sum(Kind::Float) / neg_count;
                        -PMPO_POS_TO_NEG_WEIGHT * pos_loss
                            + (1.0 - PMPO_POS_TO_NEG_WEIGHT) * neg_loss
                            + &reverse_kl_loss * PMPO_REVERSE_KL_COEF
                    }
                };

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("ret_mb", &ret_flat, _epoch, chunk_i);
                    let _ = debug_tensor_stats(
                        "new_value_logits",
                        &new_value_logits,
                        _epoch,
                        chunk_i,
                    );
                    let _ = debug_tensor_stats("adv_mb", &adv_flat, _epoch, chunk_i);
                }

                let value_loss = hl_gauss_value_loss(&self.hl_gauss, &new_value_logits, &ret_flat)
                    .mean(Kind::Float);

                let dist_entropy = dist_entropy_per_sample.mean(Kind::Float);
                let dist_entropy_detached = dist_entropy.detach();

                let ppo_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF
                    + action_loss.shallow_clone()
                    - &dist_entropy * ENTROPY_COEF;

                let alpha_loss = if self.policy_objective == PolicyObjective::Ppo
                    && RPO_ALPHA_MAX > RPO_ALPHA_MIN
                {
                    let rpo_alpha =
                        Tensor::zeros(&[1], (Kind::Float, device)).set_requires_grad(true);
                    let inv_var_mean = action_var
                        .detach()
                        .clamp_min(1e-4)
                        .reciprocal()
                        .mean(Kind::Float);
                    let d = ACTION_COUNT as f64;
                    let induced_kl = rpo_alpha.pow_tensor_scalar(2) * (d / 6.0) * inv_var_mean;
                    (induced_kl - RPO_TARGET_KL).pow_tensor_scalar(2.0) * ALPHA_LOSS_COEF
                } else {
                    Tensor::zeros([], (Kind::Float, device))
                };

                let total_loss = ppo_loss.shallow_clone() + alpha_loss.shallow_clone();

                if log_first_non_finite_tensor(
                    &mut logged_loss_non_finite,
                    "loss",
                    episode,
                    _epoch,
                    chunk_i,
                    &[
                        ("action_loss", &action_loss),
                        ("value_loss", &value_loss),
                        ("reverse_kl_loss", &reverse_kl_loss),
                        ("dist_entropy", &dist_entropy),
                        ("ppo_loss", &ppo_loss),
                        ("alpha_loss", &alpha_loss),
                        ("total_loss", &total_loss),
                    ],
                ) {
                    log_named_var_extremes(
                        "params_at_loss_failure",
                        episode,
                        _epoch,
                        chunk_i,
                        &self.named_trainable_vars,
                        false,
                        12,
                    );
                }

                fwd_time_us += fwd_start.elapsed().as_micros() as u64;
                let bwd_start = Instant::now();
                total_loss.backward();
                bwd_time_us += bwd_start.elapsed().as_micros() as u64;

                if log_first_non_finite_var(
                    &mut logged_grad_non_finite,
                    "grads_after_backward",
                    episode,
                    _epoch,
                    chunk_i,
                    &self.named_trainable_vars,
                    true,
                ) {
                    log_named_var_extremes(
                        "grads_after_backward_top_abs",
                        episode,
                        _epoch,
                        chunk_i,
                        &self.named_trainable_vars,
                        true,
                        12,
                    );
                }

                let approx_kl_val =
                    tch::no_grad(|| (log_ratio.exp() - 1.0 - &log_ratio).mean(Kind::Float));
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("approx_kl_val", &approx_kl_val, _epoch, chunk_i);
                }
                let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * minibatch_sample_count as f64));
                let _ = total_policy_loss_weighted
                    .g_add_(&(&action_loss.detach() * minibatch_sample_count as f64));
                let _ = total_value_loss_weighted
                    .g_add_(&(&value_loss.detach() * minibatch_sample_count as f64));
                let _ = total_reverse_kl_weighted
                    .g_add_(&(&reverse_kl_loss.detach() * minibatch_sample_count as f64));
                let _ = total_entropy_weighted
                    .g_add_(&(&dist_entropy_detached * minibatch_sample_count as f64));
                entropy_min = entropy_min.min_other(&dist_entropy_detached);
                entropy_max = entropy_max.max_other(&dist_entropy_detached);
                epoch_kl_count += minibatch_sample_count;
                total_sample_count += minibatch_sample_count;

                let _ = total_clipped.g_add_(&tch::no_grad(|| {
                    let dev = &ratio - 1.0;
                    let clipped_lo = dev.le(-PPO_CLIP_LOW).to_kind(Kind::Float);
                    let clipped_hi = dev.ge(PPO_CLIP_HIGH).to_kind(Kind::Float);
                    (clipped_lo + clipped_hi).sum(Kind::Float)
                }));
                total_ratio_samples += minibatch_sample_count;

                if DEBUG_NUMERICS {
                    let has_nan_grad = tch::no_grad(|| {
                        let mut found = false;
                        for v in &self.trainable_vars {
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

                let batch_grad_norm =
                    clip_grad_norm_on_device(&self.trainable_vars, MAX_GRAD_NORM, device);
                grad_norm_sum += &batch_grad_norm;
                grad_norm_count += 1;

                step_optimizer(&mut self.opt, &mut self.optimizer_step);
                if log_first_non_finite_var(
                    &mut logged_param_non_finite,
                    "params_after_step",
                    episode,
                    _epoch,
                    chunk_i,
                    &self.named_trainable_vars,
                    false,
                ) {
                    log_named_var_extremes(
                        "params_after_step_top_abs",
                        episode,
                        _epoch,
                        chunk_i,
                        &self.named_trainable_vars,
                        false,
                        12,
                    );
                }
                self.rpo_rho_step();
                self.opt.zero_grad();

                // One forward/backward per minibatch now: the minibatch's KL is
                // exactly approx_kl_val. Track the last one for end-of-epoch early stop.
                last_minibatch_kl_mean_gpu = Some(approx_kl_val.shallow_clone());
            }

            // Single end-of-epoch host sync covering both the epoch-mean KL and
            // the last-minibatch KL used for early stopping. Avoids per-minibatch
            // D2H stalls that previously blocked the training pipeline.
            let mean_epoch_kl = if let Some(last_mb) = last_minibatch_kl_mean_gpu {
                let stacked = Tensor::stack(
                    &[&(&epoch_kl_gpu / epoch_kl_count.max(1) as f64), &last_mb],
                    0,
                )
                .to_kind(Kind::Double)
                .to_device(tch::Device::Cpu);
                let vec = Vec::<f64>::try_from(stacked).unwrap_or_else(|_| vec![0.0, 0.0]);
                // Preserve prior-epoch value if this epoch somehow had zero minibatches.
                last_minibatch_approx_kl = vec[1];
                vec[0]
            } else {
                // Epoch had no minibatches; keep prior `last_minibatch_approx_kl`.
                0.0
            };
            println!(
                "Epoch {}/{}: RatioKL {:.4} (last mb {:.4})",
                _epoch + 1,
                OPTIM_EPOCHS,
                mean_epoch_kl,
                last_minibatch_approx_kl
            );
            if self.policy_objective == PolicyObjective::Ppo
                && (mean_epoch_kl > TARGET_KL * KL_STOP_MULTIPLIER
                    || last_minibatch_approx_kl > TARGET_KL * KL_STOP_MULTIPLIER)
            {
                break 'epoch_loop;
            }
        }

        if graph_update_time_us > 0 {
            println!(
                "fwd: {:.1}ms  bwd: {:.1}ms  graph_update: {:.1}ms",
                fwd_time_us as f64 / 1000.0,
                bwd_time_us as f64 / 1000.0,
                graph_update_time_us as f64 / 1000.0
            );
        } else {
            println!(
                "fwd: {:.1}ms  bwd: {:.1}ms",
                fwd_time_us as f64 / 1000.0,
                bwd_time_us as f64 / 1000.0
            );
        }

        UpdateMetrics {
            total_policy_loss_weighted,
            total_value_loss_weighted,
            total_reverse_kl_weighted,
            grad_norm_sum,
            total_sample_count,
            grad_norm_count,
            total_clipped,
            total_ratio_samples,
            total_entropy_weighted,
            entropy_min,
            entropy_max,
            last_minibatch_approx_kl,
        }
    }
}
