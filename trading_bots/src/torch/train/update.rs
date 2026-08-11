use rand::seq::SliceRandom;
use std::{env, time::Instant};
use tch::{autocast, Kind, Tensor};

use crate::torch::action_space::{beta_entropy, beta_log_prob, beta_reverse_kl};
use crate::torch::constants::{ACTION_COUNT, TICKERS_COUNT};
use crate::torch::cuda::graph::CudaGraph;
use crate::torch::model::TradingModel;
use crate::torch::value::hl_gauss::HlGaussBins;

use super::config::{
    PolicyObjective, CLIP_EPS_HIGH, CLIP_EPS_LOW, CRITIC_PRETRAIN_EPISODES, DEBUG_NUMERICS,
    ENTROPY_COEF, KL_STOP_MULTIPLIER, MAX_GRAD_NORM, OPTIM_EPOCHS, PMPO_KL_COEF,
    PMPO_POS_TO_NEG_WEIGHT, POLICY_OBJECTIVE, RET_PERC_FLOOR, RET_PERC_HI, RET_PERC_LO, TARGET_KL,
    VALUE_LOSS_COEF,
};
use super::gae::build_no_reset_windowed_layouts;
use super::numeric_debug::{
    debug_tensor_stats, log_first_non_finite_tensor, log_first_non_finite_var,
    log_named_var_extremes,
};
use super::optimizer_glue::GradClipGroups;
use super::optimizer_glue::{backward_actor_critic_with_separate_clips, step_optimizer};
use super::trainer::{AdvantageData, Trainer, UpdateMetrics};
use super::value_loss::hl_gauss_value_loss;

/// DAPO clip-higher asymmetric PPO objective.
/// Returns the policy loss and the per-sample clip gap |ratio - clamp(ratio)|,
/// a detached diagnostic of how much the clip is engaging.
pub(crate) fn asym_clip_policy_loss(advantage: &Tensor, ratio: &Tensor) -> (Tensor, Tensor) {
    let pg_loss1 = -(advantage * ratio);
    let clipped = ratio.clamp(1.0 - CLIP_EPS_LOW, 1.0 + CLIP_EPS_HIGH);
    let pg_loss2 = -(advantage * &clipped);
    let action_loss = pg_loss1.max_other(&pg_loss2).mean(Kind::Float);
    let clip_gap = tch::no_grad(|| (ratio - &clipped).abs());
    (action_loss, clip_gap)
}

/// PMPO objective: sign-based advantage-weighted MLE + closed-form reverse-KL
/// trust region. Uses RAW GAE `advantage` and the NEW-policy `log_probs`
/// directly (no ratio); old α/β feed only the KL term. Returns the scalar
/// policy loss; entropy is applied by the caller as for PPO.
fn pmpo_policy_loss(
    advantage: &Tensor,
    log_probs: &Tensor,
    old_alpha: &Tensor,
    old_beta: &Tensor,
    new_alpha: &Tensor,
    new_beta: &Tensor,
) -> Tensor {
    // adv_weight is constant across action dims, so adv_weight * beta_log_prob
    // equals sum_dims(logp_dim * adv_weight) — the summed log-prob is exact.
    let adv_weight = advantage.tanh().abs();
    let weighted = &adv_weight * log_probs;
    let pos_mask = advantage.ge(0.0).to_kind(Kind::Float);
    let neg_mask = 1.0 - &pos_mask;
    let pos_loss =
        (&weighted * &pos_mask).sum(Kind::Float) / pos_mask.sum(Kind::Float).clamp_min(1.0);
    let neg_loss =
        (&weighted * &neg_mask).sum(Kind::Float) / neg_mask.sum(Kind::Float).clamp_min(1.0);
    let pg_loss = -PMPO_POS_TO_NEG_WEIGHT * pos_loss + (1.0 - PMPO_POS_TO_NEG_WEIGHT) * neg_loss;
    let reverse_kl = beta_reverse_kl(old_alpha, old_beta, new_alpha, new_beta).mean(Kind::Float);
    pg_loss + PMPO_KL_COEF * reverse_kl
}

/// Warmup iterations run on the capture stream before capture. cuBLAS/cuDNN
/// algorithm selection and the caching allocator typically need a few steps to
/// reach steady state; capturing a not-yet-warm body risks capturing a
/// one-time allocation or autotune path. PyTorch's make_graphed_callables uses 3.
const GRAPH_WARMUP_ITERS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum GraphCaptureState {
    Warmup,
    ReadyToCapture,
    Captured,
}

pub(super) struct PpoUpdateCudaGraph {
    graph: Option<CudaGraph>,
    state: GraphCaptureState,
    disabled_reason: Option<String>,
    chunk_batch_size: i64,
    ppo_chunk_len: i64,
    minibatch_sample_count: i64,
    flat_layout_len: i64,
    so_dim: i64,
    windowed: Tensor,
    static_flat: Tensor,
    advantages: Tensor,
    returns: Tensor,
    old_log_probs: Tensor,
    actions: Tensor,
    old_alphas: Tensor,
    old_betas: Tensor,
    outputs: PpoGraphOutputs,
}

/// Persistent graph output buffers; `snapshot` returns an independent set of
/// handles to the same buffers for use as a minibatch's metrics.
struct PpoGraphOutputs {
    action_loss: Tensor,
    value_loss: Tensor,
    clip_gap_mean: Tensor,
    dist_entropy: Tensor,
    approx_kl: Tensor,
    actor_grad_norm: Tensor,
    critic_grad_norm: Tensor,
    clip_violations: Tensor,
}

impl PpoUpdateCudaGraph {
    fn new(
        device: tch::Device,
        chunk_batch_size: i64,
        ppo_chunk_len: i64,
        flat_layout_len: i64,
        so_dim: i64,
        replay_obs_kind: Kind,
    ) -> Self {
        let graph = CudaGraph::new(device);
        let (graph, disabled_reason) = match graph {
            Ok(Some(graph)) if env::var("PPO_CUDA_GRAPHS").ok().as_deref() != Some("0") => {
                (Some(graph), None)
            }
            Ok(Some(_)) => (None, Some("disabled by PPO_CUDA_GRAPHS=0".to_string())),
            Ok(None) => (None, Some("CUDA Graph support is unavailable".to_string())),
            Err(err) => (None, Some(format!("CUDA Graph init failed: {err}"))),
        };
        if let Some(reason) = &disabled_reason {
            println!("PPO CUDA graphs disabled: {reason}");
        } else {
            println!("PPO CUDA graphs enabled for full no-reset minibatches");
        }

        let minibatch_sample_count = chunk_batch_size * ppo_chunk_len;
        Self {
            graph,
            state: GraphCaptureState::Warmup,
            disabled_reason,
            chunk_batch_size,
            ppo_chunk_len,
            minibatch_sample_count,
            flat_layout_len,
            so_dim,
            windowed: Tensor::zeros(
                [minibatch_sample_count * TICKERS_COUNT, flat_layout_len],
                (replay_obs_kind, device),
            ),
            static_flat: Tensor::zeros([minibatch_sample_count, so_dim], (replay_obs_kind, device)),
            advantages: Tensor::zeros([minibatch_sample_count], (Kind::Float, device)),
            returns: Tensor::zeros([minibatch_sample_count], (Kind::Float, device)),
            old_log_probs: Tensor::zeros([minibatch_sample_count], (Kind::Float, device)),
            actions: Tensor::zeros(
                [minibatch_sample_count, ACTION_COUNT],
                (Kind::Float, device),
            ),
            old_alphas: Tensor::zeros(
                [minibatch_sample_count, ACTION_COUNT],
                (Kind::Float, device),
            ),
            old_betas: Tensor::zeros(
                [minibatch_sample_count, ACTION_COUNT],
                (Kind::Float, device),
            ),
            outputs: PpoGraphOutputs::new(device),
        }
    }

    fn matches(
        &self,
        chunk_batch_size: i64,
        ppo_chunk_len: i64,
        flat_layout_len: i64,
        so_dim: i64,
    ) -> bool {
        self.chunk_batch_size == chunk_batch_size
            && self.ppo_chunk_len == ppo_chunk_len
            && self.flat_layout_len == flat_layout_len
            && self.so_dim == so_dim
    }

    fn is_enabled(&self) -> bool {
        self.disabled_reason.is_none() && self.graph.is_some()
    }

    /// Permanently disable the graph after a capture/replay failure: record and
    /// log the reason, zero any partially-accumulated trainable grads, drop the
    /// graph, and signal the caller to fall back to the eager path (`None`).
    fn disable_and_fallback(
        &mut self,
        reason: String,
        trainable_vars: &[Tensor],
    ) -> Option<PpoGraphOutputs> {
        println!("PPO CUDA graphs disabled: {reason}");
        self.disabled_reason = Some(reason);
        zero_trainable_grads(trainable_vars);
        self.graph = None;
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        trading_model: &TradingModel,
        hl_gauss: &HlGaussBins,
        grad_clip_groups: &GradClipGroups,
        trainable_vars: &[Tensor],
        device: tch::Device,
        windowed: &Tensor,
        static_flat: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
        old_log_probs: &Tensor,
        actions: &Tensor,
        old_alphas: &Tensor,
        old_betas: &Tensor,
    ) -> Option<PpoGraphOutputs> {
        if !self.is_enabled() {
            return None;
        }

        // Input H2D copies run on the default stream; the graph's stream scope
        // orders the capture/replay stream after them via an event, so no
        // full-device host sync is needed here.
        copy_into(&self.windowed, windowed);
        copy_into(&self.static_flat, static_flat);
        copy_into(&self.advantages, advantages);
        copy_into(&self.returns, returns);
        copy_into(&self.old_log_probs, old_log_probs);
        copy_into(&self.actions, actions);
        copy_into(&self.old_alphas, old_alphas);
        copy_into(&self.old_betas, old_betas);

        match self.state {
            GraphCaptureState::Warmup => {
                // Run several warmup iterations on the capture stream so cuBLAS /
                // cuDNN / the caching allocator reach steady state before capture.
                // Warmup runs on the same stream capture will use, so allocator
                // pool selection matches.
                let graph = self.graph.as_ref().expect("enabled graph missing");
                let scope = graph.with_stream_scope(|_| {
                    for _ in 0..GRAPH_WARMUP_ITERS {
                        run_graph_body(
                            trading_model,
                            hl_gauss,
                            grad_clip_groups,
                            trainable_vars,
                            device,
                            self.minibatch_sample_count,
                            &self.windowed,
                            &self.static_flat,
                            &self.advantages,
                            &self.returns,
                            &self.old_log_probs,
                            &self.actions,
                            &self.old_alphas,
                            &self.old_betas,
                            &self.outputs,
                        );
                    }
                });
                if let Err(err) = scope {
                    return self.disable_and_fallback(
                        format!("graph warmup failed: {err}"),
                        trainable_vars,
                    );
                }
                self.state = GraphCaptureState::ReadyToCapture;
                println!("PPO CUDA graph warmup complete");
                Some(self.outputs.snapshot())
            }
            GraphCaptureState::ReadyToCapture => {
                let graph = self.graph.as_ref().expect("enabled graph missing");
                // Capture and the first replay both run inside one stream scope.
                let scope = graph
                    .with_stream_scope(|graph| {
                        graph.capture(|| {
                            run_graph_body(
                                trading_model,
                                hl_gauss,
                                grad_clip_groups,
                                trainable_vars,
                                device,
                                self.minibatch_sample_count,
                                &self.windowed,
                                &self.static_flat,
                                &self.advantages,
                                &self.returns,
                                &self.old_log_probs,
                                &self.actions,
                                &self.old_alphas,
                                &self.old_betas,
                                &self.outputs,
                            );
                        })?;
                        graph.replay()
                    })
                    .and_then(|inner| inner);
                match scope {
                    Ok(()) => {
                        self.state = GraphCaptureState::Captured;
                        println!("PPO CUDA graph captured");
                        Some(self.outputs.snapshot())
                    }
                    Err(err) => {
                        self.disable_and_fallback(format!("capture failed: {err}"), trainable_vars)
                    }
                }
            }
            GraphCaptureState::Captured => {
                let graph = self.graph.as_ref().expect("enabled graph missing");
                let replay = graph
                    .with_stream_scope(|graph| graph.replay())
                    .and_then(|inner| inner);
                match replay {
                    Ok(()) => Some(self.outputs.snapshot()),
                    Err(err) => {
                        self.disable_and_fallback(format!("replay failed: {err}"), trainable_vars)
                    }
                }
            }
        }
    }
}

impl PpoGraphOutputs {
    fn new(device: tch::Device) -> Self {
        Self {
            action_loss: Tensor::zeros([], (Kind::Float, device)),
            value_loss: Tensor::zeros([], (Kind::Float, device)),
            clip_gap_mean: Tensor::zeros([], (Kind::Float, device)),
            dist_entropy: Tensor::zeros([], (Kind::Float, device)),
            approx_kl: Tensor::zeros([], (Kind::Float, device)),
            actor_grad_norm: Tensor::zeros([], (Kind::Float, device)),
            critic_grad_norm: Tensor::zeros([], (Kind::Float, device)),
            clip_violations: Tensor::zeros([], (Kind::Float, device)),
        }
    }

    fn snapshot(&self) -> PpoGraphOutputs {
        PpoGraphOutputs {
            action_loss: self.action_loss.shallow_clone(),
            value_loss: self.value_loss.shallow_clone(),
            clip_gap_mean: self.clip_gap_mean.shallow_clone(),
            dist_entropy: self.dist_entropy.shallow_clone(),
            approx_kl: self.approx_kl.shallow_clone(),
            actor_grad_norm: self.actor_grad_norm.shallow_clone(),
            critic_grad_norm: self.critic_grad_norm.shallow_clone(),
            clip_violations: self.clip_violations.shallow_clone(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_graph_body(
    trading_model: &TradingModel,
    hl_gauss: &HlGaussBins,
    grad_clip_groups: &GradClipGroups,
    trainable_vars: &[Tensor],
    device: tch::Device,
    minibatch_sample_count: i64,
    windowed: &Tensor,
    static_flat: &Tensor,
    advantages: &Tensor,
    returns: &Tensor,
    old_log_probs: &Tensor,
    actions: &Tensor,
    old_alphas: &Tensor,
    old_betas: &Tensor,
    outputs: &PpoGraphOutputs,
) {
    let (new_value_logits, action_alpha, action_beta) = autocast(false, || {
        trading_model.windowed_replay_forward(windowed, static_flat, minibatch_sample_count)
    });
    let action_log_probs = beta_log_prob(actions, &action_alpha, &action_beta);
    let dist_entropy_per_sample = beta_entropy(&action_alpha, &action_beta);
    let log_ratio = &action_log_probs - old_log_probs;
    let ratio = log_ratio.exp();
    let ratio_diff = &ratio - 1.0;
    let (action_loss, clip_gap) = match POLICY_OBJECTIVE {
        PolicyObjective::Pmpo => {
            let loss = pmpo_policy_loss(
                advantages,
                &action_log_probs,
                old_alphas,
                old_betas,
                &action_alpha,
                &action_beta,
            );
            let clip_gap = tch::no_grad(|| Tensor::zeros_like(&ratio));
            (loss, clip_gap)
        }
        PolicyObjective::Ppo => asym_clip_policy_loss(advantages, &ratio),
    };
    let value_loss = hl_gauss_value_loss(hl_gauss, &new_value_logits, returns).mean(Kind::Float);
    let dist_entropy = dist_entropy_per_sample.mean(Kind::Float);
    let actor_loss = action_loss.shallow_clone() - &dist_entropy * ENTROPY_COEF;
    let critic_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF;
    let (actor_grad_norm, critic_grad_norm) = backward_actor_critic_with_separate_clips(
        grad_clip_groups,
        trainable_vars,
        &actor_loss,
        &critic_loss,
        MAX_GRAD_NORM,
        device,
        false,
    );

    // Analytical closed-form KL(old||new) drives the early-stop and KL-LR
    // controller for both objectives; the ratio estimator only fed PPO's
    // surrogate diagnostics.
    let approx_kl = tch::no_grad(|| {
        beta_reverse_kl(old_alphas, old_betas, &action_alpha, &action_beta).mean(Kind::Float)
    });
    let clip_violations = tch::no_grad(|| {
        ratio_diff
            .gt(CLIP_EPS_HIGH)
            .logical_or(&ratio_diff.lt(-CLIP_EPS_LOW))
            .to_kind(Kind::Float)
            .sum(Kind::Float)
    });
    tch::no_grad(|| {
        copy_into(&outputs.action_loss, &action_loss.detach());
        copy_into(&outputs.value_loss, &value_loss.detach());
        copy_into(&outputs.clip_gap_mean, &clip_gap.mean(Kind::Float));
        copy_into(&outputs.dist_entropy, &dist_entropy.detach());
        copy_into(&outputs.approx_kl, &approx_kl);
        copy_into(
            &outputs.actor_grad_norm,
            &actor_grad_norm.to_kind(Kind::Float),
        );
        copy_into(
            &outputs.critic_grad_norm,
            &critic_grad_norm.to_kind(Kind::Float),
        );
        copy_into(&outputs.clip_violations, &clip_violations);
    });
}

fn copy_into(dst: &Tensor, src: &Tensor) {
    let mut dst = dst.shallow_clone();
    dst.copy_(src);
}

fn zero_trainable_grads(trainable_vars: &[Tensor]) {
    for param in trainable_vars {
        let mut param = param.shallow_clone();
        param.zero_grad();
    }
}

/// One-shot "log the first occurrence" latches for the per-update numeric-debug
/// ladder. Each stays set once a non-finite value has been reported so the log
/// isn't flooded across minibatches.
#[derive(Default)]
struct NonFiniteLogState {
    replay_input: bool,
    forward: bool,
    loss: bool,
    grad: bool,
    param: bool,
    forward_probe: bool,
}

impl Trainer {
    /// Check the replay/forward inputs for non-finite values, logging the first
    /// occurrence (and the worst params at that point) exactly once per update.
    fn log_non_finite_replay_inputs(
        &self,
        logged: &mut bool,
        episode: usize,
        epoch: i64,
        chunk_i: usize,
        windowed: &Tensor,
        static_flat: &Tensor,
        actions_flat: &Tensor,
        old_log_probs_flat: &Tensor,
        adv_flat: &Tensor,
        ret_flat: &Tensor,
    ) {
        if log_first_non_finite_tensor(
            logged,
            "replay_inputs",
            episode,
            epoch,
            chunk_i,
            &[
                ("windowed", windowed),
                ("static_flat", static_flat),
                ("actions", actions_flat),
                ("old_log_probs", old_log_probs_flat),
                ("advantages", adv_flat),
                ("returns", ret_flat),
            ],
        ) {
            log_named_var_extremes(
                "params_at_replay_input_failure",
                episode,
                epoch,
                chunk_i,
                &self.named_trainable_vars,
                false,
                12,
            );
        }
    }

    /// Log the first non-finite loss component (and worst params at that point).
    /// Shared by the graph-replay and eager paths; the loss tensors carry
    /// identical semantics on both.
    #[allow(clippy::too_many_arguments)]
    fn log_non_finite_loss(
        &self,
        logged: &mut bool,
        episode: usize,
        epoch: i64,
        chunk_i: usize,
        action_loss: &Tensor,
        value_loss: &Tensor,
        dist_entropy: &Tensor,
        actor_loss: &Tensor,
        critic_loss: &Tensor,
        total_loss: &Tensor,
    ) {
        if log_first_non_finite_tensor(
            logged,
            "loss",
            episode,
            epoch,
            chunk_i,
            &[
                ("action_loss", action_loss),
                ("value_loss", value_loss),
                ("dist_entropy", dist_entropy),
                ("actor_loss", actor_loss),
                ("critic_loss", critic_loss),
                ("total_loss", total_loss),
            ],
        ) {
            log_named_var_extremes(
                "params_at_loss_failure",
                episode,
                epoch,
                chunk_i,
                &self.named_trainable_vars,
                false,
                12,
            );
        }
    }

    /// Log the first non-finite gradient after backward.
    fn log_non_finite_grads_after_backward(
        &self,
        logged: &mut bool,
        episode: usize,
        epoch: i64,
        chunk_i: usize,
    ) {
        if log_first_non_finite_var(
            logged,
            "grads_after_backward",
            episode,
            epoch,
            chunk_i,
            &self.named_trainable_vars,
            true,
        ) {
            log_named_var_extremes(
                "grads_after_backward_top_abs",
                episode,
                epoch,
                chunk_i,
                &self.named_trainable_vars,
                true,
                12,
            );
        }
    }

    /// Log the first non-finite parameter after the optimizer step.
    fn log_non_finite_params_after_step(
        &self,
        logged: &mut bool,
        episode: usize,
        epoch: i64,
        chunk_i: usize,
    ) {
        if log_first_non_finite_var(
            logged,
            "params_after_step",
            episode,
            epoch,
            chunk_i,
            &self.named_trainable_vars,
            false,
        ) {
            log_named_var_extremes(
                "params_after_step_top_abs",
                episode,
                epoch,
                chunk_i,
                &self.named_trainable_vars,
                false,
                12,
            );
        }
    }
}

impl Trainer {
    pub(super) fn update_policy(
        &mut self,
        episode: usize,
        adv_data: &AdvantageData,
    ) -> UpdateMetrics {
        let device = self.device;
        // Critic-only pretraining: warm up the value function (and the shared
        // trunk through it) before the actor turns on. While active, the actor
        // backward is skipped, the CUDA graph is suppressed (so it captures
        // fresh with the full actor+critic body once the actor begins), and the
        // KL early-stop is disabled since trunk drift would spuriously trip it.
        let critic_only = episode < CRITIC_PRETRAIN_EPISODES;
        let mut total_policy_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_value_loss_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut total_clip_gap_weighted = Tensor::zeros([], (Kind::Float, device));
        let mut actor_grad_norm_sum = Tensor::zeros([], (Kind::Float, device));
        let mut critic_grad_norm_sum = Tensor::zeros([], (Kind::Float, device));
        let mut total_sample_count = 0i64;
        let mut grad_norm_count = 0i64;
        let mut total_clip_violations = Tensor::zeros([], (Kind::Float, device));
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
        let mut graph_time_us = 0u64;
        let mut nf_log = NonFiniteLogState::default();

        let mut mean_epoch_approx_kl = 0.0f64;
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
                let chunk_ids_host = &perm_host[mb_start as usize..mb_end as usize];
                let chunk_ids = perm_gpu.narrow(0, mb_start, chunk_count);
                let boundary_layout = self.s_chunk_start_layouts.index_select(0, &chunk_ids);
                let so_chunk = self.s_static_obs.index_select(0, &chunk_ids);
                let step_deltas_chunk = self.s_step_deltas.index_select(0, &chunk_ids);
                let adv_mb_by_chunk = adv_data.advantages.index_select(0, &chunk_ids);
                let ret_mb_by_chunk = adv_data.returns.index_select(0, &chunk_ids);
                let old_log_probs_by_chunk = self.s_old_log_probs.index_select(0, &chunk_ids);
                let actions_by_chunk = self.s_actions.index_select(0, &chunk_ids);
                let old_alphas_by_chunk = self.s_old_alphas.index_select(0, &chunk_ids);
                let old_betas_by_chunk = self.s_old_betas.index_select(0, &chunk_ids);
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
                    && chunk_ids_host
                        .iter()
                        .any(|id| adv_data.reset_chunks_have_slots[*id as usize]);
                let windowed = if has_reset_slots {
                    let layout_rows = chunk_count * TICKERS_COUNT;
                    let mut current_layout = boundary_layout.view([layout_rows, flat_layout_len]);
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
                            let reset_chunk_idx = step_reset_slots.gt(0).nonzero().squeeze_dim(1);
                            if reset_chunk_idx.size()[0] > 0 {
                                let reset_slot_ids =
                                    step_reset_slots.index_select(0, &reset_chunk_idx) - 1;
                                let reset_slot_ids_cpu = reset_slot_ids.to_device(tch::Device::Cpu);
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
                // `adv_flat`/`ret_flat` are fresh per-minibatch tensors (gather +
                // reshape), so scaling `adv_flat` below never touches the persistent
                // raw `adv_data.advantages`.
                let adv_flat = adv_mb_by_chunk.reshape([-1]);
                let ret_flat = ret_mb_by_chunk.reshape([-1]);
                let old_log_probs_flat = old_log_probs_by_chunk.reshape([-1]);
                let actions_flat = actions_by_chunk.reshape([-1, ACTION_COUNT]);
                let old_alphas_flat = old_alphas_by_chunk.reshape([-1, ACTION_COUNT]);
                let old_betas_flat = old_betas_by_chunk.reshape([-1, ACTION_COUNT]);

                // Per-minibatch percentile return-norm ("mbpercnorm"): divide-only
                // scale by S = clamp(P95 - P5 of THIS minibatch's raw GAE returns,
                // FLOOR), recomputed fresh each minibatch (no EMA, no caching). The
                // critic still regresses raw `ret_flat`; only the policy advantage
                // is scaled. Computed eagerly here, OUTSIDE any captured graph body,
                // so the scaled `adv_flat` is what gets copied into the CUDA-graph
                // static input buffer (and what the eager path consumes) — no
                // staleness, and the graph captures a correctly-scaled static buffer
                // on every replay.
                // PMPO consumes RAW GAE advantages (its tanh weighting subsumes
                // scale normalization); only PPO applies the percentile norm.
                let adv_flat = if POLICY_OBJECTIVE == PolicyObjective::Pmpo {
                    adv_flat
                } else {
                    tch::no_grad(|| {
                        let qs = Tensor::from_slice(&[RET_PERC_LO, RET_PERC_HI])
                            .to_kind(Kind::Float)
                            .to_device(device);
                        let bounds = ret_flat.quantile(&qs, None::<i64>, false, "linear");
                        let lo = bounds.get(0);
                        let hi = bounds.get(1);
                        let scale = (hi - lo).clamp_min(RET_PERC_FLOOR);
                        adv_flat / scale
                    })
                };

                // Runs once per minibatch, shared by the graph and eager paths
                // (the graph branch may `continue`, otherwise control falls
                // through to eager without re-checking).
                self.log_non_finite_replay_inputs(
                    &mut nf_log.replay_input,
                    episode,
                    _epoch,
                    chunk_i,
                    &windowed,
                    &static_flat,
                    &actions_flat,
                    &old_log_probs_flat,
                    &adv_flat,
                    &ret_flat,
                );

                let graph_eligible = self.device.is_cuda()
                    && !DEBUG_NUMERICS
                    && !critic_only
                    && !has_reset_slots
                    && chunk_count == adv_data.chunk_batch_size;
                if graph_eligible {
                    let graph_start = Instant::now();
                    let mut graph = self.ppo_update_graph.take().unwrap_or_else(|| {
                        PpoUpdateCudaGraph::new(
                            device,
                            adv_data.chunk_batch_size,
                            self.rollout.ppo_chunk_len,
                            flat_layout_len,
                            self.so_dim,
                            self.replay_obs_kind,
                        )
                    });
                    // The captured shapes derive solely from constructor-fixed
                    // Trainer geometry and compile-time constants, so a persisted
                    // graph always matches across every update in a run; the only
                    // per-minibatch variation (the partial last minibatch) is
                    // excluded from `graph_eligible` above.
                    assert!(graph.matches(
                        adv_data.chunk_batch_size,
                        self.rollout.ppo_chunk_len,
                        flat_layout_len,
                        self.so_dim,
                    ));
                    let graph_metrics = graph.run(
                        &self.trading_model,
                        &self.hl_gauss,
                        &self.grad_clip_groups,
                        &self.trainable_vars,
                        device,
                        &windowed,
                        &static_flat,
                        &adv_flat,
                        &ret_flat,
                        &old_log_probs_flat,
                        &actions_flat,
                        &old_alphas_flat,
                        &old_betas_flat,
                    );
                    self.ppo_update_graph = Some(graph);

                    if let Some(metrics) = graph_metrics {
                        graph_time_us += graph_start.elapsed().as_micros() as u64;
                        let graph_actor_loss = metrics.action_loss.shallow_clone()
                            - &metrics.dist_entropy * ENTROPY_COEF;
                        let graph_critic_loss =
                            metrics.value_loss.shallow_clone() * VALUE_LOSS_COEF;
                        let graph_total_loss =
                            graph_actor_loss.shallow_clone() + graph_critic_loss.shallow_clone();
                        self.log_non_finite_loss(
                            &mut nf_log.loss,
                            episode,
                            _epoch,
                            chunk_i,
                            &metrics.action_loss,
                            &metrics.value_loss,
                            &metrics.dist_entropy,
                            &graph_actor_loss,
                            &graph_critic_loss,
                            &graph_total_loss,
                        );
                        self.log_non_finite_grads_after_backward(
                            &mut nf_log.grad,
                            episode,
                            _epoch,
                            chunk_i,
                        );
                        let _ = epoch_kl_gpu
                            .g_add_(&(&metrics.approx_kl * minibatch_sample_count as f64));
                        let _ = total_policy_loss_weighted
                            .g_add_(&(&metrics.action_loss * minibatch_sample_count as f64));
                        let _ = total_value_loss_weighted
                            .g_add_(&(&metrics.value_loss * minibatch_sample_count as f64));
                        let _ = total_clip_gap_weighted
                            .g_add_(&(&metrics.clip_gap_mean * minibatch_sample_count as f64));
                        let _ = total_entropy_weighted
                            .g_add_(&(&metrics.dist_entropy * minibatch_sample_count as f64));
                        entropy_min = entropy_min.min_other(&metrics.dist_entropy);
                        entropy_max = entropy_max.max_other(&metrics.dist_entropy);
                        epoch_kl_count += minibatch_sample_count;
                        total_sample_count += minibatch_sample_count;
                        let _ = total_clip_violations.g_add_(&metrics.clip_violations);
                        total_ratio_samples += minibatch_sample_count;
                        actor_grad_norm_sum += metrics.actor_grad_norm;
                        critic_grad_norm_sum += metrics.critic_grad_norm;
                        grad_norm_count += 1;
                        step_optimizer(&mut self.opt, &mut self.optimizer_step);
                        self.log_non_finite_params_after_step(
                            &mut nf_log.param,
                            episode,
                            _epoch,
                            chunk_i,
                        );
                        last_minibatch_kl_mean_gpu = Some(metrics.approx_kl.detach().copy());
                        continue;
                    }
                    graph_time_us += graph_start.elapsed().as_micros() as u64;
                }

                let (new_value_logits, action_alpha, action_beta) = autocast(false, || {
                    self.trading_model.windowed_replay_forward(
                        &windowed,
                        &static_flat,
                        minibatch_sample_count,
                    )
                });

                let action_log_probs = beta_log_prob(&actions_flat, &action_alpha, &action_beta);
                let dist_entropy_per_sample = beta_entropy(&action_alpha, &action_beta);

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("actions_mb", &actions_flat, _epoch, chunk_i);
                    let _ = debug_tensor_stats(
                        "old_log_probs_mb",
                        &old_log_probs_flat,
                        _epoch,
                        chunk_i,
                    );
                    let _ = debug_tensor_stats("action_alpha", &action_alpha, _epoch, chunk_i);
                    let _ = debug_tensor_stats("action_beta", &action_beta, _epoch, chunk_i);
                }

                let log_ratio = &action_log_probs - &old_log_probs_flat;

                if DEBUG_NUMERICS {
                    let _ =
                        debug_tensor_stats("action_log_probs", &action_log_probs, _epoch, chunk_i);
                    let _ = debug_tensor_stats("log_ratio", &log_ratio, _epoch, chunk_i);
                }
                let ratio = log_ratio.exp();
                let ratio_diff = &ratio - 1.0;

                if log_first_non_finite_tensor(
                    &mut nf_log.forward,
                    "forward",
                    episode,
                    _epoch,
                    chunk_i,
                    &[
                        ("action_alpha", &action_alpha),
                        ("action_beta", &action_beta),
                        ("action_log_probs", &action_log_probs),
                        ("old_log_probs", &old_log_probs_flat),
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
                    if !nf_log.forward_probe {
                        nf_log.forward_probe = true;
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

                let (action_loss, clip_gap) = match POLICY_OBJECTIVE {
                    PolicyObjective::Pmpo => {
                        let loss = pmpo_policy_loss(
                            &adv_flat,
                            &action_log_probs,
                            &old_alphas_flat,
                            &old_betas_flat,
                            &action_alpha,
                            &action_beta,
                        );
                        let clip_gap = tch::no_grad(|| Tensor::zeros_like(&ratio));
                        (loss, clip_gap)
                    }
                    PolicyObjective::Ppo => asym_clip_policy_loss(&adv_flat, &ratio),
                };

                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("ret_mb", &ret_flat, _epoch, chunk_i);
                    let _ =
                        debug_tensor_stats("new_value_logits", &new_value_logits, _epoch, chunk_i);
                    let _ = debug_tensor_stats("adv_mb", &adv_flat, _epoch, chunk_i);
                }

                let value_loss = hl_gauss_value_loss(&self.hl_gauss, &new_value_logits, &ret_flat)
                    .mean(Kind::Float);

                let dist_entropy = dist_entropy_per_sample.mean(Kind::Float);
                let dist_entropy_detached = dist_entropy.detach();

                let actor_loss = action_loss.shallow_clone() - &dist_entropy * ENTROPY_COEF;
                let critic_loss = value_loss.shallow_clone() * VALUE_LOSS_COEF;
                let total_loss = actor_loss.shallow_clone() + critic_loss.shallow_clone();

                self.log_non_finite_loss(
                    &mut nf_log.loss,
                    episode,
                    _epoch,
                    chunk_i,
                    &action_loss,
                    &value_loss,
                    &dist_entropy,
                    &actor_loss,
                    &critic_loss,
                    &total_loss,
                );

                fwd_time_us += fwd_start.elapsed().as_micros() as u64;
                let bwd_start = Instant::now();
                let (actor_grad_norm, critic_grad_norm) = backward_actor_critic_with_separate_clips(
                    &self.grad_clip_groups,
                    &self.trainable_vars,
                    &actor_loss,
                    &critic_loss,
                    MAX_GRAD_NORM,
                    device,
                    critic_only,
                );
                bwd_time_us += bwd_start.elapsed().as_micros() as u64;

                self.log_non_finite_grads_after_backward(
                    &mut nf_log.grad,
                    episode,
                    _epoch,
                    chunk_i,
                );

                // Analytical closed-form KL(old||new) drives the early-stop and
                // KL-LR controller for both objectives.
                let approx_kl_val = tch::no_grad(|| {
                    beta_reverse_kl(
                        &old_alphas_flat,
                        &old_betas_flat,
                        &action_alpha,
                        &action_beta,
                    )
                    .mean(Kind::Float)
                });
                if DEBUG_NUMERICS {
                    let _ = debug_tensor_stats("approx_kl_val", &approx_kl_val, _epoch, chunk_i);
                }
                let _ = epoch_kl_gpu.g_add_(&(&approx_kl_val * minibatch_sample_count as f64));
                let _ = total_policy_loss_weighted
                    .g_add_(&(&action_loss.detach() * minibatch_sample_count as f64));
                let _ = total_value_loss_weighted
                    .g_add_(&(&value_loss.detach() * minibatch_sample_count as f64));
                let _ = total_clip_gap_weighted
                    .g_add_(&(clip_gap.mean(Kind::Float) * minibatch_sample_count as f64));
                let _ = total_entropy_weighted
                    .g_add_(&(&dist_entropy_detached * minibatch_sample_count as f64));
                entropy_min = entropy_min.min_other(&dist_entropy_detached);
                entropy_max = entropy_max.max_other(&dist_entropy_detached);
                epoch_kl_count += minibatch_sample_count;
                total_sample_count += minibatch_sample_count;

                let _ = total_clip_violations.g_add_(&tch::no_grad(|| {
                    ratio_diff
                        .gt(CLIP_EPS_HIGH)
                        .logical_or(&ratio_diff.lt(-CLIP_EPS_LOW))
                        .to_kind(Kind::Float)
                        .sum(Kind::Float)
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

                actor_grad_norm_sum += actor_grad_norm.to_kind(Kind::Float);
                critic_grad_norm_sum += critic_grad_norm.to_kind(Kind::Float);
                grad_norm_count += 1;

                step_optimizer(&mut self.opt, &mut self.optimizer_step);
                self.log_non_finite_params_after_step(&mut nf_log.param, episode, _epoch, chunk_i);
                self.opt.zero_grad();

                // One forward/backward per minibatch now: the minibatch's KL is
                // exactly approx_kl_val. Track the last one for end-of-epoch diagnostics.
                last_minibatch_kl_mean_gpu = Some(approx_kl_val.shallow_clone());
            }

            // Single end-of-epoch host sync covering both the epoch-mean KL used
            // for early stopping and the last-minibatch KL diagnostic. Avoids
            // per-minibatch D2H stalls that previously blocked the training pipeline.
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
            mean_epoch_approx_kl = mean_epoch_kl;
            println!(
                "Epoch {}/{}: RatioKL {:.4} (last mb {:.4})",
                _epoch + 1,
                OPTIM_EPOCHS,
                mean_epoch_kl,
                last_minibatch_approx_kl
            );
            if !critic_only && mean_epoch_kl > TARGET_KL * KL_STOP_MULTIPLIER {
                break 'epoch_loop;
            }
        }

        println!(
            "fwd: {:.1}ms  bwd: {:.1}ms  graph: {:.1}ms",
            fwd_time_us as f64 / 1000.0,
            bwd_time_us as f64 / 1000.0,
            graph_time_us as f64 / 1000.0
        );

        UpdateMetrics {
            total_policy_loss_weighted,
            total_value_loss_weighted,
            total_clip_gap_weighted,
            actor_grad_norm_sum,
            critic_grad_norm_sum,
            total_sample_count,
            grad_norm_count,
            total_clip_violations,
            total_ratio_samples,
            total_entropy_weighted,
            entropy_min,
            entropy_max,
            mean_epoch_approx_kl,
            last_minibatch_approx_kl,
            lr_scale: 1.0,
            kl_lr_scale_next: 1.0,
            kl_lr_ema: 0.0,
            kl_lr_signal: last_minibatch_approx_kl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{asym_clip_policy_loss, CLIP_EPS_HIGH, CLIP_EPS_LOW};
    use tch::Tensor;

    #[test]
    fn asym_clip_policy_loss_uses_dapo_clip_higher_bounds() {
        // Ratios straddle both bounds: 0.7 < 1-LOW=0.80, 1.4 > 1+HIGH=1.28.
        let advantage = Tensor::from_slice(&[2.0f32, 2.0, -2.0, -2.0]);
        let ratio = Tensor::from_slice(&[1.4f32, 0.7, 1.4, 0.7]);

        let (loss, clip_gap) = asym_clip_policy_loss(&advantage, &ratio);

        let lo = 1.0 - CLIP_EPS_LOW;
        let hi = 1.0 + CLIP_EPS_HIGH;
        let clamp = |r: f64| r.clamp(lo, hi);
        let pg = |a: f64, r: f64| (-a * r).max(-a * clamp(r));
        let samples = [(2.0, 1.4), (2.0, 0.7), (-2.0, 1.4), (-2.0, 0.7)];

        let expected_loss = samples.iter().map(|&(a, r)| pg(a, r)).sum::<f64>() / 4.0;
        assert!((loss.double_value(&[]) - expected_loss).abs() < 1e-6);

        for (i, &(_, r)) in samples.iter().enumerate() {
            let expected_gap = (r - clamp(r)).abs();
            assert!((clip_gap.double_value(&[i as i64]) - expected_gap).abs() < 1e-6);
        }
    }
}
