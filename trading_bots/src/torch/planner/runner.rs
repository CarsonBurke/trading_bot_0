use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use rand::{rngs::StdRng, SeedableRng};
use tch::{nn, Device, Kind, Tensor};

use crate::torch::{
    action_space::{beta_log_prob, beta_mean, sample_beta_action},
    cuda::cfg::configure_cuda,
    optim::muon::{Muon, MuonConfig},
    train::{
        config::{LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON},
        optimizer_glue::{
            apply_lr_scale, backward_actor_critic_with_separate_clips, grad_clip_groups,
            muon_momentum_for_step, named_trainable_variables, KlLrController,
        },
    },
    value::hl_gauss::HlGaussBins,
    world_model::{world_model_metadata_path, LejepaWorldModel, WorldModelPrediction},
};

use super::{
    checkpoint::{
        load_planner_checkpoint, planner_metadata_path, planner_optimizer_state_path,
        save_planner_checkpoint, verify_optimizer_state, PlannerCheckpointMetadata,
        FANTASY_CLOSE_DELTA_MAX, FANTASY_CLOSE_DELTA_MIN, KL_CONTROLLER_HALF_LIFE, KL_MAX_LR_SCALE,
        KL_MIN_LR_SCALE, OPTIMIZATION_EPOCHS, TARGET_KL,
    },
    data::{planner_context_bars, PlannerDataSplit, PlannerDataset, PlannerEndpoint},
    gae::compute_planner_gae,
    losses::{planner_actor_critic_losses, route_value_logits, split_critic_diagnostics},
    portfolio::PlannerPortfolio,
    rollout::{
        MixedRollout, PlannerBatch, PlannerObservation, PlannerTransition, RolloutMetrics,
        RolloutSource,
    },
    PlannerForecast, WorldModelPlanner, WorldModelPlannerInput, PLANNER_PORTFOLIO_DIM,
};

pub const DEFAULT_PLANNER_HORIZON: usize = 100;
pub const DEFAULT_PLANNER_ROLLOUT_LENGTH: usize = 100;
pub const DEFAULT_PLANNER_ENVIRONMENTS: usize = 16;
pub const DEFAULT_PLANNER_OPTIMIZATION_EPOCHS: usize = OPTIMIZATION_EPOCHS;
pub const DEFAULT_PLANNER_MINIBATCH_SIZE: usize = 160;
const VALIDATION_EVERY_UPDATES: u64 = 50;
const HELD_OUT_ENDPOINTS: usize = 16;
const VALIDATION_MAX_MEAN_DRAWDOWN: f64 = 0.30;
const VALIDATION_MAX_MEAN_TURNOVER: f64 = 0.50;

#[derive(Clone, Debug)]
pub struct TrainPlannerArgs {
    pub world_model_weights: String,
    pub world_model_metadata: Option<String>,
    pub planner_weights: Option<String>,
    pub output: String,
    pub updates: usize,
    pub horizon: usize,
    pub rollout_length: usize,
    pub environments: usize,
    pub minibatch_size: usize,
    pub context_bars: Option<usize>,
    pub tickers: Option<Vec<String>>,
    pub seed: u64,
}

impl Default for TrainPlannerArgs {
    fn default() -> Self {
        Self {
            world_model_weights: "weights/pretrain_heads_best.ot".to_owned(),
            world_model_metadata: None,
            planner_weights: None,
            output: "weights/planner.ot".to_owned(),
            updates: 1_000,
            horizon: DEFAULT_PLANNER_HORIZON,
            rollout_length: DEFAULT_PLANNER_ROLLOUT_LENGTH,
            environments: DEFAULT_PLANNER_ENVIRONMENTS,
            minibatch_size: DEFAULT_PLANNER_MINIBATCH_SIZE,
            context_bars: None,
            tickers: None,
            seed: 7,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InferPlannerArgs {
    pub world_model_weights: String,
    pub world_model_metadata: Option<String>,
    pub planner_weights: String,
    pub episodes: usize,
    pub horizon: Option<usize>,
    pub rollout_length: usize,
    pub context_bars: Option<usize>,
    pub tickers: Option<Vec<String>>,
    pub split: PlannerDataSplit,
}

impl Default for InferPlannerArgs {
    fn default() -> Self {
        Self {
            world_model_weights: "weights/pretrain_heads_best.ot".to_owned(),
            world_model_metadata: None,
            planner_weights: "weights/planner.ot".to_owned(),
            episodes: 10,
            horizon: None,
            rollout_length: DEFAULT_PLANNER_ROLLOUT_LENGTH,
            context_bars: None,
            tickers: None,
            split: PlannerDataSplit::Test,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlannerInferenceEpisode {
    pub ticker: String,
    pub start_bar: usize,
    pub steps: usize,
    pub reward_sum: f64,
    pub final_wealth_ratio: f64,
    pub commissions: f64,
    pub turnover_mean: f64,
    pub action_mean: f64,
    pub max_drawdown: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlannerInferenceSummary {
    pub episodes: Vec<PlannerInferenceEpisode>,
    pub mean_reward: f64,
    pub mean_final_wealth_ratio: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct OptimizationSummary {
    actor_loss: f64,
    critic_loss: f64,
    reverse_kl: f64,
    max_reverse_kl: f64,
    entropy: f64,
    actor_grad_norm: f64,
    critic_grad_norm: f64,
    real_beta_concentration: f64,
    fantasy_beta_concentration: f64,
    real_critic_explained_variance: f64,
    fantasy_critic_explained_variance: f64,
    kl_early_stopped: bool,
    actor_steps: usize,
    steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HeldOutMetrics {
    median_wealth_ratio: f64,
    mean_max_drawdown: f64,
    mean_turnover: f64,
}

impl HeldOutMetrics {
    fn eligible(self) -> bool {
        self.median_wealth_ratio.is_finite()
            && self.mean_max_drawdown <= VALIDATION_MAX_MEAN_DRAWDOWN
            && self.mean_turnover <= VALIDATION_MAX_MEAN_TURNOVER
    }
}

pub fn train_planner(args: TrainPlannerArgs) -> Result<()> {
    validate_train_args(&args)?;
    configure_cuda();
    let device = Device::cuda_if_available();
    let metadata_path = args
        .world_model_metadata
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| world_model_metadata_path(&args.world_model_weights));
    let world_model = LejepaWorldModel::load(&args.world_model_weights, metadata_path, device)?;
    let world_lineage = world_model.lineage_sha256().to_owned();
    let world_weights_hash = world_model.metadata().checkpoint_sha256.clone();
    let dataset = PlannerDataset::load_cached(args.tickers.as_deref())?;

    let mut planner_vs = nn::VarStore::new(device);
    let planner = WorldModelPlanner::new(&planner_vs.root());
    let resumed = match &args.planner_weights {
        Some(weights) => Some(load_planner_checkpoint(
            &mut planner_vs,
            weights,
            &world_lineage,
            Some(args.horizon),
            Some(args.seed),
        )?),
        None => None,
    };
    // Context length must agree with the checkpoint on resume: derive from the
    // checkpoint when the CLI is silent, and bail if the CLI contradicts it.
    let context_bars = match &resumed {
        Some(metadata) => {
            let requested = args.context_bars.unwrap_or(metadata.context_bars);
            if requested != metadata.context_bars {
                bail!(
                    "planner resume context_bars mismatch: checkpoint={}, requested={requested}",
                    metadata.context_bars
                );
            }
            planner_context_bars(world_model.metadata(), Some(metadata.context_bars))?
        }
        None => planner_context_bars(world_model.metadata(), args.context_bars)?,
    };
    let mut optimizer_steps = resumed.as_ref().map(|m| m.optimizer_steps).unwrap_or(0);
    let base_updates = resumed.as_ref().map(|m| m.cumulative_updates).unwrap_or(0);
    let mut kl_controller = KlLrController::new(
        TARGET_KL,
        KL_CONTROLLER_HALF_LIFE,
        KL_MIN_LR_SCALE,
        KL_MAX_LR_SCALE,
    );
    if let Some(metadata) = &resumed {
        kl_controller.restore(metadata.kl_ema, metadata.kl_lr_scale);
    }
    let named_vars = named_trainable_variables(&planner_vs);
    let trainable_vars = named_vars
        .iter()
        .map(|(_, tensor)| tensor.shallow_clone())
        .collect::<Vec<_>>();
    let clip_groups = grad_clip_groups(&named_vars);
    let mut optimizer = Muon::new_named(
        &named_vars,
        MuonConfig {
            lr: MUON_LR,
            use_muon_for_2d: USE_MUON,
            momentum: MUON_MOMENTUM_WARMUP_START,
            adamw_lr: LEARNING_RATE,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            force_adamw_name_substrings: vec![
                "policy_concentration".to_owned(),
                "value_projection".to_owned(),
            ],
            ..MuonConfig::default()
        },
    );
    if let Some(metadata) = &resumed {
        if let Some(weights) = &args.planner_weights {
            let optimizer_state = planner_optimizer_state_path(weights);
            if optimizer_state.exists() {
                verify_optimizer_state(&optimizer_state, &metadata.optimizer_sha256)?;
                optimizer.load_state(&optimizer_state)?;
            } else if !metadata.optimizer_sha256.is_empty() {
                bail!(
                    "planner optimizer state {} is required by checkpoint metadata",
                    optimizer_state.display()
                );
            } else {
                // Moments and step_count are zero in a fresh optimizer; reset the
                // warmup counter to match so the momentum schedule restarts from
                // MUON_MOMENTUM_WARMUP_START instead of applying steady-state beta1
                // to zeroed EMA buffers.
                optimizer_steps = 0;
                eprintln!(
                    "warning: planner optimizer state {} absent; restarting optimizer moments and momentum warmup from zero",
                    optimizer_state.display()
                );
            }
        }
    }
    let hl_gauss = HlGaussBins::default_for(device);
    let validation_endpoints = dataset.deterministic_ticker_stratified_endpoints(
        PlannerDataSplit::Validation,
        HELD_OUT_ENDPOINTS,
        context_bars,
        args.rollout_length,
    )?;
    let best_path = best_checkpoint_path(&args.output);
    let compatible_existing_best = best_path.exists()
        && PlannerCheckpointMetadata::load(planner_metadata_path(&best_path))
            .and_then(|metadata| {
                metadata.validate(&world_lineage, Some(args.horizon), Some(args.seed))
            })
            .is_ok();
    let mut best_validation = compatible_existing_best
        .then(|| load_best_validation_metrics(&args.output))
        .flatten();
    let mut best_selected_this_run = false;
    println!(
        "planner training: device={device:?} updates={} H={} T={} N={} context={} (stateful world-model KV cache enabled)",
        args.updates, args.horizon, args.rollout_length, args.environments, context_bars
    );
    for update in 0..args.updates {
        let global_update = base_updates + update as u64 + 1;
        let rollout_seed = update_seed(args.seed, global_update, 0x524f4c4c4f5554);
        let mut rng = StdRng::seed_from_u64(rollout_seed);
        tch::manual_seed(rollout_seed as i64);
        let rollout = collect_mixed_rollout(
            &planner,
            &world_model,
            &dataset,
            &hl_gauss,
            args.horizon,
            args.rollout_length,
            args.environments,
            context_bars,
            &mut rng,
            device,
        )?;
        let rollout_metrics = rollout.metrics();
        let batch = rollout.to_batch(device)?;
        let (advantages, returns) = rollout_advantages(
            &rollout,
            &batch,
            args.rollout_length,
            args.environments,
            device,
        )?;
        let optimization = optimize_rollout(
            &planner,
            &rollout,
            &batch,
            &advantages,
            &returns,
            args.minibatch_size,
            update_seed(args.seed, global_update, 0x4f5054494d495a45),
            &hl_gauss,
            &clip_groups,
            &trainable_vars,
            &mut optimizer,
            &mut optimizer_steps,
            &mut kl_controller,
            device,
        )?;
        print_training_metrics(global_update, &rollout_metrics, optimization);
        append_training_metrics(&args.output, global_update, &rollout_metrics, optimization)?;
        save_planner_checkpoint(
            &planner_vs,
            &args.output,
            &PlannerCheckpointMetadata::new(
                &world_lineage,
                &world_weights_hash,
                args.horizon,
                context_bars,
                optimizer_steps,
                global_update,
                args.seed,
                kl_controller.ema(),
                kl_controller.scale(),
            ),
            &optimizer,
        )?;
        if global_update % VALIDATION_EVERY_UPDATES == 0 || update + 1 == args.updates {
            let (validation_summary, validation_metrics) = evaluate_real_endpoints(
                &planner,
                &world_model,
                &dataset,
                &validation_endpoints,
                args.horizon,
                args.rollout_length,
                context_bars,
                device,
            )?;
            let selected = validation_metrics.eligible()
                && best_validation.is_none_or(|best| {
                    validation_metrics.median_wealth_ratio > best.median_wealth_ratio
                });
            append_validation_metrics(&args.output, global_update, validation_metrics, selected)?;
            println!(
                "planner validation update={global_update} episodes={} median_wealth={:.6} mean_drawdown={:.6} mean_turnover={:.6} eligible={} selected={selected}",
                validation_summary.episodes.len(),
                validation_metrics.median_wealth_ratio,
                validation_metrics.mean_max_drawdown,
                validation_metrics.mean_turnover,
                validation_metrics.eligible(),
            );
            if selected {
                save_planner_checkpoint(
                    &planner_vs,
                    &best_path,
                    &PlannerCheckpointMetadata::new(
                        &world_lineage,
                        &world_weights_hash,
                        args.horizon,
                        context_bars,
                        optimizer_steps,
                        global_update,
                        args.seed,
                        kl_controller.ema(),
                        kl_controller.scale(),
                    ),
                    &optimizer,
                )?;
                best_validation = Some(validation_metrics);
                best_selected_this_run = true;
            }
        }
    }
    if best_selected_this_run {
        load_planner_checkpoint(
            &mut planner_vs,
            &best_path,
            &world_lineage,
            Some(args.horizon),
            None,
        )?;
        let test_endpoints = dataset.deterministic_ticker_stratified_endpoints(
            PlannerDataSplit::Test,
            HELD_OUT_ENDPOINTS,
            context_bars,
            args.rollout_length,
        )?;
        let (test_summary, test_metrics) = evaluate_real_endpoints(
            &planner,
            &world_model,
            &dataset,
            &test_endpoints,
            args.horizon,
            args.rollout_length,
            context_bars,
            device,
        )?;
        write_inference_csv(&best_path, PlannerDataSplit::Test, &test_summary)?;
        println!(
            "planner selected test episodes={} median_wealth={:.6} mean_drawdown={:.6} mean_turnover={:.6}",
            test_summary.episodes.len(),
            test_metrics.median_wealth_ratio,
            test_metrics.mean_max_drawdown,
            test_metrics.mean_turnover,
        );
    } else if best_validation.is_none() {
        eprintln!("planner produced no checkpoint passing real validation drawdown/turnover gates");
    }
    Ok(())
}

pub fn infer_planner(args: InferPlannerArgs) -> Result<PlannerInferenceSummary> {
    if args.episodes == 0 || args.rollout_length == 0 {
        bail!("planner inference episodes and rollout length must be positive");
    }
    configure_cuda();
    let device = Device::cuda_if_available();
    let metadata_path = args
        .world_model_metadata
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| world_model_metadata_path(&args.world_model_weights));
    let world_model = LejepaWorldModel::load(&args.world_model_weights, metadata_path, device)?;
    let world_lineage = world_model.lineage_sha256().to_owned();
    let mut planner_vs = nn::VarStore::new(device);
    let planner = WorldModelPlanner::new(&planner_vs.root());
    let checkpoint_metadata = load_planner_checkpoint(
        &mut planner_vs,
        &args.planner_weights,
        &world_lineage,
        args.horizon,
        None,
    )?;
    planner_vs.freeze();
    let horizon = args.horizon.unwrap_or(checkpoint_metadata.horizon);
    let context_bars = planner_context_bars(
        world_model.metadata(),
        args.context_bars.or(Some(checkpoint_metadata.context_bars)),
    )?;
    let dataset = PlannerDataset::load_cached(args.tickers.as_deref())?;
    let endpoints = dataset.deterministic_endpoints(
        args.split,
        args.episodes,
        context_bars,
        args.rollout_length,
    )?;
    let mut episodes = Vec::with_capacity(endpoints.len());

    for endpoint in endpoints {
        episodes.push(infer_real_episode(
            &planner,
            &world_model,
            &dataset,
            endpoint,
            horizon,
            args.rollout_length,
            context_bars,
            device,
        )?);
    }
    let mean_reward = episodes
        .iter()
        .map(|episode| episode.reward_sum)
        .sum::<f64>()
        / episodes.len() as f64;
    let mean_final_wealth_ratio = episodes
        .iter()
        .map(|episode| episode.final_wealth_ratio)
        .sum::<f64>()
        / episodes.len() as f64;
    println!(
        "planner held-out {:?}: episodes={} mean_reward={mean_reward:.6} mean_wealth={mean_final_wealth_ratio:.6}",
        args.split,
        episodes.len()
    );
    let summary = PlannerInferenceSummary {
        episodes,
        mean_reward,
        mean_final_wealth_ratio,
    };
    write_inference_csv(&args.planner_weights, args.split, &summary)?;
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn collect_mixed_rollout(
    planner: &WorldModelPlanner,
    world_model: &LejepaWorldModel,
    dataset: &PlannerDataset,
    hl_gauss: &HlGaussBins,
    horizon: usize,
    rollout_length: usize,
    environments: usize,
    context_bars: usize,
    rng: &mut StdRng,
    device: Device,
) -> Result<MixedRollout> {
    let per_source = environments / 2;
    let real_endpoints = dataset.sample_endpoints(
        PlannerDataSplit::Train,
        per_source,
        context_bars,
        rollout_length,
        rng,
    )?;
    let fantasy_endpoints =
        dataset.sample_endpoints(PlannerDataSplit::Train, per_source, context_bars, 0, rng)?;
    let fantasy_context = dataset.contexts(
        &fantasy_endpoints,
        &vec![0; per_source],
        context_bars,
        device,
    )?;
    let fantasy_session = world_model.start_session(&fantasy_context)?;
    let mut fantasy_prediction =
        fantasy_session.forecast(world_model, (horizon + rollout_length) as i64)?;
    validate_prediction_finite(&fantasy_prediction)?;
    let real_context =
        dataset.contexts(&real_endpoints, &vec![0; per_source], context_bars, device)?;
    let mut real_session = world_model.start_session(&real_context)?;
    let fantasy_clamp_mask = sanitize_fantasy_close_deltas(&mut fantasy_prediction)?;
    let fantasy_closes = fantasy_close_paths(dataset, &fantasy_endpoints, &fantasy_prediction)?;
    let mut portfolios = (0..environments)
        .map(|_| PlannerPortfolio::new(100.0))
        .collect::<Vec<_>>();
    let mut pending = (0..environments)
        .map(|_| None::<PlannerTransition>)
        .collect::<Vec<_>>();
    let world_lineage = world_model.lineage_sha256().to_owned();
    let mut rollout = MixedRollout::new(per_source * rollout_length, world_lineage)?;
    let relative_horizon = relative_horizon(environments, horizon, device);

    for decision in 0..=rollout_length {
        let real_prediction = real_session.forecast(world_model, horizon as i64)?;
        validate_prediction_finite(&real_prediction)?;
        let latent = Tensor::cat(
            &[
                real_prediction.latent.shallow_clone(),
                fantasy_prediction
                    .latent
                    .narrow(1, decision as i64, horizon as i64),
            ],
            0,
        );
        let mean = Tensor::cat(
            &[
                real_prediction.ohlc_mean.shallow_clone(),
                fantasy_prediction
                    .ohlc_mean
                    .narrow(1, decision as i64, horizon as i64),
            ],
            0,
        );
        let logvar = Tensor::cat(
            &[
                real_prediction.ohlc_logvar.shallow_clone(),
                fantasy_prediction
                    .ohlc_logvar
                    .narrow(1, decision as i64, horizon as i64),
            ],
            0,
        );
        let current_prices = current_prices(
            dataset,
            &real_endpoints,
            &fantasy_endpoints,
            &fantasy_closes,
            decision,
        );
        let portfolio_state = Tensor::from_slice(
            &portfolios
                .iter()
                .zip(&current_prices)
                .flat_map(|(portfolio, &price)| portfolio.planner_state(price))
                .collect::<Vec<_>>(),
        )
        .view([environments as i64, PLANNER_PORTFOLIO_DIM])
        .to_device(device);
        let output = tch::no_grad(|| {
            planner.forward_mixed_precision(&WorldModelPlannerInput {
                forecast: PlannerForecast {
                    latent: latent.shallow_clone(),
                    ohlc_mean: mean.shallow_clone(),
                    ohlc_log_variance: logvar.shallow_clone(),
                    relative_horizon: relative_horizon.shallow_clone(),
                },
                portfolio_state: portfolio_state.shallow_clone(),
            })
        });
        let sources = Tensor::arange(environments as i64, (Kind::Int64, device))
            .ge(per_source as i64)
            .to_kind(Kind::Int64);
        let routed_logits = route_value_logits(
            &output.real_value_logits,
            &output.fantasy_value_logits,
            &sources,
        );
        let values_cpu = hl_gauss.decode(&routed_logits).to_device(Device::Cpu);
        let stored_latent = latent.to_device(Device::Cpu).detach();
        let stored_mean = mean.to_device(Device::Cpu).detach();
        let stored_logvar = logvar.to_device(Device::Cpu).detach();
        let stored_horizon = relative_horizon.to_device(Device::Cpu).detach();
        let stored_portfolio = portfolio_state.to_device(Device::Cpu).detach();

        for environment in 0..environments {
            if let Some(mut previous) = pending[environment].take() {
                previous.next_value = Some(values_cpu.get(environment as i64));
                previous.truncated = decision == rollout_length;
                rollout.push(previous)?;
            }
        }
        if decision == rollout_length {
            break;
        }

        let actions = sample_beta_action(&output.alpha, &output.beta);
        let log_probs = beta_log_prob(&actions, &output.alpha, &output.beta);
        let actions_cpu = actions.to_device(Device::Cpu);
        let alpha_cpu = output.alpha.to_device(Device::Cpu);
        let beta_cpu = output.beta.to_device(Device::Cpu);
        let log_probs_cpu = log_probs.to_device(Device::Cpu);
        for environment in 0..environments {
            let source = if environment < per_source {
                RolloutSource::Real
            } else {
                RolloutSource::Fantasy
            };
            let next_price = if environment < per_source {
                let endpoint = real_endpoints[environment];
                dataset.series(endpoint.series).closes[endpoint.bar + decision + 1]
            } else {
                fantasy_closes[environment - per_source][decision + 1]
            };
            let action = actions_cpu.double_value(&[environment as i64, 0]);
            let step =
                portfolios[environment].step(action, current_prices[environment], next_price);
            pending[environment] = Some(PlannerTransition {
                observation: observation_at(
                    &stored_latent,
                    &stored_mean,
                    &stored_logvar,
                    &stored_horizon,
                    &stored_portfolio,
                    environment,
                ),
                source,
                environment_id: environment,
                decision_index: decision,
                action: actions_cpu.get(environment as i64),
                old_alpha: alpha_cpu.get(environment as i64),
                old_beta: beta_cpu.get(environment as i64),
                old_log_prob: log_probs_cpu.get(environment as i64),
                value: values_cpu.get(environment as i64),
                next_value: None,
                reward: step.reward as f32,
                // Long-only, no-leverage portfolio: total assets stay strictly
                // positive every step, so an episode never terminates; it can
                // only truncate at the rollout horizon.
                terminated: false,
                truncated: false,
                commission: step.commission,
                turnover: step.turnover,
                assets_before: step.assets_before,
                assets_after: step.assets_after,
                fantasy_clamped: source == RolloutSource::Fantasy
                    && fantasy_clamp_mask[environment - per_source][decision],
            });
        }
        let actual_next_bars =
            dataset.contexts(&real_endpoints, &vec![decision + 1; per_source], 1, device)?;
        real_session.append_actual_bar(world_model, &actual_next_bars)?;
    }
    rollout.validate_complete()?;
    Ok(rollout)
}

#[allow(clippy::too_many_arguments)]
fn optimize_rollout(
    planner: &WorldModelPlanner,
    rollout: &MixedRollout,
    batch: &PlannerBatch,
    advantages: &Tensor,
    returns: &Tensor,
    minibatch_size: usize,
    seed: u64,
    hl_gauss: &HlGaussBins,
    clip_groups: &crate::torch::train::optimizer_glue::GradClipGroups,
    trainable_vars: &[Tensor],
    optimizer: &mut Muon,
    optimizer_steps: &mut u64,
    kl_controller: &mut KlLrController,
    device: Device,
) -> Result<OptimizationSummary> {
    let mut summary = OptimizationSummary::default();
    let diagnostics = split_critic_diagnostics(
        &batch.sources,
        &batch.old_alpha,
        &batch.old_beta,
        &batch.values,
        returns,
    );
    summary.real_beta_concentration = diagnostics.real.beta_concentration_mean;
    summary.fantasy_beta_concentration = diagnostics.fantasy.beta_concentration_mean;
    summary.real_critic_explained_variance = diagnostics.real.critic_explained_variance;
    summary.fantasy_critic_explained_variance = diagnostics.fantasy.critic_explained_variance;

    let mut actor_loss_sum = Tensor::zeros([], (Kind::Float, device));
    let mut critic_loss_sum = Tensor::zeros([], (Kind::Float, device));
    let mut reverse_kl_sum = Tensor::zeros([], (Kind::Float, device));
    let mut entropy_sum = Tensor::zeros([], (Kind::Float, device));
    let mut actor_grad_sum = Tensor::zeros([], (Kind::Float, device));
    let mut critic_grad_sum = Tensor::zeros([], (Kind::Float, device));
    apply_lr_scale(optimizer, kl_controller.scale());
    let mut actor_stopped = false;
    for epoch in 0..DEFAULT_PLANNER_OPTIMIZATION_EPOCHS {
        for indices in
            rollout.balanced_minibatch_indices(minibatch_size, seed ^ ((epoch as u64 + 1) << 32))?
        {
            let mini = batch.select(&indices);
            let index = Tensor::from_slice(&indices).to_device(device);
            let mini_advantages = advantages.index_select(0, &index);
            let mini_returns = returns.index_select(0, &index);
            let output = planner.forward_mixed_precision(&planner_input(&mini));
            let losses = planner_actor_critic_losses(
                hl_gauss,
                &output.real_value_logits,
                &output.fantasy_value_logits,
                &mini.sources,
                &output.alpha,
                &output.beta,
                &mini.actions,
                &mini.old_alpha,
                &mini.old_beta,
                &mini_advantages,
                &mini_returns,
            );
            let minibatch_kl = losses.reverse_kl.double_value(&[]);
            if !actor_stopped {
                summary.max_reverse_kl = summary.max_reverse_kl.max(minibatch_kl);
            }
            if !actor_stopped && summary.actor_steps > 0 && minibatch_kl > TARGET_KL {
                actor_stopped = true;
                summary.kl_early_stopped = true;
            }
            optimizer.zero_grad();
            let (actor_norm, critic_norm) = backward_actor_critic_with_separate_clips(
                clip_groups,
                trainable_vars,
                &losses.actor_loss,
                &losses.critic_loss,
                MAX_GRAD_NORM,
                device,
                actor_stopped,
            );
            optimizer.set_momentum(muon_momentum_for_step(*optimizer_steps as i64));
            optimizer.step();
            *optimizer_steps += 1;
            if !actor_stopped {
                actor_loss_sum += losses.actor_loss.detach();
                reverse_kl_sum += losses.reverse_kl.detach();
                entropy_sum += losses.entropy.detach();
                actor_grad_sum += actor_norm.detach();
                summary.actor_steps += 1;
            }
            critic_loss_sum += losses.critic_loss.detach();
            critic_grad_sum += critic_norm.detach();
            summary.steps += 1;
        }
    }
    let critic_denominator = summary.steps as f64;
    let actor_denominator = summary.actor_steps.max(1) as f64;
    summary.actor_loss = actor_loss_sum.double_value(&[]) / actor_denominator;
    summary.critic_loss = critic_loss_sum.double_value(&[]) / critic_denominator;
    summary.reverse_kl = reverse_kl_sum.double_value(&[]) / actor_denominator;
    summary.entropy = entropy_sum.double_value(&[]) / actor_denominator;
    summary.actor_grad_norm = actor_grad_sum.double_value(&[]) / actor_denominator;
    summary.critic_grad_norm = critic_grad_sum.double_value(&[]) / critic_denominator;
    kl_controller.observe(summary.max_reverse_kl);
    Ok(summary)
}

fn rollout_advantages(
    rollout: &MixedRollout,
    batch: &PlannerBatch,
    rollout_length: usize,
    environments: usize,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let len = rollout_length * environments;
    if rollout.len() != len {
        bail!(
            "planner rollout is not dense: expected {len} transitions, got {}",
            rollout.len()
        );
    }
    // Map each transition (in push order) to its [decision, environment] slot and
    // assert the rollout is a dense rectangle: every slot filled exactly once.
    let mut slots = Vec::with_capacity(len);
    let mut seen = vec![false; len];
    for transition in rollout.transitions() {
        let slot = transition.decision_index * environments + transition.environment_id;
        if slot >= len || seen[slot] {
            bail!("planner rollout is not a dense [rollout_length, environments] grid");
        }
        seen[slot] = true;
        slots.push(slot as i64);
    }
    let slot_index = Tensor::from_slice(&slots).to_device(device);
    let shape = [rollout_length as i64, environments as i64];
    // Scatter push-ordered batch fields into grid order, run GAE on-device, then
    // gather advantages/returns back into push order — no host readback.
    let to_grid = |source: &Tensor| {
        Tensor::zeros([len as i64], (Kind::Float, device)).index_copy(0, &slot_index, source)
    };
    let (advantages, returns) = compute_planner_gae(
        &to_grid(&batch.rewards).view(shape),
        &to_grid(&batch.values).view(shape),
        &to_grid(&batch.next_values).view(shape),
        &to_grid(&batch.terminated).view(shape),
        &to_grid(&batch.truncated).view(shape),
    );
    Ok((
        advantages.view([len as i64]).index_select(0, &slot_index),
        returns.view([len as i64]).index_select(0, &slot_index),
    ))
}

#[allow(clippy::too_many_arguments)]
fn infer_real_episode(
    planner: &WorldModelPlanner,
    world_model: &LejepaWorldModel,
    dataset: &PlannerDataset,
    endpoint: PlannerEndpoint,
    horizon: usize,
    rollout_length: usize,
    context_bars: usize,
    device: Device,
) -> Result<PlannerInferenceEpisode> {
    let mut portfolio = PlannerPortfolio::new(100.0);
    let mut reward_sum = 0.0;
    let mut turnover = 0.0;
    let mut action_sum = 0.0;
    let mut peak_assets: f64 = 100.0;
    let mut max_drawdown: f64 = 0.0;
    let relative_horizon = relative_horizon(1, horizon, device);
    let context = dataset.contexts(&[endpoint], &[0], context_bars, device)?;
    let mut session = world_model.start_session(&context)?;
    for decision in 0..rollout_length {
        let prediction = session.forecast(world_model, horizon as i64)?;
        validate_prediction_finite(&prediction).with_context(|| {
            format!(
                "invalid world-model forecast for {} at decision {decision}",
                dataset.series(endpoint.series).ticker
            )
        })?;
        let series = dataset.series(endpoint.series);
        let current_price = series.closes[endpoint.bar + decision];
        let portfolio_state = Tensor::from_slice(&portfolio.planner_state(current_price))
            .view([1, PLANNER_PORTFOLIO_DIM])
            .to_device(device);
        let output = tch::no_grad(|| {
            planner.forward_mixed_precision(&WorldModelPlannerInput {
                forecast: PlannerForecast {
                    latent: prediction.latent,
                    ohlc_mean: prediction.ohlc_mean,
                    ohlc_log_variance: prediction.ohlc_logvar,
                    relative_horizon: relative_horizon.shallow_clone(),
                },
                portfolio_state,
            })
        });
        let action = beta_mean(&output.alpha, &output.beta).double_value(&[0, 0]);
        let step = portfolio.step(
            action,
            current_price,
            series.closes[endpoint.bar + decision + 1],
        );
        reward_sum += step.reward;
        turnover += step.turnover;
        action_sum += action;
        peak_assets = peak_assets.max(step.assets_after);
        max_drawdown = max_drawdown.max(1.0 - step.assets_after / peak_assets);
        let actual_next_bar = dataset.contexts(&[endpoint], &[decision + 1], 1, device)?;
        session.append_actual_bar(world_model, &actual_next_bar)?;
    }
    let series = dataset.series(endpoint.series);
    Ok(PlannerInferenceEpisode {
        ticker: series.ticker.clone(),
        start_bar: endpoint.bar,
        steps: rollout_length,
        reward_sum,
        final_wealth_ratio: portfolio.total_assets(series.closes[endpoint.bar + rollout_length])
            / 100.0,
        commissions: portfolio.total_commissions,
        turnover_mean: turnover / rollout_length as f64,
        action_mean: action_sum / rollout_length as f64,
        max_drawdown,
    })
}

fn planner_input(batch: &PlannerBatch) -> WorldModelPlannerInput {
    WorldModelPlannerInput {
        forecast: PlannerForecast {
            latent: batch.forecast_latent.shallow_clone(),
            ohlc_mean: batch.forecast_mean.shallow_clone(),
            ohlc_log_variance: batch.forecast_logvar.shallow_clone(),
            relative_horizon: batch.relative_horizon.shallow_clone(),
        },
        portfolio_state: batch.portfolio_state.shallow_clone(),
    }
}

fn relative_horizon(batch: usize, horizon: usize, device: Device) -> Tensor {
    (Tensor::arange(horizon as i64, (Kind::Float, device)) + 1.0)
        .view([1, horizon as i64, 1])
        .expand([batch as i64, horizon as i64, 1], false)
        / horizon as f64
}

fn update_seed(base_seed: u64, cumulative_update: u64, stream: u64) -> u64 {
    let mut value = base_seed ^ cumulative_update.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ stream;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn observation_at(
    latent: &Tensor,
    mean: &Tensor,
    logvar: &Tensor,
    relative_horizon: &Tensor,
    portfolio_state: &Tensor,
    index: usize,
) -> PlannerObservation {
    let index = index as i64;
    PlannerObservation {
        forecast_latent: latent.get(index).to_device(Device::Cpu).detach(),
        forecast_mean: mean.get(index).to_device(Device::Cpu).detach(),
        forecast_logvar: logvar.get(index).to_device(Device::Cpu).detach(),
        relative_horizon: relative_horizon.get(index).to_device(Device::Cpu).detach(),
        portfolio_state: portfolio_state.get(index).to_device(Device::Cpu).detach(),
    }
}

fn fantasy_close_paths(
    dataset: &PlannerDataset,
    endpoints: &[PlannerEndpoint],
    prediction: &WorldModelPrediction,
) -> Result<Vec<Vec<f64>>> {
    let means = prediction.ohlc_mean.to_device(Device::Cpu);
    let steps = means.size()[1] as usize;
    let mut result = Vec::with_capacity(endpoints.len());
    for (batch, endpoint) in endpoints.iter().enumerate() {
        let mut closes = Vec::with_capacity(steps + 1);
        closes.push(dataset.series(endpoint.series).closes[endpoint.bar]);
        for step in 0..steps {
            let delta = means.double_value(&[batch as i64, step as i64, 3]);
            let multiplier = 1.0 + delta;
            let next = closes.last().copied().unwrap() * multiplier;
            if !next.is_finite() || next <= 0.0 {
                bail!("fantasy close path became invalid for batch {batch} at step {step}: {next}");
            }
            closes.push(next);
        }
        result.push(closes);
    }
    Ok(result)
}

fn sanitize_fantasy_close_deltas(prediction: &mut WorldModelPrediction) -> Result<Vec<Vec<bool>>> {
    let close = prediction.ohlc_mean.narrow(2, 3, 1);
    if close.isfinite().all().int64_value(&[]) == 0 {
        bail!("fantasy close prediction contains NaN or infinity");
    }
    let clamped = close.clamp(FANTASY_CLOSE_DELTA_MIN, FANTASY_CLOSE_DELTA_MAX);
    let changed = close.ne_tensor(&clamped).to_device(Device::Cpu);
    tch::no_grad(|| prediction.ohlc_mean.narrow(2, 3, 1).copy_(&clamped));
    let batch = changed.size()[0] as usize;
    let steps = changed.size()[1] as usize;
    Ok((0..batch)
        .map(|b| {
            (0..steps)
                .map(|step| changed.int64_value(&[b as i64, step as i64, 0]) != 0)
                .collect()
        })
        .collect())
}

fn validate_prediction_finite(prediction: &WorldModelPrediction) -> Result<()> {
    for (name, tensor) in [
        ("latent", &prediction.latent),
        ("OHLC mean", &prediction.ohlc_mean),
        ("OHLC log-variance", &prediction.ohlc_logvar),
    ] {
        if tensor.isfinite().all().int64_value(&[]) == 0 {
            bail!("world-model {name} prediction contains NaN or infinity");
        }
    }
    Ok(())
}

fn current_prices(
    dataset: &PlannerDataset,
    real: &[PlannerEndpoint],
    fantasy: &[PlannerEndpoint],
    fantasy_closes: &[Vec<f64>],
    decision: usize,
) -> Vec<f64> {
    real.iter()
        .map(|endpoint| dataset.series(endpoint.series).closes[endpoint.bar + decision])
        .chain(
            fantasy
                .iter()
                .enumerate()
                .map(|(index, _)| fantasy_closes[index][decision]),
        )
        .collect()
}

fn validate_train_args(args: &TrainPlannerArgs) -> Result<()> {
    if args.updates == 0 || args.horizon == 0 || args.rollout_length == 0 {
        bail!("planner updates, horizon, and rollout length must be positive");
    }
    if args.environments == 0 || args.environments % 2 != 0 {
        bail!("planner environments must be positive and even for equal real/fantasy sources");
    }
    let samples = args.environments * args.rollout_length;
    let per_source = samples / 2;
    if args.minibatch_size == 0
        || args.minibatch_size % 2 != 0
        || samples % args.minibatch_size != 0
        || per_source % (args.minibatch_size / 2) != 0
    {
        bail!("planner minibatch size must be even and evenly partition both rollout sources");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_real_endpoints(
    planner: &WorldModelPlanner,
    world_model: &LejepaWorldModel,
    dataset: &PlannerDataset,
    endpoints: &[PlannerEndpoint],
    horizon: usize,
    rollout_length: usize,
    context_bars: usize,
    device: Device,
) -> Result<(PlannerInferenceSummary, HeldOutMetrics)> {
    let mut episodes = Vec::with_capacity(endpoints.len());
    for &endpoint in endpoints {
        episodes.push(infer_real_episode(
            planner,
            world_model,
            dataset,
            endpoint,
            horizon,
            rollout_length,
            context_bars,
            device,
        )?);
    }
    let count = episodes.len() as f64;
    let mean_reward = episodes
        .iter()
        .map(|episode| episode.reward_sum)
        .sum::<f64>()
        / count;
    let mean_final_wealth_ratio = episodes
        .iter()
        .map(|episode| episode.final_wealth_ratio)
        .sum::<f64>()
        / count;
    let mut wealth = episodes
        .iter()
        .map(|episode| episode.final_wealth_ratio)
        .collect::<Vec<_>>();
    wealth.sort_by(f64::total_cmp);
    let median_wealth_ratio = if wealth.len() % 2 == 0 {
        (wealth[wealth.len() / 2 - 1] + wealth[wealth.len() / 2]) * 0.5
    } else {
        wealth[wealth.len() / 2]
    };
    let metrics = HeldOutMetrics {
        median_wealth_ratio,
        mean_max_drawdown: episodes
            .iter()
            .map(|episode| episode.max_drawdown)
            .sum::<f64>()
            / count,
        mean_turnover: episodes
            .iter()
            .map(|episode| episode.turnover_mean)
            .sum::<f64>()
            / count,
    };
    Ok((
        PlannerInferenceSummary {
            episodes,
            mean_reward,
            mean_final_wealth_ratio,
        },
        metrics,
    ))
}

fn best_checkpoint_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    let checkpoint = checkpoint.as_ref();
    let stem = checkpoint
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("planner");
    checkpoint.with_file_name(format!("{stem}_best.ot"))
}

fn validation_metrics_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("validation.csv")
}

fn append_validation_metrics(
    checkpoint: impl AsRef<Path>,
    update: u64,
    metrics: HeldOutMetrics,
    selected: bool,
) -> Result<()> {
    let path = validation_metrics_path(checkpoint);
    let write_header = fs::metadata(&path)
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(true);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if write_header {
        writeln!(
            file,
            "update,median_wealth_ratio,mean_max_drawdown,mean_turnover,eligible,selected"
        )?;
    }
    writeln!(
        file,
        "{update},{},{},{},{},{}",
        metrics.median_wealth_ratio,
        metrics.mean_max_drawdown,
        metrics.mean_turnover,
        metrics.eligible(),
        selected,
    )?;
    Ok(())
}

fn load_best_validation_metrics(checkpoint: impl AsRef<Path>) -> Option<HeldOutMetrics> {
    let contents = fs::read_to_string(validation_metrics_path(checkpoint)).ok()?;
    contents
        .lines()
        .skip(1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .find_map(|line| {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.len() != 6 || fields[5] != "true" {
                return None;
            }
            Some(HeldOutMetrics {
                median_wealth_ratio: fields[1].parse().ok()?,
                mean_max_drawdown: fields[2].parse().ok()?,
                mean_turnover: fields[3].parse().ok()?,
            })
        })
}

fn print_training_metrics(
    update: u64,
    rollout: &RolloutMetrics,
    optimization: OptimizationSummary,
) {
    println!(
        "planner update={update} real_reward={:.6} fantasy_reward={:.6} real_wealth={:.6} fantasy_wealth={:.6} real_turnover={:.6} fantasy_turnover={:.6} fantasy_clamp={:.6} actor_loss={:.6} critic_loss={:.6} kl={:.6} max_kl={:.6} kl_stop={} entropy={:.6} real_critic_ev={:.4} fantasy_critic_ev={:.4} actor_grad={:.6} critic_grad={:.6}",
        rollout.real.reward_mean,
        rollout.fantasy.reward_mean,
        rollout.real.mean_environment_wealth_ratio,
        rollout.fantasy.mean_environment_wealth_ratio,
        rollout.real.turnover_mean,
        rollout.fantasy.turnover_mean,
        rollout.fantasy.fantasy_clamp_fraction,
        optimization.actor_loss,
        optimization.critic_loss,
        optimization.reverse_kl,
        optimization.max_reverse_kl,
        optimization.kl_early_stopped,
        optimization.entropy,
        optimization.real_critic_explained_variance,
        optimization.fantasy_critic_explained_variance,
        optimization.actor_grad_norm,
        optimization.critic_grad_norm,
    );
}

fn append_training_metrics(
    planner_checkpoint: impl AsRef<Path>,
    update: u64,
    rollout: &RolloutMetrics,
    optimization: OptimizationSummary,
) -> Result<()> {
    let path = planner_checkpoint.as_ref().with_extension("training.csv");
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let write_header = fs::metadata(&path)
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(true);
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    if write_header {
        writeln!(
            file,
            "update,real_reward_mean,fantasy_reward_mean,real_wealth_ratio,fantasy_wealth_ratio,real_turnover_mean,fantasy_turnover_mean,real_commissions,fantasy_commissions,real_action_mean,fantasy_action_mean,real_action_boundary_fraction,fantasy_action_boundary_fraction,fantasy_clamp_fraction,real_beta_concentration,fantasy_beta_concentration,real_critic_explained_variance,fantasy_critic_explained_variance,actor_loss,critic_loss,reverse_kl,max_reverse_kl,kl_early_stopped,entropy,actor_grad_norm,critic_grad_norm"
        )?;
    }
    writeln!(
        file,
        "{update},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        rollout.real.reward_mean,
        rollout.fantasy.reward_mean,
        rollout.real.mean_environment_wealth_ratio,
        rollout.fantasy.mean_environment_wealth_ratio,
        rollout.real.turnover_mean,
        rollout.fantasy.turnover_mean,
        rollout.real.commissions,
        rollout.fantasy.commissions,
        rollout.real.action_mean,
        rollout.fantasy.action_mean,
        rollout.real.action_boundary_fraction,
        rollout.fantasy.action_boundary_fraction,
        rollout.fantasy.fantasy_clamp_fraction,
        optimization.real_beta_concentration,
        optimization.fantasy_beta_concentration,
        optimization.real_critic_explained_variance,
        optimization.fantasy_critic_explained_variance,
        optimization.actor_loss,
        optimization.critic_loss,
        optimization.reverse_kl,
        optimization.max_reverse_kl,
        optimization.kl_early_stopped,
        optimization.entropy,
        optimization.actor_grad_norm,
        optimization.critic_grad_norm,
    )?;
    Ok(())
}

fn write_inference_csv(
    planner_checkpoint: impl AsRef<Path>,
    split: PlannerDataSplit,
    summary: &PlannerInferenceSummary,
) -> Result<()> {
    let path = planner_checkpoint.as_ref().with_extension("inference.csv");
    let mut file = File::create(&path)?;
    writeln!(
        file,
        "split,episode,ticker,start_bar,steps,reward_sum,final_wealth_ratio,commissions,turnover_mean,action_mean,max_drawdown"
    )?;
    for (index, episode) in summary.episodes.iter().enumerate() {
        writeln!(
            file,
            "{split:?},{index},{},{},{},{},{},{},{},{},{}",
            episode.ticker,
            episode.start_bar,
            episode.steps,
            episode.reward_sum,
            episode.final_wealth_ratio,
            episode.commissions,
            episode.turnover_mean,
            episode.action_mean,
            episode.max_drawdown,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_mixed_rollout_plan() {
        let args = TrainPlannerArgs::default();
        assert_eq!(args.horizon, 100);
        assert_eq!(args.rollout_length, 100);
        assert_eq!(args.environments, 16);
        validate_train_args(&args).unwrap();
    }

    #[test]
    fn rejects_unbalanced_environment_count() {
        let mut args = TrainPlannerArgs::default();
        args.environments = 15;
        assert!(validate_train_args(&args).is_err());
    }

    #[test]
    fn fantasy_prices_use_next_close_delta_without_off_by_one() {
        let previous = 100.0f64;
        let deltas = [0.1f64, -0.5];
        let mut prices = vec![previous];
        for delta in deltas {
            prices.push(prices.last().copied().unwrap() * (1.0 + delta));
        }
        assert_eq!(prices, vec![100.0, 110.00000000000001, 55.00000000000001]);
    }

    #[test]
    fn fantasy_sanitization_changes_both_presented_and_executed_close_delta() {
        let mut prediction = WorldModelPrediction {
            latent: Tensor::zeros([1, 2, 256], (Kind::Float, Device::Cpu)),
            ohlc_mean: Tensor::zeros([1, 2, 16], (Kind::Float, Device::Cpu)),
            ohlc_logvar: Tensor::zeros([1, 2, 16], (Kind::Float, Device::Cpu)),
        };
        let _ = prediction.ohlc_mean.get(0).get(0).get(3).fill_(-2.0);
        let _ = prediction.ohlc_mean.get(0).get(1).get(3).fill_(120.0);
        let mask = sanitize_fantasy_close_deltas(&mut prediction).unwrap();
        assert_eq!(mask, vec![vec![true, true]]);
        assert!((prediction.ohlc_mean.double_value(&[0, 0, 3]) + 0.25).abs() < 1e-6);
        assert_eq!(prediction.ohlc_mean.double_value(&[0, 1, 3]), 0.25);
    }

    #[test]
    fn fantasy_sanitization_rejects_non_finite_close_delta() {
        let mut prediction = WorldModelPrediction {
            latent: Tensor::zeros([1, 1, 256], (Kind::Float, Device::Cpu)),
            ohlc_mean: Tensor::full([1, 1, 16], f64::NAN, (Kind::Float, Device::Cpu)),
            ohlc_logvar: Tensor::zeros([1, 1, 16], (Kind::Float, Device::Cpu)),
        };
        assert!(sanitize_fantasy_close_deltas(&mut prediction).is_err());
    }

    #[test]
    fn update_seeds_are_stable_and_do_not_repeat_after_resume() {
        let first = update_seed(7, 1, 11);
        assert_eq!(first, update_seed(7, 1, 11));
        assert_ne!(first, update_seed(7, 2, 11));
        assert_ne!(first, update_seed(7, 1, 12));
    }

    #[test]
    fn validation_gates_drawdown_and_turnover_before_wealth_selection() {
        let eligible = HeldOutMetrics {
            median_wealth_ratio: 1.1,
            mean_max_drawdown: 0.2,
            mean_turnover: 0.1,
        };
        assert!(eligible.eligible());
        assert!(!HeldOutMetrics {
            mean_max_drawdown: 0.31,
            ..eligible
        }
        .eligible());
        assert!(!HeldOutMetrics {
            mean_turnover: 0.51,
            ..eligible
        }
        .eligible());
    }

    #[test]
    fn validation_log_restores_only_the_last_selected_real_score() {
        let checkpoint = std::env::temp_dir().join(format!(
            "planner-validation-{}-{}.ot",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let first = HeldOutMetrics {
            median_wealth_ratio: 1.05,
            mean_max_drawdown: 0.1,
            mean_turnover: 0.2,
        };
        let later_unselected = HeldOutMetrics {
            median_wealth_ratio: 0.9,
            mean_max_drawdown: 0.1,
            mean_turnover: 0.2,
        };
        append_validation_metrics(&checkpoint, 50, first, true).unwrap();
        append_validation_metrics(&checkpoint, 100, later_unselected, false).unwrap();
        assert_eq!(load_best_validation_metrics(&checkpoint), Some(first));
        assert_eq!(
            best_checkpoint_path(&checkpoint),
            checkpoint.with_file_name(format!(
                "{}_best.ot",
                checkpoint.file_stem().unwrap().to_str().unwrap()
            ))
        );
        let _ = fs::remove_file(validation_metrics_path(&checkpoint));
    }

    #[test]
    fn training_diagnostics_append_one_header() {
        let checkpoint = std::env::temp_dir().join(format!(
            "planner-diagnostics-{}-{}.ot",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut metrics = RolloutMetrics::default();
        metrics.real.reward_mean = 1.0;
        metrics.fantasy.reward_mean = -1.0;
        append_training_metrics(&checkpoint, 1, &metrics, OptimizationSummary::default()).unwrap();
        append_training_metrics(&checkpoint, 2, &metrics, OptimizationSummary::default()).unwrap();
        let path = checkpoint.with_extension("training.csv");
        let contents = fs::read_to_string(&path).unwrap();
        assert_eq!(contents.lines().count(), 3);
        assert_eq!(contents.matches("update,real_reward_mean").count(), 1);
        let _ = fs::remove_file(path);
    }
}
