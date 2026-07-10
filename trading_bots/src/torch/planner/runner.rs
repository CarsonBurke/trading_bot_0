use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use rand::{rngs::StdRng, SeedableRng};
use tch::{nn, Device, Kind, Tensor};

use crate::torch::{
    action_space::{beta_log_prob, beta_mean, sample_beta_action},
    cuda::cfg::configure_cuda,
    optim::muon::{Muon, MuonConfig},
    train::{
        config::{LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON},
        optimizer_glue::{
            backward_actor_critic_with_separate_clips, grad_clip_groups, muon_momentum_for_step,
            named_trainable_variables,
        },
    },
    value::hl_gauss::HlGaussBins,
    world_model::{world_model_metadata_path, LejepaWorldModel, WorldModelPrediction},
};

use super::{
    checkpoint::{load_planner_checkpoint, save_planner_checkpoint, PlannerCheckpointMetadata},
    data::{planner_context_bars, PlannerDataSplit, PlannerDataset, PlannerEndpoint},
    gae::compute_default_planner_gae,
    losses::{planner_actor_critic_losses, split_optimization_metrics, PlannerLossConfig},
    portfolio::PlannerPortfolio,
    rollout::{
        MixedRollout, PlannerBatch, PlannerObservation, PlannerTransition, RolloutMetrics,
        RolloutSource,
    },
    PlannerForecast, WorldModelPlanner, WorldModelPlannerInput,
};

pub const DEFAULT_PLANNER_HORIZON: usize = 100;
pub const DEFAULT_PLANNER_ROLLOUT_LENGTH: usize = 100;
pub const DEFAULT_PLANNER_ENVIRONMENTS: usize = 16;
pub const DEFAULT_PLANNER_OPTIMIZATION_EPOCHS: usize = 3;
pub const DEFAULT_PLANNER_MINIBATCH_SIZE: usize = 160;

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
    entropy: f64,
    actor_grad_norm: f64,
    critic_grad_norm: f64,
    real_beta_concentration: f64,
    fantasy_beta_concentration: f64,
    real_critic_explained_variance: f64,
    fantasy_critic_explained_variance: f64,
    steps: usize,
}

struct PendingTransition {
    observation: PlannerObservation,
    source: RolloutSource,
    environment_id: usize,
    decision_index: usize,
    action: Tensor,
    old_alpha: Tensor,
    old_beta: Tensor,
    old_log_prob: Tensor,
    value: Tensor,
    reward: f32,
    commission: f64,
    turnover: f64,
    assets_before: f64,
    assets_after: f64,
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
    let world_hash = world_model.metadata().checkpoint_sha256.clone();
    let context_bars = planner_context_bars(world_model.metadata(), args.context_bars)?;
    let dataset = PlannerDataset::load_cached(args.tickers.as_deref())?;

    let mut planner_vs = nn::VarStore::new(device);
    let planner = WorldModelPlanner::new(&planner_vs.root());
    let mut optimizer_steps = 0u64;
    if let Some(weights) = &args.planner_weights {
        optimizer_steps =
            load_planner_checkpoint(&mut planner_vs, weights, &world_hash, Some(args.horizon))?
                .optimizer_steps;
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
    let hl_gauss = HlGaussBins::default_for(device);
    let mut rng = StdRng::seed_from_u64(args.seed);

    println!(
        "planner training: device={device:?} updates={} H={} T={} N={} context={} (stateful world-model KV cache enabled)",
        args.updates, args.horizon, args.rollout_length, args.environments, context_bars
    );
    for update in 0..args.updates {
        let rollout = collect_mixed_rollout(
            &planner,
            &world_model,
            &dataset,
            args.horizon,
            args.rollout_length,
            args.environments,
            context_bars,
            &mut rng,
            device,
        )?;
        let rollout_metrics = rollout.metrics();
        let (advantages, returns) =
            rollout_advantages(&rollout, args.rollout_length, args.environments, device)?;
        let optimization = optimize_rollout(
            &planner,
            &rollout,
            &advantages,
            &returns,
            args.minibatch_size,
            args.seed ^ update as u64,
            &hl_gauss,
            &clip_groups,
            &trainable_vars,
            &mut optimizer,
            &mut optimizer_steps,
            device,
        )?;
        print_training_metrics(update + 1, &rollout_metrics, optimization);
        append_training_metrics(&args.output, update + 1, &rollout_metrics, optimization)?;
        save_planner_checkpoint(
            &planner_vs,
            &args.output,
            &PlannerCheckpointMetadata::new(
                &world_hash,
                args.horizon,
                context_bars,
                optimizer_steps,
            ),
        )?;
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
    let world_hash = world_model.metadata().checkpoint_sha256.clone();
    let mut planner_vs = nn::VarStore::new(device);
    let planner = WorldModelPlanner::new(&planner_vs.root());
    let checkpoint_metadata = load_planner_checkpoint(
        &mut planner_vs,
        &args.planner_weights,
        &world_hash,
        args.horizon,
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
    let fantasy_prediction =
        fantasy_session.forecast(world_model, (horizon + rollout_length) as i64)?;
    let real_context =
        dataset.contexts(&real_endpoints, &vec![0; per_source], context_bars, device)?;
    let mut real_session = world_model.start_session(&real_context)?;
    let fantasy_closes = fantasy_close_paths(dataset, &fantasy_endpoints, &fantasy_prediction)?;
    let mut portfolios = (0..environments)
        .map(|_| PlannerPortfolio::new(100.0))
        .collect::<Vec<_>>();
    let mut pending = (0..environments)
        .map(|_| None::<PendingTransition>)
        .collect::<Vec<_>>();
    let mut rollout = MixedRollout::new(per_source * rollout_length)?;
    let relative_horizon = relative_horizon(environments, horizon, device);
    let world_hash = &world_model.metadata().checkpoint_sha256;

    for decision in 0..=rollout_length {
        let real_prediction = real_session.forecast(world_model, horizon as i64)?;
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
        .view([environments as i64, 4])
        .to_device(device);
        let output = tch::no_grad(|| {
            planner.forward(&WorldModelPlannerInput {
                forecast: PlannerForecast {
                    latent: latent.shallow_clone(),
                    ohlc_mean: mean.shallow_clone(),
                    ohlc_log_variance: logvar.shallow_clone(),
                    relative_horizon: relative_horizon.shallow_clone(),
                },
                portfolio_state: portfolio_state.shallow_clone(),
            })
        });
        let values = HlGaussBins::default_for(device).decode(&output.value_logits);
        let stored_latent = latent.to_device(Device::Cpu).detach();
        let stored_mean = mean.to_device(Device::Cpu).detach();
        let stored_logvar = logvar.to_device(Device::Cpu).detach();
        let stored_horizon = relative_horizon.to_device(Device::Cpu).detach();
        let stored_portfolio = portfolio_state.to_device(Device::Cpu).detach();

        for environment in 0..environments {
            if let Some(previous) = pending[environment].take() {
                rollout.push(PlannerTransition {
                    observation: previous.observation,
                    source: previous.source,
                    environment_id: previous.environment_id,
                    decision_index: previous.decision_index,
                    action: previous.action,
                    old_alpha: previous.old_alpha,
                    old_beta: previous.old_beta,
                    old_log_prob: previous.old_log_prob,
                    value: previous.value,
                    next_value: values.get(environment as i64).to_device(Device::Cpu),
                    reward: previous.reward,
                    terminated: false,
                    truncated: decision == rollout_length,
                    commission: previous.commission,
                    turnover: previous.turnover,
                    assets_before: previous.assets_before,
                    assets_after: previous.assets_after,
                    world_model_hash: world_hash.clone(),
                })?;
            }
        }
        if decision == rollout_length {
            break;
        }

        let actions = sample_beta_action(&output.alpha, &output.beta);
        let log_probs = beta_log_prob(&actions, &output.alpha, &output.beta);
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
            let action = actions.double_value(&[environment as i64, 0]);
            let step =
                portfolios[environment].step(action, current_prices[environment], next_price);
            pending[environment] = Some(PendingTransition {
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
                action: actions.get(environment as i64).to_device(Device::Cpu),
                old_alpha: output.alpha.get(environment as i64).to_device(Device::Cpu),
                old_beta: output.beta.get(environment as i64).to_device(Device::Cpu),
                old_log_prob: log_probs.get(environment as i64).to_device(Device::Cpu),
                value: values.get(environment as i64).to_device(Device::Cpu),
                reward: step.reward as f32,
                commission: step.commission,
                turnover: step.turnover,
                assets_before: step.assets_before,
                assets_after: step.assets_after,
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
    advantages: &Tensor,
    returns: &Tensor,
    minibatch_size: usize,
    seed: u64,
    hl_gauss: &HlGaussBins,
    clip_groups: &crate::torch::train::optimizer_glue::GradClipGroups,
    trainable_vars: &[Tensor],
    optimizer: &mut Muon,
    optimizer_steps: &mut u64,
    device: Device,
) -> Result<OptimizationSummary> {
    let batch = rollout.to_batch(device)?;
    let mut summary = OptimizationSummary::default();
    let rollout_diagnostics = split_optimization_metrics(
        &batch.sources,
        &batch.actions,
        &batch.old_alpha,
        &batch.old_beta,
        &batch.old_alpha,
        &batch.old_beta,
        &batch.values,
        returns,
    );
    summary.real_beta_concentration = rollout_diagnostics.real.beta_concentration_mean;
    summary.fantasy_beta_concentration = rollout_diagnostics.fantasy.beta_concentration_mean;
    summary.real_critic_explained_variance = rollout_diagnostics.real.critic_explained_variance;
    summary.fantasy_critic_explained_variance =
        rollout_diagnostics.fantasy.critic_explained_variance;
    for epoch in 0..DEFAULT_PLANNER_OPTIMIZATION_EPOCHS {
        for indices in
            rollout.balanced_minibatch_indices(minibatch_size, seed ^ ((epoch as u64 + 1) << 32))?
        {
            let mini = batch.select(&indices);
            let index = Tensor::from_slice(&indices).to_device(device);
            let mini_advantages = advantages.index_select(0, &index);
            let mini_returns = returns.index_select(0, &index);
            let output = planner.forward(&planner_input(&mini));
            let losses = planner_actor_critic_losses(
                hl_gauss,
                &output.value_logits,
                &output.alpha,
                &output.beta,
                &mini.actions,
                &mini.old_alpha,
                &mini.old_beta,
                &mini_advantages,
                &mini_returns,
                PlannerLossConfig::default(),
            );
            optimizer.zero_grad();
            let (actor_norm, critic_norm) = backward_actor_critic_with_separate_clips(
                clip_groups,
                trainable_vars,
                &losses.actor_loss,
                &losses.critic_loss,
                MAX_GRAD_NORM,
                device,
                false,
            );
            optimizer.set_momentum(muon_momentum_for_step(*optimizer_steps as i64));
            optimizer.step();
            *optimizer_steps += 1;
            summary.actor_loss += losses.actor_loss.double_value(&[]);
            summary.critic_loss += losses.critic_loss.double_value(&[]);
            summary.reverse_kl += losses.reverse_kl.double_value(&[]);
            summary.entropy += losses.entropy.double_value(&[]);
            summary.actor_grad_norm += actor_norm.double_value(&[]);
            summary.critic_grad_norm += critic_norm.double_value(&[]);
            summary.steps += 1;
        }
    }
    let denominator = summary.steps as f64;
    summary.actor_loss /= denominator;
    summary.critic_loss /= denominator;
    summary.reverse_kl /= denominator;
    summary.entropy /= denominator;
    summary.actor_grad_norm /= denominator;
    summary.critic_grad_norm /= denominator;
    Ok(summary)
}

fn rollout_advantages(
    rollout: &MixedRollout,
    rollout_length: usize,
    environments: usize,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let len = rollout_length * environments;
    let mut rewards = vec![0.0f32; len];
    let mut values = vec![0.0f32; len];
    let mut next_values = vec![0.0f32; len];
    let mut terminated = vec![0.0f32; len];
    let mut truncated = vec![0.0f32; len];
    for transition in rollout.transitions() {
        let slot = transition.decision_index * environments + transition.environment_id;
        if slot >= len {
            bail!("planner transition index exceeds GAE tensor");
        }
        rewards[slot] = transition.reward;
        values[slot] = transition.value.double_value(&[]) as f32;
        next_values[slot] = transition.next_value.double_value(&[]) as f32;
        terminated[slot] = transition.terminated as u8 as f32;
        truncated[slot] = transition.truncated as u8 as f32;
    }
    let shape = [rollout_length as i64, environments as i64];
    let (advantages, returns) = compute_default_planner_gae(
        &Tensor::from_slice(&rewards).view(shape).to_device(device),
        &Tensor::from_slice(&values).view(shape).to_device(device),
        &Tensor::from_slice(&next_values)
            .view(shape)
            .to_device(device),
        &Tensor::from_slice(&terminated)
            .view(shape)
            .to_device(device),
        &Tensor::from_slice(&truncated).view(shape).to_device(device),
    );
    let mut rollout_advantages = Vec::with_capacity(rollout.len());
    let mut rollout_returns = Vec::with_capacity(rollout.len());
    for transition in rollout.transitions() {
        rollout_advantages.push(advantages.double_value(&[
            transition.decision_index as i64,
            transition.environment_id as i64,
        ]) as f32);
        rollout_returns.push(returns.double_value(&[
            transition.decision_index as i64,
            transition.environment_id as i64,
        ]) as f32);
    }
    Ok((
        Tensor::from_slice(&rollout_advantages).to_device(device),
        Tensor::from_slice(&rollout_returns).to_device(device),
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
    let relative_horizon = relative_horizon(1, horizon, device);
    let context = dataset.contexts(&[endpoint], &[0], context_bars, device)?;
    let mut session = world_model.start_session(&context)?;
    for decision in 0..rollout_length {
        let prediction = session.forecast(world_model, horizon as i64)?;
        let series = dataset.series(endpoint.series);
        let current_price = series.closes[endpoint.bar + decision];
        let portfolio_state = Tensor::from_slice(&portfolio.planner_state(current_price))
            .view([1, 4])
            .to_device(device);
        let output = tch::no_grad(|| {
            planner.forward(&WorldModelPlannerInput {
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
            let multiplier = if delta.is_finite() {
                (1.0 + delta).clamp(1e-4, 100.0)
            } else {
                1.0
            };
            closes.push(closes.last().copied().unwrap() * multiplier);
        }
        result.push(closes);
    }
    Ok(result)
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

fn print_training_metrics(
    update: usize,
    rollout: &RolloutMetrics,
    optimization: OptimizationSummary,
) {
    println!(
        "planner update={update} real_reward={:.6} fantasy_reward={:.6} real_wealth={:.6} fantasy_wealth={:.6} real_turnover={:.6} fantasy_turnover={:.6} actor_loss={:.6} critic_loss={:.6} kl={:.6} entropy={:.6} real_critic_ev={:.4} fantasy_critic_ev={:.4} actor_grad={:.6} critic_grad={:.6}",
        rollout.real.reward_mean,
        rollout.fantasy.reward_mean,
        rollout.real.mean_environment_wealth_ratio,
        rollout.fantasy.mean_environment_wealth_ratio,
        rollout.real.turnover_mean,
        rollout.fantasy.turnover_mean,
        optimization.actor_loss,
        optimization.critic_loss,
        optimization.reverse_kl,
        optimization.entropy,
        optimization.real_critic_explained_variance,
        optimization.fantasy_critic_explained_variance,
        optimization.actor_grad_norm,
        optimization.critic_grad_norm,
    );
}

fn append_training_metrics(
    planner_checkpoint: impl AsRef<Path>,
    update: usize,
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
            "update,real_reward_mean,fantasy_reward_mean,real_wealth_ratio,fantasy_wealth_ratio,real_turnover_mean,fantasy_turnover_mean,real_commissions,fantasy_commissions,real_action_mean,fantasy_action_mean,real_action_boundary_fraction,fantasy_action_boundary_fraction,real_beta_concentration,fantasy_beta_concentration,real_critic_explained_variance,fantasy_critic_explained_variance,actor_loss,critic_loss,reverse_kl,entropy,actor_grad_norm,critic_grad_norm"
        )?;
    }
    writeln!(
        file,
        "{update},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
        optimization.real_beta_concentration,
        optimization.fantasy_beta_concentration,
        optimization.real_critic_explained_variance,
        optimization.fantasy_critic_explained_variance,
        optimization.actor_loss,
        optimization.critic_loss,
        optimization.reverse_kl,
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
        "split,episode,ticker,start_bar,steps,reward_sum,final_wealth_ratio,commissions,turnover_mean,action_mean"
    )?;
    for (index, episode) in summary.episodes.iter().enumerate() {
        writeln!(
            file,
            "{split:?},{index},{},{},{},{},{},{},{},{}",
            episode.ticker,
            episode.start_bar,
            episode.steps,
            episode.reward_sum,
            episode.final_wealth_ratio,
            episode.commissions,
            episode.turnover_mean,
            episode.action_mean,
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
