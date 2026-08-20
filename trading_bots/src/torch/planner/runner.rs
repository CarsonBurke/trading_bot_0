use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use shared::paths::RUNS_PATH;
use shared::run_dir::RunDir;
use tch::{nn, Device, Kind, Tensor};

use crate::torch::{
    action_space::{beta_mean, sample_beta_action},
    bar_dist::DOF_R,
    cuda::cfg::configure_cuda,
    optim::muon::{Muon, MuonConfig, StepKind},
    train::{
        config::{
            PolicyObjective, LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM_WARMUP_START,
            POLICY_OBJECTIVE, USE_MUON,
        },
        optimizer_glue::{muon_momentum_for_step, named_trainable_variables, KlLrController},
    },
    value::hl_gauss::HlGaussBins,
    world_model::{world_model_metadata_path, BarWorldModel, BarWorldModelSession, RolloutMode},
};

use super::{
    checkpoint::{
        load_committed_planner_metadata, load_planner_checkpoint,
        planner_actor_optimizer_state_path, planner_critic_optimizer_state_path,
        planner_metadata_path, save_planner_checkpoint, verify_optimizer_state,
        PlannerCheckpointMetadata, KL_CONTROLLER_HALF_LIFE, KL_MAX_LR_SCALE, KL_MIN_LR_SCALE,
        OPTIMIZATION_EPOCHS, TARGET_KL,
    },
    data::{planner_context_bars, PlannerCorpus, PlannerDataSplit, PlannerEndpoint},
    gae::compute_planner_gae,
    losses::{
        critic_diagnostics, normalize_ppo_advantages, planner_actor_critic_losses,
        planner_critic_losses,
    },
    portfolio::PlannerPortfolio,
    reports::{
        cleanup_uncommitted_report_generations, write_inference_reports, PlannerEpisodeTrace,
        PlannerReportHistory, PlannerTrainingReportPoint,
    },
    rollout::{
        PlannerBatch, PlannerObservation, PlannerRollout, PlannerTransition, RolloutMetrics,
    },
    PlannerForecast, WorldModelPlanner, WorldModelPlannerInput, PLANNER_PORTFOLIO_DIM,
    PLANNER_RETURN_QUANTILES,
};

pub const DEFAULT_PLANNER_HORIZON: usize = 100;
pub const DEFAULT_PLANNER_ROLLOUT_LENGTH: usize = 100;
// Batch many environments through the world-model forecast per decision step.
// The forecast is a small-batch autoregressive decode (the dominant GPU cost);
// a wide environment batch amortizes its latency-bound kernels and is the primary
// lever for GPU saturation. minibatch_size scales proportionally so the optimizer
// still runs environments*rollout_length / minibatch_size = 10 minibatches/epoch.
pub const DEFAULT_PLANNER_ENVIRONMENTS: usize = 128;
pub const DEFAULT_PLANNER_OPTIMIZATION_EPOCHS: usize = OPTIMIZATION_EPOCHS;
pub const DEFAULT_PLANNER_MINIBATCH_SIZE: usize = 1280;
/// Ancestral samples drawn per imagined forecast. The planner never sees them
/// individually: they are summarized into a per-step belief mean and standard
/// deviation plus cumulative-return quantiles before the observation is stored,
/// so this only controls how well those moments are estimated.
pub const DEFAULT_PLANNER_FORECAST_SAMPLES: usize = 16;
/// Samples per `imagine` call. Each call forks the session KV cache and replicates
/// the cached history once per sample, so peak cache is `environments * chunk *
/// context` rows; the chunk bounds that independently of the sample count.
const PLANNER_FORECAST_SAMPLE_CHUNK: usize = 4;
/// One is the model's own predictive distribution; anything else biases the risk
/// estimate the planner trades against.
const PLANNER_FORECAST_TEMPERATURE: f64 = 1.0;
/// Symbols shorter than this cannot host a context plus a rollout plus a horizon.
const PLANNER_MIN_CORPUS_BARS: usize = 4096;
/// Cumulative-log-return quantile levels carried per horizon step. Tied to
/// [`PLANNER_RETURN_QUANTILES`] so widening the planner's risk view without
/// choosing the new levels here cannot compile.
const PLANNER_RETURN_QUANTILE_LEVELS: [f32; PLANNER_RETURN_QUANTILES as usize] = [0.1, 0.5, 0.9];
const KL_NEGATIVE_ROUNDOFF_TOLERANCE: f64 = 8.0 * f32::EPSILON as f64;
const RESUME_MANIFEST_VERSION: u32 = 2;
const PLANNER_ACTOR_LR_PATTERNS: &[&str] = &["policy."];

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PlannerResumeManifest {
    version: u32,
    run_lineage_id: String,
    update: u64,
    checkpoint_file: String,
    weights_sha256: String,
    actor_optimizer_sha256: String,
    critic_optimizer_sha256: String,
}

#[derive(Clone, Debug)]
pub struct TrainPlannerArgs {
    pub world_model_weights: String,
    pub world_model_metadata: Option<String>,
    pub planner_weights: Option<String>,
    pub output: String,
    pub run: Option<String>,
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
            world_model_weights: "weights/pretrain_best.ot".to_owned(),
            world_model_metadata: None,
            planner_weights: None,
            output: String::new(),
            run: None,
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
    pub report_root: Option<PathBuf>,
}

impl Default for InferPlannerArgs {
    fn default() -> Self {
        Self {
            world_model_weights: "weights/pretrain_best.ot".to_owned(),
            world_model_metadata: None,
            planner_weights: "weights/planner.ot".to_owned(),
            episodes: 10,
            horizon: None,
            rollout_length: DEFAULT_PLANNER_ROLLOUT_LENGTH,
            context_bars: None,
            tickers: None,
            split: PlannerDataSplit::Test,
            report_root: None,
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
    pub buy_and_hold_wealth_ratio: f64,
    pub outperformance_ratio: f64,
    pub commissions: f64,
    pub turnover_mean: f64,
    pub requested_target_weight_mean: f64,
    pub executed_stock_weight_mean: f64,
    pub max_drawdown: f64,
    pub trace: PlannerEpisodeTrace,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlannerInferenceSummary {
    pub episodes: Vec<PlannerInferenceEpisode>,
    pub mean_reward: f64,
    pub mean_final_wealth_ratio: f64,
    pub mean_buy_and_hold_wealth_ratio: f64,
    pub mean_outperformance_ratio: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PairedBenchmarkMetrics {
    mean_buy_and_hold_wealth_ratio: f64,
    mean_outperformance_ratio: f64,
    median_outperformance_ratio: f64,
    outperformance_fraction: f64,
}

struct CollectedRollout {
    rollout: PlannerRollout,
    benchmark: PairedBenchmarkMetrics,
    primary_trace: PlannerEpisodeTrace,
    deterministic: DeterministicRolloutEvaluation,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct DeterministicRolloutEvaluation {
    reward_mean: f64,
    wealth_ratio: f64,
    benchmark: PairedBenchmarkMetrics,
    turnover_mean: f64,
    commissions: f64,
    requested_target_weight_mean: f64,
    executed_stock_weight_mean: f64,
    action_boundary_fraction: f64,
    primary_trace: PlannerEpisodeTrace,
}

#[derive(Clone, Copy, Debug, Default)]
struct OptimizationSummary {
    actor_loss: f64,
    critic_loss: f64,
    aux_return_loss: f64,
    reverse_kl: f64,
    max_reverse_kl: f64,
    entropy: f64,
    actor_grad_norm: f64,
    critic_grad_norm: f64,
    beta_concentration: f64,
    critic_explained_variance: f64,
    kl_early_stopped: bool,
    actor_steps: usize,
    actor_available_steps: usize,
    critic_steps: usize,
    critic_available_steps: usize,
    actor_lr_scale: f64,
}

pub fn train_planner(mut args: TrainPlannerArgs) -> Result<()> {
    args.output = resolve_planner_output(&args)?;
    validate_train_args(&args)?;
    configure_cuda();
    let device = Device::cuda_if_available();
    let metadata_path = args
        .world_model_metadata
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| world_model_metadata_path(&args.world_model_weights));
    let world_model =
        BarWorldModel::load(Path::new(&args.world_model_weights), &metadata_path, device)?;
    if !world_model.all_parameters_frozen() {
        bail!("world model loaded for planning is still trainable; refusing to train a planner whose forecast can backpropagate into the world model");
    }
    let world_lineage = world_model.lineage_sha256().to_owned();
    let world_weights_hash = world_model.metadata().checkpoint_sha256.clone();
    let dataset = PlannerCorpus::load_filtered(
        &crate::data::ingest::bars_dir(),
        world_model.metadata().res_secs,
        PLANNER_MIN_CORPUS_BARS,
        args.tickers.as_deref(),
    )?;

    let mut planner_vs = nn::VarStore::new(device);
    let planner = WorldModelPlanner::new(&planner_vs.root());
    // The value critic's running-stats normalization buffers must be registered
    // in the planner VarStore BEFORE any checkpoint load so they participate in
    // save/load and are restored on resume. They are non-trainable, so they are
    // excluded from the optimizer's trainable-variable set.
    let hl_gauss = HlGaussBins::planner(&(planner_vs.root() / "value_running_stats"), device);
    if args.planner_weights.is_none() {
        ensure_fresh_resume_output(&args.output)?;
    }
    let resolved_resume = args
        .planner_weights
        .as_deref()
        .map(|weights| {
            resolve_resume_checkpoint(weights, &world_lineage, Some(args.horizon), Some(args.seed))
        })
        .transpose()?;
    let resumed = match &resolved_resume {
        Some((weights, _)) => Some(load_planner_checkpoint(
            &mut planner_vs,
            weights,
            &world_lineage,
            Some(args.horizon),
            Some(args.seed),
        )?),
        None => None,
    };
    let run_lineage_id = resumed
        .as_ref()
        .map(|metadata| metadata.run_lineage_id.clone())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
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
    let mut actor_optimizer_steps = resumed
        .as_ref()
        .map(|m| m.actor_optimizer_steps)
        .unwrap_or(0);
    let mut critic_optimizer_steps = resumed
        .as_ref()
        .map(|m| m.critic_optimizer_steps)
        .unwrap_or(0);
    let base_updates = resumed.as_ref().map(|m| m.cumulative_updates).unwrap_or(0);
    validate_output_manifest_for_resume(&args.output, &run_lineage_id, base_updates)?;
    cleanup_uncommitted_resume_bundles(&args.output, base_updates, &run_lineage_id)?;
    let output_path = Path::new(&args.output);
    let weights_dir = output_path
        .parent()
        .context("planner output has no parent")?;
    let run_root = weights_dir
        .parent()
        .context("planner weights dir has no parent")?;
    fs::create_dir_all(weights_dir).with_context(|| {
        format!(
            "failed creating planner weights dir {}",
            weights_dir.display()
        )
    })?;
    fs::create_dir_all(run_root.join("gens")).with_context(|| {
        format!(
            "failed creating planner reports dir {}",
            run_root.join("gens").display()
        )
    })?;
    let run_dir = RunDir::from_weights_path(output_path)?;
    ensure_fresh_planner_gens(&run_dir.gens, base_updates)?;
    cleanup_uncommitted_report_generations(&run_dir.gens, base_updates, &run_lineage_id)?;
    let mut report_history =
        PlannerReportHistory::load(&run_dir.gens, base_updates, &run_lineage_id)?;
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
    let (actor_named_vars, critic_named_vars) = split_planner_parameters(&named_vars)?;
    let actor_vars = actor_named_vars
        .iter()
        .map(|(_, tensor)| tensor.shallow_clone())
        .collect::<Vec<_>>();
    let critic_vars = critic_named_vars
        .iter()
        .map(|(_, tensor)| tensor.shallow_clone())
        .collect::<Vec<_>>();
    let mut actor_optimizer = new_planner_optimizer(&actor_named_vars);
    let mut critic_optimizer = new_planner_optimizer(&critic_named_vars);
    apply_planner_actor_lr_scale(
        &mut actor_optimizer,
        kl_controller.scale(),
        actor_vars.len(),
    )?;
    if let Some(metadata) = &resumed {
        if let Some((weights, _)) = &resolved_resume {
            let actor_state = planner_actor_optimizer_state_path(weights);
            let critic_state = planner_critic_optimizer_state_path(weights);
            verify_optimizer_state(&actor_state, &metadata.actor_optimizer_sha256)?;
            verify_optimizer_state(&critic_state, &metadata.critic_optimizer_sha256)?;
            let actor_steps = i64::try_from(metadata.actor_optimizer_steps)
                .context("planner actor optimizer step count exceeds i64")?;
            let critic_steps = i64::try_from(metadata.critic_optimizer_steps)
                .context("planner critic optimizer step count exceeds i64")?;
            actor_optimizer.validate_state_strict(
                &actor_state,
                &metadata.actor_initialized_adamw,
                actor_steps,
            )?;
            critic_optimizer.validate_state_strict(
                &critic_state,
                &metadata.critic_initialized_adamw,
                critic_steps,
            )?;
            actor_optimizer.load_state_strict(&actor_state, &metadata.actor_initialized_adamw)?;
            critic_optimizer
                .load_state_strict(&critic_state, &metadata.critic_initialized_adamw)?;
        }
    }
    println!(
        "planner training: device={device:?} updates={} H={} T={} N={} context={} (stateful world-model KV cache enabled)",
        args.updates, args.horizon, args.rollout_length, args.environments, context_bars
    );
    for update in 0..args.updates {
        let global_update = base_updates + update as u64 + 1;
        let rollout_seed = update_seed(args.seed, global_update, 0x524f4c4c4f5554);
        let mut rng = StdRng::seed_from_u64(rollout_seed);
        tch::manual_seed(rollout_seed as i64);
        let collected = collect_real_rollout(
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
        let rollout_metrics = collected.rollout.metrics();
        let benchmark = collected.benchmark;
        let primary_trace = collected.primary_trace;
        let deterministic = collected.deterministic;
        let rollout = collected.rollout;
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
            &actor_vars,
            &critic_vars,
            &mut actor_optimizer,
            &mut critic_optimizer,
            &mut actor_optimizer_steps,
            &mut critic_optimizer_steps,
            &mut kl_controller,
            device,
        )?;
        print_training_metrics(
            global_update,
            &rollout_metrics,
            benchmark,
            &deterministic,
            optimization,
        );
        let staged_reports = report_history.stage_training(
            global_update,
            training_report_point(&rollout_metrics, benchmark, &deterministic, optimization),
            &primary_trace,
            &deterministic.primary_trace,
        )?;
        let resume_path = resume_checkpoint_path(&args.output, global_update);
        ensure_immutable_checkpoint_path(&resume_path)?;
        let committed = save_planner_checkpoint(
            &planner_vs,
            &resume_path,
            &PlannerCheckpointMetadata::new(
                &world_lineage,
                &world_weights_hash,
                args.horizon,
                context_bars,
                actor_optimizer_steps,
                critic_optimizer_steps,
                global_update,
                &run_lineage_id,
                args.seed,
                kl_controller.ema(),
                kl_controller.scale(),
            ),
            &actor_optimizer,
            &critic_optimizer,
        )?;
        staged_reports.publish()?;
        commit_resume_manifest(&args.output, &resume_path, &committed)?;
    }
    // Automatic validation is disabled: training is representative enough and
    // validation is a major GPU/wall-clock sink. The canonical run output is the
    // final checkpoint (identical weights to the last per-update resume bundle);
    // write it explicitly from the final weights so `--output` is a self-contained,
    // directly loadable planner.ot bundle (weights + metadata + optimizer sidecar)
    // for infer-planner. Per-update resume bundles and their manifest are retained.
    save_planner_checkpoint(
        &planner_vs,
        output_path,
        &PlannerCheckpointMetadata::new(
            &world_lineage,
            &world_weights_hash,
            args.horizon,
            context_bars,
            actor_optimizer_steps,
            critic_optimizer_steps,
            base_updates + args.updates as u64,
            &run_lineage_id,
            args.seed,
            kl_controller.ema(),
            kl_controller.scale(),
        ),
        &actor_optimizer,
        &critic_optimizer,
    )?;
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
    let world_model =
        BarWorldModel::load(Path::new(&args.world_model_weights), &metadata_path, device)?;
    if !world_model.all_parameters_frozen() {
        bail!("world model loaded for planning is still trainable; refusing to plan against a world model whose forecast can accumulate gradient");
    }
    let world_lineage = world_model.lineage_sha256().to_owned();
    let mut planner_vs = nn::VarStore::new(device);
    let planner = WorldModelPlanner::new(&planner_vs.root());
    // Register the critic value running-stats buffers before load so the persisted
    // stats restore into this store (keeping the value function definition intact
    // even though inference reads actions, not decoded values).
    let _hl_gauss = HlGaussBins::planner(&(planner_vs.root() / "value_running_stats"), device);
    let (planner_weights, _) =
        resolve_resume_checkpoint(&args.planner_weights, &world_lineage, args.horizon, None)?;
    let checkpoint_metadata = load_planner_checkpoint(
        &mut planner_vs,
        &planner_weights,
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
    let dataset = PlannerCorpus::load_filtered(
        &crate::data::ingest::bars_dir(),
        world_model.metadata().res_secs,
        PLANNER_MIN_CORPUS_BARS,
        args.tickers.as_deref(),
    )?;
    // The last decision of the episode still imagines a full horizon, whose future
    // calendar is read off real bars, so an endpoint must leave room for the rollout
    // AND the horizon behind it.
    let endpoints = dataset.deterministic_endpoints(
        args.split,
        args.episodes,
        context_bars,
        args.rollout_length + horizon,
    )?;
    let evaluation_fingerprint = dataset.evaluation_fingerprint(
        args.split,
        &endpoints,
        horizon,
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
    let summary = inference_summary(episodes);
    println!(
        "planner held-out {:?}: episodes={} mean_reward={:.6} mean_wealth={:.6} mean_buy_hold={:.6} mean_outperformance={:.6}",
        args.split,
        summary.episodes.len(),
        summary.mean_reward,
        summary.mean_final_wealth_ratio,
        summary.mean_buy_and_hold_wealth_ratio,
        summary.mean_outperformance_ratio,
    );
    let run_dir = match args.report_root.as_deref() {
        Some(root) => RunDir::open(root)?,
        None => RunDir::create_fresh(RUNS_PATH, None)?,
    };
    write_inference_reports(
        &run_dir.gens,
        checkpoint_metadata.cumulative_updates,
        &checkpoint_metadata.run_lineage_id,
        &format!("{:?}", args.split),
        &summary
            .episodes
            .iter()
            .map(|episode| episode.trace.clone())
            .collect::<Vec<_>>(),
        &evaluation_fingerprint,
    )?;
    Ok(summary)
}

fn resolve_planner_output(args: &TrainPlannerArgs) -> Result<String> {
    if !args.output.is_empty() {
        if args.run.is_some() {
            bail!("--output and --run are mutually exclusive");
        }
        let destination = RunDir::from_weights_path_in(Path::new(&args.output), RUNS_PATH)
            .with_context(|| {
                format!(
                    "planner output must belong to a prepared run: {}",
                    args.output
                )
            })?;
        destination.activate(RUNS_PATH)?;
        return Ok(args.output.clone());
    }
    if let Some(weights) = args.planner_weights.as_deref() {
        if args.run.is_some() {
            bail!(
                "--run cannot be combined with --planner-weights; omit --run to resume in place or use --output for an explicit destination"
            );
        }
        let source =
            RunDir::from_weights_path_in(Path::new(weights), RUNS_PATH).with_context(|| {
                "planner resume weights must belong to a run when --output is omitted"
            })?;
        source.activate(RUNS_PATH)?;
        return Ok(source.weights.join("planner.ot").display().to_string());
    }
    let destination = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    Ok(destination.weights.join("planner.ot").display().to_string())
}

/// Sample-axis summary of one imagined forecast, in exactly the shapes
/// [`PlannerForecast`] accepts.
struct PlannerForecastSummary {
    /// `[B, horizon, PLANNER_LATENT_DIM]`: the per-step belief mean concatenated
    /// with its across-sample standard deviation.
    latent: Tensor,
    /// `[B, horizon, PLANNER_RETURN_QUANTILES]`: quantiles of the cumulative log
    /// return the same samples imply at that step.
    quantiles: Tensor,
}

/// Imagine `samples` ancestral continuations from `session` and fold them into the
/// two belief moments plus the cumulative-return quantiles.
///
/// Nothing per-sample survives, so the samples are drawn in chunks and reduced as
/// they arrive instead of being materialized as one `[B, samples, horizon, D]` block:
/// each `imagine` call replicates the session's cached context once per sample, and
/// that replication — not the arithmetic — is what bounds the sample count.
fn summarize_forecast(
    world_model: &BarWorldModel,
    session: &BarWorldModelSession,
    future_time_ids: &Tensor,
    samples: usize,
) -> Result<PlannerForecastSummary> {
    if samples == 0 {
        bail!("planner forecast needs at least one imagined sample");
    }
    let device = world_model.device();
    // Moments are accumulated about the decision-time belief rather than about zero.
    // Every imagined step stays near it, so the centered `E[c^2] - E[c]^2` never has to
    // cancel a large shared mean: measured uncentred, a step whose true across-sample
    // spread is exactly zero reported 2.4e-4 of the belief scale purely from squaring in
    // f32. `[B, 1, 1, D]` broadcasts against the `[B, samples, horizon, D]` beliefs.
    let shift = session.belief().unsqueeze(1).unsqueeze(1);
    tch::no_grad(|| {
        let mut sum: Option<Tensor> = None;
        let mut sum_sq: Option<Tensor> = None;
        let mut cumulative_returns = Vec::new();
        let mut drawn = 0usize;
        while drawn < samples {
            let chunk = PLANNER_FORECAST_SAMPLE_CHUNK.min(samples - drawn);
            let rollout = world_model.imagine(
                session,
                future_time_ids,
                chunk,
                PLANNER_FORECAST_TEMPERATURE,
                RolloutMode::Exact,
            );
            // The sums themselves accumulate in f64, so a long horizon cannot drift the
            // moments even though each squared deviation is only f32-accurate.
            let centered = &rollout.beliefs - &shift;
            let chunk_sum = centered.sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
            let chunk_sum_sq =
                centered
                    .square()
                    .sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
            sum = Some(match sum {
                Some(total) => total + chunk_sum,
                None => chunk_sum,
            });
            sum_sq = Some(match sum_sq {
                Some(total) => total + chunk_sum_sq,
                None => chunk_sum_sq,
            });
            // `[B, chunk, horizon]`: the log return compounds along the horizon, so the
            // quantiles below are of the cumulative return to each step, not of one bar.
            cumulative_returns.push(rollout.dof.select(-1, DOF_R as i64).cumsum(-1, Kind::Float));
            drawn += chunk;
        }
        let (Some(sum), Some(sum_sq)) = (sum, sum_sq) else {
            bail!("planner forecast summarized no imagined samples");
        };
        let count = samples as f64;
        let offset = sum / count;
        let variance = (sum_sq / count - offset.square()).clamp_min(0.0);
        let mean = shift.squeeze_dim(1) + offset;
        let latent = Tensor::cat(
            &[
                mean.to_kind(Kind::Float),
                variance.sqrt().to_kind(Kind::Float),
            ],
            -1,
        );
        let levels = Tensor::from_slice(&PLANNER_RETURN_QUANTILE_LEVELS).to_device(device);
        // `quantile` prepends the level axis, so `[B, samples, horizon]` reduces to
        // `[levels, B, horizon]` and has to be rotated into the planner's layout.
        let quantiles = Tensor::cat(&cumulative_returns, 1)
            .quantile(&levels, Some(1), false, "linear")
            .permute([1, 2, 0])
            .contiguous();
        Ok(PlannerForecastSummary { latent, quantiles })
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_real_rollout(
    planner: &WorldModelPlanner,
    world_model: &BarWorldModel,
    dataset: &PlannerCorpus,
    hl_gauss: &HlGaussBins,
    horizon: usize,
    rollout_length: usize,
    environments: usize,
    context_bars: usize,
    rng: &mut StdRng,
    device: Device,
) -> Result<CollectedRollout> {
    // The last decision still imagines a full horizon, and that horizon's calendar is
    // read off real bars, so an endpoint has to leave room for the rollout AND the
    // horizon behind it or `future_time_ids` runs past the end of the series.
    let endpoints = dataset.sample_endpoints(
        PlannerDataSplit::Train,
        environments,
        context_bars,
        rollout_length + horizon,
        rng,
    )?;
    let start = dataset.context_batch(&endpoints, &vec![0; environments], context_bars, device)?;
    let mut session = world_model.start_session(&start.dof, &start.time_ids);
    let mut portfolios = (0..environments)
        .map(|_| PlannerPortfolio::new(100.0))
        .collect::<Vec<_>>();
    let mut deterministic_portfolios = (0..environments)
        .map(|_| PlannerPortfolio::new(100.0))
        .collect::<Vec<_>>();
    let mut deterministic_reward_sum = 0.0;
    let mut deterministic_turnover_sum = 0.0;
    let mut deterministic_requested_weight_sum = 0.0;
    let mut deterministic_executed_weight_sum = 0.0;
    let mut deterministic_boundary_count = 0usize;
    let mut pending = (0..environments)
        .map(|_| None::<PlannerTransition>)
        .collect::<Vec<_>>();
    let world_lineage = world_model.lineage_sha256().to_owned();
    let mut rollout = PlannerRollout::new(environments * rollout_length, world_lineage)?;
    let relative_horizon = relative_horizon(environments, horizon, device);
    // The relative-horizon tensor is identical across every decision step, so its
    // host copy is hoisted out of the loop instead of being re-transferred ~100x.
    let stored_horizon = relative_horizon.to_device(Device::Cpu).detach();
    let primary_symbol = dataset.symbol(endpoints[0].series).to_owned();
    let primary_start_price = dataset.close(endpoints[0].series, endpoints[0].bar);
    let mut primary_trace = PlannerEpisodeTrace {
        ticker: primary_symbol.clone(),
        cash: vec![100.0],
        positioned: vec![0.0],
        total: vec![100.0],
        benchmark: vec![100.0],
        ..PlannerEpisodeTrace::default()
    };
    let mut deterministic_primary_trace = PlannerEpisodeTrace {
        ticker: primary_symbol,
        cash: vec![100.0],
        positioned: vec![0.0],
        total: vec![100.0],
        benchmark: vec![100.0],
        ..PlannerEpisodeTrace::default()
    };

    for decision in 0..=rollout_length {
        let future_time_ids =
            dataset
                .corpus()
                .future_time_ids(&endpoints, decision, horizon as i64, device)?;
        let summary = summarize_forecast(
            world_model,
            &session,
            &future_time_ids,
            DEFAULT_PLANNER_FORECAST_SAMPLES,
        )?;
        validate_forecast_finite(&summary)?;
        let current_prices = endpoints
            .iter()
            .map(|endpoint| dataset.close(endpoint.series, endpoint.bar + decision))
            .collect::<Vec<_>>();
        let portfolio_state = Tensor::from_slice(
            &portfolios
                .iter()
                .zip(&current_prices)
                .flat_map(|(portfolio, &price)| portfolio.planner_state(price))
                .collect::<Vec<_>>(),
        )
        .view([environments as i64, PLANNER_PORTFOLIO_DIM])
        .to_device(device);
        let belief = session.belief().shallow_clone();
        let planner_input = WorldModelPlannerInput {
            forecast: PlannerForecast {
                latent: summary.latent.shallow_clone(),
                relative_horizon: relative_horizon.shallow_clone(),
                return_quantiles: summary.quantiles.shallow_clone(),
            },
            belief: belief.shallow_clone(),
            portfolio_state: portfolio_state.shallow_clone(),
        };
        let (encoded_policy, output) =
            tch::no_grad(|| planner.forward_with_policy_encoding_mixed_precision(&planner_input));
        let values_cpu = hl_gauss.decode(&output.value_logits).to_device(Device::Cpu);

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

        // The observation tensors feed only the transitions created below, so their
        // host copies are deferred past the truncation break to skip the wasted
        // final-iteration transfer (horizon is constant and copied once, hoisted).
        let stored_latent = summary.latent.to_device(Device::Cpu).detach();
        let stored_quantiles = summary.quantiles.to_device(Device::Cpu).detach();
        let stored_belief = belief.to_device(Device::Cpu).detach();
        let stored_portfolio = portfolio_state.to_device(Device::Cpu).detach();

        let deterministic_portfolio_state = Tensor::from_slice(
            &deterministic_portfolios
                .iter()
                .zip(&current_prices)
                .flat_map(|(portfolio, &price)| portfolio.planner_state(price))
                .collect::<Vec<_>>(),
        )
        .view([environments as i64, PLANNER_PORTFOLIO_DIM])
        .to_device(device);
        let deterministic_output = tch::no_grad(|| {
            planner.readout_policy_encoded_mixed_precision(
                &encoded_policy,
                &deterministic_portfolio_state,
            )
        });
        // Coalesce the four per-step policy tensors (all [environments, 1]) into a
        // single device->host copy: one sync per step instead of four. The split
        // values are byte-identical to transferring each tensor separately.
        let sampled_actions = sample_beta_action(&output.alpha, &output.beta);
        let deterministic_actions = beta_mean(&deterministic_output.0, &deterministic_output.1);
        let policy_bundle_cpu = Tensor::stack(
            &[
                &sampled_actions,
                &deterministic_actions,
                &output.alpha,
                &output.beta,
            ],
            0,
        )
        .to_device(Device::Cpu);
        let actions_cpu = policy_bundle_cpu.get(0);
        let deterministic_actions_cpu = policy_bundle_cpu.get(1);
        let alpha_cpu = policy_bundle_cpu.get(2);
        let beta_cpu = policy_bundle_cpu.get(3);
        for environment in 0..environments {
            let endpoint = endpoints[environment];
            let next_price = dataset.close(endpoint.series, endpoint.bar + decision + 1);
            let action = actions_cpu.double_value(&[environment as i64, 0]);
            let step =
                portfolios[environment].step(action, current_prices[environment], next_price);
            let deterministic_action =
                deterministic_actions_cpu.double_value(&[environment as i64, 0]);
            let deterministic_step = deterministic_portfolios[environment].step(
                deterministic_action,
                current_prices[environment],
                next_price,
            );
            deterministic_reward_sum += deterministic_step.reward;
            deterministic_turnover_sum += deterministic_step.turnover;
            deterministic_requested_weight_sum += deterministic_step.requested_target_weight;
            deterministic_executed_weight_sum += deterministic_step.executed_stock_weight;
            if !(0.01..=0.99).contains(&deterministic_step.requested_target_weight) {
                deterministic_boundary_count += 1;
            }
            if environment == 0 {
                primary_trace.cash.push(step.cash_after_trade);
                primary_trace.positioned.push(step.positioned_value_after);
                primary_trace.total.push(step.assets_after);
                primary_trace
                    .benchmark
                    .push(100.0 * next_price / primary_start_price);
                primary_trace.rewards.push(step.reward);
                primary_trace.commissions.push(step.commission);
                primary_trace.turnover.push(step.turnover);
                primary_trace
                    .requested_target_weight
                    .push(step.requested_target_weight);
                primary_trace
                    .executed_stock_weight
                    .push(step.executed_stock_weight);
                deterministic_primary_trace
                    .cash
                    .push(deterministic_step.cash_after_trade);
                deterministic_primary_trace
                    .positioned
                    .push(deterministic_step.positioned_value_after);
                deterministic_primary_trace
                    .total
                    .push(deterministic_step.assets_after);
                deterministic_primary_trace
                    .benchmark
                    .push(100.0 * next_price / primary_start_price);
                deterministic_primary_trace
                    .rewards
                    .push(deterministic_step.reward);
                deterministic_primary_trace
                    .commissions
                    .push(deterministic_step.commission);
                deterministic_primary_trace
                    .turnover
                    .push(deterministic_step.turnover);
                deterministic_primary_trace
                    .requested_target_weight
                    .push(deterministic_step.requested_target_weight);
                deterministic_primary_trace
                    .executed_stock_weight
                    .push(deterministic_step.executed_stock_weight);
            }
            pending[environment] = Some(PlannerTransition {
                observation: observation_at(
                    &stored_latent,
                    &stored_horizon,
                    &stored_quantiles,
                    &stored_belief,
                    &stored_portfolio,
                    environment,
                ),
                environment_id: environment,
                decision_index: decision,
                action: actions_cpu.get(environment as i64),
                old_alpha: alpha_cpu.get(environment as i64),
                old_beta: beta_cpu.get(environment as i64),
                value: values_cpu.get(environment as i64),
                next_value: None,
                reward: step.reward as f32,
                next_log_return: (next_price / current_prices[environment]).ln() as f32,
                // Long-only, no-leverage portfolio: total assets stay strictly
                // positive every step, so an episode never terminates; it can
                // only truncate at the rollout horizon.
                terminated: false,
                truncated: false,
                commission: step.commission,
                turnover: step.turnover,
                executed_stock_weight: step.executed_stock_weight,
                assets_before: step.assets_before,
                assets_after: step.assets_after,
            });
        }
        let next =
            dataset.context_batch(&endpoints, &vec![decision + 1; environments], 1, device)?;
        world_model.advance(&mut session, &next.dof, &next.time_ids)?;
    }
    rollout.validate_complete()?;
    let policy_wealth = endpoints
        .iter()
        .enumerate()
        .map(|(environment, endpoint)| {
            let final_price = dataset.close(endpoint.series, endpoint.bar + rollout_length);
            portfolios[environment].total_assets(final_price) / 100.0
        })
        .collect::<Vec<_>>();
    let buy_and_hold_wealth = endpoints
        .iter()
        .map(|endpoint| {
            buy_and_hold_wealth_ratio(
                dataset.close(endpoint.series, endpoint.bar),
                dataset.close(endpoint.series, endpoint.bar + rollout_length),
            )
        })
        .collect::<Vec<_>>();
    let deterministic_policy_wealth = endpoints
        .iter()
        .enumerate()
        .map(|(environment, endpoint)| {
            let final_price = dataset.close(endpoint.series, endpoint.bar + rollout_length);
            deterministic_portfolios[environment].total_assets(final_price) / 100.0
        })
        .collect::<Vec<_>>();
    let samples = (environments * rollout_length) as f64;
    Ok(CollectedRollout {
        rollout,
        benchmark: paired_benchmark_metrics(&policy_wealth, &buy_and_hold_wealth)?,
        primary_trace,
        deterministic: DeterministicRolloutEvaluation {
            reward_mean: deterministic_reward_sum / samples,
            wealth_ratio: deterministic_policy_wealth.iter().sum::<f64>() / environments as f64,
            benchmark: paired_benchmark_metrics(
                &deterministic_policy_wealth,
                &buy_and_hold_wealth,
            )?,
            turnover_mean: deterministic_turnover_sum / samples,
            commissions: deterministic_portfolios
                .iter()
                .map(|portfolio| portfolio.total_commissions)
                .sum(),
            requested_target_weight_mean: deterministic_requested_weight_sum / samples,
            executed_stock_weight_mean: deterministic_executed_weight_sum / samples,
            action_boundary_fraction: deterministic_boundary_count as f64 / samples,
            primary_trace: deterministic_primary_trace,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn optimize_rollout(
    planner: &WorldModelPlanner,
    rollout: &PlannerRollout,
    batch: &PlannerBatch,
    advantages: &Tensor,
    returns: &Tensor,
    minibatch_size: usize,
    seed: u64,
    hl_gauss: &HlGaussBins,
    actor_vars: &[Tensor],
    critic_vars: &[Tensor],
    actor_optimizer: &mut Muon,
    critic_optimizer: &mut Muon,
    actor_optimizer_steps: &mut u64,
    critic_optimizer_steps: &mut u64,
    kl_controller: &mut KlLrController,
    device: Device,
) -> Result<OptimizationSummary> {
    let mut summary = OptimizationSummary::default();
    let diagnostics = critic_diagnostics(&batch.old_alpha, &batch.old_beta, &batch.values, returns);
    summary.beta_concentration = diagnostics.beta_concentration_mean;
    summary.critic_explained_variance = diagnostics.critic_explained_variance;
    summary.actor_lr_scale = kl_controller.scale();
    // Refit the value critic's running normalization on this update's GAE returns
    // (whole batch, once) BEFORE encoding any minibatch target, so every minibatch
    // in this update standardizes against the same run-level stats and the encoded
    // targets are consistent with the decode used for the next rollout/GAE/EV.
    // critic_ev above is measured against the values the critic produced during the
    // rollout (decoded with the prior stats), so it stays in raw return units.
    hl_gauss.update_running_stats(returns);

    let mut actor_loss_sum = Tensor::zeros([], (Kind::Float, device));
    let mut critic_loss_sum = Tensor::zeros([], (Kind::Float, device));
    let mut aux_return_loss_sum = Tensor::zeros([], (Kind::Float, device));
    let mut reverse_kl_sum = 0.0f64;
    let mut entropy_sum = Tensor::zeros([], (Kind::Float, device));
    let mut actor_grad_sum = Tensor::zeros([], (Kind::Float, device));
    let mut critic_grad_sum = Tensor::zeros([], (Kind::Float, device));
    apply_planner_actor_lr_scale(actor_optimizer, kl_controller.scale(), actor_vars.len())?;
    summary.actor_available_steps =
        batch.len() as usize / minibatch_size * DEFAULT_PLANNER_OPTIMIZATION_EPOCHS;
    summary.critic_available_steps = summary.actor_available_steps;
    let mut actor_stopped = false;
    for epoch in 0..DEFAULT_PLANNER_OPTIMIZATION_EPOCHS {
        for indices in
            rollout.minibatch_indices(minibatch_size, seed ^ ((epoch as u64 + 1) << 32))?
        {
            let mini = batch.select(&indices);
            let index = Tensor::from_slice(&indices).to_device(device);
            let mini_advantages = advantages.index_select(0, &index);
            let mini_returns = returns.index_select(0, &index);
            let input = planner_input(&mini);
            let (critic_loss, aux_return_loss) = if actor_stopped {
                let output = planner.forward_critic_mixed_precision(&input);
                let losses = planner_critic_losses(
                    hl_gauss,
                    &output.value_logits,
                    &mini_returns,
                    &output.next_return,
                    &mini.next_log_returns,
                );
                (losses.critic_loss, losses.aux_return_loss)
            } else {
                let output = planner.forward_mixed_precision(&input);
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
                    &output.next_return,
                    &mini.next_log_returns,
                );
                let raw_minibatch_kl = losses.reverse_kl.double_value(&[]);
                let (stop, minibatch_kl) =
                    kl_stops_before_optimizer_step(&mut summary, raw_minibatch_kl)?;
                actor_stopped = stop;
                if !stop {
                    actor_optimizer.zero_grad();
                    losses.actor_loss.backward();
                    let actor_norm = clip_gradients(actor_vars, MAX_GRAD_NORM, device);
                    actor_optimizer
                        .set_momentum(muon_momentum_for_step(*actor_optimizer_steps as i64));
                    actor_optimizer.step(StepKind::Primary);
                    *actor_optimizer_steps += 1;
                    actor_loss_sum += losses.actor_loss.detach();
                    reverse_kl_sum += minibatch_kl;
                    entropy_sum += losses.entropy.detach();
                    actor_grad_sum += actor_norm;
                    summary.actor_steps += 1;
                }
                (losses.critic_loss, losses.aux_return_loss)
            };

            critic_optimizer.zero_grad();
            critic_loss.backward();
            let critic_norm = clip_gradients(critic_vars, MAX_GRAD_NORM, device);
            critic_optimizer.set_momentum(muon_momentum_for_step(*critic_optimizer_steps as i64));
            critic_optimizer.step(StepKind::Primary);
            *critic_optimizer_steps += 1;
            critic_loss_sum += critic_loss.detach();
            aux_return_loss_sum += aux_return_loss.detach();
            critic_grad_sum += critic_norm;
            summary.critic_steps += 1;
        }
    }
    let critic_denominator = summary.critic_steps as f64;
    let actor_denominator = summary.actor_steps.max(1) as f64;
    summary.actor_loss = actor_loss_sum.double_value(&[]) / actor_denominator;
    summary.critic_loss = critic_loss_sum.double_value(&[]) / critic_denominator;
    summary.aux_return_loss = aux_return_loss_sum.double_value(&[]) / critic_denominator;
    summary.reverse_kl = reverse_kl_sum / actor_denominator;
    summary.entropy = entropy_sum.double_value(&[]) / actor_denominator;
    summary.actor_grad_norm = actor_grad_sum.double_value(&[]) / actor_denominator;
    summary.critic_grad_norm = critic_grad_sum.double_value(&[]) / critic_denominator;
    kl_controller.observe(summary.max_reverse_kl);
    Ok(summary)
}

fn apply_planner_actor_lr_scale(
    optimizer: &mut Muon,
    lr_scale: f64,
    expected: usize,
) -> Result<()> {
    let matched = optimizer.set_named_lr_scale(PLANNER_ACTOR_LR_PATTERNS, lr_scale);
    if matched != expected {
        bail!(
            "planner actor learning-rate routing matched {matched} parameters, expected {expected}"
        );
    }
    Ok(())
}

fn new_planner_optimizer(named_vars: &[(String, Tensor)]) -> Muon {
    Muon::new_named(
        named_vars,
        MuonConfig {
            lr: MUON_LR,
            use_muon_for_2d: USE_MUON,
            momentum: MUON_MOMENTUM_WARMUP_START,
            adamw_lr: LEARNING_RATE,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            force_adamw_name_substrings: vec![
                "concentration".to_owned(),
                "value_projection".to_owned(),
                "next_return_head".to_owned(),
            ],
            ..MuonConfig::default()
        },
    )
}

fn split_planner_parameters(
    named_vars: &[(String, Tensor)],
) -> Result<(Vec<(String, Tensor)>, Vec<(String, Tensor)>)> {
    let mut actor = Vec::new();
    let mut critic = Vec::new();
    for (name, tensor) in named_vars {
        let destination = if name.starts_with("policy.") {
            &mut actor
        } else if name.starts_with("critic.") {
            &mut critic
        } else {
            bail!("planner trainable parameter is outside policy/critic ownership: {name}");
        };
        destination.push((name.clone(), tensor.shallow_clone()));
    }
    if actor.is_empty() || critic.is_empty() {
        bail!("planner policy and critic parameter sets must both be non-empty");
    }
    Ok((actor, critic))
}

fn clip_gradients(parameters: &[Tensor], max_norm: f64, device: Device) -> Tensor {
    tch::no_grad(|| {
        let total_squared = parameters.iter().fold(
            Tensor::zeros([], (Kind::Float, device)),
            |sum, parameter| {
                let grad = parameter.grad();
                if grad.defined() {
                    sum + grad.square().sum(Kind::Float)
                } else {
                    sum
                }
            },
        );
        let norm = total_squared.sqrt();
        let coefficient =
            (Tensor::from(max_norm as f32).to_device(device) / (&norm + 1e-6)).clamp_max(1.0);
        for parameter in parameters {
            let mut grad = parameter.grad();
            if grad.defined() {
                let _ = grad.g_mul_(&coefficient.to_kind(grad.kind()));
            }
        }
        norm
    })
}

fn kl_stops_before_optimizer_step(
    summary: &mut OptimizationSummary,
    raw_minibatch_kl: f64,
) -> Result<(bool, f64)> {
    let minibatch_kl = validated_reverse_kl(raw_minibatch_kl)?;
    summary.max_reverse_kl = summary.max_reverse_kl.max(minibatch_kl);
    let stop = summary.actor_steps > 0 && minibatch_kl > TARGET_KL;
    summary.kl_early_stopped |= stop;
    Ok((stop, minibatch_kl))
}

fn validated_reverse_kl(raw_kl: f64) -> Result<f64> {
    if !raw_kl.is_finite() || raw_kl < -KL_NEGATIVE_ROUNDOFF_TOLERANCE {
        bail!("planner reverse KL is invalid: {raw_kl}");
    }
    Ok(raw_kl.max(0.0))
}

fn rollout_advantages(
    rollout: &PlannerRollout,
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
    let advantages = advantages.view([len as i64]).index_select(0, &slot_index);
    let returns = returns.view([len as i64]).index_select(0, &slot_index);
    // CleanRL-style advantage normalization over the ENTIRE batch (not per-minibatch):
    // stabilizes per-minibatch step scale and keeps a cluster of boundary-driven
    // advantage outliers from dominating a minibatch, which is what detaches max
    // reverse-KL from the mean. Returns are left raw — the running-stats critic owns
    // its own target normalization (no return norm). PPO only: PMPO consumes RAW GAE
    // advantages (its tanh sign-weighting subsumes scale normalization, and mean-
    // centering would flip the sign of near-mean samples and corrupt the objective).
    let advantages = match POLICY_OBJECTIVE {
        PolicyObjective::Ppo => normalize_ppo_advantages(&advantages)?,
        PolicyObjective::Pmpo => {
            if advantages.isfinite().all().int64_value(&[]) == 0 {
                bail!("planner advantage batch contains NaN or infinity");
            }
            advantages
        }
    };
    Ok((advantages, returns))
}

#[allow(clippy::too_many_arguments)]
fn infer_real_episode(
    planner: &WorldModelPlanner,
    world_model: &BarWorldModel,
    dataset: &PlannerCorpus,
    endpoint: PlannerEndpoint,
    horizon: usize,
    rollout_length: usize,
    context_bars: usize,
    device: Device,
) -> Result<PlannerInferenceEpisode> {
    let mut portfolio = PlannerPortfolio::new(100.0);
    let mut reward_sum = 0.0;
    let mut turnover = 0.0;
    let mut requested_target_weight_sum = 0.0;
    let mut executed_stock_weight_sum = 0.0;
    let mut peak_assets: f64 = 100.0;
    let mut max_drawdown: f64 = 0.0;
    let relative_horizon = relative_horizon(1, horizon, device);
    let start = dataset.context_batch(&[endpoint], &[0], context_bars, device)?;
    let mut session = world_model.start_session(&start.dof, &start.time_ids);
    let symbol = dataset.symbol(endpoint.series).to_owned();
    let start_price = dataset.close(endpoint.series, endpoint.bar);
    let mut trace = PlannerEpisodeTrace {
        ticker: symbol.clone(),
        cash: vec![100.0],
        positioned: vec![0.0],
        total: vec![100.0],
        benchmark: vec![100.0],
        ..PlannerEpisodeTrace::default()
    };
    for decision in 0..rollout_length {
        let future_time_ids =
            dataset
                .corpus()
                .future_time_ids(&[endpoint], decision, horizon as i64, device)?;
        let summary = summarize_forecast(
            world_model,
            &session,
            &future_time_ids,
            DEFAULT_PLANNER_FORECAST_SAMPLES,
        )?;
        validate_forecast_finite(&summary).with_context(|| {
            format!("invalid world-model forecast for {symbol} at decision {decision}")
        })?;
        let current_price = dataset.close(endpoint.series, endpoint.bar + decision);
        let portfolio_state = Tensor::from_slice(&portfolio.planner_state(current_price))
            .view([1, PLANNER_PORTFOLIO_DIM])
            .to_device(device);
        let belief = session.belief().shallow_clone();
        let output = tch::no_grad(|| {
            planner.forward_mixed_precision(&WorldModelPlannerInput {
                forecast: PlannerForecast {
                    latent: summary.latent.shallow_clone(),
                    relative_horizon: relative_horizon.shallow_clone(),
                    return_quantiles: summary.quantiles.shallow_clone(),
                },
                belief,
                portfolio_state,
            })
        });
        let action = beta_mean(&output.alpha, &output.beta).double_value(&[0, 0]);
        let next_price = dataset.close(endpoint.series, endpoint.bar + decision + 1);
        let step = portfolio.step(action, current_price, next_price);
        reward_sum += step.reward;
        turnover += step.turnover;
        requested_target_weight_sum += step.requested_target_weight;
        executed_stock_weight_sum += step.executed_stock_weight;
        trace.cash.push(step.cash_after_trade);
        trace.positioned.push(step.positioned_value_after);
        trace.total.push(step.assets_after);
        trace.benchmark.push(100.0 * next_price / start_price);
        trace.rewards.push(step.reward);
        trace.commissions.push(step.commission);
        trace.turnover.push(step.turnover);
        trace
            .requested_target_weight
            .push(step.requested_target_weight);
        trace.executed_stock_weight.push(step.executed_stock_weight);
        peak_assets = peak_assets.max(step.assets_after);
        max_drawdown = max_drawdown.max(1.0 - step.assets_after / peak_assets);
        let next = dataset.context_batch(&[endpoint], &[decision + 1], 1, device)?;
        world_model.advance(&mut session, &next.dof, &next.time_ids)?;
    }
    let final_price = dataset.close(endpoint.series, endpoint.bar + rollout_length);
    let final_wealth_ratio = portfolio.total_assets(final_price) / 100.0;
    let buy_and_hold_wealth_ratio =
        buy_and_hold_wealth_ratio(dataset.close(endpoint.series, endpoint.bar), final_price);
    Ok(PlannerInferenceEpisode {
        ticker: symbol,
        start_bar: endpoint.bar,
        steps: rollout_length,
        reward_sum,
        final_wealth_ratio,
        buy_and_hold_wealth_ratio,
        outperformance_ratio: final_wealth_ratio - buy_and_hold_wealth_ratio,
        commissions: portfolio.total_commissions,
        turnover_mean: turnover / rollout_length as f64,
        requested_target_weight_mean: requested_target_weight_sum / rollout_length as f64,
        executed_stock_weight_mean: executed_stock_weight_sum / rollout_length as f64,
        max_drawdown,
        trace,
    })
}

fn planner_input(batch: &PlannerBatch) -> WorldModelPlannerInput {
    WorldModelPlannerInput {
        forecast: PlannerForecast {
            latent: batch.forecast_latent.shallow_clone(),
            relative_horizon: batch.relative_horizon.shallow_clone(),
            return_quantiles: batch.return_quantiles.shallow_clone(),
        },
        belief: batch.belief.shallow_clone(),
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
    relative_horizon: &Tensor,
    return_quantiles: &Tensor,
    belief: &Tensor,
    portfolio_state: &Tensor,
    index: usize,
) -> PlannerObservation {
    let index = index as i64;
    PlannerObservation {
        forecast_latent: latent.get(index).to_device(Device::Cpu).detach(),
        relative_horizon: relative_horizon.get(index).to_device(Device::Cpu).detach(),
        return_quantiles: return_quantiles.get(index).to_device(Device::Cpu).detach(),
        belief: belief.get(index).to_device(Device::Cpu).detach(),
        portfolio_state: portfolio_state.get(index).to_device(Device::Cpu).detach(),
    }
}

fn validate_forecast_finite(summary: &PlannerForecastSummary) -> Result<()> {
    for (name, tensor) in [
        ("forecast latent", &summary.latent),
        ("forecast return quantiles", &summary.quantiles),
    ] {
        let finite: i64 = tensor.isfinite().all().int64_value(&[]);
        if finite == 0 {
            bail!("world-model {name} contains NaN or infinity");
        }
    }
    Ok(())
}

fn buy_and_hold_wealth_ratio(start_price: f64, end_price: f64) -> f64 {
    end_price / start_price
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

fn paired_benchmark_metrics(
    policy_wealth: &[f64],
    buy_and_hold_wealth: &[f64],
) -> Result<PairedBenchmarkMetrics> {
    if policy_wealth.is_empty() || policy_wealth.len() != buy_and_hold_wealth.len() {
        bail!("paired benchmark requires equally sized, non-empty policy and buy-and-hold samples");
    }
    if policy_wealth
        .iter()
        .chain(buy_and_hold_wealth)
        .any(|value| !value.is_finite())
    {
        bail!("paired benchmark wealth contains NaN or infinity");
    }
    let count = policy_wealth.len() as f64;
    let mut outperformance = policy_wealth
        .iter()
        .zip(buy_and_hold_wealth)
        .map(|(policy, benchmark)| policy - benchmark)
        .collect::<Vec<_>>();
    Ok(PairedBenchmarkMetrics {
        mean_buy_and_hold_wealth_ratio: buy_and_hold_wealth.iter().sum::<f64>() / count,
        mean_outperformance_ratio: outperformance.iter().sum::<f64>() / count,
        median_outperformance_ratio: median(&mut outperformance),
        outperformance_fraction: outperformance
            .iter()
            .filter(|outperformance| **outperformance > 0.0)
            .count() as f64
            / count,
    })
}

fn inference_summary(episodes: Vec<PlannerInferenceEpisode>) -> PlannerInferenceSummary {
    let count = episodes.len() as f64;
    PlannerInferenceSummary {
        mean_reward: episodes
            .iter()
            .map(|episode| episode.reward_sum)
            .sum::<f64>()
            / count,
        mean_final_wealth_ratio: episodes
            .iter()
            .map(|episode| episode.final_wealth_ratio)
            .sum::<f64>()
            / count,
        mean_buy_and_hold_wealth_ratio: episodes
            .iter()
            .map(|episode| episode.buy_and_hold_wealth_ratio)
            .sum::<f64>()
            / count,
        mean_outperformance_ratio: episodes
            .iter()
            .map(|episode| episode.outperformance_ratio)
            .sum::<f64>()
            / count,
        episodes,
    }
}

fn validate_train_args(args: &TrainPlannerArgs) -> Result<()> {
    if args.updates == 0 || args.horizon == 0 || args.rollout_length == 0 {
        bail!("planner updates, horizon, and rollout length must be positive");
    }
    if args.environments == 0 {
        bail!("planner environments must be positive");
    }
    let samples = args.environments * args.rollout_length;
    if args.minibatch_size == 0 || samples % args.minibatch_size != 0 {
        bail!("planner minibatch size must evenly divide the real rollout");
    }
    Ok(())
}

fn resume_checkpoint_path(checkpoint: impl AsRef<Path>, update: u64) -> PathBuf {
    let checkpoint = checkpoint.as_ref();
    let stem = checkpoint
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("planner");
    checkpoint.with_file_name(format!("{stem}_resume_u{update:08}.ot"))
}

fn resume_manifest_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("resume.json")
}

fn ensure_immutable_checkpoint_path(checkpoint: &Path) -> Result<()> {
    if checkpoint.exists()
        || planner_metadata_path(checkpoint).exists()
        || planner_actor_optimizer_state_path(checkpoint).exists()
        || planner_critic_optimizer_state_path(checkpoint).exists()
    {
        bail!(
            "refusing to overwrite immutable planner checkpoint {}",
            checkpoint.display()
        );
    }
    Ok(())
}

fn ensure_fresh_resume_output(checkpoint: impl AsRef<Path>) -> Result<()> {
    let checkpoint = checkpoint.as_ref();
    let stem = checkpoint
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("planner");
    let has_output_bundle = checkpoint.exists()
        || planner_metadata_path(checkpoint).exists()
        || planner_actor_optimizer_state_path(checkpoint).exists()
        || planner_critic_optimizer_state_path(checkpoint).exists()
        || resume_manifest_path(checkpoint).exists();
    let has_bundle = checkpoint
        .parent()
        .and_then(|parent| fs::read_dir(parent).ok())
        .is_some_and(|entries| {
            entries.filter_map(std::result::Result::ok).any(|entry| {
                entry.file_name().to_str().is_some_and(|name| {
                    name.starts_with(&format!("{stem}_resume_u"))
                        || name.starts_with(&format!("{stem}_best_u"))
                })
            })
        });
    if has_output_bundle || has_bundle {
        bail!(
            "planner output {} already contains a run; resume it explicitly or use a fresh output",
            checkpoint.display()
        );
    }
    Ok(())
}

fn ensure_fresh_planner_gens(gens: &Path, base_updates: u64) -> Result<()> {
    if base_updates == 0 && fs::read_dir(gens).is_ok_and(|mut entries| entries.next().is_some()) {
        bail!(
            "fresh planner output requires an empty gens directory: {}",
            gens.display()
        );
    }
    Ok(())
}

fn cleanup_uncommitted_resume_bundles(
    checkpoint: impl AsRef<Path>,
    committed_update: u64,
    run_lineage_id: &str,
) -> Result<()> {
    let checkpoint = checkpoint.as_ref();
    let Some(parent) = checkpoint.parent() else {
        return Ok(());
    };
    let stem = checkpoint
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("planner");
    let prefix = format!("{stem}_resume_u");
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let updates = entries
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let rest = name.to_str()?.strip_prefix(&prefix)?;
            let digits = rest
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect::<String>();
            digits.parse::<u64>().ok()
        })
        .collect::<std::collections::BTreeSet<_>>();
    for update in updates
        .into_iter()
        .filter(|update| *update > committed_update)
    {
        let resume = resume_checkpoint_path(checkpoint, update);
        let metadata_path = planner_metadata_path(&resume);
        if metadata_path.exists() {
            let metadata = PlannerCheckpointMetadata::load(&metadata_path)?;
            if metadata.run_lineage_id != run_lineage_id || metadata.cumulative_updates != update {
                bail!(
                    "uncommitted resume checkpoint {} belongs to a different run lineage or update",
                    metadata_path.display()
                );
            }
        }
        for path in [
            resume.clone(),
            planner_actor_optimizer_state_path(&resume),
            planner_critic_optimizer_state_path(&resume),
            planner_metadata_path(&resume),
        ] {
            if path.exists() {
                fs::remove_file(&path).with_context(|| {
                    format!(
                        "failed removing uncommitted planner bundle {}",
                        path.display()
                    )
                })?;
            }
        }
    }
    Ok(())
}

fn validate_output_manifest_for_resume(
    checkpoint: impl AsRef<Path>,
    run_lineage_id: &str,
    resumed_update: u64,
) -> Result<()> {
    let checkpoint = checkpoint.as_ref();
    let path = resume_manifest_path(checkpoint);
    if !path.exists() {
        return ensure_fresh_resume_output(checkpoint);
    }
    let manifest: PlannerResumeManifest = serde_json::from_slice(&fs::read(&path)?)?;
    if manifest.version != RESUME_MANIFEST_VERSION
        || manifest.run_lineage_id != run_lineage_id
        || manifest.update != resumed_update
    {
        bail!(
            "planner output manifest is newer than or unrelated to the requested resume checkpoint; resume the manifest target or use a fresh output"
        );
    }
    Ok(())
}

fn commit_resume_manifest(
    base_checkpoint: impl AsRef<Path>,
    committed_checkpoint: &Path,
    metadata: &PlannerCheckpointMetadata,
) -> Result<()> {
    let base_checkpoint = base_checkpoint.as_ref();
    let checkpoint_file = committed_checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .context("planner resume checkpoint filename is not UTF-8")?;
    if committed_checkpoint.parent() != base_checkpoint.parent() {
        bail!("planner resume checkpoint and manifest must share a directory");
    }
    let manifest = PlannerResumeManifest {
        version: RESUME_MANIFEST_VERSION,
        run_lineage_id: metadata.run_lineage_id.clone(),
        update: metadata.cumulative_updates,
        checkpoint_file: checkpoint_file.to_owned(),
        weights_sha256: metadata.weights_sha256.clone(),
        actor_optimizer_sha256: metadata.actor_optimizer_sha256.clone(),
        critic_optimizer_sha256: metadata.critic_optimizer_sha256.clone(),
    };
    let path = resume_manifest_path(base_checkpoint);
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("resume.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&manifest)?)?;
    File::open(&temporary)?.sync_all()?;
    fs::rename(&temporary, &path)?;
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn resolve_resume_checkpoint(
    requested: impl AsRef<Path>,
    world_model_lineage: &str,
    horizon: Option<usize>,
    seed: Option<u64>,
) -> Result<(PathBuf, PlannerCheckpointMetadata)> {
    let requested = requested.as_ref();
    let manifest_path = resume_manifest_path(requested);
    if manifest_path.exists() {
        let manifest: PlannerResumeManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        if manifest.version != RESUME_MANIFEST_VERSION
            || manifest.checkpoint_file.contains('/')
            || manifest.checkpoint_file.contains('\\')
        {
            bail!("planner resume manifest is incompatible or unsafe");
        }
        let checkpoint = requested
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(&manifest.checkpoint_file);
        let metadata =
            load_committed_planner_metadata(&checkpoint, world_model_lineage, horizon, seed)?;
        if metadata.cumulative_updates != manifest.update
            || metadata.run_lineage_id != manifest.run_lineage_id
            || metadata.weights_sha256 != manifest.weights_sha256
            || metadata.actor_optimizer_sha256 != manifest.actor_optimizer_sha256
            || metadata.critic_optimizer_sha256 != manifest.critic_optimizer_sha256
        {
            bail!("planner resume manifest does not match its committed checkpoint bundle");
        }
        return Ok((checkpoint, metadata));
    }
    let metadata = load_committed_planner_metadata(requested, world_model_lineage, horizon, seed)?;
    Ok((requested.to_path_buf(), metadata))
}

fn print_training_metrics(
    update: u64,
    rollout: &RolloutMetrics,
    benchmark: PairedBenchmarkMetrics,
    deterministic: &DeterministicRolloutEvaluation,
    optimization: OptimizationSummary,
) {
    println!(
        "planner update={update} sampled_reward={:.6} sampled_wealth={:.6} buy_hold={:.6} sampled_mean_outperformance={:.6} sampled_median_outperformance={:.6} sampled_outperform_fraction={:.3} sampled_turnover={:.6} deterministic_reward={:.6} deterministic_wealth={:.6} deterministic_mean_outperformance={:.6} deterministic_median_outperformance={:.6} deterministic_outperform_fraction={:.3} deterministic_turnover={:.6} actor_loss={:.6} critic_loss={:.6} aux_return_loss={:.6} kl={:.6} max_kl={:.6} kl_stop={} entropy={:.6} critic_ev={:.4} actor_grad={:.6} critic_grad={:.6}",
        rollout.reward_mean,
        rollout.mean_environment_wealth_ratio,
        benchmark.mean_buy_and_hold_wealth_ratio,
        benchmark.mean_outperformance_ratio,
        benchmark.median_outperformance_ratio,
        benchmark.outperformance_fraction,
        rollout.turnover_mean,
        deterministic.reward_mean,
        deterministic.wealth_ratio,
        deterministic.benchmark.mean_outperformance_ratio,
        deterministic.benchmark.median_outperformance_ratio,
        deterministic.benchmark.outperformance_fraction,
        deterministic.turnover_mean,
        optimization.actor_loss,
        optimization.critic_loss,
        optimization.aux_return_loss,
        optimization.reverse_kl,
        optimization.max_reverse_kl,
        optimization.kl_early_stopped,
        optimization.entropy,
        optimization.critic_explained_variance,
        optimization.actor_grad_norm,
        optimization.critic_grad_norm,
    );
}

fn training_report_point(
    rollout: &RolloutMetrics,
    benchmark: PairedBenchmarkMetrics,
    deterministic: &DeterministicRolloutEvaluation,
    optimization: OptimizationSummary,
) -> PlannerTrainingReportPoint {
    PlannerTrainingReportPoint {
        reward_mean: rollout.reward_mean,
        wealth_ratio: rollout.mean_environment_wealth_ratio,
        buy_and_hold_wealth_ratio: benchmark.mean_buy_and_hold_wealth_ratio,
        mean_outperformance_ratio: benchmark.mean_outperformance_ratio,
        median_outperformance_ratio: benchmark.median_outperformance_ratio,
        outperformance_fraction: benchmark.outperformance_fraction,
        turnover_mean: rollout.turnover_mean,
        commissions: rollout.commissions,
        requested_target_weight_mean: rollout.requested_target_weight_mean,
        executed_stock_weight_mean: rollout.executed_stock_weight_mean,
        action_boundary_fraction: rollout.action_boundary_fraction,
        deterministic_reward_mean: deterministic.reward_mean,
        deterministic_wealth_ratio: deterministic.wealth_ratio,
        deterministic_mean_outperformance_ratio: deterministic.benchmark.mean_outperformance_ratio,
        deterministic_median_outperformance_ratio: deterministic
            .benchmark
            .median_outperformance_ratio,
        deterministic_outperformance_fraction: deterministic.benchmark.outperformance_fraction,
        deterministic_turnover_mean: deterministic.turnover_mean,
        deterministic_commissions: deterministic.commissions,
        deterministic_requested_target_weight_mean: deterministic.requested_target_weight_mean,
        deterministic_executed_stock_weight_mean: deterministic.executed_stock_weight_mean,
        deterministic_action_boundary_fraction: deterministic.action_boundary_fraction,
        beta_concentration: optimization.beta_concentration,
        critic_explained_variance: optimization.critic_explained_variance,
        actor_loss: optimization.actor_loss,
        critic_loss: optimization.critic_loss,
        aux_return_loss: optimization.aux_return_loss,
        reverse_kl: optimization.reverse_kl,
        max_reverse_kl: optimization.max_reverse_kl,
        kl_early_stopped: optimization.kl_early_stopped,
        actor_steps: optimization.actor_steps as f64,
        actor_available_steps: optimization.actor_available_steps as f64,
        critic_steps: optimization.critic_steps as f64,
        critic_available_steps: optimization.critic_available_steps as f64,
        actor_lr_scale: optimization.actor_lr_scale,
        entropy: optimization.entropy,
        actor_grad_norm: optimization.actor_grad_norm,
        critic_grad_norm: optimization.critic_grad_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_real_receding_horizon_plan() {
        let args = TrainPlannerArgs::default();
        assert_eq!(args.horizon, 100);
        assert_eq!(args.rollout_length, 100);
        assert_eq!(args.environments, 128);
        assert_eq!(args.minibatch_size, 1280);
        validate_train_args(&args).unwrap();
    }

    #[test]
    fn accepts_odd_real_environment_count_and_rejects_zero() {
        let mut args = TrainPlannerArgs::default();
        args.environments = 5;
        args.minibatch_size = 100;
        validate_train_args(&args).unwrap();
        args.environments = 0;
        assert!(validate_train_args(&args).is_err());
    }

    #[test]
    fn planner_resume_does_not_silently_ignore_an_explicit_run() {
        let args = TrainPlannerArgs {
            planner_weights: Some("source/weights/planner.ot".to_owned()),
            run: Some("destination".to_owned()),
            ..TrainPlannerArgs::default()
        };
        let error = resolve_planner_output(&args).unwrap_err();
        assert!(error.to_string().contains("cannot be combined"));
    }

    #[test]
    fn update_seeds_are_stable_and_do_not_repeat_after_resume() {
        let first = update_seed(7, 1, 11);
        assert_eq!(first, update_seed(7, 1, 11));
        assert_ne!(first, update_seed(7, 2, 11));
        assert_ne!(first, update_seed(7, 1, 12));
    }

    #[test]
    fn training_reports_keep_sampled_and_deterministic_metrics_distinct() {
        let sampled = RolloutMetrics {
            reward_mean: 0.1,
            mean_environment_wealth_ratio: 1.01,
            turnover_mean: 0.3,
            ..RolloutMetrics::default()
        };
        let sampled_benchmark = PairedBenchmarkMetrics {
            mean_buy_and_hold_wealth_ratio: 1.02,
            mean_outperformance_ratio: -0.01,
            median_outperformance_ratio: -0.02,
            outperformance_fraction: 0.25,
        };
        let deterministic = DeterministicRolloutEvaluation {
            reward_mean: 0.2,
            wealth_ratio: 1.03,
            benchmark: PairedBenchmarkMetrics {
                mean_buy_and_hold_wealth_ratio: 1.02,
                mean_outperformance_ratio: 0.01,
                median_outperformance_ratio: 0.02,
                outperformance_fraction: 0.75,
            },
            turnover_mean: 0.1,
            commissions: 0.4,
            requested_target_weight_mean: 0.6,
            executed_stock_weight_mean: 0.59,
            action_boundary_fraction: 0.05,
            ..DeterministicRolloutEvaluation::default()
        };

        let point = training_report_point(
            &sampled,
            sampled_benchmark,
            &deterministic,
            OptimizationSummary {
                aux_return_loss: 0.33,
                ..OptimizationSummary::default()
            },
        );
        assert_eq!(point.reward_mean, 0.1);
        assert_eq!(point.mean_outperformance_ratio, -0.01);
        assert_eq!(point.deterministic_reward_mean, 0.2);
        assert_eq!(point.deterministic_wealth_ratio, 1.03);
        assert_eq!(point.deterministic_mean_outperformance_ratio, 0.01);
        assert_eq!(point.deterministic_turnover_mean, 0.1);
        assert_eq!(point.deterministic_commissions, 0.4);
        assert_eq!(point.aux_return_loss, 0.33);
    }

    #[test]
    fn kl_trigger_stops_before_any_parameter_update() {
        let mut summary = OptimizationSummary {
            actor_steps: 1,
            critic_steps: 1,
            ..OptimizationSummary::default()
        };
        let before_steps = summary.critic_steps;
        let vs = nn::VarStore::new(Device::Cpu);
        let parameter = vs.root().ones("weight", &[1]);
        let mut optimizer = Muon::new_named(
            &named_trainable_variables(&vs),
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        parameter.square().sum(Kind::Float).backward();
        let before_parameter = parameter.copy();
        let (stop, _) = kl_stops_before_optimizer_step(&mut summary, TARGET_KL * 2.0).unwrap();
        if !stop {
            optimizer.step(StepKind::Primary);
        }

        assert!(stop);
        assert!(summary.kl_early_stopped);
        assert_eq!(summary.max_reverse_kl, TARGET_KL * 2.0);
        assert_eq!(summary.actor_steps, 1);
        assert_eq!(summary.critic_steps, before_steps);
        assert!(parameter.equal(&before_parameter));
    }

    #[test]
    fn planner_kl_lr_routing_matches_the_entire_actor_branch() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _planner = WorldModelPlanner::new(&vs.root());
        let named_vars = named_trainable_variables(&vs);
        let matched_names = named_vars
            .iter()
            .filter(|(name, _)| {
                PLANNER_ACTOR_LR_PATTERNS
                    .iter()
                    .any(|pattern| name.contains(pattern))
            })
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();

        let (actor, critic) = split_planner_parameters(&named_vars).unwrap();
        assert_eq!(matched_names.len(), actor.len());
        assert!(matched_names.iter().all(|name| name.starts_with("policy.")));
        assert!(critic.iter().all(|(name, _)| name.starts_with("critic.")));
        assert!(actor.iter().any(|(name, _)| name.contains("trunk_0")));
        assert!(critic.iter().any(|(name, _)| name.contains("trunk_0")));

        let mut optimizer = new_planner_optimizer(&actor);
        apply_planner_actor_lr_scale(&mut optimizer, 0.25, actor.len()).unwrap();
    }

    #[test]
    fn critic_only_step_cannot_change_policy_parameters_or_output() {
        use crate::torch::planner::{PLANNER_BELIEF_DIM, PLANNER_LATENT_DIM};

        tch::manual_seed(17);
        let vs = nn::VarStore::new(Device::Cpu);
        let planner = WorldModelPlanner::new(&vs.root());
        let hl_gauss = HlGaussBins::planner(&(vs.root() / "value_running_stats"), Device::Cpu);
        let named = named_trainable_variables(&vs);
        let (actor_named, critic_named) = split_planner_parameters(&named).unwrap();
        let actor_before = actor_named
            .iter()
            .map(|(_, parameter)| parameter.copy())
            .collect::<Vec<_>>();
        let critic_before = critic_named
            .iter()
            .map(|(_, parameter)| parameter.copy())
            .collect::<Vec<_>>();
        let critic_vars = critic_named
            .iter()
            .map(|(_, parameter)| parameter.shallow_clone())
            .collect::<Vec<_>>();
        let mut critic_optimizer = new_planner_optimizer(&critic_named);
        let input = WorldModelPlannerInput {
            forecast: PlannerForecast {
                latent: Tensor::randn([2, 3, PLANNER_LATENT_DIM], (Kind::Float, Device::Cpu)),
                relative_horizon: Tensor::from_slice(&[1.0f32 / 3.0, 2.0 / 3.0, 1.0])
                    .view([1, 3, 1])
                    .expand([2, 3, 1], false),
                return_quantiles: Tensor::randn(
                    [2, 3, PLANNER_RETURN_QUANTILES],
                    (Kind::Float, Device::Cpu),
                ),
            },
            belief: Tensor::randn([2, PLANNER_BELIEF_DIM], (Kind::Float, Device::Cpu)),
            portfolio_state: Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.0])
                .view([1, 4])
                .expand([2, 4], false),
        };
        let policy_before = tch::no_grad(|| planner.forward_mixed_precision(&input));
        let critic_output_before = policy_before.value_logits.copy();
        let critic_output = planner.forward_critic_mixed_precision(&input);
        let losses = planner_critic_losses(
            &hl_gauss,
            &critic_output.value_logits,
            &Tensor::from_slice(&[0.4f32, -0.2]),
            &critic_output.next_return,
            &Tensor::from_slice(&[0.03f32, -0.01]),
        );
        critic_optimizer.zero_grad();
        losses.critic_loss.backward();
        let _ = clip_gradients(&critic_vars, MAX_GRAD_NORM, Device::Cpu);
        critic_optimizer.step(StepKind::Primary);

        let after = tch::no_grad(|| planner.forward_mixed_precision(&input));
        assert!(policy_before.alpha.equal(&after.alpha));
        assert!(policy_before.beta.equal(&after.beta));
        for ((_, parameter), before) in actor_named.iter().zip(&actor_before) {
            assert!(parameter.equal(before));
        }
        assert!(critic_named
            .iter()
            .zip(&critic_before)
            .any(|((_, parameter), before)| !parameter.equal(before)));
        assert!(!critic_output_before.equal(&after.value_logits));
    }

    #[test]
    fn actor_kl_stop_does_not_reduce_critic_minibatch_steps() {
        use crate::torch::planner::{PLANNER_BELIEF_DIM, PLANNER_LATENT_DIM};

        let transition = |decision_index: usize| PlannerTransition {
            observation: PlannerObservation {
                forecast_latent: Tensor::randn([2, PLANNER_LATENT_DIM], (Kind::Float, Device::Cpu)),
                relative_horizon: Tensor::from_slice(&[0.5f32, 1.0]).view([2, 1]),
                return_quantiles: Tensor::from_slice(&[-0.01f32, 0.0, 0.01, -0.02, 0.0, 0.02])
                    .view([2, PLANNER_RETURN_QUANTILES]),
                belief: Tensor::randn([PLANNER_BELIEF_DIM], (Kind::Float, Device::Cpu)),
                portfolio_state: Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.0]),
            },
            environment_id: 0,
            decision_index,
            action: Tensor::from_slice(&[0.5f32]),
            old_alpha: Tensor::from_slice(&[50.0f32]),
            old_beta: Tensor::from_slice(&[1.1f32]),
            value: Tensor::from_slice(&[0.0f32]),
            next_value: Some(Tensor::from_slice(&[0.0f32])),
            reward: if decision_index == 0 { 0.2 } else { -0.1 },
            next_log_return: if decision_index == 0 { 0.01 } else { -0.02 },
            terminated: false,
            truncated: decision_index == 1,
            commission: 0.0,
            turnover: 0.1,
            executed_stock_weight: 0.5,
            assets_before: 100.0,
            assets_after: 100.0,
        };
        let mut rollout = PlannerRollout::new(2, "wm-test".to_owned()).unwrap();
        rollout.push(transition(0)).unwrap();
        rollout.push(transition(1)).unwrap();
        let batch = rollout.to_batch(Device::Cpu).unwrap();
        let vs = nn::VarStore::new(Device::Cpu);
        let planner = WorldModelPlanner::new(&vs.root());
        let hl_gauss = HlGaussBins::planner(&(vs.root() / "value_running_stats"), Device::Cpu);
        let named = named_trainable_variables(&vs);
        let (actor_named, critic_named) = split_planner_parameters(&named).unwrap();
        let actor_vars = actor_named
            .iter()
            .map(|(_, parameter)| parameter.shallow_clone())
            .collect::<Vec<_>>();
        let critic_vars = critic_named
            .iter()
            .map(|(_, parameter)| parameter.shallow_clone())
            .collect::<Vec<_>>();
        let mut actor_optimizer = new_planner_optimizer(&actor_named);
        let mut critic_optimizer = new_planner_optimizer(&critic_named);
        let mut actor_optimizer_steps = 0;
        let mut critic_optimizer_steps = 0;
        let mut controller = KlLrController::new(
            TARGET_KL,
            KL_CONTROLLER_HALF_LIFE,
            KL_MIN_LR_SCALE,
            KL_MAX_LR_SCALE,
        );
        let summary = optimize_rollout(
            &planner,
            &rollout,
            &batch,
            &Tensor::from_slice(&[1.0f32, -1.0]),
            &Tensor::from_slice(&[0.4f32, -0.2]),
            1,
            7,
            &hl_gauss,
            &actor_vars,
            &critic_vars,
            &mut actor_optimizer,
            &mut critic_optimizer,
            &mut actor_optimizer_steps,
            &mut critic_optimizer_steps,
            &mut controller,
            Device::Cpu,
        )
        .unwrap();

        assert!(summary.kl_early_stopped);
        assert!(summary.actor_steps < summary.actor_available_steps);
        assert_eq!(summary.critic_steps, summary.critic_available_steps);
        assert_eq!(
            critic_optimizer_steps as usize,
            summary.critic_available_steps
        );
        assert_eq!(actor_optimizer_steps as usize, summary.actor_steps);
    }

    #[test]
    fn reverse_kl_clamps_only_float32_roundoff_negatives() {
        let tiny_negative = -0.5 * KL_NEGATIVE_ROUNDOFF_TOLERANCE;
        assert_eq!(validated_reverse_kl(tiny_negative).unwrap(), 0.0);
        assert_eq!(validated_reverse_kl(0.0).unwrap(), 0.0);
        assert_eq!(validated_reverse_kl(0.01).unwrap(), 0.01);

        let mut summary = OptimizationSummary {
            actor_steps: 1,
            ..OptimizationSummary::default()
        };
        let (stopped, sanitized) =
            kl_stops_before_optimizer_step(&mut summary, tiny_negative).unwrap();
        assert!(!stopped);
        assert_eq!(sanitized, 0.0);
        assert_eq!(summary.max_reverse_kl, 0.0);
        assert!(!summary.kl_early_stopped);
    }

    #[test]
    fn reverse_kl_rejects_material_negatives_and_non_finite_values() {
        assert!(validated_reverse_kl(-2.0 * KL_NEGATIVE_ROUNDOFF_TOLERANCE).is_err());
        assert!(validated_reverse_kl(f64::NAN).is_err());
        assert!(validated_reverse_kl(f64::INFINITY).is_err());
        assert!(validated_reverse_kl(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn resume_manifest_switches_only_after_an_immutable_bundle_commits() {
        let dir = std::env::temp_dir().join(format!(
            "planner-resume-manifest-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let weights = dir.join("weights");
        fs::create_dir_all(&weights).unwrap();
        let base = weights.join("planner.ot");
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = vs.root().zeros("weight", &[2, 2]);
        let named = named_trainable_variables(&vs);
        let save_resume = |update: u64| {
            let mut optimizer = Muon::new_named(
                &named,
                MuonConfig {
                    quiet: true,
                    ..MuonConfig::default()
                },
            );
            for _ in 0..update {
                named[0].1.square().sum(Kind::Float).backward();
                optimizer.step(StepKind::Primary);
                optimizer.zero_grad();
            }
            let path = resume_checkpoint_path(&base, update);
            let committed = save_planner_checkpoint(
                &vs,
                &path,
                &PlannerCheckpointMetadata::new(
                    "lineage-a",
                    "world-weights-a",
                    100,
                    128,
                    update,
                    update,
                    update,
                    "run-a",
                    7,
                    TARGET_KL,
                    1.0,
                ),
                &optimizer,
                &optimizer,
            )
            .unwrap();
            (path, committed)
        };

        let (first_path, first_metadata) = save_resume(1);
        commit_resume_manifest(&base, &first_path, &first_metadata).unwrap();
        assert_eq!(
            resolve_resume_checkpoint(&base, "lineage-a", Some(100), Some(7))
                .unwrap()
                .0,
            first_path
        );

        // A fully staged newer bundle is invisible until the atomic manifest
        // rename commits it, so interruption here resumes the previous update.
        let (second_path, _) = save_resume(2);
        assert!(validate_output_manifest_for_resume(&base, "run-a", 2).is_err());
        assert_eq!(
            resolve_resume_checkpoint(&base, "lineage-a", Some(100), Some(7))
                .unwrap()
                .0,
            first_path
        );
        cleanup_uncommitted_resume_bundles(&base, 1, "run-a").unwrap();
        assert!(!second_path.exists());
        assert!(!planner_metadata_path(&second_path).exists());
        assert!(!planner_actor_optimizer_state_path(&second_path).exists());
        assert!(!planner_critic_optimizer_state_path(&second_path).exists());
        let (second_path, second_metadata) = save_resume(2);
        commit_resume_manifest(&base, &second_path, &second_metadata).unwrap();
        let (_, resolved) =
            resolve_resume_checkpoint(&base, "lineage-a", Some(100), Some(7)).unwrap();
        assert_eq!(resolved.cumulative_updates, 2);
        assert_eq!(resolved.run_lineage_id, "run-a");
        assert!(validate_output_manifest_for_resume(&base, "run-a", 1).is_err());
        assert!(second_path.exists());

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resume_cleanup_refuses_mismatched_future_resume_bundle() {
        let dir = std::env::temp_dir().join(format!(
            "planner-resume-cleanup-owner-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let weights = dir.join("weights");
        fs::create_dir_all(&weights).unwrap();
        let base = weights.join("planner.ot");
        let future = resume_checkpoint_path(&base, 2);
        fs::write(&future, b"preserve").unwrap();
        let metadata = PlannerCheckpointMetadata::new(
            "lineage-a",
            "world-weights-a",
            100,
            128,
            2,
            2,
            2,
            "run-b",
            7,
            TARGET_KL,
            1.0,
        );
        fs::write(
            planner_metadata_path(&future),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();

        assert!(cleanup_uncommitted_resume_bundles(&base, 1, "run-a").is_err());
        assert_eq!(fs::read(&future).unwrap(), b"preserve");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fresh_training_rejects_reused_output_bundle_and_sidecars() {
        let dir = std::env::temp_dir().join(format!(
            "planner-fresh-output-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("planner.ot");
        ensure_fresh_resume_output(&checkpoint).unwrap();
        for occupied in [
            checkpoint.clone(),
            planner_metadata_path(&checkpoint),
            planner_actor_optimizer_state_path(&checkpoint),
            planner_critic_optimizer_state_path(&checkpoint),
            resume_manifest_path(&checkpoint),
            resume_checkpoint_path(&checkpoint, 1),
            checkpoint.with_file_name("planner_best_u00000001.ot"),
        ] {
            fs::write(&occupied, "occupied").unwrap();
            assert!(ensure_fresh_resume_output(&checkpoint).is_err());
            fs::remove_file(occupied).unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fresh_training_rejects_any_preexisting_generation_data() {
        let dir = std::env::temp_dir().join(format!(
            "planner-fresh-gens-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        ensure_fresh_planner_gens(&dir, 0).unwrap();
        fs::create_dir(dir.join("1")).unwrap();
        assert!(ensure_fresh_planner_gens(&dir, 0).is_err());
        ensure_fresh_planner_gens(&dir, 1).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn buy_and_hold_is_the_raw_index_return_at_exact_endpoint_prices() {
        let ratio = buy_and_hold_wealth_ratio(10.0, 12.0);
        assert!((ratio - 1.2).abs() < 1e-12);
    }

    #[test]
    fn benchmark_metrics_compare_each_policy_to_its_paired_market_path() {
        let metrics = paired_benchmark_metrics(&[1.4, 0.9, 1.05], &[1.2, 1.0, 1.0]).unwrap();

        assert!((metrics.mean_buy_and_hold_wealth_ratio - 1.0666666667).abs() < 1e-10);
        assert!((metrics.mean_outperformance_ratio - 0.05).abs() < 1e-12);
        assert!((metrics.median_outperformance_ratio - 0.05).abs() < 1e-12);
        assert!((metrics.outperformance_fraction - 2.0 / 3.0).abs() < 1e-12);
    }

    /// A hermetic corpus plus a structurally real, randomly-initialized frozen world
    /// model. Both are needed because the collection loop's contract is exactly the
    /// agreement between the two: the corpus decides how far an endpoint may look
    /// ahead, and the world model decides what a forecast looks like.
    struct RolloutFixture {
        dir: PathBuf,
        corpus: PlannerCorpus,
        world_model: BarWorldModel,
    }

    impl Drop for RolloutFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    impl RolloutFixture {
        const RES_SECS: u32 = 300;
        const BARS: i64 = 1_600;

        fn new(label: &str) -> Self {
            use crate::torch::world_model::{world_model_supports_path, BarModules};
            use rand::Rng;
            use shared::bars::{write_bar_file, PackedBar, FILE_EXTENSION};

            let dir = std::env::temp_dir().join(format!(
                "planner-rollout-{label}-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            fs::create_dir_all(&dir).unwrap();
            let step_ms = Self::RES_SECS as i64 * 1_000;
            for (symbol, seed) in [("AAA", 11u64), ("BBB", 12)] {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut close = 100.0f32;
                let bars = (0..Self::BARS)
                    .map(|slot| {
                        let open = close;
                        close = (close * (1.0 + rng.random_range(-0.01f32..0.01))).max(1.0);
                        let spread = rng.random_range(0.0f32..0.02) * open;
                        PackedBar {
                            ts_ms: 1_600_000_000_000i64 / step_ms * step_ms + slot * step_ms,
                            open,
                            high: open.max(close) + spread,
                            low: (open.min(close) - spread).max(0.5),
                            close,
                            volume: rng.random_range(1_000.0f32..50_000.0),
                            vwap: 0.5 * (open + close),
                            trades: rng.random_range(1u32..500),
                        }
                    })
                    .collect::<Vec<_>>();
                write_bar_file(
                    &dir.join(format!("{symbol}.{}.{FILE_EXTENSION}", Self::RES_SECS)),
                    symbol,
                    Self::RES_SECS,
                    &bars,
                )
                .unwrap();
            }
            let corpus = PlannerCorpus::load(&dir, Self::RES_SECS, 512).unwrap();

            // The loader accepts nothing but a hash-consistent weights/supports/metadata
            // triple, so the fixture writes a real one rather than reaching past `load`.
            let weights = dir.join("world_model.ot");
            let supports_path = world_model_supports_path(&weights, Self::RES_SECS);
            corpus
                .corpus()
                .fit_supports(4_096, 5)
                .save(&supports_path)
                .unwrap();
            let mut world_vs = nn::VarStore::new(Device::Cpu);
            let _modules = BarModules::new(&world_vs.root());
            world_vs.freeze();
            world_vs.save(&weights).unwrap();
            let metadata_path =
                crate::torch::world_model::BarWorldModelMetadata::save_for_checkpoint(
                    &weights,
                    &[Self::RES_SECS],
                    Self::RES_SECS,
                )
                .unwrap();
            let world_model = BarWorldModel::load(&weights, &metadata_path, Device::Cpu).unwrap();

            Self {
                dir,
                corpus,
                world_model,
            }
        }
    }

    #[test]
    fn forecast_summary_carries_belief_moments_and_ordered_return_quantiles() {
        use crate::torch::planner::{PLANNER_BELIEF_DIM, PLANNER_LATENT_DIM};

        tch::manual_seed(23);
        let fixture = RolloutFixture::new("summary");
        let horizon = 4i64;
        let endpoints = fixture
            .corpus
            .deterministic_endpoints(PlannerDataSplit::Train, 2, 64, 8)
            .unwrap();
        let start = fixture
            .corpus
            .context_batch(&endpoints, &[0, 0], 64, Device::Cpu)
            .unwrap();
        let session = fixture
            .world_model
            .start_session(&start.dof, &start.time_ids);
        let future_time_ids = fixture
            .corpus
            .corpus()
            .future_time_ids(&endpoints, 0, horizon, Device::Cpu)
            .unwrap();
        // Six samples with a chunk of four: the ragged final chunk is the case a
        // fixed-stride accumulator would silently miscount.
        let summary =
            summarize_forecast(&fixture.world_model, &session, &future_time_ids, 6).unwrap();
        validate_forecast_finite(&summary).unwrap();

        assert_eq!(summary.latent.size(), [2, horizon, PLANNER_LATENT_DIM]);
        assert_eq!(
            summary.quantiles.size(),
            [2, horizon, PLANNER_RETURN_QUANTILES]
        );
        // p10 <= p50 <= p90 at every step. This is what pins the quantile axis order:
        // a wrong permute scatters the levels across the horizon and breaks it.
        let quantiles = &summary.quantiles;
        for level in 0..PLANNER_RETURN_QUANTILES - 1 {
            let lower = quantiles.select(-1, level);
            let upper = quantiles.select(-1, level + 1);
            assert!(bool::try_from(lower.le_tensor(&upper).all()).unwrap());
        }
        // Every sample's first imagined step is drawn from the one decision-time belief,
        // so the across-sample spread there is exactly zero and grows afterwards. This
        // also pins the mean-then-standard-deviation halves of the latent.
        let deviation = summary
            .latent
            .narrow(-1, PLANNER_BELIEF_DIM, PLANNER_BELIEF_DIM);
        assert_eq!(
            deviation.select(1, 0).abs().max().double_value(&[]),
            0.0,
            "the first imagined step has a single belief, so its spread must vanish"
        );
        assert!(
            deviation
                .select(1, horizon - 1)
                .abs()
                .max()
                .double_value(&[])
                > 0.0
        );
        let mean = summary.latent.narrow(-1, 0, PLANNER_BELIEF_DIM);
        assert!(mean
            .select(1, 0)
            .allclose(session.belief(), 1e-5, 1e-6, false));
    }

    #[test]
    fn real_rollout_collection_reserves_room_for_the_horizon_behind_the_episode() {
        tch::manual_seed(29);
        let fixture = RolloutFixture::new("collect");
        let horizon = 3;
        let rollout_length = 2;
        let environments = 2;
        let planner_vs = nn::VarStore::new(Device::Cpu);
        let planner = WorldModelPlanner::new(&planner_vs.root());
        let hl_gauss =
            HlGaussBins::planner(&(planner_vs.root() / "value_running_stats"), Device::Cpu);
        let mut rng = StdRng::seed_from_u64(31);

        let collected = collect_real_rollout(
            &planner,
            &fixture.world_model,
            &fixture.corpus,
            &hl_gauss,
            horizon,
            rollout_length,
            environments,
            64,
            &mut rng,
            Device::Cpu,
        )
        .unwrap();

        // A dense [rollout_length, environments] grid of transitions, each of which had
        // to pass `validate_transition` — including the return-quantile shape — on push.
        assert_eq!(collected.rollout.len(), environments * rollout_length);
        assert_eq!(collected.primary_trace.total.len(), rollout_length + 1);
        assert!(collected
            .benchmark
            .mean_buy_and_hold_wealth_ratio
            .is_finite());
        assert!(collected.deterministic.reward_mean.is_finite());
        // The last bar a `rollout_length`-only reservation admits is
        // `series_len - rollout_length - 1`; from there the final decision's horizon runs
        // off the end of the series. Adding `horizon` to the reservation moves the
        // boundary back by exactly enough, and not more.
        let series_len = fixture.corpus.corpus().series_len(0);
        let future_at = |bar: usize| {
            fixture.corpus.corpus().future_time_ids(
                &[PlannerEndpoint { series: 0, bar }],
                rollout_length,
                horizon as i64,
                Device::Cpu,
            )
        };
        assert!(future_at(series_len - rollout_length - 1).is_err());
        assert!(future_at(series_len - rollout_length - horizon - 1).is_ok());
    }
}
