mod optimizer;
mod readout_fit;
mod reports;
mod rollout;
mod tail_ema;

use std::{
    path::Path,
    sync::{mpsc, Arc},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use tch::{autocast, nn, Device, Kind, Tensor};

use crate::torch::bar_dist::{
    bar_nll_from_logits, BarScoring, BarSupports, BAR_DOF, BAR_SUPPORTS_SEMANTICS_CONTRACT,
    RAW_SCALING_CONTRACT,
};
use crate::torch::cuda::cfg::{
    configure_cuda, disable_autograd_multithreading, enable_exact_fp32_matmul, enable_tf32_matmul,
    pin_bfloat16_autocast,
};
use crate::torch::lejepa::checkpoint::{
    save_bundle, CheckpointProvenance, CoreInitialization, CoreTrainingOrigin,
    EmissionGradientMode, WeightReadout,
};
use crate::torch::lejepa::dataset::{MseJepaDataset, MseJepaTrainHostBatch};
use crate::torch::lejepa::model::{LATENT_DIM, MAX_CONTEXT_BARS};
use crate::torch::lejepa::sigreg::{
    sample_temporal_views, sigreg_loss, sigreg_loss_with_directions, temporal_view_indices,
    SIGREG_MAX_VIEWS, SIGREG_PROJECTIONS,
};
use crate::torch::lejepa::{MseJepaForward, MseJepaModel};
use crate::torch::optim::muon::StepKind;

use super::optimizer_glue::named_trainable_variables;
use optimizer::{build_optimizer, DbwmSchedule, OptimizerPoint, RECIPE_ID};
use reports::{MseJepaReporter, StepMetrics, ValidationMetrics};
use tail_ema::TailEma;

pub const DEFAULT_BATCH_SIZE: usize = 8;
pub const DEFAULT_VALIDATION_WINDOWS: usize = DEFAULT_BATCH_SIZE;
pub const SIGREG_BATCH_SIZE: usize = 128;
pub const DEFAULT_CHECKPOINT_EVERY: usize = 2_048;
/// Flow-matching weight in
/// `L = CE_emission(beliefs; arm) + lambda_flow * L_flow + lambda_sigreg * SIGReg`.
pub const DEFAULT_LAMBDA_FLOW: f64 = 1.0;
/// Heun steps (two velocity evaluations each) per latent draw.
pub const DEFAULT_FLOW_STEPS: usize = 8;
/// Heun draws per validation position behind the sample diagnostics (energy score m).
pub const VALIDATION_SAMPLE_DRAWS: i64 = 4;
pub use crate::torch::lejepa::checkpoint::EmissionGradientMode as MseJepaEmissionGradientMode;
pub use readout_fit::{
    fit_mse_jepa_readouts, FitMseJepaReadoutsArgs, DEFAULT_READOUT_FIT_BATCH_SIZE,
    DEFAULT_READOUT_FIT_SEED, DEFAULT_READOUT_FIT_STEPS, DEFAULT_READOUT_TOKEN_ROWS,
    DEFAULT_READOUT_VALIDATION_WINDOWS,
};
pub use rollout::{
    evaluate_mse_jepa_rollout, EvaluateMseJepaRolloutArgs, DEFAULT_ROLLOUT_HORIZONS,
    DEFAULT_ROLLOUT_SAMPLES, DEFAULT_ROLLOUT_VALIDATION_WINDOWS, DEFAULT_ROLLOUT_WINDOW_CHUNK,
};

#[derive(Clone, Debug)]
pub struct MseJepaArgs {
    pub run: Option<String>,
    pub epochs: usize,
    pub steps: Option<usize>,
    pub batch_size: usize,
    pub emission_gradient_mode: EmissionGradientMode,
    pub seed: u64,
    pub data_dir: String,
    pub resolution_secs: u32,
    pub min_bars: usize,
    pub validation_windows: usize,
    pub validate_every: usize,
    pub checkpoint_every: usize,
    pub lambda_flow: f64,
    pub lambda_sigreg: f64,
    pub flow_steps: usize,
    pub split_bounds: Option<(i64, i64)>,
    pub derive_split_bounds: bool,
}

/// Fixed validation randomness, drawn once so every validation pass scores the same
/// interpolants and the same Heun base draws. Host-resident: it is only needed between
/// optimizer steps and would otherwise pin half a gigabyte of device memory for the run.
struct ValidationFlowPanel {
    x0: Tensor,
    tau: Tensor,
    sample_noise: Tensor,
}

impl ValidationFlowPanel {
    fn draw(windows: i64) -> Self {
        let options = (Kind::Float, Device::Cpu);
        tch::no_grad(|| Self {
            x0: Tensor::randn([windows, 1, MAX_CONTEXT_BARS, LATENT_DIM], options),
            tau: Tensor::randn([windows, 1, MAX_CONTEXT_BARS, 1], options).sigmoid(),
            sample_noise: Tensor::randn(
                [
                    VALIDATION_SAMPLE_DRAWS,
                    windows,
                    1,
                    MAX_CONTEXT_BARS,
                    LATENT_DIM,
                ],
                options,
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HostBatchKey {
    epoch: usize,
    batch_index: usize,
    step: usize,
}

struct HostBatchRequest {
    key: HostBatchKey,
    temporal_offsets: Vec<i64>,
}

struct MseJepaBatchPrefetcher {
    requests: mpsc::SyncSender<HostBatchRequest>,
    ready: mpsc::Receiver<(HostBatchKey, MseJepaTrainHostBatch)>,
    outstanding: Option<HostBatchKey>,
}

impl MseJepaBatchPrefetcher {
    fn new(dataset: Arc<MseJepaDataset>, prediction_batch_size: usize) -> Self {
        let (requests, incoming) = mpsc::sync_channel::<HostBatchRequest>(1);
        let (outgoing, ready) = mpsc::sync_channel::<(HostBatchKey, MseJepaTrainHostBatch)>(1);
        std::thread::Builder::new()
            .name("mse-jepa-prefetch".to_owned())
            .spawn(move || {
                let mut scratch = dataset.train_scratch();
                while let Ok(request) = incoming.recv() {
                    let batch = dataset.train_host_batch(
                        request.key.epoch,
                        request.key.batch_index,
                        prediction_batch_size,
                        request.key.step,
                        SIGREG_BATCH_SIZE,
                        &request.temporal_offsets,
                        &mut scratch,
                    );
                    if outgoing.send((request.key, batch)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawning the MSE-JEPA prefetch thread");
        Self {
            requests,
            ready,
            outstanding: None,
        }
    }

    fn request(&mut self, key: HostBatchKey, temporal_offsets: Vec<i64>) {
        assert!(
            self.outstanding.is_none(),
            "MSE-JEPA host prefetch already has an outstanding batch"
        );
        self.requests
            .send(HostBatchRequest {
                key,
                temporal_offsets,
            })
            .expect("the MSE-JEPA prefetch thread is alive");
        self.outstanding = Some(key);
    }

    fn take(&mut self, key: HostBatchKey) -> MseJepaTrainHostBatch {
        let outstanding = self
            .outstanding
            .take()
            .expect("MSE-JEPA host batch was requested");
        assert_eq!(outstanding, key, "MSE-JEPA prefetch key drifted");
        let (produced, batch) = self
            .ready
            .recv()
            .expect("the MSE-JEPA prefetch thread is alive");
        assert_eq!(produced, key, "MSE-JEPA prefetch produced the wrong batch");
        batch
    }
}

pub fn pretrain_mse_jepa(args: MseJepaArgs) -> Result<()> {
    validate_args(&args)?;
    super::pretrain::configure_threads();
    configure_cuda();
    enable_tf32_matmul()?;
    println!("CUDA training fp32 GEMMs configured for TF32 tensor cores");
    let device = Device::cuda_if_available();
    ensure!(
        device.is_cuda(),
        "pretrain-mse-jepa requires CUDA and strict FA4; CPU is supported only by focused tests"
    );
    let _autograd_guard = disable_autograd_multithreading();
    tch::manual_seed(args.seed as i64);

    let split_bounds = if args.derive_split_bounds {
        None
    } else {
        Some(
            args.split_bounds
                .unwrap_or(crate::data::ingest::PINNED_SPLIT_BOUNDS),
        )
    };
    let dataset = Arc::new(MseJepaDataset::load(
        Path::new(&args.data_dir),
        args.resolution_secs,
        args.min_bars,
        split_bounds,
        args.seed,
    )?);
    dataset.advise_random_access()?;
    let supports = load_authenticated_supports(&dataset)?;
    let core_origin = CoreTrainingOrigin {
        initialization: CoreInitialization::Fresh,
        train_seed: args.seed,
        batch_size: args.batch_size as u64,
        resolution_secs: args.resolution_secs,
        min_bars: args.min_bars as u64,
        split_bounds: dataset.split_bounds(),
        split_bounds_pinned: !args.derive_split_bounds,
        corpus_fingerprint: dataset.corpus_fingerprint(),
        supports_scoring_sha256: supports.scoring_sha256()?,
    };
    let prediction_windows_per_pass = dataset.prediction_windows_per_epoch();
    let steps_per_pass = dataset.batches_per_epoch(args.batch_size);
    ensure!(
        steps_per_pass > 0,
        "--batch-size {} exceeds the MSE-JEPA training window count",
        args.batch_size
    );
    let sigreg_batches_per_pass = dataset.sigreg_batches_per_pass(SIGREG_BATCH_SIZE);
    ensure!(
        sigreg_batches_per_pass > 0,
        "MSE-JEPA needs at least {SIGREG_BATCH_SIZE} training windows for independent SIGReg samples"
    );
    let available_steps = args.epochs.saturating_mul(steps_per_pass);
    let planned_steps = args.steps.unwrap_or(available_steps).min(available_steps);
    let schedule = DbwmSchedule::new(steps_per_pass, planned_steps);

    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    let var_store = nn::VarStore::new(device);
    let model = MseJepaModel::new(&var_store.root());
    // Core training always starts from the seed-bound fresh initialization recorded above.
    let named = named_trainable_variables(&var_store);
    let mut optimizer = build_optimizer(&named);
    let mut tail_ema = TailEma::new(&named, planned_steps)?;
    let mut reporter = MseJepaReporter::new(&run.gens, supports.marginal_nll_bar(BarScoring::Hard));
    let validation_refs = dataset.validation_refs(args.validation_windows);
    ensure!(
        !validation_refs.is_empty(),
        "MSE-JEPA validation panel is empty"
    );
    let validation_next_dof = dataset.validation_next_dof(&validation_refs, device);
    let validation_sigreg_offsets = temporal_offsets(false)?;
    let validation_sigreg_bars =
        dataset.validation_sigreg_batch(SIGREG_BATCH_SIZE, &validation_sigreg_offsets, device)?;
    let validation_sigreg_directions = fixed_validation_sigreg_directions(device, args.seed);
    // Drawn once at a fixed point after the top-level manual_seed, so raw and tail-EMA
    // validation use the identical stochastic panel.
    let validation_flow = ValidationFlowPanel::draw(validation_refs.len() as i64);
    // Validation construction consumes the process-global CPU generator in proportion to the
    // panel size. Restart both training generators here so evaluation-only settings cannot
    // change SIGReg temporal views, flow noise, or token-probe row sampling.
    tch::manual_seed(args.seed as i64);
    tch::Cuda::manual_seed_all(args.seed);
    let mut best_validation = f64::INFINITY;
    let mut last_validation = None;
    let mut global_step = 0usize;
    let mut windows_seen = 0usize;
    let mut last_validated_step = 0usize;
    let mut final_epoch = 0usize;
    let requested_steps = args.steps.unwrap_or(usize::MAX);
    let mut prefetch = MseJepaBatchPrefetcher::new(Arc::clone(&dataset), args.batch_size);
    prefetch.request(
        HostBatchKey {
            epoch: 0,
            batch_index: 0,
            step: 0,
        },
        temporal_offsets(true)?,
    );

    for epoch in 0..args.epochs {
        final_epoch = epoch;
        for batch_index in 0..steps_per_pass {
            if global_step >= requested_steps {
                break;
            }
            let step_started = Instant::now();
            let optimizer_point = schedule.point(global_step);
            optimizer_point.apply(&mut optimizer);
            optimizer.zero_grad();
            let key = HostBatchKey {
                epoch,
                batch_index,
                step: global_step,
            };
            let host_batch = prefetch.take(key);
            let prediction_windows = host_batch.prediction_windows();
            if global_step + 1 < planned_steps {
                let (next_epoch, next_batch_index) = if batch_index + 1 < steps_per_pass {
                    (epoch, batch_index + 1)
                } else {
                    (epoch + 1, 0)
                };
                prefetch.request(
                    HostBatchKey {
                        epoch: next_epoch,
                        batch_index: next_batch_index,
                        step: global_step + 1,
                    },
                    temporal_offsets(true)?,
                );
            }
            let (bars, next_dof, sigreg_bars) = host_batch.to_device(device);
            let losses = objective(
                &model,
                &supports,
                &bars,
                &next_dof,
                &sigreg_bars,
                ObjectiveWeights {
                    lambda_flow: args.lambda_flow,
                    lambda_sigreg: args.lambda_sigreg,
                },
                args.emission_gradient_mode,
                true,
                (global_step + 1) % REPRESENTATION_STATS_EVERY == 0,
                None,
                None,
            );
            losses.total.backward();
            let grad_norm = global_grad_norm_tensor(&named, device);
            let packed_metrics = pack_step_metrics(&losses, &grad_norm);
            optimizer.step(StepKind::Primary);
            let metric_values = read_packed_step_metrics(&packed_metrics);
            let completed_step_seconds = step_started.elapsed().as_secs_f64();
            global_step += 1;
            windows_seen += prediction_windows;
            tail_ema.update(global_step);

            reporter.record_step(step_metrics(
                metric_values,
                optimizer_point,
                windows_seen as f64 / prediction_windows_per_pass as f64,
                completed_step_seconds,
                prediction_windows,
                SIGREG_BATCH_SIZE,
            ));
            if args.validate_every > 0 && global_step % args.validate_every == 0 {
                last_validated_step = global_step;
                let validation = validation_metrics(
                    &model,
                    &supports,
                    &dataset,
                    &validation_refs,
                    &validation_next_dof,
                    &validation_sigreg_bars,
                    &args,
                    &validation_sigreg_directions,
                    &validation_flow,
                    device,
                )?;
                reporter.record_validation(validation);
                last_validation = Some(validation);
                if validation.total_loss < best_validation {
                    best_validation = validation.total_loss;
                    save_bundle(
                        &var_store,
                        run.weights.join("mse_jepa_best.ot"),
                        args.lambda_sigreg,
                        args.lambda_flow,
                        checkpoint_provenance(
                            global_step,
                            planned_steps,
                            steps_per_pass,
                            args.emission_gradient_mode,
                            WeightReadout::Raw,
                            &core_origin,
                        ),
                    )?;
                }
                reporter.write(epoch)?;
            }
            if args.checkpoint_every > 0 && global_step % args.checkpoint_every == 0 {
                save_bundle(
                    &var_store,
                    run.weights.join(format!("mse_jepa_step{global_step}.ot")),
                    args.lambda_sigreg,
                    args.lambda_flow,
                    checkpoint_provenance(
                        global_step,
                        planned_steps,
                        steps_per_pass,
                        args.emission_gradient_mode,
                        WeightReadout::Raw,
                        &core_origin,
                    ),
                )?;
            }
        }

        if global_step > 0 && last_validated_step != global_step {
            last_validated_step = global_step;
            let validation = validation_metrics(
                &model,
                &supports,
                &dataset,
                &validation_refs,
                &validation_next_dof,
                &validation_sigreg_bars,
                &args,
                &validation_sigreg_directions,
                &validation_flow,
                device,
            )?;
            reporter.record_validation(validation);
            last_validation = Some(validation);
            if validation.total_loss < best_validation {
                best_validation = validation.total_loss;
                save_bundle(
                    &var_store,
                    run.weights.join("mse_jepa_best.ot"),
                    args.lambda_sigreg,
                    args.lambda_flow,
                    checkpoint_provenance(
                        global_step,
                        planned_steps,
                        steps_per_pass,
                        args.emission_gradient_mode,
                        WeightReadout::Raw,
                        &core_origin,
                    ),
                )?;
            }
            reporter.write(epoch)?;
        }
        if global_step >= requested_steps {
            break;
        }
    }

    save_bundle(
        &var_store,
        run.weights.join("mse_jepa.ot"),
        args.lambda_sigreg,
        args.lambda_flow,
        checkpoint_provenance(
            global_step,
            planned_steps,
            steps_per_pass,
            args.emission_gradient_mode,
            WeightReadout::Raw,
            &core_origin,
        ),
    )?;
    if tail_ema.is_initialized() {
        ensure!(
            last_validation.is_some(),
            "tail-EMA readout requires paired raw validation"
        );
        let (tail_ema_var_store, tail_ema_model) = tail_ema.materialize(device)?;
        let tail_ema_validation = validation_metrics(
            &tail_ema_model,
            &supports,
            &dataset,
            &validation_refs,
            &validation_next_dof,
            &validation_sigreg_bars,
            &args,
            &validation_sigreg_directions,
            &validation_flow,
            device,
        )?;
        reporter.record_tail_ema_validation(tail_ema_validation);
        save_bundle(
            &tail_ema_var_store,
            run.weights.join("mse_jepa_tail_ema.ot"),
            args.lambda_sigreg,
            args.lambda_flow,
            checkpoint_provenance(
                global_step,
                planned_steps,
                steps_per_pass,
                args.emission_gradient_mode,
                tail_ema.readout()?,
                &core_origin,
            ),
        )?;
        reporter.write(final_epoch)?;
    }
    println!(
        "Saved raw and available tail-EMA MSE-JEPA bundles under {}",
        run.weights.display()
    );
    Ok(())
}
fn checkpoint_provenance(
    completed_steps: usize,
    planned_steps: usize,
    steps_per_pass: usize,
    emission_gradient_mode: EmissionGradientMode,
    weight_readout: WeightReadout,
    core_origin: &CoreTrainingOrigin,
) -> CheckpointProvenance {
    CheckpointProvenance {
        completed_steps: completed_steps as u64,
        planned_steps: planned_steps as u64,
        steps_per_pass: steps_per_pass as u64,
        optimizer_recipe_id: RECIPE_ID.to_owned(),
        emission_gradient_mode,
        weight_readout,
        core_origin: core_origin.clone(),
        head_fit: None,
    }
}

fn validate_args(args: &MseJepaArgs) -> Result<()> {
    ensure!(args.epochs > 0, "--epochs must be positive");
    ensure!(
        args.steps != Some(0),
        "--steps must be positive when specified"
    );
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    ensure!(
        args.resolution_secs == 300,
        "pretrain-mse-jepa requires the train-fitted raw 300s BarSupports"
    );
    ensure!(
        args.min_bars >= (MAX_CONTEXT_BARS + 2) as usize,
        "--min-bars must admit a full context plus target"
    );
    ensure!(
        args.validation_windows > 0,
        "--validation-windows must be positive"
    );
    ensure!(
        args.lambda_flow.is_finite() && args.lambda_flow >= 0.0,
        "--lambda-flow must be finite and non-negative"
    );
    ensure!(
        args.lambda_sigreg.is_finite() && args.lambda_sigreg >= 0.0,
        "--lambda-sigreg must be finite and non-negative"
    );
    ensure!(args.flow_steps > 0, "--flow-steps must be positive");
    ensure!(
        !(args.derive_split_bounds && args.split_bounds.is_some()),
        "--derive-split-bounds conflicts with --split-bounds"
    );
    Ok(())
}

pub(super) fn load_authenticated_supports(dataset: &MseJepaDataset) -> Result<BarSupports> {
    let supports_path = dataset.supports_path();
    let supports = BarSupports::load(&supports_path).with_context(|| {
        format!(
            "loading train-fitted raw supports {}",
            supports_path.display()
        )
    })?;
    authenticate_supports(&supports, dataset)?;
    Ok(supports)
}

pub(super) fn authenticate_supports(
    supports: &BarSupports,
    dataset: &MseJepaDataset,
) -> Result<()> {
    ensure!(
        supports.dof_scaling() == crate::torch::bar_dist::DofScaling::Raw,
        "MSE-JEPA emission requires raw 300s BarSupports, not {}",
        supports.dof_scaling()
    );
    ensure!(
        supports.bin_means_measured(),
        "raw BarSupports lack fitted bin moments"
    );
    let provenance = supports
        .provenance()
        .context("raw BarSupports lack authenticated fit provenance")?;
    ensure!(
        provenance.corpus_fingerprint == dataset.corpus_fingerprint(),
        "BarSupports corpus fingerprint does not match the training corpus"
    );
    ensure!(
        provenance.split_bounds == dataset.split_bounds(),
        "BarSupports split bounds do not match the training corpus"
    );
    ensure!(
        provenance.sample_count > 0,
        "BarSupports fit sample count is zero"
    );
    ensure!(
        provenance.fit_seed.is_some(),
        "BarSupports fit seed is absent"
    );
    ensure!(
        provenance.scaling_contract.as_deref() == Some(RAW_SCALING_CONTRACT),
        "BarSupports scaling provenance is not the raw target contract"
    );
    ensure!(
        provenance.support_semantics.as_deref() == Some(BAR_SUPPORTS_SEMANTICS_CONTRACT),
        "BarSupports semantics provenance mismatch"
    );
    Ok(())
}

pub(super) fn authenticate_core_origin(
    provenance: &CheckpointProvenance,
    dataset: &MseJepaDataset,
    supports: &BarSupports,
    resolution_secs: u32,
    min_bars: usize,
    split_bounds_pinned: bool,
) -> Result<()> {
    let origin = &provenance.core_origin;
    ensure!(
        origin.resolution_secs == resolution_secs
            && origin.resolution_secs == 300
            && origin.min_bars == min_bars as u64,
        "MSE-JEPA checkpoint resolution/min-bars origin does not match the loaded dataset"
    );
    ensure!(
        origin.split_bounds == dataset.split_bounds()
            && origin.split_bounds_pinned == split_bounds_pinned,
        "MSE-JEPA checkpoint split origin does not match the loaded dataset"
    );
    ensure!(
        origin.corpus_fingerprint == dataset.corpus_fingerprint(),
        "MSE-JEPA checkpoint corpus fingerprint does not match the loaded dataset"
    );
    ensure!(
        origin.supports_scoring_sha256 == supports.scoring_sha256()?,
        "MSE-JEPA checkpoint BarSupports scoring identity does not match the loaded supports"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct ObjectiveWeights {
    lambda_flow: f64,
    lambda_sigreg: f64,
}

/// Fixed `(x0, tau)` interpolant draws for a validation pass; training draws fresh ones.
struct FlowNoise<'a> {
    x0: &'a Tensor,
    tau: &'a Tensor,
}

struct Objective {
    forward: MseJepaForward,
    /// Exact scalar differentiated by the core-training backward.
    total: Tensor,
    emission_ce: Tensor,
    emission_ce_dof: Tensor,
    emission_hard_nll: Tensor,
    flow: Tensor,
    /// `[4]` flow loss over `tau` in `[0, .25)`, `[.25, .5)`, `[.5, .75)`, `[.75, 1)`.
    flow_quartiles: Tensor,
    persistence_mse: Tensor,
    sigreg: Tensor,
    representation_std: Tensor,
    target_std: Tensor,
}

/// Validation-only diagnostics of `VALIDATION_SAMPLE_DRAWS` Heun draws per position.
struct SampleDiagnostics {
    spread: Tensor,
    distance: Tensor,
    calibration_ratio: Tensor,
    energy_score: Tensor,
    sample_std: Tensor,
    prediction_mse: Tensor,
    skill_vs_persistence: Tensor,
    skill_vs_mean: Tensor,
}

const REPRESENTATION_STATS_EVERY: usize = 32;
const STEP_METRIC_COUNT: usize = 18;
const SAMPLE_METRIC_COUNT: usize = 8;
const VALIDATION_METRIC_COUNT: usize = STEP_METRIC_COUNT + SAMPLE_METRIC_COUNT;
const VALIDATION_SIGREG_SEED_DOMAIN: u64 = 0x4C45_574D_5641_4C31;

#[allow(clippy::too_many_arguments)]
fn objective(
    model: &MseJepaModel,
    supports: &BarSupports,
    bars: &Tensor,
    next_dof: &Tensor,
    sigreg_bars: &Tensor,
    weights: ObjectiveWeights,
    emission_gradient_mode: EmissionGradientMode,
    train: bool,
    sample_representation_stats: bool,
    sigreg_directions: Option<&Tensor>,
    flow_noise: Option<FlowNoise<'_>>,
) -> Objective {
    let forward = model.forward(bars, train);
    assert_eq!(
        next_dof.size(),
        [bars.size()[0], MAX_CONTEXT_BARS, BAR_DOF as i64],
        "emission targets must cover every predicted position"
    );

    let next_bins = supports.bin_ids(next_dof).unsqueeze(1);
    let logits = if bars.device().is_cuda() {
        pin_bfloat16_autocast();
        autocast(true, || {
            model.training_emission_logits(&forward.beliefs, &next_bins, emission_gradient_mode)
        })
    } else {
        model.training_emission_logits(&forward.beliefs, &next_bins, emission_gradient_mode)
    };
    let smoothed_targets = supports.targets(next_dof, BarScoring::Smoothed);
    let logits = logits.squeeze_dim(1);
    let (emission_ce, emission_ce_dof) = bar_nll_from_logits(&logits, &smoothed_targets);
    let emission_hard_nll = tch::no_grad(|| {
        let hard_targets = supports.targets(next_dof, BarScoring::Hard);
        bar_nll_from_logits(&logits.detach(), &hard_targets).0
    });

    let (flow, flow_quartiles) =
        flow_matching_terms(model, &forward.beliefs, &forward.targets, flow_noise);
    let persistence_mse = tch::no_grad(|| {
        mean_square_error(
            &forward.all_tokens.detach().narrow(2, 0, MAX_CONTEXT_BARS),
            &forward.targets.detach(),
        )
    });

    let sigreg_tokens = model.encode(sigreg_bars, train);
    let views = sample_temporal_views(&sigreg_tokens, false);
    assert_eq!(
        views.size()[1],
        SIGREG_BATCH_SIZE as i64,
        "SIGReg independent sample count drifted"
    );
    let sigreg = match sigreg_directions {
        Some(directions) => sigreg_loss_with_directions(&views, directions),
        None => sigreg_loss(&views),
    };
    let total = &emission_ce + &flow * weights.lambda_flow + &sigreg * weights.lambda_sigreg;
    let (representation_std, target_std) = if sample_representation_stats {
        (
            feature_std(&forward.all_tokens.detach()),
            forward.targets.detach().std(false),
        )
    } else {
        let nan = Tensor::full([], f64::NAN, (Kind::Float, bars.device()));
        (nan.shallow_clone(), nan)
    };
    Objective {
        forward,
        total,
        emission_ce,
        emission_ce_dof,
        emission_hard_nll,
        flow,
        flow_quartiles,
        persistence_mse,
        sigreg,
        representation_std,
        target_std,
    }
}

/// Conditional flow matching on the linear path `x_tau = (1 - tau) x0 + tau z1` with
/// `tau ~ LogitNormal(0, 1)`, `x0 ~ N(0, I)`, target velocity `u = z1 - x0`, one draw per
/// position. `z1` stays attached: the trunk and encoder learn from the transition too.
/// Returns `(mean per-dim MSE, [4] masked means over tau quartiles)`.
fn flow_matching_terms(
    model: &MseJepaModel,
    beliefs: &Tensor,
    targets: &Tensor,
    noise: Option<FlowNoise<'_>>,
) -> (Tensor, Tensor) {
    let (x0, tau) = match noise {
        Some(noise) => {
            assert_eq!(
                noise.x0.size(),
                targets.size(),
                "flow x0 panel shape drifted"
            );
            (noise.x0.shallow_clone(), noise.tau.shallow_clone())
        }
        None => tch::no_grad(|| {
            let mut tau_shape = targets.size();
            *tau_shape.last_mut().expect("targets carry a latent dim") = 1;
            (
                Tensor::randn_like(targets),
                Tensor::randn(tau_shape, (Kind::Float, targets.device())).sigmoid(),
            )
        }),
    };
    let target_velocity = targets - &x0;
    let x_tau = &x0 + &tau * &target_velocity;
    let velocity = if beliefs.device().is_cuda() {
        pin_bfloat16_autocast();
        autocast(true, || model.velocity(&x_tau, &tau, beliefs))
    } else {
        model.velocity(&x_tau, &tau, beliefs)
    };
    let per_position =
        (velocity - target_velocity)
            .square()
            .mean_dim([-1i64].as_slice(), false, Kind::Float);
    let flow = per_position.mean(Kind::Float);
    let quartiles = tch::no_grad(|| {
        let per_position = per_position.detach();
        let tau = tau.squeeze_dim(-1);
        // An empty bucket reads NaN (0/0), never a spurious zero.
        let masked = |lower: f64, upper: f64| {
            let mask = tau
                .ge(lower)
                .logical_and(&tau.lt(upper))
                .to_kind(Kind::Float);
            (&per_position * &mask).sum(Kind::Float) / mask.sum(Kind::Float)
        };
        Tensor::stack(
            &[
                masked(0.0, 0.25),
                masked(0.25, 0.5),
                masked(0.5, 0.75),
                masked(0.75, 1.0),
            ],
            0,
        )
    });
    (flow, quartiles)
}

/// Heun-sample diagnostics against the realized next tokens, all under `no_grad`.
fn sample_diagnostics(
    model: &MseJepaModel,
    forward: &MseJepaForward,
    persistence_mse: &Tensor,
    sample_noise: &Tensor,
    flow_steps: i64,
) -> SampleDiagnostics {
    tch::no_grad(|| {
        let samples = model.sample_next_latents_from(&forward.beliefs, sample_noise, flow_steps);
        let targets = forward.targets.detach();
        let (energy_score, distance, spread) = fan_energy_terms(&samples, &targets);
        // Nominal 1.0: against a REALIZED draw z of the same law, E||z_hat - z_hat'|| equals
        // E||z_hat - z|| exactly under calibration; the sqrt(2) folklore factor applies only
        // against the conditional MEAN and would pin this ratio at ~0.707 when healthy.
        let calibration_ratio = &spread / distance.clamp_min(1e-6);
        let mean_prediction = samples.mean_dim([0i64].as_slice(), false, Kind::Float);
        let prediction_mse = mean_square_error(&mean_prediction, &targets);
        let mean_baseline_mse = {
            let flat = targets.reshape([-1, LATENT_DIM]);
            let mean = flat.mean_dim([0i64].as_slice(), true, Kind::Float);
            (flat - mean).square().mean(Kind::Float)
        };
        let skill_vs_persistence = latent_skill(&prediction_mse, persistence_mse);
        let skill_vs_mean = latent_skill(&prediction_mse, &mean_baseline_mse);
        SampleDiagnostics {
            spread,
            distance,
            calibration_ratio,
            energy_score,
            sample_std: samples.std(false),
            prediction_mse,
            skill_vs_persistence,
            skill_vs_mean,
        }
    })
}

/// Energy score over `[m, ..lead.., LATENT_DIM]` draws against `[..lead.., LATENT_DIM]`
/// targets: `ES = E||z_hat - z|| - 0.5 * E||z_hat - z_hat'||`, a strictly proper scoring
/// rule kept as a sampler diagnostic (continuity with the v1/v2 objective). Returns
/// `(es, E||z_hat - z||, E||z_hat - z_hat'||)`, each a scalar mean over positions.
fn fan_energy_terms(samples: &Tensor, targets: &Tensor) -> (Tensor, Tensor, Tensor) {
    let m = samples.size()[0];
    assert!(m > 1, "the energy score spread term needs at least 2 draws");
    let flat_samples = samples
        .reshape([m, -1, LATENT_DIM])
        .transpose(0, 1)
        .contiguous();
    let flat_targets = targets.reshape([-1, 1, LATENT_DIM]);
    let latent_distance = Tensor::cdist(&flat_samples, &flat_targets, 2.0, None).mean(Kind::Float);
    let pairwise = Tensor::cdist(&flat_samples, &flat_samples, 2.0, None);
    let sample_spread = pairwise.sum_dim_intlist([-1i64, -2].as_slice(), false, Kind::Float)
        / ((m * (m - 1)) as f64);
    let sample_spread = sample_spread.mean(Kind::Float);
    let energy_score = &latent_distance - &sample_spread * 0.5;
    (energy_score, latent_distance, sample_spread)
}

fn mean_square_error(predictions: &Tensor, targets: &Tensor) -> Tensor {
    (predictions - targets).square().mean(Kind::Float)
}

fn latent_skill(prediction_mse: &Tensor, baseline_mse: &Tensor) -> Tensor {
    Tensor::full_like(baseline_mse, f64::NAN).where_self(
        &baseline_mse.eq(0.0),
        &(Tensor::ones_like(baseline_mse) - prediction_mse / baseline_mse),
    )
}

fn feature_std(tokens: &Tensor) -> Tensor {
    let flat = tokens.reshape([-1, LATENT_DIM]).to_kind(Kind::Float);
    let mean = flat.mean_dim([0i64].as_slice(), true, Kind::Float);
    (&flat - mean)
        .square()
        .mean_dim([0i64].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
}

fn global_grad_norm_tensor(named: &[(String, Tensor)], device: Device) -> Tensor {
    tch::no_grad(|| {
        let norms = named
            .iter()
            .map(|(_, parameter)| parameter.grad())
            .filter(|gradient| gradient.defined())
            .map(|gradient| gradient.to_kind(Kind::Float).norm())
            .collect::<Vec<_>>();
        if norms.is_empty() {
            Tensor::zeros([], (Kind::Float, device))
        } else {
            Tensor::stack(&norms, 0).norm()
        }
    })
}

fn flat_f32(tensor: &Tensor) -> Tensor {
    tensor.detach().to_kind(Kind::Float).reshape([-1])
}

fn pack_step_metrics(losses: &Objective, grad_norm: &Tensor) -> Tensor {
    Tensor::cat(
        &[
            flat_f32(&losses.total),
            flat_f32(&losses.emission_ce),
            flat_f32(&losses.emission_hard_nll),
            flat_f32(&losses.flow),
            flat_f32(&losses.flow_quartiles),
            flat_f32(&losses.persistence_mse),
            flat_f32(&losses.sigreg),
            flat_f32(&losses.representation_std),
            flat_f32(&losses.target_std),
            flat_f32(grad_norm),
            flat_f32(&losses.emission_ce_dof),
        ],
        0,
    )
}

fn pack_sample_diagnostics(diagnostics: &SampleDiagnostics) -> Tensor {
    Tensor::cat(
        &[
            flat_f32(&diagnostics.spread),
            flat_f32(&diagnostics.distance),
            flat_f32(&diagnostics.calibration_ratio),
            flat_f32(&diagnostics.energy_score),
            flat_f32(&diagnostics.sample_std),
            flat_f32(&diagnostics.prediction_mse),
            flat_f32(&diagnostics.skill_vs_persistence),
            flat_f32(&diagnostics.skill_vs_mean),
        ],
        0,
    )
}

fn read_packed_metrics<const N: usize>(packed: &Tensor) -> [f64; N] {
    let values = Vec::<f32>::try_from(packed.reshape([-1]))
        .expect("packed MSE-JEPA metrics are convertible");
    let values: [f32; N] = values.try_into().unwrap_or_else(|values: Vec<f32>| {
        panic!(
            "packed MSE-JEPA metrics carry {} entries, expected {N}",
            values.len()
        )
    });
    values.map(f64::from)
}

fn read_packed_step_metrics(packed: &Tensor) -> [f64; STEP_METRIC_COUNT] {
    read_packed_metrics(packed)
}

fn step_metrics(
    values: [f64; STEP_METRIC_COUNT],
    optimizer: OptimizerPoint,
    training_pass: f64,
    completed_step_seconds: f64,
    prediction_batch_size: usize,
    sigreg_batch_size: usize,
) -> StepMetrics {
    StepMetrics {
        total_loss: values[0],
        emission_ce: values[1],
        emission_hard_nll: values[2],
        flow: values[3],
        flow_quartiles: [values[4], values[5], values[6], values[7]],
        persistence_mse: values[8],
        sigreg: values[9],
        representation_std: values[10],
        target_std: values[11],
        grad_norm: values[12],
        emission_ce_dof: [values[13], values[14], values[15], values[16], values[17]],
        training_pass,
        learning_rate_multiplier: optimizer.learning_rate_multiplier,
        muon_learning_rate: optimizer.muon_learning_rate,
        adamw_learning_rate: optimizer.adamw_learning_rate,
        muon_momentum: optimizer.muon_momentum,
        completed_step_seconds,
        windows_per_second: prediction_batch_size as f64 / completed_step_seconds,
        sigreg_samples_per_second: sigreg_batch_size as f64 / completed_step_seconds,
        predicted_latent_positions_per_second: prediction_batch_size as f64
            * MAX_CONTEXT_BARS as f64
            / completed_step_seconds,
    }
}

fn fixed_validation_sigreg_directions(device: Device, seed: u64) -> Tensor {
    let mut rng = ChaCha12Rng::seed_from_u64(seed ^ VALIDATION_SIGREG_SEED_DOMAIN);
    let count = (LATENT_DIM * SIGREG_PROJECTIONS) as usize;
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let uniform = (1.0 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
        let radius = (-2.0 * uniform.ln()).sqrt();
        let angle = std::f64::consts::TAU * rng.random::<f64>();
        values.push((radius * angle.cos()) as f32);
        if values.len() < count {
            values.push((radius * angle.sin()) as f32);
        }
    }
    let directions = Tensor::from_slice(&values)
        .view([LATENT_DIM, SIGREG_PROJECTIONS])
        .to_device(device);
    &directions / directions.norm_scalaropt_dim(2, [0i64].as_slice(), true)
}

fn temporal_offsets(train: bool) -> Result<Vec<i64>> {
    Ok(Vec::<i64>::try_from(temporal_view_indices(
        MAX_CONTEXT_BARS + 1,
        SIGREG_MAX_VIEWS.min(MAX_CONTEXT_BARS + 1),
        train,
        Device::Cpu,
    ))?)
}

#[allow(clippy::too_many_arguments)]
fn validation_metrics(
    model: &MseJepaModel,
    supports: &BarSupports,
    dataset: &MseJepaDataset,
    refs: &[crate::torch::dataset::WindowRef],
    next_dof: &Tensor,
    sigreg_bars: &Tensor,
    args: &MseJepaArgs,
    sigreg_directions: &Tensor,
    flow_panel: &ValidationFlowPanel,
    device: Device,
) -> Result<ValidationMetrics> {
    enable_exact_fp32_matmul()?;
    let metrics = tch::no_grad(|| {
        let bars = dataset.features(refs, device);
        let x0 = flow_panel.x0.to_device(device);
        let tau = flow_panel.tau.to_device(device);
        let losses = objective(
            model,
            supports,
            &bars,
            next_dof,
            sigreg_bars,
            ObjectiveWeights {
                lambda_flow: args.lambda_flow,
                lambda_sigreg: args.lambda_sigreg,
            },
            args.emission_gradient_mode,
            false,
            false,
            Some(sigreg_directions),
            Some(FlowNoise { x0: &x0, tau: &tau }),
        );
        drop((x0, tau));
        let diagnostics = sample_diagnostics(
            model,
            &losses.forward,
            &losses.persistence_mse,
            &flow_panel.sample_noise.to_device(device),
            args.flow_steps as i64,
        );
        let packed = Tensor::cat(
            &[
                pack_step_metrics(&losses, &Tensor::full([], f64::NAN, (Kind::Float, device))),
                pack_sample_diagnostics(&diagnostics),
            ],
            0,
        );
        let values = read_packed_metrics::<VALIDATION_METRIC_COUNT>(&packed);
        ValidationMetrics {
            total_loss: values[0],
            emission_ce: values[1],
            emission_hard_nll: values[2],
            flow: values[3],
            flow_quartiles: [values[4], values[5], values[6], values[7]],
            persistence_mse: values[8],
            sigreg: values[9],
            emission_ce_dof: [values[13], values[14], values[15], values[16], values[17]],
            sample_spread: values[18],
            sample_distance: values[19],
            calibration_ratio: values[20],
            energy_score: values[21],
            sample_std: values[22],
            prediction_mse: values[23],
            skill_vs_persistence: values[24],
            skill_vs_mean: values[25],
        }
    });
    enable_tf32_matmul()?;
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::{
        fan_energy_terms, fixed_validation_sigreg_directions, flow_matching_terms, latent_skill,
        validate_args, EmissionGradientMode, FlowNoise, MseJepaArgs, ValidationFlowPanel,
        VALIDATION_SAMPLE_DRAWS,
    };
    use crate::torch::lejepa::model::LATENT_DIM;
    use crate::torch::lejepa::sigreg::DEFAULT_SIGREG_LAMBDA;
    use crate::torch::lejepa::MseJepaModel;
    use crate::torch::test_rng;
    use tch::{nn, Device, Kind, Tensor};

    #[test]
    fn fan_energy_score_rewards_spread_around_the_target() {
        let targets = Tensor::zeros([16, LATENT_DIM], (Kind::Float, Device::Cpu));
        let collapsed = Tensor::ones([4, 16, LATENT_DIM], (Kind::Float, Device::Cpu));
        let (collapsed_es, _, collapsed_spread) = fan_energy_terms(&collapsed, &targets);
        assert!(collapsed_spread.double_value(&[]).abs() < 1e-6);
        tch::manual_seed(17);
        let dispersed = Tensor::randn([4, 16, LATENT_DIM], (Kind::Float, Device::Cpu));
        let (_, _, dispersed_spread) = fan_energy_terms(&dispersed, &targets);
        assert!(dispersed_spread.double_value(&[]) > 0.0);
        let offset = collapsed_es.double_value(&[]);
        assert!(
            (offset - (LATENT_DIM as f64).sqrt()).abs() < 1e-3,
            "a collapsed unit-offset fan must score exactly its distance: {offset}"
        );
    }

    #[test]
    fn validation_flow_panel_draws_logit_normal_tau_inside_the_open_unit_interval() {
        let _torch_rng_guard = test_rng::shared();
        tch::manual_seed(17);
        let panel = ValidationFlowPanel::draw(1);
        assert_eq!(panel.tau.size(), vec![1, 1, super::MAX_CONTEXT_BARS, 1]);
        assert_eq!(
            panel.x0.size(),
            vec![1, 1, super::MAX_CONTEXT_BARS, LATENT_DIM]
        );
        assert_eq!(
            panel.sample_noise.size(),
            vec![
                VALIDATION_SAMPLE_DRAWS,
                1,
                1,
                super::MAX_CONTEXT_BARS,
                LATENT_DIM
            ]
        );
        assert!(panel.tau.min().double_value(&[]) > 0.0);
        assert!(panel.tau.max().double_value(&[]) < 1.0);
        let median = panel.tau.median().double_value(&[]);
        assert!(
            (median - 0.5).abs() < 0.05,
            "logit-normal median drifted: {median}"
        );
        let mid_mass = panel
            .tau
            .gt(0.25)
            .logical_and(&panel.tau.lt(0.75))
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        assert!(
            mid_mass > 0.65,
            "logit-normal must emphasise the middle of the path: {mid_mass}"
        );
    }

    #[test]
    fn flow_matching_terms_reduce_to_the_target_velocity_energy_at_init_and_stay_attached() {
        let _torch_rng_guard = test_rng::shared();
        tch::manual_seed(17);
        let var_store = nn::VarStore::new(Device::Cpu);
        let model = MseJepaModel::new(&var_store.root());
        let beliefs = Tensor::randn([2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu));
        let targets = Tensor::randn([2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu))
            .set_requires_grad(true);
        let x0 = Tensor::randn([2, 1, 4, LATENT_DIM], (Kind::Float, Device::Cpu));
        let tau =
            Tensor::from_slice(&[0.1f32, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8]).reshape([2, 1, 4, 1]);
        let (flow, quartiles) = flow_matching_terms(
            &model,
            &beliefs,
            &targets,
            Some(FlowNoise { x0: &x0, tau: &tau }),
        );
        let expected = (&targets - &x0).square().mean(Kind::Float);
        assert!((flow.double_value(&[]) - expected.double_value(&[])).abs() < 1e-5);
        assert_eq!(quartiles.size(), vec![4]);
        assert!(quartiles.isfinite().all().int64_value(&[]) != 0);
        let quartile_mean = quartiles.mean(Kind::Float).double_value(&[]);
        assert!((quartile_mean - flow.double_value(&[])).abs() < 1e-4);
        flow.backward();
        assert!(targets.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);

        let (fresh, _) = flow_matching_terms(&model, &beliefs, &targets.detach(), None);
        assert!(fresh.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn validation_sigreg_directions_are_seeded_and_unit_norm() {
        let first = fixed_validation_sigreg_directions(tch::Device::Cpu, 17);
        let second = fixed_validation_sigreg_directions(tch::Device::Cpu, 17);
        assert!(first.equal(&second));
        let max_norm_error = (first.norm_scalaropt_dim(2, [0i64].as_slice(), false) - 1.0)
            .abs()
            .max()
            .double_value(&[]);
        assert!(max_norm_error < 1e-5);
    }

    #[test]
    fn latent_skill_is_undefined_when_the_baseline_is_perfect() {
        let prediction_mse = Tensor::from(0.0f32);
        let baseline_mse = Tensor::from(0.0f32);
        let undefined = latent_skill(&prediction_mse, &baseline_mse).double_value(&[]);
        assert!(undefined.is_nan());

        let prediction_mse = Tensor::from(0.5f32);
        let baseline_mse = Tensor::from(1.0f32);
        let skill = latent_skill(&prediction_mse, &baseline_mse).double_value(&[]);
        assert!((skill - 0.5).abs() < 1e-7);
    }

    #[test]
    fn zero_steps_cannot_publish_an_untrained_bundle() {
        let args = MseJepaArgs {
            run: None,
            epochs: 1,
            steps: Some(0),
            batch_size: 8,
            emission_gradient_mode: EmissionGradientMode::Attached,
            seed: 0x5EED,
            data_dir: "unused".to_owned(),
            resolution_secs: 300,
            min_bars: 60_000,
            validation_windows: 8,
            validate_every: 1_000,
            checkpoint_every: 2_048,
            lambda_flow: super::DEFAULT_LAMBDA_FLOW,
            lambda_sigreg: DEFAULT_SIGREG_LAMBDA,
            flow_steps: super::DEFAULT_FLOW_STEPS,
            split_bounds: None,
            derive_split_bounds: false,
        };
        let error = validate_args(&args).unwrap_err().to_string();
        assert!(error.contains("--steps must be positive"));
        let mut no_flow_steps = args.clone();
        no_flow_steps.steps = None;
        no_flow_steps.flow_steps = 0;
        let error = validate_args(&no_flow_steps).unwrap_err().to_string();
        assert!(error.contains("--flow-steps must be positive"));
        let mut negative_flow = args;
        negative_flow.steps = None;
        negative_flow.lambda_flow = -1.0;
        let error = validate_args(&negative_flow).unwrap_err().to_string();
        assert!(error.contains("--lambda-flow must be finite and non-negative"));
    }
}
