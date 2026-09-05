use std::path::Path;

use anyhow::{ensure, Context, Result};
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{
    bar_nll_terms, BarScoring, BarSupports, DofScaling, BAR_DOF, BAR_VOLUME_EMA_SPAN, DOF_R, DOF_S,
    DOF_U, DOF_V, DOF_W,
};
use crate::torch::cuda::cfg::{configure_cuda, disable_autograd_multithreading};
use crate::torch::dataset::{time_ids_without_market, Split};
use crate::torch::lejepa::checkpoint::{
    load_authenticated, AuthenticatedCheckpoint, CheckpointKind, CoreTrainingOrigin,
    CANONICAL_READOUT_FIT_BATCH_SIZE, CANONICAL_READOUT_FIT_SEED, CANONICAL_READOUT_FIT_STEPS,
    CANONICAL_READOUT_OPTIMIZER_RECIPE_ID, CANONICAL_READOUT_TOKEN_ROWS,
    CANONICAL_READOUT_VALIDATION_WINDOWS,
};
use crate::torch::lejepa::dataset::{MseJepaDataset, MseJepaRolloutPanel};
use crate::torch::lejepa::model::{MseJepaKvCache, LATENT_DIM, MAX_CONTEXT_BARS};
use crate::torch::lejepa::MseJepaModel;
use crate::torch::train::pretrain_reports::{
    write_candle_windows_labeled, CandleFanLabels, MIN_FAN_SAMPLES,
};
use crate::torch::world_model::{
    world_model_metadata_path, BarWorldModel, RolloutMode, BAR_MAX_CONTEXT,
};

use super::reports::{
    inherit_posthoc_readout_report, inherit_source_reports, preflight_posthoc_readout_report,
    preflight_source_reports, write_rollout_reports, RolloutCoverage, RolloutReport,
    RolloutStepCoverage,
};

pub const DEFAULT_ROLLOUT_VALIDATION_WINDOWS: usize = 64;
pub const DEFAULT_ROLLOUT_WINDOW_CHUNK: usize = 2;
pub const DEFAULT_ROLLOUT_SAMPLES: usize = 32;
pub const DEFAULT_ROLLOUT_HORIZONS: [usize; 6] = [1, 4, 16, 39, 78, 100];
const MAX_CANDLE_WINDOWS: usize = 8;
const ROLLOUT_SEED_DOMAIN: u64 = 0x4442_574d_524f_4c4c;
const BASELINE_SEED_DOMAIN: u64 = 0x4c4c_4d42_4153_454c;
const COVERAGE_QUANTILES: [f64; 4] = [0.05, 0.25, 0.75, 0.95];

#[derive(Clone, Debug)]
pub struct EvaluateMseJepaRolloutArgs {
    pub weights: String,
    /// Completed LLM-style categorical `BarWorldModel` checkpoint scored on the identical
    /// panel with identical fan-marginalized conventions: the design's evaluation gate.
    pub baseline_weights: Option<String>,
    pub run: String,
    pub data_dir: String,
    pub resolution_secs: u32,
    pub min_bars: usize,
    pub validation_windows: usize,
    pub window_chunk: usize,
    pub samples: usize,
    pub horizons: Vec<usize>,
    /// Heun steps per latent-mode transition draw.
    pub flow_steps: usize,
    pub seed: u64,
    pub split_bounds: Option<(i64, i64)>,
    pub derive_split_bounds: bool,
}

/// Score both ancestral rollout modes of a fitted DBWM endpoint on a pinned validation panel:
/// bar mode (sample the posthoc-fitted emission, convert to transition7, re-encode, append) and
/// latent mode (Heun-sample the frozen core transition, append the latent directly). Per horizon
/// this reports the fan-marginalized predictive NLL `-log((1/S) sum_s p(bar | h_s))` and
/// central-interval coverage of the return DOF.
pub fn evaluate_mse_jepa_rollout(args: EvaluateMseJepaRolloutArgs) -> Result<()> {
    validate_args(&args)?;
    RunDir::ensure_creatable(RUNS_PATH, &args.run)
        .context("validating the rollout evaluation output run")?;
    super::super::pretrain::configure_threads();
    configure_cuda();
    let device = Device::cuda_if_available();
    ensure!(
        device.is_cuda(),
        "evaluate-mse-jepa-rollout requires CUDA and strict FA4; there is no CPU fallback"
    );
    let _autograd_guard = disable_autograd_multithreading();

    let split_bounds = if args.derive_split_bounds {
        None
    } else {
        Some(
            args.split_bounds
                .unwrap_or(crate::data::ingest::PINNED_SPLIT_BOUNDS),
        )
    };
    let dataset = MseJepaDataset::load(
        Path::new(&args.data_dir),
        args.resolution_secs,
        args.min_bars,
        split_bounds,
        args.seed,
    )?;
    let supports = super::load_authenticated_supports(&dataset)?;

    let mut representation_vs = tch::nn::VarStore::new(device);
    let model = MseJepaModel::new(&representation_vs.root());
    let authenticated = load_authenticated(&mut representation_vs, &args.weights)?;
    require_fitted_endpoint(&authenticated)?;
    super::authenticate_core_origin(
        &authenticated.provenance,
        &dataset,
        &supports,
        args.resolution_secs,
        args.min_bars,
        !args.derive_split_bounds,
    )
    .context("authenticating fitted endpoint data/support origin")?;
    let completed_steps = usize::try_from(authenticated.provenance.completed_steps)
        .context("MSE-JEPA checkpoint step count exceeds this platform")?;
    let source_run = RunDir::from_weights_path_in(&authenticated.path, RUNS_PATH)
        .context("resolving the authenticated MSE-JEPA source run")?;
    let source_reports = preflight_source_reports(&source_run.gens, completed_steps)
        .context("preflighting the required source MSE-JEPA reports")?;
    let posthoc_report =
        preflight_posthoc_readout_report(&source_run.gens, &authenticated.provenance)
            .context("preflighting the fitted readout report")?;
    representation_vs.freeze();
    assert_representation_frozen(&representation_vs)?;

    let max_horizon = *args.horizons.last().expect("validated non-empty horizons");
    let panel = dataset.rollout_panel(Split::Val, args.validation_windows, max_horizon, device)?;

    let fans = tch::no_grad(|| {
        sample_rollout_fans(
            &model,
            &supports,
            &panel,
            args.window_chunk,
            args.samples,
            args.flow_steps as i64,
            max_horizon,
            args.seed,
        )
    });
    assert_representation_frozen(&representation_vs)?;

    let baseline_nll = args
        .baseline_weights
        .as_deref()
        .map(|weights| {
            let baseline = load_baseline_world_model(
                weights,
                device,
                &supports,
                &authenticated.provenance.core_origin,
            )?;
            let per_trajectory = tch::no_grad(|| {
                baseline_fan_nll(
                    &baseline,
                    &panel,
                    args.window_chunk,
                    args.samples,
                    max_horizon,
                    args.seed,
                )
            });
            Ok::<_, anyhow::Error>(predictive_nll(&per_trajectory))
        })
        .transpose()?;

    let marginal_nll = supports.marginal_nll_bar(BarScoring::Hard);
    let real_r = panel
        .future_dof
        .select(-1, DOF_R as i64)
        .transpose(0, 1)
        .contiguous();
    let source_kind = match &authenticated.provenance.weight_readout {
        crate::torch::lejepa::checkpoint::WeightReadout::Raw => "raw",
        crate::torch::lejepa::checkpoint::WeightReadout::TailEma { .. } => "tail_ema",
    };
    let gradient_mode = authenticated.provenance.emission_gradient_mode.to_string();
    let origin = &authenticated.provenance.core_origin;
    let report = RolloutReport {
        bar_nll: &predictive_nll(&fans.bar_nll),
        latent_nll: &predictive_nll(&fans.latent_nll),
        baseline_nll: baseline_nll.as_deref(),
        marginal_nll,
        bar_coverage: coverage_profile(&fans.bar_step_r, &real_r),
        latent_coverage: step_coverage_profile(&fans.latent_step_r, &real_r),
        selected_horizons: &args.horizons,
        samples: args.samples,
        validation_windows: args.validation_windows,
        window_chunk: args.window_chunk,
        flow_steps: args.flow_steps,
        seed: args.seed,
        checkpoint_sha256: &authenticated.checkpoint_sha256,
        checkpoint_lineage_sha256: &authenticated.lineage_sha256,
        emission_gradient_mode: &gradient_mode,
        source_kind,
        corpus_fingerprint: &origin.corpus_fingerprint,
        split_bounds: origin.split_bounds,
        split_bounds_pinned: origin.split_bounds_pinned,
        supports_scoring_sha256: &origin.supports_scoring_sha256,
        core_train_seed: origin.train_seed,
        core_batch_size: origin.batch_size,
    };

    let candle_windows = args.validation_windows.min(MAX_CANDLE_WINDOWS);
    let drawn = fans
        .bar_dof
        .narrow(0, 0, candle_windows as i64)
        .to_device(Device::Cpu);

    let requested_root = Path::new(RUNS_PATH).join(&args.run);
    let previous_latest = RunDir::latest(RUNS_PATH)
        .ok()
        .filter(|previous| previous.root != requested_root);
    let run = RunDir::create_fresh(RUNS_PATH, Some(args.run.as_str()))?;
    let publish = (|| -> Result<()> {
        write_rollout_reports(&run.gens, &report)?;
        inherit_source_reports(&run.gens, &source_reports)
            .context("inheriting source MSE-JEPA reports into the rollout run")?;
        inherit_posthoc_readout_report(&run.gens, &posthoc_report)
            .context("inheriting fitted readout report into the rollout run")?;
        write_candle_windows_labeled(
            &run.gens.join("0").join("candle_snapshots"),
            0,
            None,
            &drawn,
            &panel
                .future_dof
                .narrow(0, 0, candle_windows as i64)
                .to_device(Device::Cpu),
            CandleFanLabels {
                title_prefix: "MSE-JEPA DBWM Bar-Mode Ancestral Rollout Fan",
                coverage_reference: None,
            },
        )
        .context("writing MSE-JEPA DBWM bar-mode rollout candle fans")?;
        Ok(())
    })();
    if let Err(error) = publish {
        return cleanup_failed_publication(Path::new(RUNS_PATH), &run, previous_latest, error);
    }
    println!(
        "Saved MSE-JEPA DBWM rollout evaluation reports to {}",
        run.gens.display()
    );
    Ok(())
}
fn require_fitted_endpoint(authenticated: &AuthenticatedCheckpoint) -> Result<()> {
    ensure!(
        authenticated.kind == CheckpointKind::FittedReadout,
        "evaluate-mse-jepa-rollout accepts only posthoc-fitted mse_jepa*_fitted.ot endpoints"
    );
    ensure!(
        authenticated.provenance.completed_steps == authenticated.provenance.planned_steps,
        "evaluate-mse-jepa-rollout requires a fully completed core lineage"
    );
    let fit = authenticated
        .provenance
        .head_fit
        .as_ref()
        .context("fitted endpoint lacks posthoc readout provenance")?;
    ensure!(
        fit.steps == CANONICAL_READOUT_FIT_STEPS
            && fit.batch_size == CANONICAL_READOUT_FIT_BATCH_SIZE
            && fit.token_rows_per_step == CANONICAL_READOUT_TOKEN_ROWS
            && fit.validation_windows == CANONICAL_READOUT_VALIDATION_WINDOWS
            && fit.seed == CANONICAL_READOUT_FIT_SEED
            && fit.optimizer_recipe_id == CANONICAL_READOUT_OPTIMIZER_RECIPE_ID,
        "evaluate-mse-jepa-rollout requires the canonical 4096/8/4096/8 posthoc fit, dedicated seed, and optimizer recipe"
    );
    Ok(())
}

fn validate_args(args: &EvaluateMseJepaRolloutArgs) -> Result<()> {
    ensure!(!args.weights.is_empty(), "--weights is required");
    ensure!(!args.run.is_empty(), "--run is required");
    ensure!(
        args.resolution_secs == 300,
        "evaluate-mse-jepa-rollout requires the train-fitted raw 300s BarSupports"
    );
    ensure!(
        args.min_bars >= (MAX_CONTEXT_BARS + 2) as usize,
        "--min-bars must admit a full context and held-out target"
    );
    ensure!(
        args.validation_windows > 0,
        "--validation-windows must be positive"
    );
    ensure!(
        args.window_chunk > 0 && args.window_chunk <= args.validation_windows,
        "--window-chunk must be in 1..=--validation-windows"
    );
    ensure!(
        args.samples >= MIN_FAN_SAMPLES,
        "--samples must carry at least {MIN_FAN_SAMPLES} fan draws"
    );
    ensure!(
        !args.horizons.is_empty()
            && args.horizons.iter().all(|horizon| *horizon > 0)
            && args.horizons.windows(2).all(|pair| pair[0] < pair[1]),
        "--horizons must be positive, sorted, and unique"
    );
    ensure!(args.flow_steps > 0, "--flow-steps must be positive");
    ensure!(
        !(args.derive_split_bounds && args.split_bounds.is_some()),
        "--derive-split-bounds conflicts with --split-bounds"
    );
    Ok(())
}

fn assert_representation_frozen(var_store: &tch::nn::VarStore) -> Result<()> {
    for (name, variable) in var_store.variables() {
        ensure!(
            !variable.requires_grad(),
            "frozen representation parameter {name} still requires gradients"
        );
        ensure!(
            !variable.grad().defined(),
            "frozen representation parameter {name} acquired a gradient"
        );
    }
    Ok(())
}

/// Reset the process generators for exactly one logical validation window. Production rollout
/// deliberately evaluates one window at a time: CUDA's global generator does not expose
/// splittable streams, so changing a batch shape can otherwise change every later draw.
/// `--window-chunk` remains a compatibility iteration grouping only; it cannot affect streams.
fn seed_window_rng(domain_seed: u64, window: i64) {
    let mut z = domain_seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(window as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    tch::manual_seed(z as i64);
    tch::Cuda::manual_seed_all(z);
}

fn window_order(windows: i64, window_chunk: usize) -> impl Iterator<Item = i64> {
    (0..windows)
        .step_by(window_chunk)
        .flat_map(move |start| start..(start + window_chunk as i64).min(windows))
}

struct RolloutFans {
    /// `[horizons, windows, samples]` per-trajectory hard NLL of the realized bar.
    bar_nll: Tensor,
    latent_nll: Tensor,
    /// `[horizons, windows, samples]` sampled return DOF fans.
    bar_step_r: Tensor,
    latent_step_r: Tensor,
    /// `[windows, samples, horizons, BAR_DOF]` bar-mode sampled DOF paths.
    bar_dof: Tensor,
}

fn sample_rollout_fans(
    model: &MseJepaModel,
    supports: &BarSupports,
    panel: &MseJepaRolloutPanel,
    window_chunk: usize,
    samples: usize,
    flow_steps: i64,
    max_horizon: usize,
    seed: u64,
) -> RolloutFans {
    let windows = panel.contexts.size()[0];
    let mut bar_nll = Vec::new();
    let mut latent_nll = Vec::new();
    let mut bar_step_r = Vec::new();
    let mut latent_step_r = Vec::new();
    let mut bar_dof = Vec::new();
    for start in window_order(windows, window_chunk) {
        seed_window_rng(seed ^ ROLLOUT_SEED_DOMAIN, start);
        let chunk = rollout_chunk(
            model,
            supports,
            &panel.contexts.narrow(0, start, 1),
            &panel.current_dof.narrow(0, start, 1),
            &panel.future_dof.narrow(0, start, 1),
            samples as i64,
            flow_steps,
            max_horizon,
        );
        bar_nll.push(chunk.bar_nll);
        latent_nll.push(chunk.latent_nll);
        bar_step_r.push(chunk.bar_step_r);
        latent_step_r.push(chunk.latent_step_r);
        bar_dof.push(chunk.bar_dof);
    }
    RolloutFans {
        bar_nll: Tensor::cat(&bar_nll, 1),
        latent_nll: Tensor::cat(&latent_nll, 1),
        bar_step_r: Tensor::cat(&bar_step_r, 1),
        latent_step_r: Tensor::cat(&latent_step_r, 1),
        bar_dof: Tensor::cat(&bar_dof, 0),
    }
}

struct RolloutChunk {
    bar_nll: Tensor,
    latent_nll: Tensor,
    bar_step_r: Tensor,
    latent_step_r: Tensor,
    bar_dof: Tensor,
}

fn rollout_chunk(
    model: &MseJepaModel,
    supports: &BarSupports,
    contexts: &Tensor,
    current_dof: &Tensor,
    future_dof: &Tensor,
    samples: i64,
    flow_steps: i64,
    max_horizon: usize,
) -> RolloutChunk {
    let chunk_windows = contexts.size()[0];
    let rows = chunk_windows * samples;
    let tokens = model.post_projector_tokens(contexts).detach();
    let mut prefill_cache = MseJepaKvCache::new();
    let prefilled = model.prefill_cached(&tokens, &mut prefill_cache);
    let last_belief = prefilled
        .narrow(2, MAX_CONTEXT_BARS - 1, 1)
        .repeat_interleave_self_int(samples, 0, None)
        .detach();
    let mut bar_cache = prefill_cache.repeat_batch(samples);

    let score = |belief: &Tensor, horizon: usize| -> Tensor {
        let flat = belief.reshape([rows, LATENT_DIM]);
        let real = future_dof
            .narrow(1, horizon as i64, 1)
            .reshape([chunk_windows, BAR_DOF as i64])
            .repeat_interleave_self_int(samples, 0, None);
        let bins = supports.bin_ids(&real);
        let logits = model.emission_logits(&flat, &bins);
        let targets = supports.targets_from_class_ids(&bins, BarScoring::Hard);
        bar_nll_terms(&logits, &targets)
            .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
            .reshape([chunk_windows, samples])
    };

    let mut bar_belief = last_belief.shallow_clone();
    let mut previous_w = current_dof
        .select(1, MAX_CONTEXT_BARS - 1)
        .select(-1, DOF_W as i64)
        .repeat_interleave_self_int(samples, 0, None)
        .detach();
    let mut bar_nll = Vec::with_capacity(max_horizon);
    let mut bar_step_r = Vec::with_capacity(max_horizon);
    let mut bar_dof = Vec::with_capacity(max_horizon);
    for horizon in 0..max_horizon {
        bar_nll.push(score(&bar_belief, horizon));
        let flat = bar_belief.reshape([rows, LATENT_DIM]);
        let sampled = model.sample_bars(&flat, supports, 1.0).detach();
        bar_step_r.push(
            sampled
                .select(-1, DOF_R as i64)
                .reshape([chunk_windows, samples]),
        );
        bar_dof.push(sampled.reshape([chunk_windows, samples, BAR_DOF as i64]));
        if horizon + 1 < max_horizon {
            let features = emitted_dof_to_transition7(&sampled, &previous_w).view([rows, 1, 1, 7]);
            let token = model.post_projector_tokens(&features).detach();
            bar_belief = model.decode_cached(&token, &mut bar_cache);
        }
        previous_w = sampled.select(-1, DOF_W as i64);
    }
    // The latent-mode cache is built only after the bar-mode one is released: both are the
    // sample-expanded footprint, so materializing them together would double peak memory
    // for a cache the bar loop never touches.
    drop(bar_cache);
    let mut latent_cache = prefill_cache.repeat_batch(samples);
    drop(prefill_cache);

    let mut latent_belief = last_belief;
    let mut latent_nll = Vec::with_capacity(max_horizon);
    let mut latent_step_r = Vec::with_capacity(max_horizon);
    for horizon in 0..max_horizon {
        latent_nll.push(score(&latent_belief, horizon));
        let flat = latent_belief.reshape([rows, LATENT_DIM]);
        let sampled = model.sample_bars(&flat, supports, 1.0).detach();
        latent_step_r.push(
            sampled
                .select(-1, DOF_R as i64)
                .reshape([chunk_windows, samples]),
        );
        if horizon + 1 < max_horizon {
            let next_latent = model
                .sample_next_latents(&latent_belief, 1, flow_steps)
                .squeeze_dim(0)
                .detach();
            latent_belief = model.decode_cached(&next_latent, &mut latent_cache);
        }
    }

    RolloutChunk {
        bar_nll: Tensor::stack(&bar_nll, 0),
        latent_nll: Tensor::stack(&latent_nll, 0),
        bar_step_r: Tensor::stack(&bar_step_r, 0),
        latent_step_r: Tensor::stack(&latent_step_r, 0),
        bar_dof: Tensor::stack(&bar_dof, 2),
    }
}

/// Fan-marginalized predictive NLL per horizon: `-log((1/S) sum_s exp(-nll_s))`, then the
/// mean over windows — a proper predictive likelihood, unlike a point-conditioned NLL.
fn predictive_nll(per_trajectory_nll: &Tensor) -> Vec<f64> {
    let samples = per_trajectory_nll.size()[2];
    let predictive = -((-per_trajectory_nll.to_kind(Kind::Float))
        .logsumexp([-1i64].as_slice(), false)
        - (samples as f64).ln());
    Vec::<f64>::try_from(
        predictive
            .mean_dim([-1i64].as_slice(), false, Kind::Float)
            .to_kind(Kind::Double)
            .to_device(Device::Cpu)
            .reshape([-1]),
    )
    .expect("predictive NLL profile is convertible")
}

/// Central-interval coverage rates from `[horizons, windows, samples]` fans against
/// `[horizons, windows]` realized values: `(50% band, 90% band)` per horizon.
fn central_band_rates(fans: &Tensor, real: &Tensor) -> (Vec<f64>, Vec<f64>) {
    let quantiles = Tensor::from_slice(&COVERAGE_QUANTILES)
        .to_kind(Kind::Float)
        .to_device(fans.device());
    let bands = fans
        .to_kind(Kind::Float)
        .quantile(&quantiles, -1, false, "linear");
    let rate = |lower: i64, upper: i64| -> Vec<f64> {
        let lower = bands.select(0, lower);
        let upper = bands.select(0, upper);
        let covered = real
            .ge_tensor(&lower)
            .logical_and(&real.le_tensor(&upper))
            .to_kind(Kind::Float)
            .mean_dim([-1i64].as_slice(), false, Kind::Float);
        Vec::<f64>::try_from(
            covered
                .to_kind(Kind::Double)
                .to_device(Device::Cpu)
                .reshape([-1]),
        )
        .expect("coverage profile is convertible")
    };
    (rate(1, 2), rate(0, 3))
}

/// Per-step and cumulative coverage for the genuine ancestral bar path, whose fan defines
/// a joint law over the whole trajectory.
fn coverage_profile(step_fans: &Tensor, real_step_r: &Tensor) -> RolloutCoverage {
    let real = real_step_r.to_kind(Kind::Float);
    let (step_50, step_90) = central_band_rates(step_fans, &real);
    let (cumulative_50, cumulative_90) = central_band_rates(
        &step_fans.to_kind(Kind::Float).cumsum(0, Kind::Float),
        &real.cumsum(0, Kind::Float),
    );
    RolloutCoverage {
        step_50,
        step_90,
        cumulative_50,
        cumulative_90,
    }
}

/// Per-step coverage only: latent-mode bar draws are per-horizon marginals whose
/// inter-step dependence the model never defines, so no cumulative band exists for them.
fn step_coverage_profile(step_fans: &Tensor, real_step_r: &Tensor) -> RolloutStepCoverage {
    let real = real_step_r.to_kind(Kind::Float);
    let (step_50, step_90) = central_band_rates(step_fans, &real);
    RolloutStepCoverage { step_50, step_90 }
}

fn load_baseline_world_model(
    weights: &str,
    device: Device,
    expected_supports: &BarSupports,
    expected_origin: &CoreTrainingOrigin,
) -> Result<BarWorldModel> {
    let checkpoint = Path::new(weights);
    let metadata = world_model_metadata_path(checkpoint);
    let baseline = BarWorldModel::load(checkpoint, &metadata, device)
        .context("loading the LLM baseline BarWorldModel")?;
    let training = baseline
        .metadata()
        .training()
        .context("the LLM baseline lacks authenticated training/scoring provenance")?;
    ensure!(
        training.dof_scaling == DofScaling::Raw && training.scoring == BarScoring::Hard.to_string(),
        "the LLM baseline scoring semantics differ from the MSE-JEPA hard/raw rollout contract"
    );
    require_baseline_origin(
        &training.corpus_fingerprint,
        training.split_bounds,
        training.split_bounds_pinned,
        training.supports_frozen,
        training.supports_corpus_fingerprint.as_deref(),
        expected_origin,
    )?;
    let baseline_supports = baseline
        .supports_for(300)
        .context("the LLM baseline carries no 300s supports and cannot score the shared panel")?;
    ensure!(
        baseline_supports.scoring_sha256()? == expected_supports.scoring_sha256()?,
        "the LLM baseline BarSupports scoring geometry/content differs from MSE-JEPA"
    );
    ensure!(
        baseline.all_parameters_frozen(),
        "the LLM baseline must load fully frozen"
    );
    Ok(baseline)
}
fn require_baseline_origin(
    corpus_fingerprint: &str,
    split_bounds: (i64, i64),
    split_bounds_pinned: bool,
    supports_frozen: bool,
    supports_corpus_fingerprint: Option<&str>,
    expected: &CoreTrainingOrigin,
) -> Result<()> {
    ensure!(
        corpus_fingerprint == expected.corpus_fingerprint,
        "the LLM baseline training corpus differs from MSE-JEPA"
    );
    ensure!(
        split_bounds == expected.split_bounds
            && split_bounds_pinned == expected.split_bounds_pinned,
        "the LLM baseline split origin differs from MSE-JEPA"
    );
    ensure!(
        !supports_frozen,
        "the LLM baseline reused supports under a training-corpus provenance mismatch"
    );
    ensure!(
        supports_corpus_fingerprint == Some(expected.corpus_fingerprint.as_str()),
        "the LLM baseline support provenance corpus differs from MSE-JEPA"
    );
    Ok(())
}

/// The LLM baseline's per-trajectory fan NLL `[horizons, windows, samples]` on the SAME
/// panel, horizons and sample budget as the DBWM modes, so the per-horizon predictive NLL
/// profiles are directly comparable: identical windows, identical ancestral fan size, and
/// per-window `logmeanexp` marginalization before the window mean.
fn baseline_fan_nll(
    baseline: &BarWorldModel,
    panel: &MseJepaRolloutPanel,
    window_chunk: usize,
    samples: usize,
    max_horizon: usize,
    seed: u64,
) -> Tensor {
    let windows = panel.current_dof.size()[0];
    let history_len = MAX_CONTEXT_BARS.min(BAR_MAX_CONTEXT);
    let history_start = MAX_CONTEXT_BARS - history_len;
    let mut nll = Vec::new();
    for start in window_order(windows, window_chunk) {
        seed_window_rng(seed ^ BASELINE_SEED_DOMAIN, start);
        let history_dof =
            panel
                .current_dof
                .narrow(0, start, 1)
                .narrow(1, history_start, history_len);
        let history_time_ids =
            panel
                .time_ids
                .narrow(0, start, 1)
                .narrow(1, history_start, history_len);
        let future_time_ids =
            panel
                .time_ids
                .narrow(0, start, 1)
                .narrow(1, MAX_CONTEXT_BARS, max_horizon as i64);
        let real = panel.future_dof.narrow(0, start, 1);
        nll.push(baseline_chunk_nll(
            baseline,
            &history_dof,
            &history_time_ids,
            &future_time_ids,
            &real,
            samples as i64,
        ));
    }
    Tensor::cat(&nll, 1)
}

fn baseline_chunk_nll(
    baseline: &BarWorldModel,
    history_dof: &Tensor,
    history_time_ids: &Tensor,
    future_time_ids: &Tensor,
    real_future_dof: &Tensor,
    samples: i64,
) -> Tensor {
    let supports = baseline
        .supports_for(300)
        .expect("validated 300s baseline supports");
    let chunk_windows = history_dof.size()[0];
    let history_len = history_dof.size()[1];
    let steps = future_time_ids.size()[1];
    let session = baseline.start_session(history_dof, history_time_ids);
    let rollout = baseline.imagine(
        &session,
        future_time_ids,
        samples as usize,
        1.0,
        RolloutMode::Exact,
    );

    // The exact per-step conditioning `imagine` drew each bar from: the last real row (with
    // live market channels) at the first step, the step's own market-less clock afterwards.
    let last_real_row = history_time_ids.narrow(1, history_len - 1, 1);
    let currents = if steps > 1 {
        Tensor::cat(
            &[
                last_real_row,
                time_ids_without_market(&future_time_ids.narrow(1, 1, steps - 1)),
            ],
            1,
        )
    } else {
        last_real_row
    };
    let conditioning = baseline
        .trunk()
        .forecast_conditioning(future_time_ids, &currents);
    let belief_dim = *rollout.beliefs.size().last().expect("belief dim");
    let conditioning_rows = conditioning
        .unsqueeze(1)
        .expand([chunk_windows, samples, steps, belief_dim], false)
        .reshape([-1, belief_dim]);
    let belief_rows = rollout.beliefs.reshape([-1, belief_dim]);
    let real_rows = real_future_dof
        .unsqueeze(1)
        .expand([chunk_windows, samples, steps, BAR_DOF as i64], false)
        .reshape([-1, BAR_DOF as i64]);
    let bins = supports.bin_ids(&real_rows);
    let logits = baseline
        .head()
        .logits(&belief_rows, &conditioning_rows, &bins);
    let targets = supports.targets_from_class_ids(&bins, BarScoring::Hard);
    bar_nll_terms(&logits, &targets)
        .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
        .reshape([chunk_windows, samples, steps])
        .permute([2, 0, 1])
        .contiguous()
}

pub(super) fn cleanup_failed_publication(
    runs_path: &Path,
    run: &RunDir,
    previous_latest: Option<RunDir>,
    error: anyhow::Error,
) -> Result<()> {
    if let Err(cleanup) = std::fs::remove_dir_all(&run.root) {
        return Err(error).context(format!(
            "failed report publication and could not remove {}: {cleanup}",
            run.root.display()
        ));
    }
    let latest = runs_path.join("latest");
    let latest_target = match std::fs::read_link(&latest) {
        Ok(target) => target,
        Err(read) if read.kind() == std::io::ErrorKind::NotFound => return Err(error),
        Err(read) => {
            return Err(error).context(format!(
                "failed report publication; partial run was removed but {} could not be read: \
                 {read}",
                latest.display()
            ));
        }
    };
    let active_root = if latest_target.is_relative() {
        runs_path.join(latest_target)
    } else {
        latest_target
    };
    if active_root != run.root {
        return Err(error);
    }
    let restore = if let Some(previous) = previous_latest {
        previous.activate(runs_path)
    } else {
        std::fs::remove_file(&latest)
            .with_context(|| format!("failed removing {}", latest.display()))
    };
    if let Err(restore) = restore {
        return Err(error).context(format!(
            "failed report publication; partial run was removed but latest could not be \
             restored: {restore:#}"
        ));
    }
    Err(error)
}

fn emitted_dof_to_transition7(emitted_dof: &Tensor, previous_w: &Tensor) -> Tensor {
    let emitted_size = emitted_dof.size();
    let lead = &emitted_size[..emitted_size.len() - 1];
    assert_eq!(
        emitted_size.last().copied(),
        Some(BAR_DOF as i64),
        "emitted raw DOF must end in BAR_DOF"
    );
    assert_eq!(
        previous_w.size().as_slice(),
        lead,
        "previous volume DOF must match emitted DOF leading dimensions"
    );
    let r = emitted_dof.select(-1, DOF_R as i64);
    let s = emitted_dof.select(-1, DOF_S as i64);
    let u = emitted_dof.select(-1, DOF_U as i64);
    let v = emitted_dof.select(-1, DOF_V as i64);
    let next_w = emitted_dof.select(-1, DOF_W as i64);
    let body = (&u - &v) * &s;
    let gap = r - &body;
    let upper = (1.0 - u.maximum(&v)) * &s;
    let lower = u.minimum(&v) * &s;
    let gk = (&s * &s * 0.5f64 - &body * &body * (2.0f64 * std::f64::consts::LN_2 - 1.0))
        .clamp_min(0.0)
        .sqrt();
    let alpha = 2.0 / (BAR_VOLUME_EMA_SPAN + 1.0);
    let volume_delta = next_w + (alpha + (1.0 - alpha) * (-previous_w).exp()).log();
    let present = Tensor::ones_like(&volume_delta);
    Tensor::stack(&[gap, body, upper, lower, gk, volume_delta, present], -1).detach()
}

#[cfg(test)]
mod tests {
    use super::{
        cleanup_failed_publication, coverage_profile, emitted_dof_to_transition7, predictive_nll,
        require_baseline_origin, require_fitted_endpoint, seed_window_rng, window_order,
    };
    use crate::torch::bar_dist::{decode_dof, BarDof, BAR_DOF, BAR_VOLUME_EMA_SPAN};
    #[test]
    fn baseline_origin_requires_the_same_unfrozen_corpus_and_split() {
        let origin = CoreTrainingOrigin {
            initialization: CoreInitialization::Fresh,
            train_seed: 0x5eed,
            batch_size: 8,
            resolution_secs: 300,
            min_bars: 60_000,
            split_bounds: (1_700_000_000_000, 1_710_000_000_000),
            split_bounds_pinned: true,
            corpus_fingerprint: "c".repeat(64),
            supports_scoring_sha256: "d".repeat(64),
        };
        assert!(require_baseline_origin(
            &origin.corpus_fingerprint,
            origin.split_bounds,
            true,
            false,
            Some(&origin.corpus_fingerprint),
            &origin,
        )
        .is_ok());
        assert!(require_baseline_origin(
            &"e".repeat(64),
            origin.split_bounds,
            true,
            false,
            Some(&origin.corpus_fingerprint),
            &origin,
        )
        .is_err());
        assert!(require_baseline_origin(
            &origin.corpus_fingerprint,
            (origin.split_bounds.0 + 1, origin.split_bounds.1),
            true,
            false,
            Some(&origin.corpus_fingerprint),
            &origin,
        )
        .is_err());
        assert!(require_baseline_origin(
            &origin.corpus_fingerprint,
            origin.split_bounds,
            true,
            true,
            Some(&origin.corpus_fingerprint),
            &origin,
        )
        .is_err());
        assert!(require_baseline_origin(
            &origin.corpus_fingerprint,
            origin.split_bounds,
            true,
            false,
            Some(&"e".repeat(64)),
            &origin,
        )
        .is_err());
    }

    use crate::torch::lejepa::checkpoint::{
        AuthenticatedCheckpoint, CheckpointKind, CheckpointProvenance, CoreInitialization,
        CoreTrainingOrigin, EmissionGradientMode, HeadFitProvenance, WeightReadout,
        CANONICAL_READOUT_FIT_BATCH_SIZE, CANONICAL_READOUT_FIT_SEED, CANONICAL_READOUT_FIT_STEPS,
        CANONICAL_READOUT_OPTIMIZER_RECIPE_ID, CANONICAL_READOUT_TOKEN_ROWS,
        CANONICAL_READOUT_VALIDATION_WINDOWS,
    };
    use crate::torch::lejepa::dataset::transition_bar_features;
    use crate::torch::test_rng;
    use shared::run_dir::RunDir;
    use std::path::PathBuf;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn predictive_nll_marginalizes_the_fan_with_log_mean_exp() {
        let per_trajectory = Tensor::from_slice(&[2.0f32, 2.0, 4.0, 8.0]).view([1, 2, 2]);
        let profile = predictive_nll(&per_trajectory);
        assert_eq!(profile.len(), 1);
        let first = -(0.5f64 * (-2.0f64).exp() + 0.5 * (-2.0f64).exp()).ln();
        let second = -(0.5f64 * (-4.0f64).exp() + 0.5 * (-8.0f64).exp()).ln();
        assert!((profile[0] - 0.5 * (first + second)).abs() < 1e-6);
    }

    #[test]
    fn failed_publication_removes_partial_run_and_restores_latest() {
        let runs =
            std::env::temp_dir().join(format!("mse-jepa-publication-{}", uuid::Uuid::new_v4()));
        let runs_text = runs.to_str().unwrap();
        let previous = RunDir::create_fresh(runs_text, Some("previous")).unwrap();
        let failed = RunDir::create_fresh(runs_text, Some("failed")).unwrap();
        let error = cleanup_failed_publication(
            &runs,
            &failed,
            Some(previous.clone()),
            anyhow::anyhow!("forced publication failure"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("forced publication failure"));
        assert!(!failed.root.exists());
        assert_eq!(RunDir::latest(runs_text).unwrap().root, previous.root);
        std::fs::remove_dir_all(runs).unwrap();
    }

    #[test]
    fn official_rollout_rejects_core_and_incomplete_endpoints() {
        let mut checkpoint = AuthenticatedCheckpoint {
            path: PathBuf::from("mse_jepa.ot"),
            kind: CheckpointKind::Core,
            checkpoint_sha256: "a".repeat(64),
            lineage_sha256: "b".repeat(64),
            sigreg_lambda: 0.09,
            flow_lambda: 1.0,
            provenance: CheckpointProvenance {
                completed_steps: 9,
                planned_steps: 10,
                steps_per_pass: 10,
                optimizer_recipe_id: "core".to_owned(),
                emission_gradient_mode: EmissionGradientMode::Attached,
                weight_readout: WeightReadout::Raw,
                core_origin: CoreTrainingOrigin {
                    initialization: CoreInitialization::Fresh,
                    train_seed: 0x5eed,
                    batch_size: 8,
                    resolution_secs: 300,
                    min_bars: 60_002,
                    split_bounds: (1_700_000_000_000, 1_710_000_000_000),
                    split_bounds_pinned: true,
                    corpus_fingerprint: "c".repeat(64),
                    supports_scoring_sha256: "d".repeat(64),
                },
                head_fit: None,
            },
        };
        assert!(require_fitted_endpoint(&checkpoint).is_err());
        checkpoint.kind = CheckpointKind::FittedReadout;
        checkpoint.provenance.head_fit = Some(HeadFitProvenance {
            source_checkpoint_sha256: "a".repeat(64),
            source_lineage_sha256: "b".repeat(64),
            source_weight_readout: WeightReadout::Raw,
            source_core_origin: checkpoint.provenance.core_origin.clone(),
            seed: 17,
            steps: 1,
            batch_size: 1,
            token_rows_per_step: 1,
            validation_windows: 1,
            optimizer_recipe_id: "fit".to_owned(),
        });
        assert!(require_fitted_endpoint(&checkpoint).is_err());
        checkpoint.provenance.completed_steps = 10;
        assert!(require_fitted_endpoint(&checkpoint).is_err());
        let fit = checkpoint.provenance.head_fit.as_mut().unwrap();
        fit.seed = CANONICAL_READOUT_FIT_SEED;
        fit.steps = CANONICAL_READOUT_FIT_STEPS;
        fit.batch_size = CANONICAL_READOUT_FIT_BATCH_SIZE;
        fit.token_rows_per_step = CANONICAL_READOUT_TOKEN_ROWS;
        fit.validation_windows = CANONICAL_READOUT_VALIDATION_WINDOWS;
        fit.optimizer_recipe_id = CANONICAL_READOUT_OPTIMIZER_RECIPE_ID.to_owned();
        assert!(require_fitted_endpoint(&checkpoint).is_ok());
    }

    #[test]
    fn per_window_random_streams_are_exactly_chunk_invariant() {
        let _rng = test_rng::exclusive();
        fn draws(window_chunk: usize) -> Tensor {
            let mut windows = Vec::new();
            for window in window_order(7, window_chunk) {
                seed_window_rng(0x1234_5678, window);
                windows.push(Tensor::randn([3, 5], (Kind::Float, Device::Cpu)));
            }
            Tensor::stack(&windows, 0)
        }
        let one = draws(1);
        assert!(one.equal(&draws(2)));
        assert!(one.equal(&draws(4)));
        assert!(one.equal(&draws(7)));
    }

    #[test]
    fn coverage_counts_realized_returns_inside_the_central_fan_bands() {
        let fans = Tensor::arange(32, (Kind::Float, Device::Cpu)).view([1, 1, 32]);
        let centered = Tensor::from_slice(&[15.5f32]).view([1, 1]);
        let covered = coverage_profile(&fans, &centered);
        assert_eq!(covered.step_50, vec![1.0]);
        assert_eq!(covered.step_90, vec![1.0]);
        let outside = Tensor::from_slice(&[100.0f32]).view([1, 1]);
        let uncovered = coverage_profile(&fans, &outside);
        assert_eq!(uncovered.step_50, vec![0.0]);
        assert_eq!(uncovered.step_90, vec![0.0]);
        assert_eq!(uncovered.cumulative_90, vec![0.0]);
    }

    #[test]
    fn emitted_dof_maps_directly_to_exact_transition7_algebra() {
        let emitted = Tensor::from_slice(&[0.03f32, 0.10, 0.80, 0.20, 0.40]).view([1, 5]);
        let previous_w = Tensor::from_slice(&[-0.30f32]);
        let actual =
            Vec::<f32>::try_from(emitted_dof_to_transition7(&emitted, &previous_w).view([-1]))
                .unwrap();
        let body = (0.80f64 - 0.20) * 0.10;
        let gk = (0.5 * 0.10f64.powi(2) - (2.0 * std::f64::consts::LN_2 - 1.0) * body.powi(2))
            .max(0.0)
            .sqrt();
        let alpha = 2.0 / (BAR_VOLUME_EMA_SPAN + 1.0);
        let expected = [
            0.03 - body,
            body,
            (1.0 - 0.80) * 0.10,
            0.20 * 0.10,
            gk,
            0.40 + (alpha + (1.0 - alpha) * 0.30f64.exp()).ln(),
            1.0,
        ];
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual as f64 - expected).abs() < 1e-6,
                "feature {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn emitted_volume_ema_algebra_matches_decoded_transition_features() {
        let previous_reference = 800.0f32;
        let previous_dof = BarDof {
            r: 0.012,
            s: 0.05,
            u: 0.65,
            v: 0.20,
            w: 1.25f32.ln(),
        };
        let previous_bar = decode_dof(100.0, &previous_dof, previous_reference);
        let alpha = (2.0 / (BAR_VOLUME_EMA_SPAN + 1.0)) as f32;
        let next_reference = alpha * previous_bar.volume + (1.0 - alpha) * previous_reference;
        let emitted = BarDof {
            r: -0.008,
            s: 0.07,
            u: 0.30,
            v: 0.75,
            w: 0.85f32.ln(),
        };
        let current_bar = decode_dof(previous_bar.close, &emitted, next_reference);
        let expected = transition_bar_features(&previous_bar, &current_bar);
        let emitted_tensor = Tensor::from_slice(&emitted.to_array()).view([1, BAR_DOF as i64]);
        let previous_w = Tensor::from_slice(&[previous_dof.w]);
        let actual = Vec::<f32>::try_from(
            emitted_dof_to_transition7(&emitted_tensor, &previous_w).view([-1]),
        )
        .unwrap();
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-5,
                "feature {index}: actual={actual}, expected={expected}"
            );
        }
    }
}
