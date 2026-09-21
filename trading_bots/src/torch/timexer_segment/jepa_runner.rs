use super::{
    cache::process_elapsed_ms,
    compute::{
        Engine, LrSchedule, MlpDownLr, OptimizerKind, RecipeKnobs, CAPTURE_AFTER_STEPS,
        NANOGPT_COOLDOWN_FLOOR, NANOGPT_COOLDOWN_FRAC,
    },
    corpus::{Corpus, CorpusContract, ResearchSamplePlan, WindowRef},
    jepa_eval,
    model::{CausalPatchModel, ModelConfig},
    reports::{self, HorizonSplit, TradingSplit, DECISION_HORIZONS},
    runner::{cross_section_origins, score, Evaluation, Prefetcher, TrainArgs},
    supervision::{self, DecimationPlan, SupervisionGeometry},
    temporal_moments,
};
use crate::torch::{hashing::file_sha256, single_ticker_timexer::runner::cuda_device};
use anyhow::{ensure, Result};
use clap::Args;
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use shared::{
    paths::RUNS_PATH,
    report::{write_report, Report, ReportKind, ReportSeries, ScaleKind},
    run_dir::RunDir,
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tch::{nn, Device, Kind, Tensor};

const SCHEMA: &str = "causal-patch-temporal-jepa-fixed-endpoint-v1";

#[derive(Serialize, Deserialize)]
pub(super) struct Checkpoint {
    schema: String,
    pub(super) model: ModelConfig,
    pub(super) data: CorpusContract,
    pub(super) seed: u64,
    pub(super) batch_size: usize,
    requested_tickers: Vec<String>,
    pub(super) validation_rows: usize,
    pub(super) probe_fit_rows: usize,
    pub(super) source_lookback_bars: usize,
    pub(super) completed_steps: usize,
    pub(super) schedule_budget: usize,
    base_learning_rate: f64,
    optimizer: OptimizerKind,
    scalar_lr_mult: f64,
    mlp_down_lr: MlpDownLr,
    pub(super) sample_plan_sha256: String,
    pub(super) cross_section_sha256: String,
    weights_sha256: String,
    executable_sha256: String,
    manifest_sha256: String,
    /// Omitted for historical manifests, preserving their canonical digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) unanchored_contract: Option<serde_json::Value>,
}

impl Checkpoint {
    fn digest(&self) -> Result<String> {
        let mut value = serde_json::to_value(self)?;
        value["manifest_sha256"] = serde_json::Value::String(String::new());
        Ok(digest(&serde_json::to_vec(&value)?))
    }

    pub(super) fn read(run_root: &Path) -> Result<Self> {
        let checkpoint: Self =
            serde_json::from_slice(&fs::read(run_root.join("weights/jepa-manifest.json"))?)?;
        ensure!(
            checkpoint.schema == SCHEMA,
            "unsupported research checkpoint schema"
        );
        ensure!(
            checkpoint.manifest_sha256 == checkpoint.digest()?,
            "research manifest digest mismatch"
        );
        checkpoint.model.validate()?;
        ensure!(
            checkpoint.unanchored_contract == unanchored_contract(&checkpoint.model),
            "unanchored objective provenance is absent or differs from the model contract"
        );
        ensure!(
            !checkpoint.model.jepa_mode.unanchored()
                || checkpoint.completed_steps == checkpoint.schedule_budget,
            "unanchored checkpoint must be the completed fixed-budget endpoint"
        );
        ensure!(
            !checkpoint.model.future_calendar
                && checkpoint.model.seq_len as usize == checkpoint.data.context
                && checkpoint.model.pred_len as usize == checkpoint.data.pred_len
                && checkpoint.model.features == checkpoint.data.features
                && jepa_eval::source_lookback_bars(&checkpoint.model)?
                    == checkpoint.source_lookback_bars,
            "research model/data/source geometry mismatch or noncausal future calendar"
        );
        ensure!(
            checkpoint.weights_sha256 == file_sha256(run_root.join("weights/jepa.safetensors"))?,
            "research checkpoint weight digest mismatch"
        );
        ensure!(
            checkpoint.sample_plan_sha256
                == digest(&fs::read(run_root.join("research-sample-plan.json"))?),
            "research panel digest mismatch"
        );
        Ok(checkpoint)
    }

    pub(super) fn identity(&self) -> Result<serde_json::Value> {
        let mut identity = serde_json::to_value(self)?;
        identity.as_object_mut().unwrap().remove("data");
        Ok(identity)
    }

    pub(super) fn load_model(
        &self,
        run_root: &Path,
        device: Device,
    ) -> Result<(nn::VarStore, CausalPatchModel)> {
        let weights = run_root.join("weights/jepa.safetensors");
        ensure!(
            self.weights_sha256 == file_sha256(&weights)?,
            "research checkpoint weights changed after authentication"
        );
        let mut store = nn::VarStore::new(device);
        let model = CausalPatchModel::new(&store.root(), &self.model);
        store.load(&weights)?;
        store.freeze();
        Ok((store, model))
    }
}

fn unanchored_contract(config: &ModelConfig) -> Option<serde_json::Value> {
    config.jepa_mode.unanchored().then(|| {
        let placement = config.sigreg_placement;
        let weight = config.jepa.sigreg_weight
            * if placement == super::jepa::SigregPlacement::Both { 0.5 } else { 1. };
        serde_json::json!({
            "schema": "temporal-jepa-unanchored-v1",
            "initialization": "fresh-random",
            "objective": "attached-temporal-jepa-plus-placement-sigreg",
            "objective_contract": config.jepa_contract().expect("unanchored contract"),
            "prediction_weight": config.jepa.prediction_weight,
            "sigreg_total_weight": config.jepa.sigreg_weight,
            "sigreg_placement": placement,
            "sigreg_site_weights": {
                "local": if placement.local() { weight } else { 0. },
                "state": if placement.state() { weight } else { 0. },
                "target": 0.0
            },
            "offsets_patches": config.jepa_mode.offsets(),
            "forecast_weight": 0.0,
            "reconstruction_weight": 0.0,
            "decision_weight": 0.0,
            "target_stop_gradient": false,
            "gradient_surgery": false,
            "forecast_parameters": "allocated-at-initialization-frozen-never-forwarded-or-optimized",
            "checkpoint_selection": "completed-fixed-budget",
            "downstream_readers": "fit-only-after-full-store-freeze"
        })
    })
}

fn digest(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Default)]
struct Curves(BTreeMap<String, Vec<f32>>);
impl Curves {
    fn put(&mut self, step: usize, name: impl Into<String>, value: f64) {
        let values = self.0.entry(name.into()).or_default();
        values.resize(step + 1, f32::NAN);
        values[step] = value as f32;
    }
    fn write(&self, output: &Path, base: &str, title: &str, unit: &str) -> Result<()> {
        let report = Report {
            title: title.to_owned(),
            x_label: Some("optimizer step".into()),
            y_label: Some(unit.into()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::MultiLine {
                series: self
                    .0
                    .iter()
                    .map(|(label, values)| ReportSeries {
                        label: label.clone(),
                        values: values.clone(),
                    })
                    .collect(),
            },
        };
        write_report(output.join(format!("{base}.report.bin")), &report)?;
        Ok(())
    }
}

fn record_forecast(
    forecast: &mut Curves,
    step: usize,
    measured: &Evaluation,
    cross: &Evaluation,
    pred_len: usize,
) -> Result<()> {
    forecast.put(step, "validation Gaussian NLL", measured.nll);
    forecast.put(step, "persistence Gaussian NLL", measured.persistence_nll);
    forecast.put(
        step,
        "validation neutral OHLC MSE ratio",
        measured.mse / measured.persistence_mse,
    );
    forecast.put(
        step,
        "validation raw OHLC MSE ratio",
        measured.absolute_mse / measured.absolute_persistence_mse,
    );
    forecast.put(
        step,
        "validation one-sigma coverage",
        measured.within_1_sigma,
    );
    forecast.put(
        step,
        "validation 95-percent coverage",
        measured.within_2_sigma,
    );
    let mut ratio = 0.;
    let mut count = 0.;
    for &h in DECISION_HORIZONS.iter().filter(|&&h| h <= pred_len) {
        let i = h - 1;
        let close = measured.trading.close_mse_ratio[i];
        if close.is_finite() {
            ratio += close;
            count += 1.;
        }
        forecast.put(step, format!("validation close MSE ratio h{h}"), close);
        forecast.put(
            step,
            format!("validation signed Pearson h{h}"),
            measured.trading.pearson[i],
        );
        forecast.put(
            step,
            format!("validation delayed close MSE ratio h{h}"),
            measured.trading.delayed_mse_ratio[i],
        );
        forecast.put(
            step,
            format!("validation cross-sectional signed IC h{h}"),
            cross.trading.cross_sectional_ic[i],
        );
    }
    ensure!(
        count > 0.,
        "research evaluation has no scored decision horizons"
    );
    forecast.put(
        step,
        "validation fixed-horizon close MSE ratio",
        ratio / count,
    );
    Ok(())
}

pub fn prepare(args: TrainArgs) -> Result<()> {
    args.model.validate()?;
    ensure!(
        args.eval_origins > 0 && args.probe_fit_origins > 0,
        "research panel counts must be positive"
    );
    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    let corpus = Corpus::load(
        &args.data_dir,
        &args.ticker,
        args.model.seq_len as usize,
        args.model.pred_len as usize,
        args.common_context,
        &args.model.features,
        args.market_min_cross_section,
        args.in_period_sections,
    )?;
    let plan = corpus.research_sample_plan(
        args.seed,
        args.eval_origins,
        args.probe_fit_origins,
        jepa_eval::source_lookback_bars(&args.model)?,
    )?;
    jepa_eval::validate_panel(
        &args.model,
        &corpus,
        &plan.probe_fit_refs,
        &plan.validation_refs,
    )?;
    fs::write(
        run.root.join("research-sample-plan.json"),
        serde_json::to_vec_pretty(&plan.manifest)?,
    )?;
    fs::write(
        run.root.join("timexer-segment-data-contract.json"),
        serde_json::to_vec_pretty(&corpus.contract)?,
    )?;
    let output = run.gens.join("0");
    reports::write_corpus(
        &output,
        &corpus.contract,
        corpus.calibration_refs.len(),
        &corpus.market,
    )?;
    let mut panel = Curves::default();
    panel.put(0, "eligible tickers", corpus.contract.tickers.len() as f64);
    panel.put(0, "training rows", corpus.train_refs.len() as f64);
    panel.put(
        0,
        "fixed validation origins",
        plan.validation_refs.len() as f64,
    );
    panel.put(
        0,
        "train-only probe fit origins",
        plan.probe_fit_refs.len() as f64,
    );
    panel.put(
        0,
        "chronology-excluded validation origins",
        plan.manifest.validation_chronology_excluded_origins as f64,
    );
    panel.write(
        &output,
        "timexer_segment_jepa_panel",
        "Authenticated research corpus census; no model loaded",
        "count",
    )?;
    println!(
        "Prepared research contract and panel in {}",
        run.root.display()
    );
    Ok(())
}

pub fn train(args: TrainArgs, learning_rate: f64) -> Result<()> {
    args.model.validate()?;
    let unanchored = args.model.jepa_mode.unanchored();
    ensure!(
        args.max_steps > CAPTURE_AFTER_STEPS,
        "research runs require an explicit --max-steps beyond capture warmup"
    );
    ensure!(!args.model.future_calendar, "research observed-bar forecasts require --future-calendar false; actual future print times are not known at the decision");
    ensure!(
        args.schedule_budget == 0 || args.schedule_budget == args.max_steps,
        "research schedule must end at --max-steps"
    );
    ensure!(
        args.probe_fit_origins >= 64 && args.eval_origins >= 64,
        "research probe and validation panels need at least 64 origins"
    );
    ensure!(
        args.probe_fit_origins <= jepa_eval::FIT_PANEL_LIMIT
            && args.eval_origins <= jepa_eval::SCORE_PANEL_LIMIT,
        "research panels exceed the explicit frozen-probe fit/score limits"
    );
    ensure!(
        !args.model.jepa_mode.enabled()
            || args
                .model
                .jepa_horizons()
                .iter()
                .all(|&h| h <= args.model.pred_len),
        "research probe horizon must cover every temporal prediction horizon"
    );
    ensure!(
        args.row_selection().is_identity(),
        "the shared research protocol uses the complete training row pool and fixed patch phase"
    );
    let wall = Instant::now();
    let device = cuda_device()?;
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    let output = run.gens.join("0");
    let mut corpus = Corpus::load(
        &args.data_dir,
        &args.ticker,
        args.model.seq_len as usize,
        args.model.pred_len as usize,
        args.common_context,
        &args.model.features,
        args.market_min_cross_section,
        args.in_period_sections,
    )?;
    corpus.prepare(device);
    let source_lookback_bars = jepa_eval::source_lookback_bars(&args.model)?;
    let plan = corpus.research_sample_plan(
        args.seed,
        args.eval_origins,
        args.probe_fit_origins,
        source_lookback_bars,
    )?;
    jepa_eval::validate_panel(
        &args.model,
        &corpus,
        &plan.probe_fit_refs,
        &plan.validation_refs,
    )?;
    let plan_bytes = serde_json::to_vec_pretty(&plan.manifest)?;
    fs::write(run.root.join("research-sample-plan.json"), &plan_bytes)?;
    fs::write(
        run.root.join("timexer-segment-data-contract.json"),
        serde_json::to_vec_pretty(&corpus.contract)?,
    )?;
    let cross = cross_section_origins(&corpus, &plan.validation_population_refs)?;
    let geometry = SupervisionGeometry::new(
        args.model.seq_len as usize,
        args.model.patch_len as usize,
        args.model.min_history as usize,
        args.model.pred_len as usize,
    )?;
    let census = supervision::select(&mut corpus, args.row_selection(), geometry.clone())?;
    let mut origins = corpus.train_refs.clone();
    let batches_per_pass = origins.len() / args.batch_size;
    ensure!(
        batches_per_pass > 0,
        "research batch exceeds training population"
    );
    let corpus = Arc::new(corpus);
    tch::manual_seed(args.seed as i64);
    tch::Cuda::manual_seed_all(args.seed);
    let mut store = nn::VarStore::new(device);
    let model = CausalPatchModel::new(&store.root(), &args.model);
    let schedule = LrSchedule::new(
        args.max_steps,
        NANOGPT_COOLDOWN_FRAC,
        NANOGPT_COOLDOWN_FLOOR,
    )?;
    let knobs = RecipeKnobs {
        scalar_lr_mult: args.scalar_lr_mult,
        x0_lambdas: args.model.x0_lambdas,
        schedule,
        mlp_down_lr: args.mlp_down_lr,
    };
    let mut engine = Engine::new(&store, learning_rate, knobs, args.fused, args.optimizer)?;
    if args.model.horizon_decimation.enabled() {
        engine.arm_horizon_decimation(
            DecimationPlan::new(&geometry, args.model.seq_len as usize)?,
            args.seed,
        );
    }
    let mut forecast = Curves::default();
    let mut objective = Curves::default();
    let mut runtime = Curves::default();
    let mut population = Curves::default();
    let mut geometry_curve = Curves::default();
    let mut moment_curves: [Curves; 6] = std::array::from_fn(|_| Curves::default());
    let startup_ms = process_elapsed_ms().unwrap_or(wall.elapsed().as_secs_f64() * 1000.);
    population.put(0, "eligible tickers", corpus.contract.tickers.len() as f64);
    population.put(0, "training rows", origins.len() as f64);
    population.put(0, "optimizer steps per row pass", batches_per_pass as f64);
    population.put(
        0,
        "fixed validation origins",
        plan.validation_refs.len() as f64,
    );
    population.put(
        0,
        "train-only probe fit origins",
        plan.probe_fit_refs.len() as f64,
    );
    population.put(0, "cross-section validation origins", cross.len() as f64);
    population.write(
        &output,
        "timexer_segment_jepa_panel",
        "Fixed research panel; terminal test untouched",
        "count",
    )?;
    reports::write_corpus(
        &output,
        &corpus.contract,
        corpus.calibration_refs.len(),
        &corpus.market,
    )?;
    let loader = Prefetcher::new(Arc::clone(&corpus));
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut step = 0usize;
    let mut sums: Option<Tensor> = None;
    let mut moment_sums: Option<Tensor> = None;
    let mut interval_steps = 0usize;
    let mut train_ms = 0.;
    let mut eval_ms = 0.;
    let mut interval = Instant::now();
    while step < args.max_steps {
        origins.shuffle(&mut rng);
        let used = &origins[..batches_per_pass * args.batch_size];
        loader.request(&used[..args.batch_size])?;
        for (index, _) in used.chunks_exact(args.batch_size).enumerate() {
            let (host, _) = loader.receive()?;
            let losses = if step == CAPTURE_AFTER_STEPS && engine.capture_ready() {
                engine.arm_step_graph(&model, &host)?
            } else {
                engine.step(&model, &host)?
            };
            step += 1;
            interval_steps += 1;
            let auxiliary = losses
                .jepa
                .unwrap_or_else(|| Tensor::zeros([0], (Kind::Float, device)));
            let scalars = Tensor::cat(
                &[
                    Tensor::stack(&[losses.nll, losses.mse, losses.objective], 0),
                    auxiliary,
                ],
                0,
            )
            .to_kind(Kind::Double);
            sums = Some(match sums {
                Some(total) => total + scalars,
                None => scalars,
            });
            if let Some(diagnostics) = losses.moments {
                let diagnostics = diagnostics.to_kind(Kind::Double);
                moment_sums = Some(match moment_sums {
                    Some(total) => total + diagnostics,
                    None => diagnostics,
                });
            }
            if step < args.max_steps && index + 1 < batches_per_pass {
                let next = (index + 1) * args.batch_size;
                loader.request(&used[next..next + args.batch_size])?;
            }
            if step % args.eval_every == 0 || step == args.max_steps {
                tch::Cuda::synchronize(0);
                let elapsed = interval.elapsed().as_secs_f64() * 1000.;
                train_ms += elapsed;
                let values = Vec::<f64>::try_from(
                    sums.take().expect("nonempty interval") / interval_steps as f64,
                )?;
                ensure!(
                    values[if unanchored { 2 } else { 0 }..]
                        .iter()
                        .all(|v| v.is_finite())
                        && (!unanchored || (values[0].is_nan() && values[1].is_nan())),
                    "nonfinite research objective at step {step}"
                );
                if !unanchored {
                    objective.put(step, "forecast NLL (not total objective)", values[0]);
                    objective.put(step, "forecast MSE", values[1]);
                }
                objective.put(step, "total optimization objective", values[2]);
                let labels = super::jepa::diagnostic_labels(&args.model);
                for (index, value) in values[3..].iter().enumerate() {
                    let reader_sigreg = !unanchored && args.model.sigreg_placement.enabled();
                    if unanchored {
                        if matches!(index, 1 | 2)
                            || (index == 8 && !args.model.sigreg_placement.local())
                            || (index == 10 && !args.model.sigreg_placement.state())
                        {
                            continue;
                        }
                    } else if reader_sigreg {
                        let placement = args.model.sigreg_placement;
                        if (index == 0 && placement == super::jepa::SigregPlacement::State)
                            || (index == 2 && placement == super::jepa::SigregPlacement::Local)
                        {
                            continue;
                        }
                    } else if (index == 1 && !args.model.jepa_mode.regularized())
                        || (index == 2
                            && args.model.jepa_mode != super::jepa::JepaMode::AnchoredReconstruct)
                    {
                        continue;
                    }
                    let destination = match index {
                        3 if reader_sigreg => &mut population,
                        11..=13 if unanchored => &mut population,
                        14 | 15 if unanchored => &mut geometry_curve,
                        4 | 5 => &mut population,
                        6 | 7 => &mut geometry_curve,
                        _ => &mut objective,
                    };
                    destination.put(step, labels[index], *value);
                }
                if let Some(total) = moment_sums.take() {
                    let values = Vec::<f64>::try_from((total / interval_steps as f64).view([-1]))?;
                    ensure!(
                        values.len()
                            == temporal_moments::DIAGNOSTIC_LABELS.len()
                                * temporal_moments::HORIZONS.len()
                            && values.iter().all(|v| v.is_finite()),
                        "invalid temporal moment diagnostics at step {step}"
                    );
                    for (index, (label, row)) in temporal_moments::DIAGNOSTIC_LABELS
                        .iter()
                        .zip(values.chunks_exact(temporal_moments::HORIZONS.len()))
                        .enumerate()
                    {
                        let chart = match index {
                            0..=2 => 0,
                            3 => 1,
                            4 => 2,
                            5 => 3,
                            6 => 4,
                            _ => unreachable!(),
                        };
                        for (&horizon, &value) in temporal_moments::HORIZONS.iter().zip(row) {
                            moment_curves[chart].put(
                                step,
                                format!("{label}; horizon {horizon}"),
                                value,
                            );
                        }
                    }
                    moment_curves[5].put(
                        step,
                        "applied temporal moment weight",
                        args.model.temporal_moment_weight,
                    );
                    moment_curves[5].put(
                        step,
                        "applied decision MSE weight",
                        args.model.decision_mse_weight,
                    );
                    for (curve, (suffix, units)) in moment_curves.iter().zip([
                        ("train", "squared normalized close moments"),
                        ("train_error", "normalized close MSE"),
                        ("train_bias", "normalized close residual"),
                        ("train_rows", "valid rows summed over dense sources"),
                        (
                            "train_pairs",
                            "ordered distinct row pairs summed over dense sources",
                        ),
                        ("train_weights", "objective coefficient"),
                    ]) {
                        curve.write(
                            &output,
                            &format!("timexer_segment_temporal_moment_{suffix}"),
                            "Training-only close conditional moments; interval means, distinct batch-row pairs (not IID evidence)",
                            units,
                        )?;
                    }
                }
                let measured = if unanchored {
                    None
                } else {
                    let evaluation_started = Instant::now();
                    let measured = score(
                        &corpus,
                        &model,
                        &plan.validation_refs,
                        args.eval_batch_size,
                        device,
                    )?;
                    let cross_measured =
                        score(&corpus, &model, &cross, args.eval_batch_size, device)?;
                    eval_ms += evaluation_started.elapsed().as_secs_f64() * 1000.;
                    record_forecast(
                        &mut forecast,
                        step,
                        &measured,
                        &cross_measured,
                        args.model.pred_len as usize,
                    )?;
                    Some((measured, cross_measured))
                };
                runtime.put(
                    step,
                    "training interval milliseconds per update",
                    elapsed / interval_steps as f64,
                );
                runtime.put(step, "startup milliseconds", startup_ms);
                runtime.put(step, "cumulative training milliseconds", train_ms);
                runtime.put(step, "cumulative routine evaluation milliseconds", eval_ms);
                runtime.put(
                    step,
                    "CUDA graph captured",
                    if engine.step_graph_captured() { 1. } else { 0. },
                );
                if !unanchored {
                    forecast.write(
                        &output,
                        "timexer_segment_jepa_forecast",
                        "Fixed endpoint research; uncalibrated predictions, no oracle rescaling",
                        "forecast score",
                    )?;
                }
                objective.write(&output, "timexer_segment_jepa_objective",
                    if unanchored {
                        "Unanchored attached temporal JEPA plus placement SIGReg only; fixed completed budget, no forecast objective"
                    } else if args.model.temporal_moments_enabled() {
                        "Forecast and close conditional moment objectives are distinct; no auxiliary-loss checkpoint selection"
                    } else if args.model.sigreg_placement != super::jepa::SigregPlacement::Off {
                        "Forecast loss and reader/local SIGReg are distinct; no geometry-loss checkpoint selection"
                    } else if args.model.jepa_mode.conditional() {
                        "Forecast and fixed conditional CF objectives are distinct; no auxiliary-loss checkpoint selection"
                    } else {
                        "Forecast and latent objectives are distinct; no latent-loss checkpoint selection"
                    }, "loss")?;
                population.write(
                    &output,
                    "timexer_segment_jepa_panel",
                    if unanchored {
                        "Attached temporal pairs and independent batch populations at actual reader interfaces"
                    } else if args.model.jepa_mode.conditional() {
                        "Fixed panel and realized conditional CF supervision counts"
                    } else if args.model.sigreg_placement != super::jepa::SigregPlacement::Off {
                        "Fixed panel and valid batch populations per causal reader view; time is not an independent population"
                    } else {
                        "Fixed panel and realized latent supervision counts"
                    },
                    "count",
                )?;
                if args.model.jepa_mode.enabled()
                    || args.model.sigreg_placement != super::jepa::SigregPlacement::Off
                {
                    geometry_curve.write(&output, "timexer_segment_jepa_geometry",
                        if unanchored {
                            "Unanchored observation, prediction and actual state population spread"
                        } else if args.model.jepa_mode.conditional() {
                            "Fixed CF targets and conditional-mean prediction spread; weak mean variation is legitimate"
                        } else if args.model.sigreg_placement != super::jepa::SigregPlacement::Off {
                            "Local observation and actual forecast-reader population spread; persistent states remain legal"
                        } else {
                            "Observation and conditional-mean prediction spread; unequal variance is expected"
                        }, "standard deviation")?;
                }
                runtime.write(
                    &output,
                    "timexer_segment_jepa_runtime",
                    "Actual fixed-step research execution costs",
                    "milliseconds",
                )?;
                if let Some((measured, cross_measured)) = &measured {
                    reports::write_horizon(
                        &output,
                        0,
                        step,
                        Some(HorizonSplit {
                            curve: &measured.horizon,
                            origins: plan.validation_refs.len(),
                        }),
                        None,
                        corpus.contract.tickers.len(),
                    )?;
                    reports::write_trading(
                        &output,
                        0,
                        step,
                        Some(TradingSplit {
                            curve: &measured.trading,
                            portfolio: &measured.portfolio,
                            origins: plan.validation_refs.len(),
                        }),
                        None,
                        Some(TradingSplit {
                            curve: &cross_measured.trading,
                            portfolio: &cross_measured.portfolio,
                            origins: cross.len(),
                        }),
                        corpus.contract.tickers.len(),
                    )?;
                    reports::write_supervision_occupancy(
                        &output,
                        0,
                        step,
                        args.batch_size,
                        &census,
                    )?;
                }
                if let Some(decimation) = engine.drain_horizon_decimation() {
                    reports::write_horizon_decimation(
                        &output,
                        0,
                        step,
                        args.batch_size,
                        &decimation,
                    )?;
                }
                interval_steps = 0;
                interval = Instant::now();
            }
            if step == args.max_steps {
                break;
            }
        }
    }
    ensure!(
        engine.step_graph_captured(),
        "research run never entered the captured execution path"
    );
    drop(loader);
    drop(engine);
    // No reader is fitted until all representation/predictor parameters have been frozen.
    store.freeze();
    crate::torch::cuda::empty_cache();
    let probe_start = Instant::now();
    jepa_eval::evaluate(
        &model,
        &corpus,
        &plan.probe_fit_refs,
        &plan.validation_refs,
        args.eval_batch_size,
        device,
        &output,
        step,
    )?;
    runtime.put(
        step,
        "frozen probe milliseconds",
        probe_start.elapsed().as_secs_f64() * 1000.,
    );
    let weights = run.weights.join("jepa.safetensors");
    store.save(&weights)?;
    let mut checkpoint = Checkpoint {
        schema: SCHEMA.into(),
        unanchored_contract: unanchored_contract(&args.model),
        model: args.model,
        data: corpus.contract.clone(),
        seed: args.seed,
        batch_size: args.batch_size,
        completed_steps: step,
        schedule_budget: args.max_steps,
        requested_tickers: args.ticker,
        validation_rows: args.eval_origins,
        probe_fit_rows: args.probe_fit_origins,
        source_lookback_bars,
        base_learning_rate: learning_rate,
        optimizer: args.optimizer,
        scalar_lr_mult: args.scalar_lr_mult,
        mlp_down_lr: args.mlp_down_lr,
        sample_plan_sha256: digest(&plan_bytes),
        cross_section_sha256: corpus.origins_sha256(&cross),
        weights_sha256: file_sha256(&weights)?,
        executable_sha256: file_sha256(std::env::current_exe()?)?,
        manifest_sha256: String::new(),
    };
    checkpoint.manifest_sha256 = checkpoint.digest()?;
    fs::write(
        run.weights.join("jepa-manifest.json"),
        serde_json::to_vec_pretty(&checkpoint)?,
    )?;
    runtime.put(
        step,
        "process wall milliseconds including probes and checkpoint",
        process_elapsed_ms().unwrap_or(wall.elapsed().as_secs_f64() * 1000.),
    );
    runtime.write(
        &output,
        "timexer_segment_jepa_runtime",
        "Actual fixed-step research costs; build and queue wait excluded",
        "milliseconds",
    )?;
    println!(
        "Completed research endpoint at {step} updates; reports {}",
        output.display()
    );
    Ok(())
}

/// Authenticate the corpus and exact fixed panels once, independently of model residency.
pub(super) fn load_panel(
    run_root: &Path,
    data_dir: &Path,
    device: Device,
) -> Result<(Checkpoint, Arc<Corpus>, ResearchSamplePlan, Vec<WindowRef>)> {
    let checkpoint = Checkpoint::read(run_root)?;
    let mut corpus = Corpus::load(
        data_dir,
        &checkpoint.requested_tickers,
        checkpoint.data.context,
        checkpoint.data.pred_len,
        checkpoint.data.common_context,
        &checkpoint.data.features,
        checkpoint.data.market_min_cross_section,
        checkpoint.data.in_period_sections,
    )?;
    ensure!(
        corpus.contract == checkpoint.data,
        "research checkpoint corpus changed"
    );
    let plan = corpus.research_sample_plan(
        checkpoint.seed,
        checkpoint.validation_rows,
        checkpoint.probe_fit_rows,
        checkpoint.source_lookback_bars,
    )?;
    let plan_value: serde_json::Value =
        serde_json::from_slice(&fs::read(run_root.join("research-sample-plan.json"))?)?;
    ensure!(
        serde_json::to_value(&plan.manifest)? == plan_value,
        "research sample plan changed"
    );
    let cross = cross_section_origins(&corpus, &plan.validation_population_refs)?;
    ensure!(
        corpus.origins_sha256(&cross) == checkpoint.cross_section_sha256,
        "research cross-section panel changed"
    );
    corpus.prepare(device);
    Ok((checkpoint, Arc::new(corpus), plan, cross))
}

#[derive(Clone, Debug, Args)]
pub struct EvaluateArgs {
    #[arg(long)]
    pub run_root: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
}

pub fn evaluate(args: EvaluateArgs) -> Result<()> {
    ensure!(
        args.batch_size > 0 && !args.output.exists(),
        "evaluation requires positive batch size and a fresh output path"
    );
    let device = cuda_device()?;
    let (checkpoint, corpus, plan, cross) = load_panel(&args.run_root, &args.data_dir, device)?;
    let (_store, model) = checkpoint.load_model(&args.run_root, device)?;
    if !checkpoint.model.jepa_mode.unanchored() {
        let measured = score(
            &corpus,
            &model,
            &plan.validation_refs,
            args.batch_size,
            device,
        )?;
        let cross_measured = score(&corpus, &model, &cross, args.batch_size, device)?;
        let mut forecast = Curves::default();
        record_forecast(
            &mut forecast,
            checkpoint.completed_steps,
            &measured,
            &cross_measured,
            checkpoint.model.pred_len as usize,
        )?;
        forecast.write(
            &args.output,
            "timexer_segment_jepa_forecast",
            "Authenticated fixed endpoint re-evaluation; uncalibrated forecasts",
            "forecast score",
        )?;
    }
    jepa_eval::evaluate(
        &model,
        &corpus,
        &plan.probe_fit_refs,
        &plan.validation_refs,
        args.batch_size,
        device,
        &args.output,
        checkpoint.completed_steps,
    )
}
