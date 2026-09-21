//! Controlled learned-memory task, deliberately not market-performance evidence.
//! Old auxiliary cues change a future price pulse; paired validation episodes have exactly
//! identical recent prices, auxiliaries, normalization inputs and future innovations. A second
//! independently trained task makes that cue irrelevant. Bayesian means are known analytically.
use std::{path::PathBuf, time::Instant};

use anyhow::{ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use shared::report::ReportSeries;
use tch::{nn, Device, Kind, Tensor};

use super::{
    compute::{Engine, LrSchedule, OptimizerKind, RecipeKnobs, CAPTURE_AFTER_STEPS},
    corpus::Batch,
    features::FeatureSet,
    jepa::{JepaMode, ReaderNorm, SigregPlacement},
    jepa_eval::{future_targets, lines, probe_geometry, ridge_with_refit, score, series, Fit},
    model::{CausalPatchModel, ModelConfig, ScaleCoupling, Statistics},
};

const CONTEXT: usize = 512;
const HORIZON: usize = 192;
const PATCH: usize = 16;
const SOURCE: usize = CONTEXT - HORIZON - 1;
const CUE_START: usize = PATCH;
const RECENT_PATCHES: i64 = 4;
const NOISE_SIGMA: f64 = 0.0005;
const PULSE_AMPLITUDE: f64 = 0.01;
const VALIDATION_STREAM: u64 = 0xB18D_604F_D2A9_5E31;
const TRAIN_STREAM: u64 = 0x71E4_93D2_A609_8B5F;
const READER_INNER_STREAM: u64 = 0xCEF4_139B_627D_A085;
const READER_HOLDOUT_STREAM: u64 = 0x54A9_F08E_B732_61CD;
const READER_INNER_ROWS: usize = 2048;
const READER_HOLDOUT_ROWS: usize = 512;
const FROZEN_READERS: [&str; 3] = [
    "frozen-ridge-full",
    "frozen-ridge-recomputed-recent",
    "frozen-ridge-local",
];

#[derive(Clone, Debug, clap::Args, Serialize)]
pub struct MemoryArgs {
    #[arg(long)]
    pub output: PathBuf,
    /// Fixed optimizer budget PER (objective, relevant/null task); never early stopped.
    #[arg(long, default_value_t = 1024)]
    pub steps: usize,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
    /// Independent matched pairs; effective validation sample size is pairs, not 2*pairs.
    #[arg(long, default_value_t = 256)]
    pub validation_pairs: usize,
    #[arg(long, default_value_t = 128)]
    pub eval_every: usize,
    #[arg(long, default_value_t = 20260919)]
    pub seed: u64,
    #[arg(long, default_value_t = 0.001)]
    pub learning_rate: f64,
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "off,latent-one,latent-multi,anchored,anchored-no-sigreg,anchored-reconstruct"
    )]
    pub jepa_modes: Vec<JepaMode>,
    /// Compare the four actual-reader SIGReg placements instead of the legacy JEPA objectives.
    #[arg(long, conflicts_with_all = ["jepa_modes", "unanchored_sigreg"])]
    pub reader_sigreg: bool,
    /// Fresh attached-target JEPA pretraining only; fit frozen readers after all updates.
    #[arg(long, conflicts_with_all = ["reader_sigreg", "jepa_modes"])]
    pub unanchored_sigreg: bool,
}

fn config(mode: JepaMode) -> ModelConfig {
    ModelConfig {
        seq_len: CONTEXT as i64,
        pred_len: HORIZON as i64,
        patch_len: PATCH as i64,
        layers: 2,
        d_model: 128,
        heads: 2,
        ffn: 256,
        dropout: 0.,
        min_history: 256,
        features: FeatureSet {
            volume: true,
            ..FeatureSet::NONE
        },
        future_calendar: false,
        scale_coupling: ScaleCoupling::Decoupled,
        jepa_mode: mode,
        ..ModelConfig::default()
    }
}

/// Zero at the context's final close, exactly rather than sin(pi)'s numerical residue.
/// Therefore centering on that close cannot smuggle the cue into any pre-event price.
fn pulse(h: usize) -> f64 {
    if h == 0 || h >= HORIZON {
        0.
    } else {
        PULSE_AMPLITUDE * (std::f64::consts::PI * h as f64 / HORIZON as f64).sin()
    }
}
fn normal(rng: &mut ChaCha8Rng) -> f64 {
    let radius = (-2. * (1. - rng.random::<f64>()).ln()).sqrt();
    radius * (std::f64::consts::TAU * rng.random::<f64>()).cos()
}

struct Innovations {
    close: Vec<f64>,
    wick: Vec<f64>,
}
impl Innovations {
    fn draw(rng: &mut ChaCha8Rng) -> Self {
        let mut close = Vec::with_capacity(CONTEXT + HORIZON);
        let mut wick = Vec::with_capacity(CONTEXT + HORIZON);
        let mut level = 0.;
        for _ in 0..CONTEXT + HORIZON {
            level += normal(rng) * NOISE_SIGMA;
            close.push(level);
            wick.push(0.0001 + rng.random::<f64>() * 0.0001);
        }
        Self { close, wick }
    }
    fn row(&self, cue: f64, relevant: bool) -> Vec<f32> {
        let length = CONTEXT + HORIZON;
        let mut row = vec![0.; Batch::row_width(CONTEXT, HORIZON, 2)];
        // This anchor is cue-independent because the pulse is exactly zero at h=192.
        let anchor = self.close[CONTEXT - 1];
        let close_at = |t: usize| {
            self.close[t]
                + if relevant && t > SOURCE {
                    cue * pulse(t - SOURCE)
                } else {
                    0.
                }
        };
        for t in 0..length {
            let close = close_at(t);
            let open = if t == 0 { close } else { close_at(t - 1) };
            let prices = [
                open,
                open.max(close) + self.wick[t],
                open.min(close) - self.wick[t],
                close,
            ];
            for (c, value) in prices.into_iter().enumerate() {
                row[t * 4 + c] = (value - anchor) as f32;
            }
            row[length * 4 + t] = 1.;
            if (CUE_START..CUE_START + PATCH).contains(&t) {
                row[length * 5 + t * 2] = cue as f32;
                row[length * 5 + t * 2 + 1] = 1.;
            }
        }
        row[length * 8] = anchor.exp() as f32;
        row
    }
}

fn host_batch(rows: &[Vec<f32>], pin: Option<Device>) -> Batch {
    let width = Batch::row_width(CONTEXT, HORIZON, 2);
    let flat: Vec<_> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    let packed = Tensor::from_slice(&flat).reshape([rows.len() as i64, width as i64]);
    let packed = match pin {
        Some(device) => packed.pin_memory(device),
        None => packed,
    };
    Batch::from_packed(packed, CONTEXT, HORIZON, 2, rows.len() * HORIZON)
}

fn training_batch(rng: &mut ChaCha8Rng, rows: usize, relevant: bool, device: Device) -> Batch {
    // Independent episodes, NOT paired rows, so SIGReg's batch population stays independent.
    let rows: Vec<_> = (0..rows)
        .map(|_| {
            let cue = if rng.random::<bool>() { 1. } else { -1. };
            Innovations::draw(rng).row(cue, relevant)
        })
        .collect();
    host_batch(&rows, Some(device))
}

fn validation(args: &MemoryArgs, relevant: bool, device: Device) -> Batch {
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed ^ VALIDATION_STREAM);
    let rows: Vec<_> = (0..args.validation_pairs)
        .flat_map(|_| {
            let innovations = Innovations::draw(&mut rng);
            [
                innovations.row(1., relevant),
                innovations.row(-1., relevant),
            ]
        })
        .collect();
    host_batch(&rows, None).to_device(device)
}

struct Measurement {
    ratio: Vec<f64>,
    error: Vec<f64>,
    correlation: Vec<f64>,
    bayes_regret: Vec<f64>,
    bayes_ratio: Vec<f64>,
    pair_error: Vec<f64>,
    pair_effect: Vec<f64>,
    pair_expected: Vec<f64>,
    pair_sign: Vec<f64>,
    state_distance: f64,
    recent_distance: f64,
    observation_distance: f64,
    input_max_difference: [f64; 7],
}

fn paired_delta(x: &Tensor) -> Tensor {
    let shape = x.size();
    x.reshape([shape[0] / 2, 2, -1]).select(1, 0) - x.reshape([shape[0] / 2, 2, -1]).select(1, 1)
}

struct SourceFeatures {
    full: Tensor,
    recent: Tensor,
    local: Tensor,
}
impl SourceFeatures {
    fn extract(model: &CausalPatchModel, batch: &Batch, source: i64) -> Self {
        let _guard = tch::no_grad_guard();
        let views = model.representation_views(batch, false);
        Self {
            full: views.state.select(1, source).contiguous(),
            recent: model.representation_state_at_recent(batch, source, RECENT_PATCHES),
            local: views.observation.select(1, source).contiguous(),
        }
    }

    fn readers(&self) -> [&Tensor; 3] {
        [&self.full, &self.recent, &self.local]
    }
}

struct ReaderPanel {
    features: SourceFeatures,
    target: Tensor,
    mask: Tensor,
}

fn reader_panel(
    args: &MemoryArgs,
    model: &CausalPatchModel,
    relevant: bool,
    source: i64,
    horizons: &[i64],
    stream: u64,
    rows: usize,
    device: Device,
) -> Result<ReaderPanel> {
    let _guard = tch::no_grad_guard();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed ^ stream);
    let mut features: [Vec<Tensor>; 3] = std::array::from_fn(|_| Vec::new());
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    for start in (0..rows).step_by(args.batch_size) {
        let batch = training_batch(
            &mut rng,
            args.batch_size.min(rows - start),
            relevant,
            device,
        )
        .to_device(device);
        let extracted = SourceFeatures::extract(model, &batch, source);
        for (values, feature) in features.iter_mut().zip(extracted.readers()) {
            ensure!(
                !feature.requires_grad(),
                "frozen memory feature retained a gradient"
            );
            values.push(feature.shallow_clone());
        }
        let (target, mask) = future_targets(&batch, SOURCE as i64, horizons);
        targets.push(target);
        masks.push(mask);
    }
    let [full, recent, local] = features;
    let panel = ReaderPanel {
        features: SourceFeatures {
            full: Tensor::cat(&full, 0),
            recent: Tensor::cat(&recent, 0),
            local: Tensor::cat(&local, 0),
        },
        target: Tensor::cat(&targets, 0),
        mask: Tensor::cat(&masks, 0),
    };
    ensure!(
        panel.target.size()[0] == rows as i64 && panel.mask.eq(1.).all().int64_value(&[]) == 1,
        "memory reader panel has missing rows or invalid targets"
    );
    Ok(panel)
}

struct PairedEvidence {
    distance: [f64; 3],
    input_max_difference: [f64; 7],
}

fn paired_evidence(
    batch: &Batch,
    stats: &Statistics,
    features: &SourceFeatures,
    source: i64,
) -> Result<PairedEvidence> {
    let distance = features
        .readers()
        .map(|x| paired_delta(x).square().mean(Kind::Float).double_value(&[]));
    let recent_start = (SOURCE + 1 - RECENT_PATCHES as usize * PATCH) as i64;
    let recent_bars = RECENT_PATCHES * PATCH as i64;
    let max_difference = |x: &Tensor| paired_delta(x).abs().max().double_value(&[]);
    let input_max_difference = [
        max_difference(&batch.log_prices.narrow(1, recent_start, recent_bars)),
        max_difference(&batch.aux.narrow(1, recent_start, recent_bars)),
        max_difference(&batch.valid.narrow(1, recent_start, recent_bars)),
        max_difference(&batch.market_cum.narrow(1, recent_start, recent_bars)),
        max_difference(&batch.anchor.unsqueeze(-1)),
        max_difference(&stats.sigma.select(1, source).unsqueeze(-1)),
        max_difference(&stats.range.select(1, source).unsqueeze(-1)),
    ];
    ensure!(
        input_max_difference.iter().all(|&v| v == 0.),
        "paired recent inputs or normalization differ: {input_max_difference:?}"
    );
    // Exact invariance is a construction check, not a learned-memory success criterion.
    ensure!(distance[1] <= 1e-12 && distance[2] <= 1e-12,
        "synthetic cue leaked into a supposedly identical recent/local input: recent={:e}, local={:e}",
        distance[1], distance[2]);
    Ok(PairedEvidence {
        distance,
        input_max_difference,
    })
}

fn measure(
    model: &CausalPatchModel,
    batch: &Batch,
    relevant: bool,
    horizons: &[i64],
    source: i64,
) -> Result<Measurement> {
    let _guard = tch::no_grad_guard();
    let stats = model.statistics(batch);
    let views = model.representation_views(batch, false);
    let output = model.output(&model.head(batch, &views.state, false));
    let mean = model.decode(&output, &stats).select(1, source).select(1, 3)
        * stats.sigma.select(1, source).unsqueeze(-1);
    let indices = Tensor::from_slice(&horizons.iter().map(|h| h - 1).collect::<Vec<_>>())
        .to_device(mean.device());
    let prediction = mean.index_select(1, &indices);
    let features = SourceFeatures {
        full: views.state.select(1, source),
        recent: model.representation_state_at_recent(batch, source, RECENT_PATCHES),
        local: views.observation.select(1, source),
    };
    let evidence = paired_evidence(batch, &stats, &features, source)?;
    measure_prediction(&prediction, batch, relevant, horizons, &evidence)
}

fn measure_prediction(
    prediction: &Tensor,
    batch: &Batch,
    relevant: bool,
    horizons: &[i64],
    evidence: &PairedEvidence,
) -> Result<Measurement> {
    ensure!(
        prediction.isfinite().all().int64_value(&[]) == 1,
        "synthetic memory forecast is nonfinite"
    );
    let (target, mask) = future_targets(batch, SOURCE as i64, horizons);
    let bayes_row: Vec<f32> = horizons
        .iter()
        .map(|&h| {
            if relevant {
                pulse(h as usize) as f32
            } else {
                0.
            }
        })
        .collect();
    let signs = Tensor::from_slice(&[1f32, -1.])
        .to_device(prediction.device())
        .repeat([batch.rows() / 2])
        .unsqueeze(-1);
    let bayes = signs
        * Tensor::from_slice(&bayes_row)
            .to_device(prediction.device())
            .unsqueeze(0);
    let predicted_scores = score(prediction, &target, &mask);
    let bayes_scores = score(&bayes, &target, &mask);
    let regret = (prediction - &bayes)
        .square()
        .mean_dim([0i64].as_slice(), false, Kind::Float)
        / PULSE_AMPLITUDE.powi(2);
    let delta = paired_delta(prediction);
    let expected = paired_delta(&bayes);
    let pair_error = (&delta - &expected)
        .square()
        .mean_dim([0i64].as_slice(), false, Kind::Float)
        / PULSE_AMPLITUDE.powi(2);
    let pair_effect = delta.mean_dim([0i64].as_slice(), false, Kind::Float);
    let pair_expected = expected.mean_dim([0i64].as_slice(), false, Kind::Float);
    let pair_sign = (&delta * &expected).gt(0.).to_kind(Kind::Float).mean_dim(
        [0i64].as_slice(),
        false,
        Kind::Float,
    );
    let values = Tensor::stack(
        &[regret, pair_error, pair_effect, pair_expected, pair_sign],
        0,
    )
    .to_device(Device::Cpu);
    let vector = |row: i64| {
        (0..horizons.len() as i64)
            .map(|i| values.double_value(&[row, i]))
            .collect::<Vec<_>>()
    };
    let mut sign = vector(4);
    for (i, &h) in horizons.iter().enumerate() {
        if !relevant || pulse(h as usize) == 0. {
            sign[i] = f64::NAN;
        }
    }
    Ok(Measurement {
        ratio: predicted_scores.iter().map(|s| s.ratio).collect(),
        error: predicted_scores.iter().map(|s| s.mse).collect(),
        correlation: predicted_scores.iter().map(|s| s.correlation).collect(),
        bayes_ratio: bayes_scores.iter().map(|s| s.ratio).collect(),
        bayes_regret: vector(0),
        pair_error: vector(1),
        pair_effect: vector(2),
        pair_expected: vector(3),
        pair_sign: sign,
        state_distance: evidence.distance[0],
        recent_distance: evidence.distance[1],
        observation_distance: evidence.distance[2],
        input_max_difference: evidence.input_max_difference,
    })
}

#[derive(Serialize)]
struct FrozenAudit {
    completed_updates: usize,
    training_episodes: usize,
    captured_training: bool,
    frozen_at_update: usize,
    reader_fit_started_after_update: usize,
    inner_rows: usize,
    holdout_rows: usize,
    refit_rows: usize,
    validation_pairs: usize,
    parameter_tensors_checked: usize,
    parameter_scalars_checked: usize,
    remaining_trainable_tensors: usize,
    max_parameter_difference: f64,
}

fn fit_frozen_readers(
    args: &MemoryArgs,
    store: &nn::VarStore,
    model: &CausalPatchModel,
    relevant: bool,
    source: i64,
    horizons: &[i64],
    history: &mut ArmHistory,
    device: Device,
) -> Result<()> {
    let _guard = tch::no_grad_guard();
    let completed_updates = history.training.last().map_or(0, |(step, _, _, _)| *step);
    ensure!(
        completed_updates == args.steps,
        "reader fitting preceded the completed pretraining budget"
    );
    let before: Vec<_> = store
        .variables()
        .into_iter()
        .map(|(name, value)| {
            ensure!(
                !value.requires_grad(),
                "memory parameter {name} was not frozen"
            );
            Ok((name, value.copy()))
        })
        .collect::<Result<_>>()?;
    ensure!(!before.is_empty(), "empty memory parameter snapshot");
    let inner = reader_panel(
        args,
        model,
        relevant,
        source,
        horizons,
        READER_INNER_STREAM,
        READER_INNER_ROWS,
        device,
    )?;
    let holdout = reader_panel(
        args,
        model,
        relevant,
        source,
        horizons,
        READER_HOLDOUT_STREAM,
        READER_HOLDOUT_ROWS,
        device,
    )?;
    let refit_target = Tensor::cat(&[&inner.target, &holdout.target], 0);
    let refit_mask = Tensor::cat(&[&inner.mask, &holdout.mask], 0);
    // Each reader selects per-horizon penalties on independent TRAINING-only episodes.
    // All fitting finishes before the paired validation set is even constructed.
    for (inner_x, holdout_x) in inner
        .features
        .readers()
        .into_iter()
        .zip(holdout.features.readers())
    {
        let refit_x = Tensor::cat(&[inner_x, holdout_x], 0);
        history.readers.push(ridge_with_refit(
            inner_x,
            &inner.target,
            &inner.mask,
            holdout_x,
            &holdout.target,
            &holdout.mask,
            Some((&refit_x, &refit_target, &refit_mask)),
        )?);
    }
    let heldout = validation(args, relevant, device);
    let features = SourceFeatures::extract(model, &heldout, source);
    let evidence = paired_evidence(&heldout, &model.statistics(&heldout), &features, source)?;
    for ((reader, x), name) in history
        .readers
        .iter()
        .zip(features.readers())
        .zip(FROZEN_READERS)
    {
        history.points.push((
            args.steps,
            name,
            measure_prediction(&reader.predict(x), &heldout, relevant, horizons, &evidence)?,
        ));
    }
    let after = store.variables();
    ensure!(
        before.len() == after.len(),
        "memory store changed during reader fitting"
    );
    let mut max_parameter_difference = 0f64;
    for (name, snapshot) in &before {
        let value = after.get(name).expect("frozen parameter disappeared");
        ensure!(
            !value.requires_grad(),
            "reader fitting unfroze parameter {name}"
        );
        let difference = (value - snapshot).abs().max().double_value(&[]);
        ensure!(
            snapshot.equal(value),
            "reader fitting changed frozen parameter {name}"
        );
        max_parameter_difference = max_parameter_difference.max(difference);
    }
    history.frozen = Some(FrozenAudit {
        completed_updates,
        training_episodes: completed_updates * args.batch_size,
        captured_training: true,
        frozen_at_update: completed_updates,
        reader_fit_started_after_update: completed_updates,
        inner_rows: inner.target.size()[0] as usize,
        holdout_rows: holdout.target.size()[0] as usize,
        refit_rows: refit_target.size()[0] as usize,
        validation_pairs: heldout.rows() as usize / 2,
        parameter_tensors_checked: before.len(),
        parameter_scalars_checked: before.iter().map(|(_, value)| value.numel()).sum(),
        remaining_trainable_tensors: after.values().filter(|value| value.requires_grad()).count(),
        max_parameter_difference,
    });
    Ok(())
}

struct ArmHistory {
    label: String,
    points: Vec<(usize, &'static str, Measurement)>,
    training: Vec<(usize, f64, f64, f64)>,
    diagnostics: Vec<(usize, Vec<f64>)>,
    diagnostic_labels: Vec<&'static str>,
    seconds: f64,
    readers: Vec<Fit>,
    frozen: Option<FrozenAudit>,
}

fn write_reader_reports(
    args: &MemoryArgs,
    horizons: &[i64],
    histories: &[ArmHistory],
    title: &str,
    labels: &str,
) -> Result<()> {
    let horizon_axis: Vec<_> = horizons.iter().map(|&h| h as u64).collect();
    let mut penalties = Vec::new();
    let mut gains = Vec::new();
    let mut spectrum_values = Vec::new();
    let mut reader_labels = Vec::new();
    for arm in histories {
        for (name, reader) in FROZEN_READERS.iter().zip(&arm.readers) {
            let label = format!("{}/{name}", arm.label);
            let penalty = reader.penalty.to_device(Device::Cpu);
            penalties.push(series(
                &label,
                (0..horizons.len() as i64).map(|h| penalty.double_value(&[h])),
            ));
            let diagnostics = reader
                .diagnostics
                .as_ref()
                .expect("memory reader refit diagnostics");
            let fit = diagnostics.gains.to_device(Device::Cpu);
            for (row, split) in [
                "inner training fit",
                "training-only penalty selection",
                "all-training refit",
            ]
            .into_iter()
            .enumerate()
            {
                gains.push(series(
                    format!("{label} {split}"),
                    (0..horizons.len() as i64).map(|h| fit.double_value(&[row as i64, h])),
                ));
            }
            // Every horizon has exactly the same complete-row feature population.
            spectrum_values.push(diagnostics.spectrum.select(1, 0).to_device(Device::Cpu));
            reader_labels.push(label);
        }
    }
    lines(&args.output, "timexer_segment_jepa_memory_reader_penalty",
        format!("{title}; selection labels never include paired validation; refit all independent training episodes"),
        "selected ridge penalty / mean centered feature eigenvalue", "future bars", &horizon_axis, penalties)?;
    lines(&args.output, "timexer_segment_jepa_memory_reader_fit",
        format!("{title}; achieved training-only fraction of raw target second moment removed; not validation evidence"),
        "1 - training-only fitted MSE / target second moment", "future bars", &horizon_axis, gains)?;
    let reader_axis: Vec<_> = (0..reader_labels.len() as u64).collect();
    let spectrum = [
        "uncentered feature second-moment trace",
        "top uncentered eigenvalue",
        "uncentered participation rank",
        "mean-feature squared norm",
    ]
    .into_iter()
    .enumerate()
    .map(|(row, name)| {
        series(
            name,
            spectrum_values
                .iter()
                .map(|values| values.double_value(&[row as i64])),
        )
    })
    .collect();
    lines(
        &args.output,
        "timexer_segment_jepa_memory_reader_spectrum",
        format!(
            "{title}; train-only refit feature population; reader order: {}",
            reader_labels.join("; ")
        ),
        "labeled uncentered feature moments / effective rank",
        "reader index",
        &reader_axis,
        spectrum,
    )?;
    let arm_axis: Vec<_> = (0..histories.len() as u64).collect();
    let frozen = [
        "completed pretraining updates",
        "complete-store freeze update",
        "reader fitting starts after update",
        "parameter tensors checked",
        "parameter scalars checked",
        "remaining trainable parameter tensors",
        "max absolute parameter change across reader fitting and evaluation",
        "captured pretraining verified",
    ]
    .into_iter()
    .enumerate()
    .map(|(field, name)| {
        series(
            name,
            histories.iter().map(|arm| {
                let audit = arm.frozen.as_ref().expect("completed frozen-reader audit");
                match field {
                    0 => audit.completed_updates as f64,
                    1 => audit.frozen_at_update as f64,
                    2 => audit.reader_fit_started_after_update as f64,
                    3 => audit.parameter_tensors_checked as f64,
                    4 => audit.parameter_scalars_checked as f64,
                    5 => audit.remaining_trainable_tensors as f64,
                    6 => audit.max_parameter_difference,
                    _ => u8::from(audit.captured_training) as f64,
                }
            }),
        )
    })
    .collect();
    lines(&args.output, "timexer_segment_jepa_memory_frozen",
        format!("{title}; every parameter compared exactly with its pre-reader snapshot; arm order: {labels}"),
        "labeled ordering/count/equality evidence", "arm index", &arm_axis, frozen)
}

fn write_histories(args: &MemoryArgs, horizons: &[i64], histories: &[ArmHistory]) -> Result<()> {
    let Some(first) = histories.first() else {
        return Ok(());
    };
    let axis: Vec<_> = (0..args.steps)
        .filter_map(|i| {
            let step = i + 1;
            (step % args.eval_every == 0 || step == args.steps).then_some(step as u64)
        })
        .collect();
    let evaluation_axis = if args.unanchored_sigreg {
        vec![args.steps as u64]
    } else {
        axis.clone()
    };
    let recipe = if args.unanchored_sigreg {
        "FRESH UNANCHORED attached temporal JEPA+SIGReg only; forecast/reconstruction/decision losses absent, no target stop-gradient or gradient surgery; reader_norm=none, full coupling/no decimation, offsets=[1,2,4,8,12], prediction weight=1, SIGReg total=.09 (both=.045 per site); full store frozen AFTER fixed budget; full/recomputed-recent/local CUDA ridge readers fit afterward on independent train-only 2048 inner + 512 selection episodes, refit all2560; validation generated only after readers fitted; forecast head never evaluated"
    } else if args.reader_sigreg {
        "full-coupling forecast; reader_norm=none in all four arms; SIGReg total .09, both=.045 per site"
    } else {
        "decoupled forecast; reader_norm=rms"
    };
    let title = format!("CONTROLLED SYNTHETIC learned delayed cue, NOT market evidence; seed={} (ChaCha8 streams train={TRAIN_STREAM:x}, validation={VALIDATION_STREAM:x}); budget={} optimizer updates per objective/task, batch={}, validation={} independent pairs, same future innovations within pairs; CausalPatch D128 L2 heads2 FF256 context512 patch16, BF16 captured forward/backward, fused Adam lr={}, linear last60% warmdown to15%, {recipe}, no future calendar; old aux cue bars={}..{}, source={}, recent={} bars, event {}..{}; analytic mean cue*0.01*sin(pi*h/192), null mean=0, innovation sigma={NOISE_SIGMA}; endpoint fixed, no validation selection", args.seed, args.steps, args.batch_size, args.validation_pairs, args.learning_rate, CUE_START, CUE_START + PATCH - 1, SOURCE, RECENT_PATCHES * PATCH as i64, SOURCE + 1, CONTEXT - 1);
    for (base, unit, field) in [
        (
            "timexer_segment_jepa_memory_ratio",
            "future close MSE / paired persistence MSE",
            0,
        ),
        (
            "timexer_segment_jepa_memory_error",
            "future close log-return MSE",
            1,
        ),
        (
            "timexer_segment_jepa_memory_correlation",
            "signed pooled forecast/realization correlation",
            2,
        ),
        (
            "timexer_segment_jepa_memory_bayes_regret",
            "mean (forecast - Bayesian mean)^2 / pulse amplitude^2",
            3,
        ),
        (
            "timexer_segment_jepa_memory_pair_error",
            "mean (paired effect - correct Bayesian effect)^2 / amplitude^2",
            4,
        ),
        (
            "timexer_segment_jepa_memory_pair_effect",
            "paired forecast effect, log-return units",
            5,
        ),
        (
            "timexer_segment_jepa_memory_pair_sign",
            "correctly signed paired effects (null/zero effect undefined)",
            6,
        ),
    ] {
        let mut curves = Vec::new();
        for arm in histories {
            let readers: &[&str] = if args.unanchored_sigreg {
                &FROZEN_READERS
            } else {
                &[""]
            };
            for reader in readers {
                let label = if reader.is_empty() {
                    arm.label.clone()
                } else {
                    format!("{}/{reader}", arm.label)
                };
                for (h, horizon) in horizons.iter().enumerate() {
                    let values = evaluation_axis.iter().map(|&step| {
                        arm.points
                            .iter()
                            .find(|(s, name, _)| *s as u64 == step && name == reader)
                            .map_or(f64::NAN, |(_, _, m)| match field {
                                0 => m.ratio[h],
                                1 => m.error[h],
                                2 => m.correlation[h],
                                3 => m.bayes_regret[h],
                                4 => m.pair_error[h],
                                5 => m.pair_effect[h],
                                _ => m.pair_sign[h],
                            })
                    });
                    curves.push(series(format!("{label} h{horizon}"), values));
                    if (field == 0 || field == 5) && *reader == readers[0] {
                        curves.push(series(
                            format!("{} h{horizon} Bayesian reference", arm.label),
                            evaluation_axis.iter().map(|&step| {
                                arm.points
                                    .iter()
                                    .find(|(s, _, _)| *s as u64 == step)
                                    .map_or(f64::NAN, |(_, _, m)| {
                                        if field == 0 {
                                            m.bayes_ratio[h]
                                        } else {
                                            m.pair_expected[h]
                                        }
                                    })
                            }),
                        ));
                    }
                }
            }
        }
        lines(
            &args.output,
            base,
            title.clone(),
            unit,
            "optimizer step",
            &evaluation_axis,
            curves,
        )?;
    }
    let mut state = Vec::new();
    let mut input_equality = Vec::new();
    for arm in histories {
        for (name, field) in [
            ("full-state", 0),
            ("actual-recent-state", 1),
            ("observation", 2),
        ] {
            state.push(series(
                format!("{} {name}", arm.label),
                evaluation_axis.iter().map(|&step| {
                    arm.points
                        .iter()
                        .find(|(s, _, _)| *s as u64 == step)
                        .map_or(f64::NAN, |(_, _, m)| match field {
                            0 => m.state_distance,
                            1 => m.recent_distance,
                            _ => m.observation_distance,
                        })
                }),
            ));
        }
        for (field, name) in [
            "recent prices",
            "recent auxiliary",
            "recent validity",
            "recent market",
            "global anchor",
            "causal sigma",
            "causal range",
        ]
        .into_iter()
        .enumerate()
        {
            input_equality.push(series(
                format!("{} {name}", arm.label),
                evaluation_axis.iter().map(|&step| {
                    arm.points
                        .iter()
                        .find(|(s, _, _)| *s as u64 == step)
                        .map_or(f64::NAN, |(_, _, m)| m.input_max_difference[field])
                }),
            ));
        }
    }
    lines(&args.output, "timexer_segment_jepa_memory_state", format!("{title}; state sensitivity alone is NOT evidence of correct memory; consult Bayesian paired effect error"), "mean paired squared representation difference", "optimizer step", &evaluation_axis, state)?;
    lines(
        &args.output,
        "timexer_segment_jepa_memory_input_equality",
        title.clone(),
        "maximum absolute paired input/statistic difference (must be zero)",
        "optimizer step",
        &evaluation_axis,
        input_equality,
    )?;
    for (base, unit, field) in [
        (
            "timexer_segment_jepa_memory_objective",
            "interval mean total training objective",
            0,
        ),
        (
            "timexer_segment_jepa_memory_train_nll",
            "interval mean forecast NLL, nats",
            1,
        ),
        (
            "timexer_segment_jepa_memory_train_mse",
            "interval mean normalized forecast MSE",
            2,
        ),
    ] {
        if args.unanchored_sigreg && field != 0 {
            continue; // No forecast objective or random-head diagnostic exists in this protocol.
        }
        let mut curves: Vec<_> = histories
            .iter()
            .map(|arm| {
                series(
                    &arm.label,
                    axis.iter().map(|&step| {
                        arm.training
                            .iter()
                            .find(|(s, _, _, _)| *s as u64 == step)
                            .map_or(f64::NAN, |(_, total, nll, mse)| match field {
                                0 => *total,
                                1 => *nll,
                                _ => *mse,
                            })
                    }),
                )
            })
            .collect();
        if args.unanchored_sigreg {
            for arm in histories {
                for (i, label) in arm.diagnostic_labels.iter().enumerate() {
                    curves.push(series(
                        format!("{} {label}", arm.label),
                        axis.iter().map(|&step| {
                            arm.diagnostics
                                .iter()
                                .find(|(s, _)| *s as u64 == step)
                                .map_or(f64::NAN, |(_, values)| values[i])
                        }),
                    ));
                }
            }
        }
        lines(
            &args.output,
            base,
            title.clone(),
            if args.unanchored_sigreg {
                "interval mean objective and labeled latent/SIGReg diagnostics"
            } else {
                unit
            },
            "optimizer step",
            &axis,
            curves,
        )?;
    }
    let arm_axis: Vec<_> = (0..histories.len() as u64).collect();
    let mut populations: Vec<ReportSeries> = vec![
        series(
            "independent validation pairs",
            histories.iter().map(|_| args.validation_pairs as f64),
        ),
        series(
            "scored rows (paired, not independent)",
            histories.iter().map(|_| (args.validation_pairs * 2) as f64),
        ),
        series(
            "declared optimizer updates",
            histories.iter().map(|_| args.steps as f64),
        ),
        series(
            "measured optimizer updates",
            histories
                .iter()
                .map(|arm| arm.training.last().map_or(0., |(s, _, _, _)| *s as f64)),
        ),
    ];
    populations.push(series(
        "training episodes at declared budget",
        histories
            .iter()
            .map(|_| (args.steps * args.batch_size) as f64),
    ));
    if args.unanchored_sigreg {
        for (name, field) in [
            ("independent inner training reader-fit rows", 0),
            ("independent training-only penalty-selection rows", 1),
            ("all training-only refit rows", 2),
            ("scored frozen readers (shared representation)", 3),
            ("actual pretraining episodes", 4),
        ] {
            populations.push(series(
                name,
                histories.iter().map(|arm| {
                    arm.frozen.as_ref().map_or(f64::NAN, |audit| match field {
                        0 => audit.inner_rows as f64,
                        1 => audit.holdout_rows as f64,
                        2 => audit.refit_rows as f64,
                        3 => arm.readers.len() as f64,
                        _ => audit.training_episodes as f64,
                    })
                }),
            ));
        }
    }
    let labels = histories
        .iter()
        .map(|h| h.label.as_str())
        .collect::<Vec<_>>()
        .join("; ");
    lines(
        &args.output,
        "timexer_segment_jepa_memory_population",
        format!("{title}; arm order: {labels}; first arm={}", first.label),
        "count",
        "arm index",
        &arm_axis,
        populations,
    )?;
    if args.unanchored_sigreg {
        write_reader_reports(args, horizons, histories, &title, &labels)?;
    }
    lines(
        &args.output,
        "timexer_segment_jepa_memory_runtime",
        format!("{title}; includes generation, capture and fixed evaluation; arm order: {labels}"),
        "elapsed seconds",
        "arm index",
        &arm_axis,
        vec![series("arm elapsed", histories.iter().map(|h| h.seconds))],
    )
}

/// Main can expose this as a clap tuple subcommand containing MemoryArgs. This command must
/// be launched through mlq like every other local training workload. It never loads market
/// data and does not claim a representative market training budget.
pub fn run(args: MemoryArgs) -> Result<()> {
    ensure!(args.steps > CAPTURE_AFTER_STEPS && args.batch_size >= 2 && args.validation_pairs > 0 && args.validation_pairs <= 1024 && args.eval_every > 0,
        "memory task needs captured-step budget, batch>=2, 1..1024 validation pairs, and positive evaluation cadence");
    ensure!(
        args.learning_rate.is_finite() && args.learning_rate > 0.,
        "invalid memory learning rate"
    );
    ensure!(
        !args.jepa_modes.is_empty(),
        "memory task requires objective arms"
    );
    ensure!(
        !(args.unanchored_sigreg && args.reader_sigreg),
        "memory SIGReg protocols are mutually exclusive"
    );
    ensure!(
        !args.jepa_modes.contains(&JepaMode::Unanchored),
        "use --unanchored-sigreg for the frozen-reader representation-only protocol"
    );
    if args.unanchored_sigreg {
        ensure!(
            args.steps == 1024 && args.batch_size == 64,
            "the unanchored memory protocol requires the full 1024-update, batch64 budget"
        );
    }
    for (i, mode) in args.jepa_modes.iter().enumerate() {
        ensure!(
            !args.jepa_modes[..i].contains(mode),
            "duplicate memory objective {mode}"
        );
    }
    let mut arms: Vec<(String, ModelConfig)> = if args.unanchored_sigreg {
        [
            SigregPlacement::Off,
            SigregPlacement::Local,
            SigregPlacement::State,
            SigregPlacement::Both,
        ]
        .into_iter()
        .map(|placement| {
            let mut model = config(JepaMode::Unanchored);
            model.reader_norm = ReaderNorm::None;
            model.sigreg_placement = placement;
            model.scale_coupling = ScaleCoupling::Full;
            model.jepa.prediction_weight = 1.;
            model.jepa.sigreg_weight = 0.09;
            model.jepa.reconstruction_weight = 0.;
            (format!("unanchored-{placement}"), model)
        })
        .collect()
    } else if args.reader_sigreg {
        [
            ("reader-none", SigregPlacement::Off),
            ("reader-local", SigregPlacement::Local),
            ("reader-state", SigregPlacement::State),
            ("reader-both", SigregPlacement::Both),
        ]
        .into_iter()
        .map(|(name, placement)| {
            let mut model = config(JepaMode::Off);
            model.reader_norm = ReaderNorm::None;
            model.sigreg_placement = placement;
            model.scale_coupling = ScaleCoupling::Full;
            (name.to_owned(), model)
        })
        .collect()
    } else {
        args.jepa_modes
            .iter()
            .map(|&mode| (mode.to_string(), config(mode)))
            .collect()
    };
    for (_, configuration) in &mut arms {
        configuration.jepa.seed = args.seed;
        configuration.validate()?;
    }
    ensure!(!args.output.exists(), "memory output already exists");
    let device = crate::torch::single_ticker_timexer::runner::cuda_device()?;
    ensure!(
        crate::torch::cuda::graph::CudaGraph::is_available(),
        "memory diagnostic requires captured CUDA training"
    );
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    std::fs::create_dir_all(&args.output)?;
    let mut provenance = serde_json::json!({
        "schema": if args.unanchored_sigreg { "causal-delayed-cue-unanchored-memory-v1" } else { "causal-delayed-cue-memory-v1" },
        "args": args,
        "arms": arms,
        "executable_sha256": crate::torch::hashing::file_sha256(std::env::current_exe()?)?,
        "training_stream_xor": TRAIN_STREAM,
        "validation_stream_xor": VALIDATION_STREAM,
        "tasks": ["relevant-cue", "irrelevant-cue-null"],
        "interpretation": "controlled causal old-cue use under matched recent inputs; not market-performance evidence",
    });
    if args.unanchored_sigreg {
        provenance["unanchored_protocol"] = serde_json::json!({
            "initialization": "fresh-random-per-placement-and-task-no-warmstart",
            "argv": std::env::args().collect::<Vec<_>>(),
            "planned_independent_tasks": arms.len() * 2,
            "planned_updates_per_task": args.steps,
            "planned_pretraining_episodes_per_task": args.steps * args.batch_size,
            "parameter_seed": args.seed,
            "pretraining_seed": args.seed ^ TRAIN_STREAM,
            "sigreg_seed": args.seed,
            "reader_inner_seed": args.seed ^ READER_INNER_STREAM,
            "reader_holdout_seed": args.seed ^ READER_HOLDOUT_STREAM,
            "paired_validation_seed": args.seed ^ VALIDATION_STREAM,
            "reader_inner_stream_xor": READER_INNER_STREAM,
            "reader_holdout_stream_xor": READER_HOLDOUT_STREAM,
            "reader_inner_rows": READER_INNER_ROWS,
            "reader_holdout_rows": READER_HOLDOUT_ROWS,
            "reader_refit_rows": READER_INNER_ROWS + READER_HOLDOUT_ROWS,
            "reader_features": FROZEN_READERS,
            "reader_targets": "source-anchored future close log returns; source bar319; horizons16,32,64,128,192",
            "reader_selection": "existing CUDA ridge penalty grid; select only independent training holdout then refit inner+holdout",
            "ordered_stages": ["fresh initialization", "1024 attached JEPA+SIGReg-only updates",
                "destroy optimizer and captured graph", "freeze complete parameter store",
                "snapshot every parameter", "generate independent train-only reader panels under no_grad",
                "select and refit three frozen readers", "generate and score held-out matched pairs",
                "assert all parameter snapshots exactly unchanged"],
            "forecast_loss": false,
            "forecast_head_evaluated": false,
            "reconstruction_loss": false,
            "decision_loss": false,
            "target_stop_gradient": false,
            "gradient_surgery": false,
            "reader_norm": "none",
            "scale_coupling": "full",
            "horizon_decimation": "none",
            "future_calendar": false,
            "latent_offsets": [1, 2, 4, 8, 12],
            "prediction_weight": 1.0,
            "sigreg_total_weight": 0.09,
            "both_site_weight": 0.045,
            "paired_innovations": "identical innovations within each +/-cue pair; same validation RNG across placements and relevant/null tasks",
            "feature_equality_assertions": "exact recent prices,aux,validity,market,anchor and source sigma/range; recent/local paired feature MSE <= 1e-12",
            "endpoint_selection": "fixed completed budget; no intermediate downstream evaluation or validation selection",
        });
    }
    std::fs::write(
        args.output.join("memory-protocol.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    let mut histories = Vec::new();
    let (_, horizons) = probe_geometry(&config(JepaMode::Off))?;
    for (mode, configuration) in &arms {
        for relevant in [true, false] {
            let started = Instant::now();
            tch::manual_seed(args.seed as i64);
            tch::Cuda::manual_seed_all(args.seed);
            let mut store = nn::VarStore::new(device);
            let model = CausalPatchModel::new(&store.root(), configuration);
            let mut knobs = RecipeKnobs::reference(configuration.x0_lambdas);
            knobs.schedule = LrSchedule::new(args.steps, 0.60, 0.15)?;
            let mut engine =
                Engine::new(&store, args.learning_rate, knobs, true, OptimizerKind::Adam)?;
            let heldout = (!args.unanchored_sigreg).then(|| validation(&args, relevant, device));
            let (source, _) = probe_geometry(configuration)?;
            let label = format!(
                "{mode}/{}",
                if relevant {
                    "relevant-cue"
                } else {
                    "irrelevant-cue-null"
                }
            );
            let mut history = ArmHistory {
                label,
                points: Vec::new(),
                training: Vec::new(),
                diagnostics: Vec::new(),
                diagnostic_labels: if args.unanchored_sigreg {
                    super::jepa::diagnostic_labels(configuration)
                } else {
                    Vec::new()
                },
                readers: Vec::new(),
                frozen: None,
                seconds: 0.,
            };
            let mut rng = ChaCha8Rng::seed_from_u64(args.seed ^ TRAIN_STREAM);
            let mut interval = Tensor::zeros([3], (Kind::Float, device));
            let mut diagnostics = Tensor::zeros(
                [history.diagnostic_labels.len() as i64],
                (Kind::Float, device),
            );
            let mut interval_steps = 0usize;
            for step in 1..=args.steps {
                let batch = training_batch(&mut rng, args.batch_size, relevant, device);
                let losses = if step == CAPTURE_AFTER_STEPS + 1 {
                    engine.arm_step_graph(&model, &batch)?
                } else {
                    engine.step(&model, &batch)?
                };
                if args.unanchored_sigreg {
                    let values = losses.jepa.as_ref().expect("unanchored JEPA diagnostics");
                    ensure!(
                        values.size() == diagnostics.size(),
                        "unanchored diagnostic shape changed"
                    );
                    diagnostics += values;
                }
                interval += Tensor::stack(&[losses.objective, losses.nll, losses.mse], 0).detach();
                interval_steps += 1;
                if step % args.eval_every == 0 || step == args.steps {
                    let average = (&interval / interval_steps as f64).to_device(Device::Cpu);
                    ensure!(
                        if args.unanchored_sigreg {
                            average.double_value(&[0]).is_finite()
                                && average.double_value(&[1]).is_nan()
                                && average.double_value(&[2]).is_nan()
                        } else {
                            average.isfinite().all().int64_value(&[]) == 1
                        },
                        "nonfinite memory training objective at step {step}"
                    );
                    history.training.push((
                        step,
                        average.double_value(&[0]),
                        average.double_value(&[1]),
                        average.double_value(&[2]),
                    ));
                    if let Some(heldout) = &heldout {
                        history.points.push((
                            step,
                            "",
                            measure(&model, heldout, relevant, &horizons, source)?,
                        ));
                    }
                    if args.unanchored_sigreg {
                        let values = (&diagnostics / interval_steps as f64).to_device(Device::Cpu);
                        ensure!(
                            values.isfinite().all().int64_value(&[]) == 1,
                            "nonfinite unanchored memory diagnostic at step {step}"
                        );
                        history.diagnostics.push((
                            step,
                            (0..history.diagnostic_labels.len())
                                .map(|i| values.double_value(&[i as i64]))
                                .collect(),
                        ));
                        let _ = diagnostics.zero_();
                    }
                    let _ = interval.zero_();
                    interval_steps = 0;
                }
            }
            ensure!(
                engine.step_graph_captured(),
                "memory training did not remain captured"
            );
            if args.unanchored_sigreg {
                drop(engine);
                store.freeze();
                fit_frozen_readers(
                    &args,
                    &store,
                    &model,
                    relevant,
                    source,
                    &horizons,
                    &mut history,
                    device,
                )?;
            }
            history.seconds = started.elapsed().as_secs_f64();
            if args.unanchored_sigreg {
                let result = serde_json::json!({
                    "schema": "causal-delayed-cue-unanchored-memory-task-v1",
                    "task": history.label,
                    "configuration": configuration,
                    "objective_contract": configuration.jepa_contract(),
                    "protocol": "memory-protocol.json",
                    "actual": history.frozen,
                    "reader_order": FROZEN_READERS,
                    "elapsed_seconds": history.seconds,
                });
                std::fs::write(
                    args.output
                        .join(format!("{}.json", history.label.replace('/', "-"))),
                    serde_json::to_vec_pretty(&result)?,
                )?;
            }
            // Persist every completed arm; no parameters or score-dependent selection.
            histories.push(history);
            write_histories(&args, &horizons, &histories)?;
        }
    }
    if args.unanchored_sigreg {
        ensure!(
            histories.len() == 8,
            "unanchored memory protocol did not complete all eight tasks"
        );
        let completion = serde_json::json!({
            "schema": "causal-delayed-cue-unanchored-memory-completion-v1",
            "protocol": "memory-protocol.json",
            "completed_independent_tasks": histories.len(),
            "frozen_reader_count": histories.iter().map(|h| h.readers.len()).sum::<usize>(),
            "completed_pretraining_updates": histories.iter().filter_map(|h| h.frozen.as_ref()).map(|a| a.completed_updates).sum::<usize>(),
            "completed_pretraining_episodes": histories.iter().filter_map(|h| h.frozen.as_ref()).map(|a| a.training_episodes).sum::<usize>(),
            "tasks": histories.iter().map(|h| (&h.label, &h.frozen)).collect::<Vec<_>>(),
        });
        std::fs::write(
            args.output.join("memory-completion.json"),
            serde_json::to_vec_pretty(&completion)?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_intervention_has_identical_recent_prices_auxiliary_and_anchor() {
        let mut rng = ChaCha8Rng::seed_from_u64(71);
        let innovations = Innovations::draw(&mut rng);
        let positive = innovations.row(1., true);
        let negative = innovations.row(-1., true);
        let length = CONTEXT + HORIZON;
        let recent = SOURCE + 1 - RECENT_PATCHES as usize * PATCH;
        assert_eq!(
            &positive[recent * 4..(SOURCE + 1) * 4],
            &negative[recent * 4..(SOURCE + 1) * 4]
        );
        assert_eq!(
            &positive[length * 5 + recent * 2..length * 5 + (SOURCE + 1) * 2],
            &negative[length * 5 + recent * 2..length * 5 + (SOURCE + 1) * 2]
        );
        assert_eq!(positive[length * 8], negative[length * 8]);
        // Stronger than recent equality: every normalization input up to the source agrees.
        assert_eq!(&positive[..(SOURCE + 1) * 4], &negative[..(SOURCE + 1) * 4]);
        assert_ne!(
            positive[length * 5 + CUE_START * 2],
            negative[length * 5 + CUE_START * 2]
        );
        for h in [16, 32, 64, 128, 192] {
            let difference = positive[(SOURCE + h) * 4 + 3] - negative[(SOURCE + h) * 4 + 3];
            assert!((difference as f64 - 2. * pulse(h)).abs() < 1e-8);
        }
    }
    #[test]
    fn irrelevant_cue_changes_no_price_or_future_label() {
        let mut rng = ChaCha8Rng::seed_from_u64(913);
        let innovations = Innovations::draw(&mut rng);
        let positive = innovations.row(1., false);
        let negative = innovations.row(-1., false);
        let length = CONTEXT + HORIZON;
        assert_eq!(&positive[..length * 5], &negative[..length * 5]);
        assert_eq!(&positive[length * 7..], &negative[length * 7..]);
        assert_ne!(
            positive[length * 5 + CUE_START * 2],
            negative[length * 5 + CUE_START * 2]
        );
    }
    #[test]
    fn intervention_endpoint_cannot_leak_through_global_centering() {
        assert_eq!(SOURCE + HORIZON, CONTEXT - 1);
        assert_eq!(pulse(HORIZON), 0.);
        assert_eq!(pulse(HORIZON + 1), 0.);
        assert!(pulse(64) > 0.);
        assert!(CUE_START + PATCH < SOURCE + 1 - RECENT_PATCHES as usize * PATCH);
    }
}
