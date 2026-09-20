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
    jepa::JepaMode,
    jepa_eval::{future_targets, lines, probe_geometry, score, series},
    model::{CausalPatchModel, ModelConfig, ScaleCoupling},
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
        .to_device(mean.device())
        .repeat([batch.rows() / 2])
        .unsqueeze(-1);
    let bayes = signs
        * Tensor::from_slice(&bayes_row)
            .to_device(mean.device())
            .unsqueeze(0);
    let predicted_scores = score(&prediction, &target, &mask);
    let bayes_scores = score(&bayes, &target, &mask);
    let regret = (&prediction - &bayes)
        .square()
        .mean_dim([0i64].as_slice(), false, Kind::Float)
        / PULSE_AMPLITUDE.powi(2);
    let delta = paired_delta(&prediction);
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
    let recent = model.representation_state_at_recent(batch, source, RECENT_PATCHES);
    let distance = |x: &Tensor| paired_delta(x).square().mean(Kind::Float).double_value(&[]);
    let recent_distance = distance(&recent);
    let observation_distance = distance(&views.observation.select(1, source));
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
    ensure!(recent_distance <= 1e-12 && observation_distance <= 1e-12,
        "synthetic cue leaked into a supposedly identical recent/observation input: recent={recent_distance:e}, observation={observation_distance:e}");
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
        state_distance: distance(&views.state.select(1, source)),
        recent_distance,
        observation_distance,
        input_max_difference,
    })
}

struct ArmHistory {
    label: String,
    points: Vec<(usize, Measurement)>,
    training: Vec<(usize, f64, f64, f64)>,
    seconds: f64,
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
    let title = format!("CONTROLLED SYNTHETIC learned delayed cue, NOT market evidence; seed={} (ChaCha8 streams train={TRAIN_STREAM:x}, validation={VALIDATION_STREAM:x}); budget={} optimizer updates per objective/task, batch={}, validation={} independent pairs, same future innovations within pairs; CausalPatch D128 L2 heads2 FF256 context512 patch16, BF16 captured forward/backward, fused Adam lr={}, linear last60% warmdown to15%, decoupled forecast, no future calendar; old aux cue bars={}..{}, source={}, recent={} bars, event {}..{}; analytic mean cue*0.01*sin(pi*h/192), null mean=0, innovation sigma={NOISE_SIGMA}; endpoint fixed, no validation selection", args.seed, args.steps, args.batch_size, args.validation_pairs, args.learning_rate, CUE_START, CUE_START + PATCH - 1, SOURCE, RECENT_PATCHES * PATCH as i64, SOURCE + 1, CONTEXT - 1);
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
            for (h, horizon) in horizons.iter().enumerate() {
                let values = axis.iter().map(|&step| {
                    arm.points
                        .iter()
                        .find(|(s, _)| *s as u64 == step)
                        .map_or(f64::NAN, |(_, m)| match field {
                            0 => m.ratio[h],
                            1 => m.error[h],
                            2 => m.correlation[h],
                            3 => m.bayes_regret[h],
                            4 => m.pair_error[h],
                            5 => m.pair_effect[h],
                            _ => m.pair_sign[h],
                        })
                });
                curves.push(series(format!("{} h{horizon}", arm.label), values));
                if field == 0 || field == 5 {
                    curves.push(series(
                        format!("{} h{horizon} Bayesian reference", arm.label),
                        axis.iter().map(|&step| {
                            arm.points.iter().find(|(s, _)| *s as u64 == step).map_or(
                                f64::NAN,
                                |(_, m)| {
                                    if field == 0 {
                                        m.bayes_ratio[h]
                                    } else {
                                        m.pair_expected[h]
                                    }
                                },
                            )
                        }),
                    ));
                }
            }
        }
        lines(
            &args.output,
            base,
            title.clone(),
            unit,
            "optimizer step",
            &axis,
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
                axis.iter().map(|&step| {
                    arm.points
                        .iter()
                        .find(|(s, _)| *s as u64 == step)
                        .map_or(f64::NAN, |(_, m)| match field {
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
                axis.iter().map(|&step| {
                    arm.points
                        .iter()
                        .find(|(s, _)| *s as u64 == step)
                        .map_or(f64::NAN, |(_, m)| m.input_max_difference[field])
                }),
            ));
        }
    }
    lines(&args.output, "timexer_segment_jepa_memory_state", format!("{title}; state sensitivity alone is NOT evidence of correct memory; consult Bayesian paired effect error"), "mean paired squared representation difference", "optimizer step", &axis, state)?;
    lines(
        &args.output,
        "timexer_segment_jepa_memory_input_equality",
        title.clone(),
        "maximum absolute paired input/statistic difference (must be zero)",
        "optimizer step",
        &axis,
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
        let curves = histories
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
        lines(
            &args.output,
            base,
            title.clone(),
            unit,
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
                .map(|arm| arm.points.last().map_or(0., |(s, _)| *s as f64)),
        ),
    ];
    populations.push(series(
        "training episodes at declared budget",
        histories
            .iter()
            .map(|_| (args.steps * args.batch_size) as f64),
    ));
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
    for (i, mode) in args.jepa_modes.iter().enumerate() {
        ensure!(
            !args.jepa_modes[..i].contains(mode),
            "duplicate memory objective {mode}"
        );
    }
    ensure!(!args.output.exists(), "memory output already exists");
    let device = crate::torch::single_ticker_timexer::runner::cuda_device()?;
    ensure!(
        crate::torch::cuda::graph::CudaGraph::is_available(),
        "memory diagnostic requires captured CUDA training"
    );
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    std::fs::create_dir_all(&args.output)?;
    let mut histories = Vec::new();
    let (_, horizons) = probe_geometry(&config(JepaMode::Off))?;
    for &mode in &args.jepa_modes {
        for relevant in [true, false] {
            let started = Instant::now();
            tch::manual_seed(args.seed as i64);
            tch::Cuda::manual_seed_all(args.seed);
            let store = nn::VarStore::new(device);
            let mut configuration = config(mode);
            configuration.jepa.seed = args.seed;
            configuration.validate()?;
            let model = CausalPatchModel::new(&store.root(), &configuration);
            let mut knobs = RecipeKnobs::reference(configuration.x0_lambdas);
            knobs.schedule = LrSchedule::new(args.steps, 0.60, 0.15)?;
            let mut engine =
                Engine::new(&store, args.learning_rate, knobs, true, OptimizerKind::Adam)?;
            let heldout = validation(&args, relevant, device);
            let (source, _) = probe_geometry(&configuration)?;
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
                seconds: 0.,
            };
            let mut rng = ChaCha8Rng::seed_from_u64(args.seed ^ TRAIN_STREAM);
            let mut interval = Tensor::zeros([3], (Kind::Float, device));
            let mut interval_steps = 0usize;
            for step in 1..=args.steps {
                let batch = training_batch(&mut rng, args.batch_size, relevant, device);
                let losses = if step == CAPTURE_AFTER_STEPS + 1 {
                    engine.arm_step_graph(&model, &batch)?
                } else {
                    engine.step(&model, &batch)?
                };
                interval += Tensor::stack(&[losses.objective, losses.nll, losses.mse], 0).detach();
                interval_steps += 1;
                if step % args.eval_every == 0 || step == args.steps {
                    let average = (&interval / interval_steps as f64).to_device(Device::Cpu);
                    ensure!(
                        average.isfinite().all().int64_value(&[]) == 1,
                        "nonfinite memory training objective at step {step}"
                    );
                    history.training.push((
                        step,
                        average.double_value(&[0]),
                        average.double_value(&[1]),
                        average.double_value(&[2]),
                    ));
                    history.points.push((
                        step,
                        measure(&model, &heldout, relevant, &horizons, source)?,
                    ));
                    let _ = interval.zero_();
                    interval_steps = 0;
                }
            }
            ensure!(
                engine.step_graph_captured(),
                "memory training did not remain captured"
            );
            history.seconds = started.elapsed().as_secs_f64();
            // Persist every completed arm; no parameters or score-dependent selection.
            histories.push(history);
            write_histories(&args, &horizons, &histories)?;
        }
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
