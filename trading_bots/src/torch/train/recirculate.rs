//! Frozen-checkpoint evaluation of fixed sequential residual recirculation.

use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use shared::report::{
    write_report, Report, ReportKind, ReportSeries, ScaleKind, PRETRAIN_REPORT_BASES,
};
use tch::Device;

use crate::torch::bar_dist::BarScoring;
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::dataset::Split;
use crate::torch::world_model::{BarWorldModel, RecirculationConfig};

use super::pretrain::{
    evaluate_with_trunk, load_corpus, pinned_blocks, CorpusFlags, EvaluationTrunk, PinnedSet,
    EVAL_WINDOW_SEED,
};
use super::pretrain_stats::{block_bootstrap, Dispersion, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED};

pub const RECIRCULATION_REPORT_BASE: &str = "pretrain_recirculation_sweep";
const PAIR_SCREEN_WINDOWS: usize = 8;
const SHORT_EVAL_TOKEN_BATCH_BUDGET: u64 = 50_000;
const SERIAL_DRIFT_TOLERANCE_NATS: f64 = 1e-4;
const MIN_CONFIRMATION_GAIN_NATS: f64 = 0.05;
const MIN_EFFECTIVE_RANK_RATIO: f64 = 0.5;
const RAMP_POSITIONS: usize = 10;
const PAPER_SCALED_PAIRS: [(usize, usize); 7] =
    [(4, 1), (4, 2), (5, 2), (5, 3), (6, 2), (6, 3), (7, 3)];
const ALPHAS: [f64; 3] = [0.05, 0.10, 0.15];

#[derive(Clone, Debug)]
pub struct RecirculateArgs {
    pub weights: String,
    pub metadata: String,
    pub output: String,
    pub screen_windows: usize,
    /// Total validation draw. Stage B is the disjoint range `screen_windows..confirmation_windows`.
    pub confirmation_windows: usize,
    pub context: i64,
    pub batch_size: usize,
    pub allow_long_eval: bool,
    pub corpus: CorpusFlags,
}

#[derive(Clone, Debug)]
struct Measurement {
    window_nll: Vec<f64>,
    dir_acc: f64,
    effective_rank: f64,
}

#[derive(Clone, Copy, Debug)]
struct Comparison {
    nll: Dispersion,
    windows: usize,
    dir_delta: f64,
    rank_ratio: f64,
}

#[derive(Clone, Debug)]
struct CandidateResult {
    index: usize,
    config: RecirculationConfig,
    stage: &'static str,
    screen: Comparison,
    confirmation: Option<Comparison>,
    test: Option<Comparison>,
    early_rejected: bool,
    selected: bool,
    passed: bool,
}

pub fn pretrain_recirculate(args: RecirculateArgs) -> Result<()> {
    validate_args(&args)?;

    configure_cuda();
    let device = Device::cuda_if_available();
    let corpus = load_corpus(&args.corpus)?;
    let world = load_frozen_checkpoint(&args, &corpus, device)?;
    let mut validation =
        PinnedSet::pinned(&corpus, Split::Val, args.context, args.confirmation_windows)?;
    ensure!(
        validation.windows.len() >= args.confirmation_windows,
        "validation supplied only {} pinned windows, fewer than the requested {}",
        validation.windows.len(),
        args.confirmation_windows
    );

    let pair_range = 0..PAIR_SCREEN_WINDOWS;
    let tune_range = PAIR_SCREEN_WINDOWS..args.screen_windows;
    let confirmation_range = args.screen_windows..args.confirmation_windows;
    let pair_blocks = blocks_for_range(&mut validation, pair_range.clone())?;
    let pair_baseline = measure_range(
        &world,
        &mut validation,
        pair_range.clone(),
        args.batch_size,
        device,
        EvaluationTrunk::Serialized,
    )?;
    let parallel_pair = measure_range(
        &world,
        &mut validation,
        pair_range.clone(),
        args.batch_size,
        device,
        EvaluationTrunk::Parallel,
    )?;
    let serial_drift = compare(&parallel_pair, &pair_baseline, &pair_blocks);
    let serial_drift_pass = serial_drift.nll.mean.abs() <= SERIAL_DRIFT_TOLERANCE_NATS;

    let mut results = Vec::with_capacity(PAPER_SCALED_PAIRS.len() + ALPHAS.len() * 2);
    for config in pair_grid()? {
        let measured = measure_range(
            &world,
            &mut validation,
            pair_range.clone(),
            args.batch_size,
            device,
            EvaluationTrunk::Recirculated(&config),
        )?;
        let screen = compare(&pair_baseline, &measured, &pair_blocks);
        let early_rejected =
            !measurement_is_valid(&measured, &pair_baseline) || screen.nll.mean > 0.0;
        results.push(CandidateResult {
            index: results.len(),
            stage: "pair",
            config,
            screen,
            confirmation: None,
            test: None,
            early_rejected,
            selected: false,
            passed: false,
        });
    }

    let Some(pair_index) = best_screen_index(&results, "pair") else {
        write_sweep_report(
            Path::new(&args.output),
            &results,
            serial_drift,
            serial_drift_pass,
        )?;
        return Ok(());
    };
    let source = results[pair_index].config.source_layer();
    let destination = results[pair_index].config.destination_layer();

    let tune_baseline = measure_range(
        &world,
        &mut validation,
        tune_range.clone(),
        args.batch_size,
        device,
        EvaluationTrunk::Serialized,
    )?;
    let tune_blocks = blocks_for_range(&mut validation, tune_range.clone())?;
    for config in tuning_grid(source, destination)? {
        let measured = measure_range(
            &world,
            &mut validation,
            tune_range.clone(),
            args.batch_size,
            device,
            EvaluationTrunk::Recirculated(&config),
        )?;
        let screen = compare(&tune_baseline, &measured, &tune_blocks);
        let early_rejected =
            !measurement_is_valid(&measured, &tune_baseline) || screen.nll.mean > 0.0;
        results.push(CandidateResult {
            index: results.len(),
            stage: "mix",
            config,
            screen,
            confirmation: None,
            test: None,
            early_rejected,
            selected: false,
            passed: false,
        });
    }

    let Some(selected_index) = best_screen_index(&results, "mix") else {
        write_sweep_report(
            Path::new(&args.output),
            &results,
            serial_drift,
            serial_drift_pass,
        )?;
        return Ok(());
    };
    results[selected_index].selected = true;

    let confirmation_baseline = measure_range(
        &world,
        &mut validation,
        confirmation_range.clone(),
        args.batch_size,
        device,
        EvaluationTrunk::Serialized,
    )?;
    let confirmation_blocks = blocks_for_range(&mut validation, confirmation_range.clone())?;
    let confirmation_measured = measure_range(
        &world,
        &mut validation,
        confirmation_range.clone(),
        args.batch_size,
        device,
        EvaluationTrunk::Recirculated(&results[selected_index].config),
    )?;
    let confirmation = compare(
        &confirmation_baseline,
        &confirmation_measured,
        &confirmation_blocks,
    );
    results[selected_index].confirmation = Some(confirmation);
    results[selected_index].passed = serial_drift_pass
        && confirmation.nll.mean <= -MIN_CONFIRMATION_GAIN_NATS
        && confirmation.nll.ci_high < 0.0
        && confirmation.dir_delta >= 0.0
        && confirmation.rank_ratio >= MIN_EFFECTIVE_RANK_RATIO;

    if results[selected_index].passed {
        let mut test = PinnedSet::pinned(
            &corpus,
            Split::Test,
            args.context,
            args.confirmation_windows,
        )?;
        let test_range = 0..args.confirmation_windows;
        let baseline = measure_range(
            &world,
            &mut test,
            test_range.clone(),
            args.batch_size,
            device,
            EvaluationTrunk::Serialized,
        )?;
        let candidate = measure_range(
            &world,
            &mut test,
            test_range.clone(),
            args.batch_size,
            device,
            EvaluationTrunk::Recirculated(&results[selected_index].config),
        )?;
        let blocks = blocks_for_range(&mut test, test_range)?;
        results[selected_index].test = Some(compare(&baseline, &candidate, &blocks));
    }

    write_sweep_report(
        Path::new(&args.output),
        &results,
        serial_drift,
        serial_drift_pass,
    )?;
    Ok(())
}

fn validate_args(args: &RecirculateArgs) -> Result<()> {
    ensure!(
        args.screen_windows > PAIR_SCREEN_WINDOWS,
        "--screen-windows must exceed the {PAIR_SCREEN_WINDOWS}-window pair screen"
    );
    ensure!(
        args.confirmation_windows > args.screen_windows,
        "--confirmation-windows must exceed --screen-windows so confirmation is non-empty and disjoint"
    );
    ensure!(args.context > 1, "--context must exceed one bar");
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    super::eval_budget::enforce(
        "pretrain-recirculate",
        "sequential token-batches",
        estimated_token_batches(args),
        SHORT_EVAL_TOKEN_BATCH_BUDGET,
        args.allow_long_eval,
    )
}

fn load_frozen_checkpoint(
    args: &RecirculateArgs,
    corpus: &crate::torch::dataset::BarCorpus,
    device: Device,
) -> Result<BarWorldModel> {
    let weights = Path::new(&args.weights);
    let metadata = Path::new(&args.metadata);
    let world = BarWorldModel::load(weights, metadata, device)?;
    ensure!(
        world.metadata().res_secs == args.corpus.resolution_secs,
        "checkpoint resolution {}s does not match --resolution-secs {}",
        world.metadata().res_secs,
        args.corpus.resolution_secs
    );
    let trained = world
        .metadata()
        .training
        .as_ref()
        .context("checkpoint has no training provenance; fixed recirculation requires the pinned evaluation contract")?;
    ensure!(
        trained.eval_window_seed == EVAL_WINDOW_SEED,
        "checkpoint used eval_window_seed {:#x}, but this build pins {EVAL_WINDOW_SEED:#x}",
        trained.eval_window_seed
    );
    let scoring: BarScoring = trained
        .scoring
        .parse()
        .map_err(|reason| anyhow!("checkpoint scoring contract cannot be parsed: {reason}"))?;
    ensure!(
        scoring == BarScoring::Hard,
        "recirculation headline requires Hard categorical NLL, but checkpoint records `{scoring}`"
    );
    let fingerprint = corpus.identity_fingerprint();
    ensure!(
        trained.corpus_fingerprint == fingerprint,
        "checkpoint corpus {} does not match loaded corpus {}",
        &trained.corpus_fingerprint[..12.min(trained.corpus_fingerprint.len())],
        &fingerprint[..12.min(fingerprint.len())]
    );
    ensure!(
        trained.split_bounds == corpus.split_bounds(),
        "checkpoint split bounds {:?} do not match loaded corpus {:?}",
        trained.split_bounds,
        corpus.split_bounds()
    );
    Ok(world)
}

fn pair_grid() -> Result<Vec<RecirculationConfig>> {
    PAPER_SCALED_PAIRS
        .into_iter()
        .map(|(source, destination)| {
            RecirculationConfig::new(source, destination, 0.10, false, RAMP_POSITIONS)
        })
        .collect()
}

fn tuning_grid(source: usize, destination: usize) -> Result<Vec<RecirculationConfig>> {
    ALPHAS
        .into_iter()
        .flat_map(|alpha| {
            [false, true]
                .into_iter()
                .map(move |beta_one| (alpha, beta_one))
        })
        .map(|(alpha, beta_one)| {
            RecirculationConfig::new(source, destination, alpha, beta_one, RAMP_POSITIONS)
        })
        .collect()
}

fn best_screen_index(results: &[CandidateResult], stage: &str) -> Option<usize> {
    results
        .iter()
        .filter(|result| {
            result.stage == stage
                && !result.early_rejected
                && result.screen.nll.mean.is_finite()
                && result.screen.nll.mean <= 0.0
        })
        .min_by(|left, right| left.screen.nll.mean.total_cmp(&right.screen.nll.mean))
        .map(|result| result.index)
}

fn estimated_token_batches(args: &RecirculateArgs) -> u64 {
    let batches = |windows: usize| windows.div_ceil(args.batch_size) as u64;
    let context = args.context as u64;
    let pair = batches(PAIR_SCREEN_WINDOWS) * (2 + PAPER_SCALED_PAIRS.len() as u64);
    let tune_windows = args.screen_windows - PAIR_SCREEN_WINDOWS;
    let tune = batches(tune_windows) * (1 + (ALPHAS.len() * 2) as u64);
    let confirmation = batches(args.confirmation_windows - args.screen_windows) * 2;
    let test = batches(args.confirmation_windows) * 2;
    context * (pair + tune + confirmation + test)
}

fn measure_range(
    world: &BarWorldModel,
    set: &mut PinnedSet,
    range: Range<usize>,
    batch_size: usize,
    device: Device,
    trunk: EvaluationTrunk<'_>,
) -> Result<Measurement> {
    ensure!(
        range.start < range.end && range.end <= set.windows.len(),
        "evaluation range {:?} is outside {} pinned windows",
        range,
        set.windows.len()
    );
    let all_windows = std::mem::take(&mut set.windows);
    set.windows = all_windows[range].to_vec();
    let result = evaluate_with_trunk(
        world.modules(),
        world.deployment_supports(),
        set,
        batch_size,
        device,
        BarScoring::Hard,
        trunk,
    );
    set.windows = all_windows;
    let stats = result?;
    Ok(Measurement {
        window_nll: stats.window_nll,
        dir_acc: stats.dir_acc,
        effective_rank: stats.effective_rank,
    })
}

fn blocks_for_range(set: &mut PinnedSet, range: Range<usize>) -> Result<Vec<u64>> {
    ensure!(
        range.start < range.end && range.end <= set.windows.len(),
        "bootstrap range {:?} is outside {} pinned windows",
        range,
        set.windows.len()
    );
    let all_windows = std::mem::take(&mut set.windows);
    set.windows = all_windows[range].to_vec();
    let blocks = pinned_blocks(set);
    set.windows = all_windows;
    Ok(blocks)
}

fn compare(baseline: &Measurement, candidate: &Measurement, blocks: &[u64]) -> Comparison {
    assert_eq!(baseline.window_nll.len(), candidate.window_nll.len());
    assert_eq!(candidate.window_nll.len(), blocks.len());
    let difference: Vec<f64> = candidate
        .window_nll
        .iter()
        .zip(&baseline.window_nll)
        .map(|(candidate, baseline)| candidate - baseline)
        .collect();
    Comparison {
        nll: block_bootstrap(&difference, blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        windows: difference.len(),
        dir_delta: candidate.dir_acc - baseline.dir_acc,
        rank_ratio: candidate.effective_rank / baseline.effective_rank,
    }
}

fn measurement_is_valid(candidate: &Measurement, baseline: &Measurement) -> bool {
    candidate.window_nll.iter().all(|value| value.is_finite())
        && candidate.dir_acc.is_finite()
        && candidate.effective_rank.is_finite()
        && baseline.effective_rank.is_finite()
        && candidate.effective_rank / baseline.effective_rank >= MIN_EFFECTIVE_RANK_RATIO
}

fn comparison_slots(comparison: Option<Comparison>) -> [f32; 7] {
    let Some(comparison) = comparison else {
        return [f32::NAN; 7];
    };
    [
        comparison.nll.mean as f32,
        comparison.nll.se as f32,
        comparison.nll.ci_low as f32,
        comparison.nll.ci_high as f32,
        comparison.windows as f32,
        comparison.dir_delta as f32,
        comparison.rank_ratio as f32,
    ]
}

fn bool_flag(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}
fn write_sweep_report(
    output: &Path,

    results: &[CandidateResult],
    serial_drift: Comparison,
    serial_drift_pass: bool,
) -> Result<()> {
    ensure!(
        PRETRAIN_REPORT_BASES.contains(&RECIRCULATION_REPORT_BASE),
        "{RECIRCULATION_REPORT_BASE} must be registered before it can be written"
    );
    std::fs::create_dir_all(output)
        .with_context(|| format!("failed to create {}", output.display()))?;
    let mut series = Vec::with_capacity(results.len() + 1);
    let drift = comparison_slots(Some(serial_drift));
    series.push(ReportSeries {
        label: format!(
            "serialized-minus-parallel alpha0 control tolerance={SERIAL_DRIFT_TOLERANCE_NATS:.1e} pass={serial_drift_pass}"
        ),
        values: drift.into_iter().chain([bool_flag(serial_drift_pass)]).collect(),
    });
    for result in results {
        let mut values = vec![
            result.index as f32,
            if result.stage == "pair" { 0.0 } else { 1.0 },
            result.config.source_layer() as f32,
            result.config.destination_layer() as f32,
            result.config.alpha() as f32,
            bool_flag(result.config.beta_one()),
            result.config.ramp_positions() as f32,
        ];
        values.extend(comparison_slots(Some(result.screen)));
        values.extend(comparison_slots(result.confirmation));
        values.extend(comparison_slots(result.test));
        values.extend([
            bool_flag(result.early_rejected),
            bool_flag(result.selected),
            bool_flag(result.passed),
        ]);
        series.push(ReportSeries {
            label: format!(
                "{} cfg{:02} source={} dest={} alpha={:.2} beta={} l2 ramp10",
                result.stage,
                result.index,
                result.config.source_layer(),
                result.config.destination_layer(),
                result.config.alpha(),
                if result.config.beta_one() {
                    "1"
                } else {
                    "1-alpha_t"
                },
            ),
            values,
        });
    }
    let report = Report {
        title: concat!(
            "Fixed Recirculation Hierarchy | candidate slots 0:index 1:stage(pair=0,mix=1) ",
            "2:source 3:dest 4:alpha 5:beta_one 6:ramp; then screen/confirmation/test each: ",
            "delta,se,ci_low,ci_high,windows,dir_delta,rank_ratio; final: rejected,selected,gate_pass. ",
            "Control series starts at delta."
        )
        .to_owned(),
        x_label: Some("encoded metric slot".to_owned()),
        y_label: Some("value (nats/bar for NLL slots)".to_owned()),
        scale: ScaleKind::Linear,
        kind: ReportKind::MultiLine { series },
    };
    write_report(
        output.join(format!("{RECIRCULATION_REPORT_BASE}.report.bin")),
        &report,
    )
    .context("failed writing fixed recirculation report")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{read_report, ReportKind};

    #[test]
    fn hierarchical_grid_limits_work_without_dropping_paper_pairs_or_mixes() {
        let pairs = pair_grid().expect("valid pair grid");
        assert_eq!(pairs.len(), PAPER_SCALED_PAIRS.len());
        assert!(pairs.iter().all(|config| {
            config.destination_layer() < config.source_layer()
                && config.alpha() == 0.10
                && !config.beta_one()
                && config.ramp_positions() == RAMP_POSITIONS
        }));
        let mixes = tuning_grid(6, 2).expect("valid mix grid");
        assert_eq!(mixes.len(), ALPHAS.len() * 2);
        assert!(mixes
            .iter()
            .all(|config| config.source_layer() == 6 && config.destination_layer() == 2));
    }

    fn budget_args(screen_windows: usize, confirmation_windows: usize) -> RecirculateArgs {
        RecirculateArgs {
            weights: String::new(),
            metadata: String::new(),
            output: String::new(),
            screen_windows,
            confirmation_windows,
            context: 896,
            batch_size: 8,
            allow_long_eval: false,
            corpus: CorpusFlags {
                data_dir: String::new(),
                resolution_secs: 300,
                min_bars: 1,
                split_bounds: None,
                derive_split_bounds: false,
                min_dollar_volume: 0.0,
                dof_scaling: crate::torch::bar_dist::DofScaling::Raw,
            },
        }
    }

    #[test]
    fn recirculation_defaults_fit_the_fail_closed_short_eval_budget() {
        let args = budget_args(24, 64);
        assert_eq!(estimated_token_batches(&args), 43_904);
        validate_args(&args).expect("short hierarchical defaults must fit");

        let old_grid = budget_args(512, 4096);
        let error = validate_args(&old_grid).expect_err("old multi-hour defaults must fail closed");
        assert!(error.to_string().contains("--allow-long-eval"));
    }

    #[test]
    fn recirculation_report_is_registered_and_readable() {
        assert!(PRETRAIN_REPORT_BASES.contains(&RECIRCULATION_REPORT_BASE));
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "recirculation_report_{}_{}",
            std::process::id(),
            unique
        ));
        let drift = Comparison {
            nll: Dispersion {
                mean: 0.0,
                se: 0.0,
                ci_low: 0.0,
                ci_high: 0.0,
                blocks: 4,
                samples: 256,
            },
            windows: 256,
            dir_delta: 0.0,
            rank_ratio: 1.0,
        };
        write_sweep_report(&dir, &[], drift, true).expect("report writes");
        let report = read_report(dir.join(format!("{RECIRCULATION_REPORT_BASE}.report.bin")))
            .expect("report reads");
        assert!(matches!(report.kind, ReportKind::MultiLine { .. }));
        std::fs::remove_dir_all(dir).ok();
    }
}
