use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::{paths::RUNS_PATH, run_dir::RunDir};

use crate::data::{
    historical::{get_historical_bars_result, set_ibkr_download_enabled},
    universe::minimum_history_bars,
};

use super::{
    backtest::{evaluate_family, MarketDataset},
    families,
    family::{GeneticFamily, StrategyFamilySpec},
    logging::{write_split_trace, GaHistory, SessionLogger},
    metrics::{population_stats, BacktestMetrics},
    tickers::TickerSet,
    GeneticArgs,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub family: GeneticFamily,
    pub generations: usize,
    pub population: usize,
    pub survivor_ratio: f64,
    pub heavy_report_every: usize,
    pub seed: u64,
    pub mutation_entropy: f64,
}

#[derive(Clone, Debug)]
pub struct SessionPaths {
    pub root: PathBuf,
    pub gens: PathBuf,
    pub weights: PathBuf,
    pub log_file: PathBuf,
}

#[derive(Clone, Debug)]
pub struct DatasetBundle {
    pub train: MarketDataset,
    pub validation: MarketDataset,
    pub test: MarketDataset,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub best_validation_generation: usize,
    pub best_validation_score: f64,
    pub best_test_metrics: BacktestMetrics,
    pub run_root: PathBuf,
}

#[derive(Clone, Serialize, Deserialize)]
struct SavedCheckpoint<G> {
    family: GeneticFamily,
    generation: usize,
    train_metrics: BacktestMetrics,
    validation_metrics: BacktestMetrics,
    genome: G,
}

#[derive(Clone)]
struct Candidate<G> {
    genome: G,
    train_metrics: BacktestMetrics,
}

pub fn run(args: GeneticArgs) -> Result<()> {
    set_ibkr_download_enabled(!args.skip_additional_downloads);
    let run_dir = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    let datasets = load_datasets(
        args.train_tickers,
        args.validation_tickers,
        args.test_tickers,
        args.skip_additional_downloads,
    )?;
    let config = TrainingConfig {
        family: args.family,
        generations: args.generations,
        population: args.population,
        survivor_ratio: args.survivor_ratio,
        heavy_report_every: args.heavy_report_every,
        seed: args.seed,
        mutation_entropy: args.mutation_entropy,
    };
    let session_paths = SessionPaths {
        root: run_dir.root,
        gens: run_dir.gens,
        weights: run_dir.weights,
        log_file: run_dir.log_file,
    };

    match args.family {
        GeneticFamily::PriceRebound => {
            run_family_with_markets(
                &families::price_rebound::Family,
                config,
                datasets,
                session_paths,
            )?;
        }
        GeneticFamily::RsiRebound => {
            run_family_with_markets(
                &families::rsi_rebound::Family,
                config,
                datasets,
                session_paths,
            )?;
        }
        GeneticFamily::TrendBreakout => {
            run_family_with_markets(
                &families::trend_breakout::Family,
                config,
                datasets,
                session_paths,
            )?;
        }
    }

    Ok(())
}

pub fn run_family_with_markets<F: StrategyFamilySpec>(
    family: &F,
    config: TrainingConfig,
    datasets: DatasetBundle,
    session_paths: SessionPaths,
) -> Result<TrainingSummary> {
    if config.population < 2 {
        bail!("population must be at least 2");
    }
    if !(0.0..1.0).contains(&config.survivor_ratio) {
        bail!("survivor_ratio must be in 0..1");
    }
    if config.mutation_entropy <= 0.0 {
        bail!("mutation_entropy must be > 0");
    }

    fs::create_dir_all(&session_paths.gens)?;
    fs::create_dir_all(&session_paths.weights)?;

    let mut logger = SessionLogger::new(&session_paths.root, &session_paths.log_file)?;
    logger.log_line(&format!(
        "starting genetic run family={:?} generations={} population={} survivor_ratio={:.3} mutation_entropy={:.3} seed={} train={:?} validation={:?} test={:?}",
        config.family,
        config.generations,
        config.population,
        config.survivor_ratio,
        config.mutation_entropy,
        config.seed,
        datasets.train.tickers,
        datasets.validation.tickers,
        datasets.test.tickers
    ))?;

    let mut history = GaHistory::default();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut population: Vec<F::Genome> = (0..config.population)
        .map(|_| family.seed_genome(&mut rng))
        .collect();

    let mut best_validation_score = f64::NEG_INFINITY;
    let mut best_validation_generation = 0usize;
    let mut best_validation_genome = population[0].clone();
    let mut best_validation_train_metrics = BacktestMetrics::default();
    let mut best_validation_metrics = BacktestMetrics::default();
    let survivor_count = survivor_count(config.population, config.survivor_ratio);
    let mut generations_since_validation_improvement = 0usize;

    for generation in 0..config.generations {
        let population_seed = rng.random::<u64>();
        let candidates = evaluate_population(family, &population, &datasets.train, population_seed);
        let mut ranked = candidates;
        ranked.sort_unstable_by(|left, right| {
            right
                .train_metrics
                .score
                .partial_cmp(&left.train_metrics.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let population_scores = ranked
            .iter()
            .map(|candidate| candidate.train_metrics.score)
            .collect::<Vec<_>>();
        let train_population_stats = population_stats(population_scores);
        let champion = ranked
            .first()
            .context("population evaluation returned no candidates")?;
        let validation_outcome =
            evaluate_family(family, &champion.genome, &datasets.validation, false);

        if validation_outcome.metrics.score > best_validation_score {
            best_validation_score = validation_outcome.metrics.score;
            best_validation_generation = generation;
            best_validation_genome = champion.genome.clone();
            best_validation_train_metrics = champion.train_metrics.clone();
            best_validation_metrics = validation_outcome.metrics.clone();
            generations_since_validation_improvement = 0;
            persist_checkpoint(
                family,
                &session_paths.weights,
                generation,
                &champion.genome,
                &champion.train_metrics,
                &validation_outcome.metrics,
            )?;
        }

        let adaptive_mutation_entropy = adaptive_mutation_entropy(
            config.mutation_entropy,
            &train_population_stats,
            generations_since_validation_improvement,
        );
        history.record_generation(
            &champion.train_metrics,
            &validation_outcome.metrics,
            best_validation_score,
            train_population_stats.clone(),
            adaptive_mutation_entropy,
        );
        let generation_dir = session_paths.gens.join(generation.to_string());
        history.write_reports(&generation_dir)?;

        logger.log_line(&format!(
            "gen={} train_score={:.3} valid_score={:.3} train_return={:.2}% valid_return={:.2}% best_valid={:.3} survivor_count={} mutation_entropy={:.3} plateau={}",
            generation,
            champion.train_metrics.score,
            validation_outcome.metrics.score,
            champion.train_metrics.return_pct,
            validation_outcome.metrics.return_pct,
            best_validation_score,
            survivor_count,
            adaptive_mutation_entropy,
            generations_since_validation_improvement,
        ))?;

        let write_heavy = generation == 0
            || generation + 1 == config.generations
            || generation % config.heavy_report_every.max(1) == 0
            || generation == best_validation_generation;
        if write_heavy {
            let (train_trace, valid_trace) = rayon::join(
                || evaluate_family(family, &champion.genome, &datasets.train, true).trace,
                || evaluate_family(family, &champion.genome, &datasets.validation, true).trace,
            );
            if let Some(trace) = train_trace.as_ref() {
                write_split_trace(&generation_dir, "train", &datasets.train.tickers, trace)?;
            }
            if let Some(trace) = valid_trace.as_ref() {
                write_split_trace(
                    &generation_dir,
                    "validation",
                    &datasets.validation.tickers,
                    trace,
                )?;
            }
        }

        if generation + 1 == config.generations {
            break;
        }

        population = evolve_population(
            family,
            &ranked,
            &mut rng,
            config.population,
            survivor_count,
            adaptive_mutation_entropy,
        );
        generations_since_validation_improvement += 1;
    }

    let final_test = evaluate_family(family, &best_validation_genome, &datasets.test, true);
    let final_generation_dir = session_paths
        .gens
        .join(config.generations.saturating_sub(1).to_string());
    if let Some(trace) = final_test.trace.as_ref() {
        write_split_trace(&final_generation_dir, "test", &datasets.test.tickers, trace)?;
    }
    write_final_metadata(
        &session_paths.root,
        &best_validation_genome,
        family,
        &best_validation_train_metrics,
        &best_validation_metrics,
        &final_test.metrics,
        best_validation_generation,
    )?;
    logger.log_line(&format!(
        "completed family={:?} best_validation_generation={} test_score={:.3} test_return={:.2}% outperformance={:.2}%",
        config.family,
        best_validation_generation,
        final_test.metrics.score,
        final_test.metrics.return_pct,
        final_test.metrics.outperformance_pct
    ))?;

    Ok(TrainingSummary {
        best_validation_generation,
        best_validation_score,
        best_test_metrics: final_test.metrics,
        run_root: session_paths.root,
    })
}

fn load_datasets(
    train_set: TickerSet,
    validation_set: TickerSet,
    test_set: TickerSet,
    cache_only: bool,
) -> Result<DatasetBundle> {
    let train = load_market(train_set, cache_only)?;
    let validation = load_market(validation_set, cache_only)?;
    let test = load_market(test_set, cache_only)?;
    Ok(DatasetBundle {
        train,
        validation,
        test,
    })
}

fn load_market(set: TickerSet, cache_only: bool) -> Result<MarketDataset> {
    let split_label = set.label();
    let min_bars = minimum_history_bars();
    let source_tickers = if cache_only {
        set.cached_eligible_tickers(min_bars)
    } else {
        set.tickers()
    };
    let mut tickers = Vec::new();
    let mut bars = Vec::new();

    for ticker in source_tickers {
        match get_historical_bars_result(&ticker).with_context(|| {
            format!(
                "failed loading {} historical data for {}",
                split_label, ticker
            )
        })? {
            Some(ticker_bars) if ticker_bars.len() >= min_bars => {
                tickers.push(ticker);
                bars.push(ticker_bars);
            }
            Some(ticker_bars) => {
                eprintln!(
                    "Skipping {} from {}: only {} bars, need at least {}",
                    ticker,
                    split_label,
                    ticker_bars.len(),
                    min_bars
                );
            }
            None => {
                eprintln!(
                    "Skipping {} from {}: downloaded data was empty or unusable",
                    ticker, split_label
                );
            }
        }
    }

    if bars.is_empty() {
        bail!("no historical bars found for {}", split_label);
    }
    let bars = align_bars_to_common_trailing_window(bars)
        .with_context(|| format!("failed to align historical bars for {}", split_label))?;
    Ok(MarketDataset::new(split_label.to_string(), tickers, bars))
}

fn align_bars_to_common_trailing_window(
    bars: Vec<Vec<ibapi::market_data::historical::Bar>>,
) -> Result<Vec<Vec<ibapi::market_data::historical::Bar>>> {
    let common_len = bars
        .iter()
        .map(|ticker_bars| ticker_bars.len())
        .min()
        .unwrap_or(0);

    if common_len == 0 {
        bail!("encountered an empty ticker series");
    }

    Ok(bars
        .into_iter()
        .map(|ticker_bars| ticker_bars[ticker_bars.len() - common_len..].to_vec())
        .collect())
}

fn evaluate_population<F: StrategyFamilySpec>(
    family: &F,
    population: &[F::Genome],
    dataset: &MarketDataset,
    seed: u64,
) -> Vec<Candidate<F::Genome>> {
    population
        .par_iter()
        .enumerate()
        .map(|(index, genome)| {
            let _local_seed = seed ^ index as u64;
            let outcome = evaluate_family(family, genome, dataset, false);
            Candidate {
                genome: genome.clone(),
                train_metrics: outcome.metrics,
            }
        })
        .collect()
}

fn evolve_population<F: StrategyFamilySpec>(
    family: &F,
    ranked: &[Candidate<F::Genome>],
    rng: &mut StdRng,
    population_size: usize,
    survivors: usize,
    mutation_entropy: f64,
) -> Vec<F::Genome> {
    let survivors = &ranked[..survivors];
    let elite_keep = (survivors.len() / 8).max(2).min(survivors.len());
    let mut next = Vec::with_capacity(population_size);
    next.extend(
        survivors
            .iter()
            .take(elite_keep)
            .map(|candidate| candidate.genome.clone()),
    );

    while next.len() < population_size {
        if next.len() % 13 == 0 {
            next.push(family.seed_genome(rng));
            continue;
        }
        let left = tournament_pick(survivors, rng);
        let right = tournament_pick(survivors, rng);
        let mut child = family.crossover(&left.genome, &right.genome, rng);
        family.mutate(&mut child, rng, mutation_entropy);
        next.push(child);
    }

    next
}

fn survivor_count(population: usize, survivor_ratio: f64) -> usize {
    ((population as f64 * survivor_ratio).ceil() as usize).clamp(2, population.saturating_sub(1))
}

fn adaptive_mutation_entropy(
    base_entropy: f64,
    population: &super::metrics::PopulationStats,
    plateau_generations: usize,
) -> f64 {
    let normalized_spread = population.spread.abs() / population.p50.abs().max(10.0);
    let diversity_boost = if normalized_spread < 0.08 {
        1.35
    } else if normalized_spread < 0.16 {
        1.15
    } else {
        1.0
    };
    let plateau_boost = 1.0 + (plateau_generations.min(12) as f64 * 0.035);
    (base_entropy * diversity_boost * plateau_boost).clamp(0.25, 4.0)
}

fn tournament_pick<'a, G>(population: &'a [Candidate<G>], rng: &mut StdRng) -> &'a Candidate<G> {
    let rounds = population.len().min(4).max(1);
    let mut best = &population[rng.random_range(0..population.len())];
    for _ in 1..rounds {
        let candidate = &population[rng.random_range(0..population.len())];
        if candidate.train_metrics.score > best.train_metrics.score {
            best = candidate;
        }
    }
    best
}

fn persist_checkpoint<F: StrategyFamilySpec>(
    family: &F,
    weights_dir: &Path,
    generation: usize,
    genome: &F::Genome,
    train_metrics: &BacktestMetrics,
    validation_metrics: &BacktestMetrics,
) -> Result<()> {
    let checkpoint = SavedCheckpoint {
        family: family.kind(),
        generation,
        train_metrics: train_metrics.clone(),
        validation_metrics: validation_metrics.clone(),
        genome: genome.clone(),
    };
    fs::write(
        weights_dir.join("ga_best_validation.json"),
        serde_json::to_vec_pretty(&checkpoint)?,
    )?;
    fs::write(
        weights_dir.join("ga_best_validation.txt"),
        family.describe(genome),
    )?;
    Ok(())
}

fn write_final_metadata<F: StrategyFamilySpec>(
    run_root: &Path,
    genome: &F::Genome,
    family: &F,
    train_metrics: &BacktestMetrics,
    validation_metrics: &BacktestMetrics,
    test_metrics: &BacktestMetrics,
    generation: usize,
) -> Result<()> {
    #[derive(Serialize)]
    struct FinalSummary<'a, G> {
        family: GeneticFamily,
        best_validation_generation: usize,
        train_metrics: &'a BacktestMetrics,
        validation_metrics: &'a BacktestMetrics,
        test_metrics: &'a BacktestMetrics,
        genome: &'a G,
    }

    let summary = FinalSummary {
        family: family.kind(),
        best_validation_generation: generation,
        train_metrics,
        validation_metrics,
        test_metrics,
        genome,
    };

    fs::write(
        run_root.join("ga_summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}
