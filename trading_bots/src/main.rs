use clap::{Parser, Subcommand, ValueEnum};
use colored::{self, Colorize};
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use trading_bot_0::torch::model::ModelVariant;
use trading_bot_0::torch::planner::PlannerDataSplit;
use trading_bot_0::torch::train::PretrainArgs;
use trading_bot_0::{genetic, torch};

/// Symbols to paper/live trade when the operator names none.
///
/// Resolved when the `paper` subcommand actually runs, never while clap builds the command
/// tree: this reads the whole packed-bar corpus, which `--help` and every other subcommand
/// have no business doing. Picks the deepest corpus histories rather than whatever happens
/// to sort first alphabetically.
fn default_paper_symbols() -> Vec<String> {
    trading_bot_0::data::universe::deepest_symbols(torch::constants::TICKERS_COUNT as usize)
}

/// `--split-bounds <b0>,<b1>` in epoch millis.
///
/// Two instants rather than a duration or a date: the split is defined by exact wall-clock
/// millis in the corpus, and a run that pins them must reproduce the derived pair to the
/// millisecond or it is scoring different windows.
fn parse_split_bounds(raw: &str) -> Result<(i64, i64), String> {
    let (first, second) = raw
        .split_once(',')
        .ok_or_else(|| format!("expected `<train_val_ms>,<val_test_ms>`, got `{raw}`"))?;
    let parse = |text: &str| -> Result<i64, String> {
        text.trim()
            .parse::<i64>()
            .map_err(|err| format!("`{text}` is not an epoch-millis instant: {err}"))
    };
    let bounds = (parse(first)?, parse(second)?);
    if bounds.0 >= bounds.1 {
        return Err(format!(
            "split bounds must ascend: {} is not before {}",
            bounds.0, bounds.1
        ));
    }
    Ok(bounds)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum StreamingModelVariant {
    #[value(
        name = "uniform-stream",
        alias = "uniform-256-stream",
        alias = "uniform256-stream"
    )]
    UniformStream,
}

impl From<StreamingModelVariant> for ModelVariant {
    fn from(value: StreamingModelVariant) -> Self {
        match value {
            StreamingModelVariant::UniformStream => Self::UniformStream,
        }
    }
}

#[derive(Parser)]
#[command(name = "trading_bot")]
#[command(about = "Trading bot with PPO training and inference", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Genetic {
        #[arg(long, value_enum, default_value_t = genetic::GeneticFamily::TrendBreakout)]
        family: genetic::GeneticFamily,

        #[arg(long)]
        run: Option<String>,

        #[arg(long, default_value_t = 600)]
        generations: usize,

        #[arg(long, default_value_t = 192)]
        population: usize,

        #[arg(long, default_value_t = 0.25)]
        survivor_ratio: f64,

        #[arg(long, value_enum, default_value_t = genetic::TickerSet::Train)]
        train_tickers: genetic::TickerSet,

        #[arg(long, value_enum, default_value_t = genetic::TickerSet::Validation)]
        validation_tickers: genetic::TickerSet,

        #[arg(long, value_enum, default_value_t = genetic::TickerSet::Test)]
        test_tickers: genetic::TickerSet,

        #[arg(long, default_value_t = 5)]
        heavy_report_every: usize,

        #[arg(long, default_value_t = 7)]
        seed: u64,

        #[arg(long, default_value_t = false)]
        skip_additional_downloads: bool,

        #[arg(long, default_value_t = 1.0)]
        mutation_entropy: f64,
    },
    Train {
        #[arg(short, long)]
        weights: Option<String>,

        #[arg(long, value_enum, default_value_t = StreamingModelVariant::UniformStream)]
        model_size: StreamingModelVariant,

        #[arg(long)]
        run: Option<String>,

        /// Reproducible environment, action-sampling, and minibatch seed.
        #[arg(long, default_value_t = 20260811)]
        seed: u64,
    },
    /// Pretrain the discrete distributional bar world model on the local bar corpus.
    Pretrain {
        /// Initialize from an existing pretrain checkpoint. Weights only: training
        /// restarts at step zero with a fresh optimizer and schedule.
        #[arg(short, long)]
        weights: Option<String>,

        #[arg(long)]
        run: Option<String>,

        /// One epoch is one pass worth of BAR-TOKENS over the training split. It is not a
        /// guaranteed pass over every unique bar: each ramp stage walks its own anchor list
        /// from the start and the token budget splits unevenly across the ramp, so stage 0
        /// covers only ~27% of the corpus. `pretrain_stage_coverage` charts the truth.
        #[arg(long, default_value_t = 3)]
        epochs: usize,

        /// Override the corpus-derived step count. Diagnostic use only: it decouples
        /// the learning-rate and ramp schedules from the corpus size.
        #[arg(long)]
        steps: Option<usize>,

        /// Batch size at the first ramp stage; the later stages use 2x and 3x.
        #[arg(long, default_value_t = 24)]
        batch_size: usize,

        /// Seeds the TRAINING sampler, support fitting and the torch/CUDA RNGs. It does NOT
        /// move the pinned evaluation windows or the PIT draws — those are pinned by the
        /// campaign constant `EVAL_WINDOW_SEED` — so seed replicates measure training noise
        /// on an unchanged bench and every run stays paired on identical windows.
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        /// Bar resolution to train on, in seconds.
        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        /// Drop symbols with fewer bars than this. The default guarantees every
        /// symbol contributes at least one full-context window to each split, and is
        /// shared with `ingest --min-bars` so the universe and the split agree on
        /// which files exist.
        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Training-split bars drawn to fit the equal-mass bin supports.
        #[arg(long, default_value_t = 4_000_000)]
        support_samples: usize,

        /// Scoring rule for the next-bar log-likelihood: `smoothed`, `hard` or `density`.
        ///
        /// THE THREE MODES ARE NOT COMPARABLE IN ABSOLUTE NATS. They differ by additive
        /// constants that depend on the binning, so a `density` figure sits tens of nats
        /// below a `smoothed` one on the identical model. The mode is written into the
        /// checkpoint metadata, folded into the lineage hash, and `pretrain-compare`
        /// refuses to pair two runs that disagree.
        ///
        /// * `density` (default) — the proper log-likelihood of the MIXED law we observe:
        ///   a probability MASS on an atom, a DENSITY inside a continuous bin. No
        ///   unreachable floor, and up to discretization error no dependence on the bin
        ///   count, which is what makes `NUM_BAR_BINS` ablatable.
        /// * `hard` — one-hot cross entropy on the containing bin. Proper for the
        ///   discretized law and floor-free, but its scale moves with the bin count.
        /// * `smoothed` — the old Gaussian label smoothing at 0.75x the local bin width.
        ///   Proper for the SMOOTHED law rather than the observed one, and it imposes an
        ///   unreachable 4.6482 nats/bar floor. Kept only so the campaign's earlier runs
        ///   stay comparable.
        #[arg(long, default_value_t = trading_bot_0::torch::bar_dist::BarScoring::default())]
        scoring: trading_bot_0::torch::bar_dist::BarScoring,

        /// Recursive latent-dynamics rollout depth.
        #[arg(long, default_value_t = 4)]
        dyn_horizon: usize,

        /// Weight on the NextLat term.
        ///
        /// `dyn` is summed over the `BAR_MODEL_DIM`-wide feature axis and averaged over
        /// tokens, so it is commensurate with `nll` and this weight means what it looks
        /// like. It previously defaulted to `1.0`, which was harmless only while the term
        /// was mean-reduced and therefore inert; the 512x reduction fix made `1.0` measure
        /// 28 against `nll` 17 — 62% of the objective — and `nll` ROSE from 16.34 to 17.19
        /// over 4000 steps. `1e-2` keeps the term a latent-shaping regularizer rather than
        /// a competing objective, and every training line now prints each term's share of
        /// the total so a repeat cannot go unnoticed for 4000 steps.
        #[arg(long, default_value_t = 1e-2)]
        lambda_dyn: f64,

        #[arg(long, default_value_t = 1.0)]
        lambda_kl: f64,

        /// Held-out windows in each pinned evaluation set. Pinned by the campaign constant
        /// `EVAL_WINDOW_SEED`, so they are identical across runs, seeds and ablations.
        #[arg(long, default_value_t = 4096)]
        validation_windows: usize,

        /// Context of the across-run diagnostic evaluation. Promotion always
        /// evaluates at the full deployed context instead.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START)]
        diagnostic_context: i64,

        /// Pinned windows carried into the candle-rollout snapshot reports.
        #[arg(long, default_value_t = 8)]
        snapshot_windows: usize,

        /// Validate every N optimizer steps. Validation also always runs at every
        /// epoch boundary and at the end of the run.
        #[arg(long, default_value_t = 1000)]
        validate_every: usize,

        /// Write a step-tagged checkpoint every N optimizer steps (0 disables).
        #[arg(long, default_value_t = 0)]
        checkpoint_every: usize,

        /// Print a training line every N optimizer steps (0 disables).
        #[arg(long, default_value_t = 20)]
        log_every: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign
        /// pin `ingest::PINNED_SPLIT_BOUNDS`.
        ///
        /// Ingest appends continuously and the bounds are percentiles of the trading-time
        /// axis, so a derived boundary moves with the corpus — after the survivorship
        /// expansion it lands 26 days EARLIER, which drops universe-ranking sessions into
        /// validation and reopens the selection leak. Pinning is therefore the default, and
        /// the pin is checked against the instant the symbol universe was ranked as of.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus instead of the campaign
        /// pin. Diagnostic use only: such a run is comparable to nothing, and
        /// `pretrain-compare` refuses to pair it with anything else.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Explicit path to the bin supports, instead of the corpus default.
        #[arg(long)]
        supports: Option<String>,

        /// Reuse cached supports whose recorded provenance does not match this corpus.
        /// Without it a mismatch is a hard error. Freezing is right mid-campaign — the
        /// supports define the nll_bar scale — but it must be a stated decision, and it is
        /// written into the checkpoint metadata.
        #[arg(long, default_value_t = false)]
        freeze_supports: bool,

        /// Train only on symbols clearing this median dollar volume; 0 uses every file on
        /// disk. The split instants are derived before the filter, so both arms of the
        /// ablation are scored over the same wall-clock held-out window.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Compare two pretraining runs PAIRED on their identical pinned evaluation windows.
    ///
    /// Comparing two absolute levels has a minimum detectable effect of ~0.41 nats, because
    /// the validation split holds only ~4 non-overlapping time slots and the market-common
    /// regime term does not average down. Differencing the same windows takes that to
    /// 0.04-0.09. Each argument is the `pretrain_best.windows.json` written beside a
    /// promoted checkpoint.
    PretrainCompare {
        /// Baseline run's per-window vector.
        #[arg(long)]
        baseline: String,
        /// Candidate run's per-window vector.
        #[arg(long)]
        candidate: String,
    },
    TrainPlanner {
        #[arg(long, default_value = "weights/pretrain_best.ot")]
        world_model_weights: String,

        #[arg(long)]
        world_model_metadata: Option<String>,

        #[arg(long)]
        planner_weights: Option<String>,

        #[arg(long)]
        output: Option<String>,

        #[arg(long)]
        run: Option<String>,

        #[arg(long, default_value_t = 1_000)]
        updates: usize,

        #[arg(long, default_value_t = 100)]
        horizon: usize,

        #[arg(long, default_value_t = 100)]
        rollout_length: usize,

        #[arg(long, default_value_t = 128)]
        environments: usize,

        #[arg(long, default_value_t = 1280)]
        minibatch_size: usize,

        #[arg(long)]
        context_bars: Option<usize>,

        #[arg(long, value_delimiter = ',')]
        tickers: Option<Vec<String>>,

        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    InferPlanner {
        #[arg(long, default_value = "weights/pretrain_best.ot")]
        world_model_weights: String,

        #[arg(long)]
        world_model_metadata: Option<String>,

        #[arg(long, default_value = "weights/planner.ot")]
        planner_weights: String,

        #[arg(long, default_value_t = 10)]
        episodes: usize,

        #[arg(long)]
        horizon: Option<usize>,

        #[arg(long, default_value_t = 100)]
        rollout_length: usize,

        #[arg(long)]
        context_bars: Option<usize>,

        #[arg(long, value_delimiter = ',')]
        tickers: Option<Vec<String>>,

        #[arg(long, value_enum, default_value_t = PlannerDataSplit::Test)]
        split: PlannerDataSplit,

        #[arg(long)]
        run: Option<String>,
    },
    Infer {
        #[arg(short, long, default_value = "weights/ppo_ep1000.ot")]
        weights: String,

        #[arg(short, long, default_value_t = 10)]
        episodes: usize,

        #[arg(short, long, default_value_t = false)]
        deterministic: bool,

        #[arg(short, long, default_value_t = 1.0)]
        temperature: f64,

        #[arg(long, value_delimiter = ',')]
        tickers: Option<Vec<String>>,

        #[arg(short, long, default_value_t = true)]
        random_start: bool,

        #[arg(long, value_enum, default_value_t = ModelVariant::UniformStream)]
        model_size: ModelVariant,

        #[arg(long)]
        run: Option<String>,
    },
    Paper {
        #[arg(short, long, default_value = "weights/ppo_ep1000.ot")]
        weights: String,

        /// Dedicated IBKR paper account to trade. Existing positions are rejected.
        #[arg(long)]
        account: String,

        /// Defaults to the deepest corpus histories, resolved when the command runs.
        #[arg(short, long, value_delimiter = ',')]
        symbols: Vec<String>,

        #[arg(short, long, default_value_t = 5)]
        interval: u64,

        #[arg(short, long, default_value_t = 500)]
        max_steps: usize,

        #[arg(long, value_enum, default_value_t = ModelVariant::UniformStream)]
        model_size: ModelVariant,
    },
    /// Download the Polygon corpus: measure the universe, then pull aggregates into
    /// `long_data/bars`.
    Ingest {
        /// Liquidity floor for corpus membership, in median dollars traded per session.
        ///
        /// A floor rather than a rank cutoff: "the top 3000 names" is an arbitrary number, whereas
        /// a dollar-volume floor states the quality threshold below which five-minute bars stop
        /// carrying intra-bar structure.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::MIN_DOLLAR_VOLUME)]
        min_dollar_volume: f64,

        #[arg(long, default_value = "5min")]
        resolution: String,

        #[arg(long, default_value_t = 5)]
        years: u32,

        #[arg(long, default_value_t = trading_bot_0::data::polygon::DEFAULT_CONCURRENCY)]
        concurrency: usize,

        /// Re-measure liquidity from the vendor instead of reusing the cached ranking.
        #[arg(long)]
        refresh_universe: bool,

        /// RFC 3339 instant the ranking sessions must precede, so universe membership carries no
        /// held-out information. Defaults to the corpus's own `train | val` boundary.
        #[arg(long, value_parser = trading_bot_0::data::ingest::parse_train_end)]
        train_end: Option<chrono::DateTime<chrono::Utc>>,

        /// Bars a symbol needs to count toward that boundary. Must match `pretrain --min-bars`.
        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Measure and report the universe, then stop before downloading anything.
        #[arg(long)]
        universe_only: bool,

        #[arg(long)]
        force: bool,

        #[arg(long, default_value_t = false)]
        daily: bool,
    },
    /// Fill the pre-2021 regime gap: pull free deep daily history into `long_data/bars`.
    ///
    /// The Polygon plan is capped at a rolling five-year window, so the intraday corpus contains
    /// no 2000 and no 2008. This pulls decades of daily bars from a free source instead.
    DeepDaily {
        /// Liquidity floor the cached ranking is filtered by, matching `ingest`.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::MIN_DOLLAR_VOLUME)]
        min_dollar_volume: f64,

        /// Take only the first N symbols of that ranking; 0 takes all of them.
        #[arg(long, default_value_t = 0)]
        limit: usize,

        #[arg(long, default_value_t = trading_bot_0::data::deep_daily::DEFAULT_CONCURRENCY)]
        concurrency: usize,

        /// Rewrite symbols whose daily file already reaches past the Polygon floor.
        #[arg(long)]
        force: bool,

        /// Fetch and audit, but write nothing.
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() {
    // Must precede the runtime: `set_var` is only sound while the process is single-threaded.
    trading_bot_0::data::load_dotenv();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed building the tokio runtime")
        .block_on(run());
}

async fn run() {
    println!("{}", "Start".green());

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Genetic {
            family,
            run,
            generations,
            population,
            survivor_ratio,
            train_tickers,
            validation_tickers,
            test_tickers,
            heavy_report_every,
            seed,
            skip_additional_downloads,
            mutation_entropy,
        }) => {
            genetic::run(genetic::GeneticArgs {
                family: *family,
                run: run.clone(),
                generations: *generations,
                population: *population,
                survivor_ratio: *survivor_ratio,
                train_tickers: *train_tickers,
                validation_tickers: *validation_tickers,
                test_tickers: *test_tickers,
                heavy_report_every: *heavy_report_every,
                seed: *seed,
                skip_additional_downloads: *skip_additional_downloads,
                mutation_entropy: *mutation_entropy,
            })
            .expect("genetic training failed");
        }
        Some(Commands::Train {
            weights,
            model_size,
            run,
            seed,
        }) => {
            torch::train::train(weights.as_deref(), (*model_size).into(), run.clone(), *seed)
                .await
                .expect("PPO training failed");
        }
        Some(Commands::Pretrain {
            weights,
            run,
            epochs,
            steps,
            batch_size,
            seed,
            data_dir,
            resolution_secs,
            min_bars,
            support_samples,
            scoring,
            dyn_horizon,
            lambda_dyn,
            lambda_kl,
            validation_windows,
            diagnostic_context,
            snapshot_windows,
            validate_every,
            checkpoint_every,
            log_every,
            split_bounds,
            derive_split_bounds,
            supports,
            freeze_supports,
            min_dollar_volume,
        }) => {
            let args = PretrainArgs {
                weights: weights.clone(),
                run: run.clone(),
                epochs: *epochs,
                steps: *steps,
                batch_size: *batch_size,
                seed: *seed,
                data_dir: data_dir.clone(),
                resolution_secs: *resolution_secs,
                min_bars: *min_bars,
                support_samples: *support_samples,
                scoring: *scoring,
                dyn_horizon: *dyn_horizon,
                lambda_dyn: *lambda_dyn,
                lambda_kl: *lambda_kl,
                validation_windows: *validation_windows,
                diagnostic_context: *diagnostic_context,
                snapshot_windows: *snapshot_windows,
                validate_every: *validate_every,
                checkpoint_every: *checkpoint_every,
                log_every: *log_every,
                split_bounds: *split_bounds,
                derive_split_bounds: *derive_split_bounds,
                supports: supports.clone(),
                freeze_supports: *freeze_supports,
                min_dollar_volume: *min_dollar_volume,
            };
            tokio::task::spawn_blocking(move || torch::train::pretrain(args))
                .await
                .expect("pretraining task panicked")
                .expect("pretraining failed");
        }
        Some(Commands::PretrainCompare {
            baseline,
            candidate,
        }) => {
            let comparison = torch::train::pretrain_stats::compare_runs(
                std::path::Path::new(baseline),
                std::path::Path::new(candidate),
            )
            .expect("paired comparison failed");
            print!("{comparison}");
        }
        Some(Commands::TrainPlanner {
            world_model_weights,
            world_model_metadata,
            planner_weights,
            output,
            run,
            updates,
            horizon,
            rollout_length,
            environments,
            minibatch_size,
            context_bars,
            tickers,
            seed,
        }) => {
            let args = torch::planner::TrainPlannerArgs {
                world_model_weights: world_model_weights.clone(),
                world_model_metadata: world_model_metadata.clone(),
                planner_weights: planner_weights.clone(),
                output: output.clone().unwrap_or_default(),
                run: run.clone(),
                updates: *updates,
                horizon: *horizon,
                rollout_length: *rollout_length,
                environments: *environments,
                minibatch_size: *minibatch_size,
                context_bars: *context_bars,
                tickers: tickers.clone(),
                seed: *seed,
            };
            tokio::task::spawn_blocking(move || torch::planner::train_planner(args))
                .await
                .expect("planner training task panicked")
                .expect("planner training failed");
        }
        Some(Commands::InferPlanner {
            world_model_weights,
            world_model_metadata,
            planner_weights,
            episodes,
            horizon,
            rollout_length,
            context_bars,
            tickers,
            split,
            run,
        }) => {
            let destination = RunDir::create_fresh(RUNS_PATH, run.as_deref())
                .expect("failed to create planner inference run dir");
            let args = torch::planner::InferPlannerArgs {
                world_model_weights: world_model_weights.clone(),
                world_model_metadata: world_model_metadata.clone(),
                planner_weights: planner_weights.clone(),
                episodes: *episodes,
                horizon: *horizon,
                rollout_length: *rollout_length,
                context_bars: *context_bars,
                tickers: tickers.clone(),
                split: *split,
                report_root: Some(destination.root),
            };
            tokio::task::spawn_blocking(move || torch::planner::infer_planner(args))
                .await
                .expect("planner inference task panicked")
                .expect("planner inference failed");
        }
        Some(Commands::Infer {
            weights,
            episodes,
            deterministic,
            temperature,
            tickers,
            random_start,
            model_size,
            run,
        }) => {
            let run_dir = RunDir::create_fresh(RUNS_PATH, run.as_deref())
                .expect("failed to create inference run dir");
            let output_dir = run_dir.root.join("inference");
            torch::infer::run_inference(
                weights,
                *episodes,
                *deterministic,
                *temperature,
                tickers.clone(),
                *random_start,
                *model_size,
                output_dir,
            )
            .expect("inference failed");
        }
        Some(Commands::Paper {
            weights,
            account,
            symbols,
            interval,
            max_steps,
            model_size,
        }) => {
            let symbols = if symbols.is_empty() {
                default_paper_symbols()
            } else {
                symbols.clone()
            };
            torch::infer::run_ibkr_paper_trading(
                weights,
                account.clone(),
                symbols,
                *interval,
                *max_steps,
                *model_size,
            )
            .expect("paper trading failed");
        }
        Some(Commands::Ingest {
            min_dollar_volume,
            resolution,
            years,
            concurrency,
            refresh_universe,
            train_end,
            min_bars,
            universe_only,
            force,
            daily,
        }) => {
            trading_bot_0::data::ingest::run(trading_bot_0::data::ingest::IngestArgs {
                min_dollar_volume: *min_dollar_volume,
                resolution: resolution.clone(),
                years: *years,
                concurrency: *concurrency,
                refresh_universe: *refresh_universe,
                train_end: *train_end,
                min_bars: *min_bars,
                universe_only: *universe_only,
                force: *force,
                daily: *daily,
            })
            .await
            .expect("ingest failed");
        }
        Some(Commands::DeepDaily {
            min_dollar_volume,
            limit,
            concurrency,
            force,
            dry_run,
        }) => {
            trading_bot_0::data::deep_daily::run(trading_bot_0::data::deep_daily::DeepDailyArgs {
                min_dollar_volume: *min_dollar_volume,
                limit: *limit,
                concurrency: *concurrency,
                force: *force,
                dry_run: *dry_run,
            })
            .await
            .expect("deep daily ingest failed");
        }
        None => {
            torch::train::train(None, ModelVariant::UniformStream, None, 20260811)
                .await
                .expect("PPO training failed");
        }
    }

    println!("{}", "End".green())
}

#[cfg(test)]
mod tests {
    use super::{default_paper_symbols, Cli, Commands, StreamingModelVariant};
    use clap::Parser;
    use trading_bot_0::torch::model::ModelVariant;

    #[test]
    fn train_defaults_are_executable_contracts() {
        let train = Cli::try_parse_from(["trading_bot", "train"]).expect("train should parse");
        assert!(matches!(
            train.command,
            Some(Commands::Train {
                model_size: StreamingModelVariant::UniformStream,
                ..
            })
        ));
    }

    /// Pretrain's defaults are load-bearing: the seed makes runs reproducible, the
    /// batch size sizes every ramp stage, and the step count must stay corpus-derived.
    #[test]
    fn pretrain_defaults_are_executable_contracts() {
        let pretrain =
            Cli::try_parse_from(["trading_bot", "pretrain"]).expect("pretrain should parse");
        let Some(Commands::Pretrain {
            epochs,
            steps,
            batch_size,
            seed,
            resolution_secs,
            dyn_horizon,
            lambda_dyn,
            lambda_kl,
            validation_windows,
            diagnostic_context,
            data_dir,
            ..
        }) = pretrain.command
        else {
            panic!("pretrain subcommand should parse as Pretrain");
        };
        assert_eq!(epochs, 3);
        assert_eq!(steps, None);
        assert_eq!(batch_size, 24);
        assert_eq!(seed, 0x5EED);
        assert_eq!(resolution_secs, 300);
        assert_eq!(dyn_horizon, 4);
        // Deliberately `1 / BAR_MODEL_DIM`: the NextLat term is now summed over the
        // feature axis, and this default is what keeps its effective strength identical
        // to the old mean-reduced term rather than 512x larger.
        assert_eq!(
            lambda_dyn,
            1.0 / trading_bot_0::torch::world_model::BAR_MODEL_DIM as f64
        );
        assert_eq!(lambda_kl, 1.0);
        assert_eq!(validation_windows, 4096);
        assert_eq!(
            diagnostic_context,
            trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START
        );
        assert!(data_dir.ends_with("bars"), "{data_dir}");
    }

    #[test]
    fn streaming_only_commands_reject_unimplemented_model_families() {
        for model in ["base", "ablation-small"] {
            let Err(error) = Cli::try_parse_from(["trading_bot", "train", "--model-size", model])
            else {
                panic!("unsupported streaming model must fail during parsing");
            };
            let message = error.to_string();
            assert!(message.contains("invalid value"));
            assert!(message.contains("uniform-stream"));
        }
    }

    /// The planner loads whatever pretraining promoted; the two defaults must agree.
    #[test]
    fn planner_defaults_point_at_the_promoted_pretrain_checkpoint() {
        for command in ["train-planner", "infer-planner"] {
            let cli = Cli::try_parse_from(["trading_bot", command])
                .expect("planner subcommand should parse");
            let weights = match cli.command {
                Some(Commands::TrainPlanner {
                    world_model_weights,
                    ..
                })
                | Some(Commands::InferPlanner {
                    world_model_weights,
                    ..
                }) => world_model_weights,
                _ => panic!("unexpected subcommand"),
            };
            assert_eq!(weights, "weights/pretrain_best.ot");
        }
    }

    #[test]
    fn offline_inference_exposes_every_implemented_model_family() {
        for (name, expected) in [
            ("base", ModelVariant::Base),
            ("uniform-stream", ModelVariant::UniformStream),
            ("ablation-small", ModelVariant::AblationSmall),
        ] {
            let cli = Cli::try_parse_from(["trading_bot", "infer", "--model-size", name])
                .expect("implemented inference model should parse");
            let Some(Commands::Infer { model_size, .. }) = cli.command else {
                panic!("infer subcommand should parse as Infer");
            };
            assert_eq!(model_size, expected);
        }
    }

    #[test]
    fn paper_symbols_are_not_resolved_while_parsing_the_cli() {
        let cli = Cli::try_parse_from(["trading_bot", "paper", "--account", "DU123"])
            .expect("paper CLI should parse");
        let Some(Commands::Paper {
            symbols, interval, ..
        }) = cli.command
        else {
            panic!("paper subcommand should parse as Paper");
        };
        assert!(
            symbols.is_empty(),
            "the corpus must not be scanned to build the command tree"
        );
        assert_eq!(interval, 5);
    }

    #[test]
    fn paper_defaults_match_model_ticker_count() {
        assert_eq!(
            default_paper_symbols().len(),
            trading_bot_0::torch::constants::TICKERS_COUNT as usize
        );
    }

    #[test]
    fn paper_rejects_unsupported_temperature_sampling() {
        assert!(Cli::try_parse_from([
            "trading_bot",
            "paper",
            "--account",
            "DU123",
            "--temperature",
            "0.8",
        ])
        .is_err());
    }
}
