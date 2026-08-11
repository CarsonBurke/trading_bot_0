use clap::{Parser, Subcommand, ValueEnum};
use colored::{self, Colorize};
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use trading_bot_0::torch::model::ModelVariant;
use trading_bot_0::torch::planner::PlannerDataSplit;
use trading_bot_0::torch::train::PretrainObjective;
use trading_bot_0::{genetic, torch};

fn default_paper_symbols() -> Vec<String> {
    trading_bot_0::constants::TICKERS
        .iter()
        .take(torch::constants::TICKERS_COUNT as usize)
        .map(|symbol| (*symbol).to_string())
        .collect()
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
    Pretrain {
        #[arg(short, long)]
        weights: Option<String>,

        #[arg(long, value_enum, default_value_t = StreamingModelVariant::UniformStream)]
        model_size: StreamingModelVariant,

        #[arg(long)]
        run: Option<String>,

        #[arg(long, default_value_t = 1)]
        epochs: usize,

        #[arg(long)]
        steps: Option<usize>,

        /// With --steps 0 and LeJEPA weights, run only the lightweight latent-skill gate.
        #[arg(long, default_value_t = false)]
        eval_skill_only: bool,

        #[arg(long, default_value_t = 256)]
        batch_size: usize,

        #[arg(long, default_value_t = 16)]
        k_patches: usize,

        #[arg(long, value_enum, default_value_t = PretrainObjective::MeanMse)]
        objective: PretrainObjective,

        #[arg(long, default_value_t = 0.0)]
        lambda_lat: f64,

        #[arg(long, default_value_t = 0.09)]
        lambda_sigreg: f64,

        #[arg(long, default_value_t = 100.0)]
        target_scale: f64,

        #[arg(long, default_value_t = 0)]
        validation_batches: usize,

        /// Run validation every N global optimizer steps within an epoch (0 disables
        /// mid-epoch validation). Validation always also runs at each epoch end.
        #[arg(long, default_value_t = 0)]
        validate_every: usize,

        /// Write a checkpoint every N global optimizer steps within an epoch (0 disables
        /// mid-epoch checkpoints).
        #[arg(long, default_value_t = 0)]
        checkpoint_every: usize,

        /// Evaluate one validation mini-batch every N training steps, folded into
        /// the pretrain_step_loss report's val overlay (0 disables).
        #[arg(long, default_value_t = 5)]
        step_val_every: usize,

        /// Write deterministic candle-rollout snapshot reports on fixed validation
        /// windows every N training steps (0 disables).
        #[arg(long, default_value_t = 500)]
        candle_snapshot_every: usize,
    },
    TrainPlanner {
        #[arg(long, default_value = "weights/pretrain_heads_best.ot")]
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
        #[arg(long, default_value = "weights/pretrain_heads_best.ot")]
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

        #[arg(short, long, value_delimiter = ',', default_values_t = default_paper_symbols())]
        symbols: Vec<String>,

        #[arg(short, long, default_value_t = 5)]
        interval: u64,

        #[arg(short, long, default_value_t = 500)]
        max_steps: usize,

        #[arg(long, value_enum, default_value_t = ModelVariant::UniformStream)]
        model_size: ModelVariant,
    },
}

#[tokio::main]
async fn main() {
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
            model_size,
            run,
            epochs,
            steps,
            eval_skill_only,
            batch_size,
            k_patches,
            objective,
            lambda_lat,
            lambda_sigreg,
            target_scale,
            validation_batches,
            validate_every,
            checkpoint_every,
            step_val_every,
            candle_snapshot_every,
        }) => {
            let args = torch::train::PretrainArgs {
                weights: weights.clone(),
                model_size: (*model_size).into(),
                run: run.clone(),
                epochs: *epochs,
                steps: *steps,
                eval_skill_only: *eval_skill_only,
                batch_size: *batch_size,
                k_patches: *k_patches,
                objective: *objective,
                lambda_lat: *lambda_lat,
                lambda_sigreg: *lambda_sigreg,
                target_scale: *target_scale,
                validation_batches: *validation_batches,
                validate_every: *validate_every,
                checkpoint_every: *checkpoint_every,
                step_val_every: *step_val_every,
                candle_snapshot_every: *candle_snapshot_every,
            };
            tokio::task::spawn_blocking(move || torch::train::pretrain(args))
                .await
                .expect("pretraining task panicked")
                .expect("pretraining failed");
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
            torch::infer::run_ibkr_paper_trading(
                weights,
                account.clone(),
                symbols.clone(),
                *interval,
                *max_steps,
                *model_size,
            )
            .expect("paper trading failed");
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
    use super::{Cli, Commands, StreamingModelVariant};
    use clap::Parser;
    use trading_bot_0::torch::model::ModelVariant;

    #[test]
    fn train_and_pretrain_defaults_are_executable_contracts() {
        let train = Cli::try_parse_from(["trading_bot", "train"]).expect("train should parse");
        assert!(matches!(
            train.command,
            Some(Commands::Train {
                model_size: StreamingModelVariant::UniformStream,
                ..
            })
        ));

        let pretrain =
            Cli::try_parse_from(["trading_bot", "pretrain"]).expect("pretrain should parse");
        assert!(matches!(
            pretrain.command,
            Some(Commands::Pretrain {
                model_size: StreamingModelVariant::UniformStream,
                epochs: 1,
                steps: None,
                ..
            })
        ));
    }

    #[test]
    fn streaming_only_commands_reject_unimplemented_model_families() {
        for command in ["train", "pretrain"] {
            for model in ["base", "ablation-small"] {
                let Err(error) =
                    Cli::try_parse_from(["trading_bot", command, "--model-size", model])
                else {
                    panic!("unsupported streaming model must fail during parsing");
                };
                let message = error.to_string();
                assert!(message.contains("invalid value"));
                assert!(message.contains("uniform-stream"));
            }
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
    fn paper_defaults_match_model_ticker_count() {
        let cli = Cli::try_parse_from(["trading_bot", "paper", "--account", "DU123"])
            .expect("paper CLI should parse");
        let Some(Commands::Paper {
            symbols, interval, ..
        }) = cli.command
        else {
            panic!("paper subcommand should parse as Paper");
        };
        assert_eq!(
            symbols.len(),
            trading_bot_0::torch::constants::TICKERS_COUNT as usize
        );
        assert_eq!(interval, 5);
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
