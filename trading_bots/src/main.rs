use clap::{Parser, Subcommand};
use colored::{self, Colorize};
use trading_bot_0::torch::model::ModelVariant;
use trading_bot_0::torch::train::PretrainObjective;
use trading_bot_0::{genetic, torch};

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

        #[arg(long, value_enum, default_value_t = ModelVariant::UniformStream)]
        model_size: ModelVariant,

        #[arg(long)]
        run: Option<String>,
    },
    Pretrain {
        #[arg(short, long)]
        weights: Option<String>,

        #[arg(long, value_enum, default_value_t = ModelVariant::UniformStream)]
        model_size: ModelVariant,

        #[arg(long)]
        run: Option<String>,

        #[arg(long, default_value_t = 4)]
        epochs: usize,

        #[arg(long)]
        steps: Option<usize>,

        #[arg(long, default_value_t = 256)]
        batch_size: usize,

        #[arg(long, default_value_t = 16)]
        k_patches: usize,

        #[arg(long, default_value_t = 512)]
        batches_per_epoch: usize,

        #[arg(long, value_enum, default_value_t = PretrainObjective::MeanMse)]
        objective: PretrainObjective,

        #[arg(long, default_value_t = 0.0)]
        lambda_lat: f64,

        #[arg(long, default_value_t = 0.09)]
        lambda_sigreg: f64,

        #[arg(long, default_value_t = 2)]
        probe_epochs: usize,

        #[arg(long, default_value_t = 100.0)]
        target_scale: f64,

        #[arg(long, default_value_t = 0)]
        validation_batches: usize,

        #[arg(long, default_value_t = 0)]
        validate_every: usize,

        #[arg(long, default_value_t = 0)]
        checkpoint_every: usize,

        #[arg(long, default_value_t = false)]
        log_step_losses: bool,
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
    },
    Paper {
        #[arg(short, long, default_value = "weights/ppo_ep1000.ot")]
        weights: String,

        #[arg(short, long, value_delimiter = ',', default_value = "TSLA,AAPL")]
        symbols: Vec<String>,

        #[arg(short, long, default_value_t = 60)]
        interval: u64,

        #[arg(short, long, default_value_t = 500)]
        max_steps: usize,

        #[arg(short, long, default_value_t = 0.8)]
        temperature: f64,

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
        }) => {
            torch::train::train(weights.as_deref(), *model_size, run.clone()).await;
        }
        Some(Commands::Pretrain {
            weights,
            model_size,
            run,
            epochs,
            steps,
            batch_size,
            k_patches,
            batches_per_epoch,
            objective,
            lambda_lat,
            lambda_sigreg,
            probe_epochs,
            target_scale,
            validation_batches,
            validate_every,
            checkpoint_every,
            log_step_losses,
        }) => {
            let args = torch::train::PretrainArgs {
                weights: weights.clone(),
                model_size: *model_size,
                run: run.clone(),
                epochs: *epochs,
                steps: *steps,
                batch_size: *batch_size,
                k_patches: *k_patches,
                batches_per_epoch: *batches_per_epoch,
                objective: *objective,
                lambda_lat: *lambda_lat,
                lambda_sigreg: *lambda_sigreg,
                probe_epochs: *probe_epochs,
                target_scale: *target_scale,
                validation_batches: *validation_batches,
                validate_every: *validate_every,
                checkpoint_every: *checkpoint_every,
                log_step_losses: *log_step_losses,
            };
            tokio::task::spawn_blocking(move || torch::train::pretrain(args))
                .await
                .expect("pretraining task panicked")
                .expect("pretraining failed");
        }
        Some(Commands::Infer {
            weights,
            episodes,
            deterministic,
            temperature,
            tickers,
            random_start,
            model_size,
        }) => {
            torch::infer::run_inference(
                weights,
                *episodes,
                *deterministic,
                *temperature,
                tickers.clone(),
                *random_start,
                *model_size,
            )
            .expect("inference failed");
        }
        Some(Commands::Paper {
            weights,
            symbols,
            interval,
            max_steps,
            temperature,
            model_size,
        }) => {
            torch::infer::run_ibkr_paper_trading(
                weights,
                symbols.clone(),
                *interval,
                *max_steps,
                *temperature,
                *model_size,
            )
            .expect("paper trading failed");
        }
        None => {
            torch::train::train(None, ModelVariant::UniformStream, None).await;
        }
    }

    println!("{}", "End".green())
}
