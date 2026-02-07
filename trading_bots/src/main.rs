// For burn Wgpu
#![recursion_limit = "256"]
#![feature(f16)]
#![feature(stdarch_x86_avx512_bf16)]
#![allow(dead_code)]

use crate::torch::model::ModelVariant;
use clap::{Parser, Subcommand};
use colored::{self, Colorize};

mod agent;
mod charts;
mod constants;
mod data;
mod history;
mod neural_net;
mod strategies;
mod torch;
mod types;
mod utils;
// mod gym;

#[derive(Parser)]
#[command(name = "trading_bot")]
#[command(about = "Trading bot with PPO training and inference", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long)]
        weights: Option<String>,

        #[arg(long, value_enum, default_value_t = ModelVariant::Base)]
        model_size: ModelVariant,
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

        #[arg(long, value_enum, default_value_t = ModelVariant::Base)]
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

        #[arg(long, value_enum, default_value_t = ModelVariant::Base)]
        model_size: ModelVariant,
    },
}

#[tokio::main]
async fn main() {
    println!("{}", "Start".green());

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Train {
            weights,
            model_size,
        }) => {
            torch::ppo::train(weights.as_deref(), *model_size).await;
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
            torch::ibkr_infer::run_ibkr_paper_trading(
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
            torch::ppo::train(None, ModelVariant::Base).await;
        }
    }

    println!("{}", "End".green())
}
