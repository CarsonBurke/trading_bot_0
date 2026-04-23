mod backtest;
mod engine;
pub mod families;
mod family;
mod logging;
mod metrics;
mod tickers;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
pub use backtest::MarketDataset;
#[allow(unused_imports)]
pub use engine::{run_family_with_markets, DatasetBundle, SessionPaths, TrainingConfig};
#[allow(unused_imports)]
pub use families::{
    price_rebound::Family as PriceReboundFamily, rsi_rebound::Family as RsiReboundFamily,
    trend_breakout::Family as TrendBreakoutFamily,
};
pub use family::GeneticFamily;
pub use tickers::TickerSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneticArgs {
    pub family: GeneticFamily,
    pub run: Option<String>,
    pub generations: usize,
    pub population: usize,
    pub survivors: usize,
    pub train_tickers: TickerSet,
    pub validation_tickers: TickerSet,
    pub test_tickers: TickerSet,
    pub heavy_report_every: usize,
    pub seed: u64,
}

impl Default for GeneticArgs {
    fn default() -> Self {
        Self {
            family: GeneticFamily::TrendBreakout,
            run: None,
            generations: 600,
            population: 192,
            survivors: 48,
            train_tickers: TickerSet::Train,
            validation_tickers: TickerSet::Validation,
            test_tickers: TickerSet::Test,
            heavy_report_every: 5,
            seed: 7,
        }
    }
}

pub fn run(args: GeneticArgs) -> Result<()> {
    engine::run(args)
}
