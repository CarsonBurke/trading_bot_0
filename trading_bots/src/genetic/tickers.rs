use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::data::universe::{training_universe, TARGET_UNIVERSE_TICKERS};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum TickerSet {
    Train,
    Validation,
    Test,
    All,
}

fn split_ranked_tickers(tickers: &[&str]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();

    for (index, ticker) in tickers.iter().copied().enumerate() {
        match index % 6 {
            0 | 1 | 2 | 3 => train.push(ticker.to_string()),
            4 => validation.push(ticker.to_string()),
            _ => test.push(ticker.to_string()),
        }
    }

    (train, validation, test)
}

impl TickerSet {
    pub fn label(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
            Self::All => "all",
        }
    }

    pub fn tickers(self) -> Vec<String> {
        let (train, validation, test) = split_ranked_tickers(TARGET_UNIVERSE_TICKERS);
        match self {
            Self::Train => train,
            Self::Validation => validation,
            Self::Test => test,
            Self::All => TARGET_UNIVERSE_TICKERS
                .iter()
                .map(|ticker| ticker.to_string())
                .collect(),
        }
    }

    pub fn resolved_tickers(self) -> Vec<String> {
        let (train, validation, test) = split_ranked_tickers(training_universe());
        match self {
            Self::Train => train,
            Self::Validation => validation,
            Self::Test => test,
            Self::All => training_universe()
                .iter()
                .map(|ticker| ticker.to_string())
                .collect(),
        }
    }
}
