use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::data::{historical::get_cached_historical_bars, universe::TARGET_UNIVERSE_TICKERS};

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

fn split_ranked_ticker_strings(tickers: &[String]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();

    for (index, ticker) in tickers.iter().enumerate() {
        match index % 6 {
            0 | 1 | 2 | 3 => train.push(ticker.clone()),
            4 => validation.push(ticker.clone()),
            _ => test.push(ticker.clone()),
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

    pub fn cached_eligible_tickers(self, min_bars: usize) -> Vec<String> {
        let eligible = TARGET_UNIVERSE_TICKERS
            .iter()
            .filter_map(|ticker| {
                let bars = get_cached_historical_bars(ticker)?;
                (bars.len() >= min_bars).then(|| (*ticker).to_string())
            })
            .collect::<Vec<_>>();

        let (train, validation, test) = split_ranked_ticker_strings(&eligible);
        match self {
            Self::Train => train,
            Self::Validation => validation,
            Self::Test => test,
            Self::All => eligible,
        }
    }
}
