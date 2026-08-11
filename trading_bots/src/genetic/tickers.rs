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
        self.filter_available(|ticker| {
            let bars = get_cached_historical_bars(ticker)?;
            (bars.len() >= min_bars).then_some(())
        })
    }

    fn filter_available(self, mut is_available: impl FnMut(&str) -> Option<()>) -> Vec<String> {
        self.tickers()
            .into_iter()
            .filter(|ticker| is_available(ticker).is_some())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::TickerSet;

    #[test]
    fn missing_data_filters_within_canonical_partitions() {
        let unavailable = HashSet::from(["NVDA".to_string()]);

        for set in [TickerSet::Train, TickerSet::Validation, TickerSet::Test] {
            let canonical = set.tickers();
            let filtered =
                set.filter_available(|ticker| (!unavailable.contains(ticker)).then_some(()));
            let expected = canonical
                .into_iter()
                .filter(|ticker| !unavailable.contains(ticker))
                .collect::<Vec<_>>();

            assert_eq!(filtered, expected);
        }

        assert!(TickerSet::Validation
            .filter_available(|ticker| (!unavailable.contains(ticker)).then_some(()))
            .contains(&"GOOGL".to_string()));
        assert!(TickerSet::Test
            .filter_available(|ticker| (!unavailable.contains(ticker)).then_some(()))
            .contains(&"AVGO".to_string()));
    }
}
