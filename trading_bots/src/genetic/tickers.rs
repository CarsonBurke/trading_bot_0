use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

use crate::data::universe::{corpus_bar_count, deepest_symbols};

/// Symbols the genetic search runs over.
///
/// `load_market` materializes every bar of every symbol in a split and holds all three
/// splits at once, so the search cannot be handed the whole corpus: at corpus depth that is
/// tens of gigabytes. The cap is close to the size of the hand-curated list this replaced,
/// so the search's cost profile is unchanged.
const GENETIC_UNIVERSE_SIZE: usize = 96;

/// The genetic universe, ranked by corpus depth, deepest first.
static GENETIC_UNIVERSE: LazyLock<Vec<String>> =
    LazyLock::new(|| deepest_symbols(GENETIC_UNIVERSE_SIZE));

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum TickerSet {
    Train,
    Validation,
    Test,
    All,
}

/// Interleave a depth-ranked universe 4/1/1, so every split spans the whole depth range
/// rather than one split getting all the deepest histories.
fn split_ranked_tickers(tickers: &[String]) -> (Vec<String>, Vec<String>, Vec<String>) {
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
        let universe = GENETIC_UNIVERSE.as_slice();
        let (train, validation, test) = split_ranked_tickers(universe);
        match self {
            Self::Train => train,
            Self::Validation => validation,
            Self::Test => test,
            Self::All => universe.to_vec(),
        }
    }

    /// Symbols in this split whose corpus history is long enough, without downloading.
    pub fn corpus_eligible_tickers(self, min_bars: usize) -> Vec<String> {
        self.filter_available(|ticker| (corpus_bar_count(ticker)? >= min_bars).then_some(()))
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
        let universe = TickerSet::All.tickers();
        assert!(!universe.is_empty(), "genetic universe must be non-empty");
        let unavailable = HashSet::from([universe[0].clone()]);

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
    }

    /// The 4/1/1 interleave must partition the universe exactly: every symbol lands in
    /// exactly one split, and no symbol is invented. Asserting membership of a NAMED
    /// symbol would be wrong here — the universe is ranked by corpus depth over thousands
    /// of symbols, so which split a given ticker falls into is a property of the data.
    #[test]
    fn splits_partition_the_universe_without_overlap() {
        let universe = TickerSet::All.tickers();
        let train = TickerSet::Train.tickers();
        let validation = TickerSet::Validation.tickers();
        let test = TickerSet::Test.tickers();

        assert_eq!(train.len() + validation.len() + test.len(), universe.len());

        let mut seen = HashSet::new();
        for ticker in train.iter().chain(&validation).chain(&test) {
            assert!(seen.insert(ticker.clone()), "{ticker} is in two splits");
            assert!(universe.contains(ticker), "{ticker} is not in the universe");
        }
        assert_eq!(seen.len(), universe.len());

        // 4/1/1 interleave: train takes four of every six ranks.
        assert_eq!(train.len(), universe.len() - universe.len() / 6 * 2);
    }
}
