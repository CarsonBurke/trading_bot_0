use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use shared::bars::{bar_file_path, parse_bar_file_name, BarFile, FILE_EXTENSION};
use shared::constants::{PRICE_DELTAS_PER_TICKER, STEPS_PER_EPISODE};

use crate::data::ingest::bars_dir;

/// Bar resolution the PPO, paper and live paths trade on, in seconds.
///
/// The same resolution the world model pretrains on, so both read the same
/// `<SYMBOL>.300.bars` files out of the same corpus directory.
pub const LIVE_RES_SECS: u32 = 300;

/// Minimum packed bars a symbol needs to enter the trading universe.
///
/// `full_episode_start_offsets` refuses to start an episode unless a symbol holds
/// `PRICE_DELTAS_PER_TICKER + STEPS_PER_EPISODE` bars, which buys exactly one start offset.
/// Ten further episodes of history keep `random_start` from degenerating onto that single
/// offset, and the surplus is what the momentum and earnings indicators warm up over.
///
/// Note the wall-clock horizon this implies. The packed corpus covers extended hours, about
/// 186 five-minute bars per session against the 78 of a regular-hours-only feed, so the
/// 6000-bar observation window spans roughly 32 sessions and an episode roughly 11. Every
/// constant expressed in bars therefore means about 2.4x less calendar time than it did on
/// the old regular-hours IBKR series. That is a modelling choice, recorded here rather than
/// silently rescaled.
pub const MIN_TRADING_BARS: usize = PRICE_DELTAS_PER_TICKER + 10 * STEPS_PER_EPISODE;

static CACHED_BAR_UNIVERSE: LazyLock<Vec<String>> =
    LazyLock::new(|| eligible_bar_universe(&bars_dir(), LIVE_RES_SECS, MIN_TRADING_BARS));

/// The trading universe: every corpus symbol with enough 5-minute history to run a
/// full episode, sorted and memoized.
///
/// This is the PPO, paper and live universe, and it is nothing but a cached
/// [`eligible_bar_universe`] over the corpus the world model also reads. There is no
/// second, hand-curated list to drift away from it.
pub fn cached_bar_universe() -> &'static [String] {
    CACHED_BAR_UNIVERSE.as_slice()
}

/// Symbols in a packed-bar corpus directory (`<dir>/<SYMBOL>.<res_secs>.bars`) holding at
/// least `min_bars` bars, sorted. The single source of truth for every universe in the
/// repository: pretraining, the planner, PPO and live trading all resolve to this.
///
/// Every rejection is reported, so a half-downloaded corpus cannot pass itself off as a
/// healthy small one.
pub fn eligible_bar_universe(dir: &Path, res_secs: u32, min_bars: usize) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("[universe] cannot read bar corpus directory {}", dir.display());
        return Vec::new();
    };
    let mut eligible = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(FILE_EXTENSION) {
            continue;
        }
        let Ok((symbol, res)) = parse_bar_file_name(&path) else {
            continue;
        };
        if res != res_secs {
            continue;
        }
        match BarFile::open(&path) {
            Ok(file) if file.len() >= min_bars => eligible.push(symbol),
            Ok(file) => println!(
                "[universe] dropping {symbol}: {} bars < min_bars {min_bars}",
                file.len()
            ),
            Err(error) => eprintln!("[universe] dropping {symbol}: {error:#}"),
        }
    }
    eligible.sort();
    eligible
}

/// Corpus file for `symbol` at the live resolution.
pub fn corpus_bar_path(symbol: &str) -> PathBuf {
    bar_file_path(bars_dir(), symbol, LIVE_RES_SECS)
}

/// Bars the corpus holds for `symbol`, read from the file header without touching a
/// single record.
pub fn corpus_bar_count(symbol: &str) -> Option<usize> {
    BarFile::open(&corpus_bar_path(symbol)).ok().map(|file| file.len())
}

/// The `count` universe symbols with the deepest history, deepest first, ties broken
/// alphabetically.
///
/// The deliberate ranking for the consumers that cannot take the whole corpus: the genetic
/// search, which materializes every bar of every symbol it is given, and the paper/live
/// default symbol. Depth is the one liquidity-adjacent property the corpus itself carries,
/// so this needs no second ranking artifact to stay in sync with.
pub fn deepest_symbols(count: usize) -> Vec<String> {
    let mut ranked: Vec<(usize, &str)> = cached_bar_universe()
        .iter()
        .filter_map(|symbol| corpus_bar_count(symbol).map(|bars| (bars, symbol.as_str())))
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    ranked
        .into_iter()
        .take(count)
        .map(|(_, symbol)| symbol.to_owned())
        .collect()
}
