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
    LazyLock::new(|| eligible_bar_universe(&bars_dir(), LIVE_RES_SECS, MIN_TRADING_BARS, None));

/// The operational PPO, paper and live universe, sorted and memoized.
///
/// Operational eligibility intentionally measures CURRENT total history: these consumers need
/// enough bars to run an episode now. Causal research admission instead supplies `train_end_ms`
/// to [`eligible_bar_universe`]. Keeping the distinction explicit prevents the live rule from
/// masquerading as the point-in-time corpus rule.
pub fn cached_bar_universe() -> &'static [String] {
    CACHED_BAR_UNIVERSE.as_slice()
}

/// Resolve eligible symbols under either current-depth (`train_end_ms == None`) or causal
/// point-in-time (`Some(train_end_ms)`) semantics.
///
/// Under a cutoff, eligibility is `file.index_at_or_after(train_end_ms) >= min_bars`: the bar at
/// the boundary does not count. The full file remains represented by its symbol after admission.
pub fn eligible_bar_universe(
    dir: &Path,
    res_secs: u32,
    min_bars: usize,
    train_end_ms: Option<i64>,
) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!(
            "[universe] cannot read bar corpus directory {}",
            dir.display()
        );
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
            Ok(file) => {
                let eligible_bars = train_end_ms
                    .map(|cutoff| file.index_at_or_after(cutoff))
                    .unwrap_or_else(|| file.len());
                if eligible_bars >= min_bars {
                    eligible.push(symbol);
                } else {
                    let rule = train_end_ms.map_or_else(
                        || "current total bars".to_owned(),
                        |cutoff| format!("training bars strictly before train_end {cutoff}"),
                    );
                    println!(
                        "[universe] dropping {symbol}: {eligible_bars} {rule} < minimum {min_bars}"
                    );
                }
            }
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
    BarFile::open(&corpus_bar_path(symbol))
        .ok()
        .map(|file| file.len())
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
