//! Historical bars for the PPO, paper and live paths.
//!
//! One on-disk format for the whole repository: the packed corpus at
//! `long_data/bars/<SYMBOL>.<res_secs>.bars`, the very files the world-model pretraining,
//! the planner and the ingest pipeline read and write. The IBKR download path survives
//! only to extend that corpus with bars fresher than the last ingest for live and paper
//! trading, and it appends packed records like every other producer.

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use chrono::{Offset, TimeZone};
use chrono_tz::America::New_York;
use hashbrown::HashSet;
use ibapi::{
    contracts::Contract,
    market_data::{
        historical::{self, BarSize, Duration as HistoricalDuration, ToDuration, WhatToShow},
        TradingHours,
    },
    Client,
};
use shared::bars::{append_bars, bar_file_path, BarFile, PackedBar};
use time::{OffsetDateTime, UtcOffset};

use crate::{
    constants::api,
    data::universe::{corpus_bar_path, LIVE_RES_SECS},
    types::MappedHistorical,
};

/// History requested from IBKR for a symbol nothing on disk covers. A symbol already on
/// disk only needs the gap since its newest bar.
const IBKR_COLD_START_YEARS: i32 = 5;

/// Longest incremental top-up expressed in days; beyond this a cold start is cheaper and
/// stays inside IBKR's per-request limits.
const IBKR_MAX_TOP_UP_DAYS: i64 = 365;

static IBKR_DOWNLOAD_ENABLED: AtomicBool = AtomicBool::new(true);

#[derive(Debug, Clone)]
pub enum HistoricalLoadError {
    Connect(String),
    Request { ticker: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalAlignmentError(String);

impl std::fmt::Display for HistoricalAlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HistoricalAlignmentError {}

impl std::fmt::Display for HistoricalLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(message) => write!(f, "{message}"),
            Self::Request { ticker, message } => {
                write!(f, "historical data request failed for {ticker}: {message}")
            }
        }
    }
}

impl std::error::Error for HistoricalLoadError {}

pub fn set_ibkr_download_enabled(enabled: bool) {
    IBKR_DOWNLOAD_ENABLED.store(enabled, Ordering::Relaxed);
}

/// IBKR live tail for a symbol, in the same packed format but a separate directory.
///
/// The corpus is Polygon data, split- and dividend-adjusted and covering extended hours,
/// and it is what the world model pretrains on. IBKR bars are unadjusted, regular hours
/// only, and report volume in round lots, so splicing them into a corpus file would leave
/// a permanent level, coverage and volume-scale step in the pretraining data that `ingest`
/// would never repair. The live tail therefore lives beside the corpus and is read only by
/// the paper and live paths, which need bars fresher than the last ingest.
fn live_tail_path(ticker: &str) -> PathBuf {
    bar_file_path(
        Path::new(shared::paths::DATA_PATH).join("live_bars"),
        ticker,
        LIVE_RES_SECS,
    )
}

fn ibkr_symbol(ticker: &str) -> Cow<'_, str> {
    if ticker.contains('.') {
        Cow::Owned(ticker.replace('.', " "))
    } else {
        Cow::Borrowed(ticker)
    }
}

/// Drop bars that cannot be traded against: non-finite or non-positive prices, and bars
/// whose close is not yet final.
fn usable(bar: &PackedBar, completed_before: OffsetDateTime, res_secs: u32) -> bool {
    let (open, high, low, close) = (bar.open, bar.high, bar.low, bar.close);
    open.is_finite()
        && high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && open > 0.0
        && high > 0.0
        && low > 0.0
        && close > 0.0
        && bar.ts() / 1_000 + i64::from(res_secs) <= completed_before.unix_timestamp()
}

/// Completed packed bars from one corpus-format file, or `None` when it is absent or holds
/// nothing usable.
fn read_bar_file(path: &Path, completed_before: OffsetDateTime) -> Option<Vec<PackedBar>> {
    let file = match BarFile::open(path) {
        Ok(file) => file,
        Err(error) => {
            if path.exists() {
                eprintln!("failed opening bar file {}: {error:#}", path.display());
            }
            return None;
        }
    };

    let all = file.bars();
    let bars: Vec<PackedBar> = all
        .iter()
        .copied()
        .filter(|bar| usable(bar, completed_before, LIVE_RES_SECS))
        .collect();
    if bars.len() != all.len() {
        eprintln!(
            "Filtered {} unusable bars from {}",
            all.len() - bars.len(),
            path.display()
        );
    }
    (!bars.is_empty()).then_some(bars)
}

/// Completed packed bars for `ticker` out of the corpus alone.
fn load_from_corpus(ticker: &str, completed_before: OffsetDateTime) -> Option<Vec<PackedBar>> {
    read_bar_file(&corpus_bar_path(ticker), completed_before)
}

/// Corpus history for `ticker` extended by the IBKR live tail newer than it.
///
/// Only the live and paper paths call this: training reads the corpus alone so its inputs
/// stay pure Polygon.
fn load_with_live_tail(ticker: &str, completed_before: OffsetDateTime) -> Option<Vec<PackedBar>> {
    let corpus = load_from_corpus(ticker, completed_before);
    let tail = read_bar_file(&live_tail_path(ticker), completed_before);
    match (corpus, tail) {
        (Some(mut corpus), Some(tail)) => {
            let newest = corpus.last().map_or(i64::MIN, PackedBar::ts);
            corpus.extend(tail.into_iter().filter(|bar| bar.ts() > newest));
            Some(corpus)
        }
        (Some(corpus), None) => Some(corpus),
        (None, tail) => tail,
    }
}

fn fetch_or_load_ticker(
    ticker: &str,
    client: &mut Option<Client>,
) -> Result<Option<Vec<PackedBar>>, HistoricalLoadError> {
    if let Some(bars) = load_from_corpus(ticker, OffsetDateTime::now_utc()) {
        return Ok(Some(bars));
    }

    if !IBKR_DOWNLOAD_ENABLED.load(Ordering::Relaxed) {
        return Ok(None);
    }

    let client = match client {
        Some(client) => client,
        None => client.insert(Client::connect(api::CONNECTION_URL, 1).map_err(|err| {
            HistoricalLoadError::Connect(format!(
                "failed connecting to TWS for historical download at {}: {err}",
                api::CONNECTION_URL
            ))
        })?),
    };

    download_live_tail(client, ticker, OffsetDateTime::now_utc())
}

/// Packed bars for `ticker`, from the corpus if present and from IBKR otherwise.
pub fn get_packed_historical_bars_result(
    ticker: &str,
) -> Result<Option<Vec<PackedBar>>, HistoricalLoadError> {
    let mut client = None;
    fetch_or_load_ticker(ticker, &mut client)
}

/// Packed bars for `tickers`, restricted to the timestamps every symbol shares.
pub fn get_packed_historical_data(tickers: &[String]) -> Vec<Vec<PackedBar>> {
    let mut data = Vec::with_capacity(tickers.len());
    let mut client = None;

    for ticker in tickers {
        let bars = fetch_or_load_ticker(ticker, &mut client)
            .unwrap_or_else(|err| panic!("{err}"))
            .unwrap_or_else(|| panic!("historical data unavailable for {ticker}"));
        data.push(bars);
    }

    align_packed_to_common_timestamps(data)
        .unwrap_or_else(|err| panic!("failed aligning historical data: {err}"))
}

/// [`get_packed_historical_data`] in the IBKR bar shape the genetic and strategy
/// backtests are written against.
pub fn get_historical_data(tickers: &[String]) -> MappedHistorical {
    get_packed_historical_data(tickers)
        .iter()
        .map(|bars| to_ibkr_bars(bars))
        .collect()
}

/// Download fresh bars for `ticker` and return its history including them.
///
/// Live and paper trading are the only callers. When the corpus already holds the symbol
/// the download is a top-up: the fresh bars are appended and the full corpus history comes
/// back. A corpus file is only ever appended to, never rewritten or created here.
pub fn refresh_historical_bars_at(
    client: &Client,
    ticker: &str,
    completed_before: OffsetDateTime,
) -> Result<Vec<PackedBar>, HistoricalLoadError> {
    download_live_tail(client, ticker, completed_before)?
        .ok_or_else(|| HistoricalLoadError::Request {
            ticker: ticker.to_string(),
            message: "IBKR returned no completed historical bars".to_string(),
        })
}

/// Restrict every series to the timestamps all of them share.
///
/// Rejects an empty series, a series with duplicate or out-of-order timestamps, and a set
/// with no timestamp in common, because each of those silently corrupts a backtest that
/// indexes every symbol by the same step.
pub fn align_packed_to_common_timestamps(
    bars_by_ticker: Vec<Vec<PackedBar>>,
) -> Result<Vec<Vec<PackedBar>>, HistoricalAlignmentError> {
    if bars_by_ticker.is_empty() || bars_by_ticker.iter().any(Vec::is_empty) {
        return Err(HistoricalAlignmentError(
            "encountered an empty ticker series".to_string(),
        ));
    }

    for (ticker_idx, bars) in bars_by_ticker.iter().enumerate() {
        for pair in bars.windows(2) {
            let (prev, next) = (pair[0].ts(), pair[1].ts());
            if next == prev {
                return Err(HistoricalAlignmentError(format!(
                    "ticker series {ticker_idx} contains duplicate timestamp {next}"
                )));
            }
            if next < prev {
                return Err(HistoricalAlignmentError(format!(
                    "ticker series {ticker_idx} is not strictly chronological at {next}"
                )));
            }
        }
    }

    if bars_by_ticker.len() == 1 {
        return Ok(bars_by_ticker);
    }

    let mut common_timestamps = bars_by_ticker[0]
        .iter()
        .map(PackedBar::ts)
        .collect::<HashSet<_>>();
    for bars in bars_by_ticker.iter().skip(1) {
        let timestamps = bars.iter().map(PackedBar::ts).collect::<HashSet<_>>();
        common_timestamps.retain(|timestamp| timestamps.contains(timestamp));
    }
    if common_timestamps.is_empty() {
        return Err(HistoricalAlignmentError(
            "ticker series have no timestamps in common".to_string(),
        ));
    }

    Ok(bars_by_ticker
        .into_iter()
        .map(|bars| {
            bars.into_iter()
                .filter(|bar| common_timestamps.contains(&bar.ts()))
                .collect()
        })
        .collect())
}

/// Bar open time in the exchange's timezone.
///
/// The corpus stores UTC epoch millis and covers extended hours, so an 8pm Eastern bar
/// falls on the following UTC day. Earnings and macro indicators key off the trading date,
/// which makes America/New_York the only correct calendar for a bar's wall clock. This is
/// also the timezone IBKR itself stamped onto the bars this corpus replaced.
pub fn exchange_time(ts_ms: i64) -> OffsetDateTime {
    let seconds = ts_ms.div_euclid(1_000);
    let utc = chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, 0)
        .expect("corpus timestamps are within the representable range");
    let offset_seconds = New_York
        .offset_from_utc_datetime(&utc.naive_utc())
        .fix()
        .local_minus_utc();
    OffsetDateTime::from_unix_timestamp(seconds)
        .expect("corpus timestamps are within the representable range")
        .to_offset(
            UtcOffset::from_whole_seconds(offset_seconds)
                .expect("America/New_York offsets are whole minutes within a day"),
        )
}

/// Packed records in the IBKR bar shape, for the backtests that still speak it.
pub fn to_ibkr_bars(bars: &[PackedBar]) -> Vec<historical::Bar> {
    bars.iter()
        .map(|bar| historical::Bar {
            date: exchange_time(bar.ts()),
            open: f64::from(bar.open),
            high: f64::from(bar.high),
            low: f64::from(bar.low),
            close: f64::from(bar.close),
            volume: f64::from(bar.volume),
            wap: f64::from(bar.vwap),
            count: i32::try_from(bar.trades).unwrap_or(i32::MAX),
        })
        .collect()
}

/// IBKR bars as packed records, dropping anything unrepresentable or not yet complete.
///
/// The corpus format requires strictly increasing timestamps, so a repeated timestamp
/// keeps its first observation rather than failing the whole append.
fn to_packed_bars(
    bars: &[historical::Bar],
    completed_before: OffsetDateTime,
) -> Vec<PackedBar> {
    let mut packed: Vec<PackedBar> = bars
        .iter()
        .map(|bar| PackedBar {
            ts_ms: (bar.date.unix_timestamp_nanos() / 1_000_000) as i64,
            open: bar.open as f32,
            high: bar.high as f32,
            low: bar.low as f32,
            close: bar.close as f32,
            volume: bar.volume as f32,
            vwap: bar.wap as f32,
            trades: u32::try_from(bar.count).unwrap_or(0),
        })
        .filter(|bar| usable(bar, completed_before, LIVE_RES_SECS))
        .collect();
    // Stable, so `dedup_by_key` keeps the first observation of a repeated timestamp.
    packed.sort_by_key(PackedBar::ts);
    packed.dedup_by_key(|bar| bar.ts());
    packed
}

/// Bars requested from IBKR: the gap since the corpus's newest bar, or a cold start when
/// the corpus has nothing for the symbol (or is more than [`IBKR_MAX_TOP_UP_DAYS`] behind,
/// where a bounded top-up would leave a hole).
fn ibkr_lookback(
    corpus_last_ts_ms: Option<i64>,
    completed_before: OffsetDateTime,
) -> HistoricalDuration {
    let Some(last_ts_ms) = corpus_last_ts_ms else {
        return IBKR_COLD_START_YEARS.years();
    };
    let gap_secs = completed_before.unix_timestamp() - last_ts_ms.div_euclid(1_000);
    // One extra day covers the partial day at each end of the gap.
    let days = gap_secs.div_euclid(86_400) + 2;
    if days > IBKR_MAX_TOP_UP_DAYS {
        return IBKR_COLD_START_YEARS.years();
    }
    (days.max(1) as i32).days()
}

/// Download bars fresher than everything on disk for `ticker`, persist them to its live
/// tail, and return the symbol's full history.
fn download_live_tail(
    client: &Client,
    ticker: &str,
    completed_before: OffsetDateTime,
) -> Result<Option<Vec<PackedBar>>, HistoricalLoadError> {
    let tail_path = live_tail_path(ticker);
    let newest_on_disk = [corpus_bar_path(ticker), tail_path.clone()]
        .iter()
        .filter_map(|path| BarFile::open(path).ok().and_then(|file| file.last_ts_ms()))
        .max();

    println!("Downloading data for {ticker}");
    let ibkr_symbol = ibkr_symbol(ticker);
    let contract = Contract::stock(ibkr_symbol.as_ref()).build();

    let historical_data = client
        .historical_data(
            &contract,
            Some(completed_before),
            ibkr_lookback(newest_on_disk, completed_before),
            BarSize::Min5,
            WhatToShow::Trades,
            TradingHours::Regular,
        )
        .map_err(|err| HistoricalLoadError::Request {
            ticker: ticker.to_string(),
            message: err.to_string(),
        })?;

    let downloaded = to_packed_bars(&historical_data.bars, completed_before);
    if downloaded.len() != historical_data.bars.len() {
        eprintln!(
            "Filtered {} invalid bars for {} from IBKR",
            historical_data.bars.len() - downloaded.len(),
            ticker
        );
    }
    if downloaded.is_empty() {
        eprintln!("Downloaded zero bars for {ticker}");
        return Ok(load_with_live_tail(ticker, completed_before));
    }

    // `append_bars` writes only the records newer than the tail's last timestamp, and
    // creates the file from its name when the symbol has no tail yet. The Polygon corpus
    // is never touched.
    if let Err(error) = append_bars(&tail_path, &downloaded) {
        eprintln!("failed extending live tail {}: {error:#}", tail_path.display());
    }

    match load_with_live_tail(ticker, completed_before) {
        Some(bars) => Ok(Some(bars)),
        // The tail write failed; trade off the download alone rather than nothing.
        None => Ok(Some(downloaded)),
    }
}

#[cfg(test)]
mod tests {
    use shared::bars::PackedBar;

    use super::{
        align_packed_to_common_timestamps, exchange_time, ibkr_lookback, to_ibkr_bars,
        to_packed_bars,
    };
    use ibapi::market_data::historical::{Bar, ToDuration};
    use time::{Duration, OffsetDateTime};

    fn bar(minute: i64) -> PackedBar {
        PackedBar {
            ts_ms: minute * 60_000,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.0,
            volume: 1_000.0,
            vwap: 100.0,
            trades: 1,
        }
    }

    #[test]
    fn aligns_different_starts_and_internal_gaps_to_strict_intersection() {
        let aligned = align_packed_to_common_timestamps(vec![
            vec![bar(0), bar(1), bar(2), bar(3)],
            vec![bar(0), bar(2), bar(3), bar(4)],
            vec![bar(2), bar(3), bar(5)],
        ])
        .expect("histories should align");

        let expected = vec![bar(2).ts(), bar(3).ts()];
        assert_eq!(aligned.len(), 3);
        for bars in aligned {
            assert_eq!(bars.iter().map(PackedBar::ts).collect::<Vec<_>>(), expected);
        }
    }

    #[test]
    fn single_ticker_preserves_the_original_series() {
        let original = vec![bar(0), bar(2), bar(5)];
        let aligned = align_packed_to_common_timestamps(vec![original.clone()])
            .expect("single ticker should remain valid");

        assert_eq!(aligned, vec![original]);
    }

    #[test]
    fn rejects_duplicate_or_non_monotonic_timestamps() {
        let duplicate = align_packed_to_common_timestamps(vec![vec![bar(0), bar(0)]])
            .expect_err("duplicate timestamps must be rejected");
        assert!(duplicate.to_string().contains("duplicate timestamp"));

        let reversed = align_packed_to_common_timestamps(vec![vec![bar(1), bar(0)]])
            .expect_err("non-monotonic timestamps must be rejected");
        assert!(reversed.to_string().contains("not strictly chronological"));
    }

    #[test]
    fn rejects_histories_without_a_common_timestamp() {
        let error = align_packed_to_common_timestamps(vec![vec![bar(0)], vec![bar(1)]])
            .expect_err("disjoint histories must be rejected");

        assert!(error.to_string().contains("no timestamps in common"));
    }

    #[test]
    fn packed_and_ibkr_shapes_round_trip() {
        let packed = vec![bar(0), bar(5), bar(10)];
        let ibkr = to_ibkr_bars(&packed);
        assert_eq!(ibkr.len(), 3);
        assert_eq!(ibkr[1].date.unix_timestamp(), 300);
        assert_eq!(ibkr[1].close, 100.0);
        assert_eq!(ibkr[1].wap, 100.0);
        assert_eq!(ibkr[1].count, 1);

        let cutoff = OffsetDateTime::UNIX_EPOCH + Duration::days(1);
        assert_eq!(to_packed_bars(&ibkr, cutoff), packed);
    }

    #[test]
    fn packing_drops_unusable_bars_and_collapses_duplicate_timestamps() {
        let cutoff = OffsetDateTime::UNIX_EPOCH + Duration::days(1);
        let template = to_ibkr_bars(&[bar(0)])[0];
        let bars = vec![
            Bar {
                close: 0.0,
                ..template
            },
            Bar {
                date: cutoff + Duration::hours(1),
                ..template
            },
            Bar {
                open: 101.0,
                ..template
            },
            template,
        ];

        let packed = to_packed_bars(&bars, cutoff);
        assert_eq!(packed.len(), 1, "one usable timestamp survives");
        assert_eq!(packed[0].open, 101.0, "the first observation is kept");
    }

    #[test]
    fn lookback_tops_up_from_the_corpus_and_cold_starts_without_one() {
        let now = OffsetDateTime::UNIX_EPOCH + Duration::days(4_000);
        assert_eq!(ibkr_lookback(None, now), 5.years());

        let two_days_ago = (now - Duration::days(2)).unix_timestamp() * 1_000;
        assert_eq!(ibkr_lookback(Some(two_days_ago), now), 4.days());

        let just_now = now.unix_timestamp() * 1_000;
        assert_eq!(ibkr_lookback(Some(just_now), now), 2.days());

        let ancient = (now - Duration::days(900)).unix_timestamp() * 1_000;
        assert_eq!(ibkr_lookback(Some(ancient), now), 5.years());
    }

    #[test]
    fn exchange_time_uses_the_trading_day_not_the_utc_day() {
        // 2024-01-03 00:30 UTC is 2024-01-02 19:30 in New York: an after-hours bar of the
        // previous trading day.
        let after_hours = exchange_time(1_704_242_400_000);
        assert_eq!(
            (
                after_hours.year(),
                after_hours.month() as u8,
                after_hours.day()
            ),
            (2024, 1, 2)
        );
        assert_eq!(after_hours.hour(), 19);
        assert_eq!(after_hours.unix_timestamp(), 1_704_242_400);

        // Daylight saving time: 2024-07-03 14:30 UTC is 10:30 in New York.
        let summer = exchange_time(1_720_017_000_000);
        assert_eq!(
            (summer.month() as u8, summer.day(), summer.hour()),
            (7, 3, 10)
        );
    }
}
