use std::{
    borrow::Cow,
    fs,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock, RwLock,
    },
};

use hashbrown::{HashMap, HashSet};
use ibapi::{
    contracts::Contract,
    market_data::{
        historical::{self, BarSize, ToDuration, WhatToShow},
        TradingHours,
    },
    Client,
};
use time::OffsetDateTime;

use crate::{
    constants::{
        api,
        files::{self, DATA_PATH},
        TICKERS,
    },
    types::MappedHistorical,
    utils::create_folder_if_not_exists,
    utils::{convert_historical, get_price_deltas},
};

static DATA_CACHE: OnceLock<RwLock<HashMap<String, Vec<historical::Bar>>>> = OnceLock::new();
static SERIES_CACHE: OnceLock<RwLock<HashMap<String, (Vec<f64>, Vec<f64>)>>> = OnceLock::new();
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

fn bars_cache() -> &'static RwLock<HashMap<String, Vec<historical::Bar>>> {
    DATA_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn series_cache() -> &'static RwLock<HashMap<String, (Vec<f64>, Vec<f64>)>> {
    SERIES_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn insert_cache_entry(ticker: &str, bars: Vec<historical::Bar>) -> Vec<historical::Bar> {
    let prices = convert_historical(&bars);
    let deltas = get_price_deltas(&bars);

    bars_cache()
        .write()
        .expect("historical bars cache poisoned")
        .insert(ticker.to_string(), bars.clone());
    series_cache()
        .write()
        .expect("historical series cache poisoned")
        .insert(ticker.to_string(), (prices, deltas));

    bars
}

fn get_cached_bars(ticker: &str) -> Option<Vec<historical::Bar>> {
    bars_cache()
        .read()
        .expect("historical bars cache poisoned")
        .get(ticker)
        .cloned()
}

fn get_cached_series(ticker: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    series_cache()
        .read()
        .expect("historical series cache poisoned")
        .get(ticker)
        .cloned()
}

fn ibkr_symbol(ticker: &str) -> Cow<'_, str> {
    if ticker.contains('.') {
        Cow::Owned(ticker.replace('.', " "))
    } else {
        Cow::Borrowed(ticker)
    }
}

fn fetch_or_load_ticker(
    ticker: &str,
    client: &mut Option<Client>,
) -> Result<Option<Vec<historical::Bar>>, HistoricalLoadError> {
    if let Some(bars) = get_cached_bars(ticker) {
        return Ok(Some(bars));
    }

    if let Some(bars) = get_historical_data_from_files(ticker) {
        return Ok(Some(insert_cache_entry(ticker, bars)));
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

    let Some(bars) = get_historical_data_from_ibkr(client, ticker)? else {
        return Ok(None);
    };
    Ok(Some(insert_cache_entry(ticker, bars)))
}

pub fn ensure_historical_data_len(ticker: &str) -> Option<usize> {
    let mut client = None;
    fetch_or_load_ticker(ticker, &mut client)
        .ok()
        .flatten()
        .map(|bars| bars.len())
}

pub fn get_historical_bars(ticker: &str) -> Option<Vec<historical::Bar>> {
    get_historical_bars_result(ticker).ok().flatten()
}

pub fn get_cached_historical_bars(ticker: &str) -> Option<Vec<historical::Bar>> {
    if let Some(bars) = get_cached_bars(ticker) {
        return Some(bars);
    }
    let bars = get_historical_data_from_files(ticker)?;
    Some(insert_cache_entry(ticker, bars))
}

pub fn get_historical_bars_result(
    ticker: &str,
) -> Result<Option<Vec<historical::Bar>>, HistoricalLoadError> {
    let mut client = None;
    fetch_or_load_ticker(ticker, &mut client)
}

pub fn refresh_historical_bars(
    client: &Client,
    ticker: &str,
) -> Result<Vec<historical::Bar>, HistoricalLoadError> {
    refresh_historical_bars_at(client, ticker, OffsetDateTime::now_utc())
}

pub fn refresh_historical_bars_at(
    client: &Client,
    ticker: &str,
    completed_before: OffsetDateTime,
) -> Result<Vec<historical::Bar>, HistoricalLoadError> {
    get_historical_data_from_ibkr_at(client, ticker, completed_before)?
        .map(|bars| insert_cache_entry(ticker, bars))
        .ok_or_else(|| HistoricalLoadError::Request {
            ticker: ticker.to_string(),
            message: "IBKR returned no completed historical bars".to_string(),
        })
}

pub fn get_historical_series(ticker: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    if let Some(series) = get_cached_series(ticker) {
        return Some(series);
    }

    let bars = get_cached_bars(ticker)?;
    let prices = convert_historical(&bars);
    let deltas = get_price_deltas(&bars);
    series_cache()
        .write()
        .expect("historical series cache poisoned")
        .insert(ticker.to_string(), (prices.clone(), deltas.clone()));
    Some((prices, deltas))
}

pub fn get_historical_data(tickers: Option<&[&str]>) -> MappedHistorical {
    let tickers = tickers.unwrap_or(TICKERS);
    let mut data = Vec::with_capacity(tickers.len());
    let mut client = None;

    for ticker in tickers {
        let bars = fetch_or_load_ticker(ticker, &mut client)
            .unwrap_or_else(|err| panic!("{err}"))
            .unwrap_or_else(|| panic!("historical data unavailable for {ticker}"));
        data.push(bars);
    }

    align_bars_to_common_timestamps(data)
        .unwrap_or_else(|err| panic!("failed aligning historical data: {err}"))
}

pub fn align_bars_to_common_timestamps(
    bars_by_ticker: MappedHistorical,
) -> Result<MappedHistorical, HistoricalAlignmentError> {
    if bars_by_ticker.is_empty() || bars_by_ticker.iter().any(Vec::is_empty) {
        return Err(HistoricalAlignmentError(
            "encountered an empty ticker series".to_string(),
        ));
    }

    for (ticker_idx, bars) in bars_by_ticker.iter().enumerate() {
        for pair in bars.windows(2) {
            if pair[1].date == pair[0].date {
                return Err(HistoricalAlignmentError(format!(
                    "ticker series {ticker_idx} contains duplicate timestamp {}",
                    pair[1].date
                )));
            }
            if pair[1].date < pair[0].date {
                return Err(HistoricalAlignmentError(format!(
                    "ticker series {ticker_idx} is not strictly chronological at {}",
                    pair[1].date
                )));
            }
        }
    }

    if bars_by_ticker.len() == 1 {
        return Ok(bars_by_ticker);
    }

    let mut common_timestamps = bars_by_ticker[0]
        .iter()
        .map(|bar| bar.date.unix_timestamp_nanos())
        .collect::<HashSet<_>>();
    for bars in bars_by_ticker.iter().skip(1) {
        let timestamps = bars
            .iter()
            .map(|bar| bar.date.unix_timestamp_nanos())
            .collect::<HashSet<_>>();
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
                .filter(|bar| common_timestamps.contains(&bar.date.unix_timestamp_nanos()))
                .collect()
        })
        .collect())
}

fn get_historical_data_from_files(ticker: &str) -> Option<Vec<historical::Bar>> {
    let path = format!("{}/{}.bin", files::DATA_PATH, ticker);
    let file = fs::read(path).ok()?;

    let mut bars: Vec<historical::Bar> = postcard::from_bytes(&file).ok()?;
    let before = bars.len();
    bars.retain(|b| {
        b.open.is_finite()
            && b.high.is_finite()
            && b.low.is_finite()
            && b.close.is_finite()
            && b.open > 0.0
            && b.high > 0.0
            && b.low > 0.0
            && b.close > 0.0
            && b.date.unix_timestamp().saturating_add(300)
                <= OffsetDateTime::now_utc().unix_timestamp()
    });
    if bars.len() != before {
        eprintln!(
            "Filtered {} invalid bars for {} from cache",
            before - bars.len(),
            ticker
        );
    }
    if bars.is_empty() {
        return None;
    }
    Some(bars)
}

fn get_historical_data_from_ibkr(
    client: &Client,
    ticker: &str,
) -> Result<Option<Vec<historical::Bar>>, HistoricalLoadError> {
    get_historical_data_from_ibkr_at(client, ticker, OffsetDateTime::now_utc())
}

fn get_historical_data_from_ibkr_at(
    client: &Client,
    ticker: &str,
    completed_before: OffsetDateTime,
) -> Result<Option<Vec<historical::Bar>>, HistoricalLoadError> {
    create_folder_if_not_exists(&files::DATA_PATH.to_string());

    println!("Downloading data for {ticker}");
    let ibkr_symbol = ibkr_symbol(ticker);
    let contract = Contract::stock(ibkr_symbol.as_ref()).build();

    let historical_data = client
        .historical_data(
            &contract,
            Some(completed_before),
            match data_path_kind() {
                "data" => 356.days(),
                "long_data" => 5.years(),
                "very_long_data" => 10.years(),
                "extra_long_data" => 20.years(),
                _ => panic!("no data path provided"),
            },
            BarSize::Min5,
            WhatToShow::Trades,
            TradingHours::Regular,
        )
        .map_err(|err| HistoricalLoadError::Request {
            ticker: ticker.to_string(),
            message: err.to_string(),
        })?;

    let mut bars = historical_data.bars;
    let before = bars.len();
    bars.retain(|b| {
        b.open.is_finite()
            && b.high.is_finite()
            && b.low.is_finite()
            && b.close.is_finite()
            && b.open > 0.0
            && b.high > 0.0
            && b.low > 0.0
            && b.close > 0.0
            && b.date.unix_timestamp().saturating_add(300) <= completed_before.unix_timestamp()
    });
    if bars.len() != before {
        eprintln!(
            "Filtered {} invalid bars for {} from IBKR",
            before - bars.len(),
            ticker
        );
    }
    if bars.is_empty() {
        eprintln!("Downloaded zero bars for {ticker}");
        return Ok(None);
    }

    let Some(encoded) = postcard::to_allocvec(&bars).ok() else {
        return Ok(Some(bars));
    };
    fs::write(
        format!("{}/{}.bin", files::DATA_PATH, ticker),
        encoded.as_slice(),
    )
    .map_err(|err| eprintln!("failed writing cached bars for {ticker}: {err}"))
    .ok();

    Ok(Some(bars))
}

fn data_path_kind() -> &'static str {
    Path::new(DATA_PATH)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use ibapi::market_data::historical::Bar;
    use time::{Duration, OffsetDateTime};

    use super::align_bars_to_common_timestamps;

    fn bar(minute: i64) -> Bar {
        Bar {
            date: OffsetDateTime::UNIX_EPOCH + Duration::minutes(minute),
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.0,
            volume: 1_000.0,
            wap: 100.0,
            count: 1,
        }
    }

    #[test]
    fn aligns_different_starts_and_internal_gaps_to_strict_intersection() {
        let aligned = align_bars_to_common_timestamps(vec![
            vec![bar(0), bar(1), bar(2), bar(3)],
            vec![bar(0), bar(2), bar(3), bar(4)],
            vec![bar(2), bar(3), bar(5)],
        ])
        .expect("histories should align");

        let expected = vec![bar(2).date, bar(3).date];
        assert_eq!(aligned.len(), 3);
        for bars in aligned {
            assert_eq!(
                bars.iter().map(|bar| bar.date).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn single_ticker_preserves_the_original_series() {
        let original = vec![bar(0), bar(2), bar(5)];
        let aligned = align_bars_to_common_timestamps(vec![original.clone()])
            .expect("single ticker should remain valid");

        assert_eq!(aligned, vec![original]);
    }

    #[test]
    fn rejects_duplicate_or_non_monotonic_timestamps() {
        let duplicate = align_bars_to_common_timestamps(vec![vec![bar(0), bar(0)]])
            .expect_err("duplicate timestamps must be rejected");
        assert!(duplicate.to_string().contains("duplicate timestamp"));

        let reversed = align_bars_to_common_timestamps(vec![vec![bar(1), bar(0)]])
            .expect_err("non-monotonic timestamps must be rejected");
        assert!(reversed.to_string().contains("not strictly chronological"));
    }

    #[test]
    fn rejects_histories_without_a_common_timestamp() {
        let error = align_bars_to_common_timestamps(vec![vec![bar(0)], vec![bar(1)]])
            .expect_err("disjoint histories must be rejected");

        assert!(error.to_string().contains("no timestamps in common"));
    }
}
