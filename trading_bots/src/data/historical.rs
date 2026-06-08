use std::{
    borrow::Cow,
    fs,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock, RwLock,
    },
};

use hashbrown::HashMap;
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

    data
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
    create_folder_if_not_exists(&files::DATA_PATH.to_string());

    println!("Downloading data for {ticker}");
    let ibkr_symbol = ibkr_symbol(ticker);
    let contract = Contract::stock(ibkr_symbol.as_ref()).build();

    let historical_data = client
        .historical_data(
            &contract,
            Some(OffsetDateTime::now_utc()),
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
