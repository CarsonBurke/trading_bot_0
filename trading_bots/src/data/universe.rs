use std::{path::Path, sync::OnceLock};

use shared::paths::DATA_PATH;

use crate::data::historical::get_cached_historical_bars;

pub const TARGET_UNIVERSE_TICKERS: &[&str] = &[
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "AVGO", "GOOG", "META", "TSLA", "WMT", "BRK.B", "JPM",
    "LLY", "XOM", "V", "MU", "JNJ", "ORCL", "AMD", "MA", "COST", "NFLX", "BAC", "CAT", "CVX",
    "PLTR", "CSCO", "ABBV", "HD", "PG", "LRCX", "INTC", "KO", "UNH", "AMAT", "GEV", "MS", "GE",
    "MRK", "GS", "PM", "WFC", "RTX", "KLAC", "IBM", "LIN", "AXP", "ANET", "C", "TXN", "MCD", "PEP",
    "TMUS", "VZ", "TMO", "NEE", "AMGN", "ADI", "DIS", "APH", "BA", "T", "TJX", "ISRG", "BLK",
    "GILD", "APP", "ETN", "SCHW", "ABT", "DE", "CRM", "UBER", "PFE", "COP", "UNP", "PANW", "QCOM",
    "GLW", "SNDK", "BKNG", "WELL", "DELL", "HON", "LOW", "SPGI", "WDC", "PLD", "DHR", "STX", "LMT",
    "CB", "SYK", "COF", "NEM", "PH", "BMY", "PGR", "CRWD", "VRT",
];

pub const AVAILABLE_TICKERS: &[&str] = TARGET_UNIVERSE_TICKERS;

static CACHED_ELIGIBLE_TRAINING_UNIVERSE: OnceLock<Vec<String>> = OnceLock::new();

pub fn minimum_history_bars() -> usize {
    const BARS_PER_TRADING_DAY: usize = 78;
    const TRADING_DAYS_PER_YEAR: usize = 252;
    const COVERAGE_RATIO: f64 = 0.90;

    let expected =
        match data_path_kind() {
            "data" => ((356.0 / 365.25) * (TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY) as f64)
                .round() as usize,
            "long_data" => 5 * TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY,
            "very_long_data" => 10 * TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY,
            "extra_long_data" => 20 * TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY,
            _ => panic!("unsupported data path: {DATA_PATH}"),
        };

    (expected as f64 * COVERAGE_RATIO).round() as usize
}

pub fn training_universe() -> &'static [&'static str] {
    TARGET_UNIVERSE_TICKERS
}

pub fn cached_eligible_training_universe() -> &'static [String] {
    CACHED_ELIGIBLE_TRAINING_UNIVERSE
        .get_or_init(|| {
            let min_bars = minimum_history_bars();
            TARGET_UNIVERSE_TICKERS
                .iter()
                .filter_map(|ticker| {
                    let bars = get_cached_historical_bars(ticker)?;
                    (bars.len() >= min_bars).then(|| (*ticker).to_string())
                })
                .collect()
        })
        .as_slice()
}

fn data_path_kind() -> &'static str {
    Path::new(DATA_PATH)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
}
