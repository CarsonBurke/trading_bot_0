mod alpha_vantage;
mod finnhub;
mod fmp;

use crate::constants::files::DATA_PATH;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarningsReport {
    pub date: String, // "2024-01-25"
    pub symbol: String,
    pub revenue: Option<f64>,
    pub revenue_growth: Option<f64>, // QoQ
    pub operating_expenses: Option<f64>,
    pub opex_growth: Option<f64>, // QoQ
    pub net_income: Option<f64>,
    pub net_income_growth: Option<f64>, // QoQ
    pub eps: Option<f64>,
    pub eps_estimated: Option<f64>,
    pub eps_surprise: Option<f64>, // (actual - estimated) / |estimated|
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarningsProvider {
    Fmp,
    Finnhub,
    AlphaVantage,
}

impl EarningsProvider {
    fn cache_suffix(&self) -> &'static str {
        match self {
            Self::Fmp => "fmp",
            Self::Finnhub => "finnhub",
            Self::AlphaVantage => "alphavantage",
        }
    }

    fn env_key(&self) -> &'static str {
        match self {
            Self::Fmp => "FMP_API_KEY",
            Self::Finnhub => "FINNHUB_API_KEY",
            Self::AlphaVantage => "ALPHA_VANTAGE_API_KEY",
        }
    }

    fn fetch(&self, ticker: &str, api_key: &str) -> Vec<EarningsReport> {
        match self {
            Self::Fmp => fmp::fetch(ticker, api_key),
            Self::Finnhub => finnhub::fetch(ticker, api_key),
            Self::AlphaVantage => alpha_vantage::fetch(ticker, api_key),
        }
    }
}

pub fn get_earnings_data(ticker: &str, provider: EarningsProvider) -> Vec<EarningsReport> {
    if let Some(reports) = load_from_cache(ticker, provider) {
        return reports;
    }

    // let key_maybe = std::env::var(provider.env_key()).ok();
    let key_maybe = Some("RRI6UL6LP5L6E9ZQ");
    let Some(key) = key_maybe else {
        eprintln!(
            "No {} set, returning empty earnings for {ticker}",
            provider.env_key()
        );
        return Vec::new();
    };

    let reports = provider.fetch(ticker, &key);
    if !reports.is_empty() {
        save_to_cache(ticker, provider, &reports);
    }
    reports
}

/// Try providers in order until one succeeds (with cache or API key)
/// First check all caches, then try API calls
pub fn get_earnings_data_any(ticker: &str) -> Vec<EarningsReport> {
    const PROVIDERS: [EarningsProvider; 3] = [
        EarningsProvider::AlphaVantage,
        EarningsProvider::Finnhub,
        EarningsProvider::Fmp,
    ];
    // Check caches first (fast, no network)
    for provider in PROVIDERS {
        if let Some(reports) = load_from_cache(ticker, provider) {
            if !reports.is_empty() {
                return reports;
            }
        }
    }
    // Only then try API calls
    for provider in PROVIDERS {
        let reports = get_earnings_data(ticker, provider);
        if !reports.is_empty() {
            return reports;
        }
    }
    Vec::new()
}

pub fn get_cached_earnings_data_any(ticker: &str) -> Vec<EarningsReport> {
    const PROVIDERS: [EarningsProvider; 3] = [
        EarningsProvider::AlphaVantage,
        EarningsProvider::Finnhub,
        EarningsProvider::Fmp,
    ];
    for provider in PROVIDERS {
        if let Some(reports) = load_from_cache(ticker, provider) {
            if !reports.is_empty() {
                return reports;
            }
        }
    }
    Vec::new()
}

fn cache_path(ticker: &str, provider: EarningsProvider) -> String {
    format!(
        "{}/{}_earnings_{}.bin",
        DATA_PATH,
        ticker,
        provider.cache_suffix()
    )
}

fn load_from_cache(ticker: &str, provider: EarningsProvider) -> Option<Vec<EarningsReport>> {
    let path = cache_path(ticker, provider);
    let file = fs::read(path).ok()?;
    postcard::from_bytes(&file).ok()
}

fn save_to_cache(ticker: &str, provider: EarningsProvider, reports: &[EarningsReport]) {
    let path = cache_path(ticker, provider);
    if let Ok(encoded) = postcard::to_allocvec(reports) {
        let _ = fs::write(path, encoded);
    }
}

pub(crate) fn calc_growth(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    match (current, previous) {
        (Some(c), Some(p)) if p.abs() > 0.001 => Some((c - p) / p.abs()),
        _ => None,
    }
}
