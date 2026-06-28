use super::{calc_growth, EarningsReport};
use serde::Deserialize;

const BASE_URL: &str = "https://finnhub.io/api/v1";

#[derive(Debug, Deserialize)]
struct EarningsEntry {
    #[serde(rename = "actual")]
    eps_actual: Option<f64>,
    #[serde(rename = "estimate")]
    eps_estimate: Option<f64>,
    period: String, // "2024-01-31"
    symbol: String,
    #[serde(rename = "surprise")]
    eps_surprise: Option<f64>,
    #[serde(rename = "surprisePercent")]
    surprise_percent: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct BasicFinancials {
    metric: Option<FinancialMetrics>,
    series: Option<FinancialSeries>,
}

#[derive(Debug, Deserialize)]
struct FinancialMetrics {
    #[serde(rename = "revenuePerShareAnnual")]
    revenue_per_share: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct FinancialSeries {
    quarterly: Option<QuarterlySeries>,
}

#[derive(Debug, Deserialize)]
struct QuarterlySeries {
    revenue: Option<Vec<SeriesPoint>>,
    #[serde(rename = "netIncome")]
    net_income: Option<Vec<SeriesPoint>>,
}

#[derive(Debug, Deserialize)]
struct SeriesPoint {
    period: String,
    v: f64,
}

pub fn fetch(ticker: &str, api_key: &str) -> Vec<EarningsReport> {
    let client = reqwest::blocking::Client::new();

    // Fetch earnings (EPS data)
    let earnings_url = format!(
        "{}/stock/earnings?symbol={}&token={}",
        BASE_URL, ticker, api_key
    );
    let earnings: Vec<EarningsEntry> = client
        .get(&earnings_url)
        .send()
        .ok()
        .and_then(|r| r.json().ok())
        .unwrap_or_default();

    // Fetch basic financials for revenue/net income series
    let financials_url = format!(
        "{}/stock/metric?symbol={}&metric=all&token={}",
        BASE_URL, ticker, api_key
    );
    let financials: BasicFinancials = client
        .get(&financials_url)
        .send()
        .ok()
        .and_then(|r| r.json().ok())
        .unwrap_or(BasicFinancials {
            metric: None,
            series: None,
        });

    // Build revenue/net_income lookups by period
    let revenue_map: std::collections::HashMap<String, f64> = financials
        .series
        .as_ref()
        .and_then(|s| s.quarterly.as_ref())
        .and_then(|q| q.revenue.as_ref())
        .map(|v| v.iter().map(|p| (p.period.clone(), p.v)).collect())
        .unwrap_or_default();

    let net_income_map: std::collections::HashMap<String, f64> = financials
        .series
        .as_ref()
        .and_then(|s| s.quarterly.as_ref())
        .and_then(|q| q.net_income.as_ref())
        .map(|v| v.iter().map(|p| (p.period.clone(), p.v)).collect())
        .unwrap_or_default();

    let mut reports: Vec<EarningsReport> = Vec::with_capacity(earnings.len());

    for (i, entry) in earnings.iter().enumerate() {
        let prev = earnings.get(i + 1);

        let revenue = revenue_map.get(&entry.period).copied();
        let prev_revenue = prev.and_then(|p| revenue_map.get(&p.period).copied());

        let net_income = net_income_map.get(&entry.period).copied();
        let prev_net_income = prev.and_then(|p| net_income_map.get(&p.period).copied());

        let eps_surprise_pct = entry.surprise_percent.map(|p| p / 100.0);

        reports.push(EarningsReport {
            date: entry.period.clone(),
            symbol: entry.symbol.clone(),
            revenue,
            revenue_growth: calc_growth(revenue, prev_revenue),
            operating_expenses: None, // Not available in Finnhub free tier
            opex_growth: None,
            net_income,
            net_income_growth: calc_growth(net_income, prev_net_income),
            eps: entry.eps_actual,
            eps_estimated: entry.eps_estimate,
            eps_surprise: eps_surprise_pct,
        });
    }

    reports.reverse();
    reports
}
