use super::{calc_growth, EarningsReport};
use serde::Deserialize;

const BASE_URL: &str = "https://www.alphavantage.co/query";

#[derive(Debug, Deserialize)]
struct EarningsResponse {
    #[serde(rename = "quarterlyEarnings")]
    quarterly_earnings: Option<Vec<QuarterlyEarning>>,
}

#[derive(Debug, Deserialize)]
struct QuarterlyEarning {
    #[serde(rename = "fiscalDateEnding")]
    fiscal_date: String,
    #[serde(rename = "reportedEPS")]
    reported_eps: Option<String>,
    #[serde(rename = "estimatedEPS")]
    estimated_eps: Option<String>,
    #[serde(rename = "surprise")]
    surprise: Option<String>,
    #[serde(rename = "surprisePercentage")]
    surprise_percentage: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IncomeResponse {
    #[serde(rename = "quarterlyReports")]
    quarterly_reports: Option<Vec<IncomeReport>>,
}

#[derive(Debug, Deserialize)]
struct IncomeReport {
    #[serde(rename = "fiscalDateEnding")]
    fiscal_date: String,
    #[serde(rename = "totalRevenue")]
    total_revenue: Option<String>,
    #[serde(rename = "operatingExpenses")]
    operating_expenses: Option<String>,
    #[serde(rename = "netIncome")]
    net_income: Option<String>,
}

fn parse_num(s: &Option<String>) -> Option<f64> {
    s.as_ref().and_then(|v| v.parse().ok())
}

pub fn fetch(ticker: &str, api_key: &str) -> Vec<EarningsReport> {
    let client = reqwest::blocking::Client::new();

    // Fetch earnings (EPS)
    let earnings_url = format!(
        "{}?function=EARNINGS&symbol={}&apikey={}",
        BASE_URL, ticker, api_key
    );
    let earnings_resp: EarningsResponse = client
        .get(&earnings_url)
        .send()
        .ok()
        .and_then(|r| r.json().ok())
        .unwrap_or(EarningsResponse {
            quarterly_earnings: None,
        });

    // Fetch income statement
    let income_url = format!(
        "{}?function=INCOME_STATEMENT&symbol={}&apikey={}",
        BASE_URL, ticker, api_key
    );
    let income_resp: IncomeResponse = client
        .get(&income_url)
        .send()
        .ok()
        .and_then(|r| r.json().ok())
        .unwrap_or(IncomeResponse {
            quarterly_reports: None,
        });

    let quarterly_earnings = earnings_resp.quarterly_earnings.unwrap_or_default();
    let quarterly_income = income_resp.quarterly_reports.unwrap_or_default();

    // Build income lookup by date
    let income_map: std::collections::HashMap<String, &IncomeReport> = quarterly_income
        .iter()
        .map(|r| (r.fiscal_date.clone(), r))
        .collect();

    let mut reports: Vec<EarningsReport> = Vec::with_capacity(quarterly_earnings.len());

    for (i, entry) in quarterly_earnings.iter().enumerate() {
        let prev = quarterly_earnings.get(i + 1);
        let income = income_map.get(&entry.fiscal_date);
        let prev_income = prev.and_then(|p| income_map.get(&p.fiscal_date));

        let revenue = income.and_then(|r| parse_num(&r.total_revenue));
        let prev_revenue = prev_income.and_then(|r| parse_num(&r.total_revenue));

        let opex = income.and_then(|r| parse_num(&r.operating_expenses));
        let prev_opex = prev_income.and_then(|r| parse_num(&r.operating_expenses));

        let net_income = income.and_then(|r| parse_num(&r.net_income));
        let prev_net_income = prev_income.and_then(|r| parse_num(&r.net_income));

        let eps_surprise = parse_num(&entry.surprise_percentage).map(|p| p / 100.0);

        reports.push(EarningsReport {
            date: entry.fiscal_date.clone(),
            symbol: ticker.to_string(),
            revenue,
            revenue_growth: calc_growth(revenue, prev_revenue),
            operating_expenses: opex,
            opex_growth: calc_growth(opex, prev_opex),
            net_income,
            net_income_growth: calc_growth(net_income, prev_net_income),
            eps: parse_num(&entry.reported_eps),
            eps_estimated: parse_num(&entry.estimated_eps),
            eps_surprise,
        });
    }

    reports.reverse();
    reports
}
