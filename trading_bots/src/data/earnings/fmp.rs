use super::{calc_growth, EarningsReport};
use serde::Deserialize;
use std::collections::HashMap;

const BASE_URL: &str = "https://financialmodelingprep.com/api/v3";

#[derive(Debug, Deserialize)]
struct IncomeStatement {
    date: String,
    symbol: String,
    revenue: Option<f64>,
    #[serde(rename = "operatingExpenses")]
    operating_expenses: Option<f64>,
    #[serde(rename = "netIncome")]
    net_income: Option<f64>,
    eps: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct EarningsCalendar {
    date: String,
    eps: Option<f64>,
    #[serde(rename = "epsEstimated")]
    eps_estimated: Option<f64>,
}

pub fn fetch(ticker: &str, api_key: &str) -> Vec<EarningsReport> {
    let client = reqwest::blocking::Client::new();

    let income_url = format!(
        "{}/income-statement/{}?period=quarter&limit=80&apikey={}",
        BASE_URL, ticker, api_key
    );
    let statements: Vec<IncomeStatement> = client
        .get(&income_url)
        .send()
        .ok()
        .and_then(|r| r.json().ok())
        .unwrap_or_default();

    let calendar_url = format!(
        "{}/historical/earning_calendar/{}?apikey={}",
        BASE_URL, ticker, api_key
    );
    let calendar: Vec<EarningsCalendar> = client
        .get(&calendar_url)
        .send()
        .ok()
        .and_then(|r| r.json().ok())
        .unwrap_or_default();

    let eps_estimates: HashMap<String, (Option<f64>, Option<f64>)> = calendar
        .iter()
        .map(|e| (e.date.clone(), (e.eps_estimated, e.eps)))
        .collect();

    let mut reports: Vec<EarningsReport> = Vec::with_capacity(statements.len());

    for (i, stmt) in statements.iter().enumerate() {
        let prev = statements.get(i + 1);

        let (eps_estimated, actual_eps) = eps_estimates
            .get(&stmt.date)
            .cloned()
            .unwrap_or((None, stmt.eps));

        let eps_surprise = match (actual_eps.or(stmt.eps), eps_estimated) {
            (Some(actual), Some(est)) if est.abs() > 0.001 => Some((actual - est) / est.abs()),
            _ => None,
        };

        reports.push(EarningsReport {
            date: stmt.date.clone(),
            symbol: stmt.symbol.clone(),
            revenue: stmt.revenue,
            revenue_growth: calc_growth(stmt.revenue, prev.and_then(|p| p.revenue)),
            operating_expenses: stmt.operating_expenses,
            opex_growth: calc_growth(
                stmt.operating_expenses,
                prev.and_then(|p| p.operating_expenses),
            ),
            net_income: stmt.net_income,
            net_income_growth: calc_growth(stmt.net_income, prev.and_then(|p| p.net_income)),
            eps: actual_eps.or(stmt.eps),
            eps_estimated,
            eps_surprise,
        });
    }

    reports.reverse();
    reports
}
