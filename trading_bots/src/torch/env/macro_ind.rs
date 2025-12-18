use crate::data::macro_econ::{MacroSeries, get_macro_data, MacroObservation};

/// Precomputed macroeconomic indicators aligned to bar dates
/// Uses publication lag to avoid lookahead bias
pub struct MacroIndicators {
    pub gdp_growth: Vec<f64>,
    pub unemployment: Vec<f64>,
    pub jobs_growth: Vec<f64>,
    pub cpi_yoy: Vec<f64>,
    pub core_cpi_yoy: Vec<f64>,
    pub fed_funds: Vec<f64>,
    pub treasury_10y: Vec<f64>,
    pub yield_spread: Vec<f64>,
    pub consumer_sentiment: Vec<f64>,
    pub initial_claims: Vec<f64>,
    pub steps_to_jobs: Vec<f64>,
    pub steps_to_cpi: Vec<f64>,
    pub steps_to_fomc: Vec<f64>,
    pub steps_to_gdp: Vec<f64>,
}

impl MacroIndicators {
    pub fn empty(n: usize) -> Self {
        Self {
            gdp_growth: vec![0.0; n],
            unemployment: vec![0.0; n],
            jobs_growth: vec![0.0; n],
            cpi_yoy: vec![0.0; n],
            core_cpi_yoy: vec![0.0; n],
            fed_funds: vec![0.0; n],
            treasury_10y: vec![0.0; n],
            yield_spread: vec![0.0; n],
            consumer_sentiment: vec![0.0; n],
            initial_claims: vec![0.0; n],
            steps_to_jobs: vec![0.0; n],
            steps_to_cpi: vec![0.0; n],
            steps_to_fomc: vec![0.0; n],
            steps_to_gdp: vec![0.0; n],
        }
    }

    pub fn compute(bar_dates: &[String]) -> Self {
        let n = bar_dates.len();
        if n == 0 {
            return Self::empty(0);
        }

        let gdp_data = get_macro_data(MacroSeries::GdpGrowth);
        let unemp_data = get_macro_data(MacroSeries::UnemploymentRate);
        let payrolls_data = get_macro_data(MacroSeries::JobsGrowth);
        let cpi_data = get_macro_data(MacroSeries::CpiAllItems);
        let core_cpi_data = get_macro_data(MacroSeries::CoreCpi);
        let fed_data = get_macro_data(MacroSeries::FedFundsRate);
        let t10y_data = get_macro_data(MacroSeries::Treasury10Y);
        let t2y_data = get_macro_data(MacroSeries::Treasury2Y);
        let sentiment_data = get_macro_data(MacroSeries::ConsumerSentiment);
        let claims_data = get_macro_data(MacroSeries::InitialClaims);

        let mut result = Self::empty(n);

        for (i, bar_date) in bar_dates.iter().enumerate() {
            result.gdp_growth[i] = get_lagged_value(&gdp_data.observations, bar_date, 30)
                .map(|v| (v / 10.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.unemployment[i] = get_lagged_value(&unemp_data.observations, bar_date, 5)
                .map(|v| ((v - 5.0) / 5.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.jobs_growth[i] = get_lagged_value(&payrolls_data.observations, bar_date, 5)
                .map(|v| (v / 2.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.cpi_yoy[i] = get_yoy_change(&cpi_data.observations, bar_date, 15)
                .map(|v| ((v - 2.0) / 5.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.core_cpi_yoy[i] = get_yoy_change(&core_cpi_data.observations, bar_date, 15)
                .map(|v| ((v - 2.0) / 5.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.fed_funds[i] = get_lagged_value(&fed_data.observations, bar_date, 0)
                .map(|v| ((v - 3.0) / 3.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            let t10y = get_lagged_value(&t10y_data.observations, bar_date, 0);
            let t2y = get_lagged_value(&t2y_data.observations, bar_date, 0);

            result.treasury_10y[i] = t10y
                .map(|v| ((v - 3.0) / 3.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.yield_spread[i] = match (t10y, t2y) {
                (Some(l), Some(s)) => ((l - s) / 2.0).clamp(-1.0, 1.0),
                _ => 0.0,
            };

            result.consumer_sentiment[i] = get_lagged_value(&sentiment_data.observations, bar_date, 15)
                .map(|v| ((v - 90.0) / 30.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.initial_claims[i] = get_lagged_value(&claims_data.observations, bar_date, 5)
                .map(|v| ((v - 250.0) / 200.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.steps_to_jobs[i] = days_to_first_friday(bar_date) as f64 / 31.0;
            result.steps_to_cpi[i] = days_to_day_of_month(bar_date, 13) as f64 / 31.0;
            result.steps_to_fomc[i] = days_to_next_fomc(bar_date) as f64 / 50.0;
            result.steps_to_gdp[i] = days_to_gdp_release(bar_date) as f64 / 90.0;
        }

        result
    }
}

fn get_lagged_value(obs: &[MacroObservation], bar_date: &str, lag_days: i32) -> Option<f64> {
    let effective_date = subtract_days(bar_date, lag_days);
    obs.iter()
        .rev()
        .find(|o| o.date.as_str() <= effective_date.as_str())
        .and_then(|o| o.value)
}

fn get_yoy_change(obs: &[MacroObservation], bar_date: &str, lag_days: i32) -> Option<f64> {
    let effective_date = subtract_days(bar_date, lag_days);
    let year_ago = subtract_days(&effective_date, 365);

    let current = obs.iter()
        .rev()
        .find(|o| o.date.as_str() <= effective_date.as_str())
        .and_then(|o| o.value)?;

    let prev = obs.iter()
        .rev()
        .find(|o| o.date.as_str() <= year_ago.as_str())
        .and_then(|o| o.value)?;

    if prev.abs() > 0.001 {
        Some((current / prev - 1.0) * 100.0)
    } else {
        None
    }
}

fn subtract_days(date: &str, days: i32) -> String {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return date.to_string();
    }
    let y: i32 = parts[0].parse().unwrap_or(2020);
    let m: i32 = parts[1].parse().unwrap_or(1);
    let d: i32 = parts[2].parse().unwrap_or(1);

    let total_days = y * 365 + m * 30 + d - days;
    let new_y = total_days / 365;
    let rem = total_days % 365;
    let new_m = (rem / 30).clamp(1, 12);
    let new_d = (rem % 30).max(1);

    format!("{:04}-{:02}-{:02}", new_y, new_m, new_d)
}

fn days_to_first_friday(date: &str) -> i32 {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 { return 15; }
    let y: i32 = parts[0].parse().unwrap_or(2020);
    let m: i32 = parts[1].parse().unwrap_or(1);
    let d: i32 = parts[2].parse().unwrap_or(1);

    let dow_of_first = (3 + (y - 2020) * 365 + (m - 1) * 30 + 1) % 7;
    let first_friday = 1 + (5 - dow_of_first + 7) % 7;

    if d <= first_friday {
        first_friday - d
    } else {
        let dow_next_first = (dow_of_first + 30) % 7;
        let next_first_friday = 1 + (5 - dow_next_first + 7) % 7;
        (30 - d) + next_first_friday
    }
}

fn days_to_day_of_month(date: &str, target_day: i32) -> i32 {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 { return 15; }
    let d: i32 = parts[2].parse().unwrap_or(1);

    if d <= target_day {
        target_day - d
    } else {
        (30 - d) + target_day
    }
}

fn days_to_next_fomc(date: &str) -> i32 {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 { return 25; }
    let m: i32 = parts[1].parse().unwrap_or(1);
    let d: i32 = parts[2].parse().unwrap_or(1);

    let fomc_months = [1, 3, 5, 6, 7, 9, 11, 12];
    let fomc_day = 15;

    for &fm in &fomc_months {
        if fm > m || (fm == m && fomc_day > d) {
            return (fm - m) * 30 + (fomc_day - d);
        }
    }
    (12 - m + 1) * 30 + (fomc_day - d)
}

fn days_to_gdp_release(date: &str) -> i32 {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 { return 45; }
    let m: i32 = parts[1].parse().unwrap_or(1);
    let d: i32 = parts[2].parse().unwrap_or(1);

    let gdp_months = [1, 4, 7, 10];
    let gdp_day = 28;

    for &gm in &gdp_months {
        if gm > m || (gm == m && gdp_day > d) {
            return (gm - m) * 30 + (gdp_day - d);
        }
    }
    (12 - m + 1) * 30 + (gdp_day - d)
}
