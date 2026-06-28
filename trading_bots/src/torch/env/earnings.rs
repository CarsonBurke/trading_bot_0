use crate::data::EarningsReport;
use chrono::{Duration, NaiveDate};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const FUNDAMENTAL_AVAILABILITY_LAG_DAYS: i64 = 90;

/// Global cache for earnings indicators (ticker -> indicators)
static EARNINGS_CACHE: OnceLock<Mutex<HashMap<String, Arc<EarningsIndicators>>>> = OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<String, Arc<EarningsIndicators>>> {
    EARNINGS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Precomputed earnings indicators per step (from cached quarterly reports)
#[derive(Debug)]
pub struct EarningsIndicators {
    pub steps_since_available: Vec<f64>,
    pub revenue_growth: Vec<f64>,
    pub opex_growth: Vec<f64>,
    pub net_profit_growth: Vec<f64>,
    pub eps: Vec<f64>,
    pub eps_surprise: Vec<f64>,
}

impl EarningsIndicators {
    pub fn get_cached(ticker: &str, prices_len: usize) -> Option<Arc<EarningsIndicators>> {
        let cache = get_cache();
        let locked = cache.lock().unwrap();
        locked
            .get(ticker)
            .filter(|cached| cached.eps.len() == prices_len)
            .cloned()
    }

    /// Get cached earnings indicators or compute if not present
    pub fn get_or_compute(
        ticker: &str,
        reports: &[EarningsReport],
        bar_dates: &[String],
        prices: &[f64],
    ) -> Arc<EarningsIndicators> {
        if let Some(cached) = Self::get_cached(ticker, prices.len()) {
            return cached;
        }
        let cache = get_cache();
        let computed = if reports.is_empty() {
            Arc::new(Self::empty(prices.len()))
        } else {
            Arc::new(Self::compute(reports, bar_dates, prices))
        };
        cache
            .lock()
            .unwrap()
            .insert(ticker.to_string(), computed.clone());
        computed
    }

    pub fn empty(n: usize) -> Self {
        Self {
            steps_since_available: vec![0.0; n],
            revenue_growth: vec![0.0; n],
            opex_growth: vec![0.0; n],
            net_profit_growth: vec![0.0; n],
            eps: vec![0.0; n],
            eps_surprise: vec![0.0; n],
        }
    }

    pub fn compute(reports: &[EarningsReport], bar_dates: &[String], prices: &[f64]) -> Self {
        let n = bar_dates.len();
        if reports.is_empty() {
            return Self::empty(n);
        }

        let mut steps_since_available = vec![0.0; n];
        let mut revenue_growth = vec![0.0; n];
        let mut opex_growth = vec![0.0; n];
        let mut net_profit_growth = vec![0.0; n];
        let mut eps = vec![0.0; n];
        let mut eps_surprise = vec![0.0; n];

        let mut available_reports: Vec<_> = reports
            .iter()
            .filter_map(|report| {
                report_available_date(report).map(|available_date| AvailableReport {
                    available_date,
                    report,
                })
            })
            .collect();
        available_reports.sort_by_key(|entry| entry.available_date);
        if available_reports.is_empty() {
            return Self::empty(n);
        }

        let mut report_idx = 0;
        for (i, bar_date) in bar_dates.iter().enumerate() {
            let Some(bar_date) = parse_date(bar_date) else {
                continue;
            };

            while report_idx + 1 < available_reports.len()
                && available_reports[report_idx + 1].available_date <= bar_date
            {
                report_idx += 1;
            }

            if available_reports[report_idx].available_date > bar_date {
                continue;
            }

            let entry = &available_reports[report_idx];
            let report = entry.report;

            let days_since_available = (bar_date - entry.available_date).num_days().max(0) as f64;
            steps_since_available[i] = (days_since_available / 90.0).clamp(0.0, 1.0);

            revenue_growth[i] = report.revenue_growth.unwrap_or(0.0).clamp(-1.0, 1.0);
            opex_growth[i] = report.opex_growth.unwrap_or(0.0).clamp(-1.0, 1.0);
            net_profit_growth[i] = report.net_income_growth.unwrap_or(0.0).clamp(-1.0, 1.0);

            if let Some(e) = report.eps {
                let price = prices[i].max(1.0);
                eps[i] = (e / price * 4.0).clamp(-0.5, 0.5);
            }

            eps_surprise[i] = report.eps_surprise.unwrap_or(0.0).clamp(-1.0, 1.0);
        }

        Self {
            steps_since_available,
            revenue_growth,
            opex_growth,
            net_profit_growth,
            eps,
            eps_surprise,
        }
    }
}

struct AvailableReport<'a> {
    available_date: NaiveDate,
    report: &'a EarningsReport,
}

fn report_available_date(report: &EarningsReport) -> Option<NaiveDate> {
    parse_date(&report.date).map(|date| date + Duration::days(FUNDAMENTAL_AVAILABILITY_LAG_DAYS))
}

fn parse_date(date: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()
}

#[cfg(test)]
mod tests {
    use super::{EarningsIndicators, FUNDAMENTAL_AVAILABILITY_LAG_DAYS};
    use crate::data::EarningsReport;
    use chrono::{Duration, NaiveDate};

    fn report(date: &str, revenue_growth: f64) -> EarningsReport {
        EarningsReport {
            date: date.to_string(),
            symbol: "TEST".to_string(),
            revenue: None,
            revenue_growth: Some(revenue_growth),
            operating_expenses: None,
            opex_growth: Some(-0.25),
            net_income: None,
            net_income_growth: Some(0.75),
            eps: Some(4.0),
            eps_estimated: Some(3.0),
            eps_surprise: Some(0.25),
        }
    }

    #[test]
    fn fundamentals_are_hidden_until_conservative_availability_lag() {
        let fiscal_date = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let availability_date = fiscal_date + Duration::days(FUNDAMENTAL_AVAILABILITY_LAG_DAYS);
        let before_date = availability_date - Duration::days(1);
        let bar_dates = vec![
            before_date.format("%Y-%m-%d").to_string(),
            availability_date.format("%Y-%m-%d").to_string(),
        ];
        let prices = vec![100.0, 100.0];

        let indicators =
            EarningsIndicators::compute(&[report("2024-01-31", 0.5)], &bar_dates, &prices);

        assert_eq!(indicators.steps_since_available[0], 0.0);
        assert_eq!(indicators.revenue_growth[0], 0.0);
        assert_eq!(indicators.opex_growth[0], 0.0);
        assert_eq!(indicators.net_profit_growth[0], 0.0);
        assert_eq!(indicators.eps[0], 0.0);
        assert_eq!(indicators.eps_surprise[0], 0.0);

        assert_eq!(indicators.steps_since_available[1], 0.0);
        assert_eq!(indicators.revenue_growth[1], 0.5);
        assert_eq!(indicators.opex_growth[1], -0.25);
        assert_eq!(indicators.net_profit_growth[1], 0.75);
        assert!((indicators.eps[1] - 0.16).abs() < 1e-12);
        assert_eq!(indicators.eps_surprise[1], 0.25);
    }

    #[test]
    fn current_report_remains_active_until_next_lagged_availability() {
        let first_date = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let second_date = NaiveDate::from_ymd_opt(2024, 4, 30).unwrap();
        let first_available = first_date + Duration::days(FUNDAMENTAL_AVAILABILITY_LAG_DAYS);
        let second_available = second_date + Duration::days(FUNDAMENTAL_AVAILABILITY_LAG_DAYS);
        let before_second = second_available - Duration::days(1);
        let bar_dates = vec![
            first_available.format("%Y-%m-%d").to_string(),
            before_second.format("%Y-%m-%d").to_string(),
            second_available.format("%Y-%m-%d").to_string(),
        ];
        let prices = vec![100.0; bar_dates.len()];
        let reports = [report("2024-01-31", 0.25), report("2024-04-30", -0.5)];

        let indicators = EarningsIndicators::compute(&reports, &bar_dates, &prices);

        assert_eq!(indicators.revenue_growth[0], 0.25);
        assert_eq!(indicators.revenue_growth[1], 0.25);
        assert_eq!(indicators.revenue_growth[2], -0.5);
    }

    #[test]
    fn staleness_grows_from_availability_and_clamps_at_full_scale() {
        let fiscal_date = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let availability_date = fiscal_date + Duration::days(FUNDAMENTAL_AVAILABILITY_LAG_DAYS);
        let before_date = availability_date - Duration::days(1);
        let just_after = availability_date + Duration::days(9);
        let long_after = availability_date + Duration::days(120);
        let bar_dates = vec![
            before_date.format("%Y-%m-%d").to_string(),
            availability_date.format("%Y-%m-%d").to_string(),
            just_after.format("%Y-%m-%d").to_string(),
            long_after.format("%Y-%m-%d").to_string(),
        ];
        let prices = vec![100.0; bar_dates.len()];

        let indicators =
            EarningsIndicators::compute(&[report("2024-01-31", 0.5)], &bar_dates, &prices);

        // Before the report is available: default value.
        assert_eq!(indicators.steps_since_available[0], 0.0);
        // On the availability date: zero days stale.
        assert_eq!(indicators.steps_since_available[1], 0.0);
        // Shortly after: small, proportional staleness (9 / 90).
        assert!((indicators.steps_since_available[2] - 0.1).abs() < 1e-12);
        // Long after (>= 90 days): clamped to the max scale.
        assert_eq!(indicators.steps_since_available[3], 1.0);
    }
}
