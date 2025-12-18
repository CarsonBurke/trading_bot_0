use crate::data::EarningsReport;

/// Precomputed earnings indicators per step (from cached quarterly reports)
#[derive(Debug)]
pub struct EarningsIndicators {
    pub steps_to_next: Vec<f64>,
    pub revenue_growth: Vec<f64>,
    pub opex_growth: Vec<f64>,
    pub net_profit_growth: Vec<f64>,
    pub eps: Vec<f64>,
    pub eps_surprise: Vec<f64>,
}

impl EarningsIndicators {
    pub fn empty(n: usize) -> Self {
        Self {
            steps_to_next: vec![0.0; n],
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

        let mut steps_to_next = vec![0.0; n];
        let mut revenue_growth = vec![0.0; n];
        let mut opex_growth = vec![0.0; n];
        let mut net_profit_growth = vec![0.0; n];
        let mut eps = vec![0.0; n];
        let mut eps_surprise = vec![0.0; n];

        let mut report_idx = 0;
        for (i, bar_date) in bar_dates.iter().enumerate() {
            while report_idx + 1 < reports.len() && reports[report_idx + 1].date <= *bar_date {
                report_idx += 1;
            }

            let report = &reports[report_idx];

            if report_idx + 1 < reports.len() {
                let next_date = &reports[report_idx + 1].date;
                let days_to_next = date_diff_days(bar_date, next_date).max(0) as f64;
                steps_to_next[i] = (days_to_next / 90.0).clamp(0.0, 1.0);
            }

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
            steps_to_next,
            revenue_growth,
            opex_growth,
            net_profit_growth,
            eps,
            eps_surprise,
        }
    }
}

fn date_diff_days(from: &str, to: &str) -> i32 {
    let parse = |s: &str| -> Option<i32> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let y: i32 = parts[0].parse().ok()?;
        let m: i32 = parts[1].parse().ok()?;
        let d: i32 = parts[2].parse().ok()?;
        Some(y * 365 + m * 30 + d)
    };
    match (parse(from), parse(to)) {
        (Some(f), Some(t)) => t - f,
        _ => 0,
    }
}
