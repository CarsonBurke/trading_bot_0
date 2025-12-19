use std::sync::{Arc, OnceLock};
use std::fs;
use serde::{Serialize, Deserialize};
use crate::data::macro_econ::{MacroSeries, get_macro_data, MacroObservation};

/// Convert YYYY-MM-DD to sortable integer (approximate day number)
#[inline]
fn date_to_int(date: &str) -> i32 {
    let b = date.as_bytes();
    if b.len() < 10 { return 0; }
    let y = (b[0] - b'0') as i32 * 1000 + (b[1] - b'0') as i32 * 100
          + (b[2] - b'0') as i32 * 10 + (b[3] - b'0') as i32;
    let m = (b[5] - b'0') as i32 * 10 + (b[6] - b'0') as i32;
    let d = (b[8] - b'0') as i32 * 10 + (b[9] - b'0') as i32;
    y * 365 + m * 30 + d
}

/// Cache for MacroIndicators (compute once, share across envs)
static MACRO_CACHE: OnceLock<Arc<MacroIndicators>> = OnceLock::new();

const MACRO_CACHE_PATH: &str = "../long_data/macro_indicators.bin";

/// Precomputed macroeconomic indicators aligned to bar dates
/// Uses publication lag to avoid lookahead bias
#[derive(Serialize, Deserialize)]
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

    /// Get cached or compute macro indicators
    pub fn get_or_compute(bar_dates: &[String]) -> Arc<MacroIndicators> {
        MACRO_CACHE.get_or_init(|| {
            // Try disk cache first
            if let Ok(bytes) = fs::read(MACRO_CACHE_PATH) {
                if let Ok(cached) = postcard::from_bytes::<MacroIndicators>(&bytes) {
                    if cached.gdp_growth.len() == bar_dates.len() {
                        eprintln!("Loaded macro indicators from cache ({} dates)", bar_dates.len());
                        return Arc::new(cached);
                    }
                }
            }

            eprintln!("Computing macro indicators for {} dates...", bar_dates.len());
            let result = Self::compute_inner(bar_dates);

            // Save to disk cache
            if let Ok(bytes) = postcard::to_allocvec(&result) {
                let _ = fs::write(MACRO_CACHE_PATH, bytes);
                eprintln!("Saved macro indicators to cache");
            }

            Arc::new(result)
        }).clone()
    }

    fn compute_inner(bar_dates: &[String]) -> Self {
        let n = bar_dates.len();
        if n == 0 {
            return Self::empty(0);
        }

        // Precompute bar dates as integers (one-time O(n))
        let bar_ints: Vec<i32> = bar_dates.iter().map(|d| date_to_int(d)).collect();

        // Load and preprocess macro series (one-time O(m) per series)
        let gdp_obs = obs_to_ints(&get_macro_data(MacroSeries::GdpGrowth).observations);
        let unemp_obs = obs_to_ints(&get_macro_data(MacroSeries::UnemploymentRate).observations);
        let payrolls_obs = obs_to_ints(&get_macro_data(MacroSeries::JobsGrowth).observations);
        let cpi_obs = obs_to_ints(&get_macro_data(MacroSeries::CpiAllItems).observations);
        let core_cpi_obs = obs_to_ints(&get_macro_data(MacroSeries::CoreCpi).observations);
        let fed_obs = obs_to_ints(&get_macro_data(MacroSeries::FedFundsRate).observations);
        let t10y_obs = obs_to_ints(&get_macro_data(MacroSeries::Treasury10Y).observations);
        let t2y_obs = obs_to_ints(&get_macro_data(MacroSeries::Treasury2Y).observations);
        let sentiment_obs = obs_to_ints(&get_macro_data(MacroSeries::ConsumerSentiment).observations);
        let claims_obs = obs_to_ints(&get_macro_data(MacroSeries::InitialClaims).observations);

        let mut result = Self::empty(n);

        // Cursor-based linear scan - O(n + m) total
        let mut gdp_idx = 0usize;
        let mut unemp_idx = 0usize;
        let mut payrolls_idx = 0usize;
        let mut cpi_idx = 0usize;
        let mut core_cpi_idx = 0usize;
        let mut fed_idx = 0usize;
        let mut t10y_idx = 0usize;
        let mut t2y_idx = 0usize;
        let mut sentiment_idx = 0usize;
        let mut claims_idx = 0usize;
        let mut cpi_yoy_idx = 0usize;
        let mut core_cpi_yoy_idx = 0usize;

        for (i, &bar_int) in bar_ints.iter().enumerate() {
            result.gdp_growth[i] = advance_and_get(&gdp_obs, &mut gdp_idx, bar_int - 30)
                .map(|v| (v / 10.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.unemployment[i] = advance_and_get(&unemp_obs, &mut unemp_idx, bar_int - 5)
                .map(|v| ((v - 5.0) / 5.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.jobs_growth[i] = advance_and_get(&payrolls_obs, &mut payrolls_idx, bar_int - 5)
                .map(|v| (v / 2.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            let cpi_current = advance_and_get(&cpi_obs, &mut cpi_idx, bar_int - 15);
            let cpi_prev = advance_and_get(&cpi_obs, &mut cpi_yoy_idx, bar_int - 15 - 365);
            result.cpi_yoy[i] = match (cpi_current, cpi_prev) {
                (Some(c), Some(p)) if p.abs() > 0.001 => (((c / p - 1.0) * 100.0 - 2.0) / 5.0).clamp(-1.0, 1.0),
                _ => 0.0,
            };

            let core_current = advance_and_get(&core_cpi_obs, &mut core_cpi_idx, bar_int - 15);
            let core_prev = advance_and_get(&core_cpi_obs, &mut core_cpi_yoy_idx, bar_int - 15 - 365);
            result.core_cpi_yoy[i] = match (core_current, core_prev) {
                (Some(c), Some(p)) if p.abs() > 0.001 => (((c / p - 1.0) * 100.0 - 2.0) / 5.0).clamp(-1.0, 1.0),
                _ => 0.0,
            };

            result.fed_funds[i] = advance_and_get(&fed_obs, &mut fed_idx, bar_int)
                .map(|v| ((v - 3.0) / 3.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            let t10y = advance_and_get(&t10y_obs, &mut t10y_idx, bar_int);
            let t2y = advance_and_get(&t2y_obs, &mut t2y_idx, bar_int);

            result.treasury_10y[i] = t10y
                .map(|v| ((v - 3.0) / 3.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.yield_spread[i] = match (t10y, t2y) {
                (Some(l), Some(s)) => ((l - s) / 2.0).clamp(-1.0, 1.0),
                _ => 0.0,
            };

            result.consumer_sentiment[i] = advance_and_get(&sentiment_obs, &mut sentiment_idx, bar_int - 15)
                .map(|v| ((v - 90.0) / 30.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.initial_claims[i] = advance_and_get(&claims_obs, &mut claims_idx, bar_int - 5)
                .map(|v| ((v - 250.0) / 200.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.steps_to_jobs[i] = days_to_first_friday_int(bar_int) as f64 / 31.0;
            result.steps_to_cpi[i] = days_to_day_of_month_int(bar_int, 13) as f64 / 31.0;
            result.steps_to_fomc[i] = days_to_next_fomc_int(bar_int) as f64 / 50.0;
            result.steps_to_gdp[i] = days_to_gdp_release_int(bar_int) as f64 / 90.0;
        }

        result
    }
}

/// Precomputed observation with integer date
struct IntObs {
    date_int: i32,
    value: Option<f64>,
}

fn obs_to_ints(obs: &[MacroObservation]) -> Vec<IntObs> {
    obs.iter()
        .map(|o| IntObs {
            date_int: date_to_int(&o.date),
            value: o.value,
        })
        .collect()
}

/// Advance cursor to find most recent observation <= target_date, return its value
#[inline]
fn advance_and_get(obs: &[IntObs], cursor: &mut usize, target_date: i32) -> Option<f64> {
    if obs.is_empty() {
        return None;
    }
    while *cursor + 1 < obs.len() && obs[*cursor + 1].date_int <= target_date {
        *cursor += 1;
    }
    if obs[*cursor].date_int <= target_date {
        obs[*cursor].value
    } else {
        None
    }
}

/// Extract (y, m, d) from integer date representation
#[inline]
fn int_to_ymd(date_int: i32) -> (i32, i32, i32) {
    let y = date_int / 365;
    let rem = date_int % 365;
    let m = (rem / 30).clamp(1, 12);
    let d = (rem % 30).max(1);
    (y, m, d)
}

#[inline]
fn days_to_first_friday_int(date_int: i32) -> i32 {
    let (y, m, d) = int_to_ymd(date_int);
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

#[inline]
fn days_to_day_of_month_int(date_int: i32, target_day: i32) -> i32 {
    let (_, _, d) = int_to_ymd(date_int);
    if d <= target_day {
        target_day - d
    } else {
        (30 - d) + target_day
    }
}

#[inline]
fn days_to_next_fomc_int(date_int: i32) -> i32 {
    let (_, m, d) = int_to_ymd(date_int);
    const FOMC_MONTHS: [i32; 8] = [1, 3, 5, 6, 7, 9, 11, 12];
    const FOMC_DAY: i32 = 15;

    for &fm in &FOMC_MONTHS {
        if fm > m || (fm == m && FOMC_DAY > d) {
            return (fm - m) * 30 + (FOMC_DAY - d);
        }
    }
    (12 - m + 1) * 30 + (FOMC_DAY - d)
}

#[inline]
fn days_to_gdp_release_int(date_int: i32) -> i32 {
    let (_, m, d) = int_to_ymd(date_int);
    const GDP_MONTHS: [i32; 4] = [1, 4, 7, 10];
    const GDP_DAY: i32 = 28;

    for &gm in &GDP_MONTHS {
        if gm > m || (gm == m && GDP_DAY > d) {
            return (gm - m) * 30 + (GDP_DAY - d);
        }
    }
    (12 - m + 1) * 30 + (GDP_DAY - d)
}
