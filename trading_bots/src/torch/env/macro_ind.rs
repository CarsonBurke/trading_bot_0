use crate::data::macro_econ::{get_macro_data, MacroDataError, MacroObservation, MacroSeries};
use chrono::{Datelike, Duration, Months, NaiveDate, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use time::OffsetDateTime;

/// Cache for MacroIndicators keyed by the exact bar-timestamp sequence.
static MACRO_CACHE: OnceLock<Mutex<HashMap<u64, Arc<MacroIndicators>>>> = OnceLock::new();
static MACRO_SOURCE: OnceLock<Result<MacroSourceData, MacroDataError>> = OnceLock::new();

fn macro_cache() -> &'static Mutex<HashMap<u64, Arc<MacroIndicators>>> {
    MACRO_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn timestamp_sequence_key(bar_times: &[OffsetDateTime]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bar_times.len().hash(&mut hasher);
    for timestamp in bar_times {
        timestamp.unix_timestamp_nanos().hash(&mut hasher);
    }
    hasher.finish()
}

struct MacroSourceData {
    gdp_obs: Vec<IntObs>,
    unemp_obs: Vec<IntObs>,
    payrolls_obs: Vec<IntObs>,
    cpi_obs: Vec<IntObs>,
    cpi_by_period: HashMap<NaiveDate, IntObs>,
    core_cpi_obs: Vec<IntObs>,
    core_cpi_by_period: HashMap<NaiveDate, IntObs>,
    fed_obs: Vec<IntObs>,
    t10y_obs: Vec<IntObs>,
    t2y_obs: Vec<IntObs>,
    sentiment_obs: Vec<IntObs>,
    claims_obs: Vec<IntObs>,
}

fn macro_source() -> Result<&'static MacroSourceData, &'static MacroDataError> {
    MACRO_SOURCE
        .get_or_init(|| {
            let cpi_obs = load_series(MacroSeries::CpiAllItems)?;
            let core_cpi_obs = load_series(MacroSeries::CoreCpi)?;
            Ok(MacroSourceData {
                gdp_obs: load_series(MacroSeries::GdpGrowth)?,
                unemp_obs: load_series(MacroSeries::UnemploymentRate)?,
                payrolls_obs: load_series(MacroSeries::JobsGrowth)?,
                cpi_by_period: index_by_period(&cpi_obs),
                cpi_obs,
                core_cpi_by_period: index_by_period(&core_cpi_obs),
                core_cpi_obs,
                fed_obs: load_series(MacroSeries::FedFundsRate)?,
                t10y_obs: load_series(MacroSeries::Treasury10Y)?,
                t2y_obs: load_series(MacroSeries::Treasury2Y)?,
                sentiment_obs: load_series(MacroSeries::ConsumerSentiment)?,
                claims_obs: load_series(MacroSeries::InitialClaims)?,
            })
        })
        .as_ref()
}

/// Precomputed macroeconomic indicators aligned to bar timestamps.
/// Uses each observation's causal availability timestamp to avoid lookahead bias.
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
    pub fn get_or_compute(bar_times: &[OffsetDateTime]) -> Arc<MacroIndicators> {
        let key = timestamp_sequence_key(bar_times);
        {
            let locked = macro_cache().lock().unwrap();
            if let Some(cached) = locked.get(&key) {
                return cached.clone();
            }
        }

        eprintln!(
            "Computing macro indicators for {} dates (key {key:016x})",
            bar_times.len()
        );
        let result = Self::compute_inner(bar_times).unwrap_or_else(|err| {
            panic!("failed to initialize required macroeconomic features: {err}")
        });
        let result = Arc::new(result);
        macro_cache().lock().unwrap().insert(key, result.clone());
        result
    }

    fn compute_inner(bar_times: &[OffsetDateTime]) -> Result<Self, MacroDataError> {
        let n = bar_times.len();
        if n == 0 {
            return Ok(Self::empty(0));
        }

        let source = macro_source().map_err(Clone::clone)?;
        Ok(Self::compute_with_source(bar_times, source))
    }

    fn compute_with_source(bar_times: &[OffsetDateTime], source: &MacroSourceData) -> Self {
        let n = bar_times.len();

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
        for (i, &bar_time) in bar_times.iter().enumerate() {
            let bar_timestamp = bar_time.unix_timestamp();
            let bar_date = NaiveDate::from_ymd_opt(
                bar_time.year(),
                bar_time.month() as u8 as u32,
                bar_time.day() as u32,
            )
            .expect("OffsetDateTime always contains a valid date");
            result.gdp_growth[i] = advance_and_get(&source.gdp_obs, &mut gdp_idx, bar_timestamp)
                .map(|v| (v / 10.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.unemployment[i] =
                advance_and_get(&source.unemp_obs, &mut unemp_idx, bar_timestamp)
                    .map(|v| ((v - 5.0) / 5.0).clamp(-1.0, 1.0))
                    .unwrap_or(0.0);

            result.jobs_growth[i] =
                advance_and_get(&source.payrolls_obs, &mut payrolls_idx, bar_timestamp)
                    .map(|v| (v / 2.0).clamp(-1.0, 1.0))
                    .unwrap_or(0.0);

            let cpi_current = advance_to_latest(&source.cpi_obs, &mut cpi_idx, bar_timestamp);
            let cpi_prev = cpi_current.and_then(|current| {
                previous_year_period(current.period_date).and_then(|period| {
                    released_period_value(&source.cpi_by_period, period, bar_timestamp)
                })
            });
            result.cpi_yoy[i] = match (cpi_current, cpi_prev) {
                (Some(IntObs { value: Some(c), .. }), Some(p)) if p.abs() > 0.001 => {
                    (((c / p - 1.0) * 100.0 - 2.0) / 5.0).clamp(-1.0, 1.0)
                }
                _ => 0.0,
            };

            let core_current =
                advance_to_latest(&source.core_cpi_obs, &mut core_cpi_idx, bar_timestamp);
            let core_prev = core_current.and_then(|current| {
                previous_year_period(current.period_date).and_then(|period| {
                    released_period_value(&source.core_cpi_by_period, period, bar_timestamp)
                })
            });
            result.core_cpi_yoy[i] = match (core_current, core_prev) {
                (Some(IntObs { value: Some(c), .. }), Some(p)) if p.abs() > 0.001 => {
                    (((c / p - 1.0) * 100.0 - 2.0) / 5.0).clamp(-1.0, 1.0)
                }
                _ => 0.0,
            };

            result.fed_funds[i] = advance_and_get(&source.fed_obs, &mut fed_idx, bar_timestamp)
                .map(|v| ((v - 3.0) / 3.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            let t10y = advance_and_get(&source.t10y_obs, &mut t10y_idx, bar_timestamp);
            let t2y = advance_and_get(&source.t2y_obs, &mut t2y_idx, bar_timestamp);

            result.treasury_10y[i] = t10y
                .map(|v| ((v - 3.0) / 3.0).clamp(-1.0, 1.0))
                .unwrap_or(0.0);

            result.yield_spread[i] = match (t10y, t2y) {
                (Some(l), Some(s)) => ((l - s) / 2.0).clamp(-1.0, 1.0),
                _ => 0.0,
            };

            result.consumer_sentiment[i] =
                advance_and_get(&source.sentiment_obs, &mut sentiment_idx, bar_timestamp)
                    .map(|v| ((v - 90.0) / 30.0).clamp(-1.0, 1.0))
                    .unwrap_or(0.0);

            result.initial_claims[i] =
                advance_and_get(&source.claims_obs, &mut claims_idx, bar_timestamp)
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

#[derive(Clone, Copy)]
struct IntObs {
    period_date: NaiveDate,
    available_at: i64,
    value: Option<f64>,
}

fn load_series(series: MacroSeries) -> Result<Vec<IntObs>, MacroDataError> {
    let data = get_macro_data(series)?;
    obs_to_dates(series, &data.observations)
}

fn obs_to_dates(
    series: MacroSeries,
    observations: &[MacroObservation],
) -> Result<Vec<IntObs>, MacroDataError> {
    let mut dated =
        observations
            .iter()
            .map(|observation| {
                let period_date = NaiveDate::parse_from_str(&observation.date, "%Y-%m-%d")
                    .map_err(|_| MacroDataError::InvalidData {
                        series: series.series_id(),
                        message: format!("invalid observation period {:?}", observation.date),
                    })?;
                OffsetDateTime::from_unix_timestamp(observation.available_at).map_err(|_| {
                    MacroDataError::InvalidData {
                        series: series.series_id(),
                        message: format!(
                            "invalid availability timestamp {:?}",
                            observation.available_at
                        ),
                    }
                })?;
                Ok(IntObs {
                    period_date,
                    available_at: observation.available_at,
                    value: observation.value,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    dated.sort_by_key(|observation| observation.available_at);
    Ok(dated)
}

fn index_by_period(observations: &[IntObs]) -> HashMap<NaiveDate, IntObs> {
    let mut by_period = HashMap::with_capacity(observations.len());
    for &observation in observations {
        by_period
            .entry(observation.period_date)
            .and_modify(|existing: &mut IntObs| {
                if observation.available_at < existing.available_at {
                    *existing = observation;
                }
            })
            .or_insert(observation);
    }
    by_period
}

fn previous_year_period(period: NaiveDate) -> Option<NaiveDate> {
    period.checked_sub_months(Months::new(12))
}

fn released_period_value(
    by_period: &HashMap<NaiveDate, IntObs>,
    period: NaiveDate,
    bar_timestamp: i64,
) -> Option<f64> {
    by_period
        .get(&period)
        .filter(|observation| observation.available_at <= bar_timestamp)
        .and_then(|observation| observation.value)
}

#[inline]
fn advance_to_latest<'a>(
    observations: &'a [IntObs],
    cursor: &mut usize,
    target_timestamp: i64,
) -> Option<&'a IntObs> {
    if observations.is_empty() {
        return None;
    }
    while *cursor + 1 < observations.len()
        && observations[*cursor + 1].available_at <= target_timestamp
    {
        *cursor += 1;
    }
    (observations[*cursor].available_at <= target_timestamp).then(|| &observations[*cursor])
}

/// Advance to the most recent observation released by the target date.
#[inline]
fn advance_and_get(
    observations: &[IntObs],
    cursor: &mut usize,
    target_timestamp: i64,
) -> Option<f64> {
    advance_to_latest(observations, cursor, target_timestamp)
        .and_then(|observation| observation.value)
}

#[inline]
fn days_to_first_friday(date: NaiveDate) -> i64 {
    let first = NaiveDate::from_ymd_opt(date.year(), date.month(), 1).unwrap();
    let offset = (Weekday::Fri.num_days_from_monday() as i64
        - first.weekday().num_days_from_monday() as i64)
        .rem_euclid(7);
    let this_month = first + Duration::days(offset);
    let target = if date <= this_month {
        this_month
    } else {
        let (year, month) = if date.month() == 12 {
            (date.year() + 1, 1)
        } else {
            (date.year(), date.month() + 1)
        };
        let first = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        let offset = (Weekday::Fri.num_days_from_monday() as i64
            - first.weekday().num_days_from_monday() as i64)
            .rem_euclid(7);
        first + Duration::days(offset)
    };
    (target - date).num_days()
}

#[inline]
fn days_to_day_of_month(date: NaiveDate, target_day: u32) -> i64 {
    let this_month = NaiveDate::from_ymd_opt(date.year(), date.month(), target_day).unwrap();
    let target = if date <= this_month {
        this_month
    } else {
        let (year, month) = if date.month() == 12 {
            (date.year() + 1, 1)
        } else {
            (date.year(), date.month() + 1)
        };
        NaiveDate::from_ymd_opt(year, month, target_day).unwrap()
    };
    (target - date).num_days()
}

#[inline]
fn days_to_next_fomc(date: NaiveDate) -> i64 {
    const FOMC_MONTHS: [u32; 8] = [1, 3, 5, 6, 7, 9, 11, 12];
    const FOMC_DAY: u32 = 15;

    for year in [date.year(), date.year() + 1] {
        for month in FOMC_MONTHS {
            let candidate = NaiveDate::from_ymd_opt(year, month, FOMC_DAY).unwrap();
            if candidate > date {
                return (candidate - date).num_days();
            }
        }
    }
    unreachable!()
}

#[inline]
fn days_to_gdp_release(date: NaiveDate) -> i64 {
    const GDP_MONTHS: [u32; 4] = [1, 4, 7, 10];
    const GDP_DAY: u32 = 28;

    for year in [date.year(), date.year() + 1] {
        for month in GDP_MONTHS {
            let candidate = NaiveDate::from_ymd_opt(year, month, GDP_DAY).unwrap();
            if candidate > date {
                return (candidate - date).num_days();
            }
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::{index_by_period, IntObs, MacroIndicators, MacroSourceData};
    use chrono::{NaiveDate, NaiveDateTime};
    use time::OffsetDateTime;

    fn date(value: &str) -> NaiveDate {
        NaiveDate::parse_from_str(value, "%Y-%m-%d").unwrap()
    }

    fn timestamp(value: &str) -> i64 {
        NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc()
            .timestamp()
    }

    fn bar(value: &str) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(timestamp(value)).unwrap()
    }

    fn observation(period_date: &str, available_at: &str, value: f64) -> IntObs {
        IntObs {
            period_date: date(period_date),
            available_at: timestamp(available_at),
            value: Some(value),
        }
    }

    fn source_with_unemployment(unemp_obs: Vec<IntObs>) -> MacroSourceData {
        MacroSourceData {
            gdp_obs: vec![],
            unemp_obs,
            payrolls_obs: vec![],
            cpi_obs: vec![],
            cpi_by_period: Default::default(),
            core_cpi_obs: vec![],
            core_cpi_by_period: Default::default(),
            fed_obs: vec![],
            t10y_obs: vec![],
            t2y_obs: vec![],
            sentiment_obs: vec![],
            claims_obs: vec![],
        }
    }

    fn source_with_cpi(mut cpi_obs: Vec<IntObs>, mut core_cpi_obs: Vec<IntObs>) -> MacroSourceData {
        cpi_obs.sort_by_key(|observation| observation.available_at);
        core_cpi_obs.sort_by_key(|observation| observation.available_at);
        MacroSourceData {
            gdp_obs: vec![],
            unemp_obs: vec![],
            payrolls_obs: vec![],
            cpi_by_period: index_by_period(&cpi_obs),
            cpi_obs,
            core_cpi_by_period: index_by_period(&core_cpi_obs),
            core_cpi_obs,
            fed_obs: vec![],
            t10y_obs: vec![],
            t2y_obs: vec![],
            sentiment_obs: vec![],
            claims_obs: vec![],
        }
    }

    #[test]
    fn same_day_release_is_hidden_until_its_exact_availability_time() {
        let source =
            source_with_unemployment(vec![observation("2024-01-01", "2024-02-02 20:00:00", 6.0)]);
        let indicators = MacroIndicators::compute_with_source(
            &[
                bar("2024-02-02 14:30:00"),
                bar("2024-02-02 19:55:00"),
                bar("2024-02-02 20:00:00"),
            ],
            &source,
        );

        assert_eq!(indicators.unemployment[0], 0.0);
        assert_eq!(indicators.unemployment[1], 0.0);
        assert!((indicators.unemployment[2] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn a_new_initial_release_replaces_the_previous_available_value() {
        let source = source_with_unemployment(vec![
            observation("2024-01-01", "2024-02-02 00:00:00", 6.0),
            observation("2024-02-01", "2024-03-08 00:00:00", 4.0),
        ]);
        let indicators = MacroIndicators::compute_with_source(
            &[
                bar("2024-02-02 14:30:00"),
                bar("2024-03-07 14:30:00"),
                bar("2024-03-08 14:30:00"),
            ],
            &source,
        );

        assert!((indicators.unemployment[0] - 0.2).abs() < 1e-12);
        assert!((indicators.unemployment[1] - 0.2).abs() < 1e-12);
        assert!((indicators.unemployment[2] + 0.2).abs() < 1e-12);
    }

    #[test]
    fn cpi_yoy_uses_matching_periods_when_release_schedules_shift() {
        let cpi = vec![
            observation("2022-12-01", "2023-01-15 00:00:00", 80.0),
            observation("2023-01-01", "2023-02-20 00:00:00", 100.0),
            observation("2024-01-01", "2024-02-10 00:00:00", 103.0),
        ];
        let core_cpi = cpi.clone();
        let source = source_with_cpi(cpi, core_cpi);
        let indicators =
            MacroIndicators::compute_with_source(&[bar("2024-02-10 14:30:00")], &source);

        assert!((indicators.cpi_yoy[0] - 0.2).abs() < 1e-12);
        assert!((indicators.core_cpi_yoy[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn cpi_yoy_requires_a_released_matching_prior_period() {
        let cpi = vec![
            observation("2022-12-01", "2023-01-15 00:00:00", 80.0),
            observation("2024-01-01", "2024-02-10 00:00:00", 103.0),
        ];
        let source = source_with_cpi(cpi, vec![]);
        let indicators =
            MacroIndicators::compute_with_source(&[bar("2024-02-10 14:30:00")], &source);

        assert_eq!(indicators.cpi_yoy[0], 0.0);
    }
}
