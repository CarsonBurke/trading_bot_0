use crate::constants::files::DATA_PATH;
use chrono::{
    Datelike, Duration as ChronoDuration, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Weekday,
};
use chrono_tz::America::New_York;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

const FRED_BASE: &str = "https://api.stlouisfed.org/fred/series/observations";
const CACHE_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MacroObservation {
    pub date: String,
    pub available_at: i64,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroData {
    pub series_id: String,
    pub observations: Vec<MacroObservation>,
}

#[derive(Debug, Clone)]
pub enum MacroDataError {
    MissingApiKey,
    Client(String),
    Request {
        series: &'static str,
        message: String,
    },
    Api {
        series: &'static str,
        message: String,
    },
    Decode {
        series: &'static str,
        message: String,
    },
    InvalidData {
        series: &'static str,
        message: String,
    },
    Cache {
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for MacroDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingApiKey => write!(f, "FRED_API_KEY is required for macroeconomic data"),
            Self::Client(message) => write!(f, "failed to create FRED client: {message}"),
            Self::Request { series, message } => {
                write!(f, "FRED request failed for {series}: {message}")
            }
            Self::Api { series, message } => {
                write!(f, "FRED returned an error for {series}: {message}")
            }
            Self::Decode { series, message } => {
                write!(f, "failed to decode FRED data for {series}: {message}")
            }
            Self::InvalidData { series, message } => {
                write!(f, "invalid FRED data for {series}: {message}")
            }
            Self::Cache { path, message } => {
                write!(f, "macro cache error at {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for MacroDataError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroSeries {
    RealGdp,
    GdpGrowth,
    UnemploymentRate,
    NonFarmPayrolls,
    JobsGrowth,
    InitialClaims,
    CpiAllItems,
    CoreCpi,
    PceInflation,
    FedFundsRate,
    Treasury10Y,
    Treasury2Y,
    ConsumerSentiment,
    RetailSales,
    IndustrialProd,
}

impl MacroSeries {
    pub fn series_id(&self) -> &'static str {
        match self {
            Self::RealGdp => "GDPC1",
            Self::GdpGrowth => "A191RL1Q225SBEA",
            Self::UnemploymentRate => "UNRATE",
            Self::NonFarmPayrolls | Self::JobsGrowth => "PAYEMS",
            Self::InitialClaims => "ICSA",
            Self::CpiAllItems => "CPIAUCSL",
            Self::CoreCpi => "CPILFESL",
            Self::PceInflation => "PCEPI",
            Self::FedFundsRate => "FEDFUNDS",
            Self::Treasury10Y => "DGS10",
            Self::Treasury2Y => "DGS2",
            Self::ConsumerSentiment => "UMCSENT",
            Self::RetailSales => "RSAFS",
            Self::IndustrialProd => "INDPRO",
        }
    }

    fn units(&self) -> &'static str {
        match self {
            Self::JobsGrowth => "pch",
            _ => "lin",
        }
    }

    fn frequency(&self) -> Option<&'static str> {
        match self {
            Self::InitialClaims => Some("m"),
            _ => None,
        }
    }

    fn cache_key(&self) -> String {
        format!(
            "{}_{}_{}",
            self.series_id(),
            self.units(),
            self.frequency().unwrap_or("native")
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedMacroData {
    schema_version: u32,
    series_id: String,
    units: String,
    frequency: Option<String>,
    observations: Vec<MacroObservation>,
}

#[derive(Deserialize)]
struct FredResponse {
    observations: Vec<FredObservation>,
}

#[derive(Deserialize)]
struct FredObservation {
    realtime_start: String,
    date: String,
    value: String,
}

#[derive(Deserialize)]
struct FredErrorResponse {
    error_message: Option<String>,
}

fn fetch_series(series: MacroSeries, api_key: &str) -> Result<MacroData, MacroDataError> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| MacroDataError::Client(err.without_url().to_string()))?;

    let mut query = vec![
        ("series_id", series.series_id()),
        ("api_key", api_key),
        ("file_type", "json"),
        ("units", series.units()),
        ("output_type", "4"),
        ("realtime_start", "1776-07-04"),
        ("realtime_end", "9999-12-31"),
    ];
    if let Some(frequency) = series.frequency() {
        query.extend([("frequency", frequency), ("aggregation_method", "avg")]);
    }

    let response =
        client
            .get(FRED_BASE)
            .query(&query)
            .send()
            .map_err(|err| MacroDataError::Request {
                series: series.series_id(),
                message: err.without_url().to_string(),
            })?;

    if !response.status().is_success() {
        let status = response.status();
        let message = response
            .json::<FredErrorResponse>()
            .ok()
            .and_then(|body| body.error_message)
            .unwrap_or_else(|| status.to_string());
        return Err(MacroDataError::Api {
            series: series.series_id(),
            message,
        });
    }

    let response = response
        .json::<FredResponse>()
        .map_err(|err| MacroDataError::Decode {
            series: series.series_id(),
            message: err.without_url().to_string(),
        })?;
    let observations = response
        .observations
        .into_iter()
        .map(|observation| {
            Ok(MacroObservation {
                date: observation.date,
                available_at: conservative_availability_timestamp(
                    series,
                    &observation.realtime_start,
                )?,
                value: observation.value.parse().ok(),
            })
        })
        .collect::<Result<Vec<_>, MacroDataError>>()?;
    let data = MacroData {
        series_id: series.series_id().to_string(),
        observations,
    };
    validate_data(series, &data)?;
    Ok(data)
}

pub fn get_macro_data(series: MacroSeries) -> Result<MacroData, MacroDataError> {
    if let Some(data) = load_from_cache(series)? {
        return Ok(data);
    }

    let api_key = env::var("FRED_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty())
        .ok_or(MacroDataError::MissingApiKey)?;
    let data = fetch_series(series, &api_key)?;
    save_to_cache(series, &data)?;
    Ok(data)
}

pub fn get_all_macro_data() -> Result<Vec<MacroData>, MacroDataError> {
    use MacroSeries::*;
    [
        RealGdp,
        GdpGrowth,
        UnemploymentRate,
        NonFarmPayrolls,
        JobsGrowth,
        InitialClaims,
        CpiAllItems,
        CoreCpi,
        PceInflation,
        FedFundsRate,
        Treasury10Y,
        Treasury2Y,
        ConsumerSentiment,
        RetailSales,
        IndustrialProd,
    ]
    .into_iter()
    .map(get_macro_data)
    .collect()
}

fn cache_path(series: MacroSeries) -> PathBuf {
    Path::new(DATA_PATH).join(format!(
        "macro_{}_v{CACHE_SCHEMA_VERSION}.bin",
        series.cache_key()
    ))
}

fn load_from_cache(series: MacroSeries) -> Result<Option<MacroData>, MacroDataError> {
    let path = cache_path(series);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(cache_error(path, err)),
    };
    let cached: CachedMacroData = match postcard::from_bytes(&bytes) {
        Ok(cached) => cached,
        Err(_) => return Ok(None),
    };
    if cached.schema_version != CACHE_SCHEMA_VERSION
        || cached.series_id != series.series_id()
        || cached.units != series.units()
        || cached.frequency.as_deref() != series.frequency()
    {
        return Ok(None);
    }

    let data = MacroData {
        series_id: cached.series_id,
        observations: cached.observations,
    };
    if validate_data(series, &data).is_err() {
        return Ok(None);
    }
    Ok(Some(data))
}

fn save_to_cache(series: MacroSeries, data: &MacroData) -> Result<(), MacroDataError> {
    validate_data(series, data)?;
    fs::create_dir_all(DATA_PATH).map_err(|err| cache_error(PathBuf::from(DATA_PATH), err))?;
    let cached = CachedMacroData {
        schema_version: CACHE_SCHEMA_VERSION,
        series_id: data.series_id.clone(),
        units: series.units().to_string(),
        frequency: series.frequency().map(str::to_string),
        observations: data.observations.clone(),
    };
    let bytes = postcard::to_allocvec(&cached).map_err(|err| MacroDataError::Cache {
        path: cache_path(series),
        message: err.to_string(),
    })?;
    let path = cache_path(series);
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temporary, bytes).map_err(|err| cache_error(temporary.clone(), err))?;
    fs::rename(&temporary, &path).map_err(|err| cache_error(path, err))
}

fn validate_data(series: MacroSeries, data: &MacroData) -> Result<(), MacroDataError> {
    if data.series_id != series.series_id() {
        return Err(invalid_data(series, "series ID does not match the request"));
    }
    if data.observations.is_empty() || !data.observations.iter().any(|item| item.value.is_some()) {
        return Err(invalid_data(
            series,
            "the series contains no numeric observations",
        ));
    }
    for observation in &data.observations {
        if NaiveDate::parse_from_str(&observation.date, "%Y-%m-%d").is_err() {
            return Err(invalid_data(
                series,
                "an observation has an invalid period date",
            ));
        }
        if chrono::DateTime::from_timestamp(observation.available_at, 0).is_none() {
            return Err(invalid_data(
                series,
                "an observation has an invalid availability timestamp",
            ));
        }
        if observation.value.is_some_and(|value| !value.is_finite()) {
            return Err(invalid_data(
                series,
                "an observation has a non-finite value",
            ));
        }
    }
    Ok(())
}

fn invalid_data(series: MacroSeries, message: &str) -> MacroDataError {
    MacroDataError::InvalidData {
        series: series.series_id(),
        message: message.to_string(),
    }
}

fn cache_error(path: PathBuf, error: io::Error) -> MacroDataError {
    MacroDataError::Cache {
        path,
        message: error.to_string(),
    }
}

pub fn get_latest_value(series: MacroSeries) -> Result<Option<f64>, MacroDataError> {
    Ok(get_macro_data(series)?
        .observations
        .into_iter()
        .max_by_key(|observation| observation.available_at)
        .and_then(|observation| observation.value))
}

fn conservative_availability_timestamp(
    series: MacroSeries,
    provider_date: &str,
) -> Result<i64, MacroDataError> {
    let mut session_date = NaiveDate::parse_from_str(provider_date, "%Y-%m-%d")
        .map_err(|_| invalid_data(series, "an observation has an invalid availability date"))?
        .succ_opt()
        .ok_or_else(|| invalid_data(series, "availability date exceeds supported range"))?;
    while !is_nyse_session(session_date) {
        session_date = session_date
            .succ_opt()
            .ok_or_else(|| invalid_data(series, "availability date exceeds supported range"))?;
    }
    // FRED exposes only a date for realtime_start, not the instant when the
    // vintage became observable. Make it visible at the following NYSE open.
    let local_open = NaiveDateTime::new(
        session_date,
        NaiveTime::from_hms_opt(9, 30, 0).expect("NYSE open is valid"),
    );
    New_York
        .from_local_datetime(&local_open)
        .single()
        .map(|open| open.timestamp())
        .ok_or_else(|| invalid_data(series, "NYSE open has an ambiguous timezone offset"))
}

fn is_nyse_session(date: NaiveDate) -> bool {
    !matches!(date.weekday(), Weekday::Sat | Weekday::Sun) && !is_nyse_holiday(date)
}

fn observed_fixed_holiday(date: NaiveDate) -> NaiveDate {
    match date.weekday() {
        Weekday::Sat => date.pred_opt().expect("supported holiday date"),
        Weekday::Sun => date.succ_opt().expect("supported holiday date"),
        _ => date,
    }
}

fn nth_weekday(year: i32, month: u32, weekday: Weekday, nth: u32) -> NaiveDate {
    let first = NaiveDate::from_ymd_opt(year, month, 1).expect("valid month");
    let offset = (7 + weekday.num_days_from_monday() as i64
        - first.weekday().num_days_from_monday() as i64)
        % 7;
    first + ChronoDuration::days(offset + 7 * (nth - 1) as i64)
}

fn last_weekday(year: i32, month: u32, weekday: Weekday) -> NaiveDate {
    let next_month = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .expect("valid next month");
    let last = next_month.pred_opt().expect("valid previous date");
    let offset = (7 + last.weekday().num_days_from_monday() as i64
        - weekday.num_days_from_monday() as i64)
        % 7;
    last - ChronoDuration::days(offset)
}

// Gregorian computus, used because Good Friday is an NYSE holiday.
fn easter_sunday(year: i32) -> NaiveDate {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = (h + l - 7 * m + 114) % 31 + 1;
    NaiveDate::from_ymd_opt(year, month as u32, day as u32).expect("valid Easter date")
}

fn is_nyse_holiday(date: NaiveDate) -> bool {
    let year = date.year();
    let new_year =
        observed_fixed_holiday(NaiveDate::from_ymd_opt(year, 1, 1).expect("valid New Year's Day"));
    let next_new_year = observed_fixed_holiday(
        NaiveDate::from_ymd_opt(year + 1, 1, 1).expect("valid next New Year's Day"),
    );
    let christmas =
        observed_fixed_holiday(NaiveDate::from_ymd_opt(year, 12, 25).expect("valid Christmas"));
    let independence = observed_fixed_holiday(
        NaiveDate::from_ymd_opt(year, 7, 4).expect("valid Independence Day"),
    );
    let juneteenth =
        observed_fixed_holiday(NaiveDate::from_ymd_opt(year, 6, 19).expect("valid Juneteenth"));

    const EXTRAORDINARY_FULL_DAY_CLOSURES: &[(i32, u32, u32)] = &[
        // September 11 attacks.
        (2001, 9, 11),
        (2001, 9, 12),
        (2001, 9, 13),
        (2001, 9, 14),
        // National days of mourning for Presidents Reagan, Ford, Bush, Carter.
        (2004, 6, 11),
        (2007, 1, 2),
        (2018, 12, 5),
        (2025, 1, 9),
        // Hurricane Sandy.
        (2012, 10, 29),
        (2012, 10, 30),
    ];

    date == new_year
        || date == next_new_year
        || (year >= 1998 && date == nth_weekday(year, 1, Weekday::Mon, 3))
        || date == nth_weekday(year, 2, Weekday::Mon, 3)
        || date == easter_sunday(year) - ChronoDuration::days(2)
        || date == last_weekday(year, 5, Weekday::Mon)
        || (year >= 2022 && date == juneteenth)
        || date == independence
        || date == nth_weekday(year, 9, Weekday::Mon, 1)
        || date == nth_weekday(year, 11, Weekday::Thu, 4)
        || date == christmas
        || EXTRAORDINARY_FULL_DAY_CLOSURES
            .iter()
            .any(|&(closure_year, month, day)| {
                date == NaiveDate::from_ymd_opt(closure_year, month, day)
                    .expect("valid extraordinary NYSE closure")
            })
}

pub fn get_value_at_timestamp(
    series: MacroSeries,
    timestamp: i64,
) -> Result<Option<f64>, MacroDataError> {
    Ok(get_macro_data(series)?
        .observations
        .into_iter()
        .filter(|observation| observation.available_at <= timestamp)
        .max_by_key(|observation| observation.available_at)
        .and_then(|observation| observation.value))
}

#[cfg(test)]
mod tests {
    use super::{cache_path, conservative_availability_timestamp, FredResponse, MacroSeries};
    use chrono::NaiveDateTime;

    fn timestamp(value: &str) -> i64 {
        NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc()
            .timestamp()
    }

    #[test]
    fn fred_date_only_availability_moves_to_the_next_session() {
        let response: FredResponse = serde_json::from_str(
            r#"{"observations":[{"realtime_start":"2024-02-02","date":"2024-01-01","value":"353.0"}]}"#,
        )
        .unwrap();
        let observation = response.observations.into_iter().next().unwrap();

        assert_eq!(observation.date, "2024-01-01");
        assert_eq!(observation.value.parse::<f64>().unwrap(), 353.0);
        assert_eq!(
            conservative_availability_timestamp(
                MacroSeries::CpiAllItems,
                &observation.realtime_start
            )
            .unwrap(),
            timestamp("2024-02-05 14:30:00")
        );
    }

    #[test]
    fn date_only_availability_starts_at_the_next_weekday_session() {
        assert_eq!(
            conservative_availability_timestamp(MacroSeries::CpiAllItems, "2024-02-08").unwrap(),
            timestamp("2024-02-09 14:30:00")
        );
        assert_eq!(
            conservative_availability_timestamp(MacroSeries::CpiAllItems, "2024-02-09").unwrap(),
            timestamp("2024-02-12 14:30:00")
        );
        assert_eq!(
            conservative_availability_timestamp(MacroSeries::CpiAllItems, "2024-01-12").unwrap(),
            timestamp("2024-01-16 14:30:00"),
            "Martin Luther King Jr. Day is not a trading session"
        );
        assert_eq!(
            conservative_availability_timestamp(MacroSeries::CpiAllItems, "2024-06-13").unwrap(),
            timestamp("2024-06-14 13:30:00"),
            "summer session open must use the daylight-saving offset"
        );
        assert_eq!(
            conservative_availability_timestamp(MacroSeries::CpiAllItems, "2006-03-16").unwrap(),
            timestamp("2006-03-17 14:30:00"),
            "pre-2007 US daylight-saving rules must be respected"
        );
        assert_eq!(
            conservative_availability_timestamp(MacroSeries::CpiAllItems, "2012-10-26").unwrap(),
            timestamp("2012-10-31 13:30:00"),
            "extraordinary full-day exchange closures must be skipped"
        );
    }

    #[test]
    fn cache_identity_includes_transform_and_schema_version() {
        let levels = cache_path(MacroSeries::NonFarmPayrolls);
        let growth = cache_path(MacroSeries::JobsGrowth);

        assert_ne!(levels, growth);
        assert!(levels.to_string_lossy().ends_with("_v3.bin"));
        assert!(growth.to_string_lossy().ends_with("_v3.bin"));
    }
}
