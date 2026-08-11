use crate::constants::files::DATA_PATH;
use chrono::NaiveDate;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

const FRED_BASE: &str = "https://api.stlouisfed.org/fred/series/observations";
const CACHE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MacroObservation {
    pub date: String,
    pub available_on: String,
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
        .map(|observation| MacroObservation {
            date: observation.date,
            available_on: observation.realtime_start,
            value: observation.value.parse().ok(),
        })
        .collect();
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
        if NaiveDate::parse_from_str(&observation.available_on, "%Y-%m-%d").is_err() {
            return Err(invalid_data(
                series,
                "an observation has an invalid availability date",
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
        .max_by(|left, right| left.available_on.cmp(&right.available_on))
        .and_then(|observation| observation.value))
}

pub fn get_value_at_date(series: MacroSeries, date: &str) -> Result<Option<f64>, MacroDataError> {
    Ok(get_macro_data(series)?
        .observations
        .into_iter()
        .filter(|observation| observation.available_on.as_str() <= date)
        .max_by(|left, right| left.available_on.cmp(&right.available_on))
        .and_then(|observation| observation.value))
}

#[cfg(test)]
mod tests {
    use super::{cache_path, FredResponse, MacroObservation, MacroSeries};

    #[test]
    fn fred_initial_release_availability_is_preserved() {
        let response: FredResponse = serde_json::from_str(
            r#"{"observations":[{"realtime_start":"2024-02-02","date":"2024-01-01","value":"353.0"}]}"#,
        )
        .unwrap();
        let observation = response.observations.into_iter().next().unwrap();
        let stored = MacroObservation {
            date: observation.date,
            available_on: observation.realtime_start,
            value: observation.value.parse().ok(),
        };

        assert_eq!(stored.date, "2024-01-01");
        assert_eq!(stored.available_on, "2024-02-02");
        assert_eq!(stored.value, Some(353.0));
    }

    #[test]
    fn cache_identity_includes_transform_and_schema_version() {
        let levels = cache_path(MacroSeries::NonFarmPayrolls);
        let growth = cache_path(MacroSeries::JobsGrowth);

        assert_ne!(levels, growth);
        assert!(levels.to_string_lossy().ends_with("_v2.bin"));
        assert!(growth.to_string_lossy().ends_with("_v2.bin"));
    }
}
