use crate::constants::files::DATA_PATH;
use serde::{Deserialize, Serialize};
use std::fs;

const FRED_BASE: &str = "https://api.stlouisfed.org/fred/series/observations";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroObservation {
    pub date: String,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroData {
    pub series_id: String,
    pub observations: Vec<MacroObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroSeries {
    // GDP
    RealGdp,   // GDPC1 - Real GDP, quarterly, billions chained 2017$
    GdpGrowth, // A191RL1Q225SBEA - Real GDP % change from preceding period

    // Labor Market
    UnemploymentRate, // UNRATE - Civilian unemployment rate, monthly
    NonFarmPayrolls,  // PAYEMS - Total nonfarm employees, thousands, monthly
    JobsGrowth,       // PAYEMS with pch transform
    InitialClaims,    // ICSA - Initial jobless claims, weekly

    // Inflation
    CpiAllItems,  // CPIAUCSL - CPI all urban consumers, monthly
    CoreCpi,      // CPILFESL - CPI less food & energy, monthly
    PceInflation, // PCEPI - Personal consumption expenditures price index

    // Interest Rates
    FedFundsRate, // FEDFUNDS - Effective federal funds rate
    Treasury10Y,  // DGS10 - 10-year treasury constant maturity rate
    Treasury2Y,   // DGS2 - 2-year treasury

    // Consumer/Business
    ConsumerSentiment, // UMCSENT - U of Michigan consumer sentiment
    RetailSales,       // RSAFS - Advance retail sales
    IndustrialProd,    // INDPRO - Industrial production index
}

impl MacroSeries {
    pub fn series_id(&self) -> &'static str {
        match self {
            Self::RealGdp => "GDPC1",
            Self::GdpGrowth => "A191RL1Q225SBEA",
            Self::UnemploymentRate => "UNRATE",
            Self::NonFarmPayrolls => "PAYEMS",
            Self::JobsGrowth => "PAYEMS",
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
            Self::JobsGrowth => "pch", // % change
            _ => "lin",                // linear (raw values)
        }
    }

    fn frequency(&self) -> Option<&'static str> {
        match self {
            Self::InitialClaims => Some("m"), // aggregate weekly to monthly
            _ => None,
        }
    }
}

#[derive(Deserialize)]
struct FredResponse {
    observations: Vec<FredObs>,
}

#[derive(Deserialize)]
struct FredObs {
    date: String,
    value: String,
}

fn fetch_series(series: MacroSeries, api_key: &str) -> MacroData {
    let client = reqwest::blocking::Client::new();

    let mut url = format!(
        "{}?series_id={}&api_key={}&file_type=json&units={}",
        FRED_BASE,
        series.series_id(),
        api_key,
        series.units()
    );

    if let Some(freq) = series.frequency() {
        url.push_str(&format!("&frequency={}&aggregation_method=avg", freq));
    }

    let observations = client
        .get(&url)
        .send()
        .ok()
        .and_then(|r| r.json::<FredResponse>().ok())
        .map(|resp| {
            resp.observations
                .into_iter()
                .map(|o| MacroObservation {
                    date: o.date,
                    value: o.value.parse().ok(),
                })
                .collect()
        })
        .unwrap_or_default();

    MacroData {
        series_id: series.series_id().to_string(),
        observations,
    }
}

pub fn get_macro_data(series: MacroSeries) -> MacroData {
    if let Some(data) = load_from_cache(series) {
        return data;
    }

    // let key = std::env::var("FRED_API_KEY").ok();
    let key = Some(
        "https://api.stlouisfed.org/fred/series/search?api_key=abcdefghijklmnopqrstuvwxyz123456"
            .to_string(),
    );
    let Some(key) = key else {
        eprintln!("No FRED_API_KEY set, returning empty data for {:?}", series);
        return MacroData {
            series_id: series.series_id().to_string(),
            observations: Vec::new(),
        };
    };

    let data = fetch_series(series, &key);
    if !data.observations.is_empty() {
        save_to_cache(series, &data);
    }
    data
}

pub fn get_all_macro_data() -> Vec<MacroData> {
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

fn cache_path(series: MacroSeries) -> String {
    format!("{}/macro_{}.bin", DATA_PATH, series.series_id())
}

fn load_from_cache(series: MacroSeries) -> Option<MacroData> {
    let path = cache_path(series);
    let file = fs::read(path).ok()?;
    postcard::from_bytes(&file).ok()
}

fn save_to_cache(series: MacroSeries, data: &MacroData) {
    let path = cache_path(series);
    if let Ok(encoded) = postcard::to_allocvec(data) {
        let _ = fs::write(path, encoded);
    }
}

pub fn get_latest_value(series: MacroSeries) -> Option<f64> {
    let data = get_macro_data(series);
    data.observations.last().and_then(|o| o.value)
}

pub fn get_value_at_date(series: MacroSeries, date: &str) -> Option<f64> {
    let data = get_macro_data(series);
    // Find most recent observation <= date
    data.observations
        .iter()
        .rev()
        .find(|o| o.date.as_str() <= date)
        .and_then(|o| o.value)
}
