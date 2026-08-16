//! Async Polygon.io REST client for market aggregates and reference data.
//!
//! The account plan grants a rolling history window; requests that reach further back answer
//! `HTTP 403` with `{"status":"NOT_AUTHORIZED"}`. That is a boundary signal rather than a failure,
//! so it is surfaced as [`Window::Unauthorized`] instead of an error. A rejected key is distinct:
//! it answers `HTTP 401` with `{"status":"ERROR"}` and is reported as one.

use anyhow::{bail, Context, Result};
use chrono::NaiveDate;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use shared::bars::PackedBar;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::Semaphore;

const BASE_URL: &str = "https://api.polygon.io";
const NOT_AUTHORIZED: &str = "NOT_AUTHORIZED";
const MAX_ATTEMPTS: u32 = 5;
const BASE_BACKOFF: Duration = Duration::from_millis(500);
const MAX_RETRY_AFTER_SECS: u64 = 60;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);
const AGG_LIMIT: usize = 50_000;
const TICKER_LIMIT: usize = 1_000;
const MAX_PAGES: usize = 512;

pub const DEFAULT_CONCURRENCY: usize = 16;

/// Security types kept for the tradable universe.
pub const KEPT_TICKER_TYPES: [&str; 3] = ["CS", "ETF", "ADRC"];

/// Payload of a request that may sit outside the plan's rolling history window.
#[derive(Clone, Debug)]
pub enum Window<T> {
    Data(T),
    Unauthorized,
}

impl<T: Default> Window<T> {
    /// Collapses the boundary signal into an empty payload.
    pub fn or_default(self) -> T {
        match self {
            Self::Data(data) => data,
            Self::Unauthorized => T::default(),
        }
    }
}

/// Reference metadata for one tradable symbol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TickerRef {
    pub ticker: String,
    #[serde(default)]
    pub name: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub primary_exchange: String,
}

/// Date-effective identity of a ticker string.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TickerIdentity {
    #[serde(default, deserialize_with = "null_as_empty")]
    pub composite_figi: String,
    #[serde(default, deserialize_with = "null_as_empty")]
    pub cik: String,
    #[serde(default, deserialize_with = "null_as_empty")]
    pub name: String,
    #[serde(rename = "type", default, deserialize_with = "null_as_empty")]
    pub kind: String,
}

impl TickerIdentity {
    /// Stable key for "which security is this". `composite_figi` is the right answer, but Polygon
    /// leaves it null for freshly listed issues (NIQ Global Intelligence plc), so fall back to the
    /// filer id and finally the name rather than treating an unnamed security as unchanged.
    pub fn key(&self) -> &str {
        [
            self.composite_figi.as_str(),
            self.cik.as_str(),
            self.name.as_str(),
        ]
        .into_iter()
        .find(|value| !value.is_empty())
        .unwrap_or_default()
    }
}

fn null_as_empty<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

/// Overrides the global in-flight request limit. Takes effect only before the first request.
pub fn configure(concurrency: usize) {
    CONCURRENCY.store(concurrency.max(1), Ordering::Relaxed);
}

/// One `[from, to]` aggregate window for a single symbol, cursor-paged to exhaustion.
pub async fn aggregates(
    symbol: &str,
    multiplier: u32,
    timespan: &str,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<Vec<PackedBar>> {
    Ok(aggregates_window(symbol, multiplier, timespan, from, to)
        .await?
        .or_default())
}

/// [`aggregates`] that keeps the plan-window boundary distinguishable from an empty window.
pub async fn aggregates_window(
    symbol: &str,
    multiplier: u32,
    timespan: &str,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<Window<Vec<PackedBar>>> {
    let url = format!(
        "{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}\
         ?adjusted=true&sort=asc&limit={AGG_LIMIT}"
    );
    let rows: Window<Vec<AggRow>> = client()?.collect(url).await?;
    Ok(match rows {
        Window::Unauthorized => Window::Unauthorized,
        Window::Data(rows) => Window::Data(rows.iter().map(AggRow::to_bar).collect()),
    })
}

/// Every symbol's daily bar for one session.
pub async fn grouped_daily(date: NaiveDate) -> Result<Vec<(String, PackedBar)>> {
    Ok(grouped_daily_window(date).await?.or_default())
}

/// [`grouped_daily`] that keeps the plan-window boundary distinguishable from a non-trading day.
///
/// The grouped endpoint stamps daily bars at the 16:00 ET close while `range/1/day` stamps them at
/// the session's midnight ET; timestamps are rewritten to the latter so `ts_ms` always denotes the
/// bar's opening instant regardless of which endpoint produced it.
pub async fn grouped_daily_window(date: NaiveDate) -> Result<Window<Vec<(String, PackedBar)>>> {
    let url = format!("{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted=true");
    let rows: Window<Vec<GroupedRow>> = client()?.collect(url).await?;
    let ts_ms = session_open_ms(date);
    Ok(match rows {
        Window::Unauthorized => Window::Unauthorized,
        Window::Data(rows) => Window::Data(
            rows.into_iter()
                .map(|row| {
                    let mut bar = row.agg.to_bar();
                    bar.ts_ms = ts_ms;
                    (row.symbol, bar)
                })
                .collect(),
        ),
    })
}

/// Midnight New York time for a session date, in UTC milliseconds.
fn session_open_ms(date: NaiveDate) -> i64 {
    let midnight = date
        .and_hms_opt(0, 0, 0)
        .expect("midnight is a valid local time");
    midnight
        .and_local_timezone(chrono_tz::America::New_York)
        .earliest()
        .map(|stamp| stamp.timestamp_millis())
        .unwrap_or_else(|| midnight.and_utc().timestamp_millis())
}

/// Every US stocks ticker of a kept type, either listed (`active`) or delisted.
///
/// The delisted half is not optional for a corpus: a reference set of names that still trade today
/// omits every bankruptcy, every acquisition and every reverse-split-to-oblivion inside the
/// window, which are the outcomes a trading model most needs to have seen. Polygon serves
/// aggregates for a delisted ticker string for as long as the plan window covers its last trade.
pub async fn tickers(active: bool) -> Result<Vec<TickerRef>> {
    let url = format!(
        "{BASE_URL}/v3/reference/tickers?market=stocks&active={active}&limit={TICKER_LIMIT}"
    );
    let refs: Vec<TickerRef> = client()?.collect(url).await?.or_default();
    Ok(refs
        .into_iter()
        .filter(|entry| KEPT_TICKER_TYPES.contains(&entry.kind.as_str()))
        .collect())
}

/// Which security a ticker string denoted on a given date. Aggregates are keyed by the STRING, so
/// a reused ticker splices unrelated companies into one series; `composite_figi` is the stable
/// identity that survives a rename and changes on reuse.
pub async fn ticker_identity(symbol: &str, date: NaiveDate) -> Result<Option<TickerIdentity>> {
    let url = format!("{BASE_URL}/v3/reference/tickers/{symbol}?date={date}");
    client()?.fetch_object(&url).await
}

static CLIENT: OnceLock<PolygonClient> = OnceLock::new();
static CONCURRENCY: AtomicUsize = AtomicUsize::new(DEFAULT_CONCURRENCY);

fn client() -> Result<&'static PolygonClient> {
    if let Some(existing) = CLIENT.get() {
        return Ok(existing);
    }
    let api_key = std::env::var("POLYGON_API_KEY")
        .context("POLYGON_API_KEY is unset; add it to <repo>/.env")?;
    let concurrency = CONCURRENCY.load(Ordering::Relaxed);
    let http = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .pool_max_idle_per_host(concurrency)
        .build()
        .context("building polygon http client")?;
    let _ = CLIENT.set(PolygonClient {
        http,
        api_key,
        permits: Arc::new(Semaphore::new(concurrency)),
    });
    Ok(CLIENT.get().expect("polygon client initialized"))
}

struct PolygonClient {
    http: reqwest::Client,
    api_key: String,
    permits: Arc<Semaphore>,
}

enum Page<T> {
    Data {
        results: Vec<T>,
        next_url: Option<String>,
    },
    Unauthorized,
}

/// Raw outcome of one HTTP attempt, before any endpoint-specific decoding.
enum Fetched {
    Body(String),
    Unauthorized,
    NotFound,
}

#[derive(Deserialize)]
struct Envelope<T> {
    status: Option<String>,
    message: Option<String>,
    error: Option<String>,
    results: Option<Vec<T>>,
    next_url: Option<String>,
}

#[derive(Deserialize)]
struct AggRow {
    t: i64,
    o: f64,
    h: f64,
    l: f64,
    c: f64,
    #[serde(default)]
    v: f64,
    vw: Option<f64>,
    #[serde(default)]
    n: u64,
}

#[derive(Deserialize)]
struct GroupedRow {
    #[serde(rename = "T")]
    symbol: String,
    #[serde(flatten)]
    agg: AggRow,
}

impl AggRow {
    fn to_bar(&self) -> PackedBar {
        PackedBar {
            ts_ms: self.t,
            open: self.o as f32,
            high: self.h as f32,
            low: self.l as f32,
            close: self.c as f32,
            volume: self.v as f32,
            vwap: self.vw.unwrap_or((self.h + self.l + self.c) / 3.0) as f32,
            trades: self.n.min(u32::MAX as u64) as u32,
        }
    }
}

impl PolygonClient {
    /// Follows the `next_url` cursor until exhausted, concatenating every page.
    async fn collect<T: DeserializeOwned>(&self, first: String) -> Result<Window<Vec<T>>> {
        let mut next = Some(first);
        let mut collected = Vec::new();
        let mut pages = 0usize;
        while let Some(url) = next {
            pages += 1;
            if pages > MAX_PAGES {
                bail!("polygon cursor exceeded {MAX_PAGES} pages: {url}");
            }
            match self.fetch_page::<T>(&url).await? {
                // Ascending pages cross the plan boundary on the first request, so a boundary
                // signal after data has arrived would silently drop rows; refuse instead.
                Page::Unauthorized if collected.is_empty() => return Ok(Window::Unauthorized),
                Page::Unauthorized => bail!(
                    "polygon reported the plan boundary on page {pages} after {} rows: {url}",
                    collected.len()
                ),
                Page::Data { results, next_url } => {
                    collected.extend(results);
                    next = next_url;
                }
            }
        }
        Ok(Window::Data(collected))
    }

    /// Decodes one list-shaped page. A missing ticker yields an empty page.
    async fn fetch_page<T: DeserializeOwned>(&self, url: &str) -> Result<Page<T>> {
        match self.request(url).await? {
            Fetched::Unauthorized => Ok(Page::Unauthorized),
            Fetched::NotFound => Ok(Page::Data {
                results: Vec::new(),
                next_url: None,
            }),
            Fetched::Body(body) => decode(&body, url),
        }
    }

    /// Decodes an object-shaped `results`. A missing or out-of-plan record yields `None`.
    async fn fetch_object<T: DeserializeOwned>(&self, url: &str) -> Result<Option<T>> {
        let Fetched::Body(body) = self.request(url).await? else {
            return Ok(None);
        };
        #[derive(Deserialize)]
        struct One<T> {
            results: Option<T>,
        }
        let envelope: One<T> = serde_json::from_str(&body)
            .with_context(|| format!("decoding polygon record for {url}: {}", clip(&body)))?;
        Ok(envelope.results)
    }

    async fn request(&self, url: &str) -> Result<Fetched> {
        let _permit = self
            .permits
            .acquire()
            .await
            .context("polygon concurrency semaphore closed")?;
        let mut backoff = BASE_BACKOFF;
        let mut transient = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            match self.http.get(url).bearer_auth(&self.api_key).send().await {
                Ok(response) => {
                    let status = response.status();
                    if status.as_u16() == 429 || status.is_server_error() {
                        transient = format!("http {status}");
                        if let Some(hint) = retry_after(&response) {
                            backoff = hint;
                        }
                    } else if status == reqwest::StatusCode::NOT_FOUND {
                        return Ok(Fetched::NotFound);
                    } else {
                        let body = response
                            .text()
                            .await
                            .with_context(|| format!("reading polygon body for {url}"))?;
                        // Out-of-plan windows answer 403 carrying a NOT_AUTHORIZED envelope.
                        if !status.is_success() {
                            if is_not_authorized(&body) {
                                return Ok(Fetched::Unauthorized);
                            }
                            bail!("polygon http {status} for {url}: {}", clip(&body));
                        }
                        return Ok(Fetched::Body(body));
                    }
                }
                Err(error) => transient = error.to_string(),
            }
            if attempt < MAX_ATTEMPTS {
                tokio::time::sleep(backoff).await;
                backoff *= 2;
            }
        }
        bail!("polygon gave up after {MAX_ATTEMPTS} attempts ({transient}) for {url}")
    }
}

/// A server-stated `Retry-After` delay in seconds, clamped to a sane ceiling.
fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    let seconds: u64 = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(seconds.min(MAX_RETRY_AFTER_SECS)))
}

fn decode<T: DeserializeOwned>(body: &str, url: &str) -> Result<Page<T>> {
    let envelope: Envelope<T> = serde_json::from_str(body)
        .with_context(|| format!("decoding polygon body for {url}: {}", clip(body)))?;
    match envelope.status.as_deref() {
        Some(NOT_AUTHORIZED) => Ok(Page::Unauthorized),
        None | Some("OK") | Some("DELAYED") => Ok(Page::Data {
            results: envelope.results.unwrap_or_default(),
            next_url: envelope.next_url,
        }),
        Some(other) => bail!(
            "polygon status {other} for {url}: {}",
            envelope
                .message
                .or(envelope.error)
                .unwrap_or_else(|| clip(body))
        ),
    }
}

/// A `NOT_AUTHORIZED` envelope means the request fell outside the plan's rolling history window,
/// which Polygon reports on both `200` and `403` responses.
fn is_not_authorized(body: &str) -> bool {
    #[derive(Deserialize)]
    struct StatusOnly {
        status: Option<String>,
    }
    serde_json::from_str::<StatusOnly>(body)
        .is_ok_and(|envelope| envelope.status.as_deref() == Some(NOT_AUTHORIZED))
}

fn clip(body: &str) -> String {
    const MAX: usize = 240;
    match body.char_indices().nth(MAX) {
        Some((end, _)) => format!("{}…", &body[..end]),
        None => body.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_open_matches_polygon_daily_stamps_across_dst() {
        // Values taken from /v2/aggs/ticker/AAPL/range/1/day for those sessions.
        assert_eq!(
            session_open_ms(NaiveDate::from_ymd_opt(2025, 8, 18).unwrap()),
            1_755_489_600_000
        );
        assert_eq!(
            session_open_ms(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap()),
            1_767_589_200_000
        );
    }

    #[test]
    fn out_of_plan_window_bodies_are_recognized() {
        assert!(is_not_authorized(
            r#"{"status":"NOT_AUTHORIZED","message":"Your plan doesn't include this data timeframe."}"#
        ));
        assert!(!is_not_authorized(r#"{"status":"OK","results":[]}"#));
        assert!(!is_not_authorized("<html>gateway error</html>"));
    }

    #[test]
    fn aggregate_rows_map_onto_packed_bars() {
        let row: AggRow = serde_json::from_str(
            r#"{"v":8211,"vw":231.2144,"o":231.18,"c":231.2,"h":231.31,"l":231.18,"t":1755504000000,"n":712}"#,
        )
        .unwrap();
        let bar = row.to_bar();
        assert_eq!(bar.ts(), 1_755_504_000_000);
        assert_eq!(bar.open, 231.18);
        assert_eq!(bar.high, 231.31);
        assert_eq!(bar.low, 231.18);
        assert_eq!(bar.close, 231.2);
        assert_eq!(bar.volume, 8211.0);
        assert_eq!(bar.vwap, 231.2144);
        assert_eq!(bar.trades, 712);
    }

    #[test]
    fn missing_vwap_and_trade_count_fall_back() {
        let row: AggRow =
            serde_json::from_str(r#"{"o":10.0,"h":12.0,"l":9.0,"c":11.0,"t":1000}"#).unwrap();
        let bar = row.to_bar();
        assert_eq!(bar.vwap, (12.0f32 + 9.0 + 11.0) / 3.0);
        assert_eq!(bar.trades, 0);
        assert_eq!(bar.volume, 0.0);
    }

    #[test]
    fn grouped_rows_carry_the_symbol_alongside_the_aggregate() {
        let row: GroupedRow = serde_json::from_str(
            r#"{"T":"FEIM","v":364737,"vw":25.9263,"o":25.96,"c":26.45,"h":26.65,"l":24.7218,"t":1754078400000,"n":6390}"#,
        )
        .unwrap();
        assert_eq!(row.symbol, "FEIM");
        assert_eq!(row.agg.to_bar().trades, 6390);
    }
}
