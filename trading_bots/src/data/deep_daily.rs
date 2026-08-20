//! Free deep-history daily ingestion, to supply the market regimes the paid intraday corpus
//! structurally cannot contain.
//!
//! The Polygon plan serves a rolling five-year window: everything before roughly 2021-08-16
//! answers `NOT_AUTHORIZED`. So `long_data/bars/*.300.bars` holds no 2000 dot-com unwind, no 2008
//! credit crisis, and only the tail of 2020. Widening that corpus sideways does not help, because
//! equity returns are cross-sectionally correlated: more tickers over the same five years buy
//! bars, not regimes. A model that has never seen a crash is maximally confident exactly when it
//! is most wrong.
//!
//! This module fills the gap along the other axis — depth rather than breadth — by pulling free
//! daily history and writing it as `<SYM>.86400.bars` beside the intraday files, for the
//! pretraining pipeline to pick up through the resolution-class channel once it loads more than
//! one resolution. Nothing here is resolution-aware beyond stamping `86400`.
//!
//! # Source
//!
//! Yahoo Finance's chart API, chosen over Stooq after probing both:
//!
//! * Yahoo serves 1970-01-02 onward (XOM, GE, KO all start there; AAPL at its 1980-12-12 IPO),
//!   needs no key and no crumb for `v8/finance/chart`, and returns split events inline so the
//!   quality gate below can tell a corporate action from a data defect. It does require a browser
//!   `User-Agent` and a `finance.yahoo.com` `Origin`/`Referer`; without them every request answers
//!   `429`, which is a header check rather than a rate limit.
//! * Stooq's per-symbol CSV is gated by a SHA-256 proof-of-work (difficulty 4, ~65k hashes). The
//!   challenge is trivial to solve and `/__verify` does return a session cookie, but the CSV
//!   endpoint then answers the literal body `Access denied` anyway, so the free path is closed.
//!   Its bulk `/db/` archives require a paid account.
//!
//! # Adjustment
//!
//! Yahoo's `indicators.quote` OHLC and volume are already split-adjusted, and `adjclose` carries
//! the additional dividend adjustment. Prices here are adjusted for BOTH: every OHLC is scaled by
//! `adjclose / close`, which is piecewise constant between ex-dividend dates. That makes the log
//! return a total return and removes the artificial ex-dividend gap, which is what a return model
//! should see. Volume is left alone, since a dividend does not change the share count.
//!
//! # Ticker reuse
//!
//! Polygon keys aggregates by ticker STRING, so [`super::ingest::identity_handover`] exists to stop a
//! reused ticker splicing two companies into one series. Yahoo keys by ENTITY instead, and is not
//! subject to that failure: `META` returns 3580 rows starting at the 2012-05-18 Facebook IPO with
//! no trace of the Roundhill Ball Metaverse ETF that held the ticker until 2022-06-09, and `BBBY`
//! returns a single continuous entity. Clamping the floor to Polygon's reference data would
//! therefore cut the deep history this module exists to fetch, for no gain, so it is not done.
//! The splice class of the anomaly scan below remains as the check that this stays true.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use chrono_tz::America::New_York;
use chrono_tz::Tz;
use serde::Deserialize;
use shared::bars::{bar_file_path, write_bar_file, BarFile, PackedBar};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use super::ingest::{bars_dir, universe_entries};
use crate::torch::dataset::{
    SymbolAnomalies, ANOMALY_HOLE_DAYS, ANOMALY_LOG_LIMIT, ANOMALY_WORST_LISTED,
};

const CHART_URL: &str = "https://query1.finance.yahoo.com/v8/finance/chart";
/// Yahoo answers `429` to any client that does not look like a browser, whatever the request rate.
const BROWSER_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36";
const MAX_ATTEMPTS: u32 = 5;
const BASE_BACKOFF: Duration = Duration::from_millis(750);
const MAX_RETRY_AFTER_SECS: u64 = 60;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const PROGRESS_EVERY: usize = 25;
const DAILY_RES_SECS: u32 = 86_400;
/// Modest by construction: one request per symbol, and the whole universe is ~3000 requests.
pub const DEFAULT_CONCURRENCY: usize = 8;
/// A split's price seam can sit a session either side of the stamped event date, and a stale
/// weekend stamp shifts it further, so a jump counts as explained within this many days of one.
const SPLIT_MATCH_DAYS: i64 = 3;
/// A daily bar is only complete once its session has closed.
const SESSION_CLOSE_HOUR: u32 = 16;

/// The first session the Polygon plan serves; anything earlier is what this module is for.
pub const POLYGON_FLOOR: (i32, u32, u32) = (2021, 8, 16);
/// Depth milestones reported by [`DeepDailySummary`], newest last.
const COVERAGE_EPOCHS: [(i32, u32, u32); 3] = [(2001, 1, 1), (2009, 1, 1), POLYGON_FLOOR];

/// Command-line surface of the `deep-daily` subcommand.
#[derive(Clone, Debug)]
pub struct DeepDailyArgs {
    /// Liquidity floor the cached ranking is filtered by, matching `ingest --min-dollar-volume`.
    pub min_dollar_volume: f64,
    /// Take only the first `limit` symbols of that ranking; `0` takes all of them.
    pub limit: usize,
    pub concurrency: usize,
    /// Rewrite symbols whose daily file already reaches past the Polygon floor.
    pub force: bool,
    /// Fetch and audit, but write nothing.
    pub dry_run: bool,
}

/// An extreme level move the source does not explain with a split. See [`scan`].
#[derive(Clone, Debug)]
pub struct Seam {
    /// Index of the bar the move lands ON, so `bars[index..]` is the history after it.
    pub index: usize,
    pub date: NaiveDate,
    pub kind: &'static str,
    pub log_return: f64,
}

/// Below this, the clean tail left by a truncation is not worth a corpus file.
const MIN_TRUNCATED_BARS: usize = 250;

/// Tally of one deep-history pass.
#[derive(Clone, Debug, Default)]
pub struct DeepDailySummary {
    pub attempted: usize,
    pub written: usize,
    pub skipped: usize,
    /// The source has no history for the symbol at all.
    pub empty: usize,
    /// Withheld entirely by the quality gate: no clean tail long enough to be worth keeping.
    pub rejected: usize,
    /// Written, but with a corrupt prefix cut away first.
    pub truncated: usize,
    /// Bars discarded by those truncations.
    pub dropped_bars: usize,
    /// Bars whose high/low envelope was widened to contain a source open or close.
    pub repaired_bars: usize,
    pub failed: usize,
    pub bars: usize,
    pub first_ts_ms: Option<i64>,
    pub last_ts_ms: Option<i64>,
    /// Written symbols whose history reaches before each of [`COVERAGE_EPOCHS`].
    pub coverage: [usize; COVERAGE_EPOCHS.len()],
    /// Anomaly counts for every symbol that passed the gate, worst rate first.
    pub anomalies: Vec<SymbolAnomalies>,
}

/// Downloads deep daily history for the cached universe and writes it into the bar corpus.
pub async fn run(args: DeepDailyArgs) -> Result<()> {
    let entries = universe_entries(args.min_dollar_volume)?
        .context("no measured universe cached; run `ingest --refresh-universe` first")?;
    let mut symbols: Vec<String> = entries.into_iter().map(|entry| entry.symbol).collect();
    if args.limit > 0 {
        symbols.truncate(args.limit);
    }
    let out_dir = bars_dir();
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating {}", out_dir.display()))?;

    println!(
        "[deep-daily] {} symbols clearing ${:.0}/day -> {} (concurrency {}{})",
        symbols.len(),
        args.min_dollar_volume,
        out_dir.display(),
        args.concurrency,
        if args.dry_run { ", dry run" } else { "" }
    );
    let summary = ingest(&symbols, &out_dir, &args).await?;
    report(&summary);
    Ok(())
}

/// Fetches every symbol under a concurrency bound, gating each one before it is written.
pub async fn ingest(
    symbols: &[String],
    out_dir: &Path,
    args: &DeepDailyArgs,
) -> Result<DeepDailySummary> {
    let client = Arc::new(YahooClient::new(args.concurrency)?);
    let mut tasks: JoinSet<SymbolOutcome> = JoinSet::new();
    for symbol in symbols {
        let client = Arc::clone(&client);
        let symbol = symbol.clone();
        let path = bar_file_path(out_dir, &symbol, DAILY_RES_SECS);
        let (force, dry_run) = (args.force, args.dry_run);
        tasks.spawn(async move { fetch_one(&client, symbol, path, force, dry_run).await });
    }
    drain(&mut tasks, symbols.len()).await
}

/// One symbol, end to end: skip check, fetch, convert, audit, gate, write.
async fn fetch_one(
    client: &YahooClient,
    symbol: String,
    path: PathBuf,
    force: bool,
    dry_run: bool,
) -> SymbolOutcome {
    if !force && already_deep(&path).await {
        return SymbolOutcome::Skipped;
    }
    let chart = match client.chart(&query_symbol(&symbol)).await {
        Ok(Some(chart)) => chart,
        Ok(None) => return SymbolOutcome::Empty(symbol),
        Err(error) => return SymbolOutcome::Failed(symbol, format!("{error:#}")),
    };
    let Converted {
        mut bars,
        splits,
        repaired,
    } = match to_bars(&chart) {
        Ok(converted) => converted,
        Err(error) => return SymbolOutcome::Failed(symbol, format!("{error:#}")),
    };
    if bars.len() < 2 {
        return SymbolOutcome::Empty(symbol);
    }
    let (anomalies, cut) = match truncate_at_last_seam(&symbol, &mut bars, &splits) {
        Ok(kept) => kept,
        Err((seams, reason)) => return SymbolOutcome::Rejected(symbol, seams, reason),
    };
    if dry_run {
        return SymbolOutcome::Written {
            bars: bars.len(),
            first_ts_ms: bars[0].ts(),
            last_ts_ms: bars[bars.len() - 1].ts(),
            anomalies,
            cut,
            repaired,
        };
    }
    persist(path, symbol, bars, anomalies, cut, repaired).await
}

/// A corrupt prefix cut away before writing.
#[derive(Debug)]
struct Truncation {
    /// Every unexplained seam found, in ascending date order; the last one is the cut point.
    seams: Vec<Seam>,
    /// Bars discarded.
    dropped: usize,
    /// First session actually written.
    kept_from: NaiveDate,
}

/// Cuts a corrupt prefix away so the rest can be kept, or reports why nothing can be.
///
/// Every seam is a boundary between two incompatible price levels, so the history AFTER the last
/// one is internally consistent even when the history before it is not. Keeping that tail recovers
/// the decades a whole-symbol reject would throw away — NVR's 1993 emergence from bankruptcy and
/// HUBB's share-class reclassification each corrupt only a prefix — while still admitting no
/// unexplained level move anywhere in what is written.
#[allow(clippy::type_complexity)]
fn truncate_at_last_seam(
    symbol: &str,
    bars: &mut Vec<PackedBar>,
    splits: &BTreeSet<NaiveDate>,
) -> Result<(SymbolAnomalies, Option<Truncation>), (Vec<Seam>, &'static str)> {
    let (anomalies, seams) = scan(symbol, bars, splits);
    let Some(last) = seams.last() else {
        return Ok((anomalies, None));
    };
    let dropped = last.index;
    if bars.len() - dropped < MIN_TRUNCATED_BARS {
        return Err((seams, "no clean tail long enough after the last seam"));
    }
    bars.drain(..dropped);
    let (rescanned, residual) = scan(symbol, bars, splits);
    // Dropping a prefix cannot manufacture an extreme return, and a leg excused as a tick recovery
    // can never be the cut point, so the cut cannot strand one. This is a tripwire on that
    // reasoning rather than an expected branch; if it fires the tail is genuinely unusable.
    if !residual.is_empty() {
        return Err((residual, "a seam survived the cut at the last seam"));
    }
    Ok((
        rescanned,
        Some(Truncation {
            seams,
            dropped,
            kept_from: session_date(bars[0].ts()),
        }),
    ))
}

/// Writes the corpus file off the async runtime; the buffer and path are already owned here.
async fn persist(
    path: PathBuf,
    symbol: String,
    bars: Vec<PackedBar>,
    anomalies: SymbolAnomalies,
    cut: Option<Truncation>,
    repaired: usize,
) -> SymbolOutcome {
    let name = symbol.clone();
    let written = tokio::task::spawn_blocking(move || {
        let first_ts_ms = bars[0].ts();
        let last_ts_ms = bars[bars.len() - 1].ts();
        write_bar_file(&path, &symbol, DAILY_RES_SECS, &bars)
            .map(|()| (bars.len(), first_ts_ms, last_ts_ms))
    })
    .await;
    match written {
        Ok(Ok((bars, first_ts_ms, last_ts_ms))) => SymbolOutcome::Written {
            bars,
            first_ts_ms,
            last_ts_ms,
            anomalies,
            cut,
            repaired,
        },
        Ok(Err(error)) => SymbolOutcome::Failed(name, format!("{error:#}")),
        Err(error) => SymbolOutcome::Failed(name, format!("write task panicked: {error}")),
    }
}

enum SymbolOutcome {
    Written {
        bars: usize,
        first_ts_ms: i64,
        last_ts_ms: i64,
        anomalies: SymbolAnomalies,
        cut: Option<Truncation>,
        repaired: usize,
    },
    Skipped,
    Empty(String),
    Rejected(String, Vec<Seam>, &'static str),
    Failed(String, String),
}

async fn drain(tasks: &mut JoinSet<SymbolOutcome>, total: usize) -> Result<DeepDailySummary> {
    let started = Instant::now();
    let mut summary = DeepDailySummary {
        attempted: total,
        ..Default::default()
    };
    let mut completed = 0usize;
    while let Some(joined) = tasks.join_next().await {
        completed += 1;
        match joined.context("deep-daily task panicked")? {
            SymbolOutcome::Written {
                bars,
                first_ts_ms,
                last_ts_ms,
                anomalies,
                cut,
                repaired,
            } => {
                summary.repaired_bars += repaired;
                if let Some(cut) = cut {
                    summary.truncated += 1;
                    summary.dropped_bars += cut.dropped;
                    for seam in &cut.seams {
                        eprintln!(
                            "[deep-daily] SEAM {} {}: unexplained {} of {:.3}x (ln {:+.3}) with \
                             no split event from the source",
                            anomalies.symbol,
                            seam.date,
                            seam.kind,
                            seam.log_return.exp(),
                            seam.log_return
                        );
                    }
                    eprintln!(
                        "[deep-daily] TRUNCATE {}: dropped {} bars before {}, kept {bars}",
                        anomalies.symbol, cut.dropped, cut.kept_from
                    );
                }
                summary.written += 1;
                summary.bars += bars;
                summary.first_ts_ms = Some(
                    summary
                        .first_ts_ms
                        .map_or(first_ts_ms, |current| current.min(first_ts_ms)),
                );
                summary.last_ts_ms = Some(
                    summary
                        .last_ts_ms
                        .map_or(last_ts_ms, |current| current.max(last_ts_ms)),
                );
                for (slot, epoch) in coverage_epochs().into_iter().enumerate() {
                    if first_ts_ms < session_open_ms(epoch) {
                        summary.coverage[slot] += 1;
                    }
                }
                summary.anomalies.push(anomalies);
            }
            SymbolOutcome::Skipped => summary.skipped += 1,
            SymbolOutcome::Empty(symbol) => {
                summary.empty += 1;
                eprintln!("[deep-daily] {symbol}: source has no history");
            }
            SymbolOutcome::Rejected(symbol, seams, reason) => {
                summary.rejected += 1;
                for seam in &seams {
                    eprintln!(
                        "[deep-daily] SEAM {symbol} {}: unexplained {} of {:.3}x (ln {:+.3}) \
                         with no split event from the source",
                        seam.date,
                        seam.kind,
                        seam.log_return.exp(),
                        seam.log_return
                    );
                }
                eprintln!(
                    "[deep-daily] REJECT {symbol}: {reason} (floor {MIN_TRUNCATED_BARS} bars)"
                );
            }
            SymbolOutcome::Failed(symbol, error) => {
                summary.failed += 1;
                eprintln!("[deep-daily] {symbol}: {error}");
            }
        }
        if completed % PROGRESS_EVERY == 0 || completed == total {
            println!(
                "[deep-daily] {completed}/{total} symbols | {} bars | {} written ({} truncated), \
                 {} rejected, {} empty, {} failed | {:.1}s",
                summary.bars,
                summary.written,
                summary.truncated,
                summary.rejected,
                summary.empty,
                summary.failed,
                started.elapsed().as_secs_f64()
            );
        }
    }
    summary.anomalies.sort_unstable_by(|a, b| {
        b.anomaly_rate()
            .total_cmp(&a.anomaly_rate())
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    Ok(summary)
}

// ---------------------------------------------------------------------------
// Quality gate
// ---------------------------------------------------------------------------

/// Classifies every extreme move in one symbol, and reports the ones that disqualify it.
///
/// The taxonomy and the `ln 4` threshold are exactly [`crate::torch::dataset::scan_anomalies`]'s,
/// reusing its constants and its [`SymbolAnomalies`] so the two audits are directly comparable.
/// The audit there is corpus-wide and counts only; this one runs on the in-memory series BEFORE
/// anything is written, because the point is to keep bad history off disk rather than to describe
/// it afterwards.
///
/// The gate is only as sharp as its threshold: it admits no unexplained level move ABOVE `ln 4`,
/// but an unadjusted 2:1 or 3:2 split (`ln 0.69`, `ln 0.41`) passes it silently. Retuning the
/// constant would decouple this audit from the corpus-wide one for a class of defect the source
/// has not been observed to produce, so the threshold is shared deliberately and the residual
/// risk is stated rather than papered over.
///
/// A `jump` or a `splice` is a level shift: every price after it is wrong relative to every price
/// before it, which is precisely how an unadjusted split looks, and a fabricated crash is worse
/// than a missing one. Such a symbol is withheld unless the source declares a split within
/// [`SPLIT_MATCH_DAYS`] of the seam. A `tick` reverts on the very next bar, so it is an isolated
/// bad print rather than a level error; it is counted and reported but does not withhold the
/// symbol, matching the corpus policy that breaking series continuity is worse than the anomaly.
///
/// The two decisions are deliberately not the same predicate. The counts above stay bit-identical
/// to the corpus audit, which classifies each return independently and therefore books a one-bar
/// spike as a tick on the way up AND a jump on the way down, since the recovery leg has no
/// successor of its own to revert into. That second leg is the same event as the first, not an
/// independent level shift — the level before the spike and the level after it agree — so the
/// GATE attributes it to the tick and does not withhold on it. Without that, the tick carve-out
/// would be unreachable and every isolated bad print would cost a whole symbol.
fn scan(
    symbol: &str,
    bars: &[PackedBar],
    splits: &BTreeSet<NaiveDate>,
) -> (SymbolAnomalies, Vec<Seam>) {
    let mut out = SymbolAnomalies {
        symbol: symbol.to_string(),
        bars: bars.len(),
        holes: 0,
        splices: 0,
        ticks: 0,
        jumps: 0,
        extreme_range: 0,
    };
    let mut seams = Vec::new();
    if bars.len() < 2 {
        return (out, seams);
    }
    let hole_ms = ANOMALY_HOLE_DAYS * 86_400_000;
    // `r` and `s` as `BarDof` defines them: ln(close / prev_close) and ln(high / low).
    let mut returns = Vec::with_capacity(bars.len() - 1);
    for window in bars.windows(2) {
        let (prev_close, high, low, close) =
            (window[0].close, window[1].high, window[1].low, window[1].close);
        if (high / low).ln() as f64 > ANOMALY_LOG_LIMIT {
            out.extreme_range += 1;
        }
        returns.push((close / prev_close).ln() as f64);
    }
    for (index, &log_return) in returns.iter().enumerate() {
        // `returns[index]` is the return of bar `index + 1` against bar `index`.
        let bar = index + 1;
        let gap = bars[bar].ts() - bars[bar - 1].ts();
        if gap >= hole_ms {
            out.holes += 1;
        }
        if log_return.abs() <= ANOMALY_LOG_LIMIT {
            continue;
        }
        // The pair `(returns[index], returns[index + 1])` is a tick exactly when the second
        // cancels the first AND no hole intervenes, since the ladder below lets a hole outrank a
        // reversion. `is_tick_recovery` is that same predicate evaluated one step back, guards
        // included: without the hole guard a leg classified `splice` would also excuse its
        // successor, and the cut at the splice would then strand that successor as the kept
        // tail's first return, tripping the residual check and losing the whole symbol.
        let reverts = |first: f64, second: f64| (first + second).abs() < 0.5 * first.abs();
        let is_tick = gap < hole_ms
            && returns
                .get(index + 1)
                .is_some_and(|&next| reverts(log_return, next));
        let is_tick_recovery = index.checked_sub(1).is_some_and(|previous| {
            let leg = returns[previous];
            leg.abs() > ANOMALY_LOG_LIMIT
                && bars[previous + 1].ts() - bars[previous].ts() < hole_ms
                && reverts(leg, log_return)
        });
        let kind = if gap >= hole_ms {
            out.splices += 1;
            "splice"
        } else if is_tick {
            out.ticks += 1;
            "tick"
        } else {
            out.jumps += 1;
            "jump"
        };
        let date = session_date(bars[bar].ts());
        if kind != "tick" && !is_tick_recovery && !near_split(date, splits) {
            seams.push(Seam {
                index: bar,
                date,
                kind,
                log_return,
            });
        }
    }
    (out, seams)
}

/// True when the source declares a split within [`SPLIT_MATCH_DAYS`] of `date`.
fn near_split(date: NaiveDate, splits: &BTreeSet<NaiveDate>) -> bool {
    let span = chrono::Duration::days(SPLIT_MATCH_DAYS);
    splits.range(date - span..=date + span).next().is_some()
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

/// One converted symbol: strictly increasing fully adjusted bars, the split dates the gate needs,
/// and how many bars had to have their high/low envelope repaired.
struct Converted {
    bars: Vec<PackedBar>,
    splits: BTreeSet<NaiveDate>,
    repaired: usize,
}

/// Converts one chart payload into daily bars.
fn to_bars(chart: &ChartResult) -> Result<Converted> {
    let stamps = chart.timestamp.as_deref().unwrap_or_default();
    let quote = chart
        .indicators
        .quote
        .first()
        .context("chart payload carries no quote series")?;
    let adjclose = chart
        .indicators
        .adjclose
        .as_ref()
        .and_then(|series| series.first())
        .and_then(|entry| entry.adjclose.as_deref());
    let splits: BTreeSet<NaiveDate> = chart
        .events
        .as_ref()
        .and_then(|events| events.splits.as_ref())
        .map(|splits| splits.values().map(|split| session_date(split.date * 1000)).collect())
        .unwrap_or_default();

    let now = Utc::now().with_timezone(&New_York);
    // A `BTreeMap` both deduplicates the occasional repeated stamp, keeping the last value, and
    // guarantees the strictly increasing order `write_bar_file` expects.
    let mut rows: BTreeMap<i64, PackedBar> = BTreeMap::new();
    let mut repaired = 0usize;
    for (index, &stamp) in stamps.iter().enumerate() {
        let (Some(open), Some(high), Some(low), Some(close)) = (
            positive(quote.open.get(index).copied().flatten()),
            positive(quote.high.get(index).copied().flatten()),
            positive(quote.low.get(index).copied().flatten()),
            positive(quote.close.get(index).copied().flatten()),
        ) else {
            continue;
        };
        let date = session_date(stamp * 1000);
        if !session_complete(date, now) {
            continue;
        }
        // `adjclose / close` is the dividend adjustment; the split adjustment is already in both.
        // A symbol with no adjclose series at all is consistently unadjusted, which is benign. A
        // symbol that HAS one but is missing this row is not: falling back to 1.0 there would
        // write a single bar at the unadjusted level among dividend-adjusted neighbours, and the
        // resulting spike-and-recovery of the cumulative dividend factor is typically well under
        // `ln 4`, so the gate would never see the move it fabricated. Skip the row instead — a
        // one-day hole is indistinguishable from a weekend, and silence is the failure to avoid.
        let factor = match adjclose {
            Some(series) => match positive(series.get(index).copied().flatten()) {
                Some(adjusted) => adjusted / close,
                None => continue,
            },
            None => 1.0,
        };
        let (open, close) = (open * factor, close * factor);
        // The source occasionally reports an open or a close outside its own high/low envelope —
        // 8 bars in 15.8M, clustered on one bad feed day. The open and the close are real prints
        // and the return is read off the close, so the envelope is widened to contain them rather
        // than the bar being dropped. This is a no-op for a well-formed bar, and it makes
        // `low <= min(open, close) <= max(open, close) <= high` hold for every bar written, which
        // the DOF encoder relies on to keep the close's position inside the log range in `[0, 1]`.
        let (raw_high, raw_low) = (high * factor, low * factor);
        let high = raw_high.max(open).max(close);
        let low = raw_low.min(open).min(close);
        if high != raw_high || low != raw_low {
            repaired += 1;
        }
        let volume = quote
            .volume
            .get(index)
            .copied()
            .flatten()
            .filter(|volume| volume.is_finite())
            .unwrap_or(0.0)
            .max(0.0);
        let ts_ms = session_open_ms(date);
        rows.insert(
            ts_ms,
            PackedBar {
                ts_ms,
                open: open as f32,
                high: high as f32,
                low: low as f32,
                close: close as f32,
                volume: volume as f32,
                // The source reports neither a daily VWAP nor a trade count.
                vwap: ((high + low + close) / 3.0) as f32,
                trades: 0,
            },
        );
    }
    Ok(Converted {
        bars: rows.into_values().collect(),
        splits,
        repaired,
    })
}

/// A price is usable only when it is finite and strictly positive; the series is read in logs.
fn positive(value: Option<f64>) -> Option<f64> {
    value.filter(|price| price.is_finite() && *price > 0.0)
}

/// True once `date`'s regular session has closed, so its daily bar can no longer change.
fn session_complete(date: NaiveDate, now: DateTime<Tz>) -> bool {
    date.and_hms_opt(SESSION_CLOSE_HOUR, 0, 0)
        .and_then(|close| close.and_local_timezone(New_York).earliest())
        .is_none_or(|close| now >= close)
}

/// Midnight New York time for a session date, in UTC milliseconds. Matches the convention
/// [`super::polygon::grouped_daily_window`] rewrites its 16:00 ET close stamps to, so a daily bar
/// from either source denotes the same instant.
fn session_open_ms(date: NaiveDate) -> i64 {
    let midnight = date
        .and_hms_opt(0, 0, 0)
        .expect("midnight is a valid local time");
    midnight
        .and_local_timezone(New_York)
        .earliest()
        .map(|stamp| stamp.timestamp_millis())
        .unwrap_or_else(|| midnight.and_utc().timestamp_millis())
}

/// The New York session date an instant belongs to.
fn session_date(ts_ms: i64) -> NaiveDate {
    New_York
        .timestamp_millis_opt(ts_ms)
        .earliest()
        .map(|stamp| stamp.date_naive())
        .unwrap_or_else(|| {
            DateTime::<Utc>::from_timestamp_millis(ts_ms)
                .unwrap_or_default()
                .date_naive()
        })
}

fn coverage_epochs() -> [NaiveDate; COVERAGE_EPOCHS.len()] {
    COVERAGE_EPOCHS.map(|(year, month, day)| {
        NaiveDate::from_ymd_opt(year, month, day).expect("coverage epoch is a valid date")
    })
}

/// Polygon writes `BRK.B`, Yahoo indexes the same security as `BRK-B`. The corpus file keeps the
/// Polygon spelling so both resolutions of one symbol stay side by side.
fn query_symbol(symbol: &str) -> String {
    symbol.replace('.', "-")
}

/// True when a daily file for this symbol already reaches past the Polygon floor, which only this
/// path can produce. Reads the header off the async runtime.
async fn already_deep(path: &Path) -> bool {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let Ok(file) = BarFile::open(&path) else {
            return false;
        };
        let floor = session_open_ms(coverage_epochs()[COVERAGE_EPOCHS.len() - 1]);
        file.bars().first().is_some_and(|bar| bar.ts() < floor)
    })
    .await
    .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn report(summary: &DeepDailySummary) {
    let epochs = coverage_epochs();
    println!(
        "[deep-daily] {} attempted | {} written, {} skipped, {} rejected, {} empty, {} failed",
        summary.attempted,
        summary.written,
        summary.skipped,
        summary.rejected,
        summary.empty,
        summary.failed
    );
    println!(
        "[deep-daily] quality gate: {} symbols written with a corrupt prefix cut away \
         ({} bars dropped), {} withheld entirely, {} bar envelopes repaired",
        summary.truncated, summary.dropped_bars, summary.rejected, summary.repaired_bars
    );
    println!(
        "[deep-daily] {} bars spanning {} .. {}",
        summary.bars,
        summary.first_ts_ms.map_or("-".to_string(), format_ts),
        summary.last_ts_ms.map_or("-".to_string(), format_ts)
    );
    for (slot, epoch) in epochs.into_iter().enumerate() {
        println!(
            "[deep-daily] symbols with data before {epoch}: {}",
            summary.coverage[slot]
        );
    }

    let bars: usize = summary.anomalies.iter().map(|entry| entry.bars).sum();
    let total: usize = summary.anomalies.iter().map(SymbolAnomalies::total).sum();
    let sum = |pick: fn(&SymbolAnomalies) -> usize| -> usize {
        summary.anomalies.iter().map(pick).sum()
    };
    println!(
        "[deep-daily] {total} anomalous bars in {bars} ({:.2}/10k) at |r| or s > ln 4: \
         {} splices, {} ticks, {} jumps, {} extreme ranges, {} interior holes",
        if bars == 0 {
            0.0
        } else {
            total as f64 * 10_000.0 / bars as f64
        },
        sum(|entry| entry.splices),
        sum(|entry| entry.ticks),
        sum(|entry| entry.jumps),
        sum(|entry| entry.extreme_range),
        sum(|entry| entry.holes),
    );
    let worst: Vec<&SymbolAnomalies> = summary
        .anomalies
        .iter()
        .filter(|entry| entry.total() > 0)
        .take(ANOMALY_WORST_LISTED)
        .collect();
    if worst.is_empty() {
        println!("[deep-daily] no anomalies in any written symbol");
        return;
    }
    println!(
        "[deep-daily] worst {} of {}: {:<8} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>8}",
        worst.len(),
        summary.anomalies.len(),
        "symbol",
        "bars",
        "splice",
        "tick",
        "jump",
        "range",
        "hole",
        "per10k"
    );
    for entry in worst {
        println!(
            "[deep-daily]              {:<8} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>8.2}",
            entry.symbol,
            entry.bars,
            entry.splices,
            entry.ticks,
            entry.jumps,
            entry.extreme_range,
            entry.holes,
            entry.anomaly_rate()
        );
    }
}

fn format_ts(ts_ms: i64) -> String {
    session_date(ts_ms).to_string()
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

struct YahooClient {
    http: reqwest::Client,
    permits: Semaphore,
}

impl YahooClient {
    fn new(concurrency: usize) -> Result<Self> {
        use reqwest::header::{HeaderMap, HeaderValue};
        let mut headers = HeaderMap::new();
        // Yahoo rejects anything that does not present as a browser navigating its own site,
        // answering `429 Too Many Requests` on the very first request when these are absent.
        headers.insert("accept", HeaderValue::from_static("application/json,text/plain,*/*"));
        headers.insert("accept-language", HeaderValue::from_static("en-US,en;q=0.9"));
        headers.insert("origin", HeaderValue::from_static("https://finance.yahoo.com"));
        headers.insert("referer", HeaderValue::from_static("https://finance.yahoo.com/"));
        headers.insert(
            "sec-ch-ua",
            HeaderValue::from_static("\"Chromium\";v=\"128\", \"Not;A=Brand\";v=\"24\""),
        );
        headers.insert("sec-ch-ua-platform", HeaderValue::from_static("\"Windows\""));
        let http = reqwest::Client::builder()
            .user_agent(BROWSER_USER_AGENT)
            .default_headers(headers)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .context("building the yahoo http client")?;
        Ok(Self {
            http,
            permits: Semaphore::new(concurrency.max(1)),
        })
    }

    /// The symbol's entire daily history. `None` when the source knows no such symbol.
    async fn chart(&self, symbol: &str) -> Result<Option<ChartResult>> {
        // `period1=0` asks for everything the source has; the epoch floor is not a real bound.
        let url = format!(
            "{CHART_URL}/{symbol}?period1=0&period2={}&interval=1d&events=div%2Csplit",
            Utc::now().timestamp() + 86_400
        );
        let Some(body) = self.get(&url).await? else {
            return Ok(None);
        };
        let envelope: ChartEnvelope = serde_json::from_str(&body)
            .with_context(|| format!("decoding yahoo chart for {symbol}"))?;
        Ok(envelope
            .chart
            .result
            .unwrap_or_default()
            .into_iter()
            .next())
    }

    /// One GET with bounded retries. `None` is a `404`, meaning the symbol does not exist.
    async fn get(&self, url: &str) -> Result<Option<String>> {
        let _permit = self
            .permits
            .acquire()
            .await
            .context("deep-daily concurrency semaphore closed")?;
        let mut backoff = BASE_BACKOFF;
        let mut transient = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            match self.http.get(url).send().await {
                Ok(response) => {
                    let status = response.status();
                    if status == reqwest::StatusCode::NOT_FOUND {
                        return Ok(None);
                    }
                    if status.as_u16() == 429 || status.is_server_error() {
                        transient = format!("http {status}");
                        if let Some(hint) = retry_after(&response) {
                            backoff = hint;
                        }
                    } else if !status.is_success() {
                        let body = response.text().await.unwrap_or_default();
                        anyhow::bail!("yahoo http {status} for {url}: {}", clip(&body));
                    } else {
                        return Ok(Some(
                            response
                                .text()
                                .await
                                .with_context(|| format!("reading yahoo body for {url}"))?,
                        ));
                    }
                }
                Err(error) => transient = error.to_string(),
            }
            if attempt < MAX_ATTEMPTS {
                tokio::time::sleep(backoff).await;
                // The permit is held across every sleep, which is the point — it throttles the
                // whole pass rather than one symbol. That makes an unbounded doubling expensive:
                // a 60s `Retry-After` would otherwise reach 480s and park a permit for minutes.
                backoff = (backoff * 2).min(Duration::from_secs(MAX_RETRY_AFTER_SECS));
            }
        }
        anyhow::bail!("yahoo gave up after {MAX_ATTEMPTS} attempts ({transient}) for {url}")
    }
}

/// A server-stated `Retry-After` delay in seconds, clamped to a sane ceiling.
fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    let seconds = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(Duration::from_secs(seconds.min(MAX_RETRY_AFTER_SECS)))
}

fn clip(body: &str) -> String {
    const LIMIT: usize = 200;
    match body.char_indices().nth(LIMIT) {
        Some((end, _)) => format!("{}...", &body[..end]),
        None => body.to_string(),
    }
}

#[derive(Deserialize)]
struct ChartEnvelope {
    chart: ChartBody,
}

#[derive(Deserialize)]
struct ChartBody {
    result: Option<Vec<ChartResult>>,
}

#[derive(Deserialize)]
struct ChartResult {
    timestamp: Option<Vec<i64>>,
    indicators: Indicators,
    events: Option<Events>,
}

#[derive(Deserialize)]
struct Indicators {
    #[serde(default)]
    quote: Vec<Quote>,
    adjclose: Option<Vec<AdjClose>>,
}

#[derive(Deserialize)]
struct Quote {
    #[serde(default)]
    open: Vec<Option<f64>>,
    #[serde(default)]
    high: Vec<Option<f64>>,
    #[serde(default)]
    low: Vec<Option<f64>>,
    #[serde(default)]
    close: Vec<Option<f64>>,
    #[serde(default)]
    volume: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct AdjClose {
    adjclose: Option<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct Events {
    splits: Option<BTreeMap<String, SplitEvent>>,
}

#[derive(Deserialize)]
struct SplitEvent {
    /// Seconds, at the session open of the split date.
    date: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike};

    fn bar(date: (i32, u32, u32), open: f32, high: f32, low: f32, close: f32) -> PackedBar {
        let date = NaiveDate::from_ymd_opt(date.0, date.1, date.2).unwrap();
        PackedBar {
            ts_ms: session_open_ms(date),
            open,
            high,
            low,
            close,
            volume: 1_000.0,
            vwap: (high + low + close) / 3.0,
            trades: 0,
        }
    }

    #[test]
    fn session_open_is_et_midnight_across_the_dst_boundary() {
        // EST is UTC-5, EDT is UTC-4, so ET midnight is 05:00Z in winter and 04:00Z in summer.
        let winter = session_open_ms(NaiveDate::from_ymd_opt(2009, 1, 2).unwrap());
        let summer = session_open_ms(NaiveDate::from_ymd_opt(2009, 7, 2).unwrap());
        assert_eq!(
            DateTime::<Utc>::from_timestamp_millis(winter).unwrap().hour(),
            5
        );
        assert_eq!(
            DateTime::<Utc>::from_timestamp_millis(summer).unwrap().hour(),
            4
        );
        assert_eq!(session_date(winter), NaiveDate::from_ymd_opt(2009, 1, 2).unwrap());
        assert_eq!(session_date(summer), NaiveDate::from_ymd_opt(2009, 7, 2).unwrap());
    }

    #[test]
    fn an_unexplained_level_jump_withholds_the_symbol() {
        // A 20x overnight drop with no split declared: an unadjusted reverse split, not a crash.
        let bars = [
            bar((2026, 5, 21), 1282.0, 1301.0, 1217.0, 1229.0),
            bar((2026, 5, 22), 1186.0, 1195.0, 1112.0, 1146.0),
            bar((2026, 5, 26), 68.0, 68.6, 61.1, 62.1),
            bar((2026, 5, 27), 57.6, 68.3, 57.6, 64.5),
        ];
        let (anomalies, seams) = scan("SOXS", &bars, &BTreeSet::new());
        assert_eq!(anomalies.jumps, 1);
        assert_eq!(seams.len(), 1);
        assert_eq!(seams[0].kind, "jump");
        assert_eq!(seams[0].date, NaiveDate::from_ymd_opt(2026, 5, 26).unwrap());
        assert_eq!(seams[0].index, 2, "the cut point is the bar the move lands on");
    }

    #[test]
    fn a_split_declared_by_the_source_explains_the_same_jump() {
        let bars = [
            bar((2026, 5, 21), 1282.0, 1301.0, 1217.0, 1229.0),
            bar((2026, 5, 22), 1186.0, 1195.0, 1112.0, 1146.0),
            bar((2026, 5, 26), 68.0, 68.6, 61.1, 62.1),
            bar((2026, 5, 27), 57.6, 68.3, 57.6, 64.5),
        ];
        let splits = BTreeSet::from([NaiveDate::from_ymd_opt(2026, 5, 26).unwrap()]);
        let (anomalies, seams) = scan("SOXS", &bars, &splits);
        assert_eq!(anomalies.jumps, 1, "the move is still counted");
        assert!(seams.is_empty(), "but the source explains it");
    }

    #[test]
    fn a_reverting_print_is_a_tick_and_does_not_withhold_the_symbol() {
        let bars = [
            bar((2009, 1, 2), 10.0, 10.2, 9.9, 10.0),
            bar((2009, 1, 5), 10.0, 10.1, 9.8, 10.0),
            bar((2009, 1, 6), 10.0, 100.0, 9.9, 100.0),
            bar((2009, 1, 7), 10.0, 10.1, 9.9, 10.0),
        ];
        let (anomalies, seams) = scan("TICK", &bars, &BTreeSet::new());
        assert_eq!(anomalies.ticks, 1);
        // The recovery leg is booked a jump, exactly as the corpus audit books it: it has no
        // successor of its own to revert into. The gate still recognises it as the tick's tail.
        assert_eq!(anomalies.jumps, 1);
        assert_eq!(anomalies.extreme_range, 1, "only the bad print's own bar spans 10x");
        assert!(
            seams.is_empty(),
            "a spike that returns to its own level is not a level shift: {seams:?}"
        );
    }

    #[test]
    fn an_extreme_move_across_an_interior_hole_is_a_splice() {
        let bars = [
            bar((2009, 1, 2), 10.0, 10.2, 9.9, 10.0),
            bar((2009, 3, 2), 100.0, 101.0, 99.0, 100.0),
            bar((2009, 3, 3), 100.0, 101.0, 99.0, 100.0),
        ];
        let (anomalies, seams) = scan("REUSE", &bars, &BTreeSet::new());
        assert_eq!(anomalies.splices, 1);
        assert_eq!(anomalies.holes, 1);
        assert_eq!(seams.len(), 1);
        assert_eq!(seams[0].kind, "splice");
    }

    #[test]
    fn a_clean_series_is_written_untouched() {
        let bars = [
            bar((2008, 9, 12), 12.0, 12.3, 11.8, 12.1),
            bar((2008, 9, 15), 12.1, 12.2, 10.4, 10.6),
            bar((2008, 9, 16), 10.6, 11.0, 9.9, 10.9),
        ];
        let (anomalies, seams) = scan("SPY", &bars, &BTreeSet::new());
        assert_eq!(anomalies.total(), 0, "a real crash day is not an anomaly");
        assert!(seams.is_empty());
    }

    /// A corrupt prefix of `lead` bars at 0.375, then `tail` clean bars at 10.25.
    fn reorganised(lead: usize, tail: usize) -> Vec<PackedBar> {
        let mut bars = Vec::with_capacity(lead + tail);
        let mut day = NaiveDate::from_ymd_opt(1993, 1, 4).unwrap();
        let push = |bars: &mut Vec<PackedBar>, day: &mut NaiveDate, close: f32| {
            bars.push(bar(
                (day.year(), day.month(), day.day()),
                close,
                close * 1.01,
                close * 0.99,
                close,
            ));
            *day = day.succ_opt().unwrap();
        };
        for _ in 0..lead {
            push(&mut bars, &mut day, 0.375);
        }
        for _ in 0..tail {
            push(&mut bars, &mut day, 10.25);
        }
        bars
    }

    #[test]
    fn the_shipped_cut_keeps_the_clean_tail_and_reports_it() {
        // NVR's shape: a corrupt pre-reorganisation prefix, then decades of clean history.
        let mut bars = reorganised(3, MIN_TRUNCATED_BARS);
        let kept_from = session_date(bars[3].ts());
        let (anomalies, cut) =
            truncate_at_last_seam("NVR", &mut bars, &BTreeSet::new()).expect("tail is long enough");

        let cut = cut.expect("a seam was cut");
        assert_eq!(cut.dropped, 3);
        assert_eq!(cut.kept_from, kept_from);
        assert_eq!(cut.seams.len(), 1, "the seam is reported, not swallowed");
        assert_eq!(bars.len(), MIN_TRUNCATED_BARS);
        assert_eq!(
            anomalies.bars,
            MIN_TRUNCATED_BARS,
            "the counts describe what is written, not what was fetched"
        );
        assert_eq!(anomalies.total(), 0, "the kept tail is clean");
        assert_eq!(
            session_date(bars[0].ts()),
            kept_from,
            "the first kept bar is the one the move landed on, not the corrupt one before it"
        );
    }

    #[test]
    fn a_tail_below_the_floor_is_withheld_rather_than_written_short() {
        let mut bars = reorganised(3, MIN_TRUNCATED_BARS - 1);
        let before = bars.len();
        let (seams, reason) =
            truncate_at_last_seam("NVR", &mut bars, &BTreeSet::new()).expect_err("tail too short");

        assert_eq!(seams.len(), 1);
        assert!(reason.contains("clean tail"), "reason was {reason:?}");
        assert_eq!(bars.len(), before, "a withheld symbol's bars are left alone");
    }

    #[test]
    fn a_clean_symbol_is_not_cut_at_all() {
        let mut bars = reorganised(0, 300);
        let (anomalies, cut) =
            truncate_at_last_seam("KO", &mut bars, &BTreeSet::new()).expect("nothing to cut");
        assert!(cut.is_none());
        assert_eq!(anomalies.total(), 0);
        assert_eq!(bars.len(), 300, "an untouched symbol keeps every bar");
    }

    #[test]
    fn a_bad_print_across_a_hole_cuts_at_the_hole_and_survives_the_rescan() {
        // The case the tick-recovery hole guard exists for: a >=14-day halt, one bad print across
        // it, then a return to the old level. Without the guard the recovery leg is excused as a
        // tick's tail, the cut lands on the splice alone, and the still-extreme leg becomes the
        // kept tail's first return — tripping the rescan and costing the whole symbol. With it
        // the leg is a seam in its own right, so the cut moves PAST the bad print and the symbol
        // survives with a clean tail, which is the outcome worth having.
        let mut bars = vec![
            bar((2009, 1, 2), 10.0, 10.1, 9.9, 10.0),
            bar((2009, 2, 2), 100.0, 101.0, 99.0, 100.0),
        ];
        let mut day = NaiveDate::from_ymd_opt(2009, 2, 3).unwrap();
        for _ in 0..MIN_TRUNCATED_BARS {
            bars.push(bar((day.year(), day.month(), day.day()), 10.0, 10.1, 9.9, 10.0));
            day = day.succ_opt().unwrap();
        }
        let (anomalies, cut) =
            truncate_at_last_seam("HALT", &mut bars, &BTreeSet::new()).expect("the tail survives");

        let cut = cut.expect("the splice is cut");
        assert_eq!(
            cut.seams.len(),
            2,
            "the splice and its recovery leg are both seams"
        );
        assert_eq!(cut.dropped, 2, "the cut clears the bad print itself");
        assert_eq!(anomalies.total(), 0, "the kept tail carries no anomaly");
        assert_eq!(bars.len(), MIN_TRUNCATED_BARS);
        assert_eq!(
            session_date(bars[0].ts()),
            NaiveDate::from_ymd_opt(2009, 2, 3).unwrap(),
            "the first kept bar is the first good one after the print"
        );
    }

    #[test]
    fn a_still_forming_session_is_not_written() {
        let date = NaiveDate::from_ymd_opt(2026, 8, 14).unwrap();
        let at = |hour: u32, minute: u32| {
            date.and_hms_opt(hour, minute, 0)
                .unwrap()
                .and_local_timezone(New_York)
                .unwrap()
        };
        assert!(!session_complete(date, at(9, 30)), "the session is still open");
        assert!(!session_complete(date, at(15, 59)), "one minute before the close");
        assert!(session_complete(date, at(16, 0)), "complete at the close");
        assert!(session_complete(
            date,
            at(9, 30) + chrono::Duration::days(1)
        ));
    }

    #[test]
    fn dot_symbols_are_translated_for_the_query_only() {
        assert_eq!(query_symbol("BRK.B"), "BRK-B");
        assert_eq!(query_symbol("AAPL"), "AAPL");
    }

    #[test]
    fn conversion_adjusts_for_dividends_dedupes_and_stamps_et_midnight() {
        // Two stamps on one session date plus a null close that must be dropped entirely.
        let payload = r#"{"chart":{"result":[{
            "timestamp":[1221451200,1221537600,1221537600,1221624000],
            "indicators":{
              "quote":[{"open":[10.0,12.0,12.1,null],"high":[10.5,12.5,12.6,13.0],
                        "low":[9.5,11.5,11.6,12.0],"close":[10.0,12.0,12.2,null],
                        "volume":[100,200,250,300]}],
              "adjclose":[{"adjclose":[5.0,6.0,6.1,null]}]},
            "events":{"splits":{"1221537600":{"date":1221537600,"numerator":2.0,
                       "denominator":1.0,"splitRatio":"2:1"}}}}]}}"#;
        let chart: ChartEnvelope = serde_json::from_str(payload).unwrap();
        let result = chart.chart.result.unwrap().into_iter().next().unwrap();
        let Converted {
            bars,
            splits,
            repaired,
        } = to_bars(&result).unwrap();
        assert_eq!(repaired, 0, "every row in this payload is well formed");

        assert_eq!(bars.len(), 2, "duplicate stamp collapsed, null row dropped");
        assert_eq!(session_date(bars[0].ts()), NaiveDate::from_ymd_opt(2008, 9, 15).unwrap());
        assert_eq!(bars[0].ts(), session_open_ms(session_date(bars[0].ts())));
        assert!(bars[1].ts() > bars[0].ts(), "strictly increasing");
        // adjclose/close = 5/10 halves every price; volume is untouched.
        assert!((bars[0].close - 5.0).abs() < 1e-5);
        assert!((bars[0].open - 5.0).abs() < 1e-5);
        assert!((bars[0].volume - 100.0).abs() < 1e-5);
        assert_eq!(bars[0].trades, 0);
        let (high, low, close) = (bars[0].high, bars[0].low, bars[0].close);
        assert!((bars[0].vwap - (high + low + close) / 3.0).abs() < 1e-5);
        // The later duplicate wins: close 12.2 * (6.1/12.2) = 6.1.
        assert!((bars[1].close - 6.1).abs() < 1e-5);
        assert_eq!(splits.len(), 1);
        assert!(near_split(NaiveDate::from_ymd_opt(2008, 9, 16).unwrap(), &splits));
        assert!(!near_split(NaiveDate::from_ymd_opt(2008, 10, 16).unwrap(), &splits));
    }

    #[test]
    fn a_close_outside_the_source_envelope_widens_it_instead_of_dropping_the_bar() {
        // UA 2021-05-05 as the source reports it: the low sits above the open.
        let payload = r#"{"chart":{"result":[{
            "timestamp":[1620187200],
            "indicators":{"quote":[{"open":[20.870001],"high":[21.825001],
                "low":[21.000000],"close":[21.129999],"volume":[1000]}]}}]}}"#;
        let chart: ChartEnvelope = serde_json::from_str(payload).unwrap();
        let result = chart.chart.result.unwrap().into_iter().next().unwrap();
        let Converted { bars, repaired, .. } = to_bars(&result).unwrap();

        assert_eq!(repaired, 1);
        let only = bars[0];
        let (open, high, low, close) = (only.open, only.high, only.low, only.close);
        assert!((low - open).abs() < 1e-5, "the low is pulled down to the open");
        assert!((high - 21.825001).abs() < 1e-4, "the high was already valid");
        assert!(low <= open && low <= close && high >= open && high >= close);
        assert!(high >= low);
    }
}
