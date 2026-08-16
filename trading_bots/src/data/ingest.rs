//! Corpus ingestion: universe selection from Polygon grouped daily bars, then bulk aggregate
//! download into the packed `.bars` format consumed by the pretraining pipeline.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Datelike, Days, Duration, NaiveDate, Utc, Weekday};
use serde::{Deserialize, Serialize};
use shared::bars::{bar_file_path, write_bar_file, BarFile, PackedBar};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::torch::dataset::BarCorpus;
use super::polygon::{self, Window};

/// Aggregate windows are bounded by the 50k-row page cap (~186 extended-hours 5m bars per day).
const MAX_WINDOW_DAYS: i64 = 269;
/// Sessions sampled when measuring universe liquidity, spread over the whole training window.
///
/// Deliberately not a trailing window, even one ending at the train boundary: a symbol that
/// delisted early in the corpus traded in none of the recent sessions, so a trailing sample would
/// erase it and re-impose exactly the survivorship bias that admitting delisted tickers removes.
const UNIVERSE_SAMPLES: usize = 60;
/// Sampled sessions a symbol must have traded in to be ranked at all. At the density above this is
/// a few months of trading, which is what separates a security from a stub listing.
const UNIVERSE_MIN_SESSIONS: usize = 3;
/// Resolution whose corpus defines the split instants. The pretrainer derives the boundary from
/// its primary resolution and hands it to every auxiliary one, so the universe must read it from
/// the same place or the two definitions of "train" drift apart.
const UNIVERSE_BOUND_RES_SECS: u32 = 300;
/// Sampling stays this far inside the plan's rolling window; a session at the very edge can answer
/// `NOT_AUTHORIZED` between the request and the vendor's own cutover.
const PLAN_EDGE_MARGIN_DAYS: i64 = 10;
/// A corpus file that ends within this many days of the request is current: a long weekend plus a
/// holiday is the widest gap an up-to-date series can legitimately show at its right edge.
const FRESH_TAIL_DAYS: i64 = 5;
/// Candidate liquidity floors reported beside the chosen one, so the cost of moving it is visible.
const FLOOR_LADDER: [f64; 10] = [
    1e5, 2.5e5, 5e5, 1e6, 2e6, 4e6, 8.5e6, 1.5e7, 3e7, 1e8,
];
/// Liquidity floor for corpus membership: median dollars traded per session, measured over the
/// sampled training-window sessions a symbol actually traded in.
///
/// A floor rather than a rank cutoff, and this one, because of what five-minute bars look like as
/// liquidity falls. Measured over one train-window quarter, the share of u/v observations landing on
/// an atom — a close or open pinned to an extreme of the bar, or a flat bar with no intra-bar shape
/// at all — rises smoothly from 37% above $100M/day to 70% in $4-8.5M, then steps onto a plateau:
/// 79% in $2-4M and no worse than 87% anywhere below it. Flat bars, which force three of the five
/// degrees of freedom, follow the same curve: 10% above $30M/day, 20% at $4-8.5M, 31% at $2-4M,
/// 58% below $25k. So there is exactly one knee in the data and nothing below it to cut on.
///
/// The floor sits BELOW that knee deliberately. Degeneracy is not corruption: a flat bar pins `s`,
/// `u` and `v`, but `r` and `w` stay informative, and a failing company's return path is the thing
/// the corpus most needs and a survivor-only universe cannot contain. Delisted share climbs
/// monotonically as liquidity falls — 9% of the top decile, 51% of the bottom — so survivorship and
/// illiquidity are one axis and the bias cannot be removed without descending it: $8.5M/day admits
/// 466 delisted names, $1M/day admits 1,029. Against that, the cost is bounded and small — 5,297
/// eligible symbols and 456M bars, 16 GB, and ~11 minutes per epoch against a 44-60 minute budget.
///
/// Below roughly $500k/day the trade does reverse, which is why the floor is not lower: more than a
/// quarter of those symbols never reach `pretrain --min-bars`, so the download is discarded, and by
/// $50-100k/day only 0.2% of them do.
pub const MIN_DOLLAR_VOLUME: f64 = 1_000_000.0;

/// The `train | val` and `val | test` instants every run over this corpus MUST be pinned to, in
/// epoch millis: 2025-10-07T12:10:00Z and 2026-03-13T18:45:00Z.
///
/// These are the instants [`corpus_train_end`] derived from the 3,000-symbol corpus at the moment
/// `long_data/universe.json` was measured, and therefore the instants the universe is a function
/// of: every one of its 60 sampled sessions (2021-09-22 .. 2025-10-06) lies strictly before the
/// first of them. They are written down rather than re-derived because the boundary is the
/// [`crate::torch::dataset::TRAIN_FRACTION`] percentile of pooled bar timestamps, so it MOVES when
/// the corpus does, and re-deriving it on the expanded corpus yields 2025-09-11T14:10:00Z — 26 days
/// EARLIER, which puts two of those sampled sessions inside val and reopens the selection leak.
///
/// The direction is worth recording, because the intuitive argument gets it backwards. The
/// expansion was expected to push the percentile LATER, on the reasoning that delisted names add
/// bar mass to the early part of the window. They do, but there are only 1,029 of them carrying
/// 49.7M bars, while the 2,728 newly added files are dominated by recent listings and by thin names
/// whose bar density RISES over the window. That is late mass, and late mass moves an upper
/// percentile earlier.
///
/// So do not "fix" this pin by re-deriving it, and do not iterate the universe against the derived
/// boundary: the universe depends on the boundary which depends on the corpus which depends on the
/// universe, and that loop has no reason to converge. Re-measure the ranking only together with an
/// update of these two numbers, and check them against
/// [`universe_train_end`] before trusting a run.
pub const PINNED_SPLIT_BOUNDS: (i64, i64) = (1_759_839_000_000, 1_773_427_500_000);
const PROGRESS_EVERY: usize = 25;
const DAILY_RES_SECS: u32 = 86_400;
/// Deepest history the subscription serves; requests beyond it answer `NOT_AUTHORIZED`.
const PLAN_WINDOW_DAYS: i64 = 5 * 365;
/// Share of symbols allowed to fail before the whole pass is reported as failed.
const FAILURE_TOLERANCE: f64 = 0.01;
/// Days to walk back looking for the newest reference-data entry for a ticker.
const IDENTITY_LOOKBACK_DAYS: i64 = 8;
/// Trading hole at a ticker handover that marks the two sides as different securities.
const SPLICE_MIN_HOLE_DAYS: i64 = 5;
/// Level jump that no corporate action explains. Deliberately as coarse as the corpus anomaly
/// threshold: a 2x five-minute move is a takeover premium or a halt release, not a new security,
/// and cutting on one would amputate most of a symbol's history for an ordinary market event.
const SPLICE_MAX_JUMP: f64 = 4.0;
/// Breaks examined against reference data per symbol, worst first.
const MAX_SPLICE_PROBES: usize = 16;
/// Reference probes per side of a break, stepping outward over unassigned days.
const IDENTITY_PROBE_STEPS: i64 = 4;
/// Days between successive reference probes on one side of a break.
const IDENTITY_PROBE_SPAN_DAYS: i64 = 5;

/// A bar resolution expressed both in seconds and as a Polygon aggregate span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolution {
    pub multiplier: u32,
    pub timespan: &'static str,
    pub res_secs: u32,
}

impl Resolution {
    /// Parses `5min`, `15m`, `1hour`, `day`, `30sec` and friends.
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim().to_ascii_lowercase();
        let split = spec
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(spec.len());
        let (count, unit) = spec.split_at(split);
        let count: u32 = if count.is_empty() {
            1
        } else {
            count.parse().with_context(|| format!("bad resolution {spec}"))?
        };
        let unit_secs = match unit {
            "" | "m" | "min" | "mins" | "minute" | "minutes" => 60,
            "s" | "sec" | "secs" | "second" | "seconds" => 1,
            "h" | "hr" | "hour" | "hours" => 3_600,
            "d" | "day" | "days" => 86_400,
            other => bail!("unsupported resolution unit `{other}` in `{spec}`"),
        };
        if count == 0 {
            bail!("resolution `{spec}` must be positive");
        }
        Self::from_secs(count * unit_secs)
    }

    /// Canonical Polygon span for a resolution given in seconds.
    pub fn from_secs(res_secs: u32) -> Result<Self> {
        let (multiplier, timespan) = if res_secs == 0 {
            bail!("resolution seconds must be positive");
        } else if res_secs % 86_400 == 0 {
            (res_secs / 86_400, "day")
        } else if res_secs % 3_600 == 0 {
            (res_secs / 3_600, "hour")
        } else if res_secs % 60 == 0 {
            (res_secs / 60, "minute")
        } else {
            (res_secs, "second")
        };
        Ok(Self {
            multiplier,
            timespan,
            res_secs,
        })
    }
}

/// One measured universe member.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniverseEntry {
    pub symbol: String,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    /// Median dollar volume over the sampled sessions this symbol actually traded in, i.e. over
    /// its own life rather than over the whole window. A symbol that delisted in 2022 has to rank
    /// on how it traded while it existed, or absence alone would disqualify it.
    pub median_dollar_volume: f64,
    /// Sampled sessions the symbol traded in.
    pub sessions: usize,
    pub first_session: NaiveDate,
    pub last_session: NaiveDate,
    /// Absent from Polygon's listed reference set: the security no longer trades. Kept
    /// deliberately — a corpus of survivors cannot teach a model what a failure looks like.
    pub delisted: bool,
}

/// The measured ranking, cached whole.
///
/// The liquidity floor is applied when this is READ, never when it is written, so moving the floor
/// is a filter over measured data rather than a re-measurement: it costs no vendor requests and
/// cannot silently change what "median dollar volume" meant.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct UniverseFile {
    generated_utc: String,
    /// The `train | val` instant this ranking is a function of. No session at or after it was
    /// sampled, so which symbols exist in the corpus carries no held-out information.
    train_end_utc: String,
    /// Sessions that returned data, out of [`UNIVERSE_SAMPLES`] requested.
    sampled_sessions: usize,
    first_sampled_session: NaiveDate,
    last_sampled_session: NaiveDate,
    listed_refs: usize,
    delisted_refs: usize,
    /// Ticker strings that named more than one security in the reference data: the ticker-reuse
    /// population [`truncate_at_splice`] exists to cut.
    reused_tickers: usize,
    entries: Vec<UniverseEntry>,
}

/// Command-line surface of the `ingest` subcommand.
#[derive(Clone, Debug)]
pub struct IngestArgs {
    /// Liquidity floor for corpus membership: minimum median dollar volume per session.
    pub min_dollar_volume: f64,
    pub resolution: String,
    pub years: u32,
    pub concurrency: usize,
    /// Re-measure liquidity from the vendor instead of reusing the cached ranking.
    pub refresh_universe: bool,
    /// The `train | val` instant the ranking must precede. Derived from the corpus when absent.
    pub train_end: Option<DateTime<Utc>>,
    /// Bars a symbol needs to enter the corpus that boundary is derived from; matches the
    /// pretrainer's own `--min-bars`, since dropping a file moves the trading-time percentile.
    pub min_bars: usize,
    /// Measure and report the universe, then stop before downloading anything.
    pub universe_only: bool,
    pub force: bool,
    pub daily: bool,
}

/// One symbol to ingest.
#[derive(Clone, Debug)]
pub struct IngestTarget {
    pub symbol: String,
    /// The security no longer trades, so its series is complete at whatever its last bar is and a
    /// stale right edge is no reason to refetch it. Without this every delisted symbol would be
    /// redownloaded on every pass, forever.
    pub settled: bool,
}

impl From<&UniverseEntry> for IngestTarget {
    fn from(entry: &UniverseEntry) -> Self {
        Self {
            symbol: entry.symbol.clone(),
            settled: entry.delisted,
        }
    }
}

/// Tally of one ingestion pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct IngestSummary {
    pub written: usize,
    pub skipped: usize,
    pub empty: usize,
    pub failed: usize,
    pub bars: usize,
    pub first_ts_ms: Option<i64>,
    pub last_ts_ms: Option<i64>,
}

/// Runs universe selection followed by the requested ingestion.
pub async fn run(args: IngestArgs) -> Result<()> {
    polygon::configure(args.concurrency);

    let resolution = if args.daily {
        Resolution::from_secs(DAILY_RES_SECS)?
    } else {
        Resolution::parse(&args.resolution)?
    };

    let train_end = match args.train_end {
        Some(instant) => instant,
        None => corpus_train_end(args.min_bars)?,
    };
    let entries = universe_ranking(train_end, args.refresh_universe).await?;
    report_liquidity(&entries, args.min_dollar_volume);
    report_provenance()?;
    let targets: Vec<IngestTarget> = entries
        .iter()
        .filter(|entry| entry.median_dollar_volume >= args.min_dollar_volume)
        .map(IngestTarget::from)
        .collect();
    if targets.is_empty() {
        bail!(
            "no symbol clears the ${:.0}/day liquidity floor",
            args.min_dollar_volume
        );
    }
    if args.universe_only {
        return Ok(());
    }
    println!(
        "[ingest] universe of {} symbols ({} delisted), resolution {}{} ({}s), {} year(s) back",
        targets.len(),
        targets.iter().filter(|target| target.settled).count(),
        resolution.multiplier,
        resolution.timespan,
        resolution.res_secs,
        args.years
    );

    let out_dir = bars_dir();
    let summary = if args.daily {
        ingest_daily(
            &targets,
            args.years,
            &out_dir,
            args.concurrency,
            args.force,
        )
        .await?
    } else {
        ingest_intraday(
            &targets,
            resolution.res_secs,
            args.years,
            &out_dir,
            args.concurrency,
            args.force,
        )
        .await?
    };

    println!(
        "[ingest] done: {} written, {} skipped, {} empty, {} failed, {} bars, span {} .. {}, dir {}",
        summary.written,
        summary.skipped,
        summary.empty,
        summary.failed,
        summary.bars,
        summary.first_ts_ms.map(format_ts).unwrap_or_else(|| "-".into()),
        summary.last_ts_ms.map(format_ts).unwrap_or_else(|| "-".into()),
        out_dir.display()
    );
    let attempted = summary.written + summary.skipped + summary.empty + summary.failed;
    if summary.failed as f64 > attempted as f64 * FAILURE_TOLERANCE {
        bail!(
            "ingest incomplete: {}/{attempted} symbol(s) failed",
            summary.failed
        );
    }
    if summary.written == 0 && summary.skipped == 0 {
        bail!("ingest produced no usable bar files");
    }
    Ok(())
}

/// The `train | val` instant the universe must precede, read from the corpus the pretrainer
/// splits, so a universe built today and a split derived at training time cannot disagree.
///
/// The boundary is the [`crate::torch::dataset::TRAIN_FRACTION`] percentile of pooled bar
/// timestamps, so it moves when the corpus does. That is a one-way dependency and must stay one:
/// admitting delisted and thinly traded names adds bar mass to the EARLY part of the window, which
/// pushes the percentile later and leaves every session sampled here inside train. Iterating this
/// to a fixed point would make the universe a function of itself; if the boundary ever moves
/// earlier than a sampled session, pin it explicitly instead.
pub fn corpus_train_end(min_bars: usize) -> Result<DateTime<Utc>> {
    let dir = bars_dir();
    let corpus = BarCorpus::load(&dir, UNIVERSE_BOUND_RES_SECS, min_bars).with_context(|| {
        format!(
            "deriving the train|val boundary from {} at {UNIVERSE_BOUND_RES_SECS}s",
            dir.display()
        )
    })?;
    let (train_val, _) = corpus.split_bounds();
    DateTime::<Utc>::from_timestamp_millis(train_val)
        .with_context(|| format!("train|val boundary {train_val} is not a valid instant"))
}

/// Parses the `--train-end` instant. It lives beside the ranking it constrains so the CLI and the
/// universe cannot disagree about what "before the boundary" means.
pub fn parse_train_end(raw: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(raw)
        .map(|instant| instant.with_timezone(&Utc))
        .map_err(|error| {
            format!("expected an RFC 3339 instant such as 2025-10-07T12:10:00Z: {error}")
        })
}

/// The full measured ranking: the cached one unless a refresh is asked for or none exists.
pub async fn universe_ranking(
    train_end: DateTime<Utc>,
    refresh: bool,
) -> Result<Vec<UniverseEntry>> {
    if !refresh {
        if let Some(file) = load_universe()? {
            println!(
                "[universe] reusing {} ({} symbols, {} sessions before {})",
                universe_path().display(),
                file.entries.len(),
                file.sampled_sessions,
                file.train_end_utc
            );
            return Ok(file.entries);
        }
    }
    rebuild_universe(train_end).await
}

/// Every symbol clearing `min_dollar_volume`, for callers that need only the names.
pub async fn build_universe(
    min_dollar_volume: f64,
    train_end: DateTime<Utc>,
) -> Result<Vec<String>> {
    Ok(universe_ranking(train_end, false)
        .await?
        .into_iter()
        .filter(|entry| entry.median_dollar_volume >= min_dollar_volume)
        .map(|entry| entry.symbol)
        .collect())
}

/// The sessions to measure liquidity on: spread over the vendor's whole window, every one of them
/// strictly before the `train | val` boundary.
fn ranking_sessions(train_end: DateTime<Utc>, today: NaiveDate) -> Result<Vec<NaiveDate>> {
    let last_session = previous_weekday(train_end.date_naive() - Duration::days(1));
    let plan_floor = today - Duration::days(PLAN_WINDOW_DAYS - PLAN_EDGE_MARGIN_DAYS);
    let span_days = (last_session - plan_floor).num_days();
    if span_days < UNIVERSE_SAMPLES as i64 {
        bail!(
            "the training window {plan_floor} .. {last_session} is too short to sample \
             {UNIVERSE_SAMPLES} sessions"
        );
    }
    let dates = sample_trading_days(UNIVERSE_SAMPLES, span_days, last_session);
    if dates.is_empty() {
        bail!("no session to sample before {last_session}");
    }
    Ok(dates)
}

/// Ranks measured dollar volumes, most liquid first.
///
/// A symbol is ranked on the MEDIAN over the sampled sessions it actually traded in, and needs
/// `min_sessions` of them. Both halves matter: a symbol that delisted mid-window traded in only the
/// early samples, so a gate that counted absence, or a median that counted absent sessions as zero,
/// would rank it below a stub listing and delete the failure outcomes from the corpus.
fn rank_symbols(
    dollar_volumes: HashMap<String, Vec<(NaiveDate, f64)>>,
    reference: &ReferenceIndex,
    min_sessions: usize,
) -> Vec<UniverseEntry> {
    let mut ranked: Vec<UniverseEntry> = dollar_volumes
        .into_iter()
        .filter_map(|(symbol, mut traded)| {
            if traded.len() < min_sessions {
                return None;
            }
            let found = reference.by_symbol.get(&symbol)?;
            let first_session = traded.iter().map(|(date, _)| *date).min()?;
            let last_session = traded.iter().map(|(date, _)| *date).max()?;
            traded.sort_unstable_by(|left, right| left.1.total_cmp(&right.1));
            Some(UniverseEntry {
                symbol,
                name: found.entry.name.clone(),
                kind: found.entry.kind.clone(),
                median_dollar_volume: traded[traded.len() / 2].1,
                sessions: traded.len(),
                first_session,
                last_session,
                delisted: !found.listed,
            })
        })
        .collect();
    ranked.sort_unstable_by(|a, b| {
        b.median_dollar_volume
            .total_cmp(&a.median_dollar_volume)
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    ranked
}

/// Re-measures liquidity from Polygon over training-window sessions only, and overwrites the cache.
///
/// Two properties make this a corpus definition rather than a leak plus a bias:
///
/// * Every sampled session lies strictly before `train_end`, so which symbols exist is a function
///   of training-period information alone. A calendar split partitions time WITHIN a symbol and
///   says nothing about WHICH symbols were admitted, so selection leak cannot be repaired
///   downstream: a symbol admitted because of how it traded during val or test contaminates every
///   held-out number computed from it.
/// * The reference set is listed AND delisted tickers, and each symbol is ranked over the sessions
///   it traded in. A universe of names that survive to today omits every bankruptcy and every
///   acquisition in the window — precisely the outcomes the model most needs to have seen, and the
///   ones it will otherwise be most confident about.
pub async fn rebuild_universe(train_end: DateTime<Utc>) -> Result<Vec<UniverseEntry>> {
    let dates = ranking_sessions(train_end, Utc::now().date_naive())?;
    println!(
        "[universe] sampling {} sessions in {} .. {} (train|val {})",
        dates.len(),
        dates[0],
        dates[dates.len() - 1],
        train_end.to_rfc3339()
    );

    let (reference, samples) = tokio::try_join!(reference_index(), grouped_samples(&dates))?;
    let (sampled_days, dollar_volumes) = samples;
    if sampled_days * 3 < dates.len() * 2 {
        bail!(
            "only {sampled_days}/{} sampled sessions returned data; ranking would be unreliable",
            dates.len()
        );
    }

    let ranked = rank_symbols(
        dollar_volumes,
        &reference,
        UNIVERSE_MIN_SESSIONS.min(sampled_days),
    );
    if ranked.is_empty() {
        bail!("universe ranking is empty after filtering");
    }

    let path = universe_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = UniverseFile {
        generated_utc: Utc::now().to_rfc3339(),
        train_end_utc: train_end.to_rfc3339(),
        sampled_sessions: sampled_days,
        first_sampled_session: dates[0],
        last_sampled_session: dates[dates.len() - 1],
        listed_refs: reference.listed,
        delisted_refs: reference.delisted,
        reused_tickers: reference.reused,
        entries: ranked,
    };
    fs::write(&path, serde_json::to_string_pretty(&file)?)
        .with_context(|| format!("writing {}", path.display()))?;
    println!(
        "[universe] wrote {} ({} symbols over {sampled_days} sessions, leader {} @ ${:.0}/day)",
        path.display(),
        file.entries.len(),
        file.entries[0].symbol,
        file.entries[0].median_dollar_volume
    );

    Ok(file.entries)
}

/// Reports the measured liquidity distribution and what a ladder of floors would admit, so the
/// floor is read off the distribution it cuts instead of guessed.
fn report_liquidity(entries: &[UniverseEntry], floor: f64) {
    if entries.is_empty() {
        return;
    }
    let total = entries.len();
    let delisted = entries.iter().filter(|entry| entry.delisted).count();
    println!(
        "[universe] {total} symbols measured, {delisted} delisted ({:.1}%)",
        100.0 * delisted as f64 / total as f64
    );
    println!("[universe] decile | symbols | median $volume range | delisted | median sessions");
    for decile in 0..10usize {
        let lo = decile * total / 10;
        let hi = ((decile + 1) * total / 10).min(total);
        if hi <= lo {
            continue;
        }
        let slice = &entries[lo..hi];
        let mut sessions: Vec<usize> = slice.iter().map(|entry| entry.sessions).collect();
        sessions.sort_unstable();
        println!(
            "[universe]   {:>2} | {:>7} | {:>14.0} .. {:<14.0} | {:>8} | {:>3}",
            decile + 1,
            slice.len(),
            slice[slice.len() - 1].median_dollar_volume,
            slice[0].median_dollar_volume,
            slice.iter().filter(|entry| entry.delisted).count(),
            sessions[sessions.len() / 2]
        );
    }
    println!("[universe] liquidity floor | symbols | delisted");
    for candidate in FLOOR_LADDER {
        let kept: Vec<&UniverseEntry> = entries
            .iter()
            .filter(|entry| entry.median_dollar_volume >= candidate)
            .collect();
        println!(
            "[universe]   >= ${candidate:>12.0}/day | {:>7} | {:>8}{}",
            kept.len(),
            kept.iter().filter(|entry| entry.delisted).count(),
            if candidate == floor { "  <- selected" } else { "" }
        );
    }
    let kept = entries
        .iter()
        .filter(|entry| entry.median_dollar_volume >= floor)
        .count();
    println!("[universe] floor ${floor:.0}/day admits {kept}/{total} symbols");
}

/// Prints the record that ties this corpus to a split: the instant the ranking was measured
/// against, the digest of the ranking itself, and whether [`PINNED_SPLIT_BOUNDS`] still matches.
///
/// A mismatch is not an error here — re-measuring the ranking is a legitimate thing to do — but it
/// means the pinned constant and every run using it are stale, so it is reported loudly rather than
/// left for a checkpoint to inherit silently.
fn report_provenance() -> Result<()> {
    let Some(train_end) = universe_train_end()? else {
        return Ok(());
    };
    let fingerprint = universe_fingerprint()?.unwrap_or_default();
    let pinned = DateTime::<Utc>::from_timestamp_millis(PINNED_SPLIT_BOUNDS.0);
    println!(
        "[universe] provenance: measured against train|val {}, sha256 {fingerprint}",
        train_end.to_rfc3339()
    );
    match pinned {
        Some(pinned) if pinned == train_end => println!(
            "[universe] pinned split bounds {} | {} match the ranking; pin them with \
             `pretrain --split-bounds {},{}`",
            pinned.to_rfc3339(),
            DateTime::<Utc>::from_timestamp_millis(PINNED_SPLIT_BOUNDS.1)
                .map(|instant| instant.to_rfc3339())
                .unwrap_or_default(),
            PINNED_SPLIT_BOUNDS.0,
            PINNED_SPLIT_BOUNDS.1
        ),
        _ => eprintln!(
            "[universe] WARNING: PINNED_SPLIT_BOUNDS.0 = {} does not match the ranking's {}; \
             update the constant together with the ranking or every pinned run is scored against \
             a boundary the universe was not selected under",
            pinned
                .map(|instant| instant.to_rfc3339())
                .unwrap_or_else(|| PINNED_SPLIT_BOUNDS.0.to_string()),
            train_end.to_rfc3339()
        ),
    }
    Ok(())
}

/// Downloads intraday aggregates for every target, paging backwards until the plan window ends.
pub async fn ingest_intraday(
    targets: &[IngestTarget],
    res_secs: u32,
    years: u32,
    out_dir: &Path,
    concurrency: usize,
    force: bool,
) -> Result<IngestSummary> {
    let resolution = Resolution::from_secs(res_secs)?;
    let end = Utc::now().date_naive();
    let floor = end - Duration::days(365 * years.max(1) as i64);
    fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let permits = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut tasks = JoinSet::new();
    for target in targets {
        let symbol = target.symbol.clone();
        let settled = target.settled;
        let out_dir = out_dir.to_path_buf();
        let permits = Arc::clone(&permits);
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await;
            let path = bar_file_path(&out_dir, &symbol, res_secs);
            if !force && is_current(path.clone(), res_secs, end, settled).await {
                return SymbolOutcome::Skipped;
            }
            let mut bars = match fetch_history(&symbol, resolution, floor, end).await {
                Ok(bars) => bars,
                Err(error) => return SymbolOutcome::Failed(symbol, format!("{error:#}")),
            };
            if bars.is_empty() {
                return SymbolOutcome::Empty(symbol);
            }
            // Cheap gate: only a series whose ticker changed hands can contain a splice.
            match identity_handover(&symbol, &bars).await {
                Ok(Some(_)) => {
                    if let Err(error) = truncate_at_splice(&symbol, &mut bars).await {
                        return SymbolOutcome::Failed(symbol, format!("{error:#}"));
                    }
                }
                Ok(None) => {}
                Err(error) => return SymbolOutcome::Failed(symbol, format!("{error:#}")),
            }
            if bars.is_empty() {
                SymbolOutcome::Empty(symbol)
            } else {
                persist(path, symbol, res_secs, bars).await
            }
        });
    }

    drain(&mut tasks, targets.len(), "symbols").await
}

/// Downloads daily bars for every symbol via grouped-daily fan-out over trading sessions.
pub async fn ingest_daily(
    targets: &[IngestTarget],
    years: u32,
    out_dir: &Path,
    concurrency: usize,
    force: bool,
) -> Result<IngestSummary> {
    let end = Utc::now().date_naive();
    let floor = end - Duration::days(365 * years.max(1) as i64);
    fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let wanted: Arc<HashSet<String>> =
        Arc::new(targets.iter().map(|target| target.symbol.clone()).collect());
    let sessions = weekdays(floor, end);
    let permits = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut fetches = JoinSet::new();
    for date in sessions.iter().copied() {
        let wanted = Arc::clone(&wanted);
        let permits = Arc::clone(&permits);
        fetches.spawn(async move {
            let _permit = permits.acquire_owned().await;
            polygon::grouped_daily(date).await.map(|rows| {
                rows.into_iter()
                    .filter(|(symbol, _)| wanted.contains(symbol))
                    .collect::<Vec<_>>()
            })
        });
    }

    let started = Instant::now();
    let mut per_symbol: HashMap<String, Vec<PackedBar>> = HashMap::new();
    let mut fetched = 0usize;
    let mut failed_sessions = 0usize;
    while let Some(joined) = fetches.join_next().await {
        fetched += 1;
        match joined.context("grouped daily task panicked")? {
            Ok(rows) => {
                for (symbol, bar) in rows {
                    per_symbol.entry(symbol).or_default().push(bar);
                }
            }
            Err(error) => {
                failed_sessions += 1;
                eprintln!("[ingest] grouped daily session failed: {error:#}");
            }
        }
        if fetched % PROGRESS_EVERY == 0 || fetched == sessions.len() {
            println!(
                "[ingest] daily sessions {fetched}/{} | symbols seen {} | {:.1}s",
                sessions.len(),
                per_symbol.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    if failed_sessions > 0 {
        bail!(
            "{failed_sessions}/{} grouped daily session(s) failed; refusing to write a corpus with holes",
            sessions.len()
        );
    }

    let mut tasks = JoinSet::new();
    let mut per_symbol: BTreeMap<String, Vec<PackedBar>> = per_symbol.into_iter().collect();
    for target in targets {
        let symbol = target.symbol.clone();
        let settled = target.settled;
        let mut bars = per_symbol.remove(&symbol).unwrap_or_default();
        let out_dir = out_dir.to_path_buf();
        let permits = Arc::clone(&permits);
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await;
            let path = bar_file_path(&out_dir, &symbol, DAILY_RES_SECS);
            if !force && is_current(path.clone(), DAILY_RES_SECS, end, settled).await {
                return SymbolOutcome::Skipped;
            }
            bars.sort_unstable_by_key(|bar| bar.ts_ms);
            bars.dedup_by_key(|bar| bar.ts_ms);
            drop_incomplete(&mut bars, DAILY_RES_SECS);
            if bars.is_empty() {
                return SymbolOutcome::Empty(symbol);
            }
            match identity_handover(&symbol, &bars).await {
                Ok(Some(_)) => {
                    if let Err(error) = truncate_at_splice(&symbol, &mut bars).await {
                        return SymbolOutcome::Failed(symbol, format!("{error:#}"));
                    }
                }
                Ok(None) => {}
                Err(error) => return SymbolOutcome::Failed(symbol, format!("{error:#}")),
            }
            if bars.is_empty() {
                return SymbolOutcome::Empty(symbol);
            }
            persist(path, symbol, DAILY_RES_SECS, bars).await
        });
    }

    drain(&mut tasks, targets.len(), "symbols").await
}

enum SymbolOutcome {
    Written {
        bars: usize,
        first_ts_ms: i64,
        last_ts_ms: i64,
    },
    Skipped,
    Empty(String),
    Failed(String, String),
}

async fn drain(
    tasks: &mut JoinSet<SymbolOutcome>,
    total: usize,
    unit: &str,
) -> Result<IngestSummary> {
    let started = Instant::now();
    let mut summary = IngestSummary::default();
    let mut completed = 0usize;
    while let Some(joined) = tasks.join_next().await {
        completed += 1;
        match joined.context("ingest task panicked")? {
            SymbolOutcome::Written {
                bars,
                first_ts_ms,
                last_ts_ms,
            } => {
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
            }
            SymbolOutcome::Skipped => summary.skipped += 1,
            SymbolOutcome::Empty(symbol) => {
                summary.empty += 1;
                eprintln!("[ingest] {symbol}: no bars in range");
            }
            SymbolOutcome::Failed(symbol, error) => {
                summary.failed += 1;
                eprintln!("[ingest] {symbol}: {error}");
            }
        }
        if completed % PROGRESS_EVERY == 0 || completed == total {
            println!(
                "[ingest] {completed}/{total} {unit} | {} bars | {:.1}s",
                summary.bars,
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(summary)
}

/// Writes the corpus file off the async runtime; the buffer and path are already owned here.
async fn persist(
    path: PathBuf,
    symbol: String,
    res_secs: u32,
    bars: Vec<PackedBar>,
) -> SymbolOutcome {
    let name = symbol.clone();
    let written = tokio::task::spawn_blocking(move || {
        let first_ts_ms = bars[0].ts_ms;
        let last_ts_ms = bars[bars.len() - 1].ts_ms;
        write_bar_file(&path, &symbol, res_secs, &bars)
            .map(|()| (bars.len(), first_ts_ms, last_ts_ms))
    })
    .await;
    match written {
        Ok(Ok((bars, first_ts_ms, last_ts_ms))) => SymbolOutcome::Written {
            bars,
            first_ts_ms,
            last_ts_ms,
        },
        Ok(Err(error)) => SymbolOutcome::Failed(name, format!("{error:#}")),
        Err(error) => SymbolOutcome::Failed(name, format!("write task panicked: {error}")),
    }
}

/// The day the ticker string changed hands, when the security it names at the END of `bars` is not
/// the one it named at their START. Polygon keys aggregates by the ticker STRING, so a reused
/// ticker splices unrelated companies into one series (ticker META was the Roundhill Ball Metaverse
/// ETF until Meta Platforms took it on 2022-06-09). `composite_figi` survives a rename and changes
/// on reuse, but it also changes on a holdco reorganisation with a continuous price series (Exxon
/// Mobil Corporation -> ExxonMobil Holdings Corporation), so this date is evidence to corroborate
/// against the bars, never a cut on its own.
///
/// Both probes are anchored on the bars themselves rather than on today. A delisted ticker's
/// reference data stops at its last trade, so anchoring the "current" side on `now` would resolve
/// to nothing and skip the gate entirely for the cohort whose strings are most likely to have been
/// recycled — and a string is usually recycled precisely BECAUSE its previous owner delisted.
pub async fn identity_handover(
    symbol: &str,
    bars: &[PackedBar],
) -> Result<Option<polygon::TickerIdentity>> {
    let (Some(first), Some(last)) = (bars.first(), bars.last()) else {
        return Ok(None);
    };
    let Some((current, _)) = current_identity(symbol, utc_date(last.ts())).await? else {
        return Ok(None);
    };
    if current.key().is_empty() {
        return Ok(None);
    }
    let at_floor = polygon::ticker_identity(symbol, utc_date(first.ts())).await?;
    if at_floor.is_some_and(|found| found.key() == current.key()) {
        return Ok(None);
    }
    Ok(Some(current))
}

/// Indices whose bar opens a discontinuity — a multi-session hole, or a level jump no corporate
/// action explains — in ascending order.
fn discontinuities(bars: &[PackedBar]) -> Vec<usize> {
    let min_hole = Duration::days(SPLICE_MIN_HOLE_DAYS).num_milliseconds();
    (1..bars.len())
        .filter(|&index| {
            let before = bars[index - 1];
            let after = bars[index];
            let ratio = after.close as f64 / before.close.max(f32::MIN_POSITIVE) as f64;
            after.ts_ms - before.ts_ms >= min_hole
                || !(1.0 / SPLICE_MAX_JUMP..=SPLICE_MAX_JUMP).contains(&ratio)
        })
        .collect()
}

/// How badly a break interrupts the series, used to spend the reference-data budget on the breaks
/// most likely to be handovers.
fn severity(bars: &[PackedBar], index: usize) -> f64 {
    let before = bars[index - 1];
    let after = bars[index];
    let hole_days = (after.ts_ms - before.ts_ms) as f64 / 86_400_000.0;
    let ratio = after.close as f64 / before.close.max(f32::MIN_POSITIVE) as f64;
    hole_days + ratio.max(f64::MIN_POSITIVE).ln().abs()
}

/// Drops everything before the most recent discontinuity the vendor confirms separates two
/// different securities.
///
/// Only breaks are candidates, so an ordinary bar is never cut, and the two SIDES of a break are
/// compared against each other rather than against today's identity: a security can change its own
/// figi later, so the newest segment need not carry the current one. Ticker COHR is the case that
/// forces this — on the handover day it resolved to II-VI Incorporated's figi, and only later became
/// Coherent Corp's. A side that cannot be resolved abstains instead of voting, because Polygon's
/// reference data has multi-day holes around exactly these transitions.
async fn truncate_at_splice(symbol: &str, bars: &mut Vec<PackedBar>) -> Result<bool> {
    let mut breaks = discontinuities(bars);
    // Newest first, so the cut lands at the latest change of security; severity only decides which
    // breaks are worth spending reference lookups on when an illiquid ticker has dozens.
    breaks.sort_unstable_by(|&left, &right| {
        severity(bars, right)
            .total_cmp(&severity(bars, left))
            .then(right.cmp(&left))
    });
    breaks.truncate(MAX_SPLICE_PROBES);
    breaks.sort_unstable_by(|left, right| right.cmp(left));
    for &split in &breaks {
        let Some(after) = identity_key_near(symbol, utc_date(bars[split].ts_ms), 1).await? else {
            continue;
        };
        let Some(before) = identity_key_near(symbol, utc_date(bars[split - 1].ts_ms), -1).await?
        else {
            continue;
        };
        if before == after {
            continue;
        }
        let previous = bars[split - 1];
        let current = bars[split];
        println!(
            "[ingest] {symbol}: ticker reused at {}, dropping {split} bars before it (hole {:.1}d, level x{:.3})",
            utc_date(current.ts_ms),
            (current.ts_ms - previous.ts_ms) as f64 / 86_400_000.0,
            current.close as f64 / previous.close.max(f32::MIN_POSITIVE) as f64
        );
        bars.drain(..split);
        return Ok(true);
    }
    Ok(false)
}

/// Identity key in force at `date`, stepping outward in `step`-signed increments over the
/// reference-data holes that surround a handover. `None` when nothing resolves nearby.
async fn identity_key_near(symbol: &str, date: NaiveDate, step: i64) -> Result<Option<String>> {
    for probe in 0..IDENTITY_PROBE_STEPS {
        let at = date + Duration::days(step * probe * IDENTITY_PROBE_SPAN_DAYS);
        if let Some(found) = polygon::ticker_identity(symbol, at).await? {
            return Ok(Some(found.key().to_string()));
        }
    }
    Ok(None)
}

/// UTC calendar date of a bar timestamp, for date-keyed reference lookups.
fn utc_date(ts_ms: i64) -> NaiveDate {
    DateTime::<Utc>::from_timestamp_millis(ts_ms)
        .map(|stamp| stamp.date_naive())
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is a valid date"))
}


/// Identity in force at `end`, walking back a few days in case the newest reference day is missing.
async fn current_identity(
    symbol: &str,
    end: NaiveDate,
) -> Result<Option<(polygon::TickerIdentity, NaiveDate)>> {
    for back in 0..IDENTITY_LOOKBACK_DAYS {
        let date = end - Duration::days(back);
        if let Some(found) = polygon::ticker_identity(symbol, date).await? {
            return Ok(Some((found, date)));
        }
    }
    Ok(None)
}

/// Pages backwards in `MAX_WINDOW_DAYS` windows until the plan window or `floor` is reached.
async fn fetch_history(
    symbol: &str,
    resolution: Resolution,
    floor: NaiveDate,
    end: NaiveDate,
) -> Result<Vec<PackedBar>> {
    let mut bars = Vec::new();
    let mut to = end;
    let mut newest = true;
    while to >= floor {
        let from = floor.max(to - Duration::days(MAX_WINDOW_DAYS));
        match polygon::aggregates_window(
            symbol,
            resolution.multiplier,
            resolution.timespan,
            from,
            to,
        )
        .await?
        {
            // The newest window always lies inside the plan window, so a refusal there means the
            // subscription no longer covers this data rather than that history ran out.
            Window::Unauthorized if newest => bail!(
                "polygon refused the most recent window {from}..{to}; check plan entitlement"
            ),
            Window::Unauthorized => break,
            Window::Data(page) => bars.extend(page),
        }
        newest = false;
        if from <= floor {
            break;
        }
        to = from - Duration::days(1);
    }
    bars.sort_unstable_by_key(|bar| bar.ts_ms);
    bars.dedup_by_key(|bar| bar.ts_ms);
    drop_incomplete(&mut bars, resolution.res_secs);
    Ok(bars)
}

/// Removes trailing bars whose interval has not elapsed yet, so no partial bar is persisted.
fn drop_incomplete(bars: &mut Vec<PackedBar>, res_secs: u32) {
    let now_ms = Utc::now().timestamp_millis();
    let span_ms = res_secs as i64 * 1_000;
    while let Some(last) = bars.last() {
        if last.ts() + span_ms <= now_ms {
            break;
        }
        bars.pop();
    }
}

/// [`current_file`] evaluated off the async runtime, since it mmaps and reads the file header.
async fn is_current(path: PathBuf, res_secs: u32, end: NaiveDate, settled: bool) -> bool {
    tokio::task::spawn_blocking(move || current_file(&path, res_secs, end, settled))
        .await
        .unwrap_or(false)
}

/// True when an existing corpus file is already everything this pass could produce.
///
/// Tests the RIGHT edge only, and not even that for a security that has stopped trading. A file's
/// left edge is set by three things the vendor decides and this process cannot: the plan's rolling
/// window, the symbol's own listing date, and the splice repair that truncates a reused ticker at
/// its handover. A left-edge test therefore reports every legitimately-late series as incomplete
/// and rewrites it — which would refetch thousands of correct files and, far worse, restore the
/// very splices [`truncate_at_splice`] cut out of META, BBBY and their kind. Rebuilding a history
/// from scratch is what `force` is for, and it is never implicit.
fn current_file(path: &Path, res_secs: u32, end: NaiveDate, settled: bool) -> bool {
    let Ok(file) = BarFile::open(path) else {
        return false;
    };
    if file.res_secs() != res_secs {
        return false;
    }
    let Some(last_ts_ms) = file.last_ts_ms() else {
        return false;
    };
    settled
        || last_ts_ms >= day_start_ms(end) - Duration::days(FRESH_TAIL_DAYS).num_milliseconds()
}

/// Reference metadata for one ticker string, and whether that string still names a live security.
struct SymbolRef {
    entry: polygon::TickerRef,
    listed: bool,
}

/// Reference data for every ticker string of a kept type, listed or not.
struct ReferenceIndex {
    by_symbol: HashMap<String, SymbolRef>,
    listed: usize,
    delisted: usize,
    reused: usize,
}

/// Listed and delisted reference data, unioned by ticker string.
///
/// On a collision the LISTED entry wins the name and type, because that is what the string denotes
/// today and what its newest bars belong to. The collisions are counted rather than dropped: that
/// count IS the ticker-reuse population, and it is the population [`truncate_at_splice`] has to cut
/// (ticker META named the Roundhill Ball Metaverse ETF before Meta Platforms took it).
async fn reference_index() -> Result<ReferenceIndex> {
    let (listed, delisted) = tokio::try_join!(polygon::tickers(true), polygon::tickers(false))?;
    let (listed_count, delisted_count) = (listed.len(), delisted.len());
    let mut by_symbol: HashMap<String, SymbolRef> =
        HashMap::with_capacity(listed_count + delisted_count);
    let mut reused = 0usize;
    // Delisted first, so a reused string keeps the LISTED entry's name and type.
    for (entries, listed) in [(delisted, false), (listed, true)] {
        for entry in entries {
            if by_symbol
                .insert(entry.ticker.clone(), SymbolRef { entry, listed })
                .is_some()
            {
                reused += 1;
            }
        }
    }
    println!(
        "[universe] reference data: {listed_count} listed, {delisted_count} delisted, \
         {} distinct tickers, {reused} reused string(s)",
        by_symbol.len()
    );
    Ok(ReferenceIndex {
        by_symbol,
        listed: listed_count,
        delisted: delisted_count,
        reused,
    })
}

/// Median inputs: per-symbol dollar volume, keyed by the session it was measured on so a symbol's
/// own trading life stays visible.
async fn grouped_samples(
    dates: &[NaiveDate],
) -> Result<(usize, HashMap<String, Vec<(NaiveDate, f64)>>)> {
    let mut tasks = JoinSet::new();
    for date in dates.iter().copied() {
        tasks.spawn(async move { (date, polygon::grouped_daily(date).await) });
    }
    let mut sampled_days = 0usize;
    let mut dollar_volumes: HashMap<String, Vec<(NaiveDate, f64)>> = HashMap::new();
    while let Some(joined) = tasks.join_next().await {
        let (date, result) = joined.context("grouped daily sample task panicked")?;
        let rows = match result {
            Ok(rows) => rows,
            Err(error) => {
                eprintln!("[universe] grouped daily sample {date} failed: {error:#}");
                continue;
            }
        };
        if rows.is_empty() {
            continue;
        }
        sampled_days += 1;
        for (symbol, bar) in rows {
            let dollar_volume = bar.close as f64 * bar.volume as f64;
            if dollar_volume > 0.0 {
                dollar_volumes
                    .entry(symbol)
                    .or_default()
                    .push((date, dollar_volume));
            }
        }
    }
    Ok((sampled_days, dollar_volumes))
}

fn load_universe() -> Result<Option<UniverseFile>> {
    let path = universe_path();
    let Ok(raw) = fs::read_to_string(&path) else {
        return Ok(None);
    };
    let file: UniverseFile = serde_json::from_str(&raw)
        .with_context(|| format!("parsing cached universe {}", path.display()))?;
    if file.entries.is_empty() {
        return Ok(None);
    }
    Ok(Some(file))
}

/// Cached entries clearing `min_dollar_volume`, for consumers that need the metadata and not just
/// the names. `None` when liquidity has never been measured.
pub fn universe_entries(min_dollar_volume: f64) -> Result<Option<Vec<UniverseEntry>>> {
    Ok(load_universe()?.map(|file| {
        file.entries
            .into_iter()
            .filter(|entry| entry.median_dollar_volume >= min_dollar_volume)
            .collect()
    }))
}

/// The `train | val` instant the cached ranking was measured against, i.e. the instant universe
/// membership is a function of. `None` when liquidity has never been measured.
///
/// Compare this against the bounds a run is training with: they must agree, or the corpus was
/// selected under a different notion of "train" than it is being scored under. It is the cheap check
/// that [`PINNED_SPLIT_BOUNDS`] has not gone stale against a re-measured universe.
pub fn universe_train_end() -> Result<Option<DateTime<Utc>>> {
    let Some(file) = load_universe()? else {
        return Ok(None);
    };
    let parsed = DateTime::parse_from_rfc3339(&file.train_end_utc)
        .with_context(|| {
            format!(
                "cached universe {} records an unparseable train|val instant {:?}",
                universe_path().display(),
                file.train_end_utc
            )
        })?
        .with_timezone(&Utc);
    Ok(Some(parsed))
}

/// SHA-256 of the cached ranking file, for checkpoint provenance. `None` when it does not exist.
///
/// A pinned boundary and a floor are only half of the record: which SYMBOLS the floor admitted is
/// the other half, and it changes whenever the ranking is re-measured. Folding this digest into a
/// checkpoint's lineage makes "which universe was this trained on" answerable from the artifact
/// instead of from memory.
pub fn universe_fingerprint() -> Result<Option<String>> {
    let path = universe_path();
    let Ok(raw) = fs::read(&path) else {
        return Ok(None);
    };
    let digest = ring::digest::digest(&ring::digest::SHA256, &raw);
    Ok(Some(
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    ))
}

fn universe_path() -> PathBuf {
    Path::new(shared::paths::DATA_PATH).join("universe.json")
}

/// Corpus directory written by `Ingest` and read by `Pretrain`.
pub fn bars_dir() -> PathBuf {
    Path::new(shared::paths::DATA_PATH).join("bars")
}

/// `count` dates spread evenly over `span_days`, each snapped back onto a weekday.
fn sample_trading_days(count: usize, span_days: i64, end: NaiveDate) -> Vec<NaiveDate> {
    let step = (span_days / count.max(1) as i64).max(1);
    let mut dates: Vec<NaiveDate> = (0..count as i64)
        .map(|index| previous_weekday(end - Duration::days(index * step)))
        .collect();
    dates.sort_unstable();
    dates.dedup();
    dates
}

fn previous_weekday(date: NaiveDate) -> NaiveDate {
    let mut date = date;
    while matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
        date = date - Days::new(1);
    }
    date
}

fn weekdays(from: NaiveDate, to: NaiveDate) -> Vec<NaiveDate> {
    let mut dates = Vec::new();
    let mut date = from;
    while date <= to {
        if !matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
            dates.push(date);
        }
        date = date + Days::new(1);
    }
    dates
}

fn day_start_ms(date: NaiveDate) -> i64 {
    date.and_hms_opt(0, 0, 0)
        .expect("midnight is a valid time")
        .and_utc()
        .timestamp_millis()
}

fn format_ts(ts_ms: i64) -> String {
    DateTime::<Utc>::from_timestamp_millis(ts_ms)
        .map(|stamp| stamp.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| ts_ms.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar_at(ts_ms: i64, close: f32) -> PackedBar {
        PackedBar {
            ts_ms,
            open: close,
            high: close,
            low: close,
            close,
            volume: 1.0,
            vwap: close,
            trades: 1,
        }
    }

    /// The cut point is localized by the data: a reused ticker leaves a hole or a level jump, a
    /// reorganisation or a continuous listing leaves neither, and the LAST break wins so a ticker
    /// that changed hands twice is cut at the most recent handover.
    #[test]
    fn discontinuities_localize_the_cut() {
        let day = 86_400_000i64;
        let at = day_start_ms(NaiveDate::from_ymd_opt(2022, 6, 9).unwrap());

        // META: an ETF at ~$12, a 131-day hole, then Meta Platforms at ~$196.
        let spliced = vec![
            bar_at(at - 131 * day - 300_000, 12.34),
            bar_at(at - 131 * day, 12.39),
            bar_at(at + 300_000, 196.46),
        ];
        assert_eq!(discontinuities(&spliced), vec![2]);

        // Holdco reorganisation: continuous series, nothing to cut.
        let reorg = vec![
            bar_at(at - 2 * day, 110.00),
            bar_at(at - day, 110.50),
            bar_at(at + 300_000, 111.20),
        ];
        assert!(discontinuities(&reorg).is_empty());

        // BBBY: an overnight handover with no hole is caught by the 144x level jump.
        let overnight = vec![bar_at(at - day, 0.06), bar_at(at + 300_000, 8.72)];
        assert_eq!(discontinuities(&overnight), vec![1]);

        // A splice followed later by an ordinary trading halt: BOTH breaks must be visible, so the
        // halt cannot hide the splice behind it (this is what PINC and NIQ looked like).
        let splice_then_halt = vec![
            bar_at(at - 900 * day, 4.0),
            bar_at(at - 500 * day, 4.1),
            bar_at(at - 100 * day, 40.0),
            bar_at(at - 99 * day, 40.5),
            bar_at(at + 300_000, 40.2),
        ];
        assert_eq!(discontinuities(&splice_then_halt), vec![1, 2, 4]);

        // A weekend is not a discontinuity.
        let weekend = vec![
            bar_at(at, 50.0),
            bar_at(at + 3 * day, 50.4),
            bar_at(at + 3 * day + 300_000, 50.6),
        ];
        assert!(discontinuities(&weekend).is_empty());
        assert!(discontinuities(&[]).is_empty());
    }

    #[test]
    fn resolution_parsing_maps_to_polygon_spans() {
        let five_min = Resolution::parse("5min").unwrap();
        assert_eq!(
            (five_min.multiplier, five_min.timespan, five_min.res_secs),
            (5, "minute", 300)
        );
        let hourly = Resolution::parse("2h").unwrap();
        assert_eq!((hourly.multiplier, hourly.timespan), (2, "hour"));
        let daily = Resolution::parse("day").unwrap();
        assert_eq!(
            (daily.multiplier, daily.timespan, daily.res_secs),
            (1, "day", 86_400)
        );
        assert!(Resolution::parse("5fortnights").is_err());
        assert!(Resolution::parse("0min").is_err());
    }

    #[test]
    fn incomplete_trailing_bars_are_dropped() {
        let now_ms = Utc::now().timestamp_millis();
        let bar = |ts_ms: i64| PackedBar {
            ts_ms,
            open: 1.0,
            high: 1.0,
            low: 1.0,
            close: 1.0,
            volume: 1.0,
            vwap: 1.0,
            trades: 1,
        };
        // Closed, exactly closed, and still forming, at a 300s resolution.
        let mut bars = vec![
            bar(now_ms - 900_000),
            bar(now_ms - 300_000),
            bar(now_ms - 60_000),
        ];
        drop_incomplete(&mut bars, 300);
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[1].ts(), now_ms - 300_000);

        let mut only_forming = vec![bar(now_ms)];
        drop_incomplete(&mut only_forming, 300);
        assert!(only_forming.is_empty());
    }

    #[test]
    fn resolution_from_secs_prefers_the_coarsest_exact_span() {
        assert_eq!(Resolution::from_secs(300).unwrap().timespan, "minute");
        assert_eq!(Resolution::from_secs(86_400).unwrap().timespan, "day");
        assert_eq!(Resolution::from_secs(90).unwrap().timespan, "second");
        assert!(Resolution::from_secs(0).is_err());
    }

    /// Universe membership must be a function of training-period information only: a symbol that
    /// entered the corpus because of how it traded during val or test contaminates every held-out
    /// number computed from it, and no calendar split can undo that afterwards.
    #[test]
    fn ranking_sessions_stay_strictly_inside_the_training_window() {
        let today = NaiveDate::from_ymd_opt(2026, 8, 15).unwrap();
        let train_end = DateTime::parse_from_rfc3339("2025-10-07T12:10:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let dates = ranking_sessions(train_end, today).unwrap();
        let boundary = train_end.date_naive();
        let plan_floor = today - Duration::days(PLAN_WINDOW_DAYS);
        assert!(dates.len() >= UNIVERSE_SAMPLES - 2, "{}", dates.len());
        for date in &dates {
            assert!(*date < boundary, "{date} is not before the train|val boundary");
            assert!(*date >= plan_floor, "{date} predates the vendor window");
            assert!(!matches!(date.weekday(), Weekday::Sat | Weekday::Sun));
        }
        // Spread over the window, not bunched at its recent end: a name that delisted early in the
        // corpus only ever appears in the old samples.
        assert!(
            dates[0] < boundary - Duration::days(1000),
            "oldest sampled session {} is too recent",
            dates[0]
        );
        assert!(dates.windows(2).all(|pair| pair[0] < pair[1]));
    }

    /// A name that traded liquidly and then delisted must outrank a thin survivor. Ranking on all
    /// sampled sessions instead of the ones it traded in would bury it, which is the survivorship
    /// bias the delisted half of the reference data exists to remove.
    #[test]
    fn ranking_measures_each_symbol_over_its_own_life() {
        let day = |offset: i64| NaiveDate::from_ymd_opt(2022, 1, 3).unwrap() + Duration::days(offset);
        let reference = ReferenceIndex {
            by_symbol: [
                ("GONE", "Acquired Industries", "CS", false),
                ("THIN", "Thin Survivor Inc", "CS", true),
                ("STUB", "Stub Listing Inc", "CS", true),
            ]
            .into_iter()
            .map(|(ticker, name, kind, listed)| {
                (
                    ticker.to_string(),
                    SymbolRef {
                        entry: polygon::TickerRef {
                            ticker: ticker.to_string(),
                            name: name.to_string(),
                            kind: kind.to_string(),
                            primary_exchange: "XNAS".to_string(),
                        },
                        listed,
                    },
                )
            })
            .collect(),
            listed: 2,
            delisted: 1,
            reused: 0,
        };
        let dollar_volumes = HashMap::from([
            // Liquid for four samples, then gone.
            (
                "GONE".to_string(),
                (0..4).map(|i| (day(i * 25), 40e6)).collect::<Vec<_>>(),
            ),
            // Present throughout, but thin.
            (
                "THIN".to_string(),
                (0..20).map(|i| (day(i * 25), 2e6)).collect::<Vec<_>>(),
            ),
            // Two sessions of life: below the presence gate whatever it traded.
            ("STUB".to_string(), vec![(day(0), 90e6), (day(25), 90e6)]),
            // Traded, but not in the reference data at all.
            ("XXUNKNOWN".to_string(), vec![(day(0), 50e6); 5]),
        ]);

        let ranked = rank_symbols(dollar_volumes, &reference, 3);
        let symbols: Vec<&str> = ranked.iter().map(|entry| entry.symbol.as_str()).collect();
        assert_eq!(symbols, vec!["GONE", "THIN"]);
        let gone = &ranked[0];
        assert!(gone.delisted);
        assert_eq!(gone.sessions, 4);
        assert_eq!(gone.median_dollar_volume, 40e6);
        assert_eq!(gone.first_session, day(0));
        assert_eq!(gone.last_session, day(75));
        assert!(!ranked[1].delisted);
    }

    /// The coverage test decides whether an existing corpus file is rewritten, so it must never
    /// judge a file by where its history STARTS: a post-floor listing and a splice-repaired series
    /// both legitimately start late, and rewriting them would restore the splices the repair cut.
    #[test]
    fn coverage_ignores_the_left_edge_and_settled_symbols() {
        let dir = std::env::temp_dir().join(format!("ingest-coverage-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 8, 14).unwrap();
        let day = 86_400_000i64;
        let late_start = day_start_ms(end) - 30 * day;

        let write = |symbol: &str, first: i64, last: i64| {
            let path = bar_file_path(&dir, symbol, 300);
            let bars = [bar_at(first, 10.0), bar_at(last, 11.0)];
            write_bar_file(&path, symbol, 300, &bars).unwrap();
            path
        };

        // Late left edge, current right edge: already everything this pass could produce.
        let repaired = write("REPAIRED", late_start, day_start_ms(end) - day);
        assert!(current_file(&repaired, 300, end, false));
        // Stale right edge: the vendor has newer bars, so it is not current...
        let stale = write(
            "STALE",
            day_start_ms(end) - 120 * day,
            day_start_ms(end) - 90 * day,
        );
        assert!(!current_file(&stale, 300, end, false));
        // ...unless the security has stopped trading, in which case there is nothing to add and
        // refetching it on every pass would rewrite it forever.
        assert!(current_file(&stale, 300, end, true));
        // Wrong resolution and missing files are never current.
        assert!(!current_file(&repaired, 86_400, end, true));
        assert!(!current_file(&dir.join("ABSENT.300.bars"), 300, end, true));

        fs::remove_dir_all(&dir).unwrap();
    }

    /// The pin exists so the boundary cannot drift under the universe. Its whole value is the
    /// invariant that no sampled ranking session reaches the pinned `train | val`, so that is what
    /// is asserted here rather than the literal digits.
    #[test]
    fn pinned_bounds_precede_every_ranking_session() {
        let (train_val, val_test) = PINNED_SPLIT_BOUNDS;
        assert!(train_val < val_test, "{train_val} .. {val_test}");
        let pinned = DateTime::<Utc>::from_timestamp_millis(train_val).expect("valid instant");
        assert_eq!(pinned.to_rfc3339(), "2025-10-07T12:10:00+00:00");
        assert_eq!(
            DateTime::<Utc>::from_timestamp_millis(val_test)
                .expect("valid instant")
                .to_rfc3339(),
            "2026-03-13T18:45:00+00:00"
        );

        let today = NaiveDate::from_ymd_opt(2026, 8, 15).expect("valid date");
        let boundary = pinned.date_naive();
        for date in ranking_sessions(pinned, today).expect("sessions") {
            assert!(date < boundary, "sampled session {date} is not inside train");
        }
    }

    #[test]
    fn weekday_enumeration_excludes_weekends() {
        let from = NaiveDate::from_ymd_opt(2026, 8, 10).unwrap();
        let to = NaiveDate::from_ymd_opt(2026, 8, 16).unwrap();
        let dates = weekdays(from, to);
        assert_eq!(dates.len(), 5);
        assert_eq!(dates[0], from);
        assert_eq!(dates[4], NaiveDate::from_ymd_opt(2026, 8, 14).unwrap());
    }
}
