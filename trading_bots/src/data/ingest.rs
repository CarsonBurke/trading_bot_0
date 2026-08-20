//! Corpus ingestion: universe selection from Polygon grouped daily bars, then bulk aggregate
//! download into the packed `.bars` format consumed by the pretraining pipeline.

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Datelike, Days, Duration, NaiveDate, Utc, Weekday};
use serde::{Deserialize, Serialize};
use shared::bars::{bar_file_path, is_temp_bar_file, write_bar_file, BarFile, PackedBar};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::signal::unix::{signal, SignalKind};
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
/// 466 delisted names, $1M/day admits 1,029. Against that, the cost is bounded and small — 5,684
/// symbols clear this floor, 456M bars, 16 GB, and ~11 minutes per epoch against a 44-60 minute
/// budget.
///
/// That count is NOT the trained population. `pretrain` globs `*.<res>.bars` and cuts on
/// `--min-bars` alone, never on this floor, so it loads 5,297 files: 5,259 that clear the floor, 29
/// ranked below it, and 9 with no ranking row at all because they listed after the ranking was
/// measured. Anything per-symbol must therefore be keyed on bars or on the series index, never on
/// this ranking: those 38 sit at the EXPENSIVE end — the thinnest is $16k/session, 62x below this
/// floor — so a join that silently drops unmatched names biases a cost aggregate optimistic. The
/// cost calibration measures ADV, price and spread from the bar stream for exactly this reason and
/// is unaffected; 2 of one 256-name traded sample fell in these sets and both priced normally.
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
/// History window the corpus already on disk was downloaded under, credited to every file the
/// ingest manifest predates.
///
/// Measured rather than assumed: every one of the 5,728 `*.300.bars` files carries an identical
/// left edge of 2021-08-17, one session after `2026-08-15 - 5*365d`, and 2026-08-15 is the day they
/// were written. So the corpus is uniformly clipped at the five-year Starter window, and this is
/// the span a pre-manifest file honestly claims.
///
/// Deliberately a SEPARATE constant from [`PLAN_WINDOW_DAYS`], which is the vendor's CURRENT
/// entitlement and feeds universe sampling. Coupling them would re-credit the old corpus with depth
/// it does not contain the moment the subscription deepens — which is exactly the situation this
/// constant exists to survive.
const LEGACY_CORPUS_WINDOW_DAYS: i64 = 5 * 365;
/// Journal of completed per-symbol downloads, beside the corpus it describes.
const MANIFEST_FILE: &str = ".ingest_manifest.jsonl";
/// Rewrite the journal when replaying it costs more than this multiple of its live entry count.
const MANIFEST_COMPACT_RATIO: usize = 3;

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
    /// Symbols a stop request reached before they started. These are the resume backlog.
    pub cancelled: usize,
    pub bars: usize,
    pub first_ts_ms: Option<i64>,
    /// Symbols whose history stopped at the vendor's entitlement boundary rather than at the
    /// requested window. A nonzero count means `--years` asks for more than the plan serves.
    pub capped: usize,
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
    let shutdown = install_shutdown_handler();
    let pass = Pass::open(
        &out_dir,
        resolution.res_secs,
        args.years,
        args.force,
        shutdown,
    )?;
    let summary = if args.daily {
        ingest_daily(&targets, &out_dir, args.concurrency, &pass).await?
    } else {
        ingest_intraday(&targets, &out_dir, args.concurrency, &pass).await?
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
    // An entitlement-capped pass is the one failure the operator cannot see in the numbers above:
    // the bars land, the manifest records the depth the vendor actually served, and every following
    // pass dutifully refetches those symbols because the request is still unsatisfied. Said loudly,
    // with the fix, rather than left to be discovered by a corpus that is shorter than believed.
    if summary.capped > 0 {
        eprintln!(
            "[ingest] WARNING: {}/{} written symbol(s) stopped at the vendor's entitlement \
             boundary, not at the requested {} year(s). The plan serves less history than asked \
             for; lower --years to what it serves, or these symbols are refetched every pass.",
            summary.capped, summary.written, args.years
        );
    }
    // A stop request with a backlog is a successful partial pass, not a failure: every completed
    // symbol is on disk and in the manifest, so the gates below would be measuring a backlog rather
    // than a defect. A signal that lands after the last task joined leaves no backlog, and must not
    // buy a failed pass an exit code of zero.
    if summary.cancelled > 0 {
        println!(
            "[ingest] STOPPED by signal: {} done this pass ({} written, {} skipped), {} still to \
             do, {} empty, {} failed",
            summary.written + summary.skipped,
            summary.written,
            summary.skipped,
            summary.cancelled,
            summary.empty,
            summary.failed
        );
        println!("[ingest] resume with:\n  {}", resume_command(&args));
        return Ok(());
    }
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
    if pass.stopping() {
        println!("[ingest] stop requested; nothing was left pending. Pass complete.");
    }
    Ok(())
}

/// The exact invocation that resumes an interrupted pass.
///
/// `--refresh-universe` is deliberately never emitted, whatever the interrupted pass was given:
/// re-measuring the ranking rewrites `long_data/universe.json`, whose digest is folded into every
/// bar checkpoint's lineage hash (see [`universe_fingerprint`]), so a resume that re-measured would
/// make every existing checkpoint refuse to load. `--force` is omitted for the same class of
/// reason — it is unconditional, so resuming with it would restart the download from zero, which is
/// the failure the manifest exists to prevent.
fn resume_command(args: &IngestArgs) -> String {
    let mut parts = vec![
        "cargo run --release -p trading_bot_0 --bin trading_bot_0 -- ingest".to_string(),
        format!("--years {}", args.years),
        format!("--concurrency {}", args.concurrency),
        format!("--min-dollar-volume {}", args.min_dollar_volume),
        format!("--min-bars {}", args.min_bars),
    ];
    if args.daily {
        parts.push("--daily".to_string());
    } else {
        parts.push(format!("--resolution {}", args.resolution));
    }
    if let Some(train_end) = args.train_end {
        parts.push(format!("--train-end {}", train_end.to_rfc3339()));
    }
    parts.join(" ")
}

/// Flips on the first SIGINT/SIGTERM so the pass stops starting new symbols; a second one exits
/// immediately.
///
/// Correctness does NOT depend on this running. The manifest is fsynced per completed symbol and
/// every corpus file is installed by rename, so a hard kill loses at most the symbols in flight and
/// the next pass refetches exactly those. This handler is the courtesy that makes a stop clean and
/// prints the resume line; the atomicity is the guarantee.
fn install_shutdown_handler() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    for (label, kind) in [
        ("SIGINT", SignalKind::interrupt()),
        ("SIGTERM", SignalKind::terminate()),
    ] {
        let mut stream = match signal(kind) {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("[ingest] cannot handle {label} ({error}); a stop will be a hard kill");
                continue;
            }
        };
        let flag = Arc::clone(&flag);
        tokio::spawn(async move {
            while stream.recv().await.is_some() {
                if flag.swap(true, Ordering::SeqCst) {
                    eprintln!("[ingest] second {label}; exiting now, in-flight symbols abandoned");
                    std::process::exit(130);
                }
                eprintln!(
                    "[ingest] {label} received: finishing the symbols in flight, then stopping"
                );
            }
        });
    }
    flag
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

/// One completed per-(symbol, resolution) download, as recorded by the pass that finished it.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestEntry {
    symbol: String,
    res_secs: u32,
    /// Window start the completed download REQUESTED, in epoch millis UTC. The comparison key, and
    /// the only field [`covered`] reads.
    ///
    /// An absolute instant rather than the `--years` it came from, because the vendor's window is
    /// ROLLING: `--years 10` resolves to a different start every single day, so "downloaded under
    /// years=10" is unfalsifiable — a later pass would compare 10 against 10, skip every symbol,
    /// and leave the operator believing they hold ten years. An instant is monotone and stays
    /// correct across a plan change in either direction.
    ///
    /// Also NOT the first bar's timestamp. A series legitimately starts after the window it was
    /// requested over — a late listing, a splice repair — so keying on the data's own left edge
    /// refetches those symbols forever; that is the trap [`covered`] documents.
    window_start_ms: i64,
    /// The `--years` that start was derived from. Provenance for a human reader, never compared.
    years: u32,
    completed_at_ms: i64,
    bars: usize,
    first_ts_ms: i64,
    last_ts_ms: i64,
}

/// Replay of the manifest journal: newest record per (symbol, resolution), and its replay cost.
struct ManifestState {
    entries: HashMap<(String, u32), ManifestEntry>,
    lines: usize,
    damaged: usize,
    /// The replay reached end-of-file without an I/O error, so `entries` covers the WHOLE journal.
    /// Only then may the journal be replaced by a compaction of it.
    clean: bool,
}

/// Replays the append-only manifest, discarding any line that does not parse.
///
/// A torn final line is the expected shape of an interrupted append, and it costs exactly the one
/// symbol it described. The alternative — a single JSON document rewritten per symbol — puts the
/// WHOLE record at risk on every one of thousands of updates, and a truncated blob there is the
/// "next run starts from zero" failure this file exists to prevent.
///
/// Framing is on raw bytes rather than [`BufRead::lines`] so that damage is LOCAL. `lines()` fails
/// the whole iterator on a line that is not UTF-8, which would make one bad line hide every record
/// appended after it — a journal that silently stops accumulating while still looking valid, and so
/// a corpus that silently refetches from zero forever.
///
/// An absent journal replays as empty, which is what makes a missing manifest degrade to the
/// pre-manifest behaviour rather than to a full redownload; see [`covered`].
fn replay_manifest(path: &Path) -> ManifestState {
    let mut state = ManifestState {
        entries: HashMap::new(),
        lines: 0,
        damaged: 0,
        clean: true,
    };
    let Ok(file) = File::open(path) else {
        return state;
    };
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => break,
            Ok(_) => {}
            // A real I/O error, not a framing one: nothing after it is reachable, and the prefix
            // decoded so far must not be mistaken for the whole journal.
            Err(_) => {
                state.damaged += 1;
                state.clean = false;
                break;
            }
        }
        let trimmed = line.strip_suffix(b"\n").unwrap_or(&line);
        if trimmed.is_empty() {
            continue;
        }
        state.lines += 1;
        match serde_json::from_slice::<ManifestEntry>(trimmed) {
            Ok(entry) => {
                state
                    .entries
                    .insert((entry.symbol.clone(), entry.res_secs), entry);
            }
            Err(_) => state.damaged += 1,
        }
    }
    state
}

/// True when the journal's last byte is not a newline, i.e. a writer was killed mid-append.
///
/// An absent or empty journal is terminated by definition.
fn unterminated(path: &Path) -> Result<bool> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| format!("reading ingest manifest {}", path.display()))
        }
    };
    let len = file
        .metadata()
        .with_context(|| format!("stat of ingest manifest {}", path.display()))?
        .len();
    if len == 0 {
        return Ok(false);
    }
    file.seek(SeekFrom::End(-1))
        .with_context(|| format!("seeking ingest manifest {}", path.display()))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)
        .with_context(|| format!("reading ingest manifest {}", path.display()))?;
    Ok(last[0] != b'\n')
}

/// Rewrites the journal as one line per live entry, atomically.
///
/// Called only before the pass opens its append handle, so the replace cannot race an append, and
/// the rename means an interrupted compaction leaves the previous journal fully intact.
fn compact_manifest(path: &Path, entries: &HashMap<(String, u32), ManifestEntry>) -> Result<()> {
    let temp = path.with_file_name(format!("{MANIFEST_FILE}.compacting"));
    let mut keys: Vec<&(String, u32)> = entries.keys().collect();
    keys.sort();
    let compacted = || -> Result<()> {
        let mut out = BufWriter::new(
            File::create(&temp).with_context(|| format!("creating {}", temp.display()))?,
        );
        for key in keys {
            serde_json::to_writer(&mut out, &entries[key])?;
            out.write_all(b"\n")?;
        }
        out.flush()?;
        out.into_inner()
            .map_err(|e| anyhow!("flushing {}: {e}", temp.display()))?
            .sync_all()
            .with_context(|| format!("syncing {}", temp.display()))?;
        fs::rename(&temp, path)
            .with_context(|| format!("renaming {} onto {}", temp.display(), path.display()))
    };
    compacted().inspect_err(|_| {
        let _ = fs::remove_file(&temp);
    })
}

/// Append-only handle on the manifest.
///
/// Append rather than atomic whole-file replace: one line per completed symbol is O(1), and on an
/// `O_APPEND` fd a sub-page write followed by `sync_data` is the smallest durable unit the
/// filesystem offers. Replacing a 10k-entry document once per symbol is quadratic in the corpus and
/// re-risks every earlier entry on every update.
struct ManifestJournal {
    file: File,
    path: PathBuf,
}

impl ManifestJournal {
    fn open(path: PathBuf) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening ingest manifest {}", path.display()))?;
        // Terminate a line a kill left half-written, so the next append is framed as its own record.
        // Without this the partial bytes and the next complete record read as ONE unparseable line
        // and a tear costs two symbols instead of the one it interrupted.
        if unterminated(&path)? {
            file.write_all(b"\n")
                .and_then(|()| file.sync_data())
                .with_context(|| format!("terminating ingest manifest {}", path.display()))?;
        }
        Ok(Self { file, path })
    }

    fn record(&mut self, entry: &ManifestEntry) -> Result<()> {
        let mut line = serde_json::to_vec(entry)
            .with_context(|| format!("encoding manifest entry for {}", entry.symbol))?;
        line.push(b'\n');
        self.file
            .write_all(&line)
            .and_then(|()| self.file.sync_data())
            .with_context(|| format!("appending to ingest manifest {}", self.path.display()))
    }
}

/// Everything one ingestion pass shares across its per-symbol tasks: what it is asking the vendor
/// for, what a previous pass already achieved, and where completions are recorded.
struct Pass {
    res_secs: u32,
    years: u32,
    /// Right edge of the request, i.e. today.
    end: NaiveDate,
    /// Left edge of the request as a date, for the vendor calls.
    floor: NaiveDate,
    /// The same left edge in epoch millis. Recorded verbatim on completion.
    requested_start_ms: i64,
    /// Window start credited to a corpus file that has no manifest record.
    bootstrap_start_ms: i64,
    recorded: HashMap<(String, u32), ManifestEntry>,
    manifest: Mutex<ManifestJournal>,
    /// Flipped by SIGINT/SIGTERM: no further symbol is started.
    shutdown: Arc<AtomicBool>,
    force: bool,
}

impl Pass {
    fn open(
        out_dir: &Path,
        res_secs: u32,
        years: u32,
        force: bool,
        shutdown: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        fs::create_dir_all(out_dir)
            .with_context(|| format!("creating {}", out_dir.display()))?;
        let swept = sweep_temp_files(out_dir);
        if swept > 0 {
            println!("[ingest] swept {swept} staging file(s) left by an interrupted pass");
        }

        let path = out_dir.join(MANIFEST_FILE);
        let state = replay_manifest(&path);
        if state.damaged > 0 {
            // Expected after a kill mid-append, and self-healing: the symbols those lines described
            // are simply refetched. Reported rather than silent so a systematically broken journal
            // is visible instead of looking like an empty one.
            eprintln!(
                "[ingest] manifest {}: discarded {} unreadable line(s), {} usable record(s)",
                path.display(),
                state.damaged,
                state.entries.len()
            );
        }
        // Compaction is a pure optimisation, and it replaces the journal wholesale. Skipping it
        // whenever the replay did not reach clean EOF is the difference between an optimisation and
        // a data loss: a replay that aborted on an I/O error returns only the PREFIX it decoded, and
        // compacting that prefix would delete every durable record beyond the bad offset.
        if state.clean && state.lines > state.entries.len() * MANIFEST_COMPACT_RATIO + 64 {
            if let Err(error) = compact_manifest(&path, &state.entries) {
                eprintln!("[ingest] manifest compaction skipped: {error:#}");
            }
        }

        let end = Utc::now().date_naive();
        let floor = end - Duration::days(365 * years.max(1) as i64);
        println!(
            "[ingest] manifest {}: {} completed record(s); requesting {} .. {}",
            path.display(),
            state.entries.len(),
            floor,
            end
        );
        Ok(Arc::new(Self {
            res_secs,
            years,
            end,
            floor,
            requested_start_ms: day_start_ms(floor),
            bootstrap_start_ms: day_start_ms(end - Duration::days(LEGACY_CORPUS_WINDOW_DAYS)),
            recorded: state.entries,
            manifest: Mutex::new(ManifestJournal::open(path)?),
            shutdown,
            force,
        }))
    }

    fn stopping(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// The window start a fetch for `symbol` must reach: the request, or deeper if a completed
    /// download already recorded deeper.
    ///
    /// [`write_bar_file`] replaces a corpus file wholesale, so a pass that refetches a symbol at a
    /// shallower floor DESTROYS history. That is reachable from a command nobody thinks of as
    /// destructive: once a file's right edge goes stale past [`FRESH_TAIL_DAYS`], [`covered`] returns
    /// false before the intent test is ever consulted, so a habitual bare `ingest` would rewrite a
    /// ten-year series as a five-year one and overwrite its own record saying so. A refresh may add
    /// to a series; it may never shorten one.
    fn floor_for(&self, symbol: &str) -> (NaiveDate, i64) {
        match self.recorded.get(&(symbol.to_string(), self.res_secs)) {
            Some(entry) if entry.window_start_ms < self.requested_start_ms => (
                DateTime::from_timestamp_millis(entry.window_start_ms)
                    .map_or(self.floor, |instant| instant.date_naive()),
                entry.window_start_ms,
            ),
            _ => (self.floor, self.requested_start_ms),
        }
    }

    /// [`covered`] evaluated off the async runtime, since it mmaps and reads the file header.
    async fn covers(&self, path: PathBuf, symbol: &str, settled: bool) -> bool {
        let recorded = self
            .recorded
            .get(&(symbol.to_string(), self.res_secs))
            .map(|entry| entry.window_start_ms);
        let (res_secs, end) = (self.res_secs, self.end);
        let (requested, bootstrap) = (self.requested_start_ms, self.bootstrap_start_ms);
        tokio::task::spawn_blocking(move || {
            covered(
                &path, res_secs, end, settled, requested, recorded, bootstrap,
            )
        })
        .await
        .unwrap_or(false)
    }

    /// Records a finished symbol. Called only after the corpus file's rename has landed, so a
    /// record always implies a complete file on disk.
    ///
    /// `window_start_ms` is what the download ACHIEVED, not what the pass asked for, so an
    /// entitlement-capped series is never stamped as complete to a depth the vendor refused.
    fn complete(
        &self,
        symbol: &str,
        window_start_ms: i64,
        bars: usize,
        first_ts_ms: i64,
        last_ts_ms: i64,
    ) -> Result<()> {
        let entry = ManifestEntry {
            symbol: symbol.to_string(),
            res_secs: self.res_secs,
            window_start_ms,
            years: self.years,
            completed_at_ms: Utc::now().timestamp_millis(),
            bars,
            first_ts_ms,
            last_ts_ms,
        };
        self.manifest
            .lock()
            .map_err(|_| anyhow!("ingest manifest lock poisoned"))?
            .record(&entry)
    }
}

/// Removes staging files an interrupted writer left behind.
///
/// They are invisible to every corpus reader — their extension is not `bars` — but one accumulates
/// per interrupted symbol, and this process is expected to be interrupted repeatedly.
fn sweep_temp_files(dir: &Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if is_temp_bar_file(&path) && fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Downloads intraday aggregates for every target, paging backwards until the plan window ends.
async fn ingest_intraday(
    targets: &[IngestTarget],
    out_dir: &Path,
    concurrency: usize,
    pass: &Arc<Pass>,
) -> Result<IngestSummary> {
    let resolution = Resolution::from_secs(pass.res_secs)?;
    let permits = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut tasks = JoinSet::new();
    for target in targets {
        let symbol = target.symbol.clone();
        let settled = target.settled;
        let out_dir = out_dir.to_path_buf();
        let permits = Arc::clone(&permits);
        let pass = Arc::clone(pass);
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await;
            // Checked after the permit, so a stop drains the queue instead of racing it.
            if pass.stopping() {
                return SymbolOutcome::Cancelled;
            }
            let path = bar_file_path(&out_dir, &symbol, pass.res_secs);
            if !pass.force && pass.covers(path.clone(), &symbol, settled).await {
                return SymbolOutcome::Skipped;
            }
            let (floor, _) = pass.floor_for(&symbol);
            let fetched = match fetch_history(&symbol, resolution, floor, pass.end).await {
                Ok(fetched) => fetched,
                Err(error) => return SymbolOutcome::Failed(symbol, format!("{error:#}")),
            };
            let mut bars = fetched.bars;
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
                persist(
                    pass,
                    path,
                    symbol,
                    bars,
                    fetched.achieved_start_ms,
                    fetched.capped,
                )
                .await
            }
        });
    }

    drain(&mut tasks, targets.len(), "symbols").await
}

/// Downloads daily bars for every symbol via grouped-daily fan-out over trading sessions.
async fn ingest_daily(
    targets: &[IngestTarget],
    out_dir: &Path,
    concurrency: usize,
    pass: &Arc<Pass>,
) -> Result<IngestSummary> {
    let wanted: Arc<HashSet<String>> =
        Arc::new(targets.iter().map(|target| target.symbol.clone()).collect());
    // Pass-level rather than per-symbol: the grouped-daily endpoint is keyed by SESSION, so one
    // fan-out serves every symbol and the span has to cover the deepest series any of them will
    // rewrite. A shallower fan-out would replace a deep file with a short one; see
    // [`Pass::floor_for`].
    let floor = targets
        .iter()
        .map(|target| pass.floor_for(&target.symbol).0)
        .min()
        .unwrap_or(pass.floor);
    let sessions = weekdays(floor, pass.end);
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
    // A partial session set must never be written: every symbol's file is replaced wholesale, so a
    // missing session is a hole punched into the corpus rather than a gap to be filled later. A stop
    // during the fan-out therefore discards the pass instead of writing what it managed to fetch.
    if pass.stopping() {
        println!("[ingest] stop requested during the grouped-daily fan-out; writing nothing");
        return Ok(IngestSummary {
            cancelled: targets.len(),
            ..IngestSummary::default()
        });
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
        let pass = Arc::clone(pass);
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await;
            if pass.stopping() {
                return SymbolOutcome::Cancelled;
            }
            let path = bar_file_path(&out_dir, &symbol, pass.res_secs);
            if !pass.force && pass.covers(path.clone(), &symbol, settled).await {
                return SymbolOutcome::Skipped;
            }
            bars.sort_unstable_by_key(|bar| bar.ts_ms);
            bars.dedup_by_key(|bar| bar.ts_ms);
            drop_incomplete(&mut bars, pass.res_secs);
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
            // The fan-out covered `floor`, so that is what this write achieved. Grouped-daily has
            // no per-window entitlement signal, hence never capped.
            persist(pass, path, symbol, bars, day_start_ms(floor), false).await
        });
    }

    drain(&mut tasks, targets.len(), "symbols").await
}

enum SymbolOutcome {
    Written {
        bars: usize,
        first_ts_ms: i64,
        last_ts_ms: i64,
        /// The vendor's entitlement ran out before the requested window did.
        capped: bool,
    },
    Skipped,
    Empty(String),
    Failed(String, String),
    /// A stop request arrived before this symbol started. It is the resume backlog, not a failure.
    Cancelled,
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
                capped,
            } => {
                summary.capped += usize::from(capped);
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
            SymbolOutcome::Cancelled => summary.cancelled += 1,
        }
        if completed % PROGRESS_EVERY == 0 || completed == total {
            println!(
                "[ingest] {completed}/{total} {unit} | {} written, {} skipped, {} pending | {} bars | {:.1}s",
                summary.written,
                summary.skipped,
                total - completed,
                summary.bars,
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(summary)
}

/// Writes the corpus file off the async runtime, then records the completion.
///
/// The order is the whole point: [`write_bar_file`] installs the file by rename, so it is complete
/// the instant it is visible, and only then is the manifest appended. A kill between the two leaves
/// a complete file with no record, and the next pass simply refetches that one symbol — the safe
/// direction. The reverse order would let a record vouch for a file that does not exist.
async fn persist(
    pass: Arc<Pass>,
    path: PathBuf,
    symbol: String,
    bars: Vec<PackedBar>,
    window_start_ms: i64,
    capped: bool,
) -> SymbolOutcome {
    let name = symbol.clone();
    let written = tokio::task::spawn_blocking(move || -> Result<(usize, i64, i64)> {
        let first_ts_ms = bars[0].ts_ms;
        let last_ts_ms = bars[bars.len() - 1].ts_ms;
        write_bar_file(&path, &symbol, pass.res_secs, &bars)?;
        pass.complete(&symbol, window_start_ms, bars.len(), first_ts_ms, last_ts_ms)?;
        Ok((bars.len(), first_ts_ms, last_ts_ms))
    })
    .await;
    match written {
        Ok(Ok((bars, first_ts_ms, last_ts_ms))) => SymbolOutcome::Written {
            bars,
            first_ts_ms,
            last_ts_ms,
            capped,
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
) -> Result<Fetched> {
    let mut bars = Vec::new();
    let mut to = end;
    let mut newest = true;
    // Deepest window start the vendor actually answered. Only meaningful once a refusal has been
    // seen; until then the walk is still on course for `floor`.
    let mut deepest_authorized = end;
    let mut capped = false;
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
            // The entitlement ran out before `floor` did. Recorded, because a pass that stamps the
            // requested window on a series the vendor refused to serve would latch that claim and
            // skip the symbol forever — including after a genuine plan upgrade.
            Window::Unauthorized => {
                capped = true;
                break;
            }
            Window::Data(page) => {
                bars.extend(page);
                deepest_authorized = from;
            }
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
    Ok(Fetched {
        bars,
        // A refused window means the vendor's boundary lies above `floor`, so the honest claim is
        // the deepest start it did answer. Never the first bar's timestamp: a security that listed
        // late has a late left edge and no refusal, and conflating the two is the left-edge trap
        // [`covered`] documents.
        achieved_start_ms: day_start_ms(if capped { deepest_authorized } else { floor }),
        capped,
    })
}

/// One symbol's downloaded history, and how deep the vendor actually let the walk go.
struct Fetched {
    bars: Vec<PackedBar>,
    achieved_start_ms: i64,
    /// The vendor refused a window above `floor`, so this series is entitlement-capped rather than
    /// complete to the request.
    capped: bool,
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

/// True when this pass can add nothing to the existing corpus file for a symbol.
///
/// Three tests, all of which must hold.
///
/// REALITY. The file is opened first, and a file that is absent, short, header-inconsistent or of
/// the wrong resolution loses to whatever the manifest claims. A record is evidence about the past,
/// never a substitute for the bytes.
///
/// RIGHT EDGE, unchanged from the pre-manifest rule. A file whose newest bar is stale has newer
/// bars waiting at the vendor — except for a security that has stopped trading, whose series is
/// complete at its last bar. Without that exception every delisted symbol would be redownloaded on
/// every pass, forever.
///
/// INTENT. The window start THIS pass asks for is compared against the window start a COMPLETED
/// download RECORDED. It is deliberately not compared against the file's own left edge. A file's
/// left edge is set by three things the vendor decides and this process cannot: the plan's rolling
/// window, the symbol's own listing date, and the splice repair that truncates a reused ticker at
/// its handover. A left-edge test therefore reports every legitimately-late series as incomplete
/// and rewrites it on every pass, forever — thousands of correct files refetched, and META, BBBY
/// and their kind rewritten from scratch each time. Recorded intent has no such fixed point: it is
/// written once the download that satisfied it finished, so the second pass over the same request
/// skips. That property is exactly what makes an interrupted run resumable, and it is why deepening
/// `--years` needs no new flag: the request is the signal and the manifest makes it safe.
///
/// The left edge does appear, but only ever as an ADDITIONAL reason to skip: a file that already
/// holds bars from before the requested start has nothing left to fetch on its left, whatever the
/// manifest says. Used in that direction it can never trigger a refetch, so it cannot reintroduce
/// the loop above. It is what stops `--daily` from re-pulling the decades [`super::deep_daily`]
/// wrote for the same symbols.
///
/// A symbol with no record at all is credited with `bootstrap_start_ms`
/// ([`LEGACY_CORPUS_WINDOW_DAYS`]), so a corpus that predates the manifest is neither invalidated —
/// a request for the span it was built under still skips, which is byte for byte the pre-manifest
/// behaviour — nor frozen, since a request for a deeper span refetches it.
///
/// Rebuilding a history from scratch regardless of any of this is what `force` is for, and it is
/// never implicit.
fn covered(
    path: &Path,
    res_secs: u32,
    end: NaiveDate,
    settled: bool,
    requested_start_ms: i64,
    recorded_start_ms: Option<i64>,
    bootstrap_start_ms: i64,
) -> bool {
    let Ok(file) = BarFile::open(path) else {
        return false;
    };
    if file.res_secs() != res_secs {
        return false;
    }
    let Some(last_ts_ms) = file.last_ts_ms() else {
        return false;
    };
    if !settled
        && last_ts_ms < day_start_ms(end) - Duration::days(FRESH_TAIL_DAYS).num_milliseconds()
    {
        return false;
    }
    let reach = recorded_start_ms
        .unwrap_or(bootstrap_start_ms)
        .min(file.first_ts_ms().unwrap_or(i64::MAX));
    reach <= requested_start_ms
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
///
/// THIS IS A CHECKPOINT LOAD GATE, not a diagnostic. It lands in
/// `BarTrainingProvenance::universe_fingerprint`, is folded into the bar world-model lineage hash,
/// and `BarWorldModelMetadata::validate_schema` re-verifies that hash on every load — so changing
/// what this hashes makes every existing bar checkpoint REFUSE to load rather than merely look
/// stale. `env::snapshot::ppo_input_fingerprint` used to share this function's name and is
/// unrelated (it digests the per-symbol PPO input files); do not unify them.
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

    /// A scratch corpus directory, removed by the caller.
    fn scratch(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ingest-{label}-{}-{seq}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_series(dir: &Path, symbol: &str, res_secs: u32, first: i64, last: i64) -> PathBuf {
        let path = bar_file_path(dir, symbol, res_secs);
        write_bar_file(&path, symbol, res_secs, &[bar_at(first, 10.0), bar_at(last, 11.0)]).unwrap();
        path
    }

    /// A pass over `dir` with no signal handler, as a fresh process would open it.
    fn pass_over(dir: &Path, res_secs: u32, years: u32, force: bool) -> Arc<Pass> {
        Pass::open(
            dir,
            res_secs,
            years,
            force,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
    }

    /// The coverage test decides whether an existing corpus file is rewritten, so it must never
    /// judge a file by where its history STARTS: a post-floor listing and a splice-repaired series
    /// both legitimately start late, and rewriting them would restore the splices the repair cut.
    #[test]
    fn coverage_ignores_the_left_edge_and_settled_symbols() {
        let dir = scratch("coverage");
        let end = NaiveDate::from_ymd_opt(2026, 8, 14).unwrap();
        let day = 86_400_000i64;
        let late_start = day_start_ms(end) - 30 * day;
        // A five-year request against a corpus credited with five years: the pre-manifest case, and
        // the one a bare `ingest` must keep behaving exactly as it always did.
        let bootstrap = day_start_ms(end - Duration::days(LEGACY_CORPUS_WINDOW_DAYS));
        let same_span = |path: &Path, res_secs: u32, settled: bool| {
            covered(path, res_secs, end, settled, bootstrap, None, bootstrap)
        };

        // Late left edge, current right edge: already everything this pass could produce.
        let repaired = write_series(&dir, "REPAIRED", 300, late_start, day_start_ms(end) - day);
        assert!(same_span(&repaired, 300, false));
        // Stale right edge: the vendor has newer bars, so it is not current...
        let stale = write_series(
            &dir,
            "STALE",
            300,
            day_start_ms(end) - 120 * day,
            day_start_ms(end) - 90 * day,
        );
        assert!(!same_span(&stale, 300, false));
        // ...unless the security has stopped trading, in which case there is nothing to add and
        // refetching it on every pass would rewrite it forever.
        assert!(same_span(&stale, 300, true));
        // Wrong resolution and missing files are never current.
        assert!(!same_span(&repaired, 86_400, true));
        assert!(!same_span(&dir.join("ABSENT.300.bars"), 300, true));

        fs::remove_dir_all(&dir).unwrap();
    }

    /// The defect this manifest exists to fix: with no record and no left-edge test, a DEEPER
    /// request found every five-year file current and downloaded nothing. Intent-keying must make
    /// the deeper request fetch while leaving the same-span request alone.
    #[test]
    fn a_deeper_request_is_not_satisfied_by_a_shallower_corpus() {
        let dir = scratch("deeper");
        let end = NaiveDate::from_ymd_opt(2026, 8, 19).unwrap();
        let day = 86_400_000i64;
        let bootstrap = day_start_ms(end - Duration::days(LEGACY_CORPUS_WINDOW_DAYS));
        let ten_years = day_start_ms(end - Duration::days(10 * 365));
        let path = write_series(&dir, "AAPL", 300, bootstrap + day, day_start_ms(end) - day);

        // No record: credited with the span the corpus on disk was built under. Five years is
        // satisfied, ten is not.
        assert!(covered(&path, 300, end, false, bootstrap, None, bootstrap));
        assert!(!covered(&path, 300, end, false, ten_years, None, bootstrap));
        // Once a ten-year download is recorded, the same request skips. This termination is the
        // whole difference from a left-edge test, which would never stop refetching a symbol whose
        // history legitimately starts after the window.
        assert!(covered(&path, 300, end, false, ten_years, Some(ten_years), bootstrap));
        let listed_late = write_series(&dir, "LATE", 300, day_start_ms(end) - 90 * day, day_start_ms(end) - day);
        assert!(covered(&listed_late, 300, end, false, ten_years, Some(ten_years), bootstrap));

        // A file that already reaches past the request skips whatever the manifest says: this is
        // the left edge used as a skip reason only, which is what keeps `--daily` from re-pulling
        // the decades `deep_daily` wrote.
        let deep = write_series(&dir, "DEEP", 86_400, ten_years - 4_000 * day, day_start_ms(end) - day);
        assert!(covered(&deep, 86_400, end, false, ten_years, None, bootstrap));

        fs::remove_dir_all(&dir).unwrap();
    }

    /// One completed symbol must survive a process boundary: that is resumability, and it is the
    /// only thing standing between a stopped ten-year pull and a restart from zero.
    #[tokio::test]
    async fn a_completed_symbol_is_skipped_by_the_next_pass() {
        let dir = scratch("resume");
        let day = 86_400_000i64;
        let now = Utc::now().timestamp_millis();
        let bars: Vec<PackedBar> = (0..8)
            .map(|i| bar_at(now - (8 - i) * 300_000, 10.0 + i as f32))
            .collect();

        let first = pass_over(&dir, 300, 10, false);
        let requested = first.requested_start_ms;
        // Nothing on disk, so neither symbol is covered.
        assert!(!first.covers(bar_file_path(&dir, "AAA", 300), "AAA", false).await);
        assert!(!first.covers(bar_file_path(&dir, "BBB", 300), "BBB", false).await);
        let outcome = persist(
            Arc::clone(&first),
            bar_file_path(&dir, "AAA", 300),
            "AAA".to_string(),
            bars.clone(),
            first.requested_start_ms,
            false,
        )
        .await;
        assert!(matches!(outcome, SymbolOutcome::Written { bars: 8, .. }));
        drop(first);

        // A fresh process over the same request: the completed symbol is skipped, the untouched one
        // is still pending, and the record says what it was downloaded under.
        let resumed = pass_over(&dir, 300, 10, false);
        assert_eq!(resumed.requested_start_ms, requested);
        assert!(resumed.covers(bar_file_path(&dir, "AAA", 300), "AAA", false).await);
        assert!(!resumed.covers(bar_file_path(&dir, "BBB", 300), "BBB", false).await);
        let entry = &resumed.recorded[&("AAA".to_string(), 300)];
        assert_eq!((entry.years, entry.bars), (10, 8));
        assert_eq!(entry.window_start_ms, requested);
        assert_eq!(entry.first_ts_ms, bars[0].ts());
        assert_eq!(entry.last_ts_ms, bars[7].ts());

        // `--force` is unconditional: it is the only way to rebuild a recorded symbol, and it is
        // never implicit.
        let forced = pass_over(&dir, 300, 10, true);
        assert!(forced.force);

        // Deleting the file behind the manifest's back must not make the record authoritative.
        fs::remove_file(bar_file_path(&dir, "AAA", 300)).unwrap();
        let after_loss = pass_over(&dir, 300, 10, false);
        assert!(!after_loss.covers(bar_file_path(&dir, "AAA", 300), "AAA", false).await);

        // And a record that is deeper than the request still skips: the pass asks for less than it
        // already has. Guards against the rolling window making a same-`years` rerun refetch.
        let shallower = pass_over(&dir, 300, 5, false);
        assert!(shallower.requested_start_ms > requested);
        write_series(&dir, "AAA", 300, bars[0].ts(), bars[7].ts());
        assert!(shallower.covers(bar_file_path(&dir, "AAA", 300), "AAA", false).await);
        // If it does fetch it anyway — which it will once the right edge goes stale, since the
        // right-edge test short-circuits before the intent test — it must fetch to the DEEPER
        // recorded floor. Refetching to the shallower request would replace ten years of
        // irreplaceable history with five and overwrite the record that said so.
        assert_eq!(shallower.floor_for("AAA").1, requested);
        assert!(shallower.floor_for("AAA").0 < shallower.floor);
        // A symbol with no record is fetched at exactly the request, unchanged.
        assert_eq!(
            shallower.floor_for("BBB"),
            (shallower.floor, shallower.requested_start_ms)
        );

        fs::remove_dir_all(&dir).unwrap();
    }

    /// An interrupted append tears the last line. The journal must lose exactly that symbol and
    /// keep every earlier one: a format where a torn write costs the whole record would send the
    /// next run back to zero, which is the failure mode this design exists to rule out.
    #[test]
    fn a_torn_manifest_costs_one_symbol_and_no_more() {
        let dir = scratch("torn");
        let path = dir.join(MANIFEST_FILE);
        let entry = |symbol: &str, start: i64| ManifestEntry {
            symbol: symbol.to_string(),
            res_secs: 300,
            window_start_ms: start,
            years: 10,
            completed_at_ms: 1_700_000_000_000,
            bars: 42,
            first_ts_ms: start,
            last_ts_ms: start + 300_000,
        };
        let mut journal = ManifestJournal::open(path.clone()).unwrap();
        for (index, symbol) in ["AAA", "BBB", "CCC"].iter().enumerate() {
            journal.record(&entry(symbol, 1_000 + index as i64)).unwrap();
        }
        drop(journal);

        // Round-trip first: what was written is what is read back.
        let intact = replay_manifest(&path);
        assert_eq!((intact.entries.len(), intact.damaged), (3, 0));
        assert_eq!(intact.entries[&("BBB".to_string(), 300)].window_start_ms, 1_001);

        // Now cut the file inside its last line, exactly as a kill mid-append would.
        let whole = fs::read(&path).unwrap();
        let last_newline = whole.iter().rposition(|&b| b == b'\n').unwrap();
        fs::write(&path, &whole[..last_newline - 20]).unwrap();
        let torn = replay_manifest(&path);
        assert_eq!(torn.entries.len(), 2, "earlier records must survive");
        assert_eq!(torn.damaged, 1);
        assert!(torn.entries.contains_key(&("AAA".to_string(), 300)));
        assert!(!torn.entries.contains_key(&("CCC".to_string(), 300)));

        // The tear costs ONE symbol, which requires terminating it before appending. Without that,
        // the partial bytes and the next complete record frame as a single unparseable line and the
        // tear costs two.
        assert!(unterminated(&path).unwrap());
        let mut journal = ManifestJournal::open(path.clone()).unwrap();
        journal.record(&entry("CCC", 1_002)).unwrap();
        drop(journal);
        assert!(!unterminated(&path).unwrap());
        let healed = replay_manifest(&path);
        assert_eq!(healed.entries.len(), 3, "the re-recorded symbol must land");
        assert_eq!(healed.entries[&("CCC".to_string(), 300)].window_start_ms, 1_002);

        // Garbage is discarded line by line rather than poisoning the replay. The non-UTF-8 line is
        // the case that matters: a whole-iterator decoder would stop here and hide every record
        // appended afterwards, so the journal would look valid and quietly stop accumulating.
        let mut damaged = Vec::from(b"{not json at all\n\n\xff\xfe not utf8\n".as_slice());
        damaged.extend_from_slice(&serde_json::to_vec(&entry("EEE", 7_000)).unwrap());
        damaged.push(b'\n');
        fs::write(&path, &damaged).unwrap();
        let after_garbage = replay_manifest(&path);
        assert_eq!(after_garbage.damaged, 2);
        assert_eq!(
            after_garbage.entries.keys().collect::<Vec<_>>(),
            vec![&("EEE".to_string(), 300)],
            "a record after a corrupt line must still be seen"
        );
        // Appending after damage still works, so a damaged journal self-heals instead of latching.
        let mut journal = ManifestJournal::open(path.clone()).unwrap();
        journal.record(&entry("DDD", 9_000)).unwrap();
        drop(journal);
        assert_eq!(replay_manifest(&path).entries.len(), 2);

        // Compaction leaves one line per live entry and is a rename, so it cannot lose the journal.
        let state = replay_manifest(&path);
        compact_manifest(&path, &state.entries).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 2);
        assert!(!dir.join(format!("{MANIFEST_FILE}.compacting")).exists());

        fs::remove_dir_all(&dir).unwrap();
    }

    /// The 17 GB corpus already on disk has no manifest. A missing manifest must therefore read as
    /// "downloaded under the five-year plan it was in fact downloaded under", not as "unknown, fetch
    /// everything": redownloading it costs days of metered bandwidth the operator cannot spend.
    #[tokio::test]
    async fn a_missing_manifest_does_not_invalidate_the_existing_corpus() {
        let dir = scratch("bootstrap");
        let now = Utc::now().timestamp_millis();
        for symbol in ["AAA", "BBB"] {
            let path = bar_file_path(&dir, symbol, 300);
            let start = day_start_ms(Utc::now().date_naive() - Duration::days(LEGACY_CORPUS_WINDOW_DAYS))
                + 86_400_000;
            write_bar_file(
                &path,
                symbol,
                300,
                &[bar_at(start, 10.0), bar_at(now - 300_000, 11.0)],
            )
            .unwrap();
        }
        assert!(!dir.join(MANIFEST_FILE).exists());

        // The span the corpus was built under: every symbol skips, and not one byte is refetched.
        let same = pass_over(&dir, 300, 5, false);
        assert!(same.recorded.is_empty());
        for symbol in ["AAA", "BBB"] {
            assert!(
                same.covers(bar_file_path(&dir, symbol, 300), symbol, false).await,
                "{symbol} would be refetched by a same-span pass"
            );
        }
        // A deeper span: every symbol is fetched, which is the whole point of the change.
        let deeper = pass_over(&dir, 300, 10, false);
        for symbol in ["AAA", "BBB"] {
            assert!(
                !deeper.covers(bar_file_path(&dir, symbol, 300), symbol, false).await,
                "{symbol} would be skipped by a ten-year pass"
            );
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    /// A corpus file must never be observable in a partial state, because the operator will be
    /// stopping this process repeatedly and a truncated `.bars` file is unrecoverable within their
    /// budget. The reader below sees either the old series or the new one at every instant; an
    /// in-place write fails this.
    #[test]
    fn an_interrupted_write_can_never_expose_a_partial_corpus_file() {
        let dir = scratch("atomic");
        let path = bar_file_path(&dir, "AAA", 300);
        let old: Vec<PackedBar> = (0..4).map(|i| bar_at(1_000_000 + i * 300_000, 1.0)).collect();
        write_bar_file(&path, "AAA", 300, &old).unwrap();
        // Large enough that the write is not instantaneous, small enough to stay well inside a
        // bounded-memory test: 400k records is 14 MB.
        let new: Vec<PackedBar> = (0..400_000)
            .map(|i| bar_at(9_000_000 + i * 300_000, 2.0))
            .collect();

        let reader_path = path.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let reader_stop = Arc::clone(&stop);
        let reader = std::thread::spawn(move || {
            let mut seen = 0usize;
            while !reader_stop.load(Ordering::Relaxed) {
                let file = BarFile::open(&reader_path).expect("the target is always openable");
                let len = file.len();
                assert!(len == 4 || len == 400_000, "partial corpus file of {len} bars");
                seen += 1;
            }
            seen
        });
        write_bar_file(&path, "AAA", 300, &new).unwrap();
        stop.store(true, Ordering::Relaxed);
        let observations = reader.join().unwrap();
        assert!(observations > 0);
        assert_eq!(BarFile::open(&path).unwrap().len(), 400_000);
        // A successful write leaves no staging file behind.
        assert_eq!(sweep_temp_files(&dir), 0);

        // What a killed writer DOES leave: a staging file. It is invisible to every corpus reader,
        // the target it was going to replace is untouched, and the next pass sweeps it.
        let staged = dir.join(format!("AAA.300{}999-0", shared::bars::TEMP_INFIX));
        fs::write(&staged, b"truncated garbage").unwrap();
        assert!(is_temp_bar_file(&staged));
        assert_ne!(staged.extension().and_then(|e| e.to_str()), Some("bars"));
        assert_eq!(BarFile::open(&path).unwrap().len(), 400_000);
        assert_eq!(sweep_temp_files(&dir), 1);
        assert!(!staged.exists());
        assert_eq!(BarFile::open(&path).unwrap().len(), 400_000);

        fs::remove_dir_all(&dir).unwrap();
    }

    /// The resume line is handed to a human who will paste it. It must reproduce the request and
    /// must NOT carry the two flags that would destroy work: `--refresh-universe` rewrites
    /// `universe.json`, whose digest gates every checkpoint load, and `--force` restarts from zero.
    #[test]
    fn the_resume_command_reproduces_the_request_without_the_destructive_flags() {
        let args = IngestArgs {
            min_dollar_volume: 1e6,
            resolution: "5min".to_string(),
            years: 10,
            concurrency: 16,
            refresh_universe: true,
            train_end: None,
            min_bars: 20_480,
            universe_only: false,
            force: true,
            daily: false,
        };
        let command = resume_command(&args);
        assert!(command.contains("--years 10"), "{command}");
        assert!(command.contains("--resolution 5min"), "{command}");
        assert!(command.contains("--min-bars 20480"), "{command}");
        assert!(!command.contains("--refresh-universe"), "{command}");
        assert!(!command.contains("--force"), "{command}");

        let daily = resume_command(&IngestArgs {
            daily: true,
            ..args
        });
        assert!(daily.contains("--daily"), "{daily}");
        assert!(!daily.contains("--resolution"), "{daily}");
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
