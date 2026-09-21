//! Persisted, authenticated derivations of the five-minute bar corpus.
//!
//! Every run before this module rescanned all 470,946,393 bars four times before optimizer step
//! one - once to find the shared partition boundaries, once to authenticate and SHA-256 each
//! ticker, and twice more to build the market grid - for a measured 172 s of pure host work on
//! a 3,000-step probe whose stepping is 430 s. None of that work depends on anything but the
//! bar bytes and a handful of scalars, so all of it is cacheable; the only question is what
//! makes a cache hit SAFE.
//!
//! The answer is two different keys for two different layers, and the layering is the whole
//! design:
//!
//! - **Identity gates the fingerprint.** [`BarIdentity`] - device, inode, size, mtime and,
//!   decisively, ctime - decides whether a stored per-ticker [`BarAudit`] still describes the
//!   mapped bytes. Linux exposes no way to set a file's ctime: `utimensat` moves atime and
//!   mtime and bumps ctime, an atomic [`shared::bars::write_bar_file`] replaces the inode
//!   outright via rename, and [`shared::bars::append_bars`] grows the size and moves both
//!   stamps. A stale audit therefore cannot survive any writer this repository owns, nor
//!   `cp`, `rsync`, `truncate`, `dd`, or a hand edit. What it does NOT survive is raw
//!   block-device surgery or a backwards system clock, and that residual is stated here rather
//!   than hidden.
//! - **Content addresses everything else.** The bounds cache and the market-steps cache are
//!   keyed on the ticker SHA-256 fingerprints themselves, never on inode identity. So a corpus
//!   restored from backup with fresh inodes rehashes once and then hits every downstream
//!   layer, and two corpora with identical bytes share a cache entry no matter how they got
//!   there.
//!
//! Each layer additionally carries the schema strings and the scalars its values depend on, so
//! a `--pred-len` change invalidates the market grid it changes and leaves the bar audits it
//! does not touch alone. That split is deliberate: bar audits are the expensive layer and are
//! independent of every training knob.
//!
//! Cache files live in `<data_dir>/.timexer-cache/`, beside the corpus they describe, so a
//! copied corpus carries its cache and a test corpus in a temp directory is isolated for free.
//! Writes go through a temp file and a rename, so two runs starting at once cannot observe a
//! half-written entry.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use ring::digest::{Context as Digest, SHA256};
use serde::{Deserialize, Serialize};
use shared::bars::BarIdentity;

use super::data::BarAudit;
use super::features::SlotMoment;

/// Bumped whenever a cached value's MEANING changes without its inputs changing - a new field,
/// a different derivation, a fixed bug. Part of every key below, so an old entry is a miss
/// rather than a wrong answer.
const CACHE_VERSION: &str = "timexer-startup-cache-v1";

const CACHE_DIRECTORY: &str = ".timexer-cache";
const AUDITS_FILE: &str = "bar-audits.bin";
const BOUNDS_FILE: &str = "shared-bounds.bin";
const MARKET_FILE: &str = "market-steps.bin";
const RANKS_FILE: &str = "cross-section-ranks.bin";

fn cache_directory(data_dir: &Path) -> PathBuf {
    data_dir.join(CACHE_DIRECTORY)
}

/// Read a postcard artifact, treating an absent or unreadable file as a miss.
///
/// A DECODE failure is a miss too, not an error: the only way to reach one is a truncated
/// write from a killed process or an artifact from a build with a different layout, and both
/// are answered correctly by recomputing. Every semantic guard - schema, key, identity - runs
/// on the decoded value afterwards, so a miss is the only thing this can silently produce.
fn read_artifact<T: for<'a> Deserialize<'a>>(path: &Path) -> Option<T> {
    let bytes = fs::read(path).ok()?;
    postcard::from_bytes(&bytes).ok()
}

/// Write a postcard artifact atomically: a killed writer leaves either the previous complete
/// artifact or the new one, never a torn file that a later run would have to distrust.
fn write_artifact<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let directory = path.parent().context("cache path has no parent")?;
    fs::create_dir_all(directory)
        .with_context(|| format!("creating corpus cache directory {}", directory.display()))?;
    let bytes = postcard::to_stdvec(value).context("encoding a corpus cache artifact")?;
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temp, &bytes)
        .with_context(|| format!("writing corpus cache artifact {}", temp.display()))?;
    fs::rename(&temp, path)
        .with_context(|| format!("publishing corpus cache artifact {}", path.display()))?;
    Ok(())
}

fn hex(digest: ring::digest::Digest) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// One ticker's cached audit, with the inode state it was taken from.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AuditEntry {
    symbol: String,
    identity: BarIdentity,
    audit: BarAudit,
}

#[derive(Debug, Serialize, Deserialize)]
struct AuditLedger {
    version: String,
    entries: Vec<AuditEntry>,
}

/// The per-ticker [`BarAudit`] layer: the corpus's dominant startup cost, and the only layer
/// keyed on inode identity.
pub(super) struct AuditCache {
    path: PathBuf,
    version: String,
    entries: Vec<AuditEntry>,
    dirty: bool,
}

impl AuditCache {
    /// Load the ledger for a corpus directory. `schema` is the [`BarAudit`] derivation's own
    /// schema string; a ledger written under a different one is discarded wholesale.
    pub(super) fn open(data_dir: &Path, schema: &str) -> Self {
        let path = cache_directory(data_dir).join(AUDITS_FILE);
        let version = format!("{CACHE_VERSION};{schema}");
        let entries = read_artifact::<AuditLedger>(&path)
            .filter(|ledger| ledger.version == version)
            .map_or_else(Vec::new, |ledger| ledger.entries);
        Self {
            path,
            version,
            entries,
            dirty: false,
        }
    }

    /// Move the stored audit for `symbol` out of the ledger, if one was taken from exactly
    /// this inode state. Moving rather than borrowing keeps the caller from cloning a
    /// quarantine list per ticker; the ledger is rebuilt wholesale by
    /// [`AuditCache::replace`] before it is stored, so a taken entry is never lost.
    pub(super) fn take(&mut self, symbol: &str, identity: BarIdentity) -> Option<BarAudit> {
        let index = self
            .entries
            .binary_search_by(|entry| entry.symbol.as_str().cmp(symbol))
            .ok()?;
        (self.entries[index].identity == identity)
            .then(|| std::mem::take(&mut self.entries[index].audit))
    }

    /// Replace the ledger with exactly `universe`, in symbol order. Entries for tickers that
    /// left the corpus are dropped rather than accumulated, so the ledger cannot outgrow the
    /// corpus it describes.
    pub(super) fn replace(&mut self, universe: Vec<(String, BarIdentity, BarAudit)>) {
        let mut entries: Vec<AuditEntry> = universe
            .into_iter()
            .map(|(symbol, identity, audit)| AuditEntry {
                symbol,
                identity,
                audit,
            })
            .collect();
        entries.sort_unstable_by(|a, b| a.symbol.cmp(&b.symbol));
        self.dirty = entries.len() != self.entries.len()
            || entries.iter().zip(&self.entries).any(|(fresh, stored)| {
                fresh.symbol != stored.symbol || fresh.identity != stored.identity
            });
        self.entries = entries;
    }

    /// Publish the ledger if [`AuditCache::replace`] changed it. A pure hit writes nothing.
    pub(super) fn store(&self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        write_artifact(
            &self.path,
            &AuditLedger {
                version: self.version.clone(),
                entries: self.entries.clone(),
            },
        )
    }
}

/// SHA-256 over the content identity of an ordered ticker set: the fingerprints themselves,
/// never the inodes. Two byte-identical corpora produce the same key.
fn universe_key(schema: &str, universe: &[(&str, &str)], extra: &[u8]) -> String {
    let mut digest = Digest::new(&SHA256);
    digest.update(CACHE_VERSION.as_bytes());
    digest.update(b";");
    digest.update(schema.as_bytes());
    digest.update(&(universe.len() as u64).to_le_bytes());
    for (symbol, fingerprint) in universe {
        digest.update(symbol.as_bytes());
        digest.update(b":");
        digest.update(fingerprint.as_bytes());
        digest.update(b";");
    }
    digest.update(extra);
    hex(digest.finish())
}

#[derive(Debug, Serialize, Deserialize)]
struct BoundsArtifact {
    key: String,
    bounds: [i64; 3],
    edges: Vec<[u64; 3]>,
}

/// The shared chronological partition boundaries: three timestamps, derived from a full scan of
/// every valid bar's slot. Keyed on the whole scanned ticker set's fingerprints, because a
/// boundary is a quantile of the union and one added ticker can move it.
///
/// It carries each ticker's three RAW bar indices at those timestamps as well, in the universe's
/// own order. Those are `partition_point` probes - three binary searches per ticker, ~17 random
/// page faults each into 28 GB that no readahead predicts - and every consumer of the boundaries
/// needs them, so recomputing them from a cache hit was the single largest cost left in a warm
/// load. They are a function of the ticker's bytes and the boundaries alone, and both are already
/// inside `key`, so they cannot outlive what they describe.
pub(super) struct BoundsCache {
    path: PathBuf,
    key: String,
}

impl BoundsCache {
    pub(super) fn open(data_dir: &Path, schema: &str, universe: &[(&str, &str)]) -> Self {
        Self {
            path: cache_directory(data_dir).join(BOUNDS_FILE),
            key: universe_key(schema, universe, b"shared-bounds"),
        }
    }

    /// The boundaries and one edge triple per universe member, or `None`. A stored artifact whose
    /// edge count disagrees with the universe is refused rather than indexed into: the key makes
    /// that unreachable, and an out-of-bounds read is the wrong way to find out it did not.
    pub(super) fn get(&self, tickers: usize) -> Option<([i64; 3], Vec<[u64; 3]>)> {
        read_artifact::<BoundsArtifact>(&self.path)
            .filter(|stored| stored.key == self.key && stored.edges.len() == tickers)
            .map(|stored| (stored.bounds, stored.edges))
    }

    pub(super) fn store(&self, bounds: [i64; 3], edges: &[[u64; 3]]) -> Result<()> {
        write_artifact(
            &self.path,
            &BoundsArtifact {
                key: self.key.clone(),
                bounds,
                edges: edges.to_vec(),
            },
        )
    }
}

/// The per-slot vectors [`super::features::MarketSteps`] is made of.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct MarketGrid {
    pub first_ts: i64,
    pub min_cross_section: u32,
    pub population: Vec<u32>,
    /// Contributing log close returns, `ln(volume)` and `ln((high - low) / close)` over the
    /// same contributing set, in that order.
    pub returns: SlotMoment,
    pub log_volume: SlotMoment,
    pub log_range: SlotMoment,
}

#[derive(Debug, Serialize, Deserialize)]
struct MarketArtifact {
    key: String,
    grid: MarketGrid,
}

/// The equal-weighted market grid: two more full passes over every bar, over the ELIGIBLE
/// ticker set. That set depends on the split boundaries and the purge, so this key carries the
/// grid geometry and the cross-section floor as well as the fingerprints, and a `--pred-len`
/// change correctly misses here while hitting every bar audit.
pub(super) struct MarketCache {
    path: PathBuf,
    key: String,
}

/// The grid geometry and cross-section floor every derived market artifact is keyed on, after
/// its own tag. Shared so the market grid and the cross-section ranks cannot drift apart on
/// what "the same grid" means.
fn grid_key(tag: &[u8], first_ts: i64, slots: usize, min_cross_section: usize) -> Vec<u8> {
    let mut extra = Vec::with_capacity(tag.len() + 24);
    extra.extend_from_slice(tag);
    extra.extend_from_slice(&first_ts.to_le_bytes());
    extra.extend_from_slice(&(slots as u64).to_le_bytes());
    extra.extend_from_slice(&(min_cross_section as u64).to_le_bytes());
    extra
}

impl MarketCache {
    pub(super) fn open(
        data_dir: &Path,
        schema: &str,
        universe: &[(&str, &str)],
        first_ts: i64,
        slots: usize,
        min_cross_section: usize,
    ) -> Self {
        // The tag names the vector SET, not just the construction: an artifact written before
        // the per-slot volume and range moments existed decodes to nothing here rather than
        // being probed for fields it does not carry.
        let extra = grid_key(
            b"market-steps-return-volume-range-moments",
            first_ts,
            slots,
            min_cross_section,
        );
        Self {
            path: cache_directory(data_dir).join(MARKET_FILE),
            key: universe_key(schema, universe, &extra),
        }
    }

    /// The stored grid, rejected unless its key matches AND its vectors have the length this
    /// grid geometry implies - the second check is what stops a same-key artifact from a
    /// different build being indexed out of bounds at every bar lookup.
    pub(super) fn get(&self, slots: usize) -> Option<MarketGrid> {
        let stored = read_artifact::<MarketArtifact>(&self.path)?;
        let aligned = |moment: &SlotMoment| {
            moment.sums.len() == slots
                && moment.squares.len() == slots
                && moment.counts.len() == slots
        };
        (stored.key == self.key
            && stored.grid.population.len() == slots
            && aligned(&stored.grid.returns)
            && aligned(&stored.grid.log_volume)
            && aligned(&stored.grid.log_range))
        .then_some(stored.grid)
    }

    pub(super) fn store(&self, grid: &MarketGrid) -> Result<()> {
        write_artifact(
            &self.path,
            &MarketArtifact {
                key: self.key.clone(),
                grid: MarketGrid {
                    first_ts: grid.first_ts,
                    min_cross_section: grid.min_cross_section,
                    population: grid.population.clone(),
                    returns: grid.returns.clone(),
                    log_volume: grid.log_volume.clone(),
                    log_range: grid.log_range.clone(),
                },
            },
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RankArtifact {
    key: String,
    /// One entry per eligible ticker, in universe order: its doubled mid-ranks as raw
    /// little-endian `u16`. Bytes rather than `Vec<u16>` because postcard varint-encodes
    /// integers, and this is the largest artifact the cache holds by two orders of magnitude.
    ranks: Vec<Vec<u8>>,
}

/// [`super::features::cross_section_ranks`]: one `u16` per valid bar of every eligible ticker.
///
/// Its own file and its own tag, so an arm that only toggles the rank channel neither
/// invalidates the market grid nor is invalidated by it. Keyed on the same universe
/// fingerprints and grid scalars as the market grid, because that is exactly what it is a
/// function of.
pub(super) struct RankCache {
    path: PathBuf,
    key: String,
}

impl RankCache {
    pub(super) fn open(
        data_dir: &Path,
        schema: &str,
        universe: &[(&str, &str)],
        first_ts: i64,
        slots: usize,
        min_cross_section: usize,
    ) -> Self {
        let extra = grid_key(
            b"cross-section-doubled-mid-ranks",
            first_ts,
            slots,
            min_cross_section,
        );
        Self {
            path: cache_directory(data_dir).join(RANKS_FILE),
            key: universe_key(schema, universe, &extra),
        }
    }

    /// The stored ranks, rejected unless the key matches and every ticker's row is exactly as
    /// long as that ticker's valid bar count.
    pub(super) fn get(&self, valid_bars: &[usize]) -> Option<Vec<Vec<u16>>> {
        let stored = read_artifact::<RankArtifact>(&self.path)?;
        if stored.key != self.key || stored.ranks.len() != valid_bars.len() {
            return None;
        }
        if stored
            .ranks
            .iter()
            .zip(valid_bars)
            .any(|(row, bars)| row.len() != bars * 2)
        {
            return None;
        }
        Some(
            stored
                .ranks
                .iter()
                .map(|row| {
                    row.chunks_exact(2)
                        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                        .collect()
                })
                .collect(),
        )
    }

    pub(super) fn store(&self, ranks: &[Vec<u16>]) -> Result<()> {
        write_artifact(
            &self.path,
            &RankArtifact {
                key: self.key.clone(),
                ranks: ranks
                    .iter()
                    .map(|row| bytemuck::cast_slice::<u16, u8>(row).to_vec())
                    .collect(),
            },
        )
    }
}

/// Milliseconds spent in each phase of [`super::corpus::Corpus::load`], and how much of the
/// corpus this run had to touch to get there. Reported as series so a startup regression is a
/// line that moves rather than a run that feels slow.
///
/// A phase that DID NOT RUN reads NaN, never zero. The two are different facts - "this work
/// was unnecessary" against "this work took no time" - and a cached startup would otherwise
/// draw a floor of zeros that looks like a measurement rather than an absence. `Default` is
/// therefore all-NaN, and a phase becomes a number only when something timed it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoadTiming {
    pub directory_scan_ms: f64,
    /// Reading the ledger and checking every mapped inode against the identity its stored audit
    /// was taken from. Always runs, warm or cold - it is the step that decides which of the two
    /// this run is - so unlike the phases below it is never NaN.
    pub audit_ledger_ms: f64,
    /// Reading and decoding the boundary and market artifacts, plus the bookkeeping around them.
    /// Additive and always finite for the same reason as the ledger check: it is what a warm run
    /// pays INSTEAD of the rebuild phases, so leaving it NaN would hide the cost of the cache.
    pub cache_read_ms: f64,
    pub bar_audit_ms: f64,
    pub shared_bounds_ms: f64,
    pub eligibility_ms: f64,
    pub contract_ms: f64,
    pub market_grid_ms: f64,
    pub exogenous_ms: f64,
    pub origin_enumeration_ms: f64,
    /// Run startup outside the corpus, filled by the trainer before its first step: creating
    /// the CUDA context, writing the corpus contract and census, building the model, the
    /// optimizer and their device state, and drawing the two held-out origin sets. Graph
    /// capture is deliberately absent - it happens at step `CAPTURE_AFTER_STEPS`, not before
    /// step one, and `timexer_segment_capture` already reports it.
    pub cuda_context_ms: f64,
    pub corpus_report_ms: f64,
    pub model_build_ms: f64,
    pub held_out_draw_ms: f64,
    /// Applying the row-pool diversity knobs and censusing what they retained: the per-ticker
    /// stride, the per-row patch phase, the uniform subsample, the per-outcome multiplicity
    /// histogram and the retained timestamp profile. Charged separately because it is the one
    /// startup phase whose cost scales with the ROW COUNT rather than with the bar count, so a
    /// regression in it would otherwise hide inside the enumeration row.
    pub row_selection_ms: f64,
    /// Process start to the instant before the first optimizer step.
    pub startup_total_ms: f64,
    pub fingerprint_ms: f64,
    pub cache_write_ms: f64,
    pub total_ms: f64,
    /// Records actually traversed this run. This is the cold-versus-warm tell: a rebuild reads
    /// every one of the corpus's ~471 million records, a cache hit reads none, and the two runs
    /// differ by three orders of magnitude in a series rather than only in their totals.
    pub bars_rescanned: u64,
    pub audits_reused: usize,
    pub audits_computed: usize,
    pub bounds_reused: bool,
    pub market_reused: bool,
}

impl Default for LoadTiming {
    fn default() -> Self {
        Self {
            directory_scan_ms: f64::NAN,
            audit_ledger_ms: f64::NAN,
            cache_read_ms: 0.,
            bar_audit_ms: f64::NAN,
            shared_bounds_ms: f64::NAN,
            eligibility_ms: f64::NAN,
            contract_ms: f64::NAN,
            market_grid_ms: f64::NAN,
            exogenous_ms: f64::NAN,
            origin_enumeration_ms: f64::NAN,
            cuda_context_ms: f64::NAN,
            corpus_report_ms: f64::NAN,
            model_build_ms: f64::NAN,
            held_out_draw_ms: f64::NAN,
            row_selection_ms: f64::NAN,
            startup_total_ms: f64::NAN,
            fingerprint_ms: f64::NAN,
            cache_write_ms: f64::NAN,
            total_ms: f64::NAN,
            bars_rescanned: 0,
            audits_reused: 0,
            audits_computed: 0,
            bounds_reused: false,
            market_reused: false,
        }
    }
}

impl LoadTiming {
    /// Every phase as `(label, milliseconds)` in load order, total last, plus the rescan
    /// counter that separates a rebuild from a cache hit.
    pub fn phases(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("directory scan and header open", self.directory_scan_ms),
            (
                "bar ledger authentication (inode identity)",
                self.audit_ledger_ms,
            ),
            ("cached artifact read and decode", self.cache_read_ms),
            ("bar audit (grid checks and SHA-256)", self.bar_audit_ms),
            ("shared partition boundaries", self.shared_bounds_ms),
            ("ticker eligibility scan", self.eligibility_ms),
            ("per-ticker contracts", self.contract_ms),
            ("market grid", self.market_grid_ms),
            ("exogenous series and SPY", self.exogenous_ms),
            ("row and origin enumeration", self.origin_enumeration_ms),
            ("corpus fingerprints", self.fingerprint_ms),
            ("cache write", self.cache_write_ms),
            ("corpus load total", self.total_ms),
            ("CUDA context and device selection", self.cuda_context_ms),
            ("corpus contract and census reports", self.corpus_report_ms),
            (
                "model, optimizer and captured-step allocation",
                self.model_build_ms,
            ),
            ("held-out and cross-section draws", self.held_out_draw_ms),
            (
                "row selection and supervision census",
                self.row_selection_ms,
            ),
            ("pre-first-step startup total", self.startup_total_ms),
            (
                "bar records rescanned (millions)",
                self.bars_rescanned as f64 / 1e6,
            ),
        ]
    }

    /// The startup line the trainer prints, one phase per row. `-` marks a phase that did not
    /// run at all, which is what a warm cache looks like.
    pub fn summary(&self) -> String {
        let mut text = String::from("CausalPatch corpus load, milliseconds per startup phase:");
        for (label, ms) in self.phases() {
            if ms.is_nan() {
                text.push_str(&format!("\n  {label:<52} {:>12}", "-"));
            } else {
                text.push_str(&format!("\n  {label:<52} {ms:>12.1}"));
            }
        }
        text.push_str(&format!(
            "\n  {:<52} {:>12}",
            "bar audits reused from the ledger", self.audits_reused
        ));
        text.push_str(&format!(
            "\n  {:<52} {:>12}",
            "bar audits recomputed", self.audits_computed
        ));
        text.push_str(&format!(
            "\n  {:<52} {:>12}",
            "partition boundaries reused", self.bounds_reused
        ));
        text.push_str(&format!(
            "\n  {:<52} {:>12}",
            "market grid reused", self.market_reused
        ));
        text
    }
}

/// Milliseconds since this PROCESS started, not since the caller's `Instant`.
///
/// Startup that an operator waits through begins at `execve`, and a good third of a second of it
/// is over before `main` runs: the dynamic linker maps libtorch and its CUDA kernels, and the
/// embedded interpreter initialises. An `Instant` taken inside `train` cannot see any of that
/// and would quietly under-report the number the target is stated against.
///
/// Field 22 of `/proc/self/stat` is the process's start time in clock ticks since boot, and
/// `/proc/uptime` is now; the difference is the process's age. Unreadable `/proc` yields `None`
/// rather than a fabricated zero.
pub fn process_elapsed_ms() -> Option<f64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    // The second field is the executable name in parentheses and may itself contain spaces, so
    // fields are counted from the last ')' rather than from the start of the line.
    let tail = &stat[stat.rfind(')')? + 1..];
    let started_ticks: f64 = tail.split_whitespace().nth(19)?.parse().ok()?;
    let uptime = fs::read_to_string("/proc/uptime").ok()?;
    let now_seconds: f64 = uptime.split_whitespace().next()?.parse().ok()?;
    let ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    (ticks_per_second > 0.).then(|| (now_seconds - started_ticks / ticks_per_second) * 1000.)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "timexer-startup-cache-{label}-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn identity(change_ns: i128) -> BarIdentity {
        BarIdentity {
            device: 1,
            inode: 2,
            size: 3,
            modified_ns: 4,
            change_ns,
        }
    }

    fn audit(fingerprint: &str) -> BarAudit {
        BarAudit {
            fingerprint: fingerprint.into(),
            source_bars: 10,
            invalid_ohlc_indices: vec![1],
        }
    }

    #[test]
    fn a_bar_audit_is_rejected_when_the_inode_state_moves() {
        let directory = scratch("audit");
        let mut cache = AuditCache::open(&directory, "schema-a");
        cache.replace(vec![("AAA".into(), identity(9), audit("ff"))]);
        cache.store().unwrap();

        let mut reopened = AuditCache::open(&directory, "schema-a");
        assert!(
            reopened.take("AAA", identity(10)).is_none(),
            "a ctime move must miss"
        );
        assert_eq!(
            reopened.take("AAA", identity(9)).map(|a| a.fingerprint),
            Some("ff".into()),
            "an unchanged inode must hit"
        );
        assert!(
            AuditCache::open(&directory, "schema-b")
                .take("AAA", identity(9))
                .is_none(),
            "a derivation schema change must discard the whole ledger"
        );
    }

    /// The justification for caching at all: a cache built against one corpus contract must be
    /// REFUSED when the contract changes, not silently reused. Both downstream layers are keyed
    /// on content, so this covers the case identity cannot - the same files, described by a
    /// different contract.
    #[test]
    fn downstream_caches_are_rejected_when_the_corpus_contract_changes() {
        let directory = scratch("contract");
        let universe = [("AAA", "ff"), ("BBB", "ee")];

        let bounds = BoundsCache::open(&directory, "schema-a", &universe);
        bounds.store([1, 2, 3], &[[0, 1, 2], [3, 4, 5]]).unwrap();
        assert_eq!(
            BoundsCache::open(&directory, "schema-a", &universe).get(2),
            Some(([1, 2, 3], vec![[0, 1, 2], [3, 4, 5]]))
        );
        assert_eq!(
            BoundsCache::open(&directory, "schema-b", &universe).get(2),
            None,
            "a corpus schema change must reject the stored boundaries"
        );
        assert_eq!(
            BoundsCache::open(&directory, "schema-a", &[("AAA", "ff")]).get(1),
            None,
            "dropping a ticker must reject the stored boundaries"
        );
        assert_eq!(
            BoundsCache::open(&directory, "schema-a", &[("AAA", "ff"), ("BBB", "00")]).get(2),
            None,
            "a changed ticker fingerprint must reject the stored boundaries"
        );
        assert_eq!(
            BoundsCache::open(&directory, "schema-a", &universe).get(3),
            None,
            "an edge list that does not cover the universe must be refused, not indexed into"
        );

        let moment = |sums: Vec<f64>, squares: Vec<f64>| SlotMoment {
            sums,
            squares,
            counts: vec![1, 1],
        };
        let grid = MarketGrid {
            first_ts: 0,
            min_cross_section: 20,
            population: vec![1, 2],
            returns: moment(vec![0.5, 0.25], vec![0.25, 0.0625]),
            log_volume: moment(vec![7.0, 8.0], vec![49.0, 64.0]),
            log_range: moment(vec![-6.0, -5.5], vec![36.0, 30.25]),
        };
        let market = MarketCache::open(&directory, "schema-a", &universe, 0, 2, 20);
        market.store(&grid).unwrap();
        assert!(
            MarketCache::open(&directory, "schema-a", &universe, 0, 2, 20)
                .get(2)
                .is_some()
        );
        assert!(
            MarketCache::open(&directory, "schema-a", &universe, 0, 2, 40)
                .get(2)
                .is_none(),
            "a different cross-section floor must reject the stored market grid"
        );
        assert!(
            MarketCache::open(&directory, "schema-a", &universe, 300_000, 2, 20)
                .get(2)
                .is_none(),
            "a shifted grid origin must reject the stored market grid"
        );
        assert!(
            MarketCache::open(&directory, "schema-a", &universe, 0, 3, 20)
                .get(3)
                .is_none(),
            "a grid of a different length must reject the stored market grid"
        );
        let ranks = RankCache::open(&directory, "schema-a", &universe, 0, 2, 20);
        ranks.store(&[vec![2u16, 0, 7], vec![4]]).unwrap();
        assert_eq!(
            ranks.get(&[3, 1]),
            Some(vec![vec![2u16, 0, 7], vec![4]]),
            "the doubled mid-ranks must round-trip through the artifact unchanged"
        );
        assert_eq!(
            ranks.get(&[3, 2]),
            None,
            "a ticker whose valid bar count moved must reject the stored ranks"
        );
        assert_eq!(
            RankCache::open(&directory, "schema-a", &universe, 0, 2, 40).get(&[3, 1]),
            None,
            "a different cross-section floor must reject the stored ranks"
        );
        // The two artifacts are keyed independently: the rank pass may miss while the market
        // grid it was derived from still hits, and vice versa.
        assert!(
            MarketCache::open(&directory, "schema-a", &universe, 0, 2, 20)
                .get(2)
                .is_some()
        );
    }

    #[test]
    fn a_torn_artifact_is_a_miss_and_not_a_failure() {
        let directory = scratch("torn");
        let path = cache_directory(&directory);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join(BOUNDS_FILE), b"not postcard at all").unwrap();
        assert_eq!(
            BoundsCache::open(&directory, "schema-a", &[("AAA", "ff")]).get(1),
            None
        );
    }
}
