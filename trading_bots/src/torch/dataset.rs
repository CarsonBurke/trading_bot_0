//! Corpus-scale bar dataset: mmap-backed symbol files, one global calendar split, and a
//! deterministic near-disjoint window sampler that emits bar degrees of freedom.
//!
//! Three properties this module exists to guarantee:
//!
//! * **No cross-symbol leak.** The train/val/test cut is two wall-clock instants shared by
//!   every symbol, placed at the 80th and 90th percentile of the *global trading-time axis*
//!   (bars, not calendar days, so holidays and listing dates cannot skew it). The previous
//!   per-symbol array-index split let one ticker's test window sit inside another ticker's
//!   train window at the same instant, which is a straightforward lookahead leak.
//! * **One epoch is one pass worth of BAR-TOKENS.** Anchors are strided by exactly `context`,
//!   so windows are near-disjoint (they share a single seam bar) instead of overlapping ~180x,
//!   and an epoch is sized to consume one corpus's worth of bar-tokens. It is NOT a guaranteed
//!   pass over every unique bar: the pretrainer's context ramp gives each stage its own anchor
//!   list and splits the token budget unevenly across them, so an early stage covers only a
//!   fraction of its list. `pretrain_stage_coverage` charts the fraction actually issued.
//! * **Bit-reproducible batches.** Ordering comes from a counter-based ChaCha stream keyed by
//!   `(seed, epoch)`, never from a thread RNG, so `(seed, epoch, index)` always yields the
//!   same tensor.
//!
//! DOF are computed lazily from the mmap'd bars rather than from a materialized on-disk DOF
//! cache. The corpus is live — `Ingest` appends to these files — so a precomputed cache is
//! stale by construction, would duplicate ~13 GB, and buys nothing: encoding is ~0.4 ms of
//! arithmetic for a `[64, 2049, 5]` batch once the pages are resident, which is far below the
//! page-in cost that both designs pay equally.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{bail, Context, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use ring::digest::{Context as DigestContext, SHA256};
use shared::bars::{parse_bar_file_name, BarFile, PackedBar, FILE_EXTENSION};
use shared::report::{Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Tensor};

use crate::torch::bar_dist::{encode_dof, BarDof, BarSupports, VolumeEma, BAR_DOF};

/// Causal bars fed to the volume EMA before the first emitted DOF of a window. The span-20
/// EMA retains `(1 - 2/21)^256 ~ 7e-12` of its seed after this many observations, so a
/// window's DOF are numerically indistinguishable from encoding the symbol's whole series
/// with [`crate::torch::bar_dist::encode_series`] and slicing. The warm-up is strictly causal
/// past and may reach back across a split boundary; the DOF-carrying bars never do.
pub const DOF_WARMUP_BARS: usize = 256;

/// Share of the global trading-time axis reserved for training, then for validation.
pub const TRAIN_FRACTION: f64 = 0.80;
pub const VAL_FRACTION: f64 = 0.10;

/// Bars a symbol needs to enter a corpus, and so to count toward the split percentiles.
///
/// The default guarantees every admitted symbol contributes at least one full-context window to
/// each split. It lives here rather than in the pretrain CLI because corpus ingestion must derive
/// the `train | val` instant from the same eligibility rule the pretrainer applies: dropping a file
/// moves the trading-time percentile, so a mismatch would put the universe and the split at odds.
pub const DEFAULT_MIN_BARS: usize = 20_480;

/// Consecutive DOF drawn per support-fitting block. Sampling in blocks amortizes the
/// [`DOF_WARMUP_BARS`] prefix over 64 usable samples, turning support fitting from an
/// `O(corpus)` read into an `O(max_samples)` one.
const SUPPORT_BLOCK: usize = 64;

/// RNG stream ids. Epoch shuffles use the epoch number itself as the stream, so every other
/// draw takes an id far outside any plausible epoch and can never collide with one.
const PINNED_STREAM: u64 = 0xE7A1_0000_0000_0001;
const SUPPORT_STREAM: u64 = 0xE7A1_0000_0000_0002;
const SUPPORT_ORDER_STREAM: u64 = 0xE7A1_0000_0000_0003;

// ---------------------------------------------------------------------------
// Exogenous calendar conditioning
//
// Bar dynamics are dominated by the clock: the open/close volatility smile, the pre/post
// liquidity cliff, and the fact that ~42% of extended-hours bars carry no intra-bar shape at
// all. Without these ids the trunk has to average over regimes that behave nothing alike.
// They are exogenous — known for every future bar — so they are always an input, never a
// prediction target.
// ---------------------------------------------------------------------------

pub const BAR_TIME_FEATURES: usize = 4;
/// Raw ET minute of the bar's open timestamp. Deliberately not a resolution-relative bucket:
/// the same integer must mean the same wall-clock instant in a 5-minute and a daily corpus,
/// otherwise one embedding table cannot serve both.
pub const TIME_MINUTE: usize = 0;
/// ET calendar weekday, Monday 0. Seven rows, not five: a holiday artefact or a corrupt
/// weekend bar must not blow an embedding index bound.
pub const TIME_WEEKDAY: usize = 1;
/// 0 overnight, 1 pre-market, 2 regular, 3 post-market, by ET wall clock.
pub const TIME_SESSION: usize = 2;
/// Resolution *class* id, not `res_secs`, so a merged multi-resolution corpus indexes a small
/// dense table.
pub const TIME_RESOLUTION: usize = 3;

pub const BAR_TIME_CARDINALITY: [i64; BAR_TIME_FEATURES] = [1440, 7, 4, 8];
pub const BAR_TIME_NAMES: [&str; BAR_TIME_FEATURES] =
    ["minute", "weekday", "session", "resolution"];
/// Lineage tag for the checkpoint metadata. Bump it whenever any id's meaning moves.
pub const BAR_TIME_CONDITIONING: &str = "et-minute-weekday-session4-resclass-v1";
/// ET minutes at which [`TIME_SESSION`] changes: 04:00, 09:30, 16:00, 20:00.
pub const SESSION_BOUNDARY_MINUTES: [i64; 4] = [240, 570, 960, 1200];
/// Resolutions with a dedicated class id; the index is the id. Anything else maps to
/// [`RESOLUTION_CLASS_OTHER`].
pub const RESOLUTION_CLASS_SECS: [u32; 7] = [60, 300, 900, 1800, 3600, 14_400, 86_400];
pub const RESOLUTION_CLASS_OTHER: i64 = 7;

/// Bar tensors that must never be handed out separately: the DOF the model predicts and the
/// calendar ids it conditions on, drawn from the same bars in the same order.
#[derive(Debug)]
pub struct BarBatch {
    /// `[N, L, BAR_DOF]` f32.
    pub dof: Tensor,
    /// `[N, L, BAR_TIME_FEATURES]` i64.
    pub time_ids: Tensor,
}

pub fn resolution_class(res_secs: u32) -> i64 {
    RESOLUTION_CLASS_SECS
        .iter()
        .position(|&secs| secs == res_secs)
        .map_or(RESOLUTION_CLASS_OTHER, |index| index as i64)
}

/// Calendar ids of a bar opening at `ts_ms`, in America/New_York with real DST.
///
/// Total in every id by construction — `minute` comes from a value reduced mod 86400, the
/// weekday from `rem_euclid(7)`, the session from an exhaustive branch and the resolution from
/// a lookup with an `other` fallback — so no timestamp, however corrupt, can produce an index
/// outside [`BAR_TIME_CARDINALITY`].
pub fn bar_time_ids(ts_ms: i64, res_secs: u32) -> [i64; BAR_TIME_FEATURES] {
    let utc = ts_ms.div_euclid(1000);
    let local = utc + et_offset_secs(utc) as i64;
    let day = local.div_euclid(SECS_PER_DAY);
    let minute = local.rem_euclid(SECS_PER_DAY) / 60;
    // 1970-01-01 was a Thursday, which is index 3 with Monday at 0.
    let weekday = (day + 3).rem_euclid(7);
    let [pre, regular, post, close] = SESSION_BOUNDARY_MINUTES;
    let session = if minute < pre || minute >= close {
        0
    } else if minute < regular {
        1
    } else if minute < post {
        2
    } else {
        3
    };
    let ids = [minute, weekday, session, resolution_class(res_secs)];
    debug_assert!(
        ids.iter()
            .zip(BAR_TIME_CARDINALITY)
            .all(|(&id, cardinality)| (0..cardinality).contains(&id)),
        "calendar ids {ids:?} escaped {BAR_TIME_CARDINALITY:?} for ts_ms {ts_ms}"
    );
    ids
}

const SECS_PER_DAY: i64 = 86_400;
/// 1990-01-01T00:00:00Z and 2100-01-01T00:00:00Z, the span the offset table covers.
const ET_TABLE_FROM: i64 = 631_152_000;
const ET_TABLE_TO: i64 = 4_102_444_800;

/// America/New_York UTC-offset transitions as `(start_utc_secs, offset_secs)`, sorted.
///
/// Built once by walking chrono-tz a day at a time and bisecting each change to the second.
/// Two hundred-odd entries, so the per-bar lookup is a binary search over a cache-resident
/// array rather than a tz-database query — the difference between ~5 ns and ~100 ns per bar,
/// which matters at 131k bars per batch.
static ET_TRANSITIONS: std::sync::LazyLock<(Vec<i64>, Vec<i32>)> =
    std::sync::LazyLock::new(build_et_transitions);

fn et_offset_at(utc_secs: i64) -> i32 {
    use chrono::Offset;
    use chrono::TimeZone;
    let naive = chrono::DateTime::from_timestamp(utc_secs, 0)
        .expect("offset table timestamps are in range")
        .naive_utc();
    chrono_tz::America::New_York
        .offset_from_utc_datetime(&naive)
        .fix()
        .local_minus_utc()
}

fn build_et_transitions() -> (Vec<i64>, Vec<i32>) {
    let mut starts = vec![ET_TABLE_FROM];
    let mut offsets = vec![et_offset_at(ET_TABLE_FROM)];
    let mut previous = offsets[0];
    let mut day = ET_TABLE_FROM;
    while day < ET_TABLE_TO {
        let next = (day + SECS_PER_DAY).min(ET_TABLE_TO);
        let offset = et_offset_at(next);
        if offset != previous {
            // Exactly one change inside a transition day, so a plain bisection finds it.
            let (mut lo, mut hi) = (day, next);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if et_offset_at(mid) == previous {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            starts.push(lo);
            offsets.push(offset);
            previous = offset;
        }
        day = next;
    }
    (starts, offsets)
}

fn et_offset_secs(utc_secs: i64) -> i32 {
    let (starts, offsets) = &*ET_TRANSITIONS;
    let index = starts.partition_point(|&start| start <= utc_secs);
    offsets[index.saturating_sub(1)]
}

/// Which region of the global timeline a sampler draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    pub const ALL: [Split; 3] = [Split::Train, Split::Val, Split::Test];

    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single training window: the symbol it comes from and the bar index of its first
/// DOF-carrying bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowRef {
    pub symbol: u32,
    pub bar_index: u32,
}

/// A single bar addressed absolutely, for consumers that step a timeline rather than draw
/// fixed-length training windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BarEndpoint {
    pub series: usize,
    pub bar: usize,
}

struct Corpus {
    dir: PathBuf,
    res_secs: u32,
    files: Vec<BarFile>,
    symbols: Vec<String>,
    total_bars: usize,
    bounds: (i64, i64),
}

/// All `<dir>/*.<res_secs>.bars` files, held open and mmap'd. Cloning is an `Arc` bump, so
/// samplers own their view of the corpus and carry no lifetime.
#[derive(Clone)]
pub struct BarCorpus {
    inner: Arc<Corpus>,
}

impl std::fmt::Debug for BarCorpus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarCorpus")
            .field("dir", &self.inner.dir)
            .field("res_secs", &self.inner.res_secs)
            .field("symbols", &self.inner.symbols.len())
            .field("total_bars", &self.inner.total_bars)
            .field("bounds", &self.inner.bounds)
            .finish()
    }
}

impl BarCorpus {
    /// Open every symbol file at `res_secs` under `dir`, dropping (and reporting) symbols with
    /// fewer than `min_bars` bars, and place the split at this resolution's own trading-time
    /// percentiles. Files stay mmap'd and are paged in on demand; the corpus is never read
    /// into RAM.
    ///
    /// One corpus is one resolution. `<dir>` holds `.300.bars` and `.86400.bars` side by side
    /// and they interleave alphabetically, so the `res_secs` filter is load-bearing: a daily
    /// bar mixed into a 5-minute corpus would be a legitimate 4x move against a threshold
    /// tuned for five minutes, and would land on a support fitted for the wrong scale.
    pub fn load(dir: &Path, res_secs: u32, min_bars: usize) -> Result<Self> {
        Self::open_files(dir, res_secs, min_bars, None, None)
    }

    /// As [`Self::load`], but with the split instants supplied rather than derived — the way
    /// an auxiliary resolution joins a run without moving the boundary, and the way a
    /// campaign pins the boundary against a corpus that grows under it.
    pub fn load_with_bounds(
        dir: &Path,
        res_secs: u32,
        min_bars: usize,
        bounds: (i64, i64),
    ) -> Result<Self> {
        Self::open_files(dir, res_secs, min_bars, Some(bounds), None)
    }

    /// As [`Self::load`], keeping only `symbols`.
    ///
    /// The split instants are derived from the FULL symbol set and only then is the
    /// restriction applied, which is what makes a symbol-universe ablation interpretable:
    /// both arms are scored over the same wall-clock held-out window, so their `nll_bar` are
    /// commensurable. Deriving the percentiles after the restriction would move the boundary
    /// with the symbol set and confound the two effects. `bounds` overrides the derivation
    /// entirely, as in [`Self::load_with_bounds`].
    pub fn load_restricted(
        dir: &Path,
        res_secs: u32,
        min_bars: usize,
        bounds: Option<(i64, i64)>,
        symbols: &HashSet<String>,
    ) -> Result<Self> {
        Self::open_files(dir, res_secs, min_bars, bounds, Some(symbols))
    }

    fn open_files(
        dir: &Path,
        res_secs: u32,
        min_bars: usize,
        bounds: Option<(i64, i64)>,
        keep: Option<&HashSet<String>>,
    ) -> Result<Self> {
        let mut paths = corpus_paths(dir, res_secs)?;
        paths.sort();
        if paths.is_empty() {
            bail!(
                "no *.{res_secs}.{FILE_EXTENSION} files under {}",
                dir.display()
            );
        }

        let opened = paths
            .par_iter()
            .map(|path| BarFile::open(path))
            .collect::<Result<Vec<_>>>()?;

        let mut files = Vec::with_capacity(opened.len());
        let mut dropped = 0usize;
        for file in opened {
            if file.len() < min_bars {
                println!(
                    "[dataset] dropping {}.{res_secs}: {} bars < min_bars {min_bars}",
                    file.symbol(),
                    file.len()
                );
                dropped += 1;
                continue;
            }
            files.push(file);
        }
        if files.is_empty() {
            bail!(
                "every one of {} symbol files at {res_secs}s under {} has fewer than {min_bars} bars",
                paths.len(),
                dir.display()
            );
        }

        let total_bars_before_restriction = files.iter().map(BarFile::len).sum();
        // Bounds first, restriction second: see `load_restricted`.
        let bounds = bounds.unwrap_or_else(|| {
            global_split_bounds(&files, res_secs, total_bars_before_restriction)
        });
        let mut restricted = 0usize;
        if let Some(keep) = keep {
            let before = files.len();
            files.retain(|file| keep.contains(file.symbol()));
            restricted = before - files.len();
            if files.is_empty() {
                bail!(
                    "the symbol restriction kept none of the {before} symbol files at \
                     {res_secs}s under {}",
                    dir.display()
                );
            }
        }

        let symbols: Vec<String> = files.iter().map(|f| f.symbol().to_string()).collect();
        let total_bars = files.iter().map(BarFile::len).sum();
        println!(
            "[dataset] {} symbols, {total_bars} bars at {res_secs}s ({dropped} dropped, \
             {restricted} filtered out), split at {} | {}",
            files.len(),
            iso_ms(bounds.0),
            iso_ms(bounds.1)
        );

        Ok(Self {
            inner: Arc::new(Corpus {
                dir: dir.to_path_buf(),
                res_secs,
                files,
                symbols,
                total_bars,
                bounds,
            }),
        })
    }

    /// This corpus's [`TIME_RESOLUTION`] id.
    pub fn resolution_class(&self) -> i64 {
        resolution_class(self.inner.res_secs)
    }

    /// The two shared wall-clock instants, in epoch millis: `train | val` and `val | test`.
    /// A bar belongs to train when `ts < .0`, to val when `.0 <= ts < .1`, else to test.
    pub fn split_bounds(&self) -> (i64, i64) {
        self.inner.bounds
    }

    /// Total bars held by the corpus. Windows are strided so that one epoch touches each of
    /// these at most once.
    pub fn unique_bars(&self) -> usize {
        self.inner.total_bars
    }

    pub fn symbols(&self) -> &[String] {
        &self.inner.symbols
    }

    pub fn res_secs(&self) -> u32 {
        self.inner.res_secs
    }

    pub fn dir(&self) -> &Path {
        &self.inner.dir
    }

    pub fn bars(&self, symbol: usize) -> &[PackedBar] {
        self.inner.files[symbol].bars()
    }

    pub fn series_count(&self) -> usize {
        self.inner.files.len()
    }

    pub fn symbol(&self, series: usize) -> &str {
        &self.inner.symbols[series]
    }

    pub fn series_len(&self, series: usize) -> usize {
        self.inner.files[series].len()
    }

    /// Raw close of one bar. Portfolio marking uses prices, not DOF.
    pub fn close(&self, series: usize, bar: usize) -> f32 {
        self.inner.files[series].bars()[bar].close
    }

    pub fn ts_ms(&self, series: usize, bar: usize) -> i64 {
        self.inner.files[series].bars()[bar].ts()
    }

    /// `[endpoints.len(), len, ..]` DOF and calendar ids on `device`. Row `i` covers bars
    /// `bar + offset - len + 1 ..= bar + offset` of its endpoint's series, where `offsets` is
    /// either a single broadcast value or one entry per endpoint.
    ///
    /// Every row is encoded with the same causal span-20 volume EMA warm-up the pretrainer
    /// uses, so a `len == 1` request still pays [`DOF_WARMUP_BARS`] encodes: ask for the whole
    /// contiguous run at once when stepping a rollout.
    pub fn dof_window(
        &self,
        endpoints: &[BarEndpoint],
        offsets: &[usize],
        len: i64,
        device: Device,
    ) -> Result<BarBatch> {
        let len = self.check_window(endpoints, offsets.len(), len)?;
        let rows = endpoints
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let end = e.bar + offsets[if offsets.len() == 1 { 0 } else { i }];
                let start = self.window_start(e, end, len)?;
                Ok((e.series, start))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(build_batch(
            &self.inner.files,
            &rows,
            len,
            self.inner.res_secs,
            device,
        ))
    }

    /// `[endpoints.len(), steps, BAR_TIME_FEATURES]` i64: the calendar of the next `steps`
    /// **real** bars after `bar + from_offset`.
    ///
    /// A rollout's future clock cannot be extrapolated as `last_ts + k * res_secs` — weekends,
    /// holidays and the 20:00 -> 04:00 gap all break that — so it is read off the corpus.
    pub fn future_time_ids(
        &self,
        endpoints: &[BarEndpoint],
        from_offset: usize,
        steps: i64,
        device: Device,
    ) -> Result<Tensor> {
        if endpoints.is_empty() {
            bail!("future_time_ids needs at least one endpoint");
        }
        if steps <= 0 {
            bail!("future_time_ids steps must be positive, got {steps}");
        }
        let steps = steps as usize;
        let row = steps * BAR_TIME_FEATURES;
        let mut flat = vec![0i64; endpoints.len() * row];
        for (out, e) in flat.chunks_mut(row).zip(endpoints) {
            let bars = self.inner.files[e.series].bars();
            let first = e.bar + from_offset + 1;
            if first + steps > bars.len() {
                bail!(
                    "future_time_ids wants bars {first}..{} of {} but {} has {}",
                    first + steps,
                    self.symbol(e.series),
                    self.symbol(e.series),
                    bars.len()
                );
            }
            for (slot, bar) in bars[first..first + steps].iter().enumerate() {
                out[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES]
                    .copy_from_slice(&bar_time_ids(bar.ts(), self.inner.res_secs));
            }
        }
        Ok(Tensor::from_slice(&flat)
            .view([
                endpoints.len() as i64,
                steps as i64,
                BAR_TIME_FEATURES as i64,
            ])
            .to_device(device))
    }

    fn check_window(&self, endpoints: &[BarEndpoint], offsets: usize, len: i64) -> Result<usize> {
        if endpoints.is_empty() {
            bail!("dof_window needs at least one endpoint");
        }
        if len <= 0 {
            bail!("dof_window length must be positive, got {len}");
        }
        if offsets != 1 && offsets != endpoints.len() {
            bail!(
                "dof_window offsets must be one broadcast value or {} values, got {offsets}",
                endpoints.len()
            );
        }
        Ok(len as usize)
    }

    fn window_start(&self, e: &BarEndpoint, end: usize, len: usize) -> Result<usize> {
        let series_len = self.series_len(e.series);
        if end >= series_len {
            bail!(
                "dof_window endpoint {} of {} ends at bar {end} but the series has {series_len}",
                e.bar,
                self.symbol(e.series)
            );
        }
        (end + 1).checked_sub(len).filter(|&s| s >= 1).with_context(|| {
            format!(
                "dof_window of {len} bars ending at {end} needs a predecessor close in {}",
                self.symbol(e.series)
            )
        })
    }

    /// Count, per symbol, the bars whose log return or log range exceeds
    /// [`ANOMALY_LOG_LIMIT`]. Counts only — nothing is dropped or winsorized, because breaking
    /// series continuity is worse than the anomaly and the right remedy depends on the cause.
    ///
    /// Reads the whole corpus, so expect it to be I/O bound the first time.
    pub fn scan_anomalies(&self) -> CorpusAnomalies {
        let mut per_symbol: Vec<SymbolAnomalies> =
            self.inner.files.par_iter().map(scan_symbol).collect();
        per_symbol.sort_unstable_by(|a, b| {
            b.anomaly_rate()
                .total_cmp(&a.anomaly_rate())
                .then_with(|| a.symbol.cmp(&b.symbol))
        });
        CorpusAnomalies {
            res_secs: self.inner.res_secs,
            limit: ANOMALY_LOG_LIMIT,
            hole_ms: ANOMALY_HOLE_DAYS * 86_400_000,
            bars: per_symbol.iter().map(|s| s.bars).sum(),
            splices: per_symbol.iter().map(|s| s.splices).sum(),
            ticks: per_symbol.iter().map(|s| s.ticks).sum(),
            jumps: per_symbol.iter().map(|s| s.jumps).sum(),
            extreme_range: per_symbol.iter().map(|s| s.extreme_range).sum(),
            holes: per_symbol.iter().map(|s| s.holes).sum(),
            per_symbol,
        }
    }

    /// SHA-256 over everything that decides which bars a split contains: the resolution, both
    /// split instants, and every symbol's name, length and timestamp span. Fold this into any
    /// evaluation fingerprint — the corpus grows under running jobs, and a fingerprint blind to
    /// the symbol set would compare two different evaluation sets as if they were one.
    pub fn identity_fingerprint(&self) -> String {
        let mut digest = DigestContext::new(&SHA256);
        digest.update(b"bar-corpus-v1");
        digest.update(&self.inner.res_secs.to_le_bytes());
        digest.update(&self.inner.bounds.0.to_le_bytes());
        digest.update(&self.inner.bounds.1.to_le_bytes());
        for file in &self.inner.files {
            digest.update(file.symbol().as_bytes());
            digest.update(&(file.len() as u64).to_le_bytes());
            digest.update(&file.first_ts_ms().unwrap_or(0).to_le_bytes());
            digest.update(&file.last_ts_ms().unwrap_or(0).to_le_bytes());
        }
        digest
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// Half-open bar-index range `[lo, hi)` of `symbol` inside `split`.
    pub fn split_range(&self, symbol: usize, split: Split) -> (usize, usize) {
        self.inner.split_range(symbol, split)
    }

    /// Bars belonging to `split` across the whole corpus.
    pub fn split_bars(&self, split: Split) -> usize {
        (0..self.inner.files.len())
            .into_par_iter()
            .map(|s| {
                let (lo, hi) = self.inner.split_range(s, split);
                hi - lo
            })
            .sum()
    }

    /// Where [`Self::fit_supports`] persists its result.
    pub fn supports_path(&self) -> PathBuf {
        self.inner
            .dir
            .join(format!("bar_supports.{}.json", self.inner.res_secs))
    }

    /// Fit equal-mass supports from the train region only, and persist them next to the
    /// corpus. Sampling never touches a bar at or after the `train | val` bound, so no
    /// normalization statistic can leak out of validation or test.
    pub fn fit_supports(&self, max_samples: usize, seed: u64) -> BarSupports {
        let samples: Vec<BarDof> = self
            .sample_train_dof(max_samples, seed)
            .into_iter()
            .map(|(_, dof)| dof)
            .collect();
        let supports = BarSupports::fit(&samples);
        let path = self.supports_path();
        match supports.save(&path) {
            Ok(()) => println!(
                "[dataset] fitted bar supports from {} train DOF -> {}",
                samples.len(),
                path.display()
            ),
            Err(error) => eprintln!(
                "[dataset] fitted bar supports from {} train DOF but could not write {}: {error:#}",
                samples.len(),
                path.display()
            ),
        }
        supports
    }

    /// Deterministically draw up to `max_samples` `(ts_ms, dof)` pairs from the train region,
    /// as a uniform sample without replacement over length-[`SUPPORT_BLOCK`] blocks.
    ///
    /// Blocks rather than individual bars because every DOF needs a [`DOF_WARMUP_BARS`]
    /// causal prefix; a block amortizes that prefix over 64 samples while keeping the draw
    /// uniform over the train timeline.
    pub fn sample_train_dof(&self, max_samples: usize, seed: u64) -> Vec<(i64, BarDof)> {
        assert!(max_samples > 0, "support fitting needs a positive budget");
        let inner = &self.inner;

        // Block anchors per symbol: `1 + k * SUPPORT_BLOCK` while the whole block stays inside
        // the train region. Index 0 is excluded because the first bar of a file has no
        // predecessor and therefore no DOF.
        let mut cumulative = Vec::with_capacity(inner.files.len() + 1);
        let mut total_blocks: u64 = 0;
        cumulative.push(0u64);
        for s in 0..inner.files.len() {
            let (_, hi) = inner.split_range(s, Split::Train);
            total_blocks += (hi.saturating_sub(1) / SUPPORT_BLOCK) as u64;
            cumulative.push(total_blocks);
        }
        assert!(
            total_blocks > 0,
            "train region holds no complete {SUPPORT_BLOCK}-bar block"
        );

        let wanted = max_samples.div_ceil(SUPPORT_BLOCK) as u64;
        // Rejection sampling degrades as the draw approaches the population, so switch to a
        // full shuffle once half the blocks are wanted. Both paths are a uniform sample
        // without replacement; only the cost differs.
        let mut chosen: Vec<u64> = if wanted.saturating_mul(2) >= total_blocks {
            (0..total_blocks).collect()
        } else {
            let mut rng = ChaCha12Rng::seed_from_u64(mix64(seed, SUPPORT_STREAM));
            let mut set = HashSet::with_capacity(wanted as usize);
            while (set.len() as u64) < wanted {
                set.insert(rng.random_range(0..total_blocks));
            }
            let mut picked: Vec<u64> = set.into_iter().collect();
            // HashSet iteration order is unspecified; sort before the shuffle so the block
            // order — and hence which block the final truncation drops — is reproducible.
            picked.sort_unstable();
            picked
        };
        chosen.shuffle(&mut ChaCha12Rng::seed_from_u64(mix64(seed, SUPPORT_ORDER_STREAM)));
        chosen.truncate(wanted as usize);

        let refs: Vec<WindowRef> = chosen
            .iter()
            .map(|&block| {
                let symbol = cumulative.partition_point(|&c| c <= block) - 1;
                let local = block - cumulative[symbol];
                WindowRef {
                    symbol: symbol as u32,
                    bar_index: (1 + local as usize * SUPPORT_BLOCK) as u32,
                }
            })
            .collect();

        let mut out: Vec<(i64, BarDof)> = refs
            .par_iter()
            .flat_map_iter(|r| {
                let bars = inner.files[r.symbol as usize].bars();
                let mut block = Vec::with_capacity(SUPPORT_BLOCK);
                for_each_window_dof(bars, r.bar_index as usize, SUPPORT_BLOCK, |bar, dof| {
                    block.push((bar.ts(), dof));
                });
                block
            })
            .collect();
        out.truncate(max_samples);
        out
    }
}

impl Corpus {
    fn split_range(&self, symbol: usize, split: Split) -> (usize, usize) {
        let file = &self.files[symbol];
        let (b0, b1) = self.bounds;
        match split {
            Split::Train => (0, file.index_at_or_after(b0)),
            Split::Val => (file.index_at_or_after(b0), file.index_at_or_after(b1)),
            Split::Test => (file.index_at_or_after(b1), file.len()),
        }
    }
}

/// Deterministic near-disjoint window sampler over one split.
pub struct BarSampler {
    corpus: Arc<Corpus>,
    split: Split,
    context: i64,
    seed: u64,
    anchors: Vec<WindowRef>,
    /// `(start, len)` of each symbol's contiguous, time-ordered run inside `anchors`.
    symbol_runs: Vec<(u32, u32)>,
    order: RwLock<EpochOrder>,
}

#[derive(Default)]
struct EpochOrder {
    epoch: Option<usize>,
    order: Vec<u32>,
}

impl std::fmt::Debug for BarSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarSampler")
            .field("split", &self.split)
            .field("context", &self.context)
            .field("seed", &self.seed)
            .field("windows", &self.anchors.len())
            .finish()
    }
}

impl BarSampler {
    /// Anchors are strided by exactly `context`, so consecutive windows of one symbol share
    /// only their seam bar. A window occupies bars `[a, a + context]` — `context + 1` DOF, the
    /// caller slices inputs `[..context]` and targets `[1..]` — and every one of those bars is
    /// required to lie inside `split`.
    pub fn new(corpus: &BarCorpus, split: Split, context: i64, seed: u64) -> Self {
        assert!(context > 0, "context must be positive");
        let inner = corpus.inner.clone();
        let ctx = context as usize;
        let mut anchors = Vec::new();
        let mut symbol_runs = Vec::with_capacity(inner.files.len());
        for symbol in 0..inner.files.len() {
            let start = anchors.len() as u32;
            let (lo, hi) = inner.split_range(symbol, split);
            // Bar 0 has no predecessor close, so it can never carry a DOF.
            let first = lo.max(1);
            if hi > first + ctx {
                for a in (first..=hi - 1 - ctx).step_by(ctx) {
                    anchors.push(WindowRef {
                        symbol: symbol as u32,
                        bar_index: a as u32,
                    });
                }
            }
            symbol_runs.push((start, anchors.len() as u32 - start));
        }
        Self {
            corpus: inner,
            split,
            context,
            seed,
            anchors,
            symbol_runs,
            order: RwLock::new(EpochOrder::default()),
        }
    }

    pub fn split(&self) -> Split {
        self.split
    }

    /// The two shared wall-clock instants this sampler's split was cut at, in epoch millis.
    /// Carried so a held-out score can state which data it was measured on.
    pub fn split_bounds(&self) -> (i64, i64) {
        self.corpus.bounds
    }

    pub fn context(&self) -> i64 {
        self.context
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Open timestamp of a window's first DOF-carrying bar, in epoch millis.
    ///
    /// This is what places a pinned window on the calendar, which is what the held-out
    /// dispersion estimate blocks on: windows sharing a symbol and a calendar month are one
    /// draw from the market, not two.
    pub fn anchor_ts_ms(&self, r: &WindowRef) -> i64 {
        self.corpus.files[r.symbol as usize].bars()[r.bar_index as usize].ts()
    }

    pub fn symbol(&self, index: u32) -> &str {
        &self.corpus.symbols[index as usize]
    }

    /// Number of near-disjoint windows in this split: `sum_symbols floor((usable - 1) / context)`
    /// where `usable` is the symbol's bar count inside the split, minus the leading bar when the
    /// split starts at the file's first record.
    pub fn windows(&self) -> usize {
        self.anchors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Whole batches in one pass over unique bars. The partial tail is dropped so every step
    /// sees the same shape.
    pub fn batches_per_epoch(&self, batch: usize) -> usize {
        assert!(batch > 0, "batch size must be positive");
        self.anchors.len() / batch
    }

    pub fn anchors(&self) -> &[WindowRef] {
        &self.anchors
    }

    /// `[batch, context + 1, ..]` DOF and calendar ids on `device`, bit-identical for a given
    /// `(seed, epoch, index, batch)` and reordered by `epoch`.
    pub fn batch(&self, epoch: usize, index: usize, batch: usize, device: Device) -> BarBatch {
        self.batch_of(&self.batch_refs(epoch, index, batch), device)
    }

    /// A fixed, seed-pinned, ticker- and time-stratified window list for model selection.
    /// Independent of `epoch`, so the same `(seed, count)` selects the same evaluation set on
    /// every run. Each symbol receives a quota proportional to its window count (largest
    /// remainder), its picks are spread evenly across its timeline, and the symbols are
    /// interleaved so that any prefix or chunk of the result is itself diverse.
    pub fn pinned_windows(&self, count: usize) -> Vec<WindowRef> {
        let count = count.min(self.anchors.len());
        if count == 0 {
            return Vec::new();
        }
        let total = self.anchors.len() as u128;
        let live: Vec<usize> = (0..self.symbol_runs.len())
            .filter(|&s| self.symbol_runs[s].1 > 0)
            .collect();

        let mut quota = vec![0usize; self.symbol_runs.len()];
        let mut assigned = 0usize;
        let mut remainders: Vec<(u128, usize)> = Vec::with_capacity(live.len());
        for &s in &live {
            let len = self.symbol_runs[s].1 as u128;
            let exact = len * count as u128;
            let floor = (exact / total) as usize;
            quota[s] = floor;
            assigned += floor;
            remainders.push((exact % total, s));
        }
        // Largest remainder, ties broken by symbol index, so the allocation is deterministic.
        remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, s) in remainders.iter() {
            if assigned == count {
                break;
            }
            if quota[s] < self.symbol_runs[s].1 as usize {
                quota[s] += 1;
                assigned += 1;
            }
        }

        let mut rng = ChaCha12Rng::seed_from_u64(mix64(self.seed, PINNED_STREAM));
        let mut round_robin = live.clone();
        round_robin.shuffle(&mut rng);
        let phase: Vec<u64> = (0..self.symbol_runs.len())
            .map(|_| rng.random::<u64>())
            .collect();

        let max_quota = quota.iter().copied().max().unwrap_or(0);
        let mut out = Vec::with_capacity(count);
        for j in 0..max_quota {
            for &s in &round_robin {
                let q = quota[s] as u128;
                if j as u128 >= q {
                    continue;
                }
                let (start, len) = self.symbol_runs[s];
                // `q` evenly spaced picks across the symbol's timeline, rotated by a pinned
                // per-symbol offset. Rotation preserves both the spacing and the uniqueness of
                // the picks while making the exact bars depend on the sampler seed.
                let span = len as u128;
                let offset = phase[s] as u128 % span;
                let local = (((2 * j as u128 + 1) * span) / (2 * q) + offset) % span;
                out.push(self.anchors[start as usize + local as usize]);
            }
        }
        out
    }

    /// Byte slab a window reads: its DOF bars plus the causal warm-up prefix.
    fn slab(&self, r: &WindowRef, len: usize) -> &[PackedBar] {
        let bars = self.corpus.files[r.symbol as usize].bars();
        let anchor = r.bar_index as usize;
        &bars[anchor.saturating_sub(DOF_WARMUP_BARS + 1)..anchor + len]
    }

    /// Queue kernel readahead for every window of `refs` without touching the pages.
    ///
    /// A window is ~83 KB of one mmap'd file and the batch scatters across ~64 files, so
    /// faulting them in during the encode serializes 64 independent read chains behind a
    /// handful of worker threads. Issuing the whole batch's readahead up front turns that into
    /// one deep NVMe queue. Call it a step ahead of [`Self::batch_of`] to overlap I/O with
    /// compute entirely.
    pub fn prefetch(&self, refs: &[WindowRef]) {
        let len = (self.context + 1) as usize;
        refs.par_iter().for_each(|r| readahead(self.slab(r, len)));
    }

    /// `[refs.len(), context + 1, ..]` DOF and calendar ids on `device`.
    pub fn batch_of(&self, refs: &[WindowRef], device: Device) -> BarBatch {
        assert!(!refs.is_empty(), "cannot build an empty batch");
        let len = (self.context + 1) as usize;
        let rows: Vec<(usize, usize)> = refs
            .iter()
            .map(|r| (r.symbol as usize, r.bar_index as usize))
            .collect();
        build_batch(
            &self.corpus.files,
            &rows,
            len,
            self.corpus.res_secs,
            device,
        )
    }

    /// Windows the epoch order will hand out at `index`, for [`Self::prefetch`].
    pub fn batch_refs(&self, epoch: usize, index: usize, batch: usize) -> Vec<WindowRef> {
        let total = self.batches_per_epoch(batch);
        assert!(
            index < total,
            "batch index {index} out of range for {total} batches per epoch"
        );
        self.with_order(epoch, |order| {
            order[index * batch..(index + 1) * batch]
                .iter()
                .map(|&i| self.anchors[i as usize])
                .collect()
        })
    }

    fn with_order<R>(&self, epoch: usize, f: impl FnOnce(&[u32]) -> R) -> R {
        {
            let guard = self.order.read().expect("sampler order lock");
            if guard.epoch == Some(epoch) {
                return f(&guard.order);
            }
        }
        let mut guard = self.order.write().expect("sampler order lock");
        if guard.epoch != Some(epoch) {
            let mut order: Vec<u32> = (0..self.anchors.len() as u32).collect();
            order.shuffle(&mut ChaCha12Rng::seed_from_u64(mix64(
                self.seed,
                epoch as u64,
            )));
            guard.order = order;
            guard.epoch = Some(epoch);
        }
        f(&guard.order)
    }
}

/// Encode `len` bars starting at `start` for every `(series, start)` row into one
/// [`BarBatch`]. The DOF and the calendar ids come off the same bar in the same pass, which is
/// the whole reason the two tensors are returned together.
fn build_batch(
    files: &[BarFile],
    rows: &[(usize, usize)],
    len: usize,
    res_secs: u32,
    device: Device,
) -> BarBatch {
    let dof_row = len * BAR_DOF;
    let time_row = len * BAR_TIME_FEATURES;
    let mut dof = vec![0f32; rows.len() * dof_row];
    let mut time = vec![0i64; rows.len() * time_row];
    dof.par_chunks_mut(dof_row)
        .zip(time.par_chunks_mut(time_row))
        .zip(rows.par_iter())
        .for_each(|((dof_out, time_out), &(series, start))| {
            let bars = files[series].bars();
            readahead(&bars[start.saturating_sub(DOF_WARMUP_BARS + 1)..start + len]);
            let mut slot = 0usize;
            for_each_window_dof(bars, start, len, |bar, encoded| {
                dof_out[slot * BAR_DOF..(slot + 1) * BAR_DOF].copy_from_slice(&encoded.to_array());
                time_out[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES]
                    .copy_from_slice(&bar_time_ids(bar.ts(), res_secs));
                slot += 1;
            });
            debug_assert_eq!(slot, len);
        });
    let n = rows.len() as i64;
    let len = len as i64;
    BarBatch {
        dof: Tensor::from_slice(&dof)
            .view([n, len, BAR_DOF as i64])
            .to_device(device),
        time_ids: Tensor::from_slice(&time)
            .view([n, len, BAR_TIME_FEATURES as i64])
            .to_device(device),
    }
}

// ---------------------------------------------------------------------------
// Corpus anomaly audit
// ---------------------------------------------------------------------------

/// A single 5-minute bar cannot legitimately move 4x. Anything past this is a data defect.
pub const ANOMALY_LOG_LIMIT: f64 = std::f64::consts::LN_2 * 2.0;
/// A gap this long is an interior hole, not a weekend or a holiday.
pub const ANOMALY_HOLE_DAYS: i64 = 14;
/// Report base name; must stay in step with `meta_chart_bases` in `tui/src/main.rs`.
pub const ANOMALY_REPORT_BASE: &str = "pretrain_corpus_anomalies";
/// Symbols listed by name in the report title and the load-time log.
pub const ANOMALY_WORST_LISTED: usize = 20;

/// Extreme-move counts for one symbol, classified by cause.
#[derive(Clone, Debug)]
pub struct SymbolAnomalies {
    pub symbol: String,
    pub bars: usize,
    /// Interior gaps of at least [`ANOMALY_HOLE_DAYS`], whatever the price did across them.
    pub holes: usize,
    /// Extreme return across an interior hole: the signature of a ticker-string reuse splicing
    /// two different entities into one series. The remedy is re-ingestion by composite FIGI.
    pub splices: usize,
    /// Extreme return that reverts on the very next bar: a single bad print. The remedy is
    /// support clipping, since the bar itself is a real observation of a broken feed.
    pub ticks: usize,
    /// Extreme return that neither spans a hole nor reverts — a genuine level move, or a
    /// split/dividend adjustment seam. Neither of the above remedies applies.
    pub jumps: usize,
    /// Bars whose own high/low range exceeds the limit.
    pub extreme_range: usize,
}

impl SymbolAnomalies {
    pub fn total(&self) -> usize {
        self.splices + self.ticks + self.jumps + self.extreme_range
    }

    /// Anomalous bars per 10k, the only cross-symbol comparable figure.
    pub fn anomaly_rate(&self) -> f64 {
        if self.bars == 0 {
            0.0
        } else {
            10_000.0 * self.total() as f64 / self.bars as f64
        }
    }
}

/// Corpus-wide anomaly audit, `per_symbol` sorted by descending rate.
#[derive(Clone, Debug)]
pub struct CorpusAnomalies {
    /// Resolution the audit ran on. A daily bar is measured against the same absolute
    /// threshold, so two audits are only comparable when this matches.
    pub res_secs: u32,
    pub limit: f64,
    pub hole_ms: i64,
    pub bars: usize,
    pub holes: usize,
    pub splices: usize,
    pub ticks: usize,
    pub jumps: usize,
    pub extreme_range: usize,
    pub per_symbol: Vec<SymbolAnomalies>,
}

impl CorpusAnomalies {
    pub fn total(&self) -> usize {
        self.splices + self.ticks + self.jumps + self.extreme_range
    }

    /// The worst offenders, already sorted.
    pub fn worst(&self, count: usize) -> &[SymbolAnomalies] {
        &self.per_symbol[..count.min(self.per_symbol.len())]
    }

    pub fn summary(&self) -> String {
        format!(
            "{} anomalous bars in {} at {}s ({:.2}/10k) at |r| or s > ln {:.0}: {} splices, {} ticks, {} jumps, {} extreme ranges, {} interior holes",
            self.total(),
            self.bars,
            self.res_secs,
            if self.bars == 0 {
                0.0
            } else {
                10_000.0 * self.total() as f64 / self.bars as f64
            },
            self.limit.exp(),
            self.splices,
            self.ticks,
            self.jumps,
            self.extreme_range,
            self.holes
        )
    }

    /// Per-symbol rates as a chart, symbols ranked worst first. The schema carries no per-point
    /// labels, so the worst [`ANOMALY_WORST_LISTED`] names go in the title where they are
    /// actually readable.
    pub fn report(&self) -> Report {
        let series = |label: &str, pick: fn(&SymbolAnomalies) -> usize| ReportSeries {
            label: label.to_string(),
            values: self
                .per_symbol
                .iter()
                .map(|s| {
                    if s.bars == 0 {
                        0.0
                    } else {
                        (10_000.0 * pick(s) as f64 / s.bars as f64) as f32
                    }
                })
                .collect(),
        };
        let worst = self
            .worst(ANOMALY_WORST_LISTED)
            .iter()
            .map(|s| format!("{} {:.1}", s.symbol, s.anomaly_rate()))
            .collect::<Vec<_>>()
            .join(", ");
        Report {
            title: format!("{} | worst: {worst}", self.summary()),
            x_label: Some("symbol rank (worst first)".to_string()),
            y_label: Some("anomalous bars per 10k".to_string()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::MultiLine {
                series: vec![
                    series("splice", |s| s.splices),
                    series("tick", |s| s.ticks),
                    series("jump", |s| s.jumps),
                    series("extreme_range", |s| s.extreme_range),
                ],
            },
        }
    }

    /// Write `<dir>/pretrain_corpus_anomalies.report.bin`.
    ///
    /// One registered base name, so a multi-resolution run should write the deployment
    /// resolution's audit here and log the auxiliaries via [`Self::summary`]; the resolution is
    /// in the chart title either way.
    pub fn write_report(&self, dir: &Path) -> Result<()> {
        let path = dir.join(format!("{ANOMALY_REPORT_BASE}.report.bin"));
        shared::report::write_report(&path, &self.report())
            .with_context(|| format!("writing {}", path.display()))
    }
}

fn scan_symbol(file: &BarFile) -> SymbolAnomalies {
    let bars = file.bars();
    let hole_ms = ANOMALY_HOLE_DAYS * 86_400_000;
    let mut out = SymbolAnomalies {
        symbol: file.symbol().to_string(),
        bars: bars.len(),
        holes: 0,
        splices: 0,
        ticks: 0,
        jumps: 0,
        extreme_range: 0,
    };
    if bars.len() < 2 {
        return out;
    }
    // Going through `encode_dof` means the audit measures exactly the r and s the supports are
    // fitted on, rather than a second, subtly different definition.
    let mut returns = Vec::with_capacity(bars.len() - 1);
    for_each_window_dof(bars, 1, bars.len() - 1, |_, dof| {
        if dof.s as f64 > ANOMALY_LOG_LIMIT {
            out.extreme_range += 1;
        }
        returns.push(dof.r);
    });
    for (i, &r) in returns.iter().enumerate() {
        // `returns[i]` is the return of bar `i + 1` against bar `i`.
        let bar = i + 1;
        let gap = bars[bar].ts() - bars[bar - 1].ts();
        if gap >= hole_ms {
            out.holes += 1;
        }
        if (r as f64).abs() <= ANOMALY_LOG_LIMIT {
            continue;
        }
        let reverts = returns
            .get(i + 1)
            .is_some_and(|&next| ((r + next) as f64).abs() < 0.5 * (r as f64).abs());
        if gap >= hole_ms {
            out.splices += 1;
        } else if reverts {
            out.ticks += 1;
        } else {
            out.jumps += 1;
        }
    }
    out
}

/// Emit the DOF of bars `[anchor, anchor + len)`, preceded by a causal [`DOF_WARMUP_BARS`]
/// volume-EMA warm-up. Uses exactly the [`VolumeEma`] / [`encode_dof`] pair that
/// [`crate::torch::bar_dist::encode_series`] uses, so the values agree with a whole-series
/// encode to EMA warm-up precision.
fn for_each_window_dof(
    bars: &[PackedBar],
    anchor: usize,
    len: usize,
    mut sink: impl FnMut(&PackedBar, BarDof),
) {
    assert!(anchor >= 1, "bar 0 has no predecessor close");
    assert!(anchor + len <= bars.len(), "window runs past the symbol");
    let start = anchor.saturating_sub(DOF_WARMUP_BARS + 1);
    let mut ema = VolumeEma::default();
    ema.observe(bars[start].volume);
    let mut prev_close = bars[start].close;
    for (offset, bar) in bars[start + 1..anchor + len].iter().enumerate() {
        let volume = bar.volume;
        let reference = ema.reference_for(volume);
        if start + 1 + offset >= anchor {
            sink(bar, encode_dof(prev_close, bar, reference));
        }
        ema.observe(volume);
        prev_close = bar.close;
    }
}

/// Ask the kernel to page in the mapped range backing `slab`, without faulting on it here.
/// Advisory: a failure only costs a later fault, so the return value is deliberately ignored.
fn readahead(slab: &[PackedBar]) {
    if slab.is_empty() {
        return;
    }
    let start = slab.as_ptr() as usize;
    let end = start + std::mem::size_of_val(slab);
    let lo = start & !(*PAGE_SIZE - 1);
    // SAFETY: the mapping `slab` borrows from is page-aligned at its base, so rounding `start`
    // down to a page boundary cannot leave it; `end` is the slab's own end. MADV_WILLNEED only
    // schedules readahead — it neither writes nor unmaps.
    unsafe {
        libc::madvise(lo as *mut libc::c_void, end - lo, libc::MADV_WILLNEED);
    }
}

/// Host page size, for aligning [`readahead`] ranges down to a page boundary.
static PAGE_SIZE: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    // SAFETY: sysconf is a pure query.
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if raw > 0 {
        raw as usize
    } else {
        4096
    }
});

/// Load several resolutions of the same corpus directory against ONE shared calendar split.
///
/// `deployment` is the resolution that is actually traded and that therefore defines the two
/// split instants; every `auxiliary` resolution inherits them verbatim. Computing the boundary
/// independently per resolution would reintroduce exactly the leak the global split exists to
/// prevent, one timeframe's test window overlapping another's train window in wall-clock time.
///
/// Returns the deployment corpus first, then the auxiliaries in the order given.
pub fn load_multi_resolution(
    dir: &Path,
    deployment: (u32, usize),
    auxiliary: &[(u32, usize)],
) -> Result<Vec<BarCorpus>> {
    let (res_secs, min_bars) = deployment;
    let primary = BarCorpus::load(dir, res_secs, min_bars)?;
    let bounds = primary.split_bounds();
    let mut out = vec![primary];
    for &(res, min) in auxiliary {
        if res == res_secs {
            bail!("resolution {res} is both the deployment and an auxiliary resolution");
        }
        out.push(BarCorpus::load_with_bounds(dir, res, min, bounds)?);
    }
    Ok(out)
}

fn corpus_paths(dir: &Path, res_secs: u32) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading bar corpus directory {}", dir.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry
            .with_context(|| format!("reading bar corpus directory {}", dir.display()))?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some(FILE_EXTENSION) {
            continue;
        }
        if matches!(parse_bar_file_name(&path), Ok((_, res)) if res == res_secs) {
            paths.push(path);
        }
    }
    Ok(paths)
}

/// The two instants at the [`TRAIN_FRACTION`] and `TRAIN_FRACTION + VAL_FRACTION` percentile
/// of the global trading-time axis, i.e. of the pooled multiset of every symbol's bar
/// timestamps. Weighting by bars rather than by calendar time keeps the split honest across
/// holidays, half-days and mid-corpus listings.
///
/// Found by bisection on the resolution grid: `bars_before(t)` is a monotone step function
/// evaluated as one binary search per symbol, so this costs ~20 * n_symbols index probes
/// instead of a full 24 GB scan.
fn global_split_bounds(files: &[BarFile], res_secs: u32, total_bars: usize) -> (i64, i64) {
    let res_ms = res_secs as i64 * 1000;
    let first = files.iter().filter_map(BarFile::first_ts_ms).min();
    let last = files.iter().filter_map(BarFile::last_ts_ms).max();
    let (Some(first), Some(last)) = (first, last) else {
        return (0, 0);
    };

    let lo_slot = first.div_euclid(res_ms);
    let hi_slot = last.div_euclid(res_ms) + 1;
    let bars_before = |ts: i64| -> usize {
        files
            .par_iter()
            .map(|file| file.index_at_or_after(ts))
            .sum()
    };
    let percentile = |fraction: f64| -> i64 {
        let target = ((total_bars as f64) * fraction).round() as usize;
        let (mut lo, mut hi) = (lo_slot, hi_slot);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if bars_before(mid * res_ms) >= target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo * res_ms
    };

    let train_val = percentile(TRAIN_FRACTION);
    let val_test = percentile(TRAIN_FRACTION + VAL_FRACTION);
    (train_val, val_test.max(train_val))
}

/// `epoch`/stream-mixing hash (splitmix64 finalizer). Adjacent seeds and adjacent epochs must
/// produce unrelated ChaCha streams, which a plain `seed ^ epoch` does not guarantee.
///
/// Public because any counter-keyed randomized statistic elsewhere in the pipeline needs the
/// same decorrelation — the evaluation PIT mixes its chunk index in through this.
pub fn mix64(seed: u64, stream: u64) -> u64 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(stream.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// UTC ISO-8601 rendering of an epoch-millis instant, for logs and reports.
pub fn iso_ms(ts_ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ts_ms)
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_else(|| format!("ts_ms={ts_ms}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::encode_series;
    use shared::bars::write_bar_file;

    const RES: u32 = 300;
    const RES_MS: i64 = RES as i64 * 1000;

    struct Fixture {
        dir: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Deterministic pseudo-random walk, so bars look like bars (positive prices, ordered
    /// OHLC, varying volume) without pulling a live corpus into the test.
    fn synth_bars(seed: u64, count: usize, first_ts_ms: i64) -> Vec<PackedBar> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut close = 100.0f32;
        (0..count)
            .map(|i| {
                let drift = rng.random_range(-0.01f32..0.01f32);
                let open = close;
                close = (close * (1.0 + drift)).max(1.0);
                let spread = rng.random_range(0.0f32..0.02f32) * open;
                let high = open.max(close) + spread;
                let low = (open.min(close) - spread).max(0.5);
                PackedBar {
                    ts_ms: first_ts_ms + i as i64 * RES_MS,
                    open,
                    high,
                    low,
                    close,
                    volume: rng.random_range(1_000.0f32..50_000.0f32),
                    vwap: 0.25 * (open + high + low + close),
                    trades: rng.random_range(1u32..500),
                }
            })
            .collect()
    }

    /// Three symbols with deliberately staggered listing dates and lengths: a per-symbol
    /// index split would place them at three different wall-clock instants.
    fn fixture(label: &str) -> (Fixture, BarCorpus) {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_dataset_{label}_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        for (symbol, seed, count, offset) in [
            ("AAA", 1u64, 5_000usize, 0i64),
            ("BBB", 2, 3_100, 1_200),
            ("CCC", 3, 4_400, 400),
            ("TINY", 4, 40, 0),
        ] {
            let bars = synth_bars(seed, count, base + offset * RES_MS);
            write_bar_file(&bar_path(&dir, symbol), symbol, RES, &bars).unwrap();
        }
        let corpus = BarCorpus::load(&dir, RES, 100).unwrap();
        (Fixture { dir }, corpus)
    }

    fn bar_path(dir: &Path, symbol: &str) -> PathBuf {
        dir.join(format!("{symbol}.{RES}.{FILE_EXTENSION}"))
    }

    #[test]
    fn undersized_symbols_are_dropped() {
        let (_fx, corpus) = fixture("drop");
        assert_eq!(corpus.symbols(), &["AAA", "BBB", "CCC"]);
        assert_eq!(corpus.unique_bars(), 5_000 + 3_100 + 4_400);
    }

    #[test]
    fn resolutions_never_mix_and_share_one_split() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_dataset_multires_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let _fx = Fixture { dir: dir.clone() };

        // Deployment 5-minute bars over ~2020-09..2021-01, and a daily series reaching back to
        // 2015 — the shape a deep-history auxiliary corpus actually has.
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        for symbol in ["AAA", "BBB"] {
            let intraday = synth_bars(1, 5_000, base);
            write_bar_file(&bar_path(&dir, symbol), symbol, RES, &intraday).unwrap();
            let day_ms = 86_400_000i64;
            // Deep history only: 1400 daily bars ending well before the intraday corpus opens,
            // which is the shape of an auxiliary crash-regime corpus.
            let mut daily = synth_bars(2, 1_400, base - 1_500 * day_ms);
            for (i, bar) in daily.iter_mut().enumerate() {
                bar.ts_ms = base - 1_500 * day_ms + i as i64 * day_ms;
            }
            let path = dir.join(format!("{symbol}.86400.{FILE_EXTENSION}"));
            write_bar_file(&path, symbol, 86_400, &daily).unwrap();
        }

        let loaded = load_multi_resolution(&dir, (RES, 100), &[(86_400, 100)]).unwrap();
        let (intraday, daily) = (&loaded[0], &loaded[1]);

        // Neither corpus may see the other's files.
        assert_eq!(intraday.unique_bars(), 2 * 5_000);
        assert_eq!(daily.unique_bars(), 2 * 1_400);
        assert_eq!(intraday.res_secs(), RES);
        assert_eq!(daily.res_secs(), 86_400);
        assert_eq!(intraday.resolution_class(), resolution_class(RES));
        assert_ne!(intraday.resolution_class(), daily.resolution_class());

        // One boundary pair, so a daily bar and a 5-minute bar from the same date agree.
        assert_eq!(intraday.split_bounds(), daily.split_bounds());
        let (b0, b1) = intraday.split_bounds();
        for corpus in [intraday, daily] {
            for s in 0..corpus.series_count() {
                let bars = corpus.bars(s);
                let (_, train_hi) = corpus.split_range(s, Split::Train);
                let (_, val_hi) = corpus.split_range(s, Split::Val);
                assert!(bars[..train_hi].iter().all(|b| b.ts() < b0));
                assert!(bars[train_hi..val_hi].iter().all(|b| b.ts() < b1));
            }
        }
        // The daily series starts long before the boundary, so it is entirely training data —
        // which is the point of an auxiliary deep-history corpus.
        assert_eq!(daily.split_bars(Split::Train), daily.unique_bars());

        // Calendar ids separate the two resolutions for the trunk.
        let intraday_ids = bar_time_ids(intraday.ts_ms(0, 0), intraday.res_secs());
        let daily_ids = bar_time_ids(daily.ts_ms(0, 0), daily.res_secs());
        assert_ne!(
            intraday_ids[TIME_RESOLUTION],
            daily_ids[TIME_RESOLUTION],
            "the resolution channel must distinguish the timeframes"
        );

        assert!(
            load_multi_resolution(&dir, (RES, 100), &[(RES, 100)]).is_err(),
            "a resolution cannot be both deployment and auxiliary"
        );
        assert_eq!(
            crate::data::universe::eligible_bar_universe(&dir, RES, 100),
            vec!["AAA".to_string(), "BBB".to_string()]
        );
        assert_eq!(intraday.scan_anomalies().res_secs, RES);
    }

    #[test]
    fn split_is_one_pair_of_instants_shared_by_every_symbol() {
        let (_fx, corpus) = fixture("global");
        let (b0, b1) = corpus.split_bounds();
        assert!(b0 < b1, "train|val {b0} must precede val|test {b1}");

        let mut fractions = Vec::new();
        for (s, symbol) in corpus.symbols().iter().enumerate() {
            let bars = corpus.bars(s);
            let (t_lo, t_hi) = corpus.split_range(s, Split::Train);
            let (v_lo, v_hi) = corpus.split_range(s, Split::Val);
            let (e_lo, e_hi) = corpus.split_range(s, Split::Test);
            assert_eq!((t_lo, t_hi), (0, v_lo), "{symbol} train/val must abut");
            assert_eq!((v_hi, e_hi), (e_lo, bars.len()), "{symbol} val/test must abut");
            assert!(bars[..t_hi].iter().all(|b| b.ts() < b0), "{symbol} train leak");
            assert!(
                bars[v_lo..v_hi].iter().all(|b| b.ts() >= b0 && b.ts() < b1),
                "{symbol} val leak"
            );
            assert!(bars[e_lo..].iter().all(|b| b.ts() >= b1), "{symbol} test leak");
            fractions.push(t_hi as f64 / bars.len() as f64);
        }
        // The whole point of a calendar split: symbols with different listing dates and
        // lengths do *not* get the same index-space cut. If they did, this test would be
        // passing for the wrong reason.
        let spread = fractions.iter().cloned().fold(f64::MIN, f64::max)
            - fractions.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 0.01, "fixture is degenerate, index split would agree");

        // Global bar mass either side of the bounds tracks the requested percentiles.
        let train = corpus.split_bars(Split::Train) as f64 / corpus.unique_bars() as f64;
        let val = corpus.split_bars(Split::Val) as f64 / corpus.unique_bars() as f64;
        assert!((train - TRAIN_FRACTION).abs() < 0.01, "train share {train}");
        assert!((val - VAL_FRACTION).abs() < 0.01, "val share {val}");
    }

    #[test]
    fn no_window_crosses_a_split_boundary() {
        let (_fx, corpus) = fixture("bounds");
        let (b0, b1) = corpus.split_bounds();
        for split in Split::ALL {
            let sampler = BarSampler::new(&corpus, split, 128, 7);
            let (lo_ts, hi_ts) = match split {
                Split::Train => (i64::MIN, b0),
                Split::Val => (b0, b1),
                Split::Test => (b1, i64::MAX),
            };
            assert!(sampler.windows() > 0, "{split} has no windows");
            for r in sampler.anchors() {
                let bars = corpus.bars(r.symbol as usize);
                let first = bars[r.bar_index as usize].ts();
                let last = bars[r.bar_index as usize + 128].ts();
                assert!(first >= lo_ts && last < hi_ts, "{split} window {r:?} crosses");
            }
        }
    }

    #[test]
    fn windows_are_near_disjoint_and_match_the_stride_formula() {
        let (_fx, corpus) = fixture("stride");
        let context = 256usize;
        for split in Split::ALL {
            let sampler = BarSampler::new(&corpus, split, context as i64, 11);
            let mut expected = 0usize;
            for s in 0..corpus.symbols().len() {
                let (lo, hi) = corpus.split_range(s, split);
                let usable = hi.saturating_sub(lo.max(1));
                expected += usable.saturating_sub(1) / context;
            }
            assert_eq!(sampler.windows(), expected, "{split} window count");

            // Stride == context, so successive anchors of one symbol overlap in exactly the
            // one seam bar they share.
            let mut prev: Option<WindowRef> = None;
            for r in sampler.anchors() {
                if let Some(p) = prev.filter(|p| p.symbol == r.symbol) {
                    assert_eq!(r.bar_index - p.bar_index, context as u32);
                }
                prev = Some(*r);
            }
        }
    }

    #[test]
    fn batches_are_bit_reproducible_and_reordered_per_epoch() {
        let (_fx, corpus) = fixture("determinism");
        let a = BarSampler::new(&corpus, Split::Train, 64, 4242);
        let b = BarSampler::new(&corpus, Split::Train, 64, 4242);
        let x0 = a.batch(3, 2, 8, Device::Cpu);
        let x1 = b.batch(3, 2, 8, Device::Cpu);
        assert_eq!(x0.dof.size(), vec![8, 65, BAR_DOF as i64]);
        assert_eq!(x0.time_ids.size(), vec![8, 65, BAR_TIME_FEATURES as i64]);
        assert_eq!(x0.time_ids.kind(), tch::Kind::Int64);
        assert!(
            bool::try_from(x0.dof.eq_tensor(&x1.dof).all()).unwrap()
                && bool::try_from(x0.time_ids.eq_tensor(&x1.time_ids).all()).unwrap(),
            "same (seed, epoch, index) must be bit-identical"
        );

        let other_epoch = a.batch(4, 2, 8, Device::Cpu);
        assert!(
            !bool::try_from(x0.dof.eq_tensor(&other_epoch.dof).all()).unwrap(),
            "a new epoch must reorder the pass"
        );
        let other_seed =
            BarSampler::new(&corpus, Split::Train, 64, 4243).batch(3, 2, 8, Device::Cpu);
        assert!(
            !bool::try_from(x0.dof.eq_tensor(&other_seed.dof).all()).unwrap(),
            "a new seed must reorder the pass"
        );

        // Every calendar id must be a legal embedding row.
        for feature in 0..BAR_TIME_FEATURES {
            let column = x0.time_ids.select(-1, feature as i64);
            let lo = column.min().int64_value(&[]);
            let hi = column.max().int64_value(&[]);
            assert!(
                lo >= 0 && hi < BAR_TIME_CARDINALITY[feature],
                "{} ids span [{lo}, {hi}], outside [0, {})",
                BAR_TIME_NAMES[feature],
                BAR_TIME_CARDINALITY[feature]
            );
        }

        // Every epoch is a permutation: one pass, each window exactly once.
        let mut seen: Vec<WindowRef> = Vec::new();
        a.with_order(9, |order| {
            for &i in order {
                seen.push(a.anchors[i as usize]);
            }
        });
        let mut sorted = seen.clone();
        sorted.sort_by_key(|r| (r.symbol, r.bar_index));
        let mut all = a.anchors().to_vec();
        all.sort_by_key(|r| (r.symbol, r.bar_index));
        assert_eq!(sorted, all);
    }

    #[test]
    fn window_dof_match_a_whole_series_encode() {
        let (_fx, corpus) = fixture("dof");
        let sampler = BarSampler::new(&corpus, Split::Val, 32, 5);
        let refs: Vec<WindowRef> = sampler.anchors().iter().copied().take(4).collect();
        let batch = sampler.batch_of(&refs, Device::Cpu);
        for (row, r) in refs.iter().enumerate() {
            let bars = corpus.bars(r.symbol as usize);
            let series = encode_series(bars);
            for step in 0..33usize {
                // encode_series aligns with bars[1..], so bar index `i` is series[i - 1].
                let bar = r.bar_index as usize + step;
                let want = series[bar - 1].to_array();
                for d in 0..BAR_DOF {
                    let got = batch.dof.double_value(&[row as i64, step as i64, d as i64]) as f32;
                    assert!(
                        (got - want[d]).abs() <= 1e-6 * want[d].abs().max(1.0),
                        "row {row} step {step} dof {d}: {got} vs {}",
                        want[d]
                    );
                }
                // The calendar row must describe the very same bar the DOF row came from.
                let want_ids = bar_time_ids(bars[bar].ts(), corpus.res_secs());
                for f in 0..BAR_TIME_FEATURES {
                    assert_eq!(
                        batch.time_ids.int64_value(&[row as i64, step as i64, f as i64]),
                        want_ids[f],
                        "row {row} step {step} {}",
                        BAR_TIME_NAMES[f]
                    );
                }
            }
        }
    }

    #[test]
    fn pinned_windows_are_stable_stratified_and_unique() {
        let (_fx, corpus) = fixture("pinned");
        let sampler = BarSampler::new(&corpus, Split::Train, 32, 99);
        let want = 64;
        assert!(sampler.windows() > 2 * want, "fixture must oversupply windows");
        let first = sampler.pinned_windows(want);
        assert_eq!(first.len(), want);
        assert_eq!(
            first,
            BarSampler::new(&corpus, Split::Train, 32, 99).pinned_windows(want)
        );
        assert_ne!(
            first,
            BarSampler::new(&corpus, Split::Train, 32, 100).pinned_windows(want)
        );
        let unique: HashSet<WindowRef> = first.iter().copied().collect();
        assert_eq!(unique.len(), first.len(), "pinned windows must not repeat");
        let symbols: HashSet<u32> = first.iter().map(|r| r.symbol).collect();
        assert_eq!(symbols.len(), corpus.symbols().len(), "every ticker represented");
        // Time-stratified: the picks straddle the whole split, not just its head.
        for s in 0..corpus.symbols().len() as u32 {
            let picks: Vec<u32> = first
                .iter()
                .filter(|r| r.symbol == s)
                .map(|r| r.bar_index)
                .collect();
            let (lo, hi) = corpus.split_range(s as usize, Split::Train);
            let span = (hi - lo) as u32;
            let reach = picks.iter().max().unwrap() - picks.iter().min().unwrap();
            assert!(reach > span / 2, "symbol {s} picks span only {reach} of {span}");
        }
        // Independent of the epoch shuffle.
        let _ = sampler.batch(17, 0, 4, Device::Cpu);
        assert_eq!(first, sampler.pinned_windows(want));
    }

    #[test]
    fn supports_are_fitted_from_train_timestamps_only() {
        let (_fx, corpus) = fixture("supports");
        let (b0, _) = corpus.split_bounds();
        let samples = corpus.sample_train_dof(4_000, 31);
        assert!(!samples.is_empty());
        assert!(samples.len() <= 4_000);
        for (ts, dof) in &samples {
            assert!(*ts < b0, "sampled {} at or after the train|val bound {b0}", iso_ms(*ts));
            assert!(dof.is_finite());
        }
        assert_eq!(samples, corpus.sample_train_dof(4_000, 31), "sampling must be pinned");

        let supports = corpus.fit_supports(4_000, 31);
        assert!(corpus.supports_path().is_file());
        let reloaded = BarSupports::load(&corpus.supports_path()).unwrap();
        for dof in 0..BAR_DOF {
            assert_eq!(supports.bin_of(dof, 0.0), reloaded.bin_of(dof, 0.0));
        }
    }

    /// `YYYY-MM-DDTHH:MM:SS` in ET, as epoch millis. Built via the same offset table the
    /// producer uses, so the test states its intent in wall-clock terms.
    fn et(text: &str) -> i64 {
        let naive = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S").unwrap();
        // Bisect the UTC instant whose ET rendering is `naive`: the offset depends on the
        // answer, so a single subtraction would be wrong across a transition.
        let guess = naive.and_utc().timestamp();
        for candidate in [guess + 4 * 3600, guess + 5 * 3600] {
            if candidate + et_offset_secs(candidate) as i64 == guess {
                return candidate * 1000;
            }
        }
        panic!("no UTC instant renders as {text} in ET");
    }

    #[test]
    fn calendar_ids_follow_et_wall_clock_through_dst() {
        // Standard time in January: 09:30 ET is minute 570 and the regular session opens.
        let open = bar_time_ids(et("2024-01-16T09:30:00"), RES);
        assert_eq!(open[TIME_MINUTE], 570);
        assert_eq!(open[TIME_WEEKDAY], 1, "2024-01-16 was a Tuesday");
        assert_eq!(open[TIME_SESSION], 2);
        assert_eq!(open[TIME_RESOLUTION], resolution_class(RES));

        // Daylight time in July: the same wall clock, an hour different in UTC. If the
        // producer used a fixed offset one of these two would be off by 60 minutes.
        let summer = bar_time_ids(et("2024-07-16T09:30:00"), RES);
        assert_eq!(summer[TIME_MINUTE], 570, "09:30 ET is minute 570 in DST too");
        assert_ne!(
            et("2024-01-16T09:30:00").rem_euclid(86_400_000),
            et("2024-07-16T09:30:00").rem_euclid(86_400_000),
            "the two instants must differ in UTC time-of-day, else DST is being ignored"
        );

        // Spring forward 2024-03-10: 02:00 ET never happens, 03:00 follows 01:59.
        assert_eq!(bar_time_ids(et("2024-03-10T01:30:00"), RES)[TIME_MINUTE], 90);
        assert_eq!(bar_time_ids(et("2024-03-10T03:30:00"), RES)[TIME_MINUTE], 210);
        // Fall back 2024-11-03: 01:00-02:00 ET happens twice; both renderings are minute 60+.
        let before = et("2024-11-03T00:30:00") + 3_600_000;
        let after = before + 3_600_000;
        assert_eq!(bar_time_ids(before, RES)[TIME_MINUTE], 90);
        assert_eq!(bar_time_ids(after, RES)[TIME_MINUTE], 90);

        // Half day: 2024-11-29, the day after Thanksgiving, closes at 13:00 ET. The session id
        // is a wall-clock regime, not an is-the-market-open flag, so 13:05 is still `regular`
        // — an early close is not derivable from a fixed boundary table and the model sees it
        // through the bars themselves (thin or absent) rather than through this id.
        assert_eq!(bar_time_ids(et("2024-11-29T12:55:00"), RES)[TIME_SESSION], 2);
        assert_eq!(bar_time_ids(et("2024-11-29T13:05:00"), RES)[TIME_SESSION], 2);
        assert_eq!(bar_time_ids(et("2024-11-29T12:55:00"), RES)[TIME_MINUTE], 775);
        assert_eq!(bar_time_ids(et("2024-11-29T13:05:00"), RES)[TIME_MINUTE], 785);
        assert_eq!(bar_time_ids(et("2024-11-29T12:55:00"), RES)[TIME_WEEKDAY], 4);

        // Every session boundary, in order.
        for (text, session) in [
            ("2024-06-12T03:59:00", 0),
            ("2024-06-12T04:00:00", 1),
            ("2024-06-12T09:29:00", 1),
            ("2024-06-12T09:30:00", 2),
            ("2024-06-12T15:59:00", 2),
            ("2024-06-12T16:00:00", 3),
            ("2024-06-12T19:59:00", 3),
            ("2024-06-12T20:00:00", 0),
        ] {
            assert_eq!(
                bar_time_ids(et(text), RES)[TIME_SESSION],
                session,
                "session at {text}"
            );
        }
    }

    #[test]
    fn calendar_ids_are_always_valid_embedding_rows() {
        // A dense sweep over five years at 7-minute steps walks every hour of every weekday
        // and both DST transitions ten times over, plus deliberately hostile inputs.
        let start = et("2021-01-01T00:00:00");
        let mut ts = start;
        while ts < start + 5 * 365 * 86_400_000 {
            let ids = bar_time_ids(ts, RES);
            for f in 0..BAR_TIME_FEATURES {
                assert!(
                    (0..BAR_TIME_CARDINALITY[f]).contains(&ids[f]),
                    "{} id {} out of range at {}",
                    BAR_TIME_NAMES[f],
                    ids[f],
                    iso_ms(ts)
                );
            }
            ts += 7 * 60 * 1000;
        }
        for hostile in [i64::MIN / 4, -1, 0, 1, i64::MAX / 4] {
            let ids = bar_time_ids(hostile, 0);
            for f in 0..BAR_TIME_FEATURES {
                assert!((0..BAR_TIME_CARDINALITY[f]).contains(&ids[f]));
            }
        }
        assert_eq!(resolution_class(300), 1);
        assert_eq!(resolution_class(86_400), 6);
        assert_eq!(resolution_class(7), RESOLUTION_CLASS_OTHER);
    }

    #[test]
    fn anomaly_scan_separates_splices_from_ticks() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_dataset_anomaly_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let _fx = Fixture { dir: dir.clone() };

        // CLEAN: an ordinary walk. SPLICE: a 90-day hole with a 100x level jump across it,
        // the ticker-reuse signature. TICK: a single 50x print that reverts on the next bar.
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let mut clean = synth_bars(9, 4_000, base);
        write_bar_file(&bar_path(&dir, "CLEAN"), "CLEAN", RES, &clean).unwrap();

        let mut spliced = clean.clone();
        let hole = 90 * 86_400_000i64;
        for bar in spliced[2_000..].iter_mut() {
            bar.ts_ms += hole;
            bar.open *= 100.0;
            bar.high *= 100.0;
            bar.low *= 100.0;
            bar.close *= 100.0;
        }
        write_bar_file(&bar_path(&dir, "SPLICE"), "SPLICE", RES, &spliced).unwrap();

        clean[3_000].high *= 50.0;
        clean[3_000].close *= 50.0;
        write_bar_file(&bar_path(&dir, "TICK"), "TICK", RES, &clean).unwrap();

        let corpus = BarCorpus::load(&dir, RES, 100).unwrap();
        let audit = corpus.scan_anomalies();
        let by_symbol: std::collections::HashMap<&str, &SymbolAnomalies> = audit
            .per_symbol
            .iter()
            .map(|s| (s.symbol.as_str(), s))
            .collect();

        let clean_stats = by_symbol["CLEAN"];
        assert_eq!(clean_stats.total(), 0, "a clean walk must trip nothing");
        assert_eq!(clean_stats.holes, 0);

        let splice_stats = by_symbol["SPLICE"];
        assert_eq!(splice_stats.holes, 1);
        assert_eq!(splice_stats.splices, 1, "level jump across a hole is a splice");
        assert_eq!(splice_stats.ticks, 0);

        let tick_stats = by_symbol["TICK"];
        assert_eq!(tick_stats.ticks, 1, "a reverting single print is a tick");
        assert_eq!(tick_stats.splices, 0);
        assert_eq!(tick_stats.holes, 0);
        assert!(tick_stats.extreme_range >= 1, "a 50x bar has an extreme range");

        assert_eq!(audit.splices, 1);
        assert_eq!(audit.ticks, 1);
        assert_eq!(audit.bars, corpus.unique_bars());
        // Sorted worst first.
        assert!(audit.per_symbol[0].anomaly_rate() >= audit.per_symbol[2].anomaly_rate());

        audit.write_report(&dir).unwrap();
        let path = dir.join(format!("{ANOMALY_REPORT_BASE}.report.bin"));
        let report = shared::report::read_report(&path).unwrap();
        let ReportKind::MultiLine { series } = &report.kind else {
            panic!("anomaly report must be a MultiLine chart");
        };
        assert_eq!(series.len(), 4);
        assert!(series.iter().all(|s| s.values.len() == 3));
        assert!(report.title.contains("SPLICE"));
    }

    /// Measured against whatever is on disk; `--ignored` because it needs the real corpus.
    /// `cargo test -p trading_bot_0 dataset::tests::bench_real_corpus -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_real_corpus() {
        let dir = Path::new(shared::paths::DATA_PATH).join("bars");
        let corpus = BarCorpus::load(&dir, RES, 10_000).unwrap();
        let (b0, b1) = corpus.split_bounds();
        println!(
            "symbols={} unique_bars={} train={} val={} test={} split={} | {}",
            corpus.symbols().len(),
            corpus.unique_bars(),
            corpus.split_bars(Split::Train),
            corpus.split_bars(Split::Val),
            corpus.split_bars(Split::Test),
            iso_ms(b0),
            iso_ms(b1)
        );

        let sampler = BarSampler::new(&corpus, Split::Train, 2048, 1);
        println!(
            "train windows={} batches_per_epoch(64)={}",
            sampler.windows(),
            sampler.batches_per_epoch(64)
        );

        let percentiles = |label: &str, mut ms: Vec<f64>| {
            ms.sort_by(f64::total_cmp);
            println!(
                "{label}: p50={:.2}ms p90={:.2}ms max={:.2}ms",
                ms[ms.len() / 2],
                ms[ms.len() * 9 / 10],
                ms[ms.len() - 1]
            );
        };

        // (0) Arithmetic floor: the same 64 slabs copied onto the heap, so no mmap page can be
        // reclaimed under us. Separating this from (a) is what decides lazy-encode vs a
        // materialized DOF cache — a cache can only ever remove the gap between the two.
        let floor_refs = sampler.batch_refs(0, 0, 64);
        let len = (sampler.context() + 1) as usize;
        let owned: Vec<Vec<PackedBar>> = floor_refs
            .iter()
            .map(|r| sampler.slab(r, len).to_vec())
            .collect();
        let anchors: Vec<usize> = floor_refs
            .iter()
            .map(|r| (r.bar_index as usize).min(DOF_WARMUP_BARS + 1))
            .collect();
        let mut ms = Vec::new();
        for _ in 0..32 {
            let start = std::time::Instant::now();
            let total: usize = owned
                .par_iter()
                .zip(anchors.par_iter())
                .map(|(bars, &anchor)| {
                    let mut n = 0usize;
                    for_each_window_dof(bars, anchor, len, |_, dof| n += dof.r.is_finite() as usize);
                    n
                })
                .sum();
            ms.push(start.elapsed().as_secs_f64() * 1e3);
            assert_eq!(total, 64 * len);
        }
        percentiles("encode-only floor [64,2049,5]", ms);

        // (a) Resident encode: the same 64 windows repeatedly, so every page is already in
        // core. This is the pure arithmetic + copy cost of building the tensor.
        let resident = sampler.batch_refs(0, 0, 64);
        let _ = sampler.batch_of(&resident, Device::Cpu);
        let mut ms = Vec::new();
        for _ in 0..32 {
            let start = std::time::Instant::now();
            let x = sampler.batch_of(&resident, Device::Cpu);
            ms.push(start.elapsed().as_secs_f64() * 1e3);
            assert_eq!(x.dof.size(), vec![64, 2049, BAR_DOF as i64]);
        }
        percentiles("batch[64,2049,5] resident", ms);

        // (a2) Where the gap between (0) and (a) actually goes.
        let rows: Vec<(usize, usize)> = resident
            .iter()
            .map(|r| (r.symbol as usize, r.bar_index as usize))
            .collect();
        let (mut advise, mut alloc, mut fill, mut build) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..32 {
            let t = std::time::Instant::now();
            sampler.prefetch(&resident);
            advise += t.elapsed().as_secs_f64() * 1e3;

            let t = std::time::Instant::now();
            let mut dof = vec![0f32; 64 * len * BAR_DOF];
            let mut time = vec![0i64; 64 * len * BAR_TIME_FEATURES];
            alloc += t.elapsed().as_secs_f64() * 1e3;

            let t = std::time::Instant::now();
            dof.par_chunks_mut(len * BAR_DOF)
                .zip(time.par_chunks_mut(len * BAR_TIME_FEATURES))
                .zip(rows.par_iter())
                .for_each(|((d, c), &(series, start))| {
                    let bars = corpus.bars(series);
                    let mut slot = 0usize;
                    for_each_window_dof(bars, start, len, |bar, encoded| {
                        d[slot * BAR_DOF..(slot + 1) * BAR_DOF]
                            .copy_from_slice(&encoded.to_array());
                        c[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES]
                            .copy_from_slice(&bar_time_ids(bar.ts(), RES));
                        slot += 1;
                    });
                });
            fill += t.elapsed().as_secs_f64() * 1e3;

            let t = std::time::Instant::now();
            let _ = Tensor::from_slice(&dof).view([64, len as i64, BAR_DOF as i64]);
            let _ = Tensor::from_slice(&time).view([64, len as i64, BAR_TIME_FEATURES as i64]);
            build += t.elapsed().as_secs_f64() * 1e3;
        }
        println!(
            "resident breakdown (mean ms): madvise={:.2} alloc={:.2} fill={:.2} tensor={:.2}",
            advise / 32.0,
            alloc / 32.0,
            fill / 32.0,
            build / 32.0
        );

        // (b) First touch: fresh windows every step, batched readahead inside `batch_of`.
        let mut ms = Vec::new();
        for i in 0..32 {
            let start = std::time::Instant::now();
            let _ = sampler.batch(1, i, 64, Device::Cpu);
            ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        percentiles("batch[64,2049,5] first-touch", ms);

        // (c) First touch with the readahead issued one step early, as a trainer would.
        let mut ms = Vec::new();
        let mut next = sampler.batch_refs(2, 0, 64);
        sampler.prefetch(&next);
        for i in 1..33 {
            let current = std::mem::replace(&mut next, sampler.batch_refs(2, i, 64));
            sampler.prefetch(&next);
            let start = std::time::Instant::now();
            let _ = sampler.batch_of(&current, Device::Cpu);
            ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        percentiles("batch[64,2049,5] first-touch, prefetched", ms);

        let fit_start = std::time::Instant::now();
        let samples = corpus.sample_train_dof(2_000_000, 1);
        println!(
            "sample_train_dof(2M) -> {} in {:.2}s",
            samples.len(),
            fit_start.elapsed().as_secs_f64()
        );
        assert!(samples.iter().all(|(ts, _)| *ts < b0));

        let scan_start = std::time::Instant::now();
        let audit = corpus.scan_anomalies();
        println!(
            "{} in {:.1}s",
            audit.summary(),
            scan_start.elapsed().as_secs_f64()
        );
        for s in audit.worst(ANOMALY_WORST_LISTED) {
            println!(
                "  {:<8} {:>8} bars  {:>7.2}/10k  splice={} tick={} jump={} range={} hole={}",
                s.symbol,
                s.bars,
                s.anomaly_rate(),
                s.splices,
                s.ticks,
                s.jumps,
                s.extreme_range,
                s.holes
            );
        }
        audit.write_report(&std::env::temp_dir()).unwrap();
    }
}
