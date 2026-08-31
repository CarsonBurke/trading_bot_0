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
//! * **One epoch is one pass over every training bar.** Anchors are strided by exactly
//!   `context`, so the TARGET spans of a symbol's consecutive windows tile its axis exactly:
//!   the window at `a` targets bars `a+1 ..= a+context` and the next one starts at
//!   `a + context`. [`PassPlan`] exploits that to partition a split's bars — not its
//!   anchors — into one disjoint share per ramp stage, sized to that stage's token budget, so
//!   the ramp tiles the corpus exactly once per epoch instead of each stage re-walking its own
//!   overlapping list. [`CoverageAudit::require_full_pass`] enforces it as an invariant: every
//!   bar is a prediction target exactly once, or it is in a named, counted remainder bucket.
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{bail, ensure, Context, Result};
use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime, TimeZone, Timelike, Weekday};
use chrono_tz::America::New_York;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use ring::digest::{Context as DigestContext, SHA256};
use shared::bars::{parse_bar_file_name, BarFile, PackedBar, FILE_EXTENSION};
use shared::report::{Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{
    encode_dof, scale_row, BarDof, BarSupports, BarSupportsProvenance, DofBinner, DofScaling,
    EncodedDof, RangeVolHar, StandardizedDof, VolumeEma, BAR_DOF, BAR_RAW_WARMUP_BARS, DOF_R,
    DOF_S, DOF_W, NUM_BAR_BINS,
};

/// Causal bars fed to the volume EMA before the first emitted DOF of a window. The span-20
/// EMA retains `(1 - 2/21)^256 ~ 7e-12` of its seed after this many observations, so a
/// window's DOF are numerically indistinguishable from encoding the symbol's whole series
/// with [`crate::torch::bar_dist::encode_series`] and slicing. The warm-up is strictly causal
/// past and may reach back across a split boundary; the DOF-carrying bars never do.
///
/// This is the [`DofScaling::Raw`] warm-up. [`DofScaling::VolStandardized`] additionally has
/// to converge the span-256 leg of [`RangeVolHar`] and so uses the longer
/// [`crate::torch::bar_dist::BAR_SIGMA_WARMUP_BARS`]; both are read through
/// [`DofScaling::warmup_bars`].
pub const DOF_WARMUP_BARS: usize = BAR_RAW_WARMUP_BARS;

/// Share of the global trading-time axis reserved for training, then for validation.
pub const TRAIN_FRACTION: f64 = 0.80;
pub const VAL_FRACTION: f64 = 0.10;

/// Longest training context in bars.
///
/// Duplicated from [`crate::torch::world_model::BAR_MAX_CONTEXT`] rather than imported:
/// `world_model` already depends on this module for its calendar ids, and the eligibility rule
/// has to be callable from `Ingest` before any tensor code is reachable. A test asserts the two
/// agree, so the duplicate cannot drift.
pub const MAX_CONTEXT_BARS: usize = 2_048;

/// Bars a symbol needs to enter the DEPLOYMENT corpus, and so to count toward the split
/// percentiles.
///
/// `10 * MAX_CONTEXT_BARS`, because the smallest split share is [`VAL_FRACTION`] = 0.10 of the
/// global trading-time axis: a symbol spread over that axis holding `N` bars puts about `N / 10`
/// of them in the val and test regions, so this is the floor that guarantees every admitted
/// symbol contributes at least one full-context window to EACH split. It lives here rather than
/// in the pretrain CLI because corpus ingestion must derive the `train | val` instant from the
/// same eligibility rule the pretrainer applies: dropping a file moves the trading-time
/// percentile, so a mismatch would put the universe and the split at odds.
pub const DEFAULT_MIN_BARS: usize = 10 * MAX_CONTEXT_BARS;

/// Bars a symbol needs to enter an AUXILIARY corpus.
///
/// A different rule from [`DEFAULT_MIN_BARS`], because an auxiliary resolution is used for a
/// different thing, and reusing the deployment floor here is the specific bug that would make a
/// multi-resolution run look like it worked while loading nothing. The 4,748 daily files hold a
/// MEDIAN of 3,540 bars and a MAXIMUM of 14,276 — 56 years of daily sessions is fewer bars than
/// five days of five-minute extended-hours trading — so a 20,480 floor rejects every one of them,
/// and `open_files` then fails the whole run rather than quietly training on nothing.
///
/// This floor means "not a stub", not "long enough to tile". The eligibility question an
/// auxiliary resolution actually poses is answered by its ramp contexts, not by a bar count: a
/// symbol shorter than the shortest auxiliary context is ADMITTED and lands in the coverage
/// audit's short-symbol remainder with its bar count against it, where it is visible in
/// `pretrain_pass_remainder`. A symbol dropped here disappears from `split_bars` entirely and is
/// invisible in the coverage accounting, which is the worse failure.
///
/// So 64, from the measured distribution rather than a round number: it drops the 42 daily files
/// holding fewer than 64 bars, admits 4,706, and keeps all 20,498,862 usable train bars. The
/// residual shortfall is then small and explicable — against the auxiliary ramp's shortest
/// context of 256, exactly 349 admitted symbols cannot tile one window and they carry 36,414
/// bars, 0.18% of the auxiliary train region. Every one of the 2,018 symbols with 2007-10..2009-03
/// history and all 1,410 with 2000-03..2002-10 history survives.
pub const AUXILIARY_MIN_BARS: usize = 64;

/// Consecutive DOF drawn per support-fitting block. Sampling in blocks amortizes the
/// [`DOF_WARMUP_BARS`] prefix over 64 usable samples, turning support fitting from an
/// `O(corpus)` read into an `O(max_samples)` one.
const SUPPORT_BLOCK: usize = 64;

/// RNG stream ids. Epoch shuffles use the epoch number itself as the stream, so every other
/// draw takes an id far outside any plausible epoch and can never collide with one.
const PINNED_STREAM: u64 = 0xE7A1_0000_0000_0001;
const SUPPORT_STREAM: u64 = 0xE7A1_0000_0000_0002;
const SUPPORT_ORDER_STREAM: u64 = 0xE7A1_0000_0000_0003;
/// Keys the per-epoch geometry of a [`PassPlan`]: which stage owns each stretch of a symbol,
/// where the symbol's uncoverable hole sits, and the issue order. Mixed with the seed FIRST
/// and with the epoch second, so a pass and a [`BarSampler`] epoch shuffle can never share a
/// stream for any `(seed, epoch)`.
const PASS_STREAM: u64 = 0xE7A1_0000_0000_0004;

// ---------------------------------------------------------------------------
// Bar conditioning ids
//
// Bar dynamics are dominated by the clock: the open/close volatility smile, the pre/post
// liquidity cliff, and the fact that ~42% of extended-hours bars carry no intra-bar shape at
// all. Without these ids the trunk has to average over regimes that behave nothing alike.
//
// The channels fall into two groups and the split is load-bearing:
//
// * EXOGENOUS — [`TIME_MINUTE`] through [`TIME_DAY_EDGE`]. A function of an exchange calendar
//   fixed before the decision, so free-running rollout timestamps come from
//   [`forecast_schedule_after`], never from future bars in the corpus.
// * OBSERVED — [`BAR_TIME_MARKET`]. The market proxy's own realized bar at the SAME instant
//   as the row's bar. Knowable only once that bar exists, so every future-facing constructor
//   emits [`MARKET_MISSING`] for it; see [`future_conditioning_ids`]. Reading the proxy's
//   NEXT bar here would be a lookahead leak of precisely the kind the global split exists to
//   prevent, and it would be invisible in a loss curve.
//
// Why the market enters at all: same-instant cross-symbol return correlation on this corpus
// is rho = 0.176 (`portfolio_cost::cross_correlation`), so ~17.6% of every name's return
// variance is a common factor that a single-symbol window cannot observe at all, and `r` is
// the only traded degree of freedom. The proxy's bar `t` does not reveal the factor at `t+1`,
// but it carries two things that do bear on `t+1`: the market-wide VOLATILITY REGIME, which
// is a far less noisy estimate of the name's conditional scale than the name's own recent
// `|r|` because volatility is the most strongly common of the moments, and LEAD-LAG, which is
// first-order for illiquid names at five minutes.
//
// Why bucketed ids rather than a widened `bar_dof_embed_w`. Three reasons, in order of
// weight. (1) The trunk's own bars enter primarily as 128-bin embeddings and only
// secondarily as a linear map, i.e. this model's established position is that a bar's
// nonlinear response deserves a table; a volatility regime and a piecewise-constant beta are
// exactly that kind of response. (2) A missing bucket is a RESERVED ROW, structurally
// distinct from any observed state, where a continuous channel would have to carry a
// side-band mask to avoid teaching the model that a coverage hole is a flat market.
// (3) Decisively: the id tensor is already threaded through every path that runs the trunk —
// training, the KV-cached session, `BarDynamics`, the planner, the horizon and portfolio
// scans — so the market channel inherits all of it and there is exactly ONE future-facing
// constructor to keep missing-safe. A new continuous tensor would add an argument at ~50
// sites, and a single omission in a future-facing one is a silent lookahead.
// ---------------------------------------------------------------------------

pub const BAR_TIME_FEATURES: usize = 9;
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
/// `floor(log2(bars elapsed since the previous bar)) + 1`, capped at [`TIME_ELAPSED_CAP`];
/// 0 means "no predecessor".
///
/// The bar axis is not a regular time grid — weekends, holidays, halts and the 20:00 -> 04:00
/// gap all break it — so `r` is a log return over an interval of UNKNOWN length. Without this
/// id an overnight return, a post-holiday return and an ordinary five-minute return are
/// pooled into one conditional distribution, which is a misspecification of the target rather
/// than merely a missing feature: their scales differ by `sqrt(delta_t)`, an order of
/// magnitude across the range this channel spans.
pub const TIME_ELAPSED: usize = 4;
/// 0 no predecessor, 1 same ET day as the previous bar, 2 first bar of a new ET day.
///
/// [`TIME_ELAPSED`] cannot answer this: a Monday 04:00 bar and a Friday 20:05 bar can carry
/// the same elapsed bucket, and a half-day or a halt puts an ordinary-looking gap inside one
/// day. [`et_local_day`] decides it exactly.
pub const TIME_DAY_EDGE: usize = 5;
/// Market-proxy log-return bucket, signed. [`MARKET_MISSING`] when the proxy has no bar at
/// this instant.
pub const TIME_MARKET_R: usize = 6;
/// Market-proxy log-range bucket. [`MARKET_MISSING`] when absent, 1 when the proxy's bar is
/// exactly flat, which is itself a strong liquidity statement.
pub const TIME_MARKET_S: usize = 7;
/// Market-proxy log relative-volume bucket, signed. [`MARKET_MISSING`] when absent.
pub const TIME_MARKET_W: usize = 8;

/// The OBSERVED channels, in tensor order. Every future-facing id constructor must leave all
/// of these at [`MARKET_MISSING`].
pub const BAR_TIME_MARKET: [usize; MARKET_FEATURES] = [TIME_MARKET_R, TIME_MARKET_S, TIME_MARKET_W];

/// The same id row with every market channel pinned to [`MARKET_MISSING`].
///
/// A bar that has not happened has no market row: the proxy's bar for that instant has not
/// printed either. Every future-facing id constructor here already reserves the row, but the
/// TRAINER reads its "next bar" ids out of the realized batch, where they ARE observed. Feeding
/// those to the one-step latent predictor would train it on a channel that is missing at every
/// deployment call site, which is a train/serve mismatch on the imagination path rather than a
/// leak. This is how the trainer drops them.
pub fn time_ids_without_market(time_ids: &Tensor) -> Tensor {
    let channels: [i64; MARKET_FEATURES] = BAR_TIME_MARKET.map(|slot| slot as i64);
    let index = Tensor::from_slice(&channels).to_device(time_ids.device());
    time_ids.index_fill(-1, &index, MARKET_MISSING)
}

pub const BAR_TIME_CARDINALITY: [i64; BAR_TIME_FEATURES] = [
    1440,
    7,
    4,
    8,
    TIME_ELAPSED_CAP + 1,
    3,
    NUM_BAR_BINS + 1,
    NUM_BAR_BINS + 1,
    NUM_BAR_BINS + 1,
];
pub const BAR_TIME_NAMES: [&str; BAR_TIME_FEATURES] = [
    "minute",
    "weekday",
    "session",
    "resolution",
    "elapsed",
    "day_edge",
    "market_r",
    "market_s",
    "market_w",
];
/// Lineage tag for the checkpoint metadata. Bump it whenever any id's meaning moves.
pub const BAR_TIME_CONDITIONING: &str =
    "et-minute-weekday-session4-resclass-elapsed13-dayedge-spy-rsw-v2";
/// ET minutes at which [`TIME_SESSION`] changes: 04:00, 09:30, 16:00, 20:00.
pub const SESSION_BOUNDARY_MINUTES: [i64; 4] = [240, 570, 960, 1200];
/// Resolutions with a dedicated class id; the index is the id. Anything else maps to
/// [`RESOLUTION_CLASS_OTHER`].
pub const RESOLUTION_CLASS_SECS: [u32; 7] = [60, 300, 900, 1800, 3600, 14_400, 86_400];
pub const RESOLUTION_CLASS_OTHER: i64 = 7;

/// Bar tensors that must never be handed out separately: the DOF the model predicts and the
/// conditioning ids it conditions on, drawn from the same bars in the same order.
#[derive(Debug)]
pub struct BarBatch {
    /// `[N, L, BAR_DOF]` f32.
    pub dof: Tensor,
    /// `[N, L, BAR_TIME_FEATURES]` i64.
    pub time_ids: Tensor,
    /// `[N, L]` f32 causal `sigma_t`: the divisor `dof`'s two scale-carrying degrees of
    /// freedom were divided by, one per row.
    ///
    /// Always present and always exact, including under [`DofScaling::Raw`] where every entry
    /// is `1.0`. It is not a normalization bookkeeping detail: it is the bridge back to an
    /// economic return, `r = sigma * z_r`, and it is measurable from bars strictly before the
    /// row, so a head may condition on it and a sizer may consume it without leakage.
    pub sigma: Tensor,
    /// `[N, L, BAR_DOF]` f32 true realized raw targets before standardization or z clamping.
    ///
    /// Present only when [`Self::dof`] is standardized. It is materialized while building this
    /// batch from the already-decoded row and is never retained by the corpus. Raw batches use
    /// `dof` itself and leave this `None`, avoiding both a duplicate host allocation and a
    /// duplicate device tensor.
    pub raw_dof: Option<Tensor>,
    /// `[N, context, 2, BAR_TIME_FEATURES]` forecast-safe target clocks for direct h2/h3.
    ///
    /// `None` for general corpus windows and samplers carrying fewer than three future bars.
    /// Every row is derived only from its decision timestamp on the deterministic exchange
    /// schedule; realized future availability and market channels never enter it.
    pub direct_time_ids: Option<Tensor>,
    /// `[N, context, 2]` mask for h2/h3 targets whose intervening realized rows exactly
    /// occupy the deterministic scheduled slots. Gap/halt rows are excluded rather than
    /// pairing an ordinary-clock forecast with a later realized bar.
    pub direct_valid: Option<Tensor>,
    /// Bars in this batch whose [`BAR_TIME_MARKET`] channels are [`MARKET_MISSING`], out of
    /// `N * L`. Reported as `pretrain_market_coverage`: a market channel that is absent for
    /// most rows is a data problem, and it is not visible in any loss.
    pub market_missing: usize,
}

/// Host staging buffers for one batch, owned by the caller and reused across steps.
///
/// A stage-2 batch stages ~6.3 MiB of DOF, id, sigma and validity rows. Allocating that per
/// step faults in fresh zeroed pages the fill then overwrites in full, so the buffers live
/// here and are only ever `resize`d. Every region IS fully overwritten — [`fill_batch_row`]
/// writes all `len` slots of its row and the direct builder writes both horizons of all
/// `context` decisions — so the resize filler is never read.
#[derive(Debug, Default)]
pub struct BatchScratch {
    dof: Vec<f32>,
    time: Vec<i32>,
    sigma: Vec<f32>,
    raw_dof: Vec<f32>,
    direct_time: Vec<i32>,
    direct_valid: Vec<u8>,
}

/// Every calendar id is bounded by [`BAR_TIME_CARDINALITY`], so the host wire format can be
/// half the width of the int64 tensor the embeddings ultimately index with.
const _: () = {
    let mut feature = 0;
    while feature < BAR_TIME_FEATURES {
        assert!(BAR_TIME_CARDINALITY[feature] <= i32::MAX as i64);
        feature += 1;
    }
};

/// The CPU half of a [`BarBatch`]: every tensor built, nothing transferred.
///
/// Split out so a producer thread can encode a batch while the GPU is still on the previous
/// step, leaving the training thread only the copies. `Tensor::to_device` is a synchronous
/// copy on the default stream, so a producer that also transferred would serialize against
/// the training kernels instead of overlapping with them.
#[derive(Debug)]
pub struct HostSample {
    dof: Tensor,
    /// `[N, L, BAR_TIME_FEATURES]` i32, widened by [`Self::to_device`].
    time_ids: Tensor,
    sigma: Tensor,
    raw_dof: Option<Tensor>,
    /// `[direct_time_ids, direct_valid]` as i32 and u8, widened by [`Self::to_device`].
    direct: Option<(Tensor, Tensor)>,
    /// [`BarBatch::market_missing`], already counted on the host.
    pub market_missing: usize,
}

impl HostSample {
    /// Transfer to `device` and widen the staged ids to the dtypes [`BarBatch`] declares.
    ///
    /// The only part of a batch build that touches CUDA, so it is the only part that must run
    /// on the thread owning the training stream.
    pub fn to_device(self, device: Device) -> BarBatch {
        let (direct_time_ids, direct_valid) = match self.direct {
            Some((ids, valid)) => (
                Some(ids.to_device(device).to_kind(Kind::Int64)),
                Some(valid.to_device(device).to_kind(Kind::Bool)),
            ),
            None => (None, None),
        };
        BarBatch {
            dof: self.dof.to_device(device),
            time_ids: self.time_ids.to_device(device).to_kind(Kind::Int64),
            sigma: self.sigma.to_device(device),
            raw_dof: self.raw_dof.map(|raw| raw.to_device(device)),
            direct_time_ids,
            direct_valid,
            market_missing: self.market_missing,
        }
    }
}

/// The pretrainer builds a batch on a producer thread holding `&BarSampler` and moves the
/// finished [`HostSample`] to the training thread, so these are load-bearing rather than
/// incidental. `HostSample` is only `Send`: `tch::Tensor` is not `Sync`.
const _: fn() = || {
    fn shared<T: Send + Sync>() {}
    fn moved<T: Send>() {}
    shared::<BarSampler>();
    shared::<BatchScratch>();
    moved::<HostSample>();
};

pub fn resolution_class(res_secs: u32) -> i64 {
    RESOLUTION_CLASS_SECS
        .iter()
        .position(|&secs| secs == res_secs)
        .map_or(RESOLUTION_CLASS_OTHER, |index| index as i64)
}

/// Conditioning ids of a bar opening at `ts_ms` whose predecessor opened at `prev_ts_ms`.
///
/// Total in every id by construction — `minute` comes from a value reduced mod 86400, the
/// weekday from `rem_euclid(7)`, the session from an exhaustive branch, the resolution from a
/// lookup with an `other` fallback, the elapsed bucket from a capped `leading_zeros`, the day
/// edge from a three-way match and every market bucket from a `partition_point` over a fixed
/// edge table — so no timestamp and no bar, however corrupt, can produce an index outside
/// [`BAR_TIME_CARDINALITY`].
///
/// `prev_ts_ms` is `None` only for a bar with no predecessor in its file, which can never
/// carry a DOF, and for a caller that genuinely does not know it; both take the reserved
/// "unknown" row rather than being modelled as adjacent.
///
/// `market` is the CONTEMPORANEOUS proxy state, joined on exact timestamp equality. Passing
/// `None` yields [`MARKET_MISSING`] on every market channel, which is what
/// [`future_conditioning_ids`] does.
pub fn bar_time_ids(
    ts_ms: i64,
    prev_ts_ms: Option<i64>,
    res_secs: u32,
    market: Option<&MarketChannel>,
) -> [i64; BAR_TIME_FEATURES] {
    bar_time_ids_from(ts_ms, prev_ts_ms, res_secs, market, 0).0
}

/// [`bar_time_ids`] for a non-decreasing scan, carrying [`MarketChannel::ids_at_from`]'s
/// cursor forward across the bars of one window.
pub fn bar_time_ids_from(
    ts_ms: i64,
    prev_ts_ms: Option<i64>,
    res_secs: u32,
    market: Option<&MarketChannel>,
    cursor: usize,
) -> ([i64; BAR_TIME_FEATURES], usize) {
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
    let ([market_r, market_s, market_w], cursor) = match market {
        Some(channel) => channel.ids_at_from(ts_ms, cursor),
        None => ([MARKET_MISSING; MARKET_FEATURES], cursor),
    };
    let ids = [
        minute,
        weekday,
        session,
        resolution_class(res_secs),
        elapsed_bars_id(ts_ms, prev_ts_ms, res_secs),
        day_edge_id(ts_ms, prev_ts_ms),
        market_r,
        market_s,
        market_w,
    ];
    debug_assert!(
        ids.iter()
            .zip(BAR_TIME_CARDINALITY)
            .all(|(&id, cardinality)| (0..cardinality).contains(&id)),
        "conditioning ids {ids:?} escaped {BAR_TIME_CARDINALITY:?} for ts_ms {ts_ms}"
    );
    (ids, cursor)
}

/// Conditioning ids of a bar that has NOT happened yet.
///
/// The exogenous channels are a function of timestamps, so they are computed exactly as for a
/// realized bar. The observed channels are not knowable and take [`MARKET_MISSING`], which is
/// enforced structurally: this function has no way to name a [`MarketChannel`]. Future-facing
/// callers first construct timestamps through [`forecast_schedule_after`], whose API likewise
/// has no future corpus input.
pub fn future_conditioning_ids(
    ts_ms: i64,
    prev_ts_ms: Option<i64>,
    res_secs: u32,
) -> [i64; BAR_TIME_FEATURES] {
    bar_time_ids(ts_ms, prev_ts_ms, res_secs, None)
}

/// Whether `date` is a regular full-day US-equity trading date.
///
/// This is a deterministic exchange calendar, not an inference from which bars happened to
/// print. It covers the recurring NYSE full-day closures used by the corpus. Early closes are
/// deliberately left on the ordinary 04:00--20:00 extended-hours grid: the early close of the
/// core session does not close the whole extended-hours venue, and a fixed full-day grid is
/// preferable to letting future print availability decide whether a forecast step exists.
pub fn is_us_equity_trading_date(date: NaiveDate) -> bool {
    let table = &*TRADING_DATES;
    match usize::try_from(date.num_days_from_ce() - table.base_ce) {
        Ok(index) if index < table.days => {
            let (word, bit) = (index / 64, index % 64);
            table.bits[word] & (1 << bit) != 0
        }
        _ => computed_us_equity_trading_date(date),
    }
}

/// One bit per date over the [`ET_TRANSITIONS`] span, set for a full-day trading date.
///
/// [`computed_us_equity_trading_date`] rebuilds ten `NaiveDate`s and an Easter computus on
/// every call, and [`direct_forecast_schedule_ids`] asks about ~150k dates per batch. Every
/// bit here is produced BY that function, so the table cannot disagree with it, and a date
/// outside the span still falls through to it.
struct TradingDates {
    base_ce: i32,
    days: usize,
    bits: Vec<u64>,
}

static TRADING_DATES: std::sync::LazyLock<TradingDates> =
    std::sync::LazyLock::new(build_trading_dates);

fn build_trading_dates() -> TradingDates {
    let first = NaiveDate::from_ymd_opt(1990, 1, 1).expect("the offset table start is a date");
    let last = NaiveDate::from_ymd_opt(2100, 1, 1).expect("the offset table end is a date");
    let days = (last - first).num_days() as usize;
    let mut bits = vec![0u64; days.div_ceil(64)];
    let mut date = first;
    for index in 0..days {
        if computed_us_equity_trading_date(date) {
            let (word, bit) = (index / 64, index % 64);
            bits[word] |= 1 << bit;
        }
        date = date.succ_opt().expect("the table span stays representable");
    }
    TradingDates {
        base_ce: first.num_days_from_ce(),
        days,
        bits,
    }
}

fn computed_us_equity_trading_date(date: NaiveDate) -> bool {
    if matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
        return false;
    }
    let year = date.year();
    let observed = |month: u32, day: u32| {
        let holiday = NaiveDate::from_ymd_opt(year, month, day).expect("a fixed holiday is valid");
        match holiday.weekday() {
            Weekday::Sat => holiday - Duration::days(1),
            Weekday::Sun => holiday + Duration::days(1),
            _ => holiday,
        }
    };
    // NYSE observes a Sunday New Year's Day on Monday, but stays open on the Friday before
    // a Saturday New Year's Day (unlike the federal calendar).
    let new_year = NaiveDate::from_ymd_opt(year, 1, 1).expect("New Year is valid");
    let observed_new_year = if new_year.weekday() == Weekday::Sun {
        new_year + Duration::days(1)
    } else {
        new_year
    };
    if date == observed_new_year
        || date == nth_weekday(year, 1, Weekday::Mon, 3)
        || date == nth_weekday(year, 2, Weekday::Mon, 3)
        || date == easter_sunday(year) - Duration::days(2)
        || date == last_weekday(year, 5, Weekday::Mon)
        || (year >= 2022 && date == observed(6, 19))
        || date == observed(7, 4)
        || date == nth_weekday(year, 9, Weekday::Mon, 1)
        || date == nth_weekday(year, 11, Weekday::Thu, 4)
        || date == observed(12, 25)
    {
        return false;
    }
    true
}

fn nth_weekday(year: i32, month: u32, weekday: Weekday, nth: u32) -> NaiveDate {
    let first = NaiveDate::from_ymd_opt(year, month, 1).expect("a calendar month starts");
    let offset = (7 + weekday.num_days_from_monday() as i64
        - first.weekday().num_days_from_monday() as i64)
        % 7;
    first + Duration::days(offset + 7 * i64::from(nth - 1))
}

fn last_weekday(year: i32, month: u32, weekday: Weekday) -> NaiveDate {
    let next = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .expect("the next calendar month starts");
    let last = next - Duration::days(1);
    let offset = (7 + last.weekday().num_days_from_monday() as i64
        - weekday.num_days_from_monday() as i64)
        % 7;
    last - Duration::days(offset)
}

/// Gregorian Easter, using the Meeus/Jones/Butcher computus.
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
    NaiveDate::from_ymd_opt(year, month as u32, day as u32).expect("computus returns a date")
}

/// Deterministic extended-hours timestamps beginning strictly after `decision_ts_ms`.
///
/// Intraday resolutions use the 04:00--20:00 ET weekday grid and skip the full-day exchange
/// holidays in [`is_us_equity_trading_date`]. Daily and coarser resolutions retain the
/// decision's ET wall-clock time and advance across trading dates. No corpus, symbol, proxy or
/// future availability is consulted, so a halt or missing future file cannot move this clock.
fn forecast_schedule_local(decision_ts_ms: i64, res_secs: u32) -> NaiveDateTime {
    assert!(
        res_secs > 0,
        "a forecast schedule needs a positive resolution"
    );
    et_naive_local(decision_ts_ms)
}

fn advance_forecast_schedule(local: &mut NaiveDateTime, res_secs: u32) -> i64 {
    if res_secs >= 86_400 {
        loop {
            *local += Duration::days(1);
            if is_us_equity_trading_date(local.date()) {
                break;
            }
        }
    } else {
        *local += Duration::seconds(i64::from(res_secs));
        loop {
            let date = local.date();
            let seconds = i64::from(local.time().num_seconds_from_midnight());
            let in_session = (4 * 3600..20 * 3600).contains(&seconds);
            let trading = is_us_equity_trading_date(date);
            if trading && in_session {
                break;
            }
            let next_date = if trading && seconds < 4 * 3600 {
                date
            } else {
                date + Duration::days(1)
            };
            *local = next_date
                .and_hms_opt(4, 0, 0)
                .expect("04:00 is a valid local wall time");
        }
    }
    New_York
        .from_local_datetime(local)
        .single()
        .expect("the extended session does not cross a DST transition")
        .timestamp_millis()
}
/// Deterministic extended-hours timestamps beginning strictly after `decision_ts_ms`.
///
/// No corpus, symbol, proxy, or future availability is consulted.
pub fn forecast_schedule_after(decision_ts_ms: i64, steps: usize, res_secs: u32) -> Vec<i64> {
    assert!(steps > 0, "a forecast schedule needs at least one step");
    let mut local = forecast_schedule_local(decision_ts_ms, res_secs);
    (0..steps)
        .map(|_| advance_forecast_schedule(&mut local, res_secs))
        .collect()
}

/// Forecast-safe time IDs on the deterministic US-equity schedule.
pub fn forecast_schedule_ids_after(
    decision_ts_ms: i64,
    steps: usize,
    res_secs: u32,
) -> Vec<[i64; BAR_TIME_FEATURES]> {
    let schedule = forecast_schedule_after(decision_ts_ms, steps, res_secs);
    let mut previous = decision_ts_ms;
    schedule
        .into_iter()
        .map(|ts_ms| {
            let ids = future_conditioning_ids(ts_ms, Some(previous), res_secs);
            previous = ts_ms;
            ids
        })
        .collect()
}

fn direct_forecast_schedule_ids(
    decision_ts_ms: i64,
    res_secs: u32,
) -> [(i64, [i64; BAR_TIME_FEATURES]); 3] {
    let mut local = forecast_schedule_local(decision_ts_ms, res_secs);
    let mut previous = decision_ts_ms;
    std::array::from_fn(|_| {
        let timestamp = advance_forecast_schedule(&mut local, res_secs);
        let ids = future_conditioning_ids(timestamp, Some(previous), res_secs);
        previous = timestamp;
        (timestamp, ids)
    })
}

/// The deterministic scheduled bar immediately before `ts_ms`.
pub fn forecast_schedule_previous(ts_ms: i64, res_secs: u32) -> i64 {
    assert!(
        res_secs > 0,
        "a forecast schedule needs a positive resolution"
    );
    assert!(
        res_secs <= 16 * 3600 || res_secs >= 86_400,
        "intraday resolutions must fit inside the 16-hour extended session"
    );
    let mut local = et_naive_local(ts_ms);
    loop {
        if res_secs >= 86_400 {
            local -= Duration::days(1);
            while !is_us_equity_trading_date(local.date()) {
                local -= Duration::days(1);
            }
            break;
        }
        local -= Duration::seconds(i64::from(res_secs));
        let date = local.date();
        let seconds = i64::from(local.time().num_seconds_from_midnight());
        if is_us_equity_trading_date(date) && (4 * 3600..20 * 3600).contains(&seconds) {
            break;
        }
        let mut previous_date = if seconds < 4 * 3600 {
            date - Duration::days(1)
        } else {
            date
        };
        while !is_us_equity_trading_date(previous_date) {
            previous_date -= Duration::days(1);
        }
        local = previous_date
            .and_hms_opt(20, 0, 0)
            .expect("20:00 is a valid local wall time")
            - Duration::seconds(i64::from(res_secs));
        break;
    }
    New_York
        .from_local_datetime(&local)
        .single()
        .expect("the extended session does not cross a DST transition")
        .timestamp_millis()
}

/// Deterministic forecast-safe IDs beginning at the scheduled `first_ts_ms` itself.
pub fn forecast_schedule_ids_from(
    first_ts_ms: i64,
    steps: usize,
    res_secs: u32,
) -> Vec<[i64; BAR_TIME_FEATURES]> {
    assert!(steps > 0, "a forecast schedule needs at least one step");
    let mut timestamps = Vec::with_capacity(steps);
    timestamps.push(first_ts_ms);
    if steps > 1 {
        timestamps.extend(forecast_schedule_after(first_ts_ms, steps - 1, res_secs));
    }
    let mut previous = forecast_schedule_previous(first_ts_ms, res_secs);
    timestamps
        .into_iter()
        .map(|ts_ms| {
            let ids = future_conditioning_ids(ts_ms, Some(previous), res_secs);
            previous = ts_ms;
            ids
        })
        .collect()
}

/// Largest [`TIME_ELAPSED`] bucket. At 300s, bucket 11 covers 512..1023 bars (1.8..3.6 days of
/// wall clock across a weekend) and this one saturates everything from 2048 bars — a hair over
/// one calendar week of extended-hours grid — upwards, which is where multi-day corpus holes
/// live.
pub const TIME_ELAPSED_CAP: i64 = 12;

fn elapsed_bars_id(ts_ms: i64, prev_ts_ms: Option<i64>, res_secs: u32) -> i64 {
    let Some(prev) = prev_ts_ms else {
        return 0;
    };
    let step = i64::from(res_secs.max(1)) * 1000;
    let bars = ts_ms.saturating_sub(prev) / step;
    if bars < 1 {
        // Sub-resolution or non-increasing spacing. `write_bar_file` rejects the latter, so
        // this is the adjacent-bar case with a truncating divide, not a corpus defect.
        return 1;
    }
    // `floor(log2 bars) + 1`, i.e. 1 for one bar, 2 for two or three, and so on.
    (64 - i64::from((bars as u64).leading_zeros())).min(TIME_ELAPSED_CAP)
}

fn day_edge_id(ts_ms: i64, prev_ts_ms: Option<i64>) -> i64 {
    match prev_ts_ms {
        None => 0,
        Some(prev) if et_local_day(prev) == et_local_day(ts_ms) => 1,
        Some(_) => 2,
    }
}

// ---------------------------------------------------------------------------
// Market / common-factor channel
// ---------------------------------------------------------------------------

/// Observed market channels per bar: proxy `r`, `s` and `w`.
pub const MARKET_FEATURES: usize = 3;

/// Reserved row meaning "the proxy has no bar at this instant".
///
/// A hole must NOT decode to 0.0. Zero is a perfectly ordinary return, an ordinary
/// relative volume and a legitimate range, so mapping a coverage hole onto it would teach the
/// trunk that a hole is a flat, average-volume market. The proxy has real gaps — extended-hours
/// instants it never printed, plus the multi-day holes the corpus audit counts — so this row is
/// exercised, not decorative.
pub const MARKET_MISSING: i64 = 0;

/// The market / common-factor proxy, by symbol.
///
/// SPY at the deployment resolution: it is already an admitted member of `universe.json`, it is
/// the most heavily traded index vehicle on the tape so its extended-hours coverage is the best
/// available, and it is a stable choice in a way a corpus-wide cross-sectional statistic is not
/// — breadth computed over the symbol set would move with `Ingest` and with every
/// [`BarCorpus::load_restricted`] ablation, making two runs' channels incomparable.
pub const MARKET_PROXY_SYMBOL: &str = "SPY";

/// Bar-DOF slots the market channel carries, parallel to [`BAR_TIME_MARKET`].
///
/// `r` because it is the traded degree of freedom and the one the common factor lives in. `s`
/// because the market's realized range is the least noisy estimator of the volatility regime
/// the name's next bar will be drawn from, and volatility is the most strongly common of the
/// moments. `w` because index participation separates a real move from an illiquid drift, and
/// because it is what distinguishes an extended-hours instant that traded from one that merely
/// printed. `u` and `v` are left out: the index's intra-bar close/open position says nothing
/// about a single name that its own `u`/`v` do not already say.
const MARKET_DOF: [usize; MARKET_FEATURES] = [DOF_R, DOF_S, DOF_W];

/// Every market channel is `1 + bin_id` over a [`NUM_BAR_BINS`]-bin equal-mass support, with
/// row 0 reserved for [`MARKET_MISSING`].
const _: () = assert!(BAR_TIME_CARDINALITY[TIME_MARKET_R] == NUM_BAR_BINS + 1);
const _: () = assert!(BAR_TIME_CARDINALITY[TIME_MARKET_S] == NUM_BAR_BINS + 1);
const _: () = assert!(BAR_TIME_CARDINALITY[TIME_MARKET_W] == NUM_BAR_BINS + 1);
const _: () = assert!(BAR_TIME_CARDINALITY[TIME_ELAPSED] == TIME_ELAPSED_CAP + 1);
const _: () = assert!(NUM_BAR_BINS + 1 <= u8::MAX as i64);

/// Proxy bars encoded per `bin_ids` call. Bounds the transient to a few hundred kilobytes
/// instead of `proxy_bars * BAR_DOF * 12` bytes; the proxy is one symbol but it is the LONGEST
/// symbol in the corpus, and this machine has been OOM-killed for less.
const MARKET_ENCODE_CHUNK: usize = 1 << 16;

/// Bucketed market-proxy state keyed by bar open timestamp.
///
/// Built once per corpus from the proxy's own packed file, streamed through the SAME
/// [`for_each_window_dof`] the trainer encodes with, so the channel measures exactly the `r`,
/// `s` and `w` a row's own DOF would carry rather than a second, subtly different definition of
/// them.
///
/// The buckets are the SAME object the model uses for its own bars: a [`NUM_BAR_BINS`]-bin
/// EQUAL-MASS [`BarSupports`] fitted by [`BarSupports::fit`] on the proxy's own distribution
/// over the TRAIN region only. Equal-mass rather than a hand-chosen grid over bps because a
/// linear grid would spend most of its rows on states the proxy almost never visits while
/// pooling the centre where it lives, and this repo has already measured that bins beat
/// continuous families on exactly this data. Fitted on the PROXY's distribution rather than
/// reusing the corpus support because SPY is far quieter than the median corpus symbol, so the
/// corpus support would leave most of its 128 rows unvisited.
///
/// Ids narrow to one byte each, so the whole channel is a couple of megabytes resident
/// regardless of corpus size, and building it costs one pass over one symbol — never a second
/// pass over the corpus.
pub struct MarketChannel {
    /// Ascending bar open timestamps of the proxy, parallel to `ids`.
    ts_ms: Vec<i64>,
    ids: Vec<[u8; MARKET_FEATURES]>,
    /// SHA-256 over the bucket GEOMETRY, folded into [`BarCorpus::identity_fingerprint`] so a
    /// silently refitted market support shows up in every artifact that records the corpus.
    support_sha256: String,
}

impl std::fmt::Debug for MarketChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarketChannel")
            .field("bars", &self.ts_ms.len())
            .field("first", &self.ts_ms.first().copied().map(iso_ms))
            .field("last", &self.ts_ms.last().copied().map(iso_ms))
            .field("support_sha256", &self.support_sha256)
            .finish()
    }
}

impl MarketChannel {
    fn new(file: &BarFile, supports: &BarSupports) -> Self {
        let bars = file.bars();
        let usable = bars.len().saturating_sub(1);
        let mut ts_ms = Vec::with_capacity(usable);
        let mut ids = Vec::with_capacity(usable);
        if usable >= 1 {
            let mut pending: Vec<f32> = Vec::with_capacity(MARKET_ENCODE_CHUNK * BAR_DOF);
            // RAW deliberately, on both this encode and `fit_market_supports`. The market
            // channel is forecast-safe CONTEXT with its own separately fitted buckets, never a
            // target, so it has nothing to gain from being in sigma units — and it is built
            // during `open_files`, before `with_dof_scaling` can have been called, so a
            // scaling-dependent channel would not even be expressible here.
            for_each_window_dof(bars, 1, usable, DofScaling::Raw, |bar, dof| {
                ts_ms.push(bar.ts());
                pending.extend_from_slice(&dof.dof.to_array());
                if pending.len() >= MARKET_ENCODE_CHUNK * BAR_DOF {
                    append_market_ids(supports, &pending, &mut ids);
                    pending.clear();
                }
            });
            append_market_ids(supports, &pending, &mut ids);
        }
        debug_assert_eq!(ts_ms.len(), ids.len());
        Self {
            ts_ms,
            ids,
            support_sha256: market_support_sha256(supports),
        }
    }

    /// Bars of the proxy that carry a channel value. Bar 0 has no predecessor close and so no
    /// DOF, hence one fewer than the proxy's file length.
    pub fn bars(&self) -> usize {
        self.ts_ms.len()
    }

    /// SHA-256 of the bucket geometry, for [`BarCorpus::identity_fingerprint`].
    pub fn support_sha256(&self) -> &str {
        &self.support_sha256
    }

    /// Ids of the proxy bar opening at exactly `ts_ms`, else [`MARKET_MISSING`] on every
    /// channel.
    ///
    /// EXACT equality, never a carry-forward of the last observed bar. A stale return is not a
    /// weaker version of the current return, it is a different quantity: returns do not
    /// persist, so carrying one forward across a three-hour gap asserts something false. And
    /// the strict join is what makes the causal argument a one-liner — the channel at a bar is
    /// a function of the proxy bar covering the SAME half-open interval and of nothing later.
    #[inline]
    pub fn ids_at(&self, ts_ms: i64) -> [i64; MARKET_FEATURES] {
        self.ids_at_from(ts_ms, 0).0
    }

    /// [`Self::ids_at`] for a non-decreasing scan, returning the cursor to pass to the next
    /// bar of the same window.
    ///
    /// A window's ~2000 bars otherwise pay ~2000 independent 21-probe bisections over the
    /// whole proxy vector, every probe a cache miss. Adjacent bars land one slot apart, so a
    /// bounded forward scan resolves the join outright; a corpus hole falls back to a
    /// bisection over the suffix rather than walking it. `cursor` must never exceed the
    /// index this returned for an earlier, no-later timestamp.
    #[inline]
    pub fn ids_at_from(&self, ts_ms: i64, cursor: usize) -> ([i64; MARKET_FEATURES], usize) {
        const SCAN: usize = 16;
        // The scan only moves forward, so a cursor resolved for a LATER timestamp cannot be
        // walked back: the join would report the reserved row for a bar the proxy covers, and
        // a wrong conditioning id is invisible to every shape check and every loss. Necessary
        // condition of the precondition, and free in release.
        debug_assert!(
            cursor == 0
                || self
                    .ts_ms
                    .get(cursor - 1)
                    .is_some_and(|&previous| previous < ts_ms),
            "market cursor {cursor} was resolved for a timestamp later than {ts_ms}"
        );
        let base = cursor.min(self.ts_ms.len());
        let rest = &self.ts_ms[base..];
        let near = rest.len().min(SCAN);
        let offset = match rest[..near].iter().position(|&ts| ts >= ts_ms) {
            Some(offset) => offset,
            None if near < SCAN => near,
            None => near + rest[near..].partition_point(|&ts| ts < ts_ms),
        };
        let index = base + offset;
        let ids = if self.ts_ms.get(index) == Some(&ts_ms) {
            self.ids[index].map(i64::from)
        } else {
            [MARKET_MISSING; MARKET_FEATURES]
        };
        (ids, index)
    }
}

/// Bin one chunk of proxy DOF and append `1 + bin_id` for the [`MARKET_DOF`] slots.
fn append_market_ids(
    supports: &BarSupports,
    pending: &[f32],
    out: &mut Vec<[u8; MARKET_FEATURES]>,
) {
    if pending.is_empty() {
        return;
    }
    let rows = (pending.len() / BAR_DOF) as i64;
    let bins = supports.bin_ids(&Tensor::from_slice(pending).view([rows, BAR_DOF as i64]));
    let flat = Vec::<i64>::try_from(bins.reshape([-1]).to_kind(Kind::Int64))
        .expect("bin ids are a dense int64 tensor");
    for row in flat.chunks_exact(BAR_DOF) {
        out.push(MARKET_DOF.map(|dof| {
            debug_assert!((0..NUM_BAR_BINS).contains(&row[dof]));
            (MARKET_MISSING + 1 + row[dof]) as u8
        }));
    }
}

/// SHA-256 over every bucket edge of the [`MARKET_DOF`] channels.
///
/// The BUCKET GEOMETRY, not the artifact bytes: two files differing only in provenance describe
/// the same conditioning and must hash alike, while a refit that moves one edge must not.
fn market_support_sha256(supports: &BarSupports) -> String {
    let mut digest = DigestContext::new(&SHA256);
    digest.update(b"bar-market-supports-v1");
    digest.update(&NUM_BAR_BINS.to_le_bytes());
    for dof in MARKET_DOF {
        for bound in supports
            .lower_bounds(dof)
            .iter()
            .chain(supports.upper_bounds(dof))
        {
            digest.update(&bound.to_bits().to_le_bytes());
        }
    }
    hex_digest(digest)
}

/// Fit the proxy's equal-mass buckets from the TRAIN region only.
///
/// `train_end` is the proxy's first bar at or after the `train | val` instant, so bars
/// `1 .. train_end` are exactly the proxy bars whose DOF are wholly inside the train region —
/// each reads its predecessor's close, which is also before the boundary. Nothing about
/// validation or test can reach a bucket edge, which is the same discipline
/// [`BarCorpus::fit_supports`] enforces for the model's own supports.
///
/// Every train bar, not a sample: it is one symbol, so the fit is exact, has no seed to record
/// and cannot move between two runs over the same file.
fn fit_market_supports(file: &BarFile, train_end: usize) -> Result<BarSupports> {
    ensure!(
        train_end >= 2,
        "{MARKET_PROXY_SYMBOL} has {train_end} bars before the train | val instant, too few to \
         fit the market channel's buckets"
    );
    let mut samples = Vec::with_capacity(train_end - 1);
    for_each_window_dof(file.bars(), 1, train_end - 1, DofScaling::Raw, |_, dof| {
        samples.push(dof.dof)
    });
    Ok(BarSupports::fit(&samples))
}

/// Where the market channel's equal-mass buckets are persisted, beside the corpus and beside
/// the model's own `bar_supports.<res>.json`.
pub fn market_supports_path(dir: &Path, res_secs: u32) -> PathBuf {
    dir.join(format!("bar_market_supports.{res_secs}.json"))
}

/// Fingerprint of the proxy bars a fit of `sample_count` DOF consumed, and of nothing else.
///
/// NOT [`BarCorpus::identity_fingerprint`], for two independent reasons. It is unconstructible
/// here — that fingerprint folds in [`MarketChannel::support_sha256`], the very geometry this
/// validates, and it is a method on a corpus that does not exist yet at the only call site
/// ([`BarCorpus::open_files`] builds the channel before it builds `Self`). And it would be
/// wrong even if it existed: it moves with the symbol universe, which the market channel is
/// deliberately built BEFORE the restriction in order to be independent of.
///
/// Keyed on the RECORDED sample count, never on the calling run's `train | val` instant. Only
/// [`BarCorpus::load_with_bounds`] callers pin that instant; [`BarCorpus::load`] derives it from
/// a live percentile that moves with every ingested bar and with each caller's `min_bars`, so a
/// boundary-keyed fingerprint would let whichever entry point wrote the artifact first lock out
/// the planner, the panels and the universe rebuild — all of which must simply CONSUME the
/// geometry their checkpoint was trained on rather than re-derive it.
///
/// The fit region alone, never the whole file: `Ingest` appends to the proxy continuously and
/// bars past the fit region cannot reach a bucket edge. CONTENT, not metadata, so an in-place
/// backfill that revises history without shifting an endpoint is still caught.
///
/// Exactly the five fields [`for_each_window_dof`] consumes, plus the timestamp that says which
/// bar this is. `vwap` and `trades` are deliberately excluded: nothing in the fit reads them, so
/// hashing them would hard-error a corpus open over a vendor revision that provably cannot move
/// an edge.
fn market_proxy_fingerprint(file: &BarFile, res_secs: u32, sample_count: usize) -> Option<String> {
    let bars = file.bars().get(..sample_count + 1)?;
    let mut digest = DigestContext::new(&SHA256);
    digest.update(b"bar-market-proxy-v2");
    digest.update(MARKET_PROXY_SYMBOL.as_bytes());
    digest.update(&res_secs.to_le_bytes());
    digest.update(&(sample_count as u64).to_le_bytes());
    update_bar_encoding_content(&mut digest, bars);
    Some(hex_digest(digest))
}

/// Add the exact per-bar fields consumed by dataset encoding to `digest`.
///
/// `vwap` and `trades` are intentionally absent: no dataset encoding path reads them, while
/// timestamps and OHLCV determine calendar ids, DOF and causal volume state.
fn update_bar_encoding_content(digest: &mut DigestContext, bars: &[PackedBar]) {
    // Chunked so the digest sees one call per few thousand bars rather than one per field.
    let mut scratch =
        Vec::with_capacity(BAR_CONTENT_FINGERPRINT_CHUNK * BAR_CONTENT_FINGERPRINT_BYTES);
    for chunk in bars.chunks(BAR_CONTENT_FINGERPRINT_CHUNK) {
        scratch.clear();
        for bar in chunk {
            let bar = *bar;
            scratch.extend_from_slice(&bar.ts_ms.to_le_bytes());
            for field in [bar.open, bar.high, bar.low, bar.close, bar.volume] {
                scratch.extend_from_slice(&field.to_le_bytes());
            }
        }
        digest.update(&scratch);
    }
}

fn bar_encoding_content_sha256(file: &BarFile) -> String {
    let mut digest = DigestContext::new(&SHA256);
    digest.update(b"bar-encoding-content-v1");
    update_bar_encoding_content(&mut digest, file.bars());
    hex_digest(digest)
}

const BAR_CONTENT_FINGERPRINT_CHUNK: usize = 8_192;
const BAR_CONTENT_FINGERPRINT_BYTES: usize = 8 + 5 * 4;

/// Process-level opt-out from the provenance guard below, set by the global
/// `--freeze-market-supports`.
///
/// A process flag rather than a parameter because a corpus is opened from eight call sites —
/// the planner, the horizon and portfolio panels, ingest's boundary derivation — none of which
/// has any basis for deciding a campaign's freeze policy, and threading one through every
/// `BarCorpus` constructor would put the decision in the wrong hands. It belongs to the
/// invocation, and `main` sets it once before dispatch so every subcommand honours it.
static FREEZE_MARKET_SUPPORTS: AtomicBool = AtomicBool::new(false);

/// Reuse market buckets whose provenance does not match this proxy, loudly and on purpose.
/// The right call mid-campaign: refitting moves every market conditioning row, so a frozen
/// artifact is what keeps two runs' inputs comparable.
pub fn set_freeze_market_supports(freeze: bool) {
    FREEZE_MARKET_SUPPORTS.store(freeze, Ordering::Relaxed);
}

pub fn freeze_market_supports() -> bool {
    FREEZE_MARKET_SUPPORTS.load(Ordering::Relaxed)
}

/// Decide whether cached market buckets may be used against this proxy. Returns whether they
/// were accepted under the freeze despite a mismatch.
///
/// The same three cases [`crate::torch::train::pretrain`] applies to the model's own supports,
/// against the same [`BarSupportsProvenance`] record, so there is one convention on disk and
/// one decision rule: provenance matches -> reuse; mismatched or absent under the freeze ->
/// reuse under a warning; otherwise -> hard error. `corpus_fingerprint` carries a
/// [`market_proxy_fingerprint`] here, which shares no preimage with a corpus fingerprint
/// because the two digests are domain-separated.
///
/// The question asked is "is this artifact still a faithful fit of the proxy history on disk",
/// recomputed at the artifact's OWN `sample_count` — not "did this caller's boundary produce
/// it". A differing `train | val` instant is reported, never refused: the planner, the panels
/// and the universe rebuild all open the corpus on a derived boundary and must reuse the
/// geometry their checkpoint was trained on, and refusing them would leave deleting the
/// artifact as the only way forward, which re-means every conditioning row mid-campaign — the
/// exact harm this guard exists to prevent.
fn require_market_supports_provenance(
    provenance: Option<&BarSupportsProvenance>,
    path: &Path,
    file: &BarFile,
    res_secs: u32,
    bounds: (i64, i64),
    freeze: bool,
) -> Result<bool> {
    let short = |s: &str| s[..12.min(s.len())].to_owned();
    let complaint = match provenance {
        Some(recorded)
            if market_proxy_fingerprint(file, res_secs, recorded.sample_count)
                .is_some_and(|current| current == recorded.corpus_fingerprint) =>
        {
            println!(
                "[dataset] market channel buckets reused from {} — fitted {} from {} \
                 {MARKET_PROXY_SYMBOL} train bars, and that history is byte-identical today",
                path.display(),
                recorded.fitted_utc,
                recorded.sample_count
            );
            if recorded.split_bounds.0 != bounds.0 {
                // Not a refusal, but never silent: this is the discrepancy that hid an artifact
                // fitted five weeks short of the campaign pin. Past the boundary is the direction
                // that would mean the buckets saw held-out data, so it is called out by name.
                let leaks = file
                    .bars()
                    .get(recorded.sample_count)
                    .is_some_and(|bar| bar.ts() >= bounds.0);
                println!(
                    "[dataset] {}: buckets were fitted to a train|val instant of {}, this run \
                     uses {}{}",
                    if leaks { "WARNING" } else { "note" },
                    iso_ms(recorded.split_bounds.0),
                    iso_ms(bounds.0),
                    if leaks {
                        " — the fit region reaches AT OR PAST this run's boundary, so the bucket \
                         edges saw bars this run holds out"
                    } else {
                        " — the fit region ends strictly inside this run's train region, so no \
                         held-out bar reached an edge"
                    }
                );
            }
            return Ok(false);
        }
        Some(recorded) => format!(
            "were fitted from {} {MARKET_PROXY_SYMBOL} bars fingerprinting {} (split {} | {}), \
             but the first {} bars of {MARKET_PROXY_SYMBOL} on disk today do not reproduce it: \
             the proxy's history was deepened, revised or truncated under the artifact",
            recorded.sample_count,
            short(&recorded.corpus_fingerprint),
            iso_ms(recorded.split_bounds.0),
            iso_ms(recorded.split_bounds.1),
            recorded.sample_count,
        ),
        None => "carry no provenance at all, so nothing can confirm which proxy history or \
                 which split instant produced them"
            .to_owned(),
    };
    ensure!(
        freeze,
        "market channel buckets {} {complaint}. They bin {MARKET_PROXY_SYMBOL} into the \
         conditioning ids every symbol's rows carry, so reusing them against a proxy history \
         they were not fitted on silently re-means all {} market conditioning rows and makes \
         this run's inputs incomparable to the fit it claims. Delete the file to refit from \
         {MARKET_PROXY_SYMBOL}'s current train region, or pass --freeze-market-supports to \
         reuse them deliberately.",
        path.display(),
        MARKET_FEATURES as i64 * NUM_BAR_BINS
    );
    println!(
        "[dataset] WARNING: reusing FROZEN market channel buckets {} — they {complaint}. The \
         market conditioning stays comparable to other runs on these buckets and to nothing else.",
        path.display()
    );
    Ok(true)
}

/// Reuse the persisted market buckets if they exist, fit and persist them if they do not.
///
/// Load-then-reuse rather than always-refit, and that is the whole point of the artifact: the
/// corpus is live, `Ingest` appends to the proxy's file and the `train | val` percentile drifts
/// under it, so a refit on every run would silently re-mean all `3 * NUM_BAR_BINS` conditioning
/// rows from one run to the next and every checkpoint trained against the old rows would keep
/// loading. With the artifact pinned, the buckets move only when an operator deletes the file,
/// and when they do move [`MarketChannel::support_sha256`] moves the corpus fingerprint with
/// them.
///
/// Pinned is not the same as correct, which is what [`require_market_supports_provenance`]
/// adds: mere file existence once let an artifact fitted under one proxy history and split
/// instant be applied to another without a word, and a bin count cannot see that.
///
/// An unreadable or wrong-geometry artifact is a hard error, never a silent refit: refitting is
/// exactly the failure this function exists to prevent, and falling back to an all-MISSING
/// channel would delete a whole input group without failing anything.
fn load_or_fit_market_supports(
    dir: &Path,
    res_secs: u32,
    file: &BarFile,
    bounds: (i64, i64),
) -> Result<BarSupports> {
    let path = market_supports_path(dir, res_secs);
    let freeze = freeze_market_supports();
    if path.exists() {
        let supports = BarSupports::load(&path).with_context(|| {
            format!(
                "market channel buckets {} are unreadable; delete the file to refit them from \
                 {MARKET_PROXY_SYMBOL}'s current train region",
                path.display()
            )
        })?;
        ensure!(
            supports.num_bins() == NUM_BAR_BINS,
            "market channel buckets {} have {} bins, this build conditions on {NUM_BAR_BINS}",
            path.display(),
            supports.num_bins()
        );
        require_market_supports_provenance(
            supports.provenance(),
            &path,
            file,
            res_secs,
            bounds,
            freeze,
        )?;
        return Ok(supports);
    }
    // The same refusal `fit_supports_at` makes for the model's own supports: a freeze that
    // silently fits a brand-new geometry is the opposite of frozen.
    ensure!(
        !freeze,
        "--freeze-market-supports was given but {} does not exist; put the frozen artifact \
         there, or drop the flag to fit a new one from {MARKET_PROXY_SYMBOL}'s train region",
        path.display()
    );
    let train_end = file.index_at_or_after(bounds.0);
    let sample_count = train_end.saturating_sub(1);
    let fingerprint = market_proxy_fingerprint(file, res_secs, sample_count)
        .context("the proxy is shorter than the train region it was just cut against")?;
    let supports = fit_market_supports(file, train_end)?.with_provenance(BarSupportsProvenance {
        corpus_fingerprint: fingerprint,
        split_bounds: bounds,
        sample_count,
        fit_seed: None,
        scaling_contract: None,
        support_semantics: None,
        fitted_utc: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
    });
    match supports.save(&path) {
        Ok(()) => println!(
            "[dataset] fitted market channel buckets from {} {MARKET_PROXY_SYMBOL} train bars \
             -> {}",
            train_end.saturating_sub(1),
            path.display()
        ),
        Err(error) => eprintln!(
            "[dataset] fitted market channel buckets from {} {MARKET_PROXY_SYMBOL} train bars \
             but could not write {}: {error:#}. The next run WILL refit them against a longer \
             corpus and re-mean every market conditioning row.",
            train_end.saturating_sub(1),
            path.display()
        ),
    }
    Ok(supports)
}

/// America/New_York calendar day of a bar opening at `ts_ms`, as days since 1970-01-01 ET.
///
/// The same `local = utc + et_offset_secs(utc)` reduction [`bar_time_ids`] performs, exposed as a
/// day index because [`TIME_WEEKDAY`] cannot answer "is this the first bar of a trading day": a
/// weekday repeats every seven days, so a gap over a holiday week carries the same id on both
/// sides. A DAY INDEX changes across every session boundary and across none other, which is the
/// only form in which "first bar of the day" is decidable from two timestamps.
pub fn et_local_day(ts_ms: i64) -> i64 {
    let utc = ts_ms.div_euclid(1000);
    (utc + et_offset_secs(utc) as i64).div_euclid(SECS_PER_DAY)
}

/// America/New_York wall clock at `ts_ms`, from the same table [`et_local_day`] reads.
///
/// Identical to `New_York.timestamp_millis_opt(ts_ms).single().naive_local()` by construction:
/// [`ET_TRANSITIONS`] is built from chrono-tz's own offsets, and the local wall clock IS the
/// UTC instant shifted by that offset. Outside the table's span the tz query is kept, so a
/// timestamp the table cannot answer for is answered exactly rather than clamped.
fn et_naive_local(ts_ms: i64) -> NaiveDateTime {
    let utc = ts_ms.div_euclid(1000);
    if (ET_TABLE_FROM..ET_TABLE_TO).contains(&utc) {
        return chrono::DateTime::from_timestamp_millis(
            ts_ms + i64::from(et_offset_secs(utc)) * 1000,
        )
        .expect("an in-table instant shifted by its offset is representable")
        .naive_utc();
    }
    New_York
        .timestamp_millis_opt(ts_ms)
        .single()
        .expect("a timestamp has one New York representation")
        .naive_local()
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
    /// Lazily cached v3 identity. Bar-file v1 has no persisted content digest, so the first
    /// identity request hashes each retained mmap; clones share the result through `Arc<Corpus>`.
    identity_fingerprint: OnceLock<String>,
    /// Bucketed [`MARKET_PROXY_SYMBOL`] state, derived before any symbol restriction is
    /// applied so a symbol-universe ablation cannot silently delete the channel. `None` when
    /// the directory holds no proxy file at this resolution, in which case every row's market
    /// ids are [`MARKET_MISSING`] and the trunk sees an honestly absent channel.
    market: Option<MarketChannel>,
    /// Target parametrization every encode through this corpus applies.
    ///
    /// Lives on the corpus rather than on each accessor because it must be the SAME on every
    /// path a run touches: the supports fit, the training batches, the pinned evaluation sets
    /// and any rollout all have to agree, and a per-call argument would let one of them
    /// disagree silently. Selected once, right after load, by
    /// [`BarCorpus::with_dof_scaling`].
    dof_scaling: DofScaling,
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
            .field("dof_scaling", &self.inner.dof_scaling)
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
        // Proxy before restriction, for the same reason the bounds are: a symbol-universe
        // ablation must not change the market channel, or the two arms condition on different
        // exogenous state and their `nll_bar` stop being commensurable.
        let market = match files
            .iter()
            .find(|file| file.symbol() == MARKET_PROXY_SYMBOL)
        {
            Some(file) => {
                let supports = load_or_fit_market_supports(dir, res_secs, file, bounds)?;
                let channel = MarketChannel::new(file, &supports);
                println!(
                    "[dataset] market channel from {MARKET_PROXY_SYMBOL}.{res_secs}: {channel:?}"
                );
                Some(channel)
            }
            None => {
                println!(
                    "[dataset] no {MARKET_PROXY_SYMBOL}.{res_secs} file under {}: every row's \
                     market channel is MISSING",
                    dir.display()
                );
                None
            }
        };
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
                identity_fingerprint: OnceLock::new(),
                bounds,
                market,
                dof_scaling: DofScaling::Raw,
            }),
        })
    }

    /// Select the target parametrization every encode through this corpus applies.
    ///
    /// Consumed immediately after load, BEFORE any sampler is built: [`BarSampler`] clones the
    /// inner `Arc`, so the policy has to be settled while the corpus is still uniquely owned.
    /// Panics rather than silently mutating one view of a shared corpus, which would leave two
    /// accessors encoding the same bar two different ways.
    #[must_use]
    pub fn with_dof_scaling(mut self, scaling: DofScaling) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect(
                "the DOF scaling policy must be selected before any sampler or clone shares \
                 the corpus",
            )
            .dof_scaling = scaling;
        self
    }

    pub fn dof_scaling(&self) -> DofScaling {
        self.inner.dof_scaling
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

    /// Stream every DOF-carrying bar of one series through `sink`, in bar order, with the bar's
    /// own index, in RAW log-return units regardless of this corpus's
    /// [`BarCorpus::dof_scaling`].
    ///
    /// Raw unconditionally because its consumers are absolute-scale audits: a split/dividend
    /// seam is a threshold on `ln(C_t / C_{t-1})` in log points, and dividing that by a
    /// conditional volatility would make the threshold mean a different thing on every bar. Use
    /// [`Self::for_each_series_dof_scaled`] for anything that has to match the fit draw.
    /// Bar 0 is skipped because it has no predecessor close and therefore no DOF at all.
    ///
    /// Bounded by construction: the sink sees one bar at a time and nothing is buffered, so a
    /// pass over the whole corpus costs whatever the caller's accumulator costs and no more.
    pub fn for_each_series_dof(
        &self,
        series: usize,
        mut sink: impl FnMut(usize, &PackedBar, BarDof),
    ) {
        self.for_each_series_dof_with(series, DofScaling::Raw, |index, bar, row| {
            sink(index, bar, row.dof)
        });
    }

    /// [`Self::for_each_series_dof`] under this corpus's own scaling, handing the sink the
    /// causal `sigma_t` alongside the row.
    ///
    /// The whole-series form of the accessor [`Self::sample_train_dof`] draws blocks with, and
    /// it goes through the SAME [`for_each_window_dof`] so a full-corpus audit measures exactly
    /// the quantities the supports were fitted on rather than a second, subtly different
    /// definition of them.
    pub fn for_each_series_dof_scaled(
        &self,
        series: usize,
        sink: impl FnMut(usize, &PackedBar, StandardizedDof),
    ) {
        self.for_each_series_dof_with(series, self.inner.dof_scaling, sink);
    }

    fn for_each_series_dof_with(
        &self,
        series: usize,
        scaling: DofScaling,
        mut sink: impl FnMut(usize, &PackedBar, StandardizedDof),
    ) {
        let bars = self.inner.files[series].bars();
        if bars.len() < 2 {
            return;
        }
        let mut index = 1;
        for_each_window_dof(bars, 1, bars.len() - 1, scaling, |bar, dof| {
            sink(index, bar, dof);
            index += 1;
        });
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
            self.inner.market.as_ref(),
            self.inner.dof_scaling,
            device,
        ))
    }

    /// `[endpoints.len(), steps, BAR_TIME_FEATURES]` i64: deterministic exogenous conditioning
    /// after `bar + from_offset`, with every observed market channel at [`MARKET_MISSING`].
    ///
    /// This realized-rollout API requires all `steps` target bars to exist in the endpoint's
    /// series, but never reads their timestamps or contents. The conditioning clock itself comes
    /// from [`forecast_schedule_after`] using only the already-observed decision timestamp, so
    /// future symbol halts and print availability cannot move it. Free-running generation that
    /// does not select realized targets must use the deterministic schedule helpers directly.
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
            let decision = e
                .bar
                .checked_add(from_offset)
                .filter(|&bar| bar < bars.len())
                .with_context(|| {
                    format!(
                        "future_time_ids decision offset {from_offset} escapes {} at bar {}",
                        self.symbol(e.series),
                        e.bar
                    )
                })?;
            let _available_targets_end = decision
                .checked_add(steps)
                .and_then(|last| last.checked_add(1))
                .filter(|&end| end <= bars.len())
                .with_context(|| {
                    format!(
                        "future_time_ids wants {steps} bars after {} bar {decision}, but it has {}",
                        self.symbol(e.series),
                        bars.len()
                    )
                })?;
            for (slot, ids) in
                forecast_schedule_ids_after(bars[decision].ts(), steps, self.inner.res_secs)
                    .into_iter()
                    .enumerate()
            {
                out[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES].copy_from_slice(&ids);
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
        (end + 1)
            .checked_sub(len)
            .filter(|&s| s >= 1)
            .with_context(|| {
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

    /// Measure what the target parametrization does to `split`: the causal volatility
    /// estimator's own distribution, the scale the encoded `r` and `s` reach
    /// [`BarSupports::fit`] at, and the share of the `r` grid each SYMBOL occupies.
    ///
    /// The three panels this feeds are the direct test of the standardized parametrization,
    /// and they are measured on EVERY run rather than only on the standardized arm: a
    /// per-symbol occupancy spread means nothing without the control's spread beside it, and a
    /// chart the control does not write cannot supply one.
    ///
    /// `sigma_t` is therefore always measured under [`DofScaling::VolStandardized`] whatever
    /// the corpus trains on. Under [`DofScaling::Raw`] the divisor that was actually applied is
    /// identically one, so scanning the run's own parametrization would report a constant and
    /// say nothing about the estimator; measuring the estimator regardless makes the control's
    /// panel the answer to "what would the arm have divided by".
    ///
    /// `r` and `s` in contrast are measured under the corpus's OWN parametrization, because the
    /// question they answer is what the fitter received. `supports` must tile the same
    /// parametrization for the same reason.
    ///
    /// Costs one streamed pass over `split` per parametrization — two only when they differ —
    /// with the per-series causal warm-up [`for_each_window_dof`] already applies. Nothing
    /// scales with the corpus except the per-symbol row and the bounded quantile subsample.
    pub fn audit_target_geometry(
        &self,
        supports: &BarSupports,
        split: Split,
    ) -> Result<TargetGeometry> {
        let scaling = self.inner.dof_scaling;
        ensure!(
            supports.dof_scaling() == scaling,
            "the target-geometry audit bins `r` on the supplied support, so it must tile the \
             corpus's own parametrization: corpus is {scaling}, support is {}",
            supports.dof_scaling()
        );
        // `for_each_window_dof` refuses bar 0, which carries no predecessor close and therefore
        // no DOF at all.
        let ranges: Vec<(usize, usize)> = (0..self.inner.files.len())
            .map(|series| {
                let (lo, hi) = self.inner.split_range(series, split);
                let lo = lo.max(1);
                (lo, hi.max(lo))
            })
            .collect();
        let mut offsets = Vec::with_capacity(ranges.len());
        let mut bars = 0u64;
        for (lo, hi) in &ranges {
            offsets.push(bars);
            bars += (hi - lo) as u64;
        }
        // One deterministic uniform stride over the whole split, so the retained rows are a
        // sample of the SPLIT rather than of whichever symbols happen to be long.
        let stride = (bars / TARGET_QUANTILE_ROWS as u64).max(1);
        let warmup = DofScaling::VolStandardized.warmup_bars();
        // The binner rather than the support: `BarSupports` owns device tensors and is not
        // `Sync`, while `DofBinner` is a `Copy` view of the two edge slices and the atom table
        // and resolves the bin by exactly `BarSupports::bin_of`.
        let binner = supports.binner(DOF_R);
        let scans: Vec<SeriesGeometry> = (0..self.inner.files.len())
            .into_par_iter()
            .map(|series| {
                let file = &self.inner.files[series];
                let (lo, hi) = ranges[series];
                let mut scan = SeriesGeometry::new(file.symbol());
                if hi <= lo {
                    return scan;
                }
                let all = file.bars();
                let len = hi - lo;
                scan.bars = len as u64;
                // Structural, not scanned: `for_each_window_dof` seeds its estimator at
                // `anchor - warmup - 1` clamped to the series start, so a bar has a fully
                // converged reference exactly when its own index is at least `warmup`.
                scan.sigma_unwarmed = warmup.min(hi).saturating_sub(lo) as u64;
                let standardized = scaling.is_standardized();
                let mut position = offsets[series];
                for_each_window_dof_detailed(
                    all,
                    lo,
                    len,
                    DofScaling::VolStandardized,
                    |_, _, encoded| {
                        let keep = position % stride == 0;
                        position += 1;
                        scan.sigma_relative_floored += encoded.diagnostics.relative_floor as u64;
                        scan.sigma_numerical_fallback +=
                            encoded.diagnostics.numerical_fallback as u64;
                        scan.r_z_clamped += encoded.diagnostics.r_clamped as u64;
                        scan.s_z_clamped += encoded.diagnostics.s_clamped as u64;
                        if keep {
                            scan.sigma_rows.push(encoded.row.sigma);
                        }
                        if standardized {
                            scan.absorb_dof(&binner, &encoded.row.dof, keep);
                        }
                    },
                );
                if !standardized {
                    let mut position = offsets[series];
                    for_each_window_dof(all, lo, len, scaling, |_, row| {
                        let keep = position % stride == 0;
                        position += 1;
                        scan.absorb_dof(&binner, &row.dof, keep);
                    });
                }
                scan
            })
            .collect();
        Ok(TargetGeometry::reduce(
            self.inner.res_secs,
            scaling,
            split,
            scans,
        ))
    }

    /// SHA-256 over everything that decides which bars a split contains and what the model is
    /// conditioned on: the resolution, both split instants, every symbol's name and exact
    /// timestamp/OHLCV encoding content, and the market channel's bucket geometry. Fold this into
    /// any evaluation fingerprint — the corpus grows under running jobs, and a fingerprint blind
    /// to the symbol set or an interior vendor correction would compare different observations as
    /// if they were one.
    ///
    /// `v3` adds each ordinary symbol's [`bar_encoding_content_sha256`]. Bar-file v1 has no
    /// persisted digest, so the first identity request computes these from the mmap in parallel.
    /// The result is cached on the corpus, avoiding both work on opens that do not need lineage and
    /// rereads by repeated lineage consumers.
    ///
    /// `v2` added the market channel. `MarketChannel::support_sha256` is a geometry hash, not a
    /// file hash, so re-persisting the same buckets does not move the fingerprint while a refit
    /// that moves one edge does — which is the whole reason the buckets are pinned to an artifact
    /// rather than refitted per run.
    pub fn identity_fingerprint(&self) -> String {
        self.inner
            .identity_fingerprint
            .get_or_init(|| {
                // Indexed parallel collection preserves deterministic file order. Each worker
                // holds one bounded scratch chunk rather than materializing bar content.
                let content_sha256: Vec<String> = self
                    .inner
                    .files
                    .par_iter()
                    .map(bar_encoding_content_sha256)
                    .collect();
                let mut digest = DigestContext::new(&SHA256);
                digest.update(b"bar-corpus-v3");
                digest.update(&self.inner.res_secs.to_le_bytes());
                digest.update(&self.inner.bounds.0.to_le_bytes());
                digest.update(&self.inner.bounds.1.to_le_bytes());
                for (file, content_sha256) in self.inner.files.iter().zip(&content_sha256) {
                    digest.update(file.symbol().as_bytes());
                    digest.update(&(file.len() as u64).to_le_bytes());
                    digest.update(content_sha256.as_bytes());
                }
                match &self.inner.market {
                    Some(channel) => {
                        digest.update(MARKET_PROXY_SYMBOL.as_bytes());
                        digest.update(&(channel.bars() as u64).to_le_bytes());
                        digest.update(channel.support_sha256().as_bytes());
                    }
                    None => digest.update(b"no-market-channel"),
                }
                hex_digest(digest)
            })
            .clone()
    }

    /// The corpus's market channel, or `None` when the directory holds no
    /// [`MARKET_PROXY_SYMBOL`] file at this resolution.
    pub fn market_channel(&self) -> Option<&MarketChannel> {
        self.inner.market.as_ref()
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

    /// Conventional cache path for fitted supports at this corpus's resolution and target
    /// parametrization.
    ///
    /// The parametrization is in the FILENAME, not just inside the artifact. Two arms of a
    /// measurement campaign share a data directory, and a standardized fit landing on the
    /// control's `bar_supports.<res>.json` would silently redefine the control's output space
    /// on its next run. [`DofScaling::Raw`] keeps the historical name exactly, so no existing
    /// cache is orphaned.
    pub fn supports_path(&self) -> PathBuf {
        let res_secs = self.inner.res_secs;
        self.inner.dir.join(match self.inner.dof_scaling {
            DofScaling::Raw => format!("bar_supports.{res_secs}.json"),
            DofScaling::VolStandardized => format!("bar_supports.volstd.{res_secs}.json"),
        })
    }

    /// Fit equal-mass supports from the train region only, under this corpus's
    /// [`BarCorpus::dof_scaling`].
    ///
    /// This operation is pure with respect to the filesystem: the caller owns provenance and
    /// destination policy and must explicitly persist the returned artifact. Sampling never
    /// touches a bar at or after the `train | val` bound, so no normalization statistic can leak
    /// out of validation or test — and under [`DofScaling::VolStandardized`] neither can
    /// `sigma_t`, which is a causal function of bars strictly before each row.
    pub fn fit_supports(&self, max_samples: usize, seed: u64) -> BarSupports {
        let rows = self.sample_train_dof_scaled(max_samples, seed);
        match self.inner.dof_scaling {
            DofScaling::Raw => {
                let samples: Vec<BarDof> = rows.iter().map(|(_, row)| row.dof).collect();
                BarSupports::fit(&samples)
            }
            DofScaling::VolStandardized => {
                let samples: Vec<StandardizedDof> = rows.iter().map(|(_, row)| *row).collect();
                BarSupports::fit_standardized(&samples)
            }
        }
    }

    /// Deterministically draw up to `max_samples` `(ts_ms, dof)` pairs from the train region,
    /// as a uniform sample without replacement over length-[`SUPPORT_BLOCK`] blocks.
    ///
    /// Blocks rather than individual bars because every DOF needs a [`DOF_WARMUP_BARS`]
    /// causal prefix; a block amortizes that prefix over 64 samples while keeping the draw
    /// uniform over the train timeline.
    pub fn sample_train_dof(&self, max_samples: usize, seed: u64) -> Vec<(i64, BarDof)> {
        self.sample_train_dof_scaled(max_samples, seed)
            .into_iter()
            .map(|(ts, row)| (ts, row.dof))
            .collect()
    }

    /// [`Self::sample_train_dof`] with the causal `sigma_t` each row was divided by carried
    /// alongside it.
    ///
    /// The SAME draw, row for row: `sample_train_dof` is this function with the divisor
    /// dropped. Under [`DofScaling::Raw`] every `sigma` is exactly `1.0`, so the pair is total
    /// and the raw path is a special case rather than a separate code path.
    pub fn sample_train_dof_scaled(
        &self,
        max_samples: usize,
        seed: u64,
    ) -> Vec<(i64, StandardizedDof)> {
        let blocks = self.train_dof_blocks(max_samples, seed);
        self.flatten_train_blocks(&blocks, max_samples, |_, _, bar, dof| (bar.ts(), dof))
    }

    /// [`Self::sample_train_dof`] with the originating series and bar index carried alongside
    /// each row.
    ///
    /// The SAME draw, row for row and in the same order: both wrappers pick their blocks with
    /// [`Self::train_dof_blocks`] and flatten them with [`Self::flatten_train_blocks`], so there is
    /// no second RNG path and no second truncation rule to disagree with. It exists because a
    /// per-row classification made elsewhere in the corpus — a corporate-action seam, say — can
    /// only be joined back onto the draw by `(series, bar)`, and a timestamp alone does not
    /// identify a bar when 5,297 symbols share one calendar. Asserted in
    /// `the_located_draw_is_the_same_draw`.
    pub fn sample_train_dof_located(
        &self,
        max_samples: usize,
        seed: u64,
    ) -> Vec<(WindowRef, i64, BarDof)> {
        let blocks = self.train_dof_blocks(max_samples, seed);
        self.flatten_train_blocks(&blocks, max_samples, |symbol, bar_index, bar, dof| {
            (
                WindowRef {
                    symbol,
                    bar_index: bar_index as u32,
                },
                bar.ts(),
                dof.dof,
            )
        })
    }

    /// The block anchors the draw is taken from, in draw order and already truncated to the
    /// blocks `max_samples` rows can reach.
    fn train_dof_blocks(&self, max_samples: usize, seed: u64) -> Vec<WindowRef> {
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
        chosen.shuffle(&mut ChaCha12Rng::seed_from_u64(mix64(
            seed,
            SUPPORT_ORDER_STREAM,
        )));
        chosen.truncate(wanted as usize);

        chosen
            .iter()
            .map(|&block| {
                let symbol = cumulative.partition_point(|&c| c <= block) - 1;
                let local = block - cumulative[symbol];
                WindowRef {
                    symbol: symbol as u32,
                    bar_index: (1 + local as usize * SUPPORT_BLOCK) as u32,
                }
            })
            .collect()
    }

    /// Emit `row(series, bar_index, bar, dof)` for every bar of every chosen block, in block
    /// order, truncated to `max_samples` rows.
    fn flatten_train_blocks<T, F>(&self, blocks: &[WindowRef], max_samples: usize, row: F) -> Vec<T>
    where
        T: Send,
        F: Fn(u32, usize, &PackedBar, StandardizedDof) -> T + Send + Sync,
    {
        let inner = &self.inner;
        let scaling = inner.dof_scaling;
        let mut out: Vec<T> = blocks
            .par_iter()
            .flat_map_iter(|r| {
                let bars = inner.files[r.symbol as usize].bars();
                let anchor = r.bar_index as usize;
                let mut block = Vec::with_capacity(SUPPORT_BLOCK);
                for_each_window_dof(bars, anchor, SUPPORT_BLOCK, scaling, |bar, dof| {
                    block.push(row(r.symbol, anchor + block.len(), bar, dof));
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
    /// Number of bars after each decision row materialized in a batch. One is the legacy
    /// next-bar contract; direct multi-horizon pretraining requests three.
    future_bars: i64,
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
    /// Legacy one-step sampler.
    pub fn new(corpus: &BarCorpus, split: Split, context: i64, seed: u64) -> Self {
        Self::new_with_future(corpus, split, context, 1, seed)
    }

    /// Sampler whose windows carry `context` decision beliefs and `future_bars` aligned
    /// target bars. Anchors remain strided by `context`; only tails that cannot supply every
    /// requested future target are excluded.
    pub fn new_with_future(
        corpus: &BarCorpus,
        split: Split,
        context: i64,
        future_bars: i64,
        seed: u64,
    ) -> Self {
        assert!(context > 0, "context must be positive");
        assert!(future_bars > 0, "future_bars must be positive");
        let inner = corpus.inner.clone();
        let ctx = context as usize;
        let future = future_bars as usize;
        let mut anchors = Vec::new();
        let mut symbol_runs = Vec::with_capacity(inner.files.len());
        for symbol in 0..inner.files.len() {
            let start = anchors.len() as u32;
            let (lo, hi) = inner.split_range(symbol, split);
            // Bar 0 has no predecessor close, so it can never carry a DOF.
            let first = lo.max(1);
            if hi >= first + ctx + future {
                for a in (first..=hi - ctx - future).step_by(ctx) {
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
            future_bars,
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

    /// The target parametrization every batch this sampler builds is encoded in.
    pub fn dof_scaling(&self) -> DofScaling {
        self.corpus.dof_scaling
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

    /// `[batch, context + future_bars, ..]` DOF and calendar ids on `device`,
    /// bit-identical for a given `(seed, epoch, index, batch)` and reordered by `epoch`.
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
        let len = (self.context + self.future_bars) as usize;
        refs.par_iter().for_each(|r| readahead(self.slab(r, len)));
    }

    /// Empty host staging for this sampler, grown on first use and kept alive across steps.
    pub fn scratch(&self) -> BatchScratch {
        BatchScratch::default()
    }

    /// `[refs.len(), context + future_bars, ..]` DOF and calendar ids on `device`.
    pub fn batch_of(&self, refs: &[WindowRef], device: Device) -> BarBatch {
        self.batch_of_into(refs, device, &mut self.scratch())
    }

    /// [`Self::batch_of`] reusing caller-owned host staging.
    pub fn batch_of_into(
        &self,
        refs: &[WindowRef],
        device: Device,
        scratch: &mut BatchScratch,
    ) -> BarBatch {
        self.host_batch_of_into(refs, scratch).to_device(device)
    }

    /// Every host-side part of a batch: the encode, the calendar ids and the staged CPU
    /// tensors. Touches no device, so it can run a step ahead on a producer thread while the
    /// GPU is still consuming the previous batch.
    pub fn host_batch_of_into(&self, refs: &[WindowRef], scratch: &mut BatchScratch) -> HostSample {
        assert!(!refs.is_empty(), "cannot build an empty batch");
        let len = (self.context + self.future_bars) as usize;
        let rows: Vec<(usize, usize)> = refs
            .iter()
            .map(|r| (r.symbol as usize, r.bar_index as usize))
            .collect();
        let mut sample = build_host_batch(
            &self.corpus.files,
            &rows,
            len,
            self.corpus.res_secs,
            self.corpus.market.as_ref(),
            self.corpus.dof_scaling,
            scratch,
        );
        if self.future_bars >= 3 {
            sample.direct = Some(build_host_direct_forecast_time_ids(
                &self.corpus.files,
                &rows,
                self.context as usize,
                self.corpus.res_secs,
                scratch,
            ));
        }
        sample
    }

    /// One forecast-safe next-bar clock for every consecutive decision row in each window.
    ///
    /// Used by exact teacher-forced comparators that must match the rolling deployment cache
    /// without consulting the realized target timestamp.
    pub fn one_step_forecast_time_ids(
        &self,
        refs: &[WindowRef],
        decisions: usize,
        device: Device,
    ) -> Tensor {
        assert!(!refs.is_empty(), "forecast clocks need at least one window");
        assert!(
            decisions <= (self.context + self.future_bars) as usize,
            "requested forecast clocks escape the sampled window"
        );
        let rows: Vec<(usize, usize)> = refs
            .iter()
            .map(|reference| (reference.symbol as usize, reference.bar_index as usize))
            .collect();
        build_one_step_forecast_time_ids(
            &self.corpus.files,
            &rows,
            decisions,
            self.corpus.res_secs,
            device,
        )
    }

    /// Forecast-safe deterministic clocks after one observed row inside each sampled window.
    ///
    /// `decision_offset` is relative to [`WindowRef::bar_index`]. Only that already-observed
    /// timestamp is read; future timestamps and bar availability never enter the result.
    pub fn forecast_time_ids(
        &self,
        refs: &[WindowRef],
        decision_offset: usize,
        steps: i64,
        device: Device,
    ) -> Result<Tensor> {
        if refs.is_empty() {
            bail!("forecast_time_ids needs at least one window");
        }
        if steps <= 0 {
            bail!("forecast_time_ids steps must be positive, got {steps}");
        }
        let steps = steps as usize;
        let row = steps * BAR_TIME_FEATURES;
        let mut flat = vec![0i64; refs.len() * row];
        for (out, reference) in flat.chunks_mut(row).zip(refs) {
            let bars = self.corpus.files[reference.symbol as usize].bars();
            let decision = reference
                .bar_index
                .checked_add(decision_offset as u32)
                .map(|bar| bar as usize)
                .filter(|&bar| bar < bars.len())
                .with_context(|| {
                    format!(
                        "forecast decision offset {decision_offset} escapes {} at bar {}",
                        self.symbol(reference.symbol),
                        reference.bar_index
                    )
                })?;
            for (slot, ids) in
                forecast_schedule_ids_after(bars[decision].ts(), steps, self.corpus.res_secs)
                    .into_iter()
                    .enumerate()
            {
                out[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES].copy_from_slice(&ids);
            }
        }
        Ok(Tensor::from_slice(&flat)
            .view([refs.len() as i64, steps as i64, BAR_TIME_FEATURES as i64])
            .to_device(device))
    }

    /// Deterministic future clock with realized market-proxy channels for teacher-forced
    /// scoring only.
    ///
    /// The timestamp grid is still derived solely from the decision row. Unlike
    /// [`Self::forecast_time_ids`], this post-generation view joins market observations at
    /// those scheduled timestamps. It must never be passed to free-running generation.
    pub fn teacher_forced_future_time_ids(
        &self,
        refs: &[WindowRef],
        decision_offset: usize,
        steps: i64,
        device: Device,
    ) -> Result<Tensor> {
        if refs.is_empty() {
            bail!("teacher_forced_future_time_ids needs at least one window");
        }
        if steps <= 0 {
            bail!("teacher_forced_future_time_ids steps must be positive, got {steps}");
        }
        let steps = steps as usize;
        let row = steps * BAR_TIME_FEATURES;
        let mut flat = vec![0i64; refs.len() * row];
        for (out, reference) in flat.chunks_mut(row).zip(refs) {
            let bars = self.corpus.files[reference.symbol as usize].bars();
            let decision = reference
                .bar_index
                .checked_add(decision_offset as u32)
                .map(|bar| bar as usize)
                .filter(|&bar| bar < bars.len())
                .with_context(|| {
                    format!(
                        "teacher-forced decision offset {decision_offset} escapes {} at bar {}",
                        self.symbol(reference.symbol),
                        reference.bar_index
                    )
                })?;
            let mut previous = bars[decision].ts();
            for (slot, ts_ms) in
                forecast_schedule_after(bars[decision].ts(), steps, self.corpus.res_secs)
                    .into_iter()
                    .enumerate()
            {
                let ids = bar_time_ids(
                    ts_ms,
                    Some(previous),
                    self.corpus.res_secs,
                    self.corpus.market.as_ref(),
                );
                out[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES].copy_from_slice(&ids);
                previous = ts_ms;
            }
        }
        Ok(Tensor::from_slice(&flat)
            .view([refs.len() as i64, steps as i64, BAR_TIME_FEATURES as i64])
            .to_device(device))
    }

    /// Realized DOF aligned to the same deterministic schedule as [`Self::forecast_time_ids`].
    ///
    /// Alignment happens only after the forecast clock has been chosen. A missing scheduled
    /// print is a flat carry of the last mark; an observed bar keeps the slot named by its own
    /// timestamp, so a halt never compresses later returns toward the decision. Because an
    /// observed bar's `r` is encoded against the previous observed close, the first print after
    /// a halt carries the whole catch-up payoff exactly once.
    pub fn aligned_future_dof(
        &self,
        refs: &[WindowRef],
        decision_offset: usize,
        steps: i64,
        device: Device,
    ) -> Result<Tensor> {
        if refs.is_empty() {
            bail!("aligned_future_dof needs at least one window");
        }
        if steps <= 0 {
            bail!("aligned_future_dof steps must be positive, got {steps}");
        }
        let steps = steps as usize;
        let row = steps * BAR_DOF;
        let mut flat = vec![0f32; refs.len() * row];
        let carried = BarDof::default().to_array();
        for out in flat.chunks_mut(BAR_DOF) {
            out.copy_from_slice(&carried);
        }

        for (out, reference) in flat.chunks_mut(row).zip(refs) {
            let bars = self.corpus.files[reference.symbol as usize].bars();
            let decision = reference
                .bar_index
                .checked_add(decision_offset as u32)
                .map(|bar| bar as usize)
                .filter(|&bar| bar < bars.len())
                .with_context(|| {
                    format!(
                        "aligned future decision offset {decision_offset} escapes {} at bar {}",
                        self.symbol(reference.symbol),
                        reference.bar_index
                    )
                })?;
            let schedule =
                forecast_schedule_after(bars[decision].ts(), steps, self.corpus.res_secs);
            let observed_start = decision + 1;
            let observed_end = bars.partition_point(|bar| bar.ts() <= schedule[steps - 1]);
            if observed_start >= observed_end {
                continue;
            }

            let mut schedule_slot = 0usize;
            // Corpus scaling, not raw: these rows are SCORING TARGETS compared against a
            // rollout the model emits in its own target space, so the two must be the same
            // space. The flat-carry default is the origin of both spaces — `r = 0`, `s = 0`,
            // `u = v = 0.5` — so a missing print means the same thing either way.
            for_each_window_dof(
                bars,
                observed_start,
                observed_end - observed_start,
                self.corpus.dof_scaling,
                |bar, encoded| {
                    while schedule_slot < steps && schedule[schedule_slot] < bar.ts() {
                        schedule_slot += 1;
                    }
                    if schedule_slot < steps && schedule[schedule_slot] == bar.ts() {
                        out[schedule_slot * BAR_DOF..(schedule_slot + 1) * BAR_DOF]
                            .copy_from_slice(&encoded.dof.to_array());
                        schedule_slot += 1;
                    }
                },
            );
        }
        Ok(Tensor::from_slice(&flat)
            .view([refs.len() as i64, steps as i64, BAR_DOF as i64])
            .to_device(device))
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

// ---------------------------------------------------------------------------
// Ramp pass plan: ONE disjoint tiling of a split across several contexts
// ---------------------------------------------------------------------------

/// One symbol's addressable stretch of a split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SymbolAxis {
    /// First bar that may anchor a window: `max(split_lo, 1)`.
    first: u32,
    /// Bars `first + 1 ..= split_hi - 1`, i.e. every bar of this symbol that a window lying
    /// wholly inside the split could ever PREDICT. The anchor bar itself is an input only.
    targets: u32,
    /// Bars of `targets` no assigned window covers, because the axis is not an exact sum of
    /// ramp contexts. Strictly below the shortest context by construction, and placed at a
    /// uniformly random block boundary that is redrawn every epoch.
    hole: u32,
}

/// Split bars a pass cannot make a prediction target, by cause.
///
/// `covered + head + short_symbol + hole == split bars`, exactly. That conservation law is
/// what turns coverage from a chart into an invariant: a bar was either predicted this pass or
/// it is in a named bucket with a count against it, and nothing is allowed to fall between the
/// two. [`CoverageAudit::require_full_pass`] checks the equality every epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassRemainder {
    /// Bars that can never be a prediction target at ANY context. Bar 0 of a file carries no
    /// DOF — there is no predecessor close to diff against — and the first window's anchor bar
    /// is an input, never a target. Two bars per symbol in a split that starts at the file's
    /// first record, one per symbol in a split that starts mid-file.
    pub head_bars: u64,
    /// Bars of symbols whose whole split axis is shorter than the SHORTEST ramp context, so
    /// not one window of any stage fits inside the split. Reported rather than silently
    /// skipped: at `--min-bars 20480` there are none, but a smaller corpus, a thin auxiliary
    /// resolution or a symbol listed days before the split instant produces them.
    pub short_symbol_bars: u64,
    /// How many symbols that is.
    pub short_symbols: usize,
    /// Summed [`SymbolAxis::hole`]: the sub-shortest-context hole every tiled symbol carries
    /// because `targets` is not an exact sum of ramp contexts.
    pub hole_bars: u64,
}

impl PassRemainder {
    pub fn total(&self) -> u64 {
        self.head_bars + self.short_symbol_bars + self.hole_bars
    }
}

/// A DISJOINT tiling of one split's predictable bars across the ramp's contexts: the object
/// that makes "one epoch" mean "every training bar was a prediction target exactly once".
///
/// **What this replaces.** The pretrainer used to build one [`BarSampler`] per ramp stage, each
/// with its own stride-`C` anchor list over the WHOLE split, and let each stage walk its own
/// list from index 0. The token budget splits across the ramp, so no stage got through its
/// list: at `--epochs 1`, base batch 24 and the flat ramp this card derives, the three stages
/// issued 83,400 / 83,376 / 83,376 windows against 408,306 / 247,488 / 177,143 available, i.e.
/// 20% / 34% / 47% of their lists. The three covered subsets were pseudorandom and mutually
/// independent, so their union came to 71.3% of the training corpus and the bars it did reach
/// were seen unevenly: 28.7% of training bars were never a prediction target at all, 45.9%
/// once, 22.3% twice and 3.2% three times. None of that is visible in a loss curve, and the
/// only thing that ever reported it was a coverage chart nobody read.
///
/// **How the partition is built.**
///
/// * **Bars are partitioned, not anchors.** Stride equals context, so the target spans of a
///   symbol's consecutive windows tile its axis exactly: the window at `a` targets
///   `a+1 ..= a+C` and the next window starts at `a+C`. A pass therefore lays a sequence of
///   BLOCKS along each symbol's axis, each block one whole window of one stage, and a bar
///   belongs to whichever block covers it. Partitioning bars is what keeps the tiling exact
///   across stages whose strides differ — a partition of stage-local anchor indices could not
///   be, because index `k` of the 896-bar list and index `k` of the 2048-bar list address
///   different bars.
/// * **Each stage's share is its token budget.** Stage `s` is entitled to
///   `batch[s] * context[s] / sum_j batch[j] * context[j]` of the pass's bars, which is
///   exactly the fraction of the run's bar-tokens its steps consume, because every stage runs
///   the same number of steps. A running deficit is carried ACROSS symbols rather than reset
///   per symbol, so the realized split lands within one window of the target GLOBALLY instead
///   of within one window per symbol per stage.
/// * **Geometry and order stay ChaCha-keyed by `(seed, epoch)`.** Which stage owns a given
///   stretch of a symbol, where that symbol's uncoverable hole sits, and the order windows are
///   handed out in are all drawn from a counter-based stream, never a thread RNG. So a pass is
///   bit-reproducible and decorrelated from corpus layout at the same time.
///
/// **Why the counts are epoch-independent while the geometry is not.** Per-stage window counts
/// are a property of the corpus and the ramp, so the step schedule can be derived from
/// [`Self::windows_per_stage`] once and every epoch then consumes exactly that many windows —
/// which is what makes the coverage invariant checkable rather than approximate. What DOES
/// move with the epoch is where inside each symbol each stage's blocks sit, so a given bar's
/// conditioning depth and its stage are redrawn every pass.
pub struct PassPlan {
    corpus: Arc<Corpus>,
    split: Split,
    contexts: Vec<i64>,
    seed: u64,
    axes: Vec<SymbolAxis>,
    /// `counts[symbol * contexts.len() + stage]`: windows of `stage` assigned to `symbol`.
    counts: Vec<u32>,
    windows_per_stage: Vec<usize>,
    covered_bars: u64,
    split_bars: u64,
    remainder: PassRemainder,
    layout: RwLock<Option<Arc<PassLayout>>>,
}

impl std::fmt::Debug for PassPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassPlan")
            .field("split", &self.split)
            .field("contexts", &self.contexts)
            .field("seed", &self.seed)
            .field("windows_per_stage", &self.windows_per_stage)
            .field("covered_bars", &self.covered_bars)
            .field("split_bars", &self.split_bars)
            .field("remainder", &self.remainder)
            .finish()
    }
}

impl PassPlan {
    /// Partition `split` across `contexts`, giving stage `s` a `token_weights[s]` share of the
    /// bars. `token_weights` need not be normalized; it is the per-stage bar-token budget,
    /// `batch[s] * context[s]`, and the caller passes the ramp it will actually RUN, not the
    /// one it declared.
    pub fn new(
        corpus: &BarCorpus,
        split: Split,
        contexts: &[i64],
        token_weights: &[f64],
        seed: u64,
    ) -> Result<Self> {
        Self::new_with_future(corpus, split, contexts, token_weights, 1, seed)
    }

    /// As [`Self::new`], while reserving `future_bars - 1` split-local bars after each
    /// primary target span for direct multi-horizon supervision.
    pub fn new_with_future(
        corpus: &BarCorpus,
        split: Split,
        contexts: &[i64],
        token_weights: &[f64],
        future_bars: usize,
        seed: u64,
    ) -> Result<Self> {
        ensure!(future_bars > 0, "a pass plan needs at least one future bar");
        ensure!(
            !contexts.is_empty(),
            "a pass plan needs at least one ramp context"
        );
        ensure!(
            contexts.len() == token_weights.len(),
            "{} ramp contexts against {} token weights",
            contexts.len(),
            token_weights.len()
        );
        ensure!(
            contexts.iter().all(|&context| context > 0),
            "ramp contexts must be positive: {contexts:?}"
        );
        let weight_sum: f64 = token_weights.iter().copied().sum();
        ensure!(
            token_weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0)
                && weight_sum > 0.0,
            "token weights must be finite, non-negative and not all zero: {token_weights:?}"
        );
        let weights: Vec<f64> = token_weights
            .iter()
            .map(|weight| weight / weight_sum)
            .collect();
        let stages = contexts.len();
        let min_context = *contexts.iter().min().expect("contexts are non-empty") as usize;
        let inner = corpus.inner.clone();

        let mut axes = Vec::with_capacity(inner.files.len());
        let mut counts = vec![0u32; inner.files.len() * stages];
        let mut windows_per_stage = vec![0usize; stages];
        let mut remainder = PassRemainder::default();
        let mut split_bars = 0u64;
        let mut covered_bars = 0u64;
        // Bars OWED to each stage, carried across symbols. Reset per symbol, every symbol
        // would round independently and the global split could drift by up to one window per
        // symbol per stage — 5,297 windows out of 82,919 on the 5-minute corpus. Carried, the
        // drift is bounded by one window in total.
        let mut deficit = vec![0f64; stages];
        for symbol in 0..inner.files.len() {
            let (lo, hi) = inner.split_range(symbol, split);
            // Bar 0 has no predecessor close, so it can never carry a DOF, and the anchor bar
            // is an input rather than a target.
            let first = lo.max(1);
            let span = hi.saturating_sub(lo);
            let targets = hi.saturating_sub(first + future_bars);
            debug_assert!(span >= targets);
            ensure!(
                hi <= u32::MAX as usize,
                "symbol {} has {hi} bars, past the u32 bar index a WindowRef carries",
                inner.symbols[symbol]
            );
            split_bars += span as u64;
            remainder.head_bars += (span - targets) as u64;

            if targets < min_context {
                remainder.short_symbol_bars += targets as u64;
                if targets > 0 {
                    remainder.short_symbols += 1;
                }
                axes.push(SymbolAxis {
                    first: first as u32,
                    targets: targets as u32,
                    hole: targets as u32,
                });
                continue;
            }

            for (owed, weight) in deficit.iter_mut().zip(weights.iter()) {
                *owed += weight * targets as f64;
            }
            let mut remaining = targets;
            loop {
                // The stage owed the most WINDOWS among those that still fit. Windows and not
                // bars: a stage 2,048 bars behind is one window behind at the deployed context
                // and two and a bit at the ramp's start, and it is windows the schedule buys.
                let mut chosen = usize::MAX;
                let mut best_owed = f64::NEG_INFINITY;
                for stage in 0..stages {
                    let context = contexts[stage] as usize;
                    if context > remaining {
                        continue;
                    }
                    let owed = deficit[stage] / contexts[stage] as f64;
                    if owed > best_owed {
                        best_owed = owed;
                        chosen = stage;
                    }
                }
                let Some(stage) = (chosen != usize::MAX).then_some(chosen) else {
                    break;
                };
                counts[symbol * stages + stage] += 1;
                windows_per_stage[stage] += 1;
                deficit[stage] -= contexts[stage] as f64;
                remaining -= contexts[stage] as usize;
            }
            debug_assert!(remaining < min_context);
            remainder.hole_bars += remaining as u64;
            covered_bars += (targets - remaining) as u64;
            axes.push(SymbolAxis {
                first: first as u32,
                targets: targets as u32,
                hole: remaining as u32,
            });
        }

        ensure!(
            covered_bars + remainder.total() == split_bars,
            "pass plan lost bars: {covered_bars} covered + {} remainder != {split_bars} \
             {split} bars",
            remainder.total()
        );
        Ok(Self {
            corpus: inner,
            split,
            contexts: contexts.to_vec(),
            seed,
            axes,
            counts,
            windows_per_stage,
            covered_bars,
            split_bars,
            remainder,
            layout: RwLock::new(None),
        })
    }

    pub fn split(&self) -> Split {
        self.split
    }

    pub fn contexts(&self) -> &[i64] {
        &self.contexts
    }

    /// Windows each stage is assigned per pass. The step schedule is derived from this, so an
    /// epoch consumes each stage's share exactly once.
    pub fn windows_per_stage(&self) -> &[usize] {
        &self.windows_per_stage
    }

    /// Bars a full pass makes a prediction target. This — NOT the split's bar count — is the
    /// honest denominator for "how much of one epoch has been delivered": the difference is
    /// [`Self::remainder`], and no schedule can reach it.
    pub fn covered_bars(&self) -> u64 {
        self.covered_bars
    }

    pub fn split_bars(&self) -> u64 {
        self.split_bars
    }

    pub fn remainder(&self) -> &PassRemainder {
        &self.remainder
    }

    /// Mean bars of history a target bar of `stage` is predicted from.
    ///
    /// With stride equal to context, the `j`-th target of a window is predicted from exactly
    /// `j` input bars, `j = 1..=context`, so the mean is `(context + 1) / 2` and it differs by
    /// stage — 448.5, 736.5 and 1024.5 bars on the deployed ramp. Since the stages own disjoint
    /// shares, which one a bar lands in decides how much history it is predicted from, which is
    /// exactly why the assignment has to be independent of symbol, calendar position and
    /// liquidity.
    pub fn mean_conditioning_bars(&self, stage: usize) -> f64 {
        (self.contexts[stage] as f64 + 1.0) / 2.0
    }

    /// Bar-weighted mean conditioning length over the whole pass.
    pub fn pass_mean_conditioning_bars(&self) -> f64 {
        if self.covered_bars == 0 {
            return 0.0;
        }
        let weighted: f64 = (0..self.contexts.len())
            .map(|stage| {
                (self.windows_per_stage[stage] as f64 * self.contexts[stage] as f64)
                    * self.mean_conditioning_bars(stage)
            })
            .sum();
        weighted / self.covered_bars as f64
    }

    /// Bars stage `s` owns this pass, and the share of the covered bars that is.
    pub fn stage_bar_shares(&self) -> Vec<f64> {
        let covered = self.covered_bars.max(1) as f64;
        (0..self.contexts.len())
            .map(|stage| {
                self.windows_per_stage[stage] as f64 * self.contexts[stage] as f64 / covered
            })
            .collect()
    }

    /// This pass's anchors for `epoch`, per stage, in issue order.
    ///
    /// Cached: one epoch's layout is built once and shared by `Arc`, and a new epoch replaces
    /// it. Building it costs one pass over the assigned windows — 250k pushes and three
    /// shuffles on the 5-minute corpus, microseconds — so it is rebuilt rather than kept for
    /// every epoch at once.
    pub fn layout(&self, epoch: usize) -> Arc<PassLayout> {
        {
            let guard = self.layout.read().expect("pass layout lock");
            if let Some(layout) = guard.as_ref() {
                if layout.epoch == epoch {
                    return layout.clone();
                }
            }
        }
        let mut guard = self.layout.write().expect("pass layout lock");
        match guard.as_ref() {
            Some(layout) if layout.epoch == epoch => layout.clone(),
            _ => {
                let built = Arc::new(self.build_layout(epoch));
                *guard = Some(built.clone());
                built
            }
        }
    }

    fn build_layout(&self, epoch: usize) -> PassLayout {
        let stages = self.contexts.len();
        let mut rng =
            ChaCha12Rng::seed_from_u64(mix64(mix64(self.seed, PASS_STREAM), epoch as u64));
        let mut per_stage: Vec<Vec<WindowRef>> = self
            .windows_per_stage
            .iter()
            .map(|&count| Vec::with_capacity(count))
            .collect();
        let mut order: Vec<u8> = Vec::new();
        // Where each symbol's hole landed THIS pass. The only difference between two completed
        // passes over one symbol: the blocks tile the reachable region identically and exactly
        // one contiguous run of `axis.hole` bars is skipped, so a hole start per symbol is the
        // entire cross-pass difference and [`PassCensus`] needs nothing else to reconstruct
        // exposure counts exactly. `u32::MAX` marks a symbol no window ever targets.
        let mut hole_starts = vec![u32::MAX; self.axes.len()];
        for (symbol, axis) in self.axes.iter().enumerate() {
            order.clear();
            for stage in 0..stages {
                let count = self.counts[symbol * stages + stage] as usize;
                order.resize(order.len() + count, stage as u8);
            }
            if order.is_empty() {
                continue;
            }
            // A uniformly random block ORDER is what decorrelates a bar's stage — and therefore
            // its conditioning depth — from its position in the symbol's timeline. A
            // deficit-driven order would be a near-periodic 0,1,2,0,1,2 pattern, which would
            // pin the first bars of every symbol to the shortest context in every epoch.
            order.shuffle(&mut rng);
            // The hole goes at a uniformly random block boundary rather than at the tail. Left
            // at the tail it would always be the bars immediately before the split instant —
            // the most recent training bars, the ones a market model can least afford to
            // skip — and the SAME bars in every epoch.
            let hole_at = rng.random_range(0..=order.len());
            let mut cursor = axis.first as usize;
            for (index, &stage) in order.iter().enumerate() {
                if index == hole_at {
                    hole_starts[symbol] = cursor as u32;
                    cursor += axis.hole as usize;
                }
                per_stage[stage as usize].push(WindowRef {
                    symbol: symbol as u32,
                    bar_index: cursor as u32,
                });
                cursor += self.contexts[stage as usize] as usize;
            }
            if hole_at == order.len() {
                hole_starts[symbol] = cursor as u32;
                cursor += axis.hole as usize;
            }
            debug_assert_eq!(cursor, axis.first as usize + axis.targets as usize);
        }
        // One global shuffle per stage. Without it a batch would be 24 consecutive windows of
        // one symbol, which is one draw from the market rather than 24, and the gradient would
        // be a per-ticker gradient for a whole step.
        for windows in per_stage.iter_mut() {
            windows.shuffle(&mut rng);
        }
        PassLayout {
            epoch,
            stages: per_stage,
            hole_starts,
        }
    }

    /// Reconcile a pass against its ledger.
    ///
    /// Panics if the two describe different epochs, which would silently audit one pass against
    /// another's issue record.
    pub fn audit(&self, layout: &PassLayout, ledger: &PassLedger) -> CoverageAudit {
        assert_eq!(
            layout.epoch, ledger.epoch,
            "auditing the epoch-{} layout against the epoch-{} ledger",
            layout.epoch, ledger.epoch
        );
        let stages = self.contexts.len();
        let mut multiplicity_bars = [0u64; MULTIPLICITY_BUCKETS];
        let mut issued_per_stage = vec![0usize; stages];
        let mut repeated_per_stage = vec![0usize; stages];
        for stage in 0..stages {
            let context = self.contexts[stage] as u64;
            for &issues in &ledger.issued[stage] {
                multiplicity_bars[(issues as usize).min(MULTIPLICITY_BUCKETS - 1)] += context;
                if issues >= 1 {
                    issued_per_stage[stage] += 1;
                }
                if issues >= 2 {
                    repeated_per_stage[stage] += 1;
                }
            }
        }
        // The remainder is, by definition, bars no window targeted.
        multiplicity_bars[0] += self.remainder.total();
        CoverageAudit {
            epoch: layout.epoch,
            split: self.split,
            split_bars: self.split_bars,
            contexts: self.contexts.clone(),
            multiplicity_bars,
            windows_per_stage: self.windows_per_stage.clone(),
            issued_per_stage,
            repeated_per_stage,
            remainder: self.remainder,
            mean_conditioning_bars: (0..stages)
                .map(|stage| self.mean_conditioning_bars(stage))
                .collect(),
        }
    }

    /// Symbol names, for reports that name the worst offenders.
    pub fn symbol(&self, index: u32) -> &str {
        &self.corpus.symbols[index as usize]
    }

    /// Reconcile the WHOLE RUN so far — every completed pass in `census` plus the pass in
    /// progress — into one bars-by-exposure-count histogram.
    ///
    /// [`Self::audit`] cannot answer this and never could. It is a PER-PASS census, and
    /// [`CoverageAudit::require_full_pass`] enforces multiplicity exactly one WITHIN a pass, so
    /// its histogram reads a single spike at one on every epoch of a multi-epoch run — an
    /// arithmetically correct statement that reads, to anyone who does not check its scope, as
    /// "no bar was ever seen twice". This is the cross-pass counterpart, on the same bar-token
    /// convention and the same denominator, so the two can be drawn on one panel and the
    /// per-pass zeros cannot be mistaken for a claim about the run.
    ///
    /// EXACT, not sampled, and bounded: a completed pass tiles each symbol's reachable region
    /// with disjoint blocks and skips exactly one CONTIGUOUS run of `axis.hole` bars, so the
    /// per-pass hole start recorded in [`PassLayout`] is the entire difference between passes.
    /// Peak state is the sorted issued-block list, one entry per issued window.
    pub fn cumulative_coverage(
        &self,
        census: &PassCensus,
        layout: &PassLayout,
        ledger: &PassLedger,
    ) -> CumulativeCoverage {
        assert_eq!(
            layout.epoch, ledger.epoch,
            "auditing the epoch-{} layout against the epoch-{} ledger",
            layout.epoch, ledger.epoch
        );
        let stages = self.contexts.len();
        let completed = census.completed_passes();
        // Every issued block of the pass IN PROGRESS, keyed by symbol so each symbol's sweep
        // sees a contiguous slice. `issues` is added rather than treated as a flag: a repeated
        // window is a consumption bug and it must show up as an exposure of two, not be
        // flattened into one by the very census meant to reveal it.
        let mut issued_now: Vec<(u32, u32, u32, u8)> = Vec::new();
        let mut issued_bars_now = 0u64;
        for stage in 0..stages {
            let context = self.contexts[stage] as u32;
            for (index, &issues) in ledger.issued[stage].iter().enumerate() {
                if issues == 0 {
                    continue;
                }
                let window = layout.stages[stage][index];
                issued_now.push((window.symbol, window.bar_index, context, issues));
                issued_bars_now += context as u64 * issues as u64;
            }
        }
        issued_now.sort_unstable();

        let mut multiplicity_bars = [0u64; MULTIPLICITY_BUCKETS];
        let mut events: Vec<(u32, i32)> = Vec::new();
        let mut cursor = 0usize;
        for (symbol, axis) in self.axes.iter().enumerate() {
            let begin = cursor;
            while cursor < issued_now.len() && issued_now[cursor].0 as usize == symbol {
                cursor += 1;
            }
            if axis.targets == 0 {
                continue;
            }
            // A symbol shorter than the shortest ramp context is never a target at any epoch,
            // so its whole reachable region stays at exposure zero for the life of the run.
            if layout.hole_starts[symbol] == u32::MAX {
                multiplicity_bars[0] += axis.targets as u64;
                continue;
            }
            // ANCHOR-EXCLUSIVE, to match `audit` exactly. A window at `bar_index` PREDICTS bars
            // `bar_index + 1 ..= bar_index + context` — the anchor is an input, never a target —
            // and `audit` credits `context` bars per window on that basis. The same +1 applies to
            // the hole, whose skipped targets begin one bar past the cursor, and to the region,
            // which is `[first + 1, first + 1 + targets)`. Sweeping `[bar_index, ..)` instead
            // would be off by one bar per window against the per-pass census it has to be
            // comparable with, and the two panels would disagree by ~250k bars for no reason a
            // reader could see.
            events.clear();
            for pass in census.holes() {
                let start = pass[symbol];
                if start == u32::MAX || axis.hole == 0 {
                    continue;
                }
                events.push((start + 1, -1));
                events.push((start + 1 + axis.hole, 1));
            }
            for &(_, anchor, context, issues) in &issued_now[begin..cursor] {
                events.push((anchor + 1, issues as i32));
                events.push((anchor + 1 + context, -(issues as i32)));
            }
            events.sort_unstable();
            let begin_bar = axis.first + 1;
            let end = begin_bar + axis.targets;
            let mut level = completed as i32;
            let mut previous = begin_bar;
            for &(at, delta) in events.iter() {
                if at > previous {
                    let bucket = (level.max(0) as usize).min(MULTIPLICITY_BUCKETS - 1);
                    multiplicity_bars[bucket] += (at - previous) as u64;
                    previous = at;
                }
                level += delta;
            }
            if end > previous {
                let bucket = (level.max(0) as usize).min(MULTIPLICITY_BUCKETS - 1);
                multiplicity_bars[bucket] += (end - previous) as u64;
            }
        }
        // Bar 0 of every file and every anchor bar: input-only at any epoch, so exposure zero
        // for the run exactly as it is for a pass.
        multiplicity_bars[0] += self.remainder.head_bars;
        CumulativeCoverage {
            completed_passes: completed,
            in_progress_epoch: layout.epoch,
            split: self.split,
            split_bars: self.split_bars,
            covered_bars_per_pass: self.covered_bars,
            bar_target_events: completed as u64 * self.covered_bars + issued_bars_now,
            multiplicity_bars,
        }
    }
}

/// One epoch's anchors, per stage, in the order they will be issued.
pub struct PassLayout {
    epoch: usize,
    stages: Vec<Vec<WindowRef>>,
    /// First bar of each symbol's skipped run THIS pass, `u32::MAX` for a symbol no window
    /// targets. Its LENGTH is `SymbolAxis::hole`, a plan constant, so this one number per
    /// symbol is the whole of what differs between two completed passes.
    hole_starts: Vec<u32>,
}

impl std::fmt::Debug for PassLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassLayout")
            .field("epoch", &self.epoch)
            .field(
                "windows",
                &self.stages.iter().map(Vec::len).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PassLayout {
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    pub fn windows(&self, stage: usize) -> &[WindowRef] {
        &self.stages[stage]
    }

    /// The next at most `batch` windows of `stage` from `cursor`.
    ///
    /// The final draw of a stage is SHORT rather than dropped. Dropping the partial tail is
    /// what `BarSampler::batches_per_epoch` does and it is defensible for a sampler that wraps
    /// forever; for a pass it would leave up to `batch - 1` windows of every stage unissued,
    /// which is a coverage hole of up to one batch per stage per epoch and would make the
    /// invariant unachievable rather than merely violated. One short step per stage per epoch
    /// costs a slightly noisier gradient on 3 steps out of ~10,000.
    pub fn draw(&self, stage: usize, cursor: usize, batch: usize) -> &[WindowRef] {
        let windows = &self.stages[stage];
        let end = cursor.saturating_add(batch).min(windows.len());
        &windows[cursor.min(end)..end]
    }
}

/// Bars-by-issue-count buckets; the last one saturates, so index 3 means "three or more".
pub const MULTIPLICITY_BUCKETS: usize = 4;

/// Per-window issue counts for one pass: the ledger the coverage invariant is checked against.
///
/// One byte per assigned window — ~250 KB for the 5-minute corpus — marked where the batch is
/// BUILT rather than where a cursor advances, so a bug that skips or repeats a draw shows up as
/// a zero or a two instead of being hidden by the same counter that caused it. Per-window and
/// not per-bar because the plan's blocks are disjoint by construction, so a window's issue
/// count IS the issue count of each of its `context` target bars.
pub struct PassLedger {
    epoch: usize,
    issued: Vec<Vec<u8>>,
}

impl std::fmt::Debug for PassLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassLedger")
            .field("epoch", &self.epoch)
            .field(
                "windows",
                &self.issued.iter().map(Vec::len).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PassLedger {
    pub fn new(layout: &PassLayout) -> Self {
        Self {
            epoch: layout.epoch,
            issued: layout
                .stages
                .iter()
                .map(|windows| vec![0u8; windows.len()])
                .collect(),
        }
    }

    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Record that the `count` windows of `stage` starting at `cursor` went into a batch.
    pub fn mark(&mut self, stage: usize, cursor: usize, count: usize) {
        for slot in self.issued[stage].iter_mut().skip(cursor).take(count) {
            *slot = slot.saturating_add(1);
        }
    }

    /// Windows of `stage` issued at least once.
    pub fn issued(&self, stage: usize) -> usize {
        self.issued[stage].iter().filter(|&&n| n > 0).count()
    }
}

/// What one epoch actually covered, in BARS, reconciled against the split.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageAudit {
    pub epoch: usize,
    pub split: Split,
    pub split_bars: u64,
    /// The ramp contexts this pass tiled with, so a consumer can turn window counts into bars
    /// without holding the plan.
    pub contexts: Vec<i64>,
    /// Bars by how many times a window targeted them this epoch. Index 0 includes the whole
    /// [`PassRemainder`]; the last index saturates.
    pub multiplicity_bars: [u64; MULTIPLICITY_BUCKETS],
    pub windows_per_stage: Vec<usize>,
    pub issued_per_stage: Vec<usize>,
    /// Windows issued more than once, per stage. Zero on a healthy pass.
    pub repeated_per_stage: Vec<usize>,
    pub remainder: PassRemainder,
    /// `(context + 1) / 2` per stage: the mean history a target bar of that stage is predicted
    /// from.
    pub mean_conditioning_bars: Vec<f64>,
}

impl CoverageAudit {
    /// Bars targeted exactly once this epoch — the only multiplicity a pass is meant to produce.
    pub fn covered_bars(&self) -> u64 {
        self.multiplicity_bars[1]
    }

    /// Share of the split targeted exactly once.
    pub fn coverage_fraction(&self) -> f64 {
        self.covered_bars() as f64 / self.split_bars.max(1) as f64
    }

    /// Fraction of each stage's ASSIGNED windows that were issued. One on a complete pass; the
    /// series `pretrain_stage_coverage` charts.
    pub fn stage_coverage(&self) -> Vec<f64> {
        self.issued_per_stage
            .iter()
            .zip(self.windows_per_stage.iter())
            .map(|(&issued, &assigned)| issued as f64 / assigned.max(1) as f64)
            .collect()
    }

    /// Windows assigned but never issued, per stage.
    pub fn unissued_per_stage(&self) -> Vec<usize> {
        self.windows_per_stage
            .iter()
            .zip(self.issued_per_stage.iter())
            .map(|(&assigned, &issued)| assigned.saturating_sub(issued))
            .collect()
    }

    /// Bars the plan assigned to a window that was never issued: the coverage hole a schedule
    /// too short for its pass leaves behind.
    pub fn unissued_bars(&self) -> u64 {
        self.unissued_per_stage()
            .iter()
            .zip(self.contexts.iter())
            .map(|(&count, &context)| count as u64 * context as u64)
            .sum()
    }

    /// One line stating the whole pass: coverage, the multiplicity histogram, the named
    /// remainder and the per-stage conditioning depth.
    pub fn summary(&self) -> String {
        let stage_windows: Vec<String> = (0..self.windows_per_stage.len())
            .map(|stage| {
                format!(
                    "{}/{} @ {:.1} bars mean history",
                    self.issued_per_stage[stage],
                    self.windows_per_stage[stage],
                    self.mean_conditioning_bars[stage]
                )
            })
            .collect();
        format!(
            "epoch {} {} pass: {} of {} bars targeted exactly once ({:.4}%); multiplicity \
             0/1/2/3+ = {}/{}/{}/{} bars; unreachable {} bars = {} head + {} short-symbol ({} \
             symbols) + {} sub-context hole; stages [{}]",
            self.epoch,
            self.split,
            self.covered_bars(),
            self.split_bars,
            100.0 * self.coverage_fraction(),
            self.multiplicity_bars[0],
            self.multiplicity_bars[1],
            self.multiplicity_bars[2],
            self.multiplicity_bars[3],
            self.remainder.total(),
            self.remainder.head_bars,
            self.remainder.short_symbol_bars,
            self.remainder.short_symbols,
            self.remainder.hole_bars,
            stage_windows.join(" | "),
        )
    }

    /// The invariant: every bar of the split was a prediction target EXACTLY ONCE this epoch,
    /// or it sits in a named remainder bucket with a count against it.
    ///
    /// An error and not a chart, because the failure it catches is silent by construction.
    /// `pretrain_stage_coverage` reported the old sampler's 20/34/47% shortfall on every run
    /// for as long as it existed and nobody read it off the chart — it was eventually found in
    /// a doc comment. A run that trains on 71% of its corpus while reporting numbers as if it
    /// had trained on all of it invalidates every downstream comparison, so it must not be
    /// allowed to finish quietly.
    pub fn require_full_pass(&self) -> Result<()> {
        let accounted: u64 = self.multiplicity_bars.iter().sum();
        ensure!(
            accounted == self.split_bars,
            "coverage accounting lost bars in epoch {}: multiplicity buckets sum to {accounted} \
             against {} {} bars. {}",
            self.epoch,
            self.split_bars,
            self.split,
            self.summary()
        );
        let unissued = self.unissued_per_stage();
        ensure!(
            unissued.iter().all(|&count| count == 0),
            "epoch {} did not complete a pass: {unissued:?} of the {:?} windows assigned to each \
             ramp stage were never issued, leaving {} bars untargeted. An epoch that trains on \
             part of the corpus while reporting a full pass invalidates every number derived \
             from this run. {}",
            self.epoch,
            self.windows_per_stage,
            self.unissued_bars(),
            self.summary()
        );
        ensure!(
            self.repeated_per_stage.iter().all(|&count| count == 0),
            "epoch {} issued {:?} windows per ramp stage more than once, so {} bars were a \
             prediction target twice while others were not reached at all. The pass partition is \
             disjoint by construction, so this is a consumption bug, not a corpus property. {}",
            self.epoch,
            self.repeated_per_stage,
            self.multiplicity_bars[2] + self.multiplicity_bars[3],
            self.summary()
        );
        Ok(())
    }
}

/// Cross-pass exposure history: what the run as a whole has shown the model, as opposed to what
/// one pass showed it.
///
/// This type exists because of a specific, expensive failure. `pretrain_pass_multiplicity`
/// charts [`CoverageAudit::multiplicity_bars`], which is a PER-PASS census, and
/// [`CoverageAudit::require_full_pass`] guarantees it reads "one time: ~99.4%, two times: 0,
/// three or more: 0" on EVERY epoch. On a three-pass run that panel is correct and it also
/// reads, at face value, as an assertion that no bar was ever seen twice. It was believed over
/// `pretrain_unique_bar_reuse` sitting beside it, which correctly showed 1.0, then 2.0, then
/// 2.85 — and an entire analysis session proceeded on the false premise. A per-pass census read
/// as a per-run claim is not fixable by wording alone: the run-scoped number has to exist and be
/// drawn next to it.
///
/// BOUNDED BY CONSTRUCTION. A per-bar exposure counter would be one byte per training bar, 368
/// MB resident in the trainer for a diagnostic. Not needed: a completed pass tiles each symbol's
/// reachable region with disjoint blocks and skips exactly ONE contiguous run of `axis.hole`
/// bars, so the hole START is the entire difference between two completed passes, and the state
/// is one `u32` per symbol per completed pass — about 21 KB per pass on the 5-minute corpus.
/// This is EXACT, not an approximation of the counter.
#[derive(Clone, Debug, Default)]
pub struct PassCensus {
    /// One row per COMPLETED pass, in pass order; each row is [`PassLayout::hole_starts`].
    holes: Vec<Vec<u32>>,
}

impl PassCensus {
    /// Fold a pass that finished into the history.
    ///
    /// Call this ONLY after [`CoverageAudit::require_full_pass`] has passed for the same
    /// layout. The exactness of the reconstruction rests on the pass having been complete:
    /// "every block issued once, one hole skipped" is what makes a hole start sufficient. A
    /// truncated pass absorbed here would be recorded as a full one.
    pub fn absorb(&mut self, layout: &PassLayout) {
        self.holes.push(layout.hole_starts.clone());
    }

    pub fn completed_passes(&self) -> usize {
        self.holes.len()
    }

    fn holes(&self) -> &[Vec<u32>] {
        &self.holes
    }
}

/// What the RUN has covered, in bars, across every pass so far.
///
/// Same bar-token convention and same denominator as [`CoverageAudit`], so the two are directly
/// comparable on one panel. Deliberately NOT merged into `CoverageAudit`: that type's whole
/// contract is one pass, `require_full_pass` asserts one-pass properties on it, and widening it
/// to sometimes mean the run is how the original confusion would be re-created inside the type
/// system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CumulativeCoverage {
    /// Passes that finished and were audited complete.
    pub completed_passes: usize,
    /// The epoch of the pass in progress, whose partial coverage is included.
    pub in_progress_epoch: usize,
    pub split: Split,
    pub split_bars: u64,
    /// Bars ONE full pass targets. The denominator of [`Self::effective_epochs`].
    pub covered_bars_per_pass: u64,
    /// Bar-target events delivered so far: `completed_passes * covered + issued this pass`.
    pub bar_target_events: u64,
    /// Bars by how many times the RUN has targeted them, 0/1/2/3-or-more. The last saturates,
    /// so a fourth pass reads as "3 or more" rather than silently wrapping.
    pub multiplicity_bars: [u64; MULTIPLICITY_BUCKETS],
}

impl CumulativeCoverage {
    /// Passes over the corpus the run has delivered, counting the partial one in progress.
    ///
    /// Above one means classical multi-epoch reuse is live and every held-out comparison in the
    /// run is a comparison against a model that has seen its training bars more than once.
    pub fn effective_epochs(&self) -> f64 {
        self.bar_target_events as f64 / self.covered_bars_per_pass.max(1) as f64
    }

    /// Share of the split the run has targeted MORE THAN ONCE. Zero for a single-pass run by
    /// construction, and the one number whose non-zero value contradicts the per-pass panel.
    pub fn reused_fraction(&self) -> f64 {
        (self.multiplicity_bars[2] + self.multiplicity_bars[3]) as f64
            / self.split_bars.max(1) as f64
    }

    /// The accounting invariant, stated on the RUN rather than on a pass: every split bar sits
    /// in exactly one exposure bucket.
    ///
    /// Not a weaker form of [`CoverageAudit::require_full_pass`] and not a substitute for it —
    /// that one asserts a pass was COMPLETE, this one asserts the cross-pass reconstruction did
    /// not lose or double-count a bar. A reconstruction that silently lost bars would understate
    /// reuse, which is the direction that caused the original error.
    pub fn require_accounted(&self) -> Result<()> {
        let accounted: u64 = self.multiplicity_bars.iter().sum();
        ensure!(
            accounted == self.split_bars,
            "cross-pass coverage accounting lost bars after {} completed passes (epoch {} in \
             progress): exposure buckets sum to {accounted} against {} {} bars. {}",
            self.completed_passes,
            self.in_progress_epoch,
            self.split_bars,
            self.split,
            self.summary()
        );
        Ok(())
    }

    /// One line stating what the RUN has shown the model, in the same shape as
    /// [`CoverageAudit::summary`] so the two read side by side.
    pub fn summary(&self) -> String {
        format!(
            "run so far: {} completed passes + epoch {} in progress = {:.4} effective epochs \
             ({} bar-target events / {} bars in one pass); RUN exposure 0/1/2/3+ = {}/{}/{}/{} \
             bars; {:.4}% of the {} split has been targeted MORE THAN ONCE",
            self.completed_passes,
            self.in_progress_epoch,
            self.effective_epochs(),
            self.bar_target_events,
            self.covered_bars_per_pass,
            self.multiplicity_bars[0],
            self.multiplicity_bars[1],
            self.multiplicity_bars[2],
            self.multiplicity_bars[3],
            100.0 * self.reused_fraction(),
            self.split,
        )
    }
}

/// Encode `len` bars starting at `start` for every `(series, start)` row into one
/// [`BarBatch`]. The DOF and the calendar ids come off the same bar in the same pass, which is
/// the whole reason the two tensors are returned together.
fn build_one_step_forecast_time_ids(
    files: &[BarFile],
    rows: &[(usize, usize)],
    decisions: usize,
    res_secs: u32,
    device: Device,
) -> Tensor {
    let output_row = decisions * BAR_TIME_FEATURES;
    let mut flat = vec![0i64; rows.len() * output_row];
    flat.par_chunks_mut(output_row)
        .zip(rows.par_iter())
        .for_each(|(output, &(series, start))| {
            let bars = files[series].bars();
            for decision in 0..decisions {
                let ids = direct_forecast_schedule_ids(bars[start + decision].ts(), res_secs)[0].1;
                let offset = decision * BAR_TIME_FEATURES;
                output[offset..offset + BAR_TIME_FEATURES].copy_from_slice(&ids);
            }
        });
    Tensor::from_slice(&flat)
        .view([
            rows.len() as i64,
            decisions as i64,
            BAR_TIME_FEATURES as i64,
        ])
        .to_device(device)
}

fn build_host_direct_forecast_time_ids(
    files: &[BarFile],
    rows: &[(usize, usize)],
    context: usize,
    res_secs: u32,
    scratch: &mut BatchScratch,
) -> (Tensor, Tensor) {
    let time_row = context * 2 * BAR_TIME_FEATURES;
    let valid_row = context * 2;
    scratch.direct_time.resize(rows.len() * time_row, 0);
    scratch.direct_valid.resize(rows.len() * valid_row, 0);
    let BatchScratch {
        direct_time,
        direct_valid,
        ..
    } = scratch;
    direct_time
        .par_chunks_mut(time_row)
        .zip(direct_valid.par_chunks_mut(valid_row))
        .zip(rows.par_iter())
        .for_each(|((time_output, valid_output), &(series, start))| {
            let bars = files[series].bars();
            for decision in 0..context {
                let horizons = direct_forecast_schedule_ids(bars[start + decision].ts(), res_secs);
                let time_offset = decision * 2 * BAR_TIME_FEATURES;
                time_output[time_offset..time_offset + BAR_TIME_FEATURES]
                    .copy_from_slice(&horizons[1].1.map(|id| id as i32));
                time_output[time_offset + BAR_TIME_FEATURES..time_offset + 2 * BAR_TIME_FEATURES]
                    .copy_from_slice(&horizons[2].1.map(|id| id as i32));
                let mut ordinary = true;
                for step in 0..3 {
                    ordinary &= bars[start + decision + step + 1].ts() == horizons[step].0;
                    if step >= 1 {
                        valid_output[decision * 2 + step - 1] = u8::from(ordinary);
                    }
                }
            }
        });
    let n = rows.len() as i64;
    let context = context as i64;
    let time_ids =
        Tensor::from_slice(direct_time.as_slice()).view([n, context, 2, BAR_TIME_FEATURES as i64]);
    let valid = Tensor::from_slice(direct_valid.as_slice()).view([n, context, 2]);
    (time_ids, valid)
}

fn build_host_batch(
    files: &[BarFile],
    rows: &[(usize, usize)],
    len: usize,
    res_secs: u32,
    market: Option<&MarketChannel>,
    scaling: DofScaling,
    scratch: &mut BatchScratch,
) -> HostSample {
    let dof_row = len * BAR_DOF;
    let time_row = len * BAR_TIME_FEATURES;
    let standardized = scaling.is_standardized();
    let raw_len = if standardized {
        rows.len() * dof_row
    } else {
        0
    };
    scratch.dof.resize(rows.len() * dof_row, 0.0);
    scratch.time.resize(rows.len() * time_row, 0);
    scratch.sigma.resize(rows.len() * len, 0.0);
    scratch.raw_dof.resize(raw_len, 0.0);
    let BatchScratch {
        dof,
        time,
        sigma,
        raw_dof,
        ..
    } = scratch;
    let market_missing: usize = if standardized {
        dof.par_chunks_mut(dof_row)
            .zip(time.par_chunks_mut(time_row))
            .zip(sigma.par_chunks_mut(len))
            .zip(raw_dof.par_chunks_mut(dof_row))
            .zip(rows.par_iter())
            .map(
                |((((dof_out, time_out), sigma_out), raw_out), &(series, start))| {
                    fill_batch_row(
                        files,
                        series,
                        start,
                        len,
                        res_secs,
                        market,
                        scaling,
                        dof_out,
                        time_out,
                        sigma_out,
                        Some(raw_out),
                    )
                },
            )
            .sum()
    } else {
        dof.par_chunks_mut(dof_row)
            .zip(time.par_chunks_mut(time_row))
            .zip(sigma.par_chunks_mut(len))
            .zip(rows.par_iter())
            .map(|(((dof_out, time_out), sigma_out), &(series, start))| {
                fill_batch_row(
                    files, series, start, len, res_secs, market, scaling, dof_out, time_out,
                    sigma_out, None,
                )
            })
            .sum()
    };
    let n = rows.len() as i64;
    let len = len as i64;
    HostSample {
        dof: Tensor::from_slice(dof.as_slice()).view([n, len, BAR_DOF as i64]),
        time_ids: Tensor::from_slice(time.as_slice()).view([n, len, BAR_TIME_FEATURES as i64]),
        sigma: Tensor::from_slice(sigma.as_slice()).view([n, len]),
        raw_dof: standardized
            .then(|| Tensor::from_slice(raw_dof.as_slice()).view([n, len, BAR_DOF as i64])),
        direct: None,
        market_missing,
    }
}

fn build_batch(
    files: &[BarFile],
    rows: &[(usize, usize)],
    len: usize,
    res_secs: u32,
    market: Option<&MarketChannel>,
    scaling: DofScaling,
    device: Device,
) -> BarBatch {
    build_host_batch(
        files,
        rows,
        len,
        res_secs,
        market,
        scaling,
        &mut BatchScratch::default(),
    )
    .to_device(device)
}

#[allow(clippy::too_many_arguments)]
fn fill_batch_row(
    files: &[BarFile],
    series: usize,
    start: usize,
    len: usize,
    res_secs: u32,
    market: Option<&MarketChannel>,
    scaling: DofScaling,
    dof_out: &mut [f32],
    time_out: &mut [i32],
    sigma_out: &mut [f32],
    mut raw_out: Option<&mut [f32]>,
) -> usize {
    let bars = files[series].bars();
    readahead(&bars[start.saturating_sub(scaling.warmup_bars() + 1)..start + len]);
    let mut slot = 0usize;
    let mut missing = 0usize;
    // The window's bars are strictly time-ordered, so the market join advances forward rather
    // than bisecting the whole proxy vector once per bar.
    let mut cursor = 0usize;
    for_each_window_dof_detailed(bars, start, len, scaling, |bar, raw, encoded| {
        dof_out[slot * BAR_DOF..(slot + 1) * BAR_DOF].copy_from_slice(&encoded.row.dof.to_array());
        sigma_out[slot] = encoded.row.sigma;
        if let Some(output) = raw_out.as_deref_mut() {
            output[slot * BAR_DOF..(slot + 1) * BAR_DOF].copy_from_slice(&raw.to_array());
        }
        // `start >= 1` for every window — bar 0 carries no DOF — so the predecessor is always
        // addressable. The market channel is joined at the row bar's own timestamp.
        let (ids, next) = bar_time_ids_from(
            bar.ts(),
            Some(bars[start + slot - 1].ts()),
            res_secs,
            market,
            cursor,
        );
        cursor = next;
        missing += usize::from(ids[TIME_MARKET_R] == MARKET_MISSING);
        time_out[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES]
            .copy_from_slice(&ids.map(|id| id as i32));
        slot += 1;
    });
    debug_assert_eq!(slot, len);
    missing
}

// ---------------------------------------------------------------------------
// Corpus anomaly audit
// ---------------------------------------------------------------------------

/// A single 5-minute bar cannot legitimately move 4x. Anything past this is a data defect.
///
/// Deliberately NOT scaled by resolution, and measured rather than assumed. The worry with a
/// fixed absolute threshold is that a real crash day gets classified as corruption and later
/// "cleaned" away — the daily corpus exists precisely to supply 2000, 2008 and 2020, so that
/// would destroy the thing it was downloaded for. It does not happen, by two measurements over
/// the whole corpus:
///
/// * The worst single-day move a 2008 survivor posts is around -30%, `|r| ~ 0.36`, a factor of
///   3.8 below this limit. Across all 21,494,382 daily returns only 40 exceed it (0.02/10k) and
///   274 bars carry `s > ln 4` (0.13/10k).
/// * The 5-minute corpus is 7x MORE anomalous against the same threshold: 0.14/10k of its
///   returns exceed it. So the fixed limit is if anything conservative at daily resolution, not
///   trigger-happy.
///
/// Scaling the limit up for daily would therefore buy nothing measurable and would decouple this
/// audit from `deep_daily`'s pre-write gate, which uses this same constant.
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
        Self::report_of(std::slice::from_ref(self))
    }

    /// One chart for a whole multi-resolution run, on the single registered base name.
    ///
    /// Every resolution contributes four series suffixed `@<res_secs>`, each ranked within its
    /// OWN resolution — which is why the x label says so. Ranking the union on one axis would be
    /// meaningless: the symbol at rank 500 of the 5-minute corpus and the symbol at rank 500 of
    /// the daily corpus are unrelated, and the two audits are not even measured over the same
    /// symbol set. A per-resolution base name was the alternative and is worse: it would need
    /// registering in [`shared::report::PRETRAIN_REPORT_BASES`] and would leave `N - 1` blank
    /// panels on every single-resolution run.
    pub fn report_of(audits: &[CorpusAnomalies]) -> Report {
        let mut series = Vec::with_capacity(4 * audits.len());
        for audit in audits {
            let suffix = if audits.len() > 1 {
                format!("@{}", audit.res_secs)
            } else {
                String::new()
            };
            for (name, pick) in [
                (
                    "splice",
                    (|s: &SymbolAnomalies| s.splices) as fn(&SymbolAnomalies) -> usize,
                ),
                ("tick", |s: &SymbolAnomalies| s.ticks),
                ("jump", |s: &SymbolAnomalies| s.jumps),
                ("extreme_range", |s: &SymbolAnomalies| s.extreme_range),
            ] {
                series.push(ReportSeries {
                    label: format!("{name}{suffix}"),
                    values: audit
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
                });
            }
        }
        let title = audits
            .iter()
            .map(|audit| {
                let worst = audit
                    .worst(ANOMALY_WORST_LISTED)
                    .iter()
                    .map(|s| format!("{} {:.1}", s.symbol, s.anomaly_rate()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} | worst: {worst}", audit.summary())
            })
            .collect::<Vec<_>>()
            .join(" || ");
        Report {
            title,
            x_label: Some("symbol rank (worst first, within its own resolution)".to_string()),
            y_label: Some("anomalous bars per 10k".to_string()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::MultiLine { series },
        }
    }

    /// Write `<dir>/pretrain_corpus_anomalies.report.bin`.
    pub fn write_report(&self, dir: &Path) -> Result<()> {
        Self::write_report_of(std::slice::from_ref(self), dir)
    }

    /// Write every resolution's audit to the one registered base.
    ///
    /// The generation directory is walked non-recursively, so this lands beside the reporter's
    /// own charts rather than in a subdirectory.
    pub fn write_report_of(audits: &[CorpusAnomalies], dir: &Path) -> Result<()> {
        ensure!(
            !audits.is_empty(),
            "the corpus anomaly report needs at least one resolution's audit"
        );
        let path = dir.join(format!("{ANOMALY_REPORT_BASE}.report.bin"));
        shared::report::write_report(&path, &Self::report_of(audits))
            .with_context(|| format!("writing {}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Target-geometry audit: what the parametrization does to the corpus
// ---------------------------------------------------------------------------

/// Quantile probabilities every target-geometry panel reports, ASCENDING.
pub const TARGET_QUANTILES: [f64; 9] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

/// Rows the strided quantile subsample retains over the whole audited split.
///
/// A quantile needs order statistics and an evaluation split runs to tens of millions of bars,
/// so the audit keeps a deterministic uniform stride of at most this many rows — the same
/// device [`BarSupports`] uses for its own marginal estimate. At this cap a median carries
/// `1.25 / sqrt(n) ~ 0.12%` of relative noise, which is three orders below the effect the
/// panels are read for.
///
/// Moments and occupancy are accumulated over EVERY bar rather than over the subsample, and
/// deliberately: the fourth moment of a fat-tailed return lives in the handful of most extreme
/// observations in the split, and a one-in-forty stride would simply lose them.
const TARGET_QUANTILE_ROWS: usize = 1 << 20;

/// Streaming first four central moments, mergeable so the audit can fold over series in
/// parallel.
///
/// Pébay's centred pairwise update rather than raw power sums. `E[r^4]` over a log return of
/// scale `1e-3` is a number near `1e-12`, and accumulating it as `sum(x^4) - 4 mu sum(x^3) +
/// ...` loses most of its significance to cancellation at exactly the corpus sizes this runs
/// on. The centred form has no cancellation to lose.
#[derive(Clone, Copy, Debug, Default)]
struct CentralMoments {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl CentralMoments {
    fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let prior = self.n;
        let n = prior + 1.0;
        let delta = x - self.mean;
        let step = delta / n;
        let step2 = step * step;
        let term = delta * step * prior;
        self.mean += step;
        self.m4 +=
            term * step2 * (n * n - 3.0 * n + 3.0) + 6.0 * step2 * self.m2 - 4.0 * step * self.m3;
        self.m3 += term * step * (n - 2.0) - 3.0 * step * self.m2;
        self.m2 += term;
        self.n = n;
    }

    fn merge(&mut self, other: &Self) {
        if other.n == 0.0 {
            return;
        }
        if self.n == 0.0 {
            *self = *other;
            return;
        }
        let (a, b) = (self.n, other.n);
        let n = a + b;
        let delta = other.mean - self.mean;
        let d2 = delta * delta;
        let m4 = self.m4
            + other.m4
            + d2 * d2 * a * b * (a * a - a * b + b * b) / (n * n * n)
            + 6.0 * d2 * (a * a * other.m2 + b * b * self.m2) / (n * n)
            + 4.0 * delta * (a * other.m3 - b * self.m3) / n;
        let m3 = self.m3
            + other.m3
            + d2 * delta * a * b * (a - b) / (n * n)
            + 3.0 * delta * (a * other.m2 - b * self.m2) / n;
        self.m2 += other.m2 + d2 * a * b / n;
        self.m3 = m3;
        self.m4 = m4;
        self.mean += b * delta / n;
        self.n = n;
    }

    fn sd(&self) -> f64 {
        if self.n < 2.0 {
            return f64::NAN;
        }
        (self.m2 / self.n).sqrt()
    }

    /// `E[(x - mu)^4] / sigma^4 - 3`: zero for a Gaussian, and large and positive for a bar
    /// return.
    fn excess_kurtosis(&self) -> f64 {
        if self.n < 4.0 || self.m2 <= 0.0 {
            return f64::NAN;
        }
        self.n * self.m4 / (self.m2 * self.m2) - 3.0
    }
}

/// The two degrees of freedom [`DofScaling::VolStandardized`] moves, in the order
/// [`TargetGeometry::dof`] carries them.
pub const TARGET_SCALE_DOF: [(usize, &str); 2] = [(DOF_R, "r"), (DOF_S, "s")];

/// One degree of freedom's scale, exactly as the supports fitter receives it.
#[derive(Clone, Debug)]
pub struct DofScale {
    pub name: &'static str,
    pub mean: f64,
    pub sd: f64,
    /// [`TARGET_QUANTILES`] of the strided subsample.
    pub quantiles: [f64; TARGET_QUANTILES.len()],
    /// `E[(x - mu)^4] / sigma^4 - 3`.
    ///
    /// The load-bearing number, and the one that can refute the parametrization rather than
    /// merely confirm it. Standardization is supposed to remove the CROSS-SECTIONAL scale
    /// mixture and leave the conditional shape alone, so the fat tail must SURVIVE it. A
    /// standardized kurtosis collapsing toward zero means the divisor is tracking the bar's own
    /// move closely enough to absorb it, which is a leak of exactly the signal the head is
    /// there to predict.
    pub excess_kurtosis: f64,
}

/// One symbol's share of the pooled `r` grid.
#[derive(Clone, Debug)]
pub struct SymbolBinOccupancy {
    pub symbol: String,
    /// Bars this entropy was measured over. Carried because it BOUNDS the statistic: a symbol
    /// with 90 evaluation bars cannot exceed `ln(90)` however well the grid fits it, and
    /// reading such a row as poor resolution would be a measurement artifact.
    pub bars: u64,
    /// Occupancy entropy in nats, from [`BarSupports::bin_occupancy_entropy`].
    pub entropy: f64,
    /// `exp(entropy)`, from [`BarSupports::effective_bins`]: the number of bins this symbol
    /// effectively has.
    pub effective_bins: f64,
}

/// Per-series accumulator of [`BarCorpus::audit_target_geometry`].
struct SeriesGeometry {
    symbol: String,
    bars: u64,
    sigma_relative_floored: u64,
    sigma_numerical_fallback: u64,
    r_z_clamped: u64,
    s_z_clamped: u64,
    sigma_unwarmed: u64,
    occupancy: Vec<f64>,
    moments: [CentralMoments; TARGET_SCALE_DOF.len()],
    sigma_rows: Vec<f32>,
    dof_rows: [Vec<f32>; TARGET_SCALE_DOF.len()],
}

impl SeriesGeometry {
    fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_owned(),
            bars: 0,
            sigma_relative_floored: 0,
            sigma_numerical_fallback: 0,
            r_z_clamped: 0,
            s_z_clamped: 0,
            sigma_unwarmed: 0,
            occupancy: vec![0.0; NUM_BAR_BINS as usize],
            moments: [CentralMoments::default(); TARGET_SCALE_DOF.len()],
            sigma_rows: Vec::new(),
            dof_rows: std::array::from_fn(|_| Vec::new()),
        }
    }

    /// Absorb one encoded row under the corpus's own parametrization.
    ///
    /// The bin comes from [`BarSupports::bin_of`], which is the same rule the emission head's
    /// targets go through, so the occupancy is the grid the model actually trains against
    /// rather than a second discretization of it.
    fn absorb_dof(&mut self, binner: &DofBinner<'_>, dof: &BarDof, keep: bool) {
        let values = [dof.r as f64, dof.s as f64];
        for (slot, value) in values.into_iter().enumerate() {
            self.moments[slot].push(value);
            if keep {
                self.dof_rows[slot].push(value as f32);
            }
        }
        self.occupancy[binner.bin_of(values[0])] += 1.0;
    }

    fn finish(&self) -> SymbolBinOccupancy {
        SymbolBinOccupancy {
            symbol: self.symbol.clone(),
            bars: self.bars,
            entropy: BarSupports::bin_occupancy_entropy(&self.occupancy),
            effective_bins: BarSupports::effective_bins(&self.occupancy),
        }
    }
}

/// What the target parametrization actually does to the corpus, measured on one split before a
/// single optimizer step.
///
/// Three panels, one object, because they are the same measurement read at three depths: the
/// divisor, the quotient, and what the quotient does to the grid the head predicts on.
#[derive(Clone, Debug)]
pub struct TargetGeometry {
    pub res_secs: u32,
    /// The parametrization `dof` and `per_symbol` were measured under, i.e. the corpus's own.
    pub scaling: DofScaling,
    pub split: Split,
    pub bars: u64,
    /// `sigma_t` quantiles in LOG-RETURN units, always measured under
    /// [`DofScaling::VolStandardized`] whatever the corpus trains on, so the control and the
    /// standardized arm report the same object.
    pub sigma_quantiles: [f64; TARGET_QUANTILES.len()],
    /// Bars whose adaptive HAR reference used the name's slow-anchor-relative floor.
    pub sigma_relative_floored: u64,
    /// Bars whose causal history contained no positive scale and therefore required the
    /// explicitly numerical fallback divisor.
    pub sigma_numerical_fallback: u64,
    /// Standardized return targets clipped to `[-BAR_Z_LIMIT, BAR_Z_LIMIT]`.
    pub r_z_clamped: u64,
    /// Standardized range targets clipped to `[0, BAR_Z_LIMIT]`.
    pub s_z_clamped: u64,
    /// Bars with fewer than [`crate::torch::bar_dist::BAR_SIGMA_WARMUP_BARS`] causal
    /// predecessors, i.e. whose reference is DEFINED but whose span-256 leg has not forgotten
    /// its seed. [`for_each_window_dof`] truncates the warm-up at the series start rather than
    /// reaching past it, so this is a property of the split's position in each symbol and is
    /// computed from the bar indices rather than scanned.
    pub sigma_unwarmed: u64,
    /// `r` then `s`, in [`TARGET_SCALE_DOF`] order.
    pub dof: [DofScale; TARGET_SCALE_DOF.len()],
    /// Pooled `r` occupancy SHARES over the audited split; sums to one.
    pub r_occupancy: Vec<f64>,
    /// Per symbol, WORST effective resolution first.
    ///
    /// This is the decomposition that can see the defect at all. A pooled equal-mass histogram
    /// is balanced by construction on either parametrization, so the pooled entropy is `ln(128)`
    /// on both and says nothing; the claim under test is about the worst symbol and about the
    /// SPREAD across symbols.
    pub per_symbol: Vec<SymbolBinOccupancy>,
}

impl TargetGeometry {
    fn reduce(
        res_secs: u32,
        scaling: DofScaling,
        split: Split,
        scans: Vec<SeriesGeometry>,
    ) -> Self {
        let bins = NUM_BAR_BINS as usize;
        let mut r_occupancy = vec![0.0f64; bins];
        let mut moments = [CentralMoments::default(); TARGET_SCALE_DOF.len()];
        let mut sigma_rows: Vec<f32> = Vec::new();
        let mut dof_rows: [Vec<f32>; TARGET_SCALE_DOF.len()] = std::array::from_fn(|_| Vec::new());
        let mut per_symbol = Vec::with_capacity(scans.len());
        let (
            mut bars,
            mut sigma_relative_floored,
            mut sigma_numerical_fallback,
            mut r_z_clamped,
            mut s_z_clamped,
            mut sigma_unwarmed,
        ) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for scan in &scans {
            bars += scan.bars;
            sigma_relative_floored += scan.sigma_relative_floored;
            sigma_numerical_fallback += scan.sigma_numerical_fallback;
            r_z_clamped += scan.r_z_clamped;
            s_z_clamped += scan.s_z_clamped;
            sigma_unwarmed += scan.sigma_unwarmed;
            for (total, count) in r_occupancy.iter_mut().zip(&scan.occupancy) {
                *total += count;
            }
            for slot in 0..TARGET_SCALE_DOF.len() {
                moments[slot].merge(&scan.moments[slot]);
                dof_rows[slot].extend_from_slice(&scan.dof_rows[slot]);
            }
            sigma_rows.extend_from_slice(&scan.sigma_rows);
            if scan.bars > 0 {
                per_symbol.push(scan.finish());
            }
        }
        let occupied: f64 = r_occupancy.iter().sum();
        if occupied > 0.0 {
            for share in r_occupancy.iter_mut() {
                *share /= occupied;
            }
        }
        // Worst first, which is the convention `CorpusAnomalies` already ranks by and the order
        // the headline is read off.
        per_symbol.sort_unstable_by(|a, b| {
            a.effective_bins
                .total_cmp(&b.effective_bins)
                .then_with(|| a.symbol.cmp(&b.symbol))
        });
        Self {
            res_secs,
            scaling,
            split,
            bars,
            sigma_quantiles: subsample_quantiles(&mut sigma_rows),
            sigma_relative_floored,
            sigma_numerical_fallback,
            r_z_clamped,
            s_z_clamped,
            sigma_unwarmed,
            dof: std::array::from_fn(|slot| DofScale {
                name: TARGET_SCALE_DOF[slot].1,
                mean: if moments[slot].n > 0.0 {
                    moments[slot].mean
                } else {
                    f64::NAN
                },
                sd: moments[slot].sd(),
                quantiles: subsample_quantiles(&mut dof_rows[slot]),
                excess_kurtosis: moments[slot].excess_kurtosis(),
            }),
            r_occupancy,
            per_symbol,
        }
    }

    /// Pooled occupancy entropy as a fraction of the uniform maximum `ln(NUM_BAR_BINS)`.
    pub fn entropy_fraction(&self) -> f64 {
        BarSupports::bin_occupancy_entropy(&self.r_occupancy) / (NUM_BAR_BINS as f64).ln()
    }

    pub fn sigma_relative_floored_share(&self) -> f64 {
        self.share(self.sigma_relative_floored)
    }

    pub fn sigma_numerical_fallback_share(&self) -> f64 {
        self.share(self.sigma_numerical_fallback)
    }

    pub fn r_z_clamped_share(&self) -> f64 {
        self.share(self.r_z_clamped)
    }

    pub fn s_z_clamped_share(&self) -> f64 {
        self.share(self.s_z_clamped)
    }

    pub fn sigma_unwarmed_share(&self) -> f64 {
        self.share(self.sigma_unwarmed)
    }

    fn share(&self, count: u64) -> f64 {
        if self.bars == 0 {
            0.0
        } else {
            count as f64 / self.bars as f64
        }
    }

    /// Effective `r`-bins of the worst-resolved symbol.
    pub fn worst_effective_bins(&self) -> f64 {
        self.per_symbol
            .first()
            .map_or(f64::NAN, |s| s.effective_bins)
    }

    pub fn best_effective_bins(&self) -> f64 {
        self.per_symbol
            .last()
            .map_or(f64::NAN, |s| s.effective_bins)
    }

    pub fn median_effective_bins(&self) -> f64 {
        if self.per_symbol.is_empty() {
            return f64::NAN;
        }
        self.per_symbol[self.per_symbol.len() / 2].effective_bins
    }

    /// THE headline. Best over worst per-symbol effective resolution, at least one by
    /// construction. The audit's defect 1 is that this is an order of magnitude on a pooled
    /// absolute grid; standardization is supposed to drive it toward one.
    pub fn effective_bin_spread(&self) -> f64 {
        self.best_effective_bins() / self.worst_effective_bins()
    }

    pub fn summary(&self) -> String {
        format!(
            "target geometry on {:?} at {}s under {}: {} bars, sigma_t p50 {:.2} bps/bar \
             (p1 {:.2}, p99 {:.2}), {} relative-floored, {} numerical-fallback, z clamps r={} \
             s={}, {} under-warmed; sd(z_{}) {:.4}, mean(z_{}) {:.4}, excess kurtosis {:.1} / \
             {:.1}; pooled r occupancy {:.4} of ln({}), per-symbol effective bins worst {:.1} \
             median {:.1} best {:.1} (spread {:.2}x over {} symbols)",
            self.split,
            self.res_secs,
            self.scaling,
            self.bars,
            1e4 * self.sigma_quantiles[TARGET_QUANTILES.len() / 2],
            1e4 * self.sigma_quantiles[0],
            1e4 * self.sigma_quantiles[TARGET_QUANTILES.len() - 1],
            self.sigma_relative_floored,
            self.sigma_numerical_fallback,
            self.r_z_clamped,
            self.s_z_clamped,
            self.sigma_unwarmed,
            self.dof[0].name,
            self.dof[0].sd,
            self.dof[1].name,
            self.dof[1].mean,
            self.dof[0].excess_kurtosis,
            self.dof[1].excess_kurtosis,
            self.entropy_fraction(),
            NUM_BAR_BINS,
            self.worst_effective_bins(),
            self.median_effective_bins(),
            self.best_effective_bins(),
            self.effective_bin_spread(),
            self.per_symbol.len(),
        )
    }
}

/// Exact [`TARGET_QUANTILES`] of the retained subsample, sorting it in place.
fn subsample_quantiles(rows: &mut [f32]) -> [f64; TARGET_QUANTILES.len()] {
    if rows.is_empty() {
        return [f64::NAN; TARGET_QUANTILES.len()];
    }
    rows.sort_unstable_by(f32::total_cmp);
    let last = rows.len() - 1;
    TARGET_QUANTILES.map(|q| rows[(q * last as f64).round() as usize] as f64)
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
    // fitted on, rather than a second, subtly different definition. RAW unconditionally:
    // `ANOMALY_LOG_LIMIT` is a statement about a 4x price move in log points, and expressing a
    // corruption threshold in sigma units would make it mean a different thing on every bar.
    let mut returns = Vec::with_capacity(bars.len() - 1);
    for_each_window_dof(bars, 1, bars.len() - 1, DofScaling::Raw, |_, row| {
        if row.dof.s as f64 > ANOMALY_LOG_LIMIT {
            out.extreme_range += 1;
        }
        returns.push(row.dof.r);
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

/// Emit the DOF of bars `[anchor, anchor + len)`, preceded by the selected scaling warm-up.
fn for_each_window_dof(
    bars: &[PackedBar],
    anchor: usize,
    len: usize,
    scaling: DofScaling,
    mut sink: impl FnMut(&PackedBar, StandardizedDof),
) {
    for_each_window_dof_detailed(bars, anchor, len, scaling, |bar, _, encoded| {
        sink(bar, encoded.row)
    });
}

/// Detailed window encoder carrying the exact pre-standardization row and scaling decisions.
///
/// The volume EMA always starts at the raw policy's warm-up boundary, independently of the
/// longer volatility warm-up. Consequently the raw row embedded in a standardized batch is
/// identical to the corresponding raw batch target rather than changing `w` merely because a
/// second estimator needed more history.
fn for_each_window_dof_detailed(
    bars: &[PackedBar],
    anchor: usize,
    len: usize,
    scaling: DofScaling,
    mut sink: impl FnMut(&PackedBar, BarDof, EncodedDof),
) {
    assert!(anchor >= 1, "bar 0 has no predecessor close");
    assert!(anchor + len <= bars.len(), "window runs past the symbol");
    let start = anchor.saturating_sub(scaling.warmup_bars() + 1);
    let volume_start = anchor.saturating_sub(BAR_RAW_WARMUP_BARS + 1);
    let mut ema = VolumeEma::default();
    let mut volume_active = start >= volume_start;
    if volume_active {
        ema.observe(bars[start].volume);
    }
    let mut har = RangeVolHar::default();
    if scaling.is_standardized() {
        har.observe(&encode_dof(
            bars[start].open,
            &bars[start],
            bars[start].volume,
        ));
    }
    let mut prev_close = bars[start].close;
    for (offset, bar) in bars[start + 1..anchor + len].iter().enumerate() {
        let index = start + 1 + offset;
        let volume = bar.volume;
        let seeded_volume_here = !volume_active && index == volume_start;
        if seeded_volume_here {
            ema.observe(volume);
            volume_active = true;
        }
        let raw = encode_dof(prev_close, bar, ema.reference_for(volume));
        if index >= anchor {
            sink(bar, raw, scale_row(raw, &har, scaling));
        }
        if scaling.is_standardized() {
            har.observe(&raw);
        }
        if volume_active && !seeded_volume_here {
            ema.observe(volume);
        }
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

/// Open every AUXILIARY resolution of a corpus directory against the split instants the
/// already-loaded `deployment` corpus is using.
///
/// Takes the loaded corpus rather than `(res_secs, min_bars, bounds)` so that the shared-split
/// guarantee is structural: there is no second derivation of the boundary to disagree with
/// however the deployment corpus was actually built — pinned, derived, or restricted to a
/// liquidity floor. Computing the boundary independently per resolution would reintroduce exactly
/// the leak the global split exists to prevent, one timeframe's test window overlapping another's
/// train window in wall-clock time.
///
/// Auxiliary corpora are TRAINING SIGNAL ONLY. An auxiliary bar dated at or after the
/// `train | val` instant covers the same market days the deployment resolution is scored on, so
/// training on it is leakage even though the resolution differs. `Split::Train` already excludes
/// every such bar — `Corpus::split_range(_, Train)` is `(0, index_at_or_after(bounds.0))` — and
/// this function reports the count that the exclusion costs, per resolution, so the drop is a
/// measured number in the run log instead of an assumption. It is asserted in
/// `an_auxiliary_resolution_trains_on_no_held_out_bar`.
///
/// `auxiliary` entries must be distinct from each other and from the deployment resolution:
/// loading one resolution twice would double-count its bars in every token budget.
pub fn load_auxiliary(
    deployment: &BarCorpus,
    auxiliary: &[(u32, usize)],
) -> Result<Vec<BarCorpus>> {
    let dir = deployment.dir();
    let bounds = deployment.split_bounds();
    let mut out: Vec<BarCorpus> = Vec::with_capacity(auxiliary.len());
    for &(res, min_bars) in auxiliary {
        ensure!(
            res != deployment.res_secs(),
            "resolution {res} is both the deployment and an auxiliary resolution"
        );
        ensure!(
            !out.iter().any(|loaded| loaded.res_secs() == res),
            "auxiliary resolution {res} is requested twice"
        );
        let corpus = BarCorpus::load_with_bounds(dir, res, min_bars, bounds)?;
        ensure!(
            corpus.split_bounds() == bounds,
            "auxiliary resolution {res} was loaded with split bounds {:?} instead of the \
             deployment's {bounds:?}",
            corpus.split_bounds()
        );
        let (train, val, test) = (
            corpus.split_bars(Split::Train),
            corpus.split_bars(Split::Val),
            corpus.split_bars(Split::Test),
        );
        println!(
            "[dataset] auxiliary {res}s: {} symbols, {train} train bars usable; {} bars \
             ({:.2}%) sit at or after {} and are DROPPED as held-out ({val} val, {test} test)",
            corpus.series_count(),
            val + test,
            if corpus.unique_bars() == 0 {
                0.0
            } else {
                100.0 * (val + test) as f64 / corpus.unique_bars() as f64
            },
            iso_ms(bounds.0),
        );
        out.push(corpus);
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

/// Lower-case hex of a finished SHA-256 context.
fn hex_digest(digest: DigestContext) -> String {
    digest
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
    use std::collections::HashMap;

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

    /// A wholesale vendor correction can preserve every metadata field bar-corpus v2 hashed.
    /// Identity must follow the interior observation the encoder will actually consume.
    #[test]
    fn corpus_fingerprint_tracks_interior_consumed_content() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_dataset_content_fp_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let fx = Fixture { dir };
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let path = bar_path(&fx.dir, "AAA");
        let original = synth_bars(7, 1_000, base);
        write_bar_file(&path, "AAA", RES, &original).unwrap();

        let baseline = BarCorpus::load(&fx.dir, RES, 100).unwrap();
        let fingerprint = baseline.identity_fingerprint();
        let unchanged = BarCorpus::load(&fx.dir, RES, 100).unwrap();
        assert_eq!(
            unchanged.identity_fingerprint(),
            fingerprint,
            "unchanged reloads must reproduce the corpus identity"
        );

        let mut corrected = original.clone();
        corrected[500].close *= 1.01;
        write_bar_file(&path, "AAA", RES, &corrected).unwrap();
        let revised_file = BarFile::open(&path).unwrap();
        assert_eq!(revised_file.len(), original.len());
        assert_eq!(revised_file.first_ts_ms(), Some(original[0].ts()));
        assert_eq!(
            revised_file.last_ts_ms(),
            Some(original.last().unwrap().ts())
        );

        let revised = BarCorpus::load(&fx.dir, RES, 100).unwrap();
        assert_eq!(revised.symbols(), baseline.symbols());
        assert_eq!(revised.unique_bars(), baseline.unique_bars());
        assert_eq!(revised.split_bounds(), baseline.split_bounds());
        assert_ne!(
            revised.identity_fingerprint(),
            fingerprint,
            "an interior OHLCV correction must change the corpus identity"
        );
    }

    /// The located draw must be the SAME draw: same rows, same order, same truncation. Two
    /// samplers over one corpus would make a per-row classification joined onto `(series, bar)`
    /// describe different rows than the ones scored, which is the whole reason the located form
    /// exists.
    #[test]
    fn the_located_draw_is_the_same_draw() {
        let (_fx, corpus) = fixture("located");
        // Not a multiple of SUPPORT_BLOCK, so the shared truncation rule is exercised too.
        let budget = 3 * SUPPORT_BLOCK + 17;
        let plain = corpus.sample_train_dof(budget, 0xC0FFEE);
        let located = corpus.sample_train_dof_located(budget, 0xC0FFEE);
        assert_eq!(plain.len(), budget);
        assert_eq!(located.len(), plain.len());
        for (row, ((ts, dof), (window, located_ts, located_dof))) in
            plain.iter().zip(located.iter()).enumerate()
        {
            assert_eq!(ts, located_ts, "row {row} timestamp");
            assert_eq!(dof, located_dof, "row {row} dof");
            assert_eq!(
                corpus.ts_ms(window.symbol as usize, window.bar_index as usize),
                *ts,
                "row {row} location points at a different bar than the row it carries"
            );
        }
    }

    /// The whole-series stream must cover every DOF-carrying bar exactly once, in bar order, and
    /// agree with the block draw wherever the two look at the same bar.
    #[test]
    fn the_series_stream_covers_every_dof_bar_in_order() {
        let (_fx, corpus) = fixture("stream");
        for series in 0..corpus.series_count() {
            let mut seen = Vec::new();
            corpus.for_each_series_dof(series, |index, bar, dof| {
                seen.push((index, bar.ts(), dof));
            });
            assert_eq!(
                seen.len(),
                corpus.series_len(series) - 1,
                "{} streamed {} of {} bars",
                corpus.symbol(series),
                seen.len(),
                corpus.series_len(series) - 1
            );
            for (offset, (index, ts, _)) in seen.iter().enumerate() {
                assert_eq!(
                    *index,
                    offset + 1,
                    "bar 0 carries no DOF and must be skipped"
                );
                assert_eq!(*ts, corpus.ts_ms(series, *index));
            }
        }
        // The two accessors must agree on the rows they share. `for_each_window_dof` warms the
        // volume EMA over at most DOF_WARMUP_BARS predecessors, so a block anchored deep in a
        // series sees the same reference the whole-series pass does only once both have warmed;
        // `r`, `s`, `u` and `v` are exact for every row because none of them reads the EMA.
        let located = corpus.sample_train_dof_located(4 * SUPPORT_BLOCK, 0xBEEF);
        let mut streamed: HashMap<(u32, u32), BarDof> = HashMap::new();
        for series in 0..corpus.series_count() {
            corpus.for_each_series_dof(series, |index, _, dof| {
                streamed.insert((series as u32, index as u32), dof);
            });
        }
        for (window, _, dof) in &located {
            let mirror = streamed
                .get(&(window.symbol, window.bar_index))
                .expect("every drawn bar is streamed");
            assert_eq!(dof.r, mirror.r, "r disagrees at {window:?}");
            assert_eq!(dof.s, mirror.s, "s disagrees at {window:?}");
            assert_eq!(dof.u, mirror.u, "u disagrees at {window:?}");
            assert_eq!(dof.v, mirror.v, "v disagrees at {window:?}");
        }
    }

    /// The ET day index must change at exactly one instant per calendar day and must survive a
    /// gap longer than a week, which is precisely where [`TIME_WEEKDAY`] cannot tell two days
    /// apart.
    #[test]
    fn the_et_day_index_changes_once_per_calendar_day() {
        // 2021-08-16T13:30:00Z is 09:30 ET, inside EDT; the day before ends at 04:00Z.
        let open = 1_629_120_600_000i64;
        let day = 86_400_000i64;
        assert_eq!(et_local_day(open), et_local_day(open + 6 * 3_600_000));
        assert_eq!(et_local_day(open) + 1, et_local_day(open + day));
        // Seven days on, the weekday id repeats while the day index does not.
        assert_eq!(
            bar_time_ids(open, None, RES, None)[TIME_WEEKDAY],
            bar_time_ids(open + 7 * day, None, RES, None)[TIME_WEEKDAY]
        );
        assert_eq!(et_local_day(open) + 7, et_local_day(open + 7 * day));
        // Midnight ET is the boundary, not midnight UTC: 2021-08-16T04:00:00Z is 00:00 EDT and
        // opens a new ET day, so the millisecond before it belongs to the previous one.
        // 04:00:00Z exactly. 1_629_090_000_000 would be 05:00Z, i.e. 01:00 EDT, which is
        // mid-day in ET terms and makes both assertions below vacuous-then-false.
        let midnight_et = 1_629_086_400_000i64;
        assert_eq!(et_local_day(midnight_et - 1) + 1, et_local_day(midnight_et));
        assert_eq!(
            et_local_day(midnight_et),
            et_local_day(midnight_et + 86_399_999)
        );
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
        // 2015 AND FORWARD PAST THE SPLIT — the shape the real auxiliary corpus has, where 4.65%
        // of daily bars land inside the 5-minute held-out window and must be dropped.
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let day_ms = 86_400_000i64;
        for symbol in ["AAA", "BBB"] {
            let intraday = synth_bars(1, 5_000, base);
            write_bar_file(&bar_path(&dir, symbol), symbol, RES, &intraday).unwrap();
            let mut daily = synth_bars(2, 1_600, base - 1_500 * day_ms);
            for (i, bar) in daily.iter_mut().enumerate() {
                bar.ts_ms = base - 1_500 * day_ms + i as i64 * day_ms;
            }
            let path = dir.join(format!("{symbol}.86400.{FILE_EXTENSION}"));
            write_bar_file(&path, symbol, 86_400, &daily).unwrap();
        }

        let intraday = BarCorpus::load(&dir, RES, 100).unwrap();
        let loaded = load_auxiliary(&intraday, &[(86_400, 100)]).unwrap();
        let daily = &loaded[0];

        // Neither corpus may see the other's files.
        assert_eq!(intraday.unique_bars(), 2 * 5_000);
        assert_eq!(daily.unique_bars(), 2 * 1_600);
        assert_eq!(intraday.res_secs(), RES);
        assert_eq!(daily.res_secs(), 86_400);
        assert_eq!(intraday.resolution_class(), resolution_class(RES));
        assert_ne!(intraday.resolution_class(), daily.resolution_class());

        // One boundary pair, so a daily bar and a 5-minute bar from the same date agree.
        assert_eq!(intraday.split_bounds(), daily.split_bounds());
        let (b0, b1) = intraday.split_bounds();
        for corpus in [&intraday, daily] {
            for s in 0..corpus.series_count() {
                let bars = corpus.bars(s);
                let (_, train_hi) = corpus.split_range(s, Split::Train);
                let (_, val_hi) = corpus.split_range(s, Split::Val);
                assert!(bars[..train_hi].iter().all(|b| b.ts() < b0));
                assert!(bars[train_hi..val_hi].iter().all(|b| b.ts() < b1));
            }
        }

        // Calendar ids separate the two resolutions for the trunk.
        let intraday_ids = bar_time_ids(intraday.ts_ms(0, 0), None, intraday.res_secs(), None);
        let daily_ids = bar_time_ids(daily.ts_ms(0, 0), None, daily.res_secs(), None);
        assert_ne!(
            intraday_ids[TIME_RESOLUTION], daily_ids[TIME_RESOLUTION],
            "the resolution channel must distinguish the timeframes"
        );

        assert!(
            load_auxiliary(&intraday, &[(RES, 100)]).is_err(),
            "a resolution cannot be both deployment and auxiliary"
        );
        assert!(
            load_auxiliary(&intraday, &[(86_400, 100), (86_400, 100)]).is_err(),
            "loading one auxiliary resolution twice would double-count its bars"
        );
        assert_eq!(
            crate::data::universe::eligible_bar_universe(&dir, RES, 100),
            vec!["AAA".to_string(), "BBB".to_string()]
        );
        assert_eq!(intraday.scan_anomalies().res_secs, RES);
    }

    /// The auxiliary corpus is training signal only: a daily bar dated inside the 5-minute
    /// held-out window covers the same market days the deployment resolution is SCORED on, so
    /// training on it is leakage even though the resolution differs. The fixture is built so the
    /// exclusion is not vacuous — the daily series deliberately straddles both boundaries.
    #[test]
    fn an_auxiliary_resolution_trains_on_no_held_out_bar() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_dataset_auxleak_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let _fx = Fixture { dir: dir.clone() };

        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let day_ms = 86_400_000i64;
        for symbol in ["AAA", "BBB"] {
            write_bar_file(
                &bar_path(&dir, symbol),
                symbol,
                RES,
                &synth_bars(1, 5_000, base),
            )
            .unwrap();
            let mut daily = synth_bars(2, 3_000, base - 1_500 * day_ms);
            for (i, bar) in daily.iter_mut().enumerate() {
                bar.ts_ms = base - 1_500 * day_ms + i as i64 * day_ms;
            }
            write_bar_file(
                &dir.join(format!("{symbol}.86400.{FILE_EXTENSION}")),
                symbol,
                86_400,
                &daily,
            )
            .unwrap();
        }

        let deployment = BarCorpus::load(&dir, RES, 100).unwrap();
        let aux = load_auxiliary(&deployment, &[(86_400, AUXILIARY_MIN_BARS)])
            .unwrap()
            .remove(0);
        let (b0, _) = deployment.split_bounds();

        // Not vacuous: the auxiliary corpus really does hold bars inside the held-out window.
        let held_out = aux.split_bars(Split::Val) + aux.split_bars(Split::Test);
        assert!(
            held_out > 0,
            "the fixture must straddle the boundary or this test proves nothing"
        );
        assert_eq!(aux.split_bars(Split::Train) + held_out, aux.unique_bars());

        // Every bar a training sampler can reach is strictly before the deployment's train|val
        // instant, target bars and DOF-carrying context alike.
        let sampler = BarSampler::new(&aux, Split::Train, 256, 11);
        assert!(
            sampler.windows() > 0,
            "the auxiliary must yield train windows"
        );
        for anchor in sampler.anchors() {
            let bars = aux.bars(anchor.symbol as usize);
            let last = anchor.bar_index as usize + 256;
            assert!(
                bars[last].ts() < b0,
                "auxiliary window ending at {} reaches into the deployment held-out window \
                 starting {}",
                iso_ms(bars[last].ts()),
                iso_ms(b0)
            );
        }

        // And the deployment floor really would have refused the whole auxiliary corpus, which is
        // the bug this constant exists to prevent: a hard failure, not a silent empty load.
        let refused = BarCorpus::load(&dir, 86_400, DEFAULT_MIN_BARS);
        assert!(
            refused.is_err(),
            "DEFAULT_MIN_BARS must reject every daily file rather than load a subset"
        );
    }

    /// One registered base carries every resolution, four series each, suffixed only when there
    /// is more than one resolution to disambiguate.
    #[test]
    fn the_anomaly_report_carries_every_resolution_on_one_base() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_dataset_multires_report_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let _fx = Fixture { dir: dir.clone() };

        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let day_ms = 86_400_000i64;
        for symbol in ["AAA", "BBB"] {
            write_bar_file(
                &bar_path(&dir, symbol),
                symbol,
                RES,
                &synth_bars(1, 5_000, base),
            )
            .unwrap();
            let mut daily = synth_bars(2, 1_200, base - 1_500 * day_ms);
            for (i, bar) in daily.iter_mut().enumerate() {
                bar.ts_ms = base - 1_500 * day_ms + i as i64 * day_ms;
            }
            write_bar_file(
                &dir.join(format!("{symbol}.86400.{FILE_EXTENSION}")),
                symbol,
                86_400,
                &daily,
            )
            .unwrap();
        }
        let deployment = BarCorpus::load(&dir, RES, 100).unwrap();
        let aux = load_auxiliary(&deployment, &[(86_400, AUXILIARY_MIN_BARS)]).unwrap();

        let audits = vec![deployment.scan_anomalies(), aux[0].scan_anomalies()];
        CorpusAnomalies::write_report_of(&audits, &dir).unwrap();
        let path = dir.join(format!("{ANOMALY_REPORT_BASE}.report.bin"));
        let report = shared::report::read_report(&path).unwrap();
        let ReportKind::MultiLine { series } = &report.kind else {
            panic!("anomaly report must be a MultiLine chart");
        };
        assert_eq!(series.len(), 8, "four classes per resolution");
        for res in [RES, 86_400] {
            for class in ["splice", "tick", "jump", "extreme_range"] {
                let label = format!("{class}@{res}");
                let found = series
                    .iter()
                    .find(|s| s.label == label)
                    .unwrap_or_else(|| panic!("missing series {label}"));
                assert_eq!(found.values.len(), 2, "{label} must cover both symbols");
            }
        }
        assert!(report.title.contains("300s") && report.title.contains("86400s"));

        // A single-resolution run writes the SAME base with unsuffixed labels, so a chart that
        // predates multi-resolution keeps its series names.
        CorpusAnomalies::write_report_of(&audits[..1], &dir).unwrap();
        let solo = shared::report::read_report(&path).unwrap();
        let ReportKind::MultiLine { series } = &solo.kind else {
            panic!("MultiLine");
        };
        assert_eq!(
            series.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(),
            vec!["splice", "tick", "jump", "extreme_range"]
        );
        assert!(CorpusAnomalies::write_report_of(&[], &dir).is_err());
    }

    /// The duplicated context constant must not drift from the one the trunk is built with.
    #[test]
    fn the_context_constant_matches_the_world_model() {
        assert_eq!(
            MAX_CONTEXT_BARS as i64,
            crate::torch::world_model::BAR_MAX_CONTEXT
        );
        assert_eq!(DEFAULT_MIN_BARS, 20_480);
        assert!(
            AUXILIARY_MIN_BARS < DEFAULT_MIN_BARS,
            "an auxiliary floor at or above the deployment floor rejects every daily file"
        );
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
            assert_eq!(
                (v_hi, e_hi),
                (e_lo, bars.len()),
                "{symbol} val/test must abut"
            );
            assert!(
                bars[..t_hi].iter().all(|b| b.ts() < b0),
                "{symbol} train leak"
            );
            assert!(
                bars[v_lo..v_hi].iter().all(|b| b.ts() >= b0 && b.ts() < b1),
                "{symbol} val leak"
            );
            assert!(
                bars[e_lo..].iter().all(|b| b.ts() >= b1),
                "{symbol} test leak"
            );
            fractions.push(t_hi as f64 / bars.len() as f64);
        }
        // The whole point of a calendar split: symbols with different listing dates and
        // lengths do *not* get the same index-space cut. If they did, this test would be
        // passing for the wrong reason.
        let spread = fractions.iter().cloned().fold(f64::MIN, f64::max)
            - fractions.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            spread > 0.01,
            "fixture is degenerate, index split would agree"
        );

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
                assert!(
                    first >= lo_ts && last < hi_ts,
                    "{split} window {r:?} crosses"
                );
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
    fn future_windows_cover_every_direct_target_and_truncate_short_tails() {
        let (_fx, corpus) = fixture("direct_future");
        for context in [32usize, 64, 256] {
            for future in [1usize, 3, 5] {
                let sampler = BarSampler::new_with_future(
                    &corpus,
                    Split::Train,
                    context as i64,
                    future as i64,
                    17,
                );
                let mut expected = 0usize;
                for symbol in 0..corpus.symbols().len() {
                    let (lo, hi) = corpus.split_range(symbol, Split::Train);
                    let usable = hi.saturating_sub(lo.max(1));
                    expected += usable.saturating_sub(future) / context;
                }
                assert_eq!(sampler.windows(), expected);
                let refs: Vec<_> = sampler.anchors().iter().copied().take(2).collect();
                if !refs.is_empty() {
                    let batch = sampler.batch_of(&refs, Device::Cpu);
                    assert_eq!(
                        batch.dof.size(),
                        [refs.len() as i64, (context + future) as i64, BAR_DOF as i64],
                    );
                    if future >= 3 {
                        let direct = batch
                            .direct_time_ids
                            .as_ref()
                            .expect("three-future-bar samplers must carry direct clocks");
                        assert_eq!(
                            direct.size(),
                            [
                                refs.len() as i64,
                                context as i64,
                                2,
                                BAR_TIME_FEATURES as i64
                            ],
                        );
                        let valid = batch
                            .direct_valid
                            .as_ref()
                            .expect("three-future-bar samplers must carry ordinary-row masks");
                        assert_eq!(valid.size(), [refs.len() as i64, context as i64, 2],);
                        assert_eq!(valid.kind(), Kind::Bool);
                        for (row, reference) in refs.iter().enumerate() {
                            let bars = corpus.inner.files[reference.symbol as usize].bars();
                            let start = reference.bar_index as usize;
                            let schedule =
                                forecast_schedule_after(bars[start].ts(), 3, corpus.res_secs());
                            let expected_h2 =
                                (1..=2).all(|step| bars[start + step].ts() == schedule[step - 1]);
                            let expected_h3 = expected_h2 && bars[start + 3].ts() == schedule[2];
                            assert_eq!(valid.int64_value(&[row as i64, 0, 0]) != 0, expected_h2);
                            assert_eq!(valid.int64_value(&[row as i64, 0, 1]) != 0, expected_h3);
                        }
                        let expected = sampler
                            .forecast_time_ids(&refs, 0, 3, Device::Cpu)
                            .expect("decision-only forecast clock")
                            .narrow(1, 1, 2);
                        assert!(bool::try_from(
                            direct
                                .narrow(1, 0, 1)
                                .squeeze_dim(1)
                                .eq_tensor(&expected)
                                .all()
                        )
                        .unwrap());
                        for market in BAR_TIME_MARKET {
                            assert!(bool::try_from(
                                direct.narrow(-1, market as i64, 1).eq(MARKET_MISSING).all()
                            )
                            .unwrap());
                        }
                    } else {
                        assert!(batch.direct_time_ids.is_none());
                        assert!(batch.direct_valid.is_none());
                    }
                }
                for window in sampler.anchors() {
                    let (_, hi) = corpus.split_range(window.symbol as usize, Split::Train);
                    assert!(
                        window.bar_index as usize + context + future <= hi,
                        "short tail admitted without every future target"
                    );
                }
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
                // The conditioning row must describe the very same bar the DOF row came from.
                let want_ids = bar_time_ids(
                    bars[bar].ts(),
                    Some(bars[bar - 1].ts()),
                    corpus.res_secs(),
                    corpus.market_channel(),
                );
                for f in 0..BAR_TIME_FEATURES {
                    assert_eq!(
                        batch
                            .time_ids
                            .int64_value(&[row as i64, step as i64, f as i64]),
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
        assert!(
            sampler.windows() > 2 * want,
            "fixture must oversupply windows"
        );
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
        assert_eq!(
            symbols.len(),
            corpus.symbols().len(),
            "every ticker represented"
        );
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
            assert!(
                reach > span / 2,
                "symbol {s} picks span only {reach} of {span}"
            );
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
            assert!(
                *ts < b0,
                "sampled {} at or after the train|val bound {b0}",
                iso_ms(*ts)
            );
            assert!(dof.is_finite());
        }
        assert_eq!(
            samples,
            corpus.sample_train_dof(4_000, 31),
            "sampling must be pinned"
        );

        let default_path = corpus.supports_path();
        let supports = corpus.fit_supports(4_000, 31);
        assert!(
            !default_path.exists(),
            "fitting in memory must not choose or mutate a persistence destination"
        );
        assert_eq!(supports.num_bins(), crate::torch::bar_dist::NUM_BAR_BINS);
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
        let open = bar_time_ids(et("2024-01-16T09:30:00"), None, RES, None);
        assert_eq!(open[TIME_MINUTE], 570);
        assert_eq!(open[TIME_WEEKDAY], 1, "2024-01-16 was a Tuesday");
        assert_eq!(open[TIME_SESSION], 2);
        assert_eq!(open[TIME_RESOLUTION], resolution_class(RES));

        // Daylight time in July: the same wall clock, an hour different in UTC. If the
        // producer used a fixed offset one of these two would be off by 60 minutes.
        let summer = bar_time_ids(et("2024-07-16T09:30:00"), None, RES, None);
        assert_eq!(
            summer[TIME_MINUTE], 570,
            "09:30 ET is minute 570 in DST too"
        );
        assert_ne!(
            et("2024-01-16T09:30:00").rem_euclid(86_400_000),
            et("2024-07-16T09:30:00").rem_euclid(86_400_000),
            "the two instants must differ in UTC time-of-day, else DST is being ignored"
        );

        // Spring forward 2024-03-10: 02:00 ET never happens, 03:00 follows 01:59.
        let spring_before = bar_time_ids(et("2024-03-10T01:30:00"), None, RES, None);
        let spring_after = bar_time_ids(et("2024-03-10T03:30:00"), None, RES, None);
        assert_eq!(spring_before[TIME_MINUTE], 90);
        assert_eq!(spring_after[TIME_MINUTE], 210);
        // Fall back 2024-11-03: 01:00-02:00 ET happens twice; both renderings are minute 60+.
        let before = et("2024-11-03T00:30:00") + 3_600_000;
        let after = before + 3_600_000;
        assert_eq!(bar_time_ids(before, None, RES, None)[TIME_MINUTE], 90);
        assert_eq!(bar_time_ids(after, None, RES, None)[TIME_MINUTE], 90);

        // Half day: 2024-11-29, the day after Thanksgiving, closes at 13:00 ET. The session id
        // is a wall-clock regime, not an is-the-market-open flag, so 13:05 is still `regular`
        // — an early close is not derivable from a fixed boundary table and the model sees it
        // through the bars themselves (thin or absent) rather than through this id.
        let half_day_close = bar_time_ids(et("2024-11-29T12:55:00"), None, RES, None);
        let half_day_after = bar_time_ids(et("2024-11-29T13:05:00"), None, RES, None);
        assert_eq!(half_day_close[TIME_SESSION], 2);
        assert_eq!(half_day_after[TIME_SESSION], 2);
        assert_eq!(half_day_close[TIME_MINUTE], 775);
        assert_eq!(half_day_after[TIME_MINUTE], 785);
        assert_eq!(half_day_close[TIME_WEEKDAY], 4);

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
                bar_time_ids(et(text), None, RES, None)[TIME_SESSION],
                session,
                "session at {text}"
            );
        }
    }

    #[test]
    fn deterministic_forecast_clock_skips_full_day_holidays_and_weekends() {
        for closed in [
            "2023-01-02", // observed New Year
            "2024-01-15", // MLK
            "2024-02-19", // Presidents
            "2024-03-29", // Good Friday
            "2024-05-27", // Memorial
            "2024-06-19", // Juneteenth
            "2024-07-04", // Independence
            "2024-09-02", // Labor
            "2024-11-28", // Thanksgiving
            "2024-12-25", // Christmas
        ] {
            let date = NaiveDate::parse_from_str(closed, "%Y-%m-%d").expect("test date");
            assert!(!is_us_equity_trading_date(date), "{closed} must be closed");
        }
        assert!(
            is_us_equity_trading_date(NaiveDate::from_ymd_opt(2021, 6, 18).expect("test date")),
            "Juneteenth was not an NYSE full-day closure before 2022"
        );

        let schedule = forecast_schedule_after(et("2024-07-03T19:55:00"), 2, RES);
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule[0], et("2024-07-05T04:00:00"));
        assert_eq!(schedule[1], et("2024-07-05T04:05:00"));
    }

    #[test]
    fn deterministic_forecast_clock_keeps_et_wall_time_across_dst() {
        let schedule = forecast_schedule_after(et("2024-03-08T19:55:00"), 2, RES);
        assert_eq!(
            schedule,
            vec![et("2024-03-11T04:00:00"), et("2024-03-11T04:05:00")]
        );
        assert_eq!(
            forecast_schedule_previous(schedule[0], RES),
            et("2024-03-08T19:55:00")
        );
    }

    #[test]
    fn daily_forecast_clock_retains_wall_time_across_weekends_holidays_and_dst() {
        let spring = forecast_schedule_after(et("2024-03-08T13:37:00"), 2, 86_400);
        assert_eq!(
            spring,
            vec![et("2024-03-11T13:37:00"), et("2024-03-12T13:37:00")],
            "the spring-DST weekend must not reset daily decisions to 04:00"
        );
        assert_eq!(
            forecast_schedule_previous(spring[0], 86_400),
            et("2024-03-08T13:37:00")
        );

        let holiday = forecast_schedule_after(et("2024-07-03T13:37:00"), 1, 86_400);
        assert_eq!(
            holiday,
            vec![et("2024-07-05T13:37:00")],
            "Independence Day must be skipped without changing the wall clock"
        );

        let fall = forecast_schedule_after(et("2024-11-01T13:37:00"), 1, 86_400);
        assert_eq!(
            fall,
            vec![et("2024-11-04T13:37:00")],
            "the fall-DST weekend must retain the decision wall clock"
        );
    }

    #[test]
    fn forecast_ids_have_exact_length_and_no_observed_future_channels() {
        let first = et("2024-11-27T19:55:00");
        let ids = forecast_schedule_ids_from(first, 100, RES);
        assert_eq!(ids.len(), 100, "H100 must emit exactly 100 ID rows");
        for row in &ids {
            for feature in BAR_TIME_MARKET {
                assert_eq!(row[feature], MARKET_MISSING);
            }
        }
        // The API accepts only the already-known first scheduled instant. Future symbol
        // timestamps and availability have no representation, so a halt cannot alter the IDs.
        assert_eq!(ids, forecast_schedule_ids_from(first, 100, RES));
        let thanksgiving = NaiveDate::from_ymd_opt(2024, 11, 28).expect("test date");
        assert!(ids.iter().all(|row| {
            !(row[TIME_WEEKDAY] == thanksgiving.weekday().num_days_from_monday() as i64)
        }));
    }

    #[test]
    fn future_symbol_halt_keeps_generation_inputs_and_gap_aligns_h100_targets() {
        let make = |label: &str, halt: bool| {
            let dir = std::env::temp_dir().join(format!(
                "trading_bot_0_clock_{label}_{}",
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("clock fixture dir");
            let mut bars = synth_bars(91, 500, et("2024-07-08T04:00:00"));
            if halt {
                // The decision is bar 1. Six scheduled prints disappear, then every later
                // observation resumes in its true wall-clock slot.
                for bar in &mut bars[2..] {
                    let mut shifted = *bar;
                    shifted.ts_ms = shifted.ts() + 30 * 60 * 1000;
                    *bar = shifted;
                }
            }
            write_bar_file(&bar_path(&dir, "AAA"), "AAA", RES, &bars).expect("clock bars");
            let corpus = BarCorpus::load(&dir, RES, 100).expect("clock corpus");
            (Fixture { dir }, corpus)
        };
        let (_base_dir, base) = make("base", false);
        let (_halt_dir, halted) = make("halt", true);
        let reference = WindowRef {
            symbol: 0,
            bar_index: 1,
        };
        let base_sampler = BarSampler::new(&base, Split::Train, 1, 7);
        let halted_sampler = BarSampler::new(&halted, Split::Train, 1, 7);
        let base_ids = base_sampler
            .forecast_time_ids(&[reference], 0, 100, Device::Cpu)
            .expect("base forecast clock");
        let halted_ids = halted_sampler
            .forecast_time_ids(&[reference], 0, 100, Device::Cpu)
            .expect("halted forecast clock");
        assert!(
            base_ids.equal(&halted_ids),
            "future observed timestamps changed the H100 sampler clock"
        );
        let base_history = base_sampler
            .batch_of(&[reference], Device::Cpu)
            .dof
            .narrow(1, 0, 1);
        let halted_history = halted_sampler
            .batch_of(&[reference], Device::Cpu)
            .dof
            .narrow(1, 0, 1);
        assert!(
            base_history.equal(&halted_history),
            "a future halt changed the observed generation history"
        );
        assert_eq!(base_ids.size(), vec![1, 100, BAR_TIME_FEATURES as i64]);

        let targets = halted_sampler
            .aligned_future_dof(&[reference], 0, 100, Device::Cpu)
            .expect("halt-aligned targets");
        assert_eq!(
            targets.size(),
            vec![1, 100, BAR_DOF as i64],
            "scoring must retain exactly H rows"
        );
        let flat = BarDof::default().to_array();
        for slot in 0..6i64 {
            for (dof, &expected) in flat.iter().enumerate() {
                assert_eq!(
                    targets.double_value(&[0, slot, dof as i64]) as f32,
                    expected,
                    "missing scheduled slot {slot} was not a flat carry"
                );
            }
        }

        let bars = halted.bars(0);
        let decision_close = bars[1].close;
        let first_return = (bars[2].close / decision_close).ln();
        assert!(
            (targets.double_value(&[0, 6, DOF_R as i64]) - first_return as f64).abs() < 1e-6,
            "the reappearance payoff did not stay at scheduled slot 6"
        );
        let cumulative = targets
            .select(-1, DOF_R as i64)
            .sum(Kind::Double)
            .double_value(&[]);
        let terminal = (bars[95].close / decision_close).ln() as f64;
        assert!(
            (cumulative - terminal).abs() < 1e-5,
            "catch-up payoff was lost or counted more than once: {cumulative} vs {terminal}"
        );
    }

    #[test]
    fn conditioning_ids_are_always_valid_embedding_rows() {
        // A dense sweep over five years at 7-minute steps walks every hour of every weekday
        // and both DST transitions ten times over, plus deliberately hostile inputs. The
        // predecessor is swept alongside so the elapsed and day-edge channels are exercised
        // over spacings from one bar to years, not only over the adjacent case.
        let start = et("2021-01-01T00:00:00");
        let mut ts = start;
        let mut gap = RES_MS;
        while ts < start + 5 * 365 * 86_400_000 {
            for prev in [None, Some(ts - gap), Some(ts), Some(ts + gap)] {
                let ids = bar_time_ids(ts, prev, RES, None);
                for f in 0..BAR_TIME_FEATURES {
                    assert!(
                        (0..BAR_TIME_CARDINALITY[f]).contains(&ids[f]),
                        "{} id {} out of range at {} with prev {prev:?}",
                        BAR_TIME_NAMES[f],
                        ids[f],
                        iso_ms(ts)
                    );
                }
            }
            ts += 7 * 60 * 1000;
            gap = gap.saturating_mul(2).min(400 * 86_400_000);
        }
        for hostile in [i64::MIN / 4, -1, 0, 1, i64::MAX / 4] {
            for prev in [None, Some(i64::MIN / 4), Some(0), Some(i64::MAX / 4)] {
                let ids = bar_time_ids(hostile, prev, 0, None);
                for f in 0..BAR_TIME_FEATURES {
                    assert!((0..BAR_TIME_CARDINALITY[f]).contains(&ids[f]));
                }
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
        assert_eq!(
            splice_stats.splices, 1,
            "level jump across a hole is a splice"
        );
        assert_eq!(splice_stats.ticks, 0);

        let tick_stats = by_symbol["TICK"];
        assert_eq!(tick_stats.ticks, 1, "a reverting single print is a tick");
        assert_eq!(tick_stats.splices, 0);
        assert_eq!(tick_stats.holes, 0);
        assert!(
            tick_stats.extreme_range >= 1,
            "a 50x bar has an extreme range"
        );

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
                    for_each_window_dof(bars, anchor, len, DofScaling::Raw, |_, row| {
                        n += row.dof.r.is_finite() as usize
                    });
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
                    for_each_window_dof(bars, start, len, DofScaling::Raw, |bar, encoded| {
                        d[slot * BAR_DOF..(slot + 1) * BAR_DOF]
                            .copy_from_slice(&encoded.dof.to_array());
                        c[slot * BAR_TIME_FEATURES..(slot + 1) * BAR_TIME_FEATURES]
                            .copy_from_slice(&bar_time_ids(
                                bar.ts(),
                                Some(bars[start + slot - 1].ts()),
                                RES,
                                corpus.market_channel(),
                            ));
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

    // -----------------------------------------------------------------------
    // Pass partition: the "one epoch is one pass over every bar" invariant
    // -----------------------------------------------------------------------

    /// Ramp contexts for the fixture corpus, scaled down from the production 896 / 1472 / 2048
    /// but with the same shape: three 64-multiples whose ratios are 1 : 1.64 : 2.29, so the
    /// deficit carry, the per-symbol tiling and the sub-context hole all behave as they do on
    /// the real corpus. The fixture's symbols are a few thousand bars, so production contexts
    /// would make every symbol shorter than one window and test nothing.
    const PASS_CONTEXTS: [i64; 3] = [56, 92, 128];

    /// The token weights of a FLAT batch ramp: `batch[s] * context[s]` with one batch unit at
    /// every stage is the contexts themselves. This is the ramp the card of record actually
    /// derives, so it is the one the partition is exercised under.
    const FLAT_WEIGHTS: [f64; 3] = [56.0, 92.0, 128.0];

    /// Issue every window of every stage exactly once, in batches, exactly as the trainer does:
    /// a sequential cursor per stage and a SHORT final batch rather than a dropped one.
    fn issue_full_pass(layout: &PassLayout, ledger: &mut PassLedger, batch: usize) {
        for stage in 0..layout.stages() {
            let mut cursor = 0usize;
            while cursor < layout.windows(stage).len() {
                let drawn = layout.draw(stage, cursor, batch).len();
                assert!(drawn > 0, "stage {stage} draw stalled at cursor {cursor}");
                ledger.mark(stage, cursor, drawn);
                cursor += drawn;
            }
        }
    }

    /// Per-bar issue counts and per-bar stage labels, derived from the LAYOUT rather than from
    /// the plan's own bookkeeping. `counts[symbol][i]` is how many windows targeted bar `i` of
    /// that symbol; `stages[symbol][i]` is the stage that did, or `u8::MAX`.
    fn per_bar_coverage(
        corpus: &BarCorpus,
        layout: &PassLayout,
        contexts: &[i64],
    ) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut counts: Vec<Vec<u8>> = (0..corpus.series_count())
            .map(|s| vec![0u8; corpus.series_len(s)])
            .collect();
        let mut stages: Vec<Vec<u8>> = (0..corpus.series_count())
            .map(|s| vec![u8::MAX; corpus.series_len(s)])
            .collect();
        for stage in 0..layout.stages() {
            for window in layout.windows(stage) {
                let symbol = window.symbol as usize;
                let anchor = window.bar_index as usize;
                for bar in anchor + 1..=anchor + contexts[stage] as usize {
                    counts[symbol][bar] += 1;
                    stages[symbol][bar] = stage as u8;
                }
            }
        }
        (counts, stages)
    }

    /// The headline invariant, checked two independent ways: through the audit's own buckets and
    /// through a per-bar tally rebuilt from the layout.
    ///
    /// Before the partition this was 45.9% of bars once, 28.7% never, 22.3% twice and 3.2% three
    /// times on the production corpus, because each ramp stage walked its own stride-`C` anchor
    /// list from index 0 and none of them got through it.
    #[test]
    fn one_pass_targets_every_training_bar_exactly_once() {
        let (_fx, corpus) = fixture("pass_once");
        let plan = PassPlan::new(&corpus, Split::Train, &PASS_CONTEXTS, &FLAT_WEIGHTS, 7).unwrap();
        let layout = plan.layout(0);
        let mut ledger = PassLedger::new(&layout);
        issue_full_pass(&layout, &mut ledger, 24);
        let audit = plan.audit(&layout, &ledger);
        audit.require_full_pass().expect("a full pass must pass");

        // The multiplicity histogram is a single spike at one, plus the named remainder at zero.
        assert_eq!(audit.multiplicity_bars[1], plan.covered_bars());
        assert_eq!(audit.multiplicity_bars[0], plan.remainder().total());
        assert_eq!(audit.multiplicity_bars[2], 0);
        assert_eq!(audit.multiplicity_bars[3], 0);
        assert_eq!(
            audit.multiplicity_bars.iter().sum::<u64>(),
            plan.split_bars(),
            "the buckets must account for every split bar"
        );
        assert_eq!(audit.stage_coverage(), vec![1.0, 1.0, 1.0]);

        // And the same fact rebuilt from the windows themselves: no bar targeted twice, and the
        // bars targeted zero times are exactly the head, hole and short-symbol remainder.
        let (counts, _) = per_bar_coverage(&corpus, &layout, &PASS_CONTEXTS);
        let mut once = 0u64;
        let mut never = 0u64;
        for symbol in 0..corpus.series_count() {
            let (lo, hi) = corpus.split_range(symbol, Split::Train);
            for bar in lo..hi {
                match counts[symbol][bar] {
                    0 => never += 1,
                    1 => once += 1,
                    n => panic!(
                        "{} bar {bar} was a prediction target {n} times",
                        corpus.symbol(symbol)
                    ),
                }
            }
            // Nothing outside the split may be targeted at all: a window whose targets crossed
            // the boundary would be a lookahead leak, not merely a coverage error.
            assert!(
                counts[symbol][..lo].iter().all(|&n| n == 0)
                    && counts[symbol][hi..].iter().all(|&n| n == 0),
                "{} has targets outside the train split",
                corpus.symbol(symbol)
            );
        }
        assert_eq!(once, plan.covered_bars());
        assert_eq!(never, plan.remainder().total());
        assert_eq!(once + never, plan.split_bars());

        // Conditioning depth is `(C + 1) / 2` at stride `C`: bar `j` of a window is predicted
        // from `j` bars of history, `j` running 1..=C.
        for stage in 0..PASS_CONTEXTS.len() {
            let want = (PASS_CONTEXTS[stage] as f64 + 1.0) / 2.0;
            assert!((plan.mean_conditioning_bars(stage) - want).abs() < 1e-12);
            assert!((audit.mean_conditioning_bars[stage] - want).abs() < 1e-12);
        }
        let pass_mean = plan.pass_mean_conditioning_bars();
        assert!(
            pass_mean > plan.mean_conditioning_bars(0)
                && pass_mean < plan.mean_conditioning_bars(2),
            "the bar-weighted mean {pass_mean} must sit between the extreme stages"
        );
    }

    /// The cross-pass census reports reuse that no per-pass audit can express, and it is EXACT
    /// against a per-bar tally rebuilt independently from the layouts.
    ///
    /// The defect this closes cost an entire analysis session. `audit` is a PER-PASS census and
    /// `require_full_pass` pins its multiplicity histogram to a single spike at one, so on the
    /// third pass of a three-pass run it reads "twice: 0, three or more: 0" exactly as on the
    /// first — a correct statement that reads as "no bar was ever seen twice". So the assertions
    /// below are deliberately paired: the per-pass audit is checked to STILL say zero, and the
    /// cumulative census is checked to say three, on the same passes, in the same call.
    #[test]
    fn the_cross_pass_census_sees_reuse_the_per_pass_audit_cannot() {
        let (_fx, corpus) = fixture("cross_pass_census");
        let plan = PassPlan::new(&corpus, Split::Train, &PASS_CONTEXTS, &FLAT_WEIGHTS, 19).unwrap();
        let contexts: Vec<i64> = PASS_CONTEXTS.to_vec();
        // An independent per-bar tally, built the same way `per_bar_coverage` builds one: the
        // reference the census is checked against is a direct count, not another formula.
        let mut exposures: Vec<Vec<u32>> = (0..corpus.series_count())
            .map(|symbol| vec![0u32; corpus.series_len(symbol)])
            .collect();
        let mut census = PassCensus::default();
        const PASSES: usize = 3;
        for epoch in 0..PASSES {
            let layout = plan.layout(epoch);
            let mut ledger = PassLedger::new(&layout);
            issue_full_pass(&layout, &mut ledger, 24);
            let audit = plan.audit(&layout, &ledger);
            audit.require_full_pass().expect("every pass must be full");
            // THE MISLEADING READING, asserted to still be produced: the per-pass panel says no
            // bar was targeted twice, on the third pass as on the first.
            assert_eq!(audit.multiplicity_bars[2], 0, "epoch {epoch}");
            assert_eq!(audit.multiplicity_bars[3], 0, "epoch {epoch}");

            let run = plan.cumulative_coverage(&census, &layout, &ledger);
            run.require_accounted()
                .expect("the run must account for every bar");
            for stage in 0..layout.stages() {
                for window in layout.windows(stage) {
                    let anchor = window.bar_index as usize;
                    for bar in anchor + 1..=anchor + contexts[stage] as usize {
                        exposures[window.symbol as usize][bar] += 1;
                    }
                }
            }
            // The census, mid-pass-complete, must equal the independent tally bar for bar.
            let mut want = [0u64; MULTIPLICITY_BUCKETS];
            for symbol in &exposures {
                for &count in symbol {
                    want[(count as usize).min(MULTIPLICITY_BUCKETS - 1)] += 1;
                }
            }
            // The tally walks EVERY bar of every file; the census denominator is the split. The
            // difference is bars outside the train split, which the tally leaves at zero, so only
            // bucket zero differs and every other bucket must match exactly.
            for bucket in 1..MULTIPLICITY_BUCKETS {
                assert_eq!(
                    run.multiplicity_bars[bucket], want[bucket],
                    "epoch {epoch} bucket {bucket}: census {:?} against the independent tally \
                     {want:?}",
                    run.multiplicity_bars
                );
            }
            assert_eq!(run.completed_passes, epoch);
            census.absorb(&layout);
        }

        // And the headline: after three full passes the RUN has targeted essentially the whole
        // covered split three times, which is the fact `audit` asserted to be zero above.
        let layout = plan.layout(PASSES);
        let ledger = PassLedger::new(&layout);
        let run = plan.cumulative_coverage(&census, &layout, &ledger);
        run.require_accounted()
            .expect("the run must account for every bar");
        assert_eq!(run.completed_passes, PASSES);
        assert_eq!(
            run.bar_target_events,
            PASSES as u64 * plan.covered_bars(),
            "three complete passes deliver exactly three passes' worth of bar-target events"
        );
        assert!(
            (run.effective_epochs() - PASSES as f64).abs() < 1e-12,
            "effective epochs {} after {PASSES} complete passes",
            run.effective_epochs()
        );
        assert!(
            run.multiplicity_bars[3] > 0,
            "the cross-pass census must place mass at three-or-more: {}",
            run.summary()
        );
        assert!(
            run.reused_fraction() > 0.5,
            "most of the split has been targeted more than once: {}",
            run.summary()
        );
        // The moving hole is the ONLY reason a bar can sit below three after three full passes,
        // and it is bounded by three holes' worth of bars. This is the term a per-bar counter
        // would have cost 368 MB to obtain and that the hole-start reconstruction gets exactly.
        let below = run.multiplicity_bars[1] + run.multiplicity_bars[2];
        assert!(
            below <= PASSES as u64 * plan.remainder().hole_bars,
            "{below} bars below three exposures exceeds three holes ({}); the reconstruction is \
             crediting or losing passes",
            plan.remainder().hole_bars
        );
    }

    /// The stages own DISJOINT stretches of every symbol's axis. This is the property that makes
    /// the tiling exact across stages whose strides differ, and it is checked on bars rather than
    /// on anchor indices because index `k` of the 56-bar list and index `k` of the 128-bar list
    /// address different bars.
    #[test]
    fn the_partition_is_disjoint_across_stages() {
        let (_fx, corpus) = fixture("pass_disjoint");
        let plan = PassPlan::new(&corpus, Split::Train, &PASS_CONTEXTS, &FLAT_WEIGHTS, 11).unwrap();
        for epoch in [0usize, 1, 5] {
            let layout = plan.layout(epoch);
            let (counts, stages) = per_bar_coverage(&corpus, &layout, &PASS_CONTEXTS);
            let mut owned = [0u64; PASS_CONTEXTS.len()];
            for symbol in 0..corpus.series_count() {
                for bar in 0..corpus.series_len(symbol) {
                    assert!(
                        counts[symbol][bar] <= 1,
                        "epoch {epoch}: {} bar {bar} is owned by more than one stage",
                        corpus.symbol(symbol)
                    );
                    if stages[symbol][bar] != u8::MAX {
                        owned[stages[symbol][bar] as usize] += 1;
                    }
                }
            }
            // Each stage's bar count is its window count times its context, exactly - which is
            // only true if no window's target span overlaps another's.
            for stage in 0..PASS_CONTEXTS.len() {
                assert_eq!(
                    owned[stage],
                    plan.windows_per_stage()[stage] as u64 * PASS_CONTEXTS[stage] as u64,
                    "epoch {epoch} stage {stage}"
                );
            }
            assert_eq!(owned.iter().sum::<u64>(), plan.covered_bars());
            // Window counts are a property of the corpus and the ramp, so the step schedule can
            // be derived from them ONCE; only the geometry moves with the epoch.
            for stage in 0..PASS_CONTEXTS.len() {
                assert_eq!(layout.windows(stage).len(), plan.windows_per_stage()[stage]);
            }
        }
    }

    /// A schedule that runs out of steps before a stage has issued its share must FAIL, not warn.
    ///
    /// This is the exact shape of the defect the partition exists to remove: the old sampler left
    /// 53% / 66% / 80% of each stage's list unissued and reported the run as `--epochs 1`. Both
    /// directions are checked, because a consumption bug can also issue a window twice while
    /// leaving another untouched.
    #[test]
    fn the_coverage_assertion_fires_on_a_truncated_pass() {
        let (_fx, corpus) = fixture("pass_truncated");
        let plan = PassPlan::new(&corpus, Split::Train, &PASS_CONTEXTS, &FLAT_WEIGHTS, 13).unwrap();
        let layout = plan.layout(0);

        // Truncated: stage 1 stops short, as a held batch or a short `--steps` would leave it.
        const HELD: usize = 5;
        assert!(
            layout.windows(1).len() > HELD,
            "the fixture must assign stage 1 more than {HELD} windows to withhold any"
        );
        let mut ledger = PassLedger::new(&layout);
        for stage in 0..layout.stages() {
            let assigned = layout.windows(stage).len();
            let issue = if stage == 1 {
                assigned - HELD
            } else {
                assigned
            };
            ledger.mark(stage, 0, issue);
        }
        let audit = plan.audit(&layout, &ledger);
        let err = audit
            .require_full_pass()
            .expect_err("a short pass must be fatal");
        let message = format!("{err:#}");
        assert!(
            message.contains("did not complete a pass"),
            "the error must name the failure: {message}"
        );
        assert_eq!(audit.unissued_per_stage(), vec![0, HELD, 0]);
        assert_eq!(audit.unissued_bars(), HELD as u64 * PASS_CONTEXTS[1] as u64);
        // The shortfall shows up as bars at multiplicity ZERO, not as a smaller denominator.
        assert_eq!(
            audit.multiplicity_bars[0],
            plan.remainder().total() + HELD as u64 * PASS_CONTEXTS[1] as u64
        );
        assert!(audit.coverage_fraction() < 1.0);

        // Over-issued: the same window handed out twice while another goes untouched.
        let mut ledger = PassLedger::new(&layout);
        issue_full_pass(&layout, &mut ledger, 24);
        ledger.mark(2, 0, 4);
        let audit = plan.audit(&layout, &ledger);
        let message = format!(
            "{:#}",
            audit
                .require_full_pass()
                .expect_err("a repeated window must be fatal")
        );
        assert!(
            message.contains("more than once"),
            "the error must name the repeat: {message}"
        );
        assert_eq!(audit.multiplicity_bars[2], 4 * PASS_CONTEXTS[2] as u64);
    }

    /// Which stage owns a bar decides how much history it is predicted from - 448 / 736 / 1024
    /// bars on the production ramp. If that assignment correlated with the symbol, some tickers
    /// would be seen only at short context forever; if it correlated with calendar position, the
    /// most recent bars would always be the shallow ones. Both are checked with a chi-square
    /// against the independence null.
    ///
    /// Liquidity needs no separate test: it is a per-symbol property, so symbol independence
    /// implies it. Calendar position is tested WITHIN each symbol's axis, which is what a
    /// staggered listing date makes distinct from wall-clock time.
    #[test]
    fn partition_assignment_is_independent_of_symbol_and_calendar_position() {
        let (_fx, corpus) = fixture("pass_independence");
        let plan = PassPlan::new(&corpus, Split::Train, &PASS_CONTEXTS, &FLAT_WEIGHTS, 17).unwrap();
        const STAGES: usize = PASS_CONTEXTS.len();
        const QUARTILES: usize = 4;
        // 50 epochs: each is an independent draw of the block order and the hole position, and
        // the table has to be large enough that a real dependence would show. The per-symbol
        // window counts do NOT move with the epoch, so the symbol table is the same every epoch
        // and is accumulated once.
        const EPOCHS: usize = 50;

        let symbols = corpus.series_count();
        let mut by_symbol = vec![0f64; STAGES * symbols];
        let mut by_quartile = vec![0f64; STAGES * QUARTILES];
        for epoch in 0..EPOCHS {
            let layout = plan.layout(epoch);
            for stage in 0..STAGES {
                for window in layout.windows(stage) {
                    let symbol = window.symbol as usize;
                    if epoch == 0 {
                        by_symbol[stage * symbols + symbol] += 1.0;
                    }
                    // Position of the window's target span inside the symbol's own train axis.
                    let (lo, hi) = corpus.split_range(symbol, Split::Train);
                    let span = (hi - lo).max(1) as f64;
                    let offset = (window.bar_index as usize - lo) as f64 / span;
                    let quartile = ((offset * QUARTILES as f64) as usize).min(QUARTILES - 1);
                    by_quartile[stage * QUARTILES + quartile] += 1.0;
                }
            }
        }

        /// Pearson chi-square of a `rows x cols` contingency table against independence.
        fn chi_square(table: &[f64], rows: usize, cols: usize) -> f64 {
            let total: f64 = table.iter().sum();
            let row_sums: Vec<f64> = (0..rows)
                .map(|r| table[r * cols..(r + 1) * cols].iter().sum())
                .collect();
            let col_sums: Vec<f64> = (0..cols)
                .map(|c| (0..rows).map(|r| table[r * cols + c]).sum())
                .collect();
            let mut stat = 0.0;
            for r in 0..rows {
                for c in 0..cols {
                    let expected = row_sums[r] * col_sums[c] / total;
                    if expected > 0.0 {
                        let diff = table[r * cols + c] - expected;
                        stat += diff * diff / expected;
                    }
                }
            }
            stat
        }

        // 0.999 quantiles: df = (3-1)(3-1) = 4 -> 18.47, df = (3-1)(4-1) = 6 -> 22.46. A
        // dependence strong enough to matter lands orders of magnitude above these.
        let symbol_stat = chi_square(&by_symbol, STAGES, symbols);
        assert!(
            symbol_stat < 18.47,
            "stage assignment is not independent of symbol: chi-square {symbol_stat} on \
             {by_symbol:?} (df {}, 0.999 quantile 18.47)",
            (STAGES - 1) * (symbols - 1)
        );
        let calendar_stat = chi_square(&by_quartile, STAGES, QUARTILES);
        assert!(
            calendar_stat < 22.46,
            "stage assignment is not independent of calendar position: chi-square \
             {calendar_stat} on {by_quartile:?} (df {}, 0.999 quantile 22.46)",
            (STAGES - 1) * (QUARTILES - 1)
        );
    }

    /// The remainder policy, on the two cases the production corpus's `--min-bars 20480` hides.
    ///
    /// A symbol shorter than the LONGEST context is covered by the stages that fit, at shorter
    /// context - it is not dropped. A symbol shorter than the SHORTEST context cannot form one
    /// window of any stage, so it is excluded, counted and reported under its own cause. Nothing
    /// falls on the floor: `covered + head + short-symbol + hole == split bars`, exactly.
    #[test]
    fn short_symbols_are_covered_at_a_shorter_context_or_counted_out() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_dataset_pass_short_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let _fx = Fixture { dir: dir.clone() };
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        // LONG sets the split instant; MID's train axis fits stage 0 and 1 windows but not one
        // stage-2 window; SHORT's fits no window of any stage.
        for (symbol, seed, count) in [
            ("LONG", 1u64, 4_000usize),
            ("MID", 2, 126),
            ("SHORT", 3, 40),
        ] {
            let bars = synth_bars(seed, count, base);
            write_bar_file(&bar_path(&dir, symbol), symbol, RES, &bars).unwrap();
        }
        // `--min-bars` below the shortest ramp context on purpose: this is the case the
        // production 20,480 hides, and it has to be exercised somewhere.
        let corpus = BarCorpus::load(&dir, RES, 20).unwrap();
        let plan = PassPlan::new(&corpus, Split::Train, &PASS_CONTEXTS, &FLAT_WEIGHTS, 19).unwrap();
        let index = |name: &str| corpus.symbols().iter().position(|s| s == name).unwrap();
        let (mid, short) = (index("MID"), index("SHORT"));
        let (mid_lo, mid_hi) = corpus.split_range(mid, Split::Train);
        let (short_lo, short_hi) = corpus.split_range(short, Split::Train);
        let mid_targets = mid_hi - mid_lo.max(1) - 1;
        let short_targets = short_hi - short_lo.max(1) - 1;
        assert!(
            mid_targets >= PASS_CONTEXTS[0] as usize && mid_targets < PASS_CONTEXTS[2] as usize,
            "MID must fit a short-context window but not the longest: {mid_targets} targets"
        );
        assert!(
            short_targets < PASS_CONTEXTS[0] as usize,
            "SHORT must fit no window at all: {short_targets} targets"
        );

        // SHORT is excluded with a counted, named reason.
        let remainder = plan.remainder();
        assert_eq!(remainder.short_symbols, 1);
        assert_eq!(remainder.short_symbol_bars, short_targets as u64);
        // Conservation: every split bar is covered or in a named bucket.
        assert_eq!(
            plan.covered_bars() + remainder.total(),
            plan.split_bars(),
            "the remainder must account for every uncovered bar"
        );
        // The head is the un-targetable bar 0 and first anchor of each symbol.
        assert!(remainder.head_bars >= corpus.series_count() as u64);

        let layout = plan.layout(0);
        let mut mid_windows = [0usize; 3];
        for stage in 0..PASS_CONTEXTS.len() {
            for window in layout.windows(stage) {
                assert_ne!(
                    window.symbol as usize, short,
                    "SHORT must own no window at any stage"
                );
                if window.symbol as usize == mid {
                    mid_windows[stage] += 1;
                }
            }
        }
        assert_eq!(mid_windows[2], 0, "MID cannot fit a stage-2 window");
        assert!(
            mid_windows[0] + mid_windows[1] > 0,
            "MID must be covered by the stages that do fit: {mid_windows:?}"
        );

        // And a full pass over the reduced corpus still satisfies the invariant, with SHORT's
        // bars sitting in the zero bucket under a stated cause rather than as an unexplained gap.
        let mut ledger = PassLedger::new(&layout);
        issue_full_pass(&layout, &mut ledger, 16);
        let audit = plan.audit(&layout, &ledger);
        audit.require_full_pass().expect("a full pass must pass");
        assert_eq!(audit.multiplicity_bars[0], remainder.total());
        assert!(
            audit.summary().contains("short-symbol"),
            "the summary must name the cause: {}",
            audit.summary()
        );
    }

    /// One name plus the market proxy over the same instants, with two deliberate features:
    /// a 200-bar proxy HOLE inside the name's span, and one proxy bar that is exactly FLAT.
    /// Those are the two states the market channel has to keep apart, and no synthetic random
    /// walk produces either on its own.
    fn market_fixture(label: &str) -> (Fixture, BarCorpus, i64) {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_market_{label}_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        write_bar_file(
            &bar_path(&dir, "AAA"),
            "AAA",
            RES,
            &synth_bars(1, 3_000, base),
        )
        .unwrap();

        let mut proxy = synth_bars(9, 3_000, base);
        // `PackedBar` is `repr(C, packed)`; edit a copy and write it back rather than binding a
        // reference to a packed field.
        let flat_price = proxy[MARKET_FLAT_BAR - 1].close;
        let mut flat = proxy[MARKET_FLAT_BAR];
        flat.open = flat_price;
        flat.high = flat_price;
        flat.low = flat_price;
        flat.close = flat_price;
        let flat_ts = flat.ts();
        proxy[MARKET_FLAT_BAR] = flat;
        proxy.drain(MARKET_HOLE_START..MARKET_HOLE_END);
        write_bar_file(
            &bar_path(&dir, MARKET_PROXY_SYMBOL),
            MARKET_PROXY_SYMBOL,
            RES,
            &proxy,
        )
        .unwrap();

        let corpus = BarCorpus::load(&dir, RES, 100).unwrap();
        (Fixture { dir }, corpus, flat_ts)
    }

    /// Proxy bar index forced exactly flat by [`market_fixture`].
    const MARKET_FLAT_BAR: usize = 800;
    /// Proxy bar indices removed by [`market_fixture`], i.e. the coverage hole.
    const MARKET_HOLE_START: usize = 1_200;
    const MARKET_HOLE_END: usize = 1_400;

    fn series_of(corpus: &BarCorpus, symbol: &str) -> usize {
        corpus
            .symbols()
            .iter()
            .position(|s| s == symbol)
            .unwrap_or_else(|| panic!("{symbol} is in the fixture"))
    }

    /// THE lookahead test. Predicting bar `t+1` may condition on the proxy's bar `t` and on
    /// nothing later, so the market ids at a slot must be the proxy's state at that slot's OWN
    /// timestamp. The test has teeth in both directions: it pins the correct join AND asserts
    /// that the one-bar-advanced join would disagree on most slots, so a shift of the market
    /// series by a single bar fails it.
    #[test]
    fn the_market_channel_joins_the_row_bar_and_never_the_next_one() {
        let (_fx, corpus, _) = market_fixture("nolook");
        let channel = corpus.market_channel().expect("the fixture has a proxy");
        let name = series_of(&corpus, "AAA");
        // Clear of the proxy hole, so every slot is an OBSERVED market row and the advanced-join
        // comparison below is not weakened by slots that are missing under both joins.
        let len = 512usize;
        let start = 300usize;
        let batch = corpus
            .dof_window(
                &[BarEndpoint {
                    series: name,
                    bar: start + len - 1,
                }],
                &[0],
                len as i64,
                Device::Cpu,
            )
            .expect("a window inside AAA");
        let bars = corpus.bars(name);

        let mut advanced_would_differ = 0usize;
        for slot in 0..len {
            let ts = bars[start + slot].ts();
            let want = channel.ids_at(ts);
            let advanced = channel.ids_at(ts + RES_MS);
            for (feature, channel_index) in BAR_TIME_MARKET.into_iter().enumerate() {
                let got = batch
                    .time_ids
                    .int64_value(&[0, slot as i64, channel_index as i64]);
                assert_eq!(
                    got,
                    want[feature],
                    "slot {slot} {} must read the proxy at its own instant {}",
                    BAR_TIME_NAMES[channel_index],
                    iso_ms(ts)
                );
            }
            if advanced != want {
                advanced_would_differ += 1;
            }
        }
        assert!(
            advanced_would_differ > len / 2,
            "only {advanced_would_differ} of {len} slots distinguish the correct join from a \
             one-bar advance, so this test would not catch a lookahead"
        );
    }

    /// A proxy coverage hole must reach the trunk as a RESERVED row, distinguishable from every
    /// observed state — above all from an exactly flat proxy bar, which is the state a
    /// zero-filled channel would be indistinguishable from.
    #[test]
    fn a_market_hole_is_a_reserved_row_and_not_a_flat_market() {
        let (_fx, corpus, flat_ts) = market_fixture("hole");
        let channel = corpus.market_channel().expect("the fixture has a proxy");

        // The flat proxy bar is OBSERVED: every channel of it is a real row, not the hole.
        let flat = channel.ids_at(flat_ts);
        for (feature, channel_index) in BAR_TIME_MARKET.into_iter().enumerate() {
            assert_ne!(
                flat[feature], MARKET_MISSING,
                "an exactly flat proxy bar is an observation, not a hole ({})",
                BAR_TIME_NAMES[channel_index]
            );
        }

        // The name's own bars over the proxy's hole must all read MISSING, and the batch must
        // count them.
        let name = series_of(&corpus, "AAA");
        let bars = corpus.bars(name);
        let len = MARKET_HOLE_END - MARKET_HOLE_START;
        let batch = corpus
            .dof_window(
                &[BarEndpoint {
                    series: name,
                    bar: MARKET_HOLE_END - 1,
                }],
                &[0],
                len as i64,
                Device::Cpu,
            )
            .expect("a window over the hole");
        for slot in 0..len {
            for channel_index in BAR_TIME_MARKET {
                assert_eq!(
                    batch
                        .time_ids
                        .int64_value(&[0, slot as i64, channel_index as i64]),
                    MARKET_MISSING,
                    "bar {} sits in the proxy hole",
                    iso_ms(bars[MARKET_HOLE_START + slot].ts())
                );
            }
        }
        assert_eq!(
            batch.market_missing, len,
            "every bar of this window is uncovered and the batch must say so"
        );

        // And a window clear of the hole is fully covered, so the counter is a measurement and
        // not a constant.
        let covered = corpus
            .dof_window(
                &[BarEndpoint {
                    series: name,
                    bar: MARKET_HOLE_START - 1,
                }],
                &[0],
                256,
                Device::Cpu,
            )
            .expect("a window before the hole");
        assert_eq!(covered.market_missing, 0);
    }

    /// Imagined bars use a deterministic exchange clock and never carry proxy state. The
    /// exogenous channels must remain invariant to which symbol supplies the already-observed
    /// decision timestamp; future per-symbol prints are not inputs.
    #[test]
    fn future_conditioning_ids_never_reveal_the_market() {
        let (_fx, corpus, _) = market_fixture("future");
        let name = series_of(&corpus, "AAA");
        let bars = corpus.bars(name);
        let from = 1_000usize;
        let steps = 64i64;
        let ids = corpus
            .future_time_ids(
                &[BarEndpoint {
                    series: name,
                    bar: from,
                }],
                0,
                steps,
                Device::Cpu,
            )
            .expect("future ids");

        let expected = forecast_schedule_ids_after(bars[from].ts(), steps as usize, RES);
        for step in 0..steps as usize {
            for channel_index in BAR_TIME_MARKET {
                assert_eq!(
                    ids.int64_value(&[0, step as i64, channel_index as i64]),
                    MARKET_MISSING,
                    "step {step} leaked the proxy's future state"
                );
            }
            for feature in 0..BAR_TIME_FEATURES {
                assert_eq!(
                    ids.int64_value(&[0, step as i64, feature as i64]),
                    expected[step][feature],
                    "step {step} {}",
                    BAR_TIME_NAMES[feature]
                );
            }
        }

        let proxy = series_of(&corpus, MARKET_PROXY_SYMBOL);
        let proxy_bars = corpus.bars(proxy);
        let same_decision = proxy_bars.partition_point(|bar| bar.ts() < bars[from].ts());
        assert_eq!(proxy_bars[same_decision].ts(), bars[from].ts());
        let proxy_ids = corpus
            .future_time_ids(
                &[BarEndpoint {
                    series: proxy,
                    bar: same_decision,
                }],
                0,
                steps,
                Device::Cpu,
            )
            .expect("proxy future ids");
        assert!(
            ids.equal(&proxy_ids),
            "symbol identity or future print availability changed the forecast clock"
        );

        // The same instant read WITH a channel is not missing, so the assertion above is about
        // the future-facing constructor and not about a corpus that has no proxy.
        let observed = bar_time_ids(
            bars[from].ts(),
            Some(bars[from - 1].ts()),
            RES,
            corpus.market_channel(),
        );
        assert_ne!(observed[TIME_MARKET_R], MARKET_MISSING);
    }

    /// `r` is a return over an interval of unknown length, and these two ids are the only thing
    /// that tells an overnight or post-holiday return apart from an ordinary five-minute one.
    #[test]
    fn the_elapsed_and_day_edge_ids_separate_a_gap_from_an_adjacent_bar() {
        // 2024-01-16T14:30:00Z is 09:30 ET.
        let open = et("2024-01-16T09:30:00");
        let adjacent = bar_time_ids(open + RES_MS, Some(open), RES, None);
        assert_eq!(adjacent[TIME_ELAPSED], 1, "one bar is the first bucket");
        assert_eq!(adjacent[TIME_DAY_EDGE], 1, "same ET day");

        // Powers of two land on consecutive buckets, which is the whole point of a log axis.
        for (bars, bucket) in [
            (1i64, 1i64),
            (2, 2),
            (3, 2),
            (4, 3),
            (7, 3),
            (8, 4),
            (100, 7),
        ] {
            assert_eq!(
                bar_time_ids(open + bars * RES_MS, Some(open), RES, None)[TIME_ELAPSED],
                bucket,
                "{bars} bars"
            );
        }

        // The overnight gap: 20:00 ET Tuesday to 04:00 ET Wednesday is 96 bars and crosses the
        // ET day boundary, so both ids move away from the adjacent case.
        let close = et("2024-01-16T20:00:00");
        let reopen = et("2024-01-17T04:00:00");
        let overnight = bar_time_ids(reopen, Some(close), RES, None);
        assert_eq!(overnight[TIME_DAY_EDGE], 2, "a new ET day");
        assert!(
            overnight[TIME_ELAPSED] > adjacent[TIME_ELAPSED],
            "an overnight return must not share a bucket with a five-minute one"
        );

        // A multi-week corpus hole saturates rather than escaping the table.
        let hole = bar_time_ids(open + 400 * 86_400_000, Some(open), RES, None);
        assert_eq!(hole[TIME_ELAPSED], TIME_ELAPSED_CAP);
        assert_eq!(hole[TIME_DAY_EDGE], 2);
        assert_eq!(bar_time_ids(open, None, RES, None)[TIME_ELAPSED], 0);
        assert_eq!(bar_time_ids(open, None, RES, None)[TIME_DAY_EDGE], 0);
    }

    /// The buckets must be reused from the persisted artifact rather than refitted, because a
    /// refit against a grown corpus silently re-means every market conditioning row.
    #[test]
    fn market_buckets_are_pinned_to_their_artifact() {
        let (fx, corpus, _) = market_fixture("pinned");
        let path = market_supports_path(&fx.dir, RES);
        assert!(path.is_file(), "the fit must persist its buckets");
        let first = corpus
            .market_channel()
            .expect("proxy")
            .support_sha256()
            .to_owned();
        let fingerprint = corpus.identity_fingerprint();

        // A second load of the same directory reuses the artifact and reproduces the geometry.
        let again = BarCorpus::load(&fx.dir, RES, 100).unwrap();
        assert_eq!(
            again.market_channel().expect("proxy").support_sha256(),
            first
        );
        assert_eq!(again.identity_fingerprint(), fingerprint);

        // Derived BEFORE the symbol restriction, so a universe ablation that drops the proxy
        // still conditions on the same market state. Without that, two arms of an ablation would
        // see different exogenous inputs and their `nll_bar` would not be comparable, which is
        // the exact failure the pre-restriction rule for the split bounds already prevents.
        let restricted =
            BarCorpus::load_restricted(&fx.dir, RES, 100, None, &HashSet::from(["AAA".to_owned()]))
                .unwrap();
        assert_eq!(restricted.symbols(), &["AAA"]);
        assert_eq!(
            restricted
                .market_channel()
                .expect("proxy survives restriction")
                .support_sha256(),
            first
        );

        // A directory with no proxy at all takes the reserved row everywhere, and says so in its
        // identity rather than silently resembling a corpus that has one.
        let empty = std::env::temp_dir().join(format!(
            "trading_bot_0_market_noproxy_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&empty).unwrap();
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        write_bar_file(
            &bar_path(&empty, "AAA"),
            "AAA",
            RES,
            &synth_bars(1, 3_000, base),
        )
        .unwrap();
        let proxyless = BarCorpus::load(&empty, RES, 100).unwrap();
        assert!(proxyless.market_channel().is_none());
        assert_ne!(
            proxyless.identity_fingerprint(),
            restricted.identity_fingerprint()
        );
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// Serialises the tests that touch [`FREEZE_MARKET_SUPPORTS`]. The flag is process-wide and
    /// the harness runs tests on threads, so an unserialised setter would flip the guard under
    /// a concurrent corpus load.
    fn freeze_serial() -> std::sync::MutexGuard<'static, ()> {
        static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The hole this closes: `load_or_fit_market_supports` once reused the artifact on mere
    /// file existence, checking only the bin count, so buckets fitted under one proxy history
    /// and one split instant were applied to another without a word.
    #[test]
    fn market_buckets_refuse_a_proxy_history_they_were_not_fitted_on() {
        let (fx, corpus, _) = market_fixture("provenance");
        let path = market_supports_path(&fx.dir, RES);
        let bounds = corpus.split_bounds();
        let first = corpus
            .market_channel()
            .expect("proxy")
            .support_sha256()
            .to_owned();
        drop(corpus);

        // The fit stamps provenance rather than leaving an unverifiable artifact on disk.
        let stamped = BarSupports::load(&path).unwrap();
        let recorded = stamped
            .provenance()
            .expect("the fit must stamp provenance")
            .clone();
        assert_eq!(recorded.split_bounds, bounds);
        assert!(recorded.sample_count > 0);

        let proxy_path = bar_path(&fx.dir, MARKET_PROXY_SYMBOL);
        let original: Vec<PackedBar> = BarFile::open(&proxy_path).unwrap().bars().to_vec();
        let last_ts = original.last().unwrap().ts();

        // Bars APPENDED after the boundary must not invalidate anything: they cannot reach a
        // bucket edge, and a guard that expired on every ingest would be switched off within a
        // week, which is how the silent hole got there in the first place.
        let mut appended = original.clone();
        appended.extend(synth_bars(21, 400, last_ts + RES_MS));
        write_bar_file(&proxy_path, MARKET_PROXY_SYMBOL, RES, &appended).unwrap();
        let grown = BarCorpus::load_with_bounds(&fx.dir, RES, 100, bounds)
            .expect("appending past the train boundary must not invalidate the buckets");
        assert_eq!(
            grown.market_channel().expect("proxy").support_sha256(),
            first
        );
        drop(grown);

        // DEEPENING the proxy's history does invalidate them: the fit region itself changed,
        // which is exactly the event a bin-count check cannot see.
        let mut deepened = synth_bars(23, 500, original[0].ts() - 500 * RES_MS);
        deepened.extend(appended.iter().copied());
        write_bar_file(&proxy_path, MARKET_PROXY_SYMBOL, RES, &deepened).unwrap();
        let refused = BarCorpus::load_with_bounds(&fx.dir, RES, 100, bounds)
            .expect_err("a proxy history the buckets were not fitted on must be refused");
        let message = format!("{refused:#}");
        assert!(
            message.contains("market channel buckets") && message.contains("do not reproduce it"),
            "the refusal must name the artifact and the mismatch: {message}"
        );
        assert!(
            message.contains("--freeze-market-supports"),
            "the refusal must name its opt-out: {message}"
        );

        // Same proxy history, a DERIVED train boundary. This must LOAD: the planner, the panels
        // and the universe rebuild all open this corpus on their own percentile and have to
        // consume the geometry the checkpoint was trained on. Refusing them would leave deleting
        // the artifact as the only way forward, which re-means every conditioning row.
        write_bar_file(&proxy_path, MARKET_PROXY_SYMBOL, RES, &original).unwrap();
        let elsewhere = (bounds.0 - 600 * RES_MS, bounds.1);
        let derived = BarCorpus::load_with_bounds(&fx.dir, RES, 100, elsewhere)
            .expect("a derived boundary must reuse rather than refuse the pinned buckets");
        assert_eq!(
            derived.market_channel().expect("proxy").support_sha256(),
            first
        );
        drop(derived);

        // And the untouched original still loads, so the guard is discriminating rather than
        // simply broken.
        let restored = BarCorpus::load_with_bounds(&fx.dir, RES, 100, bounds)
            .expect("the proxy the buckets were fitted on still loads");
        assert_eq!(
            restored.market_channel().expect("proxy").support_sha256(),
            first
        );
        drop(restored);

        // The freeze is the documented way past a refusal, and it is reachable: the global flag
        // `main` sets is the only thing standing between the operator and the artifact.
        let _serial = freeze_serial();
        write_bar_file(&proxy_path, MARKET_PROXY_SYMBOL, RES, &deepened).unwrap();
        assert!(BarCorpus::load_with_bounds(&fx.dir, RES, 100, bounds).is_err());
        set_freeze_market_supports(true);
        let frozen = BarCorpus::load_with_bounds(&fx.dir, RES, 100, bounds)
            .expect("the freeze must let a mismatched artifact through");
        assert_eq!(
            frozen.market_channel().expect("proxy").support_sha256(),
            first,
            "a frozen reuse must keep the OLD geometry, not refit"
        );
        set_freeze_market_supports(false);
        let _ = std::fs::remove_dir_all(&fx.dir);
    }

    /// The decision rule itself: reuse, frozen reuse, refusal — and the fact that the caller's
    /// own train boundary never enters the verdict.
    #[test]
    fn market_supports_provenance_decides_reuse_freeze_and_refusal() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_market_rule_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let bars = synth_bars(31, 2_000, base);
        let proxy = bar_path(&dir, MARKET_PROXY_SYMBOL);
        write_bar_file(&proxy, MARKET_PROXY_SYMBOL, RES, &bars).unwrap();
        let file = BarFile::open(&proxy).unwrap();
        let path = market_supports_path(&dir, RES);

        let samples = 1_200usize;
        let bounds = (bars[samples].ts(), bars[samples].ts() + 10 * RES_MS);
        let recorded = BarSupportsProvenance {
            corpus_fingerprint: market_proxy_fingerprint(&file, RES, samples).unwrap(),
            split_bounds: bounds,
            sample_count: samples,
            fit_seed: None,
            scaling_contract: None,
            support_semantics: None,
            fitted_utc: "2026-08-20T00:00:00Z".to_owned(),
        };

        assert!(
            !require_market_supports_provenance(Some(&recorded), &path, &file, RES, bounds, false)
                .expect("a faithful artifact is reused"),
            "a match is not a freeze"
        );

        // The verdict must not depend on the caller's boundary: the planner, the panels and the
        // universe rebuild all derive a different one against this same artifact.
        for derived in [bounds.0 - 300 * RES_MS, bounds.0 + 300 * RES_MS] {
            require_market_supports_provenance(
                Some(&recorded),
                &path,
                &file,
                RES,
                (derived, derived + 10 * RES_MS),
                false,
            )
            .expect("a derived boundary must not invalidate an otherwise faithful artifact");
        }

        // A proxy whose history no longer reproduces the recorded fit is refused.
        let mut revised = bars.clone();
        let mut touched = revised[400];
        touched.close *= 1.05;
        revised[400] = touched;
        write_bar_file(&proxy, MARKET_PROXY_SYMBOL, RES, &revised).unwrap();
        let moved = BarFile::open(&proxy).unwrap();
        let refused =
            require_market_supports_provenance(Some(&recorded), &path, &moved, RES, bounds, false)
                .expect_err("a revised proxy history must be refused")
                .to_string();
        assert!(refused.contains("do not reproduce it"), "{refused}");
        assert!(refused.contains("--freeze-market-supports"), "{refused}");
        assert!(
            require_market_supports_provenance(Some(&recorded), &path, &moved, RES, bounds, true)
                .expect("the freeze accepts a mismatch"),
            "a frozen mismatch must report itself as frozen"
        );

        // A proxy TRUNCATED below the recorded fit cannot even be checked, so it is refused
        // rather than accepted for want of a comparison.
        write_bar_file(&proxy, MARKET_PROXY_SYMBOL, RES, &bars[..500]).unwrap();
        let short = BarFile::open(&proxy).unwrap();
        require_market_supports_provenance(Some(&recorded), &path, &short, RES, bounds, false)
            .expect_err("a proxy shorter than the recorded fit must be refused");

        // Absent provenance is a refusal too, which is the exact state every artifact written
        // before this guard is in — including the one this campaign was running on.
        let bare = require_market_supports_provenance(None, &path, &file, RES, bounds, false)
            .expect_err("an unverifiable artifact must be refused")
            .to_string();
        assert!(bare.contains("no provenance at all"), "{bare}");
        assert!(
            require_market_supports_provenance(None, &path, &file, RES, bounds, true)
                .expect("the freeze accepts an unstamped artifact")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The fingerprint must key on the fit region and nothing else: not the symbol universe,
    /// not bars past the region, and not fields the fit never reads.
    #[test]
    fn market_proxy_fingerprint_tracks_the_fit_region_only() {
        let dir =
            std::env::temp_dir().join(format!("trading_bot_0_market_fp_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let bars = synth_bars(5, 2_000, base);
        let path = bar_path(&dir, MARKET_PROXY_SYMBOL);
        write_bar_file(&path, MARKET_PROXY_SYMBOL, RES, &bars).unwrap();

        let samples = 1_500usize;
        let of = |p: &Path, n: usize| market_proxy_fingerprint(&BarFile::open(p).unwrap(), RES, n);
        let baseline = of(&path, samples).expect("the fit region is inside the file");

        // A fit region longer than the file cannot be checked at all, and says so.
        assert!(of(&path, bars.len()).is_none());

        // Appending past the fit region leaves it alone.
        let mut appended = bars.clone();
        appended.extend(synth_bars(6, 300, bars.last().unwrap().ts() + RES_MS));
        write_bar_file(&path, MARKET_PROXY_SYMBOL, RES, &appended).unwrap();
        assert_eq!(
            of(&path, samples),
            Some(baseline.clone()),
            "bars past the fit region cannot reach a bucket edge"
        );

        // A different cut of the same history is a different fit sample.
        assert_ne!(of(&path, samples + 1), Some(baseline.clone()));

        // Fields the fit never reads must not invalidate it: `encode_dof` consumes OHLC and
        // `VolumeEma` consumes volume, so a vendor revision of vwap or the trade count provably
        // cannot move an edge and must not hard-error a corpus open.
        let mut cosmetic = appended.clone();
        let mut untouched = cosmetic[700];
        untouched.vwap *= 1.5;
        untouched.trades += 41;
        cosmetic[700] = untouched;
        write_bar_file(&path, MARKET_PROXY_SYMBOL, RES, &cosmetic).unwrap();
        assert_eq!(
            of(&path, samples),
            Some(baseline.clone()),
            "vwap and trades are read by nothing in the fit"
        );

        // An in-place revision INSIDE the fit region that preserves the bar count and both
        // endpoints is caught, which a count-and-span fingerprint would have missed.
        let mut revised = appended.clone();
        let mut touched = revised[700];
        touched.high *= 1.05;
        revised[700] = touched;
        write_bar_file(&path, MARKET_PROXY_SYMBOL, RES, &revised).unwrap();
        assert_ne!(
            of(&path, samples),
            Some(baseline),
            "content, not metadata: a backfill that rewrites history must move the fingerprint"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regenerate the persisted market buckets from the live proxy under the campaign pin.
    ///
    /// Fitted in a scratch directory and only then copied, so a run holding the canonical
    /// artifact open can never race the writer. Ignored because it touches the live corpus and
    /// writes an artifact; it is the recipe for that artifact, kept compiling beside the code
    /// that reads it rather than living in somebody's shell history.
    #[test]
    #[ignore = "regenerates a persisted artifact from the live corpus"]
    fn refit_market_buckets_from_the_live_proxy() {
        let bars = Path::new(shared::paths::DATA_PATH).join("bars");
        let bounds = crate::data::ingest::PINNED_SPLIT_BOUNDS;
        let scratch = std::env::temp_dir().join(format!(
            "trading_bot_0_market_refit_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&scratch).unwrap();

        let proxy = bars.join(format!("{MARKET_PROXY_SYMBOL}.{RES}.{FILE_EXTENSION}"));
        let file = BarFile::open(&proxy).expect("the live proxy opens");
        let train_end = file.index_at_or_after(bounds.0);
        let supports = load_or_fit_market_supports(&scratch, RES, &file, bounds)
            .expect("the proxy's train region fits");

        let fitted = market_supports_path(&scratch, RES);
        let recorded = supports.provenance().expect("the fit stamps provenance");
        assert_eq!(recorded.split_bounds, bounds);
        assert_eq!(recorded.sample_count, train_end - 1);
        assert_eq!(
            Some(&recorded.corpus_fingerprint),
            market_proxy_fingerprint(&file, RES, train_end - 1).as_ref()
        );

        let destination = bars.join(format!("bar_market_supports.{RES}.v7.json"));
        assert!(
            destination != market_supports_path(&bars, RES),
            "the refit must not overwrite the artifact a running job holds"
        );
        std::fs::copy(&fitted, &destination).expect("the refit lands beside the corpus");

        // The artifact is only a deliverable if the guard ACCEPTS it: reload it through the
        // same path a run takes, and confirm it is reused rather than refused or refitted.
        let geometry = market_support_sha256(&supports);
        let reused = load_or_fit_market_supports(&scratch, RES, &file, bounds)
            .expect("the guard must accept the artifact this fit just stamped");
        assert_eq!(market_support_sha256(&reused), geometry);
        let landed = BarSupports::load(&destination).expect("the copy parses");
        assert_eq!(
            Some(
                landed
                    .provenance()
                    .expect("provenance survives the copy")
                    .corpus_fingerprint
                    .clone()
            ),
            market_proxy_fingerprint(&file, RES, train_end - 1)
        );
        let bytes = std::fs::read(&destination).unwrap();
        let mut digest = DigestContext::new(&SHA256);
        digest.update(&bytes);
        println!(
            "[refit] {} bars={} train_end={} samples={} fitted={}\n[refit] proxy fingerprint {}\n\
             [refit] geometry sha256 {}\n[refit] file sha256 {}",
            destination.display(),
            file.len(),
            train_end,
            recorded.sample_count,
            recorded.fitted_utc,
            recorded.corpus_fingerprint,
            market_support_sha256(&supports),
            hex_digest(digest),
        );
        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn standardized_batches_carry_exact_raw_targets_without_duplicating_raw_batches() {
        let (_fixture, corpus) = fixture("raw-target-batch");
        let rows = [(0usize, 3_000usize), (1usize, 2_000usize)];
        let len = 32usize;
        let raw = build_batch(
            &corpus.inner.files,
            &rows,
            len,
            RES,
            corpus.inner.market.as_ref(),
            DofScaling::Raw,
            Device::Cpu,
        );
        let standardized = build_batch(
            &corpus.inner.files,
            &rows,
            len,
            RES,
            corpus.inner.market.as_ref(),
            DofScaling::VolStandardized,
            Device::Cpu,
        );
        assert!(
            raw.raw_dof.is_none(),
            "a raw batch must use dof directly, not allocate a duplicate tensor"
        );
        let realized = standardized
            .raw_dof
            .as_ref()
            .expect("a standardized batch preserves true raw targets");
        for row in 0..rows.len() {
            for step in 0..len {
                for dof in 0..BAR_DOF {
                    let index = [row as i64, step as i64, dof as i64];
                    assert_eq!(
                        realized.double_value(&index) as f32,
                        raw.dof.double_value(&index) as f32,
                        "raw target mismatch at row {row}, step {step}, DOF {dof}"
                    );
                }
            }
        }
    }

    #[test]
    fn target_geometry_reduces_each_scaling_exception_counter_exactly() {
        let mut first = SeriesGeometry::new("AAA");
        first.bars = 10;
        first.sigma_relative_floored = 2;
        first.sigma_numerical_fallback = 1;
        first.r_z_clamped = 3;
        first.s_z_clamped = 4;
        let mut second = SeriesGeometry::new("BBB");
        second.bars = 30;
        second.sigma_relative_floored = 5;
        second.sigma_numerical_fallback = 6;
        second.r_z_clamped = 7;
        second.s_z_clamped = 8;
        let geometry = TargetGeometry::reduce(
            RES,
            DofScaling::VolStandardized,
            Split::Train,
            vec![first, second],
        );
        assert_eq!(geometry.sigma_relative_floored, 7);
        assert_eq!(geometry.sigma_numerical_fallback, 7);
        assert_eq!(geometry.r_z_clamped, 10);
        assert_eq!(geometry.s_z_clamped, 12);
        assert_eq!(geometry.sigma_relative_floored_share(), 7.0 / 40.0);
        assert_eq!(geometry.sigma_numerical_fallback_share(), 7.0 / 40.0);
        assert_eq!(geometry.r_z_clamped_share(), 10.0 / 40.0);
        assert_eq!(geometry.s_z_clamped_share(), 12.0 / 40.0);
    }

    /// The bitset is a CACHE for [`computed_us_equity_trading_date`], generated by it, so the
    /// only thing that can be wrong is the index arithmetic. A sweep of a year either side of
    /// the table's span checks the interior bit for bit AND that a date the table cannot
    /// answer for is not aliased into it: an `index <= days` bound reads an unset bit past the
    /// last covered date and closes the exchange on an ordinary Monday.
    #[test]
    fn the_trading_date_bitset_agrees_with_the_computation_it_caches() {
        let end = NaiveDate::from_ymd_opt(2101, 1, 1).expect("sweep end");
        let mut date = NaiveDate::from_ymd_opt(1989, 1, 1).expect("sweep start");
        let mut open = 0usize;
        while date < end {
            let computed = computed_us_equity_trading_date(date);
            assert_eq!(
                is_us_equity_trading_date(date),
                computed,
                "the bitset disagrees with its own generator at {date}"
            );
            open += usize::from(computed);
            date = date.succ_opt().expect("the sweep stays representable");
        }
        assert!(
            open > 27_000,
            "only {open} trading dates swept, so the sweep is broken rather than the table"
        );

        // The four dates the index arithmetic decides between, named rather than left to a
        // loop that could skip them.
        let table = &*TRADING_DATES;
        let first = NaiveDate::from_ymd_opt(1990, 1, 1).expect("span start");
        let last = first + Duration::days(table.days as i64 - 1);
        assert_eq!(last, NaiveDate::from_ymd_opt(2099, 12, 31).expect("span end"));
        for date in [
            first - Duration::days(1),
            first,
            last,
            last + Duration::days(1),
        ] {
            assert_eq!(
                is_us_equity_trading_date(date),
                computed_us_equity_trading_date(date),
                "the span edge {date} must be answered exactly, not clamped"
            );
        }
        // The first ordinary weekday past the span: nothing about it is closed, so only the
        // fallthrough can produce the right answer.
        let monday = NaiveDate::from_ymd_opt(2100, 1, 4).expect("past the span");
        assert_eq!(monday.weekday(), Weekday::Mon);
        assert!(
            is_us_equity_trading_date(monday),
            "a date past the table aliased onto an unset bit"
        );

        // And the arithmetic itself, because a one-day overrun of the span aliases only
        // 2100-01-01 — permanently New Year's Day, permanently closed, permanently the same
        // answer as the unset bit it would read — so no behavioural assertion can see it.
        let index = |date: NaiveDate| usize::try_from(date.num_days_from_ce() - table.base_ce);
        assert!(index(first - Duration::days(1)).is_err());
        assert_eq!(index(first), Ok(0));
        assert_eq!(index(last), Ok(table.days - 1));
        assert_eq!(index(last + Duration::days(1)), Ok(table.days));
    }

    /// [`et_naive_local`] claims a bisected offset table reproduces chrono-tz exactly, and the
    /// only instants a table can be wrong about are its own transitions. Every one is probed on
    /// both sides at second and millisecond distance, in both directions.
    ///
    /// The comparison is `single()` because that is the call the production fallback makes, and
    /// it is total here: a UTC instant has exactly one New York wall clock. Ambiguity belongs
    /// to the local-to-UTC direction, which neither this function nor its callers take.
    #[test]
    fn the_et_offset_table_reproduces_chrono_tz_at_every_transition() {
        let (starts, offsets) = &*ET_TRANSITIONS;
        assert_eq!(starts.len(), offsets.len());
        assert!(
            starts.len() > 200,
            "only {} transitions tabulated over 110 years",
            starts.len()
        );
        let mut spring_forward = 0usize;
        let mut fall_back = 0usize;
        for (index, &start) in starts.iter().enumerate() {
            for delta_ms in [-1_000i64, -1, 0, 1, 1_000] {
                let ts_ms = start * 1_000 + delta_ms;
                let want = New_York
                    .timestamp_millis_opt(ts_ms)
                    .single()
                    .expect("a UTC instant has one New York wall clock")
                    .naive_local();
                assert_eq!(
                    et_naive_local(ts_ms),
                    want,
                    "transition {index} at {} is wrong {delta_ms} ms in",
                    iso_ms(start * 1_000)
                );
            }
            if index == 0 {
                continue;
            }
            // Non-vacuity: the wall clock really jumps an hour here, so the equality above is
            // comparing two answers that a table off by one entry would get different.
            let jump = (et_naive_local(start * 1_000)
                - et_naive_local((start - 1) * 1_000))
            .num_seconds();
            assert_eq!(
                (jump - 1).abs(),
                3_600,
                "transition {index} at {} moved the wall clock by {jump} s",
                iso_ms(start * 1_000)
            );
            if jump > 0 {
                spring_forward += 1;
            } else {
                fall_back += 1;
            }
        }
        assert!(
            spring_forward > 100 && fall_back > 100,
            "{spring_forward} gaps and {fall_back} ambiguous windows probed; one direction is \
             missing from the table"
        );
    }

    /// `Tensor::equal` rather than a tolerance: reuse is claimed to change NOTHING, and a leak
    /// from a previous, longer draw is not a small numeric difference. It is false for NaN even
    /// against the same tensor, so callers assert finiteness first.
    fn assert_batch_eq(got: &BarBatch, want: &BarBatch, what: &str) {
        let optional = |got: &Option<Tensor>, want: &Option<Tensor>| match (got, want) {
            (Some(got), Some(want)) => got.equal(want),
            (None, None) => true,
            _ => false,
        };
        assert!(got.dof.equal(&want.dof), "{what}: dof");
        assert!(got.time_ids.equal(&want.time_ids), "{what}: time_ids");
        assert!(got.sigma.equal(&want.sigma), "{what}: sigma");
        assert!(optional(&got.raw_dof, &want.raw_dof), "{what}: raw_dof");
        assert!(
            optional(&got.direct_time_ids, &want.direct_time_ids),
            "{what}: direct_time_ids"
        );
        assert!(
            optional(&got.direct_valid, &want.direct_valid),
            "{what}: direct_valid"
        );
        assert_eq!(
            got.market_missing, want.market_missing,
            "{what}: market_missing"
        );
    }

    /// A corpus whose windows straddle the proxy's coverage hole, standardized so `raw_dof` is
    /// staged, and carrying three future bars so the direct-forecast buffers are staged too.
    /// Every [`BatchScratch`] field is therefore live.
    fn scratch_fixture(label: &str) -> (Fixture, BarCorpus) {
        let (fixture, corpus, _) = market_fixture(label);
        (fixture, corpus.with_dof_scaling(DofScaling::VolStandardized))
    }

    /// A long draw and a SHORT one that is not a prefix of it, both straddling the proxy hole
    /// so `market_missing` is nonzero and the staged market ids carry the reserved row.
    fn scratch_windows(corpus: &BarCorpus) -> (Vec<WindowRef>, Vec<WindowRef>) {
        let symbol = series_of(corpus, "AAA") as u32;
        let refs = |bars: &[u32]| -> Vec<WindowRef> {
            bars.iter()
                .map(|&bar_index| WindowRef { symbol, bar_index })
                .collect()
        };
        (
            refs(&[400, 700, 900, 1_000, 1_100, 1_150, 1_180, 1_210]),
            refs(&[1_205, 600, 1_190]),
        )
    }

    /// The staging buffers outlive the batch that sized them, and a stage's last draw is
    /// deliberately SHORT. Any region a draw does not fully overwrite leaks the previous
    /// draw's values into the tensors the trunk trains on, which no shape check and no loss can
    /// see.
    ///
    /// One scratch across two CONTEXTS, because that is what `BatchPrefetcher` does: it keeps a
    /// single [`BatchScratch`] and stages every ramp stage through it. The row stride therefore
    /// changes between draws, so an unwritten slot inherits a different row's value rather than
    /// the same zero every time — the difference between a leak that shows and a leak that hides
    /// behind a fixture where every draw has the same geometry.
    #[test]
    fn a_reused_scratch_stages_the_same_batch_as_a_fresh_one() {
        let (_fx, corpus) = scratch_fixture("scratch_reuse");
        let narrow = BarSampler::new_with_future(&corpus, Split::Train, 64, 3, 7);
        let wide = BarSampler::new_with_future(&corpus, Split::Train, 96, 3, 7);
        let (long, short) = scratch_windows(&corpus);

        let fresh: Vec<BarBatch> = [
            (&wide, &long),
            (&narrow, &short),
            (&narrow, &long),
            (&wide, &short),
        ]
        .iter()
        .map(|(sampler, refs)| sampler.batch_of(refs, Device::Cpu))
        .collect();
        assert!(
            bool::try_from(fresh[0].dof.isfinite().all()).expect("finite"),
            "a NaN would make every `Tensor::equal` below fail for the wrong reason"
        );
        assert!(
            fresh.iter().all(|batch| batch.market_missing > 0),
            "every draw must contain reserved market rows"
        );
        assert!(fresh[0].raw_dof.is_some() && fresh[0].direct_valid.is_some());
        assert!(
            !fresh[1]
                .dof
                .equal(&fresh[0].dof.narrow(0, 0, short.len() as i64).narrow(1, 0, 65)),
            "the draws stage the same values, so a leak between them would be invisible here"
        );

        let mut scratch = narrow.scratch();
        for (step, (sampler, refs)) in [
            (&wide, &long),
            (&narrow, &short),
            (&narrow, &long),
            (&wide, &short),
        ]
        .iter()
        .enumerate()
        {
            let reused = sampler.batch_of_into(refs, Device::Cpu, &mut scratch);
            assert_batch_eq(&reused, &fresh[step], &format!("draw {step} through one scratch"));
        }
    }

    /// The pretrainer builds the CPU half of the next draw into the SAME buffers while the
    /// current [`HostSample`] is still in flight, and only then transfers it. That is sound
    /// only because the staged tensors OWN their data — a sample whose tensors borrowed the
    /// scratch would be rewritten under the training thread — and it needs nothing from the
    /// trainer: `host_batch_of_into` followed by `to_device` IS the split.
    #[test]
    fn a_host_sample_survives_the_next_draw_into_its_own_scratch() {
        let (_fx, corpus) = scratch_fixture("prefetch_split");
        let narrow = BarSampler::new_with_future(&corpus, Split::Train, 64, 3, 11);
        let wide = BarSampler::new_with_future(&corpus, Split::Train, 96, 3, 11);
        let (long, short) = scratch_windows(&corpus);
        let synchronous_wide = wide.batch_of(&long, Device::Cpu);
        let synchronous_narrow = narrow.batch_of(&short, Device::Cpu);

        let mut scratch = wide.scratch();
        let held = wide.host_batch_of_into(&long, &mut scratch);
        // Overwrites, shrinks AND restrides the buffers `held` was staged through, exactly as
        // the prefetch worker does one ramp stage ahead of the transfer.
        let following = narrow.host_batch_of_into(&short, &mut scratch);
        assert_eq!(held.market_missing, synchronous_wide.market_missing);
        assert_batch_eq(
            &held.to_device(Device::Cpu),
            &synchronous_wide,
            "a host sample transferred after the draw that followed it",
        );
        assert_batch_eq(
            &following.to_device(Device::Cpu),
            &synchronous_narrow,
            "the follow-on draw at another context",
        );
    }

    /// The forward cursor is an optimization of one independent bisection per bar, so it must
    /// agree with [`MarketChannel::ids_at`] everywhere. Threaded densely across the proxy's
    /// coverage hole and then at a stride wider than the bounded scan, so both the scan and the
    /// suffix bisection it falls back to are exercised, along with duplicate and off-grid
    /// timestamps and a tail past the proxy's last bar.
    #[test]
    fn the_monotonic_market_cursor_matches_an_independent_lookup_per_bar() {
        let (_fx, corpus, _) = market_fixture("cursor");
        let channel = corpus.market_channel().expect("the fixture has a proxy");
        let bars = corpus.bars(series_of(&corpus, "AAA"));
        // Non-decreasing rather than strictly increasing: each bar is asked twice, then once
        // one millisecond off the grid where no proxy bar can match.
        let dense = bars[MARKET_HOLE_START - 40..MARKET_HOLE_END + 40]
            .iter()
            .flat_map(|bar| [bar.ts(), bar.ts(), bar.ts() + 1]);
        // Wider than `SCAN`, so the bounded walk is exhausted and the suffix bisection runs.
        let sparse = bars.iter().step_by(25).map(|bar| bar.ts());
        let tail = [bars[bars.len() - 1].ts() + RES_MS, i64::MAX / 4];

        for queries in [
            dense.chain(tail).collect::<Vec<i64>>(),
            sparse.chain(tail).collect(),
        ] {
            let mut cursor = 0usize;
            let mut missing = 0usize;
            let mut observed = 0usize;
            for &ts_ms in &queries {
                let (ids, next) = channel.ids_at_from(ts_ms, cursor);
                assert_eq!(
                    ids,
                    channel.ids_at(ts_ms),
                    "the threaded cursor disagrees with an independent join at {}",
                    iso_ms(ts_ms)
                );
                assert!(next >= cursor, "the cursor moved backwards at {ts_ms}");
                cursor = next;
                if ids[0] == MARKET_MISSING {
                    missing += 1;
                } else {
                    observed += 1;
                }
            }
            assert!(
                missing > 0 && observed > 0,
                "{observed} observed and {missing} uncovered joins; the series does not \
                 distinguish the two outcomes"
            );
        }
    }

    /// The cursor is a promise, not a hint. A cursor resolved for a LATER timestamp cannot be
    /// walked back, so the join silently reports the reserved row for a bar the proxy covers —
    /// a wrong conditioning id rather than a crash. Hence the `debug_assert` in
    /// [`MarketChannel::ids_at_from`]; this pins both halves, the assert where it is compiled
    /// in and the degradation it exists to prevent where it is not.
    #[test]
    fn a_market_cursor_handed_a_later_timestamp_cannot_resolve_an_earlier_bar() {
        let (_fx, corpus, _) = market_fixture("cursor_violation");
        let channel = corpus.market_channel().expect("the fixture has a proxy");
        let early = channel.ts_ms[100];
        let (correct, index) = channel.ids_at_from(early, 0);
        assert_eq!(index, 100);
        assert_ne!(
            correct[0], MARKET_MISSING,
            "proxy bar 100 is observed, so the violation below has something to lose"
        );

        #[cfg(debug_assertions)]
        assert!(
            std::panic::catch_unwind(|| channel.ids_at_from(early, 500)).is_err(),
            "a backwards cursor must trip the precondition assert"
        );
        #[cfg(not(debug_assertions))]
        {
            let (ids, next) = channel.ids_at_from(early, 500);
            assert_eq!(
                ids,
                [MARKET_MISSING; MARKET_FEATURES],
                "a backwards cursor cannot recover the bar it skipped past"
            );
            assert_eq!(next, 500, "the scan has no way to rewind");
        }
    }
}
