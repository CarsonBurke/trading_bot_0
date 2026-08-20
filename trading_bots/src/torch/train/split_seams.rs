//! Are the extreme `r` bars in the corpus MARKET MOVES or UNADJUSTED CORPORATE-ACTION SEAMS?
//!
//! The question is not academic and it is not about the far tail for its own sake. Three live
//! numbers rest on the outermost `r` bars and on nothing else:
//!
//! - The two catch-all bins of the `r` support are placed at the
//!   [`BAR_SUPPORT_CLIP_QUANTILE`] quantiles, so bins 0 and 127 are POPULATED by exactly this
//!   region, hold 1.4474% of the mass between them, and carry the overwhelming share of the
//!   variance of any first-moment decode.
//! - The measured tail index on `|r|` is a spread of six pairwise log-log slopes, and a handful
//!   of huge artificial jumps is the classic way to inflate one.
//! - The leverage/ruin licence is `1 + F(exp(r) - 1) > 0` evaluated at the support's own reachable
//!   range, so it is set by `lo[DOF_R][0]` and `hi[DOF_R][127]` — i.e. by whatever those quantiles
//!   landed on. If they landed on a stock split, a corporate action is setting the leverage cap.
//!
//! WHAT MAKES THIS DECIDABLE RATHER THAN A MATTER OF OPINION. A split seam has four properties a
//! genuine move does not have, and they are independent of each other:
//!
//! 1. `exp(r)` sits on a SIMPLE RATIONAL — 5, 1/5, 3/2, 10 — because a split ratio IS a simple
//!    rational and both sides of the seam are quantized onto the same price ladder. A real -80%
//!    move does not land on 1/5 to six decimals.
//! 2. The discontinuity is at a SESSION BOUNDARY. Splits are effective at the open.
//! 3. The bar is otherwise UNREMARKABLE. A split does not trade: the bar's own log range `s` does
//!    not open up to contain the move, and the volume DOF `w` does not spike. A genuine 400% move
//!    does both.
//! 4. It is ISOLATED. A bad print reverts on the next bar and therefore prints the OPPOSITE
//!    extreme beside it; a split is a one-bar discontinuity in an otherwise continuous series.
//!
//! Each is counted separately and the conjunction is counted too, so the verdict is a census and
//! not a judgement call. Criterion 4 is what separates a split from the other extreme-`r`
//! population — a single bad tick at 1/5 of the price, which prints `-ln 5` and then `+ln 5` — and
//! that distinction matters, because a reverting pair is the ONLY way one symbol can put both
//! `min_r = -ln 5` and `max_r = +ln 5` into the same draw.
//!
//! WHAT THIS DOES NOT DO. It writes no corpus file, it touches no ingest path, it refits no
//! support and it moves no live constant. The cleaned support edges and the cleaned ruin licence
//! below are COUNTERFACTUALS, reported so the cost of the contamination is a number; the artifact
//! on disk is read and never written.
//!
//! RESOURCE SHAPE, load-bearing, because this pass is the only thing in the tree that reads all
//! 451,507,140 bars. Everything is a BOUNDED streaming accumulator:
//!
//! - The corpus census is a rayon fold over SERIES, and each task's accumulator holds fixed-size
//!   histograms, [`EVENT_BUFFER`] retained extreme events and at most [`SEAM_BUFFER`] eight-byte
//!   seam keys. Nothing scales with the corpus.
//! - [`BarCorpus::for_each_series_dof`] hands one bar at a time and buffers nothing, so a series
//!   costs its mmap window and no heap at all.
//! - The tail control needs the SAME 4,000,000-row draw the supports were fitted from, which is
//!   the one sizeable allocation and is the same allocation
//!   [`crate::torch::dataset::BarCorpus::fit_supports`] already makes.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::torch::bar_dist::{
    BarDof, BarSupports, DofBinner, BAR_SUPPORT_CLIP_QUANTILE, DOF_R, NUM_BAR_BINS,
};
use crate::torch::dataset::{et_local_day, iso_ms, BarCorpus, WindowRef};
use crate::torch::train::bar_family::{
    empirical_tail_slopes, upper_order_statistics, EmpiricalSlope,
};
use crate::torch::train::pretrain::{load_corpus, CorpusFlags};
use crate::torch::train::pretrain_reports::write_bar_seams;

/// Above this `|r|`, a bar enters the census.
///
/// `ln 1.5`: a 50% up or 33.3% down close-to-close move inside ONE five-minute bar. Declared
/// rather than derived from a quantile, because a quantile threshold would move when the
/// population under test moves, which is the one thing an audit of that population must not do.
/// It is far enough out that the smallest split ratio anyone uses (3:2) clears it and no ordinary
/// market bar comes close: the support's own catch-all edges sit at 883 bps, and this is 4.6x
/// further out than that.
pub const EXTREME_LOG_THRESHOLD: f64 = 0.405_465_108_108_164_4;

/// Exceedance levels the census counts `|r|` at, as log ratios.
///
/// `ln 1.5`, `ln 2`, `ln 3`, `ln 4`, `ln 5`, `ln 10`. The `ln 4` entry is there because
/// `ingest::SPLICE_MAX_JUMP` is 4.0, so the count on either side of it says whether that constant
/// is looking at anything; the `ln 5` entry is there because the 4M draw's min and max both sit on
/// it to `f32` precision.
pub const CENSUS_LOG_LEVELS: [f64; 6] = [
    0.405_465_108_108_164_4,
    std::f64::consts::LN_2,
    1.098_612_288_668_109_7,
    1.386_294_361_119_890_6,
    1.609_437_912_434_100_4,
    2.302_585_092_994_046,
];

pub const CENSUS_LEVEL_NAMES: [&str; 6] = ["1.5x", "2x", "3x", "4x", "5x", "10x"];

/// The admitted "simple rational" set: `p:q` in lowest terms with both terms at most
/// [`RATIONAL_MAX_TERM`], plus the integer ladder `n:1` and `1:n` up to
/// [`RATIONAL_LADDER_MAX`].
///
/// Every split and reverse split a US listed security executes is `p:q` with both terms tiny —
/// 2:1, 3:1, 4:1, 5:1, 10:1, 3:2, 5:4, 7:5 — or an integer reverse split, which is where the large
/// terms live: 1:20, 1:50, 1:100, 1:200 are all real and all of the form `1:n`.
///
/// THE SET MUST STAY SPARSE OR THE CRITERION IS VACUOUS, and this is not a hypothetical: a set
/// admitting `p/q` with `q` up to 100 puts 9/17 within 11 bps of 0.53, so an ordinary -47% move
/// would have claimed to be "on a rational". Sparsity is measurable and is what these two bounds
/// are chosen for. With both terms at 10, the neighbours of 5 are 9/2 and 6, i.e. relative gaps of
/// 10% and 20%; the neighbours of 3/2 are 10/7 and 8/5, gaps of 5% and 7%; and the tightest place
/// on the whole reachable axis is the top of the integer ladder, where 199 and 200 are 50 bps
/// apart. [`RATIONAL_TOLERANCE`] is below every one of those.
pub const RATIONAL_MAX_TERM: u32 = 10;
pub const RATIONAL_LADDER_MAX: u32 = 200;

/// Relative deviation from the nearest simple rational that counts as landing exactly ON it.
///
/// 1 bp. This is the criterion in "a real -80% move does not land on 1/5 to six decimals", and it
/// is satisfied only when BOTH sides of the discontinuity sit on the same price ladder — the last
/// pre-action print and the first post-action print are exact multiples of each other, which is
/// what a cent-quantized security does across a split when the market did not move in the gap.
/// It cannot be hit by accident: against the sparsest gap in the admitted set (50 bps, at the top
/// of the integer ladder) a 1 bp window covers 2% of the axis, and against the gap around a 5:1
/// ratio it covers 0.1%.
pub const RATIONAL_TOLERANCE: f64 = 1.0e-4;

/// Relative deviation that counts as CONSISTENT WITH a split ratio times one bar of market move.
///
/// 2%. The second mechanism, and the one that does NOT land on the ratio exactly: when a stored
/// series splices an unadjusted level against an adjusted one — a vendor that serves
/// `adjusted=true` and is appended to across a corporate action produces exactly this — the seam
/// ratio is the split factor times whatever the market did in the gap, which at a session boundary
/// is tens to hundreds of bps.
///
/// Counted and reported SEPARATELY and never folded into the tight count, because the two are not
/// equally strong evidence: at 2% the window covers a fifth of the axis around a 5:1 ratio and
/// most of it around 3:2, so the loose count is an UPPER BOUND on the seam population and the
/// tight count is a lower one. The verdict states both.
pub const RATIONAL_NEAR_TOLERANCE: f64 = 2.0e-2;

/// Largest `s / |r|` an "unremarkable" bar has.
///
/// A GENUINE close-to-close move of `|r|` has to trade through it, so the bar's own log range `s`
/// is at least comparable to `|r|`. A split seam is a level shift between two bars: the whole move
/// is in the gap and the bar itself has an ordinary range. At 0.5 the criterion says the bar's
/// range covers less than half the move it supposedly made.
pub const QUIET_RANGE_FRACTION: f64 = 0.5;

/// Largest `|w|` an "unremarkable" bar has: volume within a factor of two of its own 20-bar
/// causal EMA. A real 400% move is not a median-volume bar; a split is.
pub const QUIET_VOLUME_LOG: f64 = std::f64::consts::LN_2;

/// Extreme events RETAINED for the report, worst `|r|` first, per fold accumulator.
///
/// The census counts every event; this bounds only the sample that gets listed and cross-tabulated
/// by ratio. 16,384 events at ~72 bytes is 1.2 MiB per accumulator.
pub const EVENT_BUFFER: usize = 1 << 14;

/// Seam KEYS retained per fold accumulator, as `(series, bar)` pairs.
///
/// These are needed whole rather than sampled, because the cleaned tail estimate joins them onto
/// the 4M draw by location. Eight bytes each, so the cap costs 4 MiB and is three orders above any
/// plausible split population over 5,297 symbols and five years. Overflow is COUNTED and reported
/// rather than silently truncating the join.
pub const SEAM_BUFFER: usize = 1 << 19;

/// Buckets in the `s / |r|` and `w` comparison histograms.
const CONTEXT_BUCKETS: usize = 24;
/// Upper edge of the `s / |r|` histogram; the last bucket saturates.
const RANGE_RATIO_MAX: f64 = 3.0;
/// Half-width of the `w` histogram; the outer buckets saturate.
const VOLUME_LOG_MAX: f64 = 6.0;

/// Buckets in the rational-deviation histogram, geometric from 1e-9 to 1e-1 relative.
const DEVIATION_BUCKETS: usize = 16;
const DEVIATION_MIN: f64 = 1.0e-9;
const DEVIATION_MAX: f64 = 1.0e-1;

/// Ratios listed by name in the report, most populous first.
const RATIOS_LISTED: usize = 24;
/// Individual events listed by name in the log, worst `|r|` first.
const EVENTS_LISTED: usize = 32;
/// Classified seams listed by name. Larger than [`EVENTS_LISTED`], because the classified
/// population IS the deliverable and a reader has to be able to check every member of it by hand
/// against a corporate-action record.
const SEAMS_LISTED: usize = 128;

/// Every base [`write_bar_seams`] writes. The single source of truth for this module's panel set,
/// walked by this module's registry test and mirrored in
/// [`shared::report::PRETRAIN_REPORT_BASES`].
pub const BAR_SEAM_BASES: &[&str] = &[
    "bar_seam_census",
    "bar_seam_ratios",
    "bar_seam_context",
    "bar_seam_tail_r",
    "bar_seam_bin_mass",
    "bar_seam_ruin_licence",
];

// ---------------------------------------------------------------------------
// Simple rationals
// ---------------------------------------------------------------------------

/// A candidate split ratio: `num / den` in lowest terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    pub num: u32,
    pub den: u32,
}

impl Rational {
    pub fn value(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn label(self) -> String {
        if self.den == 1 {
            format!("{}:1", self.num)
        } else if self.num == 1 {
            format!("1:{}", self.den)
        } else {
            format!("{}:{}", self.num, self.den)
        }
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Every admitted simple rational other than 1, ascending in value.
///
/// Built once and shared, because the nearest-rational lookup runs on every extreme bar in the
/// corpus and the set is a few hundred entries fixed at compile time.
pub fn simple_rationals() -> &'static [Rational] {
    static SET: std::sync::LazyLock<Vec<Rational>> = std::sync::LazyLock::new(|| {
        let mut out = Vec::new();
        for num in 1..=RATIONAL_MAX_TERM {
            for den in 1..=RATIONAL_MAX_TERM {
                if num != den && gcd(num, den) == 1 {
                    out.push(Rational { num, den });
                }
            }
        }
        // The integer ladder, which is where the only large terms a real corporate action uses
        // live: a 1:200 reverse split is `1:n`, never `p:q` with both terms large.
        for n in (RATIONAL_MAX_TERM + 1)..=RATIONAL_LADDER_MAX {
            out.push(Rational { num: n, den: 1 });
            out.push(Rational { num: 1, den: n });
        }
        out.sort_unstable_by(|a, b| a.value().total_cmp(&b.value()));
        out
    });
    &SET
}

/// The admitted rational closest to `ratio` in RELATIVE terms, and that relative deviation.
///
/// Relative and never absolute: a 1% error on a ratio of 100 is 1.0 in absolute terms and a 1%
/// error on a ratio of 1/100 is 0.0001, and an absolute rule would call the first a miss and the
/// second a bullseye. Binary search over the ascending set, so this is `O(log n)` per bar.
pub fn nearest_rational(ratio: f64) -> (Rational, f64) {
    let set = simple_rationals();
    let at = set.partition_point(|candidate| candidate.value() < ratio);
    let mut best = set[at.min(set.len() - 1)];
    let mut best_deviation = f64::INFINITY;
    for index in at.saturating_sub(1)..(at + 1).min(set.len()) {
        let candidate = set[index];
        let deviation = ((ratio - candidate.value()) / candidate.value()).abs();
        if deviation < best_deviation {
            best = candidate;
            best_deviation = deviation;
        }
    }
    (best, best_deviation)
}

// ---------------------------------------------------------------------------
// One extreme event
// ---------------------------------------------------------------------------

/// One bar whose `|r|` cleared [`EXTREME_LOG_THRESHOLD`], with every criterion already decided.
#[derive(Clone, Copy, Debug)]
pub struct ExtremeEvent {
    pub series: u32,
    pub bar: u32,
    pub ts_ms: i64,
    /// `ln(close / prev_close)` exactly as [`crate::torch::bar_dist::encode_dof`] produced it.
    pub r: f64,
    pub prev_close: f32,
    pub close: f32,
    pub s: f32,
    pub w: f32,
    pub volume: f32,
    /// Nearest admitted simple rational to `exp(r)`, and the RELATIVE deviation from it.
    pub ratio: Rational,
    pub ratio_deviation: f64,
    /// First bar of its ET trading day.
    pub session_open: bool,
    /// `s < QUIET_RANGE_FRACTION * |r|` and `|w| < QUIET_VOLUME_LOG`.
    pub quiet: bool,
    /// Neither neighbour bar is itself extreme.
    pub isolated: bool,
    /// The NEXT bar undoes at least half of this one: the signature of a bad print, not a split.
    pub reverts: bool,
    /// Bin this `r` lands in under the LIVE support geometry.
    pub bin: usize,
    /// Bar is in the train region, i.e. the population the supports were fitted from.
    pub in_train: bool,
}

impl ExtremeEvent {
    /// `exp(r)` sits on the ratio EXACTLY, to [`RATIONAL_TOLERANCE`]: both sides of the
    /// discontinuity are on one price ladder and the market did not move in the gap.
    pub fn on_rational(&self) -> bool {
        self.ratio_deviation <= RATIONAL_TOLERANCE
    }

    /// `exp(r)` is consistent with the ratio times one bar of market move, to
    /// [`RATIONAL_NEAR_TOLERANCE`]. Strictly weaker than [`Self::on_rational`] and never folded
    /// into it.
    pub fn near_rational(&self) -> bool {
        self.ratio_deviation <= RATIONAL_NEAR_TOLERANCE
    }

    /// All four criteria with the EXACT ratio test. The high-confidence classification, and a
    /// LOWER bound on the seam population.
    pub fn is_seam(&self) -> bool {
        self.on_rational() && self.session_open && self.quiet && self.isolated
    }

    /// The same conjunction with the LOOSE ratio test: an UPPER bound on the seam population,
    /// which is what an adjusted-against-unadjusted splice looks like once the market has moved
    /// across the gap.
    pub fn is_near_seam(&self) -> bool {
        self.near_rational() && self.session_open && self.quiet && self.isolated
    }

    pub fn ratio(&self) -> f64 {
        self.r.exp()
    }

    /// [`Self::is_seam`] at [`TIER_EXACT`], [`Self::is_near_seam`] at [`TIER_NEAR`]. The indexed
    /// form, so the census loop cannot report one tier's count under the other's name.
    pub fn is_seam_at(&self, tier: usize) -> bool {
        match tier {
            TIER_EXACT => self.is_seam(),
            TIER_NEAR => self.is_near_seam(),
            other => unreachable!("seam tier {other} does not exist"),
        }
    }
}

/// Ordering key for the retained-worst heap.
type EventKey = (OrderedFloat<f64>, u32, u32);

fn event_key(event: &ExtremeEvent) -> EventKey {
    (OrderedFloat(event.r.abs()), event.series, event.bar)
}

// ---------------------------------------------------------------------------
// The bounded accumulator
// ---------------------------------------------------------------------------

/// Ratio-test tiers a seam classification is carried at. Index 0 is the EXACT test and a lower
/// bound on the seam population; index 1 is the LOOSE one and an upper bound. Carried as an array
/// rather than two sets of fields so no report can quote one tier while labelling it the other.
pub const TIER_EXACT: usize = 0;
pub const TIER_NEAR: usize = 1;
pub const TIERS: usize = 2;
pub const TIER_NAMES: [&str; TIERS] = ["exact ratio", "ratio x one bar of market move"];

/// Everything one fold task measures. Every field is fixed-size or explicitly capped.
#[derive(Clone)]
pub struct Census {
    pub bars: u64,
    pub train_bars: u64,
    /// Bars whose `|r|` cleared each level of [`CENSUS_LOG_LEVELS`].
    pub level_counts: [u64; CENSUS_LOG_LEVELS.len()],
    pub extremes: u64,
    /// Extreme bars whose `exp(r)` passes each tier's ratio test.
    pub on_rational: [u64; TIERS],
    pub session_open: u64,
    pub quiet: u64,
    pub isolated: u64,
    pub reverts: u64,
    /// The four-way conjunction, per tier.
    pub seams: [u64; TIERS],
    pub seam_train: [u64; TIERS],
    /// Series carrying at least one seam, per tier.
    pub seam_series: [u64; TIERS],
    /// `[bin 0, bin 127]` populations over ALL bars, and over train-region bars.
    pub catch_all: [u64; 2],
    pub catch_all_train: [u64; 2],
    /// The same two counts restricted to seams, per tier.
    pub catch_all_seams: [[u64; 2]; TIERS],
    pub catch_all_seams_train: [[u64; 2]; TIERS],
    /// The same two counts restricted to REVERTING bars, i.e. the bad-print population rather than
    /// the corporate-action one. Carried because the two are the only two explanations of an
    /// extreme `r` and reporting one without the other leaves the other inferred.
    pub catch_all_reverts: [u64; 2],
    pub catch_all_reverts_train: [u64; 2],
    /// `s / |r|` histogram for extreme bars and for every bar with `r != 0`.
    pub range_ratio_extreme: [u64; CONTEXT_BUCKETS],
    pub range_ratio_ordinary: [u64; CONTEXT_BUCKETS],
    /// `w` histogram, same two populations.
    pub volume_extreme: [u64; CONTEXT_BUCKETS],
    pub volume_ordinary: [u64; CONTEXT_BUCKETS],
    /// Relative deviation from the nearest rational, extreme bars only.
    pub deviation: [u64; DEVIATION_BUCKETS],
    /// Per nearest rational: extreme bars, then the seam count at each tier.
    pub by_ratio: HashMap<(u32, u32), (u64, [u64; TIERS])>,
    /// The worst [`EVENT_BUFFER`] events by `|r|`, as a bounded min-heap.
    events: BinaryHeap<Reverse<(EventKey, usize)>>,
    event_store: Vec<ExtremeEvent>,
    /// Seam locations per tier, for the join onto the draw. Capped; overflow is counted.
    pub seam_keys: [Vec<(u32, u32)>; TIERS],
    pub seam_keys_dropped: [u64; TIERS],
    /// Reverting-bar locations, same cap and same purpose: the join that decides whether the draw's
    /// own extreme row is a bad print.
    pub revert_keys: Vec<(u32, u32)>,
    pub revert_keys_dropped: u64,
    pub seam_max_abs_r: [f64; TIERS],
}

impl Default for Census {
    fn default() -> Self {
        Self::new()
    }
}

impl Census {
    pub fn new() -> Self {
        Self {
            bars: 0,
            train_bars: 0,
            level_counts: [0; CENSUS_LOG_LEVELS.len()],
            extremes: 0,
            on_rational: [0; TIERS],
            session_open: 0,
            quiet: 0,
            isolated: 0,
            reverts: 0,
            seams: [0; TIERS],
            seam_train: [0; TIERS],
            seam_series: [0; TIERS],
            catch_all: [0; 2],
            catch_all_train: [0; 2],
            catch_all_seams: [[0; 2]; TIERS],
            catch_all_seams_train: [[0; 2]; TIERS],
            catch_all_reverts: [0; 2],
            catch_all_reverts_train: [0; 2],
            range_ratio_extreme: [0; CONTEXT_BUCKETS],
            range_ratio_ordinary: [0; CONTEXT_BUCKETS],
            volume_extreme: [0; CONTEXT_BUCKETS],
            volume_ordinary: [0; CONTEXT_BUCKETS],
            deviation: [0; DEVIATION_BUCKETS],
            by_ratio: HashMap::new(),
            events: BinaryHeap::new(),
            event_store: Vec::new(),
            seam_keys: [Vec::new(), Vec::new()],
            seam_keys_dropped: [0; TIERS],
            revert_keys: Vec::new(),
            revert_keys_dropped: 0,
            seam_max_abs_r: [0.0; TIERS],
        }
    }

    fn retain(&mut self, event: ExtremeEvent) {
        let key = event_key(&event);
        if self.event_store.len() < EVENT_BUFFER {
            self.event_store.push(event);
            self.events.push(Reverse((key, self.event_store.len() - 1)));
            return;
        }
        let Some(Reverse((worst_kept, slot))) = self.events.peek().copied() else {
            return;
        };
        if key <= worst_kept {
            return;
        }
        self.events.pop();
        self.event_store[slot] = event;
        self.events.push(Reverse((key, slot)));
    }

    /// Absorb one extreme event: every criterion counter, every histogram, the retained sample.
    fn absorb(&mut self, event: ExtremeEvent) {
        self.extremes += 1;
        self.on_rational[TIER_EXACT] += u64::from(event.on_rational());
        self.on_rational[TIER_NEAR] += u64::from(event.near_rational());
        self.session_open += u64::from(event.session_open);
        self.quiet += u64::from(event.quiet);
        self.isolated += u64::from(event.isolated);
        self.reverts += u64::from(event.reverts);

        let magnitude = event.r.abs();
        bump(
            &mut self.range_ratio_extreme,
            range_ratio_bucket(event.s as f64 / magnitude),
        );
        bump(&mut self.volume_extreme, volume_bucket(event.w as f64));
        bump(&mut self.deviation, deviation_bucket(event.ratio_deviation));

        let slot = self
            .by_ratio
            .entry((event.ratio.num, event.ratio.den))
            .or_insert((0, [0; TIERS]));
        slot.0 += 1;
        for tier in 0..TIERS {
            if !event.is_seam_at(tier) {
                continue;
            }
            slot.1[tier] += 1;
            self.seams[tier] += 1;
            self.seam_train[tier] += u64::from(event.in_train);
            self.seam_max_abs_r[tier] = self.seam_max_abs_r[tier].max(magnitude);
            if event.bin == 0 || event.bin == NUM_BAR_BINS as usize - 1 {
                let side = usize::from(event.bin != 0);
                self.catch_all_seams[tier][side] += 1;
                if event.in_train {
                    self.catch_all_seams_train[tier][side] += 1;
                }
            }
            if self.seam_keys[tier].len() < SEAM_BUFFER {
                self.seam_keys[tier].push((event.series, event.bar));
            } else {
                self.seam_keys_dropped[tier] += 1;
            }
        }
        if event.reverts {
            if event.bin == 0 || event.bin == NUM_BAR_BINS as usize - 1 {
                let side = usize::from(event.bin != 0);
                self.catch_all_reverts[side] += 1;
                if event.in_train {
                    self.catch_all_reverts_train[side] += 1;
                }
            }
            if self.revert_keys.len() < SEAM_BUFFER {
                self.revert_keys.push((event.series, event.bar));
            } else {
                self.revert_keys_dropped += 1;
            }
        }
        self.retain(event);
    }

    fn merge(mut self, other: Self) -> Self {
        self.bars += other.bars;
        self.train_bars += other.train_bars;
        merge_histogram(&mut self.level_counts, &other.level_counts);
        self.extremes += other.extremes;
        merge_histogram(&mut self.on_rational, &other.on_rational);
        self.session_open += other.session_open;
        self.quiet += other.quiet;
        self.isolated += other.isolated;
        self.reverts += other.reverts;
        merge_histogram(&mut self.seams, &other.seams);
        merge_histogram(&mut self.seam_train, &other.seam_train);
        merge_histogram(&mut self.seam_series, &other.seam_series);
        merge_histogram(&mut self.catch_all, &other.catch_all);
        merge_histogram(&mut self.catch_all_train, &other.catch_all_train);
        for tier in 0..TIERS {
            merge_histogram(
                &mut self.catch_all_seams[tier],
                &other.catch_all_seams[tier],
            );
            merge_histogram(
                &mut self.catch_all_seams_train[tier],
                &other.catch_all_seams_train[tier],
            );
            self.seam_max_abs_r[tier] = self.seam_max_abs_r[tier].max(other.seam_max_abs_r[tier]);
            self.seam_keys_dropped[tier] += other.seam_keys_dropped[tier];
        }
        merge_histogram(&mut self.catch_all_reverts, &other.catch_all_reverts);
        merge_histogram(
            &mut self.catch_all_reverts_train,
            &other.catch_all_reverts_train,
        );
        merge_histogram(&mut self.range_ratio_extreme, &other.range_ratio_extreme);
        merge_histogram(&mut self.range_ratio_ordinary, &other.range_ratio_ordinary);
        merge_histogram(&mut self.volume_extreme, &other.volume_extreme);
        merge_histogram(&mut self.volume_ordinary, &other.volume_ordinary);
        merge_histogram(&mut self.deviation, &other.deviation);
        for (key, (total, seams)) in other.by_ratio {
            let slot = self.by_ratio.entry(key).or_insert((0, [0; TIERS]));
            slot.0 += total;
            merge_histogram(&mut slot.1, &seams);
        }
        for (tier, keys) in other.seam_keys.into_iter().enumerate() {
            for key in keys {
                if self.seam_keys[tier].len() < SEAM_BUFFER {
                    self.seam_keys[tier].push(key);
                } else {
                    self.seam_keys_dropped[tier] += 1;
                }
            }
        }
        self.revert_keys_dropped += other.revert_keys_dropped;
        for key in other.revert_keys {
            if self.revert_keys.len() < SEAM_BUFFER {
                self.revert_keys.push(key);
            } else {
                self.revert_keys_dropped += 1;
            }
        }
        // Re-cap the retained sample rather than concatenating: the merged accumulator must obey
        // the same bound its parts did, or a reduce tree over 5,297 series would grow one buffer
        // per series.
        for event in other.event_store {
            self.retain(event);
        }
        self
    }

    /// The retained events, worst `|r|` first.
    pub fn worst_events(&self) -> Vec<ExtremeEvent> {
        let mut out = self.event_store.clone();
        out.sort_unstable_by(|a, b| b.r.abs().total_cmp(&a.r.abs()));
        out
    }

    /// Seam locations at one tier, ascending, ready for a binary-search join onto the draw.
    pub fn sorted_seam_keys(&self, tier: usize) -> Vec<(u32, u32)> {
        let mut keys = self.seam_keys[tier].clone();
        keys.sort_unstable();
        keys
    }

    /// The criterion ladder charted in `bar_seam_census`: the extreme population, then each
    /// criterion on its own, then the two conjunctions.
    pub fn criterion_counts(&self) -> [u64; 7] {
        [
            self.extremes,
            self.on_rational[TIER_EXACT],
            self.session_open,
            self.quiet,
            self.isolated,
            self.seams[TIER_EXACT],
            self.seams[TIER_NEAR],
        ]
    }

    /// Reverting-bar locations, ascending, for the same join.
    pub fn sorted_revert_keys(&self) -> Vec<(u32, u32)> {
        let mut keys = self.revert_keys.clone();
        keys.sort_unstable();
        keys
    }
}

fn bump(histogram: &mut [u64], bucket: usize) {
    histogram[bucket.min(histogram.len() - 1)] += 1;
}

fn merge_histogram(into: &mut [u64], from: &[u64]) {
    for (slot, value) in into.iter_mut().zip(from.iter()) {
        *slot += value;
    }
}

fn range_ratio_bucket(ratio: f64) -> usize {
    if !ratio.is_finite() || ratio <= 0.0 {
        return 0;
    }
    ((ratio / RANGE_RATIO_MAX * CONTEXT_BUCKETS as f64) as usize).min(CONTEXT_BUCKETS - 1)
}

fn volume_bucket(w: f64) -> usize {
    if !w.is_finite() {
        return CONTEXT_BUCKETS / 2;
    }
    let shifted = (w + VOLUME_LOG_MAX) / (2.0 * VOLUME_LOG_MAX);
    ((shifted * CONTEXT_BUCKETS as f64) as isize).clamp(0, CONTEXT_BUCKETS as isize - 1) as usize
}

fn deviation_bucket(deviation: f64) -> usize {
    if !(deviation > DEVIATION_MIN) {
        return 0;
    }
    let span = (DEVIATION_MAX / DEVIATION_MIN).ln();
    let position = (deviation / DEVIATION_MIN).ln() / span;
    ((position * DEVIATION_BUCKETS as f64) as isize).clamp(0, DEVIATION_BUCKETS as isize - 1)
        as usize
}

/// Lower edge of each `s / |r|` bucket.
pub fn range_ratio_edges() -> Vec<f64> {
    (0..CONTEXT_BUCKETS)
        .map(|b| b as f64 * RANGE_RATIO_MAX / CONTEXT_BUCKETS as f64)
        .collect()
}

/// Lower edge of each `w` bucket.
pub fn volume_edges() -> Vec<f64> {
    (0..CONTEXT_BUCKETS)
        .map(|b| -VOLUME_LOG_MAX + b as f64 * 2.0 * VOLUME_LOG_MAX / CONTEXT_BUCKETS as f64)
        .collect()
}

/// Lower edge of each rational-deviation bucket, in relative terms.
pub fn deviation_edges() -> Vec<f64> {
    let span = (DEVIATION_MAX / DEVIATION_MIN).ln();
    (0..DEVIATION_BUCKETS)
        .map(|b| DEVIATION_MIN * (b as f64 / DEVIATION_BUCKETS as f64 * span).exp())
        .collect()
}

// ---------------------------------------------------------------------------
// One series
// ---------------------------------------------------------------------------

/// A candidate extreme bar held back one step, because `isolated` needs the NEXT bar's `r`.
struct Pending {
    event: ExtremeEvent,
    /// Whether the PREVIOUS bar was itself extreme.
    previous_extreme: bool,
}

/// Stream one series into `census`.
///
/// One pass, one bar at a time, with exactly one bar of lookahead held in `pending`. Nothing else
/// is retained, so a 300,000-bar series costs no heap beyond the accumulator it is folding into.
fn scan_series(
    corpus: &BarCorpus,
    binner: &DofBinner<'_>,
    series: usize,
    train_end_ms: i64,
    census: &mut Census,
) {
    let mut previous_r: Option<f64> = None;
    let mut previous_close: Option<f32> = None;
    let mut previous_day: Option<i64> = None;
    let mut pending: Option<Pending> = None;
    let seams_before = census.seams;

    let finish = |census: &mut Census, held: Pending, next_r: Option<f64>| {
        let mut event = held.event;
        let next_extreme = next_r.is_some_and(|next| next.abs() > EXTREME_LOG_THRESHOLD);
        event.isolated = !held.previous_extreme && !next_extreme;
        // The bad-print signature, using the same rule `dataset::scan_symbol` classifies a tick
        // with: the next bar gives back more than half of this one.
        event.reverts = next_r.is_some_and(|next| (event.r + next).abs() < 0.5 * event.r.abs());
        census.absorb(event);
    };

    corpus.for_each_series_dof(series, |index, bar, dof| {
        let r = dof.r as f64;
        let ts_ms = bar.ts();
        let day = et_local_day(ts_ms);
        let in_train = ts_ms < train_end_ms;

        census.bars += 1;
        census.train_bars += u64::from(in_train);
        let magnitude = r.abs();
        for (slot, level) in census.level_counts.iter_mut().zip(CENSUS_LOG_LEVELS) {
            if magnitude > level {
                *slot += 1;
            }
        }
        let bin = binner.bin_of(r);
        if bin == 0 || bin == NUM_BAR_BINS as usize - 1 {
            let side = usize::from(bin != 0);
            census.catch_all[side] += 1;
            if in_train {
                census.catch_all_train[side] += 1;
            }
        }
        if r != 0.0 {
            bump(
                &mut census.range_ratio_ordinary,
                range_ratio_bucket(dof.s as f64 / magnitude),
            );
            bump(&mut census.volume_ordinary, volume_bucket(dof.w as f64));
        }

        if let Some(held) = pending.take() {
            finish(census, held, Some(r));
        }

        if magnitude > EXTREME_LOG_THRESHOLD {
            let (ratio, deviation) = nearest_rational(r.exp());
            let event = ExtremeEvent {
                series: series as u32,
                bar: index as u32,
                ts_ms,
                r,
                prev_close: previous_close.unwrap_or(bar.close),
                close: bar.close,
                s: dof.s,
                w: dof.w,
                volume: bar.volume,
                ratio,
                ratio_deviation: deviation,
                // The FIRST bar of a series has no predecessor day, so it cannot be shown to open
                // one; counting it as a session open would credit every listing's first bar.
                session_open: previous_day.is_some_and(|previous| previous != day),
                quiet: (dof.s as f64) < QUIET_RANGE_FRACTION * magnitude
                    && (dof.w as f64).abs() < QUIET_VOLUME_LOG,
                // Both decided once the next bar arrives.
                isolated: false,
                reverts: false,
                bin,
                in_train,
            };
            pending = Some(Pending {
                event,
                previous_extreme: previous_r
                    .is_some_and(|previous| previous.abs() > EXTREME_LOG_THRESHOLD),
            });
        }

        previous_r = Some(r);
        previous_close = Some(bar.close);
        previous_day = Some(day);
    });

    if let Some(held) = pending.take() {
        finish(census, held, None);
    }
    for tier in 0..TIERS {
        census.seam_series[tier] += u64::from(census.seams[tier] > seams_before[tier]);
    }
}

// ---------------------------------------------------------------------------
// The tail control and its cleaned twin
// ---------------------------------------------------------------------------

/// One reading of the six pairwise log-log slopes on `|r|`.
pub struct TailReading {
    /// Rows the thresholds were read against: every finite row, atom rows included.
    pub rows: u64,
    pub continuous_rows: u64,
    pub thresholds: Vec<f64>,
    pub slopes: Vec<EmpiricalSlope>,
    pub min_r: f64,
    pub max_r: f64,
    /// The [`BAR_SUPPORT_CLIP_QUANTILE`] and complementary quantiles of `r`, i.e. what
    /// `fit_dof_support` would place `lo[0]` and `hi[127]` at on THIS sample.
    pub clip_lo: f64,
    pub clip_hi: f64,
}

impl TailReading {
    pub fn span(&self) -> (f64, f64) {
        self.slopes
            .iter()
            .filter(|slope| slope.alpha.is_finite())
            .fold((f64::MAX, f64::MIN), |(lo, hi), slope| {
                (lo.min(slope.alpha), hi.max(slope.alpha))
            })
    }

    /// Most leverage a LONG survives against `clip_lo`: `1 / (1 - exp(r_min))`.
    ///
    /// The identical expression [`crate::torch::train::bar_family::RuinLicence`] carries, from
    /// `1 + F(exp(r) - 1) > 0`. Reproduced here rather than imported because that type is built
    /// from a fitted family's own draw extremes and this one is built from a clipped quantile of a
    /// filtered sample; `the_ruin_licence_matches_the_live_support_edges` pins both against the
    /// artifact on disk.
    pub fn long_max_leverage(&self) -> f64 {
        1.0 / (1.0 - self.clip_lo.exp())
    }

    /// Most leverage a SHORT survives against `clip_hi`: `1 / (exp(r_max) - 1)`. Always the
    /// binding side, because `ln(1 + y) < -ln(1 - y)`.
    pub fn short_max_leverage(&self) -> f64 {
        1.0 / (self.clip_hi.exp() - 1.0)
    }

    pub fn binding_max_leverage(&self) -> f64 {
        self.long_max_leverage().min(self.short_max_leverage())
    }
}

/// Read the tail of `r` over `rows`, with the SAME estimator at the SAME levels as
/// [`crate::torch::train::bar_family`].
fn read_tail(rows: &[BarDof]) -> TailReading {
    let (ordered, total_rows, continuous_rows) = upper_order_statistics(rows);
    let (thresholds, slopes) = empirical_tail_slopes(&ordered, total_rows);
    let (min_r, max_r) = rows
        .iter()
        .map(|row| row.r as f64)
        .filter(|r| r.is_finite())
        .fold((f64::MAX, f64::MIN), |(lo, hi), r| (lo.min(r), hi.max(r)));
    let (clip_lo, clip_hi) = clipped_r_range(rows);
    TailReading {
        rows: total_rows,
        continuous_rows,
        thresholds,
        slopes,
        min_r,
        max_r,
        clip_lo,
        clip_hi,
    }
}

/// The [`BAR_SUPPORT_CLIP_QUANTILE`] and complementary quantiles of `r`, by the rule
/// `bar_dist::clipped_range` applies: sort ascending in `f32`, then index
/// `round(q * (n - 1))`.
///
/// Reproduced rather than called because `clipped_range` is private to the support fitter and
/// takes an already-sorted `f32` column. The control below asserts that this rule, run on the
/// UNFILTERED draw, reproduces the live artifact's `lo[DOF_R][0]` and `hi[DOF_R][127]` exactly —
/// which is the only evidence that makes the CLEANED figure beside it mean anything.
fn clipped_r_range(rows: &[BarDof]) -> (f64, f64) {
    let mut column: Vec<f32> = rows.iter().map(|row| row.r).filter(|r| r.is_finite()).collect();
    if column.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    column.par_sort_unstable_by(f32::total_cmp);
    let last = column.len() - 1;
    let at = |q: f64| column[((q * last as f64).round() as usize).min(last)] as f64;
    (
        at(BAR_SUPPORT_CLIP_QUANTILE),
        at(1.0 - BAR_SUPPORT_CLIP_QUANTILE),
    )
}

// ---------------------------------------------------------------------------
// Everything the pass produced
// ---------------------------------------------------------------------------

/// The draw's own extreme row on one side, named, with the classification's verdict on it.
///
/// The live leverage licence is read off the SUPPORT edges, but the ruin table beside it was argued
/// from the DRAW's worst bar, and that bar was reported at -16094.38 bps, which is `-ln 5` to `f32`
/// precision on both sides at once. Whether a 5:1 corporate action produced it is exactly the
/// question this module exists to answer, so the row is named and classified here rather than
/// inferred from the fact that the number looks like a split ratio.
#[derive(Clone, Debug)]
pub struct DrawExtreme {
    pub symbol: String,
    pub bar: u32,
    pub ts_ms: i64,
    pub r: f64,
    pub ratio: Rational,
    pub ratio_deviation: f64,
    /// Whether the corpus pass classified this row a seam, per tier.
    pub seam: [bool; TIERS],
    /// Whether the corpus pass classified this row a REVERTING bar, i.e. a bad print rather than a
    /// corporate action. The competing explanation of an extreme `r`, and the one that has to be
    /// excluded before "genuine market move" is what is left.
    pub reverts: bool,
}

/// The whole audit: the corpus census, the draw's tail before and after the seams are removed, and
/// the support geometry both were read against.
pub struct SeamAudit {
    pub census: Census,
    /// Six pairwise slopes on the FULL 4M draw. The control: it must reproduce the figure already
    /// measured for `bar_family`, or the cleaned readings beside it are not comparable to anything.
    pub control: TailReading,
    /// The same estimator on the same draw with the classified seam rows of each tier removed.
    pub cleaned: [TailReading; TIERS],
    /// Draw rows the seam join removed, per tier.
    pub draw_rows_removed: [u64; TIERS],
    pub draw_rows: u64,
    /// The draw's own most negative and most positive `r` row, named and classified. `[min, max]`.
    pub draw_extremes: [DrawExtreme; 2],
    /// Ticker per series index, so a listed event names a symbol rather than a fold-order integer.
    pub series_names: Vec<String>,
    /// The live support's own `r` range and the licences it sets.
    pub support_lo: f64,
    pub support_hi: f64,
    /// Masses of bins 0 and 127 as the artifact records them.
    pub support_catch_all_mass: [f64; 2],
    /// Path the geometry was read from, and whether the `.v5.json` sibling carries the same
    /// `DOF_R` bounds.
    pub supports_path: String,
    pub cross_check_path: Option<String>,
    pub cross_check_agrees: Option<bool>,
    pub symbols: usize,
    pub wall_seconds: f64,
    /// `getrusage(RUSAGE_SELF).ru_maxrss`, in bytes.
    pub peak_rss_bytes: u64,
}

impl SeamAudit {
    pub fn support_long_max_leverage(&self) -> f64 {
        1.0 / (1.0 - self.support_lo.exp())
    }

    pub fn support_short_max_leverage(&self) -> f64 {
        1.0 / (self.support_hi.exp() - 1.0)
    }

    pub fn support_binding_max_leverage(&self) -> f64 {
        self.support_long_max_leverage()
            .min(self.support_short_max_leverage())
    }

    /// Share of ALL bars that is a classified seam, at one tier.
    pub fn seam_share(&self, tier: usize) -> f64 {
        self.census.seams[tier] as f64 / self.census.bars.max(1) as f64
    }

    /// Seam share of the two catch-all bins' populations, `[bin 0, bin 127]`, over the whole
    /// corpus.
    pub fn catch_all_seam_share(&self, tier: usize) -> [f64; 2] {
        std::array::from_fn(|side| {
            self.census.catch_all_seams[tier][side] as f64
                / self.census.catch_all[side].max(1) as f64
        })
    }

    /// The same shares over the train region alone, which is the population the support's recorded
    /// masses actually describe.
    pub fn catch_all_seam_share_train(&self, tier: usize) -> [f64; 2] {
        std::array::from_fn(|side| {
            self.census.catch_all_seams_train[tier][side] as f64
                / self.census.catch_all_train[side].max(1) as f64
        })
    }

    /// Does the census license the conclusion that the corpus is contaminated?
    ///
    /// A verdict and not a threshold on taste: contamination means classified seams EXIST and reach
    /// the region the support's catch-alls are placed from, because that region is the only thing
    /// any live number rests on. Read at whichever tier has evidence, since a seam that carries a
    /// bar of market move across the gap is no less a seam for it — the tier only says how strong
    /// the identification is, and both counts are reported.
    pub fn contaminated(&self) -> bool {
        (0..TIERS).any(|tier| {
            self.census.seams[tier] > 0
                && (self.census.catch_all_seams[tier][0] > 0
                    || self.census.catch_all_seams[tier][1] > 0)
        })
    }

    pub fn report_lines(&self) -> Vec<String> {
        let census = &self.census;
        let mut lines = Vec::new();
        lines.push(format!(
            "corpus         {} bars over {} symbols, {} of them in the train region; geometry from \
             {}{}",
            census.bars,
            self.symbols,
            census.train_bars,
            self.supports_path,
            match (&self.cross_check_path, self.cross_check_agrees) {
                (Some(path), Some(true)) =>
                    format!(" (DOF r bounds identical to {path}, so the choice is immaterial)"),
                (Some(path), Some(false)) => format!(" (DOF r bounds DIFFER from {path})"),
                _ => String::new(),
            }
        ));
        let level_parts: Vec<String> = CENSUS_LEVEL_NAMES
            .iter()
            .zip(census.level_counts.iter())
            .map(|(name, count)| {
                format!(
                    "{name}: {count} ({:.3e})",
                    *count as f64 / census.bars.max(1) as f64
                )
            })
            .collect();
        lines.push(format!("exceedance     |r| above {}", level_parts.join(" | ")));
        lines.push(format!(
            "census         {} bars above ln 1.5; exp(r) on a simple rational EXACTLY {} \
             ({:.2}%), within {:.0}% of one {} ({:.2}%); at a session open {} ({:.2}%); \
             unremarkable in s and w {} ({:.2}%); isolated {} ({:.2}%). ALL FOUR at the exact \
             ratio {} ({:.2}%), at the loose ratio {} ({:.2}%). Reverting pairs, i.e. bad prints \
             rather than splits: {}",
            census.extremes,
            census.on_rational[TIER_EXACT],
            100.0 * census.on_rational[TIER_EXACT] as f64 / census.extremes.max(1) as f64,
            100.0 * RATIONAL_NEAR_TOLERANCE,
            census.on_rational[TIER_NEAR],
            100.0 * census.on_rational[TIER_NEAR] as f64 / census.extremes.max(1) as f64,
            census.session_open,
            100.0 * census.session_open as f64 / census.extremes.max(1) as f64,
            census.quiet,
            100.0 * census.quiet as f64 / census.extremes.max(1) as f64,
            census.isolated,
            100.0 * census.isolated as f64 / census.extremes.max(1) as f64,
            census.seams[TIER_EXACT],
            100.0 * census.seams[TIER_EXACT] as f64 / census.extremes.max(1) as f64,
            census.seams[TIER_NEAR],
            100.0 * census.seams[TIER_NEAR] as f64 / census.extremes.max(1) as f64,
            census.reverts,
        ));
        for tier in 0..TIERS {
            lines.push(format!(
                "seams {:<8} {} over {} symbols = {:.3e} of all bars ({} in the train region); \
                 worst |r| {:.2} bps; keys dropped at the {SEAM_BUFFER} cap: {}",
                if tier == TIER_EXACT { "EXACT" } else { "LOOSE" },
                census.seams[tier],
                census.seam_series[tier],
                self.seam_share(tier),
                census.seam_train[tier],
                census.seam_max_abs_r[tier] * 10_000.0,
                census.seam_keys_dropped[tier],
            ));
        }

        for (rational, total, seams) in ranked_ratios(census).iter().take(RATIOS_LISTED) {
            lines.push(format!(
                "  ratio {:<8} exp(r) = {:.6}: {total} extreme bars, {} exact-ratio seams, {} \
                 loose-ratio seams",
                rational.label(),
                rational.value(),
                seams[TIER_EXACT],
                seams[TIER_NEAR],
            ));
        }

        lines.push(format!(
            "deviation      relative distance of exp(r) from the nearest simple rational, \
             {} geometric buckets from {DEVIATION_MIN:.0e} to {DEVIATION_MAX:.0e} (first bucket is \
             everything at or below {DEVIATION_MIN:.0e}, i.e. EXACT): {}",
            DEVIATION_BUCKETS,
            census
                .deviation
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ));

        let name = |series: u32| -> &str {
            self.series_names
                .get(series as usize)
                .map_or("?", String::as_str)
        };
        let describe = |event: &ExtremeEvent| {
            format!(
                "  {} bar {} at {}: r = {:+.6} ({:.2} bps), exp(r) = {:.8} vs {} = {:.8}, rel dev \
                 {:.3e}; close {} <- {}; s = {:.5}, w = {:+.4}, volume {:.0}; bin {}; \
                 exact {} near {} open {} quiet {} isolated {} reverts {} => seam {} / near-seam {}",
                name(event.series),
                event.bar,
                iso_ms(event.ts_ms),
                event.r,
                event.r * 10_000.0,
                event.ratio(),
                event.ratio.label(),
                event.ratio.value(),
                event.ratio_deviation,
                event.close,
                event.prev_close,
                event.s,
                event.w,
                event.volume,
                event.bin,
                event.on_rational(),
                event.near_rational(),
                event.session_open,
                event.quiet,
                event.isolated,
                event.reverts,
                event.is_seam(),
                event.is_near_seam(),
            )
        };
        let worst = self.census.worst_events();
        lines.push(format!(
            "worst          the {} largest |r| bars retained of {} extremes",
            worst.len().min(EVENTS_LISTED),
            census.extremes
        ));
        for event in worst.iter().take(EVENTS_LISTED) {
            lines.push(describe(event));
        }

        // The classified population itself, which is the deliverable. Listed from the same retained
        // buffer, so a seam beyond the buffer's reach cannot appear here; the counts above are the
        // authority on how many there are and `seam_keys_dropped` on whether any were lost.
        let seams: Vec<&ExtremeEvent> = worst.iter().filter(|e| e.is_near_seam()).collect();
        lines.push(format!(
            "classified     {} of the {} retained extremes are seams at the loose tier, {} of them \
             at the exact tier; every one of them listed below",
            seams.len(),
            worst.len(),
            seams.iter().filter(|e| e.is_seam()).count(),
        ));
        for event in seams.iter().take(SEAMS_LISTED) {
            lines.push(describe(event));
        }

        for (side, extreme) in self.draw_extremes.iter().enumerate() {
            lines.push(format!(
                "draw {:<9} the draw's {} r row is {} bar {} at {}: r = {:+.9} ({:.2} bps), \
                 exp(r) = {:.8}, nearest simple rational {} = {:.8} at rel dev {:.3e}; classified \
                 a seam: exact {} loose {}; classified a REVERTING bad print: {}",
                if side == 0 { "min" } else { "max" },
                if side == 0 { "most negative" } else { "most positive" },
                extreme.symbol,
                extreme.bar,
                iso_ms(extreme.ts_ms),
                extreme.r,
                extreme.r * 10_000.0,
                extreme.r.exp(),
                extreme.ratio.label(),
                extreme.ratio.value(),
                extreme.ratio_deviation,
                extreme.seam[TIER_EXACT],
                extreme.seam[TIER_NEAR],
                extreme.reverts,
            ));
        }

        let slopes_of = |reading: &TailReading| {
            reading
                .slopes
                .iter()
                .map(|s| format!("{:.4}", s.alpha))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let thresholds_of = |reading: &TailReading| {
            reading
                .thresholds
                .iter()
                .map(|x| format!("{:.2}", x * 10_000.0))
                .collect::<Vec<_>>()
                .join("/")
        };
        let (control_lo, control_hi) = self.control.span();
        lines.push(format!(
            "tail control   {} of {} draw rows off the r == 0 atom; thresholds {} bps; six \
             pairwise slopes {}; span {:.4}-{:.4}",
            self.control.continuous_rows,
            self.control.rows,
            thresholds_of(&self.control),
            slopes_of(&self.control),
            control_lo,
            control_hi,
        ));
        for tier in 0..TIERS {
            let (lo, hi) = self.cleaned[tier].span();
            lines.push(format!(
                "tail {:<9} {} seam rows removed from {} draw rows ({} tier); thresholds {} bps; \
                 six pairwise slopes {}; span {:.4}-{:.4}",
                if tier == TIER_EXACT {
                    "cleaned"
                } else {
                    "cleaned-x"
                },
                self.draw_rows_removed[tier],
                self.draw_rows,
                TIER_NAMES[tier],
                thresholds_of(&self.cleaned[tier]),
                slopes_of(&self.cleaned[tier]),
                lo,
                hi,
            ));
        }
        lines.push(format!(
            "bin mass       artifact records bin 0 at {:.6}% and bin 127 at {:.6}% of its 4M \
             fitting draw. Over the WHOLE corpus the two hold {} and {} bars; over the TRAIN \
             region {} and {}",
            100.0 * self.support_catch_all_mass[0],
            100.0 * self.support_catch_all_mass[1],
            census.catch_all[0],
            census.catch_all[1],
            census.catch_all_train[0],
            census.catch_all_train[1],
        ));
        for tier in 0..TIERS {
            lines.push(format!(
                "bin seams {:<4} of those, {} and {} are seams at the {} tier = {:.4}% of bin 0 \
                 and {:.4}% of bin 127; over the train region {} and {} = {:.4}% and {:.4}%",
                if tier == TIER_EXACT { "EXACT" } else { "LOOSE" },
                census.catch_all_seams[tier][0],
                census.catch_all_seams[tier][1],
                TIER_NAMES[tier],
                100.0 * self.catch_all_seam_share(tier)[0],
                100.0 * self.catch_all_seam_share(tier)[1],
                census.catch_all_seams_train[tier][0],
                census.catch_all_seams_train[tier][1],
                100.0 * self.catch_all_seam_share_train(tier)[0],
                100.0 * self.catch_all_seam_share_train(tier)[1],
            ));
        }
        lines.push(format!(
            "bin prints     and {} and {} are REVERTING bad prints = {:.4}% of bin 0 and {:.4}% of \
             bin 127; over the train region {} and {}. The competing population, reported beside \
             the seams because between them they exhaust the non-market explanations of an extreme \
             r. Keys dropped at the {SEAM_BUFFER} cap: {}",
            census.catch_all_reverts[0],
            census.catch_all_reverts[1],
            100.0 * census.catch_all_reverts[0] as f64 / census.catch_all[0].max(1) as f64,
            100.0 * census.catch_all_reverts[1] as f64 / census.catch_all[1].max(1) as f64,
            census.catch_all_reverts_train[0],
            census.catch_all_reverts_train[1],
            census.revert_keys_dropped,
        ));
        lines.push(format!(
            "support edges  LIVE lo[r][0] = {:+.8} ({:.2} bps) licenses a long at {:.4}x; \
             hi[r][127] = {:+.8} ({:.2} bps) licenses a short at {:.4}x; the SHORT side binds at \
             {:.4}x",
            self.support_lo,
            self.support_lo * 10_000.0,
            self.support_long_max_leverage(),
            self.support_hi,
            self.support_hi * 10_000.0,
            self.support_short_max_leverage(),
            self.support_binding_max_leverage(),
        ));
        lines.push(format!(
            "edges control  the same {:.0e} clip quantiles recomputed on the UNFILTERED draw give \
             {:+.8} / {:+.8}, i.e. {:.4}x long and {:.4}x short, binding {:.4}x",
            BAR_SUPPORT_CLIP_QUANTILE,
            self.control.clip_lo,
            self.control.clip_hi,
            self.control.long_max_leverage(),
            self.control.short_max_leverage(),
            self.control.binding_max_leverage(),
        ));
        for tier in 0..TIERS {
            let cleaned = &self.cleaned[tier];
            lines.push(format!(
                "edges {:<9} with the {} seams removed the edges become {:+.8} ({:.2} bps) / \
                 {:+.8} ({:.2} bps), i.e. {:.4}x long and {:.4}x short, binding {:.4}x — a change \
                 of {:+.4}x on the binding side against the control",
                if tier == TIER_EXACT { "cleaned" } else { "cleaned-x" },
                TIER_NAMES[tier],
                cleaned.clip_lo,
                cleaned.clip_lo * 10_000.0,
                cleaned.clip_hi,
                cleaned.clip_hi * 10_000.0,
                cleaned.long_max_leverage(),
                cleaned.short_max_leverage(),
                cleaned.binding_max_leverage(),
                cleaned.binding_max_leverage() - self.control.binding_max_leverage(),
            ));
        }
        lines.push(format!(
            "verdict        {}",
            if self.contaminated() {
                format!(
                    "CONTAMINATED. Corporate-action seams reach the catch-all bins: {} at the \
                     exact-ratio tier over {} symbols ({:.4}% of bin 0, {:.4}% of bin 127) and {} \
                     at the loose tier over {} symbols ({:.4}% of bin 0, {:.4}% of bin 127). The \
                     exact count is a LOWER bound on the population and the loose one an UPPER \
                     bound; neither is a point estimate.",
                    census.seams[TIER_EXACT],
                    census.seam_series[TIER_EXACT],
                    100.0 * self.catch_all_seam_share(TIER_EXACT)[0],
                    100.0 * self.catch_all_seam_share(TIER_EXACT)[1],
                    census.seams[TIER_NEAR],
                    census.seam_series[TIER_NEAR],
                    100.0 * self.catch_all_seam_share(TIER_NEAR)[0],
                    100.0 * self.catch_all_seam_share(TIER_NEAR)[1],
                )
            } else if census.seams[TIER_NEAR] == 0 {
                "CLEAN on this test. No extreme bar satisfies all four split criteria at either \
                 ratio tier, so the extreme r population is market moves and the question is \
                 retired."
                    .to_owned()
            } else {
                format!(
                    "CLEAN WHERE IT MATTERS. {} exact-ratio and {} loose-ratio seams exist, but \
                     none of them reaches bin 0 or bin 127, so no support edge, no tail estimate \
                     and no ruin licence rests on one.",
                    census.seams[TIER_EXACT], census.seams[TIER_NEAR],
                )
            }
        ));
        lines.push(format!(
            "resources      {:.1}s wall, {:.2} GiB peak RSS by getrusage(RUSAGE_SELF).ru_maxrss",
            self.wall_seconds,
            self.peak_rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        ));
        lines
    }
}

/// `getrusage(RUSAGE_SELF).ru_maxrss`, in bytes. Linux reports it in kibibytes.
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` fills the supplied `rusage` and reads nothing else.
    let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0;
    if !ok {
        return 0;
    }
    // SAFETY: a zero return means the struct was written.
    let usage = unsafe { usage.assume_init() };
    (usage.ru_maxrss as u64).saturating_mul(1024)
}

// ---------------------------------------------------------------------------
// Args and entry point
// ---------------------------------------------------------------------------

/// Everything the audit needs. Deliberately separate from `PretrainArgs`: this touches no model,
/// no device and no schedule.
#[derive(Clone, Debug)]
pub struct SplitSeamArgs {
    pub corpus: CorpusFlags,
    /// Support whose `bin_of` places every bar and whose `lo[r][0]` / `hi[r][127]` set the live
    /// ruin licence. Read, never written.
    pub supports: String,
    /// A second support file whose `DOF_R` bounds are compared against the first, so the statement
    /// "which geometry this used and why it does not matter" is a measurement. Empty to skip.
    pub cross_check_supports: String,
    /// Reports directory, i.e. a run's `gens/<n>`.
    pub output: String,
    /// Rows to draw for the tail control. MUST match the support's recorded `sample_count`.
    pub samples: usize,
    /// Draw seed. MUST be the `train_seed` of the run that fitted the support.
    pub seed: u64,
}

/// Census the whole corpus for corporate-action seams, quantify what they contaminate, and emit
/// the verdict.
pub fn audit_split_seams(args: SplitSeamArgs) -> Result<()> {
    ensure!(args.samples > 0, "--samples must be positive");

    let source = Path::new(&args.supports);
    let supports = BarSupports::load(source)
        .with_context(|| format!("reading the support geometry to bin against, {}", source.display()))?;
    ensure!(
        supports.num_bins() == NUM_BAR_BINS,
        "{} has {} bins, this build uses {NUM_BAR_BINS}",
        source.display(),
        supports.num_bins()
    );

    // The choice of artifact is only immaterial if the two agree, so it is checked rather than
    // asserted in prose. `bin_of` reads `lo` and the atom set alone, so DOF r's bounds decide
    // every bin assignment this pass makes.
    let (cross_check_path, cross_check_agrees) = if args.cross_check_supports.is_empty() {
        (None, None)
    } else {
        let other = Path::new(&args.cross_check_supports);
        if other.exists() {
            let twin = BarSupports::load(other)
                .with_context(|| format!("reading the cross-check support {}", other.display()))?;
            let agrees = twin.lower_bounds(DOF_R) == supports.lower_bounds(DOF_R)
                && twin.upper_bounds(DOF_R) == supports.upper_bounds(DOF_R);
            (
                Some(other.display().to_string()),
                Some(agrees),
            )
        } else {
            (None, None)
        }
    };

    let corpus = load_corpus(&args.corpus)?;
    let provenance = supports.provenance().with_context(|| {
        format!(
            "{} carries no provenance, so the draw this pass makes for the tail control cannot be \
             identified against the support it is compared with",
            source.display()
        )
    })?;
    ensure!(
        provenance.split_bounds == corpus.split_bounds(),
        "{} was fitted against split bounds {:?} but this corpus resolves {:?}",
        source.display(),
        provenance.split_bounds,
        corpus.split_bounds()
    );
    ensure!(
        provenance.sample_count == args.samples,
        "{} records a fit sample of {} rows but --samples is {}; the tail control must be the SAME \
         draw the live figure was measured on",
        source.display(),
        provenance.sample_count,
        args.samples
    );

    println!(
        "auditing {} series at {}s against {}: split bounds {:?}, corpus fingerprint {}, extreme \
         threshold |r| > ln 1.5 = {EXTREME_LOG_THRESHOLD:.6}",
        corpus.series_count(),
        corpus.res_secs(),
        source.display(),
        provenance.split_bounds,
        provenance.corpus_fingerprint,
    );

    // One shared body, so the subcommand and the test that drives a planted split run the same
    // code: the corpus census, then the control draw and the two cleaned readings off it.
    let mut audit = build_audit(
        &corpus,
        &supports,
        args.samples,
        args.seed,
        &source.display().to_string(),
    );
    ensure!(
        audit.draw_rows > 0,
        "the train region yielded no DOF rows, so there is no tail to read"
    );
    audit.cross_check_path = cross_check_path;
    audit.cross_check_agrees = cross_check_agrees;

    for line in audit.report_lines() {
        println!("{line}");
    }
    write_bar_seams(Path::new(&args.output), &audit)?;
    println!("split seam audit charts written to {}", args.output);
    Ok(())
}

/// Whether the drawn row's bar is one of a sorted key set the corpus pass built - the seam keys of
/// one tier, or the reverting-bar keys. One join, because the draw's rows and the census's rows are
/// identified the same way whichever population is being asked about.
fn row_is_keyed(sorted_keys: &[(u32, u32)], window: &WindowRef) -> bool {
    sorted_keys
        .binary_search(&(window.symbol, window.bar_index))
        .is_ok()
}

/// Run the census over an already-loaded corpus. Split out from [`audit_split_seams`] so a test
/// can drive it on a synthetic corpus with a planted split.
///
/// The binner is borrowed out of the support ONCE, before the fold: [`BarSupports`] owns tensors
/// and is therefore not `Sync`, while [`DofBinner`] is three slices of `f64` and places bars in
/// exactly the bins [`BarSupports::bin_of`] would, because that method is defined as this one.
pub fn census_corpus(corpus: &BarCorpus, supports: &BarSupports) -> Census {
    let train_end_ms = corpus.split_bounds().0;
    let binner = supports.binner(DOF_R);
    (0..corpus.series_count())
        .into_par_iter()
        .fold(Census::new, |mut accumulator, series| {
            scan_series(corpus, &binner, series, train_end_ms, &mut accumulator);
            accumulator
        })
        .reduce(Census::new, Census::merge)
}

/// Assemble a [`SeamAudit`] from an already-loaded corpus and support. The body of
/// [`audit_split_seams`] after the argument checks, so a test drives the same code the subcommand
/// does.
pub fn build_audit(
    corpus: &BarCorpus,
    supports: &BarSupports,
    samples: usize,
    seed: u64,
    supports_path: &str,
) -> SeamAudit {
    let started = Instant::now();
    let census = census_corpus(corpus, supports);
    let located = corpus.sample_train_dof_located(samples, seed);
    let draw_rows = located.len() as u64;
    let full: Vec<BarDof> = located.iter().map(|(_, _, dof)| *dof).collect();
    let control = read_tail(&full);
    drop(full);
    // Both tiers are cleaned against the SAME draw, so the two readings differ only by which rows
    // the classification removed and never by which rows were drawn.
    let mut draw_rows_removed = [0u64; TIERS];
    let keys: [Vec<(u32, u32)>; TIERS] = std::array::from_fn(|tier| census.sorted_seam_keys(tier));
    let cleaned: [TailReading; TIERS] = std::array::from_fn(|tier| {
        let cleaned_rows: Vec<BarDof> = located
            .iter()
            .filter(|(window, _, _)| !row_is_keyed(&keys[tier], window))
            .map(|(_, _, dof)| *dof)
            .collect();
        draw_rows_removed[tier] = draw_rows - cleaned_rows.len() as u64;
        read_tail(&cleaned_rows)
    });
    // The two rows the ruin table was argued from, named. `total_cmp` rather than `partial_cmp` so
    // a non-finite row cannot silently win the comparison.
    let revert_keys = census.sorted_revert_keys();
    let name_extreme = |row: &(WindowRef, i64, BarDof)| {
        let (window, ts_ms, dof) = row;
        let r = f64::from(dof.r);
        let (ratio, ratio_deviation) = nearest_rational(r.exp());
        DrawExtreme {
            symbol: corpus.symbol(window.symbol as usize).to_owned(),
            bar: window.bar_index,
            ts_ms: *ts_ms,
            r,
            ratio,
            ratio_deviation,
            seam: std::array::from_fn(|tier| row_is_keyed(&keys[tier], window)),
            reverts: row_is_keyed(&revert_keys, window),
        }
    };
    let finite = |row: &&(WindowRef, i64, BarDof)| row.2.r.is_finite();
    let draw_extremes = [
        located
            .iter()
            .filter(finite)
            .min_by(|a, b| a.2.r.total_cmp(&b.2.r)),
        located
            .iter()
            .filter(finite)
            .max_by(|a, b| a.2.r.total_cmp(&b.2.r)),
    ]
    .map(|row| {
        row.map(name_extreme).unwrap_or_else(|| DrawExtreme {
            symbol: String::new(),
            bar: 0,
            ts_ms: 0,
            r: f64::NAN,
            ratio: Rational { num: 1, den: 1 },
            ratio_deviation: f64::NAN,
            seam: [false; TIERS],
            reverts: false,
        })
    });
    drop(located);
    let bins = NUM_BAR_BINS as usize;
    SeamAudit {
        census,
        control,
        cleaned,
        draw_rows_removed,
        draw_rows,
        draw_extremes,
        series_names: (0..corpus.series_count())
            .map(|series| corpus.symbol(series).to_owned())
            .collect(),
        support_lo: supports.lower_bounds(DOF_R)[0],
        support_hi: supports.upper_bounds(DOF_R)[bins - 1],
        support_catch_all_mass: [
            supports.bin_masses(DOF_R)[0],
            supports.bin_masses(DOF_R)[bins - 1],
        ],
        supports_path: supports_path.to_owned(),
        cross_check_path: None,
        cross_check_agrees: None,
        symbols: corpus.series_count(),
        wall_seconds: started.elapsed().as_secs_f64(),
        peak_rss_bytes: peak_rss_bytes(),
    }
}

/// The most populous nearest-rationals, descending in extreme-bar count, each with its seam count
/// at both tiers.
pub fn ranked_ratios(census: &Census) -> Vec<(Rational, u64, [u64; TIERS])> {
    let mut ratios: Vec<(Rational, u64, [u64; TIERS])> = census
        .by_ratio
        .iter()
        .map(|((num, den), (total, seams))| {
            (
                Rational {
                    num: *num,
                    den: *den,
                },
                *total,
                *seams,
            )
        })
        .collect();
    ratios.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.value().total_cmp(&b.0.value())));
    ratios
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::{encode_series, BarSupports};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;
    use shared::bars::{write_bar_file, PackedBar, FILE_EXTENSION};
    use shared::report::{read_report, ReportKind};
    use std::path::PathBuf;

    const RES: u32 = 300;
    const RES_MS: i64 = RES as i64 * 1000;
    /// Bars per synthetic trading day, so a planted seam can be placed on a session open.
    const BARS_PER_DAY: usize = 78;

    struct Fixture {
        dir: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "split_seams_{label}_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// A synthetic series of 5-minute bars on a 78-bar trading day, laid out so bar index
    /// `day * BARS_PER_DAY` is the first bar of a day in ET.
    ///
    /// `splits` maps a DAY index to `(ratio, seam-bar move)`. The ratio is applied to every price
    /// from that day onward, which is what an UNADJUSTED split looks like in a stored series: the
    /// level shifts at the open, the bar itself trades a normal range on normal volume, and nothing
    /// reverts. The seam-bar move replaces that first bar's random drift, so a split can be planted
    /// either with the two sides on ONE price ladder (move 1.0, and `exp(r)` is the ratio exactly)
    /// or with a bar of market move folded into the gap (move 1.01, and `exp(r)` is the ratio times
    /// it). Real overnight seams are the second kind, which is the whole reason the classification
    /// carries two ratio tiers.
    fn synth_series(
        seed: u64,
        days: usize,
        first_open_ms: i64,
        splits: &[(usize, f32, f32)],
    ) -> Vec<PackedBar> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut close = 40.0f32;
        let mut factor = 1.0f32;
        let mut out = Vec::with_capacity(days * BARS_PER_DAY);
        for day in 0..days {
            let planted = splits.iter().find(|(at, _, _)| *at == day);
            if let Some((_, ratio, _)) = planted {
                factor *= ratio;
            }
            let day_open = first_open_ms + day as i64 * 86_400_000;
            for slot in 0..BARS_PER_DAY {
                let drift = match planted {
                    Some((_, _, move_)) if slot == 0 => move_ - 1.0,
                    _ => rng.random_range(-0.004f32..0.004f32),
                };
                let open = close;
                close = (close * (1.0 + drift)).max(1.0);
                let spread = rng.random_range(0.0005f32..0.004f32) * open;
                out.push(PackedBar {
                    ts_ms: day_open + slot as i64 * RES_MS,
                    open: open * factor,
                    high: (open.max(close) + spread) * factor,
                    low: (open.min(close) - spread).max(0.25) * factor,
                    close: close * factor,
                    volume: rng.random_range(20_000.0f32..40_000.0f32),
                    vwap: 0.0,
                    trades: rng.random_range(50u32..500),
                });
            }
        }
        out
    }

    /// Two symbols: one clean, one carrying a 5:1 forward split and a 1:5 reverse split whose two
    /// sides sit on ONE price ladder, a 3:1 split with a 1% overnight move folded into the gap, and
    /// a single reverting bad print mid-session that must NOT be classified a seam at either tier.
    fn fixture(label: &str) -> (Fixture, BarCorpus, BarSupports) {
        let dir = scratch(label);
        // 2021-08-16T13:30:00Z = 09:30 EDT.
        let first_open = 1_629_120_600_000i64;
        let clean = synth_series(11, 60, first_open, &[]);
        let mut seamed = synth_series(
            12,
            60,
            first_open,
            &[(20, 5.0, 1.0), (40, 0.2, 1.0), (50, 3.0, 1.01)],
        );
        // A bad print at a fifth of the price, mid-session, that reverts on the next bar.
        let tick = 30 * BARS_PER_DAY + 13;
        let level = seamed[tick].close;
        seamed[tick].open = level / 5.0;
        seamed[tick].high = level / 5.0;
        seamed[tick].low = level / 5.0;
        seamed[tick].close = level / 5.0;
        for (symbol, bars) in [("CLEAN", &clean), ("SEAMED", &seamed)] {
            write_bar_file(
                &dir.join(format!("{symbol}.{RES}.{FILE_EXTENSION}")),
                symbol,
                RES,
                bars,
            )
            .expect("fixture writes");
        }
        let corpus = BarCorpus::load(&dir, RES, 100).expect("fixture loads");
        let samples: Vec<BarDof> = corpus
            .sample_train_dof(2_000, 7)
            .into_iter()
            .map(|(_, dof)| dof)
            .collect();
        let supports = BarSupports::fit(&samples);
        (Fixture { dir }, corpus, supports)
    }

    /// The nearest-rational lookup has to be exact on the ratios that matter and has to REJECT a
    /// number that merely happens to be near something. Both directions, because a lookup that
    /// accepts everything makes the whole criterion vacuous.
    #[test]
    fn the_nearest_rational_is_exact_on_split_ratios_and_rejects_ordinary_numbers() {
        for (ratio, label) in [
            (5.0, "5:1"),
            (0.2, "1:5"),
            (1.5, "3:2"),
            (10.0, "10:1"),
            (0.05, "1:20"),
            (0.01, "1:100"),
            (1.25, "5:4"),
        ] {
            let (rational, deviation) = nearest_rational(ratio);
            assert_eq!(rational.label(), label, "ratio {ratio}");
            assert!(
                deviation < 1e-15,
                "ratio {ratio} landed {deviation:e} from {label}"
            );
            assert!(deviation <= RATIONAL_TOLERANCE);
        }
        // A -47% move and a +73% move are extreme and are not split ratios. Both must miss by
        // more than the tolerance, or "on a rational" would be satisfied by anything.
        for ratio in [0.53, 1.73, 2.37, 0.4137, 6.31] {
            let (rational, deviation) = nearest_rational(ratio);
            assert!(
                deviation > RATIONAL_TOLERANCE,
                "ratio {ratio} was accepted as {} at {deviation:e}",
                rational.label()
            );
        }
        // Every admitted candidate is in lowest terms and is either a small `p:q` or a rung of the
        // integer ladder, so the set cannot grow dense enough to accept an arbitrary number.
        for candidate in simple_rationals() {
            assert_eq!(gcd(candidate.num, candidate.den), 1);
            assert!(candidate.num != candidate.den);
            let small = candidate.num <= RATIONAL_MAX_TERM && candidate.den <= RATIONAL_MAX_TERM;
            let ladder = candidate.num == 1 || candidate.den == 1;
            assert!(
                small || ladder,
                "{} is neither a small p:q nor an integer ladder rung",
                candidate.label()
            );
            assert!(candidate.num <= RATIONAL_LADDER_MAX && candidate.den <= RATIONAL_LADDER_MAX);
        }
        // The gap between adjacent admitted candidates below 10 must exceed the LOOSE tolerance,
        // or "near a rational" would be satisfied by every number in the range and both tiers
        // would be vacuous.
        let below_ten: Vec<f64> = simple_rationals()
            .iter()
            .map(|r| r.value())
            .filter(|v| *v <= 10.0)
            .collect();
        for pair in below_ten.windows(2) {
            let midpoint = 0.5 * (pair[0] + pair[1]);
            let (_, deviation) = nearest_rational(midpoint);
            assert!(
                deviation > RATIONAL_TOLERANCE,
                "the midpoint {midpoint} of {} and {} is within the EXACT tolerance of one of \
                 them, so the set is too dense to discriminate",
                pair[0],
                pair[1]
            );
        }
    }

    /// The four criteria have to separate a planted split from a planted bad print, and the two
    /// ratio tiers have to separate a split whose sides sit on one price ladder from one that
    /// carries a bar of market move across the gap. This is the discriminator the whole module
    /// exists to run, so the fixture plants one of each and the test asserts all three
    /// classifications.
    #[test]
    fn a_planted_split_is_classified_and_a_reverting_bad_print_is_not() {
        let (_fx, corpus, supports) = fixture("classify");
        let census = census_corpus(&corpus, &supports);
        let events = census.worst_events();
        assert_eq!(
            events.len(),
            census.extremes as usize,
            "the fixture is small enough that every extreme event is retained"
        );

        let exact: Vec<&ExtremeEvent> = events.iter().filter(|e| e.is_seam()).collect();
        assert_eq!(
            exact.len(),
            2,
            "exactly the two ladder-aligned splits classify at the EXACT tier, got {:?}",
            events
                .iter()
                .map(|e| (e.bar, e.r, e.ratio.label(), e.ratio_deviation, e.is_seam()))
                .collect::<Vec<_>>()
        );
        for seam in &exact {
            assert!(seam.on_rational(), "a ladder-aligned split sits ON its ratio");
            assert!(seam.near_rational(), "the exact test implies the loose one");
            assert!(seam.session_open, "a planted split is at the open");
            assert!(seam.quiet, "a planted split does not trade its move");
            assert!(seam.isolated, "a planted split does not revert");
            assert!(!seam.reverts);
            assert!(
                seam.ratio_deviation <= RATIONAL_TOLERANCE,
                "{}",
                seam.ratio_deviation
            );
        }
        let mut ratios: Vec<String> = exact.iter().map(|s| s.ratio.label()).collect();
        ratios.sort();
        assert_eq!(ratios, vec!["1:5".to_owned(), "5:1".to_owned()]);

        // The 3:1 with a 1% overnight move is a seam that the EXACT test cannot see and the LOOSE
        // one must. Without this the loose tier would be decoration rather than an upper bound.
        let near: Vec<&ExtremeEvent> = events.iter().filter(|e| e.is_near_seam()).collect();
        assert_eq!(
            near.len(),
            3,
            "the drifting 3:1 split must classify at the LOOSE tier only, got {:?}",
            events
                .iter()
                .map(|e| (e.bar, e.ratio.label(), e.ratio_deviation, e.is_near_seam()))
                .collect::<Vec<_>>()
        );
        let drifting: Vec<&&ExtremeEvent> = near.iter().filter(|e| !e.is_seam()).collect();
        assert_eq!(drifting.len(), 1, "one of the three is loose-only");
        assert_eq!(drifting[0].ratio.label(), "3:1");
        assert!(
            drifting[0].ratio_deviation > RATIONAL_TOLERANCE
                && drifting[0].ratio_deviation <= RATIONAL_NEAR_TOLERANCE,
            "a split times one bar of market move must land BETWEEN the two tolerances, got {}",
            drifting[0].ratio_deviation
        );

        // The bad print prints BOTH signs of ln 5 and must be classified on neither bar, at
        // neither tier: it is the population the isolation criterion exists to reject.
        let ticks: Vec<&ExtremeEvent> = events.iter().filter(|e| e.reverts).collect();
        assert!(
            !ticks.is_empty(),
            "the fixture plants a reverting print; the census must see it"
        );
        for tick in &ticks {
            assert!(
                !tick.is_seam() && !tick.is_near_seam(),
                "a reverting bad print at bar {} was classified a seam",
                tick.bar
            );
            assert!(!tick.isolated, "a reverting print has an extreme neighbour");
        }
        assert!(
            census.reverts > 0
                && census.seams[TIER_EXACT] == 2
                && census.seams[TIER_NEAR] == 3,
            "reverts {} seams {:?}",
            census.reverts,
            census.seams
        );
        // The clean symbol must contribute nothing, at either tier.
        assert_eq!(
            census.seam_series,
            [1, 1],
            "only the seeded symbol carries seams"
        );
    }

    /// The census must count every bar of every series exactly once, and its exceedance ladder
    /// must be monotone. A fold-and-reduce accumulator that double-counts on merge is invisible in
    /// the ratios and fatal to every share this module reports.
    #[test]
    fn the_census_counts_every_bar_once_with_a_monotone_exceedance_ladder() {
        let (_fx, corpus, supports) = fixture("counts");
        let census = census_corpus(&corpus, &supports);
        let expected: u64 = (0..corpus.series_count())
            .map(|s| corpus.series_len(s) as u64 - 1)
            .sum();
        assert_eq!(census.bars, expected, "every DOF-carrying bar exactly once");
        assert!(census.train_bars <= census.bars);
        for window in census.level_counts.windows(2) {
            assert!(
                window[0] >= window[1],
                "exceedance counts must be monotone, got {:?}",
                census.level_counts
            );
        }
        assert_eq!(
            census.extremes, census.level_counts[0],
            "the census threshold is the first exceedance level"
        );
        // Every criterion count is a subset of the extremes, the conjunction is a subset of each,
        // and the EXACT tier is a subset of the LOOSE one - which is what makes the pair a bracket
        // rather than two unrelated numbers.
        for count in census.criterion_counts() {
            assert!(count <= census.extremes);
        }
        for tier in 0..TIERS {
            assert!(
                census.seams[tier] <= census.on_rational[tier].min(census.isolated),
                "tier {tier} conjunction exceeds one of its conjuncts"
            );
            assert_eq!(census.seam_keys[tier].len() as u64, census.seams[tier]);
            assert_eq!(census.seam_keys_dropped[tier], 0);
        }
        assert!(
            census.on_rational[TIER_EXACT] <= census.on_rational[TIER_NEAR]
                && census.seams[TIER_EXACT] <= census.seams[TIER_NEAR],
            "the exact tier must be a subset of the loose one: {:?} / {:?}",
            census.on_rational,
            census.seams
        );
    }

    /// The clip-quantile rule this module reproduces must recover the LIVE artifact's own outer
    /// edges from its own draw, and the ruin licence read off those edges must reproduce the
    /// numbers in force. Without this the cleaned counterfactual beside it means nothing.
    #[test]
    fn the_ruin_licence_matches_the_live_support_edges() {
        // The two live figures, from `long_data/bars/bar_supports.300.json`.
        let live_lo = -0.088_331_513_106_822_97f64;
        let live_hi = 0.088_038_101_792_335_51f64;
        let long = 1.0 / (1.0 - live_lo.exp());
        let short = 1.0 / (live_hi.exp() - 1.0);
        assert!(
            (long - 11.8283).abs() < 1e-3,
            "long licence moved to {long}"
        );
        assert!(
            (short - 10.8661).abs() < 1e-3,
            "short licence moved to {short}"
        );
        assert!(short < long, "the short side always binds first");

        // And the same expressions, through the type the report reads them off.
        let reading = TailReading {
            rows: 1,
            continuous_rows: 1,
            thresholds: Vec::new(),
            slopes: Vec::new(),
            min_r: live_lo,
            max_r: live_hi,
            clip_lo: live_lo,
            clip_hi: live_hi,
        };
        assert!((reading.long_max_leverage() - long).abs() < 1e-12);
        assert!((reading.short_max_leverage() - short).abs() < 1e-12);
        assert_eq!(reading.binding_max_leverage(), reading.short_max_leverage());

        // The clip rule itself, against a sample whose quantiles are known by construction: 20,001
        // rows means the 1e-4 quantile is index round(1e-4 * 20000) = 2.
        let rows: Vec<BarDof> = (0..20_001)
            .map(|i| BarDof {
                r: i as f32,
                ..BarDof::default()
            })
            .collect();
        let (lo, hi) = clipped_r_range(&rows);
        assert_eq!(lo, 2.0);
        assert_eq!(hi, 19_998.0);
    }

    /// Removing the seam rows must move the tail reading in the direction a contaminated tail
    /// moves, and the control must be the reading on the untouched draw. Both are read with the
    /// SAME estimator `bar_family` uses, which is what makes them comparable to the live figure.
    #[test]
    fn the_cleaned_tail_drops_the_planted_seams_from_the_draw() {
        let (_fx, corpus, supports) = fixture("tail");
        let audit = build_audit(&corpus, &supports, 2_000, 7, "fixture");
        assert_eq!(audit.draw_rows, 2_000);
        assert_eq!(audit.control.rows, 2_000);
        assert!(audit.control.slopes.len() == 6, "six pairwise slopes");
        for tier in 0..TIERS {
            let cleaned = &audit.cleaned[tier];
            assert_eq!(
                cleaned.rows + audit.draw_rows_removed[tier],
                audit.control.rows,
                "every removed row must leave tier {tier}'s cleaned reading"
            );
            assert_eq!(cleaned.slopes.len(), 6);
            // The planted splits are the draw's extremes wherever they were drawn, so removing
            // them can only pull the outer clip quantiles in.
            assert!(audit.control.clip_lo <= cleaned.clip_lo + 1e-12);
            assert!(audit.control.clip_hi >= cleaned.clip_hi - 1e-12);
        }
        assert!(
            audit.draw_rows_removed[TIER_EXACT] <= audit.draw_rows_removed[TIER_NEAR],
            "the loose tier removes a superset of the exact tier's rows: {:?}",
            audit.draw_rows_removed
        );
        assert!(audit.control.max_r.is_finite() && audit.control.min_r.is_finite());
        assert!(audit.support_long_max_leverage() > 1.0);
        assert!(audit.support_short_max_leverage() > 1.0);
        // The named draw extremes must BE the control reading's own min and max, or the row the
        // ruin table is argued from is not the row the report names.
        assert_eq!(audit.draw_extremes[0].r, audit.control.min_r);
        assert_eq!(audit.draw_extremes[1].r, audit.control.max_r);
        for extreme in &audit.draw_extremes {
            assert!(
                audit.series_names.contains(&extreme.symbol),
                "the named extreme {} must be one of the corpus's own symbols",
                extreme.symbol
            );
        }
        assert!(!audit.report_lines().is_empty());
    }

    /// A seam that lands in a catch-all bin has to be counted there, and the verdict has to follow
    /// the counts rather than a hardcoded string.
    #[test]
    fn the_verdict_follows_the_catch_all_counts() {
        let (_fx, corpus, supports) = fixture("verdict");
        let audit = build_audit(&corpus, &supports, 2_000, 7, "fixture");
        let bins = NUM_BAR_BINS as usize;
        // A planted 3x or 5x seam is far outside any equal-mass support fitted on ordinary bars, so
        // it must land in a catch-all bin, and that is exactly the contamination being quantified.
        for tier in 0..TIERS {
            assert_eq!(
                audit.census.catch_all_seams[tier][0] + audit.census.catch_all_seams[tier][1],
                audit.census.seams[tier],
                "every planted seam must reach a catch-all bin, got {:?} of {} at tier {tier}",
                audit.census.catch_all_seams[tier],
                audit.census.seams[tier]
            );
            for side in 0..2 {
                assert!(audit.catch_all_seam_share(tier)[side] <= 1.0);
                assert!(audit.catch_all_seam_share_train(tier)[side] <= 1.0);
            }
        }
        assert!(
            audit.contaminated(),
            "the fixture is contaminated by construction"
        );
        assert!(audit
            .report_lines()
            .iter()
            .any(|line| line.contains("CONTAMINATED")));
        assert_eq!(supports.lower_bounds(DOF_R).len(), bins);
    }

    /// Every base this module writes must be registered, must land on disk and must carry a finite
    /// value. The registry side of the two-sided contract; the exemption in
    /// `pretrain_reports::tests::CYCLE_EXEMPT` names THIS test as the executor.
    #[test]
    fn the_seam_audit_writes_every_registered_base() {
        let (_fx, corpus, supports) = fixture("bases");
        let audit = build_audit(&corpus, &supports, 2_000, 7, "fixture");
        let dir = scratch("charts");
        write_bar_seams(&dir, &audit).expect("every chart writes");
        for base in BAR_SEAM_BASES {
            assert!(
                shared::report::PRETRAIN_REPORT_BASES.contains(base),
                "{base} must be registered in shared::report::PRETRAIN_REPORT_BASES or the TUI \
                 never scans for it"
            );
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was not written");
            let read = read_report(&path).expect("the report reads back");
            let ReportKind::MultiLine { series } = &read.kind else {
                panic!("{base} must be a MultiLine chart");
            };
            assert!(
                series.iter().any(|s| s.values.iter().any(|v| v.is_finite())),
                "{base} carries no finite value, so it is a blank panel"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
        // `encode_series` is the whole-series encoder the streaming accessor mirrors; naming it
        // here keeps the import honest about which definition of `r` this module measures.
        let bars = corpus.bars(0);
        let encoded = encode_series(bars);
        assert_eq!(encoded.len(), bars.len() - 1);
    }

    /// The retained-worst buffer must obey its cap through a merge, or a reduce tree over
    /// thousands of series grows one buffer per series and the pass is no longer bounded.
    #[test]
    fn the_retained_event_buffer_obeys_its_cap_through_a_merge() {
        let event = |series: u32, bar: u32, r: f64| ExtremeEvent {
            series,
            bar,
            ts_ms: 0,
            r,
            prev_close: 1.0,
            close: r.exp() as f32,
            s: 0.0,
            w: 0.0,
            volume: 1.0,
            ratio: Rational { num: 5, den: 1 },
            ratio_deviation: 0.0,
            session_open: true,
            quiet: true,
            isolated: true,
            reverts: false,
            bin: 0,
            in_train: true,
        };
        let build = |offset: u32| {
            let mut census = Census::new();
            for i in 0..(EVENT_BUFFER as u32 + 64) {
                census.absorb(event(offset, i, 1.0 + f64::from(i) * 1e-6));
            }
            census
        };
        let left = build(0);
        let right = build(1);
        assert_eq!(left.event_store.len(), EVENT_BUFFER);
        assert_eq!(left.extremes as usize, EVENT_BUFFER + 64);
        let merged = left.merge(right);
        assert_eq!(
            merged.event_store.len(),
            EVENT_BUFFER,
            "the merged accumulator must obey the same bound its parts did"
        );
        assert_eq!(merged.extremes as usize, 2 * (EVENT_BUFFER + 64));
        assert_eq!(merged.events.len(), merged.event_store.len());
        // The retained sample must be the WORST events, not an arbitrary subset.
        let worst = merged.worst_events();
        assert!(worst
            .windows(2)
            .all(|pair| pair[0].r.abs() >= pair[1].r.abs()));
        let smallest_kept = worst.last().expect("buffer is full").r.abs();
        assert!(
            smallest_kept > 1.0 + 63.0 * 1e-6,
            "the buffer kept a value the cap should have evicted: {smallest_kept}"
        );
    }
}
