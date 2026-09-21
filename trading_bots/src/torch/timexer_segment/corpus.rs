use std::{collections::BTreeSet, fs, path::Path, time::Instant};

use anyhow::{ensure, Context, Result};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::{
    bars::{bar_file_path, parse_bar_file_name, BarFile, PackedBar},
    report::CandleBar,
};
use tch::{Device, Kind, Tensor};

use super::{
    cache,
    data::{audit_bars, filtered_contract, retained_partition_end, valid_ohlc, DataContract},
    features::{
        cross_section_ranks, market_steps, single_series, AuxiliaryCursor, Exogenous, Feature,
        FeatureSet, Grid, MarketSteps, MarketSummary, SPY,
    },
};
use crate::torch::hashing::file_sha256;

pub(super) const RESOLUTION_MS: i64 = 300_000;

/// The value [`CorpusContract::cross_section_placement`] carries once the anchored placement has
/// replaced the strided one. Empty means strided, which is what every manifest on disk says by
/// omitting the field entirely.
pub(super) const ANCHORED_PLACEMENT: &str = "anchored-shared-wall-clocks";
const SCHEMA: &str = "timexer-pooled-mmap-v5;independent-ticker-rows;all-valid-source-unique-utc-grid-quantiles70:10:10:10;purge-only-at-observed-partition-boundaries;centered-log-prices-with-bar-validity;covariates-over-context-and-horizon;next-valid-observed-bars;invalid-ohlc-rows-quarantined-without-repair;disjoint-target-epoch-with-masked-remainder;validation-disjoint-complete-targets;terminal-test-locked;exogenous-variates-on-shared-utc-grid;market-cumulative-log-return-over-steps-defined-by-min-cross-section-slots-spanning-sparse-slots-centered-at-origin";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRef {
    pub ticker: usize,
    pub origin: usize,
}

/// A prefix mask, not an inferred calendar duration: horizons count valid observed bars.
#[derive(Clone, Debug, Serialize)]
pub struct ResearchOrigin {
    pub ticker: String,
    pub origin: usize,
    pub origin_ms: i64,
    pub common_source_ms: i64,
    pub context_first_ms: i64,
    pub target_first_ms: i64,
    pub target_last_ms: i64,
    pub valid_target_prefix: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResearchCoverage {
    pub ticker: String,
    pub population_origins: usize,
    pub selected_origins: usize,
    pub chronology_excluded_origins: usize,
    pub population_origin_range_ms: Option<[i64; 2]>,
    pub selected_origin_range_ms: Option<[i64; 2]>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResearchDraw {
    pub partition: String,
    pub requested_rows: usize,
    pub population_origins: usize,
    pub origins_sha256: String,
    pub target_mask_sha256: String,
    pub coverage: Vec<ResearchCoverage>,
    pub origins: Vec<ResearchOrigin>,
}

/// Ordinal endpoint certificates cover EVERY possible dense training sub-origin: all of its
/// unmasked targets are no later than `train_end - 1`. Earlier held-out context is legal;
/// only scored targets, not context or origin timestamps, belong to the reserved UTC bands.
#[derive(Clone, Debug, Serialize)]
pub struct SplitChronology {
    pub ticker: String,
    pub boundary_ordinals: [usize; 3],
    pub last_before_boundary_ms: [Option<i64>; 3],
    pub first_at_or_after_boundary_ms: [Option<i64>; 3],
    pub dense_training_target_envelope_ms: [i64; 2],
    pub calibration_target_envelope_ms: Option<[i64; 2]>,
    pub validation_target_envelope_ms: Option<[i64; 2]>,
    pub terminal_test_first_ms: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResearchSampleManifest {
    pub schema: String,
    pub seed: u64,
    pub selection: String,
    pub corpus_sha256: String,
    pub training_origins_sha256: String,
    pub training_population_origins: usize,
    pub boundary_timestamps_ms: [i64; 3],
    pub context: usize,
    pub pred_len: usize,
    pub source_lookback_bars: usize,
    pub training_last_target_ms: i64,
    pub validation_chronology_excluded_origins: usize,
    pub target_mask_encoding: String,
    pub chronology: Vec<SplitChronology>,
    pub validation: ResearchDraw,
    pub probe_fit: ResearchDraw,
}

pub struct ResearchSamplePlan {
    pub validation_refs: Vec<WindowRef>,
    /// Full validation population after the SAME global decision-time guard; a separate
    /// bounded synchronized cross-section draw may be selected from this without changing
    /// the ticker-balanced routine panel or admitting an earlier decision clock.
    pub validation_population_refs: Vec<WindowRef>,
    pub probe_fit_refs: Vec<WindowRef>,
    pub manifest: ResearchSampleManifest,
}

/// Trained rows that must BRACKET the in-period hole on each side for the hole's period to be
/// "in-sample regime" rather than an extrapolation. 64 rows is `64 · pred_len` = 12,288 bars of
/// supervised history immediately before the hole and 12,288 immediately after it, roughly 158
/// trading days on each side at 78 five-minute bars per session.
pub(super) const IN_PERIOD_BRACKET_ROWS: usize = 64;

/// Tickers that saturate ONE in-period cross-section: the trading draw's own per-timestamp cap,
/// referenced rather than repeated, because a hole whose blocks are narrower than the cap
/// silently costs the in-period draw the precision the comparison is read on.
const IN_PERIOD_SATURATING_TICKERS: usize = super::runner::CROSS_SECTION_TICKERS;

/// Tickers an in-period anchor's admissible window must cover for the anchor to be a candidate
/// at all: four times the per-timestamp cap, so that the block is saturated even after the
/// tickers that simply did not trade at that moment drop out.
const IN_PERIOD_MIN_UNIVERSE: usize = 4 * IN_PERIOD_SATURATING_TICKERS;

/// Anchor placements to try before giving up. Coverage is flat over a long plateau of candidate
/// wall clocks, so the first is almost always accepted; the retry exists because an anchor's
/// TIME OF DAY is not something the coverage sweep controls, and an anchor that lands outside
/// regular hours is held by too few tickers to form a cross-section.
const IN_PERIOD_ANCHOR_ATTEMPTS: usize = 8;

/// One ticker's placement of the shared anchors: the hole as an inclusive TARGET BAR range and,
/// per section, the origin ordinal when this ticker holds a bar EXACTLY at that anchor.
///
/// `None` per section rather than a compacted list, because the census a placement feeds - how
/// many tickers a given anchor holds - is a per-ANCHOR quantity, and it is the quantity that
/// decides whether the draw can form a cross-section at all.
struct Placement {
    hole: (usize, usize),
    origins: Vec<Option<usize>>,
}

/// The in-period hole: an interval of TARGET BARS inside the training span that no surviving
/// training row can supervise, and out of which the in-period held-out origins are cut.
///
/// It exists because a chronological split conflates two different things - an origin the
/// sampler never visited, and a market period the model never saw - and those have opposite
/// fixes. The hole holds the period fixed and varies only origin identity.
///
/// Anchored on shared WALL CLOCKS, one per section, and NOT on an ordinal offset from
/// `boundaries[0]`. That distinction is the whole construction and it is what job 5460 died
/// on: `boundaries[0]` is one shared timestamp, but `boundaries[0] - k` is not, because two
/// tickers with different bar densities over the offset land at different moments. MEASURED on
/// the 4,873-ticker corpus, the ordinal rule spread the first in-period origin over 1,914 days,
/// its widest timestamp held 26 tickers against the 40 a cross-section needs, and the run
/// refused at `cross_section_blocks`. The anchored rule holds 32 timestamps of ~2,950 tickers.
///
/// [`Corpus::validation_refs`] gets this for free: it strides FORWARD from `boundaries[1]`, so
/// its first origin is exactly the shared boundary timestamp on every ticker and carries the
/// widest block in the corpus. The hole cannot borrow that anchor - it must sit inside the
/// training span - so it derives its own.
///
/// Placed as LATE in the training span as coverage allows, because the closer the hole's regime
/// is to the validation period the less a regime difference can masquerade as an
/// origin-identity difference.
///
/// Spacing comes from the REFERENCE CLOCK: the admissible ticker with the most valid bars, ties
/// broken by symbol. On this corpus that selects `SPY`, so consecutive anchors are `pred_len`
/// SPY bars apart - the market's own clock rather than a calendar constant that would have to
/// guess how many bars a session holds.
///
/// `None` when `sections` is 0. A ticker whose history cannot carry the anchor span plus both
/// brackets gets no placement: it keeps every training row and contributes no in-period origin,
/// which is a population fact to report, not an error.
fn in_period_plan(
    tickers: &[CorpusTicker],
    sections: usize,
    pool: &rayon::ThreadPool,
) -> Result<Option<(Vec<i64>, Vec<usize>, Vec<Option<Placement>>)>> {
    if sections == 0 {
        return Ok(None);
    }
    let started = Instant::now();
    let c = &tickers[0].contract;
    let bracket = IN_PERIOD_BRACKET_ROWS * c.pred_len;
    // `context - 2` is the model-configuration-INDEPENDENT backward reach of a row's dense
    // supervision; see [`CorpusTicker::supervision_clears_hole`].
    let above = c.context.saturating_sub(2) + bracket;
    let span = sections * c.pred_len;
    let first = c.common_context + bracket;
    // The admissible wall-clock window for the FIRST anchor, per ticker. Both endpoints are
    // real bar timestamps, which is what makes them usable as candidate anchors.
    let windows: Vec<(i64, i64)> = tickers
        .iter()
        .filter_map(|ticker| {
            let c = &ticker.contract;
            let last = c.boundaries[0].checked_sub(c.purge + 1 + above + span)?;
            (first <= last && last < c.valid_bars)
                .then(|| (ticker.timestamp(first), ticker.timestamp(last)))
        })
        .collect();
    ensure!(
        !windows.is_empty(),
        "no ticker's history can carry a {sections}-section in-period hole plus its \
         {IN_PERIOD_BRACKET_ROWS} bracket rows on each side: the hole needs \
         {} bars of training history before the {} bar purge band at the 70% boundary",
        first + span + above,
        c.purge
    );
    let covered = |stamp: i64| {
        windows
            .iter()
            .filter(|(lo, hi)| *lo <= stamp && stamp <= *hi)
            .count()
    };
    let mut candidates: Vec<i64> = windows.iter().flat_map(|(lo, hi)| [*lo, *hi]).collect();
    candidates.sort_unstable();
    candidates.dedup();
    let reach = candidates
        .iter()
        .map(|stamp| covered(*stamp))
        .max()
        .unwrap_or(0);
    // A corpus smaller than the universe floor is a fixture, not a failure: ask it for
    // everything it has rather than refusing to place a hole at all.
    let floor = IN_PERIOD_MIN_UNIVERSE.min(reach);
    let mut attempts: Vec<(
        i64,
        Vec<i64>,
        Vec<usize>,
        Vec<Option<Placement>>,
        String,
        usize,
    )> = Vec::new();
    for (attempt, stamp) in candidates
        .iter()
        .rev()
        .filter(|stamp| covered(**stamp) >= floor)
        .take(IN_PERIOD_ANCHOR_ATTEMPTS)
        .enumerate()
    {
        // The reference clock spaces the anchors. Any ticker dense enough to cover the span
        // would do; the densest one is picked so that the anchors land on the moments the most
        // liquid names all trade at, which is what makes the blocks wide.
        let Some(reference) = tickers
            .iter()
            .filter(|ticker| {
                ticker
                    .valid_ordinal_at_or_before(*stamp)
                    .is_some_and(|at| at + span < ticker.contract.valid_bars)
            })
            .max_by(|a, b| {
                a.contract
                    .valid_bars
                    .cmp(&b.contract.valid_bars)
                    .then_with(|| b.contract.ticker.cmp(&a.contract.ticker))
            })
        else {
            continue;
        };
        let at = reference
            .valid_ordinal_at_or_before(*stamp)
            .expect("the reference clock was selected by holding this anchor");
        let anchors: Vec<i64> = (0..sections)
            .map(|section| reference.timestamp(at + section * c.pred_len))
            .collect();
        let placements: Vec<Option<Placement>> = pool.install(|| {
            tickers
                .par_iter()
                .map(|ticker| ticker.place(&anchors, above, bracket))
                .collect()
        });
        let census: Vec<usize> = (0..sections)
            .map(|section| {
                placements
                    .iter()
                    .flatten()
                    .filter(|placement| placement.origins[section].is_some())
                    .count()
            })
            .collect();
        let usable = saturated(&census, &placements);
        attempts.push((
            *stamp,
            anchors,
            census,
            placements,
            reference.contract.ticker.clone(),
            attempt + 1,
        ));
        if usable >= sections.div_ceil(4) {
            break;
        }
    }
    // Deliberately the WIDEST attempt rather than the first that cleared the bar, so a corpus
    // where every candidate is thin still gets its best placement and a census to explain it.
    let Some((stamp, anchors, census, placements, clock, tried)) = attempts
        .into_iter()
        .max_by_key(|(stamp, _, census, placements, _, _)| (saturated(census, placements), *stamp))
    else {
        anyhow::bail!(
            "no in-period anchor is held by the {floor} tickers a cross-section needs; the \
             corpus offers at most {reach}"
        )
    };
    let usable = saturated(&census, &placements);
    ensure!(
        usable >= sections.div_ceil(4),
        "the in-period hole anchored at {stamp} yields only {usable} usable cross-sections of \
         the {} it owes: per-anchor ticker counts are {census:?} against the {} a saturated \
         block needs. An anchor is a WALL CLOCK, so this is a time-of-day failure, not a \
         population one - re-run with a different --in-period-sections, which moves the anchor",
        sections.div_ceil(4),
        IN_PERIOD_SATURATING_TICKERS.min(placements.iter().flatten().count()),
    );
    println!(
        "CausalPatch in-period hole: {sections} anchors on the {} clock from {} to {}, held by \
         {} of {} tickers, {usable} saturated cross-sections, per-anchor widths {census:?}, \
         placed at attempt {tried} of {IN_PERIOD_ANCHOR_ATTEMPTS} in {:.1} ms",
        clock,
        anchors.first().copied().unwrap_or_default(),
        anchors.last().copied().unwrap_or_default(),
        placements.iter().flatten().count(),
        tickers.len(),
        started.elapsed().as_secs_f64() * 1000.,
    );
    Ok(Some((anchors, census, placements)))
}

/// Anchors whose block is at least as wide as the draw can consume. Saturated blocks are the
/// only ones that carry the in-period draw's precision, so they - not the anchor count - are
/// what the placement is scored on.
fn saturated(census: &[usize], placements: &[Option<Placement>]) -> usize {
    let cap = IN_PERIOD_SATURATING_TICKERS.min(placements.iter().flatten().count());
    census.iter().filter(|width| **width >= cap.max(1)).count()
}

/// The ANCHORED replacement for the strided `band` placement of a reserved partition:
/// `first = 0` is calibration `[70%, 80%)`, `first = 1` validation `[80%, 90%)`.
///
/// The strided rule puts origin `i` at `boundaries[first] - 1 + i·pred_len` on each ticker's
/// OWN valid-bar ordinal. Only `i = 0` is a shared moment; every later stride drifts apart by
/// each ticker's own bar density. MEASURED on the real corpus: 433,303 validation origins land
/// on 40,837 distinct timestamps of which exactly 100 hold the `CROSS_SECTION_FLOOR` names a
/// cross-section needs, so **99.75% of the population forms no usable cross-section**, and the
/// one aligned stride `i = 0` alone holds 3,692 tickers.
///
/// This rule keeps the partition's own first bar as the anchor - `boundaries[first]` IS a shared
/// timestamp - and then walks the REFERENCE CLOCK forward `pred_len` bars at a time, taking each
/// ticker's bar at that exact moment or nothing. Every origin is therefore on one of a few
/// hundred shared wall clocks instead of one of 40,837 near-unique ones.
///
/// Anchors whose block is thinner than `floor` are dropped - the width the draw can consume is
/// a property of the DRAW, so the caller owns it (`CROSS_SECTION_FLOOR` in production). That is not tidiness:
/// eligible labels THIN near a purge boundary because a sparse ticker's `pred_len` forward bars
/// run past `retained_partition_end`, and the portfolio sibling measured exactly this on a
/// post-training window (widths min 11, mean 153.4, max 256). A block the draw cannot use is
/// population that only makes the census harder to read.
pub(super) fn anchored_partition_refs(
    tickers: &[CorpusTicker],
    first: usize,
    boundary_stamp: i64,
    floor: usize,
    pool: &rayon::ThreadPool,
) -> Result<(Vec<i64>, Vec<usize>, Vec<Vec<usize>>)> {
    let pred_len = tickers[0].contract.pred_len;
    let reference = tickers
        .iter()
        .filter(|ticker| {
            ticker
                .valid_ordinal_at_or_before(boundary_stamp)
                .is_some_and(|at| ticker.owns_partition_targets(at, first))
        })
        .max_by(|a, b| {
            a.contract
                .valid_bars
                .cmp(&b.contract.valid_bars)
                .then_with(|| b.contract.ticker.cmp(&a.contract.ticker))
        })
        .with_context(|| {
            format!(
                "no ticker owns partition {first} targets at its own first bar {boundary_stamp}"
            )
        })?;
    let at = reference
        .valid_ordinal_at_or_before(boundary_stamp)
        .expect("the reference clock was selected by holding this boundary");
    let anchors: Vec<i64> = (0..)
        .map(|section| at + section * pred_len)
        .take_while(|origin| reference.owns_partition_targets(*origin, first))
        .map(|origin| reference.timestamp(origin))
        .collect();
    ensure!(
        !anchors.is_empty(),
        "partition {first} carries no anchor on the {} clock",
        reference.contract.ticker
    );
    // One binary search then a forward walk over the partition's own bars, per ticker: the
    // page-friendly pattern, where an independent search per anchor would be a random fault
    // per anchor per ticker into a 28 GB corpus.
    let placed: Vec<Vec<Option<usize>>> = pool.install(|| {
        tickers
            .par_iter()
            .map(|ticker| {
                let mut cursor = ticker.valid_ordinal_at_or_before(anchors[0]);
                anchors
                    .iter()
                    .map(|stamp| {
                        let mut at = cursor?;
                        while at + 1 < ticker.contract.valid_bars
                            && ticker.timestamp(at + 1) <= *stamp
                        {
                            at += 1;
                        }
                        cursor = Some(at);
                        (ticker.timestamp(at) == *stamp && ticker.owns_partition_targets(at, first))
                            .then_some(at)
                    })
                    .collect()
            })
            .collect()
    });
    let kept: Vec<usize> = (0..anchors.len())
        .filter(|section| {
            placed
                .iter()
                .filter(|origins| origins[*section].is_some())
                .count()
                >= floor
        })
        .collect();
    ensure!(
        !kept.is_empty(),
        "no anchor of partition {first} holds the {floor} tickers a cross-section needs"
    );
    let census: Vec<usize> = kept
        .iter()
        .map(|section| {
            placed
                .iter()
                .filter(|origins| origins[*section].is_some())
                .count()
        })
        .collect();
    let origins: Vec<Vec<usize>> = placed
        .iter()
        .map(|per_ticker| {
            kept.iter()
                .filter_map(|section| per_ticker[*section])
                .collect()
        })
        .collect();
    Ok((
        kept.iter().map(|section| anchors[*section]).collect(),
        census,
        origins,
    ))
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExcludedTicker {
    pub ticker: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CorpusContract {
    pub schema: String,
    pub tickers: Vec<DataContract>,
    pub boundary_timestamps: [i64; 3],
    pub context: usize,
    pub pred_len: usize,
    pub common_context: usize,
    pub purge: usize,
    pub features: FeatureSet,
    pub auxiliary_schema: String,
    pub spy_fingerprint: Option<String>,
    /// SHA-256 over the ordered universe fingerprints, partition boundaries, and the market step
    /// construction (threshold included) that define the cumulative market path every row's
    /// targets are demeaned by.
    pub market_fingerprint: String,
    /// Sources that must hold a valid bar at a grid slot for it to define a market step.
    pub market_min_cross_section: usize,
    pub minimum_source_bars: usize,
    pub minimum_training_bars: usize,
    pub train_target_bars: usize,
    pub validation_target_bars: usize,
    pub validation_remainder_bars: usize,
    /// In-period held-out geometry, every field at its zero value and SKIPPED by serialization
    /// when the feature is off, so a control run's manifest digest is byte-identical to one
    /// built before the feature existed. When it is on, these are the authenticated record that
    /// this run trained on a HOLED population: `train_target_bars` above is already net of the
    /// purge, so a holed arm's training NLL is not step-comparable to a control's.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub in_period_sections: usize,
    /// The shared WALL CLOCKS the hole was placed on, one per section, ascending. The hole's
    /// identity is these timestamps and nothing else: two runs that agree here held out the
    /// same market moments whatever each ticker's own ordinals were.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub in_period_anchors: Vec<i64>,
    /// Tickers holding a bar at each anchor, in anchor order: the width of the cross-section
    /// the in-period draw can build there, published because a thin census is the difference
    /// between a measurable comparison and a refusal at draw time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub in_period_census: Vec<usize>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub in_period_origins: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub in_period_target_bars: usize,
    /// Training rows removed so that no surviving row can supervise an in-period target bar.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub in_period_purged_rows: usize,
    /// Target bars those removed rows owned, i.e. the training population the hole cost.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub in_period_purged_target_bars: usize,
    /// How `calibration_refs` and `validation_refs` were PLACED. `"strided"` (the default, and
    /// skipped by serialization so every existing manifest digests unchanged) is the historical
    /// rule `boundaries[first] - 1 + i·pred_len` on each ticker's own ordinals;
    /// `"anchored"` is [`Corpus::anchor_cross_section_draws`]. The two populations are NOT
    /// comparable: they share a cardinality class and almost no timestamp, so any statistic
    /// pooled across a cross-section - IC, `martingale_ratio`, the trading family - means a
    /// different thing under each.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cross_section_placement: String,
    /// The shared wall clocks the anchored placement used, per partition, ascending.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calibration_anchors: Vec<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation_anchors: Vec<i64>,
    pub excluded_tickers: Vec<ExcludedTicker>,
}

pub struct CorpusTicker {
    pub contract: DataContract,
    file: BarFile,
    /// Inclusive first and last TARGET BAR of this ticker's in-period hole; see
    /// [`in_period_plan`].
    hole: Option<(usize, usize)>,
    /// This ticker's in-period held-out origins: the anchor ordinals it actually holds a bar
    /// at, ascending. Stored rather than re-derived from `hole` by a stride, because the
    /// anchors are shared WALL CLOCKS and a stride over ordinals is exactly the construction
    /// that failed.
    in_period: Vec<usize>,
    /// [`cross_section_ranks`] for this ticker, one entry per VALID bar; empty when
    /// `Feature::CrossSectionRank` is off.
    ranks: Vec<u16>,
}

impl CorpusTicker {
    pub fn timestamp(&self, origin: usize) -> i64 {
        self.bar(origin).ts()
    }

    pub fn candle_window(
        &self,
        origin: usize,
        history: usize,
        future: usize,
    ) -> Result<Vec<CandleBar>> {
        let start = origin
            .checked_sub(history)
            .context("candle history precedes corpus")?;
        let end = origin
            .checked_add(future)
            .and_then(|i| i.checked_add(1))
            .context("candle window overflow")?;
        ensure!(
            end <= self.contract.valid_bars,
            "candle window exceeds valid observed corpus"
        );
        Ok(ValidBars::new(
            self.file.bars(),
            &self.contract.invalid_ohlc_indices,
            start,
            end - start,
        )
        .map(|bar| CandleBar {
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
        })
        .collect())
    }

    pub fn bar(&self, logical_index: usize) -> &PackedBar {
        assert!(
            logical_index < self.contract.valid_bars,
            "valid observed bar exceeds corpus"
        );
        &self.file.bars()[raw_index(logical_index, &self.contract.invalid_ohlc_indices)]
    }

    /// Whether `origin` owns a complete `pred_len` of targets inside the held-out partition
    /// running from `boundaries[first]` to `boundaries[first + 1]`.
    ///
    /// ONE rule, TWO partitions: `first = 0` is the calibration partition `[70%, 80%)` and
    /// `first = 1` the validation partition `[80%, 90%)`. Targets start at the partition's own
    /// first bar and stop `purge` bars short of the next boundary, so no window's targets
    /// cross a partition edge, and the last bar any CALIBRATION target reads is at least
    /// `purge` (>= 100) bars before the first VALIDATION origin. That is what makes an
    /// amplitude curve's fit block and its scoring block disjoint by corpus construction
    /// rather than by a caller's own split arithmetic.
    ///
    /// This used to exist for `first = 1` alone. The calibration partition the schema has
    /// declared since v5 was admitted by nothing and enumerated by nothing, which is the
    /// whole reason it yielded zero origins.
    pub(super) fn owns_partition_targets(&self, origin: usize, first: usize) -> bool {
        let c = &self.contract;
        let start = c.boundaries[first].saturating_sub(1);
        let Some(end) = retained_partition_end(c.boundaries[first + 1], c.valid_bars, c.purge)
            .checked_sub(c.pred_len)
        else {
            return false;
        };
        origin >= start && origin < end
    }

    /// This ticker's in-period hole as an inclusive TARGET BAR range, or `None` when it has
    /// none. Exposed so a consumer that re-selects rows (a stride, a subsample, a per-row patch
    /// phase) can assert against the same numbers the enumeration used.
    pub fn in_period_hole(&self) -> Option<(usize, usize)> {
        self.hole
    }

    /// This ticker's in-period held-out origins, ascending. Empty when it has no hole.
    pub fn in_period_origins(&self) -> &[usize] {
        &self.in_period
    }

    /// The last valid-bar ordinal whose timestamp is at or before `stamp`, or `None` when this
    /// ticker's history starts after it.
    ///
    /// A binary search over the FILTERED ordinals, not the raw indices: quarantining an invalid
    /// OHLC bar removes it from the ordinal line but preserves monotonicity, so the search is
    /// still sound and lands on the same ordinals the row enumeration uses.
    pub(super) fn valid_ordinal_at_or_before(&self, stamp: i64) -> Option<usize> {
        let bars = self.contract.valid_bars;
        if bars == 0 || self.timestamp(0) > stamp {
            return None;
        }
        let (mut low, mut high) = (0usize, bars - 1);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if self.timestamp(mid) <= stamp {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        Some(low)
    }

    /// Place the shared `anchors` on this ticker's own ordinal line.
    ///
    /// ONE binary search followed by a forward WALK, because the anchors are ascending and
    /// close together: the walk touches the hole's own bars in order, which is the page-friendly
    /// access pattern, where 32 independent binary searches would be 32 random faults per
    /// ticker into a 28 GB corpus.
    ///
    /// `None` - no hole and no origins - unless the ticker covers the whole anchor span with
    /// strictly increasing ordinals AND leaves `bracket` rows of trained history on each side,
    /// the left bracket inside its own history and the right one before the purge band at the
    /// 70% boundary. A ticker that merely stops inside the span would otherwise contribute a
    /// degenerate hole whose last anchors all collapse onto its final bar.
    fn place(&self, anchors: &[i64], above: usize, bracket: usize) -> Option<Placement> {
        let c = &self.contract;
        let mut cursor = self.valid_ordinal_at_or_before(anchors[0])?;
        let mut origins = Vec::with_capacity(anchors.len());
        origins.push((self.timestamp(cursor) == anchors[0]).then_some(cursor));
        let first = cursor;
        for stamp in &anchors[1..] {
            let mut advanced = false;
            while cursor + 1 < c.valid_bars && self.timestamp(cursor + 1) <= *stamp {
                cursor += 1;
                advanced = true;
            }
            if !advanced {
                return None;
            }
            origins.push((self.timestamp(cursor) == *stamp).then_some(cursor));
        }
        let hole = (first + 1, cursor + c.pred_len);
        (first + 1 >= c.common_context + bracket
            && hole.1 < c.valid_bars
            && hole.1 + c.purge + above < c.boundaries[0])
            .then_some(Placement { hole, origins })
    }

    /// Whether a training row whose FINAL origin is `origin` can be trained on without any of
    /// its supervised target bars falling inside the hole.
    ///
    /// The bound is deliberately MODEL-CONFIGURATION-INDEPENDENT. A row spans bars
    /// `[origin - context + 1, origin + pred_len]`, every one of its causal sub-origins lies
    /// inside the context, and every target is at least one bar ahead of its own sub-origin, so
    /// no supervised target bar can lie outside `[origin - context + 2, origin + pred_len]`
    /// whatever `patch_len` and `min_history` are. At the current configuration the true reach
    /// is `[origin - 5743, origin + 192]` - 375 sub-origins at stride `patch_len = 16`, the
    /// first 15 of them dropped by the `min_history = 256` mask - and the conservative bound
    /// costs 255 extra bars, about 1.3 rows per ticker. That price buys a guarantee that cannot
    /// be silently broken by a change to the patch grid, the history floor, or a per-row origin
    /// phase shift: a caller that shifts a row's origin re-tests the SHIFTED origin here.
    ///
    /// A row's supervision is NOT its final 192-bar target window. Training runs the head on
    /// every causal sub-origin (`model.rs` `future_windows` with `last_only = false`), so one
    /// row supervises roughly 6,000 bars while consecutive rows advance only `pred_len`. Any
    /// exclusion written against the final window alone leaks by a factor of about 31.
    pub fn supervision_clears_hole(&self, origin: usize) -> bool {
        let Some((lo, hi)) = self.hole else {
            return true;
        };
        let c = &self.contract;
        origin + c.pred_len < lo || (origin + 2).saturating_sub(c.context) > hi
    }

    fn target_count(&self, origin: usize) -> Option<usize> {
        let c = &self.contract;
        if origin < c.common_context.max(c.context) - 1 {
            return None;
        }
        if origin < c.train_end - 1 {
            return Some(c.pred_len.min(c.train_end - origin - 1));
        }
        (self.owns_partition_targets(origin, 0) || self.owns_partition_targets(origin, 1))
            .then_some(c.pred_len)
    }
}

pub struct Corpus {
    pub contract: CorpusContract,
    pub train_refs: Vec<WindowRef>,
    /// The `[70%, 80%)` calibration partition: every origin owns a COMPLETE `pred_len` of
    /// targets, so `calibration_refs.len() * pred_len` is its target-bar count exactly.
    ///
    /// Its reason to exist is that NOTHING in a training run reads it. Checkpoint selection
    /// minimizes the objective-weighted NLL of [`Self::validation_refs`] and of the strided
    /// sample drawn from it, so a statistic fitted anywhere inside that population inherits
    /// the selection that population performed. This partition is the one held-out block
    /// selection cannot have touched, which is what lets a post-hoc amplitude calibration be
    /// fitted here and spent on the validation split with no selection contamination on the
    /// FIT side. Disjointness from `validation_refs` is proven in
    /// [`CorpusTicker::owns_partition_targets`], not asserted by the consumer.
    pub calibration_refs: Vec<WindowRef>,
    pub validation_refs: Vec<WindowRef>,
    /// The IN-PERIOD held-out draw: origins cut out of the hole inside the TRAINING
    /// chronological span, placed by the SAME rule [`Self::validation_refs`] uses - complete,
    /// non-overlapping `pred_len` target runs from the hole's own first bar - so the two draws
    /// differ in period and in nothing else. That structural identity is the whole point: any
    /// difference in construction would be a second explanation for a difference in IC.
    ///
    /// Disjointness from every surviving training row is guaranteed by
    /// [`CorpusTicker::supervision_clears_hole`], which the enumeration and every re-selecting
    /// consumer both call, and proved by
    /// `in_period_origins_share_no_origin_and_no_target_bar_with_any_training_row`.
    pub in_period_refs: Vec<WindowRef>,
    pub excluded_tickers: Vec<ExcludedTicker>,
    pub market: MarketSummary,
    tickers: Vec<CorpusTicker>,
    exogenous: Exogenous,
    gather_pool: rayon::ThreadPool,
    device: Device,
    /// Where this run's startup went, phase by phase; see [`cache::LoadTiming`].
    pub timing: cache::LoadTiming,
}

/// One row is `seq_len + pred_len` bars: `log_prices[b, t, c] = ln(price) - ln(anchor)` where
/// `anchor` is the row's last context close, `valid[b, t]` marks observed bars (context bars are
/// always valid; horizon bars are valid up to the ticker's owned targets), `aux[b, t, :]`
/// carries the covariates for every bar with history-only channels zeroed beyond the context,
/// and `market_cum[b, t]` is the cumulative market log return at the bar's timestamp minus its
/// value at the row's last context bar.
pub struct Batch {
    pub log_prices: Tensor,
    pub valid: Tensor,
    pub aux: Tensor,
    pub market_cum: Tensor,
    pub anchor: Tensor,
    pub valid_target_bars: usize,
    packed: Tensor,
    context: usize,
    pred_len: usize,
    aux_channels: usize,
}

impl Batch {
    pub(super) fn row_width(context: usize, pred_len: usize, aux_channels: usize) -> usize {
        (context + pred_len) * (6 + aux_channels) + 1
    }

    pub(super) fn from_packed(
        packed: Tensor,
        context: usize,
        pred_len: usize,
        aux_channels: usize,
        valid_target_bars: usize,
    ) -> Self {
        let b = packed.size()[0];
        let length = (context + pred_len) as i64;
        assert_eq!(
            packed.size()[1] as usize,
            Self::row_width(context, pred_len, aux_channels)
        );
        let log_prices = packed.narrow(1, 0, length * 4).reshape([b, length, 4]);
        let valid = packed.narrow(1, length * 4, length);
        let aux = packed
            .narrow(1, length * 5, length * aux_channels as i64)
            .reshape([b, length, aux_channels as i64]);
        let market_cum = packed.narrow(1, length * (5 + aux_channels as i64), length);
        let anchor = packed
            .narrow(1, length * (6 + aux_channels as i64), 1)
            .reshape([b]);
        Self {
            log_prices,
            valid,
            aux,
            market_cum,
            anchor,
            valid_target_bars,
            packed,
            context,
            pred_len,
            aux_channels,
        }
    }

    /// Rows in the packed block: the batch dimension a captured graph or a resident buffer
    /// is fixed to.
    pub fn rows(&self) -> i64 {
        self.packed.size()[0]
    }

    pub fn to_device(self, device: Device) -> Self {
        Self::from_packed(
            self.packed.to_device_(device, Kind::Float, true, false),
            self.context,
            self.pred_len,
            self.aux_channels,
            self.valid_target_bars,
        )
    }

    /// A device-resident batch of this batch's exact shape whose packed storage NEVER
    /// moves, so [`Self::upload`] can refill it in place.
    ///
    /// Two things need that. A captured CUDA graph records addresses, so its input has to
    /// be one fixed buffer rather than whatever [`Self::to_device`] allocated this step.
    /// And even eagerly, `to_device` allocates and frees the whole packed row block every
    /// step - 114 MB at batch 256 - which the caching allocator has to keep re-serving.
    pub fn resident(&self, device: Device) -> Self {
        Self::from_packed(
            Tensor::zeros(self.packed.size(), (Kind::Float, device)),
            self.context,
            self.pred_len,
            self.aux_channels,
            self.valid_target_bars,
        )
    }

    /// A host copy of this batch, in pinned memory when a device is given.
    ///
    /// The capture audit builds both to measure what the loader's `pin_memory` buys: a
    /// pageable source makes the H2D copy blocking whatever `non_blocking` says, because the
    /// driver has to stage it, so the host waits for the outstanding step to drain.
    pub fn host_copy(&self, pinned: Option<Device>) -> Self {
        let packed = self
            .packed
            .to_device_(Device::Cpu, Kind::Float, true, false);
        Self::from_packed(
            match pinned {
                Some(device) => packed.pin_memory(device),
                None => packed,
            },
            self.context,
            self.pred_len,
            self.aux_channels,
            self.valid_target_bars,
        )
    }

    /// Refill `resident` from this host batch, asynchronously.
    ///
    /// `Corpus::host_batch` pins the packed block whenever the corpus is prepared for a
    /// CUDA device, which is what makes the non-blocking copy safe: libtorch's caching
    /// host allocator holds the block until the copy retires, so the host batch may drop
    /// the moment this returns.
    pub fn upload(&self, resident: &mut Self) -> Result<()> {
        ensure!(
            self.packed.size() == resident.packed.size(),
            "resident batch shape does not match the host batch"
        );
        crate::torch::cuda::copy_nonblocking(&mut resident.packed, &self.packed)
            .map_err(|err| anyhow::anyhow!("uploading the packed batch: {err}"))?;
        resident.valid_target_bars = self.valid_target_bars;
        Ok(())
    }
}

impl Corpus {
    pub fn load(
        directory: &Path,
        requested: &[String],
        context: usize,
        pred_len: usize,
        common_context: usize,
        features: &FeatureSet,
        market_min_cross_section: usize,
        // `in_period_sections`: complete `pred_len` target runs to reserve as the IN-PERIOD
        // held-out draw, cut out of the middle of the training span. 0 disables the hole
        // entirely and leaves every existing population byte-identical.
        in_period_sections: usize,
    ) -> Result<Self> {
        ensure!(
            context > 0 && pred_len > 0 && common_context >= context,
            "invalid context or forecast length"
        );
        ensure!(
            market_min_cross_section > 0,
            "market steps need a positive minimum cross-section"
        );
        let purge = pred_len.max(100);
        let minimum_source_bars = common_context + 1;
        let minimum_training_bars = common_context + 1;
        let requested_set: BTreeSet<_> = requested.iter().cloned().collect();
        ensure!(
            requested_set.len() == requested.len(),
            "duplicate requested ticker"
        );
        let load_started = Instant::now();
        let mut timing = cache::LoadTiming::default();
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8);
        let gather_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("timexer-data-{i}"))
            .build()?;
        // Startup's opens and boundary probes are LATENCY-bound, not bandwidth-bound: every one
        // of them is a thread parked on a page fault, so the useful width is the device's queue
        // depth and not the core count that sizes the gather pool. A separate pool keeps the two
        // answers from having to be the same number, and this one dies with `load`.
        let io_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers * 4)
            .thread_name(|i| format!("timexer-corpus-io-{i}"))
            .build()?;
        // `read_dir` is one directory walk, but OPENING the 5,728 members is 5,728 independent
        // synchronous mmap-and-header-probe round trips - each one an `open`, an `fstat`, and a
        // page fault on the first and last record - and doing them one after another spent 18.8
        // of the warm run's 29 seconds waiting on a device that answers eight requests as
        // cheaply as one. The walk stays serial; the opens do not.
        let candidates: Vec<(String, std::path::PathBuf)> = fs::read_dir(directory)
            .context("reading five-minute corpus directory")?
            .map(|entry| Ok(entry?.path()))
            .filter_map(|path: Result<_>| match path {
                Ok(path) => match parse_bar_file_name(&path) {
                    Ok((ticker, 300))
                        if requested_set.is_empty() || requested_set.contains(&ticker) =>
                    {
                        Some(Ok((ticker, path)))
                    }
                    _ => None,
                },
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<_>>()?;
        let opened: Vec<(String, BarFile)> = io_pool.install(|| {
            candidates
                .into_par_iter()
                .map(|(ticker, path)| {
                    let file = BarFile::open(&path)?;
                    ensure!(
                        file.symbol() == ticker && file.res_secs() == 300,
                        "corpus filename/header mismatch"
                    );
                    Ok((ticker, file))
                })
                .collect::<Result<_>>()
        })?;
        let mut files = Vec::with_capacity(opened.len());
        let mut found = BTreeSet::new();
        let mut excluded_tickers = Vec::new();
        for (ticker, file) in opened {
            found.insert(ticker.clone());
            if file.is_empty() {
                excluded_tickers.push(ExcludedTicker {
                    ticker,
                    reason: "empty source corpus".into(),
                });
            } else {
                files.push(file);
            }
        }
        for ticker in &requested_set {
            ensure!(
                found.contains(ticker),
                "requested ticker {ticker} has no five-minute corpus"
            );
        }
        files.sort_unstable_by(|a, b| a.symbol().cmp(b.symbol()));
        ensure!(!files.is_empty(), "no eligible five-minute ticker corpora");
        timing.directory_scan_ms = load_started.elapsed().as_secs_f64() * 1000.;
        // The bar audits and the occupancy bitmap were two separate traversals of the same 17 GB
        // of records, for no reason but the order they were written in: an audit depends on the
        // bar bytes alone, and only the three `partition_point` probes at the end of
        // `filtered_contract` need the boundaries the bitmap produces. They are one pass now, and
        // a run whose ledger is intact makes zero passes.
        let phase = Instant::now();
        let mut audit_cache = cache::AuditCache::open(directory, super::data::SCHEMA);
        let identities: Vec<_> = files.iter().map(BarFile::identity).collect();
        let mut audits: Vec<Option<super::data::BarAudit>> = files
            .iter()
            .zip(&identities)
            .map(|(file, identity)| audit_cache.take(file.symbol(), *identity))
            .collect();
        let stale: Vec<usize> = audits
            .iter()
            .enumerate()
            .filter_map(|(index, audit)| audit.is_none().then_some(index))
            .collect();
        timing.audits_reused = files.len() - stale.len();
        timing.audits_computed = stale.len();
        let span = UniverseSpan::of(&files)?;
        timing.audit_ledger_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = Instant::now();
        let (mut occupancy, scanned) = gather_pool.install(|| scan(&files, &stale, span))?;
        timing.bars_rescanned = scanned.iter().map(|(_, a)| a.source_bars as u64).sum();
        for (index, audit) in scanned {
            audits[index] = Some(audit);
        }
        let audits: Vec<super::data::BarAudit> = audits
            .into_iter()
            .map(|audit| audit.context("every corpus file must carry a bar audit"))
            .collect::<Result<_>>()?;
        // Whatever the fused pass cost lands on the rebuild line when it rescanned something and
        // on the cache line when it did not, so the phases always sum to the total and a gap in
        // the chart is a real gap rather than an unattributed segment.
        let elapsed = phase.elapsed().as_secs_f64() * 1000.;
        if stale.is_empty() {
            timing.cache_read_ms += elapsed;
        } else {
            timing.bar_audit_ms = elapsed;
        }
        let phase = Instant::now();
        // Downstream layers are addressed by CONTENT - the digests themselves - not by inode, so
        // a corpus restored from backup rehashes once and then hits everything below.
        let universe: Vec<(&str, &str)> = files
            .iter()
            .zip(&audits)
            .map(|(file, audit)| (file.symbol(), audit.fingerprint.as_str()))
            .collect();
        let bounds_cache = cache::BoundsCache::open(directory, SCHEMA, &universe);
        let mut cache_write = std::time::Duration::ZERO;
        // Content keys authenticate lineage, but a decoded cache must also describe the actual
        // partition edges. Six adjacent records per source prove the monotone timestamp split
        // without repeating three binary searches or a full-corpus scan on a warm startup.
        let stored_bounds = bounds_cache.get(files.len()).filter(|(bounds, edges)| {
            let verified = io_pool.install(|| {
                files.par_iter().zip(edges).try_for_each(|(file, edge)| {
                    verify_split_edges(file.symbol(), file.bars(), *edge, *bounds)
                })
            });
            if let Err(error) = verified {
                // A corrupt artifact does not tell us whether its timestamps or its ordinal
                // edges changed. Rebuild the union quantiles, not just new edges at suspect
                // timestamps. Unchanged bar audits stay authenticated and need no rehash.
                eprintln!("Discarding unusable shared split cache; rebuilding bounds: {error:#}");
                false
            } else {
                true
            }
        });
        timing.cache_read_ms += phase.elapsed().as_secs_f64() * 1000.;
        let (bounds, edges) = match stored_bounds {
            Some((bounds, edges)) => {
                timing.bounds_reused = true;
                (bounds, edges)
            }
            None => {
                let phase = Instant::now();
                // Whatever the fused pass above did not visit still owes its occupancy. On a cold
                // corpus that set is empty; after an ingest touches one ticker it is everything
                // else, and it has to be, because a partition boundary is a quantile of the union
                // and no incremental update of a quantile exists.
                let mut fresh = vec![false; files.len()];
                for &index in &stale {
                    fresh[index] = true;
                }
                let owed: Vec<usize> = (0..files.len()).filter(|&index| !fresh[index]).collect();
                if !owed.is_empty() {
                    let rest = gather_pool.install(|| occupy(&files, &owed, &audits, span))?;
                    for (word, other) in occupancy.iter_mut().zip(rest) {
                        *word |= other;
                    }
                    timing.bars_rescanned += owed
                        .iter()
                        .map(|&index| audits[index].source_bars as u64)
                        .sum::<u64>();
                }
                let bounds = quantile_bounds(&occupancy, span.first)?;
                // The edge probes are three binary searches per ticker. They belong to this arm
                // because they are a function of the boundaries, and the boundaries have just
                // moved; the pages they touch are still hot from the scan that produced them.
                let edges: Vec<[u64; 3]> = io_pool.install(|| {
                    files
                        .par_iter()
                        .map(|file| bounds.map(|bound| file.index_at_or_after(bound) as u64))
                        .collect()
                });
                timing.shared_bounds_ms = phase.elapsed().as_secs_f64() * 1000.;
                let phase = Instant::now();
                bounds_cache.store(bounds, &edges)?;
                cache_write += phase.elapsed();
                (bounds, edges)
            }
        };
        let phase = Instant::now();
        audit_cache.replace(
            files
                .iter()
                .zip(&identities)
                .zip(&audits)
                .map(|((file, identity), audit)| {
                    (file.symbol().to_owned(), *identity, audit.clone())
                })
                .collect(),
        );
        timing.cache_read_ms += phase.elapsed().as_secs_f64() * 1000.;
        let phase = Instant::now();
        audit_cache.store()?;
        cache_write += phase.elapsed();
        let phase = Instant::now();
        let mut audited: Vec<(BarFile, super::data::BarAudit, [usize; 3])> = files
            .into_iter()
            .zip(audits)
            .zip(&edges)
            .map(|((file, audit), edge)| (file, audit, edge.map(|index| index as usize)))
            .collect();
        // Eligibility used to be a `partition_point` per ticker here - a binary search over a
        // mapped bar file is ~17 RANDOM page faults into 28 GB that no readahead predicts, and
        // over 5,728 files that was 20.6 s of a 29 s warm run, more than the cached corpus load
        // it precedes. The edge it wanted is now a cached artifact and this touches no bar at all.
        audited.retain(|(file, _, edge)| {
            let keep = edge[0] >= minimum_training_bars;
            if !keep {
                excluded_tickers.push(ExcludedTicker {
                    ticker: file.symbol().to_owned(),
                    reason: "insufficient purged training history".into(),
                });
            }
            keep
        });
        for ticker in &requested_set {
            ensure!(
                audited.iter().any(|(file, _, _)| file.symbol() == ticker),
                "requested ticker {ticker} has insufficient purged training history"
            );
        }
        ensure!(
            !audited.is_empty(),
            "no ticker has sufficient purged training history"
        );
        excluded_tickers.sort_unstable_by(|a, b| {
            a.ticker
                .cmp(&b.ticker)
                .then_with(|| a.reason.cmp(&b.reason))
        });
        timing.eligibility_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = Instant::now();
        let mut tickers: Vec<CorpusTicker> = gather_pool.install(|| {
            audited
                .into_par_iter()
                .map(|(file, audit, edge)| {
                    let contract = filtered_contract(
                        file.symbol(),
                        file.bars(),
                        &audit,
                        context,
                        pred_len,
                        common_context,
                        bounds,
                        edge,
                    )
                    .with_context(|| {
                        format!(
                            "authenticating five-minute ticker {} from {}",
                            file.symbol(),
                            file.path().display()
                        )
                    })?;
                    file.advise_random_access()?;
                    Ok(CorpusTicker {
                        contract,
                        file,
                        hole: None,
                        in_period: Vec::new(),
                        ranks: Vec::new(),
                    })
                })
                .collect::<Result<_>>()
        })?;
        timing.contract_ms = phase.elapsed().as_secs_f64() * 1000.;
        tickers.retain(|ticker| {
            let eligible = ticker.contract.train_end >= minimum_training_bars;
            if !eligible {
                excluded_tickers.push(ExcludedTicker {
                    ticker: ticker.contract.ticker.clone(),
                    reason:
                        "insufficient valid purged training history after quarantining invalid OHLC"
                            .into(),
                });
            }
            eligible
        });
        for ticker in &requested_set {
            ensure!(tickers.iter().any(|data| &data.contract.ticker == ticker), "requested ticker {ticker} has insufficient valid purged training history after quarantining invalid OHLC");
        }
        excluded_tickers.sort_unstable_by(|a, b| {
            a.ticker
                .cmp(&b.ticker)
                .then_with(|| a.reason.cmp(&b.reason))
        });
        ensure!(
            !tickers.is_empty(),
            "no ticker has sufficient valid purged training history"
        );
        // AFTER the eligibility retain, because the anchor placement is scored on how many
        // tickers hold each anchor and a ticker dropped later would inflate that census. The
        // cost is folded into `contract_ms`: it is the same per-ticker geometry phase, and on
        // the 4,873-ticker corpus it is one binary search plus a walk over the hole's own bars.
        let phase = Instant::now();
        let (in_period_anchors, in_period_census) =
            match in_period_plan(&tickers, in_period_sections, &gather_pool)? {
                Some((anchors, census, placements)) => {
                    for (ticker, placement) in tickers.iter_mut().zip(placements) {
                        if let Some(placement) = placement {
                            ticker.hole = Some(placement.hole);
                            ticker.in_period = placement.origins.into_iter().flatten().collect();
                        }
                    }
                    (anchors, census)
                }
                None => (Vec::new(), Vec::new()),
            };
        timing.contract_ms += phase.elapsed().as_secs_f64() * 1000.;
        let phase = Instant::now();
        let grid = ticker_grid(&tickers)?;
        let eligible: Vec<(&str, &str)> = tickers
            .iter()
            .map(|ticker| {
                (
                    ticker.contract.ticker.as_str(),
                    ticker.contract.fingerprint.as_str(),
                )
            })
            .collect();
        let market_cache = cache::MarketCache::open(
            directory,
            SCHEMA,
            &eligible,
            grid.first_ts(),
            grid.slots(),
            market_min_cross_section,
        );
        let stored_grid = market_cache.get(grid.slots()).and_then(|stored| {
            MarketSteps::from_parts(
                stored.first_ts,
                stored.min_cross_section,
                stored.population,
                stored.returns,
                stored.log_volume,
                stored.log_range,
            )
        });
        let sources: Vec<_> = tickers
            .iter()
            .map(|ticker| {
                (
                    ticker.file.bars(),
                    ticker.contract.invalid_ohlc_indices.as_slice(),
                )
            })
            .collect();
        let source_bars: u64 = tickers
            .iter()
            .map(|ticker| ticker.contract.source_bars as u64)
            .sum();
        let steps = match stored_grid {
            Some(steps) => {
                timing.market_reused = true;
                timing.cache_read_ms += phase.elapsed().as_secs_f64() * 1000.;
                steps
            }
            None => {
                let steps =
                    gather_pool.install(|| market_steps(&sources, grid, market_min_cross_section));
                // Two traversals of every eligible ticker: one to count each slot's cross
                // section, and one to walk returns between consecutive DEFINING slots. The second
                // cannot join the first - "defining" is a property of the completed population
                // vector - which is why this layer is cached rather than folded away.
                timing.bars_rescanned += 2 * source_bars;
                timing.market_grid_ms = phase.elapsed().as_secs_f64() * 1000.;
                let phase = Instant::now();
                let (first_ts, min_cross_section, population, returns, log_volume, log_range) =
                    steps.parts();
                market_cache.store(&cache::MarketGrid {
                    first_ts,
                    min_cross_section,
                    population: population.to_vec(),
                    returns: returns.clone(),
                    log_volume: log_volume.clone(),
                    log_range: log_range.clone(),
                })?;
                cache_write += phase.elapsed();
                steps
            }
        };
        // The rank channel is the one member of the cross-section family that needs each
        // slot's whole distribution rather than its moments, so it gets its own artifact: an
        // arm toggling only this channel neither invalidates the market grid nor waits on it.
        let mut ranks: Option<Vec<Vec<u16>>> = None;
        if features.cross_section_rank {
            let phase = Instant::now();
            let valid_bars: Vec<usize> = tickers
                .iter()
                .map(|ticker| ticker.contract.valid_bars)
                .collect();
            let rank_cache = cache::RankCache::open(
                directory,
                SCHEMA,
                &eligible,
                grid.first_ts(),
                grid.slots(),
                market_min_cross_section,
            );
            ranks = Some(match rank_cache.get(&valid_bars) {
                Some(stored) => {
                    timing.cache_read_ms += phase.elapsed().as_secs_f64() * 1000.;
                    stored
                }
                None => {
                    let built = gather_pool.install(|| cross_section_ranks(&sources, grid, &steps));
                    // Three more traversals: one to count each slot's contributions, one to
                    // scatter the contributing returns into their slot segments, and one to
                    // place every bar's own return inside its sorted segment. Charged to the
                    // market grid phase - same inputs, same derivation.
                    timing.bars_rescanned += 3 * source_bars;
                    let ms = phase.elapsed().as_secs_f64() * 1000.;
                    timing.market_grid_ms = if timing.market_grid_ms.is_nan() {
                        ms
                    } else {
                        timing.market_grid_ms + ms
                    };
                    let phase = Instant::now();
                    rank_cache.store(&built)?;
                    cache_write += phase.elapsed();
                    built
                }
            });
        }
        drop(sources);
        if let Some(ranks) = ranks {
            ensure!(
                ranks.len() == tickers.len(),
                "cross-section ranks do not cover the eligible universe"
            );
            for (ticker, row) in tickers.iter_mut().zip(ranks) {
                ensure!(
                    row.len() == ticker.contract.valid_bars,
                    "cross-section ranks for {} do not align with its valid bars",
                    ticker.contract.ticker
                );
                ticker.ranks = row;
            }
        }
        let phase = Instant::now();
        let (exogenous, market) =
            gather_pool.install(|| exogenous_series(directory, features, grid, steps))?;
        timing.exogenous_ms = phase.elapsed().as_secs_f64() * 1000.;
        let phase = Instant::now();
        let mut train_refs = Vec::new();
        let mut calibration_refs = Vec::new();
        let mut validation_refs = Vec::new();
        let mut in_period_refs = Vec::new();
        let mut train_target_bars = 0;
        let mut validation_target_bars = 0;
        let mut validation_remainder_bars = 0;
        let mut in_period_purged_rows = 0;
        let mut in_period_purged_target_bars = 0;
        // One placement rule, both held-out partitions: `first = 0` strides the calibration
        // partition `[boundaries[0], boundaries[1])` and `first = 1` the validation partition
        // `[boundaries[1], boundaries[2])`. Complete, non-overlapping `pred_len` target runs
        // from the partition's own first bar, stopping `purge` bars short of the next
        // boundary. It was written out for `first = 1` alone, and the calibration partition
        // this corpus has declared in its schema since v5 was therefore enumerated by
        // nothing: it yielded zero origins BY CONSTRUCTION, not by an off-by-one.
        let band = |c: &DataContract, ticker: usize, first: usize| -> (Vec<WindowRef>, usize) {
            let start = c.boundaries[first].max(common_context);
            let available = retained_partition_end(c.boundaries[first + 1], c.valid_bars, purge)
                .saturating_sub(start);
            (
                (0..available / pred_len)
                    .map(|i| WindowRef {
                        ticker,
                        origin: start - 1 + i * pred_len,
                    })
                    .collect(),
                available % pred_len,
            )
        };
        for (ticker, data) in tickers.iter().enumerate() {
            let c = &data.contract;
            // Row by row rather than `extend`, because two of the three quantities a holed run
            // has to report - the rows the purge cost and the target bars they owned - are
            // exactly the ones a filtered `extend` would throw away. With the hole off,
            // `owned` sums to `train_end - common_context` bar for bar (the tails tile the span
            // with the last one truncated), so `train_target_bars` is unchanged to the byte.
            for origin in (common_context - 1..c.train_end - 1).step_by(pred_len) {
                let owned = pred_len.min(c.train_end - origin - 1);
                if data.supervision_clears_hole(origin) {
                    train_refs.push(WindowRef { ticker, origin });
                    train_target_bars += owned;
                } else {
                    in_period_purged_rows += 1;
                    in_period_purged_target_bars += owned;
                }
            }
            // The placed anchors, not a stride: every origin here is a shared wall clock this
            // ticker holds a bar at, which is what lets the draw group them into
            // cross-sections. Each owns a complete `pred_len` of targets inside the hole by
            // construction in `CorpusTicker::place`.
            in_period_refs.extend(data.in_period_origins().iter().map(|origin| WindowRef {
                ticker,
                origin: *origin,
            }));
            calibration_refs.extend(band(c, ticker, 0).0);
            let (validation, remainder) = band(c, ticker, 1);
            validation_target_bars += validation.len() * pred_len;
            validation_remainder_bars += remainder;
            validation_refs.extend(validation);
        }
        timing.origin_enumeration_ms = phase.elapsed().as_secs_f64() * 1000.;
        ensure!(
            !train_refs.is_empty() && !calibration_refs.is_empty() && !validation_refs.is_empty(),
            "empty unlocked corpus split: {} training, {} calibration and {} validation origins",
            train_refs.len(),
            calibration_refs.len(),
            validation_refs.len()
        );
        println!(
            "CausalPatch held-out populations: {} calibration origins over [70%, 80%) covering {} target bars, {} validation origins over [80%, 90%) covering {} target bars with {} unused remainder",
            calibration_refs.len(),
            calibration_refs.len() * pred_len,
            validation_refs.len(),
            validation_target_bars,
            validation_remainder_bars
        );
        if in_period_sections > 0 {
            ensure!(
                !in_period_refs.is_empty(),
                "an in-period holdout of {in_period_sections} sections admitted no origin: no \
                 ticker's history carries {} bars of hole plus {IN_PERIOD_BRACKET_ROWS} bracket \
                 rows on each side",
                in_period_sections * pred_len
            );
            println!(
                "CausalPatch in-period held-out population: {} origins over {} sections per \
                 eligible ticker covering {} target bars inside the TRAINING span, bracketed by \
                 {IN_PERIOD_BRACKET_ROWS} trained rows on each side; the purge that guarantees \
                 no surviving row supervises an in-period target bar removed {} training rows \
                 owning {} target bars ({:.2}% of the {} target bars an unholed run trains on), \
                 so this arm's TRAINING losses are not step-comparable to a control's",
                in_period_refs.len(),
                in_period_sections,
                in_period_refs.len() * pred_len,
                in_period_purged_rows,
                in_period_purged_target_bars,
                100. * in_period_purged_target_bars as f64
                    / (train_target_bars + in_period_purged_target_bars) as f64,
                train_target_bars + in_period_purged_target_bars
            );
        }
        let phase = Instant::now();
        let market_fingerprint = market_fingerprint(&tickers, bounds, market_min_cross_section);
        let spy_fingerprint = features
            .spy
            .then(|| file_sha256(bar_file_path(directory, SPY, 300)))
            .transpose()?;
        timing.fingerprint_ms = phase.elapsed().as_secs_f64() * 1000.;
        if !cache_write.is_zero() {
            timing.cache_write_ms = cache_write.as_secs_f64() * 1000.;
        }
        timing.total_ms = load_started.elapsed().as_secs_f64() * 1000.;
        println!("{}", timing.summary());
        Ok(Self {
            contract: CorpusContract {
                schema: SCHEMA.into(),
                tickers: tickers.iter().map(|t| t.contract.clone()).collect(),
                boundary_timestamps: bounds,
                context,
                pred_len,
                common_context,
                purge,
                features: *features,
                auxiliary_schema: features.schema(),
                market_fingerprint,
                market_min_cross_section,
                spy_fingerprint,
                minimum_source_bars,
                minimum_training_bars,
                train_target_bars,
                validation_target_bars,
                validation_remainder_bars,
                in_period_sections,
                in_period_anchors,
                in_period_census,
                in_period_origins: in_period_refs.len(),
                in_period_target_bars: in_period_refs.len() * pred_len,
                in_period_purged_rows,
                in_period_purged_target_bars,
                // Empty means the historical strided placement. `load` never anchors: the
                // anchored placement is a post-load transformation so that both draws are
                // reachable from one process on one set of weights.
                cross_section_placement: String::new(),
                calibration_anchors: Vec::new(),
                validation_anchors: Vec::new(),
                excluded_tickers: excluded_tickers.clone(),
            },
            train_refs,
            calibration_refs,
            validation_refs,
            in_period_refs,
            excluded_tickers,
            market,
            tickers,
            exogenous,
            gather_pool,
            device: Device::Cpu,
            timing,
        })
    }

    /// SHA-256 over an origin list, so a placement change is OBSERVABLE even when it preserves
    /// cardinality.
    ///
    /// This exists because the external portfolio tape binds its cache to the exact
    /// `validation_refs` digest: the anchored placement of [`Self::anchor_cross_section_draws`]
    /// changes almost every TIMESTAMP while leaving the origin count in the same range, which is
    /// precisely the change a cache keyed on shape would miss. The ticker index is hashed by
    /// SYMBOL, not by position, so a change in universe ordering cannot forge a match either.
    pub fn origins_sha256(&self, refs: &[WindowRef]) -> String {
        let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
        for reference in refs {
            digest.update(self.ticker(*reference).contract.ticker.as_bytes());
            digest.update(&(reference.origin as u64).to_le_bytes());
        }
        digest
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// Bounded, reproducible draws; this never mutates or thins the training universe.
    ///
    /// Quotas are balanced over eligible tickers with available rows (seeded ticker order
    /// resolves a budget smaller than the universe). Each ticker's quota is stratified over
    /// its full chronological origin list. This is a ticker-balanced diagnostic estimand,
    /// NOT an unbiased estimate of the row-weighted full validation loss. Zero-coverage names
    /// and the actual origin dates are explicit in the manifest.
    pub fn research_sample_plan(
        &self,
        seed: u64,
        validation_rows: usize,
        probe_fit_rows: usize,
        source_lookback_bars: usize,
    ) -> Result<ResearchSamplePlan> {
        ensure!(
            source_lookback_bars < self.contract.context,
            "research source lookback must fit inside the causal context"
        );
        let chronology = self.split_chronology()?;
        let training_last_target_ms = chronology
            .iter()
            .map(|ticker| ticker.dense_training_target_envelope_ms[1])
            .max()
            .context("research corpus has no training targets")?;
        let mut thresholds = vec![None; self.tickers.len()];
        let mut excluded = vec![0usize; self.tickers.len()];
        let validation_population: Vec<_> = self
            .validation_refs
            .iter()
            .copied()
            .filter(|reference| {
                let minimum = *thresholds[reference.ticker].get_or_insert_with(|| {
                    let ticker = self.ticker(*reference);
                    if reference.origin >= source_lookback_bars
                        && ticker.timestamp(reference.origin - source_lookback_bars)
                            > training_last_target_ms
                    {
                        0
                    } else {
                        ticker
                            .valid_ordinal_at_or_before(training_last_target_ms)
                            .map_or(source_lookback_bars, |ordinal| {
                                ordinal + 1 + source_lookback_bars
                            })
                    }
                });
                let keep = reference.origin >= minimum;
                if !keep {
                    excluded[reference.ticker] += 1;
                }
                keep
            })
            .collect();
        let (validation_refs, mut validation) = self.research_draw(
            &validation_population,
            validation_rows,
            seed ^ 0x76616c70616e656c,
            false,
            source_lookback_bars,
        )?;
        for (coverage, count) in validation.coverage.iter_mut().zip(&excluded) {
            coverage.chronology_excluded_origins = *count;
        }
        let (probe_fit_refs, probe_fit) = self.research_draw(
            &self.train_refs,
            probe_fit_rows,
            seed ^ 0x66697470616e656c,
            true,
            source_lookback_bars,
        )?;
        let corpus_sha256 = sha256_bytes(&serde_json::to_vec(&self.contract)?);
        Ok(ResearchSamplePlan {
            validation_refs,
            validation_population_refs: validation_population,
            probe_fit_refs,
            manifest: ResearchSampleManifest {
                schema: "causalpatch-research-panel-v1".into(),
                seed,
                selection: "ticker-balanced;seeded-without-replacement;per-ticker-origin-quantile-strata;full-training-pool-unchanged".into(),
                corpus_sha256,
                training_origins_sha256: self.origins_sha256(&self.train_refs),
                training_population_origins: self.train_refs.len(),
                boundary_timestamps_ms: self.contract.boundary_timestamps,
                context: self.contract.context,
                pred_len: self.contract.pred_len,
                source_lookback_bars,
                training_last_target_ms,
                validation_chronology_excluded_origins: excluded.iter().sum(),
                target_mask_encoding: "forecast horizon h in 1..=pred_len is valid iff h<=valid_target_prefix;held-out scoring final-origin-only;dense training validity is bounded by train_end".into(),
                chronology,
                validation,
                probe_fit,
            },
        })
    }

    fn research_draw(
        &self,
        population: &[WindowRef],
        requested: usize,
        seed: u64,
        training: bool,
        source_lookback_bars: usize,
    ) -> Result<(Vec<WindowRef>, ResearchDraw)> {
        ensure!(
            requested > 0 && !population.is_empty(),
            "research draws require a nonempty population and positive budget"
        );
        ensure!(
            population
                .windows(2)
                .all(|rows| (rows[0].ticker, rows[0].origin) < (rows[1].ticker, rows[1].origin)),
            "research population must be unique and ticker-major chronological"
        );
        let mut ranges = vec![0..0; self.tickers.len()];
        let mut cursor = 0;
        for (ticker, range) in ranges.iter_mut().enumerate() {
            let start = cursor;
            while cursor < population.len() && population[cursor].ticker == ticker {
                cursor += 1;
            }
            *range = start..cursor;
        }
        ensure!(
            cursor == population.len(),
            "research population has an unknown ticker"
        );
        let wanted = requested.min(population.len());
        let mut order: Vec<_> = (0..ranges.len())
            .filter(|&i| !ranges[i].is_empty())
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        order.shuffle(&mut rng);
        let mut quotas = vec![0usize; ranges.len()];
        let mut remaining = wanted;
        while remaining > 0 {
            for &ticker in &order {
                if remaining > 0 && quotas[ticker] < ranges[ticker].len() {
                    quotas[ticker] += 1;
                    remaining -= 1;
                }
            }
        }
        let mut selected = Vec::with_capacity(wanted);
        let mut coverage = Vec::with_capacity(ranges.len());
        for (ticker, range) in ranges.iter().enumerate() {
            let start = selected.len();
            let quota = quotas[ticker];
            for stratum in 0..quota {
                let lo = stratum * range.len() / quota;
                let hi = (stratum + 1) * range.len() / quota;
                selected.push(population[range.start + rng.random_range(lo..hi)]);
            }
            let stamps = |refs: &[WindowRef]| {
                refs.first().zip(refs.last()).map(|(first, last)| {
                    [
                        self.ticker(*first).timestamp(first.origin),
                        self.ticker(*last).timestamp(last.origin),
                    ]
                })
            };
            coverage.push(ResearchCoverage {
                ticker: self.tickers[ticker].contract.ticker.clone(),
                population_origins: range.len(),
                selected_origins: quota,
                chronology_excluded_origins: 0,
                population_origin_range_ms: stamps(&population[range.clone()]),
                selected_origin_range_ms: stamps(&selected[start..]),
            });
        }
        // Independent ticker rows in each batch, not long contiguous runs of one ticker.
        selected.shuffle(&mut rng);
        let origins = selected
            .iter()
            .map(|reference| {
                let ticker = self.ticker(*reference);
                let targets = if training {
                    self.training_target_count(reference.ticker, reference.origin)
                        .context("probe fit escaped the training population")?
                } else {
                    ensure!(
                        ticker.owns_partition_targets(reference.origin, 1),
                        "research validation escaped its reserved target partition"
                    );
                    self.contract.pred_len
                };
                Ok(ResearchOrigin {
                    ticker: ticker.contract.ticker.clone(),
                    origin: reference.origin,
                    origin_ms: ticker.timestamp(reference.origin),
                    common_source_ms: ticker.timestamp(reference.origin - source_lookback_bars),
                    context_first_ms: ticker
                        .timestamp(reference.origin + 1 - self.contract.context),
                    target_first_ms: ticker.timestamp(reference.origin + 1),
                    target_last_ms: ticker.timestamp(reference.origin + targets),
                    valid_target_prefix: targets,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut mask_digest = ring::digest::Context::new(&ring::digest::SHA256);
        mask_digest.update(&(self.contract.pred_len as u64).to_le_bytes());
        for row in &origins {
            mask_digest.update(&(row.valid_target_prefix as u64).to_le_bytes());
        }
        let draw = ResearchDraw {
            partition: if training {
                "train-only-probe-fit"
            } else {
                "validation-final-origin"
            }
            .into(),
            requested_rows: requested,
            population_origins: population.len(),
            origins_sha256: self.origins_sha256(&selected),
            target_mask_sha256: digest_hex(mask_digest.finish()),
            coverage,
            origins,
        };
        Ok((selected, draw))
    }

    pub fn split_chronology(&self) -> Result<Vec<SplitChronology>> {
        self.tickers
            .iter()
            .map(|ticker| {
                let c = &ticker.contract;
                ensure!(
                    c.boundary_timestamps == self.contract.boundary_timestamps,
                    "{} does not share the corpus UTC split",
                    c.ticker
                );
                let before = c
                    .boundaries
                    .map(|edge| edge.checked_sub(1).map(|i| ticker.timestamp(i)));
                let after = c
                    .boundaries
                    .map(|edge| (edge < c.valid_bars).then(|| ticker.timestamp(edge)));
                for i in 0..3 {
                    ensure!(
                        before[i].is_none_or(|stamp| stamp < c.boundary_timestamps[i])
                            && after[i].is_none_or(|stamp| stamp >= c.boundary_timestamps[i]),
                        "{} logical edge {i} disagrees with its UTC split",
                        c.ticker
                    );
                }
                ensure!(
                    c.train_end > c.common_context && c.train_end <= c.boundaries[0],
                    "{} training target reach crosses calibration",
                    c.ticker
                );
                let envelope = |first: usize| {
                    let start = c.boundaries[first].max(c.common_context);
                    let end =
                        retained_partition_end(c.boundaries[first + 1], c.valid_bars, c.purge);
                    (start < end).then(|| [ticker.timestamp(start), ticker.timestamp(end - 1)])
                };
                Ok(SplitChronology {
                    ticker: c.ticker.clone(),
                    boundary_ordinals: c.boundaries,
                    last_before_boundary_ms: before,
                    first_at_or_after_boundary_ms: after,
                    // Conservative earliest dense target: even the first possible sub-origin
                    // cannot predict a bar before the second context bar.
                    dense_training_target_envelope_ms: [
                        ticker.timestamp(c.common_context + 1 - c.context),
                        ticker.timestamp(c.train_end - 1),
                    ],
                    calibration_target_envelope_ms: envelope(0),
                    validation_target_envelope_ms: envelope(1),
                    terminal_test_first_ms: after[2],
                })
            })
            .collect()
    }

    /// Replace the STRIDED calibration and validation placements with the anchored ones, in
    /// place, after load.
    ///
    /// A post-load transformation rather than a ninth `load` parameter, for two reasons that are
    /// both about the deliverable rather than about taste. It leaves the six existing `load`
    /// call sites untouched, and it makes the before/after comparison a property of ONE process:
    /// the strided draw is what `load` produced, the anchored draw is what this returns, so both
    /// can be scored on the same weights by the same code in one run. A flag threaded through
    /// `load` would have forced two processes and two corpus loads to compare two placements.
    ///
    /// Everything downstream of `validation_refs` moves with it BY DESIGN - `held-out full`,
    /// `held-out sample` and its persistence NLL anchor included. The 2.3947663 anchor is a
    /// property of a draw, so a run under this placement must recompute it rather than compare
    /// to it; the digest printed here is what makes the two incomparable populations impossible
    /// to confuse.
    pub fn anchor_cross_section_draws(&mut self, floor: usize) -> Result<()> {
        let started = Instant::now();
        let before = (
            self.origins_sha256(&self.calibration_refs),
            self.origins_sha256(&self.validation_refs),
        );
        let pred_len = self.contract.pred_len;
        let mut anchored = Vec::new();
        for first in [0usize, 1] {
            let (anchors, census, origins) = anchored_partition_refs(
                &self.tickers,
                first,
                self.contract.boundary_timestamps[first],
                floor,
                &self.gather_pool,
            )?;
            let refs: Vec<WindowRef> = origins
                .iter()
                .enumerate()
                .flat_map(|(ticker, origins)| {
                    origins.iter().map(move |origin| WindowRef {
                        ticker,
                        origin: *origin,
                    })
                })
                .collect();
            ensure!(
                !refs.is_empty(),
                "the anchored placement admitted no origin in partition {first}"
            );
            anchored.push((anchors, census, refs));
        }
        let (validation_anchors, validation_census, validation_refs) = anchored.pop().unwrap();
        let (calibration_anchors, calibration_census, calibration_refs) = anchored.pop().unwrap();
        let mean =
            |census: &[usize]| census.iter().sum::<usize>() as f64 / census.len().max(1) as f64;
        println!(
            "CausalPatch cross-section placement ANCHORED in {:.1} ms\n  \
             calibration: {} origins on {} shared anchors, mean width {:.1}, min {}, max {} \
             (strided: {} origins on {} timestamps)\n  \
             validation:  {} origins on {} shared anchors, mean width {:.1}, min {}, max {} \
             (strided: {} origins on {} timestamps)\n  \
             validation_refs sha256 {} -> {}",
            started.elapsed().as_secs_f64() * 1000.,
            calibration_refs.len(),
            calibration_anchors.len(),
            mean(&calibration_census),
            calibration_census.iter().min().copied().unwrap_or_default(),
            calibration_census.iter().max().copied().unwrap_or_default(),
            self.calibration_refs.len(),
            self.distinct_timestamps(&self.calibration_refs),
            validation_refs.len(),
            validation_anchors.len(),
            mean(&validation_census),
            validation_census.iter().min().copied().unwrap_or_default(),
            validation_census.iter().max().copied().unwrap_or_default(),
            self.validation_refs.len(),
            self.distinct_timestamps(&self.validation_refs),
            before.1,
            self.origins_sha256(&validation_refs),
        );
        ensure!(
            self.origins_sha256(&validation_refs) != before.1
                && self.origins_sha256(&calibration_refs) != before.0,
            "the anchored placement produced the digest of the strided one, so nothing keyed on \
             that digest can tell the two populations apart"
        );
        self.contract.validation_target_bars = validation_refs.len() * pred_len;
        self.contract.validation_remainder_bars = 0;
        self.contract.cross_section_placement = ANCHORED_PLACEMENT.into();
        self.contract.calibration_anchors = calibration_anchors;
        self.contract.validation_anchors = validation_anchors;
        self.calibration_refs = calibration_refs;
        self.validation_refs = validation_refs;
        Ok(())
    }

    /// Distinct wall clocks an origin list touches - the quantity that decides how many
    /// cross-sections it can form, and the one the strided placement destroys.
    pub fn distinct_timestamps(&self, refs: &[WindowRef]) -> usize {
        self.origin_timestamps(refs)
            .into_iter()
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Each reference's ORIGIN bar timestamp, in the caller's order.
    ///
    /// Gathered TICKER-MAJOR and in parallel, which is the whole reason this exists rather
    /// than a `map` at each call site. The bars are memory-mapped per ticker, so a reference
    /// list ordered by timestamp - which every anchored cross-section draw is - hops across
    /// 5,728 separate mappings and takes a cold page for essentially every element. Job 5544
    /// spent 91.5 s of a 364.3 s ceiling placement on host work outside its scored loop, and
    /// ~3.6 M of these lookups at ~25 us each is the bulk of it: that is major-fault latency,
    /// not compute. Walking one ticker's origins in ascending order instead shares one 4 KiB
    /// page across the ~512 `i64` headers that fall in it, and the remaining faults overlap
    /// across threads because latency is what they cost.
    ///
    /// ORDER-PRESERVING, not merely order-equivalent: the permutation is inverted on the way
    /// out, so the returned vector is element-for-element the one the serial `map` returned.
    /// No statistic downstream can distinguish them, which is a stronger guarantee than any
    /// tolerance argument.
    pub fn origin_timestamps(&self, refs: &[WindowRef]) -> Vec<i64> {
        let mut order: Vec<u32> = (0..refs.len() as u32).collect();
        order.par_sort_unstable_by_key(|index| {
            let reference = refs[*index as usize];
            (reference.ticker, reference.origin)
        });
        // Rayon splits an indexed parallel iterator into CONTIGUOUS ranges, so each worker
        // walks a run of one or a few tickers in ascending origin order - the access pattern
        // the sort was for. A scattered write-back would undo it, so the values come back in
        // sorted order and are permuted home serially, which is 3.6 M sequential stores.
        let gathered: Vec<i64> = order
            .par_iter()
            .map(|index| {
                let reference = refs[*index as usize];
                self.ticker(reference).timestamp(reference.origin)
            })
            .collect();
        let mut stamps = vec![0i64; refs.len()];
        for (index, stamp) in order.iter().zip(gathered) {
            stamps[*index as usize] = stamp;
        }
        stamps
    }

    pub fn prepare(&mut self, device: Device) {
        self.device = device;
    }
    pub fn ticker(&self, reference: WindowRef) -> &CorpusTicker {
        &self.tickers[reference.ticker]
    }

    /// Target bars a TRAINING origin owns, or `None` when it is not a trainable origin at all.
    ///
    /// The band test is spelled out here rather than delegated to [`CorpusTicker::target_count`]
    /// on purpose, and the difference matters: `target_count` also admits origins in the
    /// reserved calibration and validation partitions, so a consumer that MOVED a training
    /// origin - a per-row patch phase shifts each one forward by up to `patch_len - 1` bars -
    /// could push a row past `train_end` and have it silently accepted as a held-out origin,
    /// training on the very partition checkpoint selection and the amplitude fit are defined
    /// on. This admits exactly the interval `Corpus::load` enumerates and nothing else, and it
    /// applies the same [`CorpusTicker::supervision_clears_hole`] predicate, so a re-selecting
    /// consumer inherits both guarantees instead of restating either.
    pub fn training_target_count(&self, ticker: usize, origin: usize) -> Option<usize> {
        let data = self.tickers.get(ticker)?;
        let c = &data.contract;
        if origin + 1 < self.contract.common_context.max(c.context)
            || origin + 1 >= c.train_end
            || !data.supervision_clears_hole(origin)
        {
            return None;
        }
        Some(c.pred_len.min(c.train_end - origin - 1))
    }

    /// Context-only inference accepts any observed unlocked origin, independently of target
    /// availability. In particular a delisting or a missing future bar cannot select the universe.
    pub(super) fn host_forecast_batch(
        &self,
        refs: &[WindowRef],
        include_targets: bool,
    ) -> Result<Batch> {
        ensure!(!refs.is_empty(), "cannot construct an empty forecast batch");
        let context = self.contract.context;
        let pred_len = self.contract.pred_len;
        let features = &self.contract.features;
        let width = Batch::row_width(context, pred_len, features.channels());
        let sources = refs.iter().map(|reference| {
            let ticker = self.tickers.get(reference.ticker).context("unknown forecast ticker")?;
            ensure!(
                reference.origin + 1 >= context.max(self.contract.common_context)
                    && reference.origin < ticker.contract.valid_bars
                    && ticker.timestamp(reference.origin) < self.contract.boundary_timestamps[2],
                "forecast origin lacks causal context or reaches the locked terminal test"
            );
            let targets = if include_targets {
                ensure!(ticker.owns_partition_targets(reference.origin, 0)
                    || ticker.owns_partition_targets(reference.origin, 1),
                    "portfolio diagnostic labels must belong exclusively to reserved calibration or validation, never training or terminal test");
                pred_len
            } else { 0 };
            Ok((ticker, reference.origin, targets))
        }).collect::<Result<Vec<_>>>()?;
        // Known future calendar/gap inputs must be projected, not taken from future print
        // availability. Share one deterministic schedule across synchronized ticker rows.
        let mut schedules = std::collections::BTreeMap::new();
        for (ticker, origin, _) in &sources {
            schedules
                .entry(ticker.timestamp(*origin))
                .or_insert_with(|| {
                    crate::torch::dataset::forecast_schedule_after(
                        ticker.timestamp(*origin),
                        pred_len,
                        300,
                    )
                });
        }
        let packed = empty_host_rows(refs.len(), width, self.device)?;
        let output = unsafe {
            std::slice::from_raw_parts_mut(packed.data_ptr().cast::<f32>(), refs.len() * width)
        };
        self.gather_pool.install(|| {
            output
                .par_chunks_mut(width)
                .zip(sources.par_iter())
                .for_each(|(row, &(ticker, origin, targets))| {
                    fill_row::<true, true>(
                        row,
                        ticker.file.bars(),
                        &ticker.contract,
                        features,
                        &self.exogenous,
                        &ticker.ranks,
                        origin,
                        targets,
                    );
                    let length = context + pred_len;
                    let aux = &mut row[length * 5..length * (5 + features.channels())];
                    // Projected bars only: every channel the ranks feed reads `[0, 0]` beyond
                    // the context, so this cursor needs none of them.
                    let mut cursor = AuxiliaryCursor::new(
                        features,
                        &self.exogenous,
                        Some(ticker.bar(origin)),
                        &[],
                    );
                    for (bar, &timestamp) in schedules[&ticker.timestamp(origin)].iter().enumerate()
                    {
                        let scheduled = PackedBar {
                            ts_ms: timestamp,
                            ..PackedBar::default()
                        };
                        let offset = (context + bar) * features.channels();
                        cursor.write(
                            &scheduled,
                            &mut aux[offset..offset + features.channels()],
                            true,
                        );
                    }
                })
        });
        Ok(Batch::from_packed(
            packed,
            context,
            pred_len,
            features.channels(),
            if include_targets {
                refs.len() * pred_len
            } else {
                0
            },
        ))
    }

    /// Resolve each reference to its ticker and owned target count, in row order.
    fn sources(&self, refs: &[WindowRef]) -> Result<Vec<(&CorpusTicker, usize, usize)>> {
        ensure!(!refs.is_empty(), "cannot construct an empty batch");
        refs.iter()
            .map(|reference| {
                let ticker = self
                    .tickers
                    .get(reference.ticker)
                    .context("unknown batch ticker")?;
                let targets = ticker
                    .target_count(reference.origin)
                    .context("origin is outside unlocked purged targets")?;
                Ok((ticker, reference.origin, targets))
            })
            .collect()
    }

    /// The packed row block for `refs`, pinned when the corpus is prepared for a CUDA device.
    ///
    /// The block is allocated PINNED rather than allocated pageable and then pinned: at batch
    /// 256 the row block is 114,131,968 bytes, and `Tensor::empty(...).pin_memory(device)`
    /// allocates a second block of that size and memcpys the first one's uninitialized
    /// contents into it - 228 MB of single-threaded read+write traffic per batch, every byte
    /// of which [`fill_row`] then overwrites.
    pub fn host_batch(&self, refs: &[WindowRef]) -> Result<Batch> {
        let context = self.contract.context;
        let pred_len = self.contract.pred_len;
        let features = &self.contract.features;
        let width = Batch::row_width(context, pred_len, features.channels());
        let sources = self.sources(refs)?;
        let packed = empty_host_rows(refs.len(), width, self.device)?;
        // The new tensor owns this exclusive contiguous CPU allocation; workers receive disjoint rows.
        let output = unsafe {
            std::slice::from_raw_parts_mut(packed.data_ptr().cast::<f32>(), refs.len() * width)
        };
        self.gather_pool.install(|| {
            output
                .par_chunks_mut(width)
                .zip(sources.par_iter())
                .for_each(|(row, &(ticker, origin, targets))| {
                    fill_row::<true, true>(
                        row,
                        ticker.file.bars(),
                        &ticker.contract,
                        features,
                        &self.exogenous,
                        &ticker.ranks,
                        origin,
                        targets,
                    );
                })
        });
        Ok(Batch::from_packed(
            packed,
            context,
            pred_len,
            features.channels(),
            sources.iter().map(|(_, _, targets)| *targets).sum(),
        ))
    }

    pub fn batch(&self, refs: &[WindowRef], device: Device) -> Result<Batch> {
        Ok(self.host_batch(refs)?.to_device(device))
    }

    /// One rung of [`Self::audit_host_batch`]: a whole batch assembled into `base` with only
    /// the selected components enabled.
    ///
    /// `base` must address at least `sources.len() * row_width(context, pred_len,
    /// features.channels())` floats, which the audit guarantees by allocating at the widest
    /// feature set it measures.
    fn audit_pass<const LOG: bool, const MARKET_CUM: bool>(
        &self,
        base: *mut f32,
        features: &FeatureSet,
        sources: &[(&CorpusTicker, usize, usize)],
    ) {
        let width = Batch::row_width(
            self.contract.context,
            self.contract.pred_len,
            features.channels(),
        );
        let output = unsafe { std::slice::from_raw_parts_mut(base, sources.len() * width) };
        self.gather_pool.install(|| {
            output
                .par_chunks_mut(width)
                .zip(sources.par_iter())
                .for_each(|(row, &(ticker, origin, targets))| {
                    fill_row::<LOG, MARKET_CUM>(
                        row,
                        ticker.file.bars(),
                        &ticker.contract,
                        features,
                        &self.exogenous,
                        &ticker.ranks,
                        origin,
                        targets,
                    );
                })
        });
    }

    /// Charge each component of host batch assembly separately, over the real corpus, on the
    /// CPU, with no device involved.
    ///
    /// Every rung runs over the same references in the same thread pool and writes into the
    /// same allocation, so the components are commensurable with the production total, which
    /// is measured last. Rows named `marginal:` are a rung minus the rung below it; a reader
    /// must not subtract absolute rows himself, because the ladder's baseline is the gather.
    /// Rows named `removed:` are work this loader no longer does and are here so that the
    /// before/after is one measurement rather than two builds.
    pub fn audit_host_batch(&self, refs: &[WindowRef], rounds: usize) -> Result<Vec<LoaderPhase>> {
        ensure!(
            rounds > 0,
            "the loader audit needs at least one timed round"
        );
        let features = self.contract.features;
        let context = self.contract.context;
        let pred_len = self.contract.pred_len;
        let width = Batch::row_width(context, pred_len, features.channels());
        let sources = self.sources(refs)?;
        let rows = sources.len();
        let block = Tensor::empty([rows as i64, width as i64], (Kind::Float, Device::Cpu));
        let base = block.data_ptr().cast::<f32>();
        let mib = (rows * width * 4) as f64 / 1048576.;
        let mut phases = vec![
            LoaderPhase {
                name: format!("packed block allocation: Tensor::empty, {mib:.1} MiB"),
                ms: mean_ms(rounds, || {
                    let _ = Tensor::empty([rows as i64, width as i64], (Kind::Float, Device::Cpu));
                }),
            },
            LoaderPhase {
                name: format!(
                    "removed: pin_memory's staging copy of the whole block, {mib:.1} MiB read + written"
                ),
                ms: mean_ms(rounds, || {
                    let _ = block.copy();
                }),
            },
            LoaderPhase {
                name: format!("removed: row.fill(0.0) over every row, {mib:.1} MiB stored"),
                ms: mean_ms(rounds, || {
                    let output =
                        unsafe { std::slice::from_raw_parts_mut(base, rows * width) };
                    self.gather_pool.install(|| {
                        output.par_chunks_mut(width).for_each(|row| row.fill(0.0));
                    });
                }),
            },
        ];
        let gather = mean_ms(rounds, || {
            self.audit_pass::<false, false>(base, &FeatureSet::NONE, &sources)
        });
        phases.push(LoaderPhase {
            name: "window gather: valid-bar walk, raw OHLC store, validity flag".into(),
            ms: gather,
        });
        phases.push(LoaderPhase {
            name: "marginal: f64 ln of every OHLC element minus the anchor's".into(),
            ms: mean_ms(rounds, || {
                self.audit_pass::<true, false>(base, &FeatureSet::NONE, &sources)
            }) - gather,
        });
        phases.push(LoaderPhase {
            name: "marginal: cumulative market level looked up per bar".into(),
            ms: mean_ms(rounds, || {
                self.audit_pass::<false, true>(base, &FeatureSet::NONE, &sources)
            }) - gather,
        });
        for feature in features.features() {
            let one = one_feature(feature);
            phases.push(LoaderPhase {
                name: format!("marginal: aux {} (2 channels)", feature.name()),
                ms: mean_ms(rounds, || {
                    self.audit_pass::<false, false>(base, &one, &sources)
                }) - gather,
            });
        }
        phases.push(LoaderPhase {
            name: "production host_batch total, everything enabled".into(),
            ms: mean_ms(rounds, || {
                let _ = self.host_batch(refs);
            }),
        });
        Ok(phases)
    }
}

/// One component of host batch assembly, in milliseconds per batch of the audited shape.
pub struct LoaderPhase {
    pub name: String,
    pub ms: f64,
}

/// Mean milliseconds of `rounds` timed executions, after one discarded warmup.
///
/// The warmup matters more than usual here: the first pass over a freshly allocated block
/// faults in its pages, and the first pass over a window set faults in the mapped bar file's
/// pages, so an unwarmed first round measures the page fault handler.
fn mean_ms(rounds: usize, mut run: impl FnMut()) -> f64 {
    run();
    let started = Instant::now();
    for _ in 0..rounds {
        run();
    }
    started.elapsed().as_secs_f64() * 1000. / rounds as f64
}

/// The feature set enabling exactly `feature`.
fn one_feature(feature: Feature) -> FeatureSet {
    let mut set = FeatureSet::NONE;
    match feature {
        Feature::TimeOfDay => set.time_of_day = true,
        Feature::DayOfWeek => set.day_of_week = true,
        Feature::SessionGap => set.session_gap = true,
        Feature::Volume => set.volume = true,
        Feature::Market => set.market = true,
        Feature::Spy => set.spy = true,
        Feature::Dispersion => set.dispersion = true,
        Feature::CrossSectionZ => set.cross_section_z = true,
        Feature::CrossSectionRank => set.cross_section_rank = true,
        Feature::RelativeVolume => set.relative_volume = true,
        Feature::RangeZ => set.range_z = true,
    }
    set
}

/// The packed row block for one batch: pinned when the corpus feeds a CUDA device so the
/// upload can be asynchronous, pageable otherwise.
fn empty_host_rows(rows: usize, width: usize, device: Device) -> Result<Tensor> {
    let size = [rows as i64, width as i64];
    if device.is_cuda() {
        crate::torch::cuda::empty_pinned(&size)
            .map_err(|err| anyhow::anyhow!("allocating a pinned packed row block: {err}"))
    } else {
        Ok(Tensor::empty(size, (Kind::Float, Device::Cpu)))
    }
}

fn digest_hex(digest: ring::digest::Digest) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    digest_hex(ring::digest::digest(&ring::digest::SHA256, bytes))
}

fn verify_split_edges(
    symbol: &str,
    bars: &[PackedBar],
    edges: [u64; 3],
    bounds: [i64; 3],
) -> Result<()> {
    ensure!(
        bounds.windows(2).all(|pair| pair[0] < pair[1]),
        "shared UTC split bounds must increase"
    );
    for (edge, bound) in edges.into_iter().zip(bounds) {
        let edge = usize::try_from(edge).context("cached split edge exceeds address space")?;
        ensure!(edge <= bars.len()
            && (edge == 0 || bars[edge - 1].ts() < bound)
            && (edge == bars.len() || bars[edge].ts() >= bound),
            "{symbol}: cached split edge {edge} does not bracket UTC bound {bound}; refusing stale or corrupt lineage");
    }
    Ok(())
}

fn market_fingerprint(
    tickers: &[CorpusTicker],
    bounds: [i64; 3],
    min_cross_section: usize,
) -> String {
    let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
    for ticker in tickers {
        digest.update(ticker.contract.ticker.as_bytes());
        digest.update(b":");
        digest.update(ticker.contract.fingerprint.as_bytes());
        digest.update(b";");
    }
    for bound in bounds {
        digest.update(&bound.to_le_bytes());
    }
    digest.update(b"market-steps-min-cross-section:");
    digest.update(&(min_cross_section as u64).to_le_bytes());
    digest
        .finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The shared five-minute grid the exogenous variates and the market steps live on: the span of
/// the ELIGIBLE tickers, which is a subrange of the scanned universe's span.
fn ticker_grid(tickers: &[CorpusTicker]) -> Result<Grid> {
    let first = tickers
        .iter()
        .filter_map(|ticker| ticker.file.first_ts_ms())
        .min()
        .context("empty exogenous timestamp universe")?;
    let last = tickers
        .iter()
        .filter_map(|ticker| ticker.file.last_ts_ms())
        .max()
        .context("empty exogenous timestamp universe")?;
    Grid::new(first, last)
}

fn exogenous_series(
    directory: &Path,
    features: &FeatureSet,
    grid: Grid,
    steps: MarketSteps,
) -> Result<(Exogenous, MarketSummary)> {
    let mut exogenous = Exogenous {
        market: features.market.then(|| steps.series()),
        spy: None,
        cross_section: features.cross_section().then(|| steps.cross_section()),
        market_cum: steps.path(),
    };
    if features.spy {
        let file = BarFile::open(&bar_file_path(directory, SPY, 300))
            .with_context(|| format!("loading the {SPY} exogenous variate"))?;
        ensure!(
            file.symbol() == SPY && file.res_secs() == 300 && !file.is_empty(),
            "{SPY} exogenous corpus header mismatch or empty"
        );
        let bars = file.bars();
        for (index, bar) in bars.iter().enumerate() {
            ensure!(
                (index == 0 || bars[index - 1].ts() < bar.ts())
                    && bar.ts().rem_euclid(RESOLUTION_MS) == 0,
                "{SPY}: timestamps must be strictly increasing on the five-minute grid at raw bar {index}"
            );
        }
        let invalid: Vec<usize> = bars
            .iter()
            .enumerate()
            .filter_map(|(index, bar)| (!valid_ohlc(bar)).then_some(index))
            .collect();
        exogenous.spy = Some(single_series(bars, &invalid, grid));
    }
    Ok((exogenous, steps.summary()))
}

/// Write one packed row.
///
/// The const parameters exist for [`Corpus::audit_host_batch`], which charges each component
/// of assembly separately by instantiating subsets over the same corpus, the same windows and
/// the same thread pool. They are monomorphized, so the production instantiation
/// `fill_row::<true, true>` is exactly the loop this was before the audit existed and pays no
/// branch per bar. Nothing but the audit may instantiate anything else.
///
/// Only the tail beyond the row's written bars is zeroed. Every element at a position the
/// loop below reaches is assigned - four prices, the validity flag, every auxiliary channel
/// (`AuxiliaryCursor::write` fills its whole slice), and the market level - so zeroing the
/// block first wrote 114,131,968 bytes per batch at batch 256 to overwrite all but the tail.
/// A row whose origin owns a full `pred_len` of targets, which is every dense training
/// origin, has NO tail and is now zeroed not at all.
fn fill_row<const LOG: bool, const MARKET_CUM: bool>(
    row: &mut [f32],
    bars: &[PackedBar],
    contract: &DataContract,
    features: &FeatureSet,
    exogenous: &Exogenous,
    // The ticker's whole per-valid-bar rank array; the row's own window is sliced off here.
    ranks: &[u16],
    origin: usize,
    targets: usize,
) {
    let context = contract.context;
    let length = context + contract.pred_len;
    let start = origin + 1 - context;
    let aux_channels = features.channels();
    let invalid = &contract.invalid_ohlc_indices;
    let written = context + targets;
    let (prices, rest) = row.split_at_mut(length * 4);
    let (valid, rest) = rest.split_at_mut(length);
    let (aux, rest) = rest.split_at_mut(length * aux_channels);
    let (market_cum, anchor) = rest.split_at_mut(length);
    if written < length {
        prices[written * 4..].fill(0.0);
        valid[written..].fill(0.0);
        aux[written * aux_channels..].fill(0.0);
        market_cum[written..].fill(0.0);
    }
    let market_anchor = exogenous
        .market_cum
        .at(bars[raw_index(origin, invalid)].ts());
    let c_last = f64::from(bars[raw_index(origin, invalid)].close);
    let ln_anchor = c_last.ln();
    let mut auxiliary = AuxiliaryCursor::new(
        features,
        exogenous,
        start.checked_sub(1).map(|i| &bars[raw_index(i, invalid)]),
        ranks.get(start..).unwrap_or_default(),
    );
    for (position, bar) in ValidBars::new(bars, invalid, start, written).enumerate() {
        for (channel, value) in [bar.open, bar.high, bar.low, bar.close]
            .into_iter()
            .enumerate()
        {
            prices[position * 4 + channel] = if LOG {
                (f64::from(value).ln() - ln_anchor) as f32
            } else {
                value
            };
        }
        valid[position] = 1.0;
        if MARKET_CUM {
            market_cum[position] = (exogenous.market_cum.at(bar.ts()) - market_anchor) as f32;
        }
        if aux_channels > 0 {
            let offset = position * aux_channels;
            auxiliary.write(
                bar,
                &mut aux[offset..offset + aux_channels],
                position >= context,
            );
        }
    }
    anchor[0] = c_last as f32;
}

fn raw_index(logical_index: usize, invalid: &[usize]) -> usize {
    let (mut left, mut right) = (0, invalid.len());
    while left < right {
        let middle = left + (right - left) / 2;
        if invalid[middle] - middle <= logical_index {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    logical_index + left
}

struct ValidBars<'a> {
    bars: &'a [PackedBar],
    invalid: &'a [usize],
    invalid_cursor: usize,
    raw: usize,
    remaining: usize,
}

impl<'a> ValidBars<'a> {
    fn new(bars: &'a [PackedBar], invalid: &'a [usize], start: usize, length: usize) -> Self {
        let raw = raw_index(start, invalid);
        Self {
            bars,
            invalid,
            invalid_cursor: raw - start,
            raw,
            remaining: length,
        }
    }
}

impl<'a> Iterator for ValidBars<'a> {
    type Item = &'a PackedBar;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        while self.invalid.get(self.invalid_cursor) == Some(&self.raw) {
            self.raw += 1;
            self.invalid_cursor += 1;
        }
        let bar = &self.bars[self.raw];
        self.raw += 1;
        self.remaining -= 1;
        Some(bar)
    }
}

/// The scanned universe's five-minute grid: every non-empty corpus file's span, and the bitmap
/// width that covers it. Derived from headers alone, so it costs one mapped page per file.
#[derive(Clone, Copy)]
struct UniverseSpan {
    first: i64,
    slots: usize,
    words: usize,
}

impl UniverseSpan {
    fn of(files: &[BarFile]) -> Result<Self> {
        let first = files
            .iter()
            .filter_map(BarFile::first_ts_ms)
            .min()
            .context("empty timestamp universe")?;
        let last = files
            .iter()
            .filter_map(BarFile::last_ts_ms)
            .max()
            .context("empty timestamp universe")?;
        ensure!(
            first.rem_euclid(RESOLUTION_MS) == 0 && last.rem_euclid(RESOLUTION_MS) == 0,
            "off-grid universe timestamps"
        );
        let slots = usize::try_from((last - first) / RESOLUTION_MS + 1)?;
        ensure!(
            slots <= 20_000_000,
            "five-minute corpus spans more than 190 years"
        );
        Ok(Self {
            first,
            slots,
            words: slots.div_ceil(64),
        })
    }
}

/// Mark every valid bar's slot in `bits`. Shared by both traversals so the bitmap means exactly
/// one thing however a run arrives at it.
fn mark_occupancy(
    bits: &mut [u64],
    symbol: &str,
    bars: &[PackedBar],
    invalid: &[usize],
    span: UniverseSpan,
) -> Result<()> {
    let mut quarantined = invalid.iter().copied().peekable();
    for (index, bar) in bars.iter().enumerate() {
        let offset = bar.ts() - span.first;
        ensure!(
            offset >= 0 && offset % RESOLUTION_MS == 0,
            "off-grid timestamp in {symbol}"
        );
        if quarantined.peek() == Some(&index) {
            quarantined.next();
            continue;
        }
        let slot = usize::try_from(offset / RESOLUTION_MS)?;
        ensure!(slot < span.slots, "timestamp outside header span");
        bits[slot / 64] |= 1 << (slot % 64);
    }
    Ok(())
}

/// The fused traversal: for each file in `stale`, one walk of its records that produces BOTH its
/// [`BarAudit`] and its contribution to the shared occupancy bitmap. These used to be two
/// separate passes over the same 17 GB for no reason but ordering.
fn scan(
    files: &[BarFile],
    stale: &[usize],
    span: UniverseSpan,
) -> Result<(Vec<u64>, Vec<(usize, super::data::BarAudit)>)> {
    stale
        .par_iter()
        .try_fold(
            || (vec![0u64; span.words], Vec::new()),
            |(mut bits, mut audited), &index| -> Result<_> {
                let file = &files[index];
                let bars = file.bars();
                let audit = audit_bars(file.symbol(), bars).with_context(|| {
                    format!(
                        "authenticating five-minute ticker {} from {}",
                        file.symbol(),
                        file.path().display()
                    )
                })?;
                mark_occupancy(
                    &mut bits,
                    file.symbol(),
                    bars,
                    &audit.invalid_ohlc_indices,
                    span,
                )?;
                audited.push((index, audit));
                Ok((bits, audited))
            },
        )
        .try_reduce(
            || (vec![0u64; span.words], Vec::new()),
            |(mut bits, mut audited), (other_bits, other_audited)| {
                for (word, other) in bits.iter_mut().zip(other_bits) {
                    *word |= other;
                }
                audited.extend(other_audited);
                Ok((bits, audited))
            },
        )
}

/// Occupancy only, for files whose audit was served from the ledger but whose slots the
/// boundary quantiles still need. Empty on a cold corpus; everything but the changed tickers
/// after an ingest, because a partition boundary is a quantile of the union and no incremental
/// update of it exists.
fn occupy(
    files: &[BarFile],
    owed: &[usize],
    audits: &[super::data::BarAudit],
    span: UniverseSpan,
) -> Result<Vec<u64>> {
    owed.par_iter()
        .try_fold(
            || vec![0u64; span.words],
            |mut bits, &index| -> Result<_> {
                let file = &files[index];
                mark_occupancy(
                    &mut bits,
                    file.symbol(),
                    file.bars(),
                    &audits[index].invalid_ohlc_indices,
                    span,
                )?;
                Ok(bits)
            },
        )
        .try_reduce(
            || vec![0u64; span.words],
            |mut left, right| {
                for (word, other) in left.iter_mut().zip(right) {
                    *word |= other;
                }
                Ok(left)
            },
        )
}

fn quantile_bounds(occupied: &[u64], first: i64) -> Result<[i64; 3]> {
    let count: usize = occupied.iter().map(|word| word.count_ones() as usize).sum();
    ensure!(
        count >= 10,
        "too few distinct timestamps for four chronological partitions"
    );
    let ranks = [count * 7 / 10, count * 8 / 10, count * 9 / 10];
    let mut result = [0; 3];
    let mut seen = 0;
    let mut boundary = 0;
    for (word_index, &word) in occupied.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            if boundary < 3 && seen == ranks[boundary] {
                result[boundary] = first + (word_index * 64 + bit) as i64 * RESOLUTION_MS;
                boundary += 1;
            }
            seen += 1;
            remaining &= remaining - 1;
        }
    }
    ensure!(boundary == 3, "incomplete global timestamp quantiles");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_quantiles_count_calendar_positions_once() {
        let bits = [(1u64 << 20) - 1];
        assert_eq!(
            quantile_bounds(&bits, 300_000).unwrap(),
            [15 * 300_000, 17 * 300_000, 19 * 300_000]
        );
        let sparse = [0x55555];
        assert_eq!(
            quantile_bounds(&sparse, 0).unwrap(),
            [14 * 300_000, 16 * 300_000, 18 * 300_000]
        );
    }

    #[test]
    fn sparse_invalid_mapping_matches_valid_observed_order_for_every_small_pattern() {
        let bars = (0..8)
            .map(|i| PackedBar {
                ts_ms: i,
                close: i as f32,
                ..PackedBar::default()
            })
            .collect::<Vec<_>>();
        for pattern in 0usize..256 {
            let invalid = (0..8)
                .filter(|i| pattern & (1 << i) != 0)
                .collect::<Vec<_>>();
            let expected = (0..8).filter(|i| !invalid.contains(i)).collect::<Vec<_>>();
            for (logical, &raw) in expected.iter().enumerate() {
                assert_eq!(raw_index(logical, &invalid), raw);
            }
            for start in 0..=expected.len() {
                let actual = ValidBars::new(&bars, &invalid, start, expected.len() - start)
                    .map(|bar| bar.ts() as usize)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected[start..]);
            }
        }
    }

    #[test]
    fn pooled_rows_preserve_ticker_values_and_mask_partial_targets() {
        let bars: Vec<_> = (0..20)
            .map(|i| PackedBar {
                ts_ms: i * RESOLUTION_MS,
                open: i as f32 + 1.0,
                high: i as f32 + 3.0,
                low: i as f32 + 0.5,
                close: i as f32 + 2.0,
                volume: 100.0 + i as f32,
                ..Default::default()
            })
            .collect();
        let contract = DataContract {
            schema: String::new(),
            ticker: "ONE".into(),
            fingerprint: String::new(),
            source_bars: bars.len(),
            valid_bars: bars.len(),
            invalid_ohlc_indices: Vec::new(),
            boundaries: [10, 13, 17],
            boundary_timestamps: [0; 3],
            context: 4,
            pred_len: 3,
            purge: 1,
            train_end: 9,
            common_context: 4,
        };
        let volume = FeatureSet {
            volume: true,
            ..FeatureSet::NONE
        };
        let exogenous = Exogenous {
            market: None,
            spy: None,
            cross_section: None,
            market_cum: market_steps(
                &[(&bars, &[])],
                Grid::new(0, 19 * RESOLUTION_MS).unwrap(),
                1,
            )
            .path(),
        };
        // NaN-initialized, not zero-initialized, on purpose: `fill_row` zeroes only the tail
        // beyond the row's written bars, so every assertion below is also a claim that the
        // element it reads WAS assigned. A pre-zeroed row would pass whether or not it was.
        let mut row = vec![f32::NAN; Batch::row_width(4, 3, 2)];
        assert_eq!(row.len(), 57);
        fill_row::<true, true>(&mut row, &bars, &contract, &volume, &exogenous, &[], 6, 2);
        assert_eq!(row[3], (5.0f64 / 8.0).ln() as f32);
        assert_eq!(row[0], (4.0f64 / 8.0).ln() as f32);
        assert_eq!(row[15], 0.0);
        assert_eq!(row[19], (9.0f64 / 8.0).ln() as f32);
        assert_eq!(&row[24..28], &[0.0; 4]);
        assert_eq!(&row[28..35], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(row[35], (103.0f64.ln() - 102.0f64.ln()) as f32);
        assert_eq!(row[36], 1.0);
        assert_eq!(&row[41..43], &[(106.0f64.ln() - 105.0f64.ln()) as f32, 1.0]);
        assert_eq!(
            &row[43..49],
            &[0.0; 6],
            "history channels leak beyond the context"
        );
        for position in 0..6 {
            assert!(
                (row[49 + position] - row[position * 4 + 3]).abs() <= 1e-6,
                "a one-ticker universe's market path is the ticker's own close path"
            );
        }
        assert_eq!(row[49 + 3], 0.0);
        assert_eq!(row[55], 0.0, "masked horizon bars carry no market path");
        assert_eq!(row[56], 8.0);
        let batch = Batch::from_packed(
            Tensor::from_slice(&row).reshape([1, row.len() as i64]),
            4,
            3,
            2,
            2,
        )
        .to_device(Device::Cpu);
        assert_eq!(batch.log_prices.size(), [1, 7, 4]);
        assert_eq!(batch.valid.size(), [1, 7]);
        assert_eq!(batch.aux.size(), [1, 7, 2]);
        assert_eq!(batch.anchor.size(), [1]);
        assert_eq!(batch.anchor.double_value(&[0]), 8.0);
        assert_eq!(
            Vec::<f32>::try_from(batch.valid.reshape([-1])).unwrap(),
            &row[28..35]
        );
        assert_eq!(
            batch.log_prices.double_value(&[0, 4, 3]),
            f64::from(row[19])
        );
        let mut doubled_bars = bars.clone();
        for bar in &mut doubled_bars {
            bar.open *= 2.0;
            bar.high *= 2.0;
            bar.low *= 2.0;
            bar.close *= 2.0;
        }
        let mut doubled = vec![f32::NAN; row.len()];
        fill_row::<true, true>(
            &mut doubled,
            &doubled_bars,
            &contract,
            &volume,
            &exogenous,
            &[],
            6,
            2,
        );
        assert_eq!(
            &doubled[..56],
            &row[..56],
            "representation must be scale-free"
        );
        assert_eq!(doubled[56], 16.0);
        let mut future_bars = bars.clone();
        for bar in &mut future_bars[7..] {
            bar.open *= 1000.0;
            bar.high *= 1000.0;
            bar.low *= 1000.0;
            bar.close *= 1000.0;
        }
        let mut changed = vec![f32::NAN; row.len()];
        fill_row::<true, true>(
            &mut changed,
            &future_bars,
            &contract,
            &volume,
            &exogenous,
            &[],
            6,
            2,
        );
        assert_eq!(
            &changed[..16],
            &row[..16],
            "future targets changed historical inputs"
        );
        assert_eq!(changed[56], row[56], "future targets changed the anchor");
        let mut flat_bars = bars.clone();
        for bar in &mut flat_bars[..7] {
            bar.open = 8.0;
            bar.high = 8.0;
            bar.low = 8.0;
            bar.close = 8.0;
        }
        fill_row::<true, true>(
            &mut changed,
            &flat_bars,
            &contract,
            &volume,
            &exogenous,
            &[],
            6,
            2,
        );
        assert_eq!(&changed[..16], &[0.0; 16]);
        assert_eq!(changed[19], (9.0f64 / 8.0).ln() as f32);
    }

    #[test]
    fn pooled_epoch_keeps_delisted_tickers_and_covers_each_training_target_once() {
        use shared::bars::{bar_file_path, write_bar_file};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-corpus-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let bars: Vec<_> = (0..10_000)
            .map(|i| {
                let price = 100.0 + i as f32 * 0.01;
                PackedBar {
                    ts_ms: 1_500_000_000_000 + i * RESOLUTION_MS,
                    open: price,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price + 0.5,
                    volume: 100.0,
                    vwap: price,
                    trades: 1,
                }
            })
            .collect();
        for (symbol, source) in [
            ("ALIVE", bars.as_slice()),
            ("DELISTED", &bars[..5_000]),
            ("MINIMAL", &bars[..33]),
            ("NEW", &bars[8_000..]),
        ] {
            write_bar_file(
                &bar_file_path(&directory.0, symbol, 300),
                symbol,
                300,
                source,
            )
            .unwrap();
        }
        let mut damaged = bars[..5_000].to_vec();
        for index in [0, 1, 47, 48, 49, 4000] {
            damaged[index].open = 0.0;
            damaged[index].high = 0.0;
            damaged[index].low = 0.0;
            damaged[index].close = 0.0;
        }
        write_bar_file(
            &bar_file_path(&directory.0, "DAMAGED", 300),
            "DAMAGED",
            300,
            &damaged,
        )
        .unwrap();
        let mut invalid = bars.clone();
        for bar in &mut invalid {
            bar.ts_ms -= 10_000 * RESOLUTION_MS;
            bar.close = f32::NAN;
        }
        write_bar_file(
            &bar_file_path(&directory.0, "INVALID", 300),
            "INVALID",
            300,
            &invalid,
        )
        .unwrap();
        let ancient = bars[..2]
            .iter()
            .enumerate()
            .map(|(index, bar)| PackedBar {
                ts_ms: bars[0].ts() - (2 - index) as i64 * RESOLUTION_MS,
                ..*bar
            })
            .collect::<Vec<_>>();
        write_bar_file(
            &bar_file_path(&directory.0, "ANCIENT", 300),
            "ANCIENT",
            300,
            &ancient,
        )
        .unwrap();
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let corpus = Corpus::load(&directory.0, &[], 16, 7, 32, &features, 1, 0).unwrap();
        assert_eq!(corpus.contract.market_min_cross_section, 1);
        assert_eq!(corpus.contract.features, features);
        assert!(corpus.contract.spy_fingerprint.is_none());
        assert_eq!(
            corpus
                .contract
                .tickers
                .iter()
                .map(|t| t.ticker.as_str())
                .collect::<Vec<_>>(),
            ["ALIVE", "DAMAGED", "DELISTED", "MINIMAL"]
        );
        assert!(corpus.excluded_tickers.iter().any(|t| t.ticker == "NEW"));
        assert!(corpus.validation_refs.iter().all(|r| r.ticker == 0));
        assert_eq!(corpus.contract.boundary_timestamps[0], bars[6999].ts());
        assert_eq!(corpus.contract.tickers[0].train_end, 6999 - 100);
        assert_eq!(corpus.contract.tickers[1].train_end, 4994);
        assert_eq!(
            corpus.contract.tickers[1].invalid_ohlc_indices,
            [0, 1, 47, 48, 49, 4000]
        );
        assert_eq!(corpus.contract.tickers[2].train_end, 5000);
        assert_eq!(corpus.contract.tickers[3].train_end, 33);
        assert!(corpus
            .excluded_tickers
            .iter()
            .any(|ticker| ticker.ticker == "INVALID"));
        let reference = WindowRef {
            ticker: 1,
            origin: 46,
        };
        let source = corpus.ticker(reference);
        assert_eq!(source.timestamp(0), bars[2].ts());
        assert_eq!(source.timestamp(46), bars[51].ts());
        let actual = source.candle_window(46, 16, 7).unwrap();
        let valid_raw = (0..5000)
            .filter(|i| !source.contract.invalid_ohlc_indices.contains(i))
            .collect::<Vec<_>>();
        for (candle, &raw) in actual.iter().zip(&valid_raw[30..54]) {
            assert_eq!(candle.close, bars[raw].close);
        }
        let selected = corpus.host_batch(&[reference]).unwrap();
        let log_prices: Vec<f32> = Vec::try_from(selected.log_prices.reshape([-1])).unwrap();
        let valid: Vec<f32> = Vec::try_from(selected.valid.reshape([-1])).unwrap();
        assert_eq!(log_prices.len(), 23 * 4);
        assert_eq!(valid, vec![1.0; 23]);
        let closes: Vec<f64> = valid_raw[31..54]
            .iter()
            .map(|&raw| f64::from(bars[raw].close))
            .collect();
        let c_last = closes[15];
        assert_eq!(selected.anchor.double_value(&[0]), c_last);
        let market_cum: Vec<f32> = Vec::try_from(selected.market_cum.reshape([-1])).unwrap();
        assert_eq!(market_cum.len(), 23);
        assert_eq!(market_cum[15], 0.0);
        assert!(market_cum[..23].iter().all(|v| v.is_finite()));
        assert_eq!(corpus.contract.market_fingerprint.len(), 64);
        assert_eq!(log_prices[15 * 4 + 3], 0.0);
        for (position, close) in closes.iter().enumerate() {
            let expected = (close / c_last).ln();
            assert!((f64::from(log_prices[position * 4 + 3]) - expected).abs() <= 1e-7);
        }
        let auxiliary: Vec<f32> = Vec::try_from(selected.aux.reshape([-1])).unwrap();
        assert_eq!(auxiliary.len(), 23 * 20);
        for (position, &raw) in valid_raw[31..54].iter().enumerate() {
            let channels = &auxiliary[position * 20..position * 20 + 20];
            assert!(channels[..4].iter().all(|value| value.abs() <= 1.0));
            let gap = if raw == 50 {
                [1.0, 4.0f32.ln()]
            } else {
                [0.0, 0.0]
            };
            assert_eq!(&channels[4..6], &gap);
            if position < 16 {
                assert_eq!(&channels[6..8], &[0.0, 1.0]);
                let market =
                    (f64::from(bars[raw].close) / f64::from(bars[raw - 1].close)).ln() as f32;
                assert!((channels[8] - market).abs() < 1e-7 && channels[9] == 1.0);
                // The dispersion is defined on exactly the slots the market step is, so it is
                // live wherever the market channel is, and `ln σ` of a five-minute dispersion
                // is negative. The z needs an own five-minute return as well, so the bar after
                // the four-interval gap carries none even though its slot has a dispersion.
                assert!(channels[10] < 0.0 && channels[11] == 1.0);
                assert_eq!(channels[13], if raw == 50 { 0.0 } else { 1.0 });
                // The rank carries the z's validity rule exactly. Every contributor here
                // prints the SAME return, so each takes the middle plotting position and the
                // score is zero WHERE DEFINED - not merely blank.
                assert!(channels[14].abs() < 1e-6 && channels[15] == channels[13]);
                // Every ticker in this fixture prints the identical tape and the identical
                // range, so both level cross-sections have zero spread and neither is a
                // measurement: [0, 0], never a zero z a head could read as "average".
                assert_eq!(&channels[16..20], &[0.0; 4]);
            } else {
                assert_eq!(
                    &channels[6..20],
                    &[0.0; 14],
                    "future history channels must be blank"
                );
            }
        }
        let minimal = corpus
            .train_refs
            .iter()
            .filter(|r| r.ticker == 3)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            minimal,
            [WindowRef {
                ticker: 3,
                origin: 31
            }]
        );
        assert_eq!(corpus.host_batch(&minimal).unwrap().valid_target_bars, 1);

        let mut total = 0;
        for (ticker, metadata) in corpus.contract.tickers.iter().enumerate() {
            let mut coverage = vec![0u8; metadata.valid_bars];
            for reference in corpus.train_refs.iter().filter(|r| r.ticker == ticker) {
                let count = corpus
                    .ticker(*reference)
                    .target_count(reference.origin)
                    .unwrap();
                for used in &mut coverage[reference.origin + 1..reference.origin + 1 + count] {
                    *used += 1;
                }
                total += count;
            }
            assert!(coverage[..32].iter().all(|&v| v == 0));
            assert!(coverage[32..metadata.train_end].iter().all(|&v| v == 1));
            assert!(coverage[metadata.train_end..].iter().all(|&v| v == 0));
        }
        assert_eq!(total, corpus.contract.train_target_bars);
        let refs = [corpus.train_refs[0], *corpus.train_refs.last().unwrap()];
        let batch = corpus.host_batch(&refs).unwrap();
        assert_eq!(batch.valid.size(), [2, 23]);
        assert_eq!(batch.anchor.size(), [2]);
        assert!(batch.anchor.gt(0.0).all().int64_value(&[]) == 1);
        assert_eq!(
            batch
                .valid
                .narrow(1, 16, 7)
                .sum(Kind::Float)
                .int64_value(&[]) as usize,
            batch.valid_target_bars
        );
        assert_eq!(
            batch.valid_target_bars,
            refs.iter()
                .map(|r| corpus.ticker(*r).target_count(r.origin).unwrap())
                .sum::<usize>()
        );
    }

    /// The whole justification for caching startup at all: a cache built against one corpus
    /// contract must be REJECTED when the contract changes, never silently reused.
    ///
    /// Three axes, because they fail differently. Changing a ticker's BYTES has to invalidate
    /// the audit that authenticated them, and the only thing standing between a rewritten file
    /// and a stale digest is the mapped inode's identity. Changing the corpus GEOMETRY - here
    /// the forecast length, which moves the purge and therefore the eligible universe - has to
    /// invalidate the market grid while leaving the byte-level audits alone, because those two
    /// layers are keyed on different things on purpose. And in both cases the answer a warm run
    /// gives has to be the answer a cache-less run gives, which is the assertion that actually
    /// proves the cache is not lying: the whole contract is compared, not a summary of it.
    #[test]
    fn a_cache_built_against_one_corpus_contract_is_rejected_when_the_contract_changes() {
        use shared::bars::{bar_file_path, write_bar_file};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-cache-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let series = |offset: f32| -> Vec<PackedBar> {
            (0..4_000)
                .map(|i| {
                    let price = 100.0 + offset + i as f32 * 0.01;
                    PackedBar {
                        ts_ms: 1_500_000_000_000 + i * RESOLUTION_MS,
                        open: price,
                        high: price + 1.0,
                        low: price - 1.0,
                        close: price + 0.5,
                        volume: 1_000.0,
                        vwap: price,
                        trades: 10,
                    }
                })
                .collect()
        };
        for (symbol, offset) in [("AAA", 0.0), ("BBB", 5.0), ("CCC", 9.0)] {
            write_bar_file(
                &bar_file_path(&directory.0, symbol, 300),
                symbol,
                300,
                &series(offset),
            )
            .unwrap();
        }
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let load = |pred_len: usize, min_cross_section: usize| {
            Corpus::load(
                &directory.0,
                &[],
                16,
                pred_len,
                32,
                &features,
                min_cross_section,
                0,
            )
            .unwrap()
        };

        let cold = load(7, 1);
        assert_eq!(
            cold.timing.audits_computed, 3,
            "a cold ledger audits every ticker"
        );
        assert_eq!(cold.timing.audits_reused, 0);
        // One audit pass, two market-grid passes, three more for the cross-section ranks.
        assert_eq!(
            cold.timing.bars_rescanned,
            3 * 4_000 + 2 * 3 * 4_000 + 3 * 3 * 4_000
        );
        assert!(!cold.timing.bounds_reused && !cold.timing.market_reused);

        let warm = load(7, 1);
        assert_eq!(
            warm.timing.audits_reused, 3,
            "an untouched corpus rehashes nothing"
        );
        assert_eq!(warm.timing.audits_computed, 0);
        assert_eq!(
            warm.timing.bars_rescanned, 0,
            "a warm run touches no record at all"
        );
        assert!(warm.timing.bounds_reused && warm.timing.market_reused);
        assert_eq!(
            warm.contract, cold.contract,
            "a cache may not change the answer"
        );
        assert!(
            warm.timing.bar_audit_ms.is_nan(),
            "a phase that did not run is NaN, not zero"
        );
        assert!(warm.timing.shared_bounds_ms.is_nan());
        assert!(warm.timing.market_grid_ms.is_nan());

        // A shape-valid cached edge can still point at the wrong wall clock. It is a miss,
        // not a fatal startup error or a request for the operator to delete a cache directory.
        let universe: Vec<_> = warm
            .tickers
            .iter()
            .map(|ticker| {
                (
                    ticker.contract.ticker.as_str(),
                    ticker.contract.fingerprint.as_str(),
                )
            })
            .collect();
        let bounds = warm.contract.boundary_timestamps;
        let mut corrupt_edges: Vec<_> = warm
            .tickers
            .iter()
            .map(|ticker| bounds.map(|bound| ticker.file.index_at_or_after(bound) as u64))
            .collect();
        corrupt_edges[0][1] += 1;
        cache::BoundsCache::open(&directory.0, SCHEMA, &universe)
            .store(bounds, &corrupt_edges)
            .unwrap();
        let repaired = load(7, 1);
        assert_eq!(repaired.contract, warm.contract);
        assert!(!repaired.timing.bounds_reused);
        assert_eq!(repaired.timing.audits_computed, 0);
        assert!(load(7, 1).timing.bounds_reused);

        // Axis 1: the bytes change. `write_bar_file` publishes by rename, so the inode the
        // stored audit was taken from no longer exists and its digest cannot be reused.
        let mut edited = series(5.0);
        edited[2_500].close += 0.25;
        write_bar_file(
            &bar_file_path(&directory.0, "BBB", 300),
            "BBB",
            300,
            &edited,
        )
        .unwrap();
        let changed = load(7, 1);
        assert_eq!(
            changed.timing.audits_computed, 1,
            "only the rewritten ticker rehashes"
        );
        assert_eq!(changed.timing.audits_reused, 2);
        assert!(
            !changed.timing.bounds_reused,
            "a moved fingerprint invalidates the boundaries"
        );
        assert!(!changed.timing.market_reused);
        assert_ne!(
            changed.contract.tickers[1].fingerprint, cold.contract.tickers[1].fingerprint,
            "the rewritten ticker must not keep the digest of the bytes it replaced"
        );
        assert_eq!(
            changed.contract.tickers[0].fingerprint, cold.contract.tickers[0].fingerprint,
            "an untouched ticker's digest must survive its neighbour's rewrite"
        );

        // The load-bearing comparison: what the warm-but-invalidated run produced is exactly
        // what a run with no cache at all produces.
        fs::remove_dir_all(directory.0.join(".timexer-cache")).unwrap();
        let scratch = load(7, 1);
        assert_eq!(scratch.timing.audits_computed, 3);
        assert_eq!(scratch.contract, changed.contract);
        assert_eq!(scratch.market, changed.market);

        // Axis 2: the same bytes under a different forecast length. This one must NOT invalidate
        // the market grid, and that is the point of keying the layers differently: the grid is a
        // function of the eligible tickers' bytes and the cross-section floor, and a longer
        // horizon changes neither here. A cache that threw the grid away on every `--pred-len`
        // sweep would rebuild 470 million bar reads to arrive at the same three vectors.
        let regeometried = load(200, 1);
        assert_eq!(
            regeometried.timing.audits_reused, 3,
            "geometry does not touch the bytes"
        );
        assert_eq!(regeometried.timing.audits_computed, 0);
        assert!(regeometried.timing.bounds_reused && regeometried.timing.market_reused);
        assert_ne!(regeometried.contract.purge, changed.contract.purge);
        assert_eq!(regeometried.market, changed.market);

        // Axis 3: the market grid's own contract. The cross-section floor decides which slots
        // define a step, so it changes the grid's meaning without touching a single byte.
        let refloored = load(7, 3);
        assert_eq!(refloored.timing.audits_reused, 3);
        assert!(
            refloored.timing.bounds_reused,
            "the floor does not move the boundaries"
        );
        assert!(
            !refloored.timing.market_reused,
            "a different cross-section floor is a different market grid"
        );
        assert_ne!(
            refloored.contract.market_fingerprint,
            changed.contract.market_fingerprint
        );
    }

    /// The calibration partition, and the proof that giving it a population moved nothing.
    ///
    /// Two claims, and the second is the load-bearing one. First, `[70%, 80%)` now yields
    /// complete-target origins whose targets stop `purge` bars short of the validation
    /// partition, so a curve fitted here and spent there shares no observation. Second, the
    /// training and validation origin LISTS and every published target-bar scalar are exactly
    /// what the pre-change rule produces - restated here from the geometry rather than read
    /// back out of the implementation, so a placement change fails this even if it is
    /// self-consistent. That is what pins the held-out sample draw, which is
    /// `fixed_origins(&validation_refs, ..)`: an unchanged list under a deterministic stride
    /// is an unchanged draw, and therefore an unchanged persistence NLL.
    #[test]
    fn the_calibration_partition_has_a_population_and_perturbs_no_existing_split() {
        use shared::bars::{bar_file_path, write_bar_file};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-partition-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let bars: Vec<_> = (0..12_000)
            .map(|i| {
                let price = 100.0 + i as f32 * 0.001;
                PackedBar {
                    ts_ms: 1_500_000_000_000 + i * RESOLUTION_MS,
                    open: price,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price + 0.5,
                    volume: 1_000.0,
                    vwap: price,
                    trades: 10,
                }
            })
            .collect();
        write_bar_file(
            &bar_file_path(&directory.0, "FULL", 300),
            "FULL",
            300,
            &bars,
        )
        .unwrap();
        // A quarantined-OHLC ticker, so the valid-bar ordinals the bands are cut on really do
        // differ from the raw indices the boundaries were probed at.
        let mut holed = bars.clone();
        for index in [11, 4_444, 9_001, 9_002] {
            holed[index].low = 0.0;
        }
        write_bar_file(
            &bar_file_path(&directory.0, "HOLED", 300),
            "HOLED",
            300,
            &holed,
        )
        .unwrap();
        // A ticker that stops inside the calibration partition, so the band arithmetic has to
        // survive a `retained_partition_end` that lands past the ticker's own last bar.
        write_bar_file(
            &bar_file_path(&directory.0, "SHORT", 300),
            "SHORT",
            300,
            &bars[..9_500],
        )
        .unwrap();
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let (context, pred_len, common_context) = (16usize, 7usize, 32usize);
        let corpus = Corpus::load(
            &directory.0,
            &[],
            context,
            pred_len,
            common_context,
            &features,
            1,
            0,
        )
        .unwrap();
        let purge = corpus.contract.purge;
        assert_eq!(
            purge, 100,
            "purge is max(pred_len, 100) and the bands assume it"
        );

        // ---- claim 2: the pre-change rule, restated, for both published splits ------------
        let mut expected_train = Vec::new();
        let mut expected_validation = Vec::new();
        let (mut train_bars, mut validation_bars, mut remainder_bars) = (0usize, 0, 0);
        for (ticker, c) in corpus.contract.tickers.iter().enumerate() {
            expected_train.extend(
                (common_context - 1..c.train_end - 1)
                    .step_by(pred_len)
                    .map(|origin| WindowRef { ticker, origin }),
            );
            train_bars += c.train_end - common_context;
            let start = c.boundaries[1].max(common_context);
            let available =
                retained_partition_end(c.boundaries[2], c.valid_bars, purge).saturating_sub(start);
            expected_validation.extend((0..available / pred_len).map(|i| WindowRef {
                ticker,
                origin: start - 1 + i * pred_len,
            }));
            validation_bars += available / pred_len * pred_len;
            remainder_bars += available % pred_len;
        }
        assert_eq!(corpus.train_refs, expected_train, "a training origin moved");
        assert_eq!(
            corpus.validation_refs, expected_validation,
            "a held-out origin moved, which would move the held-out sample draw with it"
        );
        assert_eq!(corpus.contract.train_target_bars, train_bars);
        assert_eq!(corpus.contract.validation_target_bars, validation_bars);
        assert_eq!(corpus.contract.validation_remainder_bars, remainder_bars);
        // The contract is what `load_checkpoint` compares against every checkpoint manifest on
        // disk, so its FIELD SET is part of the pin, not only its values.
        let serialized: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&corpus.contract).unwrap()).unwrap();
        let mut fields: Vec<&str> = serialized
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        fields.sort_unstable();
        assert_eq!(
            fields,
            [
                "auxiliary_schema",
                "boundary_timestamps",
                "common_context",
                "context",
                "excluded_tickers",
                "features",
                "market_fingerprint",
                "market_min_cross_section",
                "minimum_source_bars",
                "minimum_training_bars",
                "pred_len",
                "purge",
                "schema",
                "spy_fingerprint",
                "tickers",
                "train_target_bars",
                "validation_remainder_bars",
                "validation_target_bars",
            ],
            "the calibration population is DERIVED from this contract; adding a field to it \
             would invalidate every authenticated checkpoint manifest on disk"
        );
        assert!(
            corpus.contract.schema.contains("quantiles70:10:10:10"),
            "the schema already declares the partition this population comes from"
        );

        // ---- claim 1: the calibration partition is usable and provably disjoint -----------
        assert!(
            !corpus.calibration_refs.is_empty(),
            "the reserved partition must not be empty; that was the defect"
        );
        let mut seen_tickers = BTreeSet::new();
        for reference in &corpus.calibration_refs {
            let c = &corpus.contract.tickers[reference.ticker];
            seen_tickers.insert(reference.ticker);
            assert_eq!(
                corpus.ticker(*reference).target_count(reference.origin),
                Some(pred_len),
                "every calibration origin owns a COMPLETE horizon"
            );
            let (first, last) = (reference.origin + 1, reference.origin + pred_len);
            assert!(
                first >= c.boundaries[0],
                "a calibration target reached back into the training partition"
            );
            assert!(
                last < retained_partition_end(c.boundaries[1], c.valid_bars, purge),
                "a calibration target crossed into the purge band before validation"
            );
            assert!(last < c.valid_bars);
        }
        assert!(
            seen_tickers.len() >= 2,
            "a one-ticker calibration population would have no cross-section at all"
        );
        // Target bands, per ticker, must not intersect. Bars, not origins: an origin's context
        // may legitimately read the neighbouring partition, a target may never be in two.
        for (ticker, c) in corpus.contract.tickers.iter().enumerate() {
            let mut owner = vec![0u8; c.valid_bars];
            for (mark, refs) in [
                (1u8, &corpus.train_refs),
                (2, &corpus.calibration_refs),
                (3, &corpus.validation_refs),
            ] {
                for reference in refs.iter().filter(|r| r.ticker == ticker) {
                    let count = corpus
                        .ticker(*reference)
                        .target_count(reference.origin)
                        .unwrap();
                    for slot in &mut owner[reference.origin + 1..reference.origin + 1 + count] {
                        assert_eq!(*slot, 0, "a target bar is claimed by two populations");
                        *slot = mark;
                    }
                }
            }
        }
        let last_calibration_target = corpus
            .calibration_refs
            .iter()
            .map(|r| corpus.ticker(*r).timestamp(r.origin + pred_len))
            .max()
            .unwrap();
        let first_validation_origin = corpus
            .validation_refs
            .iter()
            .map(|r| corpus.ticker(*r).timestamp(r.origin))
            .min()
            .unwrap();
        assert!(
            last_calibration_target < first_validation_origin,
            "the last bar a calibration target reads ({last_calibration_target}) must precede \
             the first validation ORIGIN ({first_validation_origin}), or the two blocks share \
             an observation"
        );
        // A batch really builds from these origins - the population is usable, not merely
        // enumerable.
        let probe = [
            corpus.calibration_refs[0],
            *corpus.calibration_refs.last().unwrap(),
        ];
        let batch = corpus.host_batch(&probe).unwrap();
        assert_eq!(batch.valid_target_bars, 2 * pred_len);
    }

    /// WHAT the reserved bands actually guarantee, on the real construction path.
    ///
    /// `boundaries[k]` is each ticker's own valid-bar ordinal at a SHARED boundary timestamp
    /// (`index_at_or_after`), so the guarantee is per ticker: its calibration targets stop
    /// `purge` bars before `boundaries[1]` and its first validation origin is `boundaries[1] -
    /// 1`. The realized wall clocks of those ordinals are NOT comparable across tickers - a
    /// ticker that stops trading before the boundary has its own band edge at its last bar,
    /// which can be years earlier - so the two populations' global extrema interleave.
    ///
    /// The fixture is the real corpus's shape at 1/500th of the scale: one continuously
    /// traded name and one that goes quiet for 1,400 slots across the `[80%, 90%)` boundary.
    /// It reproduces the interleave that refused job 5998 - the pooled guard fails on it - and
    /// per ticker the ordering is exact.
    #[test]
    fn the_reserved_bands_are_disjoint_per_ticker_and_interleave_globally() {
        use super::super::calibration::Blocks;
        use shared::bars::{bar_file_path, write_bar_file};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-interleave-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let bar = |slot: i64| {
            let price = 100.0 + slot as f32 * 0.001;
            PackedBar {
                ts_ms: 1_500_000_000_000 + slot * RESOLUTION_MS,
                open: price,
                high: price + 1.0,
                low: price - 1.0,
                close: price + 0.5,
                volume: 1_000.0,
                vwap: price,
                trades: 10,
            }
        };
        let full: Vec<_> = (0..12_000).map(bar).collect();
        // Quiet from slot 9,000 to slot 10,400: the shared `[80%, 90%)` boundary at slot 9,600
        // falls inside the hole, so this ticker's own boundary ordinal resolves to its first
        // bar after the hole and its first validation origin is the bar BEFORE it - slot 8,999,
        // 600 slots before the continuously traded name's last calibration target.
        let gappy: Vec<_> = (0..9_000).chain(10_400..12_000).map(bar).collect();
        write_bar_file(
            &bar_file_path(&directory.0, "FULL", 300),
            "FULL",
            300,
            &full,
        )
        .unwrap();
        write_bar_file(
            &bar_file_path(&directory.0, "GAPPY", 300),
            "GAPPY",
            300,
            &gappy,
        )
        .unwrap();
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let (context, pred_len, common_context) = (16usize, 7usize, 32usize);
        let corpus = Corpus::load(
            &directory.0,
            &[],
            context,
            pred_len,
            common_context,
            &features,
            1,
            0,
        )
        .unwrap();
        let dated = |refs: &[WindowRef]| -> Vec<(usize, i64, i64)> {
            refs.iter()
                .map(|reference| {
                    let ticker = corpus.ticker(*reference);
                    (
                        reference.ticker,
                        ticker.timestamp(reference.origin),
                        ticker.timestamp(reference.origin + pred_len),
                    )
                })
                .collect()
        };
        let calibration = dated(&corpus.calibration_refs);
        let validation = dated(&corpus.validation_refs);
        // Both names carry origins on both sides, or the per-ticker claim below is vacuous.
        for ticker in 0..corpus.contract.tickers.len() {
            assert!(
                calibration.iter().any(|(owner, _, _)| *owner == ticker)
                    && validation.iter().any(|(owner, _, _)| *owner == ticker),
                "{} carries no origins on one side of the split",
                corpus.contract.tickers[ticker].ticker
            );
        }
        // ---- the invariant: per ticker, exact ------------------------------------------
        let mut tightest = i64::MAX;
        for (ticker, contract) in corpus.contract.tickers.iter().enumerate() {
            let reach = calibration
                .iter()
                .filter(|(owner, _, _)| *owner == ticker)
                .map(|(_, _, target)| *target)
                .max()
                .unwrap();
            let opens = validation
                .iter()
                .filter(|(owner, _, _)| *owner == ticker)
                .map(|(_, origin, _)| *origin)
                .min()
                .unwrap();
            assert!(
                reach < opens,
                "{}'s calibration targets reach {reach} but its own validation block opens at \
                 {opens}",
                contract.ticker
            );
            tightest = tightest.min(opens - reach);
        }
        // ---- and globally the extrema interleave, which is why they must not be compared -
        let global_reach = calibration.iter().map(|(_, _, t)| *t).max().unwrap();
        let global_open = validation.iter().map(|(_, o, _)| *o).min().unwrap();
        assert!(
            global_reach > global_open,
            "the fixture must reproduce the interleave: reach {global_reach}, open {global_open}"
        );
        println!(
            "measured on the fixture: boundaries {:?} for {} and {:?} for {}; the pooled \
             calibration reach is {global_reach} and the pooled first validation origin is \
             {global_open}, {} ms EARLIER, while the tightest per-ticker separation is \
             {tightest} ms",
            corpus.contract.tickers[0].boundaries,
            corpus.contract.tickers[0].ticker,
            corpus.contract.tickers[1].boundaries,
            corpus.contract.tickers[1].ticker,
            global_reach - global_open
        );
        let pooled = |rows: &[(usize, i64, i64)]| -> Vec<(i64, i64)> {
            rows.iter()
                .map(|(_, origin, target)| (*origin, *target))
                .collect()
        };
        let refusal = Blocks::spanning(&pooled(&calibration), &pooled(&validation))
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("share bars"), "{refusal}");
        let blocks = Blocks::per_ticker(&calibration, &validation, pred_len, |ticker| {
            corpus.contract.tickers[ticker].ticker.clone()
        })
        .unwrap();
        assert_eq!(blocks.purge_gap_ms, tightest);
        assert_eq!(blocks.calibration_last_target_ms, global_reach);
        assert_eq!(blocks.evaluation_first_origin_ms, global_open);

        let plan = corpus.research_sample_plan(71, 23, 17, pred_len).unwrap();
        let repeat = corpus.research_sample_plan(71, 23, 17, pred_len).unwrap();
        assert_eq!(plan.validation_refs, repeat.validation_refs);
        assert_eq!(plan.probe_fit_refs, repeat.probe_fit_refs);
        assert_eq!(plan.validation_refs.len(), 23);
        assert_eq!(plan.probe_fit_refs.len(), 17);
        for reference in &plan.validation_refs {
            let ticker = corpus.ticker(*reference);
            assert!(
                ticker.timestamp(reference.origin - pred_len)
                    > plan.manifest.training_last_target_ms
            );
            assert!(
                ticker.timestamp(reference.origin + 1) >= corpus.contract.boundary_timestamps[1]
            );
            assert!(
                ticker.timestamp(reference.origin + pred_len)
                    < corpus.contract.boundary_timestamps[2]
            );
        }
        for reference in &plan.probe_fit_refs {
            assert!(corpus
                .training_target_count(reference.ticker, reference.origin)
                .is_some());
        }
        for coverage in &plan.manifest.validation.coverage {
            assert!(coverage.selected_origins > 0);
        }
        // A warm cache cannot quietly move an edge even when the stored shape still fits.
        let ticker = &corpus.tickers[0];
        let actual = corpus
            .contract
            .boundary_timestamps
            .map(|bound| ticker.file.index_at_or_after(bound) as u64);
        verify_split_edges(
            &ticker.contract.ticker,
            ticker.file.bars(),
            actual,
            corpus.contract.boundary_timestamps,
        )
        .unwrap();
        let mut wrong = actual;
        wrong[1] += 1;
        assert!(verify_split_edges(
            &ticker.contract.ticker,
            ticker.file.bars(),
            wrong,
            corpus.contract.boundary_timestamps
        )
        .is_err());
    }

    /// The corpus SCHEMA is the stamp both derived caches are keyed on, so a bump to it is a
    /// MISS at every layer below it - and the bar audits, keyed on `data::SCHEMA` instead,
    /// correctly survive. A cached corpus paired with a changed partition schema is the exact
    /// stale pairing this keying exists to make unreachable.
    #[test]
    fn bumping_the_corpus_schema_invalidates_every_derived_cache_and_no_bar_audit() {
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-schema-stamp-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let universe = [("AAA", "ff"), ("BBB", "ee")];
        let bumped = format!("{SCHEMA};a-new-partition-population");
        assert_ne!(bumped, SCHEMA);

        cache::BoundsCache::open(&directory.0, SCHEMA, &universe)
            .store([1, 2, 3], &[[0, 1, 2], [3, 4, 5]])
            .unwrap();
        assert!(
            cache::BoundsCache::open(&directory.0, SCHEMA, &universe)
                .get(2)
                .is_some(),
            "the stamp the corpus actually passes must hit its own artifact"
        );
        assert!(
            cache::BoundsCache::open(&directory.0, &bumped, &universe)
                .get(2)
                .is_none(),
            "a corpus schema bump must reject the stored boundaries"
        );

        let moment = |value: f64| super::super::features::SlotMoment {
            sums: vec![value, value],
            squares: vec![value * value, value * value],
            counts: vec![1, 1],
        };
        let grid = cache::MarketGrid {
            first_ts: 0,
            min_cross_section: 1,
            population: vec![1, 1],
            returns: moment(0.5),
            log_volume: moment(7.0),
            log_range: moment(-6.0),
        };
        cache::MarketCache::open(&directory.0, SCHEMA, &universe, 0, 2, 1)
            .store(&grid)
            .unwrap();
        assert!(
            cache::MarketCache::open(&directory.0, SCHEMA, &universe, 0, 2, 1)
                .get(2)
                .is_some()
        );
        assert!(
            cache::MarketCache::open(&directory.0, &bumped, &universe, 0, 2, 1)
                .get(2)
                .is_none(),
            "a corpus schema bump must reject the stored market grid"
        );

        // The audit layer is keyed on the DATA schema, not this one, and must be untouched by
        // a partition-geometry bump: those bar digests are the expensive layer.
        assert_ne!(SCHEMA, super::super::data::SCHEMA);
    }

    /// Population statistics of the reserved `[70%, 80%)` partition on the REAL corpus, and
    /// the golden pin that giving it a population moved neither published split: the three
    /// target-bar scalars, the ticker count and the three boundary timestamps are read back
    /// from `training/runs/timexer-control-4k/timexer-segment-data-contract.json`, an artifact
    /// written by a PRE-CHANGE binary.
    ///
    /// ```
    /// ./torch-env.sh cargo test -p trading_bot_0 --release \
    ///     timexer_segment::corpus::tests::the_real_corpus_calibration_partition \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "reads the real long_data/bars corpus at the production geometry"]
    fn the_real_corpus_calibration_partition_is_measured_against_a_published_contract() {
        let directory = crate::data::ingest::bars_dir();
        let published: CorpusContract = serde_json::from_slice(
            &fs::read(
                Path::new(shared::paths::TRAINING_PATH)
                    .join("runs/timexer-control-4k/timexer-segment-data-contract.json"),
            )
            .expect("the published contract of the run the calibration is fitted on"),
        )
        .unwrap();
        let corpus = Corpus::load(
            &directory,
            &[],
            published.context,
            published.pred_len,
            published.common_context,
            &published.features,
            published.market_min_cross_section,
            published.in_period_sections,
        )
        .unwrap();
        assert_eq!(
            corpus.contract, published,
            "the whole contract must be byte-identical to the one authenticated inside every \
             checkpoint of that run, or no checkpoint of it can be loaded again"
        );
        let pred_len = published.pred_len;
        let mut stamps: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        let mut tickers = BTreeSet::new();
        for reference in &corpus.calibration_refs {
            *stamps
                .entry(corpus.ticker(*reference).timestamp(reference.origin))
                .or_default() += 1;
            tickers.insert(reference.ticker);
        }
        let width = corpus.calibration_refs.len() as f64 / stamps.len() as f64;
        println!(
            "calibration partition: {} origins, {} target bars, {} distinct timestamps, {} \
             distinct tickers, mean cross-sectional width {width:.3}, widest timestamp {}",
            corpus.calibration_refs.len(),
            corpus.calibration_refs.len() * pred_len,
            stamps.len(),
            tickers.len(),
            stamps.values().copied().max().unwrap_or(0)
        );
        println!(
            "validation partition for comparison: {} origins, {} target bars",
            corpus.validation_refs.len(),
            corpus.contract.validation_target_bars
        );
        assert!(!corpus.calibration_refs.is_empty());
    }

    /// Every accessor on a `Batch` is a VIEW of one packed block. That is what lets a
    /// resident batch be refilled by a single copy into a fixed address - which is what a
    /// captured CUDA graph reads and what keeps a 114 MB allocate-and-free off every step.
    /// If `from_packed` ever returned copies instead, `upload` would refresh nothing and the
    /// model would train on whatever the resident batch held at construction.
    #[test]
    fn uploading_a_host_batch_refreshes_every_view_of_the_resident_batch() {
        let (context, pred_len, aux_channels) = (4, 3, 2);
        let width = Batch::row_width(context, pred_len, aux_channels);
        let batch_of = |rows: usize, offset: f32| {
            let values: Vec<f32> = (0..rows * width)
                .map(|index| offset + index as f32)
                .collect();
            Batch::from_packed(
                Tensor::from_slice(&values).reshape([rows as i64, width as i64]),
                context,
                pred_len,
                aux_channels,
                rows,
            )
        };
        let first = batch_of(2, 1.0);
        let second = batch_of(2, 1000.0);
        let mut resident = first.resident(Device::Cpu);
        first.upload(&mut resident).unwrap();
        for (mine, theirs) in [
            (&resident.log_prices, &first.log_prices),
            (&resident.valid, &first.valid),
            (&resident.aux, &first.aux),
            (&resident.market_cum, &first.market_cum),
            (&resident.anchor, &first.anchor),
        ] {
            assert!(
                mine.equal(theirs),
                "the first upload did not land in a view"
            );
        }
        second.upload(&mut resident).unwrap();
        for (mine, theirs) in [
            (&resident.log_prices, &second.log_prices),
            (&resident.valid, &second.valid),
            (&resident.aux, &second.aux),
            (&resident.market_cum, &second.market_cum),
            (&resident.anchor, &second.anchor),
        ] {
            assert!(
                mine.equal(theirs),
                "a refill left a view reading the previous batch"
            );
        }
        assert_eq!(resident.valid_target_bars, second.valid_target_bars);
        // A shape the capture never recorded must be refused, not silently truncated.
        assert!(batch_of(1, 0.0).upload(&mut resident).is_err());
    }

    #[test]
    fn corpus_can_be_shared_with_a_prefetch_worker() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Corpus>();
    }

    /// THE experiment's guarantee, restated from the geometry and checked bar by bar.
    ///
    /// The in-period draw exists to separate OVERFITTING from NON-STATIONARITY, and it can only
    /// do that if it is out-of-sample in ORIGIN IDENTITY while remaining in-sample in MARKET
    /// REGIME. Origin identity is the fragile half: training supervises every causal
    /// sub-origin, not just a row's final 192-bar window, so one training row's supervised
    /// target bars span roughly a whole context and consecutive rows advance only `pred_len`.
    /// An exclusion written against final windows alone leaks by a factor of about
    /// `context / pred_len`.
    ///
    /// So the check here is deliberately NOT "no shared final window". It is: no in-period
    /// origin is any training row's origin, and no in-period TARGET BAR lies anywhere in the
    /// conservative supervised interval `[origin - context + 2, origin + owned]` of any
    /// surviving training row. That interval is a SUPERSET of what the model actually
    /// supervises at any `patch_len`/`min_history`, so disjointness from it implies
    /// disjointness from the truth, and cannot be broken by a change to the patch grid.
    ///
    /// The regime half is checked too, because a hole with no trained data on one side of it is
    /// an extrapolation wearing an in-period label: both brackets must be populated.
    #[test]
    fn in_period_origins_share_no_origin_and_no_target_bar_with_any_training_row() {
        use shared::bars::{bar_file_path, write_bar_file};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-in-period-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let bars: Vec<_> = (0..12_000)
            .map(|i| {
                let price = 100.0 + i as f32 * 0.001;
                PackedBar {
                    ts_ms: 1_500_000_000_000 + i * RESOLUTION_MS,
                    open: price,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price + 0.5,
                    volume: 1_000.0,
                    vwap: price,
                    trades: 10,
                }
            })
            .collect();
        for ticker in ["AAA", "BBB"] {
            write_bar_file(
                &bar_file_path(&directory.0, ticker, 300),
                ticker,
                300,
                &bars,
            )
            .unwrap();
        }
        // A ticker whose valid-bar ordinals differ from its raw indices, so the hole is cut on
        // the same filtered ordinals the training rows are.
        let mut quarantined = bars.clone();
        for index in [11, 4_444, 9_001] {
            quarantined[index].low = 0.0;
        }
        write_bar_file(
            &bar_file_path(&directory.0, "CCC", 300),
            "CCC",
            300,
            &quarantined,
        )
        .unwrap();
        // THE ticker that separates a wall-clock anchor from an ordinal offset. `DDD` skips
        // every third slot over the first 3,000, so from bar 3,000 onward its ordinal for any
        // given moment runs about 1,000 behind the dense tickers' while its TIMESTAMPS still
        // coincide with theirs exactly. An ordinal-offset hole lands ~1,000 bars away from the
        // others in wall clock and shares no timestamp with them; an anchored hole shares all
        // of them. This is the fixture form of the 4,873-ticker failure job 5460 hit.
        let sparse: Vec<_> = bars
            .iter()
            .enumerate()
            .filter(|(index, _)| *index >= 3_000 || index % 3 != 2)
            .map(|(_, bar)| *bar)
            .collect();
        write_bar_file(
            &bar_file_path(&directory.0, "DDD", 300),
            "DDD",
            300,
            &sparse,
        )
        .unwrap();
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let (context, pred_len, common_context) = (16usize, 7usize, 32usize);
        let sections = 32usize;
        let load = |in_period_sections| {
            Corpus::load(
                &directory.0,
                &[],
                context,
                pred_len,
                common_context,
                &features,
                1,
                in_period_sections,
            )
            .unwrap()
        };
        let control = load(0);
        let holed = load(sections);

        // ---- claim 1: the three existing held-out draws are byte-identical ----------------
        // `held-out sample` is `fixed_origins(&validation_refs, ..)` and `held-out
        // cross-section` is `cross_section_origins(&corpus, &corpus.validation_refs)`, so an
        // unchanged `validation_refs` list IS an unchanged draw - and therefore an unchanged
        // persistence NLL, the anchor every step-matched cross-run comparison is read against.
        assert_eq!(
            control.validation_refs, holed.validation_refs,
            "the hole moved a held-out origin; every `held-out *` draw and the persistence NLL \
             anchor move with it"
        );
        assert_eq!(
            control.calibration_refs, holed.calibration_refs,
            "the hole moved a calibration origin"
        );
        assert_eq!(
            (
                control.contract.validation_target_bars,
                control.contract.validation_remainder_bars
            ),
            (
                holed.contract.validation_target_bars,
                holed.contract.validation_remainder_bars
            )
        );
        assert!(
            control.in_period_refs.is_empty() && control.contract.in_period_sections == 0,
            "0 sections must be exactly today's behaviour"
        );
        assert!(!holed.in_period_refs.is_empty());

        // ---- claim 2: the census is exact, so the population price is not an estimate -----
        assert_eq!(
            holed.train_refs.len() + holed.contract.in_period_purged_rows,
            control.train_refs.len(),
            "every control row is either kept or purged, never both and never neither"
        );
        assert_eq!(
            holed.contract.train_target_bars + holed.contract.in_period_purged_target_bars,
            control.contract.train_target_bars,
            "the purged target bars must account for the whole difference"
        );
        assert!(
            holed.contract.in_period_purged_rows > 0,
            "a hole that costs nothing has not excluded the rows whose context reaches it"
        );

        // ---- claim 3: disjointness, bar by bar, against the conservative supervised set ---
        let train_origins: BTreeSet<(usize, usize)> = holed
            .train_refs
            .iter()
            .map(|r| (r.ticker, r.origin))
            .collect();
        let mut supervised: BTreeSet<(usize, usize)> = BTreeSet::new();
        for reference in &holed.train_refs {
            let c = &holed.contract.tickers[reference.ticker];
            let owned = pred_len.min(c.train_end - reference.origin - 1);
            let first = (reference.origin + 2).saturating_sub(context);
            for bar in first..=reference.origin + owned {
                supervised.insert((reference.ticker, bar));
            }
        }
        let mut per_ticker = std::collections::BTreeMap::<usize, usize>::new();
        for reference in &holed.in_period_refs {
            let ticker = holed.ticker(*reference);
            let c = &ticker.contract;
            let (lo, hi) = ticker
                .in_period_hole()
                .expect("an in-period origin needs a hole");
            *per_ticker.entry(reference.ticker).or_default() += 1;
            assert!(
                !train_origins.contains(&(reference.ticker, reference.origin)),
                "in-period origin {reference:?} IS a training row's origin"
            );
            assert_eq!(
                ticker.target_count(reference.origin),
                Some(pred_len),
                "an in-period origin must own a COMPLETE horizon, like a validation origin"
            );
            assert!(
                reference.origin < c.train_end - 1 && reference.origin + 1 >= common_context,
                "an in-period origin must lie inside the TRAINING span; that is what makes it \
                 in-period rather than a second validation draw"
            );
            for bar in reference.origin + 1..=reference.origin + pred_len {
                assert!(
                    (lo..=hi).contains(&bar),
                    "in-period target bar {bar} escaped the hole [{lo}, {hi}]"
                );
                assert!(
                    !supervised.contains(&(reference.ticker, bar)),
                    "in-period target bar {bar} of ticker {} is supervised by a surviving \
                     training row",
                    c.ticker
                );
            }
        }
        assert_eq!(
            holed.contract.in_period_origins,
            holed.in_period_refs.len(),
            "the contract must publish the population it actually built"
        );

        // ---- claim 4: the origins are SHARED WALL CLOCKS, so a cross-section can form -----
        // The failure this replaces: an ordinal offset from a shared timestamp is not a shared
        // timestamp, so the draw found no moment held by enough tickers and the run refused at
        // `cross_section_blocks`. `DDD`'s ordinals run ~1,000 behind the dense tickers' at the
        // same moment, so an unanchored construction cannot pass this.
        let anchors = &holed.contract.in_period_anchors;
        assert_eq!(anchors.len(), sections, "one anchor per section");
        assert!(
            anchors.windows(2).all(|pair| pair[0] < pair[1]),
            "anchors must be ascending distinct moments: {anchors:?}"
        );
        let mut widths = vec![0usize; sections];
        for reference in &holed.in_period_refs {
            let stamp = holed.ticker(*reference).timestamp(reference.origin);
            let section = anchors
                .iter()
                .position(|anchor| *anchor == stamp)
                .unwrap_or_else(|| {
                    panic!("in-period origin at {stamp} is on no anchor: {anchors:?}")
                });
            widths[section] += 1;
        }
        assert_eq!(
            widths, holed.contract.in_period_census,
            "the published census must be the population actually built, anchor by anchor"
        );
        assert!(
            widths.iter().all(|width| *width >= 2),
            "every anchor must be held by at least two tickers or no cross-section exists \
             there; widths are {widths:?}"
        );
        assert!(
            per_ticker.values().any(|count| *count == sections),
            "at least one ticker must hold every anchor"
        );
        // The claim above is only worth its wall clock if the fixture is actually misaligned:
        // if every ticker's ordinal line agreed, an ordinal offset would pass too. This is the
        // teeth, asserted rather than assumed, because it is a property of the fixture that a
        // later edit could quietly remove.
        let spread = holed
            .in_period_refs
            .iter()
            .filter(|reference| holed.ticker(**reference).timestamp(reference.origin) == anchors[0])
            .map(|reference| reference.origin)
            .collect::<BTreeSet<_>>();
        assert!(
            spread.len() > 1 && spread.last().unwrap() - spread.first().unwrap() > pred_len,
            "the fixture's tickers all agree on the ordinal of anchor {}, so this test could \
             not tell a wall-clock anchor from an ordinal offset: ordinals {spread:?}",
            anchors[0]
        );
        for (ticker, count) in &per_ticker {
            assert!(
                *count <= sections,
                "ticker {ticker} carries {count} origins for {sections} anchors"
            );
            let (lo, hi) = holed.tickers[*ticker].in_period_hole().unwrap();
            let (below, above) = holed
                .train_refs
                .iter()
                .filter(|r| r.ticker == *ticker)
                .fold((0usize, 0usize), |(below, above), r| {
                    if r.origin < lo {
                        (below + 1, above)
                    } else {
                        (below, above + 1)
                    }
                });
            assert!(
                below >= IN_PERIOD_BRACKET_ROWS && above >= IN_PERIOD_BRACKET_ROWS,
                "hole [{lo}, {hi}] is bracketed by {below} rows below and {above} above, under \
                 the {IN_PERIOD_BRACKET_ROWS} each side that make its period in-sample"
            );
        }

        // ---- claim 5: the predicate is safe under a per-row origin PHASE shift ------------
        // A consumer that shifts a row's origin forward by `p` bars re-tests the SHIFTED
        // origin. Every shift of a purged row must stay purged if it still reaches the hole,
        // and the predicate must be monotone in nothing - it is re-evaluated, not assumed.
        for reference in &holed.train_refs {
            let ticker = holed.ticker(*reference);
            let (lo, hi) = ticker.in_period_hole().unwrap();
            for phase in 1..pred_len {
                let shifted = reference.origin + phase;
                let reaches =
                    shifted + pred_len >= lo && (shifted + 2).saturating_sub(context) <= hi;
                assert_eq!(
                    ticker.supervision_clears_hole(shifted),
                    !reaches,
                    "the phase-shifted origin {shifted} is misclassified against hole [{lo}, {hi}]"
                );
            }
        }
    }

    /// The failure that killed job 5460, made into a test.
    ///
    /// `--in-period-sections 32` refused with "no held-out evaluation timestamp holds the 40
    /// tickers a cross-sectional trading measurement needs at all". A per-ticker ORDINAL offset
    /// back from a shared timestamp is NOT a shared wall clock: two tickers with different bar
    /// densities over the 18,479-bar backoff land at different moments, so the timestamp
    /// grouping `cross_section_blocks` performs finds nothing wide enough. The fixture in
    /// `in_period_origins_share_no_origin_and_no_target_bar_with_any_training_row` cannot see
    /// this - its three tickers share one calendar by construction.
    ///
    /// This measures the real thing: how many tickers share the widest in-period timestamp, and
    /// whether the draw a run would build actually clears `CROSS_SECTION_FLOOR`.
    ///
    /// ```
    /// ./torch-env.sh cargo test -p trading_bot_0 --release \
    ///     timexer_segment::corpus::tests::the_real_corpus_in_period_draw \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "reads the real long_data/bars corpus at the production geometry"]
    fn the_real_corpus_in_period_draw_forms_usable_cross_sections() {
        let directory = crate::data::ingest::bars_dir();
        let published: CorpusContract = serde_json::from_slice(
            &fs::read(
                Path::new(shared::paths::TRAINING_PATH)
                    .join("runs/timexer-invsqrt-ic-6k/timexer-segment-data-contract.json"),
            )
            .expect("the published contract of the arm the observation came from"),
        )
        .unwrap();
        let sections = 32;
        let corpus = Corpus::load(
            &directory,
            &[],
            published.context,
            published.pred_len,
            published.common_context,
            &published.features,
            published.market_min_cross_section,
            sections,
        )
        .unwrap();
        assert!(!corpus.in_period_refs.is_empty());
        let mut blocks: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        for reference in &corpus.in_period_refs {
            *blocks
                .entry(corpus.ticker(*reference).timestamp(reference.origin))
                .or_default() += 1;
        }
        let mut widths: Vec<usize> = blocks.values().copied().collect();
        widths.sort_unstable_by(|a, b| b.cmp(a));
        let clearing = widths.iter().filter(|w| **w >= 40).count();
        println!(
            "in-period draw: {} origins over {} distinct timestamps; widest 12 blocks {:?}; {} \
             timestamps hold >= 40 tickers",
            corpus.in_period_refs.len(),
            blocks.len(),
            &widths[..widths.len().min(12)],
            clearing
        );
        // The out-of-period comparand, for reference: the same measurement on the draw that
        // works, so the two are read off one run.
        let mut validation: std::collections::BTreeMap<i64, usize> =
            std::collections::BTreeMap::new();
        for reference in &corpus.validation_refs {
            *validation
                .entry(corpus.ticker(*reference).timestamp(reference.origin))
                .or_default() += 1;
        }
        let mut out: Vec<usize> = validation.values().copied().collect();
        out.sort_unstable_by(|a, b| b.cmp(a));
        println!(
            "out-of-period draw: {} origins over {} distinct timestamps; widest 12 blocks {:?}; \
             {} timestamps hold >= 40 tickers",
            corpus.validation_refs.len(),
            validation.len(),
            &out[..out.len().min(12)],
            out.iter().filter(|w| **w >= 40).count()
        );
        assert!(
            clearing >= 8,
            "the in-period draw must offer at least 8 usable cross-sections; it offers \
             {clearing}, and below 1 the run refuses outright at runner.rs:74"
        );
    }

    /// The `band` fix, checked on the fixture that discriminates it.
    ///
    /// `DDD` skips every third slot over its first 3,000 bars, so from bar 3,000 onward its
    /// valid-bar ORDINAL for any given moment runs ~1,000 behind the dense tickers' while its
    /// TIMESTAMPS still coincide with theirs exactly. That is the fixture form of the real
    /// corpus's failure: the strided rule `boundaries[first] - 1 + i*pred_len` is evaluated on
    /// each ticker's own ordinals, so only `i = 0` is a shared moment and every later stride
    /// scatters. The anchored rule walks ONE reference clock and takes each ticker's bar at that
    /// exact moment or nothing, so every origin it emits sits on a published wall clock.
    ///
    /// Four claims, and the third is the one the external cache binding depends on.
    #[test]
    fn the_anchored_placement_puts_every_origin_on_a_published_wall_clock() {
        use shared::bars::{bar_file_path, write_bar_file};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-anchored-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let bars: Vec<_> = (0..12_000)
            .map(|i| {
                let price = 100.0 + i as f32 * 0.001;
                PackedBar {
                    ts_ms: 1_500_000_000_000 + i * RESOLUTION_MS,
                    open: price,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price + 0.5,
                    volume: 1_000.0,
                    vwap: price,
                    trades: 10,
                }
            })
            .collect();
        for ticker in ["AAA", "BBB", "CCC"] {
            write_bar_file(
                &bar_file_path(&directory.0, ticker, 300),
                ticker,
                300,
                &bars,
            )
            .unwrap();
        }
        let sparse: Vec<_> = bars
            .iter()
            .enumerate()
            .filter(|(index, _)| *index >= 3_000 || index % 3 != 2)
            .map(|(_, bar)| *bar)
            .collect();
        write_bar_file(
            &bar_file_path(&directory.0, "DDD", 300),
            "DDD",
            300,
            &sparse,
        )
        .unwrap();
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let mut corpus = Corpus::load(&directory.0, &[], 16, 7, 32, &features, 1, 0).unwrap();
        let strided = (
            corpus.validation_refs.clone(),
            corpus.origins_sha256(&corpus.validation_refs),
            corpus.distinct_timestamps(&corpus.validation_refs),
        );
        // ---- claim 1: the strided draw really is scattered on this fixture ---------------
        // Without this the rest of the test could pass vacuously on a corpus whose tickers all
        // share one ordinal line, which is exactly the configuration that hid the real defect.
        assert!(
            strided.2 > 1,
            "the fixture is not misaligned, so it cannot discriminate the two placements"
        );
        corpus.anchor_cross_section_draws(2).unwrap();

        // ---- claim 2: every origin sits on a PUBLISHED anchor ----------------------------
        let anchors: BTreeSet<i64> = corpus.contract.validation_anchors.iter().copied().collect();
        assert!(!anchors.is_empty(), "the anchored draw published no anchor");
        for reference in &corpus.validation_refs {
            let ticker = corpus.ticker(*reference);
            let stamp = ticker.timestamp(reference.origin);
            assert!(
                anchors.contains(&stamp),
                "{} origin {} sits at {stamp}, which is not a published anchor",
                ticker.contract.ticker,
                reference.origin
            );
            // In-sample in NOTHING: the draw is still a reserved-partition draw, so every
            // origin must still own a complete `pred_len` inside `[80%, 90%)`.
            assert!(
                ticker.owns_partition_targets(reference.origin, 1),
                "an anchored origin does not own complete validation targets"
            );
        }
        // Every anchor is a shared moment, which is the whole point: the number of distinct
        // timestamps must be the anchor count, and each anchor must hold at least the floor.
        assert_eq!(
            corpus.distinct_timestamps(&corpus.validation_refs),
            anchors.len(),
            "an anchored origin landed off the anchor grid"
        );
        for (section, anchor) in corpus.contract.validation_anchors.iter().enumerate() {
            let held = corpus
                .validation_refs
                .iter()
                .filter(|reference| {
                    corpus.ticker(**reference).timestamp(reference.origin) == *anchor
                })
                .count();
            assert!(
                held >= 2,
                "anchor {section} holds {held} tickers, below the floor the draw was given"
            );
        }
        // The misaligned ticker must be IN, on the shared clock - a placement that silently
        // dropped it would look aligned while having thrown away the population.
        let sparse_held = corpus
            .validation_refs
            .iter()
            .filter(|reference| corpus.ticker(**reference).contract.ticker == "DDD")
            .count();
        assert!(
            sparse_held > 0,
            "the anchored draw dropped the misaligned ticker instead of aligning it"
        );

        // ---- claim 3: the change is OBSERVABLE to a digest-keyed cache -------------------
        // The external portfolio tape binds its cache to the exact `validation_refs` SHA. A
        // placement change that preserved the digest would silently serve stale statistics.
        assert_ne!(
            strided.1,
            corpus.origins_sha256(&corpus.validation_refs),
            "the anchored draw digests identically to the strided one"
        );
        assert_ne!(
            strided.0, corpus.validation_refs,
            "the anchored draw is the strided draw"
        );

        // ---- claim 4: the contract states which population it is ------------------------
        assert_eq!(corpus.contract.cross_section_placement, ANCHORED_PLACEMENT);
        assert_eq!(
            corpus.contract.validation_target_bars,
            corpus.validation_refs.len() * corpus.contract.pred_len
        );
        assert!(!corpus.contract.calibration_anchors.is_empty());
        // A control load must still say NOTHING, so every manifest on disk digests unchanged.
        let control = Corpus::load(&directory.0, &[], 16, 7, 32, &features, 1, 0).unwrap();
        assert!(control.contract.cross_section_placement.is_empty());
        assert!(control.contract.validation_anchors.is_empty());
    }
}
