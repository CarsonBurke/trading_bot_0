//! The row-pool diversity knobs, and the supervision-occupancy census that makes their effect
//! an OBSERVATION rather than an arithmetic afterthought.
//!
//! # What is being counted
//!
//! A training row is `seq_len + pred_len` bars ending at a final origin `O`, and the model
//! supervises it densely: every patch boundary inside the context that clears `--min-history`
//! is a causal origin, so one row carries `origins_per_row` absolute origins at stride
//! `patch_len`, spanning `[O - window, O]`. Rows are enumerated at stride `pred_len`, and
//! `pred_len` is a multiple of `patch_len`, so consecutive rows' origin lattices coincide
//! except at the ends and each interior absolute origin is supervised by
//! `window / pred_len + 1` distinct rows PER EPOCH. That count - the per-outcome multiplicity
//! `M` - is the quantity every diversity knob here moves, and the whole reason the knobs
//! exist: a step budget spent re-presenting an outcome the model has already fitted is not the
//! same budget as one spent on a fresh outcome.
//!
//! The counting is done over ABSOLUTE ORIGINS rather than over `(origin, horizon)` outcomes
//! because multiplicity is exactly flat in the horizon for every row that owns a complete
//! `pred_len` of targets. An active origin `A = O - d` with `d > 0` has all its `pred_len`
//! target bars inside the row by construction, and a row owning `pred_len` targets marks all
//! of them valid, so `M(A, h)` does not depend on `h`. The only rows that break the flatness
//! are the per-ticker tails whose `target_count` is short, and [`SupervisionCensus`] carries
//! their count so a reader can bound the deviation instead of being asked to trust it.
//!
//! # Occupancy, in closed form
//!
//! One epoch is a shuffle of the retained pool without replacement, so after `n` consumed rows
//! the number of an outcome's supervising rows already drawn is hypergeometric:
//! `X ~ Hypergeometric(R, M, n)` with `R` the retained pool. Coverage is therefore
//!
//! ```text
//! C(n) = Σ_M w_M · P[X_M ≥ 1] / Σ_M w_M
//! ```
//!
//! over the MEASURED multiplicity histogram `w`, which is the only defensible way to state it:
//! the nominal interior `M` describes 89% of the corpus's outcomes and the edge ramps carry the
//! rest, so an interior constant quoted as a global coverage figure is wrong by more than an
//! order of magnitude in the unseen tail (interior 0.09% unseen against a global 1.51% at the
//! same step). Nothing in this module hardcodes a multiplicity.
//!
//! `P[X = 0]` needs no factorials and no `lgamma`:
//! `C(R-M, n)/C(R, n) = Π_{j<M} (R - n - j)/(R - j)`, which is `M ≤ origins_per_row` terms.
//! The rest of the tail follows from the standard ratio recurrence, so the whole curve costs a
//! few hundred flops per step and can be emitted at every report interval for free.

use std::fmt;

use anyhow::{ensure, Result};
use clap::ValueEnum;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::corpus::{Corpus, WindowRef};

/// The chart base the occupancy census is written to. Registered in
/// [`shared::report::TIMEXER_SEGMENT_REPORT_BASES`]; named here so the writer and the
/// registration test cannot disagree about the string.
pub const OCCUPANCY_BASE: &str = "timexer_segment_supervision_occupancy";

/// Independent ChaCha8 stream for row selection, so the retained pool is a function of
/// `--seed` alone and is NOT entangled with the per-epoch shuffle's stream. Two arms that
/// differ only in `--epochs` must retain the identical pool, and an arm whose pool moved
/// because the shuffle consumed a different number of draws would be unreproducible.
const SELECTION_STREAM: u64 = 0x726f_775f_7365_6c65;

/// Where the patch grid of a row starts, relative to the row's own final origin.
///
/// `fixed` is the historical corpus: every row's final origin is congruent to the same residue
/// modulo `patch_len`, so all `M` exposures of an outcome carry an IDENTICAL tokenization -
/// same patch boundaries, same RoPE positions, same token count before the origin.
///
/// `random` draws one offset in `0..patch_len` per row and shifts that row's final origin
/// forward by it. This ADDS ORIGINS, not shocks: the phased lattice covers up to `patch_len`
/// times as many absolute origins, but the corpus's information ceiling is the number of
/// distinct realized return shocks and no re-tokenization raises it - the label
/// `cum(A → A+h)` of a phased origin is an exact linear combination of two lattice-supervised
/// cumulative returns. So the support changes and the ceiling does not, which is precisely why
/// this is a regularizer against tokenization-specific memorization and not a data fix.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum PatchPhase {
    /// One shared phase for the whole universe: today's corpus, exactly.
    #[default]
    Fixed,
    /// One phase per row, drawn uniformly from `0..patch_len`.
    Random,
}

impl fmt::Display for PatchPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Fixed => "fixed",
            Self::Random => "random",
        })
    }
}

/// Which training rows an arm trains on, and how each one is tokenized.
///
/// Stamped into the run manifest whenever it is not the identity, so an arm is reproducible
/// from its checkpoint and a mismatched pairing is detectable. It is deliberately NOT part of
/// [`super::corpus::CorpusContract`]: none of these knobs touches ticker eligibility, the
/// market grid, the partition boundaries, the held-out draws or any cached artifact - they
/// subset and re-phase `Corpus::train_refs`, which is enumerated from the contract in
/// milliseconds and is cached by nothing. Putting them in the contract would invalidate the
/// bar-audit and market-grid caches for a change no cached artifact is a function of.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RowSelection {
    /// Fraction of the pool kept, drawn uniformly at random over the WHOLE pool.
    ///
    /// This is a NEGATIVE CONTROL, and it is here because it is provably inert on occupancy,
    /// not despite it. `n` rows drawn from a uniformly random `F·R`-subset are still a
    /// uniformly random `n`-subset of the original `R`, so `P[X_M = 0]` at a matched step is
    /// identical to the full pool's: the `F`-fold reduction in per-outcome multiplicity
    /// cancels the `F`-fold reduction in the pool exactly. An arm that moves under this knob
    /// moved for a reason that is NOT occupancy - block structure, regime mix, epoch
    /// boundaries - which is exactly what a control is for.
    pub fraction: f64,
    /// Keep every `K`-th row of each ticker's chronological grid, with a per-ticker residue
    /// drawn from this selection's seed so the retained lattice is not phase-aligned across
    /// the universe.
    ///
    /// This is the knob that really moves multiplicity: row stride `K · pred_len` against a
    /// dense supervised span of `window` bars leaves `window / (K · pred_len) + 1` exposures
    /// per interior outcome, so `K = window/pred_len + 1` reaches `M = 1` and a fully fresh
    /// sweep. Unlike [`Self::fraction`] it thins the pool WITHOUT thinning the outcome
    /// support: the retained rows still span every ticker and every regime, so the distinct
    /// outcome count falls only by the overlap it removes.
    pub stride_multiple: usize,
    pub patch_phase: PatchPhase,
    /// The seed both the per-ticker residues and the per-row phases are drawn from. Stamped so
    /// the arm reproduces and so two arms that claim the same knobs but drew different rows
    /// are distinguishable.
    pub seed: u64,
}

impl RowSelection {
    /// The selection every arm before these knobs existed ran: the whole pool, one shared
    /// phase. [`select`] takes no random draw at all under it, which is what makes the
    /// bit-exactness of the default provable rather than asserted.
    pub fn identity(seed: u64) -> Self {
        Self {
            fraction: 1.,
            stride_multiple: 1,
            patch_phase: PatchPhase::Fixed,
            seed,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.fraction >= 1. && self.stride_multiple <= 1 && self.patch_phase == PatchPhase::Fixed
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.fraction.is_finite() && self.fraction > 0. && self.fraction <= 1.,
            "--row-fraction must lie in (0, 1]; {} keeps no rows or more than the pool holds",
            self.fraction
        );
        ensure!(
            self.stride_multiple >= 1,
            "--row-stride-multiple must be at least 1; 1 keeps every row"
        );
        Ok(())
    }

    /// One line naming the arm, for chart titles and the startup log. Never contains `=`, so
    /// it is safe to interpolate into a report series label.
    pub fn summary(&self) -> String {
        if self.is_identity() {
            return "full row pool, one shared patch phase".to_owned();
        }
        let mut parts = Vec::new();
        if self.stride_multiple > 1 {
            parts.push(format!(
                "every {}th row per ticker",
                self.stride_multiple
            ));
        }
        if self.fraction < 1. {
            parts.push(format!("{:.4} of the pool at random", self.fraction));
        }
        if self.patch_phase == PatchPhase::Random {
            parts.push("per-row patch phase".to_owned());
        }
        format!("{} (seed {})", parts.join(", "), self.seed)
    }
}

/// The dense supervision lattice one row carries, derived from the model geometry rather than
/// restated: `patch_len` is the origin stride, `--min-history` decides which patch boundary is
/// the first ACTIVE origin, and the two together fix how far back a row's supervision reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupervisionGeometry {
    /// Absolute-bar stride between a row's causal origins: `patch_len`.
    pub stride: usize,
    /// Bars from a row's earliest active origin to its final origin.
    pub window: usize,
    /// Active causal origins per row, after the `--min-history` mask.
    pub origins_per_row: usize,
    /// Row-to-row stride in bars, `pred_len`.
    pub row_stride: usize,
    pub pred_len: usize,
}

impl SupervisionGeometry {
    pub fn new(
        seq_len: usize,
        patch_len: usize,
        min_history: usize,
        pred_len: usize,
    ) -> Result<Self> {
        ensure!(
            patch_len > 0 && seq_len % patch_len == 0,
            "patches must cover the context exactly"
        );
        ensure!(
            pred_len % patch_len == 0,
            "the row stride {pred_len} must be a multiple of the origin stride {patch_len}, or \
             the rows' origin lattices do not coincide and no per-outcome multiplicity exists"
        );
        let origins = seq_len / patch_len;
        // The completeness mask is `bars >= min_history`, and origin token `i` has
        // `(i + 1) * patch_len` context bars behind it inclusive.
        let first = min_history.div_ceil(patch_len).saturating_sub(1);
        ensure!(
            first < origins,
            "--min-history {min_history} leaves no active origin in a {seq_len}-bar context"
        );
        Ok(Self {
            stride: patch_len,
            window: (origins - 1 - first) * patch_len,
            origins_per_row: origins - first,
            row_stride: pred_len,
            pred_len,
        })
    }

    /// Per-epoch exposures of an INTERIOR outcome that sits on the row lattice itself.
    ///
    /// Stated for the report's pre-registration; the census measures the real distribution and
    /// nothing downstream trusts this number over the histogram. It is worth naming anyway
    /// because at the production geometry it is not merely the modal value, it is the value of
    /// EVERY interior origin: an origin `r` bars off the lattice is reached by
    /// `floor((window - r) / row_stride) + 1` rows with `r` running over
    /// `0, patch_len, .., row_stride - patch_len`, so the count is uniform exactly when
    /// [`Self::has_uniform_interior`] holds. It does at `window 5744, row_stride 192,
    /// patch_len 16` - `5744 mod 192` is 176, which is `192 - 16` - and that is why "each
    /// interior outcome is supervised 30 times" is a statement about all of them rather than
    /// about one residue.
    pub fn interior_multiplicity(&self) -> usize {
        self.window / self.row_stride + 1
    }

    /// Whether every interior origin carries [`Self::interior_multiplicity`] exposures, rather
    /// than the lattice residues splitting between that and one fewer.
    pub fn has_uniform_interior(&self) -> bool {
        self.window % self.row_stride >= self.row_stride - self.stride
    }
}

/// The measured supervision structure of one arm's retained row pool.
#[derive(Clone, Debug)]
pub struct SupervisionCensus {
    pub selection: RowSelection,
    pub geometry: SupervisionGeometry,
    /// Retained rows: the pool one epoch consumes, and the `R` of every occupancy figure.
    pub rows: usize,
    /// `histogram[m]` distinct absolute origins are supervised by exactly `m` retained rows.
    /// Index 0 is unused and always zero: an origin no row reaches is not in the support.
    pub histogram: Vec<u64>,
    /// Rows whose `target_count` is short of `pred_len`, i.e. the per-ticker tails that are
    /// the only place multiplicity is not flat in the horizon.
    pub partial_target_rows: usize,
    /// Rows the phase shift pushed past the training band or into an excluded in-period hole.
    pub phase_dropped_rows: usize,
    /// Share of retained rows whose final origin falls in each tenth of the training period,
    /// beside the same profile for the FULL pool. The regime-mix check: a chronological slice
    /// would put ~1 in one decile and 0 in the rest.
    pub retained_deciles: [f64; 10],
    pub full_deciles: [f64; 10],
}

impl SupervisionCensus {
    /// Distinct absolute origins any retained row supervises.
    pub fn support(&self) -> u64 {
        self.histogram.iter().sum()
    }

    /// Mean per-epoch exposures of a supervised outcome, over the measured histogram.
    pub fn mean_multiplicity(&self) -> f64 {
        let support = self.support();
        if support == 0 {
            return 0.;
        }
        let total: f64 = self
            .histogram
            .iter()
            .enumerate()
            .map(|(m, count)| m as f64 * *count as f64)
            .sum();
        total / support as f64
    }

    /// Largest absolute deviation between the retained and full decile profiles, in shares.
    /// This is the regime-mix statistic: a uniform subsample leaves it at sampling noise, a
    /// chronological slice drives it to ~0.1 per emptied decile.
    pub fn decile_drift(&self) -> f64 {
        self.retained_deciles
            .iter()
            .zip(self.full_deciles.iter())
            .map(|(retained, full)| (retained - full).abs())
            .fold(0., f64::max)
    }

    /// Fraction of this arm's distinct supervised outcomes that have received at least `passes`
    /// gradient passes after `step` optimizer steps at `batch`.
    ///
    /// Epochs are exact rather than approximated: `e` complete shuffles give every outcome its
    /// full `M` exposures, and only the partial epoch is hypergeometric. That matters as soon
    /// as an arm sweeps its pool more than once, which is the whole point of a thinned pool.
    pub fn coverage(&self, step: usize, batch: usize, passes: u32) -> f64 {
        let support = self.support();
        if support == 0 || passes == 0 {
            return f64::from(passes == 0);
        }
        let consumed = (step as u128) * (batch as u128);
        let rows = self.rows as u128;
        let epochs = (consumed / rows) as f64;
        let partial = (consumed % rows) as usize;
        let mut covered = 0.;
        for (multiplicity, count) in self.histogram.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let m = multiplicity as u32;
            let already = epochs * f64::from(m);
            let owed = f64::from(passes) - already;
            let reached = if owed <= 0. {
                1.
            } else {
                // `owed` is at most `passes`, so this saturates at a handful of terms.
                1. - hypergeometric_below(self.rows, multiplicity, partial, owed.ceil() as u32)
            };
            covered += reached * *count as f64;
        }
        covered / support as f64
    }

    /// Mean gradient passes per distinct supervised outcome after `step` steps, as a share of
    /// this arm's own mean multiplicity. Dimensionless on purpose: it shares the occupancy
    /// panel's axis, and multiplying it by [`Self::mean_multiplicity`] - which the panel title
    /// carries - recovers the raw count.
    pub fn exposure_share(&self, step: usize, batch: usize) -> f64 {
        if self.rows == 0 {
            return 0.;
        }
        (step as f64) * (batch as f64) / self.rows as f64
    }

    /// First step at which [`Self::coverage`] for a single pass reaches `threshold`. This is
    /// the SATURATION STEP: pre-registered from the census before an arm runs, and read off
    /// the chart afterwards.
    ///
    /// Monotone in the step, so a bisection is exact; the bracket is one epoch, because a
    /// complete sweep covers every supported outcome by construction.
    pub fn saturation_step(&self, threshold: f64, batch: usize) -> usize {
        let mut low = 0usize;
        let mut high = self.rows.div_ceil(batch.max(1));
        while low < high {
            let mid = low + (high - low) / 2;
            if self.coverage(mid, batch, 1) >= threshold {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    }

    /// Coverage restricted to the INTERIOR outcomes - the ones at the geometry's own
    /// multiplicity. Reported beside the global figure because the two differ by an order of
    /// magnitude in the unseen tail and the literature of this project has quoted the interior
    /// number as if it were global.
    pub fn interior_coverage(&self, step: usize, batch: usize) -> f64 {
        let interior = self.geometry.interior_multiplicity();
        let Some(count) = self.histogram.get(interior) else {
            return f64::NAN;
        };
        if *count == 0 {
            return f64::NAN;
        }
        let consumed = (step as u128) * (batch as u128);
        let rows = self.rows as u128;
        if consumed >= rows {
            return 1.;
        }
        1. - hypergeometric_below(self.rows, interior, (consumed % rows) as usize, 1)
    }

    /// Share of the support sitting at the geometry's interior multiplicity, and NaN when this
    /// arm has no outcome there at all.
    ///
    /// NaN rather than 0 because the two are different facts and a thinned arm produces the
    /// second one: `--row-stride-multiple 30` leaves every outcome at `M = 1`, so "0 of the
    /// support is interior" would read as a measured collapse when the truth is that the
    /// geometry's interior multiplicity is not a quantity that arm has. Every interior figure
    /// beside it - [`Self::interior_coverage`] - reads NaN under the same condition, so the
    /// pair cannot half-report.
    pub fn interior_share(&self) -> f64 {
        let support = self.support();
        let interior = self
            .histogram
            .get(self.geometry.interior_multiplicity())
            .copied()
            .unwrap_or(0);
        if support == 0 || interior == 0 {
            return f64::NAN;
        }
        interior as f64 / support as f64
    }
}

/// `P[X < passes]` for `X ~ Hypergeometric(population, successes, draws)`.
///
/// `P[X = 0] = C(R-M, n)/C(R, n) = Π_{j<M} (R-n-j)/(R-j)` - `M` terms, no factorial, no
/// `lgamma`, and exact in fp64 for every `(R, M, n)` this corpus produces. The tail then
/// follows from `P[X=j+1]/P[X=j] = (M-j)(n-j) / ((j+1)(R-M-n+j+1))`.
fn hypergeometric_below(population: usize, successes: usize, draws: usize, passes: u32) -> f64 {
    if passes == 0 {
        return 0.;
    }
    if successes == 0 || draws == 0 {
        return 1.;
    }
    if draws >= population {
        return 0.;
    }
    let (r, m, n) = (population as f64, successes as f64, draws as f64);
    let mut term = 1.;
    for j in 0..successes {
        let numerator = r - n - j as f64;
        if numerator <= 0. {
            // Every one of the outcome's supporting rows is already drawn.
            term = 0.;
            break;
        }
        term *= numerator / (r - j as f64);
    }
    let mut total = term;
    for j in 0..u64::from(passes - 1) {
        let j = j as f64;
        let denominator = (j + 1.) * (r - m - n + j + 1.);
        if denominator <= 0. {
            break;
        }
        term *= (m - j) * (n - j) / denominator;
        if !term.is_finite() || term <= 0. {
            break;
        }
        total += term;
    }
    total.clamp(0., 1.)
}

/// Apply `selection` to `corpus.train_refs` in place and census what it retained.
///
/// Order of operations, and it is load-bearing:
///
/// 1. per-ticker stride thinning, over each ticker's chronological row grid;
/// 2. per-row phase shift, then a re-test of admissibility - a phased row whose dense
///    supervision leaves the training band or enters an excluded in-period hole is DROPPED, so
///    every guarantee the enumeration established survives the shift rather than being
///    weakened by it;
/// 3. uniform thinning over the whole surviving pool.
///
/// Under [`RowSelection::identity`] not one random draw is taken and not one origin moves, so
/// the retained pool is the enumerated pool element for element.
pub fn select(
    corpus: &mut Corpus,
    selection: RowSelection,
    geometry: SupervisionGeometry,
) -> Result<SupervisionCensus> {
    selection.validate()?;
    ensure!(
        !corpus.train_refs.is_empty(),
        "cannot apply a row selection to an empty training pool"
    );
    let tickers = corpus.contract.tickers.len();
    let full = std::mem::take(&mut corpus.train_refs);
    let full_deciles = deciles(corpus, &full);
    let mut grouped: Vec<Vec<usize>> = vec![Vec::new(); tickers];
    for reference in &full {
        grouped[reference.ticker].push(reference.origin);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(selection.seed ^ SELECTION_STREAM);
    let mut phase_dropped = 0usize;
    let mut kept: Vec<WindowRef> = Vec::with_capacity(full.len());
    for (ticker, origins) in grouped.iter().enumerate() {
        let residue = if selection.stride_multiple > 1 {
            rng.random_range(0..selection.stride_multiple)
        } else {
            0
        };
        for (index, &origin) in origins.iter().enumerate() {
            if selection.stride_multiple > 1 && index % selection.stride_multiple != residue {
                continue;
            }
            let origin = match selection.patch_phase {
                PatchPhase::Fixed => origin,
                PatchPhase::Random => origin + rng.random_range(0..geometry.stride),
            };
            // Re-tested, not assumed: a shifted origin can leave the training band or reach
            // into an excluded in-period hole, and `Corpus::training_target_count` is the same
            // predicate the enumeration applied, so a phased pool inherits exactly the
            // guarantees the unphased one had.
            if selection.patch_phase == PatchPhase::Random
                && corpus.training_target_count(ticker, origin).is_none()
            {
                phase_dropped += 1;
                continue;
            }
            kept.push(WindowRef { ticker, origin });
        }
    }
    if selection.fraction < 1. {
        let target = (selection.fraction * kept.len() as f64).round() as usize;
        ensure!(
            target > 0,
            "--row-fraction {} keeps 0 of {} rows",
            selection.fraction,
            kept.len()
        );
        // A uniform subset of EXACT size, not a per-row coin flip: the retained count is then a
        // function of the fraction alone and two arms at the same fraction have the same pool
        // size. The order is restored afterwards so the pool stays ticker-major whatever the
        // draw was, and the per-epoch shuffle is left as the only thing that orders training.
        let mut index: Vec<u32> = (0..kept.len() as u32).collect();
        index.shuffle(&mut rng);
        index.truncate(target);
        index.sort_unstable();
        kept = index.into_iter().map(|i| kept[i as usize]).collect();
    }
    ensure!(
        !kept.is_empty(),
        "the row selection {} retained no training rows",
        selection.summary()
    );
    let retained_deciles = if selection.is_identity() {
        full_deciles
    } else {
        deciles(corpus, &kept)
    };
    let partial_target_rows = kept
        .par_iter()
        .filter(|reference| {
            corpus.training_target_count(reference.ticker, reference.origin)
                != Some(geometry.pred_len)
        })
        .count();
    let mut by_ticker: Vec<Vec<usize>> = vec![Vec::new(); tickers];
    for reference in &kept {
        by_ticker[reference.ticker].push(reference.origin);
    }
    let histogram = multiplicity_histogram(&by_ticker, geometry);
    corpus.train_refs = kept;
    Ok(SupervisionCensus {
        selection,
        geometry,
        rows: corpus.train_refs.len(),
        histogram,
        partial_target_rows,
        phase_dropped_rows: phase_dropped,
        retained_deciles,
        full_deciles,
    })
}

/// Share of `refs` whose final origin falls in each tenth of the pool's own timestamp span.
///
/// One memory-mapped bar header per row, parallel over rows; the span is taken from the pool
/// itself so the profile of the full pool is exactly `0.1` per decile up to the corpus's own
/// non-uniform density in time, and the retained profile is directly comparable to it.
fn deciles(corpus: &Corpus, refs: &[WindowRef]) -> [f64; 10] {
    if refs.is_empty() {
        return [0.; 10];
    }
    let stamps: Vec<i64> = refs
        .par_iter()
        .map(|reference| corpus.ticker(*reference).timestamp(reference.origin))
        .collect();
    let (first, last) = stamps
        .par_iter()
        .fold(
            || (i64::MAX, i64::MIN),
            |(lo, hi), &ts| (lo.min(ts), hi.max(ts)),
        )
        .reduce(
            || (i64::MAX, i64::MIN),
            |a, b| (a.0.min(b.0), a.1.max(b.1)),
        );
    let span = (last - first).max(1) as f64;
    let mut counts = [0u64; 10];
    for ts in stamps {
        let bucket = (((ts - first) as f64 / span) * 10.) as usize;
        counts[bucket.min(9)] += 1;
    }
    let total = refs.len() as f64;
    std::array::from_fn(|i| counts[i] as f64 / total)
}

/// Distinct absolute origins by exactly how many retained rows supervise them.
///
/// An origin `A` is supervised by row `O` exactly when `A ≡ O (mod stride)` and
/// `O - window ≤ A ≤ O`, so within one residue class this is interval-coverage counting: each
/// row is the interval `[O - window, O]` and the histogram is the length of each constant
/// segment of the coverage function, measured in lattice points. `window` is a multiple of
/// `stride`, so every breakpoint lands on the residue class and no lattice point is
/// miscounted. Two sorted event lists merged with a running sum, per residue, per ticker:
/// `O(rows log rows)` in total and independent of how many origins the corpus holds, which is
/// what makes this affordable at 31 million supervised origins.
fn multiplicity_histogram(
    by_ticker: &[Vec<usize>],
    geometry: SupervisionGeometry,
) -> Vec<u64> {
    let width = geometry.origins_per_row + 1;
    let stride = geometry.stride as i64;
    let window = geometry.window as i64;
    by_ticker
        .par_iter()
        .fold(
            || vec![0u64; width],
            |mut histogram, origins| {
                if origins.is_empty() {
                    return histogram;
                }
                let mut classes: Vec<Vec<i64>> = vec![Vec::new(); geometry.stride];
                for &origin in origins {
                    classes[origin % geometry.stride].push(origin as i64);
                }
                for class in &mut classes {
                    if class.is_empty() {
                        continue;
                    }
                    class.sort_unstable();
                    let (mut open, mut close, mut count) = (0usize, 0usize, 0i64);
                    let mut cursor = class[0] - window;
                    while close < class.len() {
                        let next_open = class
                            .get(open)
                            .map(|origin| origin - window)
                            .unwrap_or(i64::MAX);
                        let next_close = class[close] + stride;
                        let position = next_open.min(next_close);
                        if count > 0 && position > cursor {
                            let points = (position - cursor) / stride;
                            histogram[(count as usize).min(width - 1)] += points as u64;
                        }
                        cursor = position;
                        while open < class.len() && class[open] - window == position {
                            open += 1;
                            count += 1;
                        }
                        while close < class.len() && class[close] + stride == position {
                            close += 1;
                            count -= 1;
                        }
                    }
                }
                histogram
            },
        )
        .reduce(
            || vec![0u64; width],
            |mut a, b| {
                for (slot, value) in a.iter_mut().zip(b) {
                    *slot += value;
                }
                a
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const PRODUCTION: SupervisionGeometry = SupervisionGeometry {
        stride: 16,
        window: 5744,
        origins_per_row: 360,
        row_stride: 192,
        pred_len: 192,
    };

    #[test]
    fn the_production_geometry_is_derived_from_the_model_config_and_not_restated() {
        let geometry = SupervisionGeometry::new(6000, 16, 256, 192).unwrap();
        assert_eq!(geometry, PRODUCTION);
        assert_eq!(geometry.interior_multiplicity(), 30);
        // The property that makes "30 exposures per interior outcome" a claim about every
        // interior outcome rather than about the row lattice alone: `5744 mod 192` is exactly
        // `192 - 16`, so the off-lattice residues carry 30 too.
        assert!(geometry.has_uniform_interior());
        assert_eq!(geometry.window % geometry.row_stride, geometry.row_stride - geometry.stride);
        // And it is a property of the geometry, not a law: shift `--min-history` by one patch
        // and the interior splits between 29 and 30.
        assert!(!SupervisionGeometry::new(6000, 16, 272, 192).unwrap().has_uniform_interior());
        // `min_history` sits exactly on a patch boundary here; one bar more must not change
        // which boundary is the first active origin, and one bar past it must.
        assert_eq!(SupervisionGeometry::new(6000, 16, 241, 192).unwrap(), PRODUCTION);
        assert_eq!(
            SupervisionGeometry::new(6000, 16, 257, 192).unwrap().origins_per_row,
            359
        );
        // A row stride that is not a multiple of the origin stride has no multiplicity at all.
        assert!(SupervisionGeometry::new(6000, 16, 256, 100).is_err());
        assert!(SupervisionGeometry::new(6000, 16, 6001, 192).is_err());
    }

    /// The histogram is the object every occupancy number is computed from, so it is checked
    /// against a lattice counted by brute force rather than against the closed form that
    /// produced it.
    #[test]
    fn the_multiplicity_histogram_matches_a_brute_force_count_of_the_lattice() {
        let geometry = SupervisionGeometry::new(64, 4, 12, 8).unwrap();
        assert_eq!(geometry.window, 52);
        assert_eq!(geometry.interior_multiplicity(), 7);
        for phases in [vec![0usize; 40], (0..40).map(|i| (i * 7) % 4).collect()] {
            let origins: Vec<usize> = (0..40).map(|i| 63 + i * 8 + phases[i]).collect();
            let by_ticker = vec![origins.clone()];
            let histogram = multiplicity_histogram(&by_ticker, geometry);
            let mut brute = vec![0u64; geometry.origins_per_row + 1];
            let low = origins.iter().min().unwrap() - geometry.window;
            let high = *origins.iter().max().unwrap();
            for absolute in low..=high {
                let count = origins
                    .iter()
                    .filter(|&&origin| {
                        origin % geometry.stride == absolute % geometry.stride
                            && absolute + geometry.window >= origin
                            && absolute <= origin
                    })
                    .count();
                if count > 0 {
                    brute[count] += 1;
                }
            }
            assert_eq!(histogram, brute, "phases {phases:?}");
        }
    }

    #[test]
    fn a_phase_jittered_lattice_holds_more_origins_at_lower_multiplicity() {
        let geometry = SupervisionGeometry::new(64, 4, 12, 8).unwrap();
        let aligned: Vec<usize> = (0..64).map(|i| 63 + i * 8).collect();
        let jittered: Vec<usize> = (0..64).map(|i| 63 + i * 8 + (i * 5) % 4).collect();
        let flat = census_of(&aligned, geometry);
        let phased = census_of(&jittered, geometry);
        assert!(
            phased.support() > 3 * flat.support(),
            "a per-row phase must multiply the origin lattice: {} against {}",
            phased.support(),
            flat.support()
        );
        assert!(
            phased.mean_multiplicity() < flat.mean_multiplicity() / 2.,
            "and thin the exposures by the same factor: {} against {}",
            phased.mean_multiplicity(),
            flat.mean_multiplicity()
        );
        // Adding origins is not adding shocks: both lattices are supported by the same rows.
        assert_eq!(aligned.len(), jittered.len());
    }

    fn census_of(origins: &[usize], geometry: SupervisionGeometry) -> SupervisionCensus {
        SupervisionCensus {
            selection: RowSelection::identity(1),
            geometry,
            rows: origins.len(),
            histogram: multiplicity_histogram(&[origins.to_vec()], geometry),
            partial_target_rows: 0,
            phase_dropped_rows: 0,
            retained_deciles: [0.1; 10],
            full_deciles: [0.1; 10],
        }
    }

    /// The hypergeometric closed form against an independent product, and against the two
    /// boundaries where a coverage curve is allowed to be exactly 0 or exactly 1.
    #[test]
    fn the_unseen_probability_matches_the_falling_factorial_product() {
        let (population, successes, draws) = (2_455_276usize, 30usize, 512_000usize);
        let reference: f64 = (0..successes)
            .map(|j| {
                (population - draws - j) as f64 / (population - j) as f64
            })
            .product();
        let closed = hypergeometric_below(population, successes, draws, 1);
        assert!(
            (closed - reference).abs() < 1e-15,
            "{closed} against {reference}"
        );
        // The interior figure this project has quoted, reproduced from the formula alone.
        assert!((closed - 0.000_897_439).abs() < 1e-8, "{closed}");
        assert_eq!(hypergeometric_below(100, 5, 0, 1), 1.);
        assert_eq!(hypergeometric_below(100, 5, 96, 1), 0.);
        // `P[X < k]` is a proper CDF: nondecreasing in `k`, and 1 once `k` exceeds `M`.
        let cdf: Vec<f64> = (1..=31)
            .map(|k| hypergeometric_below(population, successes, draws, k))
            .collect();
        assert!(cdf.windows(2).all(|pair| pair[1] >= pair[0]), "{cdf:?}");
        assert!((cdf[30] - 1.).abs() < 1e-9, "{}", cdf[30]);
    }

    /// The negative control's whole justification, as a test: a uniformly thinned pool has the
    /// IDENTICAL matched-step coverage curve, because the multiplicity thinning cancels the
    /// pool thinning exactly. If this ever fails, `--row-fraction` has become an occupancy
    /// intervention and stops being a control.
    #[test]
    fn uniform_thinning_leaves_matched_step_coverage_invariant() {
        let geometry = PRODUCTION;
        let full = SupervisionCensus {
            selection: RowSelection::identity(1),
            geometry,
            rows: 2_455_276,
            histogram: histogram_at(30, 31_000_000),
            partial_target_rows: 0,
            phase_dropped_rows: 0,
            retained_deciles: [0.1; 10],
            full_deciles: [0.1; 10],
        };
        // The retained multiplicity of an outcome is not `M/4` but `Binomial(M, 1/4)`, so the
        // invariance has to be stated over that LAW rather than over its mean: an arm's own
        // support excludes the outcomes that lost every supporting row, and the survivors
        // therefore carry an above-average retained multiplicity. At `M = 30` that conditioning
        // costs `0.75^30 = 1.8e-4` of the support and is invisible, which is why the coverage
        // curves below coincide; at a small `M` it is a real effect and the corpus-level test
        // states the weaker claim it supports.
        let binomial: Vec<f64> = (0..=30)
            .map(|k| {
                let mut probability = 0.25f64.powi(k) * 0.75f64.powi(30 - k);
                for j in 0..k {
                    probability *= (30 - j) as f64 / (j + 1) as f64;
                }
                probability
            })
            .collect();
        for step in [500usize, 1000, 2000] {
            let consumed = step * 256;
            let control = hypergeometric_below(full.rows, 30, consumed, 1);
            let thinned: f64 = binomial
                .iter()
                .enumerate()
                .map(|(retained, weight)| {
                    weight * hypergeometric_below(full.rows / 4, retained, consumed, 1)
                })
                .sum();
            assert!(
                (control - thinned).abs() < 1e-3,
                "step {step}: a uniformly thinned pool must leave the per-outcome unseen \
                 probability at {control}, not {thinned}; if it does not, --row-fraction has \
                 become an occupancy intervention and stops being a control"
            );
        }
        // And the same claim on the object the CHART draws: an arm's own coverage curve over
        // its own support. The quarter arm's histogram is the binomial spread above and NOT a
        // point mass at the mean 7.5, because `E[q^M]` is not `q^E[M]` - the mean surrogate is
        // off by 6e-3 at this pool, which is 30x the tolerance here and would have been read as
        // a real occupancy effect.
        let mut spread = vec![0u64; geometry.origins_per_row + 1];
        for (retained, weight) in binomial.iter().enumerate() {
            spread[retained] = (weight * 31_000_000.).round() as u64;
        }
        // An outcome that lost every supporting row is not in the arm's support.
        spread[0] = 0;
        let quarter = SupervisionCensus {
            rows: full.rows / 4,
            histogram: spread,
            ..full.clone()
        };
        assert!(
            (quarter.mean_multiplicity() - 7.5).abs() < 0.01,
            "{}",
            quarter.mean_multiplicity()
        );
        for step in [500usize, 1000, 2000] {
            let a = full.coverage(step, 256, 1);
            let b = quarter.coverage(step, 256, 1);
            assert!(
                (a - b).abs() < 1e-3,
                "step {step}: {a} against {b} - uniform thinning must not move occupancy"
            );
        }
    }

    /// Stride thinning is the intervention: it removes overlap, not support, so coverage
    /// arrives EARLIER in the step index.
    #[test]
    fn stride_thinning_moves_the_saturation_step_earlier() {
        let geometry = PRODUCTION;
        let control = SupervisionCensus {
            selection: RowSelection::identity(1),
            geometry,
            rows: 2_455_276,
            histogram: histogram_at(30, 31_000_000),
            partial_target_rows: 0,
            phase_dropped_rows: 0,
            retained_deciles: [0.1; 10],
            full_deciles: [0.1; 10],
        };
        let fresh = SupervisionCensus {
            rows: control.rows / 30,
            histogram: histogram_at(1, 31_000_000 / 30),
            ..control.clone()
        };
        let (slow, fast) = (
            control.saturation_step(0.999, 256),
            fresh.saturation_step(0.999, 256),
        );
        assert!(
            fast * 4 < slow,
            "an M-of-1 pool must saturate far earlier: {fast} against {slow}"
        );
        // A complete sweep covers everything the pool supports, by construction.
        assert!((fresh.coverage(fresh.rows / 256 + 1, 256, 1) - 1.).abs() < 1e-12);
        // And a second sweep is pure re-presentation: single-pass coverage cannot rise, while
        // two-pass coverage must.
        assert!(fresh.coverage(2 * (fresh.rows / 256 + 1), 256, 2) > 0.99);
    }

    fn histogram_at(multiplicity: usize, count: u64) -> Vec<u64> {
        let mut histogram = vec![0u64; PRODUCTION.origins_per_row + 1];
        histogram[multiplicity.max(1)] = count;
        histogram
    }

    #[test]
    fn the_arm_summary_never_carries_an_equals_sign() {
        for selection in [
            RowSelection::identity(7),
            RowSelection {
                fraction: 0.25,
                stride_multiple: 30,
                patch_phase: PatchPhase::Random,
                seed: 7,
            },
        ] {
            assert!(!selection.summary().contains('='), "{}", selection.summary());
        }
    }

    #[test]
    fn an_unusable_selection_is_refused_before_the_pool_is_touched() {
        for selection in [
            RowSelection { fraction: 0., ..RowSelection::identity(1) },
            RowSelection { fraction: 1.5, ..RowSelection::identity(1) },
            RowSelection { fraction: f64::NAN, ..RowSelection::identity(1) },
            RowSelection { stride_multiple: 0, ..RowSelection::identity(1) },
        ] {
            assert!(selection.validate().is_err(), "{selection:?}");
        }
        assert!(RowSelection::identity(1).validate().is_ok());
    }

    /// One scratch corpus, three tickers, real `Corpus::load`, and every claim the knobs make
    /// checked against it rather than against the arithmetic that motivated them.
    ///
    /// The bit-exactness half is the load-bearing one. `--row-fraction 1 --patch-phase fixed`
    /// has to leave the row population, the origin population, every validity mask and every
    /// integer index exactly as they were, and "exactly" is checked on the PACKED BATCH the
    /// loader hands the device - a changed mask is a changed dataset, not a rounding
    /// difference, and comparing the reference list alone would not see one.
    #[test]
    fn the_default_selection_is_bit_exact_and_each_knob_moves_only_what_it_claims() {
        use crate::torch::timexer_segment::corpus::{Corpus, RESOLUTION_MS};
        use crate::torch::timexer_segment::features::FeatureSet;
        use shared::bars::{bar_file_path, write_bar_file, PackedBar};
        use std::fs;

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
            "/var/tmp/timexer-row-selection-test-{}-{nonce}",
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
            write_bar_file(&bar_file_path(&directory.0, ticker, 300), ticker, 300, &bars).unwrap();
        }
        // A ticker whose valid-bar ordinals differ from its raw indices, so the phase shift is
        // applied on the same filtered ordinals the enumeration used.
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
        let features = FeatureSet { spy: false, ..FeatureSet::ALL };
        let (context, pred_len, common_context) = (64usize, 8usize, 64usize);
        // `patch_len` 4 and `--min-history` 20 against a 64-bar context: 16 origins per row,
        // the first four masked out, so 12 active origins reaching 44 bars back and an interior
        // multiplicity of 6. `44 mod 8` is `8 - 4`, so every interior origin carries 6 and not
        // some 6 and some 5 - the same uniformity the production geometry has, which is what
        // makes this fixture structurally like production rather than merely smaller.
        let geometry = SupervisionGeometry::new(context, 4, 20, pred_len).unwrap();
        assert_eq!(geometry.interior_multiplicity(), 6);
        assert!(geometry.has_uniform_interior());
        let load = |sections| {
            Corpus::load(
                &directory.0,
                &[],
                context,
                pred_len,
                common_context,
                &features,
                1,
                sections,
            )
            .unwrap()
        };
        let mut corpus = load(0);
        let enumerated = corpus.train_refs.clone();
        let contract = serde_json::to_vec(&corpus.contract).unwrap();
        let probe: Vec<_> = enumerated.iter().copied().take(37).collect();
        let before = corpus.host_batch(&probe).unwrap();

        // ---- claim 1: the default is the identity, to the byte ---------------------------
        let census = select(&mut corpus, RowSelection::identity(20260905), geometry).unwrap();
        assert_eq!(
            corpus.train_refs, enumerated,
            "--row-fraction 1 --patch-phase fixed must not move a single row"
        );
        assert_eq!(
            serde_json::to_vec(&corpus.contract).unwrap(),
            contract,
            "a knob at its default must not perturb the corpus contract, or every cached \
             artifact keyed on it is invalidated for nothing"
        );
        let after = corpus.host_batch(&probe).unwrap();
        for (name, a, b) in [
            ("log_prices", &before.log_prices, &after.log_prices),
            ("valid", &before.valid, &after.valid),
            ("aux", &before.aux, &after.aux),
            ("market_cum", &before.market_cum, &after.market_cum),
            ("anchor", &before.anchor, &after.anchor),
        ] {
            assert!(
                a.equal(b),
                "{name} is not bit-identical under the default selection"
            );
        }
        assert_eq!(before.valid_target_bars, after.valid_target_bars);
        assert_eq!(census.rows, enumerated.len());
        assert_eq!(census.phase_dropped_rows, 0);
        assert_eq!(census.retained_deciles, census.full_deciles);
        assert_eq!(census.decile_drift(), 0.);
        // The corpus is dense in the horizon apart from at most one short tail per ticker: the
        // last row's owned target count is `train_end - origin - 1`, which equals `pred_len`
        // only when the grid happens to divide the band exactly.
        assert!(census.partial_target_rows <= corpus.contract.tickers.len());
        assert!(census.interior_share() > 0.9, "{}", census.interior_share());
        let control = census.clone();

        // ---- claim 2: stride thinning removes overlap, not support ------------------------
        corpus.train_refs = enumerated.clone();
        let strided = select(
            &mut corpus,
            RowSelection {
                stride_multiple: 7,
                ..RowSelection::identity(20260905)
            },
            geometry,
        )
        .unwrap();
        // Every ticker keeps `ceil((n_t - residue) / 7)` of its `n_t` rows, so the retained
        // count is within one row per ticker of `n_t / 7`.
        let tickers = corpus.contract.tickers.len();
        assert!(
            strided.rows * 7 >= control.rows - 7 * tickers
                && strided.rows * 7 <= control.rows + 7 * tickers,
            "{} of {} rows over {tickers} tickers",
            strided.rows,
            control.rows
        );
        // Row stride 7 x 8 bars exceeds the 44-bar supervised span, so no two retained rows can
        // reach the same origin: multiplicity is exactly 1 and the sweep is fully fresh.
        assert_eq!(
            strided.mean_multiplicity(),
            1.,
            "a stride past the supervised span must leave exactly one exposure per outcome"
        );
        // Support falls only by the overlap that is gone, so far less than the pool does.
        assert!(
            strided.support() * 3 > control.support(),
            "{} of {} outcomes",
            strided.support(),
            control.support()
        );
        assert!(
            strided.saturation_step(0.999, 32) < control.saturation_step(0.999, 32),
            "thinning the overlap must move saturation earlier: {} against {}",
            strided.saturation_step(0.999, 32),
            control.saturation_step(0.999, 32)
        );
        // Every ticker keeps rows across the whole period: a stride is not a slice.
        assert!(
            strided.decile_drift() < 0.02,
            "stride thinning moved the regime mix by {}",
            strided.decile_drift()
        );

        // ---- claim 3: the uniform fraction is a NEGATIVE control -------------------------
        corpus.train_refs = enumerated.clone();
        let thinned = select(
            &mut corpus,
            RowSelection {
                fraction: 0.25,
                ..RowSelection::identity(20260905)
            },
            geometry,
        )
        .unwrap();
        assert_eq!(thinned.rows, (0.25 * control.rows as f64).round() as usize);
        assert!(
            thinned.decile_drift() < 0.02,
            "a uniform subsample must leave the regime mix alone; drift {}",
            thinned.decile_drift()
        );
        // The property that makes it a control: the pool and the per-outcome multiplicity fall
        // TOGETHER, by the same factor, which is what cancels in the unseen probability. The
        // support barely moves - an outcome is lost only if all of its supporting rows are - so
        // this knob buys no fresh outcomes at all.
        //
        // The coverage CURVE is asserted to coincide at the production geometry, not here: an
        // arm's own coverage is conditioned on its support, the survivors carry an
        // above-average retained multiplicity, and at this toy geometry's `M = 6` that
        // conditioning is worth `0.75^6 = 18%` of the support instead of the production
        // geometry's `0.75^30 = 0.018%`.
        let expected = control.mean_multiplicity() * 0.25;
        assert!(
            thinned.mean_multiplicity() > expected
                && thinned.mean_multiplicity() < expected + 1.,
            "a quarter pool must carry about a quarter of the exposures: {} against {expected}",
            thinned.mean_multiplicity()
        );
        assert!(
            thinned.support() * 10 > control.support() * 8,
            "uniform thinning must not remove support, only overlap: {} of {} outcomes",
            thinned.support(),
            control.support()
        );

        // ---- claim 4: the phase adds origins, keeps rows, and stays admissible ------------
        corpus.train_refs = enumerated.clone();
        let phased = select(
            &mut corpus,
            RowSelection {
                patch_phase: PatchPhase::Random,
                ..RowSelection::identity(20260905)
            },
            geometry,
        )
        .unwrap();
        assert!(
            phased.rows + phased.phase_dropped_rows == control.rows,
            "every control row is either phased or dropped"
        );
        assert!(
            phased.phase_dropped_rows <= corpus.contract.tickers.len(),
            "a forward shift can only cost the last row of a ticker, not {}",
            phased.phase_dropped_rows
        );
        assert!(
            phased.support() > 3 * control.support(),
            "a per-row phase must multiply the origin lattice by about patch_len: {} against {}",
            phased.support(),
            control.support()
        );
        assert!(
            phased.mean_multiplicity() < control.mean_multiplicity() / 2.,
            "and thin the exposures by the same factor: {} against {}",
            phased.mean_multiplicity(),
            control.mean_multiplicity()
        );
        // Every phased origin is still a TRAINING origin. A shift that pushed one past
        // `train_end` would train on the reserved calibration partition.
        for reference in &corpus.train_refs {
            let owned = corpus
                .training_target_count(reference.ticker, reference.origin)
                .unwrap_or_else(|| panic!("{reference:?} is not a training origin"));
            assert!(owned > 0 && owned <= pred_len);
        }
        // The phased lattice covers residues the aligned one never reached: that IS the extra
        // support, and it is the reason this is augmentation rather than a rephrasing.
        let residues: BTreeSet<usize> = corpus
            .train_refs
            .iter()
            .map(|reference| reference.origin % geometry.stride)
            .collect();
        assert_eq!(residues.len(), geometry.stride, "{residues:?}");
        assert_eq!(
            enumerated
                .iter()
                .map(|reference| reference.origin % geometry.stride)
                .collect::<BTreeSet<usize>>()
                .len(),
            1,
            "the unphased corpus is single-residue, which is what the knob exists to change"
        );
        // Reproducibility: the same seed retains the same pool, a different seed does not.
        corpus.train_refs = enumerated.clone();
        select(
            &mut corpus,
            RowSelection { patch_phase: PatchPhase::Random, ..RowSelection::identity(20260905) },
            geometry,
        )
        .unwrap();
        let repeat = corpus.train_refs.clone();
        corpus.train_refs = enumerated.clone();
        select(
            &mut corpus,
            RowSelection { patch_phase: PatchPhase::Random, ..RowSelection::identity(1) },
            geometry,
        )
        .unwrap();
        assert_ne!(repeat, corpus.train_refs, "the seed must select the pool");

        // ---- claim 5: the phase cannot leak into an in-period hole ------------------------
        // The composition the two knobs have to survive together: the hole excludes rows whose
        // dense supervision reaches it, and a forward phase shift moves that reach.
        let mut holed = load(24);
        assert!(!holed.in_period_refs.is_empty());
        let purged = holed.contract.in_period_purged_rows;
        assert!(purged > 0);
        select(
            &mut holed,
            RowSelection { patch_phase: PatchPhase::Random, ..RowSelection::identity(20260905) },
            geometry,
        )
        .unwrap();
        for reference in &holed.train_refs {
            let ticker = holed.ticker(*reference);
            let (lo, hi) = match ticker.in_period_hole() {
                Some(hole) => hole,
                None => continue,
            };
            let owned = pred_len.min(ticker.contract.train_end - reference.origin - 1);
            let first = (reference.origin + 2).saturating_sub(context);
            assert!(
                reference.origin + owned < lo || first > hi,
                "phased row {reference:?} supervises bars [{first}, {}] which reach the hole \
                 [{lo}, {hi}]",
                reference.origin + owned
            );
        }
    }
}
