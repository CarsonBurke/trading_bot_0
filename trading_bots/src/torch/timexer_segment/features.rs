use std::{fmt, str::FromStr, sync::LazyLock};

use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::bars::PackedBar;

use super::corpus::RESOLUTION_MS;
use crate::torch::dataset::et_offset_secs;

pub const SPY: &str = "SPY";
/// `ln(delta_minutes / 5)` saturates here: about ten calendar days between valid bars.
pub const GAP_LOG_CLIP: f32 = 8.0;
/// Floor under the cross-sectional dispersion before the log and the reciprocal: 1e-6 in
/// log-return units is 0.1 basis points, three orders below the smallest plausible five-minute
/// cross-sectional dispersion, so the clamp only ever fires on a numerically degenerate slot
/// (a single contributor, or identical closes) instead of letting `ln σ` reach -inf and `1/σ`
/// reach inf.
pub const DISPERSION_FLOOR: f64 = 1e-6;
/// `|z|` saturates here. At `--market-min-cross-section 2000` the slot dispersion is measured
/// from thousands of names, so a residual past sixteen slot sigmas is a stale print, an
/// unadjusted split or a halt reopen rather than a move; the clip bounds what one such bar can
/// contribute to the patch embedding without touching any plausible value.
pub const CROSS_SECTION_Z_CLIP: f32 = 16.0;
const SECS_PER_DAY: i64 = 86_400;
const CLOCK_SLOTS: usize = (SECS_PER_DAY * 1000 / RESOLUTION_MS) as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    TimeOfDay,
    DayOfWeek,
    SessionGap,
    Volume,
    Market,
    Spy,
    Dispersion,
    CrossSectionZ,
}

impl Feature {
    /// Channel order inside every auxiliary row.
    pub const ALL: [Self; 8] = [
        Self::TimeOfDay,
        Self::DayOfWeek,
        Self::SessionGap,
        Self::Volume,
        Self::Market,
        Self::Spy,
        Self::Dispersion,
        Self::CrossSectionZ,
    ];
    pub const WIDTH: usize = 2;

    pub fn name(self) -> &'static str {
        match self {
            Self::TimeOfDay => "time-of-day",
            Self::DayOfWeek => "day-of-week",
            Self::SessionGap => "session-gap",
            Self::Volume => "volume",
            Self::Market => "market",
            Self::Spy => "spy",
            Self::Dispersion => "dispersion",
            Self::CrossSectionZ => "cross-section-z",
        }
    }

    /// Calendar and gap channels are known for future bars; volume, market, SPY and both
    /// cross-section channels are history, so they read zero beyond the context.
    pub fn known_future(self, _channel: usize) -> bool {
        matches!(self, Self::TimeOfDay | Self::DayOfWeek | Self::SessionGap)
    }

    /// Market and SPY log-return values enter the token in units of the origin's causal σ,
    /// like prices; their validity flags stay unscaled. Neither cross-section channel is
    /// scaled: `ln σ_slot` is the log of a dimensionless dispersion and the z is already
    /// standardized by that same dispersion, so dividing either by the row's causal σ would
    /// compose two normalizations and make a corpus-wide fact depend on the origin.
    pub fn sigma_scaled(self, channel: usize) -> bool {
        matches!(self, Self::Market | Self::Spy) && channel == 0
    }

    fn schema(self) -> &'static str {
        match self {
            Self::TimeOfDay => "sin,cos(2pi*America/New_York-minute-of-day/1440)",
            Self::DayOfWeek => "sin,cos(2pi*America/New_York-weekday/7)",
            Self::SessionGap => {
                "flag(previous-valid-bar-delta>5min),ln(delta-minutes/5)-clipped-at-8"
            }
            Self::Volume => "same-ticker-log-volume-difference,valid-iff-current-and-previous-positive-finite",
            Self::Market => "equal-weighted-corpus-mean-log-close-return-over-the-market-step-ending-at-bar-timestamp,validity",
            Self::Spy => "SPY-5min-log-close-return-at-bar-timestamp,validity",
            Self::Dispersion => "ln(population-stdev-of-the-contributing-log-close-returns-of-the-market-step-ending-at-bar-timestamp)-floored-at-ln(1e-6),validity",
            Self::CrossSectionZ => "(own-5min-log-close-return-minus-market-step)/slot-stdev-clipped-at-16,valid-iff-slot-defines-a-step-and-the-previous-valid-bar-is-one-interval-earlier",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSet {
    pub time_of_day: bool,
    pub day_of_week: bool,
    pub session_gap: bool,
    pub volume: bool,
    pub market: bool,
    pub spy: bool,
    /// Absent from a manifest written before the cross-section channels existed and skipped
    /// while disabled, so a control checkpoint's manifest and the digest authenticating it stay
    /// byte-identical, while `deny_unknown_fields` makes a manifest that ENABLES either channel
    /// unreadable by any build that does not have it. A new-channel checkpoint therefore cannot
    /// be loaded as an old one in either direction.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dispersion: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cross_section_z: bool,
}

impl FeatureSet {
    pub const ALL: Self = Self {
        time_of_day: true,
        day_of_week: true,
        session_gap: true,
        volume: true,
        market: true,
        spy: true,
        dispersion: true,
        cross_section_z: true,
    };
    pub const NONE: Self = Self {
        time_of_day: false,
        day_of_week: false,
        session_gap: false,
        volume: false,
        market: false,
        spy: false,
        dispersion: false,
        cross_section_z: false,
    };

    pub fn enabled(&self, feature: Feature) -> bool {
        match feature {
            Feature::TimeOfDay => self.time_of_day,
            Feature::DayOfWeek => self.day_of_week,
            Feature::SessionGap => self.session_gap,
            Feature::Volume => self.volume,
            Feature::Market => self.market,
            Feature::Spy => self.spy,
            Feature::Dispersion => self.dispersion,
            Feature::CrossSectionZ => self.cross_section_z,
        }
    }

    fn enable(&mut self, feature: Feature) -> &mut bool {
        match feature {
            Feature::TimeOfDay => &mut self.time_of_day,
            Feature::DayOfWeek => &mut self.day_of_week,
            Feature::SessionGap => &mut self.session_gap,
            Feature::Volume => &mut self.volume,
            Feature::Market => &mut self.market,
            Feature::Spy => &mut self.spy,
            Feature::Dispersion => &mut self.dispersion,
            Feature::CrossSectionZ => &mut self.cross_section_z,
        }
    }

    pub fn features(&self) -> impl Iterator<Item = Feature> + '_ {
        Feature::ALL
            .into_iter()
            .filter(move |feature| self.enabled(*feature))
    }

    /// Whether any enabled channel reads the per-slot cross-section moments.
    pub fn cross_section(&self) -> bool {
        self.dispersion || self.cross_section_z
    }

    pub fn channels(&self) -> usize {
        self.features().count() * Feature::WIDTH
    }

    /// Per-channel mask in row order; `role(feature, channel)` for each channel of every feature.
    pub fn channel_mask(&self, role: fn(Feature, usize) -> bool) -> Vec<bool> {
        self.features()
            .flat_map(|feature| (0..Feature::WIDTH).map(move |channel| role(feature, channel)))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.channels() == 0
    }

    pub fn schema(&self) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut parts: Vec<String> = self
            .features()
            .map(|feature| format!("{}:{}", feature.name(), feature.schema()))
            .collect();
        parts.push(
            "return-channels-scaled-by-causal-origin-sigma;history-channels-zero-beyond-context;calendar-gap-known-future-v3"
                .into(),
        );
        parts.join(";")
    }
}

impl fmt::Display for FeatureSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        for (index, feature) in self.features().enumerate() {
            if index > 0 {
                f.write_str(",")?;
            }
            f.write_str(feature.name())?;
        }
        Ok(())
    }
}

impl FromStr for FeatureSet {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.trim() {
            "none" => return Ok(Self::NONE),
            "all" => return Ok(Self::ALL),
            _ => {}
        }
        let mut set = Self::NONE;
        for name in text.split(',').map(str::trim) {
            let feature = Feature::ALL
                .into_iter()
                .find(|feature| feature.name() == name)
                .ok_or_else(|| {
                    format!(
                        "unknown feature {name:?}; expected a comma-separated subset of {} or none",
                        Feature::ALL.map(Feature::name).join(",")
                    )
                })?;
            let slot = set.enable(feature);
            if *slot {
                return Err(format!("duplicate feature {name:?}"));
            }
            *slot = true;
        }
        Ok(set)
    }
}

/// One value per five-minute slot of the shared timestamp grid; NaN marks a missing slot.
pub struct ExogenousSeries {
    first_ts: i64,
    values: Vec<f32>,
}

impl ExogenousSeries {
    pub fn at(&self, ts: i64) -> f32 {
        let offset = ts - self.first_ts;
        if offset < 0 || offset % RESOLUTION_MS != 0 {
            return f32::NAN;
        }
        self.values
            .get((offset / RESOLUTION_MS) as usize)
            .copied()
            .unwrap_or(f32::NAN)
    }
}

pub struct Exogenous {
    pub market: Option<ExogenousSeries>,
    pub spy: Option<ExogenousSeries>,
    /// Present exactly when [`FeatureSet::cross_section`] holds.
    pub cross_section: Option<CrossSection>,
    pub market_cum: MarketPath,
}

/// One slot's cross-section, in the transforms the channels write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotMoments {
    /// The equal-weighted market step ending at the slot.
    pub market: f32,
    pub log_sigma: f32,
    pub inv_sigma: f32,
}

/// Per-slot cross-sectional moments of the market step over the shared grid: the mean, which
/// is the [`Feature::Market`] channel, and the dispersion around it.
///
/// Stored already transformed - `ln σ` and `1/σ` - so a bar costs two loads and a multiply
/// rather than a divide and a transcendental; 12 bytes per slot. `log_sigma` and `inv_sigma`
/// are NaN together wherever the slot defines no step, so definedness is one check. `market`
/// duplicates [`MarketSteps::series`] deliberately: the z channel needs the slot mean whether
/// or not the market channel itself is enabled.
pub struct CrossSection {
    first_ts: i64,
    market: Vec<f32>,
    log_sigma: Vec<f32>,
    inv_sigma: Vec<f32>,
}

impl CrossSection {
    /// The moments at exactly `ts`; `None` off the grid, or at a slot defining no step.
    pub fn at(&self, ts: i64) -> Option<SlotMoments> {
        let offset = ts - self.first_ts;
        if offset < 0 || offset % RESOLUTION_MS != 0 {
            return None;
        }
        let slot = usize::try_from(offset / RESOLUTION_MS).ok()?;
        let log_sigma = *self.log_sigma.get(slot)?;
        log_sigma.is_finite().then(|| SlotMoments {
            market: self.market[slot],
            log_sigma,
            inv_sigma: self.inv_sigma[slot],
        })
    }
}

/// Cumulative equal-weighted market log return over the shared grid; see [`MarketSteps`]. Slots
/// that define no step carry the previous value forward. Every value uses bars at or before its
/// slot only.
pub struct MarketPath {
    first_ts: i64,
    values: Vec<f64>,
}

impl MarketPath {
    /// Value at the latest slot at or before `ts`; zero before the grid starts.
    pub fn at(&self, ts: i64) -> f64 {
        let offset = ts - self.first_ts;
        if offset < 0 {
            return 0.0;
        }
        let slot = (offset / RESOLUTION_MS) as usize;
        self.values[slot.min(self.values.len() - 1)]
    }
}

/// Equal-weighted market steps on the shared grid. A slot defines a step when at least
/// `min_cross_section` sources hold a valid bar there; the step is the mean of `ln(close /
/// close at the previous defining slot)` over the sources holding valid bars at both slots. Bars
/// at sparsely populated slots (extended hours, half-day fragments, stale prints) therefore
/// neither define steps nor inject multi-step returns, and the next defining step spans them.
pub struct MarketSteps {
    first_ts: i64,
    min_cross_section: u32,
    /// Sources holding a valid bar at each slot.
    population: Vec<u32>,
    /// Sum, sum of squares and count of contributing returns at each defining slot. The second
    /// moment rides the same streaming pass as the first: at a mean step of order 1e-5 against
    /// a dispersion of order 1e-3, `E[x²] - mean²` cancels about five decimal digits of the
    /// sixteen f64 carries, so the streaming form needs no second pass and no shifted origin.
    sums: Vec<f64>,
    squares: Vec<f64>,
    counts: Vec<u32>,
}

/// Corpus-level description of the market steps for the corpus report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarketSummary {
    pub steps: usize,
    /// Grid slots the corpus spans, so `steps / slots` is the share of the grid that defines a
    /// market step and, identically, a dispersion.
    pub slots: usize,
    pub min_contributors: u32,
    pub median_contributors: u32,
    pub step_std: f64,
    pub max_abs_step: f64,
    pub max_abs_step_ts: i64,
    /// Quartiles of `sigma_slot` over defining slots, zero when there are none: a populated,
    /// non-degenerate dispersion channel reads a median of order 1e-3 with a strictly narrower
    /// interquartile band.
    pub dispersion_p25: f64,
    pub median_dispersion: f64,
    pub dispersion_p75: f64,
}

impl MarketSteps {
    fn defined(&self, slot: usize) -> bool {
        self.population[slot] >= self.min_cross_section
    }

    /// Rebuild from persisted per-slot vectors. The three lengths are the grid, so a caller
    /// handing over vectors of different lengths is refused rather than indexed out of bounds
    /// at the first bar lookup.
    pub fn from_parts(
        first_ts: i64,
        min_cross_section: u32,
        population: Vec<u32>,
        sums: Vec<f64>,
        squares: Vec<f64>,
        counts: Vec<u32>,
    ) -> Option<Self> {
        (population.len() == sums.len()
            && sums.len() == squares.len()
            && squares.len() == counts.len())
            .then_some(Self {
                first_ts,
                min_cross_section,
                population,
                sums,
                squares,
                counts,
            })
    }

    /// The persistable state: grid origin, cross-section floor, and the four per-slot vectors
    /// the two full corpus passes produce.
    pub fn parts(&self) -> (i64, u32, &[u32], &[f64], &[f64], &[u32]) {
        (
            self.first_ts,
            self.min_cross_section,
            &self.population,
            &self.sums,
            &self.squares,
            &self.counts,
        )
    }

    /// Mean contributing return at a defining slot; `None` where no step is defined.
    pub fn step(&self, slot: usize) -> Option<f64> {
        (self.defined(slot) && self.counts[slot] > 0)
            .then(|| self.sums[slot] / f64::from(self.counts[slot]))
    }

    pub fn slots(&self) -> usize {
        self.population.len()
    }

    pub fn contributors(&self, slot: usize) -> u32 {
        self.counts[slot]
    }

    pub fn path(&self) -> MarketPath {
        let mut cumulative = 0.0;
        MarketPath {
            first_ts: self.first_ts,
            values: (0..self.slots())
                .map(|slot| {
                    cumulative += self.step(slot).unwrap_or(0.0);
                    cumulative
                })
                .collect(),
        }
    }

    /// The step ending at each defining slot as the market return channel; NaN elsewhere.
    pub fn series(&self) -> ExogenousSeries {
        ExogenousSeries {
            first_ts: self.first_ts,
            values: (0..self.slots())
                .map(|slot| self.step(slot).map_or(f32::NAN, |step| step as f32))
                .collect(),
        }
    }

    /// Population standard deviation of the contributing returns at a defining slot, floored at
    /// [`DISPERSION_FLOOR`]. `None` on exactly the slots [`Self::step`] returns `None` for, so
    /// the dispersion channel and the market channel never disagree about what a bar is.
    pub fn dispersion(&self, slot: usize) -> Option<f64> {
        let mean = self.step(slot)?;
        let count = f64::from(self.counts[slot]);
        Some(
            (self.squares[slot] / count - mean * mean)
                .max(0.0)
                .sqrt()
                .max(DISPERSION_FLOOR),
        )
    }

    /// The slot mean and dispersion in the form the two cross-section channels read.
    pub fn cross_section(&self) -> CrossSection {
        let mut moments = CrossSection {
            first_ts: self.first_ts,
            market: vec![f32::NAN; self.slots()],
            log_sigma: vec![f32::NAN; self.slots()],
            inv_sigma: vec![f32::NAN; self.slots()],
        };
        for slot in 0..self.slots() {
            let (Some(step), Some(sigma)) = (self.step(slot), self.dispersion(slot)) else {
                continue;
            };
            moments.market[slot] = step as f32;
            moments.log_sigma[slot] = sigma.ln() as f32;
            moments.inv_sigma[slot] = sigma.recip() as f32;
        }
        moments
    }

    pub fn summary(&self) -> MarketSummary {
        let mut contributors = Vec::new();
        let mut dispersions = Vec::new();
        let mut largest = (0.0f64, self.first_ts);
        let (mut sum, mut sum_squares) = (0.0f64, 0.0f64);
        for slot in 0..self.slots() {
            let Some(step) = self.step(slot) else {
                continue;
            };
            contributors.push(self.counts[slot]);
            dispersions.push(self.dispersion(slot).unwrap_or(DISPERSION_FLOOR));
            sum += step;
            sum_squares += step * step;
            if step.abs() > largest.0.abs() {
                largest = (step, self.first_ts + slot as i64 * RESOLUTION_MS);
            }
        }
        contributors.sort_unstable();
        dispersions.sort_unstable_by(f64::total_cmp);
        let steps = contributors.len();
        let mean = sum / steps.max(1) as f64;
        // Nearest-rank quantiles, the convention the contributor median above already uses.
        let quantile = |numerator: usize| {
            dispersions
                .get(steps * numerator / 4)
                .or(dispersions.last())
                .copied()
                .unwrap_or(0.0)
        };
        MarketSummary {
            steps,
            slots: self.slots(),
            min_contributors: contributors.first().copied().unwrap_or(0),
            median_contributors: contributors.get(steps / 2).copied().unwrap_or(0),
            step_std: (sum_squares / steps.max(1) as f64 - mean * mean).max(0.0).sqrt(),
            max_abs_step: largest.0,
            max_abs_step_ts: largest.1,
            dispersion_p25: quantile(1),
            median_dispersion: quantile(2),
            dispersion_p75: quantile(3),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Grid {
    first_ts: i64,
    slots: usize,
}

impl Grid {
    pub fn new(first_ts: i64, last_ts: i64) -> Result<Self> {
        ensure!(
            first_ts.rem_euclid(RESOLUTION_MS) == 0 && last_ts >= first_ts,
            "off-grid exogenous timestamp span"
        );
        let slots = usize::try_from((last_ts - first_ts) / RESOLUTION_MS + 1)?;
        ensure!(slots <= 20_000_000, "exogenous grid spans more than 190 years");
        Ok(Self { first_ts, slots })
    }

    pub fn first_ts(&self) -> i64 {
        self.first_ts
    }

    pub fn slots(&self) -> usize {
        self.slots
    }
}

/// `ln(close_t / close_{t-5min})` of one source at every valid bar whose previous valid bar is
/// exactly one interval earlier, NaN elsewhere: returns across gaps are excluded so that one slot
/// never mixes five-minute moves with overnight jumps.
pub fn single_series(bars: &[PackedBar], invalid: &[usize], grid: Grid) -> ExogenousSeries {
    let mut values = vec![f32::NAN; grid.slots];
    let mut previous: Option<(usize, f32)> = None;
    for (slot, close) in valid_slots(bars, invalid, grid) {
        if let Some((previous_slot, previous_close)) = previous {
            if slot == previous_slot + 1 {
                values[slot] = (f64::from(close) / f64::from(previous_close)).ln() as f32;
            }
        }
        previous = Some((slot, close));
    }
    ExogenousSeries {
        first_ts: grid.first_ts,
        values,
    }
}

fn accumulate<F>(
    sources: &[(&[PackedBar], &[usize])],
    grid: Grid,
    per_source: F,
) -> (Vec<f64>, Vec<f64>, Vec<u32>)
where
    F: Fn(&[PackedBar], &[usize], &mut [f64], &mut [f64], &mut [u32]) + Sync,
{
    let chunk = sources.len().div_ceil(16).max(1);
    sources
        .par_chunks(chunk)
        .map(|chunk| {
            let mut sums = vec![0.0f64; grid.slots];
            let mut squares = vec![0.0f64; grid.slots];
            let mut counts = vec![0u32; grid.slots];
            for (bars, invalid) in chunk {
                per_source(bars, invalid, &mut sums, &mut squares, &mut counts);
            }
            (sums, squares, counts)
        })
        .reduce(
            || {
                (
                    vec![0.0f64; grid.slots],
                    vec![0.0f64; grid.slots],
                    vec![0u32; grid.slots],
                )
            },
            |(mut sums, mut squares, mut counts), (other_sums, other_squares, other_counts)| {
                for (sum, other) in sums.iter_mut().zip(other_sums) {
                    *sum += other;
                }
                for (square, other) in squares.iter_mut().zip(other_squares) {
                    *square += other;
                }
                for (count, other) in counts.iter_mut().zip(other_counts) {
                    *count += other;
                }
                (sums, squares, counts)
            },
        )
}

/// Valid bars of one source as `(slot, close)` pairs inside the grid.
fn valid_slots<'a>(
    bars: &'a [PackedBar],
    invalid: &'a [usize],
    grid: Grid,
) -> impl Iterator<Item = (usize, f32)> + 'a {
    let mut invalid = invalid.iter().copied().peekable();
    bars.iter().enumerate().filter_map(move |(index, bar)| {
        if invalid.peek() == Some(&index) {
            invalid.next();
            return None;
        }
        let offset = bar.ts() - grid.first_ts;
        let slot = (offset / RESOLUTION_MS) as usize;
        (offset >= 0 && slot < grid.slots).then_some((slot, bar.close))
    })
}

fn population(sources: &[(&[PackedBar], &[usize])], grid: Grid) -> Vec<u32> {
    sources
        .par_iter()
        .fold(
            || vec![0u32; grid.slots],
            |mut population, (bars, invalid)| {
                for (slot, _) in valid_slots(bars, invalid, grid) {
                    population[slot] += 1;
                }
                population
            },
        )
        .reduce(
            || vec![0u32; grid.slots],
            |mut left, right| {
                for (a, b) in left.iter_mut().zip(right) {
                    *a += b;
                }
                left
            },
        )
}

pub fn market_steps(
    sources: &[(&[PackedBar], &[usize])],
    grid: Grid,
    min_cross_section: usize,
) -> MarketSteps {
    let min_cross_section = u32::try_from(min_cross_section.max(1)).unwrap_or(u32::MAX);
    let population = population(sources, grid);
    let mut rank = Vec::with_capacity(grid.slots);
    let mut seen = 0u32;
    for &count in &population {
        seen += u32::from(count >= min_cross_section);
        rank.push(seen);
    }
    let (sums, squares, counts) =
        accumulate(sources, grid, |bars, invalid, sums, squares, counts| {
            let mut previous: Option<(u32, f32)> = None;
            for (slot, close) in valid_slots(bars, invalid, grid) {
                if population[slot] < min_cross_section {
                    continue;
                }
                if let Some((previous_rank, previous_close)) = previous {
                    if rank[slot] - previous_rank == 1 {
                        let step = (f64::from(close) / f64::from(previous_close)).ln();
                        sums[slot] += step;
                        squares[slot] += step * step;
                        counts[slot] += 1;
                    }
                }
                previous = Some((rank[slot], close));
            }
        });
    MarketSteps {
        first_ts: grid.first_ts,
        min_cross_section,
        population,
        sums,
        squares,
        counts,
    }
}

fn angle(fraction: f64) -> [f32; 2] {
    let radians = std::f64::consts::TAU * fraction;
    [radians.sin() as f32, radians.cos() as f32]
}

static CLOCK: LazyLock<[[f32; 2]; CLOCK_SLOTS]> =
    LazyLock::new(|| std::array::from_fn(|slot| angle(slot as f64 / CLOCK_SLOTS as f64)));
static WEEKDAY: LazyLock<[[f32; 2]; 7]> =
    LazyLock::new(|| std::array::from_fn(|day| angle(day as f64 / 7.0)));

fn innovation(series: Option<&ExogenousSeries>, ts: i64) -> [f32; 2] {
    let value = series
        .expect("enabled exogenous feature has a precomputed series")
        .at(ts);
    if value.is_finite() {
        [value, 1.0]
    } else {
        [0.0, 0.0]
    }
}

/// Writes one bar's auxiliary channels in [`Feature::ALL`] order, carrying the previous valid
/// bar forward for the gap and volume innovations. Future bars keep only known channels.
pub struct AuxiliaryCursor<'a> {
    set: &'a FeatureSet,
    exogenous: &'a Exogenous,
    previous_ts: Option<i64>,
    previous_volume: f32,
    previous_close: f32,
}

impl<'a> AuxiliaryCursor<'a> {
    pub fn new(set: &'a FeatureSet, exogenous: &'a Exogenous, previous: Option<&PackedBar>) -> Self {
        Self {
            set,
            exogenous,
            previous_ts: previous.map(PackedBar::ts),
            previous_volume: previous.map_or(0.0, |bar| bar.volume),
            previous_close: previous.map_or(0.0, |bar| bar.close),
        }
    }

    pub fn write(&mut self, bar: &PackedBar, out: &mut [f32], future: bool) {
        let set = self.set;
        let ts = bar.ts();
        let mut offset = 0;
        if set.time_of_day || set.day_of_week {
            let utc = ts.div_euclid(1000);
            let local = utc + i64::from(et_offset_secs(utc));
            if set.time_of_day {
                let minute = local.rem_euclid(SECS_PER_DAY) / 60;
                out[offset..offset + 2].copy_from_slice(&CLOCK[(minute / 5) as usize]);
                offset += 2;
            }
            if set.day_of_week {
                let weekday = (local.div_euclid(SECS_PER_DAY) + 3).rem_euclid(7);
                out[offset..offset + 2].copy_from_slice(&WEEKDAY[weekday as usize]);
                offset += 2;
            }
        }
        if set.session_gap {
            let delta = self.previous_ts.map_or(0, |previous| ts - previous);
            let gap = if delta > RESOLUTION_MS {
                [
                    1.0,
                    ((delta as f64 / RESOLUTION_MS as f64).ln() as f32).min(GAP_LOG_CLIP),
                ]
            } else {
                [0.0, 0.0]
            };
            out[offset..offset + 2].copy_from_slice(&gap);
            offset += 2;
        }
        if set.volume {
            let current = bar.volume;
            let previous = self.previous_volume;
            let volume = if !future
                && current.is_finite()
                && current > 0.0
                && previous.is_finite()
                && previous > 0.0
            {
                [
                    (f64::from(current).ln() - f64::from(previous).ln()) as f32,
                    1.0,
                ]
            } else {
                [0.0, 0.0]
            };
            out[offset..offset + 2].copy_from_slice(&volume);
            offset += 2;
        }
        for (enabled, series) in [
            (set.market, self.exogenous.market.as_ref()),
            (set.spy, self.exogenous.spy.as_ref()),
        ] {
            if enabled {
                let value = if future {
                    [0.0, 0.0]
                } else {
                    innovation(series, ts)
                };
                out[offset..offset + 2].copy_from_slice(&value);
                offset += 2;
            }
        }
        if set.cross_section() {
            let moments = (!future)
                .then(|| {
                    self.exogenous
                        .cross_section
                        .as_ref()
                        .expect("enabled cross-section feature has precomputed slot moments")
                        .at(ts)
                })
                .flatten();
            if set.dispersion {
                let value = moments.map_or([0.0, 0.0], |slot| [slot.log_sigma, 1.0]);
                out[offset..offset + 2].copy_from_slice(&value);
                offset += 2;
            }
            if set.cross_section_z {
                // The own step follows `single_series`: only a previous valid bar exactly one
                // interval earlier defines a return, so no z is ever taken across a gap.
                let own = (self
                    .previous_ts
                    .is_some_and(|previous| ts - previous == RESOLUTION_MS)
                    && self.previous_close > 0.0
                    && bar.close > 0.0)
                    .then(|| (f64::from(bar.close) / f64::from(self.previous_close)).ln() as f32);
                let value = match (own, moments) {
                    (Some(own), Some(slot)) => [
                        ((own - slot.market) * slot.inv_sigma)
                            .clamp(-CROSS_SECTION_Z_CLIP, CROSS_SECTION_Z_CLIP),
                        1.0,
                    ],
                    _ => [0.0, 0.0],
                };
                out[offset..offset + 2].copy_from_slice(&value);
                offset += 2;
            }
        }
        debug_assert_eq!(offset, out.len());
        self.previous_ts = Some(ts);
        self.previous_volume = bar.volume;
        self.previous_close = bar.close;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_sets_round_trip_through_their_cli_spelling() {
        assert_eq!("all".parse::<FeatureSet>().unwrap(), FeatureSet::ALL);
        assert_eq!("none".parse::<FeatureSet>().unwrap(), FeatureSet::NONE);
        let set: FeatureSet = "volume, market".parse().unwrap();
        assert_eq!(
            set,
            FeatureSet {
                volume: true,
                market: true,
                ..FeatureSet::NONE
            }
        );
        assert_eq!(set.to_string(), "volume,market");
        assert_eq!(set.channels(), 4);
        assert_eq!(FeatureSet::ALL.to_string().parse::<FeatureSet>().unwrap(), FeatureSet::ALL);
        assert_eq!(FeatureSet::ALL.channels(), 16);
        // The step-matched control's input, by name: this exact string is what reproduces the
        // pre-cross-section feature set, so it is pinned rather than described.
        let control: FeatureSet = "time-of-day,day-of-week,session-gap,volume,market,spy"
            .parse()
            .unwrap();
        assert_eq!(control.channels(), 12);
        assert!(!control.cross_section());
        assert_eq!(
            control.to_string(),
            "time-of-day,day-of-week,session-gap,volume,market,spy"
        );
        let both: FeatureSet = "dispersion,cross-section-z".parse().unwrap();
        assert_eq!(both.channels(), 4);
        assert!(both.cross_section());
        assert!("dispersion".parse::<FeatureSet>().unwrap().cross_section());
        assert!("volume,volume".parse::<FeatureSet>().is_err());
        assert!("vol".parse::<FeatureSet>().is_err());
        assert!(FeatureSet::NONE.schema().is_empty());
    }

    #[test]
    fn a_checkpoint_written_before_the_cross_section_channels_still_authenticates() {
        // `runner::Manifest::read` re-serializes what it parsed and compares the SHA-256 it
        // carries, so a control checkpoint stays loadable only if the two absent channels stay
        // absent on the way out; and `deny_unknown_fields` is what stops a build without them
        // from reading a manifest that ENABLES them as if it were an old one.
        let old = r#"{"time_of_day":true,"day_of_week":true,"session_gap":true,"volume":true,"market":true,"spy":true}"#;
        let control: FeatureSet = serde_json::from_str(old).unwrap();
        assert_eq!(control.channels(), 12);
        assert!(!control.cross_section());
        assert_eq!(serde_json::to_string(&control).unwrap(), old);
        let enabled = serde_json::to_string(&FeatureSet::ALL).unwrap();
        assert!(
            enabled.contains(r#""dispersion":true"#) && enabled.contains(r#""cross_section_z":true"#),
            "{enabled}"
        );
        assert_eq!(
            serde_json::from_str::<FeatureSet>(&enabled).unwrap(),
            FeatureSet::ALL
        );
    }

    #[test]
    fn calendar_gap_and_exogenous_channels_follow_the_bar_clock() {
        // 2024-03-11 is the first Monday after the US DST switch: 09:30 ET == 13:30 UTC.
        let monday_open = 1_710_163_800_000;
        let friday_close = monday_open - (2 * SECS_PER_DAY + 16 * 3600 + 30 * 60) * 1000;
        let bars = [
            PackedBar {
                ts_ms: friday_close,
                close: 100.0,
                volume: 50.0,
                ..PackedBar::default()
            },
            PackedBar {
                ts_ms: monday_open,
                close: 101.0,
                volume: 200.0,
                ..PackedBar::default()
            },
            PackedBar {
                ts_ms: monday_open + RESOLUTION_MS,
                close: 102.0,
                volume: 100.0,
                ..PackedBar::default()
            },
        ];
        let grid = Grid::new(friday_close, monday_open + RESOLUTION_MS).unwrap();
        let spy = single_series(&bars, &[], grid);
        assert!(spy.at(friday_close).is_nan());
        assert!(spy.at(monday_open).is_nan());
        assert!((spy.at(monday_open + RESOLUTION_MS) - (102.0f32 / 101.0).ln()).abs() < 1e-6);
        assert!(spy.at(monday_open + 1).is_nan());
        let steps = market_steps(&[(&bars, &[]), (&bars[1..], &[])], grid, 1);
        let exogenous = Exogenous {
            market: Some(steps.series()),
            spy: Some(spy),
            cross_section: None,
            market_cum: steps.path(),
        };
        // The pre-cross-section input, by name: every index this fixture asserts on is one of
        // the twelve channels that string produces.
        let set: FeatureSet = "time-of-day,day-of-week,session-gap,volume,market,spy"
            .parse()
            .unwrap();
        let mut cursor = AuxiliaryCursor::new(&set, &exogenous, Some(&bars[0]));
        let mut row = [f32::NAN; 12];
        cursor.write(&bars[1], &mut row, false);
        let open_angle = std::f64::consts::TAU * 570.0 / 1440.0;
        assert!((f64::from(row[0]) - open_angle.sin()).abs() < 1e-6);
        assert!((f64::from(row[1]) - open_angle.cos()).abs() < 1e-6);
        assert!(row[2].abs() < 1e-6 && (row[3] - 1.0).abs() < 1e-6);
        assert_eq!(row[4], 1.0);
        let weekend_minutes = (monday_open - friday_close) as f64 / 60_000.0;
        assert!((f64::from(row[5]) - (weekend_minutes / 5.0).ln()).abs() < 1e-6);
        assert!((row[6] - 4.0f32.ln()).abs() < 1e-6);
        assert_eq!(row[7], 1.0);
        // The market channel is the step spanning the weekend; SPY's five-minute channel is not.
        assert!((row[8] - (101.0f32 / 100.0).ln()).abs() < 1e-6);
        assert_eq!(&row[9..12], &[1.0, 0.0, 0.0]);
        cursor.write(&bars[2], &mut row, false);
        assert_eq!(&row[4..6], &[0.0, 0.0]);
        assert!((row[8] - (102.0f32 / 101.0).ln()).abs() < 1e-6);
        assert_eq!(&row[9..12], &[1.0, row[8], 1.0]);
        let mut future = [f32::NAN; 12];
        let mut cursor = AuxiliaryCursor::new(&set, &exogenous, Some(&bars[1]));
        cursor.write(&bars[2], &mut future, true);
        assert_eq!(&future[..6], &row[..6]);
        assert_eq!(&future[6..12], &[0.0; 6]);
        assert_eq!(
            set.channel_mask(Feature::known_future),
            [true, true, true, true, true, true, false, false, false, false, false, false]
        );
        assert_eq!(
            set.channel_mask(Feature::sigma_scaled)[6..],
            [false, false, true, false, true, false]
        );
        let mut clipped = [0.0; 2];
        let gap = FeatureSet {
            session_gap: true,
            ..FeatureSet::NONE
        };
        let mut cursor = AuxiliaryCursor::new(&gap, &exogenous, Some(&bars[0]));
        cursor.write(
            &PackedBar {
                ts_ms: friday_close + 400 * SECS_PER_DAY * 1000,
                ..PackedBar::default()
            },
            &mut clipped,
            false,
        );
        assert_eq!(clipped, [1.0, GAP_LOG_CLIP]);
    }

    #[test]
    fn market_steps_span_gaps_and_only_adjacent_defining_slots() {
        let step = RESOLUTION_MS;
        let bar = |slot: i64, close: f32| PackedBar {
            ts_ms: slot * step,
            close,
            ..PackedBar::default()
        };
        // Slots 2..=4 are a gap for every source; B skips slot 6.
        let a = [bar(0, 100.0), bar(1, 101.0), bar(5, 102.0), bar(6, 103.0), bar(7, 104.0)];
        let b = [bar(0, 50.0), bar(1, 51.0), bar(5, 52.0), bar(7, 53.0)];
        let grid = Grid::new(0, 7 * step).unwrap();
        let path = market_steps(&[(&a, &[]), (&b, &[])], grid, 1).path();
        let ln = |num: f32, den: f32| (f64::from(num) / f64::from(den)).ln();
        let s1 = (ln(101.0, 100.0) + ln(51.0, 50.0)) / 2.0;
        let s5 = (ln(102.0, 101.0) + ln(52.0, 51.0)) / 2.0;
        let s6 = ln(103.0, 102.0);
        let s7 = ln(104.0, 103.0);
        assert_eq!(path.at(-step), 0.0);
        assert_eq!(path.at(0), 0.0);
        assert!((path.at(step) - s1).abs() < 1e-12);
        assert_eq!(path.at(3 * step), path.at(step));
        assert!((path.at(5 * step) - (s1 + s5)).abs() < 1e-12);
        assert!((path.at(6 * step) - (s1 + s5 + s6)).abs() < 1e-12);
        assert!((path.at(7 * step) - (s1 + s5 + s6 + s7)).abs() < 1e-12);
        assert_eq!(path.at(9 * step), path.at(7 * step));
        // With A's slot-6 bar invalid, slot 6 is unpopulated and B's 5->7 pair becomes adjacent.
        let path = market_steps(&[(&a, &[3]), (&b, &[])], grid, 1).path();
        let s7 = (ln(104.0, 102.0) + ln(53.0, 52.0)) / 2.0;
        assert_eq!(path.at(6 * step), path.at(5 * step));
        assert!((path.at(7 * step) - (s1 + s5 + s7)).abs() < 1e-12);
    }

    #[test]
    fn sparse_slots_define_no_step_and_the_next_step_spans_them() {
        let step = RESOLUTION_MS;
        let bar = |slot: i64, close: f32| PackedBar {
            ts_ms: slot * step,
            close,
            ..PackedBar::default()
        };
        // Slot 1 is an after-hours print of A alone with a wild move; slot 3 holds A and B only;
        // C misses slot 4 and its 2->5 return must not enter any step.
        let a = [bar(0, 100.0), bar(1, 150.0), bar(2, 101.0), bar(3, 90.0), bar(4, 102.0), bar(5, 103.0)];
        let b = [bar(0, 50.0), bar(2, 51.0), bar(3, 60.0), bar(4, 52.0), bar(5, 53.0)];
        let c = [bar(0, 20.0), bar(2, 21.0), bar(5, 40.0)];
        let d = [bar(0, 10.0), bar(2, 11.0), bar(4, 12.0), bar(5, 13.0)];
        let grid = Grid::new(0, 5 * step).unwrap();
        let sources = [
            (&a[..], &[][..]),
            (&b[..], &[][..]),
            (&c[..], &[][..]),
            (&d[..], &[][..]),
        ];
        let steps = market_steps(&sources, grid, 3);
        let ln = |num: f32, den: f32| (f64::from(num) / f64::from(den)).ln();
        let s2 = (ln(101.0, 100.0) + ln(51.0, 50.0) + ln(21.0, 20.0) + ln(11.0, 10.0)) / 4.0;
        let s4 = (ln(102.0, 101.0) + ln(52.0, 51.0) + ln(12.0, 11.0)) / 3.0;
        let s5 = (ln(103.0, 102.0) + ln(53.0, 52.0) + ln(13.0, 12.0)) / 3.0;
        assert_eq!([steps.step(1), steps.step(3)], [None, None]);
        assert_eq!(
            [steps.contributors(2), steps.contributors(4), steps.contributors(5)],
            [4, 3, 3]
        );
        let path = steps.path();
        assert_eq!(path.at(step), 0.0);
        assert!((path.at(2 * step) - s2).abs() < 1e-12);
        assert_eq!(path.at(3 * step), path.at(2 * step));
        assert!((path.at(4 * step) - (s2 + s4)).abs() < 1e-12);
        assert!((path.at(5 * step) - (s2 + s4 + s5)).abs() < 1e-12);
        let series = steps.series();
        assert!(series.at(step).is_nan() && series.at(3 * step).is_nan());
        assert!((f64::from(series.at(4 * step)) - s4).abs() < 1e-6);
        let summary = steps.summary();
        assert_eq!(
            (summary.steps, summary.min_contributors, summary.median_contributors),
            (3, 3, 3)
        );
        assert_eq!(summary.max_abs_step_ts, 2 * step);
        // What the corpus report's dispersion clause reads: the share is over the whole grid,
        // and the quartiles are the defining slots' own dispersions in order.
        assert_eq!(summary.slots, grid.slots());
        let mut sigmas = [2usize, 4, 5].map(|slot| steps.dispersion(slot).unwrap());
        sigmas.sort_by(f64::total_cmp);
        assert_eq!(
            [
                summary.dispersion_p25,
                summary.median_dispersion,
                summary.dispersion_p75
            ],
            sigmas
        );
        // A lower threshold lets slot 3 define a step and inject A's and B's swings.
        assert_eq!(market_steps(&sources, grid, 2).summary().steps, 4);
    }

    /// `(slot, close)` on the shared grid, the fixture style the market tests above use.
    fn bar(slot: i64, close: f32) -> PackedBar {
        PackedBar {
            ts_ms: slot * RESOLUTION_MS,
            close,
            ..PackedBar::default()
        }
    }

    /// One bar's auxiliary row under `set`, with every exogenous series this grid can define.
    fn auxiliary_row(
        set: &FeatureSet,
        steps: &MarketSteps,
        spy: ExogenousSeries,
        previous: &PackedBar,
        current: &PackedBar,
        future: bool,
    ) -> Vec<f32> {
        let exogenous = Exogenous {
            market: Some(steps.series()),
            spy: Some(spy),
            cross_section: Some(steps.cross_section()),
            market_cum: steps.path(),
        };
        let mut out = vec![f32::NAN; set.channels()];
        AuxiliaryCursor::new(set, &exogenous, Some(previous)).write(current, &mut out, future);
        out
    }

    #[test]
    fn a_slot_below_the_cross_section_floor_defines_neither_a_step_nor_a_dispersion() {
        let a = [bar(0, 100.0), bar(1, 101.0)];
        let b = [bar(0, 50.0), bar(1, 50.25)];
        let c = [bar(0, 20.0)];
        let grid = Grid::new(0, RESOLUTION_MS).unwrap();
        let steps = market_steps(
            &[(&a[..], &[][..]), (&b[..], &[][..]), (&c[..], &[][..])],
            grid,
            3,
        );
        // Slot 1 holds two of the three sources, so it defines no market step - and therefore
        // no dispersion, on the very same population rule.
        assert_eq!(steps.step(1), None);
        assert_eq!(steps.dispersion(1), None);
        assert!(steps.cross_section().at(RESOLUTION_MS).is_none());
        // A's own step at slot 1 IS defined - its previous valid bar is one interval earlier -
        // so what zeroes both channels here is the slot rule alone, not a missing own return.
        let set: FeatureSet = "market,dispersion,cross-section-z".parse().unwrap();
        let written = auxiliary_row(&set, &steps, single_series(&a, &[], grid), &a[0], &a[1], false);
        assert_eq!(written, vec![0.0; 6]);
    }

    #[test]
    fn the_slot_dispersion_is_the_cross_sectional_deviation_and_the_z_is_its_residual() {
        let grid = Grid::new(0, RESOLUTION_MS).unwrap();
        let a = [bar(0, 100.0), bar(1, 101.0)];
        let b = [bar(0, 50.0), bar(1, 50.25)];
        let c = [bar(0, 20.0), bar(1, 20.6)];
        let ln = |num: f32, den: f32| (f64::from(num) / f64::from(den)).ln();
        let (ra, rb, rc) = (ln(101.0, 100.0), ln(50.25, 50.0), ln(20.6, 20.0));
        let set: FeatureSet = "dispersion,cross-section-z".parse().unwrap();
        // TWO tickers: the mean is the midpoint and the population sigma is the half-spread, so
        // every z is exactly +-1. An analytic value, not a transcription of the accumulator.
        let pair = market_steps(&[(&a[..], &[][..]), (&b[..], &[][..])], grid, 2);
        let sigma = (ra - rb).abs() / 2.0;
        assert!((pair.dispersion(1).unwrap() - sigma).abs() < 1e-15);
        let written = auxiliary_row(&set, &pair, single_series(&a, &[], grid), &a[0], &a[1], false);
        assert!((f64::from(written[0]) - sigma.ln()).abs() < 1e-6);
        assert!((written[2] - 1.0).abs() < 1e-5);
        assert_eq!([written[1], written[3]], [1.0, 1.0]);
        let written = auxiliary_row(&set, &pair, single_series(&b, &[], grid), &b[0], &b[1], false);
        assert!((written[2] + 1.0).abs() < 1e-5);
        // THREE tickers: the population standard deviation as the mean of squared deviations,
        // a different arithmetic path from the streaming `E[x²] - mean²` the accumulator runs.
        let trio = market_steps(
            &[(&a[..], &[][..]), (&b[..], &[][..]), (&c[..], &[][..])],
            grid,
            3,
        );
        let mean = (ra + rb + rc) / 3.0;
        let sigma =
            (((ra - mean).powi(2) + (rb - mean).powi(2) + (rc - mean).powi(2)) / 3.0).sqrt();
        assert!((trio.step(1).unwrap() - mean).abs() < 1e-15);
        assert!((trio.dispersion(1).unwrap() - sigma).abs() < 1e-15);
        let written = auxiliary_row(&set, &trio, single_series(&c, &[], grid), &c[0], &c[1], false);
        assert!((f64::from(written[0]) - sigma.ln()).abs() < 1e-6);
        assert!((f64::from(written[2]) - (rc - mean) / sigma).abs() < 1e-5);
        assert_eq!([written[1], written[3]], [1.0, 1.0]);
    }

    #[test]
    fn a_previous_valid_bar_more_than_one_interval_back_contributes_no_own_step() {
        let grid = Grid::new(0, 2 * RESOLUTION_MS).unwrap();
        let a = [bar(0, 100.0), bar(1, 101.0), bar(2, 102.0)];
        let b = [bar(0, 50.0), bar(1, 50.25), bar(2, 50.5)];
        let c = [bar(0, 20.0), bar(1, 20.6), bar(2, 20.9)];
        // D misses slot 1 entirely, so its slot-2 return would span two intervals.
        let d = [bar(0, 10.0), bar(2, 11.0)];
        let steps = market_steps(
            &[
                (&a[..], &[][..]),
                (&b[..], &[][..]),
                (&c[..], &[][..]),
                (&d[..], &[][..]),
            ],
            grid,
            3,
        );
        assert!(steps.dispersion(2).is_some());
        // The slot defines a dispersion, so that channel is live; the z is not, because D has
        // no five-minute own return to standardize - exactly `single_series`'s rule.
        let set: FeatureSet = "dispersion,cross-section-z".parse().unwrap();
        let written = auxiliary_row(&set, &steps, single_series(&d, &[], grid), &d[0], &d[1], false);
        assert_eq!(written[1], 1.0);
        assert_eq!([written[2], written[3]], [0.0, 0.0]);
    }

    #[test]
    fn appending_the_cross_section_channels_leaves_every_earlier_channel_identical() {
        let grid = Grid::new(0, 2 * RESOLUTION_MS).unwrap();
        let volume = |slot: i64, close: f32, volume: f32| PackedBar {
            volume,
            ..bar(slot, close)
        };
        let a = [
            volume(0, 100.0, 500.0),
            volume(1, 101.0, 700.0),
            volume(2, 100.5, 300.0),
        ];
        let b = [bar(0, 50.0), bar(1, 50.25), bar(2, 50.5)];
        let c = [bar(0, 20.0), bar(1, 20.6), bar(2, 20.9)];
        let steps = market_steps(
            &[(&a[..], &[][..]), (&b[..], &[][..]), (&c[..], &[][..])],
            grid,
            3,
        );
        let control: FeatureSet = "time-of-day,day-of-week,session-gap,volume,market,spy"
            .parse()
            .unwrap();
        assert_eq!((control.channels(), FeatureSet::ALL.channels()), (12, 16));
        for future in [false, true] {
            for (previous, current) in [(&a[0], &a[1]), (&a[1], &a[2])] {
                let old = auxiliary_row(
                    &control,
                    &steps,
                    single_series(&a, &[], grid),
                    previous,
                    current,
                    future,
                );
                let all = auxiliary_row(
                    &FeatureSet::ALL,
                    &steps,
                    single_series(&a, &[], grid),
                    previous,
                    current,
                    future,
                );
                assert_eq!(old.as_slice(), &all[..12]);
                // Beyond the context both appended channels are history and read nothing.
                if future {
                    assert_eq!(all[12..], [0.0; 4]);
                }
            }
        }
        // And on a live bar the appended pair is populated, so the equality above is not the
        // trivial one that would also hold if the channels never wrote anything.
        let all = auxiliary_row(
            &FeatureSet::ALL,
            &steps,
            single_series(&a, &[], grid),
            &a[0],
            &a[1],
            false,
        );
        assert_eq!([all[13], all[15]], [1.0, 1.0]);
        assert!(all[12] < 0.0 && all[14] != 0.0);
        assert_eq!(FeatureSet::ALL.channel_mask(Feature::known_future)[12..], [false; 4]);
        assert_eq!(FeatureSet::ALL.channel_mask(Feature::sigma_scaled)[12..], [false; 4]);
    }
}
