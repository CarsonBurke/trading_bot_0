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
}

impl Feature {
    /// Channel order inside every auxiliary row.
    pub const ALL: [Self; 6] = [
        Self::TimeOfDay,
        Self::DayOfWeek,
        Self::SessionGap,
        Self::Volume,
        Self::Market,
        Self::Spy,
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
        }
    }

    /// Calendar and gap channels are known for future bars; volume, market, and SPY are history.
    pub fn known_future(self, _channel: usize) -> bool {
        matches!(self, Self::TimeOfDay | Self::DayOfWeek | Self::SessionGap)
    }

    /// Market and SPY log-return values enter the token in units of the origin's causal σ,
    /// like prices; their validity flags stay unscaled.
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
}

impl FeatureSet {
    pub const ALL: Self = Self {
        time_of_day: true,
        day_of_week: true,
        session_gap: true,
        volume: true,
        market: true,
        spy: true,
    };
    pub const NONE: Self = Self {
        time_of_day: false,
        day_of_week: false,
        session_gap: false,
        volume: false,
        market: false,
        spy: false,
    };

    pub fn enabled(&self, feature: Feature) -> bool {
        match feature {
            Feature::TimeOfDay => self.time_of_day,
            Feature::DayOfWeek => self.day_of_week,
            Feature::SessionGap => self.session_gap,
            Feature::Volume => self.volume,
            Feature::Market => self.market,
            Feature::Spy => self.spy,
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
        }
    }

    pub fn features(&self) -> impl Iterator<Item = Feature> + '_ {
        Feature::ALL
            .into_iter()
            .filter(move |feature| self.enabled(*feature))
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
    pub market_cum: MarketPath,
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
    /// Sum and count of contributing returns at each defining slot.
    sums: Vec<f64>,
    counts: Vec<u32>,
}

/// Corpus-level description of the market steps for the corpus report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarketSummary {
    pub steps: usize,
    pub min_contributors: u32,
    pub median_contributors: u32,
    pub step_std: f64,
    pub max_abs_step: f64,
    pub max_abs_step_ts: i64,
}

impl MarketSteps {
    fn defined(&self, slot: usize) -> bool {
        self.population[slot] >= self.min_cross_section
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

    pub fn summary(&self) -> MarketSummary {
        let mut contributors = Vec::new();
        let mut largest = (0.0f64, self.first_ts);
        let (mut sum, mut sum_squares) = (0.0f64, 0.0f64);
        for slot in 0..self.slots() {
            let Some(step) = self.step(slot) else {
                continue;
            };
            contributors.push(self.counts[slot]);
            sum += step;
            sum_squares += step * step;
            if step.abs() > largest.0.abs() {
                largest = (step, self.first_ts + slot as i64 * RESOLUTION_MS);
            }
        }
        contributors.sort_unstable();
        let steps = contributors.len();
        let mean = sum / steps.max(1) as f64;
        MarketSummary {
            steps,
            min_contributors: contributors.first().copied().unwrap_or(0),
            median_contributors: contributors.get(steps / 2).copied().unwrap_or(0),
            step_std: (sum_squares / steps.max(1) as f64 - mean * mean).max(0.0).sqrt(),
            max_abs_step: largest.0,
            max_abs_step_ts: largest.1,
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

fn accumulate<F>(sources: &[(&[PackedBar], &[usize])], grid: Grid, per_source: F) -> (Vec<f64>, Vec<u32>)
where
    F: Fn(&[PackedBar], &[usize], &mut [f64], &mut [u32]) + Sync,
{
    let chunk = sources.len().div_ceil(16).max(1);
    sources
        .par_chunks(chunk)
        .map(|chunk| {
            let mut sums = vec![0.0f64; grid.slots];
            let mut counts = vec![0u32; grid.slots];
            for (bars, invalid) in chunk {
                per_source(bars, invalid, &mut sums, &mut counts);
            }
            (sums, counts)
        })
        .reduce(
            || (vec![0.0f64; grid.slots], vec![0u32; grid.slots]),
            |(mut sums, mut counts), (other_sums, other_counts)| {
                for (sum, other) in sums.iter_mut().zip(other_sums) {
                    *sum += other;
                }
                for (count, other) in counts.iter_mut().zip(other_counts) {
                    *count += other;
                }
                (sums, counts)
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
    let (sums, counts) = accumulate(sources, grid, |bars, invalid, sums, counts| {
        let mut previous: Option<(u32, f32)> = None;
        for (slot, close) in valid_slots(bars, invalid, grid) {
            if population[slot] < min_cross_section {
                continue;
            }
            if let Some((previous_rank, previous_close)) = previous {
                if rank[slot] - previous_rank == 1 {
                    sums[slot] += (f64::from(close) / f64::from(previous_close)).ln();
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
}

impl<'a> AuxiliaryCursor<'a> {
    pub fn new(set: &'a FeatureSet, exogenous: &'a Exogenous, previous: Option<&PackedBar>) -> Self {
        Self {
            set,
            exogenous,
            previous_ts: previous.map(PackedBar::ts),
            previous_volume: previous.map_or(0.0, |bar| bar.volume),
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
        debug_assert_eq!(offset, out.len());
        self.previous_ts = Some(ts);
        self.previous_volume = bar.volume;
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
        assert!("volume,volume".parse::<FeatureSet>().is_err());
        assert!("vol".parse::<FeatureSet>().is_err());
        assert!(FeatureSet::NONE.schema().is_empty());
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
            market_cum: steps.path(),
        };
        let set = FeatureSet::ALL;
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
        // A lower threshold lets slot 3 define a step and inject A's and B's swings.
        assert_eq!(market_steps(&sources, grid, 2).summary().steps, 4);
    }
}
