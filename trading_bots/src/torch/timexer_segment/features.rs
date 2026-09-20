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
/// Smallest cross-sectional spread a LEVEL channel accepts as a measurement, as a fraction of
/// `1 + |mean|`. Returns are centred on zero and `E[x²] - mean²` is well conditioned on them;
/// `ln(volume)` sits near 12 and `ln((high - low)/close)` near -4, and there the same
/// difference cancels its leading digits. On a slot whose contributors all print the SAME tape
/// the streaming sigma lands at about `|mean|·sqrt(n)·2⁻⁵²` - measured at 8e-8 on three
/// contributors of `ln(2/100.5)`, and near 1e-6·|mean| at universe scale - and the stored f32
/// mean carries its own 6e-8·|mean| of rounding, so dividing by that residue turns pure noise
/// into an O(1) z. This floor sits three orders above both. It cannot reject a real slot: a
/// universe of thousands whose log volumes agree to 0.1% or whose log ranges agree to 0.5%
/// does not exist, while their true sigmas are order one.
pub const LEVEL_SPREAD_FLOOR: f64 = 1e-3;
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
    CrossSectionRank,
    RelativeVolume,
    RangeZ,
}

impl Feature {
    /// Channel order inside every auxiliary row.
    pub const ALL: [Self; 11] = [
        Self::TimeOfDay,
        Self::DayOfWeek,
        Self::SessionGap,
        Self::Volume,
        Self::Market,
        Self::Spy,
        Self::Dispersion,
        Self::CrossSectionZ,
        Self::CrossSectionRank,
        Self::RelativeVolume,
        Self::RangeZ,
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
            Self::CrossSectionRank => "cross-section-rank",
            Self::RelativeVolume => "relative-volume",
            Self::RangeZ => "range-z",
        }
    }

    /// Calendar and gap channels are known for future bars; volume, market, SPY and every
    /// cross-section channel is history, so they read zero beyond the context.
    pub fn known_future(self, _channel: usize) -> bool {
        matches!(self, Self::TimeOfDay | Self::DayOfWeek | Self::SessionGap)
    }

    /// Market and SPY log-return values enter the token in units of the origin's causal σ,
    /// like prices; their validity flags stay unscaled. NO cross-section channel is scaled:
    /// `ln σ_slot` is the log of a dimensionless dispersion, the three z channels are already
    /// standardized by their own slot dispersion, and the rank channel is a normal score, so
    /// dividing any of them by the row's causal σ would compose two normalizations and make a
    /// corpus-wide fact depend on the origin.
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
            Self::CrossSectionRank => "inverse-normal-cdf((midrank-of-the-own-5min-log-close-return-in-the-slot-contributing-returns-with-own-inserted-minus-0.5)/(contributors+1)),valid-iff-slot-defines-a-step-and-the-previous-valid-bar-is-one-interval-earlier",
            Self::RelativeVolume => "(ln(own-volume)-slot-mean-ln-volume)/slot-stdev-clipped-at-16,valid-iff-slot-defines-a-step-and-own-volume-positive-finite-and-the-slot-holds-two-or-more-positive-volumes-with-nonzero-spread",
            Self::RangeZ => "(ln((own-high-minus-own-low)/own-close)-slot-mean-log-range)/slot-stdev-clipped-at-16,valid-iff-slot-defines-a-step-and-own-high-exceeds-own-low-and-the-slot-holds-two-or-more-positive-ranges-with-nonzero-spread",
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
    /// byte-identical, while `deny_unknown_fields` makes a manifest that ENABLES any of them
    /// unreadable by any build that does not have it. A new-channel checkpoint therefore cannot
    /// be loaded as an old one in either direction.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dispersion: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cross_section_z: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cross_section_rank: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub relative_volume: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub range_z: bool,
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
        cross_section_rank: true,
        relative_volume: true,
        range_z: true,
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
        cross_section_rank: false,
        relative_volume: false,
        range_z: false,
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
            Feature::CrossSectionRank => self.cross_section_rank,
            Feature::RelativeVolume => self.relative_volume,
            Feature::RangeZ => self.range_z,
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
            Feature::CrossSectionRank => &mut self.cross_section_rank,
            Feature::RelativeVolume => &mut self.relative_volume,
            Feature::RangeZ => &mut self.range_z,
        }
    }

    pub fn features(&self) -> impl Iterator<Item = Feature> + '_ {
        Feature::ALL
            .into_iter()
            .filter(move |feature| self.enabled(*feature))
    }

    /// Whether any enabled channel reads the per-slot cross-section moments.
    pub fn cross_section(&self) -> bool {
        self.dispersion
            || self.cross_section_z
            || self.cross_section_rank
            || self.relative_volume
            || self.range_z
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
    /// Mean and reciprocal dispersion of `ln(volume)` over the same contributing set; NaN
    /// together where the slot holds fewer than two positive volumes or no spread at all.
    pub log_volume_mean: f32,
    pub inv_volume_sigma: f32,
    /// The same pair for `ln((high - low) / close)`.
    pub log_range_mean: f32,
    pub inv_range_sigma: f32,
    /// `1 / (2·(contributors + 1))`, the rank channel's plotting-position denominator. Always
    /// finite at a defining slot: the contributor count there is at least one.
    pub inv_ranked: f32,
}

/// Per-slot cross-sectional moments over the shared grid: the market step's mean, which is the
/// [`Feature::Market`] channel, the dispersion around it, and the two level cross-sections the
/// volume and range channels standardize against.
///
/// Stored already transformed - `ln σ` and `1/σ` - so a bar costs two loads and a multiply
/// rather than a divide and a transcendental; 32 bytes per slot. `log_sigma` and `inv_sigma`
/// are NaN together wherever the slot defines no step, so definedness is one check, and each
/// level pair is NaN together wherever ITS own cross-section is degenerate while the step is
/// not. `market` duplicates [`MarketSteps::series`] deliberately: the z channel needs the slot
/// mean whether or not the market channel itself is enabled.
pub struct CrossSection {
    first_ts: i64,
    market: Vec<f32>,
    log_sigma: Vec<f32>,
    inv_sigma: Vec<f32>,
    log_volume_mean: Vec<f32>,
    inv_volume_sigma: Vec<f32>,
    log_range_mean: Vec<f32>,
    inv_range_sigma: Vec<f32>,
    inv_ranked: Vec<f32>,
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
            log_volume_mean: self.log_volume_mean[slot],
            inv_volume_sigma: self.inv_volume_sigma[slot],
            log_range_mean: self.log_range_mean[slot],
            inv_range_sigma: self.inv_range_sigma[slot],
            inv_ranked: self.inv_ranked[slot],
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

/// Streaming first and second moment of one per-bar quantity over a slot's contributing set.
///
/// The second moment rides the same pass as the first. `E[x²] - mean²` is safe for every
/// quantity accumulated here: a step of order 1e-5 against a dispersion of order 1e-3 cancels
/// about five decimal digits of the sixteen an f64 carries, and `ln(volume)` and
/// `ln((high - low) / close)` cancel under two, so none of them needs a second pass or a
/// shifted origin.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SlotMoment {
    pub sums: Vec<f64>,
    pub squares: Vec<f64>,
    pub counts: Vec<u32>,
}

impl SlotMoment {
    fn zeros(slots: usize) -> Self {
        Self {
            sums: vec![0.0; slots],
            squares: vec![0.0; slots],
            counts: vec![0; slots],
        }
    }

    fn push(&mut self, slot: usize, value: f64) {
        self.sums[slot] += value;
        self.squares[slot] += value * value;
        self.counts[slot] += 1;
    }

    fn absorb(&mut self, other: &Self) {
        for (sum, other) in self.sums.iter_mut().zip(&other.sums) {
            *sum += *other;
        }
        for (square, other) in self.squares.iter_mut().zip(&other.squares) {
            *square += *other;
        }
        for (count, other) in self.counts.iter_mut().zip(&other.counts) {
            *count += *other;
        }
    }

    fn aligned(&self, slots: usize) -> bool {
        self.sums.len() == slots && self.squares.len() == slots && self.counts.len() == slots
    }

    /// Population mean and standard deviation at `slot`, `None` below two observations or
    /// below [`LEVEL_SPREAD_FLOOR`]. A z-score against a degenerate cross-section is not a
    /// small number, it is no measurement at all, so the channel reading this reports invalid
    /// instead of dividing by the accumulator's own cancellation residue.
    fn moments(&self, slot: usize) -> Option<(f64, f64)> {
        if self.counts[slot] < 2 {
            return None;
        }
        let count = f64::from(self.counts[slot]);
        let mean = self.sums[slot] / count;
        let sigma = (self.squares[slot] / count - mean * mean).max(0.0).sqrt();
        (sigma > LEVEL_SPREAD_FLOOR * (1.0 + mean.abs())).then_some((mean, sigma))
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
    /// The contributing log close returns: the market step and the dispersion around it.
    returns: SlotMoment,
    /// `ln(volume)` and `ln((high - low) / close)` of the SAME contributing bars, each dropping
    /// the bars that print no tape or no range. One definition of "the slot's cross-section"
    /// serves all five channels; a bar excluded from the step is excluded from the levels too.
    log_volume: SlotMoment,
    log_range: SlotMoment,
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

    /// Rebuild from persisted per-slot vectors. Every vector's length is the grid, so a caller
    /// handing over vectors of different lengths is refused rather than indexed out of bounds
    /// at the first bar lookup.
    pub fn from_parts(
        first_ts: i64,
        min_cross_section: u32,
        population: Vec<u32>,
        returns: SlotMoment,
        log_volume: SlotMoment,
        log_range: SlotMoment,
    ) -> Option<Self> {
        let slots = population.len();
        (returns.aligned(slots) && log_volume.aligned(slots) && log_range.aligned(slots)).then_some(
            Self {
                first_ts,
                min_cross_section,
                population,
                returns,
                log_volume,
                log_range,
            },
        )
    }

    /// The persistable state: grid origin, cross-section floor, the per-slot population and the
    /// three per-slot moment triples the two full corpus passes produce.
    pub fn parts(&self) -> (i64, u32, &[u32], &SlotMoment, &SlotMoment, &SlotMoment) {
        (
            self.first_ts,
            self.min_cross_section,
            &self.population,
            &self.returns,
            &self.log_volume,
            &self.log_range,
        )
    }

    /// Mean contributing return at a defining slot; `None` where no step is defined.
    pub fn step(&self, slot: usize) -> Option<f64> {
        (self.defined(slot) && self.returns.counts[slot] > 0)
            .then(|| self.returns.sums[slot] / f64::from(self.returns.counts[slot]))
    }

    pub fn slots(&self) -> usize {
        self.population.len()
    }

    pub fn contributors(&self, slot: usize) -> u32 {
        self.returns.counts[slot]
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
        let count = f64::from(self.returns.counts[slot]);
        Some(
            (self.returns.squares[slot] / count - mean * mean)
                .max(0.0)
                .sqrt()
                .max(DISPERSION_FLOOR),
        )
    }

    /// Every per-slot statistic the five cross-section channels read, pre-transformed.
    pub fn cross_section(&self) -> CrossSection {
        let slots = self.slots();
        let nan = || vec![f32::NAN; slots];
        let mut moments = CrossSection {
            first_ts: self.first_ts,
            market: nan(),
            log_sigma: nan(),
            inv_sigma: nan(),
            log_volume_mean: nan(),
            inv_volume_sigma: nan(),
            log_range_mean: nan(),
            inv_range_sigma: nan(),
            inv_ranked: nan(),
        };
        for slot in 0..slots {
            let (Some(step), Some(sigma)) = (self.step(slot), self.dispersion(slot)) else {
                continue;
            };
            moments.market[slot] = step as f32;
            moments.log_sigma[slot] = sigma.ln() as f32;
            moments.inv_sigma[slot] = sigma.recip() as f32;
            // The rank channel inserts the own value into the contributing multiset, so its
            // denominator is one larger than the contributor count and can never reach 0 or 1.
            moments.inv_ranked[slot] =
                (0.5 / (f64::from(self.returns.counts[slot]) + 1.0)) as f32;
            if let Some((mean, sigma)) = self.log_volume.moments(slot) {
                moments.log_volume_mean[slot] = mean as f32;
                moments.inv_volume_sigma[slot] = sigma.recip() as f32;
            }
            if let Some((mean, sigma)) = self.log_range.moments(slot) {
                moments.log_range_mean[slot] = mean as f32;
                moments.inv_range_sigma[slot] = sigma.recip() as f32;
            }
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
            contributors.push(self.returns.counts[slot]);
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
    for (_, slot, bar) in valid_bars(bars, invalid, grid) {
        if let Some((previous_slot, previous_close)) = previous {
            if slot == previous_slot + 1 {
                values[slot] = (f64::from(bar.close) / f64::from(previous_close)).ln() as f32;
            }
        }
        previous = Some((slot, bar.close));
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
) -> (SlotMoment, SlotMoment, SlotMoment)
where
    F: Fn(&[PackedBar], &[usize], &mut SlotMoment, &mut SlotMoment, &mut SlotMoment) + Sync,
{
    let chunk = sources.len().div_ceil(16).max(1);
    let zeros = || {
        (
            SlotMoment::zeros(grid.slots),
            SlotMoment::zeros(grid.slots),
            SlotMoment::zeros(grid.slots),
        )
    };
    sources
        .par_chunks(chunk)
        .map(|chunk| {
            let (mut returns, mut volumes, mut ranges) = zeros();
            for (bars, invalid) in chunk {
                per_source(bars, invalid, &mut returns, &mut volumes, &mut ranges);
            }
            (returns, volumes, ranges)
        })
        .reduce(zeros, |(mut returns, mut volumes, mut ranges), other| {
            returns.absorb(&other.0);
            volumes.absorb(&other.1);
            ranges.absorb(&other.2);
            (returns, volumes, ranges)
        })
}

/// Valid bars of one source inside the grid, as `(valid-bar ordinal, slot, bar)`.
///
/// The ordinal counts EVERY valid bar of the source, including one the grid does not cover, so
/// it indexes the same axis as `DataContract::valid_bars` and every per-bar array derived from
/// it. Bars outside the grid are counted and not yielded.
fn valid_bars<'a>(
    bars: &'a [PackedBar],
    invalid: &'a [usize],
    grid: Grid,
) -> impl Iterator<Item = (usize, usize, &'a PackedBar)> + 'a {
    let mut invalid = invalid.iter().copied().peekable();
    let mut ordinal = 0usize;
    bars.iter().enumerate().filter_map(move |(index, bar)| {
        if invalid.peek() == Some(&index) {
            invalid.next();
            return None;
        }
        let at = ordinal;
        ordinal += 1;
        let offset = bar.ts() - grid.first_ts;
        let slot = (offset / RESOLUTION_MS) as usize;
        (offset >= 0 && slot < grid.slots).then_some((at, slot, bar))
    })
}

/// `ln(volume)` of a bar. `None` for an absent, zero or non-finite tape: a bar reporting no
/// volume carries no volume information, and a zero would enter the log as `-inf`.
fn log_volume(bar: &PackedBar) -> Option<f64> {
    (bar.volume.is_finite() && bar.volume > 0.0).then(|| f64::from(bar.volume).ln())
}

/// `ln((high - low) / close)` of a bar. `None` when the bar prints NO range - `high == low` is
/// a single print, a halt or a limit lock, and would enter the log as `-inf` - or when the
/// close is not a usable denominator.
fn log_range(bar: &PackedBar) -> Option<f64> {
    let (high, low, close) = (
        f64::from(bar.high),
        f64::from(bar.low),
        f64::from(bar.close),
    );
    (high.is_finite() && low.is_finite() && close.is_finite() && high > low && close > 0.0)
        .then(|| ((high - low) / close).ln())
}

/// Which slots define a market step, and each slot's running defining-slot ordinal.
struct Defining<'a> {
    population: &'a [u32],
    min_cross_section: u32,
    rank: Vec<u32>,
}

impl<'a> Defining<'a> {
    fn new(population: &'a [u32], min_cross_section: u32) -> Self {
        let mut rank = Vec::with_capacity(population.len());
        let mut seen = 0u32;
        for &count in population {
            seen += u32::from(count >= min_cross_section);
            rank.push(seen);
        }
        Self {
            population,
            min_cross_section,
            rank,
        }
    }

    fn holds(&self, slot: usize) -> bool {
        self.population[slot] >= self.min_cross_section
    }
}

/// One source's CONTRIBUTING bars in order: a valid bar at a defining slot whose previous valid
/// bar at a defining slot sits at the immediately preceding defining slot. `visit` receives the
/// slot, the bar, and the close the bar returns against.
///
/// ONE definition, shared by the moment pass and the rank pass, so the two can never disagree
/// about which bars are in the slot's cross-section.
fn contributions(
    bars: &[PackedBar],
    invalid: &[usize],
    grid: Grid,
    defining: &Defining<'_>,
    mut visit: impl FnMut(usize, &PackedBar, f32),
) {
    let mut previous: Option<(u32, f32)> = None;
    for (_, slot, bar) in valid_bars(bars, invalid, grid) {
        if !defining.holds(slot) {
            continue;
        }
        if let Some((previous_rank, previous_close)) = previous {
            if defining.rank[slot] - previous_rank == 1 {
                visit(slot, bar, previous_close);
            }
        }
        previous = Some((defining.rank[slot], bar.close));
    }
}

fn population(sources: &[(&[PackedBar], &[usize])], grid: Grid) -> Vec<u32> {
    sources
        .par_iter()
        .fold(
            || vec![0u32; grid.slots],
            |mut population, (bars, invalid)| {
                for (_, slot, _) in valid_bars(bars, invalid, grid) {
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
    let defining = Defining::new(&population, min_cross_section);
    let (returns, volumes, ranges) =
        accumulate(sources, grid, |bars, invalid, returns, volumes, ranges| {
            contributions(bars, invalid, grid, &defining, |slot, bar, previous_close| {
                returns.push(
                    slot,
                    (f64::from(bar.close) / f64::from(previous_close)).ln(),
                );
                if let Some(volume) = log_volume(bar) {
                    volumes.push(slot, volume);
                }
                if let Some(range) = log_range(bar) {
                    ranges.push(slot, range);
                }
            });
        });
    MarketSteps {
        first_ts: grid.first_ts,
        min_cross_section,
        population,
        returns,
        log_volume: volumes,
        log_range: ranges,
    }
}

/// Inverse standard normal CDF on the open unit interval, Acklam's rational approximation.
///
/// Relative error below 1.15e-9 across the whole interval - two orders inside f32's own
/// resolution - and a fixed sequence of arithmetic, so it is bit-reproducible. Its only caller
/// passes `(r - 0.5)/N` with `1 <= r <= N` and `N >= 2`, so `p` is bounded away from both ends
/// by at least `0.25/N` and the endpoint guards below are unreachable in production.
pub fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const BREAK: f64 = 0.02425;
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if !(p > 0.0) {
        return f64::NEG_INFINITY;
    }
    if !(p < 1.0) {
        return f64::INFINITY;
    }
    if p < BREAK {
        return tail((-2.0 * p.ln()).sqrt());
    }
    if p > 1.0 - BREAK {
        return -tail((-2.0 * (1.0 - p).ln()).sqrt());
    }
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

/// A raw cursor into the shared contribution buffer.
///
/// Every `(chunk, slot)` pair owns the half-open index range starting at
/// `offsets[slot] + base[chunk][slot]` and running for that chunk's own contribution count at
/// that slot. Those ranges partition each slot's segment exactly - the bases are the exclusive
/// prefix sum of the per-chunk counts, and the segment length is their total - so no two
/// threads ever address the same element.
struct Scatter(*mut f32);

unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

/// One `u16` per valid bar of every source: `2·r`, where `r` is the mid-rank of that bar's own
/// five-minute log close return inside the slot's contributing returns WITH THE OWN VALUE
/// INSERTED, and `0` wherever [`Feature::CrossSectionRank`] is undefined. Doubling keeps the
/// half-integer a tie produces exact in an integer, and a defined rank has `2·r >= 2`, so `0`
/// is an unreachable sentinel and no second array is needed to carry validity.
///
/// Own is INSERTED rather than located, so the ranked multiset has `N = n + 1` members and the
/// plotting position `(r - 0.5)/N` is strictly inside `(0, 1)` on every slot, including a
/// single-contributor one. It has to be inserted: the own five-minute return follows
/// `single_series`'s adjacency rule while a contribution follows the DEFINING-slot rule, so a
/// bar can carry a well-defined own return that is not one of the `n` contributing values, and
/// ranking it inside a set it need not belong to would otherwise reach `p = 1` exactly.
///
/// This is the one part of the family that needs each slot's whole distribution rather than its
/// moments: a full per-slot sort, and a binary search per bar. It is paid ONCE per (corpus,
/// grid, threshold) and cached, because the same search inside host batch assembly would add
/// about a dozen dependent cache misses to every one of the 1,585,152 bars in a batch.
pub fn cross_section_ranks(
    sources: &[(&[PackedBar], &[usize])],
    grid: Grid,
    steps: &MarketSteps,
) -> Vec<Vec<u16>> {
    assert!(
        sources.len() < usize::from(u16::MAX) / 2,
        "a doubled mid-rank must fit a u16, so the universe is capped at {} sources",
        usize::from(u16::MAX) / 2
    );
    let slots = grid.slots;
    let mut offsets: Vec<usize> = Vec::with_capacity(slots + 1);
    let mut total = 0usize;
    for &count in &steps.returns.counts {
        offsets.push(total);
        total += count as usize;
    }
    offsets.push(total);
    let defining = Defining::new(&steps.population, steps.min_cross_section);
    let chunk = sources.len().div_ceil(16).max(1);
    let chunks: Vec<&[(&[PackedBar], &[usize])]> = sources.chunks(chunk).collect();
    let mut bases: Vec<Vec<u32>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut counts = vec![0u32; slots];
            for (bars, invalid) in chunk.iter() {
                contributions(bars, invalid, grid, &defining, |slot, _, _| {
                    counts[slot] += 1;
                });
            }
            counts
        })
        .collect();
    let mut running = vec![0u32; slots];
    for chunk_bases in bases.iter_mut() {
        for (slot, base) in chunk_bases.iter_mut().enumerate() {
            let count = *base;
            *base = running[slot];
            running[slot] += count;
        }
    }
    drop(running);
    let mut values = vec![0.0f32; total];
    let scatter = Scatter(values.as_mut_ptr());
    chunks
        .par_iter()
        .zip(bases.par_iter_mut())
        .for_each(|(chunk, cursor)| {
            let scatter = &scatter;
            for (bars, invalid) in chunk.iter() {
                contributions(bars, invalid, grid, &defining, |slot, bar, previous_close| {
                    let at = offsets[slot] + cursor[slot] as usize;
                    cursor[slot] += 1;
                    let step = (f64::from(bar.close) / f64::from(previous_close)).ln() as f32;
                    // SAFETY: see [`Scatter`]. `at` is inside slot `slot`'s segment and no
                    // other chunk can produce it.
                    unsafe { scatter.0.add(at).write(step) };
                });
            }
        });
    drop(bases);
    let mut rest = values.as_mut_slice();
    let mut segments: Vec<&mut [f32]> = Vec::new();
    for &count in &steps.returns.counts {
        let (head, tail) = rest.split_at_mut(count as usize);
        if count > 1 {
            segments.push(head);
        }
        rest = tail;
    }
    segments
        .par_iter_mut()
        .for_each(|segment| segment.sort_unstable_by(f32::total_cmp));
    drop(segments);
    let values = values;
    sources
        .par_iter()
        .map(|(bars, invalid)| {
            let mut ranks = vec![0u16; bars.len() - invalid.len()];
            let mut previous: Option<(usize, f32)> = None;
            for (ordinal, slot, bar) in valid_bars(bars, invalid, grid) {
                if let Some((previous_slot, previous_close)) = previous {
                    // Exactly `Feature::CrossSectionZ`'s rule: a defining slot, and a previous
                    // valid bar one interval earlier.
                    if slot == previous_slot + 1
                        && previous_close > 0.0
                        && bar.close > 0.0
                        && steps.step(slot).is_some()
                    {
                        let own = (f64::from(bar.close) / f64::from(previous_close)).ln() as f32;
                        let segment = &values[offsets[slot]..offsets[slot + 1]];
                        let below = segment.partition_point(|value| *value < own);
                        let ties = segment[below..].partition_point(|value| *value <= own);
                        ranks[ordinal] = (2 * below + ties + 2) as u16;
                    }
                }
                previous = Some((slot, bar.close));
            }
            ranks
        })
        .collect()
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
    /// [`cross_section_ranks`] output for this source, starting at the FIRST bar this cursor
    /// will be handed. Empty when the rank channel is off and on the projected-future path,
    /// where the channel reads `[0, 0]` regardless.
    ranks: &'a [u16],
    /// Bars written so far, which is this cursor's index into `ranks`.
    position: usize,
}

impl<'a> AuxiliaryCursor<'a> {
    pub fn new(
        set: &'a FeatureSet,
        exogenous: &'a Exogenous,
        previous: Option<&PackedBar>,
        ranks: &'a [u16],
    ) -> Self {
        Self {
            set,
            exogenous,
            previous_ts: previous.map(PackedBar::ts),
            previous_volume: previous.map_or(0.0, |bar| bar.volume),
            previous_close: previous.map_or(0.0, |bar| bar.close),
            ranks,
            position: 0,
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
            if set.cross_section_rank {
                // The rank was resolved against the whole slot cross-section at corpus load;
                // `0` is its "undefined" sentinel and carries the same rule as the z above.
                let doubled = self.ranks.get(self.position).copied().unwrap_or(0);
                let value = match (doubled, moments) {
                    (0, _) | (_, None) => [0.0, 0.0],
                    (doubled, Some(slot)) => [
                        normal_quantile(f64::from(doubled - 1) * f64::from(slot.inv_ranked))
                            as f32,
                        1.0,
                    ],
                };
                out[offset..offset + 2].copy_from_slice(&value);
                offset += 2;
            }
            if set.relative_volume {
                let value = match (moments, log_volume(bar)) {
                    (Some(slot), Some(volume)) if slot.inv_volume_sigma.is_finite() => [
                        (((volume - f64::from(slot.log_volume_mean))
                            * f64::from(slot.inv_volume_sigma)) as f32)
                            .clamp(-CROSS_SECTION_Z_CLIP, CROSS_SECTION_Z_CLIP),
                        1.0,
                    ],
                    _ => [0.0, 0.0],
                };
                out[offset..offset + 2].copy_from_slice(&value);
                offset += 2;
            }
            if set.range_z {
                let value = match (moments, log_range(bar)) {
                    (Some(slot), Some(range)) if slot.inv_range_sigma.is_finite() => [
                        (((range - f64::from(slot.log_range_mean))
                            * f64::from(slot.inv_range_sigma)) as f32)
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
        self.position += 1;
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
        assert_eq!(FeatureSet::ALL.channels(), 22);
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
        // The second family's input, by name: the exact string an arm passes to add it.
        let family: FeatureSet =
            "time-of-day,day-of-week,session-gap,volume,market,spy,dispersion,cross-section-z,cross-section-rank,relative-volume,range-z"
                .parse()
                .unwrap();
        assert_eq!(family, FeatureSet::ALL);
        for name in ["cross-section-rank", "relative-volume", "range-z"] {
            let one: FeatureSet = name.parse().unwrap();
            assert_eq!((one.channels(), one.cross_section()), (2, true));
            assert_eq!(one.to_string(), name);
        }
        assert!("volume,volume".parse::<FeatureSet>().is_err());
        assert!("vol".parse::<FeatureSet>().is_err());
        assert!(FeatureSet::NONE.schema().is_empty());
    }

    #[test]
    fn a_checkpoint_written_before_the_cross_section_channels_still_authenticates() {
        // `runner::Manifest::read` re-serializes what it parsed and compares the SHA-256 it
        // carries, so a control checkpoint stays loadable only if the absent channels stay
        // absent on the way out; and `deny_unknown_fields` is what stops a build without them
        // from reading a manifest that ENABLES them as if it were an old one.
        let old = r#"{"time_of_day":true,"day_of_week":true,"session_gap":true,"volume":true,"market":true,"spy":true}"#;
        let control: FeatureSet = serde_json::from_str(old).unwrap();
        assert_eq!(control.channels(), 12);
        assert!(!control.cross_section());
        assert_eq!(serde_json::to_string(&control).unwrap(), old);
        let enabled = serde_json::to_string(&FeatureSet::ALL).unwrap();
        for field in [
            "dispersion",
            "cross_section_z",
            "cross_section_rank",
            "relative_volume",
            "range_z",
        ] {
            assert!(enabled.contains(&format!(r#""{field}":true"#)), "{enabled}");
        }
        assert_eq!(
            serde_json::from_str::<FeatureSet>(&enabled).unwrap(),
            FeatureSet::ALL
        );
        // The rejection path, proved rather than described: this is the shape a build predating
        // the second family compiles, and `deny_unknown_fields` is why a manifest enabling one
        // of the three new channels FAILS to parse there instead of silently defaulting it off
        // and authenticating against a SHA that never covered it.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        #[allow(dead_code)]
        struct FirstFamilyFeatureSet {
            time_of_day: bool,
            day_of_week: bool,
            session_gap: bool,
            volume: bool,
            market: bool,
            spy: bool,
            #[serde(default)]
            dispersion: bool,
            #[serde(default)]
            cross_section_z: bool,
        }
        assert!(serde_json::from_str::<FirstFamilyFeatureSet>(old).is_ok());
        for field in ["cross_section_rank", "relative_volume", "range_z"] {
            let manifest = format!(
                r#"{{"time_of_day":true,"day_of_week":true,"session_gap":true,"volume":true,"market":true,"spy":true,"{field}":true}}"#
            );
            assert!(
                serde_json::from_str::<FirstFamilyFeatureSet>(&manifest).is_err(),
                "{field}"
            );
            let read: FeatureSet = serde_json::from_str(&manifest).unwrap();
            assert_eq!(serde_json::to_string(&read).unwrap(), manifest);
        }
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
        let mut cursor = AuxiliaryCursor::new(&set, &exogenous, Some(&bars[0]), &[]);
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
        let mut cursor = AuxiliaryCursor::new(&set, &exogenous, Some(&bars[1]), &[]);
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
        let mut cursor = AuxiliaryCursor::new(&gap, &exogenous, Some(&bars[0]), &[]);
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

    /// A bar that prints a tape and a range: `close` with a symmetric `spread` around it.
    fn rich(slot: i64, close: f32, volume: f32, spread: f32) -> PackedBar {
        PackedBar {
            high: close + spread,
            low: close - spread,
            volume,
            ..bar(slot, close)
        }
    }

    /// One bar's auxiliary row under `set`, with every exogenous series this grid can define.
    /// `ranks` starts at `current`'s valid ordinal, as the cursor's own slice does.
    fn auxiliary_row(
        set: &FeatureSet,
        steps: &MarketSteps,
        spy: ExogenousSeries,
        previous: &PackedBar,
        current: &PackedBar,
        future: bool,
        ranks: &[u16],
    ) -> Vec<f32> {
        let exogenous = Exogenous {
            market: Some(steps.series()),
            spy: Some(spy),
            cross_section: Some(steps.cross_section()),
            market_cum: steps.path(),
        };
        let mut out = vec![f32::NAN; set.channels()];
        AuxiliaryCursor::new(set, &exogenous, Some(previous), ranks).write(current, &mut out, future);
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
        let written =
            auxiliary_row(&set, &steps, single_series(&a, &[], grid), &a[0], &a[1], false, &[]);
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
        let written =
            auxiliary_row(&set, &pair, single_series(&a, &[], grid), &a[0], &a[1], false, &[]);
        assert!((f64::from(written[0]) - sigma.ln()).abs() < 1e-6);
        assert!((written[2] - 1.0).abs() < 1e-5);
        assert_eq!([written[1], written[3]], [1.0, 1.0]);
        let written =
            auxiliary_row(&set, &pair, single_series(&b, &[], grid), &b[0], &b[1], false, &[]);
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
        let written =
            auxiliary_row(&set, &trio, single_series(&c, &[], grid), &c[0], &c[1], false, &[]);
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
        let written =
            auxiliary_row(&set, &steps, single_series(&d, &[], grid), &d[0], &d[1], false, &[]);
        assert_eq!(written[1], 1.0);
        assert_eq!([written[2], written[3]], [0.0, 0.0]);
    }

    #[test]
    fn the_rank_channel_is_the_normal_score_of_the_own_return_inside_its_slot() {
        let grid = Grid::new(0, RESOLUTION_MS).unwrap();
        let a = [bar(0, 100.0), bar(1, 101.0)];
        let b = [bar(0, 50.0), bar(1, 50.25)];
        let c = [bar(0, 20.0), bar(1, 20.6)];
        let sources = [(&a[..], &[][..]), (&b[..], &[][..]), (&c[..], &[][..])];
        let steps = market_steps(&sources, grid, 3);
        let ranks = cross_section_ranks(&sources, grid, &steps);
        // Three contributors with own inserted: N = 4, and the doubled mid-ranks 3, 5, 7 put the
        // three plotting positions at exactly 1/4, 1/2 and 3/4 whatever the returns were.
        assert_eq!([ranks[0][1], ranks[1][1], ranks[2][1]], [5, 3, 7]);
        let set: FeatureSet = "cross-section-rank".parse().unwrap();
        // Phi^-1(3/4) to sixteen digits; Phi^-1(1/2) is exactly zero.
        let quartile = 0.674_489_750_196_081_7_f64;
        for (source, rank, expected) in [
            (&a[..], ranks[0][1], 0.0),
            (&b[..], ranks[1][1], -quartile),
            (&c[..], ranks[2][1], quartile),
        ] {
            let written = auxiliary_row(
                &set,
                &steps,
                single_series(source, &[], grid),
                &source[0],
                &source[1],
                false,
                &[rank],
            );
            assert!((f64::from(written[0]) - expected).abs() < 1e-6, "{written:?}");
            assert_eq!(written[1], 1.0);
        }

        // ONE contributor: inserting own makes N = 2 and the position exactly 1/2, so the score
        // is zero and VALID - the case where a uniform rank has no defined value at all.
        let solo = [(&a[..], &[][..])];
        let single = market_steps(&solo, grid, 1);
        let single_ranks = cross_section_ranks(&solo, grid, &single);
        assert_eq!(single_ranks[0][1], 3);
        let written = auxiliary_row(
            &set,
            &single,
            single_series(&a, &[], grid),
            &a[0],
            &a[1],
            false,
            &single_ranks[0][1..],
        );
        assert_eq!(written, vec![0.0, 1.0]);

        // ZERO cross-sectional variance: every contributor prints ln(1.01), so all three take
        // the same mid-rank 2r = n + 2 = 5 and every position is 1/2. Unlike the z, the rank is
        // still DEFINED on a degenerate slot - that is the point of ranking.
        let flat_a = [bar(0, 100.0), bar(1, 101.0)];
        let flat_b = [bar(0, 200.0), bar(1, 202.0)];
        let flat_c = [bar(0, 400.0), bar(1, 404.0)];
        let flat = [
            (&flat_a[..], &[][..]),
            (&flat_b[..], &[][..]),
            (&flat_c[..], &[][..]),
        ];
        let flat_steps = market_steps(&flat, grid, 3);
        let flat_ranks = cross_section_ranks(&flat, grid, &flat_steps);
        assert_eq!([flat_ranks[0][1], flat_ranks[1][1], flat_ranks[2][1]], [5, 5, 5]);
        let written = auxiliary_row(
            &set,
            &flat_steps,
            single_series(&flat_a, &[], grid),
            &flat_a[0],
            &flat_a[1],
            false,
            &flat_ranks[0][1..],
        );
        assert_eq!(written, vec![0.0, 1.0]);

        // A TIE inside a live slot: two of three print ln(1.01), so both take 2r = 4 and the
        // third 2r = 7, positions 3/8 and 3/4.
        let tied = [
            (&flat_a[..], &[][..]),
            (&flat_b[..], &[][..]),
            (&c[..], &[][..]),
        ];
        let tied_steps = market_steps(&tied, grid, 3);
        let tied_ranks = cross_section_ranks(&tied, grid, &tied_steps);
        assert_eq!([tied_ranks[0][1], tied_ranks[1][1], tied_ranks[2][1]], [4, 4, 7]);
        let written = auxiliary_row(
            &set,
            &tied_steps,
            single_series(&flat_a, &[], grid),
            &flat_a[0],
            &flat_a[1],
            false,
            &tied_ranks[0][1..],
        );
        assert!((f64::from(written[0]) + 0.318_639_363_964_375).abs() < 1e-6, "{written:?}");
        assert_eq!(written[1], 1.0);

        // An INVALID previous bar: D's previous valid bar is two intervals back, so it has no
        // own five-minute return to rank even though the slot itself defines a cross-section.
        // The sentinel is the rank array's own `0`, and the channel reads exactly [0, 0].
        let wide = Grid::new(0, 2 * RESOLUTION_MS).unwrap();
        let wa = [bar(0, 100.0), bar(1, 101.0), bar(2, 102.0)];
        let wb = [bar(0, 50.0), bar(1, 50.25), bar(2, 50.5)];
        let wc = [bar(0, 20.0), bar(1, 20.6), bar(2, 20.9)];
        let wd = [bar(0, 10.0), bar(2, 11.0)];
        let gapped = [
            (&wa[..], &[][..]),
            (&wb[..], &[][..]),
            (&wc[..], &[][..]),
            (&wd[..], &[][..]),
        ];
        let gapped_steps = market_steps(&gapped, wide, 3);
        let gapped_ranks = cross_section_ranks(&gapped, wide, &gapped_steps);
        assert!(gapped_steps.dispersion(2).is_some());
        assert_eq!(gapped_ranks[3][1], 0);
        let written = auxiliary_row(
            &set,
            &gapped_steps,
            single_series(&wd, &[], wide),
            &wd[0],
            &wd[1],
            false,
            &gapped_ranks[3][1..],
        );
        assert_eq!(written, vec![0.0, 0.0]);
    }

    #[test]
    fn the_relative_volume_channel_is_the_slot_log_volume_z_and_a_silent_tape_is_invalid() {
        let grid = Grid::new(0, RESOLUTION_MS).unwrap();
        // Volumes 1, 4 and 16 at slot 1: `ln` is equally spaced, so the population z-scores are
        // exactly -sqrt(3/2), 0 and +sqrt(3/2) whatever the spacing is.
        let a = [rich(0, 100.0, 1.0, 1.0), rich(1, 101.0, 1.0, 1.0)];
        let b = [rich(0, 50.0, 4.0, 1.0), rich(1, 50.25, 4.0, 1.0)];
        let c = [rich(0, 20.0, 16.0, 1.0), rich(1, 20.6, 16.0, 1.0)];
        let set: FeatureSet = "relative-volume".parse().unwrap();
        let expected = 1.5_f64.sqrt();
        let trio = market_steps(
            &[(&a[..], &[][..]), (&b[..], &[][..]), (&c[..], &[][..])],
            grid,
            3,
        );
        let z = |steps: &MarketSteps, source: &[PackedBar]| {
            auxiliary_row(
                &set,
                steps,
                single_series(source, &[], grid),
                &source[0],
                &source[1],
                false,
                &[],
            )
        };
        assert!((f64::from(z(&trio, &a)[0]) + expected).abs() < 1e-5);
        assert!((f64::from(z(&trio, &b)[0])).abs() < 1e-5);
        assert!((f64::from(z(&trio, &c)[0]) - expected).abs() < 1e-5);
        assert_eq!(
            [z(&trio, &a)[1], z(&trio, &b)[1], z(&trio, &c)[1]],
            [1.0; 3]
        );

        // A bar printing NO tape: it is a contributing bar of the slot's step, but it enters
        // neither the log-volume mean nor its dispersion, and its own channel is exactly [0, 0]
        // rather than a z against a distribution it never joined. The other three read the SAME
        // scores they did without it, which is what "excluded" has to mean.
        let silent = [rich(0, 10.0, 4.0, 1.0), rich(1, 10.1, 0.0, 1.0)];
        let four = market_steps(
            &[
                (&a[..], &[][..]),
                (&b[..], &[][..]),
                (&c[..], &[][..]),
                (&silent[..], &[][..]),
            ],
            grid,
            3,
        );
        assert_eq!(four.contributors(1), 4);
        assert_eq!(z(&four, &silent), vec![0.0, 0.0]);
        assert!((f64::from(z(&four, &a)[0]) + expected).abs() < 1e-5);
        assert!((f64::from(z(&four, &c)[0]) - expected).abs() < 1e-5);

        // ZERO cross-sectional spread, and a SINGLE contributing tape: both are non-measurements
        // and read [0, 0], never a zero z that a linear head would take for "average volume".
        // The identical tape is 4, not 1, so `ln` is 1.386 rather than exactly zero and
        // `E[x²] - mean²` cancels to a residue instead of a true zero - the case
        // `LEVEL_SPREAD_FLOOR` exists for, and the one that produced an O(1) z without it.
        let flat_a = [rich(0, 100.0, 4.0, 1.0), rich(1, 101.0, 4.0, 1.0)];
        let flat_c = [rich(0, 20.0, 4.0, 1.0), rich(1, 20.6, 4.0, 1.0)];
        let flat = market_steps(
            &[
                (&flat_a[..], &[][..]),
                (&b[..], &[][..]),
                (&flat_c[..], &[][..]),
            ],
            grid,
            3,
        );
        assert_eq!(z(&flat, &flat_a), vec![0.0, 0.0]);
        let mute_b = [rich(0, 50.0, 4.0, 1.0), rich(1, 50.25, 0.0, 1.0)];
        let mute_c = [rich(0, 20.0, 16.0, 1.0), rich(1, 20.6, 0.0, 1.0)];
        let alone = market_steps(
            &[
                (&a[..], &[][..]),
                (&mute_b[..], &[][..]),
                (&mute_c[..], &[][..]),
            ],
            grid,
            3,
        );
        assert_eq!(z(&alone, &a), vec![0.0, 0.0]);
    }

    #[test]
    fn the_range_channel_is_the_slot_log_range_z_and_a_locked_bar_is_invalid() {
        let grid = Grid::new(0, RESOLUTION_MS).unwrap();
        // Closes all 100 at slot 1 with half-spreads 0.5, 2 and 8: the ranges (high - low)/close
        // are 0.01, 0.04 and 0.16, equally spaced in `ln`, so the z-scores are again
        // -sqrt(3/2), 0 and +sqrt(3/2).
        let a = [rich(0, 99.0, 1.0, 0.5), rich(1, 100.0, 1.0, 0.5)];
        let b = [rich(0, 98.0, 1.0, 2.0), rich(1, 100.0, 1.0, 2.0)];
        let c = [rich(0, 97.0, 1.0, 8.0), rich(1, 100.0, 1.0, 8.0)];
        let set: FeatureSet = "range-z".parse().unwrap();
        let expected = 1.5_f64.sqrt();
        let trio = market_steps(
            &[(&a[..], &[][..]), (&b[..], &[][..]), (&c[..], &[][..])],
            grid,
            3,
        );
        let z = |steps: &MarketSteps, source: &[PackedBar]| {
            auxiliary_row(
                &set,
                steps,
                single_series(source, &[], grid),
                &source[0],
                &source[1],
                false,
                &[],
            )
        };
        assert!((f64::from(z(&trio, &a)[0]) + expected).abs() < 1e-5);
        assert!((f64::from(z(&trio, &b)[0])).abs() < 1e-5);
        assert!((f64::from(z(&trio, &c)[0]) - expected).abs() < 1e-5);
        assert_eq!(
            [z(&trio, &a)[1], z(&trio, &b)[1], z(&trio, &c)[1]],
            [1.0; 3]
        );

        // A LOCKED bar - one print, a halt or a limit lock, `high == low` - has no range to
        // standardize. It enters neither moment and reads [0, 0], and the others are unmoved.
        let locked = [rich(0, 10.0, 1.0, 0.1), rich(1, 10.1, 1.0, 0.0)];
        let four = market_steps(
            &[
                (&a[..], &[][..]),
                (&b[..], &[][..]),
                (&c[..], &[][..]),
                (&locked[..], &[][..]),
            ],
            grid,
            3,
        );
        assert_eq!(four.contributors(1), 4);
        assert_eq!(z(&four, &locked), vec![0.0, 0.0]);
        assert!((f64::from(z(&four, &a)[0]) + expected).abs() < 1e-5);
        assert!((f64::from(z(&four, &c)[0]) - expected).abs() < 1e-5);

        // ZERO spread across the slot, and a SINGLE ranged bar: both read [0, 0].
        let flat_b = [rich(0, 98.0, 1.0, 0.5), rich(1, 100.0, 1.0, 0.5)];
        let flat_c = [rich(0, 97.0, 1.0, 0.5), rich(1, 100.0, 1.0, 0.5)];
        let flat = market_steps(
            &[
                (&a[..], &[][..]),
                (&flat_b[..], &[][..]),
                (&flat_c[..], &[][..]),
            ],
            grid,
            3,
        );
        assert_eq!(z(&flat, &a), vec![0.0, 0.0]);
        let lock_b = [rich(0, 98.0, 1.0, 2.0), rich(1, 100.0, 1.0, 0.0)];
        let lock_c = [rich(0, 97.0, 1.0, 8.0), rich(1, 100.0, 1.0, 0.0)];
        let alone = market_steps(
            &[
                (&a[..], &[][..]),
                (&lock_b[..], &[][..]),
                (&lock_c[..], &[][..]),
            ],
            grid,
            3,
        );
        assert_eq!(z(&alone, &a), vec![0.0, 0.0]);
    }

    #[test]
    fn appending_the_cross_section_channels_leaves_every_earlier_channel_identical() {
        let grid = Grid::new(0, 2 * RESOLUTION_MS).unwrap();
        let a = [
            rich(0, 100.0, 500.0, 0.5),
            rich(1, 102.0, 700.0, 0.6),
            rich(2, 100.5, 300.0, 0.4),
        ];
        let b = [
            rich(0, 50.0, 900.0, 0.2),
            rich(1, 50.25, 400.0, 0.3),
            rich(2, 50.5, 250.0, 0.1),
        ];
        let c = [
            rich(0, 20.0, 150.0, 0.05),
            rich(1, 20.2, 600.0, 0.08),
            rich(2, 20.9, 100.0, 0.03),
        ];
        let sources = [(&a[..], &[][..]), (&b[..], &[][..]), (&c[..], &[][..])];
        let steps = market_steps(&sources, grid, 3);
        let ranks = cross_section_ranks(&sources, grid, &steps);
        let control: FeatureSet = "time-of-day,day-of-week,session-gap,volume,market,spy"
            .parse()
            .unwrap();
        assert_eq!((control.channels(), FeatureSet::ALL.channels()), (12, 22));
        for future in [false, true] {
            for (index, (previous, current)) in
                [(&a[0], &a[1]), (&a[1], &a[2])].into_iter().enumerate()
            {
                let old = auxiliary_row(
                    &control,
                    &steps,
                    single_series(&a, &[], grid),
                    previous,
                    current,
                    future,
                    &[],
                );
                let all = auxiliary_row(
                    &FeatureSet::ALL,
                    &steps,
                    single_series(&a, &[], grid),
                    previous,
                    current,
                    future,
                    &ranks[0][index + 1..],
                );
                assert_eq!(old.as_slice(), &all[..12]);
                // Beyond the context all ten appended channels are history and read nothing.
                if future {
                    assert_eq!(all[12..], [0.0; 10]);
                }
            }
        }
        // And on a live bar every appended channel is populated, so the equality above is not
        // the trivial one that would also hold if the channels never wrote anything.
        let all = auxiliary_row(
            &FeatureSet::ALL,
            &steps,
            single_series(&a, &[], grid),
            &a[0],
            &a[1],
            false,
            &ranks[0][1..],
        );
        assert_eq!([all[13], all[15], all[17], all[19], all[21]], [1.0; 5]);
        assert!(all[12] < 0.0);
        // A prints the slot's largest of three returns, so its plotting position is 3/4.
        assert!((f64::from(all[16]) - 0.674_489_750_196_081_7).abs() < 1e-6, "{all:?}");
        assert!([all[14], all[18], all[20]].iter().all(|value| *value != 0.0), "{all:?}");
        assert_eq!(FeatureSet::ALL.channel_mask(Feature::known_future)[12..], [false; 10]);
        assert_eq!(FeatureSet::ALL.channel_mask(Feature::sigma_scaled)[12..], [false; 10]);
    }
}
